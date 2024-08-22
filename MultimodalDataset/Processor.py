from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from ast import literal_eval
import torch
import os


class DatasetProcessor():
    """
    Processes summarized dataset and prase into huggingface dataset
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def get_loaders(self, dataset, split_type=None):
        """
        Split_type: 'train', 'valid', 'test' or None for all
        """
        loaders = {
            'train': DataLoader(dataset['train'], shuffle=False, batch_size=self.cfg["batch_size"], collate_fn=self.collate_fn),
            'valid': DataLoader(dataset['valid'], shuffle=False, batch_size=self.cfg["batch_size"], collate_fn=self.collate_fn),
            'test': DataLoader(dataset['test'], shuffle=False, batch_size=self.cfg["batch_size"], collate_fn=self.collate_fn),
        }

        if split_type is None:
            return loaders['train'], loaders['valid'], loaders['test']
        else:
            return loaders.get(split_type)

    def collate_fn(self, data):
        input_texts = [x['input_text'] for x in data]
        output_texts = [x['output_text'] for x in data]

        input_times = torch.tensor([x['input_time'] for x in data])
        output_times = torch.tensor([x['output_time'] for x in data])
        if not self.cfg['model_type'].startswith("nlinear"):
            if self.cfg['normalization'] == "window_last":
                min_values = input_times.min(dim=-2, keepdim=True)[0]
                max_values = input_times.max(dim=-2, keepdim=True)[0]
                input_times -= min_values
                output_times -= max_values
            elif self.cfg['normalization'] == "min_max":
                input_times /= torch.from_numpy(self.min_values)
                output_times /= torch.from_numpy(self.max_values)
                
        instructions = [x['instruction'] for x in data]

        return input_texts, output_texts, input_times, output_times, instructions

    def get_instruction(self):
        cfg = self.cfg
        timestep = cfg['timestep']
        input_window = cfg['input_window']
        output_window = cfg['output_window']
        instruction_template = cfg['instruction_template']
        date_template = cfg['date_template']
        text_template = cfg['text_template']
        time_template = cfg['time_template']

        output_template = [
            f"{date_template.format(timestep=timestep, index=i)}:" + '\n'
            f"{text_template.format(timestep=timestep, index=i)}:"
            for i in range(input_window + 1, input_window + output_window + 1)
        ]

        if self.cfg['model_type'] == "text_time_text_time":
            for i in range(len(output_template)):
                time_in_text = "".join(
                    [f"{time_template.format(timestep=timestep, index=i+input_window+1, col=col)}:"
                     for col in cfg['numerical_columns']])
                output_template[i] += '\n' + time_in_text

        instruction = instruction_template.format(
            input_window=input_window,
            timestep=timestep,
            output_window=output_window,
        ) + "```" + "\n" +  "\n".join(output_template) + '\n' + "```"

        return instruction

    def get_text(self, df, type):
        cfg = self.cfg
        timestep = cfg['timestep']
        input_window = cfg['input_window']
        output_window = cfg['output_window']
        date_template = cfg['date_template']
        text_template = cfg['text_template']
        time_template = cfg['time_template']
        numerical_columns = cfg['numerical_columns']

        results = []
        if type == "input":
            curr_day_range = range(input_window)
            curr_window_range = range(len(df) - input_window + 1)
        elif type == "output":
            curr_day_range = range(input_window, output_window+input_window)
            curr_window_range = range(
                len(df) - input_window - output_window + 1)

        for t in curr_window_range:
            row_text = ""

            for curr_day_idx in curr_day_range:
                row_text += f"{date_template.format(timestep=timestep, index=curr_day_idx+1)}: {df.loc[t+curr_day_idx, 'date']}" + "\n"
                row_text += f"{text_template.format(timestep=timestep, index=curr_day_idx+1)}: {df.loc[t+curr_day_idx, 'text']}"

                if self.cfg['model_type'] == "text_time_text_time":
                    time_in_text = []
                    for col in numerical_columns:
                        time_in_text.append(
                            f"{time_template.format(timestep=timestep, index=curr_day_idx+1, col=col)}: '{df.loc[t+curr_day_idx, col]}'")
                    row_text += "\n" + " ".join(time_in_text)
                row_text += '\n'
            row_text = '```' + '\n' + row_text + '```'

            results.append(row_text)

        return results

    def get_time(self, df, type):
        cfg = self.cfg
        input_window = cfg['input_window']
        output_window = cfg['output_window']
        numcerical_columns = cfg['numerical_columns']

        if type == "input":
            curr_day_range = range(input_window)
            curr_window_range = range(len(df) - input_window + 1)
        elif type == "output":
            curr_day_range = range(input_window, output_window+input_window)
            curr_window_range = range(
                len(df) - input_window - output_window + 1)
        time_arrays = [
            [[df.loc[t+curr_day_idx, col] for col in numcerical_columns]
             for curr_day_idx in curr_day_range]
            for t in curr_window_range
        ]
        # time_arrays = np.array(time_arrays)
        return time_arrays

    def create_flattened_text(self, start_index: int, window_size, df, start_timestep_num=1) -> str:
        """
        Format the json text into a flattened string
        """

        cfg = self.cfg
        timestep = cfg['timestep']
        date_template = cfg['date_template']
        text_template = cfg['text_template']
        time_template = cfg['time_template']
        numerical_columns = cfg['numerical_columns']

        try:
            data = []
            for i in range(start_index, min(start_index + window_size, len(df))):
                timestep_idx = i - start_index + start_timestep_num
                row_data = f"{date_template.format(timestep=timestep, index=timestep_idx)}: {df.loc[i, 'date']}"
                row_data += '\n'
                row_data += f"{text_template.format(timestep=timestep, index=timestep_idx)}: "

                # climate_json = literal_eval(df.loc[i, 'text'])

                # flattend_text = []
                # for k, texts in climate_json.items():
                #     texts = [text for text in texts if text != '']
                #     flattend_text.append('. '.join(str(x) for x in texts))
                climate_arr = literal_eval(df.loc[i, 'text'])
                flattend_text = ' '.join(climate_arr)
                row_data += flattend_text

                if self.cfg['model_type'] == "text_time_text_time":
                    time_in_text = []
                    for col in numerical_columns:
                        time_in_text.append(
                            f"{time_template.format(timestep=timestep, index=timestep_idx, col=col)}: '{df.loc[i, col]}'")
                    row_data += "\n" + " ".join(time_in_text)
                # row_data = "```" + '\n' + row_data + '\n' + "```"   
                data.append(row_data)
                

            return "```" + '\n' + '\n'.join(data) + '\n' + "```"
        except Exception as e:  # handle missing data
            return np.nan

    def get_df_split(self, df, train_split=0.8, valid_split=0.9):
        """
        Split dataframe into train, valid, test and min-max normalize with min max of 'input_time' and output_time' columns
        """
        df = df.copy()
        n_train = int(len(df) * train_split)
        n_valid = int(len(df) * (valid_split))

        train_df = df.iloc[:n_train]
        valid_df = df.iloc[n_train:n_valid]
        test_df = df.iloc[n_valid:]

        if self.cfg['normalization'] == "min_max":
            train_values_np = np.array([np.array(x) for x in train_df['input_time'].values])
            self.min_values = train_values_np.min(axis=(0, 1))
            self.max_values = train_values_np.max(axis=(0, 1))

        return train_df, valid_df, test_df

    def df_to_dataset(self, train_df, valid_df, test_df):
        train_dataset = Dataset.from_pandas(train_df)
        valid_dataset = Dataset.from_pandas(valid_df)
        test_dataset = Dataset.from_pandas(test_df)

        dataset_dict = DatasetDict({
            'train': train_dataset,
            'valid': valid_dataset,
            'test': test_dataset,
        })
        return dataset_dict

    # Apply the renaming function to each dataset in the DatasetDict
    def push_to_huggingface(self, dataset_dict):
        
        for split in dataset_dict:
            new_column_output = ["Not Predicted"] * len(dataset_dict[split])
            new_column_time = [[] for _ in range(len(dataset_dict[split]))]

            dataset_dict[split] = dataset_dict[split].rename_column('input_text', 'input')
            dataset_dict[split] = dataset_dict[split].rename_column('output_text', 'output')
            dataset_dict[split] = dataset_dict[split].add_column('pred_output', new_column_output)
            # dataset_dict[split] = dataset_dict[split].add_column('pred_time', new_column_time)

        token = os.getenv("HF_TOKEN")
        # Push the dataset to the Hugging Face Hub
        dataset_dict.push_to_hub(self.cfg['hf_repo'], token=token)
