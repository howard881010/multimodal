from .Processor import DatasetProcessor
from glob import glob
import numpy as np
import pandas as pd
from ast import literal_eval
from datasets import DatasetDict, concatenate_datasets
import json
import os

class MedicalDataProcessor(DatasetProcessor):
    def __init__(self, cfg):
        super().__init__(cfg)

    def get_dataset(self):
        cfg = self.cfg
        input_window = cfg['input_window']
        output_window = cfg['output_window']

        instruction_1 = self.get_instruction(1)
        print(instruction_1)
        instruction_2 = self.get_instruction(2)
        print(instruction_2)
        instruction_3 = self.get_instruction(3)
        print(instruction_3)
        instruction_4 = self.get_instruction(4)
        print(instruction_4)
        dataset_paths = glob(cfg['dataset_path'] + "/*.csv")

        datasets_list = []
        for dataset_path in dataset_paths:
            df = self.get_dataframe(dataset_path)
            input_text = self.get_text(df, 'input', 'text')
            output_text = self.get_text(df, 'output', 'text')
            input_text_time  = self.get_text(df, 'input', 'text_time')
            output_text_time  = self.get_text(df, 'output', 'text_time')
            output_time = self.get_text(df, 'output', 'time')
            input_num = self.get_time(df, "input")
            output_num = self.get_time(df, "output")

            new_df = pd.DataFrame({
                'input_text': pd.Series(input_text).reindex(df.index, fill_value=np.nan),
                'output_text': pd.Series(output_text).reindex(df.index, fill_value=np.nan),
                'input_text_time': pd.Series(input_text_time).reindex(df.index, fill_value=np.nan),
                'output_text_time': pd.Series(output_text_time).reindex(df.index, fill_value=np.nan),
                'output_time': pd.Series(output_time, dtype=object).reindex(index=np.arange(len(output_time)), fill_value=np.nan),
                'input_num': pd.Series(input_num, dtype=object).reindex(index=np.arange(len(input_num)), fill_value=np.nan),
                'output_num': pd.Series(output_num, dtype=object).reindex(index=np.arange(len(output_num)), fill_value=np.nan),
            })
            new_df['instruction-1'] = instruction_1
            new_df['instruction-2'] = instruction_2
            new_df['instruction-3'] = instruction_3
            new_df['instruction-4'] = instruction_4

            # get rid of nan tails
            new_df = new_df.iloc[:-(input_window + output_window-1)]
            train_df, valid_df, test_df = self.get_df_split(new_df)
            dataset = self.df_to_dataset(train_df, valid_df, test_df)

            datasets_list.append(dataset)

        train_datasets = []
        valid_datasets = []
        test_datasets = []

        # Iterate through the list of DatasetDicts
        for dataset in datasets_list:
            train_datasets.append(dataset['train'])
            valid_datasets.append(dataset['valid'])
            test_datasets.append(dataset['test'])

        # Concatenate datasets for each split
        concatenated_train = concatenate_datasets(train_datasets)
        concatenated_valid = concatenate_datasets(valid_datasets)
        concatenated_test = concatenate_datasets(test_datasets)

        # Create a new DatasetDict with the concatenated datasets
        concatenated_dataset = DatasetDict({
            'train': concatenated_train,
            'valid': concatenated_valid,
            'test': concatenated_test,
        })

        return concatenated_dataset

    def get_text(self, df, type, case):
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
        row_text = {}
        for t in curr_window_range:
            row_text = {}
            for curr_day_idx in curr_day_range:
                row_text[date_template.format(timestep=timestep, index=curr_day_idx+1)] = df.loc[t+curr_day_idx, 'date']
                if case not in ['time']:
                    row_text[text_template.format(timestep=timestep, index=curr_day_idx+1)] = df.loc[t+curr_day_idx, 'text']

                if case in ['text_time', 'time']:
                    for col in numerical_columns:
                        row_text[time_template.format(timestep=timestep, index=curr_day_idx+1, col=col)] = df.loc[t+curr_day_idx, col]
                    
            row_text = json.dumps(row_text, indent=4)
            # print(row_text)
            results.append(row_text)

        return results

    def get_instruction(self, case):
        cfg = self.cfg
        timestep = cfg['timestep']
        input_window = cfg['input_window']
        output_window = cfg['output_window']
        instruction_template = cfg['instruction_template']
        date_template = cfg['date_template']
        text_template = cfg['text_template']
        time_template = cfg['time_template']

        output_template = {}
        for i in range(input_window + 1, input_window + output_window + 1):
            output_template[date_template.format(timestep=timestep, index=i)] = "YYYY-MM-DD"
            if case in [1, 2, 3]:
                output_template[text_template.format(timestep=timestep, index=i)] = "Medical description"
            if case in [2, 4]:
                for col in cfg['numerical_columns']:
                    output_template[time_template.format(timestep=timestep, index=i, col=col)] = "A Float Number"
        output_template = json.dumps(output_template, indent=4)

        instruction = instruction_template.format(
            input_window=input_window,
            timestep=timestep,
            output_window=output_window,
        ) + output_template

        return instruction
    
    def get_dataframe(self, dataset_path):
        df = pd.read_csv(dataset_path)
        numerical_columns = self.cfg['numerical_columns']
        df[numerical_columns] = df[numerical_columns].round(3)
        return df

    def push_to_huggingface(self, dataset_dict):
        for split in dataset_dict:
            new_column_output = ["Not Predicted"] * len(dataset_dict[split])

            
            dataset_dict[split] = dataset_dict[split].add_column('pred_output_case1', new_column_output)
            dataset_dict[split] = dataset_dict[split].add_column('pred_output_case2', new_column_output)
            dataset_dict[split] = dataset_dict[split].add_column('pred_output_case3', new_column_output)
            dataset_dict[split] = dataset_dict[split].add_column('pred_output_case4', new_column_output)
            # dataset_dict[split] = dataset_dict[split].add_column('pred_time', new_column_time)

        token = os.getenv("HF_TOKEN")
        # Push the dataset to the Hugging Face Hub
        dataset_dict.push_to_hub(self.cfg['hf_repo'], token=token)