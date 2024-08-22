from .Processor import DatasetProcessor
from glob import glob
import numpy as np
import pandas as pd
from ast import literal_eval
from datasets import DatasetDict, concatenate_datasets

class MedicalDataProcessor(DatasetProcessor):
    def __init__(self, cfg):
        super().__init__(cfg)

    def get_dataset(self):
        cfg = self.cfg
        input_window = cfg['input_window']
        output_window = cfg['output_window']
        train_split = cfg['train_split']
        valid_split = cfg['valid_split']

        instruction = self.get_instruction()
        dataset_paths = glob(cfg['dataset_path'] + "/*.csv")

        datasets_list = []
        for dataset_path in dataset_paths:
            df = self.get_dataframe(dataset_path)
            input_text = df.apply(lambda x: self.create_flattened_text(
                x.name, input_window, df), axis=1)
            output_text = df.apply(lambda x: self.create_flattened_text(
                x.name+input_window, output_window, df, input_window+1), axis=1)
            input_time = self.get_time(df, "input")
            output_time = self.get_time(df, "output")

            new_df = pd.DataFrame({
                'input_text': pd.Series(input_text).reindex(df.index, fill_value=np.nan),
                'output_text': pd.Series(output_text).reindex(df.index, fill_value=np.nan),
                # 'input_time': pd.Series(input_time, dtype=object).reindex(index=np.arange(len(input_time)), fill_value=np.nan),
                # 'output_time': pd.Series(output_time, dtype=object).reindex(index=np.arange(len(output_time)), fill_value=np.nan),
            })
            new_df['instruction'] = instruction

            # get rid of nan tails
            new_df = new_df.iloc[:-(input_window + output_window-1)]
            train_df, valid_df, test_df = self.get_df_split(new_df, train_split, valid_split)
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

    def get_dataframe(self, dataset_path):
        df = pd.read_csv(dataset_path)
        df = df.drop(columns=['TEXT'])
        df = df.rename(columns={'CHARTTIME': 'date', 'summary': 'text'})
        numerical_columns = self.cfg['numerical_columns']
        df[numerical_columns] = df[numerical_columns].round(3)
        return df
