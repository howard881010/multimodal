from .Processor import DatasetProcessor
import pandas as pd
import numpy as np


class GasDataProcessor(DatasetProcessor):
    def __init__(self, cfg):
        super().__init__(cfg)

    def get_dataset(self):
        """
        Returns dataset with the following optional columns based on model_type
            - input_text
            - output_text
            - input_time_{column_names}
            - output_time_{column_names}
            - instruction
        """
        cfg = self.cfg
        input_window = cfg['input_window']
        output_window = cfg['output_window']

        instruction = self.get_instruction()
        df = self.get_dataframe(cfg['dataset_path'])

        input_text = self.get_text(df, 'input')
        output_text = self.get_text(df, 'output')
        input_time = self.get_time(df, 'input')
        output_time = self.get_time(df, 'output')

        new_df = pd.DataFrame({
            'input_text': pd.Series(input_text).reindex(df.index, fill_value=np.nan),
            'output_text': pd.Series(output_text).reindex(df.index, fill_value=np.nan),
            'input_time': pd.Series(input_time, dtype=object).reindex(index=np.arange(len(input_time)), fill_value=np.nan),
            'output_time': pd.Series(output_time, dtype=object).reindex(index=np.arange(len(output_time)), fill_value=np.nan),
        })
        new_df['instruction'] = instruction
        print(instruction)
        # get rid of nan tails
        new_df = new_df.iloc[:-(input_window + output_window-1)]

        train_df, valid_df, test_df = self.get_df_split(new_df)
        dataset = self.df_to_dataset(train_df, valid_df, test_df)

        return dataset

    def get_dataframe(self, dataset_path):
        # https://github.com/AdityaLab/Time-MMD

        # text_df = pd.read_csv(time_mmd_text_path)
        # text_df.drop(columns=['Unnamed: 0', 'preds', 'end_date'], inplace=True)
        # numerical_df = pd.read_csv(time_mmd_numerical_path)
        # numerical_df.columns = ["date", "OT", "East_Coast", "New_England", "Central_Atlantic", "Lower_Atlantic", "Midwest",
        #                         "Gulf_Coast", "Rocky_Mountain", "West_Coast", "start_date", "end_date"]
        # df = pd.merge(text_df, numerical_df, on="start_date")
        # df = df.drop(columns='date')
        # df.to_csv(save_path, index=False)

        numerical_columns = self.cfg['numerical_columns']
        df = pd.read_csv(dataset_path)
        # df = df.dropna()
        df = df[['start_date', 'fact'] + numerical_columns]
        df = df.rename(columns={'start_date': 'date', 'fact': 'text'})
        df[numerical_columns] = df[numerical_columns].round(3)

        return df
