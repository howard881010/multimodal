
# Given the following stock prices for each day and the summary of the financial news, predict the next 10 prices.

# Prepare Financial dataset in following format
# {
#     '100', 'summary...',
#     '105', 'summary...'
# }

import pandas as pd
import os
from glob import glob
from tqdm import tqdm

data_dir = '/data/kai/forecasting/data'
numerical_path = os.path.join(data_dir, 'numerical')
summary_path = os.path.join(data_dir, 'summary')
save_formatted_dir = os.path.join(data_dir, 'formatted')
os.makedirs(save_formatted_dir, exist_ok=True)

all_tickers = os.listdir(summary_path)
for ticker in all_tickers:
    ticker_numerical_path = os.path.join(
        numerical_path, ticker.lower() + '_numerical.csv')
    ticker_summary_paths = sorted(
        glob(f'{summary_path}/{ticker.upper()}/*_summary.txt'))

    ticker_numerical_df = pd.read_csv(ticker_numerical_path)

    data = {
        'timestamp': [],
        'price': [],
        'summary': []
    }

    missing_prices = []
    missing_summaries = []
    for current_summary_path in tqdm(ticker_summary_paths, total=len(ticker_summary_paths)):
        # '/data/kai/forecasting/data/summary/AAPL/2022-03-01_summary.txt' --> 2022-03-01
        current_timestamp = current_summary_path.split('/')[-1].split('_')[0]
        try:
            current_price = ticker_numerical_df[ticker_numerical_df['timestamp']
                                                == current_timestamp]['close'].item()
            with open(current_summary_path, 'r') as file:
                current_summary = file.read()
                if current_summary == '':
                    missing_summaries.append(current_timestamp)

            data['timestamp'].append(current_timestamp)
            data['price'].append(current_price)
            data['summary'].append(current_summary)
        except ValueError:
            missing_prices.append(current_timestamp)

    save_formatted_path = os.path.join(save_formatted_dir, ticker + ".csv")
    formatted_df = pd.DataFrame.from_dict(data)
    formatted_df.to_csv(save_formatted_path)
