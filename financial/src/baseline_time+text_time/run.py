from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import os
from src.vllm import llm_chat
from templates.PROMPTS import ForecstBaselinePrompts
from glob import glob
from tqdm import tqdm



save_dir = '/data/kai/forecasting/results'

window = 5
data_dir = '/data/kai/forecasting/data/formatted'

formatted_paths = sorted(glob(data_dir + "/*"))


# '/data/kai/forecasting/data/formatted/AAPL.csv' --> AAPL
ticker_path = formatted_paths[0]
for ticker_path in tqdm(formatted_paths, total=len(formatted_paths)):
    ticker = ticker_path.split('/')[-1].split('.csv')[0]
    ticker_df = pd.read_csv(ticker_path)
    prompts = ForecstBaselinePrompts(window)


    # format into price={}, summary={} <SEP> price={}, summmary={}...

    input_data_list = []
    ground_truth_list = []
    for i in range(0, len(ticker_df)-window*2):  # use next 5 price as ground truth
        window_df = ticker_df.iloc[i: i+window*2]
        window_prices = window_df['price'].values[:window]
        window_summaries = window_df['summary'].values[:window]
        x = []
        for d, (price, summary) in enumerate(zip(window_prices, window_summaries)):
            x.append(f'<Day {d+1} Price>{price}, summary={summary}')
        x = '<SEP>'.join(x)
        y = ', '.join([str(price)
                    for price in window_df['price'].values[window:window*2]])

        messages = [{
            "role": "system",
            "content": prompts.SYSTEM_PROMPT
        },
            {
            "role": "user",
            'content': x
        }]
        # result = llm_chat(messages)
        input_data_list.append(messages)
        ground_truth_list.append(y)


    results_with_ground_truth = []

    def save_result(future, ground_truth):
        result = future.result()
        results_with_ground_truth.append((result, ground_truth))

    with ThreadPoolExecutor(max_workers=5) as executor:
        # Create a mapping of future to ground truth
        future_to_ground_truth = {executor.submit(
            llm_chat, input_data): ground_truth for input_data, ground_truth in zip(input_data_list, ground_truth_list)}
        for future in tqdm(as_completed(future_to_ground_truth), total=len(future_to_ground_truth)):
            save_result(future, future_to_ground_truth[future])

    os.makedirs(save_dir, exist_ok=True)
    baseline_path = os.path.join(save_dir, f'{ticker}_baseline.csv')

    df = pd.DataFrame(results_with_ground_truth,
                    columns=["Result", "Ground Truth"])
    df.to_csv(baseline_path, index=False)
