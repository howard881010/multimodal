from tqdm import tqdm
import pandas as pd
import os
from glob import glob
from utils import download_raw_texts_from_urls, save_text_to, get_logger
import logging

data_dir = "/data/kai/forecasting/summary"
os.makedirs(data_dir, exist_ok=True)

paths = sorted(glob("/data/kai/forecasting/data/raw_urls/*.csv"))

for directory_path in tqdm(paths, total=len(paths)):
    df = pd.read_csv(directory_path)
    ticker = df.iloc[0]["ticker"]

    get_logger(f"logs/{ticker}_raw_texts.txt")

    logging.info(f"Starting with {len(df)}")
    df = df[~df["url"].str.contains("zacks.com", na=False)]
    df.drop_duplicates(subset=['url'], keep='last', inplace=True)

    logging.info(f"Dropped duplicate to {len(df)}")

    for idx, date_str in tqdm(enumerate(df["timestamp"].unique()), total=len(df["timestamp"].unique())):
        raw_path = f"{data_dir}/{ticker}"
        os.makedirs(raw_path, exist_ok=True)

        save_path = raw_path + f'/{date_str}_raw.txt'
        if os.path.exists(save_path):
            continue

        current_urls = df[df["timestamp"] == date_str]["url"]
        raw_texts = download_raw_texts_from_urls(current_urls)

        save_text_to(raw_texts, save_path)
