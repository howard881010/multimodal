from tqdm import tqdm
import pandas as pd
import os
from glob import glob
from utils import download_raw_texts_with_path_url, get_logger, load_text_from
import logging
from urllib.parse import urlparse

data_dir = "/data/kai/forecasting/data/summary_v2"
os.makedirs(data_dir, exist_ok=True)

paths = sorted(glob("/data/kai/forecasting/data/raw_urls/*.csv"))

all_urls = []
all_paths = []


def domain(url):
    return urlparse(url).netloc


for directory_path in tqdm(paths, total=len(paths)):
    df = pd.read_csv(directory_path)
    ticker = df.iloc[0]["ticker"]

    get_logger(f"logs/{ticker}_raw_texts.txt")

    logging.info(f"Starting with {len(df)}")
    df = df[~df["url"].str.contains("zacks.com", na=False)]
    df.drop_duplicates(subset=['url'], keep='last', inplace=True)

    logging.info(f"Dropped duplicate to {len(df)}")

    for idx, date_str in enumerate(df["timestamp"].unique()):
        raw_path = f"{data_dir}/{ticker}"
        os.makedirs(raw_path, exist_ok=True)

        save_path = raw_path + f'/{date_str}_raw.txt'
        if os.path.exists(save_path):
            continue

        current_urls = df[df["timestamp"] == date_str]["url"]
        # current_urls = [url for url in current_urls if "benzinga.com" in url]
        # raw_texts = download_raw_texts_from_urls(current_urls, max_workers=1)
        if len(current_urls) > 0:
            all_urls += list(current_urls.values)
            all_paths += [save_path] * len(current_urls)
            # save_text_to(raw_texts, save_path)
        else:
            print(raw_path, date_str, 'not found...')

scraped_urls_path = os.path.join(data_dir, "urls.txt")
try:
    scraped_urls = load_text_from(scraped_urls_path)
except:
    scraped_urls = []

js_blocked_urls = ['www.investors.com',
                   'www.business-standard.com',
                   'www.financialexpress.com',
                   'www.marketwatch.com',
                   'www.thestreet.com',
                   'www.reuters.com',
                   'www.benzinga.com']
to_process_idxs = []
for i, url in enumerate(all_urls):
    if url not in scraped_urls and "investors.com" not in url and domain(url) not in js_blocked_urls:
        to_process_idxs.append(i)
print("scraped / unscraped / total", len(scraped_urls),
      len(to_process_idxs), len(all_urls))

all_urls = [all_urls[i] for i in to_process_idxs]
all_paths = [all_paths[i] for i in to_process_idxs]
raw_texts = download_raw_texts_with_path_url(
    all_urls, all_paths, scraped_urls_path)
