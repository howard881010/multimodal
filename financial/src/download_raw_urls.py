from glob import glob
from urllib.parse import urlparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import os
from tqdm import tqdm
import time
import numpy as np


# utils
def get_domain(url):
    return urlparse(url).netloc

class Downloader:
    def __init__(self, df_paths, save_dir, max_workers=5):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com'
        }
        self.max_workers = max_workers
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        #  sorted(glob('/home/mkim/SServer/financial/data/raw_urls/*'))
        self.df_paths = df_paths
        self.domain_last_request = {}
        self.domain_lock = Lock()
        self.save_lock = Lock()
        self.domain_sleep_time = 5

    def download_url(self, url: str):
        try:
            r = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(r.text, 'html.parser')
            parsed_text = ' '.join(soup.text.replace('\n', " ").strip().split())
            return parsed_text
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return np.nan

    def process_url(self, idx, ticker, url, df):
        if type(df.loc[idx, 'text']) != str:
            domain = get_domain(url)
            
            with self.domain_lock:
                last_request_time = self.domain_last_request.get(domain, 0)
                current_time = time.time()
                if current_time - last_request_time < self.domain_sleep_time:
                    time.sleep(self.domain_sleep_time - (current_time - last_request_time))
                self.domain_last_request[domain] = time.time()

            text = self.download_url(url)
            df.loc[idx, 'text'] = text

            file_path = os.path.join(self.save_dir, f"{ticker}.csv")
            with self.save_lock:
                df.to_csv(file_path, index=False)
            print('---------------')
            print(ticker, url)
            print(text[:100])
        else:
            print('already downloaded', idx, ticker, url)

    def download_ticker(self, df_path):
        df = pd.read_csv(df_path)
        df.drop_duplicates(subset=['url'], inplace=True)

        ticker = df.iloc[0]["ticker"]
        urls = df['url']
        save_path = os.path.join(self.save_dir, f"{ticker}.csv")

        # copy dataframe
        if not os.path.isfile(save_path):
            df['text'] = np.nan
            df.to_csv(save_path, index=False)
        else:
            df = pd.read_csv(save_path)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.process_url, idx, ticker, url, df)
                for idx, url in enumerate(urls)
            ]

            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {ticker} URLs"):
                pass

    def download_tickers(self):
        for df_path in self.df_paths:
            self.download_ticker(df_path)
        
save_dir = "/home/mkim/SServer/financial/data/summary_v0.2"
df_paths =  sorted(glob('/home/mkim/SServer/financial/data/raw_urls/*'))

downloader = Downloader(df_paths, save_dir)
downloader.download_tickers()