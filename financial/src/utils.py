import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
from datetime import datetime, timedelta
from typing import Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os


def download_single_url(url):
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        parsed_text = ' '.join(soup.text.replace('\n', " ").strip().split())
        return parsed_text
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


def download_raw_texts_from_urls(current_urls, max_workers=10):
    raw_texts = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(
            download_single_url, url): url for url in current_urls}

        for future in tqdm(as_completed(futures), total=len(current_urls)):
            result = future.result()
            if result is not None:
                raw_texts.append(result)

    return raw_texts


def save_text_to(data: Union[str, list[str]], file_path: str):
    with open(file_path, 'w') as file:
        if type(data) == str:
            file.write(data)
        elif type(data) == list:
            for line in data:
                file.write(line + "\n")
                file.write("<SEP>" + "\n")


def load_text_from(file_path: str):
    with open(file_path, 'r') as file:
        data = file.read().splitlines()
    return data


def log_time(start_time: float):
    elapsed_time = time.time()
    elapsed_timedelta = timedelta(seconds=elapsed_time - start_time)
    logging.info(f"Elapsed time: {elapsed_timedelta}")

    start_datetime = datetime.fromtimestamp(start_time)
    end_datetime = datetime.fromtimestamp(elapsed_time)

    logging.info(f"Start time: {start_datetime}")
    logging.info(f"End time: {end_datetime}\n")


def get_logger(filepath="log.txt"):
    os.makedirs('/'.join(filepath.split("/")[:-1]), exist_ok=True)
    logging.basicConfig(
        filename=filepath,  # Set the filename for the log file
        # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        level=logging.INFO,
        # Set the format for log messages
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'  # Set the format for the date in log messages
    )
    logger = logging.getLogger("FinancialSummary")
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    return logger
