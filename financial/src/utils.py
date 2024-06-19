import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
from datetime import datetime, timedelta
from typing import Union

def download_raw_texts_from_urls(current_urls):
    raw_texts = []
    for url in tqdm(current_urls, total=len(current_urls)):
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        parsed_text = ' '.join(soup.text.replace('\n', " ").strip().split())
        raw_texts.append(parsed_text)
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

def print_time(start_time: float):
    elapsed_time = time.time()
    elapsed_timedelta = timedelta(seconds=elapsed_time - start_time)
    print(f"Elapsed time: {elapsed_timedelta}")

    start_datetime = datetime.fromtimestamp(start_time)
    end_datetime = datetime.fromtimestamp(elapsed_time)

    print(f"Start time: {start_datetime}")
    print(f"End time: {end_datetime}\n")
