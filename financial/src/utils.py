import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
from datetime import datetime, timedelta
from typing import Union
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import logging
import os
from urllib.parse import urlparse

def download_single_url(url):
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        parsed_text = ' '.join(soup.text.replace('\n', " ").strip().split())
        return parsed_text
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


def download_raw_texts_with_path_url(current_urls, save_paths, scraped_urls_path, max_workers=10):
    raw_texts = []
    url_queue = Queue()
    domain_locks = {}
    
    # Populate the queue
    for url, path in zip(current_urls, save_paths):
        url_queue.put((url, path))
        
    def worker():
        while not url_queue.empty():
            url, path = url_queue.get()
            domain = urlparse(url).netloc
            if domain not in domain_locks:
                domain_locks[domain] = Lock()
            with domain_locks[domain]:
                result = download_single_url(url)
                if result is not None:
                    raw_texts.append(result)
                    add_text_to(result, path)
                    add_text_to(url, scraped_urls_path)
                    if "page not found" in result.lower():
                        print('----- denied -----')
                        print(f"{'.'.join(path.split('/')[-2:])} --- {domain}")
                        print(f"-- {result[:200]}")
            url_queue.task_done()

    # Start worker threads
    threads = []
    for _ in range(max_workers):
        thread = Thread(target=worker)
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()

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

def add_text_to(data: Union[str, list[str]], file_path: str):
    with open(file_path, 'a') as file:
        if isinstance(data, str):
            file.write(data + "\n")
        elif isinstance(data, list):
            for line in data:
                file.write(line + "\n")
                file.write("<SEP>" + "\n")

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
