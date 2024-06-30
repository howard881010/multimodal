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
import json

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://www.google.com'
}
def download_single_url(url):
    try:
        r = requests.get(url, headers=headers)
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

def load_json(file_path):
    """
    Load a JSON file and return the parsed JSON data.
    
    Parameters:
        file_path (str): The path to the JSON file.
        
    Returns:
        dict: The parsed JSON data.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



def chunk_documents(tokenizer, data, max_token_length=4096, overlap=0):
    # chunk list of documents
    if type(data) == list:
        chunk_data = []
        for document in data:
            token_length = len(tokenizer.encode('\n'.join(document)))
            avg_token_per_summary = token_length / len(document)
            document_per_chunk = int(max_token_length / avg_token_per_summary)
            if token_length > max_token_length:
                chunks = [
                    document[i: i + document_per_chunk]
                    for i in range(0, len(data), document_per_chunk)
                ]
                chunk_data += chunks
            else:
                chunk_data.append(document)
        return chunk_data

    # chunk one long document
    elif type(data) == str:
        avg_token_per_data = len(tokenizer.encode(data)) / len(data)
        txt_per_chunk = int(max_token_length / avg_token_per_data)
        if overlap == 0:
            chunk_data = [data[i: i+txt_per_chunk]
                        for i in range(0, len(data), txt_per_chunk)]
        else:
            txt_overlap = int(overlap / avg_token_per_data)
            chunk_data = [data[max(0, i-txt_overlap): i+txt_per_chunk+txt_overlap]
                        for i in range(0, len(data), txt_per_chunk)]
        return chunk_data