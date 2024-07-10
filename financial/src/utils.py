import time
from datetime import datetime, timedelta
import logging
import os
from urllib.parse import urlparse
import json
import yaml


# download utils
def get_domain(url):
    return urlparse(url).netloc


# summarize_documents utils
def get_config(args):
    config = load_yaml(args.config_path)
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    return config


def load_yaml(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


# logging
def get_logger(filepath="log.txt"):
    os.makedirs('/'.join(filepath.split("/")[:-1]), exist_ok=True)
    logging.basicConfig(
        filename=filepath,  # Set the filename for the log file
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'  # Set the format for the date in log messages
    )
    logger = logging.getLogger("FinancialSummary")
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)

    # Add a StreamHandler to log to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def log_time(start_time: float):
    elapsed_time = time.time()
    elapsed_timedelta = timedelta(seconds=elapsed_time - start_time)
    logging.info(f"Elapsed time: {elapsed_timedelta}")

    start_datetime = datetime.fromtimestamp(start_time)
    end_datetime = datetime.fromtimestamp(elapsed_time)

    logging.info(f"Start time: {start_datetime}")
    logging.info(f"End time: {end_datetime}\n")
