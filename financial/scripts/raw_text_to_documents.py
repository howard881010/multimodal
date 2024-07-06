import argparse
from queue import Queue
from threading import Thread, Lock
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os

from src.utils import load_json, get_logger, get_config
from src.engine import Engine
from src.summary_pipeline import DataParser
from templates.prompts import Prompts


class SummaryPipeline():
    def __init__(self, config):
        self.config = config

        api_key = os.getenv('API_KEY', 'EMPTY')
        base_url = os.getenv('BASE_URL', 'http://localhost:8000/v1')

        self.ticker = self.config['ticker']
        self.prompts = Prompts(self.ticker)
        self.engine = Engine(self.config['model'], api_key, base_url)
        self.json_schema = load_json(os.path.join(
            self.config['root_dir'], self.config['json_schema_dir']))

        # allow tokenizer to load correct model
        if self.config['model'] == 'llama3-70b':
            model = "meta-llama/Meta-Llama-3-70B-Instruct"
        else:
            model = self.config['model']
        self.utils = DataParser(model, self.json_schema)

        self.df = self.initialize_df()
        self.queue = Queue(maxsize=self.config['queue'])
        self.lock = Lock()

    def initialize_df(self):
        document_path = f"{self.config['root_dir']}/{self.config['data_dir']}/{self.config['ticker']}.csv"
        df = pd.read_csv(document_path)
        return df

    def run(self):
        doc_workers = self.config['workers']
        doc_threads = []
        for _ in range(doc_workers):
            thread = Thread(target=self.worker_wrapper)
            thread.start()
            doc_threads.append(thread)

        for row_idx, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            if type(row['text']) == float or \
                    any(p.lower() in row['text'].strip().lower() for p in self.config['blocked_words']):
                continue

            documents = self.utils.split_documents(
                row['text'], self.config['max_chunk_token'], self.config['overlap_token'])
            for doc_idx, doc in enumerate(documents):
                self.queue.put((row_idx, doc_idx, row['timestamp'], doc))

    def worker_wrapper(self):
        results = []
        while True:
            item = self.queue.get()
            if item is None:
                break
            try:
                result = self.worker(item)
                if result is not None:
                    results.append(result)
            finally:
                self.queue.task_done()
        return results

    def worker(self, item):
        row_idx, doc_idx, timestamp, document = item

        # initialize summary chunk save path
        with self.lock:
            document_path = os.path.join(self.config['root_dir'],
                                         self.config['save_dir'],
                                         self.ticker,
                                         timestamp + ".csv")
            try:
                document_df = pd.read_csv(document_path)
            except:
                document_df = pd.DataFrame(
                    columns=["row_idx", "doc_idx", "timestamp", "summary"])
                document_df.to_csv(document_path, index=False)

        if len(document_df[(document_df['row_idx'] == row_idx) &
                           (document_df['doc_idx'] == doc_idx) &
                           (document_df['timestamp'] == timestamp)]) == 0:
            self.log(
                f"{self.ticker}, {timestamp}, {row_idx}, {doc_idx}, {document[:100]}")

            result = self.engine.run(
                self.prompts.json_summary_prompt, document, self.json_schema)
            new_row = pd.DataFrame({"row_idx": [row_idx], "doc_idx": [
                                    doc_idx], "timestamp": timestamp, "summary": [str(result)]})

            # save summary chunk
            with self.lock:
                document_df = pd.read_csv(document_path)
                document_df = pd.concat(
                    [document_df, new_row], ignore_index=True)
                document_df.to_csv(document_path, index=False)

            # return new_row
        return None

    def log(self, msg):
        logger.info(msg)
        print(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize stock news by splitting raw url into documents.")
    parser.add_argument('--ticker', type=str, required=True,
                        help='Stock ticker symbol')
    parser.add_argument('--config_path', type=str,
                        required=True, help='Yaml Config File')
    args = parser.parse_args()
    config = get_config(args)

    load_dotenv()
    logger = get_logger(os.path.join(config['root_dir'],
                                     config['log_dir'],
                                     config['ticker'] + "_summary.txt"))

    pipeline = SummaryPipeline(config)
    pipeline.run()
