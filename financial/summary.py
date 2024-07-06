import argparse
from src.utils import load_yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from queue import Queue
from threading import Lock
import pandas as pd
from tqdm import tqdm
from src.summary_utils import DataParser


class SummaryPipeline():
    def __init__(self, args):
        self.config = self.get_config(args)
        self.df = self.initialize_df()
        self.utils = DataParser(model=self.config['model'])
        self.queue = Queue(maxsize=self.config['queue'])
        self.lock = Lock()
        self.blocked_words = 

    def initialize_df(self):
        pattern = f"{self.config['root_dir']}/{self.config['data_dir']}/{self.config['ticker']}.csv"
        document_path = sorted(glob(pattern))
        df = pd.read_csv(document_path)
        return df

    def run(self):
        for row_idx, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            if type(row['text']) == float or \
                any(p.lower() in row['text'].strip().lower() for p in blocked_words):
                continue

            documents = self.utils.split_documents(row['text'], self.config[''])
        num_workers = self.config['run']['workers']
        results = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.worker_wrapper)
                       for _ in range(num_workers)]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error occurred: {e}")

            # Signal the workers to exit
            for _ in range(num_workers):
                self.queue.put(None)

        return results

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
        print(item)
        pass

    def get_config(self, args):
        config = load_yaml(args.config_path)
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
        return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize stock news by splitting raw url into documents.")
    parser.add_argument('--ticker', type=str, required=True,
                        help='Stock ticker symbol')
    parser.add_argument('--config_path', type=str,
                        required=True, help='Yaml Config File')
    args = parser.parse_args()

    pipeline = SummaryPipeline(args)
