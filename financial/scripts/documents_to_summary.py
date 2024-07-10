import argparse
from threading import Lock
import pandas as pd
from dotenv import load_dotenv
import os
from src.utils import load_json, get_logger, get_config
from src.engine import Engine
from src.summary_parser import DataParser
from templates.prompts import Prompts
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
import json
import traceback
import numpy as np


class CombinePipeline():
    def __init__(self, config):
        self.config = config

        api_key = os.getenv('API_KEY', 'EMPTY')
        base_url = os.getenv('BASE_URL', 'http://localhost:8000/v1')
        self.ticker = self.config['ticker']
        self.prompts = Prompts(self.ticker)
        self.engine = Engine(self.config['model'], api_key, base_url)
        self.json_schema = load_json(os.path.join(
            self.config['root_dir'], self.config['json_schema_dir']))

        self.utils = DataParser(self.config['model'], self.json_schema)

        self.summary_paths = sorted(
            glob(f"{config['root_dir']}/{config['data_dir']}/{self.ticker}/*.csv"))

        self.save_dir = os.path.join(self.config['root_dir'],
                                     self.config['save_dir'],
                                     self.ticker)
        os.makedirs(self.save_dir, exist_ok=True)

    def run(self):
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(self.main, path)
                       for path in self.summary_paths]
            results = [future.result() for future in as_completed(futures)]

    def main(self, path):
        try:
            # same timestamp for whole document
            timestamp = path.split('/')[-1].split('.csv')[0]
            save_path = os.path.join(self.save_dir, timestamp + ".json")
            if os.path.exists(save_path):
                logger.info(f"{save_path} already exists")
                return

            json_summaries = pd.read_csv(path)["summary"].values
            # subsample if it's too big. Amazon sometimes has 100+
            json_summaries = np.random.choice(json_summaries, min(
                self.config['max_samples'], len(json_summaries)), replace=False)

            # collapse into {key: [{metric:value}, ...]}
            cleaned_summaries = self.utils.collapse_metrics(json_summaries)

            combine_count = 1
            logger.info(
                f"Combining summaries {combine_count} with len {len(cleaned_summaries)} --- {save_path}")
            json_results, formatted_results = self.run_results(
                cleaned_summaries)
            logger.info(f"{timestamp} - {json_results}")

            while len(self.utils.combine_results(formatted_results)["summary"]) > 1:
                combine_count += 1
                logger.info(
                    f"Combining summaries {combine_count} --- {save_path}")
                json_results, formatted_results = self.run_results(
                    formatted_results)

            with open(save_path, 'w') as json_file:
                json.dump(json_results[0][0], json_file, indent=4)
        except Exception as e:
            traceback.print_exc()
            logger.info(str(e))

    def run_results(self, results):
        # json results can have empty values like in original documents summary
        # formatted results is the input for the LLM
        json_results = self.batch_llm_combine_summaries(results) # list of dicts
        formatted_results = self.utils.collapse_metrics(json_results) # list of dicts
        return json_results, formatted_results

    def batch_llm_combine_summaries(self, summaries):
        summary_chunks = self.utils.chunk_summaries(summaries)
        with ThreadPoolExecutor(max_workers=self.config['workers']) as executor:
            futures = [executor.submit(self.llm_combine_chunk, chunk)
                       for chunk in summary_chunks]
            results = [future.result() for future in as_completed(futures)]
        return results

    def llm_combine_chunk(self, document):
        output = self.engine.run(
            self.prompts.combine_json_prompt, document, self.json_schema)
        return output


if __name__ == "__main__":
    # ticker = "AMZN"
    # config_path = "config/combine_v0.2.yaml"
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
                                     config['ticker'] + "_combine.txt"))

    pipeline = CombinePipeline(config)
    pipeline.run()
