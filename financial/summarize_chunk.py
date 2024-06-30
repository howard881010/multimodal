import pandas as pd
from threading import Thread, Lock
from queue import Queue
import os
from tqdm import tqdm
from src.vllm import llm_chat, message_template
from src.utils import load_json, chunk_documents
import json
from transformers import AutoTokenizer
import json
from templates.PROMPTS import Prompts, blocked_words
import argparse


def process_document(prompt, doc, guided_json=None):
    messages = message_template(prompt, doc)
    response = llm_chat(messages, guided_json=guided_json)
    data = json.loads(response)
    return data


def combine_results(results):
    results_dict = {
        'key_numbers': [],
        'growth_trends': [],
        'overall_market_outlook': [],
        'major_stock_movements': [],
        'significant_economic_indicators': [],
        'notable_company_specific_news': [],
        'summary': []
    }

    # Append each value to the appropriate list in results_dict
    for result in results:
        for key in result:
            # For lists of dictionaries (like 'key_numbers')
            if isinstance(result[key], list):
                results_dict[key].extend(result[key])
            else:
                results_dict[key].append(result[key])
    return results_dict


def worker(queue):
    while True:
        data = queue.get()
        if data is None:
            break
        row_idx, doc_idx, timestamp, document = data

        with document_lock:
            document_df = pd.read_csv(document_path)

        if len(document_df[(document_df['row_idx'] == row_idx) &
                           (document_df['doc_idx'] == doc_idx) &
                           (document_df['timestamp'] == timestamp)]) == 0:
            result = process_document(
                prompts.JSON_SUMMARY_PROMPT, document, guided_json)
            new_row = pd.DataFrame({"row_idx": [row_idx], "doc_idx": [
                                   doc_idx], "timestamp": timestamp, "summary": [str(result)]})
            with document_lock:
                document_df = pd.read_csv(document_path)
                document_df = pd.concat(
                    [document_df, new_row], ignore_index=True)
                document_df.to_csv(document_path, index=False)
        queue.task_done()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process stock ticker information.")
    parser.add_argument('--ticker', type=str, required=True,
                        help='Stock ticker symbol')
    args = parser.parse_args()

    # intialize prompts and tokenizer
    prompts = Prompts()
    model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # initialize guided json
    guided_json = load_json('templates/guided_json_summary_v0.3.json')

    # initialize raw df
    ticker = args.ticker
    ticker_path = f"/data/kai/forecasting/data/raw_v0.2/{ticker}.csv"
    df = pd.read_csv(ticker_path)

    # initialize document df
    document_dir = "/data/kai/forecasting/data/document_v0.2"
    os.makedirs(document_dir, exist_ok=True)
    document_path = os.path.join(document_dir, f"{ticker}.csv")

    if not os.path.exists(document_path):
        document_df = pd.DataFrame(
            columns=["row_idx", "doc_idx", "timestamp", "summary"])
        document_df.to_csv(document_path, index=False)
    else:
        document_df = pd.read_csv(document_path)

    document_queue = Queue(maxsize=10)
    document_lock = Lock()

    # start document worker
    doc_workers = 10
    doc_threads = []
    for i in range(doc_workers):
        thread = Thread(target=worker, args=(document_queue,))
        thread.start()
        doc_threads.append(thread)

    # Add documents to the queue
    for row_idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        if any(p.lower() in row['text'].strip().lower() for p in blocked_words):
            continue

        documents = chunk_documents(tokenizer, row['text'], 2048, overlap=512)
        for doc_idx, doc in enumerate(documents):
            document_queue.put((row_idx, doc_idx, row['timestamp'], doc))

    # wait for worker to finish
    document_queue.join()
    for i in range(doc_workers):
        document_queue.put(None)
    for thread in doc_threads:
        thread.join()
