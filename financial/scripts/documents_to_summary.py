import pandas as pd
from glob import glob
import json
from collections import defaultdict
import ast
from src.engine import llm_chat, message_template, call_llm_chat
from templates.prompts import Prompts
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
# check size of summary value
from transformers import AutoTokenizer

import numpy as np





def llm_combine_chunk(chunk):
    output = llm_chat(message_template(
        prompts.COMBINE_JSON_PROMPT, chunk), model=model_name, guided_json=guided_json)
    cleaned_output = collapse_metrics(output)
    return cleaned_output


def batch_llm_combine_summaries(summaries):
    summary_chunks = chunk_summaries(tokenizer, summaries)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(llm_combine_chunk, chunk)
                   for chunk in summary_chunks]
        results = [future.result() for future in as_completed(futures)]
    return results


def main(path: str):
    timestamp = path.split('/')[-1].split('.csv')[0]
    save_path = os.path.join(save_dir, timestamp + ".json")

    if os.path.exists(save_path):
        print(save_path, "already exists")
        return
    json_summaries = pd.read_csv(path)['summary'].values
    cleaned_results = collapse_metrics(json_summaries)
    # list[{k: v}, {k: {k:v}}]

    print("First combination task...", save_path)
    json_results = batch_llm_combine_summaries(cleaned_results)
    formatted_results = collapse_metrics(collapse_results(
        json_results))  # this is like cleaned_results

    def get_current_summary_length(results):
        return len(combine_results(results)["summary"])

    combine_count = 0
    while get_current_summary_length(formatted_results) > 1:
        combine_count += 1
        print(
            f"Combining summaries {combine_count} ... length summaries: {get_current_summary_length(formatted_results)}", save_path)
        json_results = batch_llm_combine_summaries(formatted_results)
        formatted_results = collapse_metrics(collapse_results(
            json_results))  # this is like cleaned_results

    with open(save_path, 'w') as json_file:
        json.dump(json_results[0][0], json_file, indent=4)


summary_path = "/data/kai/forecasting/data/summary_v0.2"
ticker = "AMD"
save_dir = os.path.join(summary_path, ticker)
os.makedirs(save_dir, exist_ok=True)

model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
# model_name = "casperhansen/llama-3-70b-instruct-awq"
tokenizer = AutoTokenizer.from_pretrained(model_name)

paths = sorted(
    glob(f"/data/kai/forecasting/data/document_v0.2/{ticker}/*.csv"))
prompts = Prompts(ticker)
guided_json = json.load(open(
    "/data/kai/forecasting/multimodal/financial/templates/guided_json_summary_v0.3.json", 'r'))


with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(main, path) for path in paths]
    results = [future.result() for future in as_completed(futures)]
