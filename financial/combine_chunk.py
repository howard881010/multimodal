import pandas as pd
from glob import glob
import json
from collections import defaultdict
import ast
from src.vllm import llm_chat, message_template, call_llm_chat
from templates.PROMPTS import Prompts
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread
# check size of summary value
from transformers import AutoTokenizer
import time

import numpy as np

# parse raw JSON


def collapse_metrics(json_summaries):
    if type(json_summaries) == np.ndarray:
        data = [{k: v for k, v in ast.literal_eval(
            summary).items() if v} for summary in json_summaries]
    elif type(json_summaries) == list:
        data = [{k: v for k, v in summary.items() if v}
                for summary in json_summaries]
    elif type(json_summaries) == str:
        data = [{k: v for k, v in ast.literal_eval(json_summaries).items()}]
    collapsed_data = []

    for entry in data:
        new_entry = {}
        for key, value in entry.items():
            if isinstance(value, list):
                new_value = []
                for item in value:
                    if isinstance(item, dict):
                        if 'metric' in item and 'value' in item:
                            new_value.append(
                                f"{item['metric']}: {item['value']}")
                        elif 'key' in item and 'value' in item:
                            new_value.append(f"{item['key']}: {item['value']}")
                    else:
                        new_value.append(item)
                new_entry[key] = new_value
            else:
                new_entry[key] = value
        if len(new_entry) != 0:
            collapsed_data.append(new_entry)

    return collapsed_data

# chunk JSON summary in a list


def chunk_summaries(tokenizer, summaries: list[dict], max_split_token=4096):
    """
    Chunks summaries into summaries so that the key, item pair does not get cut off.
    """
    chunks = []
    for summary in summaries:
        for k, val in summary.items():
            new_chunk = f"{k} [{val}]" if k == "summary" else f"{k} {val}"

            if not chunks or len(tokenizer.encode(chunks[-1] + ", " + new_chunk)) > max_split_token:
                chunks.append(new_chunk)
            else:
                chunks[-1] += ", " + new_chunk
    return chunks


def collapse_results(results):
    result = []
    for r in results:
        result += r
    return result


def combine_results(results):
    results_dict = defaultdict(list)

    for result in results:
        if type(result) == str:
            json_data = ast.literal_eval(result)
        else:
            json_data = result
        for key in json_data:
            # For lists of dictionaries (like 'key_numbers')
            if isinstance(json_data[key], list):
                results_dict[key].extend(json_data[key])
            else:
                results_dict[key].append(json_data[key])
    return results_dict


def get_current_summary_length(results):
    return len(combine_results(results)["summary"])


def combine_chunk(chunk):
    output = llm_chat(message_template(
        prompts.COMBINE_JSON_PROMPT, chunk), model=model_name, guided_json=guided_json)
    cleaned_output = collapse_metrics(output)
    return cleaned_output


def llm_combine_summaries(summaries):
    summary_chunks = chunk_summaries(tokenizer, summaries)
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(combine_chunk, chunk)
                   for chunk in summary_chunks]
        results = [future.result() for future in as_completed(futures)]
    return results


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


for path in paths:
    timestamp = path.split('/')[-1].split('.csv')[0]
    save_path = os.path.join(save_dir, timestamp + ".json")

    if os.path.exists(save_path):
        continue

    json_summaries = pd.read_csv(
        "/data/kai/forecasting/data/document_v0.2/AMD/2022-05-04.csv")['summary'].values
    # json_summaries = pd.read_csv("/data/kai/forecasting/data/document_v0.2/AMD/2022-03-04.csv")['summary'].values
    cleaned_results = collapse_metrics(json_summaries)  # list[{k: v}, {k: {k:v}}]

    print("First combination task...")
    json_results = llm_combine_summaries(cleaned_results)
    formatted_results = collapse_metrics(collapse_results(
        json_results))  # this is like cleaned_results


    def get_current_summary_length(results):
        return len(combine_results(results)["summary"])


    combine_count = 0
    while get_current_summary_length(formatted_results) > 1:
        combine_count += 1
        print(
            f"Combining summaries {combine_count} ... length summaries: {get_current_summary_length(formatted_results)}")
        json_results = llm_combine_summaries(formatted_results)
        formatted_results = collapse_metrics(collapse_results(
            json_results))  # this is like cleaned_results

    with open(save_path, 'w') as json_file:
        json.dump(json_results, json_file, indent=4)
    