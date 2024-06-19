# first spin up the vLLM server. takes a while

# export CUDA_VISIBLE_DEVICES='0,1'
# python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B-Instruct --tensor-parallel-size=2 --disable-log-requests


from transformers import AutoTokenizer
from PROMPTS import Prompts
from llm import batch_call_llm_chat, llm_chat, message_template
import time
from datetime import datetime, timedelta

def print_time(start_time):
    start_datetime = datetime.fromtimestamp(start_time)
    end_datetime = datetime.fromtimestamp(time.time())

    print(f"Start time: {start_datetime}")
    print(f"End time: {end_datetime}")

def load_text_from(file_path):
    with open(file_path, 'r') as file:
        data = file.read().splitlines()
    return data


ticker = "aapl"
prompts = Prompts(ticker)

raw_data = load_text_from("/data/kai/forecasting/summary/aapl_2022-08-19_raw.txt")
raw_data = [d for d in raw_data if d != "<SEP>" and d != ""]


model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# summarize and ignore error messages
start_time = time.time()
summaries = batch_call_llm_chat(prompts.SUMMARY_PROMPT, raw_data)
print("SUMMARIZING PROMPT")
print_time(start_time)

start_time = time.time()
result = batch_call_llm_chat(prompts.IGNORE_PROMPT, summaries)
print("IGNORE PROMPT")
print_time(start_time)

valid_idxs = [i for i, r in enumerate(result) if r != "<NONE>"]
valid_summaries = [summaries[i] for i in valid_idxs]


token_length = len(tokenizer.encode('\n'.join(valid_summaries)))
max_token_length = 4096

start_time = time.time()
if token_length > max_token_length:
    # Determine how many summaries to combine per chunk
    avg_token_per_summary = token_length / len(valid_summaries)
    summaries_per_chunk = int(max_token_length / avg_token_per_summary)

    # Split the summaries into chunks
    valid_summaries_combined = [
        valid_summaries[i: i + summaries_per_chunk]
        for i in range(0, len(valid_summaries), summaries_per_chunk)
    ]

    combined_summary = batch_call_llm_chat(prompts.COMBINE_PROMPT, valid_summaries_combined)
    valid_summaries_combined.append(combined_summary)
else:
    valid_summaries_combined = valid_summaries

print("COMBINE PROMPT")
print_time(start_time)

start_time = time.time()
final_summary = llm_chat(message_template(prompts.FINAL_PROMPT, '\n'.join(valid_summaries_combined)))
print("FINAL PROMPT")
print_time(start_time)

print(final_summary)