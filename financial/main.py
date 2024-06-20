# first spin up the vLLM server. takes a while

# export CUDA_VISIBLE_DEVICES='0,1'
# python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B-Instruct --tensor-parallel-size=2 --disable-log-requests

# offline inference
# from vllm import LLM, SamplingParams
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=8000)
# llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", gpu_memory_utilization=0.9)

# from transformers import AutoTokenizer
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.eos_token

# # format to role, content format
# messages_dicts = [[{"role": "user", 'content': p}] for p in prompts]
# formatted_message = tokenizer.apply_chat_template(messages_dicts, tokenize=False, add_generation_prompt=True)
# outputs = llm.generate(formatted_message, sampling_params)


# online
from transformers import AutoTokenizer
from src.PROMPTS import Prompts
from src.vllm import batch_call_llm_chat, llm_chat, message_template
from src.utils import load_text_from, log_time
import time
from src.utils import get_logger
import logging

if __name__ == "__main__":
    ticker = "aapl"
    prompts = Prompts(ticker)
    logger = get_logger(f"logs/{ticker}_summary.txt")

    raw_data = load_text_from(
        "/data/kai/forecasting/summary/aapl_2022-08-19_raw.txt")
    raw_data = [d for d in raw_data if d != "<SEP>" and d != ""]

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    script_start_time = time.time()

    # summarize and ignore error messages
    start_time = time.time()
    summaries = batch_call_llm_chat(prompts.SUMMARY_PROMPT, raw_data)
    logging.info("SUMMARIZING PROMPT")
    log_time(start_time)

    start_time = time.time()
    result = batch_call_llm_chat(prompts.IGNORE_PROMPT, summaries)
    logging.info("IGNORE PROMPT")
    log_time(start_time)

    valid_idxs = [i for i, r in enumerate(result) if r != "<NONE>"]
    valid_summaries = [summaries[i] for i in valid_idxs]

    token_length = len(tokenizer.encode('\n'.join(valid_summaries)))
    max_token_length = 4096

    start_time = time.time()
    if token_length > max_token_length:
        logging.info(f"token_length: {token_length}")
        # Determine how many summaries to combine per chunk
        avg_token_per_summary = token_length / len(valid_summaries)
        summaries_per_chunk = int(max_token_length / avg_token_per_summary)
        logging.info(f"summaries_per_chunk: {summaries_per_chunk}")
        # Split the summaries into chunks
        valid_summaries_combined = [
            valid_summaries[i: i + summaries_per_chunk]
            for i in range(0, len(valid_summaries), summaries_per_chunk)
        ]

        combined_summary = batch_call_llm_chat(
            prompts.COMBINE_PROMPT, valid_summaries_combined)
        valid_summaries_combined += combined_summary
    else:
        valid_summaries_combined = valid_summaries

    logging.info("COMBINE PROMPT")
    log_time(start_time)

    start_time = time.time()
    final_summary = llm_chat(message_template(
        prompts.FINAL_PROMPT, '\n'.join(valid_summaries_combined)))
    logging.info("FINAL PROMPT")
    log_time(start_time)

    logging.info("------------------------")
    logging.info(final_summary)

    log_time(script_start_time)
