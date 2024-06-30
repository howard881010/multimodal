# first spin up the vLLM server. takes a while

# export CUDA_VISIBLE_DEVICES='0,1'
# python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B-Instruct --tensor-parallel-size=2 --disable-log-requests
# python -m outlines.serve.serve --model meta-llama/Meta-Llama-3-70B-Instruct --tensor-parallel-size=2 --disable-log-requests

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
from multimodal.financial.templates.PROMPTS import Prompts
from src.vllm import batch_call_llm_chat, llm_chat, message_template
from src.utils import load_text_from, log_time, save_text_to, get_logger
import logging
import time
import os
import argparse
from tqdm import tqdm

null_phrases = [
    "no relevant information",
    "no stock summaries provided",
    "since there is no",
    "no available information about",
    "no summary was provided",
    "no summaries provided"
    "not available due to lack of information",
    "please provide relevant information"
]

early_null_phrases = [
    "unfortunately",
    "no availabe information",
    "relevant",
]


def is_valid_summary(current_summary: str):
    return all([phrase not in current_summary.lower()[:150] for phrase in early_null_phrases]) and \
        all([phrase not in current_summary.lower() for phrase in null_phrases])


def chunk_documents(tokenizer, data, max_token_length=4096):
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
        token_length = len(tokenizer.encode(data))
        avg_token_per_summary = token_length / len(data)
        document_per_chunk = int(max_token_length / avg_token_per_summary)
        chunk_data = [data[i: i+document_per_chunk]
                      for i in range(0, len(data), document_per_chunk)]
        return chunk_data


def main(dir_path):
    model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    file_paths = sorted(os.listdir(dir_path))
    file_paths = [f for f in file_paths if not os.path.exists(
        os.path.join(dir_path, f.replace("raw", "summary")))]
    for path in tqdm(file_paths, total=len(file_paths)):
        script_start_time = time.time()

        chunk_path = os.path.join(
            dir_path, path.replace("raw", "chunk_summary"))
        final_path = os.path.join(
            dir_path, path.replace("raw", "final_summary"))
        print(f"Running {final_path}")
        logging.info("------------------------")
        logging.info(final_path)

        raw_data = load_text_from(os.path.join(dir_path, path))
        raw_data = [d for d in raw_data if d != "<SEP>" and d != ""]
        logging.info(f'{len(raw_data)} urls found.')

        chunk_data = chunk_documents(tokenizer, raw_data)

        summaries = batch_call_llm_chat(prompts.SUMMARY_PROMPT, chunk_data)
        save_text_to('\n'.join(summaries), chunk_path)
        # valid_summaries = batch_call_llm_chat(prompts.IGNORE_PROMPT, summaries)

        # valid_idxs = [i for i, r in enumerate(
        #     valid_summaries) if r != "<NONE>"]
        # valid_summaries = [summaries[i] for i in valid_idxs]

        if len(summaries) > 0:
            combined_summaries = chunk_documents(
                tokenizer, "\n".join(summaries))
            if len(combined_summaries) > 1:
                combined_summaries = batch_call_llm_chat(
                    prompts.COMBINE_PROMPT, combined_summaries)

            chunked_summaries = chunk_documents(tokenizer, combined_summaries)
            if len(chunked_summaries) > 1:
                chunked_summaries = batch_call_llm_chat(
                    prompts.COMBINE_PROMPT, chunked_summaries)

            final_summary = llm_chat(message_template(
                prompts.FINAL_PROMPT, '\n'.join(chunked_summaries)))

            if is_valid_summary(final_summary):
                save_text_to(final_summary, final_path)

            logging.info("------------------------")
            logging.info(final_summary)

            log_time(script_start_time)
        else:
            save_text_to("", final_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process stock ticker information.")
    parser.add_argument('--ticker', type=str, required=True,
                        help='Stock ticker symbol')
    args = parser.parse_args()

    ticker = args.ticker
    # ticker = "AMD"

    prompts = Prompts(ticker)
    logger = get_logger(f"logs/{ticker}_summary.txt")

    dir_path = f"/data/kai/forecasting/summary/{ticker}"
    main(dir_path)
