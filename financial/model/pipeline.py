import numpy as np
import pandas as pd
import requests
import os
from openai import OpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm
import transformers
import torch
from transformers import AutoTokenizer, TextStreamer, AutoModelForCausalLM
from IPython.display import clear_output

# load_dotenv()
# client = OpenAI()

def get_summary_gpt(ticker, news):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant that filters and summarizes stock news for the specific company {ticker}. Filter out irrelevant infromation and provide a concise summary including key numbers, growth trends, and the overall market outlook. Ensure to mention major stock movements, significant economic indicators, and any notable company-specific news."},
            {"role": "user", "content": news}
        ]
    )
    cost = response.usage.prompt_tokens * 0.5 / 1e6 + response.usage.completion_tokens * 1.5 / 1e6
    return response, cost

    # summary, cost = get_summary("AAPL", parsed_text)
    # total_cost += cost
    # summary_text = summary.choices[0].message.content
    # print(summary_text)
    # print(f"Total cost: {total_cost}")

class FinancePipeline:

    def __init__(self, data_dir):
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.streamer = TextStreamer(tokenizer=self.tokenizer, skip_prompt=True)
        # streamer = None

        self.model = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            streamer=self.streamer,
            max_length=8000,
        )

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.data_dir = data_dir

    def set_prompts_for_ticker(self, ticker):
        self.summary_ending_prompt = f" Filter out irrelevant information and provide a concise summary including key numbers, growth trends, and the overall market outlook. Ensure to mention major stock movements, significant economic indicators, and any notable company-specific news."
        self.filter_prompt = "Keep the query the same, but please avoid any extraneous phrases or commentary such as 'Here is the filtered text' or 'I hope this helps.'"
        self.summary_prompt = f"You are a helpful assistant that filters and summarizes stock news specifically for company with ticker symbol {ticker}."
        self.combine_prompt = self.summary_prompt + " Combine the following summaries while preserving as much information as you can: "

    def llama_batch(self, messages):
        prompt_templated = self.tokenizer.apply_chat_template(messages, tokenize=False)
        model_inputs = self.tokenizer(prompt_templated, return_tensors="pt", padding="longest").to("cuda")

        generate_ids = self.model.generate(
            model_inputs.input_ids, eos_token_id=self.terminators, attention_mask=model_inputs.attention_mask)

        output = self.tokenizer.batch_decode(generate_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # currently not sure why "assistant\n\n" is showing up as part of output
        # also when batch decoding, null happens
        # output = [o.split("assistant\n\n")[1] for o in output]
        return output

    def run_llama(self, prompts):
        """
        With pipeline

        prompts: list[str] - list of prompts alternating between system and user

        output: str - generated text
        """
        messages = [{"role": "system", "content": prompts[p]} if p % 2 == 0 else {"role": "user", "content": prompts[p]} for p in range(len(prompts))]

        outputs = self.model(
            messages,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        return outputs[0]["generated_text"][-1]["content"].replace("<|eot_id|>", "")


    def combine_summaries(self, summaries):
        """
        Combines all summaries and chunk them into 8192*0.5 tokens

        Outputs list of chunk summaries.
        """
        all_summaries = " ".join(summaries)

        total_tokens = sum([len(self.tokenizer.tokenize(s)) for s in summaries])
        text_to_token_ratio = len(all_summaries) / total_tokens

        chunk_size = int(8192*0.5 * text_to_token_ratio)
        combined_summaries = []
        # print(chunk_size)

        for i in range(0, len(all_summaries), chunk_size):
            clear_output(wait=True)
            print(f"Running {i} / {len(all_summaries)}, text_length = {chunk_size} text_token={len(self.tokenizer.encode(all_summaries[i:i+chunk_size]))}")

            out = self.run_llama([self.combine_prompt, all_summaries[i:i+chunk_size]])
            combined_summaries.append(out)

        return combined_summaries

    def get_summaries(self, processed_texts):
        summaries = []
        assert self.ticker is not None

        for i, text in enumerate(tqdm(processed_texts, total=len(processed_texts))):
            clear_output(wait=True)
            print(f"Running {i} / {len(processed_texts)}")
            out = self.run_llama([self.summary_prompt, text + self.summary_ending_prompt])
            summaries.append(out)
        return summaries


    def save_summaries(self, summaries, file_path):
        with open(file_path, 'w') as file:
            if type(summaries) == str:
                file.write(summaries)
            elif type(summaries) == list:
                for line in summaries:
                    file.write(line + "\n")
                    file.write("<SEP>" + "\n")

    def load_summaries(self, file_path, raw=True):
        with open(file_path, 'r') as file:
            data = file.read().splitlines()
        return data
