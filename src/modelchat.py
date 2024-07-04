from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM, set_seed
import pandas as pd
from utils import create_batched
import numpy as np
import torch
from torch.cuda.amp import autocast
import time
from num2words import num2words
from transformers import AutoConfig
from torch.nn import DataParallel
import xformers
from peft import PeftModel


class ChatModel:
    def __init__(self, model_name, token):
        self.model_name = model_name
        self.token = token
        self.tokenizer = self.load_tokenizer(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self, model_name):
        raise NotImplementedError("Subclasses must implement this method")

    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name, device_map="auto")

    def apply_chat_template(self, template):
        if template is not None:
            self.tokenizer.apply_chat_template(template, tokenize=False)

    def chat(self, prompt):
        raise NotImplementedError("Subclasses must implement this method")


class MistralChatModel(ChatModel):
    def __init__(self, model_name, token):
        super().__init__(model_name, token)
        self.model = self.load_model(model_name, token)
        self.device = next(self.model.parameters()).device

    def load_model(self, model_name, token):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, token=token, device_map="auto")
    
        return model

    def chat(self, prompt):
        new_prompt = self.tokenizer.apply_chat_template(
            prompt, tokenize=False)

        model_inputs = self.tokenizer(
            new_prompt, return_tensors="pt", padding=True)
        # # Get the device of the model
        model_inputs = model_inputs.to(self.device)

        # Generate text using the model
        with autocast():
            generated_ids = self.model.generate(
                **model_inputs, max_new_tokens=100)

        # Decode generated ids to text
        output_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)

        return output_text


class LLMChatModel(ChatModel):
    def __init__(self, model_name, token):
        super().__init__(model_name, token)
        self.model = self.load_model(model_name, token)
        self.device = next(self.model.parameters()).device

    def load_model(self, model_name, token):
        base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=token)
        return base_model
        # return PeftModel.from_pretrained(base_model, "Rose-STL-Lab/finance-mixed-numerical")

    def chat(self, prompt):
        new_prompt = self.tokenizer.apply_chat_template(
            prompt, tokenize=False)
        
        model_inputs = self.tokenizer(
            new_prompt, return_tensors="pt", padding="longest")
        # Get the device of the model
        model_inputs = model_inputs.to(self.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Generate text using the model
        generate_ids = self.model.generate(
            model_inputs.input_ids, max_new_tokens=150, eos_token_id=terminators, attention_mask=model_inputs.attention_mask)

        output = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True)

        return output


class GemmaChatModel(ChatModel):
    def __init__(self, model_name, token):
        super().__init__(model_name, token)
        self.model = self.load_model(model_name, token)
        self.device = next(self.model.parameters()).device
        # self.config = AutoConfig.from_pretrained(model_name)
        # self.config.hidden_act = "gelu_torch_tanh"

    def load_model(self, model_name, token):
        return AutoModelForCausalLM.from_pretrained(model_name, token=token, device_map="auto", torch_dtype=torch.bfloat16)

    def chat(self, prompt):
        new_prompt = self.tokenizer.apply_chat_template(
            prompt, tokenize=False)
        model_inputs = self.tokenizer(
            new_prompt, return_tensors="pt", padding=True)
        # Get the device of the model
        model_inputs = model_inputs.to(self.device)

        # Generate text using the model
        with autocast():
            generated_ids = self.model.generate(
                **model_inputs, max_new_tokens=150)

        # Decode generated ids to text
        output_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)

        return output_text


if __name__ == "__main__":
    np.random.seed(42)
    set_seed(42)
    # Constructing the prompt with an example
    # data = pd.read_csv("Data/Yelp/4_weeks/test_0.csv")
    # data["fut_values"] = data["fut_values"].apply(str)
    # data["hist_values"] = data["hist_values"].apply(str)
    start = time.time()

    content = [{"role": "user",
            "content": f"""The financial summaries for the last trading day are day 1: Amazon.com Inc. (AMZN) stock rises 1.5% to $3,234.00, reaching a new all-time high, with a market capitalization of $1.63 trillion, making it the second-largest publicly traded US company. In the past 12 months, AMZN shares have gained 73%, outperforming the S&P 500's 41% increase. Earnings per share (EPS) for Q3 2021 are expected to be $7.16, a 44.6% increase from Q3 2020. The e-commerce giant has been benefiting from the ongoing shift to online shopping, with its cloud computing division, Amazon Web Services (AWS), driving growth. However, the company's retail dominance faces potential competition from TikTok's e-commerce play, which may impact its market share., and the stock prices of the last 1 days are day 1: 148.84
            Given the financial summaries and stock prices from the last trading day, generate financial summaries for the current trading day"""}]
    prompt = [content]
    

    # model_chat = MistralChatModel(
    #     "mistralai/Mistral-7B-Instruct-v0.1")
    # fine-tuned model
    model_chat = LLMChatModel("meta-llama/Llama-2-7b-chat-hf")
    # model_chat = GemmaChatModel("google/gemma-7b-it")

    # model_chat.apply_chat_template(chat)

    output = model_chat.chat(prompt)
    # print(output)
    for i in range(len(output)):
        print("Out of ", i, ": ", output[0].split("\n")[-1])

    end = time.time()
    print("Time taken: ", end - start)
