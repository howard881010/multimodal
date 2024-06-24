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
    def __init__(self, model_name):
        self.model_name = model_name
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
    def __init__(self, model_name):
        super().__init__(model_name)
        self.model = self.load_model(model_name)
        self.device = next(self.model.parameters()).device

    def load_model(self, model_name):
        base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        return PeftModel.from_pretrained(base_model, "Howard881010/finance-summary")

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
            "content": f"""The financial summaries for the last 1 days are day 1: Here is a comprehensive summary of Tesla's stock (TSLA): **Stock Performance:** Tesla's stock is up 50% year-to-date, outperforming the S&P 500's 15% gain. **Earnings Report:** Q2 2022 earnings report showed revenue of $16.93 billion, beating estimates of $16.54 billion, with net income of $2.27 billion and a gross margin of 25%. **Vehicle Deliveries:** Tesla's vehicle deliveries increased 27% year-over-year to 254,695 in Q2, driven by growth in China and Europe. The company aims to increase production capacity at its factories in Fremont, California, and Shanghai. **Market Outlook:** Analysts expect Tesla's revenue to grow 34% in 2022 and 24% in 2023. The company's market capitalization is around $900 billion, making it one of the largest companies in the US. **Notable News:** The opening of Tesla's Giga Berlin factory could potentially boost its market share in Europe, where the electric vehicle market is growing rapidly. The factory has a production capacity of 500,000 units per year and is expected to increase Tesla's presence in the region., and the stock prices of the last 1 days are day 1: 921.16
            .Given the financial summaries from the last 1 days, generate financial summaries for the current day and the next 5 days"""}]
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
