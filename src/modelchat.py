from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM, set_seed
import pandas as pd
import numpy as np
import torch
from torch.cuda.amp import autocast
import time
from peft import PeftModel
import os


class ChatModel:
    def __init__(self, model_name, token):
        self.model_name = model_name
        self.token = token
        self.tokenizer = self.load_tokenizer(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self, model_name):
        raise NotImplementedError("Subclasses must implement this method")

    def load_tokenizer(self, model_name):
        # raise NotImplementedError("Subclasses must implement this method")
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
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, token=token, device_map="auto")
        # return base_model
        return PeftModel.from_pretrained(base_model, "Rose-STL-Lab/climate")
        # return PeftModel.from_pretrained(base_model, "Rose-STL-Lab/gas-mixed-mixed-fact")

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
                **model_inputs, max_new_tokens=2048)

        # Decode generated ids to text
        output_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)

        return output_text


class LLMChatModel(ChatModel):
    def __init__(self, model_name, token):
        super().__init__(model_name, token)
        self.model = self.load_model(model_name, token)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = next(self.model.parameters()).device

    def load_model(self, model_name, token):
        base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=token)
        return base_model
        # return PeftModel.from_pretrained(base_model, "Rose-STL-Lab/gas-1_day-text-text")

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
            model_inputs.input_ids, max_new_tokens=1000, eos_token_id=terminators, attention_mask=model_inputs.attention_mask)

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

    input = str({"share_price": 169.9})
    content = [{"role": "system", "content": "Given the share price for the current day, please predict the shared price in json format for next day"}, {"role": "user", "content": input}]
    prompt = [content]
    token = os.getenv("HF_TOKEN")

    # model_chat = MistralChatModel(
    #     "mistralai/Mistral-7B-Instruct-v0.1")
    # fine-tuned model
    model_chat = LLMChatModel("meta-llama/Llama-2-7b-chat-hf", token=token)
    # model_chat = GemmaChatModel("google/gemma-7b-it")

    # model_chat.apply_chat_template(chat)

    output = model_chat.chat(prompt)
    # print(output)
    for i in range(len(output)):
        print("Out of ", i, ": ", output[0])

    end = time.time()
    print("Time taken: ", end - start)
