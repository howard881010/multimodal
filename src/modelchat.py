from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM, set_seed
import numpy as np
import torch
from torch.cuda.amp import autocast
import time
from peft import PeftModel
import os


class ChatModel:
    def __init__(self, model_name, token, dataset):
        self.model_name = model_name
        self.token = token
        self.tokenizer = self.load_tokenizer(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset = dataset

    def load_model(self, model_name):
        raise NotImplementedError("Subclasses must implement this method")

    def load_tokenizer(self, model_name):
        raise NotImplementedError("Subclasses must implement this method")

    def chat(self, prompt):
        raise NotImplementedError("Subclasses must implement this method")


class MistralChatModel(ChatModel):
    def __init__(self, model_name, token, dataset):
        super().__init__(model_name, token, dataset)
        self.model = self.load_model(model_name, token, dataset)
        self.tokenizer = self.load_tokenizer(model_name)
        self.device = next(self.model.parameters()).device

    def load_model(self, model_name, token, dataset):
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, token=token, device_map="auto")
        # return base_model
        if dataset == "climate":
            return PeftModel.from_pretrained(base_model, "Rose-STL-Lab/climate-cal")
        elif dataset == "gas":
            return PeftModel.from_pretrained(base_model, "Rose-STL-Lab/gas-west")
        elif dataset == "medical":
            return PeftModel.from_pretrained(base_model, "Howard881010/medical-openai")
        # return PeftModel.from_pretrained(base_model, "Rose-STL-Lab/gas-mixed-mixed-fact")

    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name, device_map="auto")
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
    def __init__(self, model_name, token, dataset):
        super().__init__(model_name, token, dataset)
        self.model = self.load_model(model_name, token, dataset)
        self.tokenizer = self.load_tokenizer(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = next(self.model.parameters()).device

    def load_model(self, model_name, token, dataset):
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, token=token, device_map="auto")
        # return base_model
        if dataset == "climate":
            return PeftModel.from_pretrained(base_model, "Howard881010/climate-1_day")
        elif dataset == "gas":
            return PeftModel.from_pretrained(base_model, "Rose-STL-Lab/gas-west")
        elif dataset == "medical":
            return PeftModel.from_pretrained(base_model, "Rose-STL-Lab/medical")
    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name, device_map="auto", padding_side="left")
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
        print("before generate")
        # Generate text using the model
        with torch.no_grad():
            generate_ids = self.model.generate(
                model_inputs.input_ids, max_new_tokens=2048, eos_token_id=terminators, attention_mask=model_inputs.attention_mask)

        output = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True)

        return output

if __name__ == "__main__":
    np.random.seed(42)
    set_seed(42)
    start = time.time()

    input = str({"temp": 169.9})
    content = [{"role": "system", "content": "Given the temp for the current day, please predict the temp in json format for next day, the example output is \{\"temp\": 169.9\}."}, {"role": "user", "content": input}]
    prompt = [content]
    prompt.append(content)
    token = os.getenv("HF_TOKEN")

    # model_chat = MistralChatModel(
    #     "mistralai/Mistral-7B-Instruct-v0.1", token, "climate")
    # fine-tuned model
    model_chat = LLMChatModel("meta-llama/Meta-Llama-3.1-8B-Instruct", token=token, dataset="climate")

    output = model_chat.chat(prompt)
    # print(output)
    for i in range(len(output)):
        print("Out of ", i, ": ", output[0])

    end = time.time()
    print("Time taken: ", end - start)
