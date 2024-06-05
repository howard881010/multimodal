from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from datasets import Dataset
from utils import create_batched
import pandas as pd


def mistral7b_chat(prompt, chat_template=None):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1", token="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", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1", token="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")
    tokenizer.pad_token = tokenizer.eos_token
    if chat_template != None:
        tokenizer.apply_chat_template(chat_template, tokenize=False)

    # Tokenize input and ensure it is on the right device
    model_inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    # Get the device of the model
    model_device = next(model.parameters()).device
    model_inputs = model_inputs.to(model_device)

    # Generate text using the model
    generated_ids = model.generate(
        **model_inputs, max_new_tokens=100)

    # Decode generated ids to text
    output_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True)

    return output_text


if __name__ == "__main__":
    # Constructing the prompt with an example
    data = pd.read_csv("Data/Yelp/4_weeks/test_0.csv")
    data["fut_values"] = data["fut_values"].apply(str)
    data["hist_values"] = data["hist_values"].apply(str)

    batches = list(create_batched(data, 8))
    # print(batches[0])
    for batch in batches:
        prompt = []
        for index, row in batches[0].iterrows():
            row_content = row['hist_values']
            historical_window_size = 4
            content = f"""For the given {historical_window_size} historical values: {row_content}, predict the future {historical_window_size} values without producing any additional text"""
            prompt.append(content)
            output_text = mistral7b_chat(prompt)
