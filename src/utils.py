import re
import numpy as np
import os
import yaml
from transformers import AutoTokenizer


    
def open_record_directory(dataset, unit, filename, model_name, sub_dir, window_size):

    out_filename = model_name + "_output_" + filename.split("/")[-1] + ".csv"
    log_filename = model_name + "_log_" + filename.split("/")[-1] + ".csv"
    os.makedirs(f"Logs/{dataset}/{window_size}_{unit}/{sub_dir}", exist_ok=True)
    os.makedirs(f"Predictions_and_attempts/{dataset}/{window_size}_{unit}/{sub_dir}", exist_ok=True)
    log_path = f"Logs/{dataset}/{window_size}_{unit}/{sub_dir}/{log_filename}"
    res_path = f"Predictions_and_attempts/{dataset}/{window_size}_{unit}/{sub_dir}/{out_filename}"

    return log_path, res_path


def open_result_directory(dataset, sub_dir, unit, filename, model_name, window_size):
    out_filename = model_name + "_rmse_" + \
        "_".join((filename.split("/")[-1].split("_"))[2:])

    os.makedirs(f"Results/{dataset}/{window_size}_{unit}/{sub_dir}", exist_ok=True)
    out_path = f"Results/{dataset}/{window_size}_{unit}/{sub_dir}/{out_filename}"

    return out_path


def normalize_together(pred, truth):
    combined = np.concatenate((pred, truth))
    min_val = np.min(combined)
    max_val = np.max(combined)
    print(min_val, max_val)
    pred_norm = (pred - min_val) / (max_val - min_val)
    truth_norm = (truth - min_val) / (max_val - min_val)
    return pred_norm, truth_norm


def rmse(y_pred, y_true):
    y_pred = np.reshape(y_pred, -1)
    y_true = np.reshape(y_true, -1)
    # pred_norm, ground_truth_norm = normalize_together(y_pred, y_true)
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

def create_batched(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]

def create_result_file(dir, filename):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir + "/" + filename

def find_text_parts(text, num_pattern):
    modified_text = re.sub(num_pattern, "\r", text)
    
    return modified_text

def find_num_parts(text, num_pattern, window_size):
    num_matches = re.findall(num_pattern, text)
    if len(num_matches) != window_size:
        return np.nan
    else:
        return  [[float(temp)] for temp in num_matches]

def split_text(text, text_pattern):
    text_matches = re.findall(text_pattern, text, re.DOTALL)
    cleaned_matches = [match.replace('\n', ' ').replace('\r', '').replace('```', '').strip() for match in text_matches]
    # print(cleaned_matches)

    return cleaned_matches

def load_config(yaml_path):
    with open(yaml_path, 'r') as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return cfg

def apply_chat_template(tokenizer, instruction, input_text, output_text):
    prompt = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text}
    ]
    new_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
    input_tokens = tokenizer(new_prompt, return_tensors="pt")
    return input_tokens.input_ids.shape[1]

def get_max_token_size(dataset, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    max_tokens = 0
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    for split in ['train', 'valid', 'test']:
        tokens_split = max([apply_chat_template(tokenizer, row['instruction'], 
                                        row['input_text'], row['output_text']) for row in dataset[split]])
        if tokens_split > max_tokens:
            max_tokens = tokens_split

    return max_tokens + 10 # add 10 extra tokens for safety