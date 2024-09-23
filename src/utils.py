import re
import numpy as np
import os
import yaml
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
from pydantic import BaseModel, create_model
import json


    
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

def find_num_parts(text, num_key_name, window_size):
    try:
        text = json.loads(text)
        # Sort the dictionary by keys
        text = dict(sorted(text.items()))
        temps = []
        for key, value in text.items():
            if key.endswith(num_key_name):
                if value is not None:
                    temps.append(value)
        while len(temps)!= window_size:
            if len(temps) < window_size:
                temps.append(np.nan)
            else:
                temps = temps[:window_size]
            
        return  [[float(temp)] for temp in temps]
    except:
        temps = re.findall(fr'"day_\d+_{num_key_name}":\s*(\d+\.?\d*)', text, re.DOTALL)
# Convert the found values to float
        while len(temps)!= window_size:
            if len(temps) < window_size:
                temps.append(np.nan)
            else:
                temps = temps[:window_size]
        return [[float(temp)] for temp in temps]

def split_text(text, text_key_name, window_size):
    day_counter = window_size + 1
    try:
        data = json.loads(text)
        result = []
        while True:
            date_key = f"day_{day_counter}_date"
            forecast_key = f"day_{day_counter}_{text_key_name}"
            if date_key not in data or forecast_key not in data:
                break
            day_info = f"""date: {data[date_key]}, weather_forecast: {data[forecast_key]}"""
            result.append(day_info)

            day_counter += 1
        
        return result
    except:
        matches = re.findall(fr'("day_\d+_date":\s*"[^"]+").*?("day_\d+_{text_key_name}":\s*"[^"]+")', text, re.DOTALL)
        if len(matches) == 0:
            return ["No prediction" for _ in range(window_size)] 
        else:
            return [f"{date}, {forecast}" for date, forecast in matches]
        


def load_config(yaml_path):
    with open(yaml_path, 'r') as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return cfg

def apply_chat_template(tokenizer, instruction, input_text, output_text):
    alpaca_text = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output_text}"
    input_tokens = tokenizer(alpaca_text, return_tensors="pt")
    return input_tokens.input_ids.shape[1]

def get_max_token_size(dataset, input_column, output_column, instruction_column, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", ):
    max_tokens = 0
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for split in ['train', 'valid', 'test']:
        tokens_split = max([apply_chat_template(tokenizer, row[instruction_column], 
                                        row[input_column], row[output_column]) for row in dataset[split]])
        if tokens_split > max_tokens:
            max_tokens = tokens_split

    return max_tokens + 10 # add 10 extra tokens for safety

def uploadToHuf(results, hf_dataset, split, case):
    data_all = load_dataset(hf_dataset)
    data = pd.DataFrame(data_all[split])
    pred_output_column = f'pred_output_case{case}'
    data[pred_output_column] = results['pred_output']
    updated_data = Dataset.from_pandas(data)
    if split == 'validation':
        updated_dataset = DatasetDict({
            'train': data_all['train'], 
            'test': data_all['test'],
            'valid': updated_data
        })
    elif split == 'test':
        updated_dataset = DatasetDict({
            'train': data_all['train'], 
            'valid': data_all['valid'],
            'test': updated_data
        })
    updated_dataset.push_to_hub(hf_dataset)


def logToWandb(wandb, meteor_score, cos_sim_score, rouge1, rouge2, rougeL, rmse_loss, drop_rate, text_drop_count, std_error):
    # if not np.isnan(meteor_score):
    #     wandb.log({"Meteor Scores": meteor_score})
    # if not np.isnan(cos_sim_score):
    #     wandb.log({"Cos Sim Scores": cos_sim_score})
    # if not np.isnan(rouge1):
    #     wandb.log({"Rouge1 Scores": rouge1})
    # if not np.isnan(rouge2):
    #     wandb.log({"Rouge2 Scores": rouge2})
    # if not np.isnan(rougeL):
    #     wandb.log({"RougeL Scores": rougeL})
    # if not np.isnan(rmse_loss):
    #     wandb.log({"RMSE Scores": rmse_loss})
    # if not np.isnan(drop_rate):
    #     wandb.log({"Drop Rate": f"{drop_rate*100:.2f}%"})
    # if not np.isnan(text_drop_count):
    #     wandb.log({"Text Drop Count": text_drop_count})
    if not np.isnan(std_error):
        wandb.log({"Standard Error": std_error})
    
    wandb.finish()

    

