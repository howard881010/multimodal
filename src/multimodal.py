import concurrent.futures
import time
import sys
import os
import pandas as pd
import numpy as np
import wandb
from loguru import logger
from utils import find_text_parts, find_num_parts, split_text
from modelchat import LLMChatModel
from transformers import set_seed
from batch_inference_chat import batch_inference
from text_evaluation import getMeteorScore, getCosineSimilarity, getROUGEScore, getRMSEScore
from datasets import load_dataset, DatasetDict, Dataset
import torch
import multiprocessing

def runModelChat(data, case, device, num_pattern, token, dataset, window_size):
    model_chat = LLMChatModel("unsloth/Meta-Llama-3.1-8B-Instruct", token, dataset, False, case, device, window_size)
    data['idx'] = data.index
    log_path = "climate_log.csv"
    logger.remove()
    logger.add(log_path, rotation="10 MB", mode="w")

    results = [{"pred_output": "Wrong output format", "pred_time": "Wrong output format"} for _ in range(len(data))]
    batch_inference(results, model_chat, data, logger, num_pattern)

    results = pd.DataFrame(results, columns=['pred_output'])
    return results


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


def getTextScore(case, split, hf_dataset,text_pattern, number_pattern, window_size):
    data_all = load_dataset(hf_dataset)
    data = pd.DataFrame(data_all[split])
    pred_output_column = f'pred_output_case{case}'
    # number part evaluation
    if case in [2, 4]:
        data['pred_time'] = data[pred_output_column].apply(lambda x: find_num_parts(x, number_pattern, window_size))
        data_clean = data.dropna()
        drop_rate = (len(data) - len(data_clean)) / len(data)
        rmse_loss = getRMSEScore(data_clean)
    else:
        rmse_loss = np.nan
        drop_rate = np.nan
        
    # text part evaluation
    if case in [1, 2, 3]:
        output_texts = data['output_text'].apply(lambda x: split_text(x, text_pattern)).to_list()
        pred_texts = data[pred_output_column].apply(lambda x: find_text_parts(x, num_pattern)).apply(lambda x: split_text(x, text_pattern)).to_list()
        for idx, pred_text in enumerate(pred_texts):
            if len(pred_text) > window_size:
                pred_texts[idx] = pred_text[:window_size]
            while len(pred_text) < window_size:
                pred_texts[idx].append("No prediction")

        output_texts = np.reshape(output_texts, -1)
        pred_texts = np.reshape(pred_texts, -1)
        
        meteor_score = getMeteorScore(output_texts, pred_texts)
        cosine_similarity_score = getCosineSimilarity(output_texts, pred_texts)
        rouge1, rouge2, rougeL = getROUGEScore(output_texts, pred_texts)
    else:
        meteor_score = np.nan
        cosine_similarity_score = np.nan
        rouge1 = np.nan
        rouge2 = np.nan
        rougeL = np.nan
    
    return meteor_score, cosine_similarity_score, rouge1, rouge2, rougeL, rmse_loss, drop_rate

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    # add seed
    np.random.seed(42)
    set_seed(42)

    if len(sys.argv) != 5:
        print("Usage: python models/lltime_test.py <dataset> <window_size> <case> <split>")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    dataset = sys.argv[1]
    window_size = int(sys.argv[2])
    case = int(sys.argv[3])
    split = sys.argv[4]
    num_gpus = torch.cuda.device_count()

    if dataset == "climate":
        unit = "day"
        num_key_name = "temp"
        text_key_name = "weather_forecast"
    elif dataset == "medical":
        unit = "day"
        num_key_name = "Heart_Rate"
    elif dataset == "gas":
        unit = "week"
        num_key_name = "gas_price"
    
    if case == 1:
        model = "text2text"
    elif case == 2:
        model = "textTime2textTime"
    elif case == 3:
        model = "textTime2text"
    elif case == 4:
        model = "textTime2time"

    hf_dataset = f"Howard881010/{dataset}-{window_size}{unit}-finetuned"

    num_pattern = fr"{unit}_\d+_{num_key_name}: '([\d.]+)'"
    text_pattern =fr'({unit}_\d+_date:\s*\S+\s+{unit}_\d+_{text_key_name}:.*?)(?=\s{unit}_\d+_date|\Z)'

    wandb.init(project="Inference-new",
                config={"window_size": f"{window_size}-{window_size}",
                        "dataset": dataset,
                        "model": model})
    
    start_time = time.time()
    # Run models in parallel
    results = [pd.DataFrame() for _ in range(num_gpus)]
    devices = [f"cuda:{i}" for i in range(num_gpus)]
    data_all = load_dataset(hf_dataset)
    data = pd.DataFrame(data_all[split])
    dataset_parts = np.array_split(data, num_gpus)
    dataset_parts = [part.reset_index(drop=True) for part in dataset_parts]
        
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
    # Create a dictionary to map each future to its corresponding index
        future_to_index = {
            executor.submit(runModelChat, dataset_parts[i], case, devices[i], num_pattern, token, dataset, window_size): i
            for i in range(num_gpus)
        }
        # Iterate over the completed futures
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
    
    results = pd.concat(results, axis=0).reset_index(drop=True)
    
    uploadToHuf(results, hf_dataset, split)
    
    meteor_score, cos_sim_score, rouge1, rouge2, rougeL, rmse_loss, drop_rate = getTextScore(
        case, split, hf_dataset, num_pattern, window_size, text_pattern
    )


    wandb.log({"Meteor Scores": meteor_score})
    wandb.log({"Cos Sim Scores": cos_sim_score})
    wandb.log({"Rouge1 Scores": rouge1})
    wandb.log({"Rouge2 Scores": rouge2})
    wandb.log({"RougeL Scores": rougeL})
    wandb.log({"RMSE Scores": rmse_loss})
    wandb.log({"Drop Rate": f"{drop_rate*100:.2f}%"})
    
    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
