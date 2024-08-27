import concurrent.futures
import time
import sys
import os
import pandas as pd
import numpy as np
import wandb
import time
from loguru import logger
from utils import open_record_directory, find_text_parts, find_num_parts
from modelchat import LLMChatModel
from transformers import set_seed
from batch_inference_chat import batch_inference
from text_evaluation import getMeteorScore, getCosineSimilarity, getROUGEScore, getRMSEScore, getGPTScore
from datasets import load_dataset, DatasetDict, Dataset
import torch
import multiprocessing

def runModelChat(dataset_part, window_size, device, num_pattern, token):
    model_chat = LLMChatModel("unsloth/Meta-Llama-3.1-8B-Instruct", token, dataset_part, False, window_size, device)
    getSummaryOutput(
        dataset_part, model_chat, num_pattern
    )

def getSummaryOutput(data, model_chat, num_pattern):
    data['idx'] = data.index

    results = [{"pred_output": "Wrong output format", "pred_time": "Wrong output format"} for _ in range(len(data))]
    batch_inference(results, model_chat, data, logger, num_pattern)

    results = pd.DataFrame(results, columns=['pred_output'])
    return results


def uploadToHuf(results, hf_dataset, split):

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


def getTextScore(case, split, hf_dataset, text_pattern, number_pattern, window_size):
    data_all = load_dataset(hf_dataset)
    data = pd.DataFrame(data_all[split])
    if case == 2:
        data['pred_time'] = data['pred_output'].apply(lambda x: find_num_parts(x, number_pattern, window_size))
        data['pred_output'] = data['pred_output'].apply(lambda x: find_text_parts(x, number_pattern))
        data_clean = data.dropna()
        drop_rate = (len(data) - len(data_clean)) / len(data)
        rmse_loss = getRMSEScore(data_clean)
    else:
        rmse_loss = np.nan
        drop_rate = np.nan

    
    meteor_score = getMeteorScore(data)
    cosine_similarity_score = getCosineSimilarity(data)
    # cosine_similarity_score = np.nan
    rouge1, rouge2, rougeL = getROUGEScore(data)
    # gpt_score = getGPTScore(data)
    gpt_score = np.nan
    

    return meteor_score, cosine_similarity_score, rouge1, rouge2, rougeL, rmse_loss, gpt_score, drop_rate

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    # add seed
    np.random.seed(42)
    set_seed(42)

    if len(sys.argv) != 6:
        print("Usage: python models/lltime_test.py <dataset> <window_size> <model_name> <case> <split>")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    dataset = sys.argv[1]
    window_size = int(sys.argv[2])
    case = int(sys.argv[4])
    model_name = sys.argv[3]
    split = sys.argv[5]
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
    
    if case == 2:
        hf_dataset = f"Howard881010/{dataset}-{window_size}{unit}-mixed"
        sub_dir = "mixed"
    elif case == 1:
        hf_dataset = f"Howard881010/{dataset}-{window_size}{unit}"
        sub_dir = "text"

    num_pattern = fr"{unit}_\d+_{num_key_name}: '([\d.]+)'"
    text_pattern = fr'(?={unit}_\d+_date:)'

    # wandb.init(project="Inference-new",
    #             config={"window_size": f"{window_size}-{window_size}",
    #                     "dataset": dataset,
    #                     "model": model_name + ("-mixed" if case == 2 else "") + "-separate"})
    
    start_time = time.time()
    # Run models in parallel
    results = []
    devices = [f"cuda:{i}" for i in range(num_gpus)]
    data_all = load_dataset(hf_dataset)
    data = pd.DataFrame(data_all[split])
    dataset_parts = np.array_split(data, num_gpus)
        
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = [
            executor.submit(runModelChat, dataset_parts[i], window_size, devices[i], num_pattern, token)
            for i in range(num_gpus)
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    results = pd.concat(results)
    
    uploadToHuf(results, hf_dataset)
    
    meteor_score, cos_sim_score, rouge1, rouge2, rougeL, rmse_loss, gpt_score, drop_rate = getTextScore(
        case, split, hf_dataset, text_pattern, num_pattern, window_size
    )

    # wandb.log({"Meteor Scores": meteor_score})
    # wandb.log({"Cos Sim Scores": cos_sim_score})
    # wandb.log({"Rouge1 Scores": rouge1})
    # wandb.log({"Rouge2 Scores": rouge2})
    # wandb.log({"RougeL Scores": rougeL})
    # wandb.log({"RMSE Scores": rmse_loss})
    # wandb.log({"GPT Scores": np.mean(gpt_score)})
    # wandb.log({"Drop Rate": f"{drop_rate*100:.2f}%"})
    
    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
