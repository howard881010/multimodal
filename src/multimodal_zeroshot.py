import concurrent.futures
import time
import sys
import os
import pandas as pd
import numpy as np
import wandb
from loguru import logger
from modelchat import LLMChatModel
from utils import uploadToHuf
from transformers import set_seed
from batch_inference_chat import batch_inference
from text_evaluation import getTextScore
from datasets import load_dataset
import torch
import multiprocessing

def runModelChat(data, case, device, token, dataset, window_size):
    model_chat = LLMChatModel("unsloth/Meta-Llama-3.1-8B-Instruct", token, dataset, True, case, device, window_size)
    data['idx'] = data.index
    log_path = "climate_log.csv"
    logger.remove()
    logger.add(log_path, rotation="10 MB", mode="w")

    results = [{"pred_output": "Wrong output format"} for _ in range(len(data))]
    batch_inference(results, model_chat, data, logger, case)

    results = pd.DataFrame(results, columns=['pred_output'])
    return results

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    # add seed
    np.random.seed(42)
    set_seed(42)

    if len(sys.argv) != 5:
        print("Usage: python models/lltime_test.py <dataset> <window_size> <case>")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    dataset = sys.argv[1]
    window_size = int(sys.argv[2])
    case = int(sys.argv[3])
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

    hf_dataset = f"Howard881010/{dataset}-{window_size}{unit}-zeroshot"

    num_pattern = fr"{unit}_\d+_{num_key_name}: ?'?([\d.]+)'?"
    text_pattern =fr'({unit}_\d+_date:\s*\S+\s+{unit}_\d+_{text_key_name}:.*?)(?=\s{unit}_\d+_date|\Z)'

    wandb.init(project="Inference-zeroshot",
                config={"window_size": f"{window_size}-{window_size}",
                        "dataset": dataset,
                        "model": model + "-zeroshot"})
    
    start_time = time.time()
    # Run models in parallel
    results = [pd.DataFrame() for _ in range(num_gpus)]
    devices = [f"cuda:{i}" for i in range(num_gpus)]
    data_all = load_dataset(hf_dataset)
    data = pd.DataFrame(data_all['test'])
    dataset_parts = np.array_split(data, num_gpus)
    dataset_parts = [part.reset_index(drop=True) for part in dataset_parts]
        
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
    # Create a dictionary to map each future to its corresponding index
        future_to_index = {
            executor.submit(runModelChat, dataset_parts[i], case, devices[i], token, dataset, window_size): i
            for i in range(num_gpus)
        }
        # Iterate over the completed futures
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
    
    results = pd.concat(results, axis=0).reset_index(drop=True)
    
    uploadToHuf(results, hf_dataset, 'test', case)
    
    # meteor_score, cos_sim_score, rouge1, rouge2, rougeL, rmse_loss, drop_rate = getTextScore(
    #     case, split, hf_dataset, text_pattern, num_pattern, window_size
    # )


    # wandb.log({"Meteor Scores": meteor_score})
    # wandb.log({"Cos Sim Scores": cos_sim_score})
    # wandb.log({"Rouge1 Scores": rouge1})
    # wandb.log({"Rouge2 Scores": rouge2})
    # wandb.log({"RougeL Scores": rougeL})
    # wandb.log({"RMSE Scores": rmse_loss})
    # wandb.log({"Drop Rate": f"{drop_rate*100:.2f}%"})
    
    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
