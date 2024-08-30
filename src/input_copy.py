import sys
import os
import pandas as pd
import numpy as np
import wandb
import time
from loguru import logger
from transformers import set_seed
from utils import find_text_parts, find_num_parts
from text_evaluation import getMeteorScore, getCosineSimilarity, getROUGEScore, getRMSEScore
from datasets import load_dataset
from utils import find_text_parts

def getTextScore(split, hf_dataset):
    data_all = load_dataset(hf_dataset)
    data = pd.DataFrame(data_all[split])
    data['pred_time'] = data['input_time']
    data['pred_output'] = data['input']
    # data['pred_text'] = data['input'].apply(lambda x: find_text_parts(x, num_pattern))
    drop_rate = 0
    rmse_loss = getRMSEScore(data)

    
    meteor_score = getMeteorScore(data)
    cosine_similarity_score = getCosineSimilarity(data)
    # cosine_similarity_score = np.nan
    rouge1, rouge2, rougeL = getROUGEScore(data)
    gpt_score = np.nan
    

    return meteor_score, cosine_similarity_score, rouge1, rouge2, rougeL, rmse_loss, drop_rate

if __name__ == "__main__":
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

    if dataset == "climate":
        unit = "day"
        num_key_name = "temp"
    elif dataset == "medical":
        unit = "day"
    elif dataset == "gas":
        unit = "week"

    hf_dataset = f"Howard881010/{dataset}-{window_size}{unit}"
    num_pattern = fr"<{unit}_\d+_{num_key_name}> : '([\d.]+)'"    

    
    wandb.init(project="Inference-new",
               config={"window_size": f"{window_size}-{window_size}",
                       "dataset": dataset,
                       "model": model_name})
    
    start_time = time.time()
    meteor_score, cos_sim_score, rouge1, rouge2, rougeL, rmse_loss, drop_rate = getTextScore(
        split, hf_dataset
    )

    wandb.log({"Meteor Scores": meteor_score})
    wandb.log({"Cos Sim Scores": cos_sim_score})
    wandb.log({"Rouge1 Scores": rouge1})
    wandb.log({"Rouge2 Scores": rouge2})
    wandb.log({"RougeL Scores": rougeL})
    wandb.log({"RMSE Scores": rmse_loss})
    
    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
