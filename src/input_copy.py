import sys
import os
import pandas as pd
import numpy as np
import wandb
import time
from loguru import logger
from transformers import set_seed
from text_evaluation import getMeteorScore, getCosineSimilarity, getROUGEScore, getRMSEScore
from datasets import load_dataset
from utils import split_text

def getTextScore(hf_dataset, text_key_name, window_size):
    data_all = load_dataset(hf_dataset)
    data = pd.DataFrame(data_all['test'])
    # number part evaluation
    pred_values = data['input_num'].to_list()
    fut_values = data['output_num'].to_list()
    rmse_loss = getRMSEScore(pred_values, fut_values)
        
    # text part evaluation
    output_texts = data['output_text'].apply(lambda x: split_text(x, text_key_name, window_size)).to_list()
    pred_texts = data['input_text'].apply(lambda x: split_text(x, text_key_name, 0)).to_list()
    output_texts = np.reshape(output_texts, -1)
    pred_texts = np.reshape(pred_texts, -1)

    meteor_score = getMeteorScore(output_texts, pred_texts)
    cosine_similarity_score = getCosineSimilarity(output_texts, pred_texts)
    rouge1, rouge2, rougeL = getROUGEScore(output_texts, pred_texts)

    return meteor_score, cosine_similarity_score, rouge1, rouge2, rougeL, rmse_loss

if __name__ == "__main__":
    # add seed
    np.random.seed(42)
    set_seed(42)

    if len(sys.argv) != 3:
        print("Usage: python models/lltime_test.py <dataset> <window_size>")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")

    dataset = sys.argv[1]
    window_size = int(sys.argv[2])

    if dataset == "climate":
        unit = "day"
        num_key_name = "temp"
        text_key_name = "weather_forecast"
    elif dataset == "medical":
        unit = "day"
        num_key_name = "Heart_Rate"
        text_key_name = "medical_notes"
    elif dataset == "gas":
        unit = "week"
        num_key_name = "gas_price"

    hf_dataset = f"Howard881010/{dataset}-{window_size}{unit}-finetuned"

    wandb.init(project="Inference-input-copy",
                config={"window_size": f"{window_size}-{window_size}",
                        "dataset": dataset,
                        "model": "input-copy"})
    
    start_time = time.time()
    meteor_score, cos_sim_score, rouge1, rouge2, rougeL, rmse_loss = getTextScore(
        hf_dataset, text_key_name, window_size
    )

    wandb.log({"Meteor Scores": meteor_score})
    wandb.log({"Cos Sim Scores": cos_sim_score})
    wandb.log({"Rouge1 Scores": rouge1})
    wandb.log({"Rouge2 Scores": rouge2})
    wandb.log({"RougeL Scores": rougeL})
    wandb.log({"RMSE Scores": rmse_loss})
    
    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
