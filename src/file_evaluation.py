import time
import sys
import os
import pandas as pd
import numpy as np
import wandb
from transformers import set_seed
from text_evaluation import getRMSEScore, getMeteorScore, getCosineSimilarity, getROUGEScore
import re
import ast


def getTextScore(text_pattern, window_size):
    file_path = f"Data/hybrid_results_{window_size}_{window_size}.csv"

    df = pd.read_csv(file_path)
    pred_times = df['pred_times'].apply(lambda x: x.strip('[]').split()).tolist()
    output_times = df['output_times'].apply(lambda x: x.strip('[]').split()).tolist()
    rmse_loss = getRMSEScore(pred_times, output_times)

    pred_texts = df['pred_texts'].apply(lambda x: re.findall(text_pattern, x, re.DOTALL)).tolist()
    output_texts = df['output_texts'].apply(lambda x: ast.literal_eval(x)).to_list()
    for idx, pred_text in enumerate(pred_texts):
        if len(pred_text) > window_size:
            pred_texts[idx] = pred_text[:window_size]
        while len(pred_text) < window_size:
            pred_texts[idx].append("No prediction")
    
    output_texts = np.reshape(output_texts, -1)
    pred_texts = np.reshape(pred_texts, -1)
    print(len(output_texts), len(pred_texts))
    indices_to_drop = [idx for idx, pred_text in enumerate(pred_texts) if "No prediction" in pred_text]
    text_drop_count = len(indices_to_drop)
    print(text_drop_count)
    output_texts = np.delete(output_texts, indices_to_drop)
    pred_texts = np.delete(pred_texts, indices_to_drop)

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
        text_key_name = "weather_forecast"
    elif dataset == "medical":
        unit = "day"
        num_key_name = "Heart_Rate"

    text_pattern = fr'({unit}_\d+_{text_key_name}:.*?)(?=\s{unit}_\d+_date|\Z)'

    wandb.init(project="Inference-hybird",
                config={"window_size": f"{window_size}-{window_size}",
                        "dataset": dataset,
                        "model": "hybrid-model"})
    
    start_time = time.time()
    # Run models in parallel
    
    meteor_score, cos_sim_score, rouge1, rouge2, rougeL, rmse_loss = getTextScore(
        text_pattern, window_size
    )


    wandb.log({"Meteor Scores": meteor_score})
    wandb.log({"Cos Sim Scores": cos_sim_score})
    wandb.log({"Rouge1 Scores": rouge1})
    wandb.log({"Rouge2 Scores": rouge2})
    wandb.log({"RougeL Scores": rougeL})
    wandb.log({"RMSE Scores": rmse_loss})
    
    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
