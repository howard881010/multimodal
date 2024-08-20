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
from batch_inference_chat import batch_inference_inContext
from text_evaluation import getMeteorScore, getCosineSimilarity, getROUGEScore, getRMSEScore, getGPTScore
from datasets import load_dataset, DatasetDict, Dataset

def getSummaryOutput(dataset, unit, model_name, model_chat, sub_dir, window_size, split, hf_dataset, num_pattern):
    data_all = load_dataset(hf_dataset)

    data = pd.DataFrame(data_all[split])
    data['idx'] = data.index

    log_path, res_path = open_record_directory(
        dataset, unit, split, model_name, sub_dir, window_size)

    logger.remove()
    logger.add(log_path, rotation="10 MB", mode="w")

    results = [{"pred_output": "Wrong output format", "pred_time": "Wrong output format"} for _ in range(len(data))]
    batch_inference_inContext(results, model_chat, data, logger, num_pattern)

    results = pd.DataFrame(results, columns=['pred_output'])
    results['fut_summary'] = data['output'].apply(str)
    results.to_csv(res_path)
    data['pred_output'] = results['pred_output']
    data.drop(columns=['idx'], inplace=True)

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
    # cosine_similarity_score = getCosineSimilarity(data)
    cosine_similarity_score = np.nan
    rouge1, rouge2, rougeL = getROUGEScore(data)
    # gpt_score = getGPTScore(data)
    gpt_score = np.nan
    

    return meteor_score, cosine_similarity_score, rouge1, rouge2, rougeL, rmse_loss, gpt_score, drop_rate

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
        text_key_name = "weather_forecast"
    elif dataset == "medical":
        unit = "day"
        num_key_name = "Heart_Rate"
    elif dataset == "gas":
        unit = "week"
        num_key_name = "gas_price"
    
    if case == 2:
        hf_dataset = f"Howard881010/{dataset}-{window_size}{unit}-mixed-inContext"
        sub_dir = "mixed"
    elif case == 1:
        hf_dataset = f"Howard881010/{dataset}-{window_size}{unit}-inContext"
        sub_dir = "text"

    num_pattern = fr"{unit}_\d+_{num_key_name}: '([\d.]+)'"
    text_pattern = fr'(?={unit}_\d+_date:)'
    
    model_chat = LLMChatModel("unsloth/Meta-Llama-3.1-8B-Instruct", token, dataset, window_size)
    
    wandb.init(project="Inference-new",
               config={"window_size": f"{window_size}-{window_size}",
                       "dataset": dataset,
                       "model": model_name + "InContext" + ("mixed" if case == 2 else "")})
    start_time = time.time()

    getSummaryOutput(
        dataset, unit, model_name, model_chat, sub_dir, window_size, split, hf_dataset, num_pattern
    )
    meteor_score, cos_sim_score, rouge1, rouge2, rougeL, rmse_loss, gpt_score, drop_rate = getTextScore(
        case, split, hf_dataset, text_pattern, num_pattern, window_size
    )

    wandb.log({"Meteor Scores": meteor_score})
    wandb.log({"Cos Sim Scores": cos_sim_score})
    wandb.log({"Rouge1 Scores": rouge1})
    wandb.log({"Rouge2 Scores": rouge2})
    wandb.log({"RougeL Scores": rougeL})
    wandb.log({"RMSE Scores": rmse_loss})
    wandb.log({"GPT Scores": np.mean(gpt_score)})
    wandb.log({"Drop Rate": f"{drop_rate*100:.2f}%"})
    
    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
