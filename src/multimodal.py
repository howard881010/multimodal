import sys
import os
import pandas as pd
import numpy as np
import wandb
import time
from loguru import logger
from utils import open_record_directory
from modelchat import LLMChatModel
from transformers import set_seed
from batch_inference_chat import batch_inference_llama_summary
from text_evaluation import getMeteorScore, getCosineSimilarity, getROUGEScore, getRMSEScore, getGPTScore
from datasets import load_dataset, DatasetDict, Dataset

def getSummaryOutput(dataset, unit, model_name, model_chat, sub_dir, window_size, split, hf_dataset, num_pattern):
    data_all = load_dataset(hf_dataset)

    data = pd.DataFrame(data_all[split])
    data['idx'] = data.index

    log_path, res_path = open_record_directory(
        dataset=dataset, sub_dir=sub_dir, unit=unit, filename=split, model_name=model_name, window_size=window_size)

    logger.remove()
    logger.add(log_path, rotation="10 MB", mode="w")

    results = [{"pred_output": "Wrong output format", "pred_time": "Wrong output format"} for _ in range(len(data))]
    error_idx = batch_inference_llama_summary(results, model_chat, data, logger, unit, num_pattern)

    print("Error idx: ", error_idx)
    results = pd.DataFrame(results, columns=['pred_output', 'pred_time'])
    results['fut_summary'] = data['output'].apply(str)
    results.to_csv(res_path)
    data['pred_output'] = results['pred_output']
    data['pred_time'] = results['pred_time']
    data.drop(columns=['idx'], inplace=True)
    print(data.head(5))
    # updated_data = Dataset.from_pandas(data)
    # if split == 'validation':
    #     updated_dataset = DatasetDict({
    #         'train': data_all['train'], 
    #         'test': data_all['test'],
    #         'validation': updated_data
    #     })
    # elif split == 'test':
    #     updated_dataset = DatasetDict({
    #         'train': data_all['train'], 
    #         'validation': data_all['validation'],
    #         'test': updated_data
    #     })

    # updated_dataset.push_to_hub(hf_dataset)

    return res_path


def getTextScore(case, num_key_name, split,hf_dataset):
    data_all = load_dataset(hf_dataset)
    data = pd.DataFrame(data_all[split])
    data = data.iloc[:-1]

    meteor_score = getMeteorScore(data, num_key_name)
    # cosine_similarity_score = getCosineSimilarity(data, num_key_name)
    cosine_similarity_score = np.nan
    rouge1, rouge2, rougeL = getROUGEScore(data, num_key_name)
    # gpt_score = getGPTScore(data, num_key_name)
    gpt_score = np.nan

    if case == "mixed-mixed":
        print("case 2")
        rmse_loss = getRMSEScore(data, num_key_name)
    else:
        rmse_loss = np.nan
    

    return meteor_score, cosine_similarity_score, rouge1, rouge2, rougeL, rmse_loss, gpt_score

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
        hf_dataset = f"Howard881010/{dataset}-{window_size}{unit}-mixed"
        sub_dir = "mixed"
    elif case == 1:
        hf_dataset = f"Howard881010/{dataset}-{window_size}{unit}"
        sub_dir = "text"

    num_pattern = fr"<{unit}_\d+_{num_key_name}> : '([\d.]+)'"
    text_pattern = fr'(<{unit}_\d+_date>[^<]*<{unit}_\d+_{text_key_name}>[^<]*)' 
    
    model_chat = LLMChatModel("unsloth/Meta-Llama-3.1-8B-Instruct", token, dataset, window_size)
    
    # wandb.init(project="Inference-new",
    #            config={"window_size": f"{window_size}-{window_size}",
    #                    "dataset": dataset,
    #                    "model": model_name})
    start_time = time.time()

    out_filename = getSummaryOutput(
        dataset, unit, model_name, model_chat, sub_dir, window_size, split, hf_dataset, num_pattern
    )
    # meteor_score, cos_sim_score, rouge1, rouge2, rougeL, rmse_loss, gpt_score = getTextScore(
    #     case, num_key_name, split, hf_dataset
    # )

    # wandb.log({"Meteor Scores": meteor_score})
    # wandb.log({"Cos Sim Scores": cos_sim_score})
    # wandb.log({"Rouge1 Scores": rouge1})
    # wandb.log({"Rouge2 Scores": rouge2})
    # wandb.log({"RougeL Scores": rougeL})
    # wandb.log({"RMSE Scores": rmse_loss})
    # wandb.log({"GPT Scores": np.mean(gpt_score)})
    
    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
