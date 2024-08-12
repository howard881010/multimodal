import sys
import os
import pandas as pd
import numpy as np
import wandb
import time
from loguru import logger
from utils import open_record_directory, create_result_file
from modelchat import MistralChatModel, LLMChatModel
from transformers import set_seed
from batch_inference_chat import batch_inference_llama_summary
from text_evaluation import getMeteorScore, getCosineSimilarity, getROUGEScore, getRMSEScore, getGPTScore
from datasets import load_dataset, DatasetDict, Dataset

def getSummaryOutput(dataset, unit, model_name, model_chat, sub_dir, window_size, split, hf_dataset):
    data_all = load_dataset(hf_dataset)

    data = pd.DataFrame(data_all[split])
    data['idx'] = data.index

    log_path, res_path = open_record_directory(
        dataset=dataset, sub_dir=sub_dir, unit=unit, filename=split, model_name=model_name, window_size=window_size)

    logger.remove()
    logger.add(log_path, rotation="100 MB", mode="w")

    results = [{"pred_summary": "Wrong output format"} for _ in range(len(data))]
    error_idx = batch_inference_llama_summary(results, model_chat, data, logger)

    print("Error idx: ", error_idx)
    results = pd.DataFrame(results, columns=['pred_summary'])
    results['fut_summary'] = data['output'].apply(str)
    results.to_csv(res_path)
    data['pred_output'] = results['pred_summary']
    updated_data = Dataset.from_pandas(data[['input', 'output', 'instruction','pred_output']])
    if split == 'validation':
        updated_dataset = DatasetDict({
            'train': data_all['train'], 
            'test': data_all['test'],
            'validation': updated_data
        })
    elif split == 'test':
        updated_dataset = DatasetDict({
            'train': data_all['train'], 
            'validation': data_all['validation'],
            'test': updated_data
        })

    updated_dataset.push_to_hub(hf_dataset)

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

    if len(sys.argv) != 7:
        print("Usage: python models/lltime_test.py <dataset> <window_size> <model_name> <case> <finetune> <split>")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")

    dataset = sys.argv[1]
    window_size = int(sys.argv[2])
    case = int(sys.argv[4])
    model_name = sys.argv[3]
    finetune = sys.argv[5]
    split = sys.argv[6]
    if case == 2:
        case = "mixed-mixed"
    elif case == 1:
        case = "text-text"

    if dataset == "climate":
        specified_region = "cal"
        unit = "day"
        num_key_name = "temperature"
    elif dataset == "medical":
        specified_region = None
        unit = "day"
        num_key_name = "Heart_Rate"
    elif dataset == "gas":
        specified_region = "west"
        unit = "week"
        num_key_name = "gas_price"
    
    # model_chat = MistralChatModel("mistralai/Mistral-7B-Instruct-v0.2", token, dataset)
    runs_name = "Mistral-7B-Instruct-v0.2"
    # model_chat = LLMChatModel("Howard881010/climate-cal", token, dataset)
    # runs_name = "Meta-Llama-3.1-8B-Instruct"
        



    wandb.init(project="Inference",
               config={"name": runs_name,
                       "window_size": window_size,
                       "dataset": dataset,
                       "model": model_name + "-" + ("finetune" if finetune == "finetune" else "zeroshot"),
                       "case": case,
                       "specified_region": specified_region,
                       "split": split})
    
    if specified_region is not None:
        sub_dir = f"{case}-{specified_region}"
    else:
        sub_dir = case
    start_time = time.time()

    hf_dataset = f"Howard881010/{dataset}-{window_size}_{unit}-{sub_dir.split('/')[0]}"

    # out_filename = getSummaryOutput(
    #     dataset, unit, model_name, model_chat, sub_dir, window_size, split, hf_dataset
    # )
    meteor_score, cos_sim_score, rouge1, rouge2, rougeL, rmse_loss, gpt_score = getTextScore(
        case, num_key_name, split, hf_dataset
    )

    
    print("Meteor Scores: ", meteor_score)
    print("Cos Sim Scores: ", cos_sim_score)
    print("Rouge1 Scores: ", rouge1)
    print("Rouge2 Scores: ", rouge2)
    print("RougeL Scores: ", rougeL)
    print("RMSE Scores: ", rmse_loss)
    print("GPT Scores: ", np.mean(gpt_score))
    wandb.log({"Meteor Scores": meteor_score})
    wandb.log({"Cos Sim Scores": cos_sim_score})
    wandb.log({"Rouge1 Scores": rouge1})
    wandb.log({"Rouge2 Scores": rouge2})
    wandb.log({"RougeL Scores": rougeL})
    wandb.log({"RMSE Scores": rmse_loss})
    wandb.log({"GPT Scores": np.mean(gpt_score)})
    
    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
