import sys
import os
import pandas as pd
import numpy as np
import wandb
import time
from loguru import logger
from utils import open_record_directory, create_result_file, add_idx
from modelchat import LLMChatModel, MistralChatModel
from transformers import set_seed
from batch_inference_chat import batch_inference_llama_summary
from text_evaluation import getMeteorScore, getCosineSimilarity, getROUGEScore, getRMSEScore
from datasets import load_dataset

def getSummaryOutput(dataset, unit, model_name, model_chat, sub_dir, window_size, split):
    hf_dataset = load_dataset(f"Howard881010/{dataset}-{window_size}_{unit}-{sub_dir.split('/')[0]}")

# Add idx column to each split
    data = pd.DataFrame(hf_dataset[split])
    data['idx'] = data.index

    log_path, res_path = open_record_directory(
        dataset=dataset, sub_dir=sub_dir, unit=unit, filename=split, model_name=model_name, window_size=window_size)

    logger.remove()
    logger.add(log_path, rotation="100 MB", mode="w")

    results = [{"pred_summary": "Not available"} for _ in range(len(data))]
    # chat = [{"role": "system", "content": data.iloc[0]['instruction']}, {"role": "user", "content": data.iloc[0]['input']}, {"role": "assistant", "content": data.iloc[0]['output']}]
    chat = None
    error_idx = batch_inference_llama_summary(
        results, model_chat, data, logger, window_size, dataset, chat_template=chat)

    print("Error idx: ", error_idx)
    results = pd.DataFrame(results, columns=['pred_summary'])
    # results['pred_summary'] = data['input'].apply(str)
    results['fut_summary'] = data['output'].apply(str)
    results.to_csv(res_path)
    return res_path


def getTextScore(dataset, filename, unit, sub_dir, case, window_size, num_name):
    meteor_score, nan_rate = getMeteorScore(filename)
    cosine_similarity_score = getCosineSimilarity(filename)
    rouge1, rouge2, rougeL = getROUGEScore(filename)
    if case == 2:
        rmse_loss = getRMSEScore(filename, num_name)
    else:
        rmse_loss = np.nan

    path = create_result_file(
        dir = f"Results/{dataset}/{window_size}_{unit}/{sub_dir}",
        filename = (filename.split("/")[-1]).replace("output", "text_score"),
    )
    
    results = [{"meteor_score": meteor_score, "nan_rate": nan_rate, "cosine_similarity": cosine_similarity_score, "rouge1": rouge1, "rouge2": rouge2, "rougeL": rougeL, "rmse": rmse_loss}]
    results = pd.DataFrame(results)
    results.to_csv(path)

    return meteor_score, nan_rate, cosine_similarity_score, rouge1, rouge2, rougeL, rmse_loss

if __name__ == "__main__":
    # add seed
    np.random.seed(42)
    set_seed(42)

    if len(sys.argv) != 6:
        print("Usage: python models/lltime_test.py <dataset> <window_size> <model_name> <case> <finetune>")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")

    dataset = sys.argv[1]
    window_size = int(sys.argv[2])
    case = int(sys.argv[4])
    model_name = sys.argv[3]
    finetune = sys.argv[5]

    if case == 1:
        if finetune == "finetune":
            sub_dir = "text-text-cal/finetune"
        elif finetune == "zeroshot":
            sub_dir = "text-text-cal/zeroshot"
    elif case == 2:
        if finetune == "finetune":
            sub_dir = "mixed-mixed-cal/finetune"
        elif finetune == "zeroshot":
            sub_dir = "mixed-mixed-cal/zeroshot"
    
    if model_name == "mistral7b":
        model_chat = MistralChatModel("mistralai/Mistral-7B-Instruct-v0.2", token)
        runs_name = "Mistral-7B-Instruct-v0.2"
    elif model_name == "llama7b":
        model_chat = LLMChatModel("meta-llama/Meta-Llama-3-8B-Instruct", token)
        runs_name = "Llama-2-7b-chat-hf"
    
    if dataset == "gas":
        unit = "week"
        num_name = "gas_price"
    elif dataset == "climate":
        unit = "day"
        num_name = "temperature"



    wandb.init(project="Inference",
               config={"name": runs_name,
                       "window_size": window_size,
                       "dataset": dataset,
                       "model": model_name + "-" + ("finetune" if finetune == "finetune" else "zeroshot"),
                       "case": sub_dir, })
    
    start_time = time.time()
    meteor_scores = []
    nans = []
    cos_sim_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    rmse_scores = []

    if case == 1 or case == 2:
        out_filename = getSummaryOutput(
            dataset, unit, model_name, model_chat, sub_dir, window_size, "validation"
        )
        # out_filename = "/home/ubuntu/multimodal/Predictions_and_attempts/climate/1_day/mixed-mixed/zeroshot/mistral7b_output_validation.csv"

        meteor_score, nan_rate, cos_sim_score, rouge1, rouge2, rougeL, rmse_loss = getTextScore(
            dataset, out_filename, unit, sub_dir, case, window_size, num_name
        )
        meteor_scores.append(meteor_score)
        nans.append(nan_rate)
        cos_sim_scores.append(cos_sim_score)
        rouge1_scores.append(rouge1)
        rouge2_scores.append(rouge2)
        rougeL_scores.append(rougeL)
        rmse_scores.append(rmse_loss)
                    
    print("Meteor Scores: ", np.mean(meteor_scores))
    print("Nan Rate: ", np.mean(nans))
    print("Cos Sim Scores: ", np.mean(cos_sim_scores))
    print("Rouge1 Scores: ", np.mean(rouge1_scores))
    print("Rouge2 Scores: ", np.mean(rouge2_scores))
    print("RougeL Scores: ", np.mean(rougeL_scores))
    print("RMSE Scores: ", np.mean(rmse_scores))
    wandb.log({"Meteor Scores": np.mean(meteor_scores)})
    wandb.log({"nan_rate": np.mean(nans)})
    wandb.log({"cos_sim_score": np.mean(cos_sim_scores)})
    wandb.log({"Rouge1 Scores": np.mean(rouge1_scores)})
    wandb.log({"Rouge2 Scores": np.mean(rouge2_scores)})
    wandb.log({"RougeL Scores": np.mean(rougeL_scores)})
    wandb.log({"RMSE Scores": np.mean(rmse_scores)})

    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
