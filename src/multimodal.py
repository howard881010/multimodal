import sys
import os
import pandas as pd
import numpy as np
import wandb
import time
from loguru import logger
from utils import open_record_directory, create_result_file
from modelchat import MistralChatModel
from transformers import set_seed
from batch_inference_chat import batch_inference_llama_summary
from text_evaluation import getMeteorScore, getCosineSimilarity, getROUGEScore, getRMSEScore
from datasets import load_dataset, DatasetDict, Dataset

def getSummaryOutput(dataset, unit, model_name, model_chat, sub_dir, window_size, split, hf_dataset):
    dataset = load_dataset(hf_dataset)

    data = pd.DataFrame(dataset[split])
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
    updated_dataset = DatasetDict({
        'train': dataset['train'], 
        'test': dataset['test'],
        'validation': updated_data
    })

    updated_dataset.push_to_hub(hf_dataset)

    return res_path


def getTextScore(dataset, filename, unit, sub_dir, case, window_size, num_key_name, split,hf_dataset):
    dataset = load_dataset(hf_dataset)
    data = pd.DataFrame(dataset[split])

    meteor_score, nan_rate = getMeteorScore(data, num_key_name)
    cosine_similarity_score = getCosineSimilarity(data)
    rouge1, rouge2, rougeL = getROUGEScore(data)
    if case == 2:
        rmse_loss = getRMSEScore(data, num_key_name)
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
    postfix = "cal" if dataset == "climate" else "west"

    if case == 1:
        if finetune == "finetune":
            sub_dir = f"text-text-{postfix}/finetune"
        elif finetune == "zeroshot":
            sub_dir = f"text-text-{postfix}/zeroshot"
    elif case == 2:
        if finetune == "finetune":
            sub_dir = f"mixed-mixed-{postfix}/finetune"
        elif finetune == "zeroshot":
            sub_dir = f"mixed-mixed-{postfix}/zeroshot"
    
    model_chat = MistralChatModel("mistralai/Mistral-7B-Instruct-v0.2", token, dataset)
    runs_name = "Mistral-7B-Instruct-v0.2"
    
    if dataset == "gas":
        unit = "week"
        num_key_name = "gas_price"
    elif dataset == "climate":
        unit = "day"
        num_key_name = "temperature"



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
        hf_dataset = f"Howard881010/{dataset}-{window_size}_{unit}-{sub_dir.split('/')[0]}"

        out_filename = getSummaryOutput(
            dataset, unit, model_name, model_chat, sub_dir, window_size, "validation", hf_dataset
        )
        # out_filename = "/home/ubuntu/multimodal/Predictions_and_attempts/climate/1_day/mixed-mixed/finetune/mistral7b_output_validation.csv"

        meteor_score, nan_rate, cos_sim_score, rouge1, rouge2, rougeL, rmse_loss = getTextScore(
            dataset, out_filename, unit, sub_dir, case, window_size, num_key_name, "validation", hf_dataset
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
