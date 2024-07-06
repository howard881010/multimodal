import sys
import os
import pandas as pd
import numpy as np
import wandb
import time
from loguru import logger
from utils import open_record_directory, open_result_directory, rmse, nmae
from modelchat import LLMChatModel, MistralChatModel, GemmaChatModel
from transformers import set_seed
from tqdm import tqdm
import glob
from num2words import num2words
from batch_inference_chat import batch_inference_mistral, batch_inference_llama, batch_inference_gemma
import re
import json



def getLLMTIMEOutput(dataset, filename, unit, model_name, model_chat, sub_dir, historical_window_size):
    data = pd.read_csv(filename)
    data["idx"] = data.index
    for idx, row in data.iterrows():
        data.at[idx, 'fut_values'] = json.loads(row['output'])["share_price"]
    # print(data['fut_values'])

    log_path, res_path = open_record_directory(
        dataset=dataset, sub_dir=sub_dir, unit=unit, filename=filename, model_name=model_name, historical_window_size=historical_window_size)

    logger.remove()
    logger.add(log_path, rotation="100 MB", mode="w")

    results = [{"pred_values": "nan", "attempt": 0} for _ in range(len(data))]
    attempts = 0
    chat = None
    error_idx = batch_inference_llama(
        results, model_chat, data, attempts, logger, historical_window_size, dataset, chat_template=chat)

    print("Error idx: ", error_idx)
    results = pd.DataFrame(results, columns=['pred_values', 'attempts'])
    results["fut_values"] = data["fut_values"].apply(str)
    results.to_csv(res_path)
    return res_path

def getSummaryOutput(dataset, filename, unit, model_name, model_chat, sub_dir, historical_window_size):
    data = pd.read_csv(filename)
    for idx, row in data.iterrows():
        data.at[idx, 'fut_summary'] = json.loads(row['output'])["summary"]
        data.at[idx, 'fut_values'] = json.loads(row['output'])["share_price"]
    # print(data['fut_values'])

    log_path, res_path = open_record_directory(
        dataset=dataset, sub_dir=sub_dir, unit=unit, filename=filename, model_name=model_name, historical_window_size=historical_window_size)

    logger.remove()
    logger.add(log_path, rotation="100 MB", mode="w")

    results = [{"pred_values": "nan", "attempt": 0} for _ in range(len(data))]
    attempts = 0
    chat = None
    error_idx = batch_inference_llama(
        results, model_chat, data, attempts, logger, historical_window_size, dataset, chat_template=chat)

    print("Error idx: ", error_idx)
    results = pd.DataFrame(results, columns=['pred_values', 'attempts'])
    results["fut_values"] = data["fut_values"].apply(str)
    results.to_csv(res_path)
    return res_path


def getLLMTIMERMSE(dataset, filename, unit, model_name, sub_dir, historical_window_size=1):
    data = pd.read_csv(filename)
    data["fut_values"] = data["fut_values"].apply(str)
    data["pred_values"] = data["pred_values"].apply(str)

    gt_values = []
    pred_values = []
    err = 0
    nan = 0
    if dataset == "Finance":
        max_val = 10000
        penalty = 0.0

    for index, row in data.iterrows():
        fut_vals = [float(ele) for ele in row["fut_values"].split()]
        pred_vals = [float(ele) for ele in row["pred_values"].split()]

        for val in pred_vals:
            if val > max_val or val < 0:
                err += 1
            if np.isnan(val):
                nan += 1
        if (np.nan in pred_vals):
            gt_values.append([np.nan] * historical_window_size)
            pred_values.append([np.nan] * historical_window_size)
        else:
            gt_values.append(fut_vals)
            pred_values.append(pred_vals)

    test_rmse_loss = rmse(np.array(pred_values), np.array(gt_values), penalty)
    test_nmae_loss = nmae(np.array(pred_values), np.array(gt_values), penalty)
    err_rate = err / len(data) / historical_window_size
    nan_rate = nan / len(data)

    out_path = open_result_directory(
        dataset=dataset, sub_dir=sub_dir, unit=unit, filename=filename, model_name=model_name, historical_window_size=historical_window_size)
    
    results = [{"test_rmse_loss": test_rmse_loss, "err_rate": err_rate, "nan_rate": nan_rate, "test_nmae_loss": test_nmae_loss}]
    results = pd.DataFrame(results, columns=['test_rmse_loss', 'err_rate', 'nan_rate', 'test_nmae_loss'])
    results.to_csv(out_path)

    return test_rmse_loss, err_rate, nan_rate, test_nmae_loss

if __name__ == "__main__":
    # add seed
    np.random.seed(42)
    set_seed(42)

    if len(sys.argv) != 5:
        print("Usage: python models/lltime_test.py <dataset> <historical_window_size> <model_name> <case>")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")

    dataset = sys.argv[1]
    historical_window_size = int(sys.argv[2])
    case = int(sys.argv[4])
    model_name = sys.argv[3]

    if case == 1:
        sub_dir = "numerical"
    elif case == 2:
        sub_dir = "mixed-numerical"
    elif case == 3:
        sub_dir = "mixed-summary"

    if model_name == "llama7b":
        model_chat = LLMChatModel("meta-llama/Llama-2-7b-chat-hf", token)
        runs_name = "Llama-2-7b-chat-hf"
    elif model_name == "mistral7b":
        model_chat = MistralChatModel(
            "mistralai/Mistral-7B-Instruct-v0.1", token)
        runs_name = "Mistral-7B-Instruct-v0.1"
    elif model_name == "gemma7b":
        model_chat = GemmaChatModel("google/gemma-7b-it", token)
        runs_name = "gemma-7b-it"

    wandb.init(project="Inference",
               config={"name": runs_name,
                       "window_size": historical_window_size,
                       "dataset": dataset,
                       "model": model_name, })
    
    start_time = time.time()
    rmses = []
    errs = []
    nans = []
    nmaes = []

    if dataset == "Yelp":
        unit = "week/"
    elif dataset == "Mimic" or dataset == "Climate" or dataset == "Finance":
        unit = "day/"
    else:
        print("No Such Dataset")
        sys.exit(1)

    folder_path = f"Data/{dataset}/{historical_window_size}_{unit}/{sub_dir}"
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist")
        sys.exit(1)
    else:
        pattern = "test_*.csv"
        for filepath in tqdm(glob.glob(os.path.join(folder_path, pattern))):
            filename = os.path.basename(filepath)
            if "all" not in filename:
                continue
            else:
                if case <= 2:
        # filepath = "Data/Yelp/4_weeks/test_1.csv"
                    out_filename = getLLMTIMEOutput(
                        dataset, filepath, unit, model_name, model_chat, sub_dir, historical_window_size)
                    out_rmse, out_err, out_nan, out_nmae = getLLMTIMERMSE(
                        dataset, out_filename, unit, model_name, sub_dir, historical_window_size
                    )
                # elif case == 3:
                    
                if out_rmse != 0 and str(out_rmse) != "nan":
                    rmses.append(out_rmse)
                errs.append(out_err)
                nans.append(out_nan)
                nmaes.append(out_nmae)
    print("Mean RMSE: " + str(np.mean(rmses)))
    print("Mean Error Rate: " + str(np.mean(errs)))
    print("Mean NaN Rate: " + str(np.mean(nans)))
    print("Std-Dev RMSE: " + str(np.std(rmses)))
    print("Mean nmae: " + str(np.mean(nmaes)))
    wandb.log({"rmse": np.mean(rmses)})
    wandb.log({"error_rate": np.mean(errs)})
    wandb.log({"nan_rate": np.mean(nans)})
    wandb.log({"std-dev": np.std(rmses)})
    wandb.log({"nmae": np.mean(nmaes)})

    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
