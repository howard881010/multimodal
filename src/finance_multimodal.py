import sys
import os
import pandas as pd
import numpy as np
import wandb
import time
from loguru import logger
from utils import open_record_directory, open_result_directory, rmse
from modelchat import LLMChatModel, MistralChatModel, GemmaChatModel
from transformers import set_seed
from tqdm import tqdm
import glob
from num2words import num2words
from batch_inference_chat import batch_inference_mistral, batch_inference_llama, batch_inference_gemma



def getLLMTIMEOutput(dataset, historical_window_size, filename, unit, model_name, model_chat, max_retries=4, backoff_factor=2):
    data = pd.read_csv(filename)
    data["idx"] = data.index

    log_path, out_path, res_path = open_record_directory(
        dataset, historical_window_size, unit, filename, model_name)

    logger.remove()
    logger.add(log_path, rotation="100 MB", mode="w")

    results = [{"pred_values": "nan", "attempt": 0} for _ in range(len(data))]
    attempts = 0
    
    example_input = f"{data.iloc[0]['input']}. {data.iloc[0]['instruction']}"
    example_answer = data.iloc[0]['output']
    chat = [
        {"role": "user",
            "content": example_input},
        {"role": "assistant", "content": example_answer}]
    
    if model_name == 'mistral7b':
        error_idx = batch_inference_mistral(
            results, model_chat, data, attempts, logger, historical_window_size, dataset, chat_template=chat)
    elif model_name == 'llama7b':
        error_idx = batch_inference_llama(
            results, model_chat, data, attempts, logger, historical_window_size, dataset, chat_template=chat)
    elif model_name == 'gemma7b':
        error_idx = batch_inference_gemma(
            results, model_chat, data, attempts, logger, historical_window_size, dataset, chat_template=chat)

    print("Error idx: ", error_idx)
    results = pd.DataFrame(results, columns=['pred_values', 'attempts'])
    results["fut_values"] = data["fut_values"].apply(str)
    results.to_csv(res_path)
    return res_path


def getLLMTIMERMSE(dataset, historical_window_size, filename, unit, model_name):
    data = pd.read_csv(filename)
    data["fut_values"] = data["fut_values"].apply(str)
    data["pred_values"] = data["pred_values"].apply(str)

    gt_values = []
    pred_values = []
    err = 0
    nan = 0
    if dataset == "Yelp":
        max_val = 5
        penalty = 1.3
    elif dataset == "Mimic" or dataset == "Climate":
        max_val = 1
        penalty = 0.0
    elif dataset == "Finance":
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
        if (
            len(fut_vals) != historical_window_size
            or len(pred_vals) != historical_window_size
        ):
            gt_values.extend([np.nan] * historical_window_size)
            pred_values.extend([np.nan] * historical_window_size)
        else:
            gt_values.extend(fut_vals)
            pred_values.extend(pred_vals)

    test_rmse_loss = rmse(np.array(gt_values), np.array(pred_values), penalty)
    err_rate = err / len(data) / historical_window_size
    nan_rate = nan / len(data)

    out_path = open_result_directory(
        dataset, historical_window_size, unit, filename, model_name)
    
    results = [{"test_rmse_loss": test_rmse_loss, "err_rate": err_rate, "nan_rate": nan_rate}]
    results = pd.DataFrame(results, columns=['test_rmse_loss', 'err_rate', 'nan_rate'])
    results.to_csv(out_path)

    return test_rmse_loss, err_rate, nan_rate


if __name__ == "__main__":
    # add seed
    np.random.seed(42)
    set_seed(42)

    if len(sys.argv) != 4:
        print("Usage: python models/lltime_test.py <dataset> <historical_window_size> <model_name>")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")

    dataset = sys.argv[1]
    historical_window_size = int(sys.argv[2])
    model_name = sys.argv[3]

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

    # wandb.init(project="llmforecast",
    #            config={"name": runs_name,
    #                    "window_size": historical_window_size,
    #                    "dataset": dataset,
    #                    "model": model_name, })
    
    start_time = time.time()
    rmses = []
    errs = []
    nans = []

    if dataset == "Yelp":
        unit = "weeks/"
    elif dataset == "Mimic" or dataset == "Climate" or dataset == "Finance":
        unit = "days/"
    else:
        print("No Such Dataset")
        sys.exit(1)

    folder_path = f"Data/{dataset}/{historical_window_size}_{unit}"
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
        # filepath = "Data/Yelp/4_weeks/test_1.csv"
                out_filename = getLLMTIMEOutput(
                    dataset, historical_window_size, filepath, unit, model_name, model_chat)
                out_rmse, out_err, out_nan = getLLMTIMERMSE(
                    dataset, historical_window_size, out_filename, unit, model_name
                )
                if out_rmse != 0 and str(out_rmse) != "nan":
                    rmses.append(out_rmse)
                errs.append(out_err)
                nans.append(out_nan)
    print("Mean RMSE: " + str(np.mean(rmses)))
    print("Mean Error Rate: " + str(np.mean(errs)))
    print("Mean NaN Rate: " + str(np.mean(nans)))
    print("Std-Dev RMSE: " + str(np.std(rmses)))
    # wandb.log({"rmse": np.mean(rmses)})
    # wandb.log({"error_rate": np.mean(errs)})
    # wandb.log({"nan_rate": np.mean(nans)})
    # wandb.log({"std-dev": np.std(rmses)})

    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
