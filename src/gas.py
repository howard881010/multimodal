import sys
import os
import pandas as pd
import numpy as np
import wandb
import time
from loguru import logger
from utils import open_record_directory, open_result_directory
from modelchat import LLMChatModel
from transformers import set_seed
from tqdm import tqdm
import glob
from batch_inference_chat import batch_inference_llama_summary
import re
import json
from nltk.translate import meteor
from nltk import word_tokenize

def getSummaryOutput(dataset, filename, unit, model_name, model_chat, sub_dir, historical_window_size):
    data = pd.read_csv(filename)
    data["idx"] = data.index
    # for idx, row in data.iterrows():
    #     data.at[idx, 'fut_summary'] = json.loads(row['output'])["summary"]
    # print(data['fut_values'])

    log_path, res_path = open_record_directory(
        dataset=dataset, sub_dir=sub_dir, unit=unit, filename=filename, model_name=model_name, historical_window_size=historical_window_size)

    logger.remove()
    logger.add(log_path, rotation="100 MB", mode="w")

    results = [{"pred_summary": "Not available"} for _ in range(len(data))]
    # chat = [{"role": "system", "content": data.iloc[0]['instruction']}, {"role": "user", "content": data.iloc[0]['input']}, {"role": "assistant", "content": data.iloc[0]['output']}]
    chat = None
    error_idx = batch_inference_llama_summary(
        results, model_chat, data, logger, historical_window_size, dataset, chat_template=chat)

    print("Error idx: ", error_idx)
    results = pd.DataFrame(results, columns=['pred_summary'])
    results['fut_summary'] = data['output'].apply(str)
    results.to_csv(res_path)
    return res_path

def getMeteorScore(dataset, filename, unit, model_name, sub_dir, historical_window_size=1):
    df = pd.read_csv(filename)
    nan_rate = (df['pred_summary'] == "Not available").sum() / len(df)
    
    scores = [meteor([word_tokenize(x['fut_summary'])], word_tokenize(x['pred_summary'])) for idx, x in df.iterrows()]
    # print(scores)
    mean_score=np.mean(scores)
    # Display the dataframe with the METEOR scores
    out_path = open_result_directory(
        dataset=dataset, sub_dir=sub_dir, unit=unit, filename=filename, model_name=model_name, historical_window_size=historical_window_size)
    
    results = [{"meteor_score": mean_score, "nan_rate": nan_rate}]
    results = pd.DataFrame(results, columns=["meteor_score", "nan_rate"])
    results.to_csv(out_path)

    return mean_score, nan_rate




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
        sub_dir = "text-text"

    model_chat = LLMChatModel("meta-llama/Llama-2-7b-chat-hf", token)
    runs_name = "Llama-2-7b-chat-hf"

    wandb.init(project="Inference",
               config={"name": runs_name,
                       "window_size": historical_window_size,
                       "dataset": dataset,
                       "model": model_name,
                       "case": sub_dir, })
    
    start_time = time.time()
    meteor_scores = []
    nans = []

    unit = "week"

    folder_path = f"Data/{dataset}/{historical_window_size}_{unit}/{sub_dir}"
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist")
        sys.exit(1)
    else:
        pattern = "validation_*.csv"
        for filepath in tqdm(glob.glob(os.path.join(folder_path, pattern))):
            filename = os.path.basename(filepath)
            if "all" not in filename:
                continue
            else:
                if case == 1:
                    out_filename = getSummaryOutput(
                        dataset, filepath, unit, model_name, model_chat, sub_dir, historical_window_size
                    )
                    # out_filename = "/home/ubuntu/multimodal/Predictions_and_attempts/Gas/text-text/llama7b_output_all.csv"
                    mean_score, nan_rate = getMeteorScore(
                        dataset, out_filename, unit, model_name, sub_dir, historical_window_size
                    )
                    meteor_scores.append(mean_score)
                    nans.append(nan_rate)

    print("Meteor Scores: ", np.mean(meteor_scores))
    print("Nan Rate: ", np.mean(nans))
    wandb.log({"mean_score": np.mean(meteor_scores)})
    wandb.log({"nan_rate": np.mean(nans)})

    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
