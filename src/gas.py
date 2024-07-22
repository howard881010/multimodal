import sys
import os
import pandas as pd
import numpy as np
import wandb
import time
from loguru import logger
from utils import open_record_directory, create_result_file, rmse
from modelchat import LLMChatModel, MistralChatModel
from transformers import set_seed
from tqdm import tqdm
import glob
from batch_inference_chat import batch_inference_llama_summary
from nltk.translate import meteor
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from rouge_score import rouge_scorer
import json
import matplotlib.pyplot as plt

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
    # results['pred_summary'] = data['input'].apply(str)
    results['fut_summary'] = data['output'].apply(str)
    results.to_csv(res_path)
    return res_path


def getTextScore(dataset, filename, unit, sub_dir, case, historical_window_size=1):
    meteor_score, nan_rate = getMeteorScore(filename)
    cosine_similarity_score = getCosineSimilarity(filename)
    rouge1, rouge2, rougeL = getROUGEScore(filename)
    if case == 2:
        rmse_loss = getRMSEScore(filename)
    else:
        rmse_loss = 0

    path = create_result_file(
        dir = f"Results/{dataset}/{historical_window_size}_{unit}/{sub_dir}",
        filename = (filename.split("/")[-1]).replace("output", "text_score"),
    )
    
    results = [{"meteor_score": meteor_score, "nan_rate": nan_rate, "cosine_similarity": cosine_similarity_score, "rouge1": rouge1, "rouge2": rouge2, "rougeL": rougeL, "rmse": rmse_loss}]
    results = pd.DataFrame(results)
    results.to_csv(path)

    return meteor_score, nan_rate, cosine_similarity_score, rouge1, rouge2, rougeL, rmse_loss

def getMeteorScore(filename):
    df = pd.read_csv(filename)
    nan_rate = (df['pred_summary'] == "Not available").sum() / len(df)
    df.replace("Not available", np.nan, inplace=True)
    df_clean = df.dropna()

    # delete the gas_price key in fut_summary and pred_summary
    for idx, row in df_clean.iterrows():
        fut_res = json.loads(row['fut_summary'])
        pred_res = json.loads(row['pred_summary'])
        for key in fut_res.keys():
            if "gas_price" in fut_res[key].keys():
                del fut_res[key]['gas_price']
                df_clean.at[idx, 'fut_summary'] = json.dumps(fut_res)
        for key in pred_res.keys():
            if "gas_price" in pred_res[key].keys():
                del pred_res[key]['gas_price']
                df_clean.at[idx, 'pred_summary'] = json.dumps(pred_res)
                
    scores = [meteor([word_tokenize(x['fut_summary'])], word_tokenize(x['pred_summary'])) for idx, x in df_clean.iterrows()]
    plotStats(scores)

# Print statistical information
    mean_score=np.mean(scores)
    
    return mean_score, nan_rate

def getCosineSimilarity(filename):
    df = pd.read_csv(filename)
    df.replace("Not available", np.nan, inplace=True)
    df_clean = df.dropna()
    
    ground_truth = df_clean['fut_summary'].tolist()
    zero_shot = df_clean['pred_summary'].tolist()
    # print(scores)
    model = SentenceTransformer('thenlper/gte-base')
    ground_truth_embeddings = model.encode(ground_truth)
    zero_shot_embeddings = model.encode(zero_shot)

    zs_cos_sims = [cos_sim(x, y) for x, y in zip(ground_truth_embeddings, zero_shot_embeddings)]

    return np.mean(zs_cos_sims)

def getCrossEncoderScore(filename):
    df = pd.read_csv(filename)
    df.replace("Not available", np.nan, inplace=True)
    df_clean = df.dropna()
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-roberta-base')
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-roberta-base')
    model.eval()
    
    scores = []
    
    for idx, row in df_clean.iterrows():
        # Tokenize the inputs
        inputs = tokenizer(row['fut_summary'], row['pred_summary'], return_tensors='pt', truncation=True, padding=True)
        
        # Get the model's predictions
        with torch.no_grad():
            scores = model(**inputs).logits
            label_mapping = ['contradiction', 'entailment', 'neutral']
            labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
            print(labels)
            
    
    # return mean_score, nan_rate

def getROUGEScore(filename):
    # Load the dataset
    df = pd.read_csv(filename)
    df.replace("Not available", np.nan, inplace=True)
    df_clean = df.dropna()
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for idx, row in df_clean.iterrows():
        scores = scorer.score(row['fut_summary'], row['pred_summary'])
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    mean_rouge1 = np.mean(rouge1_scores)
    mean_rouge2 = np.mean(rouge2_scores)
    mean_rougeL = np.mean(rougeL_scores)
    
    return mean_rouge1, mean_rouge2, mean_rougeL

def getRMSEScore(filename):
    df = pd.read_csv(filename)
    
    fut_values = []
    pred_values = []
    
    for idx, row in df.iterrows():
        try:
            fut_res = json.loads(row['fut_summary'])
            pred_res = json.loads(row['pred_summary'])
            fut_num = [ele['gas_price'] for ele in fut_res.values()]
            pred_num = [ele['gas_price'] for ele in pred_res.values()]
            if len(fut_num) == len(pred_num):
                fut_values.append(fut_num)
                pred_values.append(pred_num)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"An error occurred: {e}, row: {idx}")
    

    rmse_loss = rmse(np.array(fut_values), np.array(pred_values))
    
    return rmse_loss

def plotStats(arr):
    scores_df = pd.DataFrame(arr, columns=['METEOR Score'])

# Print statistical information
    print(scores_df.describe())

    # Plotting the statistical information
    plt.figure(figsize=(12, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(scores_df['METEOR Score'], bins=10, edgecolor='black')
    plt.title('Histogram of METEOR Scores')
    plt.xlabel('METEOR Score')
    plt.ylabel('Frequency')

    # Boxplot
    plt.subplot(1, 2, 2)
    plt.boxplot(scores_df['METEOR Score'], vert=False)
    plt.title('Boxplot of METEOR Scores')
    plt.xlabel('METEOR Score')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # add seed
    np.random.seed(42)
    set_seed(42)

    if len(sys.argv) != 6:
        print("Usage: python models/lltime_test.py <dataset> <historical_window_size> <model_name> <case> <finetune>")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")

    dataset = sys.argv[1]
    historical_window_size = int(sys.argv[2])
    case = int(sys.argv[4])
    model_name = sys.argv[3]
    finetune = sys.argv[5]

    if case == 1:
        if finetune == "finetune":
            sub_dir = "text-text-fact/finetune"
        elif finetune == "zeroshot":
            sub_dir = "text-text-fact/zeroshot"
    elif case == 2:
        if finetune == "finetune":
            sub_dir = "mixed-mixed-fact/finetune"
        elif finetune == "zeroshot":
            sub_dir = "mixed-mixed-fact/zeroshot"
    
    if model_name == "mistral7b":
    # model_chat = LLMChatModel("meta-llama/Llama-2-7b-chat-hf", token)
        model_chat = MistralChatModel("mistralai/Mistral-7B-Instruct-v0.1", token)
        runs_name = "Mistral-7B-Instruct-v0.1"
    elif model_name == "llama7b":
        model_chat = LLMChatModel("meta-llama/Meta-Llama-3-8B-Instruct", token)
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
    cos_sim_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    rmse_scores = []

    unit = "week"

    folder_path = f"Data/{dataset}/{historical_window_size}_{unit}/{sub_dir.split('/')[0]}/"
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
                if case == 1 or case == 2:
                    out_filename = getSummaryOutput(
                        dataset, filepath, unit, model_name, model_chat, sub_dir, historical_window_size
                    )
                    # out_filename = "/home/ubuntu/multimodal/Predictions_and_attempts/Gas/2_week/mixed-mixed-fact/zeroshot/mistral7b_output_validation_all.csv"

                    meteor_score, nan_rate, cos_sim_score, rouge1, rouge2, rougeL, rmse_loss = getTextScore(
                        dataset, out_filename, unit, sub_dir, case, historical_window_size
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
