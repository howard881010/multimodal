import pandas as pd
import numpy as np
from nltk.translate import meteor
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from rouge_score import rouge_scorer
import json
from utils import rmse
import nltk
nltk.download('punkt')
nltk.download('wordnet')

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

def getRMSEScore(filename, key_name):
    df = pd.read_csv(filename)
    
    fut_values = []
    pred_values = []
    
    for idx, row in df.iterrows():
        try:
            fut_res = json.loads(row['fut_summary'])
            pred_res = json.loads(row['pred_summary'])
            fut_num_dict_list = [ele[key_name] for ele in fut_res.values()]
            pred_num_dict_list = [ele[key_name] for ele in pred_res.values()]
            pred_num = [list(pred_num_dict.values()) for pred_num_dict in pred_num_dict_list]
            fut_num = [list(fut_num_dict.values()) for fut_num_dict in fut_num_dict_list]
            pred_num = np.array(pred_num).flatten().tolist()
            fut_num = np.array(fut_num).flatten().tolist()

            if len(fut_num) == len(pred_num) and all(isinstance(element, float) for element in pred_num):
                fut_values.append(fut_num)
                pred_values.append(pred_num)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"An error occurred: {e}, row: {idx}")
                
    rmse_loss = rmse(np.array(fut_values), np.array(pred_values))
    
    return rmse_loss