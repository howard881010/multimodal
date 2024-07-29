import pandas as pd
import numpy as np
from nltk.translate import meteor
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from rouge_score import rouge_scorer
import json
from utils import rmse, convertJSONToList
import nltk
nltk.download('punkt')
nltk.download('wordnet')

def getMeteorScore(df, num_key_name):
    nan_rate = (df['pred_output'] == "Not available").sum() / len(df)
    df.replace("Wrong output format", np.nan, inplace=True)
    df_clean = df.dropna()

    # delete the number key in output and pred_output
    for idx, row in df_clean.iterrows():
        fut_res = json.loads(row['output'])
        pred_res = json.loads(row['pred_output'])
        for key in fut_res.keys():
            if num_key_name in fut_res[key].keys():
                del fut_res[key][num_key_name]
                df_clean.at[idx, 'output'] = json.dumps(fut_res)
        for key in pred_res.keys():
            if num_key_name in pred_res[key].keys():
                del pred_res[key][num_key_name]
                df_clean.at[idx, 'pred_output'] = json.dumps(pred_res)
                
    scores = [meteor([word_tokenize(x['output'])], word_tokenize(x['pred_output'])) for idx, x in df_clean.iterrows()]
    mean_score=np.mean(scores)
    
    return mean_score, nan_rate

def getCosineSimilarity(df):
    df.replace("Wrong output format", np.nan, inplace=True)
    df_clean = df.dropna()
    
    ground_truth = df_clean['output'].tolist()
    zero_shot = df_clean['pred_output'].tolist()
    # print(scores)
    model = SentenceTransformer('thenlper/gte-base')
    ground_truth_embeddings = model.encode(ground_truth)
    zero_shot_embeddings = model.encode(zero_shot)

    zs_cos_sims = [cos_sim(x, y) for x, y in zip(ground_truth_embeddings, zero_shot_embeddings)]

    return np.mean(zs_cos_sims)

def getROUGEScore(df):
    df.replace("Wrong output format", np.nan, inplace=True)
    df_clean = df.dropna()
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for idx, row in df_clean.iterrows():
        scores = scorer.score(row['output'], row['pred_output'])
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    mean_rouge1 = np.mean(rouge1_scores)
    mean_rouge2 = np.mean(rouge2_scores)
    mean_rougeL = np.mean(rougeL_scores)
    
    return mean_rouge1, mean_rouge2, mean_rougeL

def getRMSEScore(df, num_key_name):
    fut_values = []
    pred_values = []
    
    for idx, row in df.iterrows():
        pred_num = convertJSONToList(row, idx, num_key_name, "pred_output")
        fut_num = convertJSONToList(row, idx, num_key_name, "output")
        pred_num = np.array(pred_num).flatten().tolist()
        fut_num = np.array(fut_num).flatten().tolist()

        if len(fut_num) == len(pred_num) and all(isinstance(element, float) for element in pred_num):
            fut_values.append(fut_num)
            pred_values.append(pred_num)
                
    rmse_loss = rmse(np.array(fut_values), np.array(pred_values))
    
    return rmse_loss

def getBinaryPrecision(df, num_key_name):
    fut_values = []
    pred_values = []
    input_values = []
    precision = []

    for idx, row in df.iterrows():
        pred_num = convertJSONToList(row, idx, num_key_name, "pred_output")
        fut_num = convertJSONToList(row, idx, num_key_name, "output")
        input_num = convertJSONToList(row, idx, num_key_name, "input")
        pred_num = np.array(pred_num).flatten().tolist()
        fut_num = np.array(fut_num).flatten().tolist()
        input_num = np.array(input_num).flatten().tolist()

        if len(fut_num) == len(pred_num) and all(isinstance(element, float) for element in pred_num):
            fut_values.append(fut_num)
            pred_values.append(pred_num)
            input_values.append(input_num)
    
    for input_value, pred_value, fut_value in zip(input_values, pred_values, fut_values):
        if (input_value > pred_value and input_value > fut_value) or \
            (input_value < pred_value and input_value < fut_value) or \
            (input_value == pred_value and input_value == fut_value):
            precision.append(1)
        else:
            precision.append(0)

    return np.mean(precision)

    
