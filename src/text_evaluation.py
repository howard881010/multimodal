import pandas as pd
import numpy as np
from nltk.translate import meteor
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from rouge_score import rouge_scorer
import json
from utils import rmse, convertJSONToList, clean_num
import nltk
from openai import OpenAI
import os
nltk.download('punkt')
nltk.download('wordnet')


def getMeteorScore(df, num_key_name):
    nan_rate = (df['pred_output'] == "Wrong output format").sum() / len(df)
    df.replace("Wrong output format", np.nan, inplace=True)
    df_clean = df.dropna()
    df_text_part = clean_num(df_clean, num_key_name)

    scores = [meteor([word_tokenize(x['output'])], word_tokenize(x['pred_output'])) for idx, x in df_text_part.iterrows()]
    mean_score=np.mean(scores)
    
    return mean_score, nan_rate

def getCosineSimilarity(df, num_key_name):
    df.replace("Wrong output format", np.nan, inplace=True)
    df_clean = df.dropna()
    df_text_part = clean_num(df_clean, num_key_name)
    
    ground_truth = df_text_part['output'].tolist()
    zero_shot = df_text_part['pred_output'].tolist()
    # print(scores)
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    ground_truth_embeddings = model.encode(ground_truth)
    zero_shot_embeddings = model.encode(zero_shot)

    zs_cos_sims = [cos_sim(x, y) for x, y in zip(ground_truth_embeddings, zero_shot_embeddings)]

    return np.mean(zs_cos_sims)

def getROUGEScore(df, num_key_name):
    df.replace("Wrong output format", np.nan, inplace=True)
    df_clean = df.dropna()
    df_text_part = clean_num(df_clean, num_key_name)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for idx, row in df_text_part.iterrows():
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
        pred_num = np.array(pred_num).flatten()
        fut_num = np.array(fut_num).flatten()

        if type(pred_num) == type(fut_num) and len(fut_num) == len(pred_num) and all(isinstance(element, float) for element in pred_num):
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
        pred_num = np.array(pred_num).flatten()
        fut_num = np.array(fut_num).flatten()
        input_num = np.array(input_num).flatten()

        if type(pred_num) == type(fut_num) and len(fut_num) == len(pred_num) and all(isinstance(element, float) for element in pred_num):
            fut_values.append(fut_num)
            pred_values.append(pred_num)
            input_values.append(input_num[-len(pred_num):])
    
    fut_values = np.reshape(fut_values, -1)
    pred_values = np.reshape(pred_values, -1)
    input_values = np.reshape(input_values, -1)

    print(fut_values.shape, pred_values.shape, input_values.shape)
    
    for input_value, pred_value, fut_value in zip(input_values, pred_values, fut_values):
        if (input_value > pred_value and input_value > fut_value) or \
            (input_value < pred_value and input_value < fut_value) or \
            (input_value == pred_value and input_value == fut_value):
            precision.append(1)
        else:
            precision.append(0)

    return np.mean(precision)

def getGPTScore(df, num_key_name):
    key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=key)

    df.replace("Wrong output format", np.nan, inplace=True)
    
    df_clean = df.dropna()
    df_text_part = clean_num(df_clean, num_key_name)

    gpt_scores = []    
    for idx, row in df_text_part.iterrows():
        question = f"summary1: {row['output']} summary2: {row['pred_output']}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. And can give me the semantic score between two summary from 1 to 10, you should only give me the scores"},
                {"role": "user", "content": question},
            ]
        )
        if len(response.choices[0].message.content) <= 2:
            gpt_scores.append(float(response.choices[0].message.content))
    return gpt_scores
