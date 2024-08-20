import pandas as pd
import numpy as np
from nltk.translate import meteor
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from rouge_score import rouge_scorer
import json
from utils import rmse, find_text_parts
import nltk
from openai import OpenAI
import os
nltk.download('punkt')
nltk.download('wordnet')


def getRMSEScore(df):
    fut_values = df['output_time'].tolist()
    pred_values = df['pred_time'].tolist()
    rmse_loss = rmse(np.array(fut_values), np.array(pred_values))
    
    return rmse_loss

def getMeteorScore(df):
    scores = [meteor([word_tokenize(x['output'])], word_tokenize(x['pred_output'])) for idx, x in df.iterrows()]
    mean_score=np.mean(scores)
    
    return mean_score

def getCosineSimilarity(df):
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    cos_sims = [cos_sim(x, y) for x, y in zip(model.encode(df['output'].tolist()), model.encode(df['pred_output'].tolist()))]

    return np.mean(cos_sims)

def getROUGEScore(df):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for idx, row in df.iterrows():
        scores = scorer.score(row['output'], row['pred_output'])
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    mean_rouge1 = np.mean(rouge1_scores)
    mean_rouge2 = np.mean(rouge2_scores)
    mean_rougeL = np.mean(rougeL_scores)
    
    return mean_rouge1, mean_rouge2, mean_rougeL

def getGPTScore(df):
    key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=key)

    df.replace("Wrong output format", np.nan, inplace=True)
    
    gpt_scores = []    
    for idx, row in df.iterrows():
        question = f"summary1: {row['output']} summary2: {row['pred_output']}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant capable of evaluating the semantic similarity between two summaries. The semantic score you provide should be a number between 1 and 10, where 10 represents the highest level of semantic similarity (meaning the summaries convey almost identical information), and 1 represents the lowest level of semantic similarity (meaning the summaries convey entirely different or unrelated information). The score should reflect how closely the meanings and key details of the two summaries align. You should only give me the number, nothing else."},
                {"role": "user", "content": question},
            ]
        )
        if len(response.choices[0].message.content) <= 2:
            gpt_scores.append(float(response.choices[0].message.content))
    return gpt_scores