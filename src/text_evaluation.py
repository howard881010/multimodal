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
    fut_values = df['output_num'].tolist()
    pred_values = df['pred_time'].tolist()
    rmse_loss = rmse(np.array(fut_values), np.array(pred_values))
    
    return rmse_loss

def getMeteorScore(outputs, pred_outputs):
    scores = [meteor([word_tokenize(output)], word_tokenize(pred_output)) for output, pred_output in zip(outputs, pred_outputs)]
    mean_score=np.mean(scores)
    
    return mean_score

def getCosineSimilarity(outputs, pred_outputs):
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    cos_sims = [cos_sim(x, y) for x, y in zip(model.encode(outputs), model.encode(pred_outputs))]

    return np.mean(cos_sims)

def getROUGEScore(outputs, pred_outputs):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for output, pred_output in zip(outputs, pred_outputs):
        scores = scorer.score(output, pred_output)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    mean_rouge1 = np.mean(rouge1_scores)
    mean_rouge2 = np.mean(rouge2_scores)
    mean_rougeL = np.mean(rougeL_scores)
    
    return mean_rouge1, mean_rouge2, mean_rougeL