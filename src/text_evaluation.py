import pandas as pd
import numpy as np
from nltk.translate import meteor
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from rouge_score import rouge_scorer
from utils import rmse, find_text_parts
import nltk
from datasets import load_dataset
from utils import find_num_parts, split_text, find_text_parts
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

def getTextScore(case, split, hf_dataset, text_key_name, num_key_name, window_size):
    data_all = load_dataset(hf_dataset)
    data = pd.DataFrame(data_all[split])
    pred_output_column = f'pred_output_case{case}'
    # number part evaluation
    if case in [2, 4]:
        data['pred_time'] = data[pred_output_column].apply(lambda x: find_num_parts(x, num_key_name, window_size))
        data_clean = data.dropna()
        drop_rate = (len(data) - len(data_clean)) / len(data)
        rmse_loss = getRMSEScore(data_clean)
    else:
        rmse_loss = np.nan
        drop_rate = np.nan
        
    # text part evaluation
    if case in [1, 2, 3]:
        output_texts = data['output_text'].apply(lambda x: split_text(x, text_key_name, window_size)).to_list()
        pred_texts = data[pred_output_column].apply(lambda x: split_text(x, text_key_name, window_size)).to_list()
        for idx, pred_text in enumerate(pred_texts):
            if len(pred_text) > window_size:
                pred_texts[idx] = pred_text[:window_size]
            while len(pred_text) < window_size:
                pred_texts[idx].append("No prediction")
        
        output_texts = np.reshape(output_texts, -1)
        pred_texts = np.reshape(pred_texts, -1)
        print(len(output_texts), len(pred_texts))
        indices_to_drop = [idx for idx, pred_text in enumerate(pred_texts) if "No prediction" in pred_text]
        text_drop_count = len(indices_to_drop)
        print(text_drop_count)
        output_texts = np.delete(output_texts, indices_to_drop)
        pred_texts = np.delete(pred_texts, indices_to_drop)

        
        meteor_score = getMeteorScore(output_texts, pred_texts)
        cosine_similarity_score = getCosineSimilarity(output_texts, pred_texts)
        rouge1, rouge2, rougeL = getROUGEScore(output_texts, pred_texts)
    else:
        meteor_score = np.nan
        cosine_similarity_score = np.nan
        rouge1 = np.nan
        rouge2 = np.nan
        rougeL = np.nan
    
    return meteor_score, cosine_similarity_score, rouge1, rouge2, rougeL, rmse_loss, drop_rate, text_drop_count