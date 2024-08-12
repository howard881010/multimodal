import sys
import os
import pandas as pd
import numpy as np
import wandb
import time
from transformers import set_seed
from datasets import load_dataset
import pandas as pd
import numpy as np
from nltk.translate import meteor
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from rouge_score import rouge_scorer
from utils import rmse
import nltk
from openai import OpenAI
import os
nltk.download('punkt')
nltk.download('wordnet')

def getMeteorScore(df):
    print("df columns:",  df.columns)
    scores = [meteor([word_tokenize(x['output_text'])], word_tokenize(x['pred_text'])) for idx, x in df.iterrows()]
    mean_score=np.mean(scores)
    
    return mean_score

def getCosineSimilarity(df):
    
    ground_truth = df['output_text'].tolist()
    zero_shot = df['pred_text'].tolist()
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    ground_truth_embeddings = model.encode(ground_truth)
    zero_shot_embeddings = model.encode(zero_shot)
    zs_cos_sims = [cos_sim(x, y) for x, y in zip(ground_truth_embeddings, zero_shot_embeddings)]

    return np.mean(zs_cos_sims)


def getROUGEScore(df):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for idx, row in df.iterrows():
        scores = scorer.score(row['output_text'], row['pred_text'])
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

    gpt_scores = []    
    for idx, row in df.iterrows():
        question = f"summary1: {row['output_text']} summary2: {row['pred_text']}"
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

def getRMSEScore(df):

    fut_values = df['output_time'].tolist()
    pred_values = df['pred_time'].tolist()
                
    rmse_loss = rmse(np.array(fut_values), np.array(pred_values))
    
    return rmse_loss

def getTextScore(case, num_key_name, split,hf_dataset):
    data_all = load_dataset(hf_dataset)
    data = pd.DataFrame(data_all['train'])

    meteor_score = getMeteorScore(data)
    # cosine_similarity_score = getCosineSimilarity(data)
    cosine_similarity_score = np.nan
    rouge1, rouge2, rougeL = getROUGEScore(data)
    # gpt_score = getGPTScore(data)
    gpt_score = np.nan

    if case == "mixed-mixed":
        rmse_loss = getRMSEScore(data)
    else:
        rmse_loss = np.nan
    

    return meteor_score, cosine_similarity_score, rouge1, rouge2, rougeL, rmse_loss, gpt_score




if __name__ == "__main__":
    # add seed
    np.random.seed(42)
    set_seed(42)

    if len(sys.argv) != 7:
        print("Usage: python models/lltime_test.py <dataset> <window_size> <model_name> <case> <finetune> <split>")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")

    dataset = sys.argv[1]
    window_size = int(sys.argv[2])
    case = int(sys.argv[4])
    model_name = sys.argv[3]
    finetune = sys.argv[5]
    split = sys.argv[6]
    if case == 2:
        case = "mixed-mixed"
    elif case == 1:
        case = "text-text"

    if dataset == "climate":
        specified_region = "cal"
        unit = "day"
        num_key_name = "temperature"
    elif dataset == "medical":
        specified_region = None
        unit = "day"
        num_key_name = "Heart_Rate"
    elif dataset == "gas":
        specified_region = "west"
        unit = "week"
        num_key_name = "gas_price"

    wandb.init(project="Inference",
               config={"name": "hybrid",
                       "window_size": window_size,
                       "dataset": dataset,
                       "model": model_name,
                       "case": case,
                       "specified_region": specified_region,
                       "split": split})
    start_time = time.time()

    hf_dataset = "kaimkim/climate_1_1_pred"

    # out_filename = getSummaryOutput(
    #     dataset, unit, model_name, model_chat, sub_dir, window_size, split, hf_dataset
    # )
    meteor_score, cos_sim_score, rouge1, rouge2, rougeL, rmse_loss, gpt_score = getTextScore(
        case, num_key_name, split, hf_dataset
    )

    
    print("Meteor Scores: ", meteor_score)
    print("Cos Sim Scores: ", cos_sim_score)
    print("Rouge1 Scores: ", rouge1)
    print("Rouge2 Scores: ", rouge2)
    print("RougeL Scores: ", rougeL)
    print("RMSE Scores: ", rmse_loss)
    print("GPT Scores: ", np.mean(gpt_score))
    wandb.log({"Meteor Scores": meteor_score})
    wandb.log({"Cos Sim Scores": cos_sim_score})
    wandb.log({"Rouge1 Scores": rouge1})
    wandb.log({"Rouge2 Scores": rouge2})
    wandb.log({"RougeL Scores": rougeL})
    wandb.log({"RMSE Scores": rmse_loss})
    wandb.log({"GPT Scores": np.mean(gpt_score)})
    
    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
