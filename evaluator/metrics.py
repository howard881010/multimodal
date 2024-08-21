import re
import numpy as np
import os
from nltk.translate import meteor
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from rouge_score import rouge_scorer
import nltk
from openai import OpenAI
import random
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')



def meteorScore(output_text, pred_text):
    scores = meteor([word_tokenize(output_text)], word_tokenize(pred_text))
    mean_score=np.mean(scores)
    
    return mean_score

def cosineSimilarity(output_text, pred_text):
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    cos_sim_score = cos_sim(model.encode(output_text), model.encode(pred_text))

    return cos_sim_score

def rougeScore(output_text, pred_text):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(output_text, pred_text)

    return [scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure]

def rmse(output_time, pred_time):
    return np.sqrt(np.mean((output_time - pred_time) ** 2))

def extract_numbers(text):
    return

def rmse_from_text(output_text, pred_text):
    output_numbers = extract_numbers(output_text)
    pred_numbers = extract_numbers(pred_text)
    return np.sqrt(np.mean((output_numbers - pred_numbers) ** 2))

def gptScore(output_text, pred_text):
    # Example using OpenAI's GPT model
    key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=key)
    random.seed(42)
    np.random.seed(42)

    question = f"summary1: {output_text} summary2: {pred_text}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant capable of evaluating the semantic similarity between two summaries. The semantic score you provide should be a number between 1 and 10, where 10 represents the highest level of semantic similarity (meaning the summaries convey almost identical information), and 1 represents the lowest level of semantic similarity (meaning the summaries convey entirely different or unrelated information). The score should reflect how closely the meanings and key details of the two summaries align. You should only give me the number, nothing else."},
            {"role": "user", "content": question},
        ]
    )
    number = re.findall(r'\d+', response.choices[0].message.content)
    # I not sure if this is the best way to do this
    if len(number) > 0:
        gpt_score = float(number[0])
    else:
        gpt_score = 0
    print(gpt_score)
    return gpt_score
