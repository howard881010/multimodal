import re
import numpy as np
import os
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, DatasetDict, Dataset
import json


def is_valid_sequence(sequence, window_size):
    numbers = re.findall(r"\d+\.\d+", sequence)
    try:
        # Attempt to convert the sequence to floats and check its length
        return len(numbers) == window_size
    except ValueError:
        # If conversion fails or length is incorrect, return False
        return False
    
def open_record_directory(dataset, unit, filename, model_name, sub_dir, window_size):

    out_filename = model_name + "_output_" + filename.split("/")[-1] + ".csv"
    log_filename = model_name + "_log_" + filename.split("/")[-1] + ".csv"
    os.makedirs(f"Logs/{dataset}/{window_size}_{unit}/{sub_dir}", exist_ok=True)
    os.makedirs(f"Predictions_and_attempts/{dataset}/{window_size}_{unit}/{sub_dir}", exist_ok=True)
    log_path = f"Logs/{dataset}/{window_size}_{unit}/{sub_dir}/{log_filename}"
    res_path = f"Predictions_and_attempts/{dataset}/{window_size}_{unit}/{sub_dir}/{out_filename}"

    return log_path, res_path


def open_result_directory(dataset, sub_dir, unit, filename, model_name, window_size):
    out_filename = model_name + "_rmse_" + \
        "_".join((filename.split("/")[-1].split("_"))[2:])

    os.makedirs(f"Results/{dataset}/{window_size}_{unit}/{sub_dir}", exist_ok=True)
    out_path = f"Results/{dataset}/{window_size}_{unit}/{sub_dir}/{out_filename}"

    return out_path


def normalize_together(pred, truth):
    combined = np.concatenate((pred, truth))
    min_val = np.min(combined)
    max_val = np.max(combined)
    print(min_val, max_val)
    pred_norm = (pred - min_val) / (max_val - min_val)
    truth_norm = (truth - min_val) / (max_val - min_val)
    return pred_norm, truth_norm


def rmse(y_pred, y_true):
    y_pred = np.reshape(y_pred, -1)
    y_true = np.reshape(y_true, -1)
    # pred_norm, ground_truth_norm = normalize_together(y_pred, y_true)
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

def create_batched(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]

def create_result_file(dir, filename):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir + "/" + filename

def find_text_parts(text, num_pattern):
    modified_text = re.sub(num_pattern, "\r", text)
    
    return modified_text

def find_num_parts(text, num_pattern, window_size):
    num_matches = re.findall(num_pattern, text)
    if len(num_matches) != window_size:
        return np.nan
    else:
        return  [[float(temp)] for temp in num_matches]

def split_text(text, text_pattern):
    text_matches = re.findall(text_pattern, text)

    return text_matches