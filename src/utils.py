import re
import numpy as np
from dataclasses import dataclass
import os
import wandb
from num2words import num2words


def is_valid_sequence(sequence, historical_window_size):
    numbers = re.findall(r"\d+\.\d+", sequence)
    try:
        # Attempt to convert the sequence to floats and check its length
        return len(numbers) == historical_window_size
    except ValueError:
        # If conversion fails or length is incorrect, return False
        return False
    
def open_record_directory(dataset, unit, filename, model_name, sub_dir, historical_window_size):

    out_filename = model_name + "_output_" + \
        "_".join((filename.split("/")[-1].split("_"))[1:])
    log_filename = model_name + "_log_" + \
        "_".join((filename.split("/")[-1].split("_"))[1:])
    if historical_window_size != 1:
        sub_dir = str(historical_window_size) + "_" + unit + sub_dir

    os.makedirs(f"Logs/{dataset}/{sub_dir}", exist_ok=True)
    os.makedirs(f"Predictions_and_attempts/{dataset}/{sub_dir}", exist_ok=True)
    log_path = f"Logs/{dataset}/{sub_dir}/{log_filename}"
    res_path = f"Predictions_and_attempts/{dataset}/{sub_dir}/{out_filename}"

    return log_path, res_path


def open_result_directory(dataset, sub_dir, unit, filename, model_name, historical_window_size):
    out_filename = model_name + "_rmse_" + \
        "_".join((filename.split("/")[-1].split("_"))[2:])
    if historical_window_size != 1:
        sub_dir = str(historical_window_size) + "_" + unit + sub_dir

    os.makedirs(f"Results/{dataset}/{sub_dir}", exist_ok=True)
    out_path = f"Results/{dataset}/{sub_dir}/{out_filename}"

    return out_path


def rmse(y_pred, y_true, penalty):
    y_pred = np.reshape(y_pred, -1)
    y_true = np.reshape(y_true, -1)
    valid_idx = ~np.isnan(y_pred)
    invalid_len = np.sum(~valid_idx)
    err_terms = invalid_len * penalty / len(y_pred)
    return np.sqrt(np.square(y_pred[valid_idx] - y_true[valid_idx]).mean() + err_terms)

def nmae(y_pred, y_true, penalty):
    nmaes = []
    mae = []
    for row_A, row_B in zip(y_true, y_pred):
        # Skip rows where all values are NaN in either A or B
        if np.isnan(row_A).all() or np.isnan(row_B).all():
            continue
        
        # Calculate MSE for the current row
        mae = np.mean(np.abs(row_A - row_B))
        # Normalize the MSE
        if len(row_A) == 1:
            value_range = row_A[0]
        else:
            value_range = max(row_A) - min(row_A)
        if value_range == 0:
            continue
        nmae = mae / value_range
        
        # Append to the list of NMSEs
        nmaes.append(nmae)

    # Calculate the mean NMSE
    mean_nmae = np.mean(nmaes)
    
    return mean_nmae


def create_batched(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]

