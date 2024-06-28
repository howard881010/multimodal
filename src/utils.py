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
    
def open_record_directory(dataset, historical_window_size, unit, filename, model_name, transform=False, in_context=True):
    trans = "_trans" if transform == True else ""
    context = "_no_in_context" if in_context == False else ""

    out_filename = model_name + trans + context + "_output_" + \
        "_".join((filename.split("/")[-1].split("_"))[1:])
    log_filename = model_name + trans + context + "_log_" + \
        "_".join((filename.split("/")[-1].split("_"))[1:])

    os.makedirs(
        "Logs/"
        + str(dataset)
        + "/LLMTIME/"
        + str(historical_window_size)
        + "_"
        + unit,
        exist_ok=True,
    )
    os.makedirs(
        "Predictions_and_attempts/"
        + str(dataset)
        + "/LLMTIME/"
        + str(historical_window_size)
        + "_"
        + unit,
        exist_ok=True,
    )
    out_path = (
        f"Logs/"
        + str(dataset)
        + "/LLMTIME/"
        + str(historical_window_size)
        + "_"
        + unit
        + out_filename
    )
    log_path = (
        f"Logs/"
        + str(dataset)
        + "/LLMTIME/"
        + str(historical_window_size)
        + "_"
        + unit
        + log_filename
    )
    res_path = (
        f"Predictions_and_attempts/"
        + str(dataset)
        + "/LLMTIME/"
        + str(historical_window_size)
        + "_"
        + unit
        + out_filename
    )

    return log_path, out_path, res_path


def open_result_directory(dataset, historical_window_size, unit, filename, model_name, transform=False, in_context=True):
    if transform:
        if in_context:
            out_filename = model_name + "_trans_rmse_" + \
                "_".join((filename.split("/")[-1].split("_"))[3:])
        else:
            out_filename = model_name + "_trans_no_in_context_rmse_" + \
                "_".join((filename.split("/")[-1].split("_"))[6:])
    else:
        if in_context:
            out_filename = model_name + "_rmse_" + \
                "_".join((filename.split("/")[-1].split("_"))[2:])
        else:
            out_filename = model_name + "_no_in_context_rmse_" + \
                "_".join((filename.split("/")[-1].split("_"))[5:])

    os.makedirs(
        "Results/"
        + str(dataset)
        + "/LLMTIME/"
        + str(historical_window_size)
        + "_"
        + unit,
        exist_ok=True,
    )
    out_path = (
        "Results/"
        + str(dataset)
        + "/LLMTIME/"
        + str(historical_window_size)
        + "_"
        + unit
        + out_filename
    )

    return out_path


def rmse(y_pred, y_true, penalty):
    valid_idx = ~np.isnan(y_pred)
    invalid_len = np.sum(~valid_idx)
    err_terms = invalid_len * penalty / len(y_pred)
    return np.sqrt(np.square(y_pred[valid_idx] - y_true[valid_idx]).mean() + err_terms)


def create_batched(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]

