import re
from serialize import serialize_arr, deserialize_str
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


def transform_input_str(input_str, settings, dataset, max_out=1, min_out=0, alpha=0.95, beta=0.3, basic=False):
    if dataset == "Yelp":
        decimal_place = 2
    elif dataset == "Mimic" or dataset == "Climate":
        decimal_place = 3
    input_arr = np.array([float(num) for num in input_str.split()])
    # scaler = get_scaler(input_arr, max_out, min_out, decimal_place = decimal_place,
    #                     alpha=alpha, beta=beta, basic=basic)
    # transform_input_arr = scaler.transform(input_arr)
    # print("transform_input_arr: ", transform_input_arr)
    input_str = serialize_arr(input_arr, settings)
    input_str = input_str.rstrip(" ,")

    return input_str


def handle_response(response, settings):
    numbers = re.findall(r"\d+\.\d+|\d+", response)
    numbers_str = ' ,'.join(numbers)
    print("numbers_str: ", numbers_str)
    deserialized_arr = deserialize_str(numbers_str, settings)

    return deserialized_arr


@dataclass
class Scaler:
    """
    Represents a data scaler with transformation and inverse transformation functions.

    Attributes:
        transform (callable): Function to apply transformation.
        inv_transform (callable): Function to apply inverse transformation.
    """
    transform: callable = lambda x: x
    inv_transform: callable = lambda x: x


def get_scaler(history, max_out, min_out, decimal_place=2, alpha=0.95, beta=0.3, basic=False):
    """
    Generate a Scaler object based on given history data.

    Args:
        history (array-like): Data to derive scaling from.
        alpha (float, optional): Quantile for scaling. Defaults to .95.
        # Truncate inputs
        tokens = [tokeniz]
        beta (float, optional): Shift parameter. Defaults to .3.
        basic (bool, optional): If True, no shift is applied, and scaling by values below 0.01 is avoided. Defaults to False.

    Returns:
        Scaler: Configured scaler object.
    """
    history = history[~np.isnan(history)]
    if basic:
        q = np.maximum(np.quantile(np.abs(history), alpha), .001)

        def transform(x):
            return x / q

        def inv_transform(x):
            return np.round(np.clip(x * q, min_out, max_out), decimal_place)
    else:
        min_ = np.min(history) - beta*(np.max(history)-np.min(history))
        q = np.quantile(history-min_, alpha)
        if q == 0:
            q = 1

        def transform(x):
            return (x - min_) / q

        def inv_transform(x):
            return np.round(np.clip(x * q + min_, min_out, max_out), decimal_place)
    return Scaler(transform=transform, inv_transform=inv_transform)


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

