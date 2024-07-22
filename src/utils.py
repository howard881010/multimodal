import re
import numpy as np
import os
import torch
from transformers import BertTokenizer, BertModel


def is_valid_sequence(sequence, historical_window_size):
    numbers = re.findall(r"\d+\.\d+", sequence)
    try:
        # Attempt to convert the sequence to floats and check its length
        return len(numbers) == historical_window_size
    except ValueError:
        # If conversion fails or length is incorrect, return False
        return False
    
def open_record_directory(dataset, unit, filename, model_name, sub_dir, historical_window_size):

    out_filename = model_name + "_output_" + filename.split("/")[-1]
    log_filename = model_name + "_log_" + filename.split("/")[-1]
    os.makedirs(f"Logs/{dataset}/{historical_window_size}_{unit}/{sub_dir}", exist_ok=True)
    os.makedirs(f"Predictions_and_attempts/{dataset}/{historical_window_size}_{unit}/{sub_dir}", exist_ok=True)
    log_path = f"Logs/{dataset}/{historical_window_size}_{unit}/{sub_dir}/{log_filename}"
    res_path = f"Predictions_and_attempts/{dataset}/{historical_window_size}_{unit}/{sub_dir}/{out_filename}"

    return log_path, res_path


def open_result_directory(dataset, sub_dir, unit, filename, model_name, historical_window_size):
    out_filename = model_name + "_rmse_" + \
        "_".join((filename.split("/")[-1].split("_"))[2:])

    os.makedirs(f"Results/{dataset}/{historical_window_size}_{unit}/{sub_dir}", exist_ok=True)
    out_path = f"Results/{dataset}/{historical_window_size}_{unit}/{sub_dir}/{out_filename}"

    return out_path


def rmse(y_pred, y_true):
    y_pred = np.reshape(y_pred, -1)
    y_true = np.reshape(y_true, -1)
    return np.sqrt(np.square(y_pred - y_true).mean())

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

def create_result_file(dir, filename):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir + "/" + filename

def bert_model_inference(summaries):
    # Set float32 matmul precision to utilize Tensor Cores
    torch.set_float32_matmul_precision('high')  # You can also use 'medium' for less precision but potentially higher performance

    # Load pre-trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Use DataParallel to wrap the model if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    # Move model to the available GPU(s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = model.to(device)
    print(type(summaries[0]))

    # Tokenize summaries
    inputs = tokenizer(summaries, padding=True, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Define batch size
    batch_size = 8

    # Function to get batches
    def get_batches(input_ids, attention_mask, batch_size):
        for i in range(0, len(input_ids), batch_size):
            yield input_ids[i:i + batch_size], attention_mask[i:i + batch_size]

    # Create batches
    batches = list(get_batches(input_ids, attention_mask, batch_size))

    # Perform inference on each batch and collect pooled outputs
    pooled_outputs = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            input_ids_batch, attention_mask_batch = batch
            input_ids_batch = input_ids_batch.to(device)
            attention_mask_batch = attention_mask_batch.to(device)
            outputs = model(input_ids_batch, attention_mask=attention_mask_batch)
            pooled_output = outputs.pooler_output.cpu().numpy()
            pooled_outputs.append(pooled_output)

    pooled_outputs = np.vstack(pooled_outputs)  # Shape: (num_samples, 768)

    return pooled_outputs


