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


def rmse(y_pred, y_true):
    y_pred = np.reshape(y_pred, -1)
    y_true = np.reshape(y_true, -1)
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

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

def add_idx(example, idx):
    example['idx'] = idx
    return example

def split_data(file_path, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1):
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Calculate the validation and test sizes
    val_size = validation_ratio / (test_ratio + validation_ratio)
    
    # Split the data into train and temporary datasets
    train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio), random_state=42, shuffle=False)
    
    # Split the temporary dataset into validation and test datasets
    validation_data, test_data = train_test_split(temp_data, test_size=val_size, random_state=42, shuffle=False)
    
    # Save the datasets
    
    dir = file_path.split('/')[:-1]
    dir = '/'.join(dir)
    file_name = file_path.split('/')[-1]
    train_data.to_csv(f'{dir}/train_{file_name}', index=False)
    validation_data.to_csv(f'{dir}/val_{file_name}', index=False)
    test_data.to_csv(f'{dir}/test_{file_name}', index=False)

    return


def convert_to_parquet(dataframe_test, dataframe_train, dataframe_val):
    train = pd.concat(dataframe_train)
    test = pd.concat(dataframe_test)
    val = pd.concat(dataframe_val)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    val = val.reset_index(drop=True)

    train_dataset = Dataset.from_pandas(train[['input', 'output', 'instruction','pred_output']])
    test_dataset = Dataset.from_pandas(test[['input', 'output', 'instruction','pred_output']])
    val_dataset = Dataset.from_pandas(val[['input', 'output', 'instruction','pred_output']])
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    return dataset_dict

def load_from_huggingface(dataset, dataset_path, case, units):
    dataset = load_dataset(dataset_path)

    if not os.path.exists(f"../Data/{dataset}/{units}/{case}"):
        os.makedirs(f"../Data/{dataset}/{units}/{case}")

# Access the train split
    for split in ['train', 'validation', 'test']:
        train_dataset = dataset[split]

        # Convert the dataset to a Pandas DataFrame
        df = train_dataset.to_pandas()
        # Save the DataFrame to a CSV file
        df.to_csv(f"../Data/{dataset}/{units}/{case}/{split}_all.csv", index=False)
    
    return 

def combine_window(df, window_size, unit):
    json_data = []
    end_index = len(df) - window_size

    for i in range(end_index):
        combined_input = {}
        combine_output = {}
        
        for j in range(window_size):
            input_key = f"{unit}_{j+1}"
            combined_input[input_key] = json.loads(df.iloc[i + j]['input'])

        output_key = f"{unit}_{window_size+1}"
        
        combine_output[output_key] = json.loads(df.iloc[i + window_size - 1]['output'])

        combine_json = {
            "input": json.dumps(combined_input),
            "output": json.dumps(combine_output),
            "instruction": df.iloc[i]['instruction'],
            "pred_output": df.iloc[i]['pred_output']
        }
        json_data.append(combine_json)
    
    json_df = pd.DataFrame(json_data)
    return json_df

def convertJSONToList(row, idx, key_name, col_name):
    try:
        res = json.loads(row[col_name])
        num_dict_list = [ele[key_name] for ele in res.values()]
        if all(isinstance(num, (int, float)) for num in num_dict_list):
            return num_dict_list
        else:
            num_list = [list(num_dict.values()) for num_dict in num_dict_list]
            return num_list
    except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"An error occurred: {e}, row: {idx}")
