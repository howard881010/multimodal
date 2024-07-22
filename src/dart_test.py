import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from darts.models import NLinearModel
from darts import TimeSeries
import glob
import os
from tqdm import tqdm
import sys
import os
import pandas as pd
import numpy as np
import wandb
import time
from loguru import logger
from utils import open_record_directory, open_result_directory, rmse, nmae
from finance_multimodal import getLLMTIMERMSE
from transformers import set_seed
from tqdm import tqdm
import glob
import re
import json


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
    batch_size = 2

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


def nlinear_darts(train_input, train_output, test_input, historcial_window_size,train_embedding=None, test_embedding=None):

    # Convert to TimeSeries object required by Darts
    train_series = TimeSeries.from_values(train_input)
    train_output_series = TimeSeries.from_values(train_output)
    if train_embedding is not None:
        train_past_covariates = TimeSeries.from_values(train_embedding)
        test_past_covariates = TimeSeries.from_values(test_embedding)
    else:
        train_past_covariates = None
        test_past_covariates = None
    
    # Define and train the NLinearModel model
    model_NLinearModel = NLinearModel(input_chunk_length=historcial_window_size, output_chunk_length=historcial_window_size, n_epochs=100, pl_trainer_kwargs={"accelerator": "gpu", "devices": 1}, )
    model_NLinearModel.fit(train_series, past_covariates=train_past_covariates, future_covariates=train_output_series)

    pred_value = []
    test = np.array([])
    # Make predictions
    for i in range(len(test_input)):
        test = np.append(test, test_input[i])
        test_series = TimeSeries.from_values(test)
        print("input: ", test_series)
        predictions = model_NLinearModel.predict(n=historcial_window_size, series=test_series, past_covariates=test_past_covariates).all_values().flatten().tolist()
        str_res = ' '.join([str(round(num,2)) for num in predictions])
        print("Prediction: " + str_res)
        pred_value.append(str_res)
    print(pred_value)
    
    return pred_value

def getLLMTIMEOutput(dataset, filename, unit, sub_dir, historical_window_size):
    # filename for train
    data = pd.read_csv(filename)
    train_input_arr = []
    train_output_arr = []
    test_input_arr = []
    test_output_arr = []
    train_summary_arr = []
    test_summary_arr = []
    for idx, row in data.iterrows():
        train_input_arr.append(json.loads(row['input'])["share_price"])
        train_output_arr.append(json.loads(row['output'])["share_price"])
        report = json.loads(row['input'])
        del report['share_price']
        report = json.dumps(report)
        train_summary_arr.append(report)

    train_embedding = bert_model_inference(train_summary_arr)

    filename = filename.replace('train', 'validation')
    data = pd.read_csv(filename)
    for idx, row in data.iterrows():
        test_input_arr.append(json.loads(row['input'])["share_price"])
        test_output_arr.append(json.loads(row['output'])["share_price"])
        report = json.loads(row['input'])
        del report['share_price']
        report = json.dumps(report)
        test_summary_arr.append(report)
    test_embedding = bert_model_inference(test_summary_arr)
    
    train_input_arr = np.array(train_input_arr)
    train_output_arr = np.array(train_output_arr)
    test_input_arr = np.array(test_input_arr)
    test_output_arr = np.array(test_output_arr)

    log_path, res_path = open_record_directory(
        dataset=dataset, sub_dir=sub_dir, unit=unit, filename=filename, model_name="nlinear", historical_window_size=historical_window_size)

    logger.remove()
    logger.add(log_path, rotation="100 MB", mode="w")

    pred_value = nlinear_darts(train_input_arr, train_output_arr, test_input_arr, historical_window_size, train_embedding, test_embedding)
    results = [{"pred_values": pred_value[i], "fut_values": test_output_arr[i]} for i in range(len(data))]
    results = pd.DataFrame(results, columns=['pred_values', 'fut_values'])
    results.to_csv(res_path)
    return res_path



if __name__ == "__main__":
    # add seed
    np.random.seed(42)
    set_seed(42)

    if len(sys.argv) != 5:
        print("Usage: python models/lltime_test.py <dataset> <historical_window_size> <model_name> <case>")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")

    dataset = sys.argv[1]
    historical_window_size = int(sys.argv[2])
    case = int(sys.argv[4])
    model_name = sys.argv[3]

    if case == 1:
        sub_dir = "numerical"
    elif case == 2:
        sub_dir = "mixed-numerical"

    wandb.init(project="Inference",
               config={"name": "nlinear",
                       "window_size": historical_window_size,
                       "dataset": dataset,
                       "model": model_name,
                       "case": sub_dir})
    
    start_time = time.time()
    rmses = []
    errs = []
    nans = []
    nmaes = []

    if dataset == "Climate" or dataset == "Finance":
        unit = "day/"
    else:
        print("No Such Dataset")
        sys.exit(1)

    folder_path = f"Data/{dataset}/{historical_window_size}_{unit}{sub_dir}"
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist")
        sys.exit(1)
    else:
        pattern = "train_*.csv"
        for filepath in tqdm(glob.glob(os.path.join(folder_path, pattern))):
            filename = os.path.basename(filepath)
            if "all" not in filename:
                continue
            else:
        # filepath = "Data/Yelp/4_weeks/test_1.csv"
                out_filename = getLLMTIMEOutput(
                    dataset, filepath, unit, sub_dir, historical_window_size)
                out_rmse, out_err, out_nan, out_nmae = getLLMTIMERMSE(
                    dataset, out_filename, unit, model_name, sub_dir, historical_window_size
                )
                if out_rmse != 0 and str(out_rmse) != "nan":
                    rmses.append(out_rmse)
                errs.append(out_err)
                nans.append(out_nan)
                nmaes.append(out_nmae)
    print("Mean RMSE: " + str(np.mean(rmses)))
    print("Mean Error Rate: " + str(np.mean(errs)))
    print("Mean NaN Rate: " + str(np.mean(nans)))
    print("Std-Dev RMSE: " + str(np.std(rmses)))
    print("Mean nmae: " + str(np.mean(nmaes)))
    wandb.log({"rmse": np.mean(rmses)})
    wandb.log({"error_rate": np.mean(errs)})
    wandb.log({"nan_rate": np.mean(nans)})
    wandb.log({"std-dev": np.std(rmses)})
    wandb.log({"nmae": np.mean(nmaes)})

    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
