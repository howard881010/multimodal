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
from utils import open_record_directory, rmse, nmae, create_result_file, bert_model_inference
from transformers import set_seed
from tqdm import tqdm
import glob
import re
import json

def nlinear_darts(train_input, test_input, historcial_window_size, train_embedding=None, test_embedding=None):

    # Convert to TimeSeries object required by Darts
    train_series = TimeSeries.from_values(train_input)
    if train_embedding is not None:
        train_past_covariates = TimeSeries.from_values(train_embedding)
        test_past_covariates = TimeSeries.from_values(test_embedding)
    else:
        train_past_covariates = None
        test_past_covariates = None
    # Define and train the NLinearModel model
    model_NLinearModel = NLinearModel(input_chunk_length=historcial_window_size, output_chunk_length=historcial_window_size, pl_trainer_kwargs={"accelerator": "gpu", "devices": 1}, )
    model_NLinearModel.fit(train_series, past_covariates=train_past_covariates)

    pred_value = []
    # test = np.array([])
    # Make predictions
    for i in range(len(test_input)):
        # test = np.append(test, test_input[i])
        print(test_input[i])
        test_series = TimeSeries.from_values(test_input[i])
        predictions = model_NLinearModel.predict(n=historcial_window_size, series=test_series, past_covariates=test_past_covariates).all_values().flatten().tolist()
        str_res = ' '.join([str(round(num,3)) for num in predictions])
        pred_value.append(str_res)
    
    return pred_value

def getLLMTIMEOutput(dataset, filename, unit, sub_dir, historical_window_size):
    # filename for train
    data = pd.read_csv(filename)
    train_input_arr = np.array([list(json.loads(row['input']).values())[0]['gas_price'] for idx, row in data.iterrows()])
    train_summary_arr = [list(json.loads(row['input']).values())[0]['summary'] for idx, row in data.iterrows()]

    filename = filename.replace('train', 'validation')
    data = pd.read_csv(filename)
    test_input_arr = np.array([])
    test_output_arr = np.array([])
    for idx, row in data.iterrows():
        input_json = json.loads(row['input'])
        output_json = json.loads(row['output'])
        test_input_arr = np.append(test_input_arr, [[ele['gas_price'] for ele in input_json.values()]])
        test_output_arr = np.append(test_output_arr, [[ele['gas_price'] for ele in output_json.values()]])
        # test_summary_arr = [json.loads(row['input']).values()['summary'] for idx, row in data.iterrows()]
    
    p
    # train_embedding = bert_model_inference(train_summary_arr)
    # test_embedding = bert_model_inference(test_summary_arr)
    train_embedding = None
    test_embedding = None
    
    log_path, res_path = open_record_directory(
        dataset=dataset, sub_dir=sub_dir, unit=unit, filename=filename, model_name="nlinear", historical_window_size=historical_window_size)

    pred_value = nlinear_darts(train_input_arr, test_input_arr, historical_window_size, train_embedding, test_embedding)
    results = [{"pred_values": pred_value[i], "fut_values": test_output_arr[i]} for i in range(len(data))]
    results = pd.DataFrame(results, columns=['pred_values', 'fut_values'])
    results.to_csv(res_path)
    return res_path

def getLLMTIMERMSE(dataset, filename, unit, sub_dir, historical_window_size=1):
    data = pd.read_csv(filename)
    data["fut_values"] = data["fut_values"].apply(str)
    data["pred_values"] = data["pred_values"].apply(str)

    gt_values = []
    pred_values = []
    for index, row in data.iterrows():
        fut_vals = [float(ele) for ele in row["fut_values"].split()]
        pred_vals = [float(ele) for ele in row["pred_values"].split()]
        gt_values.append(fut_vals)
        pred_values.append(pred_vals)

    test_rmse_loss = rmse(np.array(pred_values), np.array(gt_values))

    path = create_result_file(
        dir = f"Results/{dataset}/{historical_window_size}_{unit}/{sub_dir}",
        filename = (filename.split("/")[-1]).replace("output", "text_score"),
    )
    
    results = [{"test_rmse_loss": test_rmse_loss}]
    results = pd.DataFrame(results, columns=['test_rmse_loss'])
    results.to_csv(path)

    return test_rmse_loss



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
        sub_dir = "mixed-mixed"

    wandb.init(project="Inference",
               config={"name": "nlinear",
                       "window_size": historical_window_size,
                       "dataset": dataset,
                       "model": model_name,
                       "case": sub_dir})
    
    start_time = time.time()
    rmses = []
    unit = "week"

    folder_path = f"Data/{dataset}/{historical_window_size}_{unit}/{sub_dir}"
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
                out_rmse = getLLMTIMERMSE(
                    dataset, out_filename, unit, sub_dir, historical_window_size
                )
                if out_rmse != 0 and str(out_rmse) != "nan":
                    rmses.append(out_rmse)
    print("Mean RMSE: " + str(np.mean(rmses)))
    print("Std-Dev RMSE: " + str(np.std(rmses)))
    wandb.log({"rmse": np.mean(rmses)})
    wandb.log({"std-dev": np.std(rmses)})

    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
