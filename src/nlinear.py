
from darts.models import NLinearModel
from darts import TimeSeries
import os
import sys
import pandas as pd
import numpy as np
import wandb
import time
from utils import open_record_directory
from transformers import set_seed
from datasets import load_dataset
import ast
from text_evaluation import getRMSEScore

def nlinear_darts(train_input, test_input, window_size):
    # Convert to TimeSeries object required by Darts
    train_series = TimeSeries.from_values(train_input)
    model_NLinearModel = NLinearModel(input_chunk_length=window_size, output_chunk_length=window_size, pl_trainer_kwargs={"accelerator": "gpu", "devices": 1}, )
    model_NLinearModel.fit(train_series)

    pred_value = []
    # Make predictions
    for i in range(len(test_input)):
        test_series = TimeSeries.from_values(np.array(test_input[i]))
        predictions = model_NLinearModel.predict(n=window_size, series=test_series).values().tolist()
        pred_value.append(predictions)
    
    return pred_value

def getLLMTIMEOutput(dataset, unit, window_size, split, hf_dataset):
    # filename for train
    data_all = load_dataset(hf_dataset)

    data = pd.DataFrame(data_all['train'])
    train_input_arr = data['input_num'].apply(lambda x: x[0]).to_list()

    data = pd.DataFrame(data_all[split])
    test_input_arr = data['input_num'].to_list()
    test_output_arr = data['output_num'].to_list()

    log_path, res_path = open_record_directory(
        dataset=dataset, sub_dir='mixed', unit=unit, filename=split, model_name="nlinear", window_size=window_size)

    pred_value = nlinear_darts(np.array(train_input_arr), test_input_arr, window_size)
    results = [{"pred_num": pred_value[i], "output_num": test_output_arr[i]} for i in range(len(test_input_arr))]
    results = pd.DataFrame(results, columns=['pred_num', 'output_num'])
    results.to_csv(res_path)
    return res_path

def numberEval(filename):
    data = pd.read_csv(filename)
    data['pred_num'] = data['pred_num'].apply(lambda x: ast.literal_eval(x))
    data['output_num'] = data['output_num'].apply(lambda x: ast.literal_eval(x))
    rmse_loss = getRMSEScore(data)
    return rmse_loss



if __name__ == "__main__":
    # add seed
    np.random.seed(42)
    set_seed(42)

    if len(sys.argv) != 3:
        print("Usage: python models/lltime_test.py <dataset> <window_size>")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")

    dataset = sys.argv[1]
    window_size = int(sys.argv[2])
    split = "test"
    
    if dataset == "climate":
        unit = "day"
    elif dataset == "medical":
        unit = "day"
    elif dataset == "gas":
        unit = "week"

    hf_dataset = f"Howard881010/{dataset}-{window_size}{unit}-finetuned"

    wandb.init(project="Inference-nlinear",
               config={"window_size": f"{window_size}-{window_size}",
                       "dataset": dataset,
                       "model": "nlinear"})
    start_time = time.time()
    
    out_filename = getLLMTIMEOutput(dataset, unit, window_size, split, hf_dataset)
    out_rmse = numberEval(
        out_filename
    )
    print("RMSE Scores: ", out_rmse)
    wandb.log({"RMSE Scores": out_rmse})

    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
