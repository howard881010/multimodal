
from darts.models import TSMixerModel
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

def tsmixer_darts(train_input, test_input, window_size):
    # Convert to TimeSeries object required by Darts
    train_series = TimeSeries.from_values(train_input)
    model_TSMixer = TSMixerModel(input_chunk_length=window_size, output_chunk_length=window_size, pl_trainer_kwargs={"accelerator": "gpu", "devices": 1}, )
    model_TSMixer.fit(train_series)

    pred_value = []
    # Make predictions
    for i in range(len(test_input)):
        test_series = TimeSeries.from_values(np.array(test_input[i]))
        predictions = model_TSMixer.predict(n=window_size, series=test_series).values().tolist()
        pred_value.append(predictions)
    
    return pred_value

def getLLMTIMEOutput(dataset, unit, window_size, split, hf_dataset, model_name):
    # filename for train
    data_all = load_dataset(hf_dataset)

    data = pd.DataFrame(data_all['train'])
    train_input_arr = data['input_time'].apply(lambda x: x[0]).to_list()

    data = pd.DataFrame(data_all[split])
    test_input_arr = data['input_time'].to_list()
    test_output_arr = data['output_time'].to_list()

    log_path, res_path = open_record_directory(
        dataset=dataset, sub_dir='mixed', unit=unit, filename=split, model_name=model_name, window_size=window_size)

    pred_value = tsmixer_darts(np.array(train_input_arr), test_input_arr, window_size)
    results = [{"pred_time": pred_value[i], "output_time": test_output_arr[i]} for i in range(len(test_input_arr))]
    results = pd.DataFrame(results, columns=['pred_time', 'output_time'])
    results.to_csv(res_path)
    return res_path

def numberEval(filename):
    data = pd.read_csv(filename)
    data['pred_time'] = data['pred_time'].apply(lambda x: ast.literal_eval(x))
    data['output_time'] = data['output_time'].apply(lambda x: ast.literal_eval(x))
    rmse_loss = getRMSEScore(data)
    return rmse_loss



if __name__ == "__main__":
    # add seed
    np.random.seed(42)
    set_seed(42)

    if len(sys.argv) != 5:
        print("Usage: python models/lltime_test.py <dataset> <window_size> <model_name> <split>")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")

    dataset = sys.argv[1]
    window_size = int(sys.argv[2])
    model_name = sys.argv[3]
    split = sys.argv[4]
    
    if dataset == "climate":
        unit = "day"
        num_key_name = "temp"
    elif dataset == "medical":
        unit = "day"
    elif dataset == "gas":
        unit = "week"

    hf_dataset = f"Howard881010/{dataset}-{window_size}{unit}-mixed"

    wandb.init(project="Inference-new",
               config={"window_size": f"{window_size}-{window_size}",
                       "dataset": dataset,
                       "model": model_name})
    start_time = time.time()
    
    out_filename = getLLMTIMEOutput(dataset, unit, window_size, split, hf_dataset, model_name)
    out_rmse = numberEval(
        out_filename
    )
    print("RMSE Scores: ", out_rmse)
    wandb.log({"RMSE Scores": out_rmse})

    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
