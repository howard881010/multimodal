
from darts.models import NLinearModel
from darts import TimeSeries
import os
import sys
import pandas as pd
import numpy as np
import wandb
import time
from utils import open_record_directory, find_text_parts, split_text
from transformers import set_seed
from datasets import load_dataset
import ast
from text_evaluation import getRMSEScore
from sentence_transformers import SentenceTransformer


def textEmbedding(summaries):
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    embeddings = []
    for summary in summaries:
        embeddings.append(model.encode(summary))

    return np.array(embeddings)

def nlinear_darts(train_input, test_input, train_text, test_text, window_size):
    # Convert to TimeSeries object required by Darts
    train_series = TimeSeries.from_values(train_input)
    print(textEmbedding(train_text).shape)
    train_embedding = TimeSeries.from_values(textEmbedding(train_text))
    model_NLinearModel = NLinearModel(input_chunk_length=window_size, output_chunk_length=window_size, pl_trainer_kwargs={"accelerator": "gpu", "devices": 1}, )
    model_NLinearModel.fit(train_series, past_covariates=train_embedding)

    pred_value = []
    # Make predictions
    for i in range(len(test_input)):
        test_series = TimeSeries.from_values(np.array(test_input[i]))
        test_embedding = TimeSeries.from_values(textEmbedding(test_text[i]))
        predictions = model_NLinearModel.predict(n=window_size, series=test_series, past_covariates=test_embedding).values().tolist()
        pred_value.append(predictions)
    
    return pred_value

def getLLMTIMEOutput(dataset, unit, window_size, split, hf_dataset, text_pattern, num_pattern):
    # filename for train
    data_all = load_dataset(hf_dataset)

    data = pd.DataFrame(data_all['train'])
    train_input_arr = data['input_time'].apply(lambda x: x[0]).to_list()
    train_text_arr = data['input'].apply(lambda x: find_text_parts(x, num_pattern)).apply(lambda x: split_text(x, text_pattern)[0]).to_list()

    data = pd.DataFrame(data_all[split])
    test_input_arr = data['input_time'].to_list()
    test_output_arr = data['output_time'].to_list()
    test_text_arr = data['input'].apply(lambda x: find_text_parts(x, num_pattern)).apply(lambda x: split_text(x, text_pattern)).to_list()

    log_path, res_path = open_record_directory(
        dataset=dataset, sub_dir='mixed', unit=unit, filename=split, model_name="nlinear_text", window_size=window_size)

    pred_value = nlinear_darts(np.array(train_input_arr), test_input_arr, train_text_arr, test_text_arr, window_size)
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
        text_key_name = "weather_forecast"
        num_key_name = "temp"
    elif dataset == "medical":
        unit = "day"
    elif dataset == "gas":
        unit = "week"

    hf_dataset = f"Howard881010/{dataset}-{window_size}{unit}-mixed"
    num_pattern = fr"{unit}_\d+_{num_key_name}: '([\d.]+)'"
    text_pattern =fr'({unit}_\d+_date:\s*\S+\s+{unit}_\d+_{text_key_name}:.*?)(?=\s{unit}_\d+_date|\Z)'
    print(text_pattern)

    wandb.init(project="Inference-new",
               config={"window_size": f"{window_size}-{window_size}",
                       "dataset": dataset,
                       "model": model_name})
    start_time = time.time()
    
    out_filename = getLLMTIMEOutput(
        dataset, unit, window_size, split, hf_dataset,text_pattern, num_pattern)
    out_rmse = numberEval(
        out_filename
    )
    print("RMSE Scores: ", out_rmse)
    wandb.log({"RMSE Scores": out_rmse})

    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
