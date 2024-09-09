
from darts.models import NLinearModel
from darts import TimeSeries
import os
import sys
import pandas as pd
import numpy as np
import wandb
import time
from utils import open_record_directory, split_text
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

def getLLMTIMEOutput(dataset, unit, window_size, split, hf_dataset, text_key_name):
    # filename for train
    data_all = load_dataset(hf_dataset)

    data = pd.DataFrame(data_all['train'])
    train_input_arr = data['input_num'].apply(lambda x: x[0]).to_list()
    train_text_arr = data['input_text'].apply(lambda x: split_text(x, text_key_name, 0)[0]).to_list()

    data = pd.DataFrame(data_all[split])
    test_input_arr = data['input_num'].to_list()
    test_output_arr = data['output_num'].to_list()
    test_text_arr = data['input_text'].apply(lambda x: split_text(x, text_key_name, 0)).to_list()

    log_path, res_path = open_record_directory(
        dataset=dataset, sub_dir='mixed', unit=unit, filename=split, model_name="nlinear_text", window_size=window_size)

    pred_value = nlinear_darts(np.array(train_input_arr), test_input_arr, train_text_arr, test_text_arr, window_size)
    results = [{"pred_num": pred_value[i], "output_num": test_output_arr[i]} for i in range(len(test_input_arr))]
    results = pd.DataFrame(results, columns=['pred_num', 'output_num'])
    results.to_csv(res_path)
    return res_path

def numberEval(filename):
    data = pd.read_csv(filename)
    pred_values = data['pred_num'].apply(lambda x: ast.literal_eval(x)).to_list()
    fut_values = data['output_num'].apply(lambda x: ast.literal_eval(x)).to_list()
    rmse_loss = getRMSEScore(pred_values, fut_values)
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
        text_key_name = "weather_forecast"
        num_key_name = "temp"
    elif dataset == "medical":
        unit = "day"
        text_key_name = "medical_notes"
        num_key_name = "Heart_Rate"

    hf_dataset = f"Howard881010/{dataset}-{window_size}{unit}-finetuned"

    wandb.init(project="Inference-nlinear",
               config={"window_size": f"{window_size}-{window_size}",
                       "dataset": dataset,
                       "model": "nlinear_textEmbedding"})
    start_time = time.time()
    
    out_filename = getLLMTIMEOutput(
        dataset, unit, window_size, split, hf_dataset, text_key_name)
    out_rmse = numberEval(
        out_filename
    )
    print("RMSE Scores: ", out_rmse)
    wandb.log({"RMSE Scores": out_rmse})

    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
