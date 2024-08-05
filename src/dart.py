import numpy as np
import pandas as pd
from darts.models import NLinearModel
from darts import TimeSeries
import os
import sys
import os
import pandas as pd
import numpy as np
import wandb
import time
from loguru import logger
from utils import open_record_directory, rmse, create_result_file, convertJSONToList
from transformers import set_seed
import json
from datasets import load_dataset

def nlinear_darts(train_input, test_input, window_size, train_embedding=None, test_embedding=None):
    # Convert to TimeSeries object required by Darts
    train_series = TimeSeries.from_values(train_input)
    print("Number of time steps:", train_series.n_timesteps)
    print("Number of dimensions:", train_series.width)
    if train_embedding is not None:
        train_past_covariates = TimeSeries.from_values(train_embedding)
        test_past_covariates = TimeSeries.from_values(test_embedding)
    else:
        train_past_covariates = None
        test_past_covariates = None
    # Define and train the NLinearModel model
    model_NLinearModel = NLinearModel(input_chunk_length=window_size, output_chunk_length=1, pl_trainer_kwargs={"accelerator": "gpu", "devices": 1}, )
    model_NLinearModel.fit(train_series, past_covariates=train_past_covariates)

    pred_value = []
    # test = np.array([])
    # Make predictions
    for i in range(len(test_input)):
        # test = np.append(test, test_input[i])
        test_series = TimeSeries.from_values(np.array(test_input[i]))
        # print("Number of time steps:", test_series.n_timesteps)
        # print("Number of dimensions:", test_series.width)
        predictions = model_NLinearModel.predict(n=1, series=test_series, past_covariates=test_past_covariates).values().tolist()
        pred_value.append(predictions)
    
    return pred_value

def getLLMTIMEOutput(dataset, unit, sub_dir, window_size, key_name):
    # filename for train
    hf_dataset = load_dataset(f"Howard881010/{dataset}-{window_size}_{unit}-{sub_dir.split('/')[0]}")

# Add idx column to each split
    data = pd.DataFrame(hf_dataset['train'])
    train_input_arr = []
    for idx, row in data.iterrows():
         input_json = json.loads(row['input'])
         input_dict_list = [ele[key_name] for ele in input_json.values()]
         input_num = [list(input_dict.values()) for input_dict in input_dict_list]
         train_input_arr.append(input_num[0])
    # train_input_arr = np.array([list(json.loads(row['input']).values())[0][key_name] for idx, row in data.iterrows()])

    data = pd.DataFrame(hf_dataset['validation'])
    # data = data[:-5]
    test_input_arr = []
    test_output_arr = []
    for idx, row in data.iterrows():
        input_num = convertJSONToList(row, idx, key_name, "input")
        output_num = convertJSONToList(row, idx, key_name, "output")
        test_input_arr.append(input_num)
        test_output_arr.append(output_num)
            
    train_embedding = None
    test_embedding = None
    
    log_path, res_path = open_record_directory(
        dataset=dataset, sub_dir=sub_dir, unit=unit, filename="validation", model_name="nlinear", window_size=window_size)

    pred_value = nlinear_darts(np.array(train_input_arr), test_input_arr, window_size, train_embedding, test_embedding)
    results = [{"pred_values": pred_value[i], "fut_values": test_output_arr[i], "input_values": test_input_arr[i]} for i in range(len(data))]
    results = pd.DataFrame(results, columns=['pred_values', 'fut_values', 'input_values'])
    results.to_csv(res_path)
    return res_path

def numberEval(dataset, filename, unit, sub_dir, window_size=1):
    data = pd.read_csv(filename)
    
    data['pred_values'] = data['pred_values'].apply(lambda x: np.array(eval(x)).flatten())
    data['fut_values'] = data['fut_values'].apply(lambda x: np.array(eval(x)).flatten())
    data['input_values'] = data['input_values'].apply(lambda x: np.array(eval(x))[-1:].flatten())
    print(data["input_values"].head(5))
    pred_values_flat = data['pred_values'].values.tolist()
    fut_values_flat = data['fut_values'].values.tolist()
    input_values_flat = data['input_values'].values.tolist()


    test_rmse_loss = rmse(pred_values_flat, fut_values_flat)

    precision = []
    input_values = np.reshape(input_values_flat, -1)
    pred_values = np.reshape(pred_values_flat, -1)
    fut_values = np.reshape(fut_values_flat, -1)

    print(input_values.shape, pred_values.shape, fut_values.shape)

    for input_value, pred_value, fut_value in zip(input_values, pred_values, fut_values):
        if (input_value > pred_value and input_value > fut_value) or \
            (input_value < pred_value and input_value < fut_value) or \
            (input_value == pred_value and input_value == fut_value):
            precision.append(1)
        else:
            precision.append(0)

    path = create_result_file(
        dir = f"Results/{dataset}/{window_size}_{unit}/{sub_dir}",
        filename = (filename.split("/")[-1]).replace("output", "text_score"),
    )
    
    results = [{"test_rmse_loss": test_rmse_loss, "precision": np.mean(precision)}]
    results = pd.DataFrame(results, columns=['test_rmse_loss'])
    results.to_csv(path)

    return test_rmse_loss, np.mean(precision)



if __name__ == "__main__":
    # add seed
    np.random.seed(42)
    set_seed(42)

    if len(sys.argv) != 5:
        print("Usage: python models/lltime_test.py <dataset> <window_size> <model_name> <case>")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")

    dataset = sys.argv[1]
    window_size = int(sys.argv[2])
    case = int(sys.argv[4])
    model_name = sys.argv[3]

    if case == 1:
        sub_dir = "mixed-mixed"
    split = 'validation'

    wandb.init(project="Inference",
               config={"name": "nlinear",
                       "window_size": window_size,
                       "dataset": dataset,
                       "model": model_name,
                       "case": dataset,
                       'split': split})
    
    start_time = time.time()
    if dataset == "gas":
        unit = "week"
        key_name = "gas_price"
    elif dataset == "climate":
        unit = "day"
        key_name = "temperature"

    
    out_filename = getLLMTIMEOutput(
        dataset, unit, sub_dir, window_size, key_name)
    # out_filename = "/home/ubuntu/multimodal/Predictions_and_attempts/gas/2_week/mixed-mixed-west/nlinear_output_validation.csv"
    out_rmse, binary_precision = numberEval(
        dataset, out_filename, unit, sub_dir, window_size
    )
    print("RMSE Scores: ", out_rmse)
    print("Binary Precision: ", binary_precision)
    wandb.log({"RMSE Scores": out_rmse})
    wandb.log({"Binary Precision": binary_precision})

    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
