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
from utils import open_record_directory, rmse, create_result_file, convertJSONToList, clean_num
from transformers import set_seed
from datasets import load_dataset
import torch
from transformers import BertTokenizer, BertModel
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
    batch_size = 4

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

def nlinear_darts(train_input, test_input, window_size, train_embedding=None, test_embedding=None):
    # Convert to TimeSeries object required by Darts
    train_series = TimeSeries.from_values(train_input)
    print("Number of time steps:", train_series.n_timesteps)
    print("Number of dimensions:", train_series.width)
    if len(train_embedding) > 0:
        print("type of train_embedding:", type(train_embedding))
        train_embedding = bert_model_inference(train_embedding)
        train_past_covariates = TimeSeries.from_values(train_embedding)
    else:
        train_past_covariates = None
    # Define and train the NLinearModel model
    model_NLinearModel = NLinearModel(input_chunk_length=window_size, output_chunk_length=1, pl_trainer_kwargs={"accelerator": "gpu", "devices": 1}, )
    model_NLinearModel.fit(train_series, past_covariates=train_past_covariates)

    pred_value = []
    # Make predictions
    for i in range(len(test_input)):
        test_series = TimeSeries.from_values(np.array(test_input[i]))
        print("Number of time steps:", test_series.n_timesteps)
        print("Number of dimensions:", test_series.width)
        if len(test_embedding) > 0:
            embedding = bert_model_inference(test_embedding[i])
            test_past_covariates = TimeSeries.from_values(embedding)
            predictions = model_NLinearModel.predict(n=1, series=test_series, past_covariates=test_past_covariates).values().tolist()
        else:
            predictions = model_NLinearModel.predict(n=1, series=test_series).values().tolist()
        pred_value.append(predictions)
    
    return pred_value

def getLLMTIMEOutput(dataset, unit, sub_dir, window_size, num_key_name, split, case):
    # filename for train
    hf_dataset = load_dataset(f"Howard881010/{dataset}-{window_size}_{unit}-{sub_dir}")

    train_embeddings = []
    test_embeddings = []
    train_input_arr = []
    test_input_arr = []
    test_output_arr = []

    data = pd.DataFrame(hf_dataset['train'])
    
    # add the numerical part
    for idx, row in data.iterrows():
        input_num = convertJSONToList(row, idx, num_key_name, "input", float)
        train_input_arr.append(input_num[0])
    # add the summary part
    if case == 2:
        data = clean_num(data, num_key_name)
        for idx, row in data.iterrows():
            data_dict = json.loads(row['input'])
            data_list = [json.dumps(data_dict[key], indent=4) for key in sorted(data_dict.keys())]
            train_embeddings.append(data_list[0])
    print("type of train_embeddings:", type(train_embeddings))
    data = pd.DataFrame(hf_dataset[split])
    # data = data[:-1]
    # add the numerical part
    for idx, row in data.iterrows():
        input_num = convertJSONToList(row, idx, num_key_name, "input", float)
        output_num = convertJSONToList(row, idx, num_key_name, "output", float)
        if type(input_num) == type(output_num):
            test_input_arr.append(input_num)
            test_output_arr.append(output_num)
    # add the summary part
    if case == 2:
        data = clean_num(data, num_key_name)
        for idx, row in data.iterrows():
            data_dict = json.loads(row['input'])
            data_list = [json.dumps(data_dict[key], indent=4) for key in sorted(data_dict.keys())]
            test_embeddings.append(data_list)

    log_path, res_path = open_record_directory(
        dataset=dataset, sub_dir=sub_dir, unit=unit, filename=split, model_name="nlinear", window_size=window_size)

    pred_value = nlinear_darts(np.array(train_input_arr), test_input_arr, window_size, train_embeddings, test_embeddings)
    results = [{"pred_values": pred_value[i], "fut_values": test_output_arr[i], "input_values": test_input_arr[i]} for i in range(len(test_input_arr))]
    results = pd.DataFrame(results, columns=['pred_values', 'fut_values', 'input_values'])
    results.to_csv(res_path)
    return res_path

def numberEval(dataset, filename, unit, sub_dir, window_size=1):
    data = pd.read_csv(filename)
    
    data['pred_values'] = data['pred_values'].apply(lambda x: np.array(eval(x)).flatten())
    # for case input == output
    # data['pred_values'] = data['input_values'].apply(lambda x: np.array(eval(x)).flatten())
    data['fut_values'] = data['fut_values'].apply(lambda x: np.array(eval(x)).flatten())
    pred_values_flat = data['pred_values'].values.tolist()
    fut_values_flat = data['fut_values'].values.tolist()


    test_rmse_loss = rmse(pred_values_flat, fut_values_flat)

    path = create_result_file(
        dir = f"Results/{dataset}/{window_size}_{unit}/{sub_dir}",
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

    if len(sys.argv) != 6:
        print("Usage: python models/lltime_test.py <dataset> <window_size> <model_name> <case> <split>")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")

    dataset = sys.argv[1]
    window_size = int(sys.argv[2])
    case = int(sys.argv[4])
    model_name = sys.argv[3]
    split = sys.argv[5]
    
    if case == 1:
        runs_name = "nlinear"
    elif case == 2:
        runs_name = "nlinear_embedding"
    
    
    if dataset == "gas":
        unit = "week"
        num_key_name = "gas_price"
        specified_region = "west"
    elif dataset == "climate":
        unit = "day"
        num_key_name = "temperature"
        specified_region = "cal"
    elif dataset == "medical":
        unit = "day"
        num_key_name = "Heart_Rate"
        specified_region = None


    wandb.init(project="Inference",
               config={"name": "nlinear",
                       "window_size": window_size,
                       "dataset": dataset,
                       "model": model_name,
                       "case": runs_name,
                       "specified_region": specified_region,
                       'split': split})

    if specified_region is not None:
        sub_dir = f"mixed-mixed-{specified_region}"
    else:
        sub_dir = "mixed-mixed"
    start_time = time.time()
    
    out_filename = getLLMTIMEOutput(
        dataset, unit, sub_dir, window_size, num_key_name, split, case)
    # out_filename = "/home/ubuntu/multimodal/Predictions_and_attempts/gas/2_week/mixed-mixed-west/nlinear_output_validation.csv"
    out_rmse = numberEval(
        dataset, out_filename, unit, sub_dir, window_size
    )
    print("RMSE Scores: ", out_rmse)
    wandb.log({"RMSE Scores": out_rmse})

    end_time = time.time()
    print("Total Time: " + str(end_time - start_time))
