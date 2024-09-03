import os
from utils import find_text_parts, split_text
from datasets import load_dataset
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluator.gpt_evaluator import GPT4Semantic, GPT4Accuracy
import time
import wandb
import sys

def processData(window_size, dataset, model, text_pattern, number_pattern, case):
    output_dir = f"/home/ubuntu/multimodal/Data/{dataset}-GPT4-Evaluation/{model}/{window_size}{unit}"
    filename = "processed.csv"
    hf_dataset = f"Howard881010/{dataset}-{window_size}{unit}-{model.split('-')[-1]}"

    data_all = load_dataset(hf_dataset)
    data = pd.DataFrame(data_all['test'])
    print(data.columns)
    pred_output_column = f'pred_output_case{case}'

    output_texts = data['output_text'].apply(lambda x: split_text(x, text_pattern)).to_list()
    pred_texts = data[pred_output_column].apply(lambda x: find_text_parts(x, number_pattern)).apply(lambda x: split_text(x, text_pattern)).to_list()
    for idx, pred_text in enumerate(pred_texts):
        if len(pred_text) > window_size:
            pred_texts[idx] = pred_text[:window_size]
        while len(pred_text) < window_size:
            pred_texts[idx].append(None)

    output_texts = np.reshape(output_texts, -1)
    pred_texts = np.reshape(pred_texts, -1)

    results = pd.DataFrame({"output_text": output_texts, "pred_text": pred_texts})

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    results.to_csv(output_path, index=False)

def wait_for_completion(job_id, processor, poll_interval=100):
    status = processor.check_status(job_id)
    while status.status not in ["completed", "failed"]:
        print(f"Current status: {status}. Waiting for {poll_interval} seconds...")
        time.sleep(poll_interval)
        status = processor.check_status(job_id)
    return status.status

def calculate_metrics(window_size, unit, dataset, model, text_pattern, number_pattern, case):
    processData(window_size, dataset, model, text_pattern, number_pattern, case)

    gpt4semantic = GPT4Semantic()

    results_dir = f"/home/ubuntu/multimodal/Data/{dataset}-GPT4-Evaluation/{model}/{window_size}{unit}"
    data = pd.read_csv(f"/home/ubuntu/multimodal/Data/{dataset}-GPT4-Evaluation/{model}/{window_size}{unit}/processed.csv")

    jsonl_path = os.path.join(results_dir, "batch.jsonl")
    semantic_output_path = os.path.join(results_dir, "semantic.txt")

    semantic_batch_object_id = gpt4semantic.create_and_run_batch_job(data, jsonl_path, output_text_column="output_text",
                                    pred_text_column="pred_text")

    job_status = wait_for_completion(semantic_batch_object_id, gpt4semantic)

    if job_status == "completed":
        print("Batch job completed successfully!")
        semantic_outputs = gpt4semantic.check_status_and_parse(semantic_batch_object_id , semantic_output_path)
        semantic_score, count_none = gpt4semantic.calculate_metrics(semantic_outputs)


    gpt4accuracy = GPT4Accuracy()
    accuracy_output_path = os.path.join(results_dir, "accuracy.txt")
    accuracy_batch_object_id = gpt4accuracy.create_and_run_batch_job(data, jsonl_path, output_text_column="output_text",
                                    pred_text_column="pred_text")

    job_status = wait_for_completion(accuracy_batch_object_id, gpt4accuracy)
    if job_status == "completed":
        print("Batch job completed successfully!")
        accuracy_outputs = gpt4accuracy.check_status_and_parse(accuracy_batch_object_id, accuracy_output_path)
        precisions, recalls, f1_scores = gpt4accuracy.calculate_metrics(accuracy_outputs)

    results = {"semantic_score": semantic_score, "count_none": count_none, "precisions": precisions, "recalls": recalls, "f1_scores": f1_scores}
    results = pd.DataFrame.from_dict(results, orient="index").T
    results.to_csv(os.path.join(results_dir, "results.csv"))
    
    return semantic_score, count_none, precisions, recalls, f1_scores

if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("Usage: python models/lltime_test.py <dataset> <window_size> <case> <method>")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    dataset = sys.argv[1]
    window_size = int(sys.argv[2])
    case = int(sys.argv[3])
    method = sys.argv[4]
    
    if dataset == "climate":
        unit = "day"
        num_key_name = "temp"
        text_key_name = "weather_forecast"
    elif dataset == "medical":
        unit = "day"
        num_key_name = "Heart_Rate"
    elif dataset == "gas":
        unit = "week"
        num_key_name = "gas_price"
    if case == 1:
        model = "text2text"
    elif case == 2:
        model = "textTime2textTime"
    elif case == 3:
        model = "textTime2text"
    elif case == 4:
        model = "textTime2time"

    model = model + "-" + method
    number_pattern = fr"{unit}_\d+_{num_key_name}: ?'?([\d.]+)'?"
    text_pattern =fr'({unit}_\d+_date:\s*\S+\s+{unit}_\d+_{text_key_name}:.*?)(?=\s{unit}_\d+_date|\Z)'

        
    
    wandb.init(project="gpt-score",
                config={"window_size": f"{window_size}-{window_size}",
                        "dataset": dataset,
                        "model": model})

    semantic_score, count_none, precisions, recalls, f1_scores = calculate_metrics(window_size, unit, dataset, model, text_pattern, number_pattern, case)

    wandb.log({"semantic_score": semantic_score, "count_none": count_none, "precisions": precisions, "recalls": recalls, "f1_scores": f1_scores})
    wandb.finish()