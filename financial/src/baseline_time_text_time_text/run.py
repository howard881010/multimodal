# python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B-Instruct --tensor-parallel-size=2 --disable-log-requests
# go to src directory and
# run python -m baseline_time_text_time_text/run.py
import os
os.sys.path.append("..")
from src.vllm import llm_chat, message_template
import time
from openai import OpenAI
from glob import glob
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import ast


openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
# model_name = "llama3-70b"
model = "meta-llama/Meta-Llama-3-70B-Instruct"

if model == "meta-llama/Meta-Llama-3-70B-Instruct":
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )


def llm_chat(messages: list[dict], guided_json=None):
    chat_response = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body={
            "guided_json": guided_json
        }
    )
    return chat_response.choices[0].message.content


def clean_data(data):
    cleaned_data = {}
    for k, v in data.items():
        if type(v) == '' and v == '':
            continue
        elif type(v) == list and len(v) == 0:
            continue
        cleaned_data[k] = v
    return cleaned_data


def format_data_to_string(data):
    result = ""
    if data.get("share_price", None) is not None:
        result += "Today's share price: " + str(data["share_price"])
    for k, v in data.items():
        if k != 'share_price':
            result += f"\nToday's <{k}>: " + str(v)
    return result


prompt = "Given today's share price and stock related summary, predict the next day's share price and summary"

result_schema = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "Summary of next day's article"
        },
        "share_price": {
            "type": "integer",
            "description": "Next day's share price prediction"
        }
    },
    "required": [
        "summary",
        "share_price",
    ]
}

data_dir = "/data/kai/forecasting/data/"
paths = sorted(glob(os.path.join(data_dir, "formatted_v0.2/AMD/*.json")))


def test_text_time(path):
    timestamp = path.split('/')[-1].split(".json")[0]
    with open(path, 'r') as f:
        data = json.load(f)
        cleaned_data = clean_data(data)

    content = format_data_to_string(cleaned_data)
    message = message_template(prompt, content)
    response = llm_chat(message, guided_json=result_schema)
    response = ast.literal_eval(response)

    final_data = {}
    final_data["timestamp"] = timestamp
    final_data["today_price"] = cleaned_data.pop("share_price")
    final_data["predicted_price"] = response["share_price"]
    final_data["today_summary"] = cleaned_data
    final_data["predicted_summary"] = response["summary"]

    print(response)
    return final_data


with ThreadPoolExecutor(max_workers=1) as executor:
    futures = [executor.submit(test_text_time, path) for path in paths]
    results = [future.result() for future in as_completed(futures)]


with open(os.path.join(data_dir, "predction_v0.2/text_time_text_time.json"), 'w') as f:
    json.dump(results, f)
