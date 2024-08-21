import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluator.gpt_evaluator import ClimateProcessor
import pandas as pd
import json
climate_processor = ClimateProcessor()

results_dir = "/home/ubuntu/multimodal/Data"
results = pd.read_csv("/home/ubuntu/multimodal/Data/weather_raw.csv")

jsonl_path = os.path.join(results_dir, "batch.jsonl")
output_path = os.path.join(results_dir, "evaluator.txt")

batch_object_id = climate_processor.create_and_run_batch_job(results, jsonl_path, input_text_column="Text")

print(climate_processor.check_status(batch_object_id))
outputs = climate_processor.check_status_and_parse(batch_object_id, output_path)

outputs = climate_processor.parse_output(output_path)

filename = "/home/ubuntu/multimodal/Data/outputs.jsonl"

# Writing to JSONL
with open(filename, 'w') as jsonlfile:
    for entry in outputs:
        # Convert dictionary to JSON string and write to file
        jsonlfile.write(json.dumps(entry) + '\n')

print(f"Data has been written to {filename}")
