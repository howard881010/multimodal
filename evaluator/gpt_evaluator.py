from openai import OpenAI
import os
import pandas as pd
from ast import literal_eval
import json

"""

GPT4 Evaluagtor USAGE
=================================


from evaluator.gpt_evaluator import GPT4Evaluator
gpt4_evaluator = GPT4Evaluator()

jsonl_path = os.path.join(results_dir, "batch.jsonl")
output_path = os.path.join(results_dir, "evaluator.txt")

batch_object_id = gpt4_evaluator.create_and_run_batch_job(results, jsonl_path, 
                        output_text_column="output_text",
                        pred_text_column="pred_text")

# if ready, outputs will be returned
outputs = gpt4_evaluator.check_status_and_parse(batch_object_id, output_path)

# # if you have the results saved already, just call
# outputs = gpt4_evaluator.parse_output(output_path)

precisions, recalls, f1_scores = gpt4_evaluator.calculate_metrics(outputs)

=================================
"""


class OpeanAIBatchProcessor():
    def __init__(self, instruction: str, json_schema: dict, prompt: str, description="Multimodal Forecasting"):
        self.key = os.environ.get("OPENAI_APIKEY")
        self.client = OpenAI(api_key=self.key)
        self.instruction = instruction
        self.json_schema = json_schema
        self.prompt = prompt
        self.description = description

    def get_messages(self, target_text, pred_text) -> list[dict]:
        prompt = self.prompt.format(
            target_text=target_text, pred_text=pred_text)
        messages = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": prompt},
        ]
        return messages

    def create_and_run_batch_job(self, results: pd.DataFrame, jsonl_path: str,
                                 output_text_column="output_text",
                                 pred_text_column="pred_text") -> str:
        """
            Creates json batch from results dataframe containing ['output_text', 'pred_text']
        """

        assert ".jsonl" in jsonl_path
        batch_jsons = self.create_batch_jsons(
            results, output_text_column, pred_text_column)
        self.save_batch_json(batch_jsons, jsonl_path)

        batch_object_id = self.create_batch(jsonl_path)
        return batch_object_id

    def check_status_and_parse(self, batch_object_id, output_path: str) -> list[dict]:
        assert ".txt" in output_path
        status = self.check_status(batch_object_id)

        # check if job is done
        if status.output_file_id is None:
            print("Not ready yet")
            return status

        # save output
        self.save_batch_response(status.output_file_id, output_path)
        return self.parse_output(output_path)

    def parse_output(self, output_path: str) -> list[dict]:
        """
        Parse the output content from batch job by reading output_path.txt file
        """
        # parse output
        parsed_contents = []
        with open(output_path, 'r') as f:
            for line in f:
                output = json.loads(line)
                content = literal_eval(
                    output['response']['body']['choices'][0]['message']['content'])
                parsed_contents.append(content)
        return parsed_contents

    def create_batch_jsons(self, results: pd.DataFrame,
                           output_text_column,
                           pred_text_column) -> list[dict]:
        """
        Creates json batch from results dataframe containing ['output_text', 'pred_text']
        """
        batch_jsons = []
        for idx, row in results.iterrows():
            messages = self.get_messages(
                row[output_text_column], row[pred_text_column])
            batch_json = {
                "custom_id": f"{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": messages,
                    # "response_format": {
                    #     "type": "json_schema",
                    #     "json_schema": self.json_schema
                    # }
                }
            }
            batch_jsons.append(batch_json)
        return batch_jsons

    def save_batch_json(self, batch_jsons: list, jsonl_path: str) -> None:
        """
        Saves batch_jsons into jsonl_path
        """
        with open(jsonl_path, 'w') as file:
            for batch_json in batch_jsons:
                file.write(json.dumps(batch_json) + '\n')

    def create_batch(self, jsonl_path: str) -> str:
        batch_input_file = self.client.files.create(
            file=open(jsonl_path, "rb"),
            purpose="batch"
        )

        batch_object = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": self.description
            }
        )
        print(f"batch job created with batch_object_id \n {batch_object.id}")
        return batch_object.id

    def check_status(self, batch_object_id):
        status = self.client.batches.retrieve(batch_object_id)
        return status

    def save_batch_response(self, output_file_id, output_path: str) -> None:
        file_response = self.client.files.content(output_file_id)
        with open(output_path, "w") as file:
            file.write(file_response.text)

    def gpt_call(self, target_text, pred_text):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.get_messages(target_text, pred_text),
            response_format={"type": "json_schema",
                             "json_schema": self.json_schema}

        )
        return response.choices[0].message.content


class GPT4Semantic(OpeanAIBatchProcessor):
    def __init__(self):
        instruction = \
"""You are a helpful assistant capable of evaluating the semantic similarity between two summaries. 
The semantic score you provide should be a number between 1 and 10, where 10 represents the highest level of semantic similarity 
(meaning the summaries convey almost identical information), and 1 represents the lowest level of semantic similarity (meaning the summaries convey entirely different or unrelated information). 
The score should reflect how closely the meanings and key details of the two summaries align. You should only give me the number, nothing else."""

#         instruction = \
#                     """You are a helpful assistant capable of evaluating the semantic accuracy of a predicted weather forecast compared to the ground truth weather forecast for the same day.
# Your task is to assess how closely the meaning of the predicted forecast aligns with the ground truth. Consider factors such as the overall weather conditions, temperature ranges, precipitation chances, and any other relevant details. 

# Output a semantic accuracy score between 0 and 100, where 100 represents perfect alignment and 0 represents no alignment. Also, include a brief explanation for your evaluation.
# """

        json_schema = {
            "name": "evaluate_semantic_accuracy",
            "description": "Assesses the semantic accuracy of predicted weather forecasts compared to the ground truth and categorizes the accuracy.",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                    },
                },
                "additionalProperties": False,
                "required": ["score"]
            },
        }
        prompt = "Ground truth weather forecast: {target_text} \n\n\n Predicted weather forecast: {pred_text}"
        super().__init__(instruction, json_schema, prompt)

class GPT4Accuracy(OpeanAIBatchProcessor):

    def __init__(self):

        instruction = \
            """You are a helpful assistant capable of evaluating the similarity, accuracy, and consistency of a ground truth weather forecast, and predicted weather forecast for the same day.
By comparing each sentences step by step, output the following number of true_positive, false_positive, and false_negative number of facts:
- (True Positive): Correct information presented in the prediction,
- (False Postivie): Incorrect information presented in the prediction,
- (False Negative): Information not presented in the prediction.
"""

        json_schema = {
            "name": "evaluate_weather_forecast",
            "description": "Compares ground truth and predicted weather forecasts and counts the number of true positive, false positive, false negative facts.",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "explanation": {
                        "type": "string",
                    },
                    "true_positive": {
                        "type": "number",
                    },
                    "false_positive": {
                        "type": "number",
                    },
                    "false_negative": {
                        "type": "number",
                    },
                },
                "additionalProperties": False,
                "required": ["explanation", "true_positive", "false_positive", "false_negative"]
            },
        }
        prompt = "Ground truth weather forecast: {target_text} \n\n\n Predicted weather forecast: {pred_text}"
        super().__init__(instruction, json_schema, prompt)

    def calculate_metrics(self, outputs):
        precisions = []
        recalls = []
        f1_scores = []
        for row in outputs:
            tp = row['true_positive']
            fp = row['false_positive']
            fn = row['false_negative']

            if (tp + fp) > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0.0

            if (tp + fn) > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0.0

            if (precision + recall) > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)
        return precisions, recalls, f1_scores

class ClimateProcessor(OpeanAIBatchProcessor):

    def __init__(self):

        instruction = \
            """
Summarize the following climate report by identifying the following key elements in a JSON format with key as category and value as a list. Return empty string if not explicitly stated:

1) Temperature Trend: [Recent observations and fluctuations in temperature with relevant context, time, and location.],
2) Precipitation Trend: [Recent observations and fluctuations in precipitation with relevant context, time, and location.],
3) Humidity Trend: [Recent observations and fluctuations in humidity with relevant context, time, and location.],
4) Wind Speed Trend: [Recent observations and fluctuations in wind speed with relevant context, time, and location.],
5) Overall Summary: [A comprehensive summary of the climate conditions, trends observed in temperature, precipitation, humidity, and wind speed, and any relevant future forecast for the climate.]
"""

        json_schema = {
            "name": "evaluate_climate_report",
            "description": "Summarizes the climate report by identifying the key elements in a JSON format with key as category and value as a list.",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "temperature_trend": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                    "precipitation_trend": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                    "humidity_trend": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                    "wind_speed_trend": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                    "overall_summary": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                },
                "additionalProperties": False,
                "required": ["temperature_trend", "precipitation_trend", "humidity_trend", "wind_speed_trend", "overall_summary"]
            },
        }
        prompt = "Climate report: {target_text}"
        super().__init__(instruction, json_schema, prompt)
    
    def get_messages(self, target_text) -> list[dict]:
        prompt = self.prompt.format(
            target_text=target_text)
        messages = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": prompt},
        ]
        return messages

    def create_and_run_batch_job(self, results: pd.DataFrame, jsonl_path: str,
                                 input_text_column: str) -> str:
        """
            Creates json batch from results dataframe containing ['output_text', 'pred_text']
        """

        assert ".jsonl" in jsonl_path
        batch_jsons = self.create_batch_jsons(
            results, input_text_column)
        self.save_batch_json(batch_jsons, jsonl_path)

        batch_object_id = self.create_batch(jsonl_path)
        return batch_object_id

    def create_batch_jsons(self, results: pd.DataFrame,
                           input_text_column,) -> list[dict]:
        """
        Creates json batch from results dataframe containing ['output_text', 'pred_text']
        """
        batch_jsons = []
        for idx, row in results.iterrows():
            messages = self.get_messages(
                row[input_text_column])
            batch_json = {
                "custom_id": f"{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": messages,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": self.json_schema
                    }
                }
            }
            batch_jsons.append(batch_json)
        return batch_jsons

    def gpt_call(self, target_text):
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.get_messages(target_text),
                response_format={"type": "json_schema",
                                "json_schema": self.json_schema}

            )
            return response.choices[0].message.content
