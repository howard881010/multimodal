import pandas as pd
from .metrics import (
    meteorScore,
    cosineSimilarity,
    rougeScore,
    rmse,
    extract_numbers,
    rmse_from_text,
    gptScore)
import numpy as np
import ast
import wandb


class Evaluator():
    """
    based on model_type, returns following score from the columns:
        - output_text
        - output_time
        - pred_text
        - pred_time

    input_copy, time_time
        - RMSE(output_time, pred_time)
    text_time_text_time (use regex to extract numbers)
        - RMSE_from_text(output_text, pred_text)
    input_copy, text_text, text_time_text_time, hybrid
        - Meteor(output_text, pred_text)
        - ROUGE(output_text, pred_text)
        - Cosine(output_text, pred_text)
        - GPTScore(output_text, pred_text)
    """

    def __init__(self, cfg):
        self.dataset_cfg = cfg['dataset']
        self.result_cfg = cfg['results']
        self.load_results()
        self.results = self.inference()

    def load_results(self):
        self.results = pd.read_csv(self.result_cfg['results_path'])

    def normalize_time(self):
        self.results['output_time'] = self.results['output_time'].apply(ast.literal_eval)
        self.results['pred_time'] = self.results['pred_time'].apply(ast.literal_eval)
        combined = np.concatenate([self.results['output_time'].tolist(), self.results['pred_time'].tolist()])
        min_value = np.min(combined)
        max_value = np.max(combined)
        self.results['output_time'] = self.results['output_time'].apply(lambda x: (x - min_value) / (max_value - min_value))
        self.results['pred_time'] = self.results['pred_time'].apply(lambda x: (x - min_value) / (max_value - min_value))

    def inference(self):
        if self.dataset_cfg['model_type'].startswith("nlinear") or self.dataset_cfg['model_type'] == "input_copy":
            self.normalize_time()

        self.results['RMSE'] = self.results.apply(
                    lambda row: rmse(row['output_time'], row['pred_time']), axis=1)
        
        if not self.dataset_cfg['model_type'].startswith("nlinear"):
            self.results['Meteor'] = self.results.apply(
                lambda row: meteorScore(row['output_text'], row['pred_text']), axis=1)
            self.results['CosineSimilarity'] = self.results.apply(
                lambda row: cosineSimilarity(row['output_text'], row['pred_text']), axis=1)
            self.results['ROUGE'] = self.results.apply(
                lambda row: rougeScore(row['output_text'], row['pred_text']), axis=1)
            self.results['GPTScore'] = self.results.apply(
            lambda row: gptScore(row['output_text'], row['pred_text']), axis=1)
        
        return self.results

    def getMeteorScore(self):
        if 'Meteor' not in self.results:
            return np.nan
        return self.results['Meteor'].mean()

    def getCosineSimilarity(self):
        if 'CosineSimilarity' not in self.results:
            return np.nan
        return self.results['CosineSimilarity'].mean()

    def getROUGEScore(self):
        if 'ROUGE' not in self.results:
            return [np.nan] * 3
        results = self.results['ROUGE'].tolist()
        return np.mean(results, axis=0)

    def getRMSEScore(self):
        if 'RMSE' not in self.results:
            return np.nan
        return self.results['RMSE'].mean()

    def getGPTScore(self):
        if 'GPTScore' not in self.results:
            return np.nan
        return self.results['GPTScore'].mean()

    def push_to_wandb(self):
        wandb.init(project=self.result_cfg['wandb_project'],
               config={"window_size": f"{self.dataset_cfg['input_window']}-{self.dataset_cfg['output_window']}",
                       "dataset": self.dataset_cfg['name'],
                       "model": self.dataset_cfg['model_type']})
        wandb.log({"Meteor Scores": self.getMeteorScore()})
        wandb.log({"Cos Sim Scores": self.getCosineSimilarity()})
        wandb.log({"Rouge1 Scores": self.getROUGEScore()[0]})
        wandb.log({"Rouge2 Scores": self.getROUGEScore()[1]})
        wandb.log({"RougeL Scores": self.getROUGEScore()[2]})
        wandb.log({"RMSE Scores": self.getRMSEScore()})
        wandb.log({"GPT Scores": self.getGPTScore()})

        wandb.finish()