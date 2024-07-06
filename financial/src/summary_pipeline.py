from collections import defaultdict
import json
from utils.utils import load_json
from transformers import AutoTokenizer
from typing import Union


class DataParser():
    def __init__(self, model="meta-llama/Meta-Llama-3-70B-Instruct", json_schema=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.json_schema = json_schema

    def combine_results(self, results: list[dict]):
        """
        Combines summary of dictionaries to one dictionary.
        Input:
            list of dictionaries: dict_1, dict_2
        Output:
            dict{
                "var_1": [dict_var_1, dict_var_2]
                "var_2": [dict_var_1, dict_var_2]
                ...
            }
        """
        results_dict = defaultdict(list)

        for result in results:
            for key in result:
                if isinstance(result[key], list):
                    results_dict[key].extend(result[key])
                else:
                    results_dict[key].append(result[key])
        return dict(results_dict)

    def split_documents(self, data: Union[str, list[str]], max_token_length: int = 4096, overlap: int = 0):
        """
        Chunk list of documents desired context length.
         Also joins list of strings with '\n'.
        Args:
            data: str or list of str prompts
            max_token_length: max str prompt length (excluding prompt)
            overlap: between documents
        Returns:

        """
        if type(data) == list:
            chunk_data = []
            for document in data:
                token_length = len(self.tokenizer.encode('\n'.join(document)))
                avg_token_per_summary = token_length / len(document)
                document_per_chunk = int(
                    max_token_length / avg_token_per_summary)
                if token_length > max_token_length:
                    chunks = [
                        document[i: i + document_per_chunk]
                        for i in range(0, len(data), document_per_chunk)
                    ]
                    chunk_data += chunks
                else:
                    chunk_data.append(document)
            return chunk_data

        # chunk one long document
        elif type(data) == str:
            avg_token_per_data = len(self.tokenizer.encode(data)) / len(data)
            txt_per_chunk = int(max_token_length / avg_token_per_data)
            if overlap == 0:
                chunk_data = [data[i: i+txt_per_chunk]
                              for i in range(0, len(data), txt_per_chunk)]
            else:
                txt_overlap = int(overlap / avg_token_per_data)
                chunk_data = [data[max(0, i-txt_overlap): i+txt_per_chunk+txt_overlap]
                              for i in range(0, len(data), txt_per_chunk)]
            return chunk_data
