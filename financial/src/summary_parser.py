from collections import defaultdict
from transformers import AutoTokenizer
from typing import Union
import numpy as np
import ast


class DataParser():
    def __init__(self, model="meta-llama/Meta-Llama-3-70B-Instruct", json_schema=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.json_schema = json_schema

    # raw_text_to_documents
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

    # documents_to_summary
    # Given documents, we format the metrics with collapse_metrics
    # Afterwards, we chunk them with chunk_summaries and send it to the model
    # Our output is then collapsed again with collapse_metrics

    # we can format again with collapse_results then collapse_metrics
    # combine_results give you total string length

    def collapse_metrics(self, json_summaries: Union[np.ndarray, list[dict], str]):
        """
        Gets a list of summary dictionaries and collapse them into one dictionary{list}.
        Note that JSON Schema was defined as "metric": metric, "value": value through prompt engineering.

        Returns list of collapsed dictionary {key: [{metric:value}, ...]}
        """
        if type(json_summaries) == np.ndarray:
            data = [{k: v for k, v in ast.literal_eval(
                summary).items() if v} for summary in json_summaries]
        elif type(json_summaries) == list:
            data = [{k: v for k, v in summary.items() if v}
                    for summary in json_summaries]
        elif type(json_summaries) == str:
            data = [{k: v for k, v in ast.literal_eval(
                json_summaries).items()}]
        collapsed_data = []

        for entry in data:
            new_entry = {}
            for key, value in entry.items():
                if isinstance(value, list):
                    new_value = []
                    for item in value:
                        if isinstance(item, dict):
                            if 'metric' in item and 'value' in item:
                                new_value.append(
                                    f"{item['metric']}: {item['value']}")
                            elif 'key' in item and 'value' in item:
                                new_value.append(
                                    f"{item['key']}: {item['value']}")
                        else:
                            new_value.append(item)
                    new_entry[key] = new_value
                else:
                    new_entry[key] = value
            if len(new_entry) != 0:
                collapsed_data.append(new_entry)

        return collapsed_data

    def chunk_summaries(self, summaries: list[dict], max_split_token=4096):
        """
        Chunks summaries into summaries so that the {key, item} does not cut off.

        Returns a list of summaries in string.
        """
        chunks = []
        for summary in summaries:
            for k, val in summary.items():
                new_chunk = f"{k} [{val}]" if k == "summary" else f"{k} {val}"

                if not chunks or len(self.tokenizer.encode(chunks[-1] + ", " + new_chunk)) > max_split_token:
                    chunks.append(new_chunk)
                else:
                    chunks[-1] += ", " + new_chunk
        return chunks

    def collapse_results(self, results):
        """
        Combine results from vLLM call.
        """
        result = []
        for r in results:
            result += r
        return result

    def combine_results(self, results):
        results_dict = defaultdict(list)

        for result in results:
            if type(result) == str:
                json_data = ast.literal_eval(result)
            else:
                json_data = result
            for key in json_data:
                # For lists of dictionaries (like 'key_numbers')
                if isinstance(json_data[key], list):
                    results_dict[key].extend(json_data[key])
                else:
                    results_dict[key].append(json_data[key])
        return results_dict
