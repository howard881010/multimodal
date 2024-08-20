from utils import create_batched
import json
from tqdm import tqdm
import re

def batch_inference_llama_summary(
    results,
    model_chat,
    data,
    logger,
    num_pattern,
):
    batches = list(create_batched(data, 24))
    example_input = data.iloc[0]['input']
    example_output = data.iloc[0]['output']

    for batch in tqdm(batches):
        prompt, cur_idx = create_batch_prompt(batch)
        # prompt, cur_idx = create_batch_prompt_in_Context(batch, example_input, example_output)
        output_texts = model_chat.chat(prompt)

        for index, output_text in enumerate(output_texts):
            response = output_text.split("assistant")[-1]

            logger.info("Content for row: " + str(cur_idx[index]) + " Content: " + prompt[index][-1]['content'])
            logger.info("Response for row: " + str(cur_idx[index]) +  " Content: " + response)

            num_matches = re.findall(num_pattern, response)
            if len(num_matches) == 0:
                formatted_nums = []
            else:
                formatted_nums = [[float(temp)] for temp in num_matches]
            
            results[cur_idx[index]] = (
                    {"pred_output": response, "pred_time": formatted_nums})


def create_batch_prompt(data):
    prompt = []
    cur_idx = []

    for index, row in data.iterrows():
        content = [{"role": "system", "content": row['instruction']}, {"role": "user", "content": row['input']}]
        prompt.append(content)
        cur_idx.append(row['idx'])

    return prompt, cur_idx

def create_batch_prompt_in_Context(data, example_input, example_output):
    prompt = []
    cur_idx = []

    for index, row in data.iterrows():
        content = [{"role": "system", "content": example_input}, {"role": "user", "content": example_output}, {"role": "user", "content": row['input']}]
        prompt.append(content)
        cur_idx.append(row['idx'])

    return prompt, cur_idx
