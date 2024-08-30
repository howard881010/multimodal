from utils import create_batched
from tqdm import tqdm
import re
import random

def batch_inference(
    results,
    model_chat,
    data,
    logger,
    num_pattern,
):
    batches = list(create_batched(data, 8))

    for batch in tqdm(batches):
        prompt, cur_idx = create_batch_prompt(batch)
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

def batch_inference_inContext(
    results,
    model_chat,
    data,
    logger,
    num_pattern,
    data_train
):
    batches = list(create_batched(data, 2))

    for batch in tqdm(batches):
        prompt, cur_idx = create_batch_prompt_in_Context(batch, data_train)
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

def create_batch_prompt_in_Context(data, data_train):
    prompt = []
    cur_idx = []
    randaom_loc = random.randint(0, len(data_train) - 1)
    example_input = data_train.iloc[randaom_loc]['input']
    example_output = data_train.iloc[randaom_loc]['output']

    for index, row in data.iterrows():
        content = [{"role": "system", "content": row['instruction']}, {"role": "user", "content": example_input}, {"role": "assistant", "content": example_output}, {"role": "user", "content": row['input']}]
        prompt.append(content)
        cur_idx.append(row['idx'])

    return prompt, cur_idx
