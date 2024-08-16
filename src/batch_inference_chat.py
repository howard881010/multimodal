from utils import create_batched
import json
from tqdm import tqdm
import re

def batch_inference_llama_summary(
    results,
    model_chat,
    data,
    logger,
    unit,
    num_key_name
):
    batches = list(create_batched(data, 32))
    err_idx = []

    for batch in tqdm(batches):
        prompt, cur_idx = create_batch_prompt(batch)
        print("chat start")
        output_texts = model_chat.chat(prompt)

        for index, output_text in enumerate(output_texts):
            response = output_text.split("assistant")[-1]

            logger.info("Content for row: " + str(cur_idx[index]) + " Content: " + prompt[index][-1]['content'])
            logger.info("Response for row: " + str(cur_idx[index]) +  " Content: " + response)

            num_pattern = fr"<{unit}_\d+_{num_key_name}> : '([\d.]+)'"
            num_matches = re.findall(num_pattern, response)
            # Format the extracted temperatures
            formatted_nums = [[float(temp)] for temp in num_matches]

            # Remove the number part in the text (like <day_1_temp>)
            modified_text = re.sub(num_pattern, "", response)
            try:
                results[cur_idx[index]] = (
                    {"pred_summary": modified_text, "pred_num": formatted_nums})
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                err_idx.append(cur_idx[index])
                print(f"An error occurred: {e} for row: {cur_idx[index]}")

    return err_idx


def create_batch_prompt(data):
    prompt = []
    cur_idx = []

    for index, row in data.iterrows():
        content = [{"role": "system", "content": row['instruction']}, {"role": "user", "content": row['input']}]
        # content = [{"role": "user", "content": row_content}]

        prompt.append(content)
        cur_idx.append(row['idx'])

    return prompt, cur_idx
