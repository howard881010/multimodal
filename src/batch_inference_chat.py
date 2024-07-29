from utils import create_batched
from num2words import num2words
from modelchat import MistralChatModel, LLMChatModel, GemmaChatModel
import re
import json
from tqdm import tqdm

def batch_inference_llama_summary(
    results,
    model_chat,
    data,
    logger
):
    batches = list(create_batched(data, 16))
    err_idx = []

    for batch in tqdm(batches):
        prompt, cur_idx = create_batch_prompt(batch)

        output_texts = model_chat.chat(prompt)

        for index, output_text in enumerate(output_texts):
            response = output_text.split('[/INST]')[-1]
            # response = output_text.split("\n")[-1]

            logger.info("Content for row: " + str(cur_idx[index]) + " Content: " + prompt[index][-1]['content'])
            logger.info("Response for row: " + str(cur_idx[index]) +  " Content: " + response)

            first_brace_index = response.find('{')
            end_brace_index = response.rfind('}')
            json_response = response[first_brace_index:end_brace_index+1]
            try:
                json_format = json.loads(json_response)
                results[cur_idx[index]] = (
                    {"pred_summary": json_response})
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
