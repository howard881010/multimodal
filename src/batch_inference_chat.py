from utils import create_batched
from num2words import num2words
from modelchat import MistralChatModel, LLMChatModel, GemmaChatModel
import re
import json
from tqdm import tqdm


def batch_inference_mistral(
    results,
    model_chat: MistralChatModel,
    data,
    attempts,
    logger,
    historical_window_size,
    dataset,
    chat_template=None,
):
    batches = list(create_batched(data, 32))
    err_idx = []
    #  model_chat.apply_chat_template(chat_template)

    for batch in batches:
        prompt, cur_idx = create_batch_prompt(
            batch, historical_window_size, chat_template, dataset)
        # model_chat.apply_chat_template(chat_template)
        output_texts = model_chat.chat(prompt)

        for index, output_text in enumerate(output_texts):
            response = output_text.split('[/INST]')[:]

            logger.info("Content for row: " + str(cur_idx[index]) +
                        " Attempt: " + str(attempts + 1) + " Content: " + prompt[index][2]['content'])
            logger.info("Response for row: " + str(cur_idx[index]) + " Attempt: " + str(
                attempts + 1) + " Content: " + response)

            numbers = re.findall(r"\d+\.\d+", response)
            if len(numbers) != historical_window_size:
                err_idx.append(cur_idx[index])
                results[cur_idx[index]] = (
                    {"pred_values": "nan", "attempt": attempts + 1})
            else:
                str_res = ' '.join([str(num) for num in numbers])
                results[cur_idx[index]] = (
                    {"pred_values": str_res, "attempt": attempts + 1})

    return err_idx


def batch_inference_llama(
    results,
    model_chat: LLMChatModel,
    data,
    logger,
    historical_window_size,
    dataset,
    chat_template=None
):
    batches = list(create_batched(data, 64))
    err_idx = []
    #  model_chat.apply_chat_template(chat_template)

    for batch in batches:
        prompt, cur_idx = create_batch_prompt(
            batch, historical_window_size, chat_template, dataset)

        output_texts = model_chat.chat(prompt)

        for index, output_text in enumerate(output_texts):
            response = output_text.split('[/INST]')[-1]

            logger.info("Content for row: " + str(cur_idx[index]) + " Content: " + prompt[index][-1]['content'])
            logger.info("Response for row: " + str(cur_idx[index]) +  " Content: " + response)

            numbers = re.findall(r"\d+\.\d+", response)
            if len(numbers) != historical_window_size:
                err_idx.append(cur_idx[index])
                results[cur_idx[index]] = (
                    {"pred_values": "nan"})
            else:
                str_res = ' '.join([str(num) for num in numbers])
                results[cur_idx[index]] = (
                    {"pred_values": str_res})

    return err_idx

def batch_inference_llama_summary(
    results,
    model_chat,
    data,
    logger,
    historical_window_size,
    dataset,
    chat_template=None
):
    batches = list(create_batched(data, 16))
    err_idx = []

    for batch in tqdm(batches):
        prompt, cur_idx = create_batch_prompt(
            batch, historical_window_size, chat_template, dataset)

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


def batch_inference_gemma(
    results,
    model_chat,
    data,
    attempts,
    logger,
    historical_window_size,
    dataset,
    chat_template=None
):
    batches = list(create_batched(data, 12))
    err_idx = []
    #  model_chat.apply_chat_template(chat_template)

    for batch in batches:
        prompt, cur_idx = create_batch_prompt(
            batch, historical_window_size, chat_template, dataset)
        output_texts = model_chat.chat(prompt)

        for index, output_text in enumerate(output_texts):
            response = output_text.split("\n")[-1]

            logger.info("Content for row: " + str(cur_idx[index]) +
                        " Attempt: " + str(attempts + 1) + " Content: " + prompt[index][2]['content'])
            logger.info("Response for row: " + str(cur_idx[index]) + " Attempt: " + str(
                attempts + 1) + " Content: " + response)

            numbers = re.findall(r"\d+\.\d+", response)
            if len(numbers) != historical_window_size:
                err_idx.append(cur_idx[index])
                results[cur_idx[index]] = (
                    {"pred_values": "nan", "attempt": attempts + 1})
            else:
                str_res = ' '.join([str(num) for num in numbers])
                results[cur_idx[index]] = (
                    {"pred_values": str_res, "attempt": attempts + 1})
    return err_idx


def create_batch_prompt(data, historical_window_size, chat_template=None, dataset=None):
    prompt = []
    cur_idx = []

    for index, row in data.iterrows():
        if chat_template is not None:
            content = chat_template + [{"role": "user", "content": row['input']}]
        else:
            content = [{"role": "system", "content": row['instruction']}, {"role": "user", "content": row['input']}]
        # content = [{"role": "user", "content": row_content}]

        prompt.append(content)
        cur_idx.append(row['idx'])

    return prompt, cur_idx
