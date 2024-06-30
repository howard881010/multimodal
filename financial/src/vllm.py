from openai import OpenAI
import threading
import queue
import logging

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def llm_chat(messages: list[dict], model="meta-llama/Meta-Llama-3-70B-Instruct", guided_json=None):
    chat_response = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body={
            "guided_json": guided_json
        }
    )
    return chat_response.choices[0].message.content


def message_template(prompt: str, content: str):
    messages = [{
        "role": "system",
        "content": prompt,
    }, {
        "role": "user",
        "content": content
    }]
    return messages


def call_llm_chat(prompt: str, messages: list[dict], thread_id: int, result_queue: queue.Queue):
    try:
        response = llm_chat(message_template(prompt, messages))
        result_queue.put((thread_id, response))
    except Exception as e:
        result_queue.put((thread_id, str(e)))


def batch_call_llm_chat(prompt: str, data: list[str]):
    """
    Prompt: for system prompt
    data: list of string
    """
    result_queue = queue.Queue()
    threads = []
    for thread_id, content in enumerate(data):
        thread = threading.Thread(target=call_llm_chat, args=(
            prompt, content, thread_id, result_queue))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    responses = [None] * len(data)
    while not result_queue.empty():
        thread_id, response = result_queue.get()
        responses[thread_id] = response

    return responses
