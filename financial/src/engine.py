from openai import OpenAI
import json


class Engine():
    def __init__(self,
                 model="meta-llama/Meta-Llama-3-70B-Instruct",
                 api_key="EMPTY",
                 base_url="http://localhost:8000/v1"):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def run(self, prompt, doc, guided_json=None):
        messages = self.message_template(prompt, doc)
        response = self.llm_chat(
            messages, guided_json=guided_json)
        if guided_json is not None:
            data = json.loads(response)
            return data
        else:
            return response

    def message_template(self, prompt: str, content: str):
        messages = [{
            "role": "system",
            "content": prompt,
        }, {
            "role": "user",
            "content": content
        }]
        return messages

    def llm_chat(self, messages: list[dict], guided_json=None):
        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body={
                "guided_json": guided_json
            }
        )
        # return chat_response
        return chat_response.choices[0].message.content


#     def call_llm_chat(prompt: str, messages: list[dict], thread_id: int, result_queue: queue.Queue):
#         try:
#             formatted = self.message_template(prompt, messages)
#             response = self.llm_chat(formatted)
#             result_queue.put((thread_id, response))
#         except Exception as e:
#             result_queue.put((thread_id, str(e)))


# def batch_call_llm_chat(prompt: str, data: list[str]):
#     """
#     Prompt: for system prompt
#     data: list of string
#     """
#     result_queue = queue.Queue()
#     threads = []
#     for thread_id, content in enumerate(data):
#         thread = threading.Thread(target=call_llm_chat, args=(
#             prompt, content, thread_id, result_queue))
#         threads.append(thread)
#         thread.start()

#     # Wait for all threads to complete
#     for thread in threads:
#         thread.join()

#     responses = [None] * len(data)
#     while not result_queue.empty():
#         thread_id, response = result_queue.get()
#         responses[thread_id] = response

#     return responses
