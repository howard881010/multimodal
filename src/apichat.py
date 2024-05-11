# from openai import OpenAI
# import openai
import google.generativeai as genai
import yaml
import time
import asyncio
import fastapi_poe as fp
import requests
import re
import json
from p_tqdm import p_map
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


last_chat_time = None
system_prompt = "You are a helpful assistant."


class APITranslationFailure(Exception):
    def __init__(self, message="API connection failed after retries.", *args):
        super().__init__(message, *args)


class APIChatApp:
    def __init__(self, api_key, model_name, temperature):
        self.api_key = api_key
        self.model_name = model_name
        self.messages = [{"role": "system", "content": system_prompt}]
        self.response = None
        self.temperature = temperature

    def chat(self, message):
        raise NotImplementedError("Subclasses must implement this method")


class MistralChatApp(APIChatApp):
    def __init__(self, api_key,model_name, temperature=0.7):
        super().__init__(api_key, model_name, temperature)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token="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")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, token="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")

    def chat(self, question):
        self.messages.append({"role": "user", "content": question})
        self.encodeds = self.tokenizer.apply_chat_template(self.messages)
        
        self.generated_ids = self.model.generate(
            self.encodeds,
            max_new_tokens=1000,
            do_sample=True,
        )
        decoded = self.tokenizer.batch_decode(self.generated_ids)
        answer = decoded[0]
        self.messages.append({"role": "assistant", "content": answer})
        return answer
class OpenAIChatApp(APIChatApp):
    def __init__(self, api_key, model_name, temperature=0.7):
        super().__init__(api_key, model_name, temperature)
        base_url = "http://localhost:34567/v1"

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def chat(self, message):
        self.messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": message
            }
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                temperature=self.temperature
            )
            self.messages = [{"role": "assistant", "content": response.choices[0].message.content}]
            self.response = response
            return response.choices[0].message.content
        except openai.APIError as e:
            raise APITranslationFailure(f"OpenAI API connection failed: {str(e)}")


class GoogleChatApp(APIChatApp):
    def __init__(self, api_key, model_name, temperature=0.2):
        super().__init__(api_key, model_name, temperature)
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def chat(self, message):
        global last_chat_time

        if last_chat_time is not None:
            elapsed_time = time.time() - last_chat_time
            if elapsed_time < 1:
                time_to_wait = 1 - elapsed_time
                time.sleep(time_to_wait)
        last_chat_time = time.time()

        self.messages.append({"role": "user", "content": message})
        prompt = "".join([m["content"] for m in self.messages])
        try:
            response = self.model.generate_content(
                prompt,
                safety_settings=[
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"},
                ],
                generation_config={
                    "temperature": self.temperature, "max_output_tokens": 8192}
            )
            if 'block_reason' in response.prompt_feedback:
                print(vars(response))
                raise APITranslationFailure(
                    "Content generation blocked due to safety settings.")
            self.messages = [{"role": "assistant", "content": response.text}]
            return response.text
        except Exception as e:
            raise APITranslationFailure(
                f"Google API connection failed: {str(e)}")


class PoeAPIChatApp:
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        self.messages = []

    def chat(self, message):
        return asyncio.run(self._async_chat(message))

    async def _async_chat(self, message):
        self.messages.append({"role": "user", "content": message})
        final_message = ""
        try:
            async for partial in fp.get_bot_response(messages=self.messages, bot_name=self.model_name,
                                                     api_key=self.api_key):
                final_message += partial.text
        except Exception as e:
            raise APITranslationFailure(f"Poe API connection failed: {str(e)}")
        return final_message


class BaichuanChatApp:
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        self.url = 'https://api.baichuan-ai.com/v1/chat/completions'
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

    def chat(self, message, temperature=0.3, top_p=0.85, max_tokens=2048):
        payload = {
            "model": self.model_name,
            "messages": [{
                "role": "user",
                "content": message
            }],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "with_search_enhance": False,
            "stream": True
        }

        response = requests.post(
            self.url, headers=self.headers, data=json.dumps(payload))
        if response.status_code == 200:
            raw_stream_response = response.text
            matches = ''.join(re.findall(
                r'\"content\":\"(.*?)\"', raw_stream_response, re.DOTALL))
            finish_reason_matches = re.findall(
                r'\"finish_reason\":\"(.*?)\"', raw_stream_response)
            if not finish_reason_matches or "stop" not in finish_reason_matches:
                err_msg = "\n".join(response.text.split('\n')[-5:])
                raise APITranslationFailure(
                    f"Baichuan API translation terminated: {err_msg}")
            else:
                return matches.replace('\\n', '\n')
        else:
            raise APITranslationFailure(
                f"Baichuan API connection failed: {response.text}")


if __name__ == "__main__":

    # with open("translation.yaml", "r") as f:
    #     translation_config = yaml.load(f, Loader=yaml.FullLoader)
    # openai_chat = OpenAIChatApp(
    #     api_key='EMPTY',
    #     model_name=''
    # )
    # google_chat = GoogleChatApp(
    #     api_key=translation_config['Gemini-Pro-api']['key'],
    #     model_name='gemini-pro'
    # )
    # google_chat.messages = [
    #     {
    #         "role": "user",
    #         "content": """For the given 4 historical ratings: 4.21 4.00 4.38 3.90 and the following reviews for those historical ratings: Chargrilled oysters, seafood platter and oyster shooters! <SEP> Acme Oyster House is one of New Orleans' most famous restaurants, but what is it like to eat there? <SEP> I've been to Seafood Etoufee before and it's the best oysters I've ever had. We went with a half dozen fresh oysters, half dozen chargrilled oysters, the seafood etoufee, and the New Orleans Medley. <SEP> I've been to this place before and it's always packed. . Predict the future 4 ratings."""

    #     },
    #     {
    #         "role": "bot",
    #         "content": "4.41 4.61 4.14 4.11"
    #     }
    # ]
    mistral_chat = MistralChatApp(
        api_key='EMPTY',
        model_name='mistralai/Mistral-7B-Instruct-v0.2'
    )

    prompt = f"For the given 4 historical ratings: 4.00 4.38 3.90 4.41 and the following reviews for those historical ratings: Acme Oyster House is one of New Orleans' most famous restaurants, but what is it like to eat there? <SEP> I've been to Seafood Etoufee before and it's the best oysters I've ever had. We went with a half dozen fresh oysters, half dozen chargrilled oysters, the seafood etoufee, and the New Orleans Medley. <SEP> I've been to this place before and it's always packed. <SEP> The chargrilled oysters were delicious. . Predict the future 4 ratings. "
    print(mistral_chat.chat(prompt))
