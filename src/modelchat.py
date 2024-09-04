from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import numpy as np
import torch
import time
from peft import PeftModel
import os
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser, RegexParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn


class AnswerFormat(BaseModel):
    day_3_date: str
    day_3_weather_forecast: str
    day_4_date: str
    day_4_weather_forecast: str

class ChatModel:
    def __init__(self, model_name, token, dataset, zeroshot, case, device, window_size):
        self.model_name = model_name
        self.zeroshot = zeroshot
        self.token = token
        self.tokenizer = self.load_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset = dataset
        self.case = case
        self.device = device
        self.window_size = window_size

    def load_model(self):
        raise NotImplementedError("Subclasses must implement this method")

    def load_tokenizer(self):
        raise NotImplementedError("Subclasses must implement this method")

    def chat(self, prompt):
        raise NotImplementedError("Subclasses must implement this method")

class LLMChatModel(ChatModel):
    def __init__(self, model_name, token, dataset, zeroshot, case, device, window_size):
        super().__init__(model_name, token, dataset, zeroshot, case, device, window_size)
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        # self.device = next(self.model.parameters()).device

    def load_model(self):
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, token=self.token).to(self.device)

        if self.zeroshot == True:
            return base_model
        else: 
            return PeftModel.from_pretrained(base_model, f"Howard881010/{self.dataset}-{self.window_size}day")
    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
    def chat(self, prompt):
        new_prompt = self.tokenizer.apply_chat_template(
            prompt, tokenize=False)
        model_inputs = self.tokenizer(
            new_prompt, return_tensors="pt", padding="longest")
        model_inputs = model_inputs.to(self.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        parser = JsonSchemaParser(
            AnswerFormat.model_json_schema()
        )
        prefix_function = build_transformers_prefix_allowed_tokens_fn(
            self.tokenizer, parser
        )
        # Generate text using the model
        with torch.no_grad():
            generate_ids = self.model.generate(
                model_inputs.input_ids, max_new_tokens=4096, eos_token_id=terminators, attention_mask=model_inputs.attention_mask, prefix_allowed_tokens_fn=prefix_function)

        output = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True)

        return output

if __name__ == "__main__":
    np.random.seed(42)
    set_seed(42)
    start = time.time()

    input = str({"temp": 169.9})
    content = [{"role": "system", "content": """Given the weather forecast of the first 2 day, predict the weather forecast of the next 2 day. Output the result **ONLY** in the following JSON format: {
  "day_3_date": "",
  "day_3_weather_forecast": "",
  "day_4_date": "",
  "day_4_weather_forecast": ""
}
"""}, 
               {"role": "user", "content": """{
  "day_3_date": "2022-12-07",
  "day_3_weather_forecast": "Heavy coastal rains and significant mountain snow are expected across the West and Rockies this weekend through Monday, with heavy rainfall and severe weather threats in the Mississippi Valley, along with snow, ice, and freezing rain in the north-central U.S. Rain and snow are anticipated for the Great Lakes region into the northern Mid-Atlantic and Northeast starting Saturday night. A dynamic system will form, bringing widespread and significant precipitation over the West and enhancing rainfall in the Mississippi Valley early next week. Temperatures will be mostly below normal over the West, with 10-15°F below normal highs, while the eastern U.S. will experience near to above normal temperatures, especially in morning lows with anomalies of 15-25°F in the southern tier. Highs in the central and east-central U.S. may reach 10-15°F above normal. Precipitation includes moderate to heavy rainfall in the southeastern U.S. and potential local flooding. Heavy snowfall is expected in higher elevations like the Sierra Nevada and Rockies, with significant snow probability increasing in the Northern Plains and Upper Midwest. Severe weather, including strong convection, is likely in the Lower and Middle Mississippi Valley starting Monday. Expected hazards include heavy precipitation across the Great Lakes and Upper Mississippi Valley, heavy rain in California and the Southwest, and heavy snow across various regions on specified dates from December 10 to December 14.",
  "day_4_date": "2022-12-08",
  "day_4_weather_forecast": "Heavy coastal rains and widespread mountain snow are expected across the West and Rockies from Sunday into Monday. A significant rainfall and severe weather threat will spread eastward from the southern U.S. next week. Rain and snow are anticipated from the Great Lakes into the northern Mid-Atlantic and Northeast due to an amplified trough. Moderate to heavy rainfall is likely in the southeastern U.S. over the weekend, while plowable snow is expected in the northern regions. A deep storm will lead to significant terrain-enhanced rain and snow in the West, with a potential blizzard threat in the north-central U.S. Heavy precipitation is expected in the Lower and Middle Mississippi Valley starting Monday and intensifying through midweek, along with severe convection possible. Below normal high temperatures (10-15°F below) are expected in the West, while the southern and eastern regions will see near to above normal temperatures, particularly warmer morning lows. Significant snowfall, ice, and freezing rain risks are forecast for the Northern Plains, Upper Midwest, Great Lakes, and Northeast. Winds associated with winter weather conditions will impact several regions. Potential for severe weather exists from the Middle to Lower Mississippi Valley and Southern Plains, especially next week."
}
"""}]
    prompt = [content]
    prompt.append(content)
    token = os.getenv("HF_TOKEN")

    # model_chat = MistralChatModel(
    #     "mistralai/Mistral-7B-Instruct-v0.1", token, "climate")
    # fine-tuned model
    model_chat = LLMChatModel("unsloth/Meta-Llama-3.1-8B-Instruct", token, "climate", True, 1, "cuda:0", 3)

    output = model_chat.chat(prompt)
    # print(output)
    print(output[0])

    end = time.time()
    print("Time taken: ", end - start)
