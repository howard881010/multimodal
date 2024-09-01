from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import numpy as np
import torch
import time
from peft import PeftModel
import os


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
            return PeftModel.from_pretrained(base_model, f"Howard881010/{self.dataset}-{self.window_size}day" + ("-mixed" if self.case == 2 else ""))
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

        # Generate text using the model
        with torch.no_grad():
            generate_ids = self.model.generate(
                model_inputs.input_ids, max_new_tokens=4096, eos_token_id=terminators, attention_mask=model_inputs.attention_mask)

        output = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True)

        return output

if __name__ == "__main__":
    np.random.seed(42)
    set_seed(42)
    start = time.time()

    input = str({"temp": 169.9})
    content = [{"role": "system", "content": "Given the medical notes and heart rates of the first 3 day, predict the medical notes and heart rates of the next 3 day.Output the result **ONLY** in the following YAML format: ``` day_4_date: day_4_medical_notes: day_4_Heart_Rate: day_5_date: day_5_medical_notes: day_5_Heart_Rate: day_6_date: day_6_medical_notes: day_6_Heart_Rate: ```"}, 
               {"role": "user", "content": "``` day_1_date: 2131-08-07 day_1_medical_notes: **Respiratory Rate:** - Ranges from 36-64 breaths per minute. - Mild subcostal retractions noted. - No apnea spells. **Heart Rate:** - 150-182 bpm, elevated but stable. **SaO2:** - Oxygen saturations >95% initially, transitioning to room air with saturations >93%. - Occasionally dips to the high 80s but self-resolves. **FiO2:** - Initially 100% oxygen, transitioned to room air at 2200. **Plan:** - Continue monitoring respiratory status maintaining saturations >93%. - Monitor heart rate and blood pressure closely. day_1_Heart_Rate: '158.955' day_2_date: 2131-08-08 day_2_medical_notes: **Respiratory:** - Respiratory Rate: 40-60 breaths per minute - Oxygen Saturation: Initially >94%, drifted to 90% - Breath Sounds: Clear, mild subcostal retractions noted - Assessment: Breathing comfortably despite saturation drift - Plan: Continue monitoring respiratory status **Cardiovascular:** - Heart Rate: Not explicitly documented in notes - Blood Pressure: 86/54 (mean 65) - Glucose: D-stick 83, Electrolytes: Sodium 137, Potassium 4.6 - Assessment: Well perfused, no heart murmur detected - Plan: Monitor vital signs regularly **Oxygenation:** - FiO2: Patient on room air - SaO2: Initially >92%, some mild desaturations observed - Assessment: Stable in room air - Plan: Continue monitoring oxygen saturation **Nutritional Support:** - Enteral feeds: Currently receiving 40 cc/kg/day of BM20, tolerated well - Total Fluid: Administering 150 cc/kg - Assessment: Patient tolerating feeds, abdomen soft and full with active bowel sounds - Plan: Advance feeds as tolerated and monitor closely day_2_Heart_Rate: '168.08' day_3_date: 2131-08-09 day_3_medical_notes: **Respiratory Status:** - Patient on nasal cannula at 100% FiO2. - Flow rate: 13-25 cc. - Respiratory Rate (RR): 40-60 breaths per minute, with mild subcostal retractions. - Lungs clear; no episodes of apnea, bradycardia, or desaturation. - Current SaO2: above 93%. - Plan to wean off oxygen as tolerated; Diuril restarted. **Cardiovascular Status:** - Heart Rate (HR): 160 beats per minute. - No murmurs or cardiovascular issues observed. **Interventions:** - Diuretic (Diuril) and potassium chloride (KCl) supplementation restarted. - Monitoring for readiness to discontinue oxygen and Diuril in the coming days. **Assessment:** - Patient appears active and engaged, with stable vital signs. - Monitoring for respiratory and nutritional status ongoing. day_3_Heart_Rate: '163.667' ```"}]
    prompt = [content]
    prompt.append(content)
    token = os.getenv("HF_TOKEN")

    # model_chat = MistralChatModel(
    #     "mistralai/Mistral-7B-Instruct-v0.1", token, "climate")
    # fine-tuned model
    model_chat = LLMChatModel("unsloth/Meta-Llama-3.1-8B-Instruct", token, "climate", False, 2, "cuda:0", 3)

    output = model_chat.chat(prompt)
    # print(output)
    for i in range(len(output)):
        print("Out of ", i, ": ", output[0])

    end = time.time()
    print("Time taken: ", end - start)