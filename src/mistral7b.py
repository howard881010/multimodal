from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# Prepare the prompt
prompt = "For the given 4 historical ratings: 4.00 4.38 3.90 4.41. Predict the future 4 ratings without generating any text."

# Tokenize input and ensure it is on the right device
model_inputs = tokenizer(prompt, return_tensors="pt")
model_device = next(model.parameters()).device  # Get the device of the model
model_inputs = model_inputs.to(model_device)

# Generate text using the model
generated_ids = model.generate(
    **model_inputs, max_new_tokens=100, do_sample=True)

# Decode generated ids to text
output_text = tokenizer.batch_decode(generated_ids)[0]
print(output_text)
