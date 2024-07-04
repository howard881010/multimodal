import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from darts.models import NLinearModel
from darts import TimeSeries
from pytorch_lightning import Trainer

# Set float32 matmul precision to utilize Tensor Cores
torch.set_float32_matmul_precision('high')  # You can also use 'medium' for less precision but potentially higher performance

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Use DataParallel to wrap the model if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print("Using {} GPUs".format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

# Move model to the available GPU(s)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = model.to(device)

# Example summaries
summaries = [
    "Summary one.",
    "Summary two.",
    "Summary three.",
    "Summary four.",
    "Summary five.",
    "Summary six."
]

# Example current values and target values (using dummy data for illustration)
current_values = np.array([i for i in range(100)])
target_values = np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

# Tokenize summaries
# inputs = tokenizer(summaries, padding=True, truncation=True, return_tensors="pt")
# input_ids = inputs["input_ids"]
# attention_mask = inputs["attention_mask"]

# # Define batch size
# batch_size = 2

# # Function to get batches
# def get_batches(input_ids, attention_mask, batch_size):
#     for i in range(0, len(input_ids), batch_size):
#         yield input_ids[i:i + batch_size], attention_mask[i:i + batch_size]

# # Create batches
# batches = list(get_batches(input_ids, attention_mask, batch_size))

# # Perform inference on each batch and collect pooled outputs
# pooled_outputs = []
# model.eval()
# with torch.no_grad():
#     for batch in batches:
#         input_ids_batch, attention_mask_batch = batch
#         input_ids_batch = input_ids_batch.to(device)
#         attention_mask_batch = attention_mask_batch.to(device)
#         outputs = model(input_ids_batch, attention_mask=attention_mask_batch)
#         pooled_output = outputs.pooler_output.cpu().numpy()
#         pooled_outputs.append(pooled_output)

# pooled_outputs = np.vstack(pooled_outputs)  # Shape: (num_samples, 768)

# # Combine pooled outputs with current values to form the input for the NLinearModel model
# inputs_for_NLinearModel = np.hstack((pooled_outputs, current_values.reshape(-1, 1)))

# Convert to TimeSeries object required by Darts
series = TimeSeries.from_values(current_values)
target_series = TimeSeries.from_values(target_values)
print("Series:", series)
print("Target series:", target_series)
# Ensure the series are properly aligned
# Define an appropriate input_chunk_length and output_chunk_length
input_chunk_length = 5
output_chunk_length = 5


# Define and train the NLinearModel model
model_NLinearModel = NLinearModel(input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, n_epochs=100, pl_trainer_kwargs={"accelerator": "gpu", "devices": 1}, )
model_NLinearModel.fit(series)
test_series = TimeSeries.from_values(np.array([i for i in range(1000, 1005)]))

# Make predictions
predictions = model_NLinearModel.predict(n=output_chunk_length, series=test_series)

predicted_target_values = predictions.values()[:, -1]

print("Predictions:", predicted_target_values)
