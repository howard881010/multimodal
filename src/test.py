import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Generate Text Embeddings with LLaMA2-7B
def generate_text_embedding(text):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto")
    
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs, output_hidden_states=True)
    text_embedding = outputs.hidden_states[-1]  # Use the last hidden state
    return text_embedding

# Step 2: Generate Numerical Embeddings
class NumericalModel(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(NumericalModel, self).__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        return self.linear(x).unsqueeze(0)

def generate_numerical_embedding(data, model):
    data_tensor = torch.tensor(data, dtype=torch.float32)
    numerical_embedding = model(data_tensor)
    return numerical_embedding

# Step 3: Combine the Embeddings
def combine_embeddings(text_embedding, numerical_embedding):
    combined_embedding = torch.cat((text_embedding, numerical_embedding), dim=-1)
    return combined_embedding

# Step 4: Define Multimodal Model
class MultimodalModel(nn.Module):
    def __init__(self, combined_dim, output_dim):
        super(MultimodalModel, self).__init__()
        self.fc = nn.Linear(combined_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# Example usage
def main():
    text = "The output will increase"
    numerical_data = [1.0]

    # Initialize models
    numerical_model = NumericalModel(input_dim=len(numerical_data), embedding_dim=4096)  # Assuming LLaMA2-7B embedding size is 4096
    multimodal_model = MultimodalModel(combined_dim=4096 + 4096, output_dim=1)

    # Generate embeddings
    text_embedding = generate_text_embedding(text)
    print("Text embedding shape:", text_embedding.shape)
    numerical_embedding = generate_numerical_embedding(numerical_data, numerical_model)
    print("Numerical embedding shape:", numerical_embedding.shape)

    # Combine embeddings
    combined_embedding = combine_embeddings(text_embedding, numerical_embedding)

    # Feed combined embedding into multimodal model
    output = multimodal_model(combined_embedding)

    print(output)

if __name__ == "__main__":
    main()
