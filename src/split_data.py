import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data(file_path, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1):
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Calculate the validation and test sizes
    val_size = validation_ratio / (test_ratio + validation_ratio)
    
    # Split the data into train and temporary datasets
    train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio), random_state=42)
    
    # Split the temporary dataset into validation and test datasets
    validation_data, test_data = train_test_split(temp_data, test_size=val_size, random_state=42)
    
    # Save the datasets
    
    dir = file_path.split('/')[:-1]
    dir = '/'.join(dir)
    file_name = file_path.split('/')[-1]
    train_data.to_csv(f'{dir}/train_{file_name}', index=False)
    validation_data.to_csv(f'{dir}/val_{file_name}', index=False)
    test_data.to_csv(f'{dir}/test_{file_name}', index=False)

# List of your CSV files
dir = 'Dataset/financial_llama3_70B_summary/formatted_v0.1'

# Loop through each file and split the data
for filename in os.listdir(dir):
    if filename.endswith(".csv"):
        path = os.path.join(dir, filename)
        split_data(path)  # Adjust the directory path as needed
    
