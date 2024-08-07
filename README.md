# Get the server and know how to use the GPU on nautilus
- ask zihao to have a local server
- know how to use the [toolbox](https://github.com/Rose-STL-Lab/Zihao-s-Toolbox)
# Copy the github and Create a conda environment in your local server
```bash
git clone git@github.com:howard881010/multimodal.git
conda env create -n your-name --file environment.yml
conda activate your-name
poetry install
## Add dependencies interactively or through poetry add
## Examples:
poetry source add --priority=explicit pytorch-gpu-src https://download.pytorch.org/whl/<cuda_version>
poetry add --source pytorch-gpu-src torch
```
# Prepare the dataset
- Get the processed data from [MinIO](https://rosedata.ucsd.edu:9091/browser/m2forecast)
- Upload the dataset to your huggingface account (reference: upload_*.ipynb) and in alpaca format
- Example: [climate](https://huggingface.co/datasets/Howard881010/climate-cal)
# Finetune the model
- Use [axolotl](https://github.com/axolotl-ai-cloud/axolotl), (reference: finetune/*.yaml)
- Or try to use [unsloth](https://github.com/unslothai/unsloth)
# Inference & Evaluation
- Use your finetune model for inference (reference: modelchat.py for choose your finetune model, and multimodal.py for the inference part and evaluation)
```bash
python src/multimodal.py climate 1 mistral7b 1 finetune
```

