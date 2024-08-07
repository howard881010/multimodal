# Get the server and know how to use the GPU on nautilus
- ask zihao to have a local server
- know how to use the [toolbox](https://github.com/Rose-STL-Lab/Zihao-s-Toolbox)
# Prepare the dataset
- Get the processed data from [MinIO](https://rosedata.ucsd.edu:9091/browser/m2forecast)
- Upload the dataset to your huggingface account (reference: upload_*.ipynb) and in alpaca format
- Example: [climate](https://huggingface.co/datasets/Howard881010/climate-cal)
# Finetune the model
- Use [axolotl](https://github.com/axolotl-ai-cloud/axolotl), (reference: finetune/*.yaml)
- Or try to use [unsloth](https://github.com/unslothai/unsloth)
# Inference & Evaluation
- Use your finetune model for inference (reference: modelchat.py for choose your finetune model, and multimodal.py for the inference part and evaluation)

