"""
Saves each worker's model state dict directly to a checkpoint, which enables a
fast load path for large tensor-parallel models where each worker only needs to
read its own shard rather than the entire checkpoint.

Example usage:

python3 save_sharded_state.py \
    --model /root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-70B-Instruct \
    --quantization deepspeedfp \
    --tensor-parallel-size 8 \
    --output /root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-70B-Instruct-sharded-8

python3 save_sharded_state.py \
--model /root/.cache/huggingface/hub/models--casperhansen--llama-3-70b-instruct-awq \
--quantization deepspeedfp \
--tensor-parallel-size 8 \
--output /root/.cache/huggingface/hub/models--casperhansen--llama-3-70b-instruct-awq-sharded-8

python3 -m vllm.entrypoints.openai.api_server \
--model meta-llama/Meta-Llama-3-70B-instruct \
--api-key 67c21f73-9d1c-40f7-8a2f-adf2c9274f46 --trust-remote-code \
--gpu-memory-utilization=0.9 \
--tensor-parallel-size=8

python3 -m vllm.entrypoints.openai.api_server \
--model="/root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-70B-Instruct" \
--quantization="deepspeedfp" \
--load_format="sharded_state"
--api-key 67c21f73-9d1c-40f7-8a2f-adf2c9274f46 --trust-remote-code \
--gpu-memory-utilization=0.9 \
--tensor-parallel-size=8


python3 -m vllm.entrypoints.openai.api_server \
--model casperhansen/llama-3-70b-instruct-awq \
--api-key 67c21f73-9d1c-40f7-8a2f-adf2c9274f46 --trust-remote-code \
--gpu-memory-utilization=0.9 \
--tensor-parallel-size=2 \
--host="0.0.0.0" \
--port=8000


Then, the model can be loaded with

llm = LLM(
    model="/path/to/save",
    load_format="sharded_state",
    quantization="deepspeedfp",
    tensor_parallel_size=8,
)
"""
import dataclasses
import os
import shutil
from pathlib import Path

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser

parser = FlexibleArgumentParser()
EngineArgs.add_cli_args(parser)
parser.add_argument("--output",
                    "-o",
                    required=True,
                    type=str,
                    help="path to output checkpoint")
parser.add_argument("--file-pattern",
                    type=str,
                    help="string pattern of saved filenames")
parser.add_argument("--max-file-size",
                    type=str,
                    default=5 * 1024**3,
                    help="max size (in bytes) of each safetensors file")


def main(args):
    engine_args = EngineArgs.from_cli_args(args)
    if engine_args.enable_lora:
        raise ValueError("Saving with enable_lora=True is not supported!")
    model_path = engine_args.model
    if not Path(model_path).is_dir():
        raise ValueError("model path must be a local directory")
    # Create LLM instance from arguments
    llm = LLM(**dataclasses.asdict(engine_args))
    # Prepare output directory
    Path(args.output).mkdir(exist_ok=True)
    # Dump worker states to output directory
    model_executor = llm.llm_engine.model_executor
    model_executor.save_sharded_state(path=args.output,
                                      pattern=args.file_pattern,
                                      max_size=args.max_file_size)
    # Copy metadata files to output directory
    for file in os.listdir(model_path):
        if os.path.splitext(file)[1] not in (".bin", ".pt", ".safetensors"):
            if os.path.isdir(os.path.join(model_path, file)):
                shutil.copytree(os.path.join(model_path, file),
                                os.path.join(args.output, file))
            else:
                shutil.copy(os.path.join(model_path, file), args.output)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)