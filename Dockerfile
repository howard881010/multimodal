FROM gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/minimal

USER root

# Install dependency
RUN apt update && apt install -y make rsync git vim build-essential cmake

# Build LLaMA.CPP
WORKDIR /root/
RUN git clone https://github.com/ggerganov/llama.cpp.git && mkdir llama.cpp/build
WORKDIR /root/llama.cpp/build
RUN cmake .. -DLLAMA_CUBLAS=ON
RUN cmake --build . --config Release

# Pull the latest project
WORKDIR /root/
RUN git clone --depth=1 https://gitlab.nrp-nautilus.io/Howard/mistral7b.git
WORKDIR /root/llamacpp/

# Handle git submodule
RUN git config --global url."https://github.com/".insteadOf git@github.com:; \
    git config --global url."https://".insteadOf git://; \
    git submodule update --init --recursive

# Install conda environment
RUN conda update --all
RUN conda install -c conda-forge conda-lock
RUN conda-lock install --name llamacpp
RUN conda clean -qafy

# Activate the new conda environment and install poetry
SHELL ["/opt/conda/bin/conda", "run", "-n", "llamacpp", "/bin/bash", "-c"]
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install 'llama-cpp-python[server]'
RUN pip install wandb boto3
