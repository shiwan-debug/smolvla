# =============================================================================
# SmolVLA Docker Image
# Base: Ubuntu 22.04 + CUDA 12.8 + Python 3.12
# =============================================================================
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

LABEL maintainer="your_email@xxx.com"
LABEL description="SmolVLA inference environment: lerobot 0.6.1 + SmolVLM2-500M-Video-Instruct"

# =============================================================================
# System dependencies
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    vim \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Miniconda + Python 3.12
# =============================================================================
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# =============================================================================
# Create conda environment: smolvla (Python 3.12)
# =============================================================================
RUN conda create -n smolvla python=3.12 -y

# =============================================================================
# Install PyTorch with CUDA 12.8
# =============================================================================
RUN /opt/conda/envs/smolvla/bin/pip install \
    torch==2.11.0 \
    torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128

# =============================================================================
# Install lerobot dependencies
# av>=15.0.0,<16.0.0 is required by lerobot 0.6.1
# =============================================================================
RUN /opt/conda/envs/smolvla/bin/pip install \
    "numpy<2.3.0,>=2.0.0" \
    "packaging<26.0,>=24.2" \
    pillow \
    "opencv-python-headless>=4.9.0" \
    einops \
    draccus==0.10.0 \
    "huggingface-hub<2.0.0,>=1.0.0" \
    "safetensors<1.0.0,>=0.4.3" \
    "termcolor<4.0.0,>=2.4.0" \
    "tqdm<5.0.0,>=4.66.0" \
    "gymnasium<2.0.0,>=1.1.1" \
    "diffusers>=0.27.2" \
    datasets \
    transformers \
    "datasets" \
    num2words \
    "cmake>=3.29.0.1"

# Install av (PyAV) - required by lerobot 0.6.1
# Check if Linux wheel available, otherwise build from source
RUN /opt/conda/envs/smolvla/bin/pip install "av>=15.0.0,<16.0.0" || \
    /opt/conda/envs/smolvla/bin/pip install av

# =============================================================================
# Copy smolvla source code into container
# =============================================================================
WORKDIR /smolvla
COPY . /smolvla

# =============================================================================
# Install smolvla (lerobot 0.6.1) from local source
# =============================================================================
RUN /opt/conda/envs/smolvla/bin/pip install -e .

# =============================================================================
# Environment variables (offline mode for inference)
# =============================================================================
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

# =============================================================================
# Default command: run inference test
# =============================================================================
WORKDIR /smolvla
CMD ["/opt/conda/envs/smolvla/bin/python", "test_smolvla.py"]
