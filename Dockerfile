# =============================================================================
# SmolVLA Training Docker Image
# =============================================================================
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

LABEL description="SmolVLA training environment (lerobot 0.6.1 + SmolVLM2-500M)"

# =============================================================================
# System packages
# =============================================================================
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git curl vim libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Miniconda
# =============================================================================
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh && \
    bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh
ENV PATH=/opt/conda/bin:$PATH

# =============================================================================
# Conda env + PyTorch (CUDA 12.4)
# =============================================================================
RUN conda create -n smolvla python=3.12 -y
RUN /opt/conda/envs/smolvla/bin/pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# =============================================================================
# Python dependencies (pinned from requirements.txt)
# =============================================================================
COPY requirements.txt /tmp/requirements.txt
RUN /opt/conda/envs/smolvla/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# =============================================================================
# Install lerobot from source
# =============================================================================
WORKDIR /smolvla
COPY src/ /smolvla/src/
COPY train.py train_config.json test_smolvla.py /smolvla/

# Use setup.py from src/lerobot root
RUN cd /smolvla && /opt/conda/envs/smolvla/bin/pip install --no-cache-dir -e .

# =============================================================================
# Directories for data and outputs
# =============================================================================
RUN mkdir -p /smolvla/datasets /smolvla/outputs /smolvla/models

# =============================================================================
# Environment
# =============================================================================
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# =============================================================================
# ENTRYPOINT: training
# Usage:
#   docker run --gpus all -v /path/to/data:/smolvla/datasets -v /path/to/model:/smolvla/models $IMAGE
# =============================================================================
WORKDIR /smolvla
CMD ["/opt/conda/envs/smolvla/bin/python", "train.py"]
