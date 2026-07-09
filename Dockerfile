# =============================================================================
# SmolVLA Training Docker Image
# Based on PyTorch official image (comes with Python 3.12 + CUDA + PyTorch)
# =============================================================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

LABEL description="SmolVLA training environment (lerobot 0.6.1)"

ENV DEBIAN_FRONTEND=noninteractive

# =============================================================================
# System packages
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl vim \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Python dependencies
# =============================================================================
COPY requirements-docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# =============================================================================
# Install lerobot from source
# =============================================================================
WORKDIR /smolvla
COPY src/ /smolvla/src/
COPY train.py train_config.json test_smolvla.py /smolvla/

RUN pip install --no-cache-dir -e .

# =============================================================================
# Directories
# =============================================================================
RUN mkdir -p /smolvla/datasets /smolvla/outputs /smolvla/models

# =============================================================================
# Environment
# =============================================================================
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

WORKDIR /smolvla
CMD ["python", "train.py"]
