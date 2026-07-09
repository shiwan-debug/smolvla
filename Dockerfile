# =============================================================================
# SmolVLA Training Docker Image
# =============================================================================
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

LABEL description="SmolVLA training environment (lerobot 0.6.1 + SmolVLM2-500M)"

ENV DEBIAN_FRONTEND=noninteractive

# =============================================================================
# System packages (wget required for miniconda)
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git curl vim \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Miniconda
# =============================================================================
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh && \
    bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh && \
    /opt/conda/bin/conda --version

# =============================================================================
# Create env + install PyTorch
# =============================================================================
RUN /opt/conda/bin/conda create -n smolvla python=3.12 -y && \
    /opt/conda/envs/smolvla/bin/pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# =============================================================================
# Python dependencies
# =============================================================================
COPY requirements.txt /tmp/requirements.txt
RUN /opt/conda/envs/smolvla/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# =============================================================================
# Install lerobot from source
# =============================================================================
WORKDIR /smolvla
COPY src/ /smolvla/src/
COPY train.py train_config.json test_smolvla.py /smolvla/

RUN /opt/conda/envs/smolvla/bin/pip install --no-cache-dir -e .

# =============================================================================
# Data / output / model dirs
# =============================================================================
RUN mkdir -p /smolvla/datasets /smolvla/outputs /smolvla/models

# =============================================================================
# Environment variables
# =============================================================================
ENV PATH=/opt/conda/envs/smolvla/bin:$PATH
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# =============================================================================
# Default: training
# =============================================================================
WORKDIR /smolvla

CMD ["/opt/conda/envs/smolvla/bin/python", "train.py"]
