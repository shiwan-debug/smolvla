# SmolVLA training image — based on lerobot project structure
ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04

ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl \
    PATH=/lerobot/.venv/bin:/usr/local/bin:$PATH \
    DEVICE=cuda \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_LINK_MODE=copy

# ---- System packages ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgeos-dev \
    libgl1 \
    libglib2.0-0 \
    ninja-build \
    pkg-config \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && uv python install ${PYTHON_VERSION} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /lerobot

ENV HOME=/root \
    HF_HOME=/root/.cache/huggingface \
    HF_LEROBOT_HOME=/root/.cache/huggingface/lerobot \
    TORCH_HOME=/root/.cache/torch

# ---- Create venv ----
RUN uv venv --python ${PYTHON_VERSION}

# ---- Install lerobot ----
COPY setup.py pyproject.toml uv.lock README.md MANIFEST.in ./
COPY src/ src/

RUN uv sync --locked --extra all --no-cache

# ---- Copy custom scripts ----
COPY train.py train_config.json test_smolvla.py requirements.txt ./

# ---- Directories ----
RUN mkdir -p /lerobot/datasets /lerobot/outputs /lerobot/models

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

CMD ["/bin/bash"]
