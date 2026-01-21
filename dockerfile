FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VER=3.11

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl git openssh-client \
      build-essential pkg-config \
      ffmpeg \
      libavcodec-dev libavformat-dev libavdevice-dev libavutil-dev \
      libswscale-dev libswresample-dev \
      libgl1 libglib2.0-0 libsm6 libxext6 \
      software-properties-common \
      fonts-noto-cjk fonts-wqy-microhei fonts-wqy-zenhei \
    && rm -rf /var/lib/apt/lists/*

# ---- Python 3.11 ----
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python${PYTHON_VER} python${PYTHON_VER}-dev python${PYTHON_VER}-venv \
    && rm -rf /var/lib/apt/lists/*

# ---- venv ----
ENV VENV_PATH=/opt/venv
RUN python${PYTHON_VER} -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

# ---- pip mirror (TUNA) ----
RUN python -m pip install -U pip setuptools wheel && \
    python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# ---- PyTorch cu124 ----
RUN python -m pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

RUN python -m pip install --no-cache-dir \
    numpy==1.26.4 \
    protobuf==5.28.3 \
    h5py==3.14.0 \
    opencv-python==4.11.0.86 \
    av==16.0.1 \
    mcap==1.2.2 \
    mcap-protobuf-support==0.5.3

# ---- Existing common deps (you had) ----
RUN python -m pip install --no-cache-dir \
    scipy pandas \
    matplotlib tqdm pillow einops \
    tensorboard wandb \
    transformers accelerate safetensors huggingface_hub \
    sentencepiece \
    peft \
    bitsandbytes==0.43.3

# ---- Jupyter (optional) ----
RUN python -m pip install --no-cache-dir \
    ipython \
    "jupyterlab>=4.1.0,<5.0.0a0" \
    jupyter nbclassic notebook \
    jupyterlab-lsp anywidget kaleido pyviz_comms \
    lckr_jupyterlab_variableinspector jupyterlab-spreadsheet-editor \
    jupyterlab-spreadsheet jupyterlabcodetoc

# ---- Runtime env ----
ENV TOKENIZERS_PARALLELISM=false \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
    TORCH_HOME=/workspace/.cache/torch \
    MPLBACKEND=Agg

WORKDIR /workspace
EXPOSE 8888

CMD ["bash", "-lc", "python -V && python -c \"import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\" && python -c \"import av, mcap, h5py, cv2, numpy, google.protobuf; print('av', av.__version__, 'mcap', mcap.__version__)\" && bash"]
