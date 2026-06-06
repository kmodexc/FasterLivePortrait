# syntax=docker/dockerfile:1
#
# TensorRT runtime for Ada Lovelace GPUs such as RTX 40-series.
# The previous ONNX-focused Dockerfile is kept as Dockerfile.ada.onnx.

ARG TRT_BASE=nvcr.io/nvidia/tensorrt:24.12-py3
ARG CUDA_ARCHITECTURES=89
ARG PLUGIN_BUILD_PARALLELISM=4
ARG TENSORRT_ROOT=/usr

FROM ${TRT_BASE} AS plugin-builder

ARG CUDA_ARCHITECTURES
ARG PLUGIN_BUILD_PARALLELISM
ARG TENSORRT_ROOT
ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git

RUN git clone --depth 1 https://github.com/SeanWangJS/grid-sample3d-trt-plugin \
    /opt/grid-sample3d-trt-plugin

WORKDIR /opt/grid-sample3d-trt-plugin

RUN cmake -S . -B build \
      -DTensorRT_ROOT="${TENSORRT_ROOT}" \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" && \
    cmake --build build --parallel "${PLUGIN_BUILD_PARALLELISM}" && \
    mkdir -p /opt/fasterliveportrait/plugins && \
    cp "$(find build -name libgrid_sample_3d_plugin.so -print -quit)" \
      /opt/fasterliveportrait/plugins/libgrid_sample_3d_plugin.so

FROM ${TRT_BASE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3-venv \
    espeak-ng espeak-ng-data ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 \
    git wget && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --system-site-packages /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    FLP_TRT_PLUGIN_PATH=/opt/fasterliveportrait/plugins/libgrid_sample_3d_plugin.so \
    PHONEMIZER_ESPEAK_DATA=/usr/lib/x86_64-linux-gnu/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/lib/x86_64-linux-gnu/espeak-ng-data \
    ESPEAKNG_DATA_PATH=/usr/lib/x86_64-linux-gnu/espeak-ng-data

COPY --from=plugin-builder /opt/fasterliveportrait/plugins /opt/fasterliveportrait/plugins

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

WORKDIR /root/FasterLivePortrait
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary pycuda

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary insightface

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary -r requirements.txt

RUN python3 -c 'import tensorrt as trt; import pycuda; import torch; print("TensorRT:", trt.__version__); print("Torch CUDA:", torch.version.cuda)' && \
    test -f "${FLP_TRT_PLUGIN_PATH}"

COPY . .

ENV FLIP_IP=0.0.0.0
ENV FLIP_PORT=9871

EXPOSE 9870 9871

CMD ["python3", "webui.py", "--mode", "trt", "--host_ip", "0.0.0.0", "--port", "9870"]
