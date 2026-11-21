# docker build -t track4 .
# docker run --rm --gpus all -v /path/to/videos:/data track4 /data/P01_11.MP4 /data/P01_11.json
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH PYTHON=python3
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /app
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt

COPY . .
# CLIP weights are cached into weights/hf at build time, so inference needs no network
RUN chmod +x run.sh weights/download_weights.sh && weights/download_weights.sh

ENTRYPOINT ["./run.sh"]
