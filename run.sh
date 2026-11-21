#!/usr/bin/env bash
# Single-video inference entry point: ./run.sh <video.mp4> <out.json>
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: $0 <video.mp4> <out.json>" >&2
  exit 2
fi
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
if [ ! -f "$DIR/weights/tsm_rgb_epic100_fp16.pt" ]; then
  echo "weights/tsm_rgb_epic100_fp16.pt missing; run weights/download_weights.sh" >&2
  exit 1
fi
export HF_HOME="${HF_HOME:-$DIR/weights/hf}"
exec "$PY" "$DIR/code/infer.py" "$1" "$2"
