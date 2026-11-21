#!/usr/bin/env bash
# Fetch model weights.
#  * TSM: weights/tsm_rgb_epic100_fp16.pt is committed (converted from the official checkpoint).
#    This re-creates it from the original EPIC-KITCHENS-100 release if needed.
#  * CLIP ViT-B-32 (laion2b_s34b_b79k) is cached into weights/hf via open_clip / Hugging Face.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
if [ ! -f "$DIR/weights/tsm_rgb_epic100_fp16.pt" ]; then
  curl -L -o "$DIR/weights/tsm_rgb.ckpt" "https://www.dropbox.com/s/5yxnzubch7b6niu/tsm_rgb.ckpt?dl=1"
  "$PY" "$DIR/tools/convert_weights.py" "$DIR/weights/tsm_rgb.ckpt" "$DIR/weights/tsm_rgb_epic100_fp16.pt"
fi
HF_HOME="$DIR/weights/hf" "$PY" -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')"
echo "weights ready"
