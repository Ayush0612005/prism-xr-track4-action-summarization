#!/usr/bin/env python3
"""Convert the official Lightning TSM checkpoint into a compact fp16 state dict.

usage: python tools/convert_weights.py tsm_rgb.ckpt weights/tsm_rgb_epic100_fp16.pt
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from tsm import load_tsm  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
model = load_tsm(src)  # strict: raises on any key/shape mismatch
state = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
torch.save(state, dst)
print(f"saved {dst} ({os.path.getsize(dst) / 1e6:.1f} MB, {len(state)} tensors)")
