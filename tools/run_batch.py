#!/usr/bin/env python3
"""Run the pipeline over every video in a folder, loading the models once.

Equivalent to calling ./run.sh per video. Skips videos whose JSON exists unless --force.
usage: python tools/run_batch.py <video_dir> <out_dir> [--force]
"""
import argparse
import glob
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

import torch  # noqa: E402
from clip_model import ClipScorer  # noqa: E402
from infer import WEIGHTS, process  # noqa: E402
from tsm import load_tsm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video_dir")
ap.add_argument("out_dir")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
models = (load_tsm(WEIGHTS, device), ClipScorer(device))
vids = sorted(p for p in glob.glob(os.path.join(a.video_dir, "*")) if p.lower().endswith((".mp4", ".mkv", ".webm")))
fails = []
for p in vids:
    out = os.path.join(a.out_dir, os.path.splitext(os.path.basename(p))[0] + ".json")
    if os.path.exists(out) and not a.force:
        continue
    for attempt in (1, 2):
        try:
            process(p, out, device, models)
            break
        except Exception:
            traceback.print_exc()
            if attempt == 2:
                fails.append(p)
os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
with open(os.path.join(ROOT, "logs", "failures.txt"), "a") as f:
    for p in fails:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')}\t{os.path.basename(p)}\tinference failed\n")
print(f"done: {len(vids)} videos, {len(fails)} failures")
