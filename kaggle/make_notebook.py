#!/usr/bin/env python3
"""Generate kaggle/run_epic.ipynb (and per-shard copies) from the cells below."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "https://github.com/Ayush0612005/prism-xr-track4-action-summarization"


def nb(shard: str) -> dict:
    cells = [
        "# EPIC-KITCHENS-100 batch run (Track 4)\n"
        "Clones the public repo, downloads each manifest EPIC video with the official scripts,\n"
        f"runs the same pipeline as `run.sh`, deletes the video. Shard: `{shard}`.",
        "!nvidia-smi -L; ffmpeg -version | head -1; df -h /kaggle/working /tmp | tail -2",
        "!pip install -q open_clip_torch av imageio-ffmpeg",
        f"!rm -rf /tmp/repo && git clone -q --depth 1 {REPO} /tmp/repo && ls /tmp/repo/weights",
        # resume: attach an earlier version's output as input data; its JSONs are reused
        "import glob, os\nprev = glob.glob('/kaggle/input/*/outputs')\nprint(prev)",
        f"!cd /tmp/repo && python kaggle/run_epic.py --shard {shard} --out /kaggle/working/outputs "
        "--work /tmp/epic $( [ -n \"$(ls -d /kaggle/input/*/outputs 2>/dev/null | head -1)\" ] && "
        "echo --resume-from $(ls -d /kaggle/input/*/outputs | head -1) )",
        "!ls /kaggle/working/outputs | wc -l; cat /kaggle/working/failures.txt 2>/dev/null; "
        "cd /kaggle/working && zip -q -r outputs.zip outputs && ls -la outputs.zip",
    ]
    out = []
    for i, c in enumerate(cells):
        md = i == 0
        out.append({"cell_type": "markdown" if md else "code", "metadata": {},
                    "source": c, **({} if md else {"outputs": [], "execution_count": None})})
    return {"cells": out, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                     "language": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


for shard in ("0/2", "1/2"):
    name = f"run_epic_shard{shard[0]}.ipynb"
    json.dump(nb(shard), open(os.path.join(HERE, name), "w"), indent=1)
    print("wrote", name)
