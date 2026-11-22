#!/usr/bin/env python3
"""Batch driver for the EPIC-KITCHENS-100 part of the manifest (used on Kaggle).

For each EPIC video id in data/manifest.csv: download with the official
epic-kitchens-download-scripts (next video prefetched in a background thread), run the
same pipeline as ./run.sh, write <id>.json, delete the video. Resumable (skips ids whose
JSON exists) and shardable (--shard k/n) so two sessions can split the list.

This file only orchestrates downloads; every JSON is produced by code/infer.py.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))


def epic_ids(manifest):
    with open(manifest, newline="") as f:
        return [r["video_id"] for r in csv.DictReader(f) if r["dataset"] == "epic_kitchens_100"]


def shard_ids(manifest, k, n):
    """Longest-first greedy split of the EPIC ids into n shards of similar total duration."""
    with open(manifest, newline="") as f:
        rows = [(float(r["duration_sec"]), r["video_id"]) for r in csv.DictReader(f)
                if r["dataset"] == "epic_kitchens_100"]
    load, assign = [0.0] * n, [[] for _ in range(n)]
    for d, vid in sorted(rows, reverse=True):
        j = load.index(min(load))
        load[j] += d
        assign[j].append(vid)
    return assign[k]


def expected_md5(scripts, vid):
    """MD5 of <vid>.MP4 from the downloader's own data/md5.csv (None if not listed)."""
    try:
        with open(os.path.join(scripts, "data", "md5.csv"), newline="") as f:
            for r in csv.DictReader(f):
                if r["file_remote_path"].endswith(f"/{vid}.MP4"):
                    return r["md5"]
    except OSError:
        pass
    return None


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(vid, scripts, work, log, attempts=3, timeout_s=40 * 60):
    """Official downloader, then verify the file against its published MD5 (a truncated
    download once exited 0 and produced an undecodable file)."""
    dest = os.path.join(work, vid)
    want = expected_md5(scripts, vid)
    for attempt in range(1, attempts + 1):
        try:
            subprocess.run([sys.executable, "epic_downloader.py", "--videos", "--specific-videos", vid,
                            "--output-path", dest], cwd=scripts, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=timeout_s)
            hits = glob.glob(os.path.join(dest, "**", f"{vid}.MP4"), recursive=True) + \
                glob.glob(os.path.join(dest, "**", f"{vid}.mp4"), recursive=True)
            if not hits:
                raise FileNotFoundError(f"{vid}.MP4 not found after download")
            got = file_md5(hits[0])
            if want and got != want:
                raise ValueError(f"MD5 mismatch ({os.path.getsize(hits[0])} bytes, {got} != {want})")
            log(f"download {vid} ok ({os.path.getsize(hits[0]) / 1e6:.0f} MB, md5 {'verified' if want else 'not listed'})")
            return hits[0]
        except Exception as e:
            log(f"download {vid} attempt {attempt} failed: {e}")
            shutil.rmtree(dest, ignore_errors=True)
            time.sleep(10 * attempt)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/outputs")
    ap.add_argument("--work", default="/tmp/epic")
    ap.add_argument("--scripts", default="/tmp/epic-kitchens-download-scripts")
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data", "manifest.csv"))
    ap.add_argument("--shard", default="0/1", help="k/n: shard k of n (duration-balanced)")
    ap.add_argument("--resume-from", default=None, help="dir with JSONs from an earlier session")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="comma-separated manifest ids to (re)process instead of the full list")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.work, exist_ok=True)
    logf = open(os.path.join(a.out, "..", "epic_run.log"), "a")
    fails = os.path.join(a.out, "..", "failures.txt")

    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    if a.resume_from and os.path.isdir(a.resume_from):
        for p in glob.glob(os.path.join(a.resume_from, "*.json")):
            if not os.path.exists(os.path.join(a.out, os.path.basename(p))):
                shutil.copy(p, a.out)
    if not os.path.isdir(a.scripts):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/epic-kitchens/epic-kitchens-download-scripts", a.scripts], check=True)

    k, n = (int(x) for x in a.shard.split("/"))
    if a.ids:  # re-run a subset of manifest ids (e.g. earlier failures), one in every n
        valid = set(epic_ids(a.manifest))
        sel = [v for v in a.ids.split(",") if v in valid]
        ids = sel[k::n]
    else:
        ids = shard_ids(a.manifest, k, n)
    todo = [v for v in ids if not os.path.exists(os.path.join(a.out, f"{v}.json"))]
    if a.limit:
        todo = todo[: a.limit]
    log(f"shard {a.shard}: {len(ids)} ids, {len(todo)} to do")

    import torch
    from clip_model import ClipScorer
    from infer import WEIGHTS, process
    from tsm import load_tsm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = (load_tsm(WEIGHTS, device), ClipScorer(device))
    log(f"models loaded on {device}")

    pending = {}

    def prefetch(vid):
        pending[vid] = download(vid, a.scripts, a.work, log)

    th = None
    if todo:
        th = threading.Thread(target=prefetch, args=(todo[0],))
        th.start()
    t_start, done_sec = time.time(), 0.0
    for i, vid in enumerate(todo):
        th.join()
        path = pending.pop(vid)
        if i + 1 < len(todo):  # keep at most 2 videos on disk
            th = threading.Thread(target=prefetch, args=(todo[i + 1],))
            th.start()
        if path is None:
            open(fails, "a").write(f"{vid}\tdownload failed\n")
            continue
        out = os.path.join(a.out, f"{vid}.json")
        ok = False
        for attempt in (1, 2):
            try:
                t0 = time.time()
                res = process(path, out, device, models)
                done_sec += res["duration_sec"]
                el = time.time() - t_start
                log(f"{vid} ok in {time.time() - t0:.0f}s | {i + 1}/{len(todo)} | "
                    f"video-hours/hour {done_sec / max(el, 1):.2f}")
                ok = True
                break
            except Exception:
                log(f"{vid} attempt {attempt} failed:\n{traceback.format_exc()}")
                torch.cuda.empty_cache()
        if not ok:
            open(fails, "a").write(f"{vid}\tinference failed\n")
        shutil.rmtree(os.path.join(a.work, vid), ignore_errors=True)
    log(f"finished; {len(glob.glob(os.path.join(a.out, '*.json')))} JSONs in {a.out}")


if __name__ == "__main__":
    main()
