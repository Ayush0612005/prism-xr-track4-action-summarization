#!/usr/bin/env python3
"""Download the FPVSum videos listed in the manifest with yt-dlp.

The YouTube id comes from FPVSum's crawler/FPVSum_videolist.csv (video_name -> youtube_id;
the *_1/_3 clips have their own YouTube ids). Video-only <=480p MP4 is enough because the
pipeline works on 224x224 crops. Duration is checked against the manifest (+-0.5 s).

usage: python tools/fetch_fpvsum.py --videolist <fpvsum>/crawler/FPVSum_videolist.csv --out <dir>
"""
import argparse
import csv
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from video_io import probe_duration  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videolist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data", "manifest.csv"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    yt = {r["video_name"]: r["youtube_id"] for r in csv.DictReader(open(a.videolist))}
    rows = [r for r in csv.DictReader(open(a.manifest)) if r["dataset"] == "fpvsum"]
    bad = []
    for r in rows:
        vid, want = r["video_id"], float(r["duration_sec"])
        path = os.path.join(a.out, f"{vid}.mp4")
        if not os.path.exists(path):
            ytid = yt.get(vid)
            if ytid is None:
                bad.append((vid, "not in FPVSum video list"))
                continue
            for attempt in (1, 2):
                rc = subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings",
                                     "-f", "bv*[height<=480][ext=mp4]/b[height<=480][ext=mp4]/bv*[ext=mp4]/b",
                                     "-o", path, f"https://www.youtube.com/watch?v={ytid}"]).returncode
                if rc == 0 and os.path.exists(path):
                    break
        if not os.path.exists(path):
            bad.append((vid, "download failed"))
            continue
        got = probe_duration(path)
        status = "ok" if abs(got - want) <= 0.5 else "DURATION MISMATCH"
        if status != "ok":
            bad.append((vid, f"duration {got:.2f} vs manifest {want:.2f}"))
        print(f"{vid}: {got:.2f}s (manifest {want:.2f}) {status}", flush=True)
    print(f"{len(rows) - len(bad)}/{len(rows)} ok")
    for b in bad:
        print("FAIL", *b)


if __name__ == "__main__":
    main()
