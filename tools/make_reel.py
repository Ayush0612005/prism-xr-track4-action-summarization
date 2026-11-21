#!/usr/bin/env python3
"""Render a highlight reel (the JSON's highlight segments, concatenated) for visual spot checks.

usage: python tools/make_reel.py <video.mp4> <out.json> <reel.mp4>
"""
import json
import shutil
import subprocess
import sys

try:
    import imageio_ffmpeg
    FFMPEG = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = "ffmpeg"

video, js, out = sys.argv[1:4]
segs = [h["segment"] for h in json.load(open(js))["highlights"]]
parts = []
for i, (a, b) in enumerate(segs):
    parts.append(f"[0:v]trim=start={a}:end={b},setpts=PTS-STARTPTS,scale=-2:480,"
                 f"drawtext=text='{a:.1f}-{b:.1f}s':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5[v{i}]")
graph = ";".join(parts) + ";" + "".join(f"[v{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=0[out]"
cmd = [FFMPEG, "-y", "-v", "error", "-i", video, "-filter_complex", graph, "-map", "[out]",
       "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", out]
try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError:  # ffmpeg builds without drawtext: drop the timestamp overlay
    plain = [p.split(",drawtext=")[0] + f"[v{i}]" for i, p in enumerate(parts)]
    graph = ";".join(plain) + ";" + "".join(f"[v{i}]" for i in range(len(segs))) + \
        f"concat=n={len(segs)}:v=1:a=0[out]"
    cmd[cmd.index("-filter_complex") + 1] = graph
    subprocess.run(cmd, check=True)
print(f"wrote {out}: {len(segs)} segments, {sum(b - a for a, b in segs):.1f} s")
