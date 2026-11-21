#!/usr/bin/env python3
"""Development-time report: predicted label vs FPVSum category (taken from the file name).

Measurement only. The category in the file name is never read by the pipeline (code/);
this script only compares finished outputs against it.
usage: python tools/label_report.py outputs/
"""
import collections
import glob
import json
import os
import sys

MAP = {"bikepolo": "bikepolo", "rockclimbing": "rockclimbing"}
rows = []
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    d = json.load(open(p))
    if d["video_id"].startswith("P") and d["video_id"][1:3].isdigit():
        continue  # EPIC
    cat = d["video_id"].split("_")[0].lower()
    rows.append((d["video_id"], MAP.get(cat, cat), d["label"]["name"], d["label"]["score"]))
ok = sum(t == p for _, t, p, _ in rows)
conf = collections.Counter((t, p) for _, t, p, _ in rows if t != p)
print(f"label accuracy {ok}/{len(rows)} = {ok / max(len(rows), 1):.3f}")
for (t, p), c in conf.most_common():
    print(f"  {t:>13} -> {p:<13} x{c}")
