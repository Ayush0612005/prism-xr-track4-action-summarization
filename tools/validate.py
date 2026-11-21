#!/usr/bin/env python3
"""Validator for Track 4 output JSONs.

The kit README lists a validate.py that was not included in the distributed zip, so this
re-implements every check the README describes, plus manifest/vocabulary checks:
  * exactly the 6 top-level fields, correct types, finite numbers
  * team_id non-empty and identical across files; video_id == filename (and in manifest)
  * duration_sec > 0 and within +-0.5 s of manifest.csv
  * label.name in supported labels, label.score in [0, 1]
  * actions: 0-2000 items, 0 <= start < end <= duration + 0.5, score in [0, 1], verb/noun in vocabulary
  * highlights: >= 1 item, sorted, strictly non-overlapping, >= 1 segment of >= 4.0 s,
    warnings for segments < 4.0 s and for total > 15% of duration
  * timestamps with at most 2 decimals (warning)

Usage: python tools/validate.py --file out.json -v
       python tools/validate.py --dir outputs/ [--manifest data/manifest.csv] [--expect-all]
Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
KEYS = {"team_id", "video_id", "duration_sec", "label", "actions", "highlights"}


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _decimals_ok(x) -> bool:
    return abs(round(x, 2) - x) < 1e-9


def load_manifest(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {r["video_id"]: float(r["duration_sec"]) for r in csv.DictReader(f)}


def load_vocab(path):
    with open(path) as f:
        d = json.load(f)
    return set(d["labels"]), set(d["verbs"]), set(d["nouns"])


def validate_obj(obj, fname, manifest, vocab):
    """Returns (errors, warnings)."""
    E, W = [], []
    labels, verbs, nouns = vocab
    if not isinstance(obj, dict):
        return ["top level is not an object"], W
    keys = set(obj)
    if keys != KEYS:
        if KEYS - keys:
            E.append(f"missing fields {sorted(KEYS - keys)}")
        if keys - KEYS:
            E.append(f"unexpected fields {sorted(keys - KEYS)}")
        if KEYS - keys:
            return E, W
    if not isinstance(obj["team_id"], str) or not obj["team_id"].strip():
        E.append("team_id must be a non-empty string")
    vid = obj["video_id"]
    if not isinstance(vid, str) or not vid:
        E.append("video_id must be a non-empty string")
    elif fname is not None and os.path.splitext(os.path.basename(fname))[0] != vid:
        E.append(f"video_id '{vid}' does not match filename '{os.path.basename(fname)}'")
    elif manifest and vid not in manifest:
        E.append(f"video_id '{vid}' not in manifest")
    dur = obj["duration_sec"]
    if not _num(dur) or dur <= 0:
        E.append("duration_sec must be a positive finite number")
        return E, W
    if manifest and isinstance(vid, str) and vid in manifest and abs(dur - manifest[vid]) > 0.5:
        E.append(f"duration_sec {dur} differs from manifest {manifest[vid]} by > 0.5 s")

    lab = obj["label"]
    if not isinstance(lab, dict) or set(lab) != {"name", "score"}:
        E.append("label must be {name, score}")
    else:
        if not isinstance(lab["name"], str) or lab["name"] not in labels:
            E.append(f"label.name '{lab.get('name')}' not in supported labels")
        if not _num(lab["score"]) or not 0 <= lab["score"] <= 1:
            E.append("label.score must be in [0, 1]")

    acts = obj["actions"]
    if not isinstance(acts, list):
        E.append("actions must be an array")
    else:
        if len(acts) > 2000:
            E.append(f"actions has {len(acts)} items (max 2000)")
        for i, a in enumerate(acts):
            if not isinstance(a, dict) or set(a) != {"segment", "verb", "noun", "score"}:
                E.append(f"actions[{i}] must be {{segment, verb, noun, score}}")
                continue
            seg = a["segment"]
            if not (isinstance(seg, list) and len(seg) == 2 and all(_num(t) for t in seg)):
                E.append(f"actions[{i}].segment must be [start, end] numbers")
                continue
            s, e = seg
            if not (0 <= s < e <= dur + 0.5):
                E.append(f"actions[{i}].segment {seg} violates 0 <= start < end <= duration+0.5")
            if not (_decimals_ok(s) and _decimals_ok(e)):
                W.append(f"actions[{i}].segment has more than 2 decimals")
            if a["verb"] not in verbs:
                E.append(f"actions[{i}].verb '{a['verb']}' not in vocabulary")
            if a["noun"] not in nouns:
                E.append(f"actions[{i}].noun '{a['noun']}' not in vocabulary")
            if not _num(a["score"]) or not 0 <= a["score"] <= 1:
                E.append(f"actions[{i}].score must be in [0, 1]")

    hl = obj["highlights"]
    if not isinstance(hl, list) or not hl:
        E.append("highlights must be a non-empty array")
        return E, W
    prev_end, total, longest = None, 0.0, 0.0
    for i, h in enumerate(hl):
        if not isinstance(h, dict) or set(h) != {"segment"}:
            E.append(f"highlights[{i}] must be {{segment}}")
            continue
        seg = h["segment"]
        if not (isinstance(seg, list) and len(seg) == 2 and all(_num(t) for t in seg)):
            E.append(f"highlights[{i}].segment must be [start, end] numbers")
            continue
        s, e = seg
        if not (0 <= s < e <= dur + 0.5):
            E.append(f"highlights[{i}].segment {seg} violates 0 <= start < end <= duration+0.5")
        if prev_end is not None and s < prev_end:
            E.append(f"highlights[{i}] overlaps or is out of order")
        if prev_end is not None and s == prev_end:
            W.append(f"highlights[{i}] touches the previous segment (merge them)")
        if not (_decimals_ok(s) and _decimals_ok(e)):
            W.append(f"highlights[{i}].segment has more than 2 decimals")
        if e - s < 4.0:
            W.append(f"highlights[{i}] is {e - s:.2f} s (< 4.0 s, ignored in scoring)")
        prev_end = e
        total += e - s
        longest = max(longest, e - s)
    if longest < 4.0:
        E.append("no highlight segment is >= 4.0 s")
    if total > 0.15 * dur + 1e-6 and not (0.15 * dur < 4.0 and total <= 4.0 + 1e-6):
        W.append(f"highlights total {total:.2f} s exceeds 15% of duration ({0.15 * dur:.2f} s)")
    return E, W


def validate_file(path, manifest=None, vocab=None):
    vocab = vocab or load_vocab(os.path.join(DATA, "supported_classes.json"))
    try:
        with open(path) as f:
            obj = json.load(f)
    except Exception as e:
        return [f"cannot parse JSON: {e}"], [], None
    E, W = validate_obj(obj, path, manifest or {}, vocab)
    return E, W, obj.get("team_id") if isinstance(obj, dict) else None


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file")
    g.add_argument("--dir")
    ap.add_argument("--manifest", default=os.path.join(DATA, "manifest.csv"))
    ap.add_argument("--classes", default=os.path.join(DATA, "supported_classes.json"))
    ap.add_argument("--expect-all", action="store_true", help="require one JSON per manifest video")
    ap.add_argument("-v", "--verbose", action="store_true", help="print warnings")
    args = ap.parse_args()
    manifest, vocab = load_manifest(args.manifest), load_vocab(args.classes)
    files = [args.file] if args.file else sorted(
        os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith(".json"))
    n_err, n_warn, teams = 0, 0, set()
    for p in files:
        E, W, team = validate_file(p, manifest, vocab)
        if team:
            teams.add(team)
        n_err += len(E)
        n_warn += len(W)
        for m in E:
            print(f"ERROR {os.path.basename(p)}: {m}")
        if args.verbose:
            for m in W:
                print(f"WARN  {os.path.basename(p)}: {m}")
    if len(teams) > 1:
        print(f"ERROR team_id differs across files: {sorted(teams)}")
        n_err += 1
    if args.dir and args.expect_all and manifest:
        have = {os.path.splitext(os.path.basename(p))[0] for p in files}
        missing = sorted(set(manifest) - have)
        if missing:
            print(f"ERROR {len(missing)} manifest videos have no JSON, e.g. {missing[:5]}")
            n_err += 1
    print(f"{len(files)} file(s), {n_err} error(s), {n_warn} warning(s)")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
