#!/usr/bin/env python3
"""Action-aware video summarisation: one first-person MP4 -> one Track 4 JSON.

usage: python code/infer.py <video.mp4> <out.json>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import highlights as H  # noqa: E402
from clip_model import ClipScorer, novelty  # noqa: E402
from tsm import INPUT_MEAN, INPUT_STD, NUM_SEGMENTS, NUM_VERBS, load_tsm  # noqa: E402
from validate import load_vocab, validate_obj  # noqa: E402
from video_io import decode, probe_duration  # noqa: E402

TEAM_ID = os.environ.get("TEAM_ID", "Toothpaste Eater")
FPS = float(os.environ.get("DECODE_FPS", "4"))
WINDOWS = tuple(int(x) for x in os.environ.get("TSM_WINDOWS", "2,4,8").split(","))
WEIGHTS = os.environ.get("TSM_WEIGHTS", os.path.join(ROOT, "weights", "tsm_rgb_epic100_fp16.pt"))
CLASSES = os.path.join(ROOT, "data", "supported_classes.json")


def log(msg):
    print(f"[infer {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- TSM windows
def window_indices(n_frames: int, duration: float, length: float, fps: float):
    """Start times (stride 1 s) and the 8 frame indices sampled uniformly in each window."""
    if duration <= length:
        starts = np.array([0.0])
        lens = np.array([duration])
    else:
        starts = np.arange(0, math.floor(duration - length) + 1, 1.0)
        lens = np.full(len(starts), float(length))
    offs = (np.arange(NUM_SEGMENTS) + 0.5) / NUM_SEGMENTS
    t = starts[:, None] + offs[None, :] * lens[:, None]
    idx = np.clip(np.floor(t * fps).astype(int), 0, n_frames - 1)
    return starts, starts + lens, idx


@torch.no_grad()
def run_tsm(model, frames: np.ndarray, idx: np.ndarray, device: str, batch: int):
    mean = torch.tensor(INPUT_MEAN, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(INPUT_STD, device=device).view(1, 1, 3, 1, 1)
    vp, np_ = [], []
    for i in range(0, len(idx), batch):
        x = torch.from_numpy(frames[idx[i:i + batch]]).to(device, non_blocking=True)
        x = x.permute(0, 1, 4, 2, 3).float().div_(255.0)
        x = (x - mean) / std
        with torch.autocast("cuda", enabled=device.startswith("cuda")):
            v, n = model(x)
        vp.append(v.float().softmax(-1).cpu())
        np_.append(n.float().softmax(-1).cpu())
    return torch.cat(vp).numpy(), torch.cat(np_).numpy()


# ---------------------------------------------------------------- actions (mAP)
def propose_actions(per_scale, top_v=5, top_n=5, keep=3):
    S, E, V, N, P = [], [], [], [], []
    for starts, ends, pv, pn in per_scale:
        vi = np.argsort(-pv, axis=1)[:, :top_v]
        ni = np.argsort(-pn, axis=1)[:, :top_n]
        vs = np.take_along_axis(pv, vi, 1)
        ns = np.take_along_axis(pn, ni, 1)
        pair = (vs[:, :, None] * ns[:, None, :]).reshape(len(pv), -1)
        best = np.argsort(-pair, axis=1)[:, :keep]
        for k in range(keep):
            b = best[:, k]
            S.append(starts)
            E.append(ends)
            V.append(vi[np.arange(len(b)), b // top_n])
            N.append(ni[np.arange(len(b)), b % top_n])
            P.append(pair[np.arange(len(b)), b])
    return tuple(np.concatenate(a) for a in (S, E, V, N, P))


def temporal_nms(s, e, v, n, p, iou_thr=0.5, max_out=2000, pre_top=30000):
    order = np.argsort(-p)[:pre_top]
    s, e, v, n, p = s[order], e[order], v[order], n[order], p[order]
    key = v * 1000 + n
    keep = []
    for k in np.unique(key):
        ids = np.where(key == k)[0]  # already sorted by score
        cs, ce = s[ids], e[ids]
        alive = np.ones(len(ids), dtype=bool)
        for j in range(len(ids)):
            if not alive[j]:
                continue
            keep.append(ids[j])
            inter = np.clip(np.minimum(ce, ce[j]) - np.maximum(cs, cs[j]), 0, None)
            union = (ce - cs) + (ce[j] - cs[j]) - inter
            sup = inter / np.maximum(union, 1e-6) >= iou_thr
            sup[: j + 1] = False
            alive &= ~sup
    keep = np.array(sorted(keep, key=lambda i: -p[i])[:max_out], dtype=int)
    return s[keep], e[keep], v[keep], n[keep], p[keep]


# ---------------------------------------------------------------- per-second signals
def per_second(values: np.ndarray, fps: float, n_sec: int) -> np.ndarray:
    """Average frame-level values into 1 s bins."""
    f = int(round(fps))
    out = np.zeros(n_sec)
    for s in range(n_sec):
        chunk = values[s * f:(s + 1) * f]
        out[s] = chunk.mean() if len(chunk) else (out[s - 1] if s else 0.0)
    return out


def zscore(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-8 else np.zeros_like(x)


def smooth(x, k=3):
    if len(x) < k:
        return x
    return np.convolve(np.pad(x, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def pair_probability_per_second(scale, n_sec, pairs_v, pairs_n):
    """q[s, p]: mean over the windows covering second s of p_v * p_n."""
    starts, ends, pv, pn = scale
    acc = np.zeros((n_sec, len(pairs_v)))
    cnt = np.zeros(n_sec)
    q = pv[:, pairs_v] * pn[:, pairs_n]
    for w in range(len(starts)):
        a, b = int(starts[w]), min(n_sec, int(math.ceil(ends[w])))
        acc[a:b] += q[w]
        cnt[a:b] += 1
    return acc / np.maximum(cnt, 1)[:, None]


# ---------------------------------------------------------------- main
def process(video: str, out_path: str, device: str | None = None, models=None) -> dict:
    t0 = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    video_id = os.path.splitext(os.path.basename(video))[0]
    vocab_json = json.load(open(CLASSES))
    labels, verbs, nouns = vocab_json["labels"], vocab_json["verbs"], vocab_json["nouns"]

    duration = probe_duration(video)
    frames = decode(video, FPS)
    log(f"{video_id}: duration {duration:.2f}s, {len(frames)} frames @ {FPS} fps "
        f"({time.time() - t0:.1f}s)")
    duration_eff = min(duration, len(frames) / FPS + 1.0 / FPS)
    n_sec = int(math.ceil(duration_eff))

    tsm, clip = models if models else (load_tsm(WEIGHTS, device), ClipScorer(device))

    # 1. CLIP: label + interestingness + novelty
    emb = clip.images(frames)
    label, label_score = clip.label(emb, labels)
    interest = clip.interestingness(emb, label)
    nov = novelty(emb, int(round(FPS)))
    log(f"label={label} ({label_score:.2f}) ({time.time() - t0:.1f}s)")

    # 2. TSM sliding windows
    batch = 32 if device.startswith("cuda") else 4
    per_scale = []
    for L in WINDOWS:
        st, en, idx = window_indices(len(frames), duration_eff, L, FPS)
        pv, pn = run_tsm(tsm, frames, idx, device, batch)
        per_scale.append((st, en, pv, pn))
    log(f"TSM windows done ({sum(len(x[0]) for x in per_scale)}) ({time.time() - t0:.1f}s)")

    # 3. actions
    s, e, v, n, p = temporal_nms(*propose_actions(per_scale))
    actions = [{"segment": [round(float(a), 2), round(float(min(b, duration)), 2)],
                "verb": verbs[int(vi)], "noun": nouns[int(ni)], "score": round(float(sc), 4)}
               for a, b, vi, ni, sc in zip(s, e, v, n, p) if round(float(min(b, duration)), 2) > round(float(a), 2)]

    # 4. per-second signals
    motion_f = np.concatenate([[0.0], np.abs(np.diff(frames.astype(np.int16), axis=0)).mean(axis=(1, 2, 3))]) \
        if len(frames) > 1 else np.zeros(len(frames))
    motion = per_second(motion_f, FPS, n_sec)
    interest_s = per_second(interest, FPS, n_sec)
    nov_s = per_second(nov, FPS, n_sec)
    pv2 = per_scale[0][2]
    actionness_w = 1.0 - (-(pv2 * np.log(pv2 + 1e-12)).sum(1) / math.log(NUM_VERBS))
    actionness = np.zeros(n_sec)
    for w, (a, b) in enumerate(zip(per_scale[0][0], per_scale[0][1])):
        actionness[int(a):min(n_sec, int(math.ceil(b)))] = np.maximum(
            actionness[int(a):min(n_sec, int(math.ceil(b)))], actionness_w[w])

    # 5/6/7. highlights, strategy chosen by the predicted label
    if label == "cooking":
        # candidate pair vocabulary = distinct pairs among the proposals
        uniq = sorted({(int(a), int(b)) for a, b in zip(v, n)}, key=lambda t: t)
        pairs_v = np.array([u[0] for u in uniq], dtype=int)
        pairs_n = np.array([u[1] for u in uniq], dtype=int)
        q = np.zeros((n_sec, len(uniq)))
        for sc in per_scale[:2]:  # 2 s and 4 s windows localise best
            q = np.maximum(q, pair_probability_per_second(sc, n_sec, pairs_v, pairs_n))
        segs = H.coverage_greedy(q, duration_eff)
        fallback = actionness
        strategy = "coverage"
    else:
        importance = smooth(zscore(interest_s) + 0.5 * zscore(motion) + 0.5 * zscore(nov_s), 5)
        segs = H.importance_greedy(importance, duration_eff)
        fallback = importance
        strategy = "importance"
    hl = H.finalize(segs, min(duration, duration_eff), fallback)

    out = {
        "team_id": TEAM_ID,
        "video_id": video_id,
        "duration_sec": round(float(duration), 2),
        "label": {"name": label, "score": round(min(1.0, max(0.0, label_score)), 4)},
        "actions": actions,
        "highlights": [{"segment": seg} for seg in hl],
    }
    errors, warnings = validate_obj(out, video_id + ".json", {}, load_vocab(CLASSES))
    if errors:
        raise RuntimeError(f"output failed validation: {errors[:5]}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    tot = sum(b - a for a, b in hl)
    log(f"{video_id}: {strategy} highlights {len(hl)} segs {tot:.1f}s ({100 * tot / duration:.1f}%), "
        f"{len(actions)} actions, warnings={len(warnings)}, total {time.time() - t0:.1f}s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    process(a.video, a.out, a.device)


if __name__ == "__main__":
    main()
