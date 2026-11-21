"""Highlight selection under the 15% budget with segments of at least 4 s."""
from __future__ import annotations

import math

import numpy as np

MIN_SEG = 4.0
BUDGET_FRAC = 0.148  # 15% rule with a little rounding headroom


def budget(duration: float) -> float:
    return BUDGET_FRAC * duration


def _sliding_max(q: np.ndarray, length: int) -> np.ndarray:
    """q: (S, K) -> (S-length+1, K) max over each window of `length` rows."""
    out = q[: q.shape[0] - length + 1].copy()
    for i in range(1, length):
        np.maximum(out, q[i: q.shape[0] - length + 1 + i], out=out)
    return out


def coverage_greedy(pair_q: np.ndarray, duration: float, lengths=(4, 6, 8)) -> list[tuple[float, float]]:
    """Weighted max-coverage over action pairs.

    pair_q[s, p] is the probability that action pair p happens during second s.
    A window covers pair p with q = max over its seconds; coverage of p over the chosen
    windows is 1 - prod(1 - q). Greedily add the non-overlapping window with the best
    coverage gain per second until the budget is spent.
    """
    S = pair_q.shape[0]
    usable = int(math.floor(min(S, duration)))
    total = budget(duration)
    if total < MIN_SEG or usable < MIN_SEG:
        return []
    starts, lens, qs = [], [], []
    for L in lengths:
        if L > usable:
            continue
        q = _sliding_max(pair_q[:usable], L)
        starts.append(np.arange(q.shape[0]))
        lens.append(np.full(q.shape[0], L))
        qs.append(q)
    if not qs:
        return []
    starts, lens = np.concatenate(starts), np.concatenate(lens)
    Q = np.concatenate(qs).astype(np.float64)
    cov = np.zeros(Q.shape[1])
    alive = np.ones(len(starts), dtype=bool)
    chosen, remaining = [], total
    while remaining >= MIN_SEG and alive.any():
        alive &= lens <= remaining + 1e-9
        if not alive.any():
            break
        gain = Q @ (1.0 - cov)
        ratio = np.where(alive, gain / lens, -np.inf)
        i = int(np.argmax(ratio))
        if not np.isfinite(ratio[i]) or ratio[i] <= 1e-12:
            break
        s, L = int(starts[i]), int(lens[i])
        chosen.append((float(s), float(s + L)))
        cov = 1.0 - (1.0 - cov) * (1.0 - Q[i])
        remaining -= L
        # drop candidates overlapping the chosen interval (touching is fine; merged later)
        alive &= (starts + lens <= s) | (starts >= s + L)
    return chosen


def importance_greedy(score: np.ndarray, duration: float, lengths=(4, 5, 6, 7, 8)) -> list[tuple[float, float]]:
    """Pick the highest-mean non-overlapping windows (4-8 s) until the budget is spent."""
    usable = int(math.floor(min(len(score), duration)))
    total = budget(duration)
    if total < MIN_SEG or usable < MIN_SEG:
        return []
    cs = np.concatenate([[0.0], np.cumsum(score[:usable])])
    starts, lens, means = [], [], []
    for L in lengths:
        if L > usable:
            continue
        s = np.arange(usable - L + 1)
        starts.append(s)
        lens.append(np.full_like(s, L))
        means.append((cs[s + L] - cs[s]) / L)
    starts, lens, means = map(np.concatenate, (starts, lens, means))
    alive = np.ones(len(starts), dtype=bool)
    chosen, remaining = [], total
    while remaining >= MIN_SEG:
        alive &= lens <= remaining + 1e-9
        if not alive.any():
            break
        i = int(np.argmax(np.where(alive, means, -np.inf)))
        s, L = int(starts[i]), int(lens[i])
        chosen.append((float(s), float(s + L)))
        remaining -= L
        alive &= (starts + lens <= s) | (starts >= s + L)
    return chosen


def best_single(score_per_sec: np.ndarray, duration: float) -> list[tuple[float, float]]:
    """For very short videos (15% < 4 s): one 4.0 s segment at the best position."""
    L = MIN_SEG
    if duration <= L:
        return [(0.0, round(duration, 2))]
    n = int(math.floor(duration - L)) + 1
    sc = [float(np.mean(score_per_sec[s:s + int(L)])) if len(score_per_sec) else 0.0 for s in range(n)]
    s = int(np.argmax(sc))
    return [(float(s), float(s) + L)]


def finalize(segs: list[tuple[float, float]], duration: float, fallback_score: np.ndarray) -> list[list[float]]:
    """Sort, merge touching/overlapping, clamp to the video, enforce >= 4 s, round."""
    segs = [(max(0.0, a), min(duration, b)) for a, b in segs if b > a]
    segs = [s for s in segs if s[1] - s[0] >= MIN_SEG - 1e-6]
    if not segs:
        segs = best_single(fallback_score, duration)
    segs.sort()
    merged = [list(segs[0])]
    for a, b in segs[1:]:
        if a <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        a, b = round(a, 2), round(b, 2)
        if out and a <= out[-1][1]:
            a = round(out[-1][1] + 0.01, 2)
        if b - a >= MIN_SEG - 1e-6 or not out:
            out.append([a, b])
    return out
