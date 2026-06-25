#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Iterable


def safe_ratio(num: float, den: float) -> float:
    if abs(float(den)) < 1e-12:
        return 0.0
    return float(num) / float(den)


def mean_or_none(values: Iterable[float]) -> float | None:
    seq = [float(v) for v in values]
    if not seq:
        return None
    return sum(seq) / float(len(seq))


def median_or_none(values: Iterable[float]) -> float | None:
    seq = sorted(float(v) for v in values)
    if not seq:
        return None
    mid = len(seq) // 2
    if len(seq) % 2 == 1:
        return seq[mid]
    return (seq[mid - 1] + seq[mid]) / 2.0


def summarize_latencies(values: Iterable[int]) -> dict[str, float | int | None]:
    seq = sorted(int(v) for v in values)
    if not seq:
        return {"count": 0, "min": None, "median": None, "max": None, "mean": None}
    return {
        "count": len(seq),
        "min": seq[0],
        "median": median_or_none(seq),
        "max": seq[-1],
        "mean": mean_or_none(seq),
    }


def top2_margin(scores: Iterable[float]) -> float:
    seq = sorted((float(v) for v in scores), reverse=True)
    if not seq:
        return 0.0
    if len(seq) == 1:
        return seq[0]
    return seq[0] - seq[1]


def normalized_entropy(scores: Iterable[float]) -> float:
    seq = [max(float(v), 0.0) for v in scores]
    if not seq:
        return 0.0
    total = sum(seq)
    if total <= 1e-12:
        return 0.0
    probs = [v / total for v in seq if v > 0.0]
    if len(probs) <= 1:
        return 0.0
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    return float(entropy / math.log(len(probs)))
