#!/usr/bin/env python3
"""TrustTrack Association v1: margin-calibrated appearance veto.

This is a narrow online intervention on the *real* primary association cost:
- keep the original DMM/BoT-SORT tracker, detections, Kalman state and lifecycle;
- call the original assoc_cost() to obtain the exact baseline IoU/ReID costs;
- when ReID wins a pair but its row/column competition margin is weak while
  IoU is more discriminative, add a small penalty to the ReID cost;
- run the original Hungarian matching on the adjusted final matrix.

The goal is to reduce identity-confusing appearance takeovers in crowded scenes
without globally replacing the baseline fusion rule.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import dmm_base_tracker_shadow_pda as spda
import dmm_base_tracker_shadow_pda_v3 as v3

_ORIG_ASSOC_COST = spda.base.assoc_cost
_STATS: Dict[str, float] = {
    "assoc_calls": 0,
    "candidate_pairs": 0,
    "valid_pairs": 0,
    "app_winning_pairs": 0,
    "penalized_pairs": 0,
    "changed_pairs": 0,
    "penalty_sum": 0.0,
    "penalty_max_seen": 0.0,
}
_CFG = None


def _pair_margin(cost: np.ndarray, valid: np.ndarray, single_margin: float = 1.0) -> np.ndarray:
    """Signed local separation for every pair in a lower-is-better cost matrix.

    margin(i,j) = min(best alternative in row i, best alternative in column j)
                  - cost(i,j)
    Positive: this pair beats both row/column competitors.
    Negative: at least one competitor is better.
    """
    n_t, n_d = cost.shape
    out = np.full((n_t, n_d), -1.0, dtype=np.float32)
    if n_t == 0 or n_d == 0:
        return out
    big = 1e6
    safe = np.where(valid & np.isfinite(cost), cost, big).astype(np.float32)
    for i in range(n_t):
        for j in range(n_d):
            if not valid[i, j]:
                continue
            chosen = float(safe[i, j])
            row = safe[i]
            col = safe[:, j]
            if n_d > 1:
                row_other = float(np.min(np.concatenate([row[:j], row[j + 1:]])))
            else:
                row_other = chosen + single_margin
            if n_t > 1:
                col_other = float(np.min(np.concatenate([col[:i], col[i + 1:]])))
            else:
                col_other = chosen + single_margin
            if row_other >= big * 0.5:
                row_other = chosen + single_margin
            if col_other >= big * 0.5:
                col_other = chosen + single_margin
            out[i, j] = min(row_other - chosen, col_other - chosen)
    return out


def _patched_assoc_cost(tracks, detections, cfg):
    baseline, debug = _ORIG_ASSOC_COST(tracks, detections, cfg)
    conf = _CFG
    if conf is None or not conf.enable or baseline.size == 0 or cfg.assoc_mode != "botsort_reid":
        return baseline, debug

    raw_iou = np.asarray(debug.get("raw_iou"), dtype=np.float32)
    emb_raw = debug.get("emb")
    if emb_raw is None:
        return baseline, debug
    emb_raw = np.asarray(emb_raw, dtype=np.float32)
    if raw_iou.shape != baseline.shape or emb_raw.shape != baseline.shape:
        return baseline, debug

    # Reconstruct the exact valid regions of the baseline MOT20 fusion.
    iou_valid = np.isfinite(raw_iou) & (raw_iou <= float(cfg.proximity_thresh))
    app_valid = iou_valid & np.isfinite(emb_raw) & (emb_raw <= float(cfg.appearance_thresh))
    app_margin = _pair_margin(emb_raw, app_valid)
    iou_margin = _pair_margin(raw_iou, iou_valid)

    # Baseline chooses ReID only when it is strictly cheaper than IoU.
    app_wins = app_valid & (emb_raw + float(conf.win_epsilon) < raw_iou)

    track_age = np.asarray([int(getattr(t, "tracklet_len", 0)) for t in tracks], dtype=np.int32)
    age_ok = track_age[:, None] >= int(conf.min_track_age)

    weak_app = app_margin < float(conf.app_margin_thresh)
    iou_more_reliable = iou_margin >= (app_margin + float(conf.iou_advantage_margin))
    penalize = app_wins & age_ok & weak_app & iou_more_reliable

    # Continuous penalty: strongest when app margin is far below the threshold.
    denom = max(float(conf.app_margin_thresh), 1e-6)
    risk = np.clip((float(conf.app_margin_thresh) - app_margin) / denom, 0.0, 1.0)
    penalty = np.where(penalize, float(conf.max_penalty) * risk, 0.0).astype(np.float32)

    adjusted_app = emb_raw.copy()
    adjusted_app[~app_valid] = 1.0
    adjusted_app = np.minimum(1.0, adjusted_app + penalty)

    # MOT20 baseline does not score-fuse IoU. Preserve its hard proximity mask.
    adjusted = np.minimum(raw_iou, adjusted_app)
    adjusted[~iou_valid] = 1.0
    adjusted = adjusted.astype(np.float32)

    changed = np.abs(adjusted - baseline) > 1e-8
    _STATS["assoc_calls"] += 1
    _STATS["candidate_pairs"] += int(baseline.size)
    _STATS["valid_pairs"] += int(np.sum(iou_valid))
    _STATS["app_winning_pairs"] += int(np.sum(app_wins))
    _STATS["penalized_pairs"] += int(np.sum(penalize))
    _STATS["changed_pairs"] += int(np.sum(changed))
    _STATS["penalty_sum"] += float(np.sum(penalty))
    _STATS["penalty_max_seen"] = max(float(_STATS["penalty_max_seen"]), float(np.max(penalty)) if penalty.size else 0.0)

    debug = dict(debug)
    debug["trust_app_margin"] = app_margin
    debug["trust_iou_margin"] = iou_margin
    debug["trust_app_penalty"] = penalty
    debug["trust_final_before"] = np.asarray(baseline, dtype=np.float32).copy()
    debug["final"] = adjusted.copy()
    return adjusted, debug


class TrustAssocConfig:
    def __init__(
        self,
        enable: bool = False,
        app_margin_thresh: float = 0.03,
        iou_advantage_margin: float = 0.01,
        max_penalty: float = 0.05,
        min_track_age: int = 5,
        win_epsilon: float = 1e-6,
        stats_json: str = "",
    ) -> None:
        self.enable = bool(enable)
        self.app_margin_thresh = float(app_margin_thresh)
        self.iou_advantage_margin = float(iou_advantage_margin)
        self.max_penalty = float(max_penalty)
        self.min_track_age = int(min_track_age)
        self.win_epsilon = float(win_epsilon)
        self.stats_json = str(stats_json or "")


def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--trust-assoc-enable", action="store_true")
    p.add_argument("--trust-app-margin-thresh", type=float, default=0.03)
    p.add_argument("--trust-iou-advantage-margin", type=float, default=0.01)
    p.add_argument("--trust-max-penalty", type=float, default=0.05)
    p.add_argument("--trust-min-track-age", type=int, default=5)
    p.add_argument("--trust-win-epsilon", type=float, default=1e-6)
    p.add_argument("--trust-assoc-stats-json", type=str, default="")
    trust_args, remaining = p.parse_known_args()
    cfg = TrustAssocConfig(
        enable=trust_args.trust_assoc_enable,
        app_margin_thresh=trust_args.trust_app_margin_thresh,
        iou_advantage_margin=trust_args.trust_iou_advantage_margin,
        max_penalty=trust_args.trust_max_penalty,
        min_track_age=trust_args.trust_min_track_age,
        win_epsilon=trust_args.trust_win_epsilon,
        stats_json=trust_args.trust_assoc_stats_json,
    )
    return cfg, remaining


def main() -> None:
    global _CFG
    _CFG, remaining = parse_args()
    spda.base.assoc_cost = _patched_assoc_cost
    old_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]] + remaining
        v3.main()
    finally:
        sys.argv = old_argv
        spda.base.assoc_cost = _ORIG_ASSOC_COST
        if _CFG.stats_json:
            out = Path(_CFG.stats_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "config": _CFG.__dict__,
                "stats": dict(_STATS),
                "mean_penalty_on_penalized": float(_STATS["penalty_sum"]) / max(1, int(_STATS["penalized_pairs"])),
            }
            out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
