#!/usr/bin/env python3
"""Exact-key ablation wrapper for TrustTrack-soft diagnostics.

Filters planned soft updates by exact (frame, track_id, det_global_idx) keys.
This is an offline causal-diagnostic tool, not an online method.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Set, Tuple

import dmm_base_tracker_trusttrack_soft as soft

Key = Tuple[int, int, int]


def parse_key(value: str) -> Key:
    try:
        frame_text, track_text, det_text = value.split(":", 2)
        key = (int(frame_text), int(track_text), int(det_text))
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"key must be FRAME:TRACK_ID:DET_GLOBAL_IDX, got {value!r}"
        ) from exc
    if key[0] < 1:
        raise argparse.ArgumentTypeError(f"invalid frame in key {value!r}")
    return key


def load_keys(path: str) -> Set[Key]:
    if not path:
        return set()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("keys", payload.get("events", []))
    if not isinstance(payload, list):
        raise ValueError("key JSON must contain a list or a dict with keys/events")
    keys: Set[Key] = set()
    for item in payload:
        if isinstance(item, dict):
            keys.add(
                (
                    int(item["frame"]),
                    int(item.get("track_id", item.get("track"))),
                    int(item.get("det_global_idx", item.get("det"))),
                )
            )
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            keys.add((int(item[0]), int(item[1]), int(item[2])))
        elif isinstance(item, str):
            keys.add(parse_key(item))
        else:
            raise ValueError(f"unsupported key item: {item!r}")
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ablation-mode", choices=("only", "exclude"), required=True)
    parser.add_argument("--ablation-key", action="append", type=parse_key, default=[])
    parser.add_argument("--ablation-keys-json", default="")
    parser.add_argument("--ablation-meta-json", default="")
    args, remaining = parser.parse_known_args()

    selected_keys = set(args.ablation_key)
    selected_keys.update(load_keys(args.ablation_keys_json))
    if not selected_keys:
        raise ValueError("at least one ablation key is required")

    original_compute = soft.compute_soft_map
    stats = {
        "compute_calls": 0,
        "planned_before_filter": 0,
        "planned_after_filter": 0,
        "selected_key_hits": 0,
    }

    def filtered_compute(tracker, boxes, scores, feats, det_ids, cfg):
        alpha_map, reason_map = original_compute(
            tracker, boxes, scores, feats, det_ids, cfg
        )
        frame = int(getattr(tracker, "frame_id", 0)) + 1
        stats["compute_calls"] += 1
        stats["planned_before_filter"] += len(alpha_map)

        hit_pairs = {
            (track_id, det_id)
            for track_id, det_id in alpha_map
            if (frame, int(track_id), int(det_id)) in selected_keys
        }
        stats["selected_key_hits"] += len(hit_pairs)
        if args.ablation_mode == "only":
            keep_pairs = hit_pairs
        else:
            keep_pairs = set(alpha_map) - hit_pairs

        filtered_alpha = {
            key: value for key, value in alpha_map.items() if key in keep_pairs
        }
        filtered_reason = {
            key: value for key, value in reason_map.items() if key in keep_pairs
        }
        soft._TRUST_PAIR_META = {
            key: value
            for key, value in soft._TRUST_PAIR_META.items()
            if key in keep_pairs
        }
        stats["planned_after_filter"] += len(filtered_alpha)
        return filtered_alpha, filtered_reason

    old_argv = sys.argv
    soft.compute_soft_map = filtered_compute
    try:
        sys.argv = [sys.argv[0]] + remaining
        soft.main()
    finally:
        sys.argv = old_argv
        soft.compute_soft_map = original_compute

    if args.ablation_meta_json:
        path = Path(args.ablation_meta_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": args.ablation_mode,
            "keys": [list(value) for value in sorted(selected_keys)],
            "num_keys": len(selected_keys),
            "stats": stats,
            "diagnostic_only": True,
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
