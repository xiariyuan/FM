#!/usr/bin/env python3
"""Frame-range ablation wrapper for TrustTrack-soft diagnostics.

The wrapper changes only whether compute_soft_map is allowed to return planned
updates on selected frame ranges. It is intended for full-sequence
counterfactual diagnosis, not as an online method.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import dmm_base_tracker_trusttrack_soft as soft


def parse_range(value: str) -> Tuple[int, int]:
    try:
        start_text, end_text = value.split(":", 1)
        start, end = int(start_text), int(end_text)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"range must be START:END, got {value!r}") from exc
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError(f"invalid range {value!r}")
    return start, end


def selected(frame: int, ranges: List[Tuple[int, int]]) -> bool:
    return any(start <= frame <= end for start, end in ranges)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ablation-mode", choices=("only", "exclude"), required=True)
    parser.add_argument("--ablation-range", action="append", type=parse_range, required=True)
    parser.add_argument("--ablation-meta-json", default="")
    args, remaining = parser.parse_known_args()

    original_compute = soft.compute_soft_map

    def filtered_compute(tracker, boxes, scores, feats, det_ids, cfg):
        frame = int(getattr(tracker, "frame_id", 0)) + 1
        keep = selected(frame, args.ablation_range)
        enabled = keep if args.ablation_mode == "only" else not keep
        if not enabled:
            soft._TRUST_PAIR_META = {}
            return {}, {}
        return original_compute(tracker, boxes, scores, feats, det_ids, cfg)

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
        path.write_text(
            json.dumps(
                {
                    "mode": args.ablation_mode,
                    "ranges": [list(value) for value in args.ablation_range],
                    "diagnostic_only": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
