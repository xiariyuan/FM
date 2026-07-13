#!/usr/bin/env python3
"""Run TrustTrack-soft v1 with association-aligned observe-v2 logging.

This wrapper combines the existing soft intervention with the observe-only
association callback. It must reproduce the original soft tracker output
exactly. The observe CSV records planned soft-map hits at the real chosen pair;
the existing --trust-pair-log-csv records whether DMMTrack.update actually
applied the alpha change.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import dmm_base_tracker_shadow_pda as spda
import dmm_base_tracker_trusttrack_observe_v2 as observe
import dmm_base_tracker_trusttrack_soft as soft


def _reset_soft_state() -> None:
    soft._TRUST_SOFT_ALPHA_MAP = {}
    soft._TRUST_SOFT_REASON_MAP = {}
    soft._TRUST_PAIR_META = {}
    soft._TRUST_MATCH_ROWS.clear()
    soft._TRUST_STATS.clear()
    soft._TRUST_STATS.update(
        {
            "frames": 0,
            "candidate_pairs": 0,
            "soft_pairs_predicted": 0,
            "feature_updates_soft": 0,
            "feature_updates_normal": 0,
            "soft_alpha_sum": 0.0,
        }
    )
    soft._PATCHED = False


def main() -> None:
    observe_args, remaining = observe.parse_observe_args()
    if observe_args.observe_top_k < 0:
        raise ValueError("--observe-top-k must be >= 0")
    if "--debug-csv" in remaining:
        raise ValueError("Do not pass --debug-csv; use --observe-csv")
    if "--trust-pair-log-csv" not in remaining:
        raise ValueError(
            "--trust-pair-log-csv is required so planned and actual soft updates can be audited"
        )
    if "--debug-assoc" not in remaining:
        remaining.append("--debug-assoc")

    csv_path = Path(observe_args.observe_csv)
    summary_path = Path(observe_args.observe_summary_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    observe._reset_stats()
    observe._OBSERVE_CONFIG = observe_args
    _reset_soft_state()

    old_argv = sys.argv
    original_record = spda.base.DMMBaseTracker._record_assoc_debug
    original_update = spda.base.DMMTrack.update
    failure: str | None = None
    try:
        observe._HANDLE = csv_path.open("w", newline="", encoding="utf-8")
        observe._WRITER = csv.DictWriter(
            observe._HANDLE,
            fieldnames=observe._FIELDS,
        )
        observe._WRITER.writeheader()
        spda.base.DMMBaseTracker._record_assoc_debug = observe._patched_record_assoc_debug
        sys.argv = [sys.argv[0]] + remaining
        soft.main()
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sys.argv = old_argv
        spda.base.DMMBaseTracker._record_assoc_debug = original_record
        spda.base.DMMTrack.update = original_update
        soft._PATCHED = False
        if observe._HANDLE is not None:
            observe._HANDLE.flush()
            observe._HANDLE.close()
        observe._WRITER = None
        payload = {
            "observe_csv": str(csv_path),
            "top_k": int(observe_args.observe_top_k),
            "failure": failure,
            "stats": observe._jsonable_stats(),
            "soft_stats": dict(soft._TRUST_STATS),
        }
        summary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    stats = observe._jsonable_stats()
    if int(stats["invalid_match_indices"]) != 0:
        raise RuntimeError(f"invalid match indices observed: {stats}")
    if int(stats["chosen_pairs_logged"]) != int(stats["chosen_pairs_total"]):
        raise RuntimeError(f"chosen-pair logging incomplete: {stats}")
    if int(stats["duplicate_chosen_keys"]) != 0:
        raise RuntimeError(f"duplicate chosen keys observed: {stats}")


if __name__ == "__main__":
    main()
