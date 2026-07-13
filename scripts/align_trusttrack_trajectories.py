#!/usr/bin/env python3
"""Align tracker IDs across two runs by shared chosen detections.

Raw tracker IDs are run-local and may shift after any lifecycle change. This
script builds a one-to-one maximum-weight bipartite alignment using shared
(frame, det_global_idx) chosen-association keys. It can optionally annotate an
intervention/event CSV with aligned baseline IDs.

No GT is required. The alignment is diagnostic-only and must not be used by the
online tracker.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

Key = Tuple[int, int]

_STAGE_PRIORITY = {
    "primary": 3,
    "secondary_tracked": 2,
    "secondary": 2,
    "unconfirmed": 1,
}


def _load_chosen_index(path: Path) -> Tuple[Dict[Key, dict], dict]:
    index: Dict[Key, dict] = {}
    duplicates = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if "chosen" in row and int(float(row.get("chosen", 0) or 0)) != 1:
                continue
            key = (int(float(row["frame"])), int(float(row["det_global_idx"])))
            old = index.get(key)
            if old is None:
                index[key] = row
                continue
            duplicates += 1
            old_priority = _STAGE_PRIORITY.get(str(old.get("stage", "")), 0)
            new_priority = _STAGE_PRIORITY.get(str(row.get("stage", "")), 0)
            if new_priority > old_priority:
                index[key] = row
    stats = {
        "rows_indexed": len(index),
        "duplicate_detection_keys": duplicates,
    }
    return index, stats


def _build_alignment(variant: Dict[Key, dict], baseline: Dict[Key, dict]) -> dict:
    shared = sorted(set(variant) & set(baseline))
    variant_ids = sorted({int(float(variant[key]["track_id"])) for key in shared})
    baseline_ids = sorted({int(float(baseline[key]["track_id"])) for key in shared})
    vi = {track_id: index for index, track_id in enumerate(variant_ids)}
    bi = {track_id: index for index, track_id in enumerate(baseline_ids)}
    counts = np.zeros((len(variant_ids), len(baseline_ids)), dtype=np.int64)
    variant_support = Counter()
    baseline_support = Counter()
    raw_equal = 0
    for key in shared:
        variant_id = int(float(variant[key]["track_id"]))
        baseline_id = int(float(baseline[key]["track_id"]))
        counts[vi[variant_id], bi[baseline_id]] += 1
        variant_support[variant_id] += 1
        baseline_support[baseline_id] += 1
        raw_equal += int(variant_id == baseline_id)

    mapping: Dict[int, int] = {}
    pair_support: Dict[int, int] = {}
    if counts.size:
        rows, cols = linear_sum_assignment(-counts.astype(np.float64))
        for row, col in zip(rows.tolist(), cols.tolist()):
            support = int(counts[row, col])
            if support <= 0:
                continue
            variant_id = variant_ids[row]
            baseline_id = baseline_ids[col]
            mapping[variant_id] = baseline_id
            pair_support[variant_id] = support

    aligned_equal = 0
    aligned_comparable = 0
    mismatch_examples: List[dict] = []
    for key in shared:
        variant_id = int(float(variant[key]["track_id"]))
        baseline_id = int(float(baseline[key]["track_id"]))
        mapped = mapping.get(variant_id)
        if mapped is None:
            continue
        aligned_comparable += 1
        aligned_equal += int(mapped == baseline_id)
        if mapped != baseline_id and len(mismatch_examples) < 100:
            mismatch_examples.append(
                {
                    "frame": key[0],
                    "det_global_idx": key[1],
                    "variant_track_id": variant_id,
                    "baseline_track_id": baseline_id,
                    "mapped_baseline_track_id": mapped,
                }
            )

    mapping_rows = []
    for variant_id in variant_ids:
        mapped = mapping.get(variant_id)
        support = pair_support.get(variant_id, 0)
        total = int(variant_support[variant_id])
        mapping_rows.append(
            {
                "variant_track_id": variant_id,
                "baseline_track_id": mapped,
                "pair_support": support,
                "variant_shared_support": total,
                "alignment_purity": support / max(1, total),
                "baseline_shared_support": (
                    int(baseline_support[mapped]) if mapped is not None else 0
                ),
            }
        )

    return {
        "shared_detection_keys": len(shared),
        "variant_track_count": len(variant_ids),
        "baseline_track_count": len(baseline_ids),
        "mapped_variant_tracks": len(mapping),
        "raw_equal": raw_equal,
        "raw_equal_rate": raw_equal / max(1, len(shared)),
        "aligned_comparable": aligned_comparable,
        "aligned_equal": aligned_equal,
        "aligned_equal_rate": aligned_equal / max(1, aligned_comparable),
        "mapping_weighted_purity": sum(pair_support.values()) / max(1, len(shared)),
        "mapping_rows": mapping_rows,
        "mismatch_examples": mismatch_examples,
        "mapping": mapping,
        "pair_support": pair_support,
        "variant_support": dict(variant_support),
    }


def _event_track_key(row: dict) -> str:
    for key in ("track_id", "soft_track", "variant_track_id"):
        if key in row:
            return key
    raise KeyError("event CSV has no track_id/soft_track/variant_track_id column")


def _annotate_events(
    input_path: Path,
    output_path: Path,
    mapping: Dict[int, int],
    pair_support: Dict[int, int],
    variant_support: Dict[int, int],
) -> dict:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("event CSV is empty")
    track_key = _event_track_key(rows[0])
    comparable = 0
    same = 0
    for row in rows:
        variant_id = int(float(row[track_key]))
        mapped = mapping.get(variant_id)
        baseline_value = row.get("baseline_track")
        baseline_id = (
            int(float(baseline_value))
            if baseline_value not in (None, "", "nan")
            else None
        )
        support = int(pair_support.get(variant_id, 0))
        total = int(variant_support.get(variant_id, 0))
        aligned_same = int(
            mapped is not None and baseline_id is not None and mapped == baseline_id
        )
        row.update(
            {
                "aligned_baseline_track_id": mapped,
                "alignment_pair_support": support,
                "alignment_variant_support": total,
                "alignment_purity": support / max(1, total),
                "same_aligned_trajectory": aligned_same,
                "same_raw_tracker_id_deprecated": int(
                    baseline_id is not None and variant_id == baseline_id
                ),
            }
        )
        if mapped is not None and baseline_id is not None:
            comparable += 1
            same += aligned_same

    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "event_rows": len(rows),
        "event_aligned_comparable": comparable,
        "event_aligned_equal": same,
        "event_aligned_equal_rate": same / max(1, comparable),
        "track_column": track_key,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-observe-csv", required=True)
    parser.add_argument("--baseline-observe-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--event-csv", default="")
    parser.add_argument("--out-event-csv", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variant, variant_stats = _load_chosen_index(Path(args.variant_observe_csv))
    baseline, baseline_stats = _load_chosen_index(Path(args.baseline_observe_csv))
    result = _build_alignment(variant, baseline)
    mapping = result.pop("mapping")
    pair_support = result.pop("pair_support")
    variant_support = result.pop("variant_support")
    result.update(
        {
            "variant_observe_csv": args.variant_observe_csv,
            "baseline_observe_csv": args.baseline_observe_csv,
            "variant_index": variant_stats,
            "baseline_index": baseline_stats,
            "diagnostic_only": True,
            "uses_ground_truth": False,
        }
    )
    if args.event_csv or args.out_event_csv:
        if not args.event_csv or not args.out_event_csv:
            raise ValueError("--event-csv and --out-event-csv must be supplied together")
        result["event_annotation"] = _annotate_events(
            Path(args.event_csv),
            Path(args.out_event_csv),
            mapping,
            pair_support,
            variant_support,
        )
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
