#!/usr/bin/env python3
from __future__ import annotations

"""Fixed-detection-match HOTA teacher for ID-only tracker transactions.

This is a fast *screening* teacher, not the official final evaluator.  It uses
the official MOT20 preprocessing cached by M23-37 and freezes the parent's
per-frame Hungarian detection assignments.  Candidate tracker IDs only change
the global association contingency tables.  Every shortlisted policy must
still be verified with official TrackEval.

GT use is teacher-only.  Under strict sequence LOSO this script may only label
outer-training sequences or sequences whose GT has already been exposed for
diagnosis.  It must never select a policy on an unseen outer-held sequence.
"""

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]


def load_m23_37():
    path = REPO / "scripts/m23_research/m23_37_fast_exact_hota_teacher.py"
    spec = importlib.util.spec_from_file_location("m23_37_fast_exact_hota_teacher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FixedMatchHOTATeacher:
    """Recompute HOTA association with frozen parent detection assignments."""

    def __init__(self, prepared) -> None:
        self.prepared = prepared

    def evaluate_row_ids(self, row_ids: np.ndarray) -> Dict[str, float]:
        start = time.perf_counter()
        row_ids = np.asarray(row_ids, dtype=np.int64)
        if row_ids.shape != (self.prepared.num_parent_rows,):
            raise ValueError(
                f"expected {self.prepared.num_parent_rows} row IDs, got {row_ids.shape}"
            )

        candidate_ids_by_frame, candidate_ids_flat, unique_ids = (
            self.prepared._candidate_processed_ids(row_ids)
        )
        tracker_count = np.bincount(
            candidate_ids_flat, minlength=len(unique_ids)
        ).astype(float)
        result = self.prepared._empty_raw_result()
        matches_counts = [
            np.zeros(
                (int(self.prepared.data["num_gt_ids"]), len(unique_ids)),
                dtype=float,
            )
            for _ in self.prepared.metric.array_labels
        ]

        # The row/column assignments index detections within a frame, so they
        # remain valid when only the tracker-ID partition changes.
        for gt_ids, tracker_ids, similarity, assignment in zip(
            self.prepared.data["gt_ids"],
            candidate_ids_by_frame,
            self.prepared.data["similarity_scores"],
            self.prepared.baseline_assignments,
        ):
            match_rows, match_cols = assignment
            self.prepared._add_frame_statistics(
                result,
                matches_counts,
                gt_ids,
                tracker_ids,
                similarity,
                match_rows,
                match_cols,
                sign=1.0,
            )

        metrics = self.prepared._finalize_raw_result(
            result, matches_counts, tracker_count
        )
        metrics.update(
            {
                "evaluation_seconds": time.perf_counter() - start,
                "candidate_partitions": int(len(unique_ids)),
                "processed_detections": int(len(candidate_ids_flat)),
            }
        )
        return metrics

    def evaluate_tracker_file(
        self, tracker_file: Path, run_incremental_exact: bool = False
    ) -> Dict[str, object]:
        align_start = time.perf_counter()
        row_ids, alignment_mode = self._ids_from_ordered_tracker_file(tracker_file)
        alignment_seconds = time.perf_counter() - align_start
        metrics = self.evaluate_row_ids(row_ids)
        metrics.update(
            {
                "alignment_mode": alignment_mode,
                "alignment_seconds": alignment_seconds,
                "end_to_end_candidate_seconds": (
                    alignment_seconds + float(metrics["evaluation_seconds"])
                ),
            }
        )
        if run_incremental_exact:
            incremental = self.prepared.evaluate_row_ids_incremental(row_ids)
            metrics["incremental_exact"] = incremental
        return metrics

    def _ids_from_ordered_tracker_file(self, tracker_file: Path):
        """Use linear/hash alignment, with the M23-37 search as last fallback."""
        candidate_rows: List[List[str]] = []
        with tracker_file.open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split(",")
                if len(fields) >= 6:
                    candidate_rows.append(fields)
        parent_rows = self.prepared.parent_rows
        if len(candidate_rows) == len(parent_rows):
            strings_match = all(
                candidate[0] == parent[0] and candidate[2:6] == parent[2:6]
                for candidate, parent in zip(candidate_rows, parent_rows)
            )
            if strings_match:
                return (
                    np.asarray(
                        [int(float(fields[1])) for fields in candidate_rows],
                        dtype=np.int64,
                    ),
                    "ordered_exact_text",
                )

            # Formatting may differ even when the sorted detections are equal.
            candidate_geometry = np.asarray(
                [
                    [float(fields[0]), *(float(value) for value in fields[2:6])]
                    for fields in candidate_rows
                ],
                dtype=float,
            )
            parent_geometry = np.asarray(
                [
                    [float(fields[0]), *(float(value) for value in fields[2:6])]
                    for fields in parent_rows
                ],
                dtype=float,
            )
            if np.allclose(
                candidate_geometry, parent_geometry, rtol=0.0, atol=1e-4
            ):
                return (
                    np.asarray(
                        [int(float(fields[1])) for fields in candidate_rows],
                        dtype=np.int64,
                    ),
                    "ordered_numeric",
                )

        # Generated ID-only trackers can be sorted differently.  A quantized
        # (frame, box) queue preserves duplicates while keeping alignment O(N).
        def geometry_key(fields: Sequence[str]):
            return (
                int(float(fields[0])),
                *(int(round(float(value) * 10000.0)) for value in fields[2:6]),
            )

        candidates_by_geometry = defaultdict(deque)
        for fields in candidate_rows:
            candidates_by_geometry[geometry_key(fields)].append(int(float(fields[1])))
        ids = np.empty(len(parent_rows), dtype=np.int64)
        hash_aligned = len(candidate_rows) == len(parent_rows)
        if hash_aligned:
            for row_index, fields in enumerate(parent_rows):
                queue = candidates_by_geometry.get(geometry_key(fields))
                if not queue:
                    hash_aligned = False
                    break
                ids[row_index] = queue.popleft()
            hash_aligned = hash_aligned and not any(candidates_by_geometry.values())
        if hash_aligned:
            return ids, "frame_box_hash"

        return self.prepared.ids_from_tracker_file(tracker_file), "frame_box_search"


def parse_exact_hota(values: Sequence[str], candidate_count: int) -> List[float | None]:
    if not values:
        return [None] * candidate_count
    if len(values) != candidate_count:
        raise ValueError(
            f"--exact-hota count {len(values)} != candidate count {candidate_count}"
        )
    output: List[float | None] = []
    for value in values:
        output.append(None if value.lower() in {"none", "nan"} else float(value))
    return output


def validation_summary(
    rows: Sequence[Dict[str, object]],
    max_abs_error: float,
    max_seconds: float,
    min_spearman: float,
) -> Dict[str, object]:
    paired = [row for row in rows if row["exact_HOTA"] is not None]
    errors = [float(row["abs_HOTA_error"]) for row in paired]
    seconds = [float(row["evaluation_seconds"]) for row in rows]

    sign_checks = []
    for left, right in combinations(paired, 2):
        exact_delta = float(right["exact_HOTA"]) - float(left["exact_HOTA"])
        fast_delta = float(right["HOTA"]) - float(left["HOTA"])
        exact_sign = int(np.sign(exact_delta))
        fast_sign = int(np.sign(fast_delta))
        sign_checks.append(
            {
                "left": left["tracker"],
                "right": right["tracker"],
                "exact_delta": exact_delta,
                "fast_delta": fast_delta,
                "agrees": exact_sign == fast_sign,
            }
        )

    if len(paired) >= 2:
        correlation = float(
            spearmanr(
                [float(row["exact_HOTA"]) for row in paired],
                [float(row["HOTA"]) for row in paired],
            ).statistic
        )
    else:
        correlation = None

    error_ok = bool(errors) and max(errors) <= max_abs_error
    speed_ok = bool(seconds) and max(seconds) <= max_seconds
    signs_ok = bool(sign_checks) and all(item["agrees"] for item in sign_checks)
    rank_ok = correlation is not None and correlation >= min_spearman
    return {
        "accepted": bool(error_ok and speed_ok and signs_ok and rank_ok),
        "max_abs_HOTA_error": max(errors) if errors else None,
        "max_evaluation_seconds": max(seconds) if seconds else None,
        "pairwise_delta_sign_agreement": (
            float(np.mean([item["agrees"] for item in sign_checks]))
            if sign_checks
            else None
        ),
        "spearman_HOTA": correlation,
        "criteria": {
            "max_abs_HOTA_error": max_abs_error,
            "max_evaluation_seconds": max_seconds,
            "pairwise_delta_sign_agreement": 1.0,
            "min_spearman_HOTA": min_spearman,
        },
        "checks": {
            "error_ok": error_ok,
            "speed_ok": speed_ok,
            "signs_ok": signs_ok,
            "rank_ok": rank_ok,
        },
        "pairwise_checks": sign_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True)
    parser.add_argument("--parent-tracker", required=True)
    parser.add_argument("--candidate-tracker", action="append", required=True)
    parser.add_argument("--exact-hota", action="append", default=[])
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-abs-error", type=float, default=0.01)
    parser.add_argument("--max-seconds", type=float, default=0.5)
    parser.add_argument("--min-spearman", type=float, default=0.98)
    parser.add_argument("--run-incremental-exact", action="store_true")
    args = parser.parse_args()

    m23_37 = load_m23_37()
    prepare_start = time.perf_counter()
    prepared = m23_37.PreparedExactHOTA(
        args.seq,
        Path(args.parent_tracker),
        Path(args.cache_root),
    )
    prepare_seconds = time.perf_counter() - prepare_start
    teacher = FixedMatchHOTATeacher(prepared)
    exact_values = parse_exact_hota(args.exact_hota, len(args.candidate_tracker))

    rows = []
    for tracker_raw, exact_hota in zip(args.candidate_tracker, exact_values):
        tracker = Path(tracker_raw)
        metrics = teacher.evaluate_tracker_file(
            tracker, run_incremental_exact=args.run_incremental_exact
        )
        row: Dict[str, object] = {
            "tracker": str(tracker),
            **metrics,
            "exact_HOTA": exact_hota,
        }
        row["HOTA_error"] = (
            float(row["HOTA"]) - exact_hota if exact_hota is not None else None
        )
        row["abs_HOTA_error"] = (
            abs(float(row["HOTA_error"])) if row["HOTA_error"] is not None else None
        )
        if args.run_incremental_exact:
            incremental_hota = float(row["incremental_exact"]["HOTA"])
            row["incremental_exact_HOTA_error"] = (
                incremental_hota - exact_hota if exact_hota is not None else None
            )
            row["incremental_exact_abs_HOTA_error"] = (
                abs(float(row["incremental_exact_HOTA_error"]))
                if row["incremental_exact_HOTA_error"] is not None
                else None
            )
        rows.append(row)

    validation = validation_summary(
        rows, args.max_abs_error, args.max_seconds, args.min_spearman
    )
    report = {
        "status": "completed",
        "teacher_only": True,
        "official_final_evaluation": False,
        "protocol": "official preprocessing; parent Hungarian assignments frozen; candidate ID association recomputed over all 19 HOTA alphas",
        "seq": args.seq,
        "parent_tracker": args.parent_tracker,
        "prepare_seconds": prepare_seconds,
        "num_parent_rows": prepared.num_parent_rows,
        "candidates": rows,
        "validation": validation,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
