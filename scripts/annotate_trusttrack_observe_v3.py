#!/usr/bin/env python3
"""Offline GT annotation v3 for TrustTrack observe logs.

This script preserves all observe-v2 annotations and adds three explicitly
separated detection identity views:

1. legacy chosen-only Hungarian GT (the v2 behavior);
2. direct-best valid pedestrian GT for the chosen detection;
3. same-stage all-detection Hungarian GT after TrackEval-style distractor
   preprocessing.

It also records top-1/top-2 IoUs, their gap, and multi-GT overlap counts. The
online tracker is never given GT. This script is diagnostic-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import annotate_trusttrack_observe_v2 as v2
import current_output_id_oracle_v2 as oracle

Box = Tuple[float, float, float, float]


def _float_or_none(value):
    if value in (None, "", "nan", "None"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int_or_none(value):
    number = _float_or_none(value)
    return int(number) if number is not None else None


def _load_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_dump(path: Path):
    dump = np.load(path, allow_pickle=True)
    detections = np.asarray(dump["detections"])
    offsets = np.asarray(dump["frame_offsets"], dtype=np.int64)
    columns = {str(name): index for index, name in enumerate(dump["columns"].tolist())}
    required = {"frame", "global_det_idx", "x1", "y1", "x2", "y2", "score"}
    missing = sorted(required - set(columns))
    if missing:
        raise KeyError(f"dump is missing columns: {missing}")
    return detections, offsets, columns


def _box_from_row(row: np.ndarray, columns: dict) -> Box:
    return (
        float(row[columns["x1"]]),
        float(row[columns["y1"]]),
        float(row[columns["x2"]]),
        float(row[columns["y2"]]),
    )


def _valid_gt_rows(gt_rows: Sequence[oracle.GTRow]) -> List[oracle.GTRow]:
    return [row for row in gt_rows if row[2] != 0 and row[3] == 1]


def _direct_best_fields(
    box: Box,
    gt_rows: Sequence[oracle.GTRow],
    *,
    threshold: float,
) -> dict:
    valid_gt = _valid_gt_rows(gt_rows)
    if not valid_gt:
        return {
            "det_gt_direct_best": None,
            "det_gt_direct_best_iou": None,
            "det_gt_direct_second": None,
            "det_gt_direct_second_iou": None,
            "det_gt_direct_iou_gap": None,
            "det_num_valid_gt_iou_ge_03": 0,
            "det_num_valid_gt_iou_ge_05": 0,
        }
    overlaps = oracle.iou_matrix([box], [row[1] for row in valid_gt])[0]
    order = np.argsort(-overlaps, kind="mergesort")
    top_index = int(order[0])
    top_iou = float(overlaps[top_index])
    second_index = int(order[1]) if len(order) > 1 else None
    second_iou = float(overlaps[second_index]) if second_index is not None else None
    direct_gt = int(valid_gt[top_index][0]) if top_iou + np.finfo(float).eps >= threshold else None
    return {
        "det_gt_direct_best": direct_gt,
        "det_gt_direct_best_iou": top_iou,
        "det_gt_direct_second": int(valid_gt[second_index][0]) if second_index is not None else None,
        "det_gt_direct_second_iou": second_iou,
        "det_gt_direct_iou_gap": (
            float(top_iou - second_iou) if second_iou is not None else None
        ),
        "det_num_valid_gt_iou_ge_03": int(np.sum(overlaps >= 0.3 - np.finfo(float).eps)),
        "det_num_valid_gt_iou_ge_05": int(np.sum(overlaps >= 0.5 - np.finfo(float).eps)),
    }


def _stage_candidate_indices(
    frame_rows: np.ndarray,
    columns: dict,
    stage: str,
    *,
    high_thresh: float,
    low_thresh: float,
    primary_matched_det_ids: set[int] | None = None,
) -> np.ndarray:
    scores = np.asarray(frame_rows[:, columns["score"]], dtype=np.float32)
    stage = str(stage)
    if stage.startswith("primary"):
        return np.flatnonzero(scores > float(high_thresh))
    if stage.startswith("unconfirmed"):
        high_indices = np.flatnonzero(scores > float(high_thresh))
        matched = primary_matched_det_ids or set()
        return np.asarray(
            [
                int(index)
                for index in high_indices.tolist()
                if int(frame_rows[index, columns["global_det_idx"]]) not in matched
            ],
            dtype=np.int64,
        )
    if stage.startswith("secondary"):
        return np.flatnonzero(
            (scores > float(low_thresh)) & (scores < float(high_thresh))
        )
    # Unknown stages are deliberately left unmapped rather than guessed.
    return np.zeros((0,), dtype=np.int64)


def _build_stage_maps(
    observe_rows: Sequence[dict],
    detections: np.ndarray,
    offsets: np.ndarray,
    columns: dict,
    gt_by_frame: Dict[int, List[oracle.GTRow]],
    *,
    benchmark: str,
    iou_threshold: float,
    preproc_iou_threshold: float,
    high_thresh: float,
    low_thresh: float,
):
    needed = {(int(row["frame"]), str(row["stage"])) for row in observe_rows}
    primary_matched_by_frame: Dict[int, set[int]] = defaultdict(set)
    for row in observe_rows:
        if str(row["stage"]).startswith("primary"):
            primary_matched_by_frame[int(row["frame"])].add(int(row["det_global_idx"]))
    maps: Dict[Tuple[int, str], Dict[int, Tuple[int, float]]] = {}
    removed: Dict[Tuple[int, str], set[int]] = {}
    candidate_counts: Dict[Tuple[int, str], int] = {}
    for frame, stage in sorted(needed):
        start = int(offsets[frame - 1])
        end = int(offsets[frame])
        frame_rows = detections[start:end]
        indices = _stage_candidate_indices(
            frame_rows,
            columns,
            stage,
            high_thresh=high_thresh,
            low_thresh=low_thresh,
            primary_matched_det_ids=primary_matched_by_frame.get(frame, set()),
        )
        items = [
            (
                int(frame_rows[index, columns["global_det_idx"]]),
                _box_from_row(frame_rows[index], columns),
            )
            for index in indices.tolist()
        ]
        item_to_gt, _, removed_ids = v2.fixed_iou_frame_mapping(
            items,
            gt_by_frame.get(frame, []),
            benchmark=benchmark,
            iou_threshold=iou_threshold,
            preproc_iou_threshold=preproc_iou_threshold,
        )
        key = (frame, stage)
        maps[key] = item_to_gt
        removed[key] = removed_ids
        candidate_counts[key] = len(items)
    return maps, removed, candidate_counts


def annotate_v3(args: argparse.Namespace) -> dict:
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Reuse v2 for its TrackEval-aligned output/GT history logic. The temporary
    # legacy file is removed after enrichment.
    with tempfile.TemporaryDirectory(prefix="trusttrack_annotate_v3_") as temp_dir:
        legacy_csv = Path(temp_dir) / "legacy_v2.csv"
        legacy_summary = Path(temp_dir) / "legacy_v2_summary.json"
        legacy_args = argparse.Namespace(
            seq=args.seq,
            observe_csv=args.observe_csv,
            result_txt=args.result_txt,
            dump_npz=args.dump_npz,
            gt_txt=args.gt_txt,
            out_csv=str(legacy_csv),
            summary_json=str(legacy_summary),
            benchmark=args.benchmark,
            iou_threshold=args.iou_threshold,
            preproc_iou_threshold=args.preproc_iou_threshold,
        )
        v2_summary = v2.annotate(legacy_args)
        rows = _load_csv(legacy_csv)

    detections, offsets, columns = _load_dump(Path(args.dump_npz))
    gt_by_frame = oracle.load_all_gt(Path(args.gt_txt))
    stage_maps, stage_removed, stage_candidate_counts = _build_stage_maps(
        rows,
        detections,
        offsets,
        columns,
        gt_by_frame,
        benchmark=args.benchmark,
        iou_threshold=args.iou_threshold,
        preproc_iou_threshold=args.preproc_iou_threshold,
        high_thresh=args.track_high_thresh,
        low_thresh=args.track_low_thresh,
    )

    det_by_global: Dict[int, Tuple[int, Box, float]] = {}
    duplicate_global_ids = 0
    for row in detections:
        global_id = int(row[columns["global_det_idx"]])
        if global_id in det_by_global:
            duplicate_global_ids += 1
            continue
        det_by_global[global_id] = (
            int(row[columns["frame"]]),
            _box_from_row(row, columns),
            float(row[columns["score"]]),
        )
    if duplicate_global_ids:
        raise RuntimeError(f"duplicate global detection IDs: {duplicate_global_ids}")

    mismatch_counts = Counter()
    stage_counts = Counter()
    enriched_rows: List[dict] = []
    for row in rows:
        frame = int(row["frame"])
        stage = str(row["stage"])
        global_id = int(row["det_global_idx"])
        if global_id not in det_by_global:
            raise KeyError(f"missing detection global ID: {global_id}")
        det_frame, box, score = det_by_global[global_id]
        if det_frame != frame:
            raise RuntimeError(
                f"detection frame mismatch for {global_id}: {det_frame} != {frame}"
            )

        direct = _direct_best_fields(
            box,
            gt_by_frame.get(frame, []),
            threshold=args.iou_threshold,
        )
        stage_key = (frame, stage)
        stage_match = stage_maps.get(stage_key, {}).get(global_id)
        stage_gt = int(stage_match[0]) if stage_match else None
        stage_iou = float(stage_match[1]) if stage_match else None
        stage_preproc_removed = global_id in stage_removed.get(stage_key, set())
        legacy_gt = _int_or_none(row.get("det_gt"))
        direct_gt = direct["det_gt_direct_best"]

        enriched = dict(row)
        enriched.update(
            {
                "det_gt_legacy_chosen_hungarian": legacy_gt,
                "det_gt_legacy_chosen_hungarian_iou": _float_or_none(
                    row.get("det_gt_iou")
                ),
                **direct,
                "det_gt_stage_all_hungarian": stage_gt,
                "det_gt_stage_all_hungarian_iou": stage_iou,
                "det_stage_all_preproc_removed": int(stage_preproc_removed),
                "det_stage_candidate_count": int(
                    stage_candidate_counts.get(stage_key, 0)
                ),
                "det_score_v3": score,
                "legacy_vs_direct_same": int(
                    legacy_gt is not None and direct_gt is not None and legacy_gt == direct_gt
                ),
                "legacy_vs_stage_all_same": int(
                    legacy_gt is not None and stage_gt is not None and legacy_gt == stage_gt
                ),
                "direct_vs_stage_all_same": int(
                    direct_gt is not None and stage_gt is not None and direct_gt == stage_gt
                ),
                "legacy_direct_comparable": int(
                    legacy_gt is not None and direct_gt is not None
                ),
                "legacy_stage_all_comparable": int(
                    legacy_gt is not None and stage_gt is not None
                ),
                "direct_stage_all_comparable": int(
                    direct_gt is not None and stage_gt is not None
                ),
            }
        )
        enriched_rows.append(enriched)
        stage_counts[stage] += 1
        if legacy_gt is not None and direct_gt is not None:
            mismatch_counts["legacy_direct_comparable"] += 1
            mismatch_counts["legacy_direct_mismatch"] += int(legacy_gt != direct_gt)
        if legacy_gt is not None and stage_gt is not None:
            mismatch_counts["legacy_stage_comparable"] += 1
            mismatch_counts["legacy_stage_mismatch"] += int(legacy_gt != stage_gt)
        if direct_gt is not None and stage_gt is not None:
            mismatch_counts["direct_stage_comparable"] += 1
            mismatch_counts["direct_stage_mismatch"] += int(direct_gt != stage_gt)

    _write_csv(out_path, enriched_rows)

    summary = {
        "seq": args.seq,
        "observe_csv": args.observe_csv,
        "result_txt": args.result_txt,
        "dump_npz": args.dump_npz,
        "gt_txt": args.gt_txt,
        "out_csv": args.out_csv,
        "benchmark": args.benchmark,
        "iou_threshold": float(args.iou_threshold),
        "preproc_iou_threshold": float(args.preproc_iou_threshold),
        "track_high_thresh": float(args.track_high_thresh),
        "track_low_thresh": float(args.track_low_thresh),
        "rows": len(enriched_rows),
        "stage_counts": dict(stage_counts),
        "v2_summary": v2_summary,
        "legacy_chosen_gt_assigned": sum(
            _int_or_none(row.get("det_gt_legacy_chosen_hungarian")) is not None
            for row in enriched_rows
        ),
        "direct_best_gt_assigned": sum(
            _int_or_none(row.get("det_gt_direct_best")) is not None
            for row in enriched_rows
        ),
        "stage_all_gt_assigned": sum(
            _int_or_none(row.get("det_gt_stage_all_hungarian")) is not None
            for row in enriched_rows
        ),
        "stage_all_preproc_removed": sum(
            int(row["det_stage_all_preproc_removed"]) for row in enriched_rows
        ),
        "multi_gt_iou_ge_03": sum(
            int(row["det_num_valid_gt_iou_ge_03"]) >= 2 for row in enriched_rows
        ),
        "multi_gt_iou_ge_05": sum(
            int(row["det_num_valid_gt_iou_ge_05"]) >= 2 for row in enriched_rows
        ),
        "mismatch_counts": dict(mismatch_counts),
        "diagnostic_only": True,
        "uses_ground_truth": True,
    }
    for prefix, comparable_key, mismatch_key in (
        ("legacy_direct", "legacy_direct_comparable", "legacy_direct_mismatch"),
        ("legacy_stage", "legacy_stage_comparable", "legacy_stage_mismatch"),
        ("direct_stage", "direct_stage_comparable", "direct_stage_mismatch"),
    ):
        comparable = int(mismatch_counts[comparable_key])
        mismatch = int(mismatch_counts[mismatch_key])
        summary[f"{prefix}_mismatch_rate"] = mismatch / max(1, comparable)

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--observe-csv", required=True)
    parser.add_argument("--result-txt", required=True)
    parser.add_argument("--dump-npz", required=True)
    parser.add_argument("--gt-txt", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--benchmark", default="MOT20")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--preproc-iou-threshold", type=float, default=0.5)
    parser.add_argument("--track-high-thresh", type=float, default=0.6)
    parser.add_argument("--track-low-thresh", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = annotate_v3(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
