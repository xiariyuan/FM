#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.spot_common.io_utils import (
    append_registry,
    ensure_dir,
    upsert_plan,
    write_json,
    write_manifest,
    write_markdown,
    write_single_row_csv,
)
from scripts.spot_common.metrics import median_or_none, normalized_entropy, safe_ratio, top2_margin
from scripts.spot_common.mot_format import iou_tlwh


SUMMARY_FIELDS = [
    "status",
    "error",
    "dataset",
    "split",
    "seq_name",
    "analysis_scope",
    "groups_with_gt",
    "wrong_selected_groups",
    "fixable_groups",
    "fixable_percent",
    "median_positive_rank",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle 0C proxy for local cost reranking.")
    parser.add_argument("--pair-csv", required=True)
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--split", default="unknown")
    parser.add_argument("--seq-name", default="unknown_seq")
    parser.add_argument("--out-dir", default="outputs/oracle_gate/0C_cost_reranking")
    parser.add_argument("--max-margin-gap", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-groups", type=int, default=0, help="Optional cap on processed groups; 0 means full file.")
    parser.add_argument("--max-frame", type=int, default=0, help="Optional cap on frame_id; 0 means no frame cap.")
    return parser.parse_args()


def _iter_group_rows(path: str | Path) -> Iterator[tuple[str, list[dict]]]:
    with Path(path).expanduser().open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        current_key: str | None = None
        bucket: list[dict] = []
        for row in reader:
            key = _group_key(row)
            if current_key is None:
                current_key = key
            if key != current_key:
                yield current_key, bucket
                bucket = []
                current_key = key
            if not _is_positive(row) and _infer_positive_from_boxes(row):
                row["_box_positive"] = "1"
            bucket.append(row)
        if current_key is not None and bucket:
            yield current_key, bucket


def _float(row: dict, keys: tuple[str, ...], default: float = 0.0) -> float:
    for key in keys:
        if key in row and row[key] not in ("", None):
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return float(default)


def _int(row: dict, keys: tuple[str, ...], default: int = 0) -> int:
    for key in keys:
        if key in row and row[key] not in ("", None):
            try:
                return int(float(row[key]))
            except (TypeError, ValueError):
                continue
    return int(default)


def _group_key(row: dict) -> str:
    if row.get("group_id"):
        return str(row["group_id"])
    seq = row.get("seq") or row.get("seq_name") or "unknown_seq"
    frame = _int(row, ("frame", "frame_id"), 0)
    det_idx = _int(row, ("det_index", "det_id"), 0)
    return f"{seq}:{frame}:{det_idx}"


def _candidate_score(row: dict) -> float:
    return _float(row, ("refined_score", "s_final", "base_score", "anchor_sim"), 0.0)


def _selected_flag(row: dict) -> int:
    return _int(row, ("matched_by_host", "is_selected"), 0)


def _is_positive(row: dict) -> bool:
    det_gt = _int(row, ("det_gt_id", "gt_id_det"), -1)
    track_gt = _int(row, ("track_gt_id", "gt_id_track"), -1)
    if det_gt > 0 and track_gt > 0:
        return det_gt == track_gt
    label = _int(row, ("label", "is_positive"), -1)
    return label == 1


def _parse_tlwh(raw: str) -> tuple[float, float, float, float] | None:
    if not raw:
        return None
    parts = [chunk.strip() for chunk in str(raw).split(",")]
    if len(parts) < 4:
        return None
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
    except (TypeError, ValueError):
        return None


def _infer_positive_from_boxes(row: dict, iou_thresh: float = 0.7) -> bool:
    det_box = _parse_tlwh(str(row.get("det_tlwh", "")))
    track_box = _parse_tlwh(str(row.get("track_tlwh", "")))
    if det_box is None or track_box is None:
        return False
    return iou_tlwh(det_box, track_box) >= iou_thresh


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    summary_csv = out_dir / "summary.csv"
    script_path = str(Path(__file__).resolve().relative_to(REPO_ROOT))
    variant = out_dir.name
    tag = variant
    summary_row = {
        "status": "running",
        "error": "",
        "dataset": args.dataset,
        "split": args.split,
        "seq_name": args.seq_name,
        "analysis_scope": "partial" if (int(args.max_groups) > 0 or int(args.max_frame) > 0) else "full",
        "groups_with_gt": 0,
        "wrong_selected_groups": 0,
        "fixable_groups": 0,
        "fixable_percent": 0.0,
        "median_positive_rank": "",
    }
    write_single_row_csv(summary_csv, summary_row, SUMMARY_FIELDS)
    append_registry(
        kind="analysis",
        status="running",
        script=script_path,
        dataset=args.dataset,
        split=args.split,
        tracker_family="spot_oracle_0C",
        variant=variant,
        tag=tag,
        run_root=out_dir,
        summary_csv=summary_csv,
        notes=f"cost rerank oracle running for {args.seq_name}",
    )
    upsert_plan(
        status="running",
        kind="analysis",
        script=script_path,
        dataset=args.dataset,
        split=args.split,
        tracker_family="spot_oracle_0C",
        variant=variant,
        tag=tag,
        run_root=out_dir,
        summary_csv=summary_csv,
        notes=f"cost rerank oracle running for {args.seq_name}",
        key=f"spot_oracle_0C:{out_dir}",
    )

    try:
        wrong_selected_groups = 0
        fixable_groups = 0
        groups_with_gt = 0
        processed_groups = 0
        positive_ranks: list[int] = []
        fixable_events: list[dict] = []
        for group_id, group_rows in _iter_group_rows(args.pair_csv):
            if args.max_groups and processed_groups >= int(args.max_groups):
                break
            frame_id = _int(group_rows[0], ("frame", "frame_id"), 0) if group_rows else 0
            if args.max_frame and frame_id > int(args.max_frame):
                break
            ordered = sorted(group_rows, key=_candidate_score, reverse=True)
            positives = [row for row in ordered if _is_positive(row) or row.get("_box_positive") == "1"]
            if not positives:
                processed_groups += 1
                continue
            groups_with_gt += 1
            selected = next((row for row in ordered if _selected_flag(row) == 1), ordered[0])
            selected_correct = _is_positive(selected)
            pos_rank = next((idx for idx, row in enumerate(ordered) if _is_positive(row)), None)
            if pos_rank is not None:
                positive_ranks.append(int(pos_rank))
            if selected_correct:
                processed_groups += 1
                continue
            wrong_selected_groups += 1
            positive = positives[0]
            rank = next(idx for idx, row in enumerate(ordered) if row is positive)
            score_gap = _candidate_score(selected) - _candidate_score(positive)
            margin = top2_margin(_candidate_score(row) for row in ordered)
            entropy = normalized_entropy(max(_candidate_score(row), 0.0) for row in ordered)
            is_fixable = int(rank < int(args.top_k) and score_gap <= float(args.max_margin_gap))
            if is_fixable:
                fixable_groups += 1
            fixable_events.append(
                {
                    "group_id": group_id,
                    "fixable": is_fixable,
                    "positive_rank": rank,
                    "selected_track_id": _int(selected, ("track_id",), -1),
                    "positive_track_id": _int(positive, ("track_id",), -1),
                    "selected_score": _candidate_score(selected),
                    "positive_score": _candidate_score(positive),
                    "score_gap": score_gap,
                    "margin": margin,
                    "entropy": entropy,
                }
            )
            processed_groups += 1

        analysis_scope = "full"
        if int(args.max_groups) > 0 or int(args.max_frame) > 0:
            analysis_scope = "partial"

        summary_row.update(
            {
                "status": "completed",
                "analysis_scope": analysis_scope,
                "groups_with_gt": groups_with_gt,
                "wrong_selected_groups": wrong_selected_groups,
                "fixable_groups": fixable_groups,
                "fixable_percent": round(100.0 * safe_ratio(fixable_groups, wrong_selected_groups), 6),
                "median_positive_rank": median_or_none(positive_ranks),
            }
        )
        write_single_row_csv(summary_csv, summary_row, SUMMARY_FIELDS)
        metrics = {
            "groups_with_gt": groups_with_gt,
            "wrong_selected_groups": wrong_selected_groups,
            "fixable_groups": fixable_groups,
            "fixable_percent": summary_row["fixable_percent"],
            "median_positive_rank": summary_row["median_positive_rank"],
            "max_margin_gap": float(args.max_margin_gap),
            "top_k": int(args.top_k),
            "analysis_scope": analysis_scope,
            "processed_groups": processed_groups,
            "max_groups": int(args.max_groups),
            "max_frame": int(args.max_frame),
        }
        write_json(metrics, out_dir / "oracle_cost_rerank_metrics.json")
        write_json({"events": fixable_events}, out_dir / "fixable_events.json")
        write_markdown(
            "\n".join(
                [
                    "# Oracle 0C Cost Rerank",
                    "",
                    f"- seq_name: {args.seq_name}",
                    f"- groups_with_gt: {groups_with_gt}",
                    f"- wrong_selected_groups: {wrong_selected_groups}",
                    f"- fixable_groups: {fixable_groups}",
                    f"- fixable_percent: {summary_row['fixable_percent']}",
                    f"- median_positive_rank: {summary_row['median_positive_rank']}",
                    f"- analysis_scope: {analysis_scope}",
                    f"- processed_groups: {processed_groups}",
                ]
            ),
            out_dir / "oracle_cost_rerank_report.md",
        )
        write_manifest(
            out_dir,
            phase="oracle_0C_cost_rerank",
            script=script_path,
            args=vars(args),
            status="ok",
            metrics=metrics,
            artifacts={
                "summary_csv": str(summary_csv),
                "metrics_json": str(out_dir / "oracle_cost_rerank_metrics.json"),
                "events_json": str(out_dir / "fixable_events.json"),
            },
            notes=f"cost rerank oracle for {args.seq_name}",
        )
        append_registry(
            kind="analysis",
            status="success",
            script=script_path,
            dataset=args.dataset,
            split=args.split,
            tracker_family="spot_oracle_0C",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"cost rerank oracle complete for {args.seq_name}",
        )
        upsert_plan(
            status="completed",
            kind="analysis",
            script=script_path,
            dataset=args.dataset,
            split=args.split,
            tracker_family="spot_oracle_0C",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"cost rerank oracle complete for {args.seq_name}",
            key=f"spot_oracle_0C:{out_dir}",
        )
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = str(exc)
        write_single_row_csv(summary_csv, summary_row, SUMMARY_FIELDS)
        append_registry(
            kind="analysis",
            status="failed",
            script=script_path,
            dataset=args.dataset,
            split=args.split,
            tracker_family="spot_oracle_0C",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"cost rerank oracle failed: {exc}",
        )
        upsert_plan(
            status="failed",
            kind="analysis",
            script=script_path,
            dataset=args.dataset,
            split=args.split,
            tracker_family="spot_oracle_0C",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"cost rerank oracle failed: {exc}",
            key=f"spot_oracle_0C:{out_dir}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
