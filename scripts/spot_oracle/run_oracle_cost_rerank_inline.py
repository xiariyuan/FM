#!/usr/bin/env python3
"""Oracle 0C cost rerank with inline GT lookup.

Reads a RGSA-style pairbank CSV and GT MOT txt directly, computes
per-frame one-to-one GT matches for detections and tracks, then
evaluates whether cost reranking could fix wrong host selections.

This avoids building a massive intermediate trusted_pairbank file.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

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

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


SUMMARY_FIELDS = [
    "status",
    "error",
    "dataset",
    "split",
    "seq_name",
    "analysis_scope",
    "trusted",
    "label_source",
    "groups_with_gt",
    "wrong_selected_groups",
    "fixable_groups",
    "fixable_percent",
    "median_positive_rank",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle 0C with inline GT one-to-one matching.")
    parser.add_argument("--pair-csv", required=True)
    parser.add_argument("--gt-txt", required=True)
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--split", default="unknown")
    parser.add_argument("--seq-name", default="unknown_seq")
    parser.add_argument("--out-dir", default="outputs/oracle_gate/0C_cost_reranking")
    parser.add_argument("--max-margin-gap", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--max-frame", type=int, default=0)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    return parser.parse_args()


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


def _load_gt(gt_path: str, iou_thresh: float) -> dict[int, list[tuple[int, tuple[float, float, float, float]]]]:
    gt: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fid = int(float(parts[0]))
            gid = int(float(parts[1]))
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            gt[fid].append((gid, (x, y, w, h)))
    return dict(gt)


def _one_to_one_gt(
    boxes: list[tuple[str, tuple[float, float, float, float]]],
    gt_entries: list[tuple[int, tuple[float, float, float, float]]],
    *,
    iou_thresh: float,
) -> dict[str, tuple[int, float]]:
    if not boxes or not gt_entries:
        return {}
    ious = [[iou_tlwh(box, gt_tlwh) for _, gt_tlwh in gt_entries] for _, box in boxes]
    out: dict[str, tuple[int, float]] = {}
    if linear_sum_assignment is not None:
        costs = [[1.0 - s for s in row] for row in ious]
        ri, ci = linear_sum_assignment(costs)
        for r, c in zip(ri, ci):
            score = float(ious[int(r)][int(c)])
            if score >= iou_thresh:
                out[boxes[int(r)][0]] = (int(gt_entries[int(c)][0]), score)
        return out
    candidates: list[tuple[float, int, int]] = []
    for r, row in enumerate(ious):
        for c, score in enumerate(row):
            if score >= iou_thresh:
                candidates.append((float(score), r, c))
    used_r: set[int] = set()
    used_c: set[int] = set()
    for score, r, c in sorted(candidates, reverse=True):
        if r in used_r or c in used_c:
            continue
        out[boxes[r][0]] = (int(gt_entries[c][0]), score)
        used_r.add(r)
        used_c.add(c)
    return out


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
        "trusted": 1,
        "label_source": "inline_gt_one_to_one",
        "groups_with_gt": 0,
        "wrong_selected_groups": 0,
        "fixable_groups": 0,
        "fixable_percent": 0.0,
        "median_positive_rank": "",
    }
    write_single_row_csv(summary_csv, summary_row, SUMMARY_FIELDS)
    append_registry(
        kind="analysis", status="running", script=script_path,
        dataset=args.dataset, split=args.split,
        tracker_family="spot_oracle_0C", variant=variant, tag=tag,
        run_root=out_dir, summary_csv=summary_csv,
        notes=f"inline GT 0C running for {args.seq_name}",
    )
    upsert_plan(
        status="running", kind="analysis", script=script_path,
        dataset=args.dataset, split=args.split,
        tracker_family="spot_oracle_0C", variant=variant, tag=tag,
        run_root=out_dir, summary_csv=summary_csv,
        notes=f"inline GT 0C running for {args.seq_name}",
        key=f"spot_oracle_0C:{out_dir}",
    )

    try:
        gt_data = _load_gt(args.gt_txt, float(args.iou_thresh))
        wrong_selected_groups = 0
        fixable_groups = 0
        groups_with_gt = 0
        processed_groups = 0
        positive_ranks: list[int] = []
        fixable_events: list[dict] = []

        # Stream pairbank: group rows by (seq, frame, det_id)
        with Path(args.pair_csv).expanduser().open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            current_key: str | None = None
            bucket: list[dict] = []

            def _process_bucket(key: str, rows: list[dict]):
                nonlocal wrong_selected_groups, fixable_groups, groups_with_gt, processed_groups
                if not rows:
                    return
                frame_id = _int(rows[0], ("frame", "frame_id"), 0)
                gt_entries = gt_data.get(frame_id, [])

                # Compute GT match for each unique det and track in this group
                det_boxes: dict[str, tuple[float, float, float, float]] = {}
                track_boxes: dict[str, tuple[float, float, float, float]] = {}
                for row in rows:
                    did = str(row.get("det_id") or row.get("det_index") or "")
                    db = _parse_tlwh(str(row.get("det_tlwh", "")))
                    if did and db is not None:
                        det_boxes.setdefault(did, db)
                    tid = str(row.get("track_id") or "")
                    tb = _parse_tlwh(str(row.get("track_tlwh", "")))
                    if tid and tb is not None:
                        track_boxes.setdefault(tid, tb)

                det_map = _one_to_one_gt(list(det_boxes.items()), gt_entries, iou_thresh=float(args.iou_thresh))
                track_map = _one_to_one_gt(list(track_boxes.items()), gt_entries, iou_thresh=float(args.iou_thresh))

                # Annotate rows inline
                for row in rows:
                    did = str(row.get("det_id") or row.get("det_index") or "")
                    tid = str(row.get("track_id") or "")
                    det_gt, det_iou = det_map.get(did, (-1, 0.0))
                    track_gt, track_iou = track_map.get(tid, (-1, 0.0))
                    row["det_gt_id"] = str(det_gt)
                    row["track_gt_id"] = str(track_gt)
                    row["_is_positive"] = "1" if (det_gt > 0 and track_gt > 0 and det_gt == track_gt) else "0"

                ordered = sorted(rows, key=_candidate_score, reverse=True)
                positives = [row for row in ordered if row.get("_is_positive") == "1"]
                if not positives:
                    processed_groups += 1
                    return
                groups_with_gt += 1
                selected = next((row for row in ordered if _selected_flag(row) == 1), ordered[0])
                selected_correct = selected.get("_is_positive") == "1"
                pos_rank = next((idx for idx, row in enumerate(ordered) if row.get("_is_positive") == "1"), None)
                if pos_rank is not None:
                    positive_ranks.append(int(pos_rank))
                if selected_correct:
                    processed_groups += 1
                    return
                wrong_selected_groups += 1
                positive = positives[0]
                rank = next(idx for idx, row in enumerate(ordered) if row is positive)
                score_gap = _candidate_score(selected) - _candidate_score(positive)
                margin = top2_margin(_candidate_score(row) for row in ordered)
                entropy = normalized_entropy(max(_candidate_score(row), 0.0) for row in ordered)
                is_fixable = int(rank < int(args.top_k) and score_gap <= float(args.max_margin_gap))
                if is_fixable:
                    fixable_groups += 1
                fixable_events.append({
                    "group_id": key,
                    "fixable": is_fixable,
                    "positive_rank": rank,
                    "selected_track_id": _int(selected, ("track_id",), -1),
                    "positive_track_id": _int(positive, ("track_id",), -1),
                    "selected_score": _candidate_score(selected),
                    "positive_score": _candidate_score(positive),
                    "score_gap": score_gap,
                    "margin": margin,
                    "entropy": entropy,
                })
                processed_groups += 1

            for row in reader:
                if args.max_groups and processed_groups >= int(args.max_groups):
                    break
                frame_id = _int(row, ("frame", "frame_id"), 0)
                if args.max_frame and frame_id > int(args.max_frame):
                    break
                key = _group_key(row)
                if current_key is None:
                    current_key = key
                if key != current_key:
                    _process_bucket(current_key, bucket)
                    bucket = []
                    current_key = key
                bucket.append(row)
            if current_key is not None and bucket:
                if not (args.max_groups and processed_groups >= int(args.max_groups)):
                    _process_bucket(current_key, bucket)

        analysis_scope = "full"
        if int(args.max_groups) > 0 or int(args.max_frame) > 0:
            analysis_scope = "partial"
        trusted = int(analysis_scope == "full" and wrong_selected_groups > 0)

        summary_row.update({
            "status": "completed",
            "analysis_scope": analysis_scope,
            "trusted": trusted,
            "label_source": "inline_gt_one_to_one",
            "groups_with_gt": groups_with_gt,
            "wrong_selected_groups": wrong_selected_groups,
            "fixable_groups": fixable_groups,
            "fixable_percent": round(100.0 * safe_ratio(fixable_groups, wrong_selected_groups), 6),
            "median_positive_rank": median_or_none(positive_ranks),
        })
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
            "trusted": trusted,
            "label_source": "inline_gt_one_to_one",
            "processed_groups": processed_groups,
            "max_groups": int(args.max_groups),
            "max_frame": int(args.max_frame),
        }
        write_json(metrics, out_dir / "oracle_cost_rerank_metrics.json")
        write_json({"events": fixable_events}, out_dir / "fixable_events.json")
        write_markdown(
            "\n".join([
                "# Oracle 0C Cost Rerank (inline GT)",
                "",
                f"- seq_name: {args.seq_name}",
                f"- groups_with_gt: {groups_with_gt}",
                f"- wrong_selected_groups: {wrong_selected_groups}",
                f"- fixable_groups: {fixable_groups}",
                f"- fixable_percent: {summary_row['fixable_percent']}",
                f"- median_positive_rank: {summary_row['median_positive_rank']}",
                f"- analysis_scope: {analysis_scope}",
                f"- trusted: {trusted}",
                f"- label_source: inline_gt_one_to_one",
                f"- processed_groups: {processed_groups}",
            ]),
            out_dir / "oracle_cost_rerank_report.md",
        )
        write_manifest(
            out_dir,
            phase="oracle_0C_cost_rerank_inline",
            script=script_path,
            args=vars(args),
            status="ok",
            metrics=metrics,
            artifacts={
                "summary_csv": str(summary_csv),
                "metrics_json": str(out_dir / "oracle_cost_rerank_metrics.json"),
                "events_json": str(out_dir / "fixable_events.json"),
            },
            notes=f"inline GT cost rerank oracle for {args.seq_name}",
        )
        append_registry(
            kind="analysis", status="success", script=script_path,
            dataset=args.dataset, split=args.split,
            tracker_family="spot_oracle_0C", variant=variant, tag=tag,
            run_root=out_dir, summary_csv=summary_csv,
            notes=f"inline GT 0C complete for {args.seq_name}",
        )
        upsert_plan(
            status="completed", kind="analysis", script=script_path,
            dataset=args.dataset, split=args.split,
            tracker_family="spot_oracle_0C", variant=variant, tag=tag,
            run_root=out_dir, summary_csv=summary_csv,
            notes=f"inline GT 0C complete for {args.seq_name}",
            key=f"spot_oracle_0C:{out_dir}",
        )
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = str(exc)
        write_single_row_csv(summary_csv, summary_row, SUMMARY_FIELDS)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
