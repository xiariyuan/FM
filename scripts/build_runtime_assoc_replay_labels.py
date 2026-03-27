#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_gt_pseudotrack_groups import (  # noqa: E402
    _assign_detections_to_gt,
    _history_file_name,
    _read_gt_rows,
    _read_seqinfo,
    _split_gt_frame,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Label runtime association candidate dumps with GT-derived detection/track ids.")
    ap.add_argument("--dump-root", required=True, help="Directory containing per-sequence assoc dump CSV files.")
    ap.add_argument("--dataset", default="MOT17", choices=["MOT17", "MOT20"])
    ap.add_argument("--data-root", default="/gemini/code/datasets")
    ap.add_argument("--split", default="train")
    ap.add_argument("--split-part", default="full", choices=["full", "train_half", "val_half"])
    ap.add_argument("--iou-pos", type=float, default=0.7)
    ap.add_argument("--iou-ignore", type=float, default=0.5)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--summary-json", default="")
    ap.add_argument("--out-group-jsonl", default="")
    ap.add_argument("--out-recoverability-json", default="")
    ap.add_argument("--topk", type=int, default=8, help="Top-k cutoff for recoverability diagnostics. Use <=0 to mean all dumped rows.")
    ap.add_argument("--rank-score-col", default="refined_score", choices=["base_score", "refined_score"])
    ap.add_argument("--ambiguity-margin", type=float, default=0.10)
    return ap.parse_args()


def _cxcywh_to_tlbr(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    return np.asarray([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dtype=np.float32)


def _maybe_denormalize_tlbr(tlbr: np.ndarray, seq_w: float, seq_h: float) -> np.ndarray:
    arr = np.asarray(tlbr, dtype=np.float32).copy()
    if float(np.max(np.abs(arr))) <= 2.0:
        arr[[0, 2]] *= float(seq_w)
        arr[[1, 3]] *= float(seq_h)
    return arr


def _match_single_box(
    tlbr: np.ndarray,
    gt_positive: list[dict[str, float]],
    gt_ignore: list[dict[str, float]],
    iou_pos: float,
    iou_ignore: float,
) -> tuple[int, int]:
    assigned, ignore = _assign_detections_to_gt(
        det_tlbrs=np.asarray([tlbr], dtype=np.float32),
        gt_positive=gt_positive,
        gt_ignore=gt_ignore,
        iou_pos=float(iou_pos),
        iou_ignore=float(iou_ignore),
    )
    return int(assigned[0]), int(ignore[0])


def _read_dump_rows(path: Path) -> dict[tuple[int, int], list[dict[str, str]]]:
    groups: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"frame", "det_index", "track_rank", "track_id", "base_score", "refined_score", "assoc_mode"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            return groups
        for row in reader:
            frame = int(row["frame"])
            det_index = int(row["det_index"])
            groups[(frame, det_index)].append(row)
    return groups


def _resolve_seq_name(row_seq: str | None, csv_path: Path) -> str:
    if row_seq:
        return Path(str(row_seq)).name
    return csv_path.stem


def _group_id(seq_name: str, frame: int, det_index: int) -> str:
    return f"{seq_name}:{int(frame)}:{int(det_index)}"


def _score_entropy(scores: list[float]) -> float:
    if not scores:
        return 0.0
    arr = np.asarray(scores, dtype=np.float32)
    arr = np.clip(arr, 1e-4, 1.0 - 1e-4)
    logits = np.log(arr / np.clip(1.0 - arr, 1e-4, 1.0))
    logits = logits - float(np.max(logits))
    prob = np.exp(logits)
    prob = prob / np.clip(float(np.sum(prob)), 1e-8, None)
    return float(-(prob * np.log(np.clip(prob, 1e-8, None))).sum())


def main() -> None:
    args = parse_args()
    dump_root = Path(args.dump_root).resolve()
    out_csv = Path(args.out_csv).resolve()
    skip_paths = {out_csv}
    if args.summary_json:
        skip_paths.add(Path(args.summary_json).resolve())
    if args.out_group_jsonl:
        skip_paths.add(Path(args.out_group_jsonl).resolve())
    if args.out_recoverability_json:
        skip_paths.add(Path(args.out_recoverability_json).resolve())
    csv_paths = sorted(p for p in dump_root.rglob("*.csv") if p.resolve() not in skip_paths)
    if not csv_paths:
        raise FileNotFoundError(f"No dump CSV files found under {dump_root}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "groups": 0,
        "rows": 0,
        "positive_rows": 0,
        "background_groups": 0,
        "ignored_rows": 0,
        "det_positive_groups": 0,
        "positive_groups": 0,
        "ambiguous_groups": 0,
        "recoverable_groups": 0,
        "positive_in_topk_groups": 0,
        "rank_top1_correct_groups": 0,
        "ambiguous_rank_top1_correct_groups": 0,
        "rank_score_col": str(args.rank_score_col),
        "topk_eval": int(args.topk),
        "ambiguity_margin": float(args.ambiguity_margin),
    }

    gt_cache: dict[str, dict[int, list[dict[str, float]]]] = {}
    seq_hw_cache: dict[str, tuple[float, float]] = {}
    fieldnames = [
        "seq",
        "frame",
        "assoc_mode",
        "group_id",
        "det_index",
        "track_rank",
        "track_id",
        "is_selected",
        "det_score",
        "base_score",
        "refined_score",
        "motion_score",
        "track_gap",
        "track_hist_len",
        "det_gt_id",
        "track_gt_id",
        "det_ignore",
        "track_ignore",
        "label",
        "valid_train_row",
        "group_has_positive",
        "group_size",
        "candidate_count_total",
        "base_margin",
        "refined_margin",
        "rank_score_col",
        "rank_margin",
        "rank_entropy",
        "rank_top1_correct",
        "positive_in_topk",
        "positive_rank",
        "group_is_ambiguous",
        "group_is_background",
        "group_is_recoverable",
        "det_cx",
        "det_cy",
        "det_w",
        "det_h",
        "track_cx",
        "track_cy",
        "track_w",
        "track_h",
        "source_csv",
    ]

    group_jsonl_fp = None
    if args.out_group_jsonl:
        out_group_jsonl = Path(args.out_group_jsonl).resolve()
        out_group_jsonl.parent.mkdir(parents=True, exist_ok=True)
        group_jsonl_fp = out_group_jsonl.open("w", encoding="utf-8")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for csv_path in csv_paths:
            grouped = _read_dump_rows(csv_path)
            if not grouped:
                continue

            any_row = next(iter(next(iter(grouped.values()))))
            seq_name = _resolve_seq_name(any_row.get("seq", ""), csv_path)
            if seq_name not in gt_cache:
                seq_dir = Path(args.data_root) / args.dataset / args.split / seq_name
                gt_path = _history_file_name(seq_dir, "gt", args.split_part)
                gt_cache[seq_name] = _read_gt_rows(gt_path)
                seqinfo = _read_seqinfo(seq_dir)
                seq_hw_cache[seq_name] = (
                    float(seqinfo.get("imWidth", seqinfo.get("imwidth"))),
                    float(seqinfo.get("imHeight", seqinfo.get("imheight"))),
                )
            gt_rows = gt_cache[seq_name]
            seq_w, seq_h = seq_hw_cache[seq_name]

            for (frame, det_index), rows in grouped.items():
                rows = sorted(rows, key=lambda r: int(r["track_rank"]))
                group_id = rows[0].get("group_id", "") or _group_id(seq_name, frame, det_index)
                det_box = _maybe_denormalize_tlbr(_cxcywh_to_tlbr(
                    float(rows[0]["det_cx"]),
                    float(rows[0]["det_cy"]),
                    float(rows[0]["det_w"]),
                    float(rows[0]["det_h"]),
                ), seq_w=seq_w, seq_h=seq_h)
                gt_positive, gt_ignore = _split_gt_frame(gt_rows.get(frame, []))
                det_gt_id, det_ignore = _match_single_box(
                    tlbr=det_box,
                    gt_positive=gt_positive,
                    gt_ignore=gt_ignore,
                    iou_pos=args.iou_pos,
                    iou_ignore=args.iou_ignore,
                )

                base_vals = [float(r["base_score"]) for r in rows]
                refined_vals = [float(r["refined_score"]) for r in rows]
                base_margin = 0.0 if len(base_vals) <= 1 else float(sorted(base_vals, reverse=True)[0] - sorted(base_vals, reverse=True)[1])
                refined_margin = 0.0 if len(refined_vals) <= 1 else float(sorted(refined_vals, reverse=True)[0] - sorted(refined_vals, reverse=True)[1])
                rank_vals = refined_vals if args.rank_score_col == "refined_score" else base_vals
                rank_margin = refined_margin if args.rank_score_col == "refined_score" else base_margin
                rank_entropy = _score_entropy(rank_vals)
                topk_eval = len(rows) if int(args.topk) <= 0 else min(int(args.topk), len(rows))
                candidate_count_total = int(float(rows[0].get("candidate_count_total", len(rows))))

                labeled_rows: list[dict[str, object]] = []
                for row in rows:
                    gap = int(float(row["track_gap"]))
                    track_frame = int(frame - gap) if gap >= 0 else int(frame)
                    track_box = _maybe_denormalize_tlbr(_cxcywh_to_tlbr(
                        float(row["track_cx"]),
                        float(row["track_cy"]),
                        float(row["track_w"]),
                        float(row["track_h"]),
                    ), seq_w=seq_w, seq_h=seq_h)
                    hist_positive, hist_ignore = _split_gt_frame(gt_rows.get(track_frame, []))
                    track_gt_id, track_ignore = _match_single_box(
                        tlbr=track_box,
                        gt_positive=hist_positive,
                        gt_ignore=hist_ignore,
                        iou_pos=args.iou_pos,
                        iou_ignore=args.iou_ignore,
                    )
                    label = 1 if det_gt_id > 0 and track_gt_id > 0 and det_gt_id == track_gt_id else 0
                    valid_train_row = 1 if det_ignore == 0 and track_ignore == 0 else 0
                    labeled_rows.append(
                        {
                            "seq": seq_name,
                            "frame": int(frame),
                            "assoc_mode": str(row.get("assoc_mode", "")),
                            "group_id": group_id,
                            "det_index": int(det_index),
                            "track_rank": int(row["track_rank"]),
                            "track_id": int(row["track_id"]),
                            "is_selected": int(row["is_selected"]),
                            "det_score": float(row["det_score"]),
                            "base_score": float(row["base_score"]),
                            "refined_score": float(row["refined_score"]),
                            "motion_score": float(row["motion_score"]),
                            "track_gap": gap,
                            "track_hist_len": int(float(row["track_hist_len"])),
                            "det_gt_id": int(det_gt_id),
                            "track_gt_id": int(track_gt_id),
                            "det_ignore": int(det_ignore),
                            "track_ignore": int(track_ignore),
                            "label": int(label),
                            "valid_train_row": int(valid_train_row),
                            "group_has_positive": 0,  # filled below
                            "group_size": int(len(rows)),
                            "candidate_count_total": candidate_count_total,
                            "base_margin": base_margin,
                            "refined_margin": refined_margin,
                            "rank_score_col": str(args.rank_score_col),
                            "rank_margin": rank_margin,
                            "rank_entropy": rank_entropy,
                            "rank_top1_correct": 0,  # filled below
                            "positive_in_topk": 0,  # filled below
                            "positive_rank": -1,  # filled below
                            "group_is_ambiguous": 0,  # filled below
                            "group_is_background": 0,  # filled below
                            "group_is_recoverable": 0,  # filled below
                            "det_cx": float(row["det_cx"]),
                            "det_cy": float(row["det_cy"]),
                            "det_w": float(row["det_w"]),
                            "det_h": float(row["det_h"]),
                            "track_cx": float(row["track_cx"]),
                            "track_cy": float(row["track_cy"]),
                            "track_w": float(row["track_w"]),
                            "track_h": float(row["track_h"]),
                            "source_csv": str(csv_path),
                        }
                    )

                positive_rows = [
                    idx for idx, r in enumerate(labeled_rows)
                    if int(r["label"]) == 1 and int(r["valid_train_row"]) == 1
                ]
                group_has_positive = 1 if positive_rows else 0
                positive_rank = min((int(labeled_rows[idx]["track_rank"]) for idx in positive_rows), default=-1)
                positive_in_topk = 1 if positive_rank > 0 and positive_rank <= topk_eval else 0
                rank_arr = np.asarray(rank_vals, dtype=np.float32)
                rank_top1_idx = int(np.argmax(rank_arr)) if rank_arr.size > 0 else -1
                rank_top1_correct = 1 if rank_top1_idx in positive_rows else 0
                group_is_background = 1 if group_has_positive == 0 else 0
                group_is_ambiguous = 1 if (
                    group_has_positive == 1 and (rank_top1_correct == 0 or float(rank_margin) < float(args.ambiguity_margin))
                ) else 0
                group_is_recoverable = 1 if (
                    group_has_positive == 1 and rank_top1_correct == 0 and positive_in_topk == 1
                ) else 0

                if det_gt_id > 0:
                    summary["det_positive_groups"] += 1
                if group_has_positive == 0:
                    summary["background_groups"] += 1
                else:
                    summary["positive_groups"] += 1
                    summary["positive_in_topk_groups"] += positive_in_topk
                    summary["rank_top1_correct_groups"] += rank_top1_correct
                    if group_is_ambiguous:
                        summary["ambiguous_groups"] += 1
                        summary["ambiguous_rank_top1_correct_groups"] += rank_top1_correct
                    if group_is_recoverable:
                        summary["recoverable_groups"] += 1
                summary["groups"] += 1

                if group_jsonl_fp is not None:
                    group_record = {
                        "group_id": group_id,
                        "seq": seq_name,
                        "frame": int(frame),
                        "det_index": int(det_index),
                        "assoc_mode": str(rows[0].get("assoc_mode", "")),
                        "group_size": int(len(rows)),
                        "candidate_count_total": candidate_count_total,
                        "det_gt_id": int(det_gt_id),
                        "det_ignore": int(det_ignore),
                        "group_has_positive": int(group_has_positive),
                        "group_is_background": int(group_is_background),
                        "group_is_ambiguous": int(group_is_ambiguous),
                        "group_is_recoverable": int(group_is_recoverable),
                        "positive_rank": int(positive_rank),
                        "positive_in_topk": int(positive_in_topk),
                        "rank_score_col": str(args.rank_score_col),
                        "rank_margin": float(rank_margin),
                        "rank_entropy": float(rank_entropy),
                        "rank_top1_correct": int(rank_top1_correct),
                        "base_margin": float(base_margin),
                        "refined_margin": float(refined_margin),
                        "candidates": [
                            {
                                "track_rank": int(r["track_rank"]),
                                "track_id": int(r["track_id"]),
                                "label": int(r["label"]),
                                "valid_train_row": int(r["valid_train_row"]),
                                "base_score": float(r["base_score"]),
                                "refined_score": float(r["refined_score"]),
                                "motion_score": float(r["motion_score"]),
                                "track_gap": int(r["track_gap"]),
                                "track_hist_len": int(r["track_hist_len"]),
                            }
                            for r in labeled_rows
                        ],
                    }
                    group_jsonl_fp.write(json.dumps(group_record, ensure_ascii=False) + "\n")

                for row in labeled_rows:
                    row["group_has_positive"] = group_has_positive
                    row["rank_top1_correct"] = rank_top1_correct
                    row["positive_in_topk"] = positive_in_topk
                    row["positive_rank"] = positive_rank
                    row["group_is_ambiguous"] = group_is_ambiguous
                    row["group_is_background"] = group_is_background
                    row["group_is_recoverable"] = group_is_recoverable
                    writer.writerow(row)
                    summary["rows"] += 1
                    summary["positive_rows"] += int(row["label"])
                    summary["ignored_rows"] += 1 if int(row["valid_train_row"]) == 0 else 0

    if group_jsonl_fp is not None:
        group_jsonl_fp.close()

    positive_groups = max(int(summary["positive_groups"]), 1)
    ambiguous_groups = max(int(summary["ambiguous_groups"]), 1)
    recoverability = {
        "groups": int(summary["groups"]),
        "positive_groups": int(summary["positive_groups"]),
        "background_groups": int(summary["background_groups"]),
        "ambiguous_groups": int(summary["ambiguous_groups"]),
        "recoverable_groups": int(summary["recoverable_groups"]),
        "rank_score_col": str(args.rank_score_col),
        "topk_eval": int(args.topk),
        "ambiguity_margin": float(args.ambiguity_margin),
        "positive_in_topk_rate": float(summary["positive_in_topk_groups"]) / float(positive_groups),
        "rank_top1_acc_positive": float(summary["rank_top1_correct_groups"]) / float(positive_groups),
        "rank_top1_acc_ambiguous": (
            float(summary["ambiguous_rank_top1_correct_groups"]) / float(ambiguous_groups)
            if int(summary["ambiguous_groups"]) > 0 else 0.0
        ),
        "recoverable_rate_among_positive": float(summary["recoverable_groups"]) / float(positive_groups),
        "recoverable_rate_among_ambiguous": (
            float(summary["recoverable_groups"]) / float(ambiguous_groups)
            if int(summary["ambiguous_groups"]) > 0 else 0.0
        ),
    }
    summary["recoverability"] = recoverability

    if args.summary_json:
        summary_path = Path(args.summary_json).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.out_recoverability_json:
        recoverability_path = Path(args.out_recoverability_json).resolve()
        recoverability_path.parent.mkdir(parents=True, exist_ok=True)
        recoverability_path.write_text(json.dumps(recoverability, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
