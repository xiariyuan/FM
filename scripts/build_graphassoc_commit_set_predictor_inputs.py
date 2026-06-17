#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert graph-assoc commit cluster_examples.jsonl into rows_csv and group_jsonl "
            "for the local-conflict set predictor."
        )
    )
    parser.add_argument("--cluster-jsonl", required=True, help="Input graph_assoc_commit_data/cluster_examples.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-tag", default="graphassoc_commit_set_predictor")
    parser.add_argument("--feature-version", default="graphassoc_commit_v1")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--train-sequences", default="")
    parser.add_argument("--val-sequences", default="")
    parser.add_argument("--strict-sequence-split", action="store_true")
    return parser.parse_args()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_csv_tokens(raw: Any) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [token.strip() for token in text.split(",") if token.strip()]


def _sequence_aliases(seq: str) -> set[str]:
    raw = str(seq or "").strip()
    if not raw:
        return set()
    name = Path(raw).name
    aliases = {raw, name}
    if name.count("-") >= 2:
        aliases.add(name.rsplit("-", 1)[0])
    return {token for token in aliases if token}


def _sequence_matches(seq: str, tokens: list[str]) -> bool:
    if not tokens:
        return False
    aliases = _sequence_aliases(seq)
    for token in tokens:
        token = str(token or "").strip()
        if not token:
            continue
        for alias in aliases:
            if alias == token or alias.startswith(token):
                return True
    return False


def _determine_split_tag(
    *,
    seq: str,
    explicit_split_tag: str,
    train_tokens: list[str],
    val_tokens: list[str],
    strict_sequence_split: bool,
) -> str:
    explicit = str(explicit_split_tag or "").strip().lower()
    if explicit in {"train", "val", "unused"}:
        return explicit
    if _sequence_matches(seq, val_tokens):
        return "val"
    if train_tokens:
        if _sequence_matches(seq, train_tokens):
            return "train"
        return "unused" if strict_sequence_split else "train"
    if val_tokens:
        return "train"
    return "all"


def _entropy_from_scores(scores: list[float]) -> float:
    if not scores:
        return 0.0
    arr = np.asarray(scores, dtype=np.float32)
    arr = arr - float(np.max(arr))
    probs = np.exp(arr)
    denom = float(np.sum(probs))
    if denom <= 0.0:
        return 0.0
    probs = probs / denom
    return float(-(probs * np.log(np.clip(probs, 1e-8, None))).sum())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _neutral_box_fields() -> tuple[float, float, float, float]:
    return 0.0, 0.0, 1.0, 1.0


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_path = Path(args.cluster_jsonl).resolve()
    if not cluster_path.is_file():
        raise FileNotFoundError(f"Missing cluster jsonl: {cluster_path}")

    examples = _read_jsonl(cluster_path)
    if not examples:
        raise ValueError(f"Empty cluster jsonl: {cluster_path}")

    train_tokens = _parse_csv_tokens(args.train_sequences)
    val_tokens = _parse_csv_tokens(args.val_sequences)

    rows_out: list[dict[str, Any]] = []
    groups_out: list[dict[str, Any]] = []
    summary = Counter()
    split_counter: dict[str, Counter[str]] = defaultdict(Counter)
    seq_counter: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    seen_sequences: set[str] = set()
    seen_sources: set[str] = set()
    seen_host_variants: set[str] = set()

    for example in examples:
        seq = str(example.get("seq", "")).strip()
        if not seq:
            continue
        split_tag = _determine_split_tag(
            seq=seq,
            explicit_split_tag=str(example.get("split_tag", "")),
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            strict_sequence_split=bool(args.strict_sequence_split),
        )
        if split_tag == "unused":
            summary["unused_examples"] += 1
            continue

        cluster_id = str(example.get("cluster_id", "")).strip() or f"{seq}:{_safe_int(example.get('frame', 0), 0)}"
        source_tag = str(example.get("source_tag", "")).strip() or "source_000"
        host_variant = str(example.get("host_variant", "")).strip() or "unknown"
        feature_version = str(example.get("feature_version", "")).strip() or str(args.feature_version)
        frame = _safe_int(example.get("frame", 0), 0)
        det_rows = [int(x) for x in example.get("det_rows", []) if _safe_int(x, -1) >= 0]
        track_ids = [int(x) for x in example.get("track_ids", []) if _safe_int(x, -1) >= 0]
        det_features = list(example.get("det_features", []))
        track_features = list(example.get("track_features", []))
        edge_features = list(example.get("edge_features", []))
        edge_det_index = [int(x) for x in example.get("edge_det_index", [])]
        edge_track_index = [int(x) for x in example.get("edge_track_index", [])]
        compact_edges = list(example.get("compact_edges", []))

        if not det_rows or not track_ids or not edge_features:
            summary["skipped_missing_fields"] += 1
            continue

        edge_map: dict[tuple[int, int], dict[str, Any]] = {}
        for idx, (det_idx, track_idx, feat) in enumerate(zip(edge_det_index, edge_track_index, edge_features)):
            is_positive = 0
            if idx < len(compact_edges):
                is_positive = _safe_int(compact_edges[idx].get("is_positive", 0), 0)
            edge_map[(int(det_idx), int(track_idx))] = {
                "base_score": float(feat[0]) if len(feat) > 0 else 0.0,
                "refined_score": float(feat[1]) if len(feat) > 1 else 0.0,
                "motion_score": float(feat[2]) if len(feat) > 2 else 0.0,
                "track_gap": float(feat[3]) if len(feat) > 3 else 0.0,
                "track_hist_len": float(feat[4]) if len(feat) > 4 else 0.0,
                "rank_frac": float(feat[5]) if len(feat) > 5 else 1.0,
                "label": int(is_positive),
            }

        summary["clusters"] += 1
        summary["detections"] += int(len(det_rows))
        summary["tracks"] += int(len(track_ids))
        summary["edges"] += int(len(edge_map))
        seen_sequences.add(seq)
        seen_sources.add(source_tag)
        seen_host_variants.add(host_variant)
        split_counter[split_tag]["clusters"] += 1
        split_counter[split_tag]["detections"] += int(len(det_rows))
        split_counter[split_tag]["tracks"] += int(len(track_ids))
        split_counter[split_tag]["edges"] += int(len(edge_map))
        seq_counter[(seq, split_tag)]["clusters"] += 1
        seq_counter[(seq, split_tag)]["detections"] += int(len(det_rows))
        seq_counter[(seq, split_tag)]["tracks"] += int(len(track_ids))
        seq_counter[(seq, split_tag)]["edges"] += int(len(edge_map))

        det_feature_by_local: dict[int, list[float]] = {
            int(idx): [float(v) for v in feat]
            for idx, feat in enumerate(det_features)
            if isinstance(feat, (list, tuple))
        }
        track_feature_by_local: dict[int, list[float]] = {
            int(idx): [float(v) for v in feat]
            for idx, feat in enumerate(track_features)
            if isinstance(feat, (list, tuple))
        }

        for det_local_idx, det_row in enumerate(det_rows):
            group_id = f"{cluster_id}|det{int(det_row)}"
            det_feat = det_feature_by_local.get(int(det_local_idx), [0.0, 0.0, 0.0, 0.0])
            det_score = float(det_feat[0]) if len(det_feat) > 0 else 0.0
            det_cx, det_cy, det_w, det_h = _neutral_box_fields()
            track_candidates: list[dict[str, Any]] = []
            for track_local_idx, track_id in enumerate(track_ids):
                edge = edge_map.get((int(det_local_idx), int(track_local_idx)))
                if edge is None:
                    continue
                track_feat = track_feature_by_local.get(int(track_local_idx), [0.0, 0.0, 0.0])
                track_gap = float(edge.get("track_gap", 0.0))
                track_hist_len = float(edge.get("track_hist_len", 0.0))
                if track_gap <= 0.0 and len(track_feat) > 0:
                    track_gap = float(track_feat[0])
                if track_hist_len <= 0.0 and len(track_feat) > 1:
                    track_hist_len = float(track_feat[1])
                track_cx, track_cy, track_w, track_h = _neutral_box_fields()
                candidate = {
                    "seq": seq,
                    "frame": int(frame),
                    "assoc_mode": "graphassoc_commit",
                    "group_id": group_id,
                    "det_index": int(det_row),
                    "track_rank": 0,
                    "track_id": int(track_id),
                    "is_selected": int(edge["label"]),
                    "det_score": float(det_score),
                    "base_score": float(edge["base_score"]),
                    "refined_score": float(edge["refined_score"]),
                    "motion_score": float(edge["motion_score"]),
                    "track_gap": float(track_gap),
                    "track_hist_len": float(track_hist_len),
                    "det_gt_id": -1,
                    "track_gt_id": -1,
                    "det_ignore": 0,
                    "track_ignore": 0,
                    "label": int(edge["label"]),
                    "valid_train_row": 1,
                    "det_cx": float(det_cx),
                    "det_cy": float(det_cy),
                    "det_w": float(det_w),
                    "det_h": float(det_h),
                    "track_cx": float(track_cx),
                    "track_cy": float(track_cy),
                    "track_w": float(track_w),
                    "track_h": float(track_h),
                    "source_csv": str(cluster_path),
                    "source_tag": source_tag,
                    "host_variant": host_variant,
                    "split_tag": split_tag,
                    "feature_version": feature_version,
                    "dataset_tag": str(args.dataset_tag),
                    "cluster_id": cluster_id,
                }
                track_candidates.append(candidate)

            if not track_candidates:
                continue

            ranked = sorted(
                track_candidates,
                key=lambda row: (-float(row["refined_score"]), int(row["track_id"])),
            )
            for rank, cand in enumerate(ranked, 1):
                cand["track_rank"] = int(rank)

            refined_scores = [float(row["refined_score"]) for row in ranked]
            base_scores = [float(row["base_score"]) for row in ranked]
            positive_ranks = [int(row["track_rank"]) for row in ranked if int(row["label"]) > 0]
            positive_rank = min(positive_ranks) if positive_ranks else -1
            rank_top1_correct = 1 if ranked and int(ranked[0]["label"]) > 0 else 0
            positive_in_topk = 1 if positive_rank > 0 and positive_rank <= int(args.topk) else 0
            rank_margin = float(refined_scores[0] - refined_scores[1]) if len(refined_scores) > 1 else float(refined_scores[0])
            base_margin = float(base_scores[0] - base_scores[1]) if len(base_scores) > 1 else float(base_scores[0])
            rank_entropy = float(_entropy_from_scores(refined_scores))
            group_has_positive = 1 if positive_ranks else 0
            group_is_background = 1 if not positive_ranks else 0
            group_is_ambiguous = 1 if (group_has_positive and (rank_top1_correct == 0 or rank_margin < 0.10)) else 0
            group_is_recoverable = 1 if (group_has_positive and rank_top1_correct == 0 and positive_in_topk == 1) else 0
            candidate_count_total = int(len(ranked))

            group_record = {
                "group_id": group_id,
                "seq": seq,
                "frame": int(frame),
                "det_index": int(det_row),
                "assoc_mode": "graphassoc_commit",
                "group_size": candidate_count_total,
                "candidate_count_total": candidate_count_total,
                "det_gt_id": -1,
                "det_ignore": 0,
                "group_has_positive": int(group_has_positive),
                "group_is_background": int(group_is_background),
                "group_is_ambiguous": int(group_is_ambiguous),
                "group_is_recoverable": int(group_is_recoverable),
                "positive_rank": int(positive_rank),
                "positive_in_topk": int(positive_in_topk),
                "rank_score_col": "refined_score",
                "rank_margin": float(rank_margin),
                "rank_entropy": float(rank_entropy),
                "rank_top1_correct": int(rank_top1_correct),
                "base_margin": float(base_margin),
                "refined_margin": float(rank_margin),
                "source_tag": source_tag,
                "host_variant": host_variant,
                "split_tag": split_tag,
                "feature_version": feature_version,
                "dataset_tag": str(args.dataset_tag),
                "source_cluster_jsonl": str(cluster_path),
                "cluster_id": cluster_id,
                "candidates": [
                    {
                        "track_rank": int(row["track_rank"]),
                        "track_id": int(row["track_id"]),
                        "label": int(row["label"]),
                        "valid_train_row": int(row["valid_train_row"]),
                        "base_score": float(row["base_score"]),
                        "refined_score": float(row["refined_score"]),
                        "motion_score": float(row["motion_score"]),
                        "track_gap": float(row["track_gap"]),
                        "track_hist_len": float(row["track_hist_len"]),
                        "det_score": float(row["det_score"]),
                        "det_cx": float(row["det_cx"]),
                        "det_cy": float(row["det_cy"]),
                        "det_w": float(row["det_w"]),
                        "det_h": float(row["det_h"]),
                        "track_cx": float(row["track_cx"]),
                        "track_cy": float(row["track_cy"]),
                        "track_w": float(row["track_w"]),
                        "track_h": float(row["track_h"]),
                    }
                    for row in ranked
                ],
            }
            groups_out.append(group_record)

            for row in ranked:
                row_out = {
                    "seq": row["seq"],
                    "frame": int(row["frame"]),
                    "assoc_mode": str(row["assoc_mode"]),
                    "group_id": str(row["group_id"]),
                    "det_index": int(row["det_index"]),
                    "track_rank": int(row["track_rank"]),
                    "track_id": int(row["track_id"]),
                    "is_selected": int(row["is_selected"]),
                    "det_score": float(row["det_score"]),
                    "base_score": float(row["base_score"]),
                    "refined_score": float(row["refined_score"]),
                    "motion_score": float(row["motion_score"]),
                    "track_gap": float(row["track_gap"]),
                    "track_hist_len": float(row["track_hist_len"]),
                    "det_gt_id": int(row["det_gt_id"]),
                    "track_gt_id": int(row["track_gt_id"]),
                    "det_ignore": int(row["det_ignore"]),
                    "track_ignore": int(row["track_ignore"]),
                    "label": int(row["label"]),
                    "valid_train_row": int(row["valid_train_row"]),
                    "group_has_positive": int(group_has_positive),
                    "group_size": int(candidate_count_total),
                    "candidate_count_total": int(candidate_count_total),
                    "base_margin": float(base_margin),
                    "refined_margin": float(rank_margin),
                    "rank_score_col": "refined_score",
                    "rank_margin": float(rank_margin),
                    "rank_entropy": float(rank_entropy),
                    "rank_top1_correct": int(rank_top1_correct),
                    "positive_in_topk": int(positive_in_topk),
                    "positive_rank": int(positive_rank),
                    "group_is_ambiguous": int(group_is_ambiguous),
                    "group_is_background": int(group_is_background),
                    "group_is_recoverable": int(group_is_recoverable),
                    "det_cx": float(row["det_cx"]),
                    "det_cy": float(row["det_cy"]),
                    "det_w": float(row["det_w"]),
                    "det_h": float(row["det_h"]),
                    "track_cx": float(row["track_cx"]),
                    "track_cy": float(row["track_cy"]),
                    "track_w": float(row["track_w"]),
                    "track_h": float(row["track_h"]),
                    "source_csv": str(cluster_path),
                    "source_tag": source_tag,
                    "host_variant": host_variant,
                    "split_tag": split_tag,
                    "feature_version": feature_version,
                    "dataset_tag": str(args.dataset_tag),
                    "cluster_id": cluster_id,
                }
                rows_out.append(row_out)

            summary["groups"] += 1
            summary["rows"] += int(candidate_count_total)
            summary["positive_rows"] += int(sum(int(row["label"]) for row in ranked))
            summary["positive_groups"] += int(group_has_positive)
            summary["background_groups"] += int(group_is_background)
            summary["ambiguous_groups"] += int(group_is_ambiguous)
            summary["recoverable_groups"] += int(group_is_recoverable)
            summary["positive_in_topk_groups"] += int(positive_in_topk)
            summary["rank_top1_correct_groups"] += int(rank_top1_correct)
            split_counter[split_tag]["groups"] += 1
            split_counter[split_tag]["rows"] += int(candidate_count_total)
            split_counter[split_tag]["positive_rows"] += int(sum(int(row["label"]) for row in ranked))
            split_counter[split_tag]["positive_groups"] += int(group_has_positive)
            split_counter[split_tag]["background_groups"] += int(group_is_background)
            split_counter[split_tag]["ambiguous_groups"] += int(group_is_ambiguous)
            split_counter[split_tag]["recoverable_groups"] += int(group_is_recoverable)
            split_counter[split_tag]["positive_in_topk_groups"] += int(positive_in_topk)
            split_counter[split_tag]["rank_top1_correct_groups"] += int(rank_top1_correct)
            seq_counter[(seq, split_tag)]["groups"] += 1
            seq_counter[(seq, split_tag)]["rows"] += int(candidate_count_total)
            seq_counter[(seq, split_tag)]["positive_rows"] += int(sum(int(row["label"]) for row in ranked))
            seq_counter[(seq, split_tag)]["positive_groups"] += int(group_has_positive)
            seq_counter[(seq, split_tag)]["background_groups"] += int(group_is_background)
            seq_counter[(seq, split_tag)]["ambiguous_groups"] += int(group_is_ambiguous)
            seq_counter[(seq, split_tag)]["recoverable_groups"] += int(group_is_recoverable)
            seq_counter[(seq, split_tag)]["positive_in_topk_groups"] += int(positive_in_topk)
            seq_counter[(seq, split_tag)]["rank_top1_correct_groups"] += int(rank_top1_correct)

    rows_csv = out_dir / "rows.csv"
    group_jsonl = out_dir / "groups.jsonl"
    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"

    _write_csv(rows_csv, rows_out)
    with group_jsonl.open("w", encoding="utf-8") as f:
        for group in groups_out:
            f.write(json.dumps(group, ensure_ascii=False) + "\n")

    summary_row = {
        "dataset_tag": str(args.dataset_tag),
        "feature_version": str(args.feature_version),
        "source_cluster_jsonl": str(cluster_path),
        "rows_csv": str(rows_csv),
        "group_jsonl": str(group_jsonl),
        "clusters": int(summary.get("clusters", 0)),
        "groups": int(summary.get("groups", 0)),
        "rows": int(summary.get("rows", 0)),
        "positive_rows": int(summary.get("positive_rows", 0)),
        "positive_groups": int(summary.get("positive_groups", 0)),
        "background_groups": int(summary.get("background_groups", 0)),
        "ambiguous_groups": int(summary.get("ambiguous_groups", 0)),
        "recoverable_groups": int(summary.get("recoverable_groups", 0)),
        "positive_in_topk_groups": int(summary.get("positive_in_topk_groups", 0)),
        "rank_top1_correct_groups": int(summary.get("rank_top1_correct_groups", 0)),
        "unused_examples": int(summary.get("unused_examples", 0)),
        "train_sequences": ",".join(sorted(seq for (seq, split_tag) in seq_counter.keys() if split_tag == "train")),
        "val_sequences": ",".join(sorted(seq for (seq, split_tag) in seq_counter.keys() if split_tag == "val")),
        "all_sequences": ",".join(sorted(seen_sequences)),
        "all_sources": ",".join(sorted(seen_sources)),
        "all_host_variants": ",".join(sorted(seen_host_variants)),
        "status": "ok",
        "error": "",
    }
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)
    summary_json.write_text(json.dumps(summary_row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary_row, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
