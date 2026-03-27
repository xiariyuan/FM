#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.local_conflict_graph_common import (
    build_group_components_from_group_rows,
    compute_component_degree_features,
    filter_local_conflict_clusters_by_size,
    mine_centered_subcomponents_from_group_rows,
    solve_assignment_with_private_defer,
)
from models.local_conflict_set_predictor import (
    CLUSTER_FEATURE_NAMES,
    DET_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
    FEATURE_VERSION,
    TRACK_FEATURE_NAMES,
    entropy_from_probs,
    pair_geometry_features,
    softmax_probs_1d,
    zscore_1d,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cluster-level local-conflict set-predictor training data.")
    parser.add_argument("--rows-csv", default="")
    parser.add_argument("--group-jsonl", default="")
    parser.add_argument("--source-manifest", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--min-detections", type=int, default=2)
    parser.add_argument("--min-committed-matches", type=int, default=2)
    parser.add_argument("--max-detections", type=int, default=8)
    parser.add_argument("--max-tracks", type=int, default=32)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--train-sequences", default="")
    parser.add_argument("--val-sequences", default="")
    parser.add_argument("--strict-sequence-split", action="store_true")
    parser.add_argument("--feature-version", default=FEATURE_VERSION)
    parser.add_argument("--dataset-tag", default="local_conflict_set_predictor")
    parser.add_argument(
        "--teacher-mode",
        choices=[
            "oracle_commit",
            "delta_utility",
            "edit_utility",
            "rescue_utility",
            "sparse_edit",
            "bridge_commit_contract_utility",
        ],
        default="bridge_commit_contract_utility",
    )
    parser.add_argument("--edit-utility-commit-cost", type=float, default=0.20)
    parser.add_argument("--edit-utility-min-gain", type=float, default=0.75)
    parser.add_argument("--edit-utility-force-defer-gain", type=float, default=0.50)
    parser.add_argument("--runtime-host-match-thresh", type=float, default=0.90)
    parser.add_argument("--soft-rescue-weight", type=float, default=0.75)
    parser.add_argument("--rescue-force-defer-gain", type=float, default=0.50)
    parser.add_argument("--rescue-min-gain", type=float, default=0.50)
    parser.add_argument("--bridge-crowded-row-degree-thresh", type=int, default=4)
    parser.add_argument("--bridge-crowded-bonus", type=float, default=0.25)
    parser.add_argument("--bridge-large-component-bonus", type=float, default=0.50)
    parser.add_argument("--bridge-commit-cost", type=float, default=0.20)
    parser.add_argument("--bridge-min-gain", type=float, default=0.75)
    parser.add_argument(
        "--large-component-max-subclusters",
        type=int,
        default=0,
        help="If >0, mine up to this many suspicious-centered subclusters from each oversized component.",
    )
    return parser.parse_args()


def _write_single_row_csv(path: Path, row: dict[str, Any]) -> None:
    fieldnames = list(row.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


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


def _load_groups_by_frame(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    groups_by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            group = json.loads(line)
            seq = str(group.get("seq", ""))
            frame = _safe_int(group.get("frame", 0))
            groups_by_frame[(seq, frame)].append(group)
    return groups_by_frame


def _load_rows_by_group(path: Path) -> dict[str, list[dict[str, str]]]:
    rows_by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group_id = str(row.get("group_id", "")).strip()
            if not group_id:
                continue
            rows_by_group[group_id].append(dict(row))
    for group_id, rows in rows_by_group.items():
        rows.sort(key=lambda row: _safe_int(row.get("track_rank", 0), 0))
    return rows_by_group


def _load_sources(args: argparse.Namespace) -> tuple[list[dict[str, str]], str]:
    if str(args.source_manifest or "").strip():
        manifest_path = Path(args.source_manifest).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = [dict(row) for row in reader]
        if not rows:
            raise ValueError(f"Source manifest is empty: {manifest_path}")
        sources: list[dict[str, str]] = []
        for idx, row in enumerate(rows):
            rows_csv_raw = str(row.get("rows_csv", "")).strip()
            group_jsonl_raw = str(row.get("group_jsonl", "")).strip()
            if not rows_csv_raw or not group_jsonl_raw:
                raise ValueError(f"Manifest row {idx} is missing rows_csv/group_jsonl")
            rows_csv = Path(rows_csv_raw)
            if not rows_csv.is_absolute():
                rows_csv = (manifest_path.parent / rows_csv).resolve()
            group_jsonl = Path(group_jsonl_raw)
            if not group_jsonl.is_absolute():
                group_jsonl = (manifest_path.parent / group_jsonl).resolve()
            source_tag = str(row.get("source_tag", "")).strip() or f"source_{idx:03d}"
            sources.append(
                {
                    "rows_csv": str(rows_csv),
                    "group_jsonl": str(group_jsonl),
                    "host_variant": str(row.get("host_variant", "")).strip() or "unknown",
                    "source_tag": source_tag,
                    "split_tag": str(row.get("split_tag", "")).strip(),
                    "dataset_tag": str(row.get("dataset_tag", "")).strip() or str(args.dataset_tag),
                    "feature_version": str(row.get("feature_version", "")).strip() or str(args.feature_version),
                }
            )
        return sources, str(manifest_path)

    rows_csv = Path(str(args.rows_csv or "")).resolve()
    group_jsonl = Path(str(args.group_jsonl or "")).resolve()
    if not str(args.rows_csv or "").strip() or not str(args.group_jsonl or "").strip():
        raise ValueError("Use either --source-manifest or both --rows-csv and --group-jsonl")
    return (
        [
            {
                "rows_csv": str(rows_csv),
                "group_jsonl": str(group_jsonl),
                "host_variant": "unknown",
                "source_tag": "source_000",
                "split_tag": "",
                "dataset_tag": str(args.dataset_tag),
                "feature_version": str(args.feature_version),
            }
        ],
        "",
    )


def _candidate_rows(rows_by_group: dict[str, list[dict[str, str]]], group_id: str, topk: int) -> list[dict[str, str]]:
    rows = list(rows_by_group.get(group_id, []))
    rows.sort(key=lambda row: _safe_int(row.get("track_rank", 0), 0))
    valid_rows = [row for row in rows if _safe_int(row.get("valid_train_row", 1), 1) > 0]
    return valid_rows[: max(int(topk), 1)]


def _assign_pairs_to_target(
    *,
    num_detections: int,
    pairs: set[tuple[int, int]],
) -> list[int]:
    target = [-1 for _ in range(int(num_detections))]
    for det_local_idx, track_local_idx in pairs:
        if 0 <= int(det_local_idx) < int(num_detections):
            target[int(det_local_idx)] = int(track_local_idx)
    return target


def _sorted_pair_list(pairs: set[tuple[int, int]]) -> list[list[int]]:
    return [[int(det_local_idx), int(track_local_idx)] for det_local_idx, track_local_idx in sorted(pairs)]


def _pairs_to_action_by_det(
    *,
    num_detections: int,
    pairs: set[tuple[int, int]],
) -> list[int]:
    actions = [-1 for _ in range(int(num_detections))]
    for det_local_idx, track_local_idx in pairs:
        if 0 <= int(det_local_idx) < int(num_detections):
            actions[int(det_local_idx)] = int(track_local_idx)
    return actions


def _row_action_type(*, host_action: int, oracle_action: int) -> str:
    if int(host_action) == int(oracle_action):
        return "keep"
    if int(host_action) < 0 and int(oracle_action) >= 0:
        return "add_commit"
    if int(host_action) >= 0 and int(oracle_action) >= 0:
        return "reassign_commit"
    if int(host_action) >= 0 and int(oracle_action) < 0:
        return "force_defer"
    return "keep"


def _row_stats(candidate_rows: list[dict[str, str]]) -> dict[str, Any]:
    refined = torch.tensor(
        [_safe_float(row.get("refined_score", 0.0), 0.0) for row in candidate_rows],
        dtype=torch.float32,
    )
    base = torch.tensor(
        [_safe_float(row.get("base_score", 0.0), 0.0) for row in candidate_rows],
        dtype=torch.float32,
    )
    motion = torch.tensor(
        [_safe_float(row.get("motion_score", 0.0), 0.0) for row in candidate_rows],
        dtype=torch.float32,
    )
    refined_probs = softmax_probs_1d(refined)
    margin = 0.0
    if refined.numel() > 1:
        top2 = torch.topk(refined, k=min(2, refined.numel()), dim=0, sorted=True).values
        margin = float((top2[0] - top2[1]).item())
    return {
        "base_z": zscore_1d(base),
        "refined_z": zscore_1d(refined),
        "motion_z": zscore_1d(motion),
        "refined_softmax": refined_probs,
        "refined_top1": float(refined.max().item()) if refined.numel() > 0 else 0.0,
        "row_top1_minus_top2": float(margin),
        "row_entropy": float(entropy_from_probs(refined_probs).item()) if refined_probs.numel() > 0 else 0.0,
        "candidate_count": int(len(candidate_rows)),
    }


def _box_feature_values(row: dict[str, str], prefix: str) -> tuple[float, float, float, float]:
    cx = _safe_float(row.get(f"{prefix}_cx", 0.0), 0.0)
    cy = _safe_float(row.get(f"{prefix}_cy", 0.0), 0.0)
    w = max(_safe_float(row.get(f"{prefix}_w", 0.0), 0.0), 1e-6)
    h = max(_safe_float(row.get(f"{prefix}_h", 0.0), 0.0), 1e-6)
    return cx, cy, w, h


def _solve_runtime_aligned_host_pairs(
    *,
    score_sub: torch.Tensor,
    match_thresh: float,
) -> set[tuple[int, int]]:
    if score_sub.numel() == 0:
        return set()
    min_score = max(1.0 - float(match_thresh), 0.0)
    feasible_mask = score_sub >= float(min_score)
    if not bool(feasible_mask.any().item()):
        return set()
    assignments = solve_assignment_with_private_defer(
        score_sub=score_sub,
        feasible_mask=feasible_mask,
        defer_scores=torch.zeros((int(score_sub.shape[0]),), dtype=score_sub.dtype),
        use_hungarian=True,
    )
    return {
        (int(assignment["det_local_idx"]), int(assignment["track_local_idx"]))
        for assignment in assignments
        if assignment.get("track_local_idx", None) is not None
    }


def _select_bridge_commit_pair(
    *,
    candidate_pairs: set[tuple[int, int]],
    row_sparse_action_type: list[str],
    row_degree: list[float],
    is_large_component: bool,
    soft_rescue_weight: float,
    bridge_crowded_row_degree_thresh: int,
    bridge_crowded_bonus: float,
    bridge_large_component_bonus: float,
    bridge_commit_cost: float,
    bridge_min_gain: float,
) -> tuple[set[tuple[int, int]], list[int], list[dict[str, float]], float, int]:
    candidate_scores: list[tuple[float, tuple[int, int], str]] = []
    utility_rows: list[dict[str, float]] = []
    for det_local_idx, track_local_idx in sorted(candidate_pairs):
        action_type = (
            str(row_sparse_action_type[int(det_local_idx)])
            if 0 <= int(det_local_idx) < len(row_sparse_action_type)
            else "keep"
        )
        base_gain = 1.0 if action_type == "add_commit" else float(soft_rescue_weight)
        crowded_bonus = (
            float(bridge_crowded_bonus)
            if 0 <= int(det_local_idx) < len(row_degree)
            and float(row_degree[int(det_local_idx)]) >= float(bridge_crowded_row_degree_thresh)
            else 0.0
        )
        large_component_bonus = float(bridge_large_component_bonus) if bool(is_large_component) else 0.0
        utility = float(base_gain + crowded_bonus + large_component_bonus - float(bridge_commit_cost))
        utility_rows.append(
            {
                "det_local_idx": int(det_local_idx),
                "track_local_idx": int(track_local_idx),
                "base_gain": float(base_gain),
                "crowded_bonus": float(crowded_bonus),
                "large_component_bonus": float(large_component_bonus),
                "commit_cost": float(bridge_commit_cost),
                "utility": float(utility),
            }
        )
        candidate_scores.append((float(utility), (int(det_local_idx), int(track_local_idx)), action_type))

    if not candidate_scores:
        return set(), [0 for _ in row_sparse_action_type], utility_rows, 0.0, 0

    candidate_scores.sort(
        key=lambda item: (
            float(item[0]),
            1 if str(item[2]) == "add_commit" else 0,
            -int(item[1][0]),
            -int(item[1][1]),
        ),
        reverse=True,
    )
    best_utility, best_pair, _ = candidate_scores[0]
    if float(best_utility) < float(bridge_min_gain):
        return set(), [0 for _ in row_sparse_action_type], utility_rows, float(best_utility), len(candidate_scores)
    row_bridge_mask = [0 for _ in row_sparse_action_type]
    if 0 <= int(best_pair[0]) < len(row_bridge_mask):
        row_bridge_mask[int(best_pair[0])] = 1
    return {best_pair}, row_bridge_mask, utility_rows, float(best_utility), len(candidate_scores)


def _cluster_example(
    *,
    seq: str,
    frame: int,
    component: dict[str, Any],
    frame_groups: list[dict[str, Any]],
    rows_by_group: dict[str, list[dict[str, str]]],
    topk: int,
    min_committed_matches: int,
    teacher_mode: str,
    source_tag: str,
    host_variant: str,
    split_tag: str,
    feature_version: str,
    source_rows_csv: str,
    source_group_jsonl: str,
    edit_utility_commit_cost: float,
    edit_utility_min_gain: float,
    edit_utility_force_defer_gain: float,
    runtime_host_match_thresh: float,
    soft_rescue_weight: float,
    rescue_force_defer_gain: float,
    rescue_min_gain: float,
    bridge_crowded_row_degree_thresh: int,
    bridge_crowded_bonus: float,
    bridge_large_component_bonus: float,
    bridge_commit_cost: float,
    bridge_min_gain: float,
) -> dict[str, Any] | None:
    group_by_det = {_safe_int(group.get("det_index", -1), -1): group for group in frame_groups}
    det_rows = [int(x) for x in component.get("det_rows", [])]
    track_ids = [int(x) for x in component.get("track_ids", [])]
    if not det_rows or not track_ids:
        return None

    track_to_local = {track_id: idx for idx, track_id in enumerate(track_ids)}
    det_features: list[list[float]] = []
    det_feature_meta: list[dict[str, float]] = []
    edge_records: list[dict[str, Any]] = []
    col_refined_scores: dict[int, list[float]] = defaultdict(list)
    track_gap_acc: dict[int, list[float]] = defaultdict(list)
    track_hist_acc: dict[int, list[float]] = defaultdict(list)
    track_box_acc: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    row_entropy_values: list[float] = []
    row_margin_values: list[float] = []

    for local_det_idx, det_row in enumerate(det_rows):
        group = group_by_det.get(det_row)
        if group is None:
            return None
        group_id = str(group.get("group_id", ""))
        candidates = _candidate_rows(rows_by_group, group_id, topk)
        if not candidates:
            return None
        stats = _row_stats(candidates)
        first_row = candidates[0]
        det_cx, det_cy, det_w, det_h = _box_feature_values(first_row, "det")
        det_log_w = math.log(max(det_w, 1e-6))
        det_log_h = math.log(max(det_h, 1e-6))
        det_aspect = det_w / max(det_h, 1e-6)
        row_entropy_values.append(float(stats["row_entropy"]))
        row_margin_values.append(float(stats["row_top1_minus_top2"]))
        det_features.append(
            [
                _safe_float(first_row.get("det_score", 0.0), 0.0),
                0.0,
                float(stats["row_top1_minus_top2"]),
                float(stats["row_entropy"]),
                float(det_cx),
                float(det_cy),
                float(det_log_w),
                float(det_log_h),
                float(det_aspect),
            ]
        )
        det_feature_meta.append(
            {
                "det_cx": float(det_cx),
                "det_cy": float(det_cy),
                "det_w": float(det_w),
                "det_h": float(det_h),
            }
        )

        for row_idx, candidate in enumerate(candidates):
            track_id = _safe_int(candidate.get("track_id", -1), -1)
            if track_id not in track_to_local:
                continue
            local_track_idx = int(track_to_local[track_id])
            track_gap_raw = _safe_float(candidate.get("track_gap", 0.0), 0.0)
            track_hist_raw = _safe_float(candidate.get("track_hist_len", 0.0), 0.0)
            track_gap_acc[local_track_idx].append(float(track_gap_raw))
            track_hist_acc[local_track_idx].append(float(track_hist_raw))
            track_box = _box_feature_values(candidate, "track")
            track_box_acc[local_track_idx].append(track_box)
            det_box_tensor = torch.tensor([[det_cx, det_cy, det_w, det_h]], dtype=torch.float32)
            track_box_tensor = torch.tensor([[track_box[0], track_box[1], track_box[2], track_box[3]]], dtype=torch.float32)
            geom = pair_geometry_features(det_box_tensor, track_box_tensor)
            refined_score = _safe_float(candidate.get("refined_score", 0.0), 0.0)
            col_refined_scores[local_track_idx].append(float(refined_score))
            edge_records.append(
                {
                    "det_local_idx": int(local_det_idx),
                    "track_local_idx": int(local_track_idx),
                    "track_id": int(track_id),
                    "base_score_raw": _safe_float(candidate.get("base_score", 0.0), 0.0),
                    "refined_score_raw": float(refined_score),
                    "motion_score_raw": _safe_float(candidate.get("motion_score", 0.0), 0.0),
                    "base_score_row_z": float(stats["base_z"][row_idx].item()),
                    "refined_score_row_z": float(stats["refined_z"][row_idx].item()),
                    "motion_score_row_z": float(stats["motion_z"][row_idx].item()),
                    "refined_score_row_softmax": float(stats["refined_softmax"][row_idx].item()),
                    "refined_gap_to_row_top1": float(stats["refined_top1"] - refined_score),
                    "rank_frac": float(row_idx + 1) / float(max(int(stats["candidate_count"]), 1)),
                    "iou": float(geom["iou"].view(-1)[0].item()),
                    "bbox_dist_score": float(geom["bbox_dist_score"].view(-1)[0].item()),
                    "delta_cx_norm": float(geom["delta_cx_norm"].view(-1)[0].item()),
                    "delta_cy_norm": float(geom["delta_cy_norm"].view(-1)[0].item()),
                    "delta_log_w": float(geom["delta_log_w"].view(-1)[0].item()),
                    "delta_log_h": float(geom["delta_log_h"].view(-1)[0].item()),
                    "edge_is_gt_positive": int(_safe_int(candidate.get("label", 0), 0) == 1),
                }
            )

    if not edge_records:
        return None

    edge_det_index = [int(record["det_local_idx"]) for record in edge_records]
    edge_track_index = [int(record["track_local_idx"]) for record in edge_records]
    degree = compute_component_degree_features(
        num_detections=len(det_rows),
        num_tracks=len(track_ids),
        edge_det_index=edge_det_index,
        edge_track_index=edge_track_index,
    )
    row_degree = degree["row_degree"].tolist()
    col_degree = degree["col_degree"].tolist()
    for det_idx, feat in enumerate(det_features):
        feat[1] = float(row_degree[det_idx]) if det_idx < len(row_degree) else 0.0

    col_z_by_track: dict[int, dict[float, float]] = {}
    for local_track_idx, refined_scores in col_refined_scores.items():
        score_tensor = torch.tensor(refined_scores, dtype=torch.float32)
        col_z = zscore_1d(score_tensor).tolist()
        col_z_by_track[local_track_idx] = {
            idx: float(col_z[pos]) for pos, idx in enumerate(range(len(refined_scores)))
        }

    col_offset: dict[int, int] = defaultdict(int)
    edge_features: list[list[float]] = []
    edge_is_gt_positive: list[int] = []
    for record in edge_records:
        local_track_idx = int(record["track_local_idx"])
        local_pos = int(col_offset[local_track_idx])
        col_offset[local_track_idx] += 1
        record["refined_score_col_z"] = float(col_z_by_track.get(local_track_idx, {}).get(local_pos, 0.0))
        record["refined_minus_base"] = float(record["refined_score_raw"] - record["base_score_raw"])
        record["motion_minus_refined"] = float(record["motion_score_raw"] - record["refined_score_raw"])
        edge_features.append(
            [
                float(record["base_score_raw"]),
                float(record["refined_score_raw"]),
                float(record["motion_score_raw"]),
                float(record["base_score_row_z"]),
                float(record["refined_score_row_z"]),
                float(record["motion_score_row_z"]),
                float(record["refined_score_row_softmax"]),
                float(record["refined_gap_to_row_top1"]),
                float(record["rank_frac"]),
                float(record["refined_score_col_z"]),
                float(record["refined_minus_base"]),
                float(record["motion_minus_refined"]),
                float(record["iou"]),
                float(record["bbox_dist_score"]),
                float(record["delta_cx_norm"]),
                float(record["delta_cy_norm"]),
                float(record["delta_log_w"]),
                float(record["delta_log_h"]),
            ]
        )
        edge_is_gt_positive.append(int(record["edge_is_gt_positive"]))

    track_features: list[list[float]] = []
    for local_track_idx in range(len(track_ids)):
        gap_values = track_gap_acc.get(local_track_idx, [])
        hist_values = track_hist_acc.get(local_track_idx, [])
        track_boxes = track_box_acc.get(local_track_idx, [])
        gap_mean = float(sum(gap_values) / len(gap_values)) if gap_values else 0.0
        hist_mean = float(sum(hist_values) / len(hist_values)) if hist_values else 0.0
        if track_boxes:
            track_cx = float(sum(box[0] for box in track_boxes) / len(track_boxes))
            track_cy = float(sum(box[1] for box in track_boxes) / len(track_boxes))
            track_w = float(sum(box[2] for box in track_boxes) / len(track_boxes))
            track_h = float(sum(box[3] for box in track_boxes) / len(track_boxes))
        else:
            track_cx = track_cy = 0.0
            track_w = track_h = 1e-6
        track_features.append(
            [
                float(math.log1p(max(gap_mean, 0.0))),
                float(math.log1p(max(hist_mean, 0.0))),
                float(col_degree[local_track_idx]) if local_track_idx < len(col_degree) else 0.0,
                float(track_cx),
                float(track_cy),
                float(math.log(max(track_w, 1e-6))),
                float(math.log(max(track_h, 1e-6))),
                float(track_w / max(track_h, 1e-6)),
            ]
        )

    row_degree_tensor = torch.tensor(row_degree, dtype=torch.float32)
    col_degree_tensor = torch.tensor(col_degree, dtype=torch.float32)
    row_entropy_tensor = torch.tensor(row_entropy_values, dtype=torch.float32)
    row_margin_tensor = torch.tensor(row_margin_values, dtype=torch.float32)
    cluster_features = [
        float(len(det_rows)),
        float(len(track_ids)),
        float(len(edge_records)),
        float(row_degree_tensor.mean().item()) if row_degree_tensor.numel() > 0 else 0.0,
        float(row_degree_tensor.max().item()) if row_degree_tensor.numel() > 0 else 0.0,
        float(col_degree_tensor.mean().item()) if col_degree_tensor.numel() > 0 else 0.0,
        float(col_degree_tensor.max().item()) if col_degree_tensor.numel() > 0 else 0.0,
        float(row_entropy_tensor.mean().item()) if row_entropy_tensor.numel() > 0 else 0.0,
        float(row_entropy_tensor.max().item()) if row_entropy_tensor.numel() > 0 else 0.0,
        float(row_margin_tensor.mean().item()) if row_margin_tensor.numel() > 0 else 0.0,
        float(row_margin_tensor.max().item()) if row_margin_tensor.numel() > 0 else 0.0,
    ]

    positive_score_sub = torch.zeros((len(det_rows), len(track_ids)), dtype=torch.float32)
    positive_mask = torch.zeros_like(positive_score_sub, dtype=torch.bool)
    host_score_sub = torch.zeros_like(positive_score_sub)
    host_mask = torch.zeros_like(positive_mask)
    for record in edge_records:
        det_local_idx = int(record["det_local_idx"])
        track_local_idx = int(record["track_local_idx"])
        refined_score = float(record["refined_score_raw"])
        host_mask[det_local_idx, track_local_idx] = True
        host_score_sub[det_local_idx, track_local_idx] = refined_score
        if int(record["edge_is_gt_positive"]) <= 0:
            continue
        positive_mask[det_local_idx, track_local_idx] = True
        positive_score_sub[det_local_idx, track_local_idx] = refined_score

    host_assignments = solve_assignment_with_private_defer(
        score_sub=host_score_sub,
        feasible_mask=host_mask,
        defer_scores=torch.zeros((len(det_rows),), dtype=torch.float32),
        use_hungarian=True,
    )
    host_pairs_local = {
        (int(assignment["det_local_idx"]), int(assignment["track_local_idx"]))
        for assignment in host_assignments
        if assignment.get("track_local_idx", None) is not None
    }
    host_pairs_local_runtime = _solve_runtime_aligned_host_pairs(
        score_sub=host_score_sub,
        match_thresh=float(runtime_host_match_thresh),
    )

    assignments = solve_assignment_with_private_defer(
        score_sub=positive_score_sub,
        feasible_mask=positive_mask,
        defer_scores=torch.zeros((len(det_rows),), dtype=torch.float32),
        use_hungarian=True,
    )
    matched_pairs = {
        (int(assignment["det_local_idx"]), int(assignment["track_local_idx"]))
        for assignment in assignments
        if assignment.get("track_local_idx", None) is not None
    }
    matched_count = len(matched_pairs)
    oracle_trigger_pass = int(matched_count >= int(min_committed_matches))
    oracle_commit_pairs = matched_pairs if oracle_trigger_pass else set()
    delta_commit_pairs = set(oracle_commit_pairs) - set(host_pairs_local)
    cluster_should_intervene = int(len(delta_commit_pairs) > 0)
    cluster_utility_gain = float(len(delta_commit_pairs))
    host_equals_oracle = int(host_pairs_local == oracle_commit_pairs)
    host_runtime_equals_oracle = int(host_pairs_local_runtime == oracle_commit_pairs)

    host_action_by_det = _pairs_to_action_by_det(
        num_detections=len(det_rows),
        pairs=set(host_pairs_local),
    )
    oracle_action_by_det = _pairs_to_action_by_det(
        num_detections=len(det_rows),
        pairs=set(oracle_commit_pairs),
    )
    row_edit_mask: list[int] = []
    row_action_type: list[str] = []
    edit_commit_pairs: set[tuple[int, int]] = set()
    cluster_edit_gain = 0.0
    num_add_commit_rows = 0
    num_reassign_rows = 0
    num_force_defer_rows = 0
    for det_local_idx, (host_action, oracle_action) in enumerate(zip(host_action_by_det, oracle_action_by_det)):
        action_type = _row_action_type(host_action=int(host_action), oracle_action=int(oracle_action))
        row_action_type.append(action_type)
        is_edit = int(action_type != "keep")
        row_edit_mask.append(is_edit)
        if action_type == "add_commit":
            cluster_edit_gain += 1.0
            num_add_commit_rows += 1
        elif action_type == "reassign_commit":
            cluster_edit_gain += 1.0
            num_reassign_rows += 1
        elif action_type == "force_defer":
            cluster_edit_gain += float(edit_utility_force_defer_gain)
            num_force_defer_rows += 1
        if is_edit and int(oracle_action) >= 0:
            edit_commit_pairs.add((int(det_local_idx), int(oracle_action)))

    cluster_edit_cost = float(edit_utility_commit_cost) * float(len(edit_commit_pairs))
    cluster_edit_utility_gain = float(cluster_edit_gain - cluster_edit_cost)
    cluster_should_intervene_edit = int(
        int(sum(row_edit_mask)) > 0 and float(cluster_edit_utility_gain) >= float(edit_utility_min_gain)
    )

    host_action_by_det_runtime = _pairs_to_action_by_det(
        num_detections=len(det_rows),
        pairs=set(host_pairs_local_runtime),
    )
    row_rescue_mask: list[int] = []
    row_rescue_action_type: list[str] = []
    hard_delta_pairs: set[tuple[int, int]] = set()
    soft_rescue_pairs: set[tuple[int, int]] = set()
    rescue_commit_pairs: set[tuple[int, int]] = set()
    cluster_soft_utility_gain = 0.0
    num_hard_delta_rows = 0
    num_soft_rescue_rows = 0
    num_rescue_force_defer_rows = 0
    for det_local_idx, (host_action_runtime, oracle_action) in enumerate(zip(host_action_by_det_runtime, oracle_action_by_det)):
        action_type = _row_action_type(host_action=int(host_action_runtime), oracle_action=int(oracle_action))
        row_rescue_action_type.append(action_type)
        is_rescue = int(action_type in {"add_commit", "reassign_commit"})
        row_rescue_mask.append(is_rescue)
        if action_type == "add_commit":
            pair = (int(det_local_idx), int(oracle_action))
            hard_delta_pairs.add(pair)
            rescue_commit_pairs.add(pair)
            cluster_soft_utility_gain += 1.0
            num_hard_delta_rows += 1
        elif action_type == "reassign_commit":
            pair = (int(det_local_idx), int(oracle_action))
            soft_rescue_pairs.add(pair)
            rescue_commit_pairs.add(pair)
            cluster_soft_utility_gain += float(soft_rescue_weight)
            num_soft_rescue_rows += 1
        elif action_type == "force_defer":
            num_rescue_force_defer_rows += 1
    cluster_should_intervene_soft = int(
        int(len(rescue_commit_pairs)) > 0 and float(cluster_soft_utility_gain) >= float(rescue_min_gain)
    )

    host_runtime_matched_tracks = {
        int(track_local_idx)
        for _, track_local_idx in host_pairs_local_runtime
    }
    row_sparse_edit_mask: list[int] = []
    row_sparse_action_type: list[str] = []
    sparse_edit_pairs: set[tuple[int, int]] = set()
    cluster_sparse_utility_gain = 0.0
    num_sparse_add_rows = 0
    num_sparse_reassign_free_rows = 0
    num_sparse_blocked_reassign_rows = 0
    for det_local_idx, (host_action_runtime, oracle_action) in enumerate(zip(host_action_by_det_runtime, oracle_action_by_det)):
        action_type = _row_action_type(host_action=int(host_action_runtime), oracle_action=int(oracle_action))
        sparse_action = "keep"
        if action_type == "add_commit":
            pair = (int(det_local_idx), int(oracle_action))
            sparse_edit_pairs.add(pair)
            row_sparse_edit_mask.append(1)
            sparse_action = "add_commit"
            cluster_sparse_utility_gain += 1.0
            num_sparse_add_rows += 1
        elif action_type == "reassign_commit":
            oracle_track_local_idx = int(oracle_action)
            if oracle_track_local_idx not in host_runtime_matched_tracks:
                pair = (int(det_local_idx), oracle_track_local_idx)
                sparse_edit_pairs.add(pair)
                row_sparse_edit_mask.append(1)
                sparse_action = "reassign_free_track"
                cluster_sparse_utility_gain += float(soft_rescue_weight)
                num_sparse_reassign_free_rows += 1
            else:
                row_sparse_edit_mask.append(0)
                sparse_action = "blocked_reassign_occupied_track"
                num_sparse_blocked_reassign_rows += 1
        else:
            row_sparse_edit_mask.append(0)
            sparse_action = action_type
        row_sparse_action_type.append(sparse_action)
    cluster_should_intervene_sparse = int(
        int(len(sparse_edit_pairs)) > 0 and float(cluster_sparse_utility_gain) >= float(rescue_min_gain)
    )
    is_large_component = int(
        bool(component.get("mined_from_large_component", 0))
        or int(component.get("source_component_num_detections", len(det_rows))) > int(len(det_rows))
        or int(component.get("source_component_num_tracks", len(track_ids))) > int(len(track_ids))
    )
    bridge_commit_pairs, row_bridge_mask, bridge_candidate_rows, cluster_bridge_utility_gain, num_bridge_candidates = _select_bridge_commit_pair(
        candidate_pairs=set(sparse_edit_pairs),
        row_sparse_action_type=row_sparse_action_type,
        row_degree=[float(x) for x in row_degree],
        is_large_component=bool(is_large_component),
        soft_rescue_weight=float(soft_rescue_weight),
        bridge_crowded_row_degree_thresh=int(bridge_crowded_row_degree_thresh),
        bridge_crowded_bonus=float(bridge_crowded_bonus),
        bridge_large_component_bonus=float(bridge_large_component_bonus),
        bridge_commit_cost=float(bridge_commit_cost),
        bridge_min_gain=float(bridge_min_gain),
    )
    cluster_should_intervene_bridge = int(len(bridge_commit_pairs) > 0)

    if str(teacher_mode) == "delta_utility":
        target_pairs = set(delta_commit_pairs)
        trigger_pass = int(cluster_should_intervene)
        target_committed_matches = int(len(delta_commit_pairs))
        target_by_det = _assign_pairs_to_target(
            num_detections=len(det_rows),
            pairs=set(delta_commit_pairs),
        )
        active_cluster_should_intervene = int(cluster_should_intervene)
        active_cluster_utility_gain = float(cluster_utility_gain)
    elif str(teacher_mode) == "edit_utility":
        target_pairs = set(edit_commit_pairs)
        trigger_pass = int(oracle_trigger_pass)
        target_committed_matches = int(len(edit_commit_pairs))
        target_by_det = [int(x) for x in oracle_action_by_det]
        active_cluster_should_intervene = int(cluster_should_intervene_edit)
        active_cluster_utility_gain = float(cluster_edit_utility_gain)
    elif str(teacher_mode) == "rescue_utility":
        target_pairs = set(rescue_commit_pairs)
        trigger_pass = int(cluster_should_intervene_soft)
        target_committed_matches = int(len(rescue_commit_pairs))
        target_by_det = [int(x) for x in oracle_action_by_det]
        active_cluster_should_intervene = int(cluster_should_intervene_soft)
        active_cluster_utility_gain = float(cluster_soft_utility_gain)
    elif str(teacher_mode) == "sparse_edit":
        target_pairs = set(sparse_edit_pairs)
        trigger_pass = int(cluster_should_intervene_sparse)
        target_committed_matches = int(len(sparse_edit_pairs))
        target_by_det = _assign_pairs_to_target(
            num_detections=len(det_rows),
            pairs=set(sparse_edit_pairs),
        )
        active_cluster_should_intervene = int(cluster_should_intervene_sparse)
        active_cluster_utility_gain = float(cluster_sparse_utility_gain)
    elif str(teacher_mode) == "bridge_commit_contract_utility":
        target_pairs = set(bridge_commit_pairs)
        trigger_pass = int(cluster_should_intervene_bridge)
        target_committed_matches = int(len(bridge_commit_pairs))
        target_by_det = _assign_pairs_to_target(
            num_detections=len(det_rows),
            pairs=set(bridge_commit_pairs),
        )
        active_cluster_should_intervene = int(cluster_should_intervene_bridge)
        active_cluster_utility_gain = float(cluster_bridge_utility_gain)
    else:
        target_pairs = set(oracle_commit_pairs)
        trigger_pass = int(oracle_trigger_pass)
        target_committed_matches = int(len(oracle_commit_pairs))
        target_by_det = _assign_pairs_to_target(
            num_detections=len(det_rows),
            pairs=set(oracle_commit_pairs),
        )
        active_cluster_should_intervene = int(oracle_trigger_pass)
        active_cluster_utility_gain = float(len(oracle_commit_pairs))

    edge_is_oracle_commit = [
        int((int(record["det_local_idx"]), int(record["track_local_idx"])) in oracle_commit_pairs)
        for record in edge_records
    ]
    edge_is_delta_commit = [
        int((int(record["det_local_idx"]), int(record["track_local_idx"])) in delta_commit_pairs)
        for record in edge_records
    ]
    edge_is_edit_commit = [
        int((int(record["det_local_idx"]), int(record["track_local_idx"])) in edit_commit_pairs)
        for record in edge_records
    ]
    edge_is_soft_rescue = [
        int((int(record["det_local_idx"]), int(record["track_local_idx"])) in rescue_commit_pairs)
        for record in edge_records
    ]
    edge_is_sparse_edit = [
        int((int(record["det_local_idx"]), int(record["track_local_idx"])) in sparse_edit_pairs)
        for record in edge_records
    ]
    edge_is_bridge_commit = [
        int((int(record["det_local_idx"]), int(record["track_local_idx"])) in bridge_commit_pairs)
        for record in edge_records
    ]
    target_by_det_oracle = _assign_pairs_to_target(
        num_detections=len(det_rows),
        pairs=set(oracle_commit_pairs),
    )
    target_by_det_delta = _assign_pairs_to_target(
        num_detections=len(det_rows),
        pairs=set(delta_commit_pairs),
    )
    target_by_det_edit = [int(x) for x in oracle_action_by_det]
    target_by_det_rescue = [int(x) for x in oracle_action_by_det]
    target_by_det_sparse_edit = _assign_pairs_to_target(
        num_detections=len(det_rows),
        pairs=set(sparse_edit_pairs),
    )
    target_by_det_bridge = _assign_pairs_to_target(
        num_detections=len(det_rows),
        pairs=set(bridge_commit_pairs),
    )
    target_by_det_pairs = _assign_pairs_to_target(
        num_detections=len(det_rows),
        pairs=set(target_pairs),
    )

    compact_edges = []
    for record, oracle_commit, delta_commit, edit_commit, soft_rescue, sparse_edit, bridge_commit in zip(
        edge_records,
        edge_is_oracle_commit,
        edge_is_delta_commit,
        edge_is_edit_commit,
        edge_is_soft_rescue,
        edge_is_sparse_edit,
        edge_is_bridge_commit,
    ):
        compact_edges.append(
            {
                "det_local_idx": int(record["det_local_idx"]),
                "track_local_idx": int(record["track_local_idx"]),
                "track_id": int(record["track_id"]),
                "edge_is_gt_positive": int(record["edge_is_gt_positive"]),
                "edge_is_oracle_commit": int(oracle_commit),
                "edge_is_delta_commit": int(delta_commit),
                "edge_is_edit_commit": int(edit_commit),
                "edge_is_soft_rescue": int(soft_rescue),
                "edge_is_sparse_edit": int(sparse_edit),
                "edge_is_bridge_commit": int(bridge_commit),
            }
        )

    component_suffix = str(component.get("component_id_suffix", "") or "").strip()
    cluster_id = f"{source_tag}|{seq}:{frame}:{'-'.join(str(x) for x in det_rows)}"
    if component_suffix:
        cluster_id = f"{cluster_id}|{component_suffix}"

    return {
        "cluster_id": cluster_id,
        "seq": seq,
        "frame": int(frame),
        "source_tag": source_tag,
        "host_variant": host_variant,
        "split_tag": split_tag,
        "feature_version": feature_version,
        "teacher_mode": str(teacher_mode),
        "source_rows_csv": source_rows_csv,
        "source_group_jsonl": source_group_jsonl,
        "mined_from_large_component": int(component.get("mined_from_large_component", 0) or 0),
        "seed_det_rows": [int(x) for x in component.get("seed_det_rows", [])],
        "source_component_num_detections": int(component.get("source_component_num_detections", len(det_rows))),
        "source_component_num_tracks": int(component.get("source_component_num_tracks", len(track_ids))),
        "det_rows": [int(x) for x in det_rows],
        "track_ids": [int(x) for x in track_ids],
        "det_features": [[float(x) for x in row] for row in det_features],
        "track_features": [[float(x) for x in row] for row in track_features],
        "edge_features": [[float(x) for x in row] for row in edge_features],
        "edge_det_index": [int(x) for x in edge_det_index],
        "edge_track_index": [int(x) for x in edge_track_index],
        "cluster_features": [float(x) for x in cluster_features],
        "target_by_det": [int(x) for x in target_by_det],
        "target_by_det_oracle": [int(x) for x in target_by_det_oracle],
        "target_by_det_delta": [int(x) for x in target_by_det_delta],
        "target_by_det_edit": [int(x) for x in target_by_det_edit],
        "target_by_det_rescue": [int(x) for x in target_by_det_rescue],
        "target_by_det_sparse_edit": [int(x) for x in target_by_det_sparse_edit],
        "target_by_det_bridge": [int(x) for x in target_by_det_bridge],
        "target_by_det_pairs": [int(x) for x in target_by_det_pairs],
        "trigger_pass": int(trigger_pass),
        "cluster_should_intervene": int(active_cluster_should_intervene),
        "cluster_should_intervene_delta": int(cluster_should_intervene),
        "cluster_should_intervene_edit": int(cluster_should_intervene_edit),
        "cluster_should_intervene_soft": int(cluster_should_intervene_soft),
        "cluster_should_intervene_sparse": int(cluster_should_intervene_sparse),
        "cluster_should_intervene_bridge": int(cluster_should_intervene_bridge),
        "cluster_utility_gain": float(active_cluster_utility_gain),
        "cluster_utility_gain_delta": float(cluster_utility_gain),
        "cluster_edit_gain": float(cluster_edit_gain),
        "cluster_edit_cost": float(cluster_edit_cost),
        "cluster_edit_utility_gain": float(cluster_edit_utility_gain),
        "cluster_soft_utility_gain": float(cluster_soft_utility_gain),
        "cluster_sparse_utility_gain": float(cluster_sparse_utility_gain),
        "cluster_bridge_utility_gain": float(cluster_bridge_utility_gain),
        "is_large_component": int(is_large_component),
        "num_bridge_candidates": int(num_bridge_candidates),
        "host_equals_oracle": int(host_equals_oracle),
        "host_runtime_equals_oracle": int(host_runtime_equals_oracle),
        "target_committed_matches": int(target_committed_matches),
        "oracle_committed_matches": int(len(oracle_commit_pairs)),
        "host_committed_matches": int(len(host_pairs_local)),
        "host_runtime_committed_matches": int(len(host_pairs_local_runtime)),
        "delta_committed_matches": int(len(delta_commit_pairs)),
        "edit_committed_matches": int(len(edit_commit_pairs)),
        "rescue_committed_matches": int(len(rescue_commit_pairs)),
        "hard_delta_committed_matches": int(len(hard_delta_pairs)),
        "soft_rescue_committed_matches": int(len(soft_rescue_pairs)),
        "sparse_edit_committed_matches": int(len(sparse_edit_pairs)),
        "bridge_committed_matches": int(len(bridge_commit_pairs)),
        "host_pairs_local": _sorted_pair_list(set(host_pairs_local)),
        "host_pairs_local_runtime": _sorted_pair_list(set(host_pairs_local_runtime)),
        "oracle_pairs_local": _sorted_pair_list(set(oracle_commit_pairs)),
        "delta_commit_pairs": _sorted_pair_list(set(delta_commit_pairs)),
        "edit_commit_pairs": _sorted_pair_list(set(edit_commit_pairs)),
        "hard_delta_pairs": _sorted_pair_list(set(hard_delta_pairs)),
        "soft_rescue_pairs": _sorted_pair_list(set(soft_rescue_pairs)),
        "rescue_commit_pairs": _sorted_pair_list(set(rescue_commit_pairs)),
        "sparse_edit_pairs": _sorted_pair_list(set(sparse_edit_pairs)),
        "bridge_commit_pairs": _sorted_pair_list(set(bridge_commit_pairs)),
        "bridge_candidate_rows": bridge_candidate_rows,
        "host_action_by_det": [int(x) for x in host_action_by_det],
        "host_action_by_det_runtime": [int(x) for x in host_action_by_det_runtime],
        "oracle_action_by_det": [int(x) for x in oracle_action_by_det],
        "row_edit_mask": [int(x) for x in row_edit_mask],
        "row_rescue_mask": [int(x) for x in row_rescue_mask],
        "row_sparse_edit_mask": [int(x) for x in row_sparse_edit_mask],
        "row_bridge_mask": [int(x) for x in row_bridge_mask],
        "row_action_type": [str(x) for x in row_action_type],
        "row_rescue_action_type": [str(x) for x in row_rescue_action_type],
        "row_sparse_action_type": [str(x) for x in row_sparse_action_type],
        "num_edit_rows": int(sum(row_edit_mask)),
        "num_rescue_rows": int(sum(row_rescue_mask)),
        "num_sparse_edit_rows": int(sum(row_sparse_edit_mask)),
        "num_bridge_rows": int(sum(row_bridge_mask)),
        "num_add_commit_rows": int(num_add_commit_rows),
        "num_reassign_rows": int(num_reassign_rows),
        "num_force_defer_rows": int(num_force_defer_rows),
        "num_hard_delta_rows": int(num_hard_delta_rows),
        "num_soft_rescue_rows": int(num_soft_rescue_rows),
        "num_rescue_force_defer_rows": int(num_rescue_force_defer_rows),
        "num_sparse_add_rows": int(num_sparse_add_rows),
        "num_sparse_reassign_free_rows": int(num_sparse_reassign_free_rows),
        "num_sparse_blocked_reassign_rows": int(num_sparse_blocked_reassign_rows),
        "edge_is_gt_positive": [int(x) for x in edge_is_gt_positive],
        "edge_is_oracle_commit": [int(x) for x in edge_is_oracle_commit],
        "edge_is_delta_commit": [int(x) for x in edge_is_delta_commit],
        "edge_is_edit_commit": [int(x) for x in edge_is_edit_commit],
        "edge_is_soft_rescue": [int(x) for x in edge_is_soft_rescue],
        "edge_is_sparse_edit": [int(x) for x in edge_is_sparse_edit],
        "edge_is_bridge_commit": [int(x) for x in edge_is_bridge_commit],
        "num_detections": int(len(det_rows)),
        "num_tracks": int(len(track_ids)),
        "num_edges": int(len(edge_records)),
        "compact_edges": compact_edges,
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_jsonl = out_dir / "cluster_examples.jsonl"
    cluster_sample_jsonl = out_dir / "cluster_examples.sample.jsonl"
    cluster_summary_csv = out_dir / "cluster_summary.csv"
    seq_summary_csv = out_dir / "sequence_cluster_summary.csv"
    summary_json = out_dir / "summary.json"
    summary_csv = out_dir / "summary.csv"

    sources, manifest_path = _load_sources(args)
    train_tokens = _parse_csv_tokens(args.train_sequences)
    val_tokens = _parse_csv_tokens(args.val_sequences)

    summary_counter: Counter[str] = Counter()
    seq_counter: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    split_counter: dict[str, Counter[str]] = defaultdict(Counter)
    cluster_rows: list[dict[str, Any]] = []
    cluster_examples: list[dict[str, Any]] = []
    seen_sequences: set[str] = set()
    seen_hosts: set[str] = set()
    seen_sources: set[str] = set()

    for source_idx, source in enumerate(sources):
        rows_csv = Path(source["rows_csv"]).resolve()
        group_jsonl = Path(source["group_jsonl"]).resolve()
        if not rows_csv.is_file():
            raise FileNotFoundError(f"Missing rows csv: {rows_csv}")
        if not group_jsonl.is_file():
            raise FileNotFoundError(f"Missing group jsonl: {group_jsonl}")

        rows_by_group = _load_rows_by_group(rows_csv)
        groups_by_frame = _load_groups_by_frame(group_jsonl)
        source_tag = str(source["source_tag"])
        host_variant = str(source["host_variant"])
        feature_version = str(source["feature_version"] or args.feature_version)
        summary_counter["sources"] += 1
        summary_counter["source_frames"] += int(len(groups_by_frame))
        seen_hosts.add(host_variant)
        seen_sources.add(source_tag)

        for (seq, frame), frame_groups in sorted(groups_by_frame.items(), key=lambda x: (x[0][0], x[0][1])):
            split_tag = _determine_split_tag(
                seq=seq,
                explicit_split_tag=str(source.get("split_tag", "")),
                train_tokens=train_tokens,
                val_tokens=val_tokens,
                strict_sequence_split=bool(args.strict_sequence_split),
            )
            if split_tag == "unused":
                summary_counter["unused_frames"] += 1
                continue

            components = build_group_components_from_group_rows(frame_groups, topk=int(args.topk))
            eligible_components, skipped_large = filter_local_conflict_clusters_by_size(
                components,
                min_detections=int(args.min_detections),
                max_detections=int(args.max_detections),
                max_tracks=int(args.max_tracks),
            )
            oversized_components = []
            max_d = int(args.max_detections) if int(args.max_detections) > 0 else None
            max_t = int(args.max_tracks) if int(args.max_tracks) > 0 else None
            for component in components:
                num_d = int(component.get("num_detections", len(component.get("det_rows", [])) or 0))
                num_t = int(component.get("num_tracks", len(component.get("track_ids", [])) or 0))
                if num_d < int(args.min_detections):
                    continue
                if (max_d is not None and num_d > max_d) or (max_t is not None and num_t > max_t):
                    oversized_components.append(component)
            mined_components: list[dict[str, Any]] = []
            if int(args.large_component_max_subclusters) > 0 and oversized_components:
                for component in oversized_components:
                    mined = mine_centered_subcomponents_from_group_rows(
                        frame_groups,
                        component=component,
                        topk=int(args.topk),
                        min_detections=int(args.min_detections),
                        max_detections=int(args.max_detections),
                        max_tracks=int(args.max_tracks),
                        max_subcomponents=int(args.large_component_max_subclusters),
                    )
                    if mined:
                        mined_components.extend(mined)
                        summary_counter["recovered_large_components"] += 1
                        split_counter[split_tag]["recovered_large_components"] += 1
                        seq_counter[(seq, split_tag)]["recovered_large_components"] += 1
            combined_components = list(eligible_components) + mined_components
            summary_counter["frames"] += 1
            summary_counter["raw_components"] += int(len(components))
            summary_counter["skipped_large_clusters"] += int(skipped_large)
            summary_counter["oversized_components"] += int(len(oversized_components))
            summary_counter["mined_large_subclusters"] += int(len(mined_components))
            split_counter[split_tag]["frames"] += 1
            split_counter[split_tag]["raw_components"] += int(len(components))
            split_counter[split_tag]["skipped_large_clusters"] += int(skipped_large)
            split_counter[split_tag]["oversized_components"] += int(len(oversized_components))
            split_counter[split_tag]["mined_large_subclusters"] += int(len(mined_components))
            seq_counter[(seq, split_tag)]["oversized_components"] += int(len(oversized_components))
            seq_counter[(seq, split_tag)]["mined_large_subclusters"] += int(len(mined_components))
            if combined_components:
                summary_counter["frames_with_eligible_clusters"] += 1
                split_counter[split_tag]["frames_with_eligible_clusters"] += 1
                seq_counter[(seq, split_tag)]["frames_with_eligible_clusters"] += 1

            seen_sequences.add(seq)
            for component in combined_components:
                example = _cluster_example(
                    seq=seq,
                    frame=int(frame),
                    component=component,
                    frame_groups=frame_groups,
                    rows_by_group=rows_by_group,
                    topk=int(args.topk),
                    min_committed_matches=int(args.min_committed_matches),
                    teacher_mode=str(args.teacher_mode),
                    source_tag=source_tag,
                    host_variant=host_variant,
                    split_tag=split_tag,
                    feature_version=feature_version,
                    source_rows_csv=str(rows_csv),
                    source_group_jsonl=str(group_jsonl),
                    edit_utility_commit_cost=float(args.edit_utility_commit_cost),
                    edit_utility_min_gain=float(args.edit_utility_min_gain),
                    edit_utility_force_defer_gain=float(args.edit_utility_force_defer_gain),
                    runtime_host_match_thresh=float(args.runtime_host_match_thresh),
                    soft_rescue_weight=float(args.soft_rescue_weight),
                    rescue_force_defer_gain=float(args.rescue_force_defer_gain),
                    rescue_min_gain=float(args.rescue_min_gain),
                    bridge_crowded_row_degree_thresh=int(args.bridge_crowded_row_degree_thresh),
                    bridge_crowded_bonus=float(args.bridge_crowded_bonus),
                    bridge_large_component_bonus=float(args.bridge_large_component_bonus),
                    bridge_commit_cost=float(args.bridge_commit_cost),
                    bridge_min_gain=float(args.bridge_min_gain),
                )
                if example is None:
                    continue
                cluster_examples.append(example)
                row = {
                    "cluster_id": example["cluster_id"],
                    "seq": seq,
                    "frame": int(frame),
                    "source_tag": source_tag,
                    "host_variant": host_variant,
                    "split_tag": split_tag,
                    "feature_version": feature_version,
                    "teacher_mode": str(args.teacher_mode),
                    "mined_from_large_component": int(example.get("mined_from_large_component", 0)),
                    "seed_det_rows": ",".join(str(x) for x in example.get("seed_det_rows", [])),
                    "source_component_num_detections": int(example.get("source_component_num_detections", example["num_detections"])),
                    "source_component_num_tracks": int(example.get("source_component_num_tracks", example["num_tracks"])),
                    "num_detections": example["num_detections"],
                    "num_tracks": example["num_tracks"],
                    "num_edges": example["num_edges"],
                    "target_committed_matches": example["target_committed_matches"],
                    "trigger_pass": example["trigger_pass"],
                    "cluster_should_intervene": example["cluster_should_intervene"],
                    "cluster_utility_gain": example["cluster_utility_gain"],
                    "cluster_edit_gain": example["cluster_edit_gain"],
                    "cluster_edit_cost": example["cluster_edit_cost"],
                    "cluster_edit_utility_gain": example["cluster_edit_utility_gain"],
                    "cluster_should_intervene_soft": example["cluster_should_intervene_soft"],
                    "cluster_soft_utility_gain": example["cluster_soft_utility_gain"],
                    "cluster_should_intervene_sparse": example["cluster_should_intervene_sparse"],
                    "cluster_sparse_utility_gain": example["cluster_sparse_utility_gain"],
                    "cluster_should_intervene_bridge": example["cluster_should_intervene_bridge"],
                    "cluster_bridge_utility_gain": example["cluster_bridge_utility_gain"],
                    "bridge_committed_matches": example["bridge_committed_matches"],
                    "num_bridge_rows": example["num_bridge_rows"],
                    "num_bridge_candidates": example["num_bridge_candidates"],
                    "is_large_component": example["is_large_component"],
                    "host_equals_oracle": example["host_equals_oracle"],
                    "host_runtime_equals_oracle": example["host_runtime_equals_oracle"],
                    "host_committed_matches": example["host_committed_matches"],
                    "host_runtime_committed_matches": example["host_runtime_committed_matches"],
                    "oracle_committed_matches": example["oracle_committed_matches"],
                    "delta_committed_matches": example["delta_committed_matches"],
                    "edit_committed_matches": example["edit_committed_matches"],
                    "rescue_committed_matches": example["rescue_committed_matches"],
                    "hard_delta_committed_matches": example["hard_delta_committed_matches"],
                    "soft_rescue_committed_matches": example["soft_rescue_committed_matches"],
                    "sparse_edit_committed_matches": example["sparse_edit_committed_matches"],
                    "num_edit_rows": example["num_edit_rows"],
                    "num_rescue_rows": example["num_rescue_rows"],
                    "num_sparse_edit_rows": example["num_sparse_edit_rows"],
                    "num_add_commit_rows": example["num_add_commit_rows"],
                    "num_reassign_rows": example["num_reassign_rows"],
                    "num_force_defer_rows": example["num_force_defer_rows"],
                    "num_hard_delta_rows": example["num_hard_delta_rows"],
                    "num_soft_rescue_rows": example["num_soft_rescue_rows"],
                    "num_rescue_force_defer_rows": example["num_rescue_force_defer_rows"],
                    "num_sparse_add_rows": example["num_sparse_add_rows"],
                    "num_sparse_reassign_free_rows": example["num_sparse_reassign_free_rows"],
                    "num_sparse_blocked_reassign_rows": example["num_sparse_blocked_reassign_rows"],
                }
                cluster_rows.append(row)
                summary_counter["eligible_clusters"] += 1
                summary_counter["mined_component_clusters"] += int(example.get("mined_from_large_component", 0))
                summary_counter["detections"] += int(example["num_detections"])
                summary_counter["tracks"] += int(example["num_tracks"])
                summary_counter["edges"] += int(example["num_edges"])
                summary_counter["trigger_pass_clusters"] += int(example["trigger_pass"])
                summary_counter["trigger_fail_clusters"] += int(1 - int(example["trigger_pass"]))
                summary_counter["committed_matches"] += int(example["target_committed_matches"])
                summary_counter["cluster_should_intervene_clusters"] += int(example["cluster_should_intervene"])
                summary_counter["cluster_utility_gain"] += float(example["cluster_utility_gain"])
                summary_counter["cluster_edit_gain"] += float(example["cluster_edit_gain"])
                summary_counter["cluster_edit_cost"] += float(example["cluster_edit_cost"])
                summary_counter["cluster_edit_utility_gain"] += float(example["cluster_edit_utility_gain"])
                summary_counter["cluster_should_intervene_soft_clusters"] += int(example["cluster_should_intervene_soft"])
                summary_counter["cluster_soft_utility_gain"] += float(example["cluster_soft_utility_gain"])
                summary_counter["cluster_should_intervene_sparse_clusters"] += int(example["cluster_should_intervene_sparse"])
                summary_counter["cluster_sparse_utility_gain"] += float(example["cluster_sparse_utility_gain"])
                summary_counter["cluster_should_intervene_bridge_clusters"] += int(example["cluster_should_intervene_bridge"])
                summary_counter["cluster_bridge_utility_gain"] += float(example["cluster_bridge_utility_gain"])
                summary_counter["bridge_committed_matches"] += int(example["bridge_committed_matches"])
                summary_counter["bridge_rows"] += int(example["num_bridge_rows"])
                summary_counter["bridge_candidates"] += int(example["num_bridge_candidates"])
                summary_counter["large_component_clusters"] += int(example["is_large_component"])
                summary_counter["host_equals_oracle_clusters"] += int(example["host_equals_oracle"])
                summary_counter["host_runtime_equals_oracle_clusters"] += int(example["host_runtime_equals_oracle"])
                summary_counter["host_committed_matches"] += int(example["host_committed_matches"])
                summary_counter["host_runtime_committed_matches"] += int(example["host_runtime_committed_matches"])
                summary_counter["oracle_committed_matches"] += int(example["oracle_committed_matches"])
                summary_counter["delta_committed_matches"] += int(example["delta_committed_matches"])
                summary_counter["edit_committed_matches"] += int(example["edit_committed_matches"])
                summary_counter["rescue_committed_matches"] += int(example["rescue_committed_matches"])
                summary_counter["hard_delta_committed_matches"] += int(example["hard_delta_committed_matches"])
                summary_counter["soft_rescue_committed_matches"] += int(example["soft_rescue_committed_matches"])
                summary_counter["sparse_edit_committed_matches"] += int(example["sparse_edit_committed_matches"])
                summary_counter["edit_rows"] += int(example["num_edit_rows"])
                summary_counter["rescue_rows"] += int(example["num_rescue_rows"])
                summary_counter["sparse_edit_rows"] += int(example["num_sparse_edit_rows"])
                summary_counter["add_commit_rows"] += int(example["num_add_commit_rows"])
                summary_counter["reassign_rows"] += int(example["num_reassign_rows"])
                summary_counter["force_defer_rows"] += int(example["num_force_defer_rows"])
                summary_counter["hard_delta_rows"] += int(example["num_hard_delta_rows"])
                summary_counter["soft_rescue_rows"] += int(example["num_soft_rescue_rows"])
                summary_counter["rescue_force_defer_rows"] += int(example["num_rescue_force_defer_rows"])
                summary_counter["sparse_add_rows"] += int(example["num_sparse_add_rows"])
                summary_counter["sparse_reassign_free_rows"] += int(example["num_sparse_reassign_free_rows"])
                summary_counter["sparse_blocked_reassign_rows"] += int(example["num_sparse_blocked_reassign_rows"])
                split_counter[split_tag]["eligible_clusters"] += 1
                split_counter[split_tag]["mined_component_clusters"] += int(example.get("mined_from_large_component", 0))
                split_counter[split_tag]["detections"] += int(example["num_detections"])
                split_counter[split_tag]["tracks"] += int(example["num_tracks"])
                split_counter[split_tag]["edges"] += int(example["num_edges"])
                split_counter[split_tag]["trigger_pass_clusters"] += int(example["trigger_pass"])
                split_counter[split_tag]["committed_matches"] += int(example["target_committed_matches"])
                split_counter[split_tag]["cluster_should_intervene_clusters"] += int(example["cluster_should_intervene"])
                split_counter[split_tag]["cluster_utility_gain"] += float(example["cluster_utility_gain"])
                split_counter[split_tag]["cluster_edit_gain"] += float(example["cluster_edit_gain"])
                split_counter[split_tag]["cluster_edit_cost"] += float(example["cluster_edit_cost"])
                split_counter[split_tag]["cluster_edit_utility_gain"] += float(example["cluster_edit_utility_gain"])
                split_counter[split_tag]["cluster_should_intervene_soft_clusters"] += int(example["cluster_should_intervene_soft"])
                split_counter[split_tag]["cluster_soft_utility_gain"] += float(example["cluster_soft_utility_gain"])
                split_counter[split_tag]["cluster_should_intervene_sparse_clusters"] += int(example["cluster_should_intervene_sparse"])
                split_counter[split_tag]["cluster_sparse_utility_gain"] += float(example["cluster_sparse_utility_gain"])
                split_counter[split_tag]["cluster_should_intervene_bridge_clusters"] += int(example["cluster_should_intervene_bridge"])
                split_counter[split_tag]["cluster_bridge_utility_gain"] += float(example["cluster_bridge_utility_gain"])
                split_counter[split_tag]["bridge_committed_matches"] += int(example["bridge_committed_matches"])
                split_counter[split_tag]["bridge_rows"] += int(example["num_bridge_rows"])
                split_counter[split_tag]["bridge_candidates"] += int(example["num_bridge_candidates"])
                split_counter[split_tag]["large_component_clusters"] += int(example["is_large_component"])
                split_counter[split_tag]["host_equals_oracle_clusters"] += int(example["host_equals_oracle"])
                split_counter[split_tag]["host_runtime_equals_oracle_clusters"] += int(example["host_runtime_equals_oracle"])
                split_counter[split_tag]["host_committed_matches"] += int(example["host_committed_matches"])
                split_counter[split_tag]["host_runtime_committed_matches"] += int(example["host_runtime_committed_matches"])
                split_counter[split_tag]["oracle_committed_matches"] += int(example["oracle_committed_matches"])
                split_counter[split_tag]["delta_committed_matches"] += int(example["delta_committed_matches"])
                split_counter[split_tag]["edit_committed_matches"] += int(example["edit_committed_matches"])
                split_counter[split_tag]["rescue_committed_matches"] += int(example["rescue_committed_matches"])
                split_counter[split_tag]["hard_delta_committed_matches"] += int(example["hard_delta_committed_matches"])
                split_counter[split_tag]["soft_rescue_committed_matches"] += int(example["soft_rescue_committed_matches"])
                split_counter[split_tag]["sparse_edit_committed_matches"] += int(example["sparse_edit_committed_matches"])
                split_counter[split_tag]["edit_rows"] += int(example["num_edit_rows"])
                split_counter[split_tag]["rescue_rows"] += int(example["num_rescue_rows"])
                split_counter[split_tag]["sparse_edit_rows"] += int(example["num_sparse_edit_rows"])
                split_counter[split_tag]["add_commit_rows"] += int(example["num_add_commit_rows"])
                split_counter[split_tag]["reassign_rows"] += int(example["num_reassign_rows"])
                split_counter[split_tag]["force_defer_rows"] += int(example["num_force_defer_rows"])
                split_counter[split_tag]["hard_delta_rows"] += int(example["num_hard_delta_rows"])
                split_counter[split_tag]["soft_rescue_rows"] += int(example["num_soft_rescue_rows"])
                split_counter[split_tag]["rescue_force_defer_rows"] += int(example["num_rescue_force_defer_rows"])
                split_counter[split_tag]["sparse_add_rows"] += int(example["num_sparse_add_rows"])
                split_counter[split_tag]["sparse_reassign_free_rows"] += int(example["num_sparse_reassign_free_rows"])
                split_counter[split_tag]["sparse_blocked_reassign_rows"] += int(example["num_sparse_blocked_reassign_rows"])
                seq_counter[(seq, split_tag)]["eligible_clusters"] += 1
                seq_counter[(seq, split_tag)]["mined_component_clusters"] += int(example.get("mined_from_large_component", 0))
                seq_counter[(seq, split_tag)]["detections"] += int(example["num_detections"])
                seq_counter[(seq, split_tag)]["tracks"] += int(example["num_tracks"])
                seq_counter[(seq, split_tag)]["edges"] += int(example["num_edges"])
                seq_counter[(seq, split_tag)]["trigger_pass_clusters"] += int(example["trigger_pass"])
                seq_counter[(seq, split_tag)]["committed_matches"] += int(example["target_committed_matches"])
                seq_counter[(seq, split_tag)]["cluster_should_intervene_clusters"] += int(example["cluster_should_intervene"])
                seq_counter[(seq, split_tag)]["cluster_utility_gain"] += float(example["cluster_utility_gain"])
                seq_counter[(seq, split_tag)]["cluster_edit_gain"] += float(example["cluster_edit_gain"])
                seq_counter[(seq, split_tag)]["cluster_edit_cost"] += float(example["cluster_edit_cost"])
                seq_counter[(seq, split_tag)]["cluster_edit_utility_gain"] += float(example["cluster_edit_utility_gain"])
                seq_counter[(seq, split_tag)]["cluster_should_intervene_soft_clusters"] += int(example["cluster_should_intervene_soft"])
                seq_counter[(seq, split_tag)]["cluster_soft_utility_gain"] += float(example["cluster_soft_utility_gain"])
                seq_counter[(seq, split_tag)]["cluster_should_intervene_sparse_clusters"] += int(example["cluster_should_intervene_sparse"])
                seq_counter[(seq, split_tag)]["cluster_sparse_utility_gain"] += float(example["cluster_sparse_utility_gain"])
                seq_counter[(seq, split_tag)]["cluster_should_intervene_bridge_clusters"] += int(example["cluster_should_intervene_bridge"])
                seq_counter[(seq, split_tag)]["cluster_bridge_utility_gain"] += float(example["cluster_bridge_utility_gain"])
                seq_counter[(seq, split_tag)]["bridge_committed_matches"] += int(example["bridge_committed_matches"])
                seq_counter[(seq, split_tag)]["bridge_rows"] += int(example["num_bridge_rows"])
                seq_counter[(seq, split_tag)]["bridge_candidates"] += int(example["num_bridge_candidates"])
                seq_counter[(seq, split_tag)]["large_component_clusters"] += int(example["is_large_component"])
                seq_counter[(seq, split_tag)]["host_equals_oracle_clusters"] += int(example["host_equals_oracle"])
                seq_counter[(seq, split_tag)]["host_runtime_equals_oracle_clusters"] += int(example["host_runtime_equals_oracle"])
                seq_counter[(seq, split_tag)]["host_committed_matches"] += int(example["host_committed_matches"])
                seq_counter[(seq, split_tag)]["host_runtime_committed_matches"] += int(example["host_runtime_committed_matches"])
                seq_counter[(seq, split_tag)]["oracle_committed_matches"] += int(example["oracle_committed_matches"])
                seq_counter[(seq, split_tag)]["delta_committed_matches"] += int(example["delta_committed_matches"])
                seq_counter[(seq, split_tag)]["edit_committed_matches"] += int(example["edit_committed_matches"])
                seq_counter[(seq, split_tag)]["rescue_committed_matches"] += int(example["rescue_committed_matches"])
                seq_counter[(seq, split_tag)]["hard_delta_committed_matches"] += int(example["hard_delta_committed_matches"])
                seq_counter[(seq, split_tag)]["soft_rescue_committed_matches"] += int(example["soft_rescue_committed_matches"])
                seq_counter[(seq, split_tag)]["sparse_edit_committed_matches"] += int(example["sparse_edit_committed_matches"])
                seq_counter[(seq, split_tag)]["edit_rows"] += int(example["num_edit_rows"])
                seq_counter[(seq, split_tag)]["rescue_rows"] += int(example["num_rescue_rows"])
                seq_counter[(seq, split_tag)]["sparse_edit_rows"] += int(example["num_sparse_edit_rows"])
                seq_counter[(seq, split_tag)]["add_commit_rows"] += int(example["num_add_commit_rows"])
                seq_counter[(seq, split_tag)]["reassign_rows"] += int(example["num_reassign_rows"])
                seq_counter[(seq, split_tag)]["force_defer_rows"] += int(example["num_force_defer_rows"])
                seq_counter[(seq, split_tag)]["hard_delta_rows"] += int(example["num_hard_delta_rows"])
                seq_counter[(seq, split_tag)]["soft_rescue_rows"] += int(example["num_soft_rescue_rows"])
                seq_counter[(seq, split_tag)]["rescue_force_defer_rows"] += int(example["num_rescue_force_defer_rows"])
                seq_counter[(seq, split_tag)]["sparse_add_rows"] += int(example["num_sparse_add_rows"])
                seq_counter[(seq, split_tag)]["sparse_reassign_free_rows"] += int(example["num_sparse_reassign_free_rows"])
                seq_counter[(seq, split_tag)]["sparse_blocked_reassign_rows"] += int(example["num_sparse_blocked_reassign_rows"])
                seq_counter[(seq, split_tag)]["source_index"] = source_idx

    with cluster_jsonl.open("w", encoding="utf-8") as f:
        for row in cluster_examples:
            f.write(json.dumps(row) + "\n")

    with cluster_sample_jsonl.open("w", encoding="utf-8") as f:
        for row in cluster_examples[: max(int(args.sample_size), 0)]:
            f.write(json.dumps(row) + "\n")

    if cluster_rows:
        with cluster_summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(cluster_rows[0].keys()))
            writer.writeheader()
            for row in cluster_rows:
                writer.writerow(row)

    seq_rows: list[dict[str, Any]] = []
    for (seq, split_tag), counter in sorted(seq_counter.items(), key=lambda item: (item[0][0], item[0][1])):
        seq_rows.append(
            {
                "seq": seq,
                "split_tag": split_tag,
                "frames_with_eligible_clusters": int(counter.get("frames_with_eligible_clusters", 0)),
                "eligible_clusters": int(counter.get("eligible_clusters", 0)),
                "oversized_components": int(counter.get("oversized_components", 0)),
                "recovered_large_components": int(counter.get("recovered_large_components", 0)),
                "mined_large_subclusters": int(counter.get("mined_large_subclusters", 0)),
                "mined_component_clusters": int(counter.get("mined_component_clusters", 0)),
                "detections": int(counter.get("detections", 0)),
                "tracks": int(counter.get("tracks", 0)),
                "edges": int(counter.get("edges", 0)),
                "trigger_pass_clusters": int(counter.get("trigger_pass_clusters", 0)),
                "committed_matches": int(counter.get("committed_matches", 0)),
                "cluster_should_intervene_clusters": int(counter.get("cluster_should_intervene_clusters", 0)),
                "cluster_utility_gain": float(counter.get("cluster_utility_gain", 0.0)),
                "cluster_edit_gain": float(counter.get("cluster_edit_gain", 0.0)),
                "cluster_edit_cost": float(counter.get("cluster_edit_cost", 0.0)),
                "cluster_edit_utility_gain": float(counter.get("cluster_edit_utility_gain", 0.0)),
                "cluster_should_intervene_soft_clusters": int(counter.get("cluster_should_intervene_soft_clusters", 0)),
                "cluster_soft_utility_gain": float(counter.get("cluster_soft_utility_gain", 0.0)),
                "cluster_should_intervene_sparse_clusters": int(counter.get("cluster_should_intervene_sparse_clusters", 0)),
                "cluster_sparse_utility_gain": float(counter.get("cluster_sparse_utility_gain", 0.0)),
                "cluster_should_intervene_bridge_clusters": int(counter.get("cluster_should_intervene_bridge_clusters", 0)),
                "cluster_bridge_utility_gain": float(counter.get("cluster_bridge_utility_gain", 0.0)),
                "bridge_committed_matches": int(counter.get("bridge_committed_matches", 0)),
                "bridge_rows": int(counter.get("bridge_rows", 0)),
                "bridge_candidates": int(counter.get("bridge_candidates", 0)),
                "large_component_clusters": int(counter.get("large_component_clusters", 0)),
                "host_equals_oracle_clusters": int(counter.get("host_equals_oracle_clusters", 0)),
                "host_runtime_equals_oracle_clusters": int(counter.get("host_runtime_equals_oracle_clusters", 0)),
                "host_committed_matches": int(counter.get("host_committed_matches", 0)),
                "host_runtime_committed_matches": int(counter.get("host_runtime_committed_matches", 0)),
                "oracle_committed_matches": int(counter.get("oracle_committed_matches", 0)),
                "delta_committed_matches": int(counter.get("delta_committed_matches", 0)),
                "edit_committed_matches": int(counter.get("edit_committed_matches", 0)),
                "rescue_committed_matches": int(counter.get("rescue_committed_matches", 0)),
                "hard_delta_committed_matches": int(counter.get("hard_delta_committed_matches", 0)),
                "soft_rescue_committed_matches": int(counter.get("soft_rescue_committed_matches", 0)),
                "sparse_edit_committed_matches": int(counter.get("sparse_edit_committed_matches", 0)),
                "edit_rows": int(counter.get("edit_rows", 0)),
                "rescue_rows": int(counter.get("rescue_rows", 0)),
                "sparse_edit_rows": int(counter.get("sparse_edit_rows", 0)),
                "add_commit_rows": int(counter.get("add_commit_rows", 0)),
                "reassign_rows": int(counter.get("reassign_rows", 0)),
                "force_defer_rows": int(counter.get("force_defer_rows", 0)),
                "hard_delta_rows": int(counter.get("hard_delta_rows", 0)),
                "soft_rescue_rows": int(counter.get("soft_rescue_rows", 0)),
                "rescue_force_defer_rows": int(counter.get("rescue_force_defer_rows", 0)),
                "sparse_add_rows": int(counter.get("sparse_add_rows", 0)),
                "sparse_reassign_free_rows": int(counter.get("sparse_reassign_free_rows", 0)),
                "sparse_blocked_reassign_rows": int(counter.get("sparse_blocked_reassign_rows", 0)),
            }
        )
    if seq_rows:
        with seq_summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(seq_rows[0].keys()))
            writer.writeheader()
            for row in seq_rows:
                writer.writerow(row)

    summary_row = {
        "exp_name": "build_local_conflict_set_predictor_dataset",
        "dataset_tag": str(args.dataset_tag),
        "feature_version": str(args.feature_version),
        "teacher_mode": str(args.teacher_mode),
        "edit_utility_commit_cost": float(args.edit_utility_commit_cost),
        "edit_utility_min_gain": float(args.edit_utility_min_gain),
        "edit_utility_force_defer_gain": float(args.edit_utility_force_defer_gain),
        "runtime_host_match_thresh": float(args.runtime_host_match_thresh),
        "soft_rescue_weight": float(args.soft_rescue_weight),
        "rescue_force_defer_gain": float(args.rescue_force_defer_gain),
        "rescue_min_gain": float(args.rescue_min_gain),
        "bridge_crowded_row_degree_thresh": int(args.bridge_crowded_row_degree_thresh),
        "bridge_crowded_bonus": float(args.bridge_crowded_bonus),
        "bridge_large_component_bonus": float(args.bridge_large_component_bonus),
        "bridge_commit_cost": float(args.bridge_commit_cost),
        "bridge_min_gain": float(args.bridge_min_gain),
        "large_component_max_subclusters": int(args.large_component_max_subclusters),
        "source_manifest": manifest_path,
        "sources": int(summary_counter.get("sources", 0)),
        "source_frames": int(summary_counter.get("source_frames", 0)),
        "frames": int(summary_counter.get("frames", 0)),
        "unused_frames": int(summary_counter.get("unused_frames", 0)),
        "raw_components": int(summary_counter.get("raw_components", 0)),
        "eligible_clusters": int(summary_counter.get("eligible_clusters", 0)),
        "oversized_components": int(summary_counter.get("oversized_components", 0)),
        "recovered_large_components": int(summary_counter.get("recovered_large_components", 0)),
        "mined_large_subclusters": int(summary_counter.get("mined_large_subclusters", 0)),
        "mined_component_clusters": int(summary_counter.get("mined_component_clusters", 0)),
        "frames_with_eligible_clusters": int(summary_counter.get("frames_with_eligible_clusters", 0)),
        "trigger_pass_clusters": int(summary_counter.get("trigger_pass_clusters", 0)),
        "trigger_fail_clusters": int(summary_counter.get("trigger_fail_clusters", 0)),
        "committed_matches": int(summary_counter.get("committed_matches", 0)),
        "cluster_should_intervene_clusters": int(summary_counter.get("cluster_should_intervene_clusters", 0)),
        "cluster_utility_gain": float(summary_counter.get("cluster_utility_gain", 0.0)),
        "cluster_edit_gain": float(summary_counter.get("cluster_edit_gain", 0.0)),
        "cluster_edit_cost": float(summary_counter.get("cluster_edit_cost", 0.0)),
        "cluster_edit_utility_gain": float(summary_counter.get("cluster_edit_utility_gain", 0.0)),
        "cluster_should_intervene_soft_clusters": int(summary_counter.get("cluster_should_intervene_soft_clusters", 0)),
        "cluster_soft_utility_gain": float(summary_counter.get("cluster_soft_utility_gain", 0.0)),
        "cluster_should_intervene_sparse_clusters": int(summary_counter.get("cluster_should_intervene_sparse_clusters", 0)),
        "cluster_sparse_utility_gain": float(summary_counter.get("cluster_sparse_utility_gain", 0.0)),
        "cluster_should_intervene_bridge_clusters": int(summary_counter.get("cluster_should_intervene_bridge_clusters", 0)),
        "cluster_bridge_utility_gain": float(summary_counter.get("cluster_bridge_utility_gain", 0.0)),
        "bridge_committed_matches": int(summary_counter.get("bridge_committed_matches", 0)),
        "bridge_rows": int(summary_counter.get("bridge_rows", 0)),
        "bridge_candidates": int(summary_counter.get("bridge_candidates", 0)),
        "large_component_clusters": int(summary_counter.get("large_component_clusters", 0)),
        "host_equals_oracle_clusters": int(summary_counter.get("host_equals_oracle_clusters", 0)),
        "host_runtime_equals_oracle_clusters": int(summary_counter.get("host_runtime_equals_oracle_clusters", 0)),
        "host_committed_matches": int(summary_counter.get("host_committed_matches", 0)),
        "host_runtime_committed_matches": int(summary_counter.get("host_runtime_committed_matches", 0)),
        "oracle_committed_matches": int(summary_counter.get("oracle_committed_matches", 0)),
        "delta_committed_matches": int(summary_counter.get("delta_committed_matches", 0)),
        "edit_committed_matches": int(summary_counter.get("edit_committed_matches", 0)),
        "rescue_committed_matches": int(summary_counter.get("rescue_committed_matches", 0)),
        "hard_delta_committed_matches": int(summary_counter.get("hard_delta_committed_matches", 0)),
        "soft_rescue_committed_matches": int(summary_counter.get("soft_rescue_committed_matches", 0)),
        "sparse_edit_committed_matches": int(summary_counter.get("sparse_edit_committed_matches", 0)),
        "edit_rows": int(summary_counter.get("edit_rows", 0)),
        "rescue_rows": int(summary_counter.get("rescue_rows", 0)),
        "sparse_edit_rows": int(summary_counter.get("sparse_edit_rows", 0)),
        "add_commit_rows": int(summary_counter.get("add_commit_rows", 0)),
        "reassign_rows": int(summary_counter.get("reassign_rows", 0)),
        "force_defer_rows": int(summary_counter.get("force_defer_rows", 0)),
        "hard_delta_rows": int(summary_counter.get("hard_delta_rows", 0)),
        "soft_rescue_rows": int(summary_counter.get("soft_rescue_rows", 0)),
        "rescue_force_defer_rows": int(summary_counter.get("rescue_force_defer_rows", 0)),
        "sparse_add_rows": int(summary_counter.get("sparse_add_rows", 0)),
        "sparse_reassign_free_rows": int(summary_counter.get("sparse_reassign_free_rows", 0)),
        "sparse_blocked_reassign_rows": int(summary_counter.get("sparse_blocked_reassign_rows", 0)),
        "detections": int(summary_counter.get("detections", 0)),
        "tracks": int(summary_counter.get("tracks", 0)),
        "edges": int(summary_counter.get("edges", 0)),
        "skipped_large_clusters": int(summary_counter.get("skipped_large_clusters", 0)),
        "train_sequences": ",".join(sorted(seq for (seq, split_tag) in seq_counter.keys() if split_tag == "train")),
        "val_sequences": ",".join(sorted(seq for (seq, split_tag) in seq_counter.keys() if split_tag == "val")),
        "host_variants": ",".join(sorted(seen_hosts)),
        "source_tags": ",".join(sorted(seen_sources)),
        "det_feature_dim": len(DET_FEATURE_NAMES),
        "track_feature_dim": len(TRACK_FEATURE_NAMES),
        "edge_feature_dim": len(EDGE_FEATURE_NAMES),
        "cluster_feature_dim": len(CLUSTER_FEATURE_NAMES),
        "status": "ok",
    }
    summary_payload = {
        **summary_row,
        "split_breakdown": {key: dict(counter) for key, counter in sorted(split_counter.items())},
        "sequences": sorted(seen_sequences),
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    _write_single_row_csv(summary_csv, summary_row)
    print(json.dumps(summary_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
