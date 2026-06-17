#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.local_conflict_graph_common import compute_component_degree_features, solve_assignment_with_private_defer
from scripts.build_gt_pseudotrack_groups import (  # noqa: E402
    _assign_detections_to_gt,
    _history_file_name,
    _read_gt_rows,
    _read_seqinfo,
    _split_gt_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build graph-association local-conflict commit training data.")
    parser.add_argument("--rows-jsonl", default="")
    parser.add_argument("--source-manifest", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", default="MOT20", choices=["MOT17", "MOT20", "DanceTrack"])
    parser.add_argument("--data-root", default="/gemini/code/datasets")
    parser.add_argument("--split", default="train")
    parser.add_argument("--split-part", default="val_half", choices=["full", "train_half", "val_half"])
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--min-detections", type=int, default=2)
    parser.add_argument("--min-positive-matches", type=int, default=1)
    parser.add_argument("--max-detections", type=int, default=8)
    parser.add_argument("--max-tracks", type=int, default=32)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--train-sequences", default="")
    parser.add_argument("--val-sequences", default="")
    parser.add_argument("--strict-sequence-split", action="store_true")
    parser.add_argument("--feature-version", default="graph_assoc_v1")
    parser.add_argument("--dataset-tag", default="graph_assoc_commit")
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


def _safe_int_list(values: Any) -> list[int]:
    if not isinstance(values, (list, tuple)):
        return []
    out: list[int] = []
    for value in values:
        try:
            out.append(int(float(value)))
        except Exception:
            continue
    return out


def _safe_pair_list(values: Any) -> list[list[int]]:
    if not isinstance(values, (list, tuple)):
        return []
    out: list[list[int]] = []
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        out.append([_safe_int(value[0], -1), _safe_int(value[1], -1)])
    return out


def _action_from_gain(gt_gain: int) -> tuple[int, str]:
    if int(gt_gain) > 0:
        return 0, "rewrite"
    if int(gt_gain) < 0:
        return 2, "reject"
    return 1, "defer"


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
            rows_jsonl_raw = str(row.get("rows_jsonl", "")).strip()
            if not rows_jsonl_raw:
                raise ValueError(f"Manifest row {idx} is missing rows_jsonl")
            rows_jsonl = Path(rows_jsonl_raw)
            if not rows_jsonl.is_absolute():
                rows_jsonl = (manifest_path.parent / rows_jsonl).resolve()
            sources.append(
                {
                    "rows_jsonl": str(rows_jsonl),
                    "source_tag": str(row.get("source_tag", "")).strip() or f"source_{idx:03d}",
                    "host_variant": str(row.get("host_variant", "")).strip() or "unknown",
                    "split_tag": str(row.get("split_tag", "")).strip(),
                    "dataset": str(row.get("dataset", "")).strip() or str(args.dataset),
                    "data_root": str(row.get("data_root", "")).strip() or str(args.data_root),
                    "split": str(row.get("split", "")).strip() or str(args.split),
                    "split_part": str(row.get("split_part", "")).strip() or str(args.split_part),
                    "seq_name": str(row.get("seq_name", "")).strip(),
                    "dataset_tag": str(row.get("dataset_tag", "")).strip() or str(args.dataset_tag),
                    "feature_version": str(row.get("feature_version", "")).strip() or str(args.feature_version),
                }
            )
        return sources, str(manifest_path)

    rows_jsonl = Path(str(args.rows_jsonl or "")).resolve()
    if not str(args.rows_jsonl or "").strip():
        raise ValueError("Use either --source-manifest or --rows-jsonl")
    return (
        [
            {
                "rows_jsonl": str(rows_jsonl),
                "source_tag": "source_000",
                "host_variant": "unknown",
                "split_tag": "",
                "dataset": str(args.dataset),
                "data_root": str(args.data_root),
                "split": str(args.split),
                "split_part": str(args.split_part),
                "seq_name": "",
                "dataset_tag": str(args.dataset_tag),
                "feature_version": str(args.feature_version),
            }
        ],
        "",
    )


def _strip_known_suffixes(name: str) -> str:
    out = str(name)
    for suffix in [".jsonl.partial", ".jsonl", ".partial"]:
        if out.endswith(suffix):
            out = out[: -len(suffix)]
    for suffix in ["_candidates", "_events"]:
        if out.endswith(suffix):
            out = out[: -len(suffix)]
    return out


def _infer_seq_name(source: dict[str, str], row: dict[str, Any]) -> str:
    seq_name = str(source.get("seq_name", "")).strip()
    if seq_name:
        return seq_name
    row_seq = str(row.get("seq_name", "")).strip()
    if row_seq:
        return row_seq
    return _strip_known_suffixes(Path(str(source["rows_jsonl"])).name)


def _start_frame_for_half_val(seq_length: int) -> int:
    return int(seq_length) // 2 + 2


def _remap_frame_for_split(frame_id: int, split_part: str, seq_length: int) -> int | None:
    frame_id = int(frame_id)
    seq_length = int(seq_length)
    if frame_id <= 0 or seq_length <= 0:
        return None
    split = str(split_part or "").strip().lower()
    if split in {"", "full"}:
        return frame_id if frame_id <= seq_length else None

    start_frame = _start_frame_for_half_val(seq_length)
    train_half_length = max(start_frame - 1, 0)
    val_half_length = max(seq_length - start_frame + 1, 0)

    if split == "train_half":
        if 1 <= frame_id <= train_half_length:
            return frame_id
        return None

    if split == "val_half":
        if 1 <= frame_id <= val_half_length:
            return frame_id
        if start_frame <= frame_id <= seq_length:
            return frame_id - start_frame + 1
        return None

    return frame_id if frame_id <= seq_length else None


def _tlwh_to_tlbr(tlwh: Any) -> np.ndarray:
    arr = np.asarray(tlwh, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 4:
        raise ValueError(f"Invalid tlwh: {tlwh}")
    x, y, w, h = [float(v) for v in arr[:4]]
    return np.asarray([x, y, x + w, y + h], dtype=np.float32)


def _maybe_denormalize_tlbr(tlbr: np.ndarray, seq_w: float, seq_h: float) -> np.ndarray:
    arr = np.asarray(tlbr, dtype=np.float32).copy()
    if arr.size != 4:
        return np.zeros((4,), dtype=np.float32)
    if float(np.max(np.abs(arr))) <= 2.0:
        arr[[0, 2]] *= float(seq_w)
        arr[[1, 3]] *= float(seq_h)
    return arr


def _match_single_box(
    *,
    tlwh: Any,
    gt_positive: list[dict[str, float]],
    gt_ignore: list[dict[str, float]],
    iou_pos: float,
    iou_ignore: float,
    seq_w: float,
    seq_h: float,
) -> tuple[int, int]:
    tlbr = _maybe_denormalize_tlbr(_tlwh_to_tlbr(tlwh), seq_w=seq_w, seq_h=seq_h)
    assigned, ignore = _assign_detections_to_gt(
        det_tlbrs=np.asarray([tlbr], dtype=np.float32),
        gt_positive=gt_positive,
        gt_ignore=gt_ignore,
        iou_pos=float(iou_pos),
        iou_ignore=float(iou_ignore),
    )
    return int(assigned[0]), int(ignore[0])


def _score_entropy(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float32)
    arr = arr - float(np.max(arr))
    prob = np.exp(arr)
    prob = prob / np.clip(float(np.sum(prob)), 1e-8, None)
    return float(-(prob * np.log(np.clip(prob, 1e-8, None))).sum())


def _pair_local_map(pairs: list[list[int]] | list[tuple[int, int]], col_to_local: dict[int, int], row_to_local: dict[int, int]) -> dict[int, int]:
    local_map: dict[int, int] = {}
    for pair in pairs:
        if len(pair) < 2:
            continue
        row_idx = _safe_int(pair[0], -1)
        col_idx = _safe_int(pair[1], -1)
        if row_idx not in row_to_local or col_idx not in col_to_local:
            continue
        local_map[int(col_to_local[col_idx])] = int(row_to_local[row_idx])
    return local_map


def _infer_edge_rank_frac(edge_meta_rows: list[dict[str, Any]]) -> dict[tuple[int, int], float]:
    by_col: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in edge_meta_rows:
        col_idx = _safe_int(row.get("col_idx", -1), -1)
        if col_idx < 0:
            continue
        by_col[col_idx].append(row)
    rank_frac: dict[tuple[int, int], float] = {}
    for col_idx, rows in by_col.items():
        rows_sorted = sorted(
            rows,
            key=lambda item: (-_safe_float(item.get("utility", 0.0), 0.0), _safe_float(item.get("cost", 1.0), 1.0)),
        )
        denom = max(len(rows_sorted), 1)
        for rank, row in enumerate(rows_sorted, 1):
            row_idx = _safe_int(row.get("row_idx", -1), -1)
            if row_idx < 0:
                continue
            rank_frac[(int(row_idx), int(col_idx))] = float(rank) / float(denom)
    return rank_frac


def _merge_required_edge_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    merged: dict[tuple[int, int], dict[str, Any]] = {}
    for key in ("edge_meta", "baseline_pair_meta", "chosen_pair_meta"):
        for edge in list(row.get(key, [])):
            row_idx = _safe_int(edge.get("row_idx", -1), -1)
            col_idx = _safe_int(edge.get("col_idx", -1), -1)
            if row_idx < 0 or col_idx < 0:
                continue
            merged[(int(row_idx), int(col_idx))] = dict(edge)
    return list(merged.values())


def _read_row_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_example(
    *,
    row: dict[str, Any],
    seq: str,
    source: dict[str, str],
    gt_rows: dict[int, list[dict[str, float]]],
    seq_w: float,
    seq_h: float,
    seq_length: int,
    split_part: str,
    topk: int,
    min_positive_matches: int,
) -> dict[str, Any] | None:
    frame_id = _safe_int(row.get("frame_id", 0), 0)
    frame_lookup = _remap_frame_for_split(frame_id, split_part=split_part, seq_length=seq_length)
    if frame_lookup is None:
        return None
    row_states = list(row.get("row_states", []))
    col_states = list(row.get("col_states", []))
    edge_meta_rows = _merge_required_edge_rows(row)
    if not row_states or not col_states or not edge_meta_rows:
        return None

    track_rows = [_safe_int(v.get("row_idx", -1), -1) for v in row_states]
    det_rows = [_safe_int(v.get("col_idx", -1), -1) for v in col_states]
    if any(v < 0 for v in track_rows) or any(v < 0 for v in det_rows):
        return None

    row_to_local = {int(row_idx): local_idx for local_idx, row_idx in enumerate(track_rows)}
    col_to_local = {int(col_idx): local_idx for local_idx, col_idx in enumerate(det_rows)}
    track_ids = [_safe_int(v.get("track_id", track_rows[idx]), track_rows[idx]) for idx, v in enumerate(row_states)]

    det_gt_map: dict[int, tuple[int, int]] = {}
    gt_positive_frame, gt_ignore_frame = _split_gt_frame(gt_rows.get(int(frame_lookup), []))
    for det in col_states:
        col_idx = _safe_int(det.get("col_idx", -1), -1)
        if col_idx < 0:
            continue
        det_gt_map[int(col_idx)] = _match_single_box(
            tlwh=det.get("tlwh", []),
            gt_positive=gt_positive_frame,
            gt_ignore=gt_ignore_frame,
            iou_pos=0.7,
            iou_ignore=0.5,
            seq_w=seq_w,
            seq_h=seq_h,
        )

    track_gt_map: dict[int, tuple[int, int]] = {}
    for track in row_states:
        row_idx = _safe_int(track.get("row_idx", -1), -1)
        if row_idx < 0:
            continue
        track_frame_full = _safe_int(track.get("frame_id", frame_id), frame_id)
        track_frame = _remap_frame_for_split(track_frame_full, split_part=split_part, seq_length=seq_length)
        if track_frame is None:
            track_gt_map[int(row_idx)] = (-1, 0)
            continue
        gt_positive, gt_ignore = _split_gt_frame(gt_rows.get(int(track_frame), []))
        track_gt_map[int(row_idx)] = _match_single_box(
            tlwh=track.get("tlwh", []),
            gt_positive=gt_positive,
            gt_ignore=gt_ignore,
            iou_pos=0.7,
            iou_ignore=0.5,
            seq_w=seq_w,
            seq_h=seq_h,
        )

    rank_frac = _infer_edge_rank_frac(edge_meta_rows)
    edge_features: list[list[float]] = []
    edge_det_index: list[int] = []
    edge_track_index: list[int] = []
    compact_edges: list[dict[str, Any]] = []
    utility_by_det: dict[int, list[float]] = defaultdict(list)
    positive_score_map: dict[tuple[int, int], float] = {}
    edge_count_per_track: Counter[int] = Counter()
    edge_count_per_det: Counter[int] = Counter()

    for edge in edge_meta_rows:
        row_idx = _safe_int(edge.get("row_idx", -1), -1)
        col_idx = _safe_int(edge.get("col_idx", -1), -1)
        if row_idx not in row_to_local or col_idx not in col_to_local:
            continue
        local_track_idx = int(row_to_local[row_idx])
        local_det_idx = int(col_to_local[col_idx])
        local_track = row_states[local_track_idx]
        utility = _safe_float(edge.get("utility", 0.0), 0.0)
        cost = _safe_float(edge.get("cost", 1.0), 1.0)
        box_iou = _safe_float(edge.get("box_iou", 0.0), 0.0)
        track_gap = _safe_float(local_track.get("gap", 0.0), 0.0)
        track_hist_len = _safe_float(local_track.get("tracklet_len", 0.0), 0.0)
        feat = [
            1.0 - cost,
            utility,
            box_iou,
            track_gap,
            track_hist_len,
            float(rank_frac.get((row_idx, col_idx), 1.0)),
        ]
        edge_features.append([float(x) for x in feat])
        edge_det_index.append(local_det_idx)
        edge_track_index.append(local_track_idx)
        edge_count_per_track[local_track_idx] += 1
        edge_count_per_det[local_det_idx] += 1
        utility_by_det[local_det_idx].append(float(utility))

        det_gt_id, det_ignore = det_gt_map.get(int(col_idx), (-1, 0))
        track_gt_id, track_ignore = track_gt_map.get(int(row_idx), (-1, 0))
        is_positive = int(
            det_gt_id > 0
            and track_gt_id > 0
            and not bool(det_ignore)
            and not bool(track_ignore)
            and int(det_gt_id) == int(track_gt_id)
        )
        if is_positive:
            positive_score_map[(local_det_idx, local_track_idx)] = max(
                positive_score_map.get((local_det_idx, local_track_idx), -1e6),
                float(utility),
            )

        compact_edges.append(
            {
                "det_local_idx": int(local_det_idx),
                "track_local_idx": int(local_track_idx),
                "features": [float(x) for x in feat],
                "is_positive": int(is_positive),
            }
        )

    degree = compute_component_degree_features(
        num_detections=len(col_states),
        num_tracks=len(row_states),
        edge_det_index=edge_det_index,
        edge_track_index=edge_track_index,
    )
    row_degree = degree["row_degree"].tolist()
    col_degree = degree["col_degree"].tolist()

    det_features: list[list[float]] = []
    for local_det_idx, det in enumerate(col_states):
        utilities = sorted(utility_by_det.get(local_det_idx, []), reverse=True)
        top_margin = 0.0
        if len(utilities) == 1:
            top_margin = float(utilities[0])
        elif len(utilities) >= 2:
            top_margin = float(utilities[0] - utilities[1])
        det_features.append(
            [
                _safe_float(det.get("score", 0.0), 0.0),
                float(row_degree[local_det_idx]) if local_det_idx < len(row_degree) else 0.0,
                float(top_margin),
                float(_score_entropy(utilities)),
            ]
        )

    track_features: list[list[float]] = []
    for local_track_idx, track in enumerate(row_states):
        track_features.append(
            [
                _safe_float(track.get("gap", 0.0), 0.0),
                _safe_float(track.get("tracklet_len", 0.0), 0.0),
                float(col_degree[local_track_idx]) if local_track_idx < len(col_degree) else float(edge_count_per_track.get(local_track_idx, 0)),
            ]
        )

    row_degree_tensor = torch.tensor(row_degree, dtype=torch.float32)
    col_degree_tensor = torch.tensor(col_degree, dtype=torch.float32)
    cluster_features = [
        float(len(col_states)),
        float(len(row_states)),
        float(len(edge_features)),
        float(row_degree_tensor.mean().item()) if row_degree_tensor.numel() > 0 else 0.0,
        float(row_degree_tensor.max().item()) if row_degree_tensor.numel() > 0 else 0.0,
        float(col_degree_tensor.mean().item()) if col_degree_tensor.numel() > 0 else 0.0,
        float(col_degree_tensor.max().item()) if col_degree_tensor.numel() > 0 else 0.0,
        float(_safe_float(row.get("utility_gain", 0.0), 0.0)),
        float(_safe_float(row.get("cost_delta", 0.0), 0.0)),
        float(_safe_float(row.get("baseline_cost", 0.0), 0.0)),
        float(_safe_float(row.get("chosen_cost", 0.0), 0.0)),
        float(len(_safe_int_list(row.get("introduced_rows", [])))),
        float(len(_safe_int_list(row.get("suppressed_rows", [])))),
        float(len(_safe_int_list(row.get("recent_owner_rows", [])))),
        float(len(_safe_int_list(row.get("protected_young_active_rows", [])))),
    ]

    score_sub = torch.zeros((len(col_states), len(row_states)), dtype=torch.float32)
    feasible_mask = torch.zeros_like(score_sub, dtype=torch.bool)
    for (local_det_idx, local_track_idx), score in positive_score_map.items():
        feasible_mask[local_det_idx, local_track_idx] = True
        score_sub[local_det_idx, local_track_idx] = float(score)

    assignments = solve_assignment_with_private_defer(
        score_sub=score_sub,
        feasible_mask=feasible_mask,
        defer_scores=torch.zeros((len(col_states),), dtype=torch.float32),
        use_hungarian=True,
    )
    matched_count = sum(1 for item in assignments if item.get("track_local_idx", None) is not None)
    target_by_det = [-1 for _ in col_states]
    for assignment in assignments:
        det_local_idx = _safe_int(assignment.get("det_local_idx", -1), -1)
        track_local_idx = assignment.get("track_local_idx", None)
        if det_local_idx < 0 or track_local_idx is None:
            continue
        target_by_det[int(det_local_idx)] = int(track_local_idx)

    baseline_local = _pair_local_map(list(row.get("baseline_pairs", [])), col_to_local=col_to_local, row_to_local=row_to_local)
    chosen_local = _pair_local_map(list(row.get("chosen_pairs", [])), col_to_local=col_to_local, row_to_local=row_to_local)
    baseline_tp = 0
    chosen_tp = 0
    for det_local_idx, track_local_idx in baseline_local.items():
        if 0 <= det_local_idx < len(col_states) and 0 <= track_local_idx < len(row_states):
            det_gt_id, det_ignore = det_gt_map.get(int(det_rows[det_local_idx]), (-1, 0))
            track_gt_id, track_ignore = track_gt_map.get(int(track_rows[track_local_idx]), (-1, 0))
            if det_gt_id > 0 and track_gt_id > 0 and not bool(det_ignore) and not bool(track_ignore) and int(det_gt_id) == int(track_gt_id):
                baseline_tp += 1
    for det_local_idx, track_local_idx in chosen_local.items():
        if 0 <= det_local_idx < len(col_states) and 0 <= track_local_idx < len(row_states):
            det_gt_id, det_ignore = det_gt_map.get(int(det_rows[det_local_idx]), (-1, 0))
            track_gt_id, track_ignore = track_gt_map.get(int(track_rows[track_local_idx]), (-1, 0))
            if det_gt_id > 0 and track_gt_id > 0 and not bool(det_ignore) and not bool(track_ignore) and int(det_gt_id) == int(track_gt_id):
                chosen_tp += 1

    gt_gain = int(chosen_tp - baseline_tp)
    if gt_gain > 0:
        gt_decision = "positive"
    elif gt_gain < 0:
        gt_decision = "negative"
    else:
        gt_decision = "neutral"
    action_label, action_name = _action_from_gain(gt_gain)

    accepted = bool(row.get("accepted", row.get("decision", "") == "accepted"))
    introduced_rows = _safe_int_list(row.get("introduced_rows", []))
    active_introduced_rows = _safe_int_list(row.get("active_introduced_rows", []))
    suppressed_rows = _safe_int_list(row.get("suppressed_rows", []))
    recent_owner_rows = _safe_int_list(row.get("recent_owner_rows", []))
    protected_young_active_rows = _safe_int_list(row.get("protected_young_active_rows", []))
    baseline_pairs = _safe_pair_list(row.get("baseline_pairs", []))
    chosen_pairs = _safe_pair_list(row.get("chosen_pairs", []))
    return {
        "cluster_id": f"{source['source_tag']}|{seq}:{frame_id}:block{_safe_int(row.get('block_id', 0), 0)}",
        "seq": seq,
        "frame": int(frame_id),
        "frame_lookup": int(frame_lookup),
        "block_id": int(_safe_int(row.get("block_id", 0), 0)),
        "source_tag": str(source["source_tag"]),
        "host_variant": str(source["host_variant"]),
        "split_tag": str(source["split_tag"]),
        "feature_version": str(source["feature_version"]),
        "dataset_tag": str(source["dataset_tag"]),
        "source_rows_jsonl": str(source["rows_jsonl"]),
        "det_rows": [int(v) for v in det_rows],
        "track_ids": [int(v) for v in track_ids],
        "det_features": [[float(x) for x in feat] for feat in det_features],
        "track_features": [[float(x) for x in feat] for feat in track_features],
        "edge_features": [[float(x) for x in feat] for feat in edge_features],
        "edge_det_index": [int(v) for v in edge_det_index],
        "edge_track_index": [int(v) for v in edge_track_index],
        "cluster_features": [float(x) for x in cluster_features],
        "target_by_det": [int(v) for v in target_by_det],
        "target_committed_matches": int(matched_count),
        "trigger_pass": int(matched_count >= int(min_positive_matches)),
        "num_detections": int(len(col_states)),
        "num_tracks": int(len(row_states)),
        "num_edges": int(len(edge_features)),
        "compact_edges": compact_edges,
        "accepted": int(accepted),
        "decision": "accepted" if accepted else "rejected",
        "decision_source": str(row.get("decision_source", "")),
        "skip_reason": str(row.get("skip_reason", "")),
        "learned_commit_gate_only": int(bool(row.get("learned_commit_gate_only", False))),
        "learned_commit_replace_rules": int(bool(row.get("learned_commit_replace_rules", False))),
        "rules_passed_before_learned_gate": int(bool(row.get("rules_passed_before_learned_gate", False))),
        "utility_gain": float(_safe_float(row.get("utility_gain", 0.0), 0.0)),
        "cost_delta": float(_safe_float(row.get("cost_delta", 0.0), 0.0)),
        "baseline_cost": float(_safe_float(row.get("baseline_cost", 0.0), 0.0)),
        "chosen_cost": float(_safe_float(row.get("chosen_cost", 0.0), 0.0)),
        "baseline_utility": float(_safe_float(row.get("baseline_utility", 0.0), 0.0)),
        "chosen_utility": float(_safe_float(row.get("chosen_utility", 0.0), 0.0)),
        "introduced_rows": introduced_rows,
        "active_introduced_rows": active_introduced_rows,
        "suppressed_rows": suppressed_rows,
        "recent_owner_rows": recent_owner_rows,
        "protected_young_active_rows": protected_young_active_rows,
        "introduced_row_count": int(len(introduced_rows)),
        "active_introduced_row_count": int(len(active_introduced_rows)),
        "suppressed_row_count": int(len(suppressed_rows)),
        "recent_owner_row_count": int(len(recent_owner_rows)),
        "protected_young_active_row_count": int(len(protected_young_active_rows)),
        "baseline_true_matches": int(baseline_tp),
        "chosen_true_matches": int(chosen_tp),
        "gt_gain": int(gt_gain),
        "gt_decision": gt_decision,
        "action_label": int(action_label),
        "action_name": action_name,
        "baseline_pairs": baseline_pairs,
        "chosen_pairs": chosen_pairs,
        "baseline_pairs_local": {str(k): int(v) for k, v in baseline_local.items()},
        "chosen_pairs_local": {str(k): int(v) for k, v in chosen_local.items()},
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

    gt_cache: dict[tuple[str, str, str, str, str], tuple[dict[int, list[dict[str, float]]], float, float]] = {}
    summary_counter: Counter[str] = Counter()
    split_counter: dict[str, Counter[str]] = defaultdict(Counter)
    seq_counter: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    cluster_rows: list[dict[str, Any]] = []
    cluster_examples: list[dict[str, Any]] = []
    seen_sequences: set[str] = set()
    seen_sources: set[str] = set()

    for source in sources:
        rows_path = Path(source["rows_jsonl"]).resolve()
        if not rows_path.is_file():
            raise FileNotFoundError(f"Missing rows jsonl: {rows_path}")
        source_rows = _read_row_jsonl(rows_path)
        summary_counter["sources"] += 1
        summary_counter["source_rows"] += int(len(source_rows))
        seen_sources.add(str(source["source_tag"]))

        for row in source_rows:
            seq = _infer_seq_name(source, row)
            split_tag = _determine_split_tag(
                seq=seq,
                explicit_split_tag=str(source.get("split_tag", "")),
                train_tokens=train_tokens,
                val_tokens=val_tokens,
                strict_sequence_split=bool(args.strict_sequence_split),
            )
            if split_tag == "unused":
                summary_counter["unused_rows"] += 1
                continue

            key = (
                str(source["dataset"]),
                str(source["data_root"]),
                str(source["split"]),
                str(source["split_part"]),
                str(seq),
            )
            if key not in gt_cache:
                seq_dir = Path(str(source["data_root"])) / str(source["dataset"]) / str(source["split"]) / str(seq)
                gt_path = _history_file_name(seq_dir, "gt", str(source["split_part"]))
                gt_rows = _read_gt_rows(gt_path)
                seqinfo = _read_seqinfo(seq_dir)
                seq_length = int(seqinfo.get("seqLength", seqinfo.get("seqlength")))
                seq_w = float(seqinfo.get("imWidth", seqinfo.get("imwidth")))
                seq_h = float(seqinfo.get("imHeight", seqinfo.get("imheight")))
                gt_cache[key] = (gt_rows, seq_w, seq_h, seq_length)
            gt_rows, seq_w, seq_h, seq_length = gt_cache[key]

            example = _build_example(
                row=row,
                seq=seq,
                source=source,
                gt_rows=gt_rows,
                seq_w=seq_w,
                seq_h=seq_h,
                seq_length=seq_length,
                split_part=str(source["split_part"]),
                topk=int(args.topk),
                min_positive_matches=int(args.min_positive_matches),
            )
            if example is None:
                summary_counter["skipped_missing_fields"] += 1
                split_counter[split_tag]["skipped_missing_fields"] += 1
                continue
            if int(example["num_detections"]) < int(args.min_detections):
                summary_counter["skipped_small_clusters"] += 1
                split_counter[split_tag]["skipped_small_clusters"] += 1
                continue
            if int(example["num_detections"]) > int(args.max_detections) or int(example["num_tracks"]) > int(args.max_tracks):
                summary_counter["skipped_large_clusters"] += 1
                split_counter[split_tag]["skipped_large_clusters"] += 1
                continue

            example["split_tag"] = split_tag
            cluster_examples.append(example)
            seen_sequences.add(seq)
            summary_counter["clusters"] += 1
            summary_counter["detections"] += int(example["num_detections"])
            summary_counter["tracks"] += int(example["num_tracks"])
            summary_counter["edges"] += int(example["num_edges"])
            summary_counter["trigger_pass_clusters"] += int(example["trigger_pass"])
            summary_counter[f"decision_{example['decision']}"] += 1
            summary_counter[f"gt_{example['gt_decision']}"] += 1
            summary_counter[f"action_{example['action_name']}"] += 1
            if int(example["target_committed_matches"]) == 0:
                summary_counter["background_clusters"] += 1
            split_counter[split_tag]["clusters"] += 1
            split_counter[split_tag]["detections"] += int(example["num_detections"])
            split_counter[split_tag]["tracks"] += int(example["num_tracks"])
            split_counter[split_tag]["edges"] += int(example["num_edges"])
            split_counter[split_tag]["trigger_pass_clusters"] += int(example["trigger_pass"])
            split_counter[split_tag][f"decision_{example['decision']}"] += 1
            split_counter[split_tag][f"gt_{example['gt_decision']}"] += 1
            split_counter[split_tag][f"action_{example['action_name']}"] += 1
            seq_counter[(seq, split_tag)]["clusters"] += 1
            seq_counter[(seq, split_tag)]["detections"] += int(example["num_detections"])
            seq_counter[(seq, split_tag)]["tracks"] += int(example["num_tracks"])
            seq_counter[(seq, split_tag)]["edges"] += int(example["num_edges"])
            seq_counter[(seq, split_tag)]["trigger_pass_clusters"] += int(example["trigger_pass"])
            seq_counter[(seq, split_tag)][f"decision_{example['decision']}"] += 1
            seq_counter[(seq, split_tag)][f"gt_{example['gt_decision']}"] += 1
            seq_counter[(seq, split_tag)][f"action_{example['action_name']}"] += 1

            cluster_rows.append(
                {
                    "cluster_id": example["cluster_id"],
                    "seq": seq,
                    "frame": int(example["frame"]),
                    "source_tag": str(example["source_tag"]),
                    "host_variant": str(example["host_variant"]),
                    "split_tag": str(split_tag),
                    "num_detections": int(example["num_detections"]),
                    "num_tracks": int(example["num_tracks"]),
                    "num_edges": int(example["num_edges"]),
                    "target_committed_matches": int(example["target_committed_matches"]),
                    "trigger_pass": int(example["trigger_pass"]),
                    "decision": str(example["decision"]),
                    "decision_source": str(example.get("decision_source", "")),
                    "rules_passed_before_learned_gate": int(example.get("rules_passed_before_learned_gate", 0)),
                    "gt_decision": str(example["gt_decision"]),
                    "gt_gain": int(example["gt_gain"]),
                    "action_name": str(example["action_name"]),
                    "action_label": int(example["action_label"]),
                    "skip_reason": str(example["skip_reason"]),
                }
            )

    with cluster_jsonl.open("w", encoding="utf-8") as f:
        for row in cluster_examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with cluster_sample_jsonl.open("w", encoding="utf-8") as f:
        for row in cluster_examples[: max(int(args.sample_size), 0)]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if cluster_rows:
        with cluster_summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(cluster_rows[0].keys()))
            writer.writeheader()
            writer.writerows(cluster_rows)

    seq_rows: list[dict[str, Any]] = []
    for (seq, split_tag), counts in sorted(seq_counter.items(), key=lambda item: (item[0][0], item[0][1])):
        seq_rows.append(
            {
                "seq": seq,
                "split_tag": split_tag,
                "clusters": int(counts.get("clusters", 0)),
                "detections": int(counts.get("detections", 0)),
                "tracks": int(counts.get("tracks", 0)),
                "edges": int(counts.get("edges", 0)),
                "trigger_pass_clusters": int(counts.get("trigger_pass_clusters", 0)),
                "accepted_clusters": int(counts.get("decision_accepted", 0)),
                "rejected_clusters": int(counts.get("decision_rejected", 0)),
                "gt_positive_clusters": int(counts.get("gt_positive", 0)),
                "gt_negative_clusters": int(counts.get("gt_negative", 0)),
                "gt_neutral_clusters": int(counts.get("gt_neutral", 0)),
                "action_rewrite_clusters": int(counts.get("action_rewrite", 0)),
                "action_defer_clusters": int(counts.get("action_defer", 0)),
                "action_reject_clusters": int(counts.get("action_reject", 0)),
            }
        )
    if seq_rows:
        with seq_summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(seq_rows[0].keys()))
            writer.writeheader()
            writer.writerows(seq_rows)

    summary_row = {
        "dataset_tag": str(args.dataset_tag),
        "feature_version": str(args.feature_version),
        "source_manifest": manifest_path,
        "rows_jsonl": str(Path(args.rows_jsonl).resolve()) if str(args.rows_jsonl or "").strip() else "",
        "sources": int(summary_counter.get("sources", 0)),
        "source_rows": int(summary_counter.get("source_rows", 0)),
        "clusters": int(summary_counter.get("clusters", 0)),
        "detections": int(summary_counter.get("detections", 0)),
        "tracks": int(summary_counter.get("tracks", 0)),
        "edges": int(summary_counter.get("edges", 0)),
        "trigger_pass_clusters": int(summary_counter.get("trigger_pass_clusters", 0)),
        "accepted_clusters": int(summary_counter.get("decision_accepted", 0)),
        "rejected_clusters": int(summary_counter.get("decision_rejected", 0)),
        "gt_positive_clusters": int(summary_counter.get("gt_positive", 0)),
        "gt_negative_clusters": int(summary_counter.get("gt_negative", 0)),
        "gt_neutral_clusters": int(summary_counter.get("gt_neutral", 0)),
        "action_rewrite_clusters": int(summary_counter.get("action_rewrite", 0)),
        "action_defer_clusters": int(summary_counter.get("action_defer", 0)),
        "action_reject_clusters": int(summary_counter.get("action_reject", 0)),
        "background_clusters": int(summary_counter.get("background_clusters", 0)),
        "unused_rows": int(summary_counter.get("unused_rows", 0)),
        "skipped_missing_fields": int(summary_counter.get("skipped_missing_fields", 0)),
        "skipped_small_clusters": int(summary_counter.get("skipped_small_clusters", 0)),
        "skipped_large_clusters": int(summary_counter.get("skipped_large_clusters", 0)),
        "train_sequences": ",".join(sorted({seq for (seq, split_tag) in seq_counter.keys() if split_tag == "train"})),
        "val_sequences": ",".join(sorted({seq for (seq, split_tag) in seq_counter.keys() if split_tag == "val"})),
        "all_sequences": ",".join(sorted(seen_sequences)),
        "all_sources": ",".join(sorted(seen_sources)),
        "status": "ok",
        "error": "",
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_row, f, indent=2, ensure_ascii=False)

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
