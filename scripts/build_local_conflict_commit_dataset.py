#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.local_conflict_commit import (
    CLUSTER_FEATURE_NAMES,
    DET_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
    TRACK_FEATURE_NAMES,
)
from models.local_conflict_graph_common import (
    build_group_components_from_group_rows,
    compute_component_degree_features,
    filter_local_conflict_clusters_by_size,
    solve_assignment_with_private_defer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cluster-level local-conflict commit training data.")
    parser.add_argument("--group-jsonl", default="")
    parser.add_argument("--cases-csv", default="")
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
    parser.add_argument("--feature-version", default="v1_raw")
    parser.add_argument("--dataset-tag", default="local_conflict_commit")
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


def _load_case_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {str(row["group_id"]): dict(row) for row in reader}


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
            group_jsonl_raw = str(row.get("group_jsonl", "")).strip()
            cases_csv_raw = str(row.get("cases_csv", "")).strip()
            if not group_jsonl_raw or not cases_csv_raw:
                raise ValueError(f"Manifest row {idx} is missing group_jsonl/cases_csv")
            group_jsonl = Path(group_jsonl_raw)
            if not group_jsonl.is_absolute():
                group_jsonl = (manifest_path.parent / group_jsonl).resolve()
            cases_csv = Path(cases_csv_raw)
            if not cases_csv.is_absolute():
                cases_csv = (manifest_path.parent / cases_csv).resolve()
            source_tag = str(row.get("source_tag", "")).strip() or f"source_{idx:03d}"
            sources.append(
                {
                    "group_jsonl": str(group_jsonl),
                    "cases_csv": str(cases_csv),
                    "host_variant": str(row.get("host_variant", "")).strip() or "unknown",
                    "source_tag": source_tag,
                    "split_tag": str(row.get("split_tag", "")).strip(),
                    "dataset_tag": str(row.get("dataset_tag", "")).strip() or str(args.dataset_tag),
                    "feature_version": str(row.get("feature_version", "")).strip() or str(args.feature_version),
                }
            )
        return sources, str(manifest_path)

    group_jsonl = Path(str(args.group_jsonl or "")).resolve()
    cases_csv = Path(str(args.cases_csv or "")).resolve()
    if not str(args.group_jsonl or "").strip() or not str(args.cases_csv or "").strip():
        raise ValueError("Use either --source-manifest or both --group-jsonl and --cases-csv")
    return (
        [
            {
                "group_jsonl": str(group_jsonl),
                "cases_csv": str(cases_csv),
                "host_variant": "unknown",
                "source_tag": "source_000",
                "split_tag": "",
                "dataset_tag": str(args.dataset_tag),
                "feature_version": str(args.feature_version),
            }
        ],
        "",
    )


def _sorted_candidates(group: dict[str, Any], topk: int) -> list[dict[str, Any]]:
    candidates = sorted(
        list(group.get("candidates", [])),
        key=lambda row: _safe_int(row.get("track_rank", 0), 0),
    )
    return candidates[: max(int(topk), 1)]


def _cluster_example(
    *,
    seq: str,
    frame: int,
    component: dict[str, Any],
    frame_groups: list[dict[str, Any]],
    case_rows: dict[str, dict[str, str]],
    topk: int,
    min_committed_matches: int,
    source_tag: str,
    host_variant: str,
    split_tag: str,
    feature_version: str,
    source_group_jsonl: str,
    source_cases_csv: str,
) -> dict[str, Any] | None:
    group_by_det = {_safe_int(group.get("det_index", -1), -1): group for group in frame_groups}
    det_rows = [int(x) for x in component.get("det_rows", [])]
    track_ids = [int(x) for x in component.get("track_ids", [])]
    if not det_rows or not track_ids:
        return None

    track_to_local = {track_id: idx for idx, track_id in enumerate(track_ids)}
    det_features: list[list[float]] = []
    edge_features: list[list[float]] = []
    edge_det_index: list[int] = []
    edge_track_index: list[int] = []
    positive_score_map: dict[tuple[int, int], float] = {}
    track_gap_acc: dict[int, list[float]] = defaultdict(list)
    track_hist_acc: dict[int, list[float]] = defaultdict(list)

    for local_det_idx, det_row in enumerate(det_rows):
        group = group_by_det.get(det_row)
        if group is None:
            return None
        group_id = str(group.get("group_id", ""))
        case = case_rows.get(group_id, {})
        rank_margin = _safe_float(group.get("rank_margin", case.get("rank_margin", 0.0)), 0.0)
        rank_entropy = _safe_float(group.get("rank_entropy", case.get("rank_entropy", 0.0)), 0.0)
        det_score = _safe_float(case.get("det_score", 0.0), 0.0)
        det_features.append([det_score, 0.0, rank_margin, rank_entropy])

        candidates = _sorted_candidates(group, topk)
        k = max(len(candidates), 1)
        for cand in candidates:
            if _safe_int(cand.get("valid_train_row", 1), 1) <= 0:
                continue
            track_id = _safe_int(cand.get("track_id", -1), -1)
            if track_id not in track_to_local:
                continue
            local_track_idx = int(track_to_local[track_id])
            track_gap = _safe_float(cand.get("track_gap", 0.0), 0.0)
            track_hist_len = _safe_float(cand.get("track_hist_len", 0.0), 0.0)
            track_rank = max(_safe_int(cand.get("track_rank", 1), 1), 1)
            edge_det_index.append(local_det_idx)
            edge_track_index.append(local_track_idx)
            edge_features.append(
                [
                    _safe_float(cand.get("base_score", 0.0), 0.0),
                    _safe_float(cand.get("refined_score", 0.0), 0.0),
                    _safe_float(cand.get("motion_score", 0.0), 0.0),
                    track_gap,
                    track_hist_len,
                    float(track_rank) / float(k),
                ]
            )
            track_gap_acc[local_track_idx].append(track_gap)
            track_hist_acc[local_track_idx].append(track_hist_len)
            if _safe_int(cand.get("label", 0), 0) == 1:
                key = (local_det_idx, local_track_idx)
                positive_score_map[key] = max(
                    positive_score_map.get(key, -1e6),
                    _safe_float(cand.get("refined_score", 0.0), 0.0),
                )

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

    track_features: list[list[float]] = []
    for local_track_idx in range(len(track_ids)):
        gaps = track_gap_acc.get(local_track_idx, [])
        hists = track_hist_acc.get(local_track_idx, [])
        gap_mean = float(sum(gaps) / float(len(gaps))) if gaps else 0.0
        hist_mean = float(sum(hists) / float(len(hists))) if hists else 0.0
        col_deg = float(col_degree[local_track_idx]) if local_track_idx < len(col_degree) else 0.0
        track_features.append([gap_mean, hist_mean, col_deg])

    num_edges = len(edge_features)
    row_degree_tensor = torch.tensor(row_degree, dtype=torch.float32)
    col_degree_tensor = torch.tensor(col_degree, dtype=torch.float32)
    cluster_features = [
        float(len(det_rows)),
        float(len(track_ids)),
        float(num_edges),
        float(row_degree_tensor.mean().item()) if row_degree_tensor.numel() > 0 else 0.0,
        float(row_degree_tensor.max().item()) if row_degree_tensor.numel() > 0 else 0.0,
        float(col_degree_tensor.mean().item()) if col_degree_tensor.numel() > 0 else 0.0,
        float(col_degree_tensor.max().item()) if col_degree_tensor.numel() > 0 else 0.0,
    ]

    score_sub = torch.zeros((len(det_rows), len(track_ids)), dtype=torch.float32)
    feasible_mask = torch.zeros_like(score_sub, dtype=torch.bool)
    for (local_det_idx, local_track_idx), score in positive_score_map.items():
        feasible_mask[local_det_idx, local_track_idx] = True
        score_sub[local_det_idx, local_track_idx] = float(score)

    assignments = solve_assignment_with_private_defer(
        score_sub=score_sub,
        feasible_mask=feasible_mask,
        defer_scores=torch.zeros((len(det_rows),), dtype=torch.float32),
        use_hungarian=True,
    )
    matched_count = sum(1 for row in assignments if row.get("track_local_idx", None) is not None)
    trigger_pass = int(matched_count >= int(min_committed_matches))
    target_by_det = [-1 for _ in det_rows]
    if trigger_pass:
        for assignment in assignments:
            det_local_idx = int(assignment.get("det_local_idx", -1))
            track_local_idx = assignment.get("track_local_idx", None)
            if track_local_idx is None:
                continue
            target_by_det[det_local_idx] = int(track_local_idx)

    compact_edges = []
    for det_idx, track_idx, feat in zip(edge_det_index, edge_track_index, edge_features):
        compact_edges.append(
            {
                "det_local_idx": int(det_idx),
                "track_local_idx": int(track_idx),
                "features": [float(x) for x in feat],
                "is_positive": int((int(det_idx), int(track_idx)) in positive_score_map),
            }
        )

    return {
        "cluster_id": f"{source_tag}|{seq}:{frame}:{'-'.join(str(x) for x in det_rows)}",
        "seq": seq,
        "frame": int(frame),
        "source_tag": source_tag,
        "host_variant": host_variant,
        "split_tag": split_tag,
        "feature_version": feature_version,
        "source_group_jsonl": source_group_jsonl,
        "source_cases_csv": source_cases_csv,
        "det_rows": det_rows,
        "track_ids": track_ids,
        "det_features": [[float(x) for x in row] for row in det_features],
        "track_features": [[float(x) for x in row] for row in track_features],
        "edge_features": [[float(x) for x in row] for row in edge_features],
        "edge_det_index": [int(x) for x in edge_det_index],
        "edge_track_index": [int(x) for x in edge_track_index],
        "cluster_features": [float(x) for x in cluster_features],
        "target_by_det": [int(x) for x in target_by_det],
        "target_committed_matches": int(matched_count),
        "trigger_pass": trigger_pass,
        "num_detections": int(len(det_rows)),
        "num_tracks": int(len(track_ids)),
        "num_edges": int(num_edges),
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
        group_jsonl = Path(source["group_jsonl"]).resolve()
        cases_csv = Path(source["cases_csv"]).resolve()
        if not group_jsonl.is_file():
            raise FileNotFoundError(f"Missing group jsonl: {group_jsonl}")
        if not cases_csv.is_file():
            raise FileNotFoundError(f"Missing cases csv: {cases_csv}")

        case_rows = _load_case_rows(cases_csv)
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
            summary_counter["frames"] += 1
            summary_counter["raw_components"] += int(len(components))
            summary_counter["skipped_large_clusters"] += int(skipped_large)
            split_counter[split_tag]["frames"] += 1
            split_counter[split_tag]["raw_components"] += int(len(components))
            split_counter[split_tag]["skipped_large_clusters"] += int(skipped_large)
            if eligible_components:
                summary_counter["frames_with_eligible_clusters"] += 1
                split_counter[split_tag]["frames_with_eligible_clusters"] += 1
                seq_counter[(seq, split_tag)]["frames_with_eligible_clusters"] += 1

            seen_sequences.add(seq)
            for component in eligible_components:
                example = _cluster_example(
                    seq=seq,
                    frame=int(frame),
                    component=component,
                    frame_groups=frame_groups,
                    case_rows=case_rows,
                    topk=int(args.topk),
                    min_committed_matches=int(args.min_committed_matches),
                    source_tag=source_tag,
                    host_variant=host_variant,
                    split_tag=split_tag,
                    feature_version=feature_version,
                    source_group_jsonl=str(group_jsonl),
                    source_cases_csv=str(cases_csv),
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
                    "num_detections": example["num_detections"],
                    "num_tracks": example["num_tracks"],
                    "num_edges": example["num_edges"],
                    "target_committed_matches": example["target_committed_matches"],
                    "trigger_pass": example["trigger_pass"],
                }
                cluster_rows.append(row)
                summary_counter["eligible_clusters"] += 1
                summary_counter["detections"] += int(example["num_detections"])
                summary_counter["tracks"] += int(example["num_tracks"])
                summary_counter["edges"] += int(example["num_edges"])
                summary_counter["trigger_pass_clusters"] += int(example["trigger_pass"])
                summary_counter["trigger_fail_clusters"] += int(1 - int(example["trigger_pass"]))
                summary_counter["committed_matches"] += int(example["target_committed_matches"])
                split_counter[split_tag]["eligible_clusters"] += 1
                split_counter[split_tag]["detections"] += int(example["num_detections"])
                split_counter[split_tag]["tracks"] += int(example["num_tracks"])
                split_counter[split_tag]["edges"] += int(example["num_edges"])
                split_counter[split_tag]["trigger_pass_clusters"] += int(example["trigger_pass"])
                split_counter[split_tag]["committed_matches"] += int(example["target_committed_matches"])
                seq_counter[(seq, split_tag)]["eligible_clusters"] += 1
                seq_counter[(seq, split_tag)]["detections"] += int(example["num_detections"])
                seq_counter[(seq, split_tag)]["tracks"] += int(example["num_tracks"])
                seq_counter[(seq, split_tag)]["edges"] += int(example["num_edges"])
                seq_counter[(seq, split_tag)]["trigger_pass_clusters"] += int(example["trigger_pass"])
                seq_counter[(seq, split_tag)]["committed_matches"] += int(example["target_committed_matches"])
                seq_counter[(seq, split_tag)]["source_index"] = source_idx

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
    for (seq, split_tag), counter in sorted(seq_counter.items(), key=lambda item: (item[0][1], item[0][0])):
        eligible = max(int(counter["eligible_clusters"]), 1)
        seq_rows.append(
            {
                "seq": seq,
                "split_tag": split_tag,
                "eligible_clusters": int(counter["eligible_clusters"]),
                "frames_with_eligible_clusters": int(counter["frames_with_eligible_clusters"]),
                "detections": int(counter["detections"]),
                "tracks": int(counter["tracks"]),
                "edges": int(counter["edges"]),
                "trigger_pass_clusters": int(counter["trigger_pass_clusters"]),
                "committed_matches": int(counter["committed_matches"]),
                "avg_detections": float(counter["detections"]) / float(eligible),
                "avg_tracks": float(counter["tracks"]) / float(eligible),
                "avg_edges": float(counter["edges"]) / float(eligible),
            }
        )
    if seq_rows:
        with seq_summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(seq_rows[0].keys()))
            writer.writeheader()
            writer.writerows(seq_rows)

    eligible_clusters = max(int(summary_counter["eligible_clusters"]), 1)
    split_rows = {
        f"{split_tag}_examples": int(counter["eligible_clusters"])
        for split_tag, counter in sorted(split_counter.items())
    }
    split_rows.update(
        {
            f"{split_tag}_frames": int(counter["frames"])
            for split_tag, counter in sorted(split_counter.items())
        }
    )
    summary = {
        "dataset_tag": str(args.dataset_tag),
        "feature_version": str(args.feature_version),
        "source_manifest": manifest_path,
        "group_jsonl": str(args.group_jsonl or ""),
        "cases_csv": str(args.cases_csv or ""),
        "out_dir": str(out_dir),
        "topk": int(args.topk),
        "min_detections": int(args.min_detections),
        "min_committed_matches": int(args.min_committed_matches),
        "max_detections": int(args.max_detections),
        "max_tracks": int(args.max_tracks),
        "train_sequences": ",".join(train_tokens),
        "val_sequences": ",".join(val_tokens),
        "strict_sequence_split": int(bool(args.strict_sequence_split)),
        "det_feature_names": list(DET_FEATURE_NAMES),
        "track_feature_names": list(TRACK_FEATURE_NAMES),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "cluster_feature_names": list(CLUSTER_FEATURE_NAMES),
        "sources": int(summary_counter["sources"]),
        "source_tags": ",".join(sorted(seen_sources)),
        "host_variants": ",".join(sorted(seen_hosts)),
        "sequences": int(len(seen_sequences)),
        "frames": int(summary_counter["frames"]),
        "frames_with_eligible_clusters": int(summary_counter["frames_with_eligible_clusters"]),
        "unused_frames": int(summary_counter["unused_frames"]),
        "raw_components": int(summary_counter["raw_components"]),
        "eligible_clusters": int(summary_counter["eligible_clusters"]),
        "skipped_large_clusters": int(summary_counter["skipped_large_clusters"]),
        "trigger_pass_clusters": int(summary_counter["trigger_pass_clusters"]),
        "trigger_fail_clusters": int(summary_counter["trigger_fail_clusters"]),
        "committed_matches": int(summary_counter["committed_matches"]),
        "detections": int(summary_counter["detections"]),
        "tracks": int(summary_counter["tracks"]),
        "edges": int(summary_counter["edges"]),
        "avg_detections": float(summary_counter["detections"]) / float(eligible_clusters),
        "avg_tracks": float(summary_counter["tracks"]) / float(eligible_clusters),
        "avg_edges": float(summary_counter["edges"]) / float(eligible_clusters),
        "train_examples": int(split_counter["train"]["eligible_clusters"]),
        "val_examples": int(split_counter["val"]["eligible_clusters"]),
        "all_examples": int(split_counter["all"]["eligible_clusters"]),
        "status": "ok",
    }
    summary.update(split_rows)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_single_row_csv(summary_csv, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
