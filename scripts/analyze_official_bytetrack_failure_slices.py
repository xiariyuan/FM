#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.local_conflict_graph_common import (  # noqa: E402
    build_group_components_from_group_rows,
    filter_local_conflict_clusters_by_size,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Diagnose official ByteTrack first-stage failure slices.")
    ap.add_argument("--group-jsonl", required=True, help="Path to labeled replay group jsonl.")
    ap.add_argument(
        "--cluster-jsonl",
        default="",
        help="Optional cluster_examples.jsonl for current plugin dataset coverage/alignment analysis.",
    )
    ap.add_argument(
        "--rows-csv",
        default="",
        help="Optional labeled replay csv. When provided, det_score is merged into group-level analysis.",
    )
    ap.add_argument("--out-dir", required=True, help="Directory to write diagnostic artifacts.")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--min-detections", type=int, default=2)
    ap.add_argument("--max-detections", type=int, default=8)
    ap.add_argument("--max-tracks", type=int, default=32)
    ap.add_argument(
        "--train-sequences",
        default="",
        help="Optional comma-separated sequence list for train-split coverage analysis. "
        "If omitted and --cluster-jsonl is given, inferred from cluster examples.",
    )
    return ap.parse_args()


def _parse_csv_tokens(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [token.strip() for token in text.split(",") if token.strip()]


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


def _load_det_scores(rows_csv: Path) -> dict[str, float]:
    det_score_by_group: dict[str, float] = {}
    with rows_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group_id = str(row.get("group_id", "")).strip()
            if not group_id or group_id in det_score_by_group:
                continue
            det_score_by_group[group_id] = _safe_float(row.get("det_score", 0.0), 0.0)
    return det_score_by_group


def _load_group_records(group_jsonl: Path, det_score_by_group: dict[str, float] | None = None) -> tuple[pd.DataFrame, dict[tuple[str, int], list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    frame_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    with group_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            group = json.loads(line)
            seq = str(group.get("seq", ""))
            frame = _safe_int(group.get("frame", 0), 0)
            det_index = _safe_int(group.get("det_index", 0), 0)
            group_id = str(group.get("group_id", "")).strip()
            frame_groups[(seq, frame)].append(group)

            candidates = list(group.get("candidates", []))
            positive_candidate = next((cand for cand in candidates if _safe_int(cand.get("label", 0), 0) > 0), None)
            rows.append(
                {
                    "group_id": group_id,
                    "seq": seq,
                    "frame": frame,
                    "det_index": det_index,
                    "group_has_positive": _safe_int(group.get("group_has_positive", 0), 0),
                    "group_is_background": _safe_int(group.get("group_is_background", 0), 0),
                    "group_is_ambiguous": _safe_int(group.get("group_is_ambiguous", 0), 0),
                    "group_is_recoverable": _safe_int(group.get("group_is_recoverable", 0), 0),
                    "rank_top1_correct": _safe_int(group.get("rank_top1_correct", 0), 0),
                    "positive_rank": _safe_int(group.get("positive_rank", -1), -1),
                    "rank_margin": _safe_float(group.get("rank_margin", 0.0), 0.0),
                    "rank_entropy": _safe_float(group.get("rank_entropy", 0.0), 0.0),
                    "candidate_count_total": _safe_int(group.get("candidate_count_total", 0), 0),
                    "group_size": _safe_int(group.get("group_size", 0), 0),
                    "positive_gap": _safe_int(positive_candidate.get("track_gap", 0), 0) if positive_candidate else -1,
                    "positive_hist_len": _safe_int(positive_candidate.get("track_hist_len", 0), 0) if positive_candidate else -1,
                    "positive_base_score": _safe_float(positive_candidate.get("base_score", 0.0), 0.0) if positive_candidate else 0.0,
                    "positive_refined_score": _safe_float(positive_candidate.get("refined_score", 0.0), 0.0) if positive_candidate else 0.0,
                    "det_score": (
                        float(det_score_by_group.get(group_id, 0.0))
                        if det_score_by_group is not None
                        else None
                    ),
                }
            )
    df = pd.DataFrame(rows)
    df["top1_error"] = ((df["group_has_positive"] == 1) & (df["rank_top1_correct"] == 0)).astype(int)
    return df, frame_groups


def _quantile_slice_rows(df: pd.DataFrame, column: str, q: int = 5) -> list[dict[str, Any]]:
    valid = df[[column, "top1_error"]].dropna()
    if valid.empty or valid[column].nunique() < 2:
        return []
    try:
        buckets = pd.qcut(valid[column], q=q, duplicates="drop")
    except Exception:
        return []
    total_errors = int(valid["top1_error"].sum())
    grouped = valid.groupby(buckets, observed=False)["top1_error"].agg(["count", "sum", "mean"]).reset_index()
    rows: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        bucket = row[column]
        error_count = int(row["sum"])
        rows.append(
            {
                "slice_family": column,
                "slice_value": str(bucket),
                "group_count": int(row["count"]),
                "error_count": error_count,
                "error_rate": float(row["mean"]),
                "error_share": float(error_count / total_errors) if total_errors > 0 else 0.0,
            }
        )
    return rows


def _binary_slice_rows(df: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    valid = df[[column, "top1_error"]].dropna()
    if valid.empty:
        return []
    total_errors = int(valid["top1_error"].sum())
    grouped = valid.groupby(column)["top1_error"].agg(["count", "sum", "mean"]).reset_index()
    rows: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        error_count = int(row["sum"])
        rows.append(
            {
                "slice_family": column,
                "slice_value": str(int(row[column])),
                "group_count": int(row["count"]),
                "error_count": error_count,
                "error_rate": float(row["mean"]),
                "error_share": float(error_count / total_errors) if total_errors > 0 else 0.0,
            }
        )
    return rows


def _load_cluster_records(cluster_jsonl: Path, group_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_map = {
        (str(row.seq), int(row.frame), int(row.det_index)): row._asdict()
        for row in group_df.itertuples(index=False)
    }
    cluster_rows: list[dict[str, Any]] = []
    covered_rows: list[dict[str, Any]] = []
    with cluster_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cluster = json.loads(line)
            seq = str(cluster.get("seq", ""))
            frame = _safe_int(cluster.get("frame", 0), 0)
            det_rows = [int(x) for x in cluster.get("det_rows", [])]
            group_members = []
            for det_index in det_rows:
                key = (seq, frame, int(det_index))
                group_row = group_map.get(key)
                if group_row is not None:
                    group_members.append(group_row)
                covered_rows.append(
                    {
                        "seq": seq,
                        "frame": frame,
                        "det_index": int(det_index),
                        "cluster_id": str(cluster.get("cluster_id", "")),
                    }
                )
            cluster_rows.append(
                {
                    "cluster_id": str(cluster.get("cluster_id", "")),
                    "seq": seq,
                    "frame": frame,
                    "num_rows": len(det_rows),
                    "num_tracks": _safe_int(cluster.get("num_tracks", 0), 0),
                    "cluster_should_intervene": _safe_int(cluster.get("cluster_should_intervene", 0), 0),
                    "cluster_should_intervene_edit": _safe_int(cluster.get("cluster_should_intervene_edit", 0), 0),
                    "cluster_should_intervene_soft": _safe_int(cluster.get("cluster_should_intervene_soft", 0), 0),
                    "host_runtime_equals_oracle": _safe_int(cluster.get("host_runtime_equals_oracle", 0), 0),
                    "num_edit_rows": _safe_int(cluster.get("num_edit_rows", 0), 0),
                    "num_rescue_rows": _safe_int(cluster.get("num_rescue_rows", 0), 0),
                    "num_ambiguous_rows": int(sum(int(row.get("group_is_ambiguous", 0)) for row in group_members)),
                    "num_recoverable_rows": int(sum(int(row.get("group_is_recoverable", 0)) for row in group_members)),
                    "num_top1_error_rows": int(sum(int(row.get("top1_error", 0)) for row in group_members)),
                    "has_error": int(any(int(row.get("top1_error", 0)) > 0 for row in group_members)),
                    "has_ambiguous": int(any(int(row.get("group_is_ambiguous", 0)) > 0 for row in group_members)),
                }
            )
    cluster_df = pd.DataFrame(cluster_rows)
    covered_df = pd.DataFrame(covered_rows).drop_duplicates(["seq", "frame", "det_index"])
    return cluster_df, covered_df


def _coverage_rows(base_df: pd.DataFrame, covered_df: pd.DataFrame, scope_name: str) -> list[dict[str, Any]]:
    merged = base_df.merge(covered_df.assign(covered=1), on=["seq", "frame", "det_index"], how="left")
    merged["covered"] = merged["covered"].fillna(0).astype(int)
    result: list[dict[str, Any]] = []
    selectors = {
        "positive_groups": merged["group_has_positive"] == 1,
        "ambiguous_groups": merged["group_is_ambiguous"] == 1,
        "recoverable_groups": merged["group_is_recoverable"] == 1,
        "top1_error_groups": merged["top1_error"] == 1,
    }
    for label, mask in selectors.items():
        sub = merged.loc[mask]
        count = int(len(sub))
        covered_count = int(sub["covered"].sum()) if count > 0 else 0
        result.append(
            {
                "scope": scope_name,
                "slice_name": label,
                "count": count,
                "covered_count": covered_count,
                "covered_ratio": float(covered_count / count) if count > 0 else 0.0,
            }
        )
    return result


def _recoverable_reason_rows(
    *,
    group_df: pd.DataFrame,
    frame_groups: dict[tuple[str, int], list[dict[str, Any]]],
    scope_sequences: set[str],
    topk: int,
    min_detections: int,
    max_detections: int,
    max_tracks: int,
) -> list[dict[str, Any]]:
    recoverable_df = group_df[
        (group_df["seq"].isin(scope_sequences)) & (group_df["group_is_recoverable"] == 1)
    ].copy()
    rows: list[dict[str, Any]] = []
    for item in recoverable_df.itertuples(index=False):
        groups = frame_groups.get((str(item.seq), int(item.frame)), [])
        components = build_group_components_from_group_rows(groups, topk=int(topk))
        owning = None
        for component in components:
            if int(item.det_index) in component.get("det_rows", []):
                owning = component
                break
        if owning is None:
            reason = "no_component"
            num_dets = -1
            num_tracks = -1
        else:
            eligible, _ = filter_local_conflict_clusters_by_size(
                [owning],
                min_detections=int(min_detections),
                max_detections=int(max_detections),
                max_tracks=int(max_tracks),
            )
            num_dets = int(owning.get("num_detections", 0))
            num_tracks = int(owning.get("num_tracks", 0))
            if eligible:
                reason = "eligible"
            elif num_dets < int(min_detections):
                reason = "singleton_small"
            elif num_dets > int(max_detections) or num_tracks > int(max_tracks):
                reason = "skipped_large"
            else:
                reason = "filtered_other"
        rows.append(
            {
                "seq": str(item.seq),
                "frame": int(item.frame),
                "det_index": int(item.det_index),
                "reason": reason,
                "component_num_detections": num_dets,
                "component_num_tracks": num_tracks,
            }
        )
    return rows


def _teacher_alignment_rows(cluster_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if cluster_df.empty:
        return rows
    for column in (
        "cluster_should_intervene",
        "cluster_should_intervene_edit",
        "cluster_should_intervene_soft",
    ):
        if column not in cluster_df.columns:
            continue
        positives = cluster_df[cluster_df[column] == 1]
        count = int(len(positives))
        rows.append(
            {
                "teacher_key": column,
                "positive_clusters": count,
                "precision_vs_error_cluster": float(positives["has_error"].mean()) if count > 0 else 0.0,
                "precision_vs_ambiguous_cluster": float(positives["has_ambiguous"].mean()) if count > 0 else 0.0,
                "host_runtime_equals_oracle_rate": (
                    float(positives["host_runtime_equals_oracle"].mean()) if count > 0 else 0.0
                ),
                "mean_recoverable_rows": float(positives["num_recoverable_rows"].mean()) if count > 0 else 0.0,
                "mean_edit_rows": float(positives["num_edit_rows"].mean()) if count > 0 else 0.0,
                "mean_rescue_rows": float(positives["num_rescue_rows"].mean()) if count > 0 else 0.0,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    det_score_by_group = None
    if str(args.rows_csv or "").strip():
        det_score_by_group = _load_det_scores(Path(args.rows_csv).resolve())

    group_df, frame_groups = _load_group_records(
        Path(args.group_jsonl).resolve(),
        det_score_by_group=det_score_by_group,
    )

    positive_df = group_df[group_df["group_has_positive"] == 1].copy()
    total_positive = int(len(positive_df))
    total_errors = int(positive_df["top1_error"].sum())

    per_seq_df = (
        group_df.groupby("seq")
        .agg(
            groups=("group_id", "count"),
            positive_groups=("group_has_positive", "sum"),
            background_groups=("group_is_background", "sum"),
            ambiguous_groups=("group_is_ambiguous", "sum"),
            recoverable_groups=("group_is_recoverable", "sum"),
            top1_error_groups=("top1_error", "sum"),
        )
        .reset_index()
    )
    per_seq_df["positive_top1_error_rate"] = per_seq_df["top1_error_groups"] / per_seq_df["positive_groups"].clip(lower=1)
    per_seq_df = per_seq_df.sort_values(["top1_error_groups", "recoverable_groups", "seq"], ascending=[False, False, True])

    slice_rows: list[dict[str, Any]] = []
    for binary_col in ("group_is_ambiguous", "group_is_recoverable"):
        slice_rows.extend(_binary_slice_rows(positive_df, binary_col))
    for quant_col in (
        "rank_margin",
        "rank_entropy",
        "positive_hist_len",
        "positive_base_score",
        "positive_refined_score",
        "det_score",
    ):
        if quant_col in positive_df.columns and positive_df[quant_col].notna().any():
            slice_rows.extend(_quantile_slice_rows(positive_df, quant_col))
    slice_df = pd.DataFrame(slice_rows)
    if not slice_df.empty:
        slice_df = slice_df.sort_values(["error_share", "error_count", "slice_family"], ascending=[False, False, True])

    summary: dict[str, Any] = {
        "group_jsonl": str(Path(args.group_jsonl).resolve()),
        "cluster_jsonl": str(Path(args.cluster_jsonl).resolve()) if str(args.cluster_jsonl).strip() else "",
        "rows_csv": str(Path(args.rows_csv).resolve()) if str(args.rows_csv).strip() else "",
        "groups": int(len(group_df)),
        "positive_groups": total_positive,
        "positive_top1_error_groups": total_errors,
        "positive_top1_error_rate": float(total_errors / total_positive) if total_positive > 0 else 0.0,
        "ambiguous_groups": int(group_df["group_is_ambiguous"].sum()),
        "recoverable_groups": int(group_df["group_is_recoverable"].sum()),
        "top_error_sequences": per_seq_df.head(3).to_dict(orient="records"),
    }

    coverage_df = pd.DataFrame()
    cluster_df = pd.DataFrame()
    teacher_df = pd.DataFrame()
    recoverable_reason_df = pd.DataFrame()
    if str(args.cluster_jsonl or "").strip():
        cluster_df, covered_df = _load_cluster_records(Path(args.cluster_jsonl).resolve(), group_df=group_df)
        inferred_sequences = sorted(set(cluster_df["seq"].tolist())) if not cluster_df.empty else []
        scope_sequences = set(_parse_csv_tokens(args.train_sequences)) or set(inferred_sequences)

        coverage_rows = _coverage_rows(group_df, covered_df, "all_sequences")
        if scope_sequences:
            coverage_rows.extend(
                _coverage_rows(group_df[group_df["seq"].isin(scope_sequences)].copy(), covered_df, "cluster_sequences")
            )
        coverage_df = pd.DataFrame(coverage_rows)

        teacher_df = pd.DataFrame(_teacher_alignment_rows(cluster_df))

        if scope_sequences:
            recoverable_reason_rows = _recoverable_reason_rows(
                group_df=group_df,
                frame_groups=frame_groups,
                scope_sequences=scope_sequences,
                topk=int(args.topk),
                min_detections=int(args.min_detections),
                max_detections=int(args.max_detections),
                max_tracks=int(args.max_tracks),
            )
            recoverable_reason_df = pd.DataFrame(recoverable_reason_rows)
            if not recoverable_reason_df.empty:
                summary["cluster_scope_sequences"] = sorted(scope_sequences)
                summary["cluster_scope_recoverable_groups"] = int(len(recoverable_reason_df))
                reason_counter = Counter(recoverable_reason_df["reason"].tolist())
                summary["recoverable_coverage_reasons"] = dict(reason_counter)

        summary["cluster_examples"] = int(len(cluster_df))
        summary["cluster_error_examples"] = int(cluster_df["has_error"].sum()) if not cluster_df.empty else 0

    per_seq_df.to_csv(out_dir / "per_sequence.csv", index=False)
    if not slice_df.empty:
        slice_df.to_csv(out_dir / "slice_summary.csv", index=False)
    if not coverage_df.empty:
        coverage_df.to_csv(out_dir / "cluster_coverage.csv", index=False)
    if not teacher_df.empty:
        teacher_df.to_csv(out_dir / "teacher_alignment.csv", index=False)
    if not recoverable_reason_df.empty:
        recoverable_reason_df.to_csv(out_dir / "recoverable_group_coverage_reasons.csv", index=False)
        recoverable_reason_df.groupby("reason").agg(
            count=("reason", "count"),
            mean_component_num_detections=("component_num_detections", "mean"),
            mean_component_num_tracks=("component_num_tracks", "mean"),
        ).reset_index().to_csv(out_dir / "recoverable_coverage_reason_summary.csv", index=False)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_row = dict(summary)
    for key, value in list(summary_row.items()):
        if isinstance(value, (dict, list)):
            summary_row[key] = json.dumps(value, sort_keys=True)
    pd.DataFrame([summary_row]).to_csv(out_dir / "summary.csv", index=False)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
