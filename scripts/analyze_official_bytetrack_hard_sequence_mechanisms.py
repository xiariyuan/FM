#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.local_conflict_graph_common import (  # noqa: E402
    build_group_components_from_group_rows,
    filter_local_conflict_clusters_by_size,
)


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
REGISTRY_SCRIPT = REPO_ROOT / "scripts" / "append_experiment_record.py"
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

DEFAULT_GROUP_JSONL = (
    REPO_ROOT
    / "outputs"
    / "official_bytetrack_editutility_rawgate_select_v1_20260326_143126"
    / "labeled_replay_top8.groups.jsonl"
)
DEFAULT_ROWS_CSV = (
    REPO_ROOT
    / "outputs"
    / "official_bytetrack_editutility_rawgate_select_v1_20260326_143126"
    / "labeled_replay_top8.csv"
)
DEFAULT_CLUSTER_JSONL = (
    REPO_ROOT
    / "outputs"
    / "official_bytetrack_editutility_largecoverdiag_v1_20260326_175532"
    / "cluster_set_predictor_data"
    / "cluster_examples.jsonl"
)
DEFAULT_PLUGIN_DIAGS = {
    "seed42_largecover": (
        REPO_ROOT
        / "outputs"
        / "official_bytetrack_largecover_gateedit_pair_v1_deltadiag_20260326_223500"
        / "01_host_plus_plugin"
        / "diagnostics"
    ),
    "seed43_thresh060": (
        REPO_ROOT
        / "outputs"
        / "official_bytetrack_largecover_gateedit_pair_confirm_best_v2_thresh060_deltadiag_20260326_223500"
        / "01_host_plus_plugin"
        / "diagnostics"
    ),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Diagnose official ByteTrack hard-sequence local association mechanisms on targeted sequences."
    )
    ap.add_argument("--group-jsonl", default=str(DEFAULT_GROUP_JSONL))
    ap.add_argument("--rows-csv", default=str(DEFAULT_ROWS_CSV))
    ap.add_argument("--cluster-jsonl", default=str(DEFAULT_CLUSTER_JSONL))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sequences", default="MOT17-02-FRCNN,MOT17-05-FRCNN,MOT17-10-FRCNN")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--min-detections", type=int, default=2)
    ap.add_argument("--max-detections", type=int, default=8)
    ap.add_argument("--max-tracks", type=int, default=32)
    ap.add_argument("--near-tie-margin-quantile", type=float, default=0.2)
    ap.add_argument("--high-entropy-quantile", type=float, default=0.8)
    ap.add_argument("--low-det-score-quantile", type=float, default=0.2)
    ap.add_argument("--low-refined-score-quantile", type=float, default=0.2)
    ap.add_argument("--long-gap-quantile", type=float, default=0.8)
    ap.add_argument("--short-hist-quantile", type=float, default=0.2)
    ap.add_argument(
        "--plugin-diag",
        action="append",
        default=[],
        help="Optional run_name=/path/to/diagnostics_dir. If omitted, built-in plugin delta diagnostics are used.",
    )
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return ap.parse_args()


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


def _parse_sequences(raw: str) -> list[str]:
    return [token.strip() for token in str(raw or "").split(",") if token.strip()]


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _parse_plugin_specs(items: list[str]) -> dict[str, Path]:
    if not items:
        return {name: path.resolve() for name, path in DEFAULT_PLUGIN_DIAGS.items()}
    specs: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --plugin-diag spec, expected name=path: {item}")
        name, raw_path = item.split("=", 1)
        specs[name.strip()] = Path(raw_path).expanduser().resolve()
    return specs


def _find_positive_candidate(group: dict[str, Any]) -> dict[str, Any] | None:
    for cand in group.get("candidates", []):
        if _safe_int(cand.get("label", 0), 0) > 0:
            return cand
    return None


def _find_top1_candidate(group: dict[str, Any]) -> dict[str, Any] | None:
    candidates = sorted(
        list(group.get("candidates", [])),
        key=lambda row: _safe_int(row.get("track_rank", 0), 0),
    )
    return candidates[0] if candidates else None


def _load_det_scores(rows_csv: Path) -> dict[str, float]:
    score_by_group: dict[str, float] = {}
    with rows_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group_id = str(row.get("group_id", "")).strip()
            if not group_id or group_id in score_by_group:
                continue
            score_by_group[group_id] = _safe_float(row.get("det_score", 0.0), 0.0)
    return score_by_group


def _load_groups(group_jsonl: Path, *, det_score_by_group: dict[str, float]) -> tuple[pd.DataFrame, dict[tuple[str, int], list[dict[str, Any]]]]:
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
            det_index = _safe_int(group.get("det_index", -1), -1)
            group_id = str(group.get("group_id", "")).strip()
            positive = _find_positive_candidate(group)
            top1 = _find_top1_candidate(group)
            rows.append(
                {
                    "seq": seq,
                    "frame": frame,
                    "det_index": det_index,
                    "group_id": group_id,
                    "group_has_positive": _safe_int(group.get("group_has_positive", 0), 0),
                    "group_is_ambiguous": _safe_int(group.get("group_is_ambiguous", 0), 0),
                    "group_is_background": _safe_int(group.get("group_is_background", 0), 0),
                    "group_is_recoverable": _safe_int(group.get("group_is_recoverable", 0), 0),
                    "rank_top1_correct": _safe_int(group.get("rank_top1_correct", 0), 0),
                    "positive_rank": _safe_int(group.get("positive_rank", -1), -1),
                    "rank_margin": _safe_float(group.get("rank_margin", 0.0), 0.0),
                    "rank_entropy": _safe_float(group.get("rank_entropy", 0.0), 0.0),
                    "candidate_count_total": _safe_int(group.get("candidate_count_total", 0), 0),
                    "group_size": _safe_int(group.get("group_size", 0), 0),
                    "det_score": _safe_float(det_score_by_group.get(group_id, 0.0), 0.0),
                    "positive_track_id": _safe_int(positive.get("track_id", -1), -1) if positive else -1,
                    "positive_gap": _safe_int(positive.get("track_gap", -1), -1) if positive else -1,
                    "positive_hist_len": _safe_int(positive.get("track_hist_len", -1), -1) if positive else -1,
                    "positive_base_score": _safe_float(positive.get("base_score", 0.0), 0.0) if positive else 0.0,
                    "positive_refined_score": _safe_float(positive.get("refined_score", 0.0), 0.0) if positive else 0.0,
                    "top1_track_id": _safe_int(top1.get("track_id", -1), -1) if top1 else -1,
                    "top1_gap": _safe_int(top1.get("track_gap", -1), -1) if top1 else -1,
                    "top1_hist_len": _safe_int(top1.get("track_hist_len", -1), -1) if top1 else -1,
                    "top1_refined_score": _safe_float(top1.get("refined_score", 0.0), 0.0) if top1 else 0.0,
                }
            )
            frame_groups[(seq, frame)].append(group)
    df = pd.DataFrame(rows)
    df["top1_error"] = ((df["group_has_positive"] == 1) & (df["rank_top1_correct"] == 0)).astype(int)
    return df, frame_groups


def _load_cluster_map(cluster_jsonl: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    cluster_map: dict[tuple[str, int, int], dict[str, Any]] = {}
    with cluster_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cluster = json.loads(line)
            seq = str(cluster.get("seq", ""))
            frame = _safe_int(cluster.get("frame", 0), 0)
            for det_index in cluster.get("det_rows", []):
                key = (seq, frame, _safe_int(det_index, -1))
                if key in cluster_map:
                    continue
                cluster_map[key] = {
                    "cluster_id": str(cluster.get("cluster_id", "")),
                    "cluster_should_intervene_soft": _safe_int(cluster.get("cluster_should_intervene_soft", 0), 0),
                    "cluster_should_intervene": _safe_int(cluster.get("cluster_should_intervene", 0), 0),
                    "host_runtime_equals_oracle": _safe_int(cluster.get("host_runtime_equals_oracle", 0), 0),
                    "num_edit_rows": _safe_int(cluster.get("num_edit_rows", 0), 0),
                    "num_rescue_rows": _safe_int(cluster.get("num_rescue_rows", 0), 0),
                    "num_soft_rescue_rows": _safe_int(cluster.get("num_soft_rescue_rows", 0), 0),
                    "num_detections": _safe_int(cluster.get("num_detections", len(cluster.get("det_rows", []))), 0),
                    "num_tracks": _safe_int(cluster.get("num_tracks", 0), 0),
                    "mined_from_large_component": _safe_int(cluster.get("mined_from_large_component", 0), 0),
                }
    return cluster_map


def _build_plugin_seq_df(plugin_specs: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_name, diag_dir in plugin_specs.items():
        if not diag_dir.is_dir():
            raise FileNotFoundError(f"Plugin diagnostics dir not found: {diag_dir}")
        for path in sorted(diag_dir.glob("*.json")):
            obj = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "run_name": run_name,
                    "seq": path.stem,
                    "eligible_clusters": _safe_int(obj.get("eligible_clusters", 0), 0),
                    "gate_pass_clusters": _safe_int(obj.get("gate_pass_clusters", 0), 0),
                    "replaced_clusters": _safe_int(obj.get("replaced_clusters", 0), 0),
                    "delta_commit_pairs": _safe_int(obj.get("delta_commit_pairs", 0), 0),
                    "delta_drop_pairs": _safe_int(obj.get("delta_drop_pairs", 0), 0),
                    "delta_replaced_clusters": _safe_int(obj.get("delta_replaced_clusters", 0), 0),
                }
            )
    return pd.DataFrame(rows)


def _quantile_or_default(series: pd.Series, q: float, default: float) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return float(default)
    return float(valid.quantile(float(q)))


def append_registry(args: argparse.Namespace, *, out_dir: Path, summary_csv: Path, status: str) -> None:
    cmd = [
        args.python_bin,
        str(REGISTRY_SCRIPT),
        "--csv",
        str(Path(args.registry_csv).resolve()),
        "--kind",
        "analysis",
        "--status",
        status,
        "--script",
        "scripts/analyze_official_bytetrack_hard_sequence_mechanisms.py",
        "--dataset",
        "MOT17",
        "--split",
        "val_half_targeted_hard_sequences",
        "--tracker-family",
        "official_bytetrack",
        "--variant",
        "hard_sequence_mechanisms",
        "--tag",
        out_dir.name,
        "--run-root",
        str(out_dir.resolve()),
        "--summary-csv",
        str(summary_csv.resolve()),
        "--notes",
        "Targeted local mechanism audit for MOT17-02/05/10 on official ByteTrack host-only replay labels.",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    result_csv = out_dir / "result.csv"

    running_row = {
        "group_jsonl": str(Path(args.group_jsonl).resolve()),
        "rows_csv": str(Path(args.rows_csv).resolve()),
        "cluster_jsonl": str(Path(args.cluster_jsonl).resolve()),
        "sequences": json.dumps(_parse_sequences(args.sequences)),
        "status": "running",
        "error": "",
    }
    _write_csv(summary_csv, pd.DataFrame([running_row]))
    _write_csv(result_csv, pd.DataFrame([running_row]))

    try:
        sequences = _parse_sequences(args.sequences)
        det_score_by_group = _load_det_scores(Path(args.rows_csv).resolve())
        group_df, frame_groups = _load_groups(Path(args.group_jsonl).resolve(), det_score_by_group=det_score_by_group)
        cluster_map = _load_cluster_map(Path(args.cluster_jsonl).resolve())
        plugin_specs = _parse_plugin_specs(args.plugin_diag)
        plugin_seq_df = _build_plugin_seq_df(plugin_specs)

        focus_df = group_df[group_df["seq"].isin(sequences)].copy()
        positive_df = group_df[group_df["group_has_positive"] == 1].copy()

        thresholds = {
            "near_tie_margin_max": _quantile_or_default(positive_df["rank_margin"], args.near_tie_margin_quantile, 0.2),
            "high_entropy_min": _quantile_or_default(positive_df["rank_entropy"], args.high_entropy_quantile, 0.8),
            "low_det_score_max": _quantile_or_default(positive_df["det_score"], args.low_det_score_quantile, 0.85),
            "low_refined_score_max": _quantile_or_default(
                positive_df["positive_refined_score"],
                args.low_refined_score_quantile,
                0.8,
            ),
            "long_gap_min": _quantile_or_default(positive_df["positive_gap"], args.long_gap_quantile, 5.0),
            "short_hist_max": _quantile_or_default(positive_df["positive_hist_len"], args.short_hist_quantile, 32.0),
        }

        group_records: list[dict[str, Any]] = []
        component_rows: list[dict[str, Any]] = []
        seen_components: set[tuple[str, int, tuple[int, ...]]] = set()

        for (seq, frame), frame_sub in focus_df.groupby(["seq", "frame"], sort=True):
            raw_groups = list(frame_groups.get((str(seq), int(frame)), []))
            if not raw_groups:
                continue
            components = build_group_components_from_group_rows(raw_groups, topk=int(args.topk))
            filtered_components, _ = filter_local_conflict_clusters_by_size(
                components,
                min_detections=int(args.min_detections),
                max_detections=int(args.max_detections),
                max_tracks=int(args.max_tracks),
            )
            filtered_keys = {
                tuple(sorted(int(det_idx) for det_idx in comp.get("det_rows", [])))
                for comp in filtered_components
            }

            group_lookup = {int(group["det_index"]): group for group in raw_groups if _safe_int(group.get("det_index", -1), -1) >= 0}
            candidates_by_det = {
                det_idx: sorted(
                    list(group_lookup[det_idx].get("candidates", [])),
                    key=lambda row: _safe_int(row.get("track_rank", 0), 0),
                )
                for det_idx in group_lookup
            }

            component_for_det: dict[int, dict[str, Any]] = {}
            for comp in components:
                det_rows = [int(det_idx) for det_idx in comp.get("det_rows", [])]
                for det_idx in det_rows:
                    component_for_det[int(det_idx)] = comp

            for comp in components:
                det_rows = tuple(sorted(int(det_idx) for det_idx in comp.get("det_rows", [])))
                comp_key = (str(seq), int(frame), det_rows)
                if comp_key in seen_components:
                    continue
                seen_components.add(comp_key)
                top1_track_ids = [group_lookup[det]["candidates"][0]["track_id"] for det in det_rows if group_lookup.get(det) and group_lookup[det].get("candidates")]
                top1_collision_groups = len(top1_track_ids) - len(set(int(x) for x in top1_track_ids))
                component_rows.append(
                    {
                        "seq": str(seq),
                        "frame": int(frame),
                        "component_det_rows": ",".join(str(x) for x in det_rows),
                        "num_detections": _safe_int(comp.get("num_detections", len(det_rows)), len(det_rows)),
                        "num_tracks": _safe_int(comp.get("num_tracks", len(comp.get("track_ids", []))), 0),
                        "eligible_under_current_caps": int(det_rows in filtered_keys),
                        "top1_collision_groups": int(max(top1_collision_groups, 0)),
                        "top1_error_groups": int(
                            sum(
                                int(
                                    focus_df[
                                        (focus_df["seq"] == str(seq))
                                        & (focus_df["frame"] == int(frame))
                                        & (focus_df["det_index"] == int(det_idx))
                                    ]["top1_error"].max()
                                )
                                for det_idx in det_rows
                            )
                        ),
                    }
                )

            for row in frame_sub.itertuples(index=False):
                if int(row.top1_error) <= 0:
                    continue
                comp = component_for_det.get(int(row.det_index))
                if comp is None:
                    continue
                det_rows = [int(det_idx) for det_idx in comp.get("det_rows", [])]
                comp_det_key = tuple(sorted(det_rows))
                num_dets = _safe_int(comp.get("num_detections", len(det_rows)), len(det_rows))
                num_tracks = _safe_int(comp.get("num_tracks", len(comp.get("track_ids", []))), 0)
                eligible = int(
                    (num_dets >= int(args.min_detections))
                    and (num_dets <= int(args.max_detections))
                    and (num_tracks <= int(args.max_tracks))
                )
                skipped_large = int(
                    (num_dets >= int(args.min_detections))
                    and ((num_dets > int(args.max_detections)) or (num_tracks > int(args.max_tracks)))
                )

                positive_track_id = int(row.positive_track_id)
                top1_track_id = int(row.top1_track_id)
                positive_track_comp_count = 0
                top1_track_comp_count = 0
                for det_idx in det_rows:
                    cand_list = candidates_by_det.get(int(det_idx), [])
                    cand_track_ids = [int(c.get("track_id", -1)) for c in cand_list]
                    if positive_track_id >= 0 and positive_track_id in cand_track_ids:
                        positive_track_comp_count += 1
                    if cand_list and int(cand_list[0].get("track_id", -1)) == top1_track_id and top1_track_id >= 0:
                        top1_track_comp_count += 1
                same_column_collision = int(max(positive_track_comp_count, top1_track_comp_count) > 1)
                near_tie = int(
                    (_safe_float(row.rank_margin, 0.0) <= thresholds["near_tie_margin_max"])
                    and (_safe_float(row.rank_entropy, 0.0) >= thresholds["high_entropy_min"])
                )
                positive_gap = _safe_int(row.positive_gap, -1)
                positive_hist_len = _safe_int(row.positive_hist_len, -1)
                long_gap_or_stale = int(
                    (positive_gap > max(int(round(thresholds["long_gap_min"])), 1))
                    or (
                        positive_hist_len >= 0
                        and positive_hist_len <= thresholds["short_hist_max"]
                        and positive_gap > 1
                    )
                )
                low_score_noise = int(
                    (_safe_float(row.det_score, 0.0) <= thresholds["low_det_score_max"])
                    and (_safe_float(row.positive_refined_score, 0.0) <= thresholds["low_refined_score_max"])
                )
                cluster_info = cluster_map.get((str(row.seq), int(row.frame), int(row.det_index)), {})

                if skipped_large:
                    primary = "skipped_large_component"
                elif same_column_collision and near_tie:
                    primary = "same_column_near_tie"
                elif same_column_collision:
                    primary = "same_column_collision"
                elif long_gap_or_stale:
                    primary = "long_gap_or_stale_history"
                elif low_score_noise:
                    primary = "low_score_or_detection_noise"
                elif int(row.group_is_recoverable) == 1:
                    primary = "recoverable_other"
                else:
                    primary = "other_top1_error"

                group_records.append(
                    {
                        "seq": str(row.seq),
                        "frame": int(row.frame),
                        "det_index": int(row.det_index),
                        "group_id": str(row.group_id),
                        "group_is_ambiguous": int(row.group_is_ambiguous),
                        "group_is_recoverable": int(row.group_is_recoverable),
                        "positive_rank": int(row.positive_rank),
                        "rank_margin": float(row.rank_margin),
                        "rank_entropy": float(row.rank_entropy),
                        "det_score": float(row.det_score),
                        "positive_refined_score": float(row.positive_refined_score),
                        "positive_gap": int(row.positive_gap),
                        "positive_hist_len": int(row.positive_hist_len),
                        "top1_track_id": top1_track_id,
                        "positive_track_id": positive_track_id,
                        "component_num_detections": num_dets,
                        "component_num_tracks": num_tracks,
                        "component_det_rows": ",".join(str(x) for x in comp_det_key),
                        "component_eligible_under_current_caps": eligible,
                        "component_skipped_large": skipped_large,
                        "positive_track_component_count": int(positive_track_comp_count),
                        "top1_track_component_count": int(top1_track_comp_count),
                        "near_tie_error": near_tie,
                        "same_column_collision": same_column_collision,
                        "long_gap_or_stale_history": long_gap_or_stale,
                        "low_score_or_detection_noise": low_score_noise,
                        "cluster_covered": int(bool(cluster_info)),
                        "cluster_should_intervene_soft": int(cluster_info.get("cluster_should_intervene_soft", 0)),
                        "cluster_should_intervene": int(cluster_info.get("cluster_should_intervene", 0)),
                        "cluster_host_runtime_equals_oracle": int(cluster_info.get("host_runtime_equals_oracle", 0)),
                        "cluster_num_edit_rows": int(cluster_info.get("num_edit_rows", 0)),
                        "cluster_num_rescue_rows": int(cluster_info.get("num_rescue_rows", 0)),
                        "cluster_num_soft_rescue_rows": int(cluster_info.get("num_soft_rescue_rows", 0)),
                        "cluster_mined_from_large_component": int(cluster_info.get("mined_from_large_component", 0)),
                        "primary_mechanism": primary,
                    }
                )

        group_df_out = pd.DataFrame(group_records)
        component_df = pd.DataFrame(component_rows)
        plugin_focus_df = plugin_seq_df[plugin_seq_df["seq"].isin(sequences)].copy()

        if group_df_out.empty:
            raise ValueError("No top1 error groups found for requested sequences.")

        mechanism_rows: list[dict[str, Any]] = []
        mechanism_flags = [
            "same_column_collision",
            "near_tie_error",
            "long_gap_or_stale_history",
            "low_score_or_detection_noise",
            "component_skipped_large",
            "cluster_covered",
            "cluster_should_intervene_soft",
        ]
        for seq, sub in group_df_out.groupby("seq", sort=True):
            error_count = int(len(sub))
            row: dict[str, Any] = {
                "seq": seq,
                "top1_error_groups": error_count,
                "ambiguous_error_groups": int(sub["group_is_ambiguous"].sum()),
                "recoverable_error_groups": int(sub["group_is_recoverable"].sum()),
                "same_column_collision_groups": int(sub["same_column_collision"].sum()),
                "near_tie_error_groups": int(sub["near_tie_error"].sum()),
                "long_gap_or_stale_history_groups": int(sub["long_gap_or_stale_history"].sum()),
                "low_score_or_detection_noise_groups": int(sub["low_score_or_detection_noise"].sum()),
                "skipped_large_groups": int(sub["component_skipped_large"].sum()),
                "cluster_covered_error_groups": int(sub["cluster_covered"].sum()),
                "cluster_soft_positive_error_groups": int(sub["cluster_should_intervene_soft"].sum()),
                "component_eligible_error_groups": int(sub["component_eligible_under_current_caps"].sum()),
                "mean_rank_margin": float(sub["rank_margin"].mean()),
                "mean_rank_entropy": float(sub["rank_entropy"].mean()),
                "mean_component_num_detections": float(sub["component_num_detections"].mean()),
                "mean_component_num_tracks": float(sub["component_num_tracks"].mean()),
                "primary_mechanism_histogram": json.dumps(dict(Counter(sub["primary_mechanism"].tolist())), sort_keys=True),
            }
            for flag in mechanism_flags:
                row[f"{flag}_rate"] = float(sub[flag].mean()) if error_count > 0 else 0.0
            mechanism_rows.append(row)
        mechanism_df = pd.DataFrame(mechanism_rows)

        primary_df = (
            group_df_out.groupby(["seq", "primary_mechanism"], sort=True)
            .size()
            .reset_index(name="count")
            .sort_values(["seq", "count", "primary_mechanism"], ascending=[True, False, True])
        )

        summary_row = {
            "group_jsonl": str(Path(args.group_jsonl).resolve()),
            "rows_csv": str(Path(args.rows_csv).resolve()),
            "cluster_jsonl": str(Path(args.cluster_jsonl).resolve()),
            "sequences": json.dumps(sequences),
            "top1_error_groups": int(len(group_df_out)),
            "sequence_top1_error_counts": json.dumps(
                {str(row["seq"]): int(row["top1_error_groups"]) for row in mechanism_rows},
                sort_keys=True,
            ),
            "thresholds": json.dumps(thresholds, sort_keys=True),
            "plugin_focus_sequences": json.dumps(sorted(plugin_focus_df["seq"].unique().tolist())),
            "status": "success",
            "error": "",
        }

        _write_csv(summary_csv, pd.DataFrame([summary_row]))
        _write_csv(result_csv, pd.DataFrame([summary_row]))
        _write_csv(out_dir / "error_group_mechanisms.csv", group_df_out)
        _write_csv(out_dir / "mechanism_summary.csv", mechanism_df)
        _write_csv(out_dir / "primary_mechanism_summary.csv", primary_df)
        _write_csv(out_dir / "component_summary.csv", component_df)
        _write_csv(out_dir / "plugin_focus_summary.csv", plugin_focus_df)

        append_registry(args, out_dir=out_dir, summary_csv=summary_csv, status="success")
        print(json.dumps(summary_row, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failed_row = dict(running_row)
        failed_row.update({"status": "failed", "error": str(exc)})
        _write_csv(summary_csv, pd.DataFrame([failed_row]))
        _write_csv(result_csv, pd.DataFrame([failed_row]))
        try:
            append_registry(args, out_dir=out_dir, summary_csv=summary_csv, status="failed")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
