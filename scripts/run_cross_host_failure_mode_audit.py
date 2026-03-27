#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
REGISTRY_SCRIPT = REPO_ROOT / "scripts" / "append_experiment_record.py"
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

DEFAULT_CROSS_HOST_PER_SEQ = (
    REPO_ROOT
    / "outputs"
    / "cross_host_baseline_audit_mot17_frcnn_valhalf_20260326_225500_fix2"
    / "per_seq.csv"
)
DEFAULT_OFFICIAL_FAILURE_PER_SEQ = (
    REPO_ROOT
    / "outputs"
    / "official_bytetrack_failure_slices_crosshost_focus_20260326_214500"
    / "per_sequence.csv"
)
DEFAULT_RECOVERABLE_REASONS = (
    REPO_ROOT
    / "outputs"
    / "official_bytetrack_failure_slices_crosshost_focus_20260326_214500"
    / "recoverable_group_coverage_reasons.csv"
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
        description="Bridge clean cross-host baseline difficulty with official ByteTrack local failure semantics."
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cross-host-per-seq", default=str(DEFAULT_CROSS_HOST_PER_SEQ))
    ap.add_argument("--official-failure-per-seq", default=str(DEFAULT_OFFICIAL_FAILURE_PER_SEQ))
    ap.add_argument("--recoverable-reasons-csv", default=str(DEFAULT_RECOVERABLE_REASONS))
    ap.add_argument(
        "--plugin-diag",
        action="append",
        default=[],
        help="Optional run_name=/path/to/diagnostics_dir. If omitted, built-in official delta-diagnostic runs are used.",
    )
    ap.add_argument("--top-k-common-hard", type=int, default=3)
    ap.add_argument("--official-top-k-error", type=int, default=3)
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return ap.parse_args()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_plugin_specs(items: list[str]) -> dict[str, Path]:
    if not items:
        return {name: path.resolve() for name, path in DEFAULT_PLUGIN_DIAGS.items()}
    specs: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --plugin-diag spec, expected name=path: {item}")
        name, raw_path = item.split("=", 1)
        specs[name.strip()] = Path(raw_path).expanduser().resolve()
    return specs


def load_cross_host_sequence_table(per_seq_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(per_seq_csv)
    if df.empty:
        raise ValueError(f"Empty cross-host per-seq csv: {per_seq_csv}")
    df["HOTA"] = df["HOTA"].astype(float)
    df["AssA"] = df["AssA"].astype(float)
    df["IDF1"] = df["IDF1"].astype(float)
    df["MOTA"] = df["MOTA"].astype(float)
    df["IDSW"] = df["IDSW"].astype(float)
    carriers = sorted(df["carrier"].dropna().astype(str).unique().tolist())

    pivot = df.pivot(index="seq", columns="carrier")
    pivot.columns = [f"{carrier}_{metric}" for metric, carrier in pivot.columns]
    seq_df = pivot.reset_index()

    grouped = df.groupby("seq", sort=True)
    seq_df["carriers_present"] = grouped["carrier"].nunique().reindex(seq_df["seq"]).fillna(0).astype(int).values
    seq_df["mean_HOTA"] = grouped["HOTA"].mean().reindex(seq_df["seq"]).values
    seq_df["min_HOTA"] = grouped["HOTA"].min().reindex(seq_df["seq"]).values
    seq_df["max_HOTA"] = grouped["HOTA"].max().reindex(seq_df["seq"]).values
    seq_df["spread_HOTA"] = seq_df["max_HOTA"] - seq_df["min_HOTA"]
    seq_df["mean_AssA"] = grouped["AssA"].mean().reindex(seq_df["seq"]).values
    seq_df["mean_IDF1"] = grouped["IDF1"].mean().reindex(seq_df["seq"]).values
    seq_df["mean_MOTA"] = grouped["MOTA"].mean().reindex(seq_df["seq"]).values
    seq_df["mean_IDSW"] = grouped["IDSW"].mean().reindex(seq_df["seq"]).values

    for carrier in carriers:
        mask = df["carrier"] == carrier
        carrier_rank = (
            df.loc[mask, ["seq", "HOTA"]]
            .sort_values(["HOTA", "seq"], ascending=[True, True])
            .reset_index(drop=True)
        )
        carrier_rank[f"{carrier}_HOTA_rank"] = range(1, len(carrier_rank) + 1)
        seq_df = seq_df.merge(carrier_rank[["seq", f"{carrier}_HOTA_rank"]], on="seq", how="left")

    rank_cols = [f"{carrier}_HOTA_rank" for carrier in carriers]
    seq_df["mean_hard_rank"] = seq_df[rank_cols].mean(axis=1)
    seq_df = seq_df.sort_values(["mean_HOTA", "mean_IDF1", "seq"], ascending=[True, True, True]).reset_index(drop=True)
    seq_df["common_hard_rank"] = range(1, len(seq_df) + 1)
    return seq_df, carriers


def load_official_failure_table(per_seq_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(per_seq_csv)
    if df.empty:
        raise ValueError(f"Empty official failure per-seq csv: {per_seq_csv}")
    df["official_ambiguous_rate"] = df["ambiguous_groups"] / df["groups"].clip(lower=1)
    df["official_recoverable_rate"] = df["recoverable_groups"] / df["positive_groups"].clip(lower=1)
    total_top1_errors = max(int(df["top1_error_groups"].sum()), 1)
    df["official_top1_error_share"] = df["top1_error_groups"] / float(total_top1_errors)
    rename_map = {
        "groups": "official_groups",
        "positive_groups": "official_positive_groups",
        "background_groups": "official_background_groups",
        "ambiguous_groups": "official_ambiguous_groups",
        "recoverable_groups": "official_recoverable_groups",
        "top1_error_groups": "official_top1_error_groups",
        "positive_top1_error_rate": "official_positive_top1_error_rate",
    }
    return df.rename(columns=rename_map)


def load_recoverable_reason_table(reasons_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(reasons_csv)
    if df.empty:
        return pd.DataFrame(columns=["seq"])
    summary_rows: list[dict[str, Any]] = []
    for seq, sub in df.groupby("seq", sort=True):
        total = int(len(sub))
        eligible = int((sub["reason"] == "eligible").sum())
        skipped_large = int((sub["reason"] == "skipped_large").sum())
        summary_rows.append(
            {
                "seq": seq,
                "recoverable_reason_total": total,
                "recoverable_eligible_count": eligible,
                "recoverable_skipped_large_count": skipped_large,
                "recoverable_skipped_large_share": (float(skipped_large) / float(total)) if total > 0 else 0.0,
                "recoverable_max_component_detections": _safe_int(sub["component_num_detections"].max(), 0),
                "recoverable_max_component_tracks": _safe_int(sub["component_num_tracks"].max(), 0),
            }
        )
    return pd.DataFrame(summary_rows)


def load_plugin_sequence_table(plugin_specs: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_name, diag_dir in plugin_specs.items():
        if not diag_dir.is_dir():
            raise FileNotFoundError(f"Plugin diagnostics dir not found: {diag_dir}")
        files = sorted(diag_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No plugin diagnostic json files found in {diag_dir}")
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            eligible = _safe_int(payload.get("eligible_clusters", 0), 0)
            gate_pass = _safe_int(payload.get("gate_pass_clusters", 0), 0)
            replaced = _safe_int(payload.get("replaced_clusters", 0), 0)
            delta_replaced = _safe_int(payload.get("delta_replaced_clusters", 0), 0)
            delta_add = _safe_int(payload.get("delta_commit_pairs", 0), 0)
            delta_drop = _safe_int(payload.get("delta_drop_pairs", 0), 0)
            rows.append(
                {
                    "run_name": run_name,
                    "seq": path.stem,
                    "diag_dir": str(diag_dir.resolve()),
                    "eligible_clusters": eligible,
                    "gate_pass_clusters": gate_pass,
                    "gate_pass_rate": (float(gate_pass) / float(eligible)) if eligible > 0 else 0.0,
                    "replaced_clusters": replaced,
                    "host_same_commit_clusters": _safe_int(payload.get("host_same_commit_clusters", 0), 0),
                    "delta_replaced_clusters": delta_replaced,
                    "delta_commit_pairs": delta_add,
                    "delta_drop_pairs": delta_drop,
                    "net_delta_pairs": delta_add - delta_drop,
                    "delta_add_per_replaced": (float(delta_add) / float(delta_replaced)) if delta_replaced > 0 else 0.0,
                    "delta_drop_per_replaced": (float(delta_drop) / float(delta_replaced)) if delta_replaced > 0 else 0.0,
                    "all_defer_clusters": _safe_int(payload.get("all_defer_clusters", 0), 0),
                    "empty_pair_candidate_clusters": _safe_int(payload.get("empty_pair_candidate_clusters", 0), 0),
                    "post_filter_empty_clusters": _safe_int(payload.get("post_filter_empty_clusters", 0), 0),
                    "blocked_tracks": _safe_int(payload.get("blocked_tracks", 0), 0),
                    "budget_filtered_clusters": _safe_int(payload.get("budget_filtered_clusters", 0), 0),
                    "margin_filtered_pairs": _safe_int(payload.get("margin_filtered_pairs", 0), 0),
                    "capped_commit_pairs": _safe_int(payload.get("capped_commit_pairs", 0), 0),
                }
            )
    return pd.DataFrame(rows)


def build_plugin_aggregate(plugin_seq_df: pd.DataFrame) -> pd.DataFrame:
    if plugin_seq_df.empty:
        return pd.DataFrame(columns=["seq"])
    agg_rows: list[dict[str, Any]] = []
    for seq, sub in plugin_seq_df.groupby("seq", sort=True):
        delta_add_total = int(sub["delta_commit_pairs"].sum())
        delta_drop_total = int(sub["delta_drop_pairs"].sum())
        agg_rows.append(
            {
                "seq": seq,
                "plugin_runs": int(sub["run_name"].nunique()),
                "plugin_gate_pass_rate_mean": float(sub["gate_pass_rate"].mean()),
                "plugin_replaced_clusters_mean": float(sub["replaced_clusters"].mean()),
                "plugin_delta_commit_pairs_total": delta_add_total,
                "plugin_delta_drop_pairs_total": delta_drop_total,
                "plugin_net_delta_pairs_total": delta_add_total - delta_drop_total,
                "plugin_delta_add_per_replaced_mean": float(sub["delta_add_per_replaced"].mean()),
                "plugin_delta_drop_per_replaced_mean": float(sub["delta_drop_per_replaced"].mean()),
                "plugin_any_positive_add": int((sub["delta_commit_pairs"] > 0).any()),
                "plugin_all_drop_dominant": int(((sub["net_delta_pairs"] < 0) & (sub["delta_drop_pairs"] > 0)).all()),
                "plugin_all_zero_add": int((sub["delta_commit_pairs"] <= 0).all()),
                "plugin_all_zero_or_negative_net": int((sub["net_delta_pairs"] <= 0).all()),
            }
        )
    return pd.DataFrame(agg_rows)


def build_carrier_correlation_rows(per_seq_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(per_seq_csv)
    rows: list[dict[str, Any]] = []
    carriers = sorted(df["carrier"].dropna().astype(str).unique().tolist())
    for metric in ("HOTA", "IDF1", "AssA"):
        pivot = df.pivot(index="seq", columns="carrier", values=metric)
        rank_pivot = pivot.rank(axis=0, method="average", ascending=True)
        for idx, carrier_a in enumerate(carriers):
            for carrier_b in carriers[idx + 1 :]:
                pair = pivot[[carrier_a, carrier_b]].dropna()
                rank_pair = rank_pivot[[carrier_a, carrier_b]].dropna()
                pearson = pair[carrier_a].corr(pair[carrier_b], method="pearson") if len(pair) >= 2 else float("nan")
                spearman = (
                    rank_pair[carrier_a].corr(rank_pair[carrier_b], method="pearson")
                    if len(rank_pair) >= 2
                    else float("nan")
                )
                rows.append(
                    {
                        "metric": metric,
                        "carrier_a": carrier_a,
                        "carrier_b": carrier_b,
                        "shared_sequences": int(len(pair)),
                        "pearson": float(pearson) if pd.notna(pearson) else float("nan"),
                        "spearman": float(spearman) if pd.notna(spearman) else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def build_hypothesis_rows(
    seq_df: pd.DataFrame,
    *,
    top_k_common_hard: int,
    official_top_k_error: int,
) -> pd.DataFrame:
    if seq_df.empty:
        return pd.DataFrame(columns=["hypothesis"])
    common_hard_df = (
        seq_df.sort_values(["mean_HOTA", "mean_IDF1", "seq"], ascending=[True, True, True])
        .head(int(top_k_common_hard))
        .copy()
    )
    official_top_error_df = (
        seq_df.sort_values(
            ["official_top1_error_groups", "official_positive_top1_error_rate", "seq"],
            ascending=[False, False, True],
        )
        .head(int(official_top_k_error))
        .copy()
    )
    common_hard_seqs = set(common_hard_df["seq"].astype(str).tolist())
    official_top_error_seqs = set(official_top_error_df["seq"].astype(str).tolist())
    overlap = sorted(common_hard_seqs & official_top_error_seqs)
    common_but_not_official = sorted(common_hard_seqs - official_top_error_seqs)
    skipped_large_series = (
        common_hard_df["recoverable_skipped_large_count"]
        if "recoverable_skipped_large_count" in common_hard_df.columns
        else pd.Series([0] * len(common_hard_df), index=common_hard_df.index)
    )
    plugin_drop_series = (
        common_hard_df["plugin_all_drop_dominant"]
        if "plugin_all_drop_dominant" in common_hard_df.columns
        else pd.Series([0] * len(common_hard_df), index=common_hard_df.index)
    )
    skipped_large_common = sorted(
        common_hard_df.loc[skipped_large_series.fillna(0) > 0, "seq"].astype(str).tolist()
    )
    plugin_drop_common = sorted(
        common_hard_df.loc[plugin_drop_series.fillna(0) > 0, "seq"].astype(str).tolist()
    )
    rows = [
        {
            "hypothesis": "common_hard_consensus_sequences",
            "sequence_count": int(len(common_hard_df)),
            "sequences": ",".join(common_hard_df["seq"].astype(str).tolist()),
            "support_value": float(common_hard_df["mean_HOTA"].mean()) if len(common_hard_df) > 0 else 0.0,
            "note": "Bottom sequences by mean HOTA across clean official ByteTrack / BoT-SORT / StrongSORT baselines.",
        },
        {
            "hypothesis": "common_hard_and_official_assoc_error_overlap",
            "sequence_count": int(len(overlap)),
            "sequences": ",".join(overlap),
            "support_value": float(len(overlap) / max(len(common_hard_df), 1)),
            "note": "Shared hard sequences that also rank in official ByteTrack top association-error slices.",
        },
        {
            "hypothesis": "common_hard_but_not_official_assoc_error_heavy",
            "sequence_count": int(len(common_but_not_official)),
            "sequences": ",".join(common_but_not_official),
            "support_value": float(len(common_but_not_official) / max(len(common_hard_df), 1)),
            "note": "Shared hard sequences not explained by official local top1-error concentration alone.",
        },
        {
            "hypothesis": "common_hard_with_skipped_large_recoverables",
            "sequence_count": int(len(skipped_large_common)),
            "sequences": ",".join(skipped_large_common),
            "support_value": float(len(skipped_large_common) / max(len(common_hard_df), 1)),
            "note": "Shared hard sequences where focused official recoverable audit already shows component-size coverage gaps.",
        },
        {
            "hypothesis": "plugin_drop_dominant_on_common_hard",
            "sequence_count": int(len(plugin_drop_common)),
            "sequences": ",".join(plugin_drop_common),
            "support_value": float(len(plugin_drop_common) / max(len(common_hard_df), 1)),
            "note": "Shared hard sequences where current learned plugin is consistently host-relative drop dominant across audited recipes.",
        },
    ]
    return pd.DataFrame(rows)


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
        "scripts/run_cross_host_failure_mode_audit.py",
        "--dataset",
        "MOT17",
        "--split",
        "val_half",
        "--tracker-family",
        "cross_host_failure_modes",
        "--variant",
        f"top{int(args.top_k_common_hard)}_officialerr{int(args.official_top_k_error)}",
        "--tag",
        out_dir.name,
        "--run-root",
        str(out_dir.resolve()),
        "--summary-csv",
        str(summary_csv.resolve()),
        "--notes",
        "Bridge clean cross-host hard sequences with official ByteTrack local failure semantics and plugin delta diagnostics.",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    result_csv = out_dir / "result.csv"

    running_row = {
        "cross_host_per_seq": str(Path(args.cross_host_per_seq).resolve()),
        "official_failure_per_seq": str(Path(args.official_failure_per_seq).resolve()),
        "recoverable_reasons_csv": str(Path(args.recoverable_reasons_csv).resolve()),
        "top_k_common_hard": int(args.top_k_common_hard),
        "official_top_k_error": int(args.official_top_k_error),
        "status": "running",
        "error": "",
    }
    write_summary(summary_csv, [running_row])
    write_summary(result_csv, [running_row])

    status = "failed"
    error = ""
    try:
        per_seq_csv = Path(args.cross_host_per_seq).resolve()
        official_failure_csv = Path(args.official_failure_per_seq).resolve()
        recoverable_reasons_csv = Path(args.recoverable_reasons_csv).resolve()
        plugin_specs = parse_plugin_specs(args.plugin_diag)

        seq_df, carriers = load_cross_host_sequence_table(per_seq_csv)
        official_df = load_official_failure_table(official_failure_csv)
        recoverable_df = load_recoverable_reason_table(recoverable_reasons_csv)
        plugin_seq_df = load_plugin_sequence_table(plugin_specs)
        plugin_agg_df = build_plugin_aggregate(plugin_seq_df)

        merged = seq_df.merge(official_df, on="seq", how="left")
        if not recoverable_df.empty:
            merged = merged.merge(recoverable_df, on="seq", how="left")
        if not plugin_agg_df.empty:
            merged = merged.merge(plugin_agg_df, on="seq", how="left")

        for column, fill_value in (
            ("recoverable_skipped_large_count", 0),
            ("plugin_all_drop_dominant", 0),
            ("plugin_any_positive_add", 0),
            ("plugin_delta_commit_pairs_total", 0),
            ("plugin_delta_drop_pairs_total", 0),
        ):
            if column not in merged.columns:
                merged[column] = fill_value

        merged["is_common_hard"] = 0
        common_hard_index = (
            merged.sort_values(["mean_HOTA", "mean_IDF1", "seq"], ascending=[True, True, True])
            .head(int(args.top_k_common_hard))
            .index
        )
        merged.loc[common_hard_index, "is_common_hard"] = 1

        official_error_index = (
            merged.sort_values(
                ["official_top1_error_groups", "official_positive_top1_error_rate", "seq"],
                ascending=[False, False, True],
            )
            .head(int(args.official_top_k_error))
            .index
        )
        merged["is_official_top_error_sequence"] = 0
        merged.loc[official_error_index, "is_official_top_error_sequence"] = 1

        ambiguous_median = (
            float(merged["official_ambiguous_rate"].median())
            if "official_ambiguous_rate" in merged.columns
            else 0.0
        )
        recoverable_median = (
            float(merged["official_recoverable_rate"].median())
            if "official_recoverable_rate" in merged.columns
            else 0.0
        )
        tags: list[list[str]] = []
        for row in merged.itertuples(index=False):
            row_tags: list[str] = []
            if int(getattr(row, "is_common_hard", 0)) == 1:
                row_tags.append("common_hard")
            if int(getattr(row, "is_official_top_error_sequence", 0)) == 1:
                row_tags.append("official_assoc_error_heavy")
            if _safe_int(getattr(row, "recoverable_skipped_large_count", 0), 0) > 0:
                row_tags.append("skipped_large_recoverables")
            if _safe_int(getattr(row, "plugin_all_drop_dominant", 0), 0) == 1:
                row_tags.append("plugin_drop_dominant")
            if _safe_int(getattr(row, "plugin_any_positive_add", 0), 0) == 1:
                row_tags.append("plugin_has_any_positive_add")
            if _safe_float(getattr(row, "official_ambiguous_rate", 0.0), 0.0) > ambiguous_median:
                row_tags.append("above_median_ambiguity")
            if _safe_float(getattr(row, "official_recoverable_rate", 0.0), 0.0) > recoverable_median:
                row_tags.append("above_median_recoverable")
            tags.append(row_tags)
        merged["hypothesis_tags"] = [",".join(item) for item in tags]

        corr_df = build_carrier_correlation_rows(per_seq_csv)
        hypothesis_df = build_hypothesis_rows(
            merged,
            top_k_common_hard=int(args.top_k_common_hard),
            official_top_k_error=int(args.official_top_k_error),
        )

        merged.to_csv(out_dir / "sequence_failure_modes.csv", index=False)
        plugin_seq_df.to_csv(out_dir / "plugin_sequence_summary.csv", index=False)
        corr_df.to_csv(out_dir / "carrier_rank_correlation.csv", index=False)
        hypothesis_df.to_csv(out_dir / "hypothesis_summary.csv", index=False)

        common_hard_sequences = (
            merged.loc[merged["is_common_hard"] == 1, "seq"].astype(str).tolist()
        )
        official_top_error_sequences = (
            merged.loc[merged["is_official_top_error_sequence"] == 1, "seq"].astype(str).tolist()
        )
        overlap_sequences = sorted(set(common_hard_sequences) & set(official_top_error_sequences))
        common_only_sequences = sorted(set(common_hard_sequences) - set(official_top_error_sequences))
        plugin_drop_dominant_common = (
            merged.loc[
                (merged["is_common_hard"] == 1) & (merged["plugin_all_drop_dominant"].fillna(0) == 1),
                "seq",
            ]
            .astype(str)
            .tolist()
        )

        summary_row = {
            "cross_host_per_seq": str(per_seq_csv),
            "official_failure_per_seq": str(official_failure_csv),
            "recoverable_reasons_csv": str(recoverable_reasons_csv),
            "plugin_runs": json.dumps(sorted(plugin_specs.keys())),
            "carriers": json.dumps(carriers),
            "sequence_count": int(len(merged)),
            "top_k_common_hard": int(args.top_k_common_hard),
            "official_top_k_error": int(args.official_top_k_error),
            "common_hard_sequences": json.dumps(common_hard_sequences),
            "official_top_error_sequences": json.dumps(official_top_error_sequences),
            "common_vs_official_error_overlap": json.dumps(overlap_sequences),
            "common_hard_but_not_official_error_heavy": json.dumps(common_only_sequences),
            "plugin_drop_dominant_common_hard": json.dumps(plugin_drop_dominant_common),
            "mean_common_hard_HOTA": float(
                merged.loc[merged["is_common_hard"] == 1, "mean_HOTA"].mean()
            )
            if int((merged["is_common_hard"] == 1).sum()) > 0
            else 0.0,
            "mean_official_error_rate_common_hard": float(
                merged.loc[merged["is_common_hard"] == 1, "official_positive_top1_error_rate"].mean()
            )
            if int((merged["is_common_hard"] == 1).sum()) > 0
            else 0.0,
            "status": "success",
            "error": "",
        }
        write_summary(summary_csv, [summary_row])
        write_summary(result_csv, [summary_row])
        append_registry(args, out_dir=out_dir, summary_csv=summary_csv, status="success")
        status = "success"
    except Exception as exc:
        error = str(exc)
        failed_row = dict(running_row)
        failed_row.update({"status": "failed", "error": error})
        write_summary(summary_csv, [failed_row])
        write_summary(result_csv, [failed_row])
        try:
            append_registry(args, out_dir=out_dir, summary_csv=summary_csv, status="failed")
        except Exception:
            pass
        raise

    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
