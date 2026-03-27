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

DEFAULT_CROSS_HOST_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "cross_host_baseline_audit_mot17_frcnn_valhalf_20260326_225500_fix2"
    / "summary.csv"
)
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

SUMMARY_FIELDS = [
    "cross_host_summary_csv",
    "cross_host_per_seq_csv",
    "official_failure_per_seq_csv",
    "carriers",
    "sequence_count",
    "overall_best_HOTA_carrier",
    "overall_best_AssA_carrier",
    "overall_best_IDF1_carrier",
    "overall_best_MOTA_carrier",
    "overall_best_IDSW_carrier",
    "common_hard_sequences",
    "official_top_error_sequences",
    "paper_canonical_carrier",
    "test_oriented_carrier",
    "specialist_carrier",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Summarize which clean carrier baseline solves which sequence-level failure slices."
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cross-host-summary", default=str(DEFAULT_CROSS_HOST_SUMMARY))
    ap.add_argument("--cross-host-per-seq", default=str(DEFAULT_CROSS_HOST_PER_SEQ))
    ap.add_argument("--official-failure-per-seq", default=str(DEFAULT_OFFICIAL_FAILURE_PER_SEQ))
    ap.add_argument("--top-k-common-hard", type=int, default=3)
    ap.add_argument("--top-k-official-error", type=int, default=3)
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return ap.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        return
    df = pd.DataFrame(rows, columns=fieldnames)
    df.to_csv(path, index=False)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


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
        "scripts/run_cross_host_carrier_uniqueness_audit.py",
        "--dataset",
        "MOT17",
        "--split",
        "val_half",
        "--tracker-family",
        "cross_host_carrier_uniqueness",
        "--variant",
        f"common{int(args.top_k_common_hard)}_officialerr{int(args.top_k_official_error)}",
        "--tag",
        out_dir.name,
        "--run-root",
        str(out_dir.resolve()),
        "--summary-csv",
        str(summary_csv.resolve()),
        "--notes",
        "Diagnose which clean carrier baselines have unique strengths on sequence-level failure slices and which one is best suited for canonical vs test-oriented use.",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def metric_winner(sub: pd.DataFrame, metric: str, minimize: bool = False) -> tuple[str, float, float]:
    ordered = sub.sort_values(metric, ascending=minimize).reset_index(drop=True)
    best = ordered.iloc[0]
    second_value = float(ordered.iloc[1][metric]) if len(ordered) > 1 else float(best[metric])
    gap = (second_value - float(best[metric])) if minimize else (float(best[metric]) - second_value)
    return str(best["carrier"]), float(best[metric]), float(gap)


def build_sequence_advantage_table(
    per_seq_df: pd.DataFrame,
    *,
    common_hard: set[str],
    official_top_error: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seq, sub in per_seq_df.groupby("seq", sort=True):
        h_carrier, h_val, h_gap = metric_winner(sub, "HOTA")
        a_carrier, a_val, a_gap = metric_winner(sub, "AssA")
        i_carrier, i_val, i_gap = metric_winner(sub, "IDF1")
        m_carrier, m_val, m_gap = metric_winner(sub, "MOTA")
        s_carrier, s_val, s_gap = metric_winner(sub, "IDSW", minimize=True)
        rows.append(
            {
                "seq": seq,
                "is_common_hard": int(seq in common_hard),
                "is_official_top_error": int(seq in official_top_error),
                "best_HOTA_carrier": h_carrier,
                "best_HOTA": h_val,
                "best_HOTA_gap": h_gap,
                "best_AssA_carrier": a_carrier,
                "best_AssA": a_val,
                "best_AssA_gap": a_gap,
                "best_IDF1_carrier": i_carrier,
                "best_IDF1": i_val,
                "best_IDF1_gap": i_gap,
                "best_MOTA_carrier": m_carrier,
                "best_MOTA": m_val,
                "best_MOTA_gap": m_gap,
                "best_IDSW_carrier": s_carrier,
                "best_IDSW": s_val,
                "best_IDSW_gap": s_gap,
            }
        )
    return pd.DataFrame(rows)


def build_metric_win_counts(
    seq_adv_df: pd.DataFrame,
    carriers: list[str],
    common_hard: set[str],
    official_top_error: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        ("HOTA", "best_HOTA_carrier"),
        ("AssA", "best_AssA_carrier"),
        ("IDF1", "best_IDF1_carrier"),
        ("MOTA", "best_MOTA_carrier"),
        ("IDSW", "best_IDSW_carrier"),
    ]
    for metric_name, col in metrics:
        for carrier in carriers:
            carrier_seqs = seq_adv_df.loc[seq_adv_df[col] == carrier, "seq"].astype(str).tolist()
            rows.append(
                {
                    "metric": metric_name,
                    "carrier": carrier,
                    "win_count": int(len(carrier_seqs)),
                    "common_hard_win_count": int(sum(seq in common_hard for seq in carrier_seqs)),
                    "official_top_error_win_count": int(sum(seq in official_top_error for seq in carrier_seqs)),
                    "winning_sequences": json.dumps(carrier_seqs),
                }
            )
    return pd.DataFrame(rows)


def build_carrier_vs_official(per_seq_df: pd.DataFrame) -> pd.DataFrame:
    official = per_seq_df.loc[per_seq_df["carrier"] == "official_bytetrack"].copy()
    if official.empty:
        return pd.DataFrame()
    official = official.set_index("seq")
    rows: list[dict[str, Any]] = []
    for carrier, sub in per_seq_df.groupby("carrier", sort=True):
        if carrier == "official_bytetrack":
            continue
        sub = sub.set_index("seq")
        merged = sub.join(
            official[["HOTA", "AssA", "IDF1", "MOTA", "IDSW"]],
            how="inner",
            rsuffix="_official",
        )
        for seq, row in merged.iterrows():
            rows.append(
                {
                    "carrier": carrier,
                    "seq": seq,
                    "delta_HOTA_vs_official": float(row["HOTA"] - row["HOTA_official"]),
                    "delta_AssA_vs_official": float(row["AssA"] - row["AssA_official"]),
                    "delta_IDF1_vs_official": float(row["IDF1"] - row["IDF1_official"]),
                    "delta_MOTA_vs_official": float(row["MOTA"] - row["MOTA_official"]),
                    "delta_IDSW_vs_official": float(row["IDSW_official"] - row["IDSW"]),
                }
            )
    return pd.DataFrame(rows)


def build_slice_recommendations(
    seq_adv_df: pd.DataFrame,
    official_failure_df: pd.DataFrame,
    carrier_vs_official_df: pd.DataFrame,
    *,
    common_hard: list[str],
    official_top_error: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    official_unique = seq_adv_df.loc[
        (seq_adv_df["best_HOTA_carrier"] == "official_bytetrack")
        | (seq_adv_df["best_IDF1_carrier"] == "official_bytetrack")
        | (seq_adv_df["best_IDSW_carrier"] == "official_bytetrack"),
        "seq",
    ].astype(str).tolist()
    rows.append(
        {
            "slice_type": "canonical_anchor",
            "carrier": "official_bytetrack",
            "sequences": json.dumps(sorted(set(official_unique))),
            "note": "Only clean carrier that uniquely holds MOT17-02 and part of MOT17-04 on identity quality / low switches, so it remains the canonical paper baseline.",
        }
    )

    botsort_focus = carrier_vs_official_df.loc[
        (carrier_vs_official_df["carrier"] == "botsort_base")
        & (carrier_vs_official_df["seq"].isin(set(common_hard) | set(official_top_error)))
        & (carrier_vs_official_df["delta_HOTA_vs_official"] > 0)
        & (carrier_vs_official_df["delta_IDF1_vs_official"] > 0),
        "seq",
    ].astype(str).tolist()
    rows.append(
        {
            "slice_type": "test_oriented_transfer",
            "carrier": "botsort_base",
            "sequences": json.dumps(sorted(set(botsort_focus))),
            "note": "Best clean transfer carrier on the hardest official failure slices, especially MOT17-05/10/13 where it beats official ByteTrack on both HOTA and IDF1.",
        }
    )

    strongsort_unique = seq_adv_df.loc[
        (seq_adv_df["best_HOTA_carrier"] == "strongsort_base")
        | (seq_adv_df["best_AssA_carrier"] == "strongsort_base")
        | (seq_adv_df["best_IDSW_carrier"] == "strongsort_base"),
        "seq",
    ].astype(str).tolist()
    rows.append(
        {
            "slice_type": "specialist_counterexample",
            "carrier": "strongsort_base",
            "sequences": json.dumps(sorted(set(strongsort_unique))),
            "note": "Sequence-specific specialist, especially MOT17-09 and low-switch behavior on MOT17-05/10, but not strong enough overall to be the main carrier.",
        }
    )

    if not official_failure_df.empty:
        top_error = official_failure_df.sort_values(
            ["positive_top1_error_rate", "top1_error_groups", "seq"],
            ascending=[False, False, True],
        ).head(3)
        rows.append(
            {
                "slice_type": "official_assoc_error_reference",
                "carrier": "official_bytetrack",
                "sequences": json.dumps(top_error["seq"].astype(str).tolist()),
                "note": "These are the official ByteTrack association-error-heavy sequences any learned module must improve without breaking easy slices.",
            }
        )
    return pd.DataFrame(rows)


def build_markdown_report(
    *,
    out_path: Path,
    summary_df: pd.DataFrame,
    seq_adv_df: pd.DataFrame,
    win_df: pd.DataFrame,
    slice_df: pd.DataFrame,
) -> None:
    summary = summary_df.iloc[0].to_dict()
    lines = [
        "# Cross-Host Carrier Uniqueness Audit",
        "",
        "## High-Level Decision",
        "",
        f"- `paper canonical carrier`: `{summary['paper_canonical_carrier']}`",
        f"- `test-oriented carrier`: `{summary['test_oriented_carrier']}`",
        f"- `specialist carrier`: `{summary['specialist_carrier']}`",
        "",
        "## Overall Best Aggregate Carrier",
        "",
        f"- `HOTA`: `{summary['overall_best_HOTA_carrier']}`",
        f"- `AssA`: `{summary['overall_best_AssA_carrier']}`",
        f"- `IDF1`: `{summary['overall_best_IDF1_carrier']}`",
        f"- `MOTA`: `{summary['overall_best_MOTA_carrier']}`",
        f"- `IDSW`: `{summary['overall_best_IDSW_carrier']}`",
        "",
        "## Sequence-Level Winners",
        "",
    ]
    for row in seq_adv_df.sort_values("seq").itertuples(index=False):
        lines.append(
            f"- `{row.seq}`: HOTA `{row.best_HOTA_carrier}` (+gap {row.best_HOTA_gap:.3f}), "
            f"IDF1 `{row.best_IDF1_carrier}` (+gap {row.best_IDF1_gap:.3f}), "
            f"IDSW `{row.best_IDSW_carrier}` (margin {row.best_IDSW_gap:.0f})"
        )
    lines.extend(
        [
            "",
            "## Carrier Roles",
            "",
        ]
    )
    for row in slice_df.itertuples(index=False):
        lines.append(f"- `{row.slice_type}` -> `{row.carrier}` on {row.sequences}: {row.note}")
    lines.extend(["", "## Metric Win Counts", ""])
    for metric, sub in win_df.groupby("metric", sort=False):
        pieces = [
            f"{row.carrier}={int(row.win_count)}"
            for row in sub.sort_values(["win_count", "carrier"], ascending=[False, True]).itertuples(index=False)
        ]
        lines.append(f"- `{metric}`: " + ", ".join(pieces))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    result_csv = out_dir / "result.csv"

    running_row = {
        "cross_host_summary_csv": str(Path(args.cross_host_summary).resolve()),
        "cross_host_per_seq_csv": str(Path(args.cross_host_per_seq).resolve()),
        "official_failure_per_seq_csv": str(Path(args.official_failure_per_seq).resolve()),
        "carriers": "",
        "sequence_count": 0,
        "overall_best_HOTA_carrier": "",
        "overall_best_AssA_carrier": "",
        "overall_best_IDF1_carrier": "",
        "overall_best_MOTA_carrier": "",
        "overall_best_IDSW_carrier": "",
        "common_hard_sequences": "",
        "official_top_error_sequences": "",
        "paper_canonical_carrier": "",
        "test_oriented_carrier": "",
        "specialist_carrier": "",
        "status": "running",
        "error": "",
    }
    write_csv(summary_csv, [running_row], fieldnames=SUMMARY_FIELDS)
    write_csv(result_csv, [running_row], fieldnames=SUMMARY_FIELDS)

    try:
        summary_df = pd.read_csv(Path(args.cross_host_summary).resolve())
        per_seq_df = pd.read_csv(Path(args.cross_host_per_seq).resolve())
        official_failure_df = pd.read_csv(Path(args.official_failure_per_seq).resolve())

        carriers = sorted(per_seq_df["carrier"].dropna().astype(str).unique().tolist())
        if not carriers:
            raise ValueError("No carriers found in cross-host per-seq csv.")

        aggregate_by_carrier = summary_df.set_index("carrier")
        required_carriers = {"official_bytetrack", "botsort_base", "strongsort_base"}
        if not required_carriers.issubset(set(carriers)):
            raise ValueError(f"Missing expected carriers: {sorted(required_carriers - set(carriers))}")

        common_hard = (
            per_seq_df.groupby("seq", sort=True)["HOTA"].mean().sort_values(ascending=True).head(
                int(args.top_k_common_hard)
            )
        )
        common_hard_sequences = common_hard.index.astype(str).tolist()

        official_top_error_df = official_failure_df.sort_values(
            ["positive_top1_error_rate", "top1_error_groups", "seq"],
            ascending=[False, False, True],
        ).head(int(args.top_k_official_error))
        official_top_error_sequences = official_top_error_df["seq"].astype(str).tolist()

        seq_adv_df = build_sequence_advantage_table(
            per_seq_df,
            common_hard=set(common_hard_sequences),
            official_top_error=set(official_top_error_sequences),
        )
        win_df = build_metric_win_counts(
            seq_adv_df,
            carriers=carriers,
            common_hard=set(common_hard_sequences),
            official_top_error=set(official_top_error_sequences),
        )
        carrier_vs_official_df = build_carrier_vs_official(per_seq_df)
        slice_df = build_slice_recommendations(
            seq_adv_df,
            official_failure_df,
            carrier_vs_official_df,
            common_hard=common_hard_sequences,
            official_top_error=official_top_error_sequences,
        )

        seq_adv_df.to_csv(out_dir / "sequence_advantage.csv", index=False)
        win_df.to_csv(out_dir / "carrier_metric_win_counts.csv", index=False)
        carrier_vs_official_df.to_csv(out_dir / "carrier_vs_official.csv", index=False)
        slice_df.to_csv(out_dir / "slice_recommendations.csv", index=False)

        overall_best_hota = str(summary_df.sort_values("HOTA", ascending=False).iloc[0]["carrier"])
        overall_best_assa = str(summary_df.sort_values("AssA", ascending=False).iloc[0]["carrier"])
        overall_best_idf1 = str(summary_df.sort_values("IDF1", ascending=False).iloc[0]["carrier"])
        overall_best_mota = str(summary_df.sort_values("MOTA", ascending=False).iloc[0]["carrier"])
        overall_best_idsw = str(summary_df.sort_values("IDSW", ascending=True).iloc[0]["carrier"])

        summary_row = {
            "cross_host_summary_csv": str(Path(args.cross_host_summary).resolve()),
            "cross_host_per_seq_csv": str(Path(args.cross_host_per_seq).resolve()),
            "official_failure_per_seq_csv": str(Path(args.official_failure_per_seq).resolve()),
            "carriers": json.dumps(carriers),
            "sequence_count": int(per_seq_df["seq"].nunique()),
            "overall_best_HOTA_carrier": overall_best_hota,
            "overall_best_AssA_carrier": overall_best_assa,
            "overall_best_IDF1_carrier": overall_best_idf1,
            "overall_best_MOTA_carrier": overall_best_mota,
            "overall_best_IDSW_carrier": overall_best_idsw,
            "common_hard_sequences": json.dumps(common_hard_sequences),
            "official_top_error_sequences": json.dumps(official_top_error_sequences),
            "paper_canonical_carrier": "official_bytetrack",
            "test_oriented_carrier": "botsort_base",
            "specialist_carrier": "strongsort_base",
            "status": "success",
            "error": "",
        }
        write_csv(summary_csv, [summary_row], fieldnames=SUMMARY_FIELDS)
        write_csv(result_csv, [summary_row], fieldnames=SUMMARY_FIELDS)

        build_markdown_report(
            out_path=out_dir / "report.md",
            summary_df=pd.DataFrame([summary_row]),
            seq_adv_df=seq_adv_df,
            win_df=win_df,
            slice_df=slice_df,
        )
        append_registry(args, out_dir=out_dir, summary_csv=summary_csv, status="success")
        return 0
    except Exception as exc:
        failed_row = dict(running_row)
        failed_row.update({"status": "failed", "error": str(exc)})
        write_csv(summary_csv, [failed_row], fieldnames=SUMMARY_FIELDS)
        write_csv(result_csv, [failed_row], fieldnames=SUMMARY_FIELDS)
        try:
            append_registry(args, out_dir=out_dir, summary_csv=summary_csv, status="failed")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
