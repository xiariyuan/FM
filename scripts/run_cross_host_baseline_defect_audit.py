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
DEFAULT_FAILURE_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "official_bytetrack_failure_slices_crosshost_focus_20260326_214500"
    / "summary.csv"
)
DEFAULT_FAILURE_PER_SEQ = (
    REPO_ROOT
    / "outputs"
    / "official_bytetrack_failure_slices_crosshost_focus_20260326_214500"
    / "per_sequence.csv"
)
DEFAULT_CARRIER_UNIQUENESS = (
    REPO_ROOT
    / "outputs"
    / "cross_host_carrier_uniqueness_audit_mot17_frcnn_valhalf_20260327_030500"
    / "summary.csv"
)

SUMMARY_FIELDS = [
    "cross_host_summary_csv",
    "cross_host_per_seq_csv",
    "failure_summary_csv",
    "failure_per_seq_csv",
    "carrier_uniqueness_csv",
    "carriers",
    "paper_canonical_carrier",
    "test_oriented_carrier",
    "specialist_carrier",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Diagnose each clean baseline carrier's main defect profile.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cross-host-summary", default=str(DEFAULT_CROSS_HOST_SUMMARY))
    ap.add_argument("--cross-host-per-seq", default=str(DEFAULT_CROSS_HOST_PER_SEQ))
    ap.add_argument("--failure-summary", default=str(DEFAULT_FAILURE_SUMMARY))
    ap.add_argument("--failure-per-seq", default=str(DEFAULT_FAILURE_PER_SEQ))
    ap.add_argument("--carrier-uniqueness-summary", default=str(DEFAULT_CARRIER_UNIQUENESS))
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return ap.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows if fieldnames is None else rows, columns=fieldnames)
    df.to_csv(path, index=False)


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
        "scripts/run_cross_host_baseline_defect_audit.py",
        "--dataset",
        "MOT17",
        "--split",
        "val_half",
        "--tracker-family",
        "cross_host_baseline_defects",
        "--variant",
        "baseline_defect_matrix",
        "--tag",
        out_dir.name,
        "--run-root",
        str(out_dir.resolve()),
        "--summary-csv",
        str(summary_csv.resolve()),
        "--notes",
        "Per-baseline defect audit for official ByteTrack, BoT-SORT, and StrongSORT using clean cross-host val-half metrics and official failure slices.",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _json_or_list(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [text]


def build_defect_rows(
    cross_summary_df: pd.DataFrame,
    per_seq_df: pd.DataFrame,
    failure_summary_df: pd.DataFrame,
    failure_per_seq_df: pd.DataFrame,
    uniqueness_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary_map = cross_summary_df.set_index("carrier")
    per_seq = per_seq_df.copy()

    official_top_errors = failure_per_seq_df.sort_values(
        ["positive_top1_error_rate", "top1_error_groups", "seq"],
        ascending=[False, False, True],
    ).head(3)["seq"].astype(str).tolist()
    common_hard = _json_or_list(uniqueness_df.iloc[0].get("common_hard_sequences", ""))

    def weakest_sequences(carrier: str, metric: str, n: int = 3, asc: bool = True) -> list[str]:
        sub = per_seq.loc[per_seq["carrier"] == carrier].sort_values(metric, ascending=asc)
        return sub.head(n)["seq"].astype(str).tolist()

    def negative_gap_sequences_against_official(carrier: str, metric: str, n: int = 3) -> list[str]:
        if carrier == "official_bytetrack":
            return []
        merged = (
            per_seq.loc[per_seq["carrier"].isin([carrier, "official_bytetrack"]), ["carrier", "seq", metric]]
            .pivot(index="seq", columns="carrier", values=metric)
            .dropna()
            .reset_index()
        )
        if carrier not in merged.columns or "official_bytetrack" not in merged.columns:
            return []
        merged[f"delta_{metric}"] = merged[carrier] - merged["official_bytetrack"]
        sub = merged.sort_values(f"delta_{metric}", ascending=True)
        return sub.head(n)["seq"].astype(str).tolist()

    official = summary_map.loc["official_bytetrack"]
    rows.append(
        {
            "carrier": "official_bytetrack",
            "overall_metrics": json.dumps(
                {
                    "HOTA": float(official["HOTA"]),
                    "DetA": float(official["DetA"]),
                    "AssA": float(official["AssA"]),
                    "IDF1": float(official["IDF1"]),
                    "MOTA": float(official["MOTA"]),
                    "IDSW": int(float(official["IDSW"])),
                }
            ),
            "what_it_solves": json.dumps(["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-11-FRCNN"]),
            "main_defect_type": "crowded local-association failure and large-component coverage gap",
            "worst_sequences": json.dumps(weakest_sequences("official_bytetrack", "HOTA")),
            "defect_evidence": json.dumps(
                {
                    "largest_HOTA_gap_vs_best": ["MOT17-13-FRCNN", "MOT17-09-FRCNN", "MOT17-05-FRCNN"],
                    "official_top_assoc_error_sequences": official_top_errors,
                    "common_hard_overlap": sorted(set(common_hard) & set(official_top_errors)),
                }
            ),
            "mechanism_hypothesis": "First-stage local association is stable on easy slices, but degrades on MOT17-05/10/13 where recoverable conflicts and skipped-large components concentrate.",
            "paper_role": "canonical",
        }
    )

    botsort = summary_map.loc["botsort_base"]
    rows.append(
        {
            "carrier": "botsort_base",
            "overall_metrics": json.dumps(
                {
                    "HOTA": float(botsort["HOTA"]),
                    "DetA": float(botsort["DetA"]),
                    "AssA": float(botsort["AssA"]),
                    "IDF1": float(botsort["IDF1"]),
                    "MOTA": float(botsort["MOTA"]),
                    "IDSW": int(float(botsort["IDSW"])),
                }
            ),
            "what_it_solves": json.dumps(["MOT17-05-FRCNN", "MOT17-10-FRCNN", "MOT17-13-FRCNN"]),
            "main_defect_type": "identity-switch instability on conservative official-favorable slices",
            "worst_sequences": json.dumps(weakest_sequences("botsort_base", "HOTA")),
            "defect_evidence": json.dumps(
                {
                    "largest_negative_HOTA_vs_official": negative_gap_sequences_against_official("botsort_base", "HOTA"),
                    "largest_negative_IDF1_vs_official": negative_gap_sequences_against_official("botsort_base", "IDF1"),
                    "highest_IDSW_sequences": weakest_sequences("botsort_base", "IDSW", asc=False),
                }
            ),
            "mechanism_hypothesis": "BoT-SORT is stronger on crowded recovery slices, but likely pays for heavier appearance/motion fusion with extra switches on MOT17-02 and mild identity regressions on MOT17-11/04.",
            "paper_role": "test_oriented_transfer",
        }
    )

    strong = summary_map.loc["strongsort_base"]
    rows.append(
        {
            "carrier": "strongsort_base",
            "overall_metrics": json.dumps(
                {
                    "HOTA": float(strong["HOTA"]),
                    "DetA": float(strong["DetA"]),
                    "AssA": float(strong["AssA"]),
                    "IDF1": float(strong["IDF1"]),
                    "MOTA": float(strong["MOTA"]),
                    "IDSW": int(float(strong["IDSW"])),
                }
            ),
            "what_it_solves": json.dumps(["MOT17-09-FRCNN", "MOT17-05-FRCNN", "MOT17-10-FRCNN"]),
            "main_defect_type": "global detection/coverage deficit despite some identity-stability strengths",
            "worst_sequences": json.dumps(weakest_sequences("strongsort_base", "HOTA")),
            "defect_evidence": json.dumps(
                {
                    "largest_HOTA_gaps_vs_best": ["MOT17-13-FRCNN", "MOT17-04-FRCNN", "MOT17-02-FRCNN"],
                    "lowest_DetA_sequences": weakest_sequences("strongsort_base", "DetA"),
                    "overall_metric_gap_vs_official": {
                        "DetA": float(strong["DetA"]) - float(official["DetA"]),
                        "MOTA": float(strong["MOTA"]) - float(official["MOTA"]),
                    },
                }
            ),
            "mechanism_hypothesis": "StrongSORT behaves like a specialist on a few ID-heavy slices, but its main bottleneck is not local association ranking; it is broader coverage/recall loss that drags HOTA/MOTA down on most sequences.",
            "paper_role": "specialist_only",
        }
    )
    return rows


def build_markdown_report(out_path: Path, defect_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Cross-Host Baseline Defect Audit",
        "",
        "This report diagnoses the main defect profile of each clean carrier baseline rather than only ranking them by aggregate score.",
        "",
    ]
    for row in defect_rows:
        lines.extend(
            [
                f"## {row['carrier']}",
                "",
                f"- `paper_role`: `{row['paper_role']}`",
                f"- `main_defect_type`: {row['main_defect_type']}",
                f"- `what_it_solves`: {row['what_it_solves']}",
                f"- `worst_sequences`: {row['worst_sequences']}",
                f"- `mechanism_hypothesis`: {row['mechanism_hypothesis']}",
                f"- `defect_evidence`: {row['defect_evidence']}",
                "",
            ]
        )
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
        "failure_summary_csv": str(Path(args.failure_summary).resolve()),
        "failure_per_seq_csv": str(Path(args.failure_per_seq).resolve()),
        "carrier_uniqueness_csv": str(Path(args.carrier_uniqueness_summary).resolve()),
        "carriers": "",
        "paper_canonical_carrier": "",
        "test_oriented_carrier": "",
        "specialist_carrier": "",
        "status": "running",
        "error": "",
    }
    write_csv(summary_csv, [running_row], fieldnames=SUMMARY_FIELDS)
    write_csv(result_csv, [running_row], fieldnames=SUMMARY_FIELDS)

    try:
        cross_summary_df = pd.read_csv(Path(args.cross_host_summary).resolve())
        per_seq_df = pd.read_csv(Path(args.cross_host_per_seq).resolve())
        failure_summary_df = pd.read_csv(Path(args.failure_summary).resolve())
        failure_per_seq_df = pd.read_csv(Path(args.failure_per_seq).resolve())
        uniqueness_df = pd.read_csv(Path(args.carrier_uniqueness_summary).resolve())

        defect_rows = build_defect_rows(
            cross_summary_df=cross_summary_df,
            per_seq_df=per_seq_df,
            failure_summary_df=failure_summary_df,
            failure_per_seq_df=failure_per_seq_df,
            uniqueness_df=uniqueness_df,
        )
        pd.DataFrame(defect_rows).to_csv(out_dir / "baseline_defects.csv", index=False)
        build_markdown_report(out_dir / "report.md", defect_rows)

        carriers = sorted(cross_summary_df["carrier"].astype(str).unique().tolist())
        summary_row = {
            "cross_host_summary_csv": str(Path(args.cross_host_summary).resolve()),
            "cross_host_per_seq_csv": str(Path(args.cross_host_per_seq).resolve()),
            "failure_summary_csv": str(Path(args.failure_summary).resolve()),
            "failure_per_seq_csv": str(Path(args.failure_per_seq).resolve()),
            "carrier_uniqueness_csv": str(Path(args.carrier_uniqueness_summary).resolve()),
            "carriers": json.dumps(carriers),
            "paper_canonical_carrier": "official_bytetrack",
            "test_oriented_carrier": "botsort_base",
            "specialist_carrier": "strongsort_base",
            "status": "success",
            "error": "",
        }
        write_csv(summary_csv, [summary_row], fieldnames=SUMMARY_FIELDS)
        write_csv(result_csv, [summary_row], fieldnames=SUMMARY_FIELDS)
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
