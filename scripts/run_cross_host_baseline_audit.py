#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_botsort_halfval_trackeval.py"
REGISTRY_SCRIPT = REPO_ROOT / "scripts" / "append_experiment_record.py"
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

DEFAULT_CARRIERS = {
    "official_bytetrack": {
        "path": REPO_ROOT
        / "outputs"
        / "official_bytetrack_largecover_gateedit_pair_confirm_best_v2_thresh060_deltadiag_20260326_223500"
        / "00_host_only"
        / "track_results",
        "remap_from_fullval": False,
    },
    "botsort_base": {
        "path": REPO_ROOT
        / "external"
        / "BoT-SORT-main"
        / "YOLOX_outputs"
        / "laplace_mot17_val_base"
        / "track_results",
        "remap_from_fullval": False,
    },
    "strongsort_base": {
        "path": REPO_ROOT
        / "outputs"
        / "strongsort_ltra"
        / "MOT17_val"
        / "results"
        / "base",
        "remap_from_fullval": True,
    },
}

SUMMARY_FIELDS = [
    "carrier",
    "dataset",
    "split",
    "detector_ext",
    "source_results_dir",
    "subset_results_dir",
    "remap_from_fullval",
    "eval_dir",
    "tracker_name",
    "HOTA",
    "DetA",
    "AssA",
    "IDF1",
    "MOTA",
    "IDSW",
    "status",
    "error",
]

PER_SEQ_FIELDS = [
    "carrier",
    "seq",
    "HOTA",
    "DetA",
    "AssA",
    "IDF1",
    "MOTA",
    "IDSW",
]

COMMON_SEQ_FIELDS = [
    "seq",
    "carrier_count",
    "mean_HOTA",
    "mean_AssA",
    "mean_IDF1",
    "min_HOTA",
    "max_HOTA",
    "min_IDF1",
    "max_IDF1",
    "carriers",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a cross-host baseline audit on MOT17 half-val.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset", default="MOT17", choices=["MOT17", "MOT20"])
    ap.add_argument("--split", default="val_half")
    ap.add_argument("--detector-ext", default="FRCNN")
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--data-root", default="/gemini/code/datasets")
    ap.add_argument(
        "--carrier",
        action="append",
        default=[],
        help="Carrier spec name=/abs/or/rel/results_dir. If omitted, built-in carriers are used.",
    )
    ap.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return ap.parse_args()


def parse_carrier_specs(items: list[str]) -> dict[str, dict[str, Any]]:
    if not items:
        return {
            name: {
                "path": Path(spec["path"]).resolve(),
                "remap_from_fullval": bool(spec.get("remap_from_fullval", False)),
            }
            for name, spec in DEFAULT_CARRIERS.items()
        }
    carriers: dict[str, dict[str, Any]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --carrier spec, expected name=path: {item}")
        name, raw_path = item.split("=", 1)
        carriers[name.strip()] = {
            "path": Path(raw_path).expanduser().resolve(),
            "remap_from_fullval": False,
        }
    return carriers


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def filter_results_dir(src_dir: Path, dst_dir: Path, detector_ext: str) -> list[str]:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    suffix = f"-{detector_ext.upper()}.TXT"
    for path in sorted(src_dir.glob("*.txt")):
        if not path.name.upper().endswith(suffix):
            continue
        shutil.copy2(path, dst_dir / path.name)
        kept.append(path.stem)
    return kept


def parse_trackeval_summary(summary_path: Path) -> dict[str, str]:
    lines = [line.strip() for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Invalid TrackEval summary: {summary_path}")
    keys = lines[0].split()
    vals = lines[1].split()
    if len(keys) != len(vals):
        raise ValueError(f"TrackEval summary mismatch: {summary_path}")
    return dict(zip(keys, vals))


def load_per_seq(detailed_csv: Path) -> list[dict[str, str]]:
    with detailed_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row.get("seq", "") != "COMBINED"]


def run_eval_for_carrier(
    args: argparse.Namespace,
    *,
    carrier: str,
    src_dir: Path,
    remap_from_fullval: bool,
    out_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    subset_dir = out_dir / "subset_results" / carrier
    eval_work_dir = out_dir / "eval" / carrier
    tracker_name = f"{carrier}_{args.dataset.lower()}_{args.detector_ext.lower()}_{args.split}"
    summary_row: dict[str, Any] = {
        "carrier": carrier,
        "dataset": args.dataset,
        "split": args.split,
        "detector_ext": args.detector_ext,
        "source_results_dir": str(src_dir.resolve()),
        "subset_results_dir": str(subset_dir.resolve()),
        "remap_from_fullval": int(bool(remap_from_fullval)),
        "eval_dir": "",
        "tracker_name": tracker_name,
        "HOTA": "",
        "DetA": "",
        "AssA": "",
        "IDF1": "",
        "MOTA": "",
        "IDSW": "",
        "status": "running",
        "error": "",
    }
    per_seq_rows: list[dict[str, Any]] = []

    try:
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Results dir not found: {src_dir}")
        kept = filter_results_dir(src_dir, subset_dir, args.detector_ext)
        if not kept:
            raise FileNotFoundError(f"No {args.detector_ext} result files found in {src_dir}")
        cmd = [
            args.python_bin,
            str(EVAL_SCRIPT),
            "--dataset",
            args.dataset,
            "--data-root",
            str(Path(args.data_root).resolve()),
            "--results-dir",
            str(subset_dir.resolve()),
            "--tracker-name",
            tracker_name,
            "--work-dir",
            str(eval_work_dir.resolve()),
        ]
        if remap_from_fullval:
            cmd.append("--remap-results-from-fullval")
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        summary_path = eval_work_dir / "eval" / tracker_name / "pedestrian_summary.txt"
        detailed_path = eval_work_dir / "eval" / tracker_name / "pedestrian_detailed.csv"
        metrics = parse_trackeval_summary(summary_path)
        summary_row.update(
            {
                "eval_dir": str((eval_work_dir / "eval" / tracker_name).resolve()),
                "HOTA": metrics.get("HOTA", ""),
                "DetA": metrics.get("DetA", ""),
                "AssA": metrics.get("AssA", ""),
                "IDF1": metrics.get("IDF1", ""),
                "MOTA": metrics.get("MOTA", ""),
                "IDSW": metrics.get("IDSW", ""),
                "status": "success",
                "error": "",
            }
        )
        for row in load_per_seq(detailed_path):
            per_seq_rows.append(
                {
                    "carrier": carrier,
                    "seq": row.get("seq", ""),
                    "HOTA": row.get("HOTA___AUC", ""),
                    "DetA": row.get("DetA___AUC", ""),
                    "AssA": row.get("AssA___AUC", ""),
                    "IDF1": row.get("IDF1", ""),
                    "MOTA": row.get("MOTA", ""),
                    "IDSW": row.get("IDSW", ""),
                }
            )
    except Exception as exc:
        summary_row.update({"status": "failed", "error": str(exc)})

    return summary_row, per_seq_rows


def build_common_seq_rows(per_seq_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_seq_rows:
        grouped[str(row["seq"])].append(row)
    rows: list[dict[str, Any]] = []
    for seq, items in grouped.items():
        h_vals = [float(item["HOTA"]) for item in items]
        a_vals = [float(item["AssA"]) for item in items]
        i_vals = [float(item["IDF1"]) for item in items]
        rows.append(
            {
                "seq": seq,
                "carrier_count": len(items),
                "mean_HOTA": f"{sum(h_vals) / len(h_vals):.6f}",
                "mean_AssA": f"{sum(a_vals) / len(a_vals):.6f}",
                "mean_IDF1": f"{sum(i_vals) / len(i_vals):.6f}",
                "min_HOTA": f"{min(h_vals):.6f}",
                "max_HOTA": f"{max(h_vals):.6f}",
                "min_IDF1": f"{min(i_vals):.6f}",
                "max_IDF1": f"{max(i_vals):.6f}",
                "carriers": ",".join(sorted(str(item["carrier"]) for item in items)),
            }
        )
    rows.sort(key=lambda row: (float(row["mean_HOTA"]), float(row["mean_IDF1"]), row["seq"]))
    return rows


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
        "scripts/run_cross_host_baseline_audit.py",
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--tracker-family",
        "cross_host_baselines",
        "--variant",
        f"{args.dataset.lower()}_{args.detector_ext.lower()}_{args.split}",
        "--tag",
        out_dir.name,
        "--run-root",
        str(out_dir.resolve()),
        "--summary-csv",
        str(summary_csv.resolve()),
        "--notes",
        "Cross-host baseline audit on shared MOT17 half-val protocol",
        "--extra",
        f"detector_ext={args.detector_ext}",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    carriers = parse_carrier_specs(args.carrier)
    summary_csv = out_dir / "summary.csv"
    result_csv = out_dir / "result.csv"
    per_seq_csv = out_dir / "per_seq.csv"
    common_seq_csv = out_dir / "common_hard_sequences.csv"

    summary_rows: list[dict[str, Any]] = []
    per_seq_rows: list[dict[str, Any]] = []
    for carrier, spec in carriers.items():
        src_dir = Path(spec["path"]).resolve()
        remap_from_fullval = bool(spec.get("remap_from_fullval", False))
        summary_rows.append(
            {
                "carrier": carrier,
                "dataset": args.dataset,
                "split": args.split,
                "detector_ext": args.detector_ext,
                "source_results_dir": str(src_dir.resolve()),
                "subset_results_dir": str((out_dir / "subset_results" / carrier).resolve()),
                "remap_from_fullval": int(remap_from_fullval),
                "eval_dir": "",
                "tracker_name": f"{carrier}_{args.dataset.lower()}_{args.detector_ext.lower()}_{args.split}",
                "HOTA": "",
                "DetA": "",
                "AssA": "",
                "IDF1": "",
                "MOTA": "",
                "IDSW": "",
                "status": "running",
                "error": "",
            }
        )
    write_rows(summary_csv, SUMMARY_FIELDS, summary_rows)
    write_rows(result_csv, SUMMARY_FIELDS, summary_rows)
    write_rows(per_seq_csv, PER_SEQ_FIELDS, per_seq_rows)
    write_rows(common_seq_csv, COMMON_SEQ_FIELDS, build_common_seq_rows(per_seq_rows))

    for idx, (carrier, spec) in enumerate(carriers.items()):
        src_dir = Path(spec["path"]).resolve()
        remap_from_fullval = bool(spec.get("remap_from_fullval", False))
        summary_row, carrier_per_seq = run_eval_for_carrier(
            args,
            carrier=carrier,
            src_dir=src_dir,
            remap_from_fullval=remap_from_fullval,
            out_dir=out_dir,
        )
        summary_rows[idx] = summary_row
        per_seq_rows.extend(carrier_per_seq)
        write_rows(summary_csv, SUMMARY_FIELDS, summary_rows)
        write_rows(result_csv, SUMMARY_FIELDS, summary_rows)
        write_rows(per_seq_csv, PER_SEQ_FIELDS, per_seq_rows)
        write_rows(common_seq_csv, COMMON_SEQ_FIELDS, build_common_seq_rows(per_seq_rows))

    status = "success" if all(str(row.get("status")) == "success" for row in summary_rows) else "failed"
    append_registry(args, out_dir=out_dir, summary_csv=summary_csv, status=status)


if __name__ == "__main__":
    main()
