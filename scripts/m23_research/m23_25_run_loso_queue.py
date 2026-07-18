#!/usr/bin/env python3
from __future__ import annotations

"""Run the remaining strict M23-25 FastReID and transaction outer folds."""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
SUMMARY_FIELDS = [
    "sequence",
    "status",
    "stage",
    "HOTA",
    "DetA",
    "AssA",
    "IDSW",
    "delta_HOTA",
    "selected_actions",
    "train_dir",
    "graph_root",
    "result_root",
    "message",
]


def run(command: List[str]) -> None:
    print("[M23-25 queue] run:", " ".join(command), flush=True)
    completed = subprocess.run(command)
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {command}"
        )


def write_summary(path: Path, rows: Dict[str, Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for seq in SEQUENCES:
            if seq in rows:
                writer.writerow(
                    {key: rows[seq].get(key, "") for key in SUMMARY_FIELDS}
                )


def read_metrics(path: Path) -> Dict[str, object]:
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    output: Dict[str, object] = dict(row)
    for key in [
        "HOTA",
        "DetA",
        "AssA",
        "delta_HOTA",
        "delta_AssA",
        "delta_IDSW",
    ]:
        if row.get(key):
            output[key] = float(row[key])
    for key in ["IDSW", "selected_actions"]:
        if row.get(key):
            output[key] = int(float(row[key]))
    return output


def all_phase_outputs_exist(root: Path) -> bool:
    return all(
        (root / seq / "dump_yolox_reid.npz").is_file() for seq in SEQUENCES
    )


def all_graph_outputs_exist(root: Path) -> bool:
    required = ("candidate_edges.parquet", "microtracklets.parquet", "prototypes.f16.npy")
    return all((root / seq / name).is_file() for seq in SEQUENCES for name in required)


def evaluate_combined(track_results: Path, output_root: Path) -> Dict[str, float]:
    tracker_name = "m23_25_strict_sequence_loso_combined"
    work_dir = output_root / "eval_work"
    command = [
        sys.executable,
        "scripts/eval_motstyle_trackeval.py",
        "--benchmark-name",
        "MOT20",
        "--split-to-eval",
        "train",
        "--gt-root",
        "datasets/MOT20/train",
        "--results-dir",
        str(track_results),
        "--tracker-name",
        tracker_name,
        "--work-dir",
        str(work_dir),
        "--seqs",
        *SEQUENCES,
    ]
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    (output_root / "eval.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stdout[-5000:])
    detailed = work_dir / "eval" / tracker_name / "pedestrian_detailed.csv"
    rows = list(csv.DictReader(detailed.open(encoding="utf-8")))
    row = next(item for item in rows if item["seq"] == "COMBINED")
    return {
        "HOTA": 100.0 * float(row["HOTA___AUC"]),
        "DetA": 100.0 * float(row["DetA___AUC"]),
        "AssA": 100.0 * float(row["AssA___AUC"]),
        "IDSW": int(float(row["IDSW"])),
    }


def write_combined_metrics(path: Path, metrics: Dict[str, float]) -> None:
    fields = ["status", "HOTA", "DetA", "AssA", "IDSW", "target", "promoted"]
    row = {
        "status": "success",
        **metrics,
        "target": 80.0,
        "promoted": int(metrics["HOTA"] > 80.0),
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequences", default="MOT20-01,MOT20-03,MOT20-05"
    )
    parser.add_argument(
        "--base-root", default="outputs/mot20_m23_20260718"
    )
    parser.add_argument(
        "--m02-result-root",
        default=(
            "outputs/mot20_m23_20260718/"
            "m23_25_sequence_calibrated_graph_m02_full_v1"
        ),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=220)
    args = parser.parse_args()

    requested = [item.strip() for item in args.sequences.split(",") if item.strip()]
    invalid = sorted(set(requested) - set(SEQUENCES))
    if invalid:
        raise ValueError(f"invalid sequences: {invalid}")
    base = Path(args.base_root)
    queue_root = base / "m23_25_loso_queue_v1"
    summary_path = queue_root / "summary.csv"
    rows: Dict[str, Dict[str, object]] = {}

    m02_root = Path(args.m02_result_root)
    m02_metrics = read_metrics(m02_root / "metrics.csv")
    if m02_metrics.get("status") != "success":
        raise RuntimeError("MOT20-02 prerequisite result is not successful")
    rows["MOT20-02"] = {
        "sequence": "MOT20-02",
        "status": "success",
        "stage": "completed",
        **m02_metrics,
        "result_root": str(m02_root),
        "message": "reused completed strict outer fold",
    }
    write_summary(summary_path, rows)

    for held in requested:
        if held == "MOT20-02":
            continue
        train_dir = base / "m23_24_fastreid_loso_full" / held
        phase_root = base / "m23_24_phase_loso" / held
        graph_root = base / "m23_24_micrograph_loso" / held
        result_root = base / "m23_25_sequence_calibrated_graph_loso" / held
        rows[held] = {
            "sequence": held,
            "status": "running",
            "stage": "train",
            "train_dir": str(train_dir),
            "graph_root": str(graph_root),
            "result_root": str(result_root),
            "message": "",
        }
        write_summary(summary_path, rows)
        try:
            if not (train_dir / "summary.json").is_file():
                run(
                    [
                        sys.executable,
                        "-u",
                        "scripts/m23_research/m23_24_train_fastreid_loso.py",
                        "--held-seq",
                        held,
                        "--output-dir",
                        str(train_dir),
                        "--epochs",
                        str(args.epochs),
                        "--batch-size",
                        "64",
                        "--num-instances",
                        "4",
                        "--frame-stride",
                        "3",
                        "--max-samples-per-id",
                        "60",
                        "--val-max-items",
                        "4000",
                        "--num-workers",
                        "8",
                        "--freeze-backbone-epochs",
                        "1",
                        "--early-stop-patience",
                        "3",
                        "--amp",
                    ]
                )
            rows[held]["stage"] = "reembed"
            write_summary(summary_path, rows)
            if not all_phase_outputs_exist(phase_root):
                run(
                    [
                        sys.executable,
                        "-u",
                        "scripts/m23_research/m23_24_reembed_phase0.py",
                        "--checkpoint",
                        str(train_dir / "model_best.pth"),
                        "--held-seq",
                        held,
                        "--output-root",
                        str(phase_root),
                        "--batch-size",
                        "96",
                    ]
                )
            rows[held]["stage"] = "micrograph"
            write_summary(summary_path, rows)
            if not all_graph_outputs_exist(graph_root):
                run(
                    [
                        sys.executable,
                        "-u",
                        "scripts/m23_research/m23_24_build_finetuned_micrograph.py",
                        "--held-seq",
                        held,
                        "--phase-root",
                        str(phase_root),
                        "--out-root",
                        str(graph_root),
                    ]
                )
            rows[held]["stage"] = "transaction_graph"
            write_summary(summary_path, rows)
            metrics_path = result_root / "metrics.csv"
            metrics = read_metrics(metrics_path) if metrics_path.is_file() else {}
            if metrics.get("status") != "success":
                run(
                    [
                        sys.executable,
                        "-u",
                        "scripts/m23_research/m23_25_sequence_calibrated_transaction_graph.py",
                        "--held-seq",
                        held,
                        "--graph-root",
                        str(graph_root),
                        "--output-root",
                        str(result_root),
                        "--max-iter",
                        str(args.max_iter),
                    ]
                )
                metrics = read_metrics(metrics_path)
            if metrics.get("status") != "success":
                raise RuntimeError(f"{held}: M23-25 metrics did not finish successfully")
            rows[held].update(metrics)
            rows[held]["status"] = "success"
            rows[held]["stage"] = "completed"
            rows[held]["message"] = "strict outer fold completed"
            write_summary(summary_path, rows)
        except Exception as exc:
            rows[held]["status"] = "failed"
            rows[held]["message"] = f"{type(exc).__name__}: {exc}"
            write_summary(summary_path, rows)
            raise

    if set(rows) == set(SEQUENCES) and all(
        rows[seq].get("status") == "success" for seq in SEQUENCES
    ):
        combined_root = base / "m23_25_sequence_calibrated_graph_loso" / "combined_oof"
        track_results = combined_root / "track_results"
        track_results.mkdir(parents=True, exist_ok=True)
        sources = {
            "MOT20-02": m02_root,
            **{
                seq: base / "m23_25_sequence_calibrated_graph_loso" / seq
                for seq in SEQUENCES
                if seq != "MOT20-02"
            },
        }
        for seq, root in sources.items():
            shutil.copy2(root / "track_results" / f"{seq}.txt", track_results / f"{seq}.txt")
        combined = evaluate_combined(track_results, combined_root)
        write_combined_metrics(combined_root / "metrics.csv", combined)
        report = {
            "status": "completed",
            "protocol": "four strict outer-held trackers concatenated before one TrackEval",
            "outer_gt_use": "final TrackEval only",
            "folds": rows,
            "combined": combined,
            "target": 80.0,
            "promoted": bool(combined["HOTA"] > 80.0),
        }
        (combined_root / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
