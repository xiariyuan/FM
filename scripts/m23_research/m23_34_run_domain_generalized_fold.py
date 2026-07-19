#!/usr/bin/env python3
from __future__ import annotations

"""Run one strict M23-34 domain-generalized ReID outer fold end to end."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
FIELDS = (
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
    "phase_root",
    "graph_root",
    "result_root",
    "message",
)


def run(command: List[str]) -> None:
    print("[M23-34 fold] run:", " ".join(command), flush=True)
    completed = subprocess.run(command)
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {command}"
        )


def write_status(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})


def read_metrics(path: Path) -> Dict[str, object]:
    with path.open(newline="", encoding="utf-8") as handle:
        row = dict(next(csv.DictReader(handle)))
    for key in ("HOTA", "DetA", "AssA", "delta_HOTA", "delta_AssA"):
        if row.get(key):
            row[key] = float(row[key])
    for key in ("IDSW", "selected_actions"):
        if row.get(key):
            row[key] = int(float(row[key]))
    return row


def phase_complete(root: Path) -> bool:
    return all(
        (root / seq / "dump_yolox_reid.npz").is_file() for seq in SEQUENCES
    )


def graph_complete(root: Path) -> bool:
    names = ("candidate_edges.parquet", "microtracklets.parquet", "prototypes.f16.npy")
    return all((root / seq / name).is_file() for seq in SEQUENCES for name in names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-seq", default="MOT20-02", choices=SEQUENCES)
    parser.add_argument("--base-root", default="outputs/mot20_m23_20260718")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=220)
    args = parser.parse_args()

    base = Path(args.base_root)
    held = args.held_seq
    queue_root = base / "m23_34_domain_generalized_fold_v1" / held
    status_path = queue_root / "summary.csv"
    train_dir = base / "m23_34_dg_reid_loso_full" / held
    phase_root = base / "m23_34_phase_loso" / held
    graph_root = base / "m23_34_micrograph_loso" / held
    result_root = base / "m23_34_sequence_calibrated_graph_loso" / held
    row: Dict[str, object] = {
        "sequence": held,
        "status": "running",
        "stage": "train",
        "train_dir": str(train_dir),
        "phase_root": str(phase_root),
        "graph_root": str(graph_root),
        "result_root": str(result_root),
        "message": (
            "strict outer sequence LOSO; held GT unavailable until frozen TrackEval"
        ),
    }
    write_status(status_path, row)

    try:
        if not (train_dir / "summary.json").is_file():
            run(
                [
                    sys.executable,
                    "-u",
                    "scripts/m23_research/m23_34_domain_generalized_fastreid_loso.py",
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
                    "6000",
                    "--num-workers",
                    "8",
                    "--freeze-backbone-epochs",
                    "1",
                    "--early-stop-patience",
                    "3",
                    "--sequence-temperature",
                    "0.5",
                    "--domain-adversarial-weight",
                    "0.15",
                    "--domain-warmup-epochs",
                    "1",
                    "--amp",
                ]
            )

        row["stage"] = "reembed"
        write_status(status_path, row)
        if not phase_complete(phase_root):
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

        row["stage"] = "micrograph"
        write_status(status_path, row)
        if not graph_complete(graph_root):
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

        row["stage"] = "transaction_graph"
        write_status(status_path, row)
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
            raise RuntimeError("M23-34 downstream metrics did not finish successfully")
        row.update(metrics)
        row["status"] = "success"
        row["stage"] = "completed"
        row["message"] = "strict M23-34 outer fold completed"
        write_status(status_path, row)
        report = {
            "status": "completed",
            "experiment": "M23-34 domain-generalized ReID plus frozen M23-25 graph policy",
            "protocol": "strict outer sequence LOSO",
            "outer_held_sequence": held,
            "outer_held_gt_use": "final TrackEval only",
            "candidate_inference_gt_use": "none",
            "result": row,
        }
        (queue_root / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    except Exception as exc:
        row["status"] = "failed"
        row["message"] = f"{type(exc).__name__}: {exc}"
        write_status(status_path, row)
        raise


if __name__ == "__main__":
    main()
