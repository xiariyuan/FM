#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
BOTSORT_ROOT = REPO_ROOT / "external" / "BoT-SORT-main"
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

QUEUE_FIELDS = [
    "step",
    "name",
    "status",
    "out_dir",
    "summary_csv",
    "log_path",
    "started_at",
    "finished_at",
    "notes",
]

ONLINE_COMPARE_FIELDS = [
    "name",
    "HOTA",
    "DetA",
    "AssA",
    "IDF1",
    "MOTA",
    "IDSW",
    "Frag",
    "run_dir",
]

ONLINE_DELTA_FIELDS = [
    "name",
    "delta_HOTA",
    "delta_DetA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDSW",
    "delta_Frag",
]

PER_SEQUENCE_FIELDS = [
    "name",
    "seq",
    "HOTA",
    "DetA",
    "AssA",
    "IDF1",
    "MOTA",
    "IDSW",
    "Frag",
]

OFFLINE_COMPARE_FIELDS = [
    "name",
    "mode",
    "groups",
    "ambiguous_groups",
    "auc",
    "ambiguous_auc",
    "top1",
    "ambiguous_top1",
    "wrong_to_right_rate",
    "right_to_wrong_rate",
    "summary_csv",
]

OFFLINE_DELTA_FIELDS = [
    "name",
    "delta_auc",
    "delta_ambiguous_auc",
    "delta_top1",
    "delta_ambiguous_top1",
    "delta_wrong_to_right_rate",
    "delta_right_to_wrong_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the strict FCAA shared-checkpoint half-split protocol.")
    parser.add_argument("--dataset-root", default="/gemini/code/datasets")
    parser.add_argument("--benchmark", default="MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seq-names", nargs="*", default=["MOT17-05-FRCNN", "MOT17-10-FRCNN", "MOT17-13-FRCNN"])
    parser.add_argument("--pairbank-device", default="cuda")
    parser.add_argument("--track-device", default="gpu")
    parser.add_argument("--optimizer", choices=["adamw", "lbfgs"], default="lbfgs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-weight", type=float, default=3.0)
    parser.add_argument("--ambiguous-oversample", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--fcaa-trigger-mode", choices=["row_margin", "shared_det_top1", "shared_det_top1_margin"], default="shared_det_top1")
    parser.add_argument("--trigger-margin", type=float, default=0.05)
    parser.add_argument("--fcaa-lambda", type=float, default=0.3)
    parser.add_argument("--fcaa-topk", type=int, default=3)
    parser.add_argument("--out-root", default="")
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp_tag() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def update_row(rows: List[Dict[str, object]], step: str, **updates: object) -> None:
    for row in rows:
        if str(row["step"]) == step:
            row.update(updates)
            return
    raise KeyError(f"Missing queue step: {step}")


def run_step(cmd: List[str], log_path: Path, *, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[started_at] {now_iso()}\n")
        handle.write(f"[cwd] {cwd}\n")
        handle.write("[cmd] " + " ".join(cmd) + "\n\n")
        handle.flush()
        process = subprocess.run(cmd, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[finished_at] {now_iso()}\n")
        handle.write(f"[return_code] {process.returncode}\n")
    return int(process.returncode)


def append_registry(summary_csv: Path, run_root: Path, status: str, notes: str, registry_csv: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(registry_csv),
        "--kind",
        "other",
        "--status",
        status,
        "--script",
        "scripts/run_fcaa_shared_ckpt_halfsplit.py",
        "--dataset",
        "MOT17",
        "--split",
        "train_half_val_half",
        "--tracker-family",
        "botsort_fcaa",
        "--variant",
        run_root.name,
        "--tag",
        "fcaa_shared_ckpt_halfsplit",
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def ensure_success(summary_csv: Path) -> None:
    rows = read_rows(summary_csv)
    if not rows:
        raise FileNotFoundError(f"Missing summary rows: {summary_csv}")
    statuses = {str(row.get("status", "")).strip() for row in rows}
    if statuses != {"success"}:
        raise RuntimeError(f"Unexpected status in {summary_csv}: {sorted(statuses)}")


def parse_summary_txt(path: Path) -> Dict[str, float]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Invalid TrackEval summary: {path}")
    headers = lines[0].split()
    values = lines[1].split()
    if len(headers) != len(values):
        raise ValueError(f"Header/value mismatch in {path}")
    parsed = {key: float(value) for key, value in zip(headers, values)}
    return parsed


def load_online_metrics(eval_dir: Path, tracker_name: str) -> Dict[str, float | str]:
    summary_txt = eval_dir / "eval" / tracker_name / "pedestrian_summary.txt"
    metrics = parse_summary_txt(summary_txt)
    return {
        "name": tracker_name,
        "HOTA": float(metrics["HOTA"]),
        "DetA": float(metrics["DetA"]),
        "AssA": float(metrics["AssA"]),
        "IDF1": float(metrics["IDF1"]),
        "MOTA": float(metrics["MOTA"]),
        "IDSW": int(round(float(metrics["IDSW"]))),
        "Frag": int(round(float(metrics["Frag"]))),
        "run_dir": str(eval_dir),
    }


def load_per_sequence_metrics(eval_dir: Path, tracker_name: str) -> List[Dict[str, float | str | int]]:
    detailed_csv = eval_dir / "eval" / tracker_name / "pedestrian_detailed.csv"
    rows: List[Dict[str, float | str | int]] = []
    with detailed_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seq = str(row.get("seq", ""))
            if not seq or seq == "COMBINED":
                continue
            rows.append(
                {
                    "name": tracker_name,
                    "seq": seq,
                    "HOTA": float(row["HOTA___AUC"]) * 100.0,
                    "DetA": float(row["DetA___AUC"]) * 100.0,
                    "AssA": float(row["AssA___AUC"]) * 100.0,
                    "IDF1": float(row["IDF1"]) * 100.0,
                    "MOTA": float(row["MOTA"]) * 100.0,
                    "IDSW": int(round(float(row["IDSW"]))),
                    "Frag": int(round(float(row["Frag"]))),
                }
            )
    return rows


def load_single_summary(summary_csv: Path) -> Dict[str, str]:
    rows = read_rows(summary_csv)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {summary_csv}, got {len(rows)}")
    return rows[0]


def seq_ids_from_names(seq_names: List[str]) -> List[str]:
    seq_ids: List[str] = []
    for seq in seq_names:
        core = seq.split("-")
        if len(core) < 2:
            raise ValueError(f"Unexpected MOT sequence name: {seq}")
        seq_ids.append(str(int(core[1])))
    return seq_ids


def mode_experiment_name(queue_name: str, mode: str) -> str:
    return f"{queue_name}_{mode}"


def seq_note(seq_names: List[str]) -> str:
    return "|".join(seq_names)


def build_track_cmd(
    *,
    dataset_root: str,
    benchmark: str,
    split: str,
    seq_names: List[str],
    experiment_name: str,
    track_device: str,
    analysis_dir: Path,
    checkpoint: str = "",
    fcaa_trigger_mode: str,
    trigger_margin: float,
    fcaa_lambda: float,
    fcaa_topk: int,
) -> List[str]:
    cmd = [
        sys.executable,
        "tools/track.py",
        str(Path(dataset_root) / benchmark),
        "--benchmark",
        benchmark,
        "--eval",
        split,
        "--seq-ids",
        *seq_ids_from_names(seq_names),
        "--mot17-detector-exts",
        "FRCNN",
        "--default-parameters",
        "--with-reid",
        "--device",
        track_device,
        "--experiment-name",
        experiment_name,
        "--fcaa-analysis-dir",
        str(analysis_dir),
    ]
    if checkpoint:
        cmd.extend(
            [
                "--fcaa-enable",
                "--fcaa-scorer-checkpoint",
                str(checkpoint),
                "--fcaa-trigger-mode",
                str(fcaa_trigger_mode),
                "--fcaa-trigger-margin",
                str(trigger_margin),
                "--fcaa-lambda",
                str(fcaa_lambda),
                "--fcaa-topk",
                str(fcaa_topk),
            ]
        )
    return cmd


def build_halfval_eval_cmd(*, benchmark: str, dataset_root: str, results_dir: Path, tracker_name: str, work_dir: Path) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_botsort_halfval_trackeval.py"),
        "--dataset",
        benchmark,
        "--data-root",
        dataset_root,
        "--results-dir",
        str(results_dir),
        "--tracker-name",
        tracker_name,
        "--work-dir",
        str(work_dir),
        "--remap-results-from-fullval",
    ]


def finalize_compare_files(
    *,
    out_root: Path,
    offline_control_summary: Path,
    offline_freq_summary: Path,
    online_rows: List[Dict[str, float | str]],
    per_sequence_rows: List[Dict[str, float | str | int]],
) -> None:
    control_row = load_single_summary(offline_control_summary)
    freq_row = load_single_summary(offline_freq_summary)
    offline_rows = [
        {
            "name": "control",
            "mode": control_row.get("mode", "control"),
            "groups": control_row.get("groups", ""),
            "ambiguous_groups": control_row.get("ambiguous_groups", ""),
            "auc": control_row.get("auc", ""),
            "ambiguous_auc": control_row.get("ambiguous_auc", ""),
            "top1": control_row.get("top1", ""),
            "ambiguous_top1": control_row.get("ambiguous_top1", ""),
            "wrong_to_right_rate": control_row.get("wrong_to_right_rate", ""),
            "right_to_wrong_rate": control_row.get("right_to_wrong_rate", ""),
            "summary_csv": str(offline_control_summary),
        },
        {
            "name": "freq",
            "mode": freq_row.get("mode", "freq"),
            "groups": freq_row.get("groups", ""),
            "ambiguous_groups": freq_row.get("ambiguous_groups", ""),
            "auc": freq_row.get("auc", ""),
            "ambiguous_auc": freq_row.get("ambiguous_auc", ""),
            "top1": freq_row.get("top1", ""),
            "ambiguous_top1": freq_row.get("ambiguous_top1", ""),
            "wrong_to_right_rate": freq_row.get("wrong_to_right_rate", ""),
            "right_to_wrong_rate": freq_row.get("right_to_wrong_rate", ""),
            "summary_csv": str(offline_freq_summary),
        },
    ]
    write_rows(out_root / "offline_compare.csv", OFFLINE_COMPARE_FIELDS, offline_rows)
    offline_delta = [
        {
            "name": "freq_minus_control",
            "delta_auc": float(freq_row.get("auc", 0.0)) - float(control_row.get("auc", 0.0)),
            "delta_ambiguous_auc": float(freq_row.get("ambiguous_auc", 0.0)) - float(control_row.get("ambiguous_auc", 0.0)),
            "delta_top1": float(freq_row.get("top1", 0.0)) - float(control_row.get("top1", 0.0)),
            "delta_ambiguous_top1": float(freq_row.get("ambiguous_top1", 0.0)) - float(control_row.get("ambiguous_top1", 0.0)),
            "delta_wrong_to_right_rate": float(freq_row.get("wrong_to_right_rate", 0.0)) - float(control_row.get("wrong_to_right_rate", 0.0)),
            "delta_right_to_wrong_rate": float(freq_row.get("right_to_wrong_rate", 0.0)) - float(control_row.get("right_to_wrong_rate", 0.0)),
        }
    ]
    write_rows(out_root / "offline_delta.csv", OFFLINE_DELTA_FIELDS, offline_delta)

    write_rows(out_root / "metrics_compare.csv", ONLINE_COMPARE_FIELDS, online_rows)
    raw_row = next((row for row in online_rows if str(row["name"]).endswith("_raw")), None)
    if raw_row is None:
        raise ValueError("Missing raw online metrics row.")
    delta_rows = []
    for row in online_rows:
        if row is raw_row:
            continue
        delta_rows.append(
            {
                "name": str(row["name"]),
                "delta_HOTA": float(row["HOTA"]) - float(raw_row["HOTA"]),
                "delta_DetA": float(row["DetA"]) - float(raw_row["DetA"]),
                "delta_AssA": float(row["AssA"]) - float(raw_row["AssA"]),
                "delta_IDF1": float(row["IDF1"]) - float(raw_row["IDF1"]),
                "delta_MOTA": float(row["MOTA"]) - float(raw_row["MOTA"]),
                "delta_IDSW": int(row["IDSW"]) - int(raw_row["IDSW"]),
                "delta_Frag": int(row["Frag"]) - int(raw_row["Frag"]),
            }
        )
    write_rows(out_root / "metrics_delta.csv", ONLINE_DELTA_FIELDS, delta_rows)
    write_rows(out_root / "per_sequence_metrics.csv", PER_SEQUENCE_FIELDS, per_sequence_rows)


def main() -> None:
    args = parse_args()
    queue_name = Path(args.out_root).name if args.out_root else f"fcaa_shared_ckpt_halfsplit_{timestamp_tag()}"
    out_root = Path(args.out_root) if args.out_root else REPO_ROOT / "outputs" / queue_name
    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "summary.csv"
    seqs_note = seq_note(list(args.seq_names))

    pairbank_train_dir = out_root / "pairbank_train_half"
    pairbank_val_dir = out_root / "pairbank_val_half"
    control_train_dir = out_root / "train_control"
    freq_train_dir = out_root / "train_freq"
    control_eval_dir = out_root / "offline_eval_control"
    freq_eval_dir = out_root / "offline_eval_freq"

    rows: List[Dict[str, object]] = [
        {
            "step": "pairbank_train_half",
            "name": f"{queue_name}_pairbank_train_half",
            "status": "pending",
            "out_dir": str(pairbank_train_dir),
            "summary_csv": str(pairbank_train_dir / "summary.csv"),
            "log_path": str(out_root / "logs" / "pairbank_train_half.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"subset=train_half grouping=shared_det_top1 seqs={seqs_note}",
        },
        {
            "step": "pairbank_val_half",
            "name": f"{queue_name}_pairbank_val_half",
            "status": "pending",
            "out_dir": str(pairbank_val_dir),
            "summary_csv": str(pairbank_val_dir / "summary.csv"),
            "log_path": str(out_root / "logs" / "pairbank_val_half.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"subset=val_half grouping=shared_det_top1 seqs={seqs_note}",
        },
        {
            "step": "train_control",
            "name": f"{queue_name}_control",
            "status": "pending",
            "out_dir": str(control_train_dir),
            "summary_csv": str(control_train_dir / "summary.csv"),
            "log_path": str(out_root / "logs" / "train_control.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"mode=control optimizer={args.optimizer} trigger_mode={args.fcaa_trigger_mode}",
        },
        {
            "step": "train_freq",
            "name": f"{queue_name}_freq",
            "status": "pending",
            "out_dir": str(freq_train_dir),
            "summary_csv": str(freq_train_dir / "summary.csv"),
            "log_path": str(out_root / "logs" / "train_freq.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"mode=freq optimizer={args.optimizer} trigger_mode={args.fcaa_trigger_mode}",
        },
        {
            "step": "offline_eval_control",
            "name": f"{queue_name}_offline_eval_control",
            "status": "pending",
            "out_dir": str(control_eval_dir),
            "summary_csv": str(control_eval_dir / "summary.csv"),
            "log_path": str(out_root / "logs" / "offline_eval_control.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"control checkpoint on val_half pair-bank trigger_mode={args.fcaa_trigger_mode}",
        },
        {
            "step": "offline_eval_freq",
            "name": f"{queue_name}_offline_eval_freq",
            "status": "pending",
            "out_dir": str(freq_eval_dir),
            "summary_csv": str(freq_eval_dir / "summary.csv"),
            "log_path": str(out_root / "logs" / "offline_eval_freq.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"freq checkpoint on val_half pair-bank trigger_mode={args.fcaa_trigger_mode}",
        },
        {
            "step": "online_raw_eval",
            "name": mode_experiment_name(queue_name, "raw"),
            "status": "pending",
            "out_dir": str(BOTSORT_ROOT / "YOLOX_outputs" / mode_experiment_name(queue_name, "raw")),
            "summary_csv": str(out_root / "metrics_compare.csv"),
            "log_path": str(out_root / "logs" / "online_raw_eval.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"shared raw BoT-SORT over {seqs_note} with half-val TrackEval trigger_mode={args.fcaa_trigger_mode}",
        },
        {
            "step": "online_control_eval",
            "name": mode_experiment_name(queue_name, "control"),
            "status": "pending",
            "out_dir": str(BOTSORT_ROOT / "YOLOX_outputs" / mode_experiment_name(queue_name, "control")),
            "summary_csv": str(out_root / "metrics_compare.csv"),
            "log_path": str(out_root / "logs" / "online_control_eval.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"shared control checkpoint over {seqs_note} with half-val TrackEval trigger_mode={args.fcaa_trigger_mode}",
        },
        {
            "step": "online_freq_eval",
            "name": mode_experiment_name(queue_name, "freq"),
            "status": "pending",
            "out_dir": str(BOTSORT_ROOT / "YOLOX_outputs" / mode_experiment_name(queue_name, "freq")),
            "summary_csv": str(out_root / "metrics_compare.csv"),
            "log_path": str(out_root / "logs" / "online_freq_eval.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"shared freq checkpoint over {seqs_note} with half-val TrackEval trigger_mode={args.fcaa_trigger_mode}",
        },
        {
            "step": "compare",
            "name": f"{queue_name}_compare",
            "status": "pending",
            "out_dir": str(out_root),
            "summary_csv": str(out_root / "metrics_compare.csv"),
            "log_path": str(out_root / "logs" / "compare.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"write offline/online comparison CSVs trigger_mode={args.fcaa_trigger_mode}",
        },
    ]
    write_rows(summary_csv, QUEUE_FIELDS, rows)

    try:
        update_row(rows, "pairbank_train_half", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        train_pairbank_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_fcaa_pairbank.py"),
            "--dataset-root",
            args.dataset_root,
            "--benchmark",
            args.benchmark,
            "--split",
            args.split,
            "--subset",
            "train_half",
            "--grouping",
            "shared_det_top1",
            "--seq-names",
            *args.seq_names,
            "--out-dir",
            str(pairbank_train_dir),
            "--dataset-name",
            f"{queue_name}_train_half_pairbank",
            "--top-k",
            str(args.top_k),
            "--device",
            args.pairbank_device,
        ]
        rc = run_step(train_pairbank_cmd, Path(rows[0]["log_path"]), cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"pairbank_train_half failed with exit code {rc}")
        ensure_success(pairbank_train_dir / "summary.csv")
        update_row(rows, "pairbank_train_half", status="success", finished_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        update_row(rows, "pairbank_val_half", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        val_pairbank_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_fcaa_pairbank.py"),
            "--dataset-root",
            args.dataset_root,
            "--benchmark",
            args.benchmark,
            "--split",
            args.split,
            "--subset",
            "val_half",
            "--grouping",
            "shared_det_top1",
            "--seq-names",
            *args.seq_names,
            "--out-dir",
            str(pairbank_val_dir),
            "--dataset-name",
            f"{queue_name}_val_half_pairbank",
            "--top-k",
            str(args.top_k),
            "--device",
            args.pairbank_device,
        ]
        rc = run_step(val_pairbank_cmd, Path(rows[1]["log_path"]), cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"pairbank_val_half failed with exit code {rc}")
        ensure_success(pairbank_val_dir / "summary.csv")
        update_row(rows, "pairbank_val_half", status="success", finished_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        pairbank_train_jsonl = pairbank_train_dir / "pairbank.jsonl"
        pairbank_val_jsonl = pairbank_val_dir / "pairbank.jsonl"

        update_row(rows, "train_control", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        control_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_fcaa_pair_scorer.py"),
            "--pairbank-jsonl",
            str(pairbank_train_jsonl),
            "--val-pairbank-jsonl",
            str(pairbank_val_jsonl),
            "--out-dir",
            str(control_train_dir),
            "--run-name",
            f"{queue_name}_control",
            "--mode",
            "control",
            "--optimizer",
            args.optimizer,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--positive-weight",
            str(args.positive_weight),
            "--ambiguous-oversample",
            str(args.ambiguous_oversample),
            "--seed",
            str(args.seed),
        ]
        rc = run_step(control_cmd, Path(rows[2]["log_path"]), cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"train_control failed with exit code {rc}")
        ensure_success(control_train_dir / "summary.csv")
        update_row(rows, "train_control", status="success", finished_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        update_row(rows, "train_freq", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        freq_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_fcaa_pair_scorer.py"),
            "--pairbank-jsonl",
            str(pairbank_train_jsonl),
            "--val-pairbank-jsonl",
            str(pairbank_val_jsonl),
            "--out-dir",
            str(freq_train_dir),
            "--run-name",
            f"{queue_name}_freq",
            "--mode",
            "freq",
            "--optimizer",
            args.optimizer,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--positive-weight",
            str(args.positive_weight),
            "--ambiguous-oversample",
            str(args.ambiguous_oversample),
            "--seed",
            str(args.seed),
        ]
        rc = run_step(freq_cmd, Path(rows[3]["log_path"]), cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"train_freq failed with exit code {rc}")
        ensure_success(freq_train_dir / "summary.csv")
        update_row(rows, "train_freq", status="success", finished_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        control_ckpt = control_train_dir / "best.pt"
        freq_ckpt = freq_train_dir / "best.pt"

        update_row(rows, "offline_eval_control", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        control_eval_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "eval_fcaa_pairbank.py"),
            "--pairbank-jsonl",
            str(pairbank_val_jsonl),
            "--checkpoint",
            str(control_ckpt),
            "--out-dir",
            str(control_eval_dir),
            "--name",
            f"{queue_name}_offline_eval_control",
        ]
        rc = run_step(control_eval_cmd, Path(rows[4]["log_path"]), cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"offline_eval_control failed with exit code {rc}")
        ensure_success(control_eval_dir / "summary.csv")
        update_row(rows, "offline_eval_control", status="success", finished_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        update_row(rows, "offline_eval_freq", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        freq_eval_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "eval_fcaa_pairbank.py"),
            "--pairbank-jsonl",
            str(pairbank_val_jsonl),
            "--checkpoint",
            str(freq_ckpt),
            "--out-dir",
            str(freq_eval_dir),
            "--name",
            f"{queue_name}_offline_eval_freq",
        ]
        rc = run_step(freq_eval_cmd, Path(rows[5]["log_path"]), cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"offline_eval_freq failed with exit code {rc}")
        ensure_success(freq_eval_dir / "summary.csv")
        update_row(rows, "offline_eval_freq", status="success", finished_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        online_rows: List[Dict[str, float | str]] = []
        per_sequence_rows: List[Dict[str, float | str | int]] = []
        for step_name, mode, checkpoint in [
            ("online_raw_eval", "raw", ""),
            ("online_control_eval", "control", str(control_ckpt)),
            ("online_freq_eval", "freq", str(freq_ckpt)),
        ]:
            exp_name = mode_experiment_name(queue_name, mode)
            log_path = Path(next(row["log_path"] for row in rows if row["step"] == step_name))
            update_row(rows, step_name, status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)

            analysis_dir = out_root / f"analysis_{mode}"
            track_cmd = build_track_cmd(
                dataset_root=args.dataset_root,
                benchmark=args.benchmark,
                split=args.split,
                seq_names=list(args.seq_names),
                experiment_name=exp_name,
                track_device=args.track_device,
                analysis_dir=analysis_dir,
                checkpoint=checkpoint,
                fcaa_trigger_mode=str(args.fcaa_trigger_mode),
                trigger_margin=float(args.trigger_margin),
                fcaa_lambda=float(args.fcaa_lambda),
                fcaa_topk=int(args.fcaa_topk),
            )
            rc = run_step(track_cmd, log_path, cwd=BOTSORT_ROOT)
            if rc != 0:
                raise RuntimeError(f"{step_name} track failed with exit code {rc}")

            results_dir = BOTSORT_ROOT / "YOLOX_outputs" / exp_name / "track_results"
            work_dir = out_root / f"eval_{mode}"
            eval_cmd = build_halfval_eval_cmd(
                benchmark=args.benchmark,
                dataset_root=args.dataset_root,
                results_dir=results_dir,
                tracker_name=exp_name,
                work_dir=work_dir,
            )
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("\n[eval_cmd] " + " ".join(eval_cmd) + "\n\n")
                handle.flush()
                process = subprocess.run(eval_cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
            if process.returncode != 0:
                raise RuntimeError(f"{step_name} TrackEval failed with exit code {process.returncode}")

            online_rows.append(load_online_metrics(work_dir, exp_name))
            per_sequence_rows.extend(load_per_sequence_metrics(work_dir, exp_name))
            update_row(rows, step_name, status="success", finished_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)

        update_row(rows, "compare", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        finalize_compare_files(
            out_root=out_root,
            offline_control_summary=control_eval_dir / "summary.csv",
            offline_freq_summary=freq_eval_dir / "summary.csv",
            online_rows=online_rows,
            per_sequence_rows=per_sequence_rows,
        )
        compare_log = Path(next(row["log_path"] for row in rows if row["step"] == "compare"))
        compare_log.parent.mkdir(parents=True, exist_ok=True)
        compare_log.write_text(
            "\n".join(
                [
                    f"[finished_at] {now_iso()}",
                    f"offline_compare={out_root / 'offline_compare.csv'}",
                    f"offline_delta={out_root / 'offline_delta.csv'}",
                    f"metrics_compare={out_root / 'metrics_compare.csv'}",
                    f"metrics_delta={out_root / 'metrics_delta.csv'}",
                    f"per_sequence_metrics={out_root / 'per_sequence_metrics.csv'}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        update_row(rows, "compare", status="success", finished_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        append_registry(summary_csv, out_root, "success", "fcaa shared-checkpoint half-split queue finished", args.registry_csv)
    except Exception as exc:
        for row in rows:
            if str(row.get("status", "")) == "running":
                row["status"] = "failed"
                row["finished_at"] = now_iso()
                row["notes"] = f"{row.get('notes', '')} error={exc}".strip()
                break
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        append_registry(summary_csv, out_root, "failed", f"fcaa shared-checkpoint half-split queue failed: {exc}", args.registry_csv)
        raise


if __name__ == "__main__":
    main()
