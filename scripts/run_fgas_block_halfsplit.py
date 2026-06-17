#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
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
    "feature_mode",
    "input_dim",
    "best_epoch",
    "best_metric",
    "val_row_top1",
    "val_ambiguous_row_top1",
    "val_edge_bce",
    "val_row_ce",
    "summary_csv",
    "checkpoint",
]

OFFLINE_DELTA_FIELDS = [
    "name",
    "delta_val_row_top1",
    "delta_val_ambiguous_row_top1",
    "delta_val_edge_bce",
    "delta_val_row_ce",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FGAS block-level half-split protocol on BoT-SORT.")
    parser.add_argument("--dataset-root", default="/gemini/code/datasets")
    parser.add_argument("--benchmark", default="MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--seq-names",
        nargs="*",
        default=[
            "MOT17-02-FRCNN",
            "MOT17-04-FRCNN",
            "MOT17-05-FRCNN",
            "MOT17-09-FRCNN",
            "MOT17-10-FRCNN",
            "MOT17-11-FRCNN",
            "MOT17-13-FRCNN",
        ],
    )
    parser.add_argument(
        "--train-jsonl",
        default=str(REPO_ROOT / "outputs" / "fgas_blockbank_train_half_looseB_20260331_1" / "blockbank.jsonl"),
    )
    parser.add_argument(
        "--val-jsonl",
        default=str(REPO_ROOT / "outputs" / "fgas_blockbank_val_half_looseB_20260331_1" / "blockbank.jsonl"),
    )
    parser.add_argument("--out-root", default="")
    parser.add_argument("--track-profile", default="mot17_public_ctrl_base")
    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--track-device", default="gpu")
    parser.add_argument("--arch", choices=["v1", "v2_trackdet", "v3_trackquery"], default="v1")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--stage-embed-dim", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-attn-layers", type=int, default=2)
    parser.add_argument("--ambiguous-oversample", type=float, default=2.0)
    parser.add_argument("--col-bce-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--nofreq-checkpoint", default="")
    parser.add_argument("--full-checkpoint", default="")
    parser.add_argument("--track-high-thresh", type=float, default=0.5)
    parser.add_argument("--proximity-thresh", type=float, default=0.9)
    parser.add_argument("--appearance-thresh", type=float, default=0.25)
    parser.add_argument("--fgas-topk", type=int, default=5)
    parser.add_argument("--fgas-max-rows", type=int, default=3)
    parser.add_argument("--fgas-max-cols", type=int, default=3)
    parser.add_argument("--fgas-blend-weight", type=float, default=0.5)
    parser.add_argument("--fgas-assignment-mode", choices=["blend", "replace"], default="blend")
    parser.add_argument("--fgas-row-nomatch-weight", type=float, default=0.0)
    parser.add_argument("--fgas-controller-enable", action="store_true")
    parser.add_argument("--fgas-controller-edge-thresh", type=float, default=0.6)
    parser.add_argument("--fgas-controller-row-defer-thresh", type=float, default=0.6)
    parser.add_argument("--fgas-controller-col-newborn-thresh", type=float, default=0.6)
    parser.add_argument("--fgas-controller-margin-thresh", type=float, default=0.05)
    parser.add_argument("--fgas-controller-ambiguity-margin", type=float, default=0.05)
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
        "scripts/run_fgas_block_halfsplit.py",
        "--dataset",
        "MOT17",
        "--split",
        "train_half_val_half",
        "--tracker-family",
        "botsort_fgas",
        "--variant",
        run_root.name,
        "--tag",
        "fgas_block_halfsplit",
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


def materialize_skip_train_artifacts(checkpoint: Path, train_dir: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve(strict=False)
    source_dir = checkpoint.parent
    source_summary = source_dir / "summary.csv"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not source_summary.is_file():
        raise FileNotFoundError(f"Missing source summary for skipped train: {source_summary}")

    train_dir.mkdir(parents=True, exist_ok=True)
    target_summary = train_dir / "summary.csv"
    shutil.copy2(source_summary, target_summary)

    source_metrics = source_dir / "metrics.jsonl"
    if source_metrics.is_file():
        shutil.copy2(source_metrics, train_dir / "metrics.jsonl")

    return target_summary


def parse_summary_txt(path: Path) -> Dict[str, float]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Invalid TrackEval summary: {path}")
    headers = lines[0].split()
    values = lines[1].split()
    if len(headers) != len(values):
        raise ValueError(f"Header/value mismatch in {path}")
    return {key: float(value) for key, value in zip(headers, values)}


def load_online_metrics(eval_dir: Path, tracker_name: str, label: str) -> Dict[str, float | str]:
    summary_txt = eval_dir / "eval" / tracker_name / "pedestrian_summary.txt"
    metrics = parse_summary_txt(summary_txt)
    return {
        "name": label,
        "HOTA": float(metrics["HOTA"]),
        "DetA": float(metrics["DetA"]),
        "AssA": float(metrics["AssA"]),
        "IDF1": float(metrics["IDF1"]),
        "MOTA": float(metrics["MOTA"]),
        "IDSW": int(round(float(metrics["IDSW"]))),
        "Frag": int(round(float(metrics["Frag"]))),
        "run_dir": str(eval_dir),
    }


def load_per_sequence_metrics(eval_dir: Path, tracker_name: str, label: str) -> List[Dict[str, float | str | int]]:
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
                    "name": label,
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


def build_train_cmd(
    *,
    train_jsonl: str,
    val_jsonl: str,
    out_dir: Path,
    device: str,
    arch: str,
    feature_mode: str,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    stage_embed_dim: int,
    num_heads: int,
    num_attn_layers: int,
    ambiguous_oversample: float,
    col_bce_weight: float,
    lr: float,
    weight_decay: float,
    seed: int,
) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_fgas_block_resolver.py"),
        "--train-jsonl",
        str(train_jsonl),
        "--val-jsonl",
        str(val_jsonl),
        "--out-dir",
        str(out_dir),
        "--device",
        str(device),
        "--arch",
        str(arch),
        "--feature-mode",
        str(feature_mode),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--hidden-dim",
        str(hidden_dim),
        "--stage-embed-dim",
        str(stage_embed_dim),
        "--num-heads",
        str(num_heads),
        "--num-attn-layers",
        str(num_attn_layers),
        "--ambiguous-oversample",
        str(ambiguous_oversample),
        "--col-bce-weight",
        str(col_bce_weight),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--seed",
        str(seed),
    ]


def build_track_cmd(
    *,
    dataset_root: str,
    benchmark: str,
    split: str,
    seq_names: List[str],
    experiment_name: str,
    run_manifest: Path,
    track_profile: str,
    track_device: str,
    track_high_thresh: float,
    proximity_thresh: float,
    appearance_thresh: float,
    fgas_topk: int,
    fgas_max_rows: int,
    fgas_max_cols: int,
    fgas_blend_weight: float,
    fgas_assignment_mode: str,
    fgas_row_nomatch_weight: float,
    fgas_controller_enable: bool,
    fgas_controller_edge_thresh: float,
    fgas_controller_row_defer_thresh: float,
    fgas_controller_col_newborn_thresh: float,
    fgas_controller_margin_thresh: float,
    fgas_controller_ambiguity_margin: float,
    checkpoint: str = "",
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
        "--exp-profile",
        str(track_profile),
        "--experiment-name",
        experiment_name,
        "--run-manifest-path",
        str(run_manifest),
        "--device",
        track_device,
        "--with-reid",
        "--track_high_thresh",
        str(track_high_thresh),
        "--proximity_thresh",
        str(proximity_thresh),
        "--appearance_thresh",
        str(appearance_thresh),
    ]
    if checkpoint:
        cmd.extend(
            [
                "--fgas-enable",
                "--fgas-resolver-checkpoint",
                str(Path(checkpoint).expanduser().resolve(strict=False)),
                "--fgas-topk",
                str(fgas_topk),
                "--fgas-max-rows",
                str(fgas_max_rows),
                "--fgas-max-cols",
                str(fgas_max_cols),
                "--fgas-blend-weight",
                str(fgas_blend_weight),
                "--fgas-assignment-mode",
                str(fgas_assignment_mode),
                "--fgas-row-nomatch-weight",
                str(fgas_row_nomatch_weight),
            ]
        )
        if fgas_controller_enable:
            cmd.extend(
                [
                    "--fgas-controller-enable",
                    "--fgas-controller-edge-thresh",
                    str(fgas_controller_edge_thresh),
                    "--fgas-controller-row-defer-thresh",
                    str(fgas_controller_row_defer_thresh),
                    "--fgas-controller-col-newborn-thresh",
                    str(fgas_controller_col_newborn_thresh),
                    "--fgas-controller-margin-thresh",
                    str(fgas_controller_margin_thresh),
                    "--fgas-controller-ambiguity-margin",
                    str(fgas_controller_ambiguity_margin),
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


def collect_fgas_runtime_rows(experiment_dir: Path, label: str) -> List[Dict[str, object]]:
    summary_dir = experiment_dir / "fgas_analysis"
    if not summary_dir.is_dir():
        return []
    rows: List[Dict[str, object]] = []
    for path in sorted(summary_dir.glob("*_summary.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                payload = dict(row)
                payload["name"] = label
                payload["summary_csv"] = str(path)
                rows.append(payload)
    return rows


def finalize_compare_files(
    *,
    out_root: Path,
    nofreq_summary: Path,
    full_summary: Path,
    online_rows: List[Dict[str, float | str]],
    per_sequence_rows: List[Dict[str, float | str | int]],
    fgas_runtime_rows: List[Dict[str, object]],
) -> None:
    nofreq_row = load_single_summary(nofreq_summary)
    full_row = load_single_summary(full_summary)
    offline_rows = [
        {
            "name": "nofreq",
            "feature_mode": nofreq_row.get("feature_mode", "nofreq"),
            "input_dim": nofreq_row.get("input_dim", ""),
            "best_epoch": nofreq_row.get("best_epoch", ""),
            "best_metric": nofreq_row.get("best_metric", ""),
            "val_row_top1": nofreq_row.get("val_row_top1", ""),
            "val_ambiguous_row_top1": nofreq_row.get("val_ambiguous_row_top1", ""),
            "val_edge_bce": nofreq_row.get("val_edge_bce", ""),
            "val_row_ce": nofreq_row.get("val_row_ce", ""),
            "summary_csv": str(nofreq_summary),
            "checkpoint": str(nofreq_summary.parent / "best.pt"),
        },
        {
            "name": "full",
            "feature_mode": full_row.get("feature_mode", "full"),
            "input_dim": full_row.get("input_dim", ""),
            "best_epoch": full_row.get("best_epoch", ""),
            "best_metric": full_row.get("best_metric", ""),
            "val_row_top1": full_row.get("val_row_top1", ""),
            "val_ambiguous_row_top1": full_row.get("val_ambiguous_row_top1", ""),
            "val_edge_bce": full_row.get("val_edge_bce", ""),
            "val_row_ce": full_row.get("val_row_ce", ""),
            "summary_csv": str(full_summary),
            "checkpoint": str(full_summary.parent / "best.pt"),
        },
    ]
    write_rows(out_root / "offline_compare.csv", OFFLINE_COMPARE_FIELDS, offline_rows)
    offline_delta = [
        {
            "name": "full_minus_nofreq",
            "delta_val_row_top1": float(full_row.get("val_row_top1", 0.0)) - float(nofreq_row.get("val_row_top1", 0.0)),
            "delta_val_ambiguous_row_top1": float(full_row.get("val_ambiguous_row_top1", 0.0)) - float(nofreq_row.get("val_ambiguous_row_top1", 0.0)),
            "delta_val_edge_bce": float(full_row.get("val_edge_bce", 0.0)) - float(nofreq_row.get("val_edge_bce", 0.0)),
            "delta_val_row_ce": float(full_row.get("val_row_ce", 0.0)) - float(nofreq_row.get("val_row_ce", 0.0)),
        }
    ]
    write_rows(out_root / "offline_delta.csv", OFFLINE_DELTA_FIELDS, offline_delta)

    write_rows(out_root / "metrics_compare.csv", ONLINE_COMPARE_FIELDS, online_rows)
    raw_row = next((row for row in online_rows if str(row["name"]) == "raw"), None)
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
    if fgas_runtime_rows:
        runtime_fields = list(dict.fromkeys(["name", "seq_name", *fgas_runtime_rows[0].keys()]))
        write_rows(out_root / "fgas_runtime_compare.csv", runtime_fields, fgas_runtime_rows)


def main() -> None:
    args = parse_args()
    queue_name = Path(args.out_root).name if args.out_root else f"fgas_block_halfsplit_{timestamp_tag()}"
    out_root = Path(args.out_root) if args.out_root else REPO_ROOT / "outputs" / queue_name
    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "summary.csv"
    seqs_note = seq_note(list(args.seq_names))

    train_nofreq_dir = out_root / "train_nofreq"
    train_full_dir = out_root / "train_full"

    rows: List[Dict[str, object]] = [
        {
            "step": "train_nofreq",
            "name": f"{queue_name}_train_nofreq",
            "status": "pending",
            "out_dir": str(train_nofreq_dir),
            "summary_csv": str(train_nofreq_dir / "summary.csv"),
            "log_path": str(out_root / "logs" / "train_nofreq.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"arch={args.arch} feature_mode=nofreq train_jsonl={args.train_jsonl} val_jsonl={args.val_jsonl}",
        },
        {
            "step": "train_full",
            "name": f"{queue_name}_train_full",
            "status": "pending",
            "out_dir": str(train_full_dir),
            "summary_csv": str(train_full_dir / "summary.csv"),
            "log_path": str(out_root / "logs" / "train_full.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"arch={args.arch} feature_mode=full train_jsonl={args.train_jsonl} val_jsonl={args.val_jsonl}",
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
            "notes": f"raw BoT-SORT half-val over {seqs_note} aligned_to_looseB",
        },
        {
            "step": "online_nofreq_eval",
            "name": mode_experiment_name(queue_name, "nofreq"),
            "status": "pending",
            "out_dir": str(BOTSORT_ROOT / "YOLOX_outputs" / mode_experiment_name(queue_name, "nofreq")),
            "summary_csv": str(out_root / "metrics_compare.csv"),
            "log_path": str(out_root / "logs" / "online_nofreq_eval.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"FGAS nofreq half-val over {seqs_note} aligned_to_looseB",
        },
        {
            "step": "online_full_eval",
            "name": mode_experiment_name(queue_name, "full"),
            "status": "pending",
            "out_dir": str(BOTSORT_ROOT / "YOLOX_outputs" / mode_experiment_name(queue_name, "full")),
            "summary_csv": str(out_root / "metrics_compare.csv"),
            "log_path": str(out_root / "logs" / "online_full_eval.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"FGAS full half-val over {seqs_note} aligned_to_looseB",
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
            "notes": "write offline/online FGAS comparison CSVs",
        },
    ]
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    append_registry(summary_csv, out_root, "running", "fgas block half-split queue running", args.registry_csv)

    try:
        nofreq_ckpt = Path(args.nofreq_checkpoint) if args.nofreq_checkpoint else train_nofreq_dir / "best.pt"
        full_ckpt = Path(args.full_checkpoint) if args.full_checkpoint else train_full_dir / "best.pt"

        if args.skip_train:
            nofreq_local_summary = materialize_skip_train_artifacts(nofreq_ckpt, train_nofreq_dir)
            full_local_summary = materialize_skip_train_artifacts(full_ckpt, train_full_dir)
            ensure_success(nofreq_local_summary)
            ensure_success(full_local_summary)
            update_row(
                rows,
                "train_nofreq",
                status="success",
                started_at=now_iso(),
                finished_at=now_iso(),
                notes=f"skipped train; using checkpoint={Path(nofreq_ckpt).expanduser().resolve(strict=False)} source_summary={Path(nofreq_ckpt).expanduser().resolve(strict=False).parent / 'summary.csv'}",
            )
            update_row(
                rows,
                "train_full",
                status="success",
                started_at=now_iso(),
                finished_at=now_iso(),
                notes=f"skipped train; using checkpoint={Path(full_ckpt).expanduser().resolve(strict=False)} source_summary={Path(full_ckpt).expanduser().resolve(strict=False).parent / 'summary.csv'}",
            )
            write_rows(summary_csv, QUEUE_FIELDS, rows)
        else:
            update_row(rows, "train_nofreq", status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            nofreq_cmd = build_train_cmd(
                train_jsonl=args.train_jsonl,
                val_jsonl=args.val_jsonl,
                out_dir=train_nofreq_dir,
                device=args.train_device,
                arch=args.arch,
                feature_mode="nofreq",
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                hidden_dim=int(args.hidden_dim),
                stage_embed_dim=int(args.stage_embed_dim),
                num_heads=int(args.num_heads),
                num_attn_layers=int(args.num_attn_layers),
                ambiguous_oversample=float(args.ambiguous_oversample),
                col_bce_weight=float(args.col_bce_weight),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                seed=int(args.seed),
            )
            rc = run_step(nofreq_cmd, Path(next(row["log_path"] for row in rows if row["step"] == "train_nofreq")), cwd=REPO_ROOT)
            if rc != 0:
                raise RuntimeError(f"train_nofreq failed with exit code {rc}")
            ensure_success(train_nofreq_dir / "summary.csv")
            update_row(rows, "train_nofreq", status="success", finished_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)

            update_row(rows, "train_full", status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            full_cmd = build_train_cmd(
                train_jsonl=args.train_jsonl,
                val_jsonl=args.val_jsonl,
                out_dir=train_full_dir,
                device=args.train_device,
                arch=args.arch,
                feature_mode="full",
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                hidden_dim=int(args.hidden_dim),
                stage_embed_dim=int(args.stage_embed_dim),
                num_heads=int(args.num_heads),
                num_attn_layers=int(args.num_attn_layers),
                ambiguous_oversample=float(args.ambiguous_oversample),
                col_bce_weight=float(args.col_bce_weight),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                seed=int(args.seed),
            )
            rc = run_step(full_cmd, Path(next(row["log_path"] for row in rows if row["step"] == "train_full")), cwd=REPO_ROOT)
            if rc != 0:
                raise RuntimeError(f"train_full failed with exit code {rc}")
            ensure_success(train_full_dir / "summary.csv")
            update_row(rows, "train_full", status="success", finished_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)

        online_rows: List[Dict[str, float | str]] = []
        per_sequence_rows: List[Dict[str, float | str | int]] = []
        fgas_runtime_rows: List[Dict[str, object]] = []

        for step_name, mode, checkpoint in [
            ("online_raw_eval", "raw", ""),
            ("online_nofreq_eval", "nofreq", str(nofreq_ckpt)),
            ("online_full_eval", "full", str(full_ckpt)),
        ]:
            exp_name = mode_experiment_name(queue_name, mode)
            log_path = Path(next(row["log_path"] for row in rows if row["step"] == step_name))
            update_row(rows, step_name, status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)

            run_manifest = BOTSORT_ROOT / "YOLOX_outputs" / exp_name / "run_manifest.json"
            track_cmd = build_track_cmd(
                dataset_root=args.dataset_root,
                benchmark=args.benchmark,
                split=args.split,
                seq_names=list(args.seq_names),
                experiment_name=exp_name,
                run_manifest=run_manifest,
                track_profile=args.track_profile,
                track_device=args.track_device,
                track_high_thresh=float(args.track_high_thresh),
                proximity_thresh=float(args.proximity_thresh),
                appearance_thresh=float(args.appearance_thresh),
                fgas_topk=int(args.fgas_topk),
                fgas_max_rows=int(args.fgas_max_rows),
                fgas_max_cols=int(args.fgas_max_cols),
                fgas_blend_weight=float(args.fgas_blend_weight),
                fgas_assignment_mode=str(args.fgas_assignment_mode),
                fgas_row_nomatch_weight=float(args.fgas_row_nomatch_weight),
                fgas_controller_enable=bool(args.fgas_controller_enable),
                fgas_controller_edge_thresh=float(args.fgas_controller_edge_thresh),
                fgas_controller_row_defer_thresh=float(args.fgas_controller_row_defer_thresh),
                fgas_controller_col_newborn_thresh=float(args.fgas_controller_col_newborn_thresh),
                fgas_controller_margin_thresh=float(args.fgas_controller_margin_thresh),
                fgas_controller_ambiguity_margin=float(args.fgas_controller_ambiguity_margin),
                checkpoint=checkpoint,
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

            experiment_dir = BOTSORT_ROOT / "YOLOX_outputs" / exp_name
            online_rows.append(load_online_metrics(work_dir, exp_name, mode))
            per_sequence_rows.extend(load_per_sequence_metrics(work_dir, exp_name, mode))
            if checkpoint:
                fgas_runtime_rows.extend(collect_fgas_runtime_rows(experiment_dir, mode))
            update_row(rows, step_name, status="success", finished_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)

        update_row(rows, "compare", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        finalize_compare_files(
            out_root=out_root,
            nofreq_summary=train_nofreq_dir / "summary.csv",
            full_summary=train_full_dir / "summary.csv",
            online_rows=online_rows,
            per_sequence_rows=per_sequence_rows,
            fgas_runtime_rows=fgas_runtime_rows,
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
                    f"fgas_runtime_compare={out_root / 'fgas_runtime_compare.csv'}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        update_row(rows, "compare", status="success", finished_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        append_registry(summary_csv, out_root, "success", "fgas block half-split queue finished", args.registry_csv)
    except Exception as exc:
        for row in rows:
            if str(row.get("status", "")) == "running":
                row["status"] = "failed"
                row["finished_at"] = now_iso()
                row["notes"] = f"{row.get('notes', '')} error={exc}".strip()
                break
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        append_registry(summary_csv, out_root, "failed", f"fgas block half-split queue failed: {exc}", args.registry_csv)
        raise


if __name__ == "__main__":
    main()
