#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

from queue_deep_ocsort_preassoc_force_recovery_anchor_gate_dance3 import (
    anchored_variant_base,
)
from queue_deep_ocsort_preassoc_force_rewrite_next2h import (
    QUEUE_FIELDS,
    REPO_ROOT,
    REGISTRY_CSV,
    build_cmd,
    child_finished_at,
    ensure_child_success,
    now_iso,
    read_rows,
    run_step,
    timestamp_tag,
    update_row,
    write_rows,
)


CURRENT_SWEEP_ROOT = (
    REPO_ROOT / "outputs" / "deep_ocsort_preassoc_force_recovery_anchor_threshold_sweep_histfeat_dance3_20260411_1"
)
CURRENT_SWEEP_SUMMARY = CURRENT_SWEEP_ROOT / "summary.csv"
CURRENT_SWEEP_DECISION = CURRENT_SWEEP_ROOT / "decision_summary.csv"
CURRENT_HISTFEAT_CKPT = (
    REPO_ROOT / "outputs" / "train_recovery_anchor_gate_histfeat_soft_future_h16do01_seed701_20260411_1" / "best.pt"
)
HISTFEAT_SPLIT_ROOT = REPO_ROOT / "outputs" / "recovery_anchor_sequence_split_histfeat_soft_future_full_20260411_1"
FULLVAL_REUSE_RAW_ROOT = REPO_ROOT / "outputs" / "deep_ocsort_local_contention_export_dance_best_20260406_1"

TRAIN_SELECTION_FIELDS = [
    "rank",
    "variant_name",
    "summary_csv",
    "checkpoint",
    "hidden_dim",
    "dropout",
    "seed",
    "pos_weight",
    "best_metric",
    "val_balanced_accuracy",
    "val_precision",
    "val_recall",
    "val_f1",
    "best_threshold",
    "selected",
    "notes",
]

SWEEP_SELECTION_FIELDS = [
    "rank",
    "source",
    "variant_name",
    "decision_csv",
    "checkpoint",
    "threshold",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDs",
    "delta_Frag",
    "score",
    "selected",
    "notes",
]

FULL_DANCE_SEQS = [
    "dancetrack0004",
    "dancetrack0005",
    "dancetrack0007",
    "dancetrack0010",
    "dancetrack0014",
    "dancetrack0018",
    "dancetrack0019",
    "dancetrack0025",
    "dancetrack0026",
    "dancetrack0030",
    "dancetrack0034",
    "dancetrack0035",
    "dancetrack0041",
    "dancetrack0043",
    "dancetrack0047",
    "dancetrack0058",
    "dancetrack0063",
    "dancetrack0065",
    "dancetrack0073",
    "dancetrack0077",
    "dancetrack0079",
    "dancetrack0081",
    "dancetrack0090",
    "dancetrack0094",
    "dancetrack0097",
]

TRAIN_VARIANTS = [
    {
        "name": "histfeat_pw6_h16do01_seed701",
        "hidden_dim": 16,
        "dropout": 0.1,
        "seed": 701,
        "pos_weight": 6.0,
    },
    {
        "name": "histfeat_pw4_h16do01_seed701",
        "hidden_dim": 16,
        "dropout": 0.1,
        "seed": 701,
        "pos_weight": 4.0,
    },
    {
        "name": "histfeat_pw6_h8do00_seed701",
        "hidden_dim": 8,
        "dropout": 0.0,
        "seed": 701,
        "pos_weight": 6.0,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autopilot queue for the next 5+ hours of recovery-anchor histfeat experiments."
    )
    parser.add_argument("--out-root", default="")
    parser.add_argument("--wait-summary-csv", default=str(CURRENT_SWEEP_SUMMARY))
    parser.add_argument("--wait-decision-csv", default=str(CURRENT_SWEEP_DECISION))
    parser.add_argument(
        "--wait-process-pattern",
        default=str(CURRENT_SWEEP_ROOT.name),
        help="substring expected in the live process table while the waited queue is still running",
    )
    parser.add_argument("--histfeat-split-root", default=str(HISTFEAT_SPLIT_ROOT))
    parser.add_argument("--reference-checkpoint", default=str(CURRENT_HISTFEAT_CKPT))
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--train-epochs", type=int, default=80)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--train-lr", type=float, default=1e-3)
    parser.add_argument("--train-weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-device", default="cuda")
    parser.add_argument(
        "--sweep-thresholds",
        nargs="*",
        type=float,
        default=[0.55, 0.60, 0.66, 0.72],
    )
    parser.add_argument(
        "--seq-names",
        nargs="*",
        default=["dancetrack0081", "dancetrack0090", "dancetrack0094"],
    )
    parser.add_argument("--fullval-reuse-raw-from", default=str(FULLVAL_REUSE_RAW_ROOT))
    parser.add_argument("--fullval-frame-budget", type=int, default=12000)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


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
        "scripts/queue_recovery_anchor_histfeat_next5h.py",
        "--dataset",
        "DanceTrack",
        "--split",
        "recovery_anchor_histfeat_next5h",
        "--tracker-family",
        "deep_ocsort_preassoc_force_recovery_anchor",
        "--variant",
        run_root.name,
        "--tag",
        "recovery_anchor_histfeat_next5h",
        "--run-root",
        str(run_root.resolve()),
        "--summary-csv",
        str(summary_csv.resolve()),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def write_log(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(str(line).rstrip())
            handle.write("\n")


def make_row(step: str, out_dir: Path, log_path: Path, notes: str, summary_path: Path | None = None) -> Dict[str, object]:
    return {
        "step": step,
        "name": step,
        "status": "pending",
        "out_dir": str(out_dir.resolve()),
        "summary_csv": str((summary_path or (out_dir / "summary.csv")).resolve()),
        "log_path": str(log_path.resolve()),
        "started_at": "",
        "finished_at": "",
        "notes": notes,
    }


def process_alive(pattern: str) -> bool:
    result = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True, check=False)
    for line in result.stdout.splitlines():
        if pattern in line and "ps -eo cmd" not in line:
            return True
    return False


def wait_for_queue(summary_csv: Path, wait_process_pattern: str, poll_seconds: int) -> str:
    while True:
        rows = read_rows(summary_csv)
        statuses = {str(row.get("status", "")).strip() for row in rows}
        if rows and statuses == {"success"}:
            return child_finished_at(summary_csv)
        if "failed" in statuses:
            failed_steps = [str(row.get("step", "")) for row in rows if str(row.get("status", "")) == "failed"]
            raise RuntimeError(f"waited queue failed at steps: {failed_steps}")
        if rows and not process_alive(wait_process_pattern):
            raise RuntimeError(
                f"waited queue not finished, but no live process matches pattern {wait_process_pattern!r}"
            )
        time.sleep(max(int(poll_seconds), 15))


def read_single_row(summary_csv: Path) -> Dict[str, str]:
    rows = read_rows(summary_csv)
    if not rows:
        raise FileNotFoundError(f"missing summary rows: {summary_csv}")
    return rows[0]


def parse_float(row: Dict[str, str], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def parse_train_candidate(variant: Dict[str, object], out_dir: Path) -> Dict[str, object]:
    row = read_single_row(out_dir / "summary.csv")
    return {
        "variant_name": str(variant["name"]),
        "summary_csv": str((out_dir / "summary.csv").resolve()),
        "checkpoint": str((out_dir / "best.pt").resolve()),
        "hidden_dim": int(variant["hidden_dim"]),
        "dropout": float(variant["dropout"]),
        "seed": int(variant["seed"]),
        "pos_weight": float(variant["pos_weight"]),
        "best_metric": float(row.get("best_metric", 0.0) or 0.0),
        "val_balanced_accuracy": float(row.get("val_balanced_accuracy", 0.0) or 0.0),
        "val_precision": float(row.get("val_precision", 0.0) or 0.0),
        "val_recall": float(row.get("val_recall", 0.0) or 0.0),
        "val_f1": float(row.get("val_f1", 0.0) or 0.0),
        "best_threshold": float(row.get("best_threshold", 0.0) or 0.0),
    }


def train_rank_key(row: Dict[str, object]) -> tuple[float, float, float, float]:
    return (
        float(row.get("val_balanced_accuracy", 0.0)),
        float(row.get("val_precision", 0.0)),
        float(row.get("val_f1", 0.0)),
        float(row.get("best_metric", 0.0)),
    )


def decision_rank_key(row: Dict[str, object]) -> tuple[float, float, float, float, float]:
    delta_hota = float(row.get("delta_HOTA", 0.0))
    delta_assa = float(row.get("delta_AssA", 0.0))
    delta_idf1 = float(row.get("delta_IDF1", 0.0))
    composite = delta_hota + delta_assa + delta_idf1
    threshold = float(row.get("threshold", 0.0))
    return (composite, delta_assa, delta_hota, delta_idf1, -abs(threshold - 0.66))


def read_best_decision(decision_csv: Path, *, source: str, variant_name: str) -> Dict[str, object]:
    rows = [row for row in read_rows(decision_csv) if str(row.get("status", "")) == "success"]
    if not rows:
        raise FileNotFoundError(f"missing successful decision rows: {decision_csv}")
    best = max(
        rows,
        key=lambda row: decision_rank_key(
            {
                "delta_HOTA": parse_float(row, "delta_HOTA"),
                "delta_AssA": parse_float(row, "delta_AssA"),
                "delta_IDF1": parse_float(row, "delta_IDF1"),
                "threshold": parse_float(row, "recovery_anchor_gate_thresh"),
            }
        ),
    )
    delta_hota = parse_float(best, "delta_HOTA")
    delta_assa = parse_float(best, "delta_AssA")
    delta_idf1 = parse_float(best, "delta_IDF1")
    composite = delta_hota + delta_assa + delta_idf1
    return {
        "source": source,
        "variant_name": variant_name,
        "decision_csv": str(decision_csv.resolve()),
        "checkpoint": str(best.get("recovery_anchor_gate_checkpoint", "") or ""),
        "threshold": parse_float(best, "recovery_anchor_gate_thresh"),
        "delta_HOTA": delta_hota,
        "delta_AssA": delta_assa,
        "delta_IDF1": delta_idf1,
        "delta_MOTA": parse_float(best, "delta_MOTA"),
        "delta_IDs": parse_float(best, "delta_IDs"),
        "delta_Frag": parse_float(best, "delta_Frag"),
        "score": composite,
        "notes": str(best.get("step", "")),
    }


def promising_enough(row: Dict[str, object]) -> bool:
    return (
        float(row.get("delta_HOTA", 0.0)) >= 0.05
        and float(row.get("delta_AssA", 0.0)) >= 0.08
        and float(row.get("delta_IDF1", 0.0)) >= 0.12
    )


def build_train_cmd(args: argparse.Namespace, *, variant: Dict[str, object], out_dir: Path) -> List[str]:
    split_root = Path(args.histfeat_split_root).expanduser().resolve()
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_fgas_acceptance_gate.py"),
        "--train-jsonl",
        str((split_root / "train.jsonl").resolve()),
        "--val-jsonl",
        str((split_root / "val.jsonl").resolve()),
        "--out-dir",
        str(out_dir.resolve()),
        "--device",
        str(args.train_device),
        "--epochs",
        str(int(args.train_epochs)),
        "--batch-size",
        str(int(args.train_batch_size)),
        "--lr",
        str(float(args.train_lr)),
        "--weight-decay",
        str(float(args.train_weight_decay)),
        "--hidden-dim",
        str(int(variant["hidden_dim"])),
        "--dropout",
        str(float(variant["dropout"])),
        "--pos-weight",
        str(float(variant["pos_weight"])),
        "--seed",
        str(int(variant["seed"])),
        "--registry-dataset",
        "DanceTrack",
        "--registry-split",
        "recovery_anchor_seqsplit",
        "--registry-tracker-family",
        "deep_ocsort_preassoc_force_recovery_anchor",
        "--registry-variant",
        out_dir.name,
        "--registry-tag",
        out_dir.name,
        "--registry-notes",
        f"precision-biased histfeat recovery anchor gate training {variant['name']}",
    ]


def build_sweep_cmd(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    out_root: Path,
) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "queue_deep_ocsort_preassoc_force_recovery_anchor_threshold_sweep_dance3.py"),
        "--out-root",
        str(out_root.resolve()),
        "--recovery-anchor-gate-checkpoint",
        str(checkpoint.resolve()),
        "--thresholds",
        *[str(float(value)) for value in list(args.sweep_thresholds or [])],
    ]


def build_fullval_cmd(
    args: argparse.Namespace,
    *,
    checkpoint: str,
    threshold: float,
    out_root: Path,
) -> List[str]:
    variant = {
        **anchored_variant_base(list(FULL_DANCE_SEQS)),
        "acceptance_gate_checkpoint": str(
            REPO_ROOT
            / "outputs"
            / "local_contention_acceptance_gate_mot17_mot20_dance_seqholdout_20260408_1"
            / "best.pt"
        ),
        "acceptance_gate_thresh": 0.9995,
        "recovery_anchor_gate_checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "recovery_anchor_gate_thresh": float(threshold),
    }
    cmd = build_cmd(
        out_dir=out_root.resolve(),
        reuse_raw_from=Path(args.fullval_reuse_raw_from).expanduser().resolve(),
        seq_names=list(FULL_DANCE_SEQS),
        variant=variant,
    )
    cmd.extend(
        [
            "--competition-track-max-frames-per-batch",
            str(int(args.fullval_frame_budget)),
        ]
    )
    return cmd


def mark_step(
    queue_rows: List[Dict[str, object]],
    summary_csv: Path,
    *,
    step: str,
    status: str,
    notes: str,
) -> None:
    update_row(queue_rows, step, status=status, finished_at=now_iso(), notes=notes)
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)


def main() -> int:
    args = parse_args()
    run_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else (REPO_ROOT / "outputs" / f"queue_recovery_anchor_histfeat_next5h_{timestamp_tag()}").resolve()
    )
    logs_dir = run_root / "logs"
    trains_dir = run_root / "trains"
    sweeps_dir = run_root / "sweeps"
    fullval_dir = run_root / "fullval"

    summary_csv = run_root / "summary.csv"
    train_selection_csv = run_root / "train_selection.csv"
    sweep_selection_csv = run_root / "sweep_selection.csv"

    queue_rows: List[Dict[str, object]] = [
        make_row(
            "wait_current_histfeat_sweep",
            CURRENT_SWEEP_ROOT,
            logs_dir / "wait_current_histfeat_sweep.log",
            "wait for the running histfeat DanceTrack three-sequence threshold sweep to finish",
            Path(args.wait_summary_csv).expanduser().resolve(),
        ),
        make_row(
            "train_histfeat_pw6_h16do01_seed701",
            trains_dir / "histfeat_pw6_h16do01_seed701",
            logs_dir / "train_histfeat_pw6_h16do01_seed701.log",
            "train histfeat gate with lower positive weight 6.0 to suppress false positives",
        ),
        make_row(
            "train_histfeat_pw4_h16do01_seed701",
            trains_dir / "histfeat_pw4_h16do01_seed701",
            logs_dir / "train_histfeat_pw4_h16do01_seed701.log",
            "train histfeat gate with even lower positive weight 4.0 for a stronger precision push",
        ),
        make_row(
            "train_histfeat_pw6_h8do00_seed701",
            trains_dir / "histfeat_pw6_h8do00_seed701",
            logs_dir / "train_histfeat_pw6_h8do00_seed701.log",
            "train a smaller histfeat gate with positive weight 6.0 to test a lower-capacity precision-biased branch",
        ),
        make_row(
            "select_top2_train_variants",
            run_root,
            logs_dir / "select_top2_train_variants.log",
            "rank all precision-biased training branches and keep the top two for three-sequence rescreening",
            train_selection_csv,
        ),
        make_row(
            "sweep_rank1_variant",
            sweeps_dir / "rank1",
            logs_dir / "sweep_rank1_variant.log",
            "run a four-threshold three-sequence sweep for the best precision-biased training branch",
        ),
        make_row(
            "sweep_rank2_variant",
            sweeps_dir / "rank2",
            logs_dir / "sweep_rank2_variant.log",
            "run a four-threshold three-sequence sweep for the runner-up precision-biased training branch",
        ),
        make_row(
            "select_best_sweep_variant",
            run_root,
            logs_dir / "select_best_sweep_variant.log",
            "compare the current running histfeat reference against the two new sweeps and choose a winner",
            sweep_selection_csv,
        ),
        make_row(
            "fullval_best_variant_if_promising",
            fullval_dir / "winner_danceval",
            logs_dir / "fullval_best_variant_if_promising.log",
            "run full DanceTrack validation only if the best sweep result is strong enough to justify the cost",
        ),
    ]

    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
    write_rows(train_selection_csv, TRAIN_SELECTION_FIELDS, [])
    write_rows(sweep_selection_csv, SWEEP_SELECTION_FIELDS, [])
    append_registry(summary_csv, run_root, "running", "started 5h histfeat recovery-anchor autopilot queue", args.registry_csv)

    overall_status = "success"
    overall_notes = "completed 5h histfeat recovery-anchor autopilot queue"
    had_failed_step = False

    try:
        wait_step = "wait_current_histfeat_sweep"
        update_row(queue_rows, wait_step, status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
        finished_at = wait_for_queue(
            Path(args.wait_summary_csv).expanduser().resolve(),
            str(args.wait_process_pattern),
            int(args.poll_seconds),
        )
        current_reference = read_best_decision(
            Path(args.wait_decision_csv).expanduser().resolve(),
            source="current_histfeat_reference",
            variant_name="current_histfeat_reference",
        )
        mark_step(
            queue_rows,
            summary_csv,
            step=wait_step,
            status="success",
            notes=(
                f"finished_at={finished_at} "
                f"best_ref_threshold={float(current_reference['threshold']):.3f} "
                f"ref_delta_HOTA={float(current_reference['delta_HOTA']):+.3f} "
                f"ref_delta_AssA={float(current_reference['delta_AssA']):+.3f} "
                f"ref_delta_IDF1={float(current_reference['delta_IDF1']):+.3f}"
            ),
        )

        train_candidates: List[Dict[str, object]] = []
        for variant in TRAIN_VARIANTS:
            step = f"train_{variant['name']}"
            out_dir = trains_dir / str(variant["name"])
            log_path = logs_dir / f"{step}.log"
            update_row(queue_rows, step, status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
            rc = run_step(build_train_cmd(args, variant=variant, out_dir=out_dir), log_path, cwd=REPO_ROOT)
            if rc != 0:
                had_failed_step = True
                mark_step(queue_rows, summary_csv, step=step, status="failed", notes=f"child return code {rc}")
                continue
            child_summary = out_dir / "summary.csv"
            child_row = read_single_row(child_summary)
            if str(child_row.get("status", "")) != "success":
                had_failed_step = True
                mark_step(
                    queue_rows,
                    summary_csv,
                    step=step,
                    status="failed",
                    notes=f"unexpected child status {child_row.get('status', '')}",
                )
                continue
            candidate = parse_train_candidate(variant, out_dir)
            train_candidates.append(candidate)
            mark_step(
                queue_rows,
                summary_csv,
                step=step,
                status="success",
                notes=(
                    f"best_metric={float(candidate['best_metric']):.3f} "
                    f"val_precision={float(candidate['val_precision']):.3f} "
                    f"val_recall={float(candidate['val_recall']):.3f} "
                    f"best_threshold={float(candidate['best_threshold']):.3f}"
                ),
            )

        if not train_candidates:
            raise RuntimeError("all precision-biased histfeat training branches failed")

        ranked_train = sorted(train_candidates, key=train_rank_key, reverse=True)
        selected_train = ranked_train[: min(2, len(ranked_train))]
        train_rows: List[Dict[str, object]] = []
        for index, row in enumerate(ranked_train, start=1):
            train_rows.append(
                {
                    "rank": int(index),
                    "variant_name": str(row["variant_name"]),
                    "summary_csv": str(row["summary_csv"]),
                    "checkpoint": str(row["checkpoint"]),
                    "hidden_dim": int(row["hidden_dim"]),
                    "dropout": float(row["dropout"]),
                    "seed": int(row["seed"]),
                    "pos_weight": float(row["pos_weight"]),
                    "best_metric": float(row["best_metric"]),
                    "val_balanced_accuracy": float(row["val_balanced_accuracy"]),
                    "val_precision": float(row["val_precision"]),
                    "val_recall": float(row["val_recall"]),
                    "val_f1": float(row["val_f1"]),
                    "best_threshold": float(row["best_threshold"]),
                    "selected": int(row in selected_train),
                    "notes": "selected for dance3 sweep" if row in selected_train else "",
                }
            )
        write_rows(train_selection_csv, TRAIN_SELECTION_FIELDS, train_rows)
        mark_step(
            queue_rows,
            summary_csv,
            step="select_top2_train_variants",
            status="success",
            notes="selected " + "|".join(str(row["variant_name"]) for row in selected_train),
        )

        sweep_candidates: List[Dict[str, object]] = [dict(current_reference)]
        sweep_steps = ["sweep_rank1_variant", "sweep_rank2_variant"]
        for step, row in zip(sweep_steps, selected_train):
            sweep_out_root = sweeps_dir / str(row["variant_name"])
            log_path = logs_dir / f"{step}.log"
            update_row(queue_rows, step, status="running", started_at=now_iso(), out_dir=str(sweep_out_root.resolve()))
            write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
            rc = run_step(
                build_sweep_cmd(args, checkpoint=Path(str(row["checkpoint"])), out_root=sweep_out_root),
                log_path,
                cwd=REPO_ROOT,
            )
            if rc != 0:
                had_failed_step = True
                mark_step(queue_rows, summary_csv, step=step, status="failed", notes=f"child return code {rc}")
                continue
            ensure_child_success(sweep_out_root / "summary.csv")
            best_row = read_best_decision(
                sweep_out_root / "decision_summary.csv",
                source=str(row["variant_name"]),
                variant_name=str(row["variant_name"]),
            )
            sweep_candidates.append(best_row)
            mark_step(
                queue_rows,
                summary_csv,
                step=step,
                status="success",
                notes=(
                    f"best_threshold={float(best_row['threshold']):.3f} "
                    f"delta_HOTA={float(best_row['delta_HOTA']):+.3f} "
                    f"delta_AssA={float(best_row['delta_AssA']):+.3f} "
                    f"delta_IDF1={float(best_row['delta_IDF1']):+.3f}"
                ),
            )

        ranked_sweeps = sorted(sweep_candidates, key=decision_rank_key, reverse=True)
        sweep_rows: List[Dict[str, object]] = []
        for index, row in enumerate(ranked_sweeps, start=1):
            sweep_rows.append(
                {
                    "rank": int(index),
                    "source": str(row["source"]),
                    "variant_name": str(row["variant_name"]),
                    "decision_csv": str(row["decision_csv"]),
                    "checkpoint": str(row["checkpoint"]),
                    "threshold": float(row["threshold"]),
                    "delta_HOTA": float(row["delta_HOTA"]),
                    "delta_AssA": float(row["delta_AssA"]),
                    "delta_IDF1": float(row["delta_IDF1"]),
                    "delta_MOTA": float(row["delta_MOTA"]),
                    "delta_IDs": float(row["delta_IDs"]),
                    "delta_Frag": float(row["delta_Frag"]),
                    "score": float(row["score"]),
                    "selected": int(index == 1),
                    "notes": str(row.get("notes", "")),
                }
            )
        write_rows(sweep_selection_csv, SWEEP_SELECTION_FIELDS, sweep_rows)
        winner = ranked_sweeps[0]
        mark_step(
            queue_rows,
            summary_csv,
            step="select_best_sweep_variant",
            status="success",
            notes=(
                f"winner={winner['variant_name']} "
                f"threshold={float(winner['threshold']):.3f} "
                f"delta_HOTA={float(winner['delta_HOTA']):+.3f} "
                f"delta_AssA={float(winner['delta_AssA']):+.3f} "
                f"delta_IDF1={float(winner['delta_IDF1']):+.3f}"
            ),
        )

        fullval_step = "fullval_best_variant_if_promising"
        if not promising_enough(winner):
            mark_step(
                queue_rows,
                summary_csv,
                step=fullval_step,
                status="success",
                notes=(
                    "skipped fullval: winner not strong enough "
                    f"(delta_HOTA={float(winner['delta_HOTA']):+.3f}, "
                    f"delta_AssA={float(winner['delta_AssA']):+.3f}, "
                    f"delta_IDF1={float(winner['delta_IDF1']):+.3f})"
                ),
            )
        else:
            out_root = fullval_dir / f"{str(winner['variant_name'])}_t{int(round(float(winner['threshold']) * 1000)):04d}"
            log_path = logs_dir / f"{fullval_step}.log"
            update_row(
                queue_rows,
                fullval_step,
                status="running",
                started_at=now_iso(),
                out_dir=str(out_root.resolve()),
            )
            write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
            rc = run_step(
                build_fullval_cmd(
                    args,
                    checkpoint=str(winner["checkpoint"]),
                    threshold=float(winner["threshold"]),
                    out_root=out_root,
                ),
                log_path,
                cwd=REPO_ROOT,
            )
            if rc != 0:
                had_failed_step = True
                mark_step(queue_rows, summary_csv, step=fullval_step, status="failed", notes=f"child return code {rc}")
            else:
                ensure_child_success(out_root / "summary.csv")
                mark_step(
                    queue_rows,
                    summary_csv,
                    step=fullval_step,
                    status="success",
                    notes=(
                        f"launched full DanceTrack val for winner={winner['variant_name']} "
                        f"threshold={float(winner['threshold']):.3f}"
                    ),
                )

    except Exception as exc:
        overall_status = "failed"
        overall_notes = f"autopilot queue failed: {exc}"
    finally:
        if overall_status == "success" and had_failed_step:
            overall_status = "failed"
            overall_notes = "autopilot queue finished with at least one failed child step"
        append_registry(summary_csv, run_root, overall_status, overall_notes, args.registry_csv)

    return 0 if overall_status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
