#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
PLAN_CSV = REPO_ROOT / "outputs" / "experiment_plan.csv"

TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_graph_assoc_commit_policy.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "run_botsort_graphassoc_mot20_eval.py"

DEFAULT_DATA_JSONL = (
    REPO_ROOT
    / "outputs"
    / "graph_assoc_commit_mot20_expand_20260503_1100"
    / "graph_assoc_commit_data"
    / "cluster_examples.jsonl"
)
DEFAULT_SOURCE_MANIFEST = REPO_ROOT / "outputs" / "graph_assoc_commit_mot20_expand_20260503_1100" / "source_manifest.csv"
DEFAULT_INIT_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "graph_assoc_commit_policy_routed_moe_psign_mot20_expand_20260503_1310"
    / "best.pt"
)

QUEUE_FIELDS = [
    "step",
    "name",
    "status",
    "run_root",
    "summary_csv",
    "log_path",
    "started_at",
    "finished_at",
    "checkpoint",
    "best_epoch",
    "best_metric",
    "best_threshold",
    "decision_mode",
    "score_margin",
    "delta_hota",
    "delta_assa",
    "delta_idf1",
    "delta_mota",
    "delta_ids",
    "delta_frag",
    "identical_count",
    "notes",
    "params_json",
]

WIDE_GRAPH_ARGS = [
    "--graph-assoc-top-k",
    "4",
    "--graph-assoc-max-rows",
    "5",
    "--graph-assoc-max-cols",
    "5",
    "--graph-assoc-row-margin",
    "0.06",
    "--graph-assoc-col-margin",
    "0.06",
    "--graph-assoc-min-reclaim-tracklet-len",
    "12",
    "--graph-assoc-recent-owner-max-tracklet-len",
    "14",
    "--graph-assoc-min-box-iou",
    "0.50",
    "--graph-assoc-reclaim-bonus",
    "0.16",
    "--graph-assoc-recent-owner-penalty",
    "0.08",
    "--graph-assoc-iou-bonus",
    "0.06",
    "--graph-assoc-score-bonus",
    "0.02",
    "--graph-assoc-min-assignment-gain",
    "0.003",
    "--graph-assoc-max-cost-delta",
    "0.10",
    "--graph-assoc-row-involved-min-assignment-gain",
    "0.003",
    "--graph-assoc-col-only-min-assignment-gain",
    "0.003",
    "--graph-assoc-col-only-max-cost-delta",
    "0.10",
    "--graph-assoc-force-match-cost",
    "0.0",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Train and evaluate a calibrated residual graph-assoc commit policy.")
    parser.add_argument("--run-root", default=str(REPO_ROOT / "outputs" / f"graphassoc_commit_calibrated_next5h_{ts}"))
    parser.add_argument("--queue-name", default=f"graphassoc_commit_calibrated_next5h_{ts}")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="resume from an existing queue summary if present and skip completed steps",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--data-jsonl", default=str(DEFAULT_DATA_JSONL))
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_INIT_CHECKPOINT))
    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--eval-device", default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--policy-hidden-dim", type=int, default=192)
    parser.add_argument("--token-dim", type=int, default=192)
    parser.add_argument("--num-slots", type=int, default=6)
    parser.add_argument("--num-encoder-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=3)
    parser.add_argument("--router-hidden-dim", type=int, default=192)
    parser.add_argument("--router-temperature", type=float, default=0.8)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--gain-loss-weight", type=float, default=1.0)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--router-balance-loss-weight", type=float, default=0.02)
    parser.add_argument("--expert-aux-loss-weight", type=float, default=0.2)
    parser.add_argument("--rank-loss-weight", type=float, default=0.5)
    parser.add_argument("--rank-margin", type=float, default=0.1)
    parser.add_argument("--ordinal-rank-loss-weight", type=float, default=0.25)
    parser.add_argument("--ordinal-rank-margin", type=float, default=0.08)
    parser.add_argument("--train-selection-head-only", action="store_true")
    parser.add_argument("--selection-metric", default="policy_sign_acc", choices=["policy_sign_acc", "action_macro_f1", "gain_sign_acc"])
    parser.add_argument("--seq-ids", nargs="+", type=int, default=[2, 5])
    parser.add_argument("--max-hours", type=float, default=5.0)
    return parser.parse_args()


def write_rows(path: Path, fieldnames: Iterable[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def append_row(
    rows: List[Dict[str, object]],
    *,
    step: str,
    name: str,
    status: str,
    run_root: Path,
    summary_csv: Path,
    log_path: Path,
    checkpoint: str = "",
    best_epoch: str = "",
    best_metric: str = "",
    best_threshold: str = "",
    decision_mode: str = "",
    score_margin: str = "",
    delta_hota: str = "",
    delta_assa: str = "",
    delta_idf1: str = "",
    delta_mota: str = "",
    delta_ids: str = "",
    delta_frag: str = "",
    identical_count: str = "",
    notes: str = "",
    params_json: str = "",
) -> None:
    rows.append(
        {
            "step": step,
            "name": name,
            "status": status,
            "run_root": str(run_root),
            "summary_csv": str(summary_csv),
            "log_path": str(log_path),
            "started_at": "",
            "finished_at": "",
            "checkpoint": checkpoint,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "best_threshold": best_threshold,
            "decision_mode": decision_mode,
            "score_margin": score_margin,
            "delta_hota": delta_hota,
            "delta_assa": delta_assa,
            "delta_idf1": delta_idf1,
            "delta_mota": delta_mota,
            "delta_ids": delta_ids,
            "delta_frag": delta_frag,
            "identical_count": identical_count,
            "notes": notes,
            "params_json": params_json,
        }
    )


def find_row(rows: List[Dict[str, object]], step: str) -> Dict[str, object] | None:
    for row in rows:
        if str(row.get("step", "")) == str(step):
            return row
    return None


def update_row(rows: List[Dict[str, object]], step: str, **updates: object) -> None:
    row = find_row(rows, step)
    if row is None:
        raise KeyError(f"Missing queue step: {step}")
    row.update(updates)


def append_log_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def run_step(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[started_at] {now_iso()}\n")
        handle.write("[cmd] " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[finished_at] {now_iso()}\n")
        handle.write(f"[return_code] {proc.returncode}\n")
    return int(proc.returncode)


def parse_single_row_csv(path: Path) -> Dict[str, str]:
    rows = read_csv_rows(path)
    return rows[0] if rows else {}


def parse_train_summary(summary_csv: Path) -> Dict[str, object]:
    row = parse_single_row_csv(summary_csv)
    return {
        "best_epoch": int(float(row.get("best_epoch", 0) or 0)) if str(row.get("best_epoch", "")).strip() else "",
        "best_metric": float(row.get("best_metric", 0.0) or 0.0),
        "best_threshold": float(row.get("best_threshold", 0.0) or 0.0),
        "train_rows": int(float(row.get("train_rows", 0) or 0)),
        "val_rows": int(float(row.get("val_rows", 0) or 0)),
        "status": str(row.get("status", "")).strip(),
    }


def parse_metrics_delta(run_root: Path) -> Dict[str, object]:
    rows = read_csv_rows(run_root / "metrics_delta.csv")
    if not rows:
        raise FileNotFoundError(f"Missing metrics_delta.csv under {run_root}")
    row = rows[0]
    return {
        "delta_hota": float(row.get("delta_HOTA", 0.0) or 0.0),
        "delta_assa": float(row.get("delta_AssA", 0.0) or 0.0),
        "delta_idf1": float(row.get("delta_IDF1", 0.0) or 0.0),
        "delta_mota": float(row.get("delta_MOTA", 0.0) or 0.0),
        "delta_ids": int(round(float(row.get("delta_IDs", 0.0) or 0.0))),
        "delta_frag": int(round(float(row.get("delta_Frag", 0.0) or 0.0))),
    }


def parse_identical_count(run_root: Path) -> int:
    rows = read_csv_rows(run_root / "track_diff_summary.csv")
    return int(sum(int(float(row.get("identical", 0) or 0)) for row in rows))


def queue_plan_status(args: argparse.Namespace, status: str, summary_csv: Path, log_path: Path, notes: str = "") -> None:
    cmd = [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "upsert_experiment_plan.py"),
        "--csv",
        str(PLAN_CSV),
        "--key",
        f"run_root:{Path(args.run_root).expanduser().resolve()}",
        "--status",
        status,
        "--kind",
        "analysis",
        "--script",
        "scripts/run_graphassoc_commit_calibrated_next5h.py",
        "--dataset",
        "MOT20",
        "--split",
        "val_half",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        "graphassoc_commit_calibrated_next5h",
        "--tag",
        args.queue_name,
        "--run-root",
        str(Path(args.run_root).expanduser().resolve()),
        "--summary-csv",
        str(summary_csv),
        "--log-path",
        str(log_path),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def queue_registry(args: argparse.Namespace, status: str, summary_csv: Path, log_path: Path, notes: str = "") -> None:
    cmd = [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(REGISTRY_CSV),
        "--kind",
        "analysis",
        "--status",
        status,
        "--script",
        "scripts/run_graphassoc_commit_calibrated_next5h.py",
        "--dataset",
        "MOT20",
        "--split",
        "val_half",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        "graphassoc_commit_calibrated_next5h",
        "--tag",
        args.queue_name,
        "--run-root",
        str(Path(args.run_root).expanduser().resolve()),
        "--summary-csv",
        str(summary_csv),
        "--log-path",
        str(log_path),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def train_cmd(args: argparse.Namespace, train_dir: Path) -> List[str]:
    cmd = [
        args.python_bin,
        str(TRAIN_SCRIPT),
        "--data-jsonl",
        str(Path(args.data_jsonl).expanduser().resolve()),
        "--out-dir",
        str(train_dir),
        "--device",
        str(args.train_device),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--hidden-dim",
        str(args.hidden_dim),
        "--policy-hidden-dim",
        str(args.policy_hidden_dim),
        "--token-dim",
        str(args.token_dim),
        "--num-slots",
        str(args.num_slots),
        "--num-encoder-layers",
        str(args.num_encoder_layers),
        "--num-heads",
        str(args.num_heads),
        "--num-experts",
        str(args.num_experts),
        "--router-hidden-dim",
        str(args.router_hidden_dim),
        "--router-temperature",
        str(args.router_temperature),
        "--policy-score-mode",
        "gated_blend",
        "--train-sequences",
        "MOT20-01,MOT20-02,MOT20-03",
        "--val-sequences",
        "MOT20-05",
        "--strict-sequence-split",
        "--selection-metric",
        str(args.selection_metric),
        "--model-arch",
        "routed_moe_v1",
        "--use-balanced-sampler",
    ]
    if bool(args.train_selection_head_only):
        cmd.append("--train-selection-head-only")
    cmd.extend(
        [
            "--action-loss-weight",
            str(args.action_loss_weight),
            "--gain-loss-weight",
            str(args.gain_loss_weight),
            "--policy-loss-weight",
            str(args.policy_loss_weight),
            "--router-balance-loss-weight",
            str(args.router_balance_loss_weight),
            "--expert-aux-loss-weight",
            str(args.expert_aux_loss_weight),
            "--rank-loss-weight",
            str(args.rank_loss_weight),
            "--rank-margin",
            str(args.rank_margin),
            "--ordinal-rank-loss-weight",
            str(args.ordinal_rank_loss_weight),
            "--ordinal-rank-margin",
            str(args.ordinal_rank_margin),
            "--init-checkpoint",
            str(Path(args.init_checkpoint).expanduser().resolve()),
            "--dataset-tag",
            "graph_assoc_commit_policy_mot20_calibrated",
            "--source-manifest",
            str(Path(args.source_manifest).expanduser().resolve()),
        ]
    )
    return cmd


def eval_cmd(
    args: argparse.Namespace,
    *,
    run_root: Path,
    experiment_name: str,
    checkpoint: Path,
    decision_mode: str,
    score_margin: float,
    threshold: float,
) -> List[str]:
    return [
        args.python_bin,
        str(EVAL_SCRIPT),
        "--run-root",
        str(run_root),
        "--experiment-name",
        experiment_name,
        "--variant-name",
        "botsort_graphassoc_gate_structural",
        "--seq-ids",
        *[str(v) for v in args.seq_ids],
        *WIDE_GRAPH_ARGS,
        "--graph-assoc-commit-checkpoint",
        str(checkpoint),
        "--graph-assoc-commit-device",
        str(args.eval_device),
        "--graph-assoc-commit-score-margin",
        str(score_margin),
        "--graph-assoc-commit-decision-mode",
        decision_mode,
        "--graph-assoc-commit-threshold",
        str(threshold),
        "--graph-assoc-commit-replace-rules",
        "--graph-assoc-dump-candidate-rows",
    ]


def rank_results(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        results,
        key=lambda row: (
            float(row.get("delta_hota", -999.0)),
            float(row.get("delta_idf1", -999.0)),
            float(row.get("delta_assa", -999.0)),
            float(row.get("delta_mota", -999.0)),
            -abs(int(row.get("delta_ids", 999999))),
            -abs(int(row.get("delta_frag", 999999))),
            -int(row.get("identical_count", 999999)),
        ),
        reverse=True,
    )


def main() -> int:
    args = parse_args()
    queue_root = Path(args.run_root).expanduser().resolve()
    logs_dir = queue_root / "logs"
    summary_csv = queue_root / "summary.csv"
    queue_log = logs_dir / "queue.log"
    train_dir = queue_root / "train_policy"
    eval_root = queue_root / "mot20_eval"
    deadline = datetime.now().timestamp() + max(0.5, float(args.max_hours)) * 3600.0

    queue_root.mkdir(parents=True, exist_ok=True)
    queue_params = json.dumps(
        {
            "data_jsonl": str(Path(args.data_jsonl).expanduser().resolve()),
            "source_manifest": str(Path(args.source_manifest).expanduser().resolve()),
            "init_checkpoint": str(Path(args.init_checkpoint).expanduser().resolve()),
            "train_device": str(args.train_device),
            "eval_device": str(args.eval_device),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "hidden_dim": int(args.hidden_dim),
            "policy_hidden_dim": int(args.policy_hidden_dim),
            "token_dim": int(args.token_dim),
            "num_slots": int(args.num_slots),
            "num_encoder_layers": int(args.num_encoder_layers),
            "num_heads": int(args.num_heads),
            "num_experts": int(args.num_experts),
            "router_hidden_dim": int(args.router_hidden_dim),
            "router_temperature": float(args.router_temperature),
            "action_loss_weight": float(args.action_loss_weight),
            "gain_loss_weight": float(args.gain_loss_weight),
            "policy_loss_weight": float(args.policy_loss_weight),
            "router_balance_loss_weight": float(args.router_balance_loss_weight),
            "expert_aux_loss_weight": float(args.expert_aux_loss_weight),
            "rank_loss_weight": float(args.rank_loss_weight),
            "rank_margin": float(args.rank_margin),
            "ordinal_rank_loss_weight": float(args.ordinal_rank_loss_weight),
            "ordinal_rank_margin": float(args.ordinal_rank_margin),
            "train_selection_head_only": bool(args.train_selection_head_only),
            "selection_metric": str(args.selection_metric),
            "seq_ids": [int(v) for v in args.seq_ids],
            "max_hours": float(args.max_hours),
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    eval_specs = [
        ("01_posminus_t003", "positive_minus_neutral", 0.0, 0.03, "compare the current best rule on the new calibrated checkpoint"),
        ("02_policy_t000", "policy_score", 0.0, 0.0, "use the learned calibrated policy score directly"),
        ("03_policy_t003", "policy_score", 0.0, 0.03, "policy score with the old conservative threshold"),
    ]

    rows: List[Dict[str, object]] = read_csv_rows(summary_csv) if bool(args.resume_existing) and summary_csv.is_file() else []
    if rows:
        queue_row = find_row(rows, "queue")
        if queue_row is None:
            append_row(
                rows,
                step="queue",
                name=str(args.queue_name),
                status="running",
                run_root=queue_root,
                summary_csv=summary_csv,
                log_path=queue_log,
                notes="calibrated residual commit queue resumed",
                params_json=queue_params,
            )
        else:
            update_row(
                rows,
                "queue",
                status="running",
                run_root=queue_root,
                summary_csv=summary_csv,
                log_path=queue_log,
                notes="calibrated residual commit queue resumed",
                params_json=queue_params,
            )

        if find_row(rows, "train_policy") is None:
            append_row(
                rows,
                step="train_policy",
                name="train_graph_assoc_commit_policy_calibrated_residual",
                status="pending",
                run_root=queue_root,
                summary_csv=train_dir / "summary.csv",
                log_path=logs_dir / "train_policy.log",
                checkpoint=str(train_dir / "best.pt"),
                notes="train calibrated residual commit policy",
                params_json=queue_params,
            )
        for step, decision_mode, score_margin, threshold, notes in eval_specs:
            if find_row(rows, step) is None:
                append_row(
                    rows,
                    step=step,
                    name=f"eval_{decision_mode}_{threshold:.2f}",
                    status="pending",
                    run_root=eval_root / step,
                    summary_csv=eval_root / step / "summary.csv",
                    log_path=logs_dir / f"{step}.log",
                    checkpoint=str(train_dir / "best.pt"),
                    decision_mode=decision_mode,
                    score_margin=f"{score_margin}",
                    notes=notes,
                    params_json=json.dumps(
                        {
                            "decision_mode": decision_mode,
                            "score_margin": score_margin,
                            "threshold": threshold,
                        },
                        ensure_ascii=False,
                    ),
                )
    else:
        append_row(
            rows,
            step="queue",
            name=str(args.queue_name),
            status="running",
            run_root=queue_root,
            summary_csv=summary_csv,
            log_path=queue_log,
            notes="calibrated residual commit queue started",
            params_json=queue_params,
        )
        append_row(
            rows,
            step="train_policy",
            name="train_graph_assoc_commit_policy_calibrated_residual",
            status="pending",
            run_root=queue_root,
            summary_csv=train_dir / "summary.csv",
            log_path=logs_dir / "train_policy.log",
            checkpoint=str(train_dir / "best.pt"),
            notes="train calibrated residual commit policy",
            params_json=queue_params,
        )
        for step, decision_mode, score_margin, threshold, notes in eval_specs:
            append_row(
                rows,
                step=step,
                name=f"eval_{decision_mode}_{threshold:.2f}",
                status="pending",
                run_root=eval_root / step,
                summary_csv=eval_root / step / "summary.csv",
                log_path=logs_dir / f"{step}.log",
                checkpoint=str(train_dir / "best.pt"),
                decision_mode=decision_mode,
                score_margin=f"{score_margin}",
                notes=notes,
                params_json=json.dumps(
                    {
                        "decision_mode": decision_mode,
                        "score_margin": score_margin,
                        "threshold": threshold,
                    },
                    ensure_ascii=False,
                ),
            )

    write_rows(summary_csv, QUEUE_FIELDS, rows)
    queue_plan_status(args, "running", summary_csv, queue_log, notes="calibrated residual commit queue started")
    queue_registry(args, "running", summary_csv, queue_log, notes="calibrated residual commit queue started")

    try:
        train_row = find_row(rows, "train_policy")
        best_ckpt = train_dir / "best.pt"
        train_summary: Dict[str, object]
        train_done = (
            bool(args.resume_existing)
            and train_row is not None
            and str(train_row.get("status", "")).strip() == "success"
            and best_ckpt.is_file()
        )
        if train_done:
            train_summary = parse_train_summary(train_dir / "summary.csv")
        else:
            if datetime.now().timestamp() >= deadline:
                raise TimeoutError("Queue deadline reached before training started.")
            update_row(rows, "train_policy", status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            train_command = train_cmd(args, train_dir)
            rc = run_step(train_command, logs_dir / "train_policy.log")
            if rc != 0:
                raise RuntimeError(f"train_graph_assoc_commit_policy failed with return code {rc}")
            train_summary = parse_train_summary(train_dir / "summary.csv")
            update_row(
                rows,
                "train_policy",
                status="success",
                finished_at=now_iso(),
                checkpoint=str(best_ckpt),
                best_epoch=str(train_summary["best_epoch"]),
                best_metric=str(train_summary["best_metric"]),
                best_threshold=str(train_summary["best_threshold"]),
                notes=(
                    f"training complete, best_epoch={train_summary['best_epoch']} "
                    f"best_metric={train_summary['best_metric']:.4f} best_threshold={train_summary['best_threshold']:.3f}"
                ),
            )
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            queue_plan_status(
                args,
                "running",
                summary_csv,
                queue_log,
                notes=f"training complete, best_epoch={train_summary['best_epoch']} best_metric={train_summary['best_metric']:.4f}",
            )
            queue_registry(
                args,
                "running",
                summary_csv,
                queue_log,
                notes=f"training complete, best_epoch={train_summary['best_epoch']} best_metric={train_summary['best_metric']:.4f}",
            )

        eval_results: List[Dict[str, object]] = []
        for step, decision_mode, score_margin, threshold, _notes in eval_specs:
            step_row = find_row(rows, step)
            if bool(args.resume_existing) and step_row is not None and str(step_row.get("status", "")).strip() == "success":
                continue
            if datetime.now().timestamp() >= deadline:
                raise TimeoutError("Queue deadline reached before evaluation completed.")
            update_row(rows, step, status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            child_root = eval_root / step
            experiment_name = f"{args.queue_name}_{step}"
            rc = run_step(
                eval_cmd(
                    args,
                    run_root=child_root,
                    experiment_name=experiment_name,
                    checkpoint=best_ckpt,
                    decision_mode=decision_mode,
                    score_margin=score_margin,
                    threshold=threshold,
                ),
                logs_dir / f"{step}.log",
            )
            if rc != 0:
                raise RuntimeError(f"run_botsort_graphassoc_mot20_eval failed for {step} with return code {rc}")
            metrics = parse_metrics_delta(child_root)
            identical_count = parse_identical_count(child_root)
            result = {
                "step": step,
                "decision_mode": decision_mode,
                "score_margin": score_margin,
                "threshold": threshold,
                "run_root": str(child_root),
                "checkpoint": str(best_ckpt),
                "identical_count": identical_count,
                **metrics,
            }
            eval_results.append(result)
            update_row(
                rows,
                step,
                status="success",
                finished_at=now_iso(),
                checkpoint=str(best_ckpt),
                delta_hota=str(metrics["delta_hota"]),
                delta_assa=str(metrics["delta_assa"]),
                delta_idf1=str(metrics["delta_idf1"]),
                delta_mota=str(metrics["delta_mota"]),
                delta_ids=str(metrics["delta_ids"]),
                delta_frag=str(metrics["delta_frag"]),
                identical_count=str(identical_count),
                notes=(
                    f"{decision_mode} threshold={threshold:.3f} complete, "
                    f"delta_HOTA={metrics['delta_hota']:.3f} delta_IDF1={metrics['delta_idf1']:.3f}"
                ),
            )
            write_rows(summary_csv, QUEUE_FIELDS, rows)

        ranked = rank_results(eval_results)
        best = ranked[0] if ranked else None
        if best is not None:
            append_log_line(
                queue_log,
                "best_eval="
                + str(best["step"])
                + f" decision_mode={best['decision_mode']}"
                + f" delta_HOTA={float(best['delta_hota']):.6f}"
                + f" delta_AssA={float(best['delta_assa']):.6f}"
                + f" delta_IDF1={float(best['delta_idf1']):.6f}",
            )

        final_note = (
            f"calibrated residual commit queue completed, best_eval={best['step']} "
            f"delta_HOTA={float(best['delta_hota']):.3f}" if best is not None else "calibrated residual commit queue completed"
        )
        update_row(rows, "queue", status="completed", finished_at=now_iso(), notes=final_note)
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        queue_plan_status(args, "completed", summary_csv, queue_log, notes=final_note)
        queue_registry(args, "success", summary_csv, queue_log, notes=final_note)
        return 0
    except Exception as exc:
        finished_at = now_iso()
        for row in rows:
            status = str(row.get("status", "")).strip().lower()
            if status == "running":
                row["status"] = "failed"
                row["finished_at"] = finished_at
                row["notes"] = f"{row.get('notes', '')} | failed: {exc}".strip()
            elif status == "pending":
                row["status"] = "cancelled"
                row["finished_at"] = finished_at
                row["notes"] = f"{row.get('notes', '')} | cancelled_after_failure".strip()
        update_row(rows, "queue", status="failed", finished_at=finished_at, notes=str(exc))
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        queue_plan_status(args, "failed", summary_csv, queue_log, notes=str(exc))
        queue_registry(args, "failed", summary_csv, queue_log, notes=str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
