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
BOT_ROOT = REPO_ROOT / "external" / "BoT-SORT-main"
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
PLAN_CSV = REPO_ROOT / "outputs" / "experiment_plan.csv"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "run_botsort_graphassoc_mot20_eval.py"

DEFAULT_GATE_CKPT = (
    REPO_ROOT
    / "outputs"
    / "graphassoc_gate_trainonly_mix_rankv2_decsurf_all5_hn050w3_20260429_1146"
    / "train_gate"
    / "best.pt"
)
DEFAULT_REFERENCE_RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "graphassoc_gate_mix_rankv2_decsurf_all5_hn050w3_evalqueued_20260429_1150"
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
    "seq_ids",
    "HOTA",
    "AssA",
    "IDF1",
    "MOTA",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDs",
    "delta_Frag",
    "changed_blocks",
    "forced_matches",
    "accepted_candidates",
    "rejected_candidates",
    "notes",
    "params_json",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Run a narrow calibration sweep for graph-association gate on MOT20 val_half.")
    parser.add_argument("--run-root", default=str(REPO_ROOT / "outputs" / f"graphassoc_gate_calibration_next5h_{ts}"))
    parser.add_argument("--queue-name", default=f"graphassoc_gate_calibration_next5h_{ts}")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--gate-checkpoint", default=str(DEFAULT_GATE_CKPT))
    parser.add_argument("--reference-run-root", default=str(DEFAULT_REFERENCE_RUN_ROOT))
    parser.add_argument("--seq-ids", nargs="+", type=int, default=[2, 5])
    parser.add_argument("--commit-device", default="cuda")
    parser.add_argument(
        "--start-from-step",
        type=int,
        default=1,
        help="1-based variant index to start from; use this to resume only the remaining calibration points.",
    )
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


def update_row(rows: List[Dict[str, object]], step: str, **updates: object) -> None:
    for row in rows:
        if str(row.get("step", "")) == str(step):
            row.update(updates)
            return
    raise KeyError(f"Missing queue step: {step}")


def find_row(rows: List[Dict[str, object]], step: str) -> Dict[str, object] | None:
    for row in rows:
        if str(row.get("step", "")) == str(step):
            return row
    return None


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
        "scripts/run_graphassoc_gate_calibration_next5h.py",
        "--dataset",
        "MOT20",
        "--split",
        "val_half",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        "graphassoc_gate_calibration",
        "--tag",
        args.queue_name,
        "--run-root",
        str(Path(args.run_root).expanduser().resolve()),
        "--summary-csv",
        str(summary_csv),
        "--log-path",
        str(log_path),
        "--checkpoint",
        str(Path(args.gate_checkpoint).expanduser().resolve()),
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
        "scripts/run_graphassoc_gate_calibration_next5h.py",
        "--dataset",
        "MOT20",
        "--split",
        "val_half",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        "graphassoc_gate_calibration",
        "--tag",
        args.queue_name,
        "--run-root",
        str(Path(args.run_root).expanduser().resolve()),
        "--summary-csv",
        str(summary_csv),
        "--checkpoint",
        str(Path(args.gate_checkpoint).expanduser().resolve()),
        "--log-path",
        str(log_path),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


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


def parse_metrics_compare(run_root: Path) -> Dict[str, object]:
    rows = read_csv_rows(run_root / "metrics_compare.csv")
    for row in rows:
        if str(row.get("name", "")) == "graph_assoc":
            return {
                "HOTA": float(row.get("HOTA", 0.0) or 0.0),
                "AssA": float(row.get("AssA", 0.0) or 0.0),
                "IDF1": float(row.get("IDF1", 0.0) or 0.0),
                "MOTA": float(row.get("MOTA", 0.0) or 0.0),
            }
    if rows:
        row = rows[-1]
        return {
            "HOTA": float(row.get("HOTA", 0.0) or 0.0),
            "AssA": float(row.get("AssA", 0.0) or 0.0),
            "IDF1": float(row.get("IDF1", 0.0) or 0.0),
            "MOTA": float(row.get("MOTA", 0.0) or 0.0),
        }
    raise FileNotFoundError(f"missing metrics_compare.csv under {run_root}")


def parse_metrics_delta(run_root: Path) -> Dict[str, object]:
    rows = read_csv_rows(run_root / "metrics_delta.csv")
    if not rows:
        raise FileNotFoundError(f"missing metrics_delta.csv under {run_root}")
    row = rows[0]
    return {
        "delta_HOTA": float(row.get("delta_HOTA", 0.0) or 0.0),
        "delta_AssA": float(row.get("delta_AssA", 0.0) or 0.0),
        "delta_IDF1": float(row.get("delta_IDF1", 0.0) or 0.0),
        "delta_MOTA": float(row.get("delta_MOTA", 0.0) or 0.0),
        "delta_IDs": int(round(float(row.get("delta_IDs", 0.0) or 0.0))),
        "delta_Frag": int(round(float(row.get("delta_Frag", 0.0) or 0.0))),
    }


def parse_runtime_compare(run_root: Path) -> Dict[str, object]:
    rows = read_csv_rows(run_root / "runtime_compare.csv")
    if not rows:
        raise FileNotFoundError(f"missing runtime_compare.csv under {run_root}")
    row = rows[-1]
    return {
        "changed_blocks": int(float(row.get("changed_blocks", 0) or 0)),
        "forced_matches": int(float(row.get("forced_matches", 0) or 0)),
        "accepted_candidates": int(float(row.get("learned_commit_margin_accept_count", 0) or 0)),
        "rejected_candidates": int(float(row.get("learned_commit_margin_reject_count", 0) or 0)),
    }


def build_eval_command(args: argparse.Namespace, run_root: Path, exp_name: str, threshold: float, neutral_risk_weight: float) -> List[str]:
    gate_checkpoint = str(Path(args.gate_checkpoint).expanduser().resolve())
    common_graph_args = [
        "--graph-assoc-top-k",
        "3",
        "--graph-assoc-max-rows",
        "4",
        "--graph-assoc-max-cols",
        "4",
        "--graph-assoc-row-margin",
        "0.03",
        "--graph-assoc-col-margin",
        "0.03",
        "--graph-assoc-min-reclaim-time-since-update",
        "1",
        "--graph-assoc-max-reclaim-time-since-update",
        "8",
        "--graph-assoc-min-reclaim-tracklet-len",
        "20",
        "--graph-assoc-recent-owner-max-time-since-update",
        "1",
        "--graph-assoc-recent-owner-max-tracklet-len",
        "8",
        "--graph-assoc-young-active-max-time-since-update",
        "1",
        "--graph-assoc-young-active-max-tracklet-len",
        "20",
        "--graph-assoc-young-active-min-reclaim-gap",
        "2",
        "--graph-assoc-young-active-max-cost-delta",
        "-1.0",
        "--graph-assoc-stale-lost-owner-min-time-since-update",
        "9",
        "--graph-assoc-stale-lost-owner-min-tracklet-len",
        "100",
        "--graph-assoc-stale-lost-owner-active-max-time-since-update",
        "1",
        "--graph-assoc-stale-lost-owner-min-introduced-edge-utility",
        "0.0",
        "--graph-assoc-min-box-iou",
        "0.6",
        "--graph-assoc-reclaim-bonus",
        "0.08",
        "--graph-assoc-recent-owner-penalty",
        "0.05",
        "--graph-assoc-iou-bonus",
        "0.04",
        "--graph-assoc-score-bonus",
        "0.02",
        "--graph-assoc-min-assignment-gain",
        "0.01",
        "--graph-assoc-max-cost-delta",
        "0.05",
        "--graph-assoc-row-involved-min-assignment-gain",
        "0.01",
        "--graph-assoc-col-only-min-assignment-gain",
        "0.01",
        "--graph-assoc-col-only-max-cost-delta",
        "0.05",
        "--graph-assoc-force-match-cost",
        "0.0",
        "--graph-assoc-commit-checkpoint",
        gate_checkpoint,
        "--graph-assoc-commit-device",
        str(args.commit_device),
        "--graph-assoc-commit-score-margin",
        "0.0",
        "--graph-assoc-commit-replace-rules",
        "--graph-assoc-commit-decision-mode",
        "positive_minus_weighted_neutral",
        "--graph-assoc-commit-threshold",
        f"{threshold}",
        "--graph-assoc-commit-neutral-risk-weight",
        f"{neutral_risk_weight}",
    ]
    return [
        args.python_bin,
        str(EVAL_SCRIPT),
        "--run-root",
        str(run_root),
        "--experiment-name",
        exp_name,
        "--variant-name",
        "botsort_graphassoc_gate_calibration",
        "--seq-ids",
        *[str(v) for v in args.seq_ids],
        *common_graph_args,
    ]


def variant_catalog() -> List[Dict[str, object]]:
    thresholds = [0.18, 0.22, 0.26]
    risk_weights = [1.25, 1.4]
    variants: List[Dict[str, object]] = []
    for threshold in thresholds:
        for risk_weight in risk_weights:
            variants.append(
                {
                    "name": f"pmn_t{int(round(threshold * 1000)):03d}_rw{int(round(risk_weight * 100)):03d}",
                    "threshold": threshold,
                    "neutral_risk_weight": risk_weight,
                    "notes": "narrow calibration sweep around the current best gate operating point",
                }
            )
    return variants


def append_reference_row(args: argparse.Namespace, rows: List[Dict[str, object]], summary_csv: Path) -> None:
    if find_row(rows, "reference_current_best") is not None:
        return
    ref_root = Path(args.reference_run_root).expanduser().resolve()
    metrics = parse_metrics_compare(ref_root)
    delta = parse_metrics_delta(ref_root)
    runtime = parse_runtime_compare(ref_root)
    rows.append(
        {
            "step": "reference_current_best",
            "name": ref_root.name,
            "status": "reference",
            "run_root": str(ref_root),
            "summary_csv": str(ref_root / "summary.csv"),
            "log_path": str(ref_root / "logs" / "compare.log"),
            "started_at": "",
            "finished_at": "",
            "seq_ids": "|".join(str(v) for v in args.seq_ids),
            **metrics,
            **delta,
            **runtime,
            "notes": "current best graph-assoc checkpoint used as calibration reference",
            "params_json": json.dumps(
                {
                    "reference_run_root": str(ref_root),
                    "gate_checkpoint": str(Path(args.gate_checkpoint).expanduser().resolve()),
                },
                ensure_ascii=False,
            ),
        }
    )
    write_rows(summary_csv, QUEUE_FIELDS, rows)


def main() -> None:
    args = parse_args()
    queue_root = Path(args.run_root).expanduser().resolve()
    logs_dir = queue_root / "logs"
    summary_csv = queue_root / "summary.csv"
    queue_log = logs_dir / "queue.log"
    rows: List[Dict[str, object]] = [dict(row) for row in read_csv_rows(summary_csv)]

    queue_root.mkdir(parents=True, exist_ok=True)
    queue_plan_status(args, "running", summary_csv, queue_log, notes="graphassoc calibration queue started")
    queue_registry(args, "running", summary_csv, queue_log, notes="graphassoc calibration queue started")
    append_reference_row(args, rows, summary_csv)

    any_failed = False
    try:
        for idx, variant in enumerate(variant_catalog(), start=1):
            if idx < args.start_from_step:
                continue
            step = f"{idx:02d}_{variant['name']}"
            child_root = queue_root / "runs" / step
            child_log = logs_dir / f"{step}.log"
            exp_name = f"{args.queue_name}_{step}"
            if find_row(rows, step) is None:
                rows.append(
                    {
                        "step": step,
                        "name": variant["name"],
                        "status": "pending",
                        "run_root": str(child_root),
                        "summary_csv": str(child_root / "summary.csv"),
                        "log_path": str(child_log),
                        "started_at": "",
                        "finished_at": "",
                        "seq_ids": "|".join(str(v) for v in args.seq_ids),
                        "HOTA": "",
                        "AssA": "",
                        "IDF1": "",
                        "MOTA": "",
                        "delta_HOTA": "",
                        "delta_AssA": "",
                        "delta_IDF1": "",
                        "delta_MOTA": "",
                        "delta_IDs": "",
                        "delta_Frag": "",
                        "changed_blocks": "",
                        "forced_matches": "",
                        "accepted_candidates": "",
                        "rejected_candidates": "",
                        "notes": variant["notes"],
                        "params_json": json.dumps(
                            {
                                "threshold": variant["threshold"],
                                "neutral_risk_weight": variant["neutral_risk_weight"],
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
            if (child_root / "metrics_delta.csv").is_file():
                metrics = parse_metrics_compare(child_root)
                delta = parse_metrics_delta(child_root)
                runtime = parse_runtime_compare(child_root)
                existing = find_row(rows, step) or {}
                update_row(
                    rows,
                    step,
                    status="success",
                    started_at=existing.get("started_at", ""),
                    finished_at=existing.get("finished_at", "") or now_iso(),
                    **metrics,
                    **delta,
                    **runtime,
                    notes=f"{variant['notes']} | complete (cached)",
                )
                write_rows(summary_csv, QUEUE_FIELDS, rows)
                continue

            update_row(rows, step, status="running", started_at=now_iso(), notes=variant["notes"])
            write_rows(summary_csv, QUEUE_FIELDS, rows)

            cmd = build_eval_command(
                args=args,
                run_root=child_root,
                exp_name=exp_name,
                threshold=float(variant["threshold"]),
                neutral_risk_weight=float(variant["neutral_risk_weight"]),
            )
            rc = run_step(cmd, child_log)
            if rc != 0:
                any_failed = True
                update_row(
                    rows,
                    step,
                    status="failed",
                    finished_at=now_iso(),
                    notes=f"{variant['notes']} | return_code={rc}",
                )
                write_rows(summary_csv, QUEUE_FIELDS, rows)
                continue

            metrics = parse_metrics_compare(child_root)
            delta = parse_metrics_delta(child_root)
            runtime = parse_runtime_compare(child_root)
            update_row(
                rows,
                step,
                status="success",
                finished_at=now_iso(),
                **metrics,
                **delta,
                **runtime,
                notes=f"{variant['notes']} | complete",
            )
            write_rows(summary_csv, QUEUE_FIELDS, rows)

        final_status = "completed"
        final_notes = "graphassoc calibration queue completed"
        if any_failed:
            final_notes = "graphassoc calibration queue completed with one or more failed child runs"
        queue_plan_status(args, final_status, summary_csv, queue_log, notes=final_notes)
        queue_registry(args, "success", summary_csv, queue_log, notes=final_notes)
    except Exception as exc:
        for row in rows:
            if str(row.get("status", "")) in {"pending", "running"}:
                row["status"] = "failed"
                row["finished_at"] = now_iso()
                row["notes"] = f"{row.get('notes', '')} | queue_exception={exc}".strip()
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        queue_plan_status(args, "failed", summary_csv, queue_log, notes=str(exc))
        queue_registry(args, "failed", summary_csv, queue_log, notes=str(exc))
        raise


if __name__ == "__main__":
    main()
