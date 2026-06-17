#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
PLAN_CSV = REPO_ROOT / "outputs" / "experiment_plan.csv"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "run_botsort_graphassoc_mot20_eval.py"
DEFAULT_GATE_CKPT = REPO_ROOT / "outputs" / "graph_assoc_gate_stage7_dualhead_full_pw1_np1_nl05_rw125_ord20_rm012_20260420_1" / "best.pt"

QUEUE_FIELDS = [
    "step",
    "name",
    "status",
    "run_root",
    "log_path",
    "started_at",
    "finished_at",
    "seq_ids",
    "delta_hota",
    "delta_assa",
    "delta_idf1",
    "delta_mota",
    "delta_ids",
    "delta_frag",
    "notes",
    "params_json",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Run the next five hours of dual-head graph-association experiments.")
    parser.add_argument("--run-root", default=str(REPO_ROOT / "outputs" / f"graphassoc_dualhead_next5h_{ts}"))
    parser.add_argument("--queue-name", default=f"graphassoc_dualhead_next5h_{ts}")
    parser.add_argument(
        "--wait-summary-csv",
        default=str(
            REPO_ROOT
            / "outputs"
            / "botsort_graphassoc_gate_stage7_dualhead_runtime_seq5_diff10_margin037_20260420_1"
            / "summary.csv"
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--max-hours", type=float, default=5.0)
    parser.add_argument("--single-head-best-delta", type=float, default=-0.008)
    parser.add_argument("--gate-checkpoint", default=str(DEFAULT_GATE_CKPT))
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
        if str(row.get("step")) == str(step):
            row.update(updates)
            return
    raise KeyError(f"Missing queue step: {step}")


def append_row(
    rows: List[Dict[str, object]],
    step: str,
    name: str,
    run_root: Path | str,
    log_path: Path | str,
    seq_ids: List[int],
    params_json: str,
    notes: str = "",
) -> None:
    rows.append(
        {
            "step": step,
            "name": name,
            "status": "pending",
            "run_root": str(run_root),
            "log_path": str(log_path),
            "started_at": "",
            "finished_at": "",
            "seq_ids": "|".join(str(v) for v in seq_ids),
            "delta_hota": "",
            "delta_assa": "",
            "delta_idf1": "",
            "delta_mota": "",
            "delta_ids": "",
            "delta_frag": "",
            "notes": notes,
            "params_json": params_json,
        }
    )


def summary_complete(path: Path) -> bool:
    rows = read_csv_rows(path)
    if not rows:
        return False
    active = {"running", "pending", "queued"}
    return all(str(row.get("status", "")).strip().lower() not in active for row in rows)


def wait_for_summary(path: Path, poll_seconds: int, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[wait_start] {now_iso()} waiting for {path}\n")
        handle.flush()
        while not summary_complete(path):
            handle.write(f"[wait_poll] {now_iso()} summary not finished yet\n")
            handle.flush()
            time.sleep(max(5, int(poll_seconds)))
        handle.write(f"[wait_done] {now_iso()} summary finished\n")


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
        "scripts/run_graphassoc_dualhead_next5h.py",
        "--dataset",
        "MOT20",
        "--split",
        "val_half",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        "graphassoc_dualhead_next5h",
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
        "scripts/run_graphassoc_dualhead_next5h.py",
        "--dataset",
        "MOT20",
        "--split",
        "val_half",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        "graphassoc_dualhead_next5h",
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


def parse_metrics_delta(run_root: Path) -> Dict[str, float | int]:
    rows = read_csv_rows(run_root / "metrics_delta.csv")
    if not rows:
        return {
            "delta_hota": -999.0,
            "delta_assa": -999.0,
            "delta_idf1": -999.0,
            "delta_mota": -999.0,
            "delta_ids": 0,
            "delta_frag": 0,
        }
    row = rows[0]
    return {
        "delta_hota": float(row.get("delta_HOTA", 0.0) or 0.0),
        "delta_assa": float(row.get("delta_AssA", 0.0) or 0.0),
        "delta_idf1": float(row.get("delta_IDF1", 0.0) or 0.0),
        "delta_mota": float(row.get("delta_MOTA", 0.0) or 0.0),
        "delta_ids": int(round(float(row.get("delta_IDs", 0.0) or 0.0))),
        "delta_frag": int(round(float(row.get("delta_Frag", 0.0) or 0.0))),
    }


def make_variant(name: str, seq_ids: List[int], extra_args: List[str], notes: str) -> Dict[str, object]:
    return {
        "name": name,
        "seq_ids": list(seq_ids),
        "extra_args": list(extra_args),
        "notes": notes,
    }


def build_common_args(checkpoint_path: str) -> List[str]:
    return [
        "--graph-assoc-top-k", "3",
        "--graph-assoc-max-rows", "4",
        "--graph-assoc-max-cols", "4",
        "--graph-assoc-row-margin", "0.05",
        "--graph-assoc-col-margin", "0.05",
        "--graph-assoc-min-reclaim-time-since-update", "1",
        "--graph-assoc-max-reclaim-time-since-update", "8",
        "--graph-assoc-min-reclaim-tracklet-len", "15",
        "--graph-assoc-recent-owner-max-time-since-update", "1",
        "--graph-assoc-recent-owner-max-tracklet-len", "12",
        "--graph-assoc-young-active-max-time-since-update", "1",
        "--graph-assoc-young-active-max-tracklet-len", "20",
        "--graph-assoc-young-active-min-reclaim-gap", "2",
        "--graph-assoc-young-active-max-cost-delta", "-1.0",
        "--graph-assoc-stale-lost-owner-min-time-since-update", "9",
        "--graph-assoc-stale-lost-owner-min-tracklet-len", "100",
        "--graph-assoc-stale-lost-owner-active-max-time-since-update", "1",
        "--graph-assoc-stale-lost-owner-min-introduced-edge-utility", "0.0",
        "--graph-assoc-min-box-iou", "0.55",
        "--graph-assoc-reclaim-bonus", "0.15",
        "--graph-assoc-recent-owner-penalty", "0.08",
        "--graph-assoc-iou-bonus", "0.05",
        "--graph-assoc-score-bonus", "0.02",
        "--graph-assoc-min-assignment-gain", "0.005",
        "--graph-assoc-max-cost-delta", "0.08",
        "--graph-assoc-row-involved-min-assignment-gain", "0.005",
        "--graph-assoc-col-only-min-assignment-gain", "0.005",
        "--graph-assoc-col-only-max-cost-delta", "0.08",
        "--graph-assoc-force-match-cost", "0.0",
        "--graph-assoc-commit-checkpoint", str(Path(checkpoint_path).expanduser().resolve()),
        "--graph-assoc-commit-device", "cpu",
        "--graph-assoc-commit-gate-only",
        "--graph-assoc-dump-candidate-rows",
    ]


def build_eval_command(args: argparse.Namespace, run_root: Path, exp_name: str, seq_ids: List[int], extra_args: List[str]) -> List[str]:
    return [
        args.python_bin,
        str(EVAL_SCRIPT),
        "--run-root",
        str(run_root),
        "--experiment-name",
        exp_name,
        "--variant-name",
        "botsort_graphassoc_dualhead_runtime",
        "--seq-ids",
        *[str(v) for v in seq_ids],
        *extra_args,
    ]


def run_variant(
    args: argparse.Namespace,
    rows: List[Dict[str, object]],
    summary_csv: Path,
    logs_dir: Path,
    queue_root: Path,
    step: str,
    variant: Dict[str, object],
) -> Optional[Dict[str, object]]:
    child_run_root = queue_root / "runs" / step
    child_name = f"{args.queue_name}_{step}"
    child_log = logs_dir / f"{step}.log"
    append_row(
        rows,
        step,
        str(variant["name"]),
        child_run_root,
        child_log,
        list(variant["seq_ids"]),
        json.dumps({"extra_args": variant["extra_args"]}, ensure_ascii=False),
        notes=str(variant["notes"]),
    )
    update_row(rows, step, status="running", started_at=now_iso())
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    cmd = build_eval_command(args, child_run_root, child_name, list(variant["seq_ids"]), list(variant["extra_args"]))
    rc = run_step(cmd, child_log)
    if rc != 0:
        update_row(rows, step, status="failed", finished_at=now_iso(), notes=f"{variant['notes']} | return_code={rc}")
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        return None

    metrics = parse_metrics_delta(child_run_root)
    update_row(
        rows,
        step,
        status="success",
        finished_at=now_iso(),
        delta_hota=metrics["delta_hota"],
        delta_assa=metrics["delta_assa"],
        delta_idf1=metrics["delta_idf1"],
        delta_mota=metrics["delta_mota"],
        delta_ids=metrics["delta_ids"],
        delta_frag=metrics["delta_frag"],
        notes=f"{variant['notes']} | delta_HOTA={metrics['delta_hota']:.3f}",
    )
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    result = dict(variant)
    result.update(metrics)
    result["run_root"] = str(child_run_root)
    return result


def better_result(candidate: Optional[Dict[str, object]], incumbent: Optional[Dict[str, object]]) -> bool:
    if candidate is None:
        return False
    if incumbent is None:
        return True
    return (
        float(candidate.get("delta_hota", -999.0)),
        float(candidate.get("delta_idf1", -999.0)),
        float(candidate.get("delta_assa", -999.0)),
    ) > (
        float(incumbent.get("delta_hota", -999.0)),
        float(incumbent.get("delta_idf1", -999.0)),
        float(incumbent.get("delta_assa", -999.0)),
    )


def should_promote_to_seq25(result: Optional[Dict[str, object]], single_head_best_delta: float) -> bool:
    if result is None:
        return False
    delta = float(result.get("delta_hota", -999.0))
    return delta >= 0.0 or delta > float(single_head_best_delta)


def main() -> None:
    args = parse_args()
    queue_root = Path(args.run_root).expanduser().resolve()
    logs_dir = queue_root / "logs"
    summary_csv = queue_root / "summary.csv"
    queue_log = logs_dir / "queue.log"
    rows: List[Dict[str, object]] = []
    deadline = time.time() + max(0.5, float(args.max_hours)) * 3600.0

    queue_plan_status(args, "running", summary_csv, queue_log, notes="dual-head graph-association 5h queue started")
    queue_registry(args, "running", summary_csv, queue_log, notes="dual-head graph-association 5h queue started")

    current_variant = make_variant(
        "seq5_diff10_margin037",
        [5],
        build_common_args(args.gate_checkpoint)
        + [
            "--graph-assoc-commit-score-margin", "0.37",
            "--graph-assoc-commit-decision-mode", "positive_minus_neutral",
            "--graph-assoc-commit-threshold", "-0.28",
            "--graph-assoc-commit-neutral-risk-weight", "1.0",
        ],
        notes="current running dual-head diff10 runtime, effective acceptance threshold is 0.09",
    )

    followup_variants = [
        make_variant(
            "seq5_diff10_margin034",
            [5],
            build_common_args(args.gate_checkpoint)
            + [
                "--graph-assoc-commit-score-margin", "0.34",
                "--graph-assoc-commit-decision-mode", "positive_minus_neutral",
                "--graph-assoc-commit-threshold", "-0.28",
                "--graph-assoc-commit-neutral-risk-weight", "1.0",
            ],
            notes="same diff10 rule with a slightly looser margin to recover borderline positive cases",
        ),
        make_variant(
            "seq5_dualthr_g025_n054",
            [5],
            build_common_args(args.gate_checkpoint)
            + [
                "--graph-assoc-commit-score-margin", "0.00",
                "--graph-assoc-commit-decision-mode", "dual_threshold",
                "--graph-assoc-commit-positive-threshold", "0.25",
                "--graph-assoc-commit-neutral-threshold", "0.54",
            ],
            notes="dual-threshold gate that only commits when gain is high enough and neutral risk is low enough",
        ),
    ]

    best_seq5: Optional[Dict[str, object]] = None

    try:
        wait_summary_csv = Path(args.wait_summary_csv).expanduser().resolve()
        append_row(
            rows,
            "wait_current",
            "wait_current_seq5_runtime",
            wait_summary_csv.parent,
            queue_log,
            [5],
            json.dumps({"summary_csv": str(wait_summary_csv)}, ensure_ascii=False),
            notes=f"wait for current runtime result: {wait_summary_csv}",
        )
        update_row(rows, "wait_current", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        wait_for_summary(wait_summary_csv, int(args.poll_seconds), queue_log)

        current_metrics = parse_metrics_delta(wait_summary_csv.parent)
        update_row(
            rows,
            "wait_current",
            status="success",
            finished_at=now_iso(),
            delta_hota=current_metrics["delta_hota"],
            delta_assa=current_metrics["delta_assa"],
            delta_idf1=current_metrics["delta_idf1"],
            delta_mota=current_metrics["delta_mota"],
            delta_ids=current_metrics["delta_ids"],
            delta_frag=current_metrics["delta_frag"],
            notes=f"current runtime finished | delta_HOTA={current_metrics['delta_hota']:.3f}",
        )
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        best_seq5 = dict(current_variant)
        best_seq5.update(current_metrics)
        best_seq5["run_root"] = str(wait_summary_csv.parent)

        candidate_variants = list(followup_variants)
        if float(current_metrics["delta_hota"]) <= -900.0:
            candidate_variants = [current_variant] + candidate_variants

        for idx, variant in enumerate(candidate_variants, start=1):
            if time.time() >= deadline:
                break
            if best_seq5 is not None and float(best_seq5.get("delta_hota", -999.0)) >= 0.0:
                break
            step = f"seq5_{idx}_{variant['name']}"
            result = run_variant(args, rows, summary_csv, logs_dir, queue_root, step, variant)
            if better_result(result, best_seq5):
                best_seq5 = result

        if should_promote_to_seq25(best_seq5, float(args.single_head_best_delta)) and time.time() < deadline:
            assert best_seq5 is not None
            promote_variant = make_variant(
                f"seq25_confirm_{best_seq5['name']}",
                [2, 5],
                list(best_seq5["extra_args"]),
                notes="promote the best seq5 dual-head decision rule to seq2+5 confirmation",
            )
            run_variant(args, rows, summary_csv, logs_dir, queue_root, "seq25_confirm", promote_variant)

        final_note = "queue completed"
        if best_seq5 is not None:
            final_note = f"best_seq5={best_seq5['name']} delta_HOTA={float(best_seq5.get('delta_hota', -999.0)):.3f}"
        queue_plan_status(args, "completed", summary_csv, queue_log, notes=final_note)
        queue_registry(args, "completed", summary_csv, queue_log, notes=final_note)
    except Exception as exc:
        finished_at = now_iso()
        for row in rows:
            status = str(row.get("status", ""))
            if status == "running":
                row["status"] = "failed"
                row["finished_at"] = finished_at
                row["notes"] = f"{row.get('notes', '')} | failed: {exc}".strip()
            elif status == "pending":
                row["status"] = "cancelled"
                row["finished_at"] = finished_at
                row["notes"] = f"{row.get('notes', '')} | cancelled_after_failure".strip()
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        queue_plan_status(args, "failed", summary_csv, queue_log, notes=str(exc))
        queue_registry(args, "failed", summary_csv, queue_log, notes=str(exc))
        raise


if __name__ == "__main__":
    main()
