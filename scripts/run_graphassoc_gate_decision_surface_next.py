#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from run_graphassoc_gate_structural_next5h import (
    DEFAULT_GATE_CKPT,
    DEFAULT_REFERENCE_METRICS,
    QUEUE_FIELDS,
    append_reference_row,
    build_eval_command,
    find_row,
    now_iso,
    parse_metrics_delta,
    parse_runtime,
    queue_plan_status,
    queue_registry,
    read_csv_rows,
    run_step,
    update_row,
    write_rows,
)


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Run focused graph-association gate decision-surface follow-up experiments.")
    parser.add_argument("--run-root", default=str(REPO_ROOT / "outputs" / f"graphassoc_gate_decision_surface_{ts}"))
    parser.add_argument("--queue-name", default=f"graphassoc_gate_decision_surface_{ts}")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--gate-checkpoint", default=str(DEFAULT_GATE_CKPT))
    parser.add_argument("--seq-ids", nargs="+", type=int, default=[2, 5])
    parser.add_argument("--reference-metrics", default=str(DEFAULT_REFERENCE_METRICS))
    parser.add_argument("--commit-device", default="cpu")
    return parser.parse_args()


def common_graph_args(args: argparse.Namespace) -> List[str]:
    return [
        "--graph-assoc-commit-checkpoint",
        str(Path(args.gate_checkpoint).expanduser().resolve()),
        "--graph-assoc-commit-device",
        str(args.commit_device),
        "--graph-assoc-commit-score-margin",
        "0.0",
        "--graph-assoc-commit-replace-rules",
        "--graph-assoc-dump-candidate-rows",
    ]


def wide_graph_args() -> List[str]:
    return [
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


def variant_catalog(args: argparse.Namespace) -> List[Dict[str, object]]:
    wide = wide_graph_args()
    common = common_graph_args(args)
    return [
        {
            "name": "wide_posminus_t002",
            "extra_args": wide
            + common
            + [
                "--graph-assoc-commit-decision-mode",
                "positive_minus_neutral",
                "--graph-assoc-commit-threshold",
                "0.02",
            ],
            "notes": "lower the current best positive-minus-neutral threshold slightly to test whether more accepted rewrites add useful association gain",
        },
        {
            "name": "wide_posminus_t004",
            "extra_args": wide
            + common
            + [
                "--graph-assoc-commit-decision-mode",
                "positive_minus_neutral",
                "--graph-assoc-commit-threshold",
                "0.04",
            ],
            "notes": "raise the current best positive-minus-neutral threshold slightly to test whether cleaner rewrites reduce ID risk without losing much HOTA",
        },
        {
            "name": "wide_weighted_w110_t000",
            "extra_args": wide
            + common
            + [
                "--graph-assoc-commit-decision-mode",
                "positive_minus_weighted_neutral",
                "--graph-assoc-commit-neutral-risk-weight",
                "1.10",
                "--graph-assoc-commit-threshold",
                "0.0",
            ],
            "notes": "interpolate between the current best and the safer weighted run by reducing neutral-risk penalty from 1.25 to 1.10",
        },
        {
            "name": "wide_weighted_w115_tm002",
            "extra_args": wide
            + common
            + [
                "--graph-assoc-commit-decision-mode",
                "positive_minus_weighted_neutral",
                "--graph-assoc-commit-neutral-risk-weight",
                "1.15",
                "--graph-assoc-commit-threshold",
                "-0.02",
            ],
            "notes": "balanced weighted decision surface expected to accept between the best HOTA run and the safer low-ID-switch run",
        },
        {
            "name": "wide_times_t025",
            "extra_args": wide
            + common
            + [
                "--graph-assoc-commit-decision-mode",
                "positive_times_one_minus_neutral",
                "--graph-assoc-commit-threshold",
                "0.25",
            ],
            "notes": "nonlinear decision surface that requires positive evidence while sharply suppressing neutral-risk candidates",
        },
    ]


def main() -> None:
    args = parse_args()
    queue_root = Path(args.run_root).expanduser().resolve()
    logs_dir = queue_root / "logs"
    summary_csv = queue_root / "summary.csv"
    queue_log = logs_dir / "queue.log"
    rows: List[Dict[str, object]] = [dict(row) for row in read_csv_rows(summary_csv)]

    queue_root.mkdir(parents=True, exist_ok=True)
    queue_plan_status(args, "running", summary_csv, queue_log, notes="graphassoc decision-surface follow-up queue started")
    queue_registry(args, "running", summary_csv, queue_log, notes="graphassoc decision-surface follow-up queue started")
    append_reference_row(args, rows, summary_csv)

    try:
        for idx, variant in enumerate(variant_catalog(args), start=1):
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
                        "notes": variant["notes"],
                        "params_json": json.dumps({"extra_args": variant["extra_args"]}, ensure_ascii=False),
                    }
                )
            if (child_root / "metrics_delta.csv").is_file():
                metrics = parse_metrics_delta(child_root)
                runtime = parse_runtime(child_root)
                existing = find_row(rows, step) or {}
                update_row(
                    rows,
                    step,
                    status="success",
                    started_at=existing.get("started_at", ""),
                    finished_at=existing.get("finished_at", "") or now_iso(),
                    **metrics,
                    **runtime,
                    notes=f"{variant['notes']} | complete",
                )
                write_rows(summary_csv, QUEUE_FIELDS, rows)
                continue

            update_row(rows, step, status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            cmd = build_eval_command(args, child_root, exp_name, list(variant["extra_args"]))
            rc = run_step(cmd, child_log)
            if rc != 0:
                update_row(rows, step, status="failed", finished_at=now_iso(), notes=f"{variant['notes']} | return_code={rc}")
                write_rows(summary_csv, QUEUE_FIELDS, rows)
                continue
            metrics = parse_metrics_delta(child_root)
            runtime = parse_runtime(child_root)
            update_row(
                rows,
                step,
                status="success",
                finished_at=now_iso(),
                **metrics,
                **runtime,
                notes=f"{variant['notes']} | complete",
            )
            write_rows(summary_csv, QUEUE_FIELDS, rows)

        queue_plan_status(args, "completed", summary_csv, queue_log, notes="graphassoc decision-surface follow-up queue completed")
        queue_registry(args, "success", summary_csv, queue_log, notes="graphassoc decision-surface follow-up queue completed")
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
