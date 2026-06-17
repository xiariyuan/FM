#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from queue_deep_ocsort_preassoc_force_rewrite_next2h import (
    DECISION_FIELDS,
    QUEUE_FIELDS,
    REPO_ROOT,
    REGISTRY_CSV,
    append_registry,
    build_cmd,
    child_finished_at,
    ensure_child_success,
    now_iso,
    record_decision_from_child,
    run_step,
    timestamp_tag,
    update_row,
    write_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tail-fill queue to keep the local rewrite-gain sweep active when the main two-hour queue ends early."
    )
    parser.add_argument("--out-root", default="")
    parser.add_argument(
        "--reuse-raw-from",
        default=str(REPO_ROOT / "outputs" / "deep_ocsort_preassoc_acceptgate_smoke_20260408_2"),
    )
    parser.add_argument("--seq-name", default="dancetrack0090")
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else (REPO_ROOT / "outputs" / f"deep_ocsort_preassoc_force_rewrite_tailfill_{timestamp_tag()}").resolve()
    )
    reuse_raw_from = Path(args.reuse_raw_from).expanduser().resolve()

    variants = [
        {
            "step": "seq0090_owner150_shared015",
            "name": "seq0090_owner150_shared015",
            "notes": "push rewrite-owner benefit harder while easing shared-alt conflict penalty",
            "seq_names": [args.seq_name],
            "force_rewrite_min_score": 0.65,
            "force_rewrite_min_box_iou": 0.55,
            "force_rewrite_min_neighborhood_gain": 0.0,
            "force_rewrite_max_age_gap": 200,
            "force_rewrite_max_owner_alt_det_box_iou": 0.50,
            "keep_challenger_alt_weight": 0.25,
            "rewrite_owner_alt_weight": 1.50,
            "shared_alt_penalty": 0.15,
        },
        {
            "step": "seq0090_iou050_owner125_shared015",
            "name": "seq0090_iou050_owner125_shared015",
            "notes": "loosen rewrite box-overlap gate slightly while keeping owner rewrite reward elevated",
            "seq_names": [args.seq_name],
            "force_rewrite_min_score": 0.65,
            "force_rewrite_min_box_iou": 0.50,
            "force_rewrite_min_neighborhood_gain": 0.0,
            "force_rewrite_max_age_gap": 200,
            "force_rewrite_max_owner_alt_det_box_iou": 0.50,
            "keep_challenger_alt_weight": 0.25,
            "rewrite_owner_alt_weight": 1.25,
            "shared_alt_penalty": 0.15,
        },
    ]

    summary_csv = run_root / "summary.csv"
    decision_csv = run_root / "decision_summary.csv"
    logs_dir = run_root / "logs"
    runs_dir = run_root / "runs"

    queue_rows: List[Dict[str, object]] = []
    for variant in variants:
        step = str(variant["step"])
        out_dir = (runs_dir / step).resolve()
        queue_rows.append(
            {
                "step": step,
                "name": f"{run_root.name}_{step}",
                "status": "pending",
                "out_dir": str(out_dir),
                "summary_csv": str((out_dir / "summary.csv").resolve()),
                "log_path": str((logs_dir / f"{step}.log").resolve()),
                "started_at": "",
                "finished_at": "",
                "notes": str(variant["notes"]),
            }
        )

    decision_rows: List[Dict[str, object]] = []
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
    write_rows(decision_csv, DECISION_FIELDS, decision_rows)

    overall_status = "success"
    overall_notes = "completed rewrite tail-fill queue"

    for index, variant in enumerate(variants):
        step = str(variant["step"])
        started_at = now_iso()
        update_row(queue_rows, step, status="running", started_at=started_at)
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
        if index == 0:
            append_registry(
                summary_csv,
                run_root,
                "running",
                "started rewrite tail-fill queue",
                args.registry_csv,
            )

        status = "success"
        finished_at = now_iso()
        notes = str(variant["notes"])

        try:
            out_dir = (runs_dir / step).resolve()
            log_path = (logs_dir / f"{step}.log").resolve()
            cmd = build_cmd(
                out_dir=out_dir,
                reuse_raw_from=reuse_raw_from,
                seq_names=list(variant["seq_names"]),
                variant=variant,
            )
            rc = run_step(cmd, log_path, cwd=REPO_ROOT)
            child_summary = out_dir / "summary.csv"
            finished_at = now_iso()
            if rc != 0:
                status = "failed"
                notes = f"child return code {rc}"
            else:
                ensure_child_success(child_summary)
                finished_at = child_finished_at(child_summary)
                decision = record_decision_from_child(
                    decision_rows=decision_rows,
                    decision_csv=decision_csv,
                    step=step,
                    variant_name=str(variant["name"]),
                    child_root=out_dir,
                    variant=variant,
                    notes=str(variant["notes"]),
                )
                notes = (
                    f"delta_HOTA={float(decision['delta_HOTA']):+.3f} "
                    f"delta_AssA={float(decision['delta_AssA']):+.3f} "
                    f"selected_matches={int(decision['selected_matches'])} "
                    f"force_rewrite_accepted={int(decision['force_rewrite_accepted_rows'])}"
                )
        except Exception as exc:
            status = "failed"
            finished_at = now_iso()
            notes = f"run step failed: {exc}"

        update_row(queue_rows, step, status=status, finished_at=finished_at, notes=notes)
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

        if status == "failed":
            overall_status = "failed"
            overall_notes = f"{step} failed: {notes}"

    append_registry(summary_csv, run_root, overall_status, overall_notes, args.registry_csv)
    return 0 if overall_status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
