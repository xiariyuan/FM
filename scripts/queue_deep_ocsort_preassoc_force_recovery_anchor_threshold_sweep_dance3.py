#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from queue_deep_ocsort_preassoc_force_recovery_anchor_gate_dance3 import (
    NEW_GATE_H16DO01,
    OLD_GATE_CKPT,
    anchored_variant_base,
)
from queue_deep_ocsort_preassoc_force_rewrite_next2h import (
    DECISION_FIELDS,
    QUEUE_FIELDS,
    REPO_ROOT,
    REGISTRY_CSV,
    build_cmd,
    child_finished_at,
    ensure_child_success,
    now_iso,
    read_rows,
    record_decision_from_child,
    run_step,
    timestamp_tag,
    update_row,
    write_rows,
)


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
        "scripts/queue_deep_ocsort_preassoc_force_recovery_anchor_threshold_sweep_dance3.py",
        "--dataset",
        "DanceTrack",
        "--split",
        "val",
        "--tracker-family",
        "deep_ocsort_preassoc_force_recovery_anchor_threshold_sweep",
        "--variant",
        run_root.name,
        "--tag",
        "deep_ocsort_preassoc_force_recovery_anchor_threshold_sweep_dance3",
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run three-sequence DanceTrack threshold calibration for the h16 second-stage recovery-anchor gate."
    )
    parser.add_argument("--out-root", default="")
    parser.add_argument(
        "--reuse-raw-from",
        default=str(REPO_ROOT / "outputs" / "deep_ocsort_preassoc_acceptgate_smoke_20260408_2"),
    )
    parser.add_argument(
        "--seq-names",
        nargs="*",
        default=["dancetrack0081", "dancetrack0090", "dancetrack0094"],
    )
    parser.add_argument("--old-gate-checkpoint", default=str(OLD_GATE_CKPT))
    parser.add_argument("--old-gate-thresh", type=float, default=0.9995)
    parser.add_argument("--recovery-anchor-gate-checkpoint", default=str(NEW_GATE_H16DO01))
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=[0.57, 0.385, 0.29],
        help="Thresholds to sweep for the h16 second-stage recovery-anchor gate.",
    )
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def threshold_tag(value: float) -> str:
    return f"{int(round(float(value) * 1000)):04d}"


def build_variants(
    seq_names: List[str],
    *,
    old_gate_checkpoint: str,
    old_gate_thresh: float,
    recovery_anchor_gate_checkpoint: str,
    thresholds: List[float],
) -> List[Dict[str, object]]:
    base = anchored_variant_base(seq_names)
    variants: List[Dict[str, object]] = []
    ordered_thresholds = [float(value) for value in list(thresholds or [])]
    for index, threshold in enumerate(ordered_thresholds):
        tag = threshold_tag(threshold)
        variants.append(
            {
                **base,
                "step": f"anchor_h16_t{tag}_dance3",
                "name": f"anchor_h16_t{tag}_dance3",
                "notes": (
                    "threshold sweep for the h16 second-stage recovery-anchor gate "
                    f"at threshold {threshold:.3f}"
                ),
                "acceptance_gate_checkpoint": str(Path(old_gate_checkpoint).expanduser().resolve()),
                "acceptance_gate_thresh": float(old_gate_thresh),
                "recovery_anchor_gate_checkpoint": str(Path(recovery_anchor_gate_checkpoint).expanduser().resolve()),
                "recovery_anchor_gate_thresh": float(threshold),
                "variant_rank": int(index),
            }
        )
    return variants


def main() -> int:
    args = parse_args()
    run_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else (
            REPO_ROOT / "outputs" / f"deep_ocsort_preassoc_force_recovery_anchor_threshold_sweep_dance3_{timestamp_tag()}"
        ).resolve()
    )
    reuse_raw_from = Path(args.reuse_raw_from).expanduser().resolve()
    seq_names = list(args.seq_names)
    variants = build_variants(
        seq_names,
        old_gate_checkpoint=str(args.old_gate_checkpoint),
        old_gate_thresh=float(args.old_gate_thresh),
        recovery_anchor_gate_checkpoint=str(args.recovery_anchor_gate_checkpoint),
        thresholds=[float(value) for value in list(args.thresholds or [])],
    )

    summary_csv = run_root / "summary.csv"
    decision_csv = run_root / "decision_summary.csv"
    logs_dir = run_root / "logs"
    runs_dir = run_root / "runs"

    existing_queue_rows = {str(row.get("step", "")): row for row in read_rows(summary_csv)}
    existing_decision_rows = read_rows(decision_csv)

    queue_rows: List[Dict[str, object]] = []
    for variant in variants:
        step = str(variant["step"])
        out_dir = (runs_dir / step).resolve()
        row = {
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
        existing_row = existing_queue_rows.get(step)
        if existing_row:
            row.update(
                {
                    "status": str(existing_row.get("status", row["status"])),
                    "started_at": str(existing_row.get("started_at", row["started_at"])),
                    "finished_at": str(existing_row.get("finished_at", row["finished_at"])),
                    "notes": str(existing_row.get("notes", row["notes"])),
                }
            )
        queue_rows.append(row)

    decision_rows: List[Dict[str, object]] = list(existing_decision_rows)
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
    write_rows(decision_csv, DECISION_FIELDS, decision_rows)

    overall_status = "success"
    overall_notes = "completed h16 second-stage recovery-anchor threshold sweep on three DanceTrack sequences"

    for index, variant in enumerate(variants):
        step = str(variant["step"])
        out_dir = (runs_dir / step).resolve()
        child_summary = out_dir / "summary.csv"
        existing_status = next(
            (str(row.get("status", "")) for row in queue_rows if str(row.get("step", "")) == step),
            "",
        )

        if child_summary.is_file():
            try:
                ensure_child_success(child_summary)
                finished_at = child_finished_at(child_summary)
                decision = record_decision_from_child(
                    decision_rows=decision_rows,
                    decision_csv=decision_csv,
                    step=step,
                    variant_name=str(variant["name"]),
                    child_root=out_dir,
                    variant=variant,
                    notes=(
                        f"{variant['notes']} "
                        f"gate_ckpt={Path(str(variant['acceptance_gate_checkpoint'])).name} "
                        f"gate_thresh={variant['acceptance_gate_thresh']} "
                        f"recovery_gate_ckpt={Path(str(variant['recovery_anchor_gate_checkpoint'])).name} "
                        f"recovery_gate_thresh={variant['recovery_anchor_gate_thresh']}"
                    ),
                )
                notes = (
                    f"delta_HOTA={float(decision['delta_HOTA']):+.3f} "
                    f"delta_AssA={float(decision['delta_AssA']):+.3f} "
                    f"delta_IDF1={float(decision['delta_IDF1']):+.3f} "
                    f"selected_matches={int(decision['selected_matches'])} "
                    f"force_rewrite_accepted={int(decision['force_rewrite_accepted_rows'])}"
                )
                update_row(queue_rows, step, status="success", finished_at=finished_at, notes=notes)
                write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
                continue
            except Exception:
                if existing_status == "success":
                    update_row(queue_rows, step, status="pending", finished_at="", notes=str(variant["notes"]))
                    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

        started_at = now_iso()
        update_row(queue_rows, step, status="running", started_at=started_at)
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
        if index == 0:
            append_registry(
                summary_csv,
                run_root,
                "running",
                "started h16 second-stage recovery-anchor threshold sweep on three DanceTrack sequences",
                args.registry_csv,
            )

        status = "success"
        finished_at = now_iso()
        notes = str(variant["notes"])

        try:
            log_path = (logs_dir / f"{step}.log").resolve()
            cmd = build_cmd(
                out_dir=out_dir,
                reuse_raw_from=reuse_raw_from,
                seq_names=list(variant["seq_names"]),
                variant=variant,
            )
            rc = run_step(cmd, log_path, cwd=REPO_ROOT)
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
                    notes=(
                        f"{variant['notes']} "
                        f"gate_ckpt={Path(str(variant['acceptance_gate_checkpoint'])).name} "
                        f"gate_thresh={variant['acceptance_gate_thresh']} "
                        f"recovery_gate_ckpt={Path(str(variant['recovery_anchor_gate_checkpoint'])).name} "
                        f"recovery_gate_thresh={variant['recovery_anchor_gate_thresh']}"
                    ),
                )
                notes = (
                    f"delta_HOTA={float(decision['delta_HOTA']):+.3f} "
                    f"delta_AssA={float(decision['delta_AssA']):+.3f} "
                    f"delta_IDF1={float(decision['delta_IDF1']):+.3f} "
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
