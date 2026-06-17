#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
        description="Run three-sequence DanceTrack confirmation for the recovery-anchor override branch inside force-rewrite acceptance."
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
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def build_variants(seq_names: List[str]) -> List[Dict[str, object]]:
    base = anchored_variant_base(seq_names)
    common = {
        **base,
        "acceptance_gate_checkpoint": str(OLD_GATE_CKPT),
        "acceptance_gate_thresh": 0.9995,
        "recovery_anchor_gate_checkpoint": str(NEW_GATE_H16DO01),
        "recovery_anchor_gate_thresh": 0.29,
    }
    return [
        {
            **common,
            "step": "anchor_t0290_ref_dance3",
            "name": "anchor_t0290_ref_dance3",
            "notes": "reference run with recovery-anchor threshold 0.29 and no force-rewrite override branch",
        },
        {
            **common,
            "step": "anchor_t0290_override_n001_dance3",
            "name": "anchor_t0290_override_n001_dance3",
            "notes": "enable recovery-anchor override and allow raw neighborhood gain down to -0.01 so only the near-tie boundary row can pass",
            "force_rewrite_recovery_anchor_override_enable": True,
            "force_rewrite_recovery_anchor_min_raw_neighborhood_gain": -0.01,
        },
        {
            **common,
            "step": "anchor_t0290_override_n006_dance3",
            "name": "anchor_t0290_override_n006_dance3",
            "notes": "enable recovery-anchor override and allow raw neighborhood gain down to -0.06 to test whether admitting both boundary rows is too loose",
            "force_rewrite_recovery_anchor_override_enable": True,
            "force_rewrite_recovery_anchor_min_raw_neighborhood_gain": -0.06,
        },
    ]


def main() -> int:
    args = parse_args()
    run_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else (
            REPO_ROOT / "outputs" / f"deep_ocsort_preassoc_force_recovery_anchor_override_dance3_{timestamp_tag()}"
        ).resolve()
    )
    reuse_raw_from = Path(args.reuse_raw_from).expanduser().resolve()
    seq_names = list(args.seq_names)
    variants = build_variants(seq_names)

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
    overall_notes = "completed three-sequence recovery-anchor override confirmation on DanceTrack"

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
                "started three-sequence recovery-anchor override confirmation on DanceTrack",
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
                    notes=(
                        f"{variant['notes']} "
                        f"gate_ckpt={Path(str(variant['acceptance_gate_checkpoint'])).name} "
                        f"gate_thresh={variant['acceptance_gate_thresh']} "
                        f"recovery_gate_ckpt={Path(str(variant['recovery_anchor_gate_checkpoint'])).name} "
                        f"recovery_gate_thresh={variant['recovery_anchor_gate_thresh']} "
                        f"override_enable={int(bool(variant.get('force_rewrite_recovery_anchor_override_enable', False)))} "
                        f"override_min_raw_gain={variant.get('force_rewrite_recovery_anchor_min_raw_neighborhood_gain', '')}"
                    ),
                )
                notes = (
                    f"delta_HOTA={float(decision['delta_HOTA']):+.3f} "
                    f"delta_AssA={float(decision['delta_AssA']):+.3f} "
                    f"delta_IDF1={float(decision['delta_IDF1']):+.3f} "
                    f"selected_matches={int(decision['selected_matches'])} "
                    f"force_rewrite_accepted={int(decision['force_rewrite_accepted_rows'])} "
                    f"override_rows={int(decision['force_rewrite_recovery_anchor_override_rows'])}"
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
