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


OLD_GATE_CKPT = REPO_ROOT / "outputs" / "local_contention_acceptance_gate_mot17_mot20_dance_seqholdout_20260408_1" / "best.pt"
NEW_GATE_H16DO01 = (
    REPO_ROOT
    / "outputs"
    / "queue_recovery_anchor_threshold_confirm_until_dawn_20260410_1"
    / "dflt_h16do01_seed701"
    / "best.pt"
)
NEW_GATE_H32DO00 = (
    REPO_ROOT
    / "outputs"
    / "queue_recovery_anchor_threshold_confirm_until_dawn_20260410_1"
    / "dflt_h32do00_seed701"
    / "best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run three-sequence DanceTrack confirmation for anchored recovery structure with old-vs-new acceptance gates."
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
    parser.add_argument("--recovery-anchor-h16-checkpoint", default=str(NEW_GATE_H16DO01))
    parser.add_argument("--recovery-anchor-h16-thresh", type=float, default=0.0)
    parser.add_argument("--recovery-anchor-h32-checkpoint", default=str(NEW_GATE_H32DO00))
    parser.add_argument("--recovery-anchor-h32-thresh", type=float, default=0.0)
    parser.add_argument(
        "--disable-h32",
        action="store_true",
        help="Skip the h32 recovery-anchor gate variant when only validating the updated h16 model.",
    )
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def anchored_variant_base(seq_names: List[str]) -> Dict[str, object]:
    return {
        "seq_names": list(seq_names),
        "force_rewrite_min_score": 0.65,
        "force_rewrite_min_box_iou": 0.55,
        "force_rewrite_min_neighborhood_gain": 0.0,
        "trapped_owner_min_neighborhood_gain": 0.0,
        "reroute_ready_min_neighborhood_gain": 0.0,
        "trapped_owner_negative_gain_min_challenger_alt_box_iou": -1.0,
        "force_rewrite_max_age_gap": 200,
        "force_rewrite_max_owner_alt_det_box_iou": 0.50,
        "reroute_ready_min_box_iou": 0.80,
        "keep_challenger_alt_weight": 0.25,
        "rewrite_owner_alt_weight": 1.00,
        "shared_alt_penalty": 0.25,
        "trapped_owner_bonus": 0.75,
        "reroute_ready_penalty": 0.60,
        "recovery_memory_enable": True,
        "recovery_memory_max_frame_gap": 2,
        "recovery_memory_min_score": 0.85,
        "recovery_memory_min_box_iou": 0.80,
        "recovery_memory_min_challenger_alt_box_iou": 0.12,
        "recovery_memory_warmup_min_neighborhood_gain": -0.08,
        "recovery_memory_bonus": 0.05,
        "recovery_memory_bonus_max_streak": 2,
        "recovery_memory_gate_bonus": 0.0001,
        "recovery_memory_anchor_enable": True,
        "recovery_memory_anchor_min_raw_neighborhood_gain": 0.05,
        "recovery_memory_anchor_max_edge_deficit": 0.35,
        "recovery_memory_extension_max_edge_deficit_delta": 0.15,
    }


def main() -> int:
    args = parse_args()
    run_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else (
            REPO_ROOT / "outputs" / f"deep_ocsort_preassoc_force_recovery_anchor_gate_dance3_{timestamp_tag()}"
        ).resolve()
    )
    reuse_raw_from = Path(args.reuse_raw_from).expanduser().resolve()
    seq_names = list(args.seq_names)

    variants: List[Dict[str, object]] = [
        {
            **anchored_variant_base(seq_names),
            "step": "anchor_old_gate_dance3",
            "name": "anchor_old_gate_dance3",
            "notes": "anchored recovery structure with only the old stage-one local-contention gate and the old fixed high threshold",
            "acceptance_gate_checkpoint": str(Path(args.old_gate_checkpoint).expanduser().resolve()),
            "acceptance_gate_thresh": float(args.old_gate_thresh),
        },
        {
            **anchored_variant_base(seq_names),
            "step": "anchor_new_gate_h16do01_dance3",
            "name": "anchor_new_gate_h16do01_dance3",
            "notes": "anchored recovery structure with the old stage-one gate plus the new second-stage recovery-anchor gate h16do01 and its learned threshold",
            "acceptance_gate_checkpoint": str(Path(args.old_gate_checkpoint).expanduser().resolve()),
            "acceptance_gate_thresh": float(args.old_gate_thresh),
            "recovery_anchor_gate_checkpoint": str(Path(args.recovery_anchor_h16_checkpoint).expanduser().resolve()),
            "recovery_anchor_gate_thresh": float(args.recovery_anchor_h16_thresh),
        },
    ]
    if not bool(args.disable_h32):
        variants.append(
            {
                **anchored_variant_base(seq_names),
                "step": "anchor_new_gate_h32do00_dance3",
                "name": "anchor_new_gate_h32do00_dance3",
                "notes": "anchored recovery structure with the old stage-one gate plus the new second-stage recovery-anchor gate h32do00 and its learned threshold",
                "acceptance_gate_checkpoint": str(Path(args.old_gate_checkpoint).expanduser().resolve()),
                "acceptance_gate_thresh": float(args.old_gate_thresh),
                "recovery_anchor_gate_checkpoint": str(
                    Path(args.recovery_anchor_h32_checkpoint).expanduser().resolve()
                ),
                "recovery_anchor_gate_thresh": float(args.recovery_anchor_h32_thresh),
            }
        )

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
    overall_notes = "completed three-sequence gate confirmation for anchored recovery structure"

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
                "started three-sequence gate confirmation for anchored recovery structure",
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
                        f"recovery_gate_ckpt={Path(str(variant.get('recovery_anchor_gate_checkpoint', ''))).name if variant.get('recovery_anchor_gate_checkpoint') else 'none'} "
                        f"recovery_gate_thresh={variant.get('recovery_anchor_gate_thresh', '')}"
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
