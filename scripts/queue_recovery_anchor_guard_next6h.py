#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

from queue_deep_ocsort_preassoc_force_rewrite_next2h import (
    QUEUE_FIELDS,
    REPO_ROOT,
    REGISTRY_CSV,
    child_finished_at,
    ensure_child_success,
    now_iso,
    read_metrics_delta,
    read_rows,
    read_runtime,
    run_step,
    timestamp_tag,
    update_row,
    write_rows,
)


DEFAULT_WAIT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "deep_ocsort_preassoc_force_recovery_anchor_gate_histfeat_gtreclaimsoftneg005_t064_pairedfair_diag10_stableonly_h6g010_20260415_1"
)

DEFAULT_SEQ_NAMES = [
    "dancetrack0004",
    "dancetrack0007",
    "dancetrack0019",
    "dancetrack0025",
    "dancetrack0034",
    "dancetrack0058",
    "dancetrack0063",
    "dancetrack0073",
    "dancetrack0077",
    "dancetrack0090",
]

DECISION_FIELDS = [
    "step",
    "variant",
    "status",
    "out_dir",
    "summary_csv",
    "metrics_delta_csv",
    "runtime_compare_csv",
    "reuse_raw_from",
    "max_owner_alt_det_box_iou",
    "stable_owner_min_hits",
    "stable_owner_min_raw_neighborhood_gain",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDs",
    "delta_Frag",
    "selected_matches",
    "force_rewrite_accepted_rows",
    "candidate_rows",
    "decision",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive 5-6 hour queue for recovery-anchor owner-guard follow-up experiments."
    )
    parser.add_argument("--out-root", default="")
    parser.add_argument("--wait-summary-csv", default=str(DEFAULT_WAIT_ROOT / "summary.csv"))
    parser.add_argument("--wait-run-root", default=str(DEFAULT_WAIT_ROOT))
    parser.add_argument(
        "--wait-process-pattern",
        default=str(DEFAULT_WAIT_ROOT.name),
        help="Substring expected in the live process table while the waited run is still active.",
    )
    parser.add_argument("--seq-names", nargs="*", default=list(DEFAULT_SEQ_NAMES))
    parser.add_argument("--poll-seconds", type=int, default=120)
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
        "scripts/queue_recovery_anchor_guard_next6h.py",
        "--dataset",
        "DanceTrack",
        "--split",
        "val",
        "--tracker-family",
        "deep_ocsort_preassoc_force_recovery_anchor",
        "--variant",
        run_root.name,
        "--tag",
        "recovery_anchor_guard_next6h",
        "--run-root",
        str(run_root.resolve()),
        "--summary-csv",
        str(summary_csv.resolve()),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def process_alive(pattern: str) -> bool:
    result = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True, check=False)
    for line in result.stdout.splitlines():
        if pattern in line and "ps -eo cmd" not in line:
            return True
    return False


def wait_for_run(summary_csv: Path, process_pattern: str, poll_seconds: int) -> str:
    while True:
        rows = read_rows(summary_csv)
        if rows:
            compare_rows = [row for row in rows if str(row.get("step", "")) == "compare"]
            if compare_rows:
                status = str(compare_rows[0].get("status", "") or "").strip()
                if status == "success":
                    return child_finished_at(summary_csv)
                if status == "failed":
                    raise RuntimeError(f"waited run failed: {summary_csv}")
        if rows and not process_alive(process_pattern):
            raise RuntimeError(
                f"waited run still not successful, but no live process matches pattern {process_pattern!r}"
            )
        time.sleep(max(int(poll_seconds), 15))


def make_row(step: str, out_dir: Path, log_path: Path, notes: str, summary_csv: Path | None = None) -> Dict[str, object]:
    return {
        "step": step,
        "name": step,
        "status": "pending",
        "out_dir": str(out_dir.resolve()),
        "summary_csv": str((summary_csv or (out_dir / "summary.csv")).resolve()),
        "log_path": str(log_path.resolve()),
        "started_at": "",
        "finished_at": "",
        "notes": notes,
    }


def upsert_decision_row(rows: List[Dict[str, object]], new_row: Dict[str, object]) -> None:
    for index, row in enumerate(rows):
        if str(row.get("step", "")) == str(new_row.get("step", "")):
            merged = dict(row)
            merged.update(new_row)
            rows[index] = merged
            return
    rows.append(dict(new_row))


def build_eval_cmd(
    *,
    out_dir: Path,
    reuse_raw_from: Path,
    seq_names: List[str],
    max_owner_alt_det_box_iou: float,
    stable_owner_min_hits: int,
    stable_owner_min_raw_neighborhood_gain: float,
) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_deep_ocsort_preassoc_competition_dataset_eval.py"),
        "--benchmark",
        "DanceTrack",
        "--seq-names",
        *seq_names,
        "--out-root",
        str(out_dir),
        "--reuse-raw-from",
        str(reuse_raw_from),
        "--preassoc-stale-competition-min-time-since-update",
        "2",
        "--preassoc-stale-competition-max-time-since-update",
        "8",
        "--preassoc-stale-competition-min-hits",
        "8",
        "--preassoc-stale-competition-min-box-iou",
        "0.5",
        "--preassoc-stale-competition-min-edge-score",
        "0.0",
        "--preassoc-stale-competition-bias",
        "0.1",
        "--preassoc-stale-competition-iou-scale",
        "0.0",
        "--preassoc-stale-competition-require-raw-owner",
        "--preassoc-stale-competition-min-age-gap-vs-owner",
        "10",
        "--preassoc-stale-competition-owner-edge-penalty",
        "0.05",
        "--preassoc-stale-competition-takeover-soft-margin-floor",
        "0.1",
        "--preassoc-stale-competition-takeover-soft-edge-advantage-floor",
        "0.05",
        "--preassoc-stale-competition-owner-alt-det-bias",
        "0.3",
        "--preassoc-stale-competition-owner-alt-det-min-score",
        "0.35",
        "--preassoc-stale-competition-owner-alt-det-min-box-iou",
        "0.30",
        "--preassoc-stale-competition-max-owner-edge-deficit",
        "-1.0",
        "--preassoc-stale-competition-block-owner-on-reclaim",
        "--preassoc-stale-competition-force-owner-edge-deficit-arg",
        "--preassoc-stale-competition-force-rewrite-enable",
        "--preassoc-stale-competition-force-rewrite-min-score",
        "0.65",
        "--preassoc-stale-competition-force-rewrite-gate-weight",
        "0.45",
        "--preassoc-stale-competition-force-rewrite-iou-weight",
        "0.30",
        "--preassoc-stale-competition-force-rewrite-ranker-weight",
        "0.15",
        "--preassoc-stale-competition-force-rewrite-age-weight",
        "0.0",
        "--preassoc-stale-competition-force-rewrite-age-cap",
        "20",
        "--preassoc-stale-competition-force-rewrite-owner-alt-bonus",
        "0.0",
        "--preassoc-stale-competition-force-rewrite-neighborhood-enable",
        "--preassoc-stale-competition-force-rewrite-min-neighborhood-gain",
        "0.0",
        "--preassoc-stale-competition-force-rewrite-trapped-owner-min-neighborhood-gain",
        "0.0",
        "--preassoc-stale-competition-force-rewrite-reroute-ready-min-neighborhood-gain",
        "0.0",
        "--preassoc-stale-competition-force-rewrite-trapped-owner-negative-gain-min-challenger-alt-box-iou",
        "-1.0",
        "--preassoc-stale-competition-force-rewrite-neighborhood-keep-challenger-alt-weight",
        "0.25",
        "--preassoc-stale-competition-force-rewrite-neighborhood-rewrite-owner-alt-weight",
        "1.0",
        "--preassoc-stale-competition-force-rewrite-neighborhood-shared-alt-penalty",
        "0.25",
        "--preassoc-stale-competition-force-rewrite-neighborhood-trapped-owner-bonus",
        "0.75",
        "--preassoc-stale-competition-force-rewrite-neighborhood-reroute-ready-penalty",
        "0.6",
        "--preassoc-stale-competition-force-rewrite-min-box-iou",
        "0.55",
        "--preassoc-stale-competition-force-rewrite-max-age-gap",
        "200",
        "--preassoc-stale-competition-force-rewrite-max-owner-alt-det-box-iou",
        str(max_owner_alt_det_box_iou),
        "--preassoc-stale-competition-force-rewrite-stable-owner-min-hits",
        str(stable_owner_min_hits),
        "--preassoc-stale-competition-force-rewrite-stable-owner-min-raw-neighborhood-gain",
        str(stable_owner_min_raw_neighborhood_gain),
        "--preassoc-stale-competition-force-rewrite-reroute-ready-min-box-iou",
        "0.8",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-max-frame-gap",
        "2",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-min-score",
        "0.85",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-min-box-iou",
        "0.8",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-min-challenger-alt-box-iou",
        "0.12",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-warmup-min-neighborhood-gain",
        "-0.08",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-bonus",
        "0.05",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-bonus-max-streak",
        "2",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-gate-bonus",
        "0.0001",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-anchor-min-raw-neighborhood-gain",
        "0.05",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-anchor-max-edge-deficit",
        "0.35",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-extension-max-edge-deficit-delta",
        "0.15",
        "--preassoc-stale-competition-export-jsonl",
        str(out_dir / "preassoc_candidates.jsonl"),
        "--local-contention-export-jsonl",
        str(out_dir / "local_contention_units.jsonl"),
        "--local-contention-topk",
        "3",
        "--local-contention-min-box-iou",
        "0.5",
        "--local-contention-max-time-since-update",
        "8",
        "--local-contention-min-challenger-hits",
        "3",
        "--local-contention-owner-weak-hits",
        "8",
        "--local-contention-ranker-checkpoint",
        str(REPO_ROOT / "outputs" / "local_contention_ranker_mot17_mot20_dance_seqholdout_20260406_1" / "model.pt"),
        "--local-contention-ranker-thresh",
        "0.99",
        "--local-contention-ranker-bias",
        "0.0",
        "--local-contention-ranker-min-margin-to-second",
        "0.05",
        "--local-contention-ranker-margin-bias",
        "0.5",
        "--preassoc-stale-competition-acceptance-gate-checkpoint",
        str(REPO_ROOT / "outputs" / "local_contention_acceptance_gate_mot17_mot20_dance_seqholdout_20260408_1" / "best.pt"),
        "--preassoc-stale-competition-acceptance-gate-thresh",
        "0.9995",
        "--preassoc-stale-competition-recovery-anchor-gate-checkpoint",
        str(REPO_ROOT / "outputs" / "train_recovery_anchor_gate_histfeat_gtreclaimsoftneg005_h16do01_seed701_20260413_1" / "best.pt"),
        "--preassoc-stale-competition-recovery-anchor-gate-thresh",
        "0.64",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-enable",
        "--preassoc-stale-competition-force-rewrite-recovery-memory-anchor-enable",
        "--competition-track-max-frames-per-batch",
        "4000",
    ]


def record_decision(
    *,
    decision_rows: List[Dict[str, object]],
    decision_csv: Path,
    step: str,
    variant_name: str,
    child_root: Path,
    reuse_raw_from: Path,
    max_owner_alt_det_box_iou: float,
    stable_owner_min_hits: int,
    stable_owner_min_raw_neighborhood_gain: float,
    notes: str,
) -> Dict[str, object]:
    metrics = read_metrics_delta(child_root / "metrics_delta.csv")
    runtime = read_runtime(child_root / "runtime_compare.csv")
    row = {
        "step": step,
        "variant": variant_name,
        "status": "success",
        "out_dir": str(child_root.resolve()),
        "summary_csv": str((child_root / "summary.csv").resolve()),
        "metrics_delta_csv": str((child_root / "metrics_delta.csv").resolve()),
        "runtime_compare_csv": str((child_root / "runtime_compare.csv").resolve()),
        "reuse_raw_from": str(reuse_raw_from.resolve()),
        "max_owner_alt_det_box_iou": max_owner_alt_det_box_iou,
        "stable_owner_min_hits": stable_owner_min_hits,
        "stable_owner_min_raw_neighborhood_gain": stable_owner_min_raw_neighborhood_gain,
        "delta_HOTA": metrics["delta_HOTA"],
        "delta_AssA": metrics["delta_AssA"],
        "delta_IDF1": metrics["delta_IDF1"],
        "delta_MOTA": metrics["delta_MOTA"],
        "delta_IDs": metrics["delta_IDs"],
        "delta_Frag": metrics["delta_Frag"],
        "selected_matches": int(runtime["selected_matches"]),
        "force_rewrite_accepted_rows": int(runtime["force_rewrite_accepted_rows"]),
        "candidate_rows": int(runtime["candidate_rows"]),
        "decision": "",
        "notes": notes,
    }
    upsert_decision_row(decision_rows, row)
    write_rows(decision_csv, DECISION_FIELDS, decision_rows)
    return row


def result_score(row: Dict[str, object]) -> tuple[float, float, float, float, float, float]:
    return (
        float(row.get("delta_HOTA", 0.0) or 0.0),
        float(row.get("delta_MOTA", 0.0) or 0.0),
        float(row.get("delta_AssA", 0.0) or 0.0),
        float(row.get("delta_IDF1", 0.0) or 0.0),
        -float(row.get("delta_Frag", 0.0) or 0.0),
        -float(row.get("delta_IDs", 0.0) or 0.0),
    )


def write_schedule(path: Path, queue_root: Path, wait_root: Path) -> None:
    lines = [
        "# Recovery Anchor Guard Next 6 Hours",
        "",
        f"- Queue root: `{queue_root}`",
        f"- Wait for current reference run: `{wait_root.name}`",
        "- Planned order:",
        "  1. Wait for the current stable-owner gain 0.10 run to finish.",
        "  2. Run owner-alt-only tightening at 0.25 using reused raw results.",
        "  3. Run owner-alt-only tightening at 0.20 using reused raw results.",
        "  4. Pick the better owner-alt threshold and combine it with stable-owner gain 0.10.",
        "",
        "- Goal:",
        "  - Test whether the real bottleneck is owner alternate-detection reroute freedom rather than the mild stable-owner gain guard.",
        "  - Keep the search on one axis at a time, then confirm with one combination run.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else (REPO_ROOT / "outputs" / f"queue_recovery_anchor_guard_next6h_{timestamp_tag()}").resolve()
    )
    wait_summary_csv = Path(args.wait_summary_csv).expanduser().resolve()
    wait_run_root = Path(args.wait_run_root).expanduser().resolve()
    seq_names = list(args.seq_names)
    summary_csv = run_root / "summary.csv"
    decision_csv = run_root / "decision_summary.csv"
    schedule_md = run_root / "night_schedule.md"

    queue_rows: List[Dict[str, object]] = [
        make_row(
            "wait_current_h6g010",
            wait_run_root,
            run_root / "logs" / "wait_current_h6g010.log",
            f"wait for {wait_summary_csv}",
            summary_csv=wait_summary_csv,
        ),
        make_row(
            "owneralt025_only",
            run_root / "owneralt025_only",
            run_root / "logs" / "owneralt025_only.log",
            "reuse raw from current h6g010; max_owner_alt_det_box_iou=0.25; stable-owner guard disabled",
        ),
        make_row(
            "owneralt020_only",
            run_root / "owneralt020_only",
            run_root / "logs" / "owneralt020_only.log",
            "reuse raw from current h6g010; max_owner_alt_det_box_iou=0.20; stable-owner guard disabled",
        ),
        make_row(
            "combo_best_owner_with_h6g010",
            run_root / "combo_best_owner_with_h6g010",
            run_root / "logs" / "combo_best_owner_with_h6g010.log",
            "reuse raw from current h6g010; owner-alt threshold chosen from the stronger owner-only run; stable-owner gain=0.10",
        ),
    ]
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
    write_rows(decision_csv, DECISION_FIELDS, [])
    write_schedule(schedule_md, run_root, wait_run_root)
    append_registry(
        summary_csv,
        run_root,
        "running",
        f"queue started; waiting for {wait_run_root.name}, then owner-alt-only 0.25/0.20 plus one combo confirmation",
        args.registry_csv,
    )

    update_row(queue_rows, "wait_current_h6g010", status="running", started_at=now_iso())
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

    try:
        finished_at = wait_for_run(wait_summary_csv, args.wait_process_pattern, args.poll_seconds)
        ensure_child_success(wait_summary_csv)
    except Exception as exc:
        update_row(
            queue_rows,
            "wait_current_h6g010",
            status="failed",
            finished_at=now_iso(),
            notes=f"wait failed: {exc}",
        )
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
        append_registry(summary_csv, run_root, "failed", f"wait step failed: {exc}", args.registry_csv)
        raise

    decision_rows: List[Dict[str, object]] = []
    reference_row = record_decision(
        decision_rows=decision_rows,
        decision_csv=decision_csv,
        step="reference_h6g010_waited",
        variant_name=wait_run_root.name,
        child_root=wait_run_root,
        reuse_raw_from=wait_run_root,
        max_owner_alt_det_box_iou=-1.0,
        stable_owner_min_hits=6,
        stable_owner_min_raw_neighborhood_gain=0.10,
        notes="waited reference run; used as the anchor for the next owner-alt follow-ups",
    )
    reference_row["decision"] = "reference"
    upsert_decision_row(decision_rows, reference_row)
    write_rows(decision_csv, DECISION_FIELDS, decision_rows)

    update_row(
        queue_rows,
        "wait_current_h6g010",
        status="success",
        finished_at=finished_at,
        notes=f"waited successfully; reference delta_HOTA={float(reference_row['delta_HOTA']):+.3f}",
    )
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

    owner_variants = [
        {
            "step": "owneralt025_only",
            "name": "owneralt025_only",
            "max_owner_alt_det_box_iou": 0.25,
            "stable_owner_min_hits": -1,
            "stable_owner_min_raw_neighborhood_gain": -1.0,
        },
        {
            "step": "owneralt020_only",
            "name": "owneralt020_only",
            "max_owner_alt_det_box_iou": 0.20,
            "stable_owner_min_hits": -1,
            "stable_owner_min_raw_neighborhood_gain": -1.0,
        },
    ]

    owner_results: List[Dict[str, object]] = []
    for variant in owner_variants:
        step = str(variant["step"])
        child_root = run_root / step
        log_path = run_root / "logs" / f"{step}.log"
        update_row(queue_rows, step, status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
        cmd = build_eval_cmd(
            out_dir=child_root,
            reuse_raw_from=wait_run_root,
            seq_names=seq_names,
            max_owner_alt_det_box_iou=float(variant["max_owner_alt_det_box_iou"]),
            stable_owner_min_hits=int(variant["stable_owner_min_hits"]),
            stable_owner_min_raw_neighborhood_gain=float(variant["stable_owner_min_raw_neighborhood_gain"]),
        )
        return_code = run_step(cmd, log_path, cwd=REPO_ROOT)
        if return_code != 0:
            update_row(queue_rows, step, status="failed", finished_at=now_iso(), notes=f"return_code={return_code}")
            write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
            append_registry(summary_csv, run_root, "failed", f"{step} failed with return_code={return_code}", args.registry_csv)
            return return_code
        ensure_child_success(child_root / "summary.csv")
        row = record_decision(
            decision_rows=decision_rows,
            decision_csv=decision_csv,
            step=step,
            variant_name=str(variant["name"]),
            child_root=child_root,
            reuse_raw_from=wait_run_root,
            max_owner_alt_det_box_iou=float(variant["max_owner_alt_det_box_iou"]),
            stable_owner_min_hits=int(variant["stable_owner_min_hits"]),
            stable_owner_min_raw_neighborhood_gain=float(variant["stable_owner_min_raw_neighborhood_gain"]),
            notes=f"owner-alt-only run with max_owner_alt_det_box_iou={variant['max_owner_alt_det_box_iou']:.2f}",
        )
        owner_results.append(row)
        update_row(
            queue_rows,
            step,
            status="success",
            finished_at=child_finished_at(child_root / "summary.csv"),
            notes=(
                f"delta_HOTA={float(row['delta_HOTA']):+.3f} "
                f"delta_MOTA={float(row['delta_MOTA']):+.3f} "
                f"owner_alt_iou={float(variant['max_owner_alt_det_box_iou']):.2f}"
            ),
        )
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

    best_owner = max(owner_results, key=result_score)
    best_owner_threshold = float(best_owner["max_owner_alt_det_box_iou"])
    best_owner["decision"] = "selected_for_combo"
    upsert_decision_row(decision_rows, best_owner)
    write_rows(decision_csv, DECISION_FIELDS, decision_rows)

    combo_step = "combo_best_owner_with_h6g010"
    combo_root = run_root / combo_step
    combo_log = run_root / "logs" / f"{combo_step}.log"
    update_row(queue_rows, combo_step, status="running", started_at=now_iso())
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
    combo_cmd = build_eval_cmd(
        out_dir=combo_root,
        reuse_raw_from=wait_run_root,
        seq_names=seq_names,
        max_owner_alt_det_box_iou=best_owner_threshold,
        stable_owner_min_hits=6,
        stable_owner_min_raw_neighborhood_gain=0.10,
    )
    return_code = run_step(combo_cmd, combo_log, cwd=REPO_ROOT)
    if return_code != 0:
        update_row(queue_rows, combo_step, status="failed", finished_at=now_iso(), notes=f"return_code={return_code}")
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
        append_registry(summary_csv, run_root, "failed", f"{combo_step} failed with return_code={return_code}", args.registry_csv)
        return return_code

    ensure_child_success(combo_root / "summary.csv")
    combo_row = record_decision(
        decision_rows=decision_rows,
        decision_csv=decision_csv,
        step=combo_step,
        variant_name=f"combo_owneralt{best_owner_threshold:.2f}_h6g010",
        child_root=combo_root,
        reuse_raw_from=wait_run_root,
        max_owner_alt_det_box_iou=best_owner_threshold,
        stable_owner_min_hits=6,
        stable_owner_min_raw_neighborhood_gain=0.10,
        notes=(
            "combo run using the stronger owner-alt-only threshold and stable-owner gain 0.10; "
            f"chosen owner-alt threshold={best_owner_threshold:.2f}"
        ),
    )
    update_row(
        queue_rows,
        combo_step,
        status="success",
        finished_at=child_finished_at(combo_root / "summary.csv"),
        notes=(
            f"delta_HOTA={float(combo_row['delta_HOTA']):+.3f} "
            f"delta_MOTA={float(combo_row['delta_MOTA']):+.3f} "
            f"chosen_owner_alt_iou={best_owner_threshold:.2f}"
        ),
    )

    best_overall = max(decision_rows, key=result_score)
    best_overall["decision"] = "best_overall"
    upsert_decision_row(decision_rows, best_overall)
    write_rows(decision_csv, DECISION_FIELDS, decision_rows)
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

    append_registry(
        summary_csv,
        run_root,
        "success",
        (
            f"queue finished; best_overall={best_overall['step']} "
            f"delta_HOTA={float(best_overall['delta_HOTA']):+.3f} "
            f"delta_MOTA={float(best_overall['delta_MOTA']):+.3f}"
        ),
        args.registry_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
