#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

QUEUE_FIELDS = [
    "step",
    "name",
    "status",
    "out_dir",
    "summary_csv",
    "log_path",
    "started_at",
    "finished_at",
    "notes",
]

DECISION_FIELDS = [
    "step",
    "variant",
    "status",
    "out_dir",
    "summary_csv",
    "metrics_delta_csv",
    "runtime_compare_csv",
    "acceptance_gate_checkpoint",
    "acceptance_gate_thresh",
    "recovery_anchor_gate_checkpoint",
    "recovery_anchor_gate_thresh",
    "force_rewrite_min_score",
    "force_rewrite_min_box_iou",
    "force_rewrite_min_neighborhood_gain",
    "trapped_owner_min_neighborhood_gain",
    "reroute_ready_min_neighborhood_gain",
    "trapped_owner_negative_gain_min_challenger_alt_box_iou",
    "keep_challenger_alt_weight",
    "rewrite_owner_alt_weight",
    "shared_alt_penalty",
    "trapped_owner_bonus",
    "reroute_ready_penalty",
    "reroute_ready_min_box_iou",
    "force_rewrite_recovery_anchor_override_enable",
    "force_rewrite_recovery_anchor_min_raw_neighborhood_gain",
    "recovery_memory_enable",
    "recovery_memory_max_frame_gap",
    "recovery_memory_min_score",
    "recovery_memory_min_box_iou",
    "recovery_memory_min_challenger_alt_box_iou",
    "recovery_memory_warmup_min_neighborhood_gain",
    "recovery_memory_bonus",
    "recovery_memory_bonus_max_streak",
    "recovery_memory_gate_bonus",
    "recovery_memory_anchor_enable",
    "recovery_memory_anchor_min_raw_neighborhood_gain",
    "recovery_memory_anchor_max_edge_deficit",
    "recovery_memory_extension_max_edge_deficit_delta",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDs",
    "delta_Frag",
    "selected_matches",
    "force_rewrite_accepted_rows",
    "force_rewrite_recovery_anchor_override_rows",
    "candidate_rows",
    "decision",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous two-hour follow-up queue for local rewrite-gain pre-association experiments."
    )
    parser.add_argument("--out-root", default="")
    parser.add_argument(
        "--wait-summary-csv",
        default=str(
            REPO_ROOT
            / "outputs"
            / "deep_ocsort_preassoc_force_rewrite_neighborhood_v2_seq0090_20260409_1"
            / "summary.csv"
        ),
    )
    parser.add_argument(
        "--wait-run-root",
        default=str(REPO_ROOT / "outputs" / "deep_ocsort_preassoc_force_rewrite_neighborhood_v2_seq0090_20260409_1"),
    )
    parser.add_argument(
        "--reuse-raw-from",
        default=str(REPO_ROOT / "outputs" / "deep_ocsort_preassoc_acceptgate_smoke_20260408_2"),
    )
    parser.add_argument("--seq-name", default="dancetrack0090")
    parser.add_argument(
        "--confirm-seq-names",
        nargs="*",
        default=["dancetrack0081", "dancetrack0090", "dancetrack0094"],
    )
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp_tag() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def update_row(rows: List[Dict[str, object]], step: str, **updates: object) -> None:
    for row in rows:
        if str(row.get("step", "")) == step:
            row.update(updates)
            return
    raise KeyError(f"Missing queue step: {step}")


def upsert_decision_row(rows: List[Dict[str, object]], new_row: Dict[str, object]) -> None:
    for index, row in enumerate(rows):
        if str(row.get("step", "")) == str(new_row.get("step", "")):
            merged = dict(row)
            merged.update(new_row)
            rows[index] = merged
            return
    rows.append(dict(new_row))


def run_step(cmd: List[str], log_path: Path, *, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[started_at] {now_iso()}\n")
        handle.write(f"[cwd] {cwd}\n")
        handle.write("[cmd] " + " ".join(cmd) + "\n\n")
        handle.flush()
        process = subprocess.run(cmd, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[finished_at] {now_iso()}\n")
        handle.write(f"[return_code] {process.returncode}\n")
    return int(process.returncode)


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
        "scripts/queue_deep_ocsort_preassoc_force_rewrite_next2h.py",
        "--dataset",
        "DanceTrack",
        "--split",
        "val",
        "--tracker-family",
        "deep_ocsort_preassoc_force_rewrite",
        "--variant",
        run_root.name,
        "--tag",
        "deep_ocsort_preassoc_force_rewrite_next2h",
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def ensure_child_success(summary_csv: Path) -> None:
    rows = read_rows(summary_csv)
    if not rows:
        raise FileNotFoundError(f"Missing child summary rows: {summary_csv}")
    statuses = {str(row.get("status", "")).strip() for row in rows}
    if statuses != {"success"}:
        raise RuntimeError(f"Unexpected child status in {summary_csv}: {sorted(statuses)}")


def child_finished_at(summary_csv: Path) -> str:
    rows = read_rows(summary_csv)
    latest = ""
    compare_finished = ""
    for row in rows:
        finished_at = str(row.get("finished_at", "") or "")
        if finished_at and finished_at > latest:
            latest = finished_at
        if str(row.get("step", "")) == "compare" and finished_at:
            compare_finished = finished_at
    return compare_finished or latest or now_iso()


def read_metrics_delta(metrics_delta_csv: Path) -> Dict[str, float]:
    rows = read_rows(metrics_delta_csv)
    if not rows:
        raise FileNotFoundError(f"Missing metrics delta rows: {metrics_delta_csv}")
    row = rows[0]
    return {
        "delta_HOTA": float(row.get("delta_HOTA", 0.0) or 0.0),
        "delta_AssA": float(row.get("delta_AssA", 0.0) or 0.0),
        "delta_IDF1": float(row.get("delta_IDF1", 0.0) or 0.0),
        "delta_MOTA": float(row.get("delta_MOTA", 0.0) or 0.0),
        "delta_IDs": float(row.get("delta_IDs", 0.0) or 0.0),
        "delta_Frag": float(row.get("delta_Frag", 0.0) or 0.0),
    }


def read_runtime(runtime_compare_csv: Path) -> Dict[str, float]:
    rows = read_rows(runtime_compare_csv)
    for row in rows:
        if str(row.get("name", "")) == "competition":
            return {
                "candidate_rows": float(row.get("preassoc_stale_competition_candidate_rows", 0.0) or 0.0),
                "selected_matches": float(row.get("preassoc_stale_competition_selected_matches", 0.0) or 0.0),
                "force_rewrite_accepted_rows": float(
                    row.get("preassoc_stale_competition_force_rewrite_accepted_rows", 0.0) or 0.0
                ),
                "force_rewrite_recovery_anchor_override_rows": float(
                    row.get("preassoc_stale_competition_force_rewrite_recovery_anchor_override_rows", 0.0) or 0.0
                ),
            }
    raise ValueError(f"Missing competition runtime row in {runtime_compare_csv}")


def build_cmd(
    *,
    out_dir: Path,
    reuse_raw_from: Path,
    seq_names: List[str],
    variant: Dict[str, object],
) -> List[str]:
    acceptance_gate_checkpoint = str(
        variant.get(
            "acceptance_gate_checkpoint",
            REPO_ROOT / "outputs" / "local_contention_acceptance_gate_mot17_mot20_dance_seqholdout_20260408_1" / "best.pt",
        )
    )
    acceptance_gate_thresh = str(variant.get("acceptance_gate_thresh", 0.9995))
    recovery_anchor_gate_checkpoint = str(variant.get("recovery_anchor_gate_checkpoint", "") or "")
    recovery_anchor_gate_thresh = str(variant.get("recovery_anchor_gate_thresh", 0.0))
    force_rewrite_recovery_anchor_override_enable = bool(
        variant.get("force_rewrite_recovery_anchor_override_enable", False)
    )
    force_rewrite_recovery_anchor_min_raw_neighborhood_gain = str(
        variant.get("force_rewrite_recovery_anchor_min_raw_neighborhood_gain", -1.0)
    )
    cmd = [
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
        str(variant["force_rewrite_min_score"]),
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
        str(variant["force_rewrite_min_neighborhood_gain"]),
        "--preassoc-stale-competition-force-rewrite-trapped-owner-min-neighborhood-gain",
        str(variant.get("trapped_owner_min_neighborhood_gain", 999.0)),
        "--preassoc-stale-competition-force-rewrite-reroute-ready-min-neighborhood-gain",
        str(variant.get("reroute_ready_min_neighborhood_gain", 999.0)),
        "--preassoc-stale-competition-force-rewrite-trapped-owner-negative-gain-min-challenger-alt-box-iou",
        str(variant.get("trapped_owner_negative_gain_min_challenger_alt_box_iou", -1.0)),
        "--preassoc-stale-competition-force-rewrite-neighborhood-keep-challenger-alt-weight",
        str(variant["keep_challenger_alt_weight"]),
        "--preassoc-stale-competition-force-rewrite-neighborhood-rewrite-owner-alt-weight",
        str(variant["rewrite_owner_alt_weight"]),
        "--preassoc-stale-competition-force-rewrite-neighborhood-shared-alt-penalty",
        str(variant["shared_alt_penalty"]),
        "--preassoc-stale-competition-force-rewrite-neighborhood-trapped-owner-bonus",
        str(variant.get("trapped_owner_bonus", 0.0)),
        "--preassoc-stale-competition-force-rewrite-neighborhood-reroute-ready-penalty",
        str(variant.get("reroute_ready_penalty", 0.0)),
        "--preassoc-stale-competition-force-rewrite-min-box-iou",
        str(variant["force_rewrite_min_box_iou"]),
        "--preassoc-stale-competition-force-rewrite-max-age-gap",
        str(variant["force_rewrite_max_age_gap"]),
        "--preassoc-stale-competition-force-rewrite-max-owner-alt-det-box-iou",
        str(variant["force_rewrite_max_owner_alt_det_box_iou"]),
        "--preassoc-stale-competition-force-rewrite-reroute-ready-min-box-iou",
        str(variant.get("reroute_ready_min_box_iou", -1.0)),
        "--preassoc-stale-competition-force-rewrite-recovery-memory-max-frame-gap",
        str(variant.get("recovery_memory_max_frame_gap", 3)),
        "--preassoc-stale-competition-force-rewrite-recovery-memory-min-score",
        str(variant.get("recovery_memory_min_score", 0.85)),
        "--preassoc-stale-competition-force-rewrite-recovery-memory-min-box-iou",
        str(variant.get("recovery_memory_min_box_iou", 0.80)),
        "--preassoc-stale-competition-force-rewrite-recovery-memory-min-challenger-alt-box-iou",
        str(variant.get("recovery_memory_min_challenger_alt_box_iou", 0.12)),
        "--preassoc-stale-competition-force-rewrite-recovery-memory-warmup-min-neighborhood-gain",
        str(variant.get("recovery_memory_warmup_min_neighborhood_gain", -0.08)),
        "--preassoc-stale-competition-force-rewrite-recovery-memory-bonus",
        str(variant.get("recovery_memory_bonus", 0.06)),
        "--preassoc-stale-competition-force-rewrite-recovery-memory-bonus-max-streak",
        str(variant.get("recovery_memory_bonus_max_streak", 2)),
        "--preassoc-stale-competition-force-rewrite-recovery-memory-gate-bonus",
        str(variant.get("recovery_memory_gate_bonus", 1e-4)),
        "--preassoc-stale-competition-force-rewrite-recovery-memory-anchor-min-raw-neighborhood-gain",
        str(variant.get("recovery_memory_anchor_min_raw_neighborhood_gain", 0.05)),
        "--preassoc-stale-competition-force-rewrite-recovery-memory-anchor-max-edge-deficit",
        str(variant.get("recovery_memory_anchor_max_edge_deficit", 0.35)),
        "--preassoc-stale-competition-force-rewrite-recovery-memory-extension-max-edge-deficit-delta",
        str(variant.get("recovery_memory_extension_max_edge_deficit_delta", 0.15)),
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
        acceptance_gate_checkpoint,
        "--preassoc-stale-competition-acceptance-gate-thresh",
        acceptance_gate_thresh,
    ]
    if recovery_anchor_gate_checkpoint:
        cmd.extend(
            [
                "--preassoc-stale-competition-recovery-anchor-gate-checkpoint",
                recovery_anchor_gate_checkpoint,
                "--preassoc-stale-competition-recovery-anchor-gate-thresh",
                recovery_anchor_gate_thresh,
            ]
        )
    if force_rewrite_recovery_anchor_override_enable:
        cmd.extend(
            [
                "--preassoc-stale-competition-force-rewrite-recovery-anchor-override-enable",
                "--preassoc-stale-competition-force-rewrite-recovery-anchor-min-raw-neighborhood-gain",
                force_rewrite_recovery_anchor_min_raw_neighborhood_gain,
            ]
        )
    if bool(variant.get("recovery_memory_enable", False)):
        cmd.append("--preassoc-stale-competition-force-rewrite-recovery-memory-enable")
    if bool(variant.get("recovery_memory_anchor_enable", False)):
        cmd.append("--preassoc-stale-competition-force-rewrite-recovery-memory-anchor-enable")
    return cmd


def wait_for_existing_run(summary_csv: Path, poll_sec: int = 20, timeout_sec: int = 3 * 3600) -> Tuple[str, str]:
    deadline = time.time() + float(timeout_sec)
    last_status = "missing"
    while time.time() < deadline:
        rows = read_rows(summary_csv)
        if rows:
            compare_rows = [row for row in rows if str(row.get("step", "")) == "compare"]
            if compare_rows:
                status = str(compare_rows[0].get("status", "") or "")
                last_status = status
                if status not in {"", "pending", "running"}:
                    return status, str(compare_rows[0].get("notes", "") or "")
        time.sleep(float(poll_sec))
    raise TimeoutError(f"Timed out waiting for {summary_csv}; last compare status={last_status}")


def record_decision_from_child(
    *,
    decision_rows: List[Dict[str, object]],
    decision_csv: Path,
    step: str,
    variant_name: str,
    child_root: Path,
    variant: Dict[str, object],
    notes: str,
) -> Dict[str, object]:
    metrics = read_metrics_delta(child_root / "metrics_delta.csv")
    runtime = read_runtime(child_root / "runtime_compare.csv")
    decision = {
        "step": step,
        "variant": variant_name,
        "status": "success",
        "out_dir": str(child_root.resolve()),
        "summary_csv": str((child_root / "summary.csv").resolve()),
        "metrics_delta_csv": str((child_root / "metrics_delta.csv").resolve()),
        "runtime_compare_csv": str((child_root / "runtime_compare.csv").resolve()),
        "acceptance_gate_checkpoint": str(variant.get("acceptance_gate_checkpoint", "")),
        "acceptance_gate_thresh": variant.get("acceptance_gate_thresh", ""),
        "recovery_anchor_gate_checkpoint": str(variant.get("recovery_anchor_gate_checkpoint", "")),
        "recovery_anchor_gate_thresh": variant.get("recovery_anchor_gate_thresh", ""),
        "force_rewrite_min_score": variant["force_rewrite_min_score"],
        "force_rewrite_min_box_iou": variant["force_rewrite_min_box_iou"],
        "force_rewrite_min_neighborhood_gain": variant["force_rewrite_min_neighborhood_gain"],
        "trapped_owner_min_neighborhood_gain": variant.get("trapped_owner_min_neighborhood_gain", ""),
        "reroute_ready_min_neighborhood_gain": variant.get("reroute_ready_min_neighborhood_gain", ""),
        "trapped_owner_negative_gain_min_challenger_alt_box_iou": variant.get(
            "trapped_owner_negative_gain_min_challenger_alt_box_iou",
            "",
        ),
        "keep_challenger_alt_weight": variant["keep_challenger_alt_weight"],
        "rewrite_owner_alt_weight": variant["rewrite_owner_alt_weight"],
        "shared_alt_penalty": variant["shared_alt_penalty"],
        "trapped_owner_bonus": variant.get("trapped_owner_bonus", ""),
        "reroute_ready_penalty": variant.get("reroute_ready_penalty", ""),
        "reroute_ready_min_box_iou": variant.get("reroute_ready_min_box_iou", ""),
        "force_rewrite_recovery_anchor_override_enable": int(
            bool(variant.get("force_rewrite_recovery_anchor_override_enable", False))
        ),
        "force_rewrite_recovery_anchor_min_raw_neighborhood_gain": variant.get(
            "force_rewrite_recovery_anchor_min_raw_neighborhood_gain",
            "",
        ),
        "recovery_memory_enable": int(bool(variant.get("recovery_memory_enable", False))),
        "recovery_memory_max_frame_gap": variant.get("recovery_memory_max_frame_gap", ""),
        "recovery_memory_min_score": variant.get("recovery_memory_min_score", ""),
        "recovery_memory_min_box_iou": variant.get("recovery_memory_min_box_iou", ""),
        "recovery_memory_min_challenger_alt_box_iou": variant.get(
            "recovery_memory_min_challenger_alt_box_iou",
            "",
        ),
        "recovery_memory_warmup_min_neighborhood_gain": variant.get(
            "recovery_memory_warmup_min_neighborhood_gain",
            "",
        ),
        "recovery_memory_bonus": variant.get("recovery_memory_bonus", ""),
        "recovery_memory_bonus_max_streak": variant.get("recovery_memory_bonus_max_streak", ""),
        "recovery_memory_gate_bonus": variant.get("recovery_memory_gate_bonus", ""),
        "recovery_memory_anchor_enable": int(bool(variant.get("recovery_memory_anchor_enable", False))),
        "recovery_memory_anchor_min_raw_neighborhood_gain": variant.get(
            "recovery_memory_anchor_min_raw_neighborhood_gain",
            "",
        ),
        "recovery_memory_anchor_max_edge_deficit": variant.get("recovery_memory_anchor_max_edge_deficit", ""),
        "recovery_memory_extension_max_edge_deficit_delta": variant.get(
            "recovery_memory_extension_max_edge_deficit_delta",
            "",
        ),
        "delta_HOTA": metrics["delta_HOTA"],
        "delta_AssA": metrics["delta_AssA"],
        "delta_IDF1": metrics["delta_IDF1"],
        "delta_MOTA": metrics["delta_MOTA"],
        "delta_IDs": metrics["delta_IDs"],
        "delta_Frag": metrics["delta_Frag"],
        "selected_matches": int(runtime["selected_matches"]),
        "force_rewrite_accepted_rows": int(runtime["force_rewrite_accepted_rows"]),
        "force_rewrite_recovery_anchor_override_rows": int(
            runtime["force_rewrite_recovery_anchor_override_rows"]
        ),
        "candidate_rows": int(runtime["candidate_rows"]),
        "decision": "",
        "notes": notes,
    }
    upsert_decision_row(decision_rows, decision)
    write_rows(decision_csv, DECISION_FIELDS, decision_rows)
    return decision


def main() -> int:
    args = parse_args()
    run_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else (REPO_ROOT / "outputs" / f"deep_ocsort_preassoc_force_rewrite_next2h_{timestamp_tag()}").resolve()
    )
    wait_summary_csv = Path(args.wait_summary_csv).expanduser().resolve()
    wait_run_root = Path(args.wait_run_root).expanduser().resolve()
    reuse_raw_from = Path(args.reuse_raw_from).expanduser().resolve()

    variants = [
        {
            "step": "wait_current_v2",
            "name": "wait_current_v2",
            "notes": "wait for the currently running neighborhood-v2 seq0090 experiment and record its outcome",
            "kind": "wait",
        },
        {
            "step": "seq0090_owner125",
            "name": "seq0090_owner125",
            "notes": "increase owner-alt rewrite weight while keeping low challenger-alt keep weight",
            "kind": "run",
            "seq_names": [args.seq_name],
            "force_rewrite_min_score": 0.65,
            "force_rewrite_min_box_iou": 0.55,
            "force_rewrite_min_neighborhood_gain": 0.0,
            "force_rewrite_max_age_gap": 200,
            "force_rewrite_max_owner_alt_det_box_iou": 0.50,
            "keep_challenger_alt_weight": 0.25,
            "rewrite_owner_alt_weight": 1.25,
            "shared_alt_penalty": 0.25,
        },
        {
            "step": "seq0090_keep050_owner125",
            "name": "seq0090_keep050_owner125",
            "notes": "raise challenger-alt keep weight to test whether the previous sweep relied on more conservative keep-plan scoring",
            "kind": "run",
            "seq_names": [args.seq_name],
            "force_rewrite_min_score": 0.65,
            "force_rewrite_min_box_iou": 0.55,
            "force_rewrite_min_neighborhood_gain": 0.0,
            "force_rewrite_max_age_gap": 200,
            "force_rewrite_max_owner_alt_det_box_iou": 0.50,
            "keep_challenger_alt_weight": 0.50,
            "rewrite_owner_alt_weight": 1.25,
            "shared_alt_penalty": 0.25,
        },
        {
            "step": "seq0090_owner125_gain005",
            "name": "seq0090_owner125_gain005",
            "notes": "keep the stronger owner-alt rewrite weighting but require slightly positive neighborhood gain",
            "kind": "run",
            "seq_names": [args.seq_name],
            "force_rewrite_min_score": 0.65,
            "force_rewrite_min_box_iou": 0.55,
            "force_rewrite_min_neighborhood_gain": 0.05,
            "force_rewrite_max_age_gap": 200,
            "force_rewrite_max_owner_alt_det_box_iou": 0.50,
            "keep_challenger_alt_weight": 0.25,
            "rewrite_owner_alt_weight": 1.25,
            "shared_alt_penalty": 0.25,
        },
        {
            "step": "dance3_best_confirm",
            "name": "dance3_best_confirm",
            "notes": "conditionally run a 3-sequence confirmation on the best neighborhood variant from this two-hour block",
            "kind": "confirm",
        },
    ]

    summary_csv = run_root / "summary.csv"
    decision_csv = run_root / "decision_summary.csv"
    logs_dir = run_root / "logs"
    runs_dir = run_root / "runs"

    queue_rows: List[Dict[str, object]] = []
    for variant in variants:
        step = str(variant["step"])
        if variant["kind"] == "wait":
            out_dir = wait_run_root
            child_summary = wait_summary_csv
        else:
            out_dir = (runs_dir / step).resolve()
            child_summary = out_dir / "summary.csv"
        queue_rows.append(
            {
                "step": step,
                "name": f"{run_root.name}_{step}",
                "status": "pending",
                "out_dir": str(out_dir),
                "summary_csv": str(child_summary.resolve()),
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
    overall_notes = "completed scheduled two-hour neighborhood rewrite queue"

    for variant in variants:
        step = str(variant["step"])
        started_at = now_iso()
        update_row(queue_rows, step, status="running", started_at=started_at)
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
        if step == str(variants[0]["step"]):
            append_registry(
                summary_csv,
                run_root,
                "running",
                "started scheduled two-hour neighborhood rewrite queue",
                args.registry_csv,
            )

        status = "success"
        finished_at = now_iso()
        notes = str(variant["notes"])

        try:
            if variant["kind"] == "wait":
                waited_status, waited_notes = wait_for_existing_run(wait_summary_csv)
                finished_at = now_iso()
                if waited_status != "success":
                    status = "failed"
                    notes = f"waited current run ended with status={waited_status} notes={waited_notes}"
                else:
                    record_decision_from_child(
                        decision_rows=decision_rows,
                        decision_csv=decision_csv,
                        step=step,
                        variant_name="current_v2_waited",
                        child_root=wait_run_root,
                        variant={
                            "force_rewrite_min_score": 0.65,
                            "force_rewrite_min_box_iou": 0.55,
                            "force_rewrite_min_neighborhood_gain": 0.0,
                            "keep_challenger_alt_weight": 0.25,
                            "rewrite_owner_alt_weight": 1.0,
                            "shared_alt_penalty": 0.25,
                        },
                        notes="waited for externally started neighborhood-v2 run and recorded its result",
                    )
                    metrics = read_metrics_delta(wait_run_root / "metrics_delta.csv")
                    runtime = read_runtime(wait_run_root / "runtime_compare.csv")
                    notes = (
                        f"current_v2 done delta_HOTA={metrics['delta_HOTA']:+.3f} "
                        f"delta_AssA={metrics['delta_AssA']:+.3f} "
                        f"selected_matches={int(runtime['selected_matches'])}"
                    )
            elif variant["kind"] == "run":
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
            else:
                successful_single_seq = [
                    row
                    for row in decision_rows
                    if str(row.get("status", "")) == "success"
                    and str(row.get("step", "")).startswith("seq0090_")
                ]
                if not successful_single_seq:
                    status = "cancelled"
                    notes = "no successful seq0090 neighborhood variants available for confirmation"
                else:
                    best = max(
                        successful_single_seq,
                        key=lambda row: (
                            float(row.get("delta_HOTA", 0.0) or 0.0),
                            float(row.get("delta_AssA", 0.0) or 0.0),
                            -abs(int(row.get("selected_matches", 0) or 0) - 9),
                        ),
                    )
                    if float(best.get("delta_HOTA", 0.0) or 0.0) < 0.03:
                        status = "cancelled"
                        notes = f"best seq0090 variant too weak for confirmation: {best.get('step')} delta_HOTA={float(best.get('delta_HOTA', 0.0) or 0.0):+.3f}"
                    else:
                        source_variant = None
                        for item in variants:
                            if str(item.get("step", "")) == str(best.get("step", "")):
                                source_variant = item
                                break
                        if source_variant is None:
                            status = "cancelled"
                            notes = f"could not recover source config for {best.get('step')}"
                        else:
                            out_dir = (runs_dir / step).resolve()
                            log_path = (logs_dir / f"{step}.log").resolve()
                            cmd = build_cmd(
                                out_dir=out_dir,
                                reuse_raw_from=reuse_raw_from,
                                seq_names=list(args.confirm_seq_names),
                                variant=source_variant,
                            )
                            rc = run_step(cmd, log_path, cwd=REPO_ROOT)
                            child_summary = out_dir / "summary.csv"
                            finished_at = now_iso()
                            if rc != 0:
                                status = "failed"
                                notes = f"confirm child return code {rc}"
                            else:
                                ensure_child_success(child_summary)
                                finished_at = child_finished_at(child_summary)
                                decision = record_decision_from_child(
                                    decision_rows=decision_rows,
                                    decision_csv=decision_csv,
                                    step=step,
                                    variant_name=f"confirm_{best.get('step')}",
                                    child_root=out_dir,
                                    variant=source_variant,
                                    notes=f"3-sequence confirm sourced from {best.get('step')}",
                                )
                                notes = (
                                    f"confirm {best.get('step')} delta_HOTA={float(decision['delta_HOTA']):+.3f} "
                                    f"delta_AssA={float(decision['delta_AssA']):+.3f} "
                                    f"selected_matches={int(decision['selected_matches'])}"
                                )
        except Exception as exc:
            status = "failed"
            finished_at = now_iso()
            notes = f"{variant['kind']} step failed: {exc}"

        update_row(queue_rows, step, status=status, finished_at=finished_at, notes=notes)
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

        if status == "failed":
            overall_status = "failed"
            overall_notes = f"{step} failed: {notes}"

    append_registry(summary_csv, run_root, overall_status, overall_notes, args.registry_csv)
    return 0 if overall_status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
