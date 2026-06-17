#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


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
    "force_rewrite_min_score",
    "force_rewrite_min_box_iou",
    "force_rewrite_max_age_gap",
    "force_rewrite_max_owner_alt_det_box_iou",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDs",
    "delta_Frag",
    "selected_matches",
    "candidate_rows",
    "force_rewrite_accepted_rows",
    "acceptance_gate_accepted_rows",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recorded single-sequence calibration queue for the new pre-association force-rewrite branch."
    )
    parser.add_argument("--out-root", default="")
    parser.add_argument(
        "--reuse-raw-from",
        default=str(REPO_ROOT / "outputs" / "deep_ocsort_preassoc_acceptgate_smoke_20260408_2"),
    )
    parser.add_argument("--seq-names", nargs="*", default=["dancetrack0090"])
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
        "scripts/queue_deep_ocsort_preassoc_force_rewrite_calibration.py",
        "--dataset",
        "DanceTrack",
        "--split",
        "val",
        "--tracker-family",
        "deep_ocsort_preassoc_force_rewrite",
        "--variant",
        run_root.name,
        "--tag",
        "deep_ocsort_preassoc_force_rewrite_calibration",
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
                "acceptance_gate_accepted_rows": float(
                    row.get("preassoc_stale_competition_acceptance_gate_accepted_rows", 0.0) or 0.0
                ),
            }
    raise ValueError(f"Missing competition runtime row in {runtime_compare_csv}")


def build_dataset_eval_cmd(*, out_dir: Path, reuse_raw_from: Path, seq_names: List[str], variant: Dict[str, object]) -> List[str]:
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
        str(variant["max_owner_edge_deficit"]),
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
        "--preassoc-stale-competition-force-rewrite-min-box-iou",
        str(variant["force_rewrite_min_box_iou"]),
        "--preassoc-stale-competition-force-rewrite-max-age-gap",
        str(variant["force_rewrite_max_age_gap"]),
        "--preassoc-stale-competition-force-rewrite-max-owner-alt-det-box-iou",
        str(variant["force_rewrite_max_owner_alt_det_box_iou"]),
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
    ]


def main() -> int:
    args = parse_args()
    run_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else (REPO_ROOT / "outputs" / f"deep_ocsort_preassoc_force_rewrite_calib_{timestamp_tag()}").resolve()
    )
    reuse_raw_from = Path(args.reuse_raw_from).expanduser().resolve()
    seq_names = list(args.seq_names)

    variants = [
        {
            "step": "seq0090_i070_age200_alt050_s065",
            "force_rewrite_min_score": 0.65,
            "force_rewrite_min_box_iou": 0.70,
            "force_rewrite_max_age_gap": 200,
            "force_rewrite_max_owner_alt_det_box_iou": 0.50,
            "max_owner_edge_deficit": -1.0,
            "notes": "中等收紧，先砍掉超大年龄差和 owner 备选过强的接管",
        },
        {
            "step": "seq0090_i070_age160_alt050_s070",
            "force_rewrite_min_score": 0.70,
            "force_rewrite_min_box_iou": 0.70,
            "force_rewrite_max_age_gap": 160,
            "force_rewrite_max_owner_alt_det_box_iou": 0.50,
            "max_owner_edge_deficit": -1.0,
            "notes": "进一步收紧年龄差和最低重写分数",
        },
        {
            "step": "seq0090_i075_age120_alt050_s070",
            "force_rewrite_min_score": 0.70,
            "force_rewrite_min_box_iou": 0.75,
            "force_rewrite_max_age_gap": 120,
            "force_rewrite_max_owner_alt_det_box_iou": 0.50,
            "max_owner_edge_deficit": -1.0,
            "notes": "更强调几何可靠性，目标是让强制接管数量明显回落",
        },
        {
            "step": "seq0090_i070_age200_alt035_s065",
            "force_rewrite_min_score": 0.65,
            "force_rewrite_min_box_iou": 0.70,
            "force_rewrite_max_age_gap": 200,
            "force_rewrite_max_owner_alt_det_box_iou": 0.35,
            "max_owner_edge_deficit": -1.0,
            "notes": "只额外压低 owner 备选框重叠，测试 owner 备选过强是否是误接管主因",
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
    overall_notes = "all force-rewrite calibration variants completed"
    failed_step = ""

    for variant in variants:
        step = str(variant["step"])
        out_dir = (runs_dir / step).resolve()
        log_path = (logs_dir / f"{step}.log").resolve()
        started_at = now_iso()
        update_row(queue_rows, step, status="running", started_at=started_at)
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

        cmd = build_dataset_eval_cmd(
            out_dir=out_dir,
            reuse_raw_from=reuse_raw_from,
            seq_names=seq_names,
            variant=variant,
        )
        rc = run_step(cmd, log_path, cwd=REPO_ROOT)
        child_summary_csv = out_dir / "summary.csv"
        metrics_delta_csv = out_dir / "metrics_delta.csv"
        runtime_compare_csv = out_dir / "runtime_compare.csv"
        finished_at = now_iso()
        status = "failed"
        notes = f"child return code {rc}"
        metrics = {
            "delta_HOTA": 0.0,
            "delta_AssA": 0.0,
            "delta_IDF1": 0.0,
            "delta_MOTA": 0.0,
            "delta_IDs": 0.0,
            "delta_Frag": 0.0,
        }
        runtime = {
            "candidate_rows": 0.0,
            "selected_matches": 0.0,
            "force_rewrite_accepted_rows": 0.0,
            "acceptance_gate_accepted_rows": 0.0,
        }

        if rc == 0:
            try:
                ensure_child_success(child_summary_csv)
                finished_at = child_finished_at(child_summary_csv)
                metrics = read_metrics_delta(metrics_delta_csv)
                runtime = read_runtime(runtime_compare_csv)
                status = "success"
                notes = (
                    f"delta_HOTA={metrics['delta_HOTA']:+.3f} "
                    f"delta_IDF1={metrics['delta_IDF1']:+.3f} "
                    f"selected_matches={int(runtime['selected_matches'])} "
                    f"force_rewrite_accepted={int(runtime['force_rewrite_accepted_rows'])}"
                )
            except Exception as exc:
                status = "failed"
                notes = f"post-check failed: {exc}"

        update_row(queue_rows, step, status=status, finished_at=finished_at, notes=notes)
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

        decision_row = {
            "step": step,
            "variant": step,
            "status": status,
            "out_dir": str(out_dir),
            "summary_csv": str(child_summary_csv.resolve()),
            "metrics_delta_csv": str(metrics_delta_csv.resolve()),
            "runtime_compare_csv": str(runtime_compare_csv.resolve()),
            "force_rewrite_min_score": variant["force_rewrite_min_score"],
            "force_rewrite_min_box_iou": variant["force_rewrite_min_box_iou"],
            "force_rewrite_max_age_gap": variant["force_rewrite_max_age_gap"],
            "force_rewrite_max_owner_alt_det_box_iou": variant["force_rewrite_max_owner_alt_det_box_iou"],
            "delta_HOTA": metrics["delta_HOTA"],
            "delta_AssA": metrics["delta_AssA"],
            "delta_IDF1": metrics["delta_IDF1"],
            "delta_MOTA": metrics["delta_MOTA"],
            "delta_IDs": metrics["delta_IDs"],
            "delta_Frag": metrics["delta_Frag"],
            "selected_matches": int(runtime["selected_matches"]),
            "candidate_rows": int(runtime["candidate_rows"]),
            "force_rewrite_accepted_rows": int(runtime["force_rewrite_accepted_rows"]),
            "acceptance_gate_accepted_rows": int(runtime["acceptance_gate_accepted_rows"]),
            "notes": notes,
        }
        upsert_decision_row(decision_rows, decision_row)
        write_rows(decision_csv, DECISION_FIELDS, decision_rows)

        if status != "success":
            overall_status = "failed"
            overall_notes = f"{step} failed: {notes}"
            failed_step = step
            break

    if failed_step:
        cancelled_at = now_iso()
        for row in queue_rows:
            if str(row.get("status", "")) == "pending":
                row["status"] = "cancelled"
                row["finished_at"] = cancelled_at
                row["notes"] = f"queue stopped after {failed_step} failed"
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

    append_registry(summary_csv, run_root, overall_status, overall_notes, args.registry_csv)
    return 0 if overall_status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
