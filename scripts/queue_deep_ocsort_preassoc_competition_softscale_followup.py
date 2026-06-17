#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
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
    "metrics_compare_csv",
    "metrics_delta_csv",
    "per_sequence_csv",
    "runtime_compare_csv",
    "competition_HOTA",
    "competition_AssA",
    "competition_IDF1",
    "competition_MOTA",
    "competition_IDs",
    "competition_Frag",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDs",
    "delta_Frag",
    "seq0090_HOTA",
    "seq0090_AssA",
    "seq0090_IDF1",
    "seq0090_MOTA",
    "seq0090_IDs",
    "seq0090_Frag",
    "seq0090_delta_HOTA",
    "seq0090_delta_AssA",
    "seq0090_delta_IDF1",
    "candidate_rows",
    "biased_edges",
    "takeover_risk_rejected_rows",
    "owner_alt_biased_edges",
    "owner_alt_risk_rejected_rows",
    "selected_matches",
    "decision",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queued DanceTrack smoke follow-ups around the softscale stale-competition controller."
    )
    parser.add_argument("--out-root", default="")
    parser.add_argument(
        "--wait-run-root",
        default=str(
            REPO_ROOT
            / "outputs"
            / "deep_ocsort_local_contention_ranker_owneralt_bias030_softscale_margin010_edge005_force075_smoke_20260408_1"
        ),
    )
    parser.add_argument(
        "--reference-run-root",
        default=str(REPO_ROOT / "outputs" / "deep_ocsort_local_contention_ranker_owneralt_bias030_iou030_smoke_20260407_1"),
    )
    parser.add_argument(
        "--reuse-raw-from",
        default=str(REPO_ROOT / "outputs" / "deep_ocsort_local_contention_export_dance_best_20260406_1"),
    )
    parser.add_argument(
        "--seq-names",
        nargs="*",
        default=[
            "dancetrack0019",
            "dancetrack0047",
            "dancetrack0063",
            "dancetrack0090",
        ],
    )
    parser.add_argument("--poll-seconds", type=int, default=120)
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
        process = subprocess.run(cmd, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=False)
        handle.write(f"\n[finished_at] {now_iso()}\n")
        handle.write(f"[return_code] {process.returncode}\n")
    return int(process.returncode)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
        "scripts/queue_deep_ocsort_preassoc_competition_softscale_followup.py",
        "--dataset",
        "DanceTrack",
        "--split",
        "val",
        "--tracker-family",
        "deep_ocsort_preassoc_competition",
        "--variant",
        run_root.name,
        "--tag",
        "deep_ocsort_preassoc_competition_softscale_followup",
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def mark_running_rows_failed(rows: List[Dict[str, object]], summary_csv: Path, reason: str) -> None:
    finished_at = now_iso()
    changed = False
    for row in rows:
        status = str(row.get("status", ""))
        if status == "running":
            row["status"] = "failed"
            row["finished_at"] = finished_at
            row["notes"] = f"{row.get('notes', '')} | failed: {reason}".strip()
            changed = True
        elif status == "pending":
            row["status"] = "cancelled"
            row["finished_at"] = finished_at
            row["notes"] = f"{row.get('notes', '')} | cancelled_after_failure: {reason}".strip()
            changed = True
    if changed:
        write_rows(summary_csv, QUEUE_FIELDS, rows)


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


def dependency_process_running(wait_run_root: Path) -> bool:
    result = subprocess.run(
        ["ps", "-eo", "cmd"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    target = str(wait_run_root)
    if not target:
        return False
    for line in (result.stdout or "").splitlines():
        if target in line and "queue_deep_ocsort_preassoc_competition_softscale_followup.py" not in line:
            return True
    return False


def wait_for_dependency(wait_run_root: Path, poll_seconds: int) -> str:
    summary_csv = wait_run_root / "summary.csv"
    metrics_compare_csv = wait_run_root / "metrics_compare.csv"
    checks = 0
    while True:
        checks += 1
        process_running = dependency_process_running(wait_run_root)
        summary_rows = read_rows(summary_csv)
        summary_statuses = sorted({str(row.get("status", "")).strip() for row in summary_rows if row})
        if not process_running:
            if metrics_compare_csv.is_file():
                return f"dependency process exited after {checks} checks; metrics ready at {metrics_compare_csv}"
            if summary_rows and "running" not in summary_statuses and "pending" not in summary_statuses:
                return f"dependency process exited after {checks} checks; summary status={','.join(summary_statuses)}"
            return f"dependency process exited after {checks} checks; proceeding without further wait"
        time.sleep(max(30, int(poll_seconds)))


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


def read_metrics_compare(metrics_compare_csv: Path) -> Dict[str, float]:
    rows = read_rows(metrics_compare_csv)
    for row in rows:
        if str(row.get("name", "")) == "competition":
            return {
                "competition_HOTA": float(row.get("HOTA", 0.0) or 0.0),
                "competition_AssA": float(row.get("AssA", 0.0) or 0.0),
                "competition_IDF1": float(row.get("IDF1", 0.0) or 0.0),
                "competition_MOTA": float(row.get("MOTA", 0.0) or 0.0),
                "competition_IDs": float(row.get("IDs", 0.0) or 0.0),
                "competition_Frag": float(row.get("Frag", 0.0) or 0.0),
            }
    raise ValueError(f"Missing competition row in {metrics_compare_csv}")


def read_per_sequence(per_sequence_csv: Path, seq_name: str) -> Dict[str, float]:
    raw_row: Dict[str, str] | None = None
    comp_row: Dict[str, str] | None = None
    for row in read_rows(per_sequence_csv):
        if str(row.get("seq", "")) != seq_name:
            continue
        if str(row.get("name", "")) == "raw":
            raw_row = row
        elif str(row.get("name", "")) == "competition":
            comp_row = row
    if raw_row is None or comp_row is None:
        raise ValueError(f"Missing per-sequence raw/competition rows for {seq_name} in {per_sequence_csv}")
    comp_hota = float(comp_row.get("HOTA", 0.0) or 0.0)
    comp_assa = float(comp_row.get("AssA", 0.0) or 0.0)
    comp_idf1 = float(comp_row.get("IDF1", 0.0) or 0.0)
    return {
        "seq0090_HOTA": comp_hota,
        "seq0090_AssA": comp_assa,
        "seq0090_IDF1": comp_idf1,
        "seq0090_MOTA": float(comp_row.get("MOTA", 0.0) or 0.0),
        "seq0090_IDs": float(comp_row.get("IDs", 0.0) or 0.0),
        "seq0090_Frag": float(comp_row.get("Frag", 0.0) or 0.0),
        "seq0090_delta_HOTA": comp_hota - float(raw_row.get("HOTA", 0.0) or 0.0),
        "seq0090_delta_AssA": comp_assa - float(raw_row.get("AssA", 0.0) or 0.0),
        "seq0090_delta_IDF1": comp_idf1 - float(raw_row.get("IDF1", 0.0) or 0.0),
    }


def read_runtime(runtime_compare_csv: Path) -> Dict[str, float]:
    rows = read_rows(runtime_compare_csv)
    for row in rows:
        if str(row.get("name", "")) == "competition":
            return {
                "candidate_rows": float(row.get("preassoc_stale_competition_candidate_rows", 0.0) or 0.0),
                "biased_edges": float(row.get("preassoc_stale_competition_biased_edges", 0.0) or 0.0),
                "takeover_risk_rejected_rows": float(row.get("preassoc_stale_competition_takeover_risk_rejected_rows", 0.0) or 0.0),
                "owner_alt_biased_edges": float(row.get("preassoc_stale_competition_owner_alt_biased_edges", 0.0) or 0.0),
                "owner_alt_risk_rejected_rows": float(row.get("preassoc_stale_competition_owner_alt_risk_rejected_rows", 0.0) or 0.0),
                "selected_matches": float(row.get("preassoc_stale_competition_selected_matches", 0.0) or 0.0),
            }
    raise ValueError(f"Missing competition runtime row in {runtime_compare_csv}")


def build_variants() -> List[Dict[str, object]]:
    return [
        {
            "step": "softscale_force060",
            "suffix": "softscale_margin010_edge005_force060",
            "takeover_soft_margin_floor": 0.10,
            "takeover_soft_edge_advantage_floor": 0.05,
            "takeover_min_force_risk_scale": 0.60,
            "owner_alt_bias": 0.30,
            "owner_alt_min_score": 0.35,
            "owner_alt_min_box_iou": 0.30,
            "owner_edge_penalty": 0.05,
            "notes": "降低硬接管风险阈值，检查当前 0.75 是否压得过狠",
        },
        {
            "step": "softscale_soft005_force075",
            "suffix": "softscale_margin005_edge000_force075",
            "takeover_soft_margin_floor": 0.05,
            "takeover_soft_edge_advantage_floor": 0.00,
            "takeover_min_force_risk_scale": 0.75,
            "owner_alt_bias": 0.30,
            "owner_alt_min_score": 0.35,
            "owner_alt_min_box_iou": 0.30,
            "owner_edge_penalty": 0.05,
            "notes": "减轻连续缩放强度，保留高风险硬接管门槛",
        },
        {
            "step": "softscale_soft005_force060",
            "suffix": "softscale_margin005_edge000_force060",
            "takeover_soft_margin_floor": 0.05,
            "takeover_soft_edge_advantage_floor": 0.00,
            "takeover_min_force_risk_scale": 0.60,
            "owner_alt_bias": 0.30,
            "owner_alt_min_score": 0.35,
            "owner_alt_min_box_iou": 0.30,
            "owner_edge_penalty": 0.05,
            "notes": "同时放松软缩放与硬接管门槛，测试是否只是当前门控过严",
        },
        {
            "step": "fixed_owneralt025",
            "suffix": "fixed_owneralt_bias025_iou030",
            "takeover_soft_margin_floor": -1.0,
            "takeover_soft_edge_advantage_floor": -1.0,
            "takeover_min_force_risk_scale": -1.0,
            "owner_alt_bias": 0.25,
            "owner_alt_min_score": 0.35,
            "owner_alt_min_box_iou": 0.30,
            "owner_edge_penalty": 0.05,
            "notes": "固定强版控制组，只下调 owner_alt 偏置，单独验证 0090 负迁移是否来自重路由压力",
        },
    ]


def build_eval_cmd(out_dir: Path, reuse_raw_from: Path, seq_names: List[str], variant: Dict[str, object]) -> List[str]:
    local_contention_jsonl = out_dir / "local_contention_units.jsonl"
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_deep_ocsort_preassoc_competition_dataset_eval.py"),
        "--benchmark",
        "DanceTrack",
        "--seq-names",
        *seq_names,
        "--reuse-raw-from",
        str(reuse_raw_from),
        "--out-root",
        str(out_dir),
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
        "--preassoc-stale-competition-owner-max-hits",
        "12",
        "--preassoc-stale-competition-owner-edge-penalty",
        str(variant["owner_edge_penalty"]),
        "--preassoc-stale-competition-takeover-soft-margin-floor",
        str(variant["takeover_soft_margin_floor"]),
        "--preassoc-stale-competition-takeover-soft-edge-advantage-floor",
        str(variant["takeover_soft_edge_advantage_floor"]),
        "--preassoc-stale-competition-takeover-min-force-risk-scale",
        str(variant["takeover_min_force_risk_scale"]),
        "--preassoc-stale-competition-owner-alt-det-bias",
        str(variant["owner_alt_bias"]),
        "--preassoc-stale-competition-owner-alt-det-min-score",
        str(variant["owner_alt_min_score"]),
        "--preassoc-stale-competition-owner-alt-det-min-box-iou",
        str(variant["owner_alt_min_box_iou"]),
        "--preassoc-stale-competition-max-owner-edge-deficit",
        "-1.0",
        "--preassoc-stale-competition-force-owner-edge-deficit-arg",
        "--preassoc-stale-competition-block-owner-on-reclaim",
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
        "--local-contention-export-jsonl",
        str(local_contention_jsonl),
        "--competition-track-max-frames-per-batch",
        "2500",
    ]


def main() -> None:
    args = parse_args()
    queue_name = (
        Path(args.out_root).name
        if args.out_root
        else f"deep_ocsort_local_contention_softscale_followup_queue_{timestamp_tag()}"
    )
    out_root = Path(args.out_root).resolve() if args.out_root else (REPO_ROOT / "outputs" / queue_name).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir = out_root / "logs"
    summary_csv = out_root / "summary.csv"
    decision_csv = out_root / "decision.csv"
    registry_csv = str(Path(args.registry_csv).resolve())
    wait_run_root = Path(args.wait_run_root).resolve()
    reference_run_root = Path(args.reference_run_root).resolve()
    reuse_raw_from = Path(args.reuse_raw_from).resolve()
    variants = build_variants()

    queue_rows: List[Dict[str, object]] = [
        {
            "step": "wait_dependency",
            "name": f"{queue_name}_wait_dependency",
            "status": "pending",
            "out_dir": str(wait_run_root),
            "summary_csv": str(wait_run_root / "summary.csv"),
            "log_path": str(logs_dir / "wait_dependency.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"wait current softscale run to release resources: {wait_run_root}",
        }
    ]
    decision_rows: List[Dict[str, object]] = []

    reference_metrics_compare_csv = reference_run_root / "metrics_compare.csv"
    reference_metrics_delta_csv = reference_run_root / "metrics_delta.csv"
    reference_per_sequence_csv = reference_run_root / "per_sequence_metrics.csv"
    reference_runtime_compare_csv = reference_run_root / "runtime_compare.csv"
    ref_comp = read_metrics_compare(reference_metrics_compare_csv)
    ref_delta = read_metrics_delta(reference_metrics_delta_csv)
    ref_seq0090 = read_per_sequence(reference_per_sequence_csv, "dancetrack0090")
    ref_runtime = read_runtime(reference_runtime_compare_csv)
    decision_rows.append(
        {
            "step": "reference_best_fixed",
            "variant": "best fixed owner_alt=0.30 iou=0.30",
            "status": "success",
            "out_dir": str(reference_run_root),
            "summary_csv": str(reference_run_root / "summary.csv"),
            "metrics_compare_csv": str(reference_metrics_compare_csv),
            "metrics_delta_csv": str(reference_metrics_delta_csv),
            "per_sequence_csv": str(reference_per_sequence_csv),
            "runtime_compare_csv": str(reference_runtime_compare_csv),
            **ref_comp,
            **ref_delta,
            **ref_seq0090,
            **ref_runtime,
            "decision": "reference",
            "notes": "当前已知最强固定版本，用来对齐后续跟进实验",
        }
    )

    for variant in variants:
        step = str(variant["step"])
        child_out = out_root / step
        queue_rows.append(
            {
                "step": step,
                "name": child_out.name,
                "status": "pending",
                "out_dir": str(child_out),
                "summary_csv": str(child_out / "summary.csv"),
                "log_path": str(logs_dir / f"{step}.log"),
                "started_at": "",
                "finished_at": "",
                "notes": str(variant["notes"]),
            }
        )
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
    write_rows(decision_csv, DECISION_FIELDS, decision_rows)
    append_registry(summary_csv, out_root, "running", "softscale follow-up queue started", registry_csv)

    try:
        update_row(queue_rows, "wait_dependency", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
        wait_note = wait_for_dependency(wait_run_root, int(args.poll_seconds))
        wait_log = logs_dir / "wait_dependency.log"
        write_text(wait_log, wait_note + "\n")
        update_row(
            queue_rows,
            "wait_dependency",
            status="success",
            finished_at=now_iso(),
            log_path=str(wait_log),
            notes=wait_note,
        )
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

        for variant in variants:
            step = str(variant["step"])
            child_out = out_root / step
            child_summary_csv = child_out / "summary.csv"
            metrics_compare_csv = child_out / "metrics_compare.csv"
            metrics_delta_csv = child_out / "metrics_delta.csv"
            per_sequence_csv = child_out / "per_sequence_metrics.csv"
            runtime_compare_csv = child_out / "runtime_compare.csv"
            log_path = logs_dir / f"{step}.log"

            update_row(queue_rows, step, status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

            return_code = run_step(
                build_eval_cmd(child_out, reuse_raw_from, list(args.seq_names), variant),
                log_path,
                cwd=REPO_ROOT,
            )
            if return_code != 0:
                update_row(queue_rows, step, status="failed", finished_at=now_iso(), log_path=str(log_path))
                write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
                raise RuntimeError(f"Step failed: {step} return_code={return_code}")

            ensure_child_success(child_summary_csv)
            metrics_compare = read_metrics_compare(metrics_compare_csv)
            metrics_delta = read_metrics_delta(metrics_delta_csv)
            seq0090 = read_per_sequence(per_sequence_csv, "dancetrack0090")
            runtime = read_runtime(runtime_compare_csv)
            decision = "candidate" if (
                metrics_compare["competition_HOTA"] >= ref_comp["competition_HOTA"]
                or seq0090["seq0090_delta_HOTA"] > ref_seq0090["seq0090_delta_HOTA"]
            ) else "followup_only"
            note = (
                f"HOTA={metrics_compare['competition_HOTA']:.3f} "
                f"IDF1={metrics_compare['competition_IDF1']:.3f} "
                f"0090_dHOTA={seq0090['seq0090_delta_HOTA']:.3f} "
                f"selected={int(round(runtime['selected_matches']))}"
            )

            update_row(
                queue_rows,
                step,
                status="success",
                finished_at=child_finished_at(child_summary_csv),
                out_dir=str(child_out),
                summary_csv=str(child_summary_csv),
                log_path=str(log_path),
                notes=note,
            )
            upsert_decision_row(
                decision_rows,
                {
                    "step": step,
                    "variant": str(variant["suffix"]),
                    "status": "success",
                    "out_dir": str(child_out),
                    "summary_csv": str(child_summary_csv),
                    "metrics_compare_csv": str(metrics_compare_csv),
                    "metrics_delta_csv": str(metrics_delta_csv),
                    "per_sequence_csv": str(per_sequence_csv),
                    "runtime_compare_csv": str(runtime_compare_csv),
                    **metrics_compare,
                    **metrics_delta,
                    **seq0090,
                    **runtime,
                    "decision": decision,
                    "notes": str(variant["notes"]),
                },
            )
            write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
            write_rows(decision_csv, DECISION_FIELDS, decision_rows)

        append_registry(summary_csv, out_root, "success", "softscale follow-up queue completed", registry_csv)
    except Exception as exc:
        mark_running_rows_failed(queue_rows, summary_csv, str(exc))
        append_registry(summary_csv, out_root, "failed", f"softscale follow-up queue failed: {exc}", registry_csv)
        raise


if __name__ == "__main__":
    main()
