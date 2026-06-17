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
    "kind",
    "benchmark",
    "seq_label",
    "gate_thresh",
    "force_risk_scale",
    "status",
    "out_dir",
    "summary_csv",
    "runtime_compare_csv",
    "metrics_delta_csv",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDs",
    "delta_Frag",
    "selected_matches",
    "forced_gate_rejected_rows",
    "takeover_risk_rejected_rows",
    "acceptance_gate_scored_rows",
    "acceptance_gate_rejected_rows",
    "acceptance_gate_accepted_rows",
    "notes",
]

COMMON_FLAGS = [
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
    "0.20",
    "--preassoc-stale-competition-block-owner-on-reclaim",
    "--local-contention-ranker-checkpoint",
    "outputs/local_contention_ranker_mot17_mot20_dance_seqholdout_20260406_1/model.pt",
    "--local-contention-ranker-thresh",
    "0.99",
    "--local-contention-ranker-bias",
    "0.0",
    "--local-contention-ranker-min-margin-to-second",
    "0.05",
    "--local-contention-ranker-margin-bias",
    "0.5",
    "--preassoc-stale-competition-acceptance-gate-checkpoint",
    "outputs/local_contention_acceptance_gate_mot17_mot20_dance_seqholdout_20260408_1/best.pt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queued 6h-7h experiment batch for force-gated Deep-OC-SORT pre-association competition."
    )
    parser.add_argument("--out-root", default="")
    parser.add_argument(
        "--seed-run-root",
        default="outputs/deep_ocsort_preassoc_acceptgate_forcegatecalib_seq0090_thresh09999_20260409_1",
        help="existing calibration run root to wait on before the queue begins",
    )
    parser.add_argument("--poll-sec", type=int, default=60)
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
        "scripts/queue_deep_ocsort_forcegate_6h.py",
        "--dataset",
        "DanceTrack",
        "--split",
        "val",
        "--tracker-family",
        "deep_ocsort_forcegate_queue",
        "--variant",
        run_root.name,
        "--tag",
        "deep_ocsort_forcegate_6h",
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


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


def ensure_child_success(summary_csv: Path) -> None:
    rows = read_rows(summary_csv)
    if not rows:
        raise FileNotFoundError(f"Missing child summary rows: {summary_csv}")
    statuses = {str(row.get("status", "")).strip() for row in rows}
    if statuses != {"success"}:
        raise RuntimeError(f"Unexpected child status in {summary_csv}: {sorted(statuses)}")


def wait_existing_run(summary_csv: Path, *, poll_sec: int, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[started_at] {now_iso()}\n")
        handle.write(f"[summary_csv] {summary_csv}\n")
        handle.write(f"[poll_sec] {poll_sec}\n\n")
        handle.flush()
        while True:
            rows = read_rows(summary_csv)
            if rows:
                statuses = {str(row.get("status", "")).strip() for row in rows}
                handle.write(f"[heartbeat_at] {now_iso()} statuses={sorted(statuses)}\n")
                handle.flush()
                if "failed" in statuses:
                    raise RuntimeError(f"Seed run failed: {summary_csv}")
                if statuses == {"success"}:
                    break
            else:
                handle.write(f"[heartbeat_at] {now_iso()} statuses=[]\n")
                handle.flush()
            time.sleep(max(5, int(poll_sec)))
        handle.write(f"\n[finished_at] {now_iso()}\n")
        handle.write("[status] success\n")


def read_runtime(runtime_compare_csv: Path) -> Dict[str, float]:
    rows = read_rows(runtime_compare_csv)
    for row in rows:
        if str(row.get("name", "")) == "competition":
            return {
                "selected_matches": float(row.get("preassoc_stale_competition_selected_matches", 0) or 0),
                "forced_gate_rejected_rows": float(row.get("preassoc_stale_competition_forced_gate_rejected_rows", 0) or 0),
                "takeover_risk_rejected_rows": float(row.get("preassoc_stale_competition_takeover_risk_rejected_rows", 0) or 0),
                "acceptance_gate_scored_rows": float(row.get("preassoc_stale_competition_acceptance_gate_scored_rows", 0) or 0),
                "acceptance_gate_rejected_rows": float(row.get("preassoc_stale_competition_acceptance_gate_rejected_rows", 0) or 0),
                "acceptance_gate_accepted_rows": float(row.get("preassoc_stale_competition_acceptance_gate_accepted_rows", 0) or 0),
            }
    raise ValueError(f"Missing competition runtime row in {runtime_compare_csv}")


def read_delta(metrics_delta_csv: Path) -> Dict[str, float]:
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


def build_child_cmd(
    *,
    benchmark: str,
    seqs: List[str],
    out_root: Path,
    gate_thresh: str,
    force_risk_scale: str,
    reuse_raw_from: str,
) -> List[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_deep_ocsort_preassoc_competition_dataset_eval.py"),
        "--benchmark",
        benchmark,
        "--seq-names",
        *seqs,
        "--out-root",
        str(out_root),
        "--preassoc-stale-competition-takeover-min-force-risk-scale",
        str(force_risk_scale),
        "--preassoc-stale-competition-acceptance-gate-thresh",
        str(gate_thresh),
        "--local-contention-export-jsonl",
        str(out_root / "local_contention_units.jsonl"),
        *COMMON_FLAGS,
    ]
    if str(reuse_raw_from).strip():
        cmd.extend(["--reuse-raw-from", str(reuse_raw_from)])
    return cmd


def collect_decision_row(
    *,
    step: str,
    kind: str,
    benchmark: str,
    seqs: List[str],
    gate_thresh: str,
    force_risk_scale: str,
    out_dir: Path,
) -> Dict[str, object]:
    runtime_compare_csv = out_dir / "runtime_compare.csv"
    metrics_delta_csv = out_dir / "metrics_delta.csv"
    runtime = read_runtime(runtime_compare_csv)
    delta = read_delta(metrics_delta_csv)
    return {
        "step": step,
        "kind": kind,
        "benchmark": benchmark,
        "seq_label": "|".join(seqs),
        "gate_thresh": str(gate_thresh),
        "force_risk_scale": str(force_risk_scale),
        "status": "success",
        "out_dir": str(out_dir),
        "summary_csv": str(out_dir / "summary.csv"),
        "runtime_compare_csv": str(runtime_compare_csv),
        "metrics_delta_csv": str(metrics_delta_csv),
        "delta_HOTA": delta["delta_HOTA"],
        "delta_AssA": delta["delta_AssA"],
        "delta_IDF1": delta["delta_IDF1"],
        "delta_MOTA": delta["delta_MOTA"],
        "delta_IDs": delta["delta_IDs"],
        "delta_Frag": delta["delta_Frag"],
        "selected_matches": int(runtime["selected_matches"]),
        "forced_gate_rejected_rows": int(runtime["forced_gate_rejected_rows"]),
        "takeover_risk_rejected_rows": int(runtime["takeover_risk_rejected_rows"]),
        "acceptance_gate_scored_rows": int(runtime["acceptance_gate_scored_rows"]),
        "acceptance_gate_rejected_rows": int(runtime["acceptance_gate_rejected_rows"]),
        "acceptance_gate_accepted_rows": int(runtime["acceptance_gate_accepted_rows"]),
        "notes": "",
    }


def find_row(rows: List[Dict[str, object]], step: str) -> Dict[str, object]:
    for row in rows:
        if str(row.get("step", "")) == step:
            return row
    raise KeyError(f"Missing queue row: {step}")


def upsert_decision_row(rows: List[Dict[str, object]], new_row: Dict[str, object]) -> None:
    step = str(new_row.get("step", ""))
    for index, row in enumerate(rows):
        if str(row.get("step", "")) == step:
            rows[index] = dict(new_row)
            return
    rows.append(dict(new_row))


def init_summary_rows(
    *,
    steps: List[Dict[str, object]],
    queue_name: str,
    logs_dir: Path,
    seed_summary_csv: Path,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in steps:
        step = str(item["step"])
        rows.append(
            {
                "step": step,
                "name": f"{queue_name}_{step}",
                "status": "pending",
                "out_dir": str(item["out_dir"]),
                "summary_csv": str(Path(str(item["out_dir"])) / "summary.csv")
                if str(item["kind"]) == "run"
                else str(seed_summary_csv),
                "log_path": str(logs_dir / f"{step}.log"),
                "started_at": "",
                "finished_at": "",
                "notes": str(item["notes"]),
            }
        )
    return rows


def child_run_is_success(out_dir: Path) -> bool:
    child_summary_csv = out_dir / "summary.csv"
    runtime_compare_csv = out_dir / "runtime_compare.csv"
    metrics_delta_csv = out_dir / "metrics_delta.csv"
    if not child_summary_csv.is_file() or not runtime_compare_csv.is_file() or not metrics_delta_csv.is_file():
        return False
    rows = read_rows(child_summary_csv)
    if not rows:
        return False
    statuses = {str(row.get("status", "")).strip() for row in rows}
    return statuses == {"success"}


def main() -> None:
    args = parse_args()
    queue_name = Path(args.out_root).name if args.out_root else f"deep_ocsort_forcegate_6h_{timestamp_tag()}"
    out_root = Path(args.out_root).resolve() if args.out_root else (REPO_ROOT / "outputs" / queue_name).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir = out_root / "logs"
    summary_csv = out_root / "summary.csv"
    decision_csv = out_root / "decision.csv"
    registry_csv = str(Path(args.registry_csv).resolve())

    seed_run_root = Path(args.seed_run_root)
    if not seed_run_root.is_absolute():
        seed_run_root = (REPO_ROOT / seed_run_root).resolve()
    seed_summary_csv = seed_run_root / "summary.csv"

    steps = [
        {
            "step": "wait_seed_seq0090_t09999",
            "kind": "wait",
            "benchmark": "DanceTrack",
            "seqs": ["dancetrack0090"],
            "gate_thresh": "0.9999",
            "force_risk_scale": "0.6",
            "reuse_raw_from": "",
            "out_dir": str(seed_run_root),
            "notes": "wait for current seq0090 calibration t=0.9999 r=0.6 to finish",
        },
        {
            "step": "calib_seq0090_t09995",
            "kind": "run",
            "benchmark": "DanceTrack",
            "seqs": ["dancetrack0090"],
            "gate_thresh": "0.99995",
            "force_risk_scale": "0.6",
            "reuse_raw_from": "outputs/deep_ocsort_preassoc_acceptgate_smoke_20260408_2",
            "out_dir": str(out_root / "runs" / "calib_seq0090_t09995"),
            "notes": "single-sequence calibration around high-confidence gate",
        },
        {
            "step": "calib_seq0090_t0995",
            "kind": "run",
            "benchmark": "DanceTrack",
            "seqs": ["dancetrack0090"],
            "gate_thresh": "0.9995",
            "force_risk_scale": "0.6",
            "reuse_raw_from": "outputs/deep_ocsort_preassoc_acceptgate_smoke_20260408_2",
            "out_dir": str(out_root / "runs" / "calib_seq0090_t0995"),
            "notes": "relax gate to test whether effective takeovers reappear on seq0090",
        },
        {
            "step": "dance3_t09995_r060",
            "kind": "run",
            "benchmark": "DanceTrack",
            "seqs": ["dancetrack0081", "dancetrack0090", "dancetrack0094"],
            "gate_thresh": "0.99995",
            "force_risk_scale": "0.6",
            "reuse_raw_from": "outputs/deep_ocsort_preassoc_acceptgate_smoke_20260408_2",
            "out_dir": str(out_root / "runs" / "dance3_t09995_r060"),
            "notes": "3-sequence smoke with stricter gate and current force risk scale",
        },
        {
            "step": "dance3_t0995_r060",
            "kind": "run",
            "benchmark": "DanceTrack",
            "seqs": ["dancetrack0081", "dancetrack0090", "dancetrack0094"],
            "gate_thresh": "0.9995",
            "force_risk_scale": "0.6",
            "reuse_raw_from": "outputs/deep_ocsort_preassoc_acceptgate_smoke_20260408_2",
            "out_dir": str(out_root / "runs" / "dance3_t0995_r060"),
            "notes": "3-sequence smoke with looser gate and current force risk scale",
        },
        {
            "step": "dance3_t09995_r075",
            "kind": "run",
            "benchmark": "DanceTrack",
            "seqs": ["dancetrack0081", "dancetrack0090", "dancetrack0094"],
            "gate_thresh": "0.99995",
            "force_risk_scale": "0.75",
            "reuse_raw_from": "outputs/deep_ocsort_preassoc_acceptgate_smoke_20260408_2",
            "out_dir": str(out_root / "runs" / "dance3_t09995_r075"),
            "notes": "test whether tighter force-risk scale stabilizes strict gate",
        },
        {
            "step": "dance3_t0995_r075",
            "kind": "run",
            "benchmark": "DanceTrack",
            "seqs": ["dancetrack0081", "dancetrack0090", "dancetrack0094"],
            "gate_thresh": "0.9995",
            "force_risk_scale": "0.75",
            "reuse_raw_from": "outputs/deep_ocsort_preassoc_acceptgate_smoke_20260408_2",
            "out_dir": str(out_root / "runs" / "dance3_t0995_r075"),
            "notes": "test whether tighter force-risk scale can tame a looser gate",
        },
        {
            "step": "dance3_t09995_r050",
            "kind": "run",
            "benchmark": "DanceTrack",
            "seqs": ["dancetrack0081", "dancetrack0090", "dancetrack0094"],
            "gate_thresh": "0.99995",
            "force_risk_scale": "0.5",
            "reuse_raw_from": "outputs/deep_ocsort_preassoc_acceptgate_smoke_20260408_2",
            "out_dir": str(out_root / "runs" / "dance3_t09995_r050"),
            "notes": "test whether relaxed force-risk scale lets strict gate regain useful actions",
        },
        {
            "step": "dance3_t0995_r050",
            "kind": "run",
            "benchmark": "DanceTrack",
            "seqs": ["dancetrack0081", "dancetrack0090", "dancetrack0094"],
            "gate_thresh": "0.9995",
            "force_risk_scale": "0.5",
            "reuse_raw_from": "outputs/deep_ocsort_preassoc_acceptgate_smoke_20260408_2",
            "out_dir": str(out_root / "runs" / "dance3_t0995_r050"),
            "notes": "most permissive force-gated DanceTrack smoke in this block",
        },
    ]

    if summary_csv.is_file():
        summary_rows = [dict(row) for row in read_rows(summary_csv)]
        existing_steps = {str(row.get("step", "")) for row in summary_rows}
        for row in init_summary_rows(
            steps=steps,
            queue_name=queue_name,
            logs_dir=logs_dir,
            seed_summary_csv=seed_summary_csv,
        ):
            if str(row.get("step", "")) not in existing_steps:
                summary_rows.append(row)
        decision_rows = [dict(row) for row in read_rows(decision_csv)]
        append_registry(summary_csv, out_root, "running", "force-gated 6h queue resumed", registry_csv)
    else:
        summary_rows = init_summary_rows(
            steps=steps,
            queue_name=queue_name,
            logs_dir=logs_dir,
            seed_summary_csv=seed_summary_csv,
        )
        decision_rows: List[Dict[str, object]] = []
        write_rows(summary_csv, QUEUE_FIELDS, summary_rows)
        write_rows(decision_csv, DECISION_FIELDS, decision_rows)
        append_registry(summary_csv, out_root, "running", "force-gated 6h queue started", registry_csv)

    try:
        for item in steps:
            step = str(item["step"])
            log_path = logs_dir / f"{step}.log"
            current_row = find_row(summary_rows, step)
            current_status = str(current_row.get("status", "")).strip() or "pending"

            if str(item["kind"]) == "wait":
                if current_status == "success":
                    continue
                if seed_summary_csv.is_file():
                    seed_rows = read_rows(seed_summary_csv)
                    seed_statuses = {str(row.get("status", "")).strip() for row in seed_rows} if seed_rows else set()
                    if seed_statuses == {"success"}:
                        seed_runtime_csv = seed_run_root / "runtime_compare.csv"
                        seed_delta_csv = seed_run_root / "metrics_delta.csv"
                        runtime = read_runtime(seed_runtime_csv)
                        delta = read_delta(seed_delta_csv)
                        upsert_decision_row(
                            decision_rows,
                            {
                                "step": step,
                                "kind": "wait_existing",
                                "benchmark": str(item["benchmark"]),
                                "seq_label": "|".join(item["seqs"]),
                                "gate_thresh": str(item["gate_thresh"]),
                                "force_risk_scale": str(item["force_risk_scale"]),
                                "status": "success",
                                "out_dir": str(seed_run_root),
                                "summary_csv": str(seed_summary_csv),
                                "runtime_compare_csv": str(seed_runtime_csv),
                                "metrics_delta_csv": str(seed_delta_csv),
                                "delta_HOTA": delta["delta_HOTA"],
                                "delta_AssA": delta["delta_AssA"],
                                "delta_IDF1": delta["delta_IDF1"],
                                "delta_MOTA": delta["delta_MOTA"],
                                "delta_IDs": delta["delta_IDs"],
                                "delta_Frag": delta["delta_Frag"],
                                "selected_matches": int(runtime["selected_matches"]),
                                "forced_gate_rejected_rows": int(runtime["forced_gate_rejected_rows"]),
                                "takeover_risk_rejected_rows": int(runtime["takeover_risk_rejected_rows"]),
                                "acceptance_gate_scored_rows": int(runtime["acceptance_gate_scored_rows"]),
                                "acceptance_gate_rejected_rows": int(runtime["acceptance_gate_rejected_rows"]),
                                "acceptance_gate_accepted_rows": int(runtime["acceptance_gate_accepted_rows"]),
                                "notes": "seed calibration finished and ingested into queue decisions",
                            },
                        )
                        write_rows(decision_csv, DECISION_FIELDS, decision_rows)
                        update_row(
                            summary_rows,
                            step,
                            status="success",
                            finished_at=str(current_row.get("finished_at", "")).strip() or now_iso(),
                            notes=(
                                "seed run finished "
                                f"selected_matches={int(runtime['selected_matches'])} "
                                f"forced_gate_rejected={int(runtime['forced_gate_rejected_rows'])}"
                            ),
                        )
                        write_rows(summary_csv, QUEUE_FIELDS, summary_rows)
                        continue
            else:
                child_out_dir = Path(str(item["out_dir"]))
                if child_run_is_success(child_out_dir):
                    upsert_decision_row(
                        decision_rows,
                        collect_decision_row(
                            step=step,
                            kind="run_child",
                            benchmark=str(item["benchmark"]),
                            seqs=list(item["seqs"]),
                            gate_thresh=str(item["gate_thresh"]),
                            force_risk_scale=str(item["force_risk_scale"]),
                            out_dir=child_out_dir,
                        ),
                    )
                    write_rows(decision_csv, DECISION_FIELDS, decision_rows)
                    latest = next(row for row in decision_rows if str(row.get("step", "")) == step)
                    update_row(
                        summary_rows,
                        step,
                        status="success",
                        finished_at=str(current_row.get("finished_at", "")).strip() or now_iso(),
                        notes=(
                            f"{item['notes']} "
                            f"delta_HOTA={float(latest['delta_HOTA']):+.3f} "
                            f"selected_matches={int(latest['selected_matches'])} "
                            f"forced_gate_rejected={int(latest['forced_gate_rejected_rows'])}"
                        ),
                    )
                    write_rows(summary_csv, QUEUE_FIELDS, summary_rows)
                    continue

            update_row(summary_rows, step, status="running", started_at=str(current_row.get("started_at", "")).strip() or now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, summary_rows)

            if str(item["kind"]) == "wait":
                wait_existing_run(seed_summary_csv, poll_sec=int(args.poll_sec), log_path=log_path)
                seed_runtime_csv = seed_run_root / "runtime_compare.csv"
                seed_delta_csv = seed_run_root / "metrics_delta.csv"
                runtime = read_runtime(seed_runtime_csv)
                delta = read_delta(seed_delta_csv)
                upsert_decision_row(
                    decision_rows,
                    {
                        "step": step,
                        "kind": "wait_existing",
                        "benchmark": str(item["benchmark"]),
                        "seq_label": "|".join(item["seqs"]),
                        "gate_thresh": str(item["gate_thresh"]),
                        "force_risk_scale": str(item["force_risk_scale"]),
                        "status": "success",
                        "out_dir": str(seed_run_root),
                        "summary_csv": str(seed_summary_csv),
                        "runtime_compare_csv": str(seed_runtime_csv),
                        "metrics_delta_csv": str(seed_delta_csv),
                        "delta_HOTA": delta["delta_HOTA"],
                        "delta_AssA": delta["delta_AssA"],
                        "delta_IDF1": delta["delta_IDF1"],
                        "delta_MOTA": delta["delta_MOTA"],
                        "delta_IDs": delta["delta_IDs"],
                        "delta_Frag": delta["delta_Frag"],
                        "selected_matches": int(runtime["selected_matches"]),
                        "forced_gate_rejected_rows": int(runtime["forced_gate_rejected_rows"]),
                        "takeover_risk_rejected_rows": int(runtime["takeover_risk_rejected_rows"]),
                        "acceptance_gate_scored_rows": int(runtime["acceptance_gate_scored_rows"]),
                        "acceptance_gate_rejected_rows": int(runtime["acceptance_gate_rejected_rows"]),
                        "acceptance_gate_accepted_rows": int(runtime["acceptance_gate_accepted_rows"]),
                        "notes": "seed calibration finished and ingested into queue decisions",
                    },
                )
                write_rows(decision_csv, DECISION_FIELDS, decision_rows)
                update_row(
                    summary_rows,
                    step,
                    status="success",
                    finished_at=now_iso(),
                    notes=(
                        "seed run finished "
                        f"selected_matches={int(runtime['selected_matches'])} "
                        f"forced_gate_rejected={int(runtime['forced_gate_rejected_rows'])}"
                    ),
                )
                write_rows(summary_csv, QUEUE_FIELDS, summary_rows)
                continue

            child_out_dir = Path(str(item["out_dir"]))
            cmd = build_child_cmd(
                benchmark=str(item["benchmark"]),
                seqs=list(item["seqs"]),
                out_root=child_out_dir,
                gate_thresh=str(item["gate_thresh"]),
                force_risk_scale=str(item["force_risk_scale"]),
                reuse_raw_from=str(item["reuse_raw_from"]),
            )
            rc = run_step(cmd, log_path, cwd=REPO_ROOT)
            if rc != 0:
                raise RuntimeError(f"{step} failed with return code {rc}")
            ensure_child_success(child_out_dir / "summary.csv")
            upsert_decision_row(
                decision_rows,
                collect_decision_row(
                    step=step,
                    kind="run_child",
                    benchmark=str(item["benchmark"]),
                    seqs=list(item["seqs"]),
                    gate_thresh=str(item["gate_thresh"]),
                    force_risk_scale=str(item["force_risk_scale"]),
                    out_dir=child_out_dir,
                ),
            )
            write_rows(decision_csv, DECISION_FIELDS, decision_rows)
            latest = next(row for row in decision_rows if str(row.get("step", "")) == step)
            update_row(
                summary_rows,
                step,
                status="success",
                finished_at=now_iso(),
                notes=(
                    f"{item['notes']} "
                    f"delta_HOTA={float(latest['delta_HOTA']):+.3f} "
                    f"selected_matches={int(latest['selected_matches'])} "
                    f"forced_gate_rejected={int(latest['forced_gate_rejected_rows'])}"
                ),
            )
            write_rows(summary_csv, QUEUE_FIELDS, summary_rows)

        append_registry(summary_csv, out_root, "success", "force-gated 6h queue finished", registry_csv)
    except Exception as exc:
        for row in summary_rows:
            if str(row.get("status", "")) == "running":
                row["status"] = "failed"
                row["finished_at"] = now_iso()
                row["notes"] = f"queue aborted: {exc}"
        write_rows(summary_csv, QUEUE_FIELDS, summary_rows)
        write_rows(decision_csv, DECISION_FIELDS, decision_rows)
        append_registry(summary_csv, out_root, "failed", f"force-gated 6h queue failed: {exc}", registry_csv)
        raise


if __name__ == "__main__":
    main()
