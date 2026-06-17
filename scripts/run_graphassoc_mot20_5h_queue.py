#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
PLAN_CSV = REPO_ROOT / "outputs" / "experiment_plan.csv"

QUEUE_FIELDS = [
    "step",
    "name",
    "status",
    "run_root",
    "log_path",
    "started_at",
    "finished_at",
    "seq_ids",
    "delta_hota",
    "delta_assa",
    "delta_idf1",
    "delta_mota",
    "delta_ids",
    "delta_frag",
    "identical_count",
    "notes",
    "params_json",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Run a five-hour graph-association MOT20 experiment queue.")
    parser.add_argument("--run-root", default=str(REPO_ROOT / "outputs" / f"graphassoc_mot20_5h_queue_{ts}"))
    parser.add_argument("--queue-name", default=f"graphassoc_mot20_5h_queue_{ts}")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--wait-summary-csv", default="")
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def write_rows(path: Path, fieldnames: Iterable[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def append_row(rows: List[Dict[str, object]], step: str, name: str, run_root: Path | str, log_path: Path | str, seq_ids: List[int], params_json: str, notes: str = "") -> None:
    rows.append(
        {
            "step": step,
            "name": name,
            "status": "pending",
            "run_root": str(run_root),
            "log_path": str(log_path),
            "started_at": "",
            "finished_at": "",
            "seq_ids": "|".join(str(v) for v in seq_ids),
            "delta_hota": "",
            "delta_assa": "",
            "delta_idf1": "",
            "delta_mota": "",
            "delta_ids": "",
            "delta_frag": "",
            "identical_count": "",
            "notes": notes,
            "params_json": params_json,
        }
    )


def update_row(rows: List[Dict[str, object]], step: str, **updates: object) -> None:
    for row in rows:
        if str(row.get("step")) == str(step):
            row.update(updates)
            return
    raise KeyError(f"Missing queue step: {step}")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def summary_complete(path: Path) -> bool:
    rows = read_csv_rows(path)
    if not rows:
        return False
    active = {"running", "pending", "queued"}
    return all(str(row.get("status", "")).strip().lower() not in active for row in rows)


def wait_for_summary(path: Path, poll_seconds: int, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[wait_start] {now_iso()} waiting for {path}\n")
        handle.flush()
        while not summary_complete(path):
            handle.write(f"[wait_poll] {now_iso()} summary not finished yet\n")
            handle.flush()
            time.sleep(max(5, int(poll_seconds)))
        handle.write(f"[wait_done] {now_iso()} summary finished\n")


def queue_plan_status(args: argparse.Namespace, status: str, summary_csv: Path, log_path: Path, notes: str = "") -> None:
    cmd = [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "upsert_experiment_plan.py"),
        "--csv",
        str(PLAN_CSV),
        "--key",
        f"run_root:{Path(args.run_root).expanduser().resolve()}",
        "--status",
        status,
        "--kind",
        "analysis",
        "--script",
        "scripts/run_graphassoc_mot20_5h_queue.py",
        "--dataset",
        "MOT20",
        "--split",
        "val_half",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        "graphassoc_5h_queue",
        "--tag",
        args.queue_name,
        "--run-root",
        str(Path(args.run_root).expanduser().resolve()),
        "--summary-csv",
        str(summary_csv),
        "--log-path",
        str(log_path),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def queue_registry(args: argparse.Namespace, status: str, summary_csv: Path, log_path: Path, notes: str = "") -> None:
    cmd = [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(REGISTRY_CSV),
        "--kind",
        "analysis",
        "--status",
        status,
        "--script",
        "scripts/run_graphassoc_mot20_5h_queue.py",
        "--dataset",
        "MOT20",
        "--split",
        "val_half",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        "graphassoc_5h_queue",
        "--tag",
        args.queue_name,
        "--run-root",
        str(Path(args.run_root).expanduser().resolve()),
        "--summary-csv",
        str(summary_csv),
        "--log-path",
        str(log_path),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def build_experiment_command(args: argparse.Namespace, run_root: Path, exp_name: str, seq_ids: List[int], extra_args: List[str]) -> List[str]:
    return [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "run_botsort_graphassoc_mot20_eval.py"),
        "--run-root",
        str(run_root),
        "--experiment-name",
        exp_name,
        "--seq-ids",
        *[str(v) for v in seq_ids],
        *extra_args,
    ]


def run_step(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[started_at] {now_iso()}\n")
        handle.write("[cmd] " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[finished_at] {now_iso()}\n")
        handle.write(f"[return_code] {proc.returncode}\n")
    return int(proc.returncode)


def parse_metrics_delta(run_root: Path) -> Dict[str, float | int]:
    rows = read_csv_rows(run_root / "metrics_delta.csv")
    if not rows:
        return {
            "delta_hota": 0.0,
            "delta_assa": 0.0,
            "delta_idf1": 0.0,
            "delta_mota": 0.0,
            "delta_ids": 0,
            "delta_frag": 0,
        }
    row = rows[0]
    return {
        "delta_hota": float(row.get("delta_HOTA", 0.0) or 0.0),
        "delta_assa": float(row.get("delta_AssA", 0.0) or 0.0),
        "delta_idf1": float(row.get("delta_IDF1", 0.0) or 0.0),
        "delta_mota": float(row.get("delta_MOTA", 0.0) or 0.0),
        "delta_ids": int(round(float(row.get("delta_IDs", 0.0) or 0.0))),
        "delta_frag": int(round(float(row.get("delta_Frag", 0.0) or 0.0))),
    }


def parse_identical_count(run_root: Path) -> int:
    rows = read_csv_rows(run_root / "track_diff_summary.csv")
    return int(sum(int(float(row.get("identical", 0) or 0)) for row in rows))


def make_variant(name: str, seq_ids: List[int], extra_args: List[str], notes: str = "") -> Dict[str, object]:
    return {
        "name": name,
        "seq_ids": list(seq_ids),
        "extra_args": list(extra_args),
        "notes": notes,
    }


def rank_variants(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        results,
        key=lambda row: (
            float(row.get("delta_hota", -999.0)),
            float(row.get("delta_idf1", -999.0)),
            -abs(int(row.get("delta_ids", 0))),
            -abs(int(row.get("delta_frag", 0))),
            -int(row.get("identical_count", 999999)),
        ),
        reverse=True,
    )


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    logs_dir = run_root / "logs"
    summary_csv = run_root / "summary.csv"
    queue_log = logs_dir / "queue.log"
    rows: List[Dict[str, object]] = []

    queue_plan_status(args, "running", summary_csv, queue_log, notes="graph-assoc 5h queue started")
    queue_registry(args, "running", summary_csv, queue_log, notes="graph-assoc 5h queue started")

    try:
        if args.wait_summary_csv:
            wait_path = Path(args.wait_summary_csv).expanduser().resolve()
            append_row(rows, "wait_current", "wait_current_run", wait_path.parent, queue_log, [], "", notes=f"wait for {wait_path}")
            update_row(rows, "wait_current", status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            wait_for_summary(wait_path, args.poll_seconds, queue_log)
            update_row(rows, "wait_current", status="success", finished_at=now_iso(), notes=f"upstream run finished: {wait_path}")
            write_rows(summary_csv, QUEUE_FIELDS, rows)

        seq2_variants = [
            make_variant(
                "seq2_softforce003",
                [2],
                [
                    "--graph-assoc-row-margin", "0.05",
                    "--graph-assoc-col-margin", "0.05",
                    "--graph-assoc-min-reclaim-tracklet-len", "15",
                    "--graph-assoc-recent-owner-max-tracklet-len", "12",
                    "--graph-assoc-min-box-iou", "0.55",
                    "--graph-assoc-reclaim-bonus", "0.15",
                    "--graph-assoc-recent-owner-penalty", "0.08",
                    "--graph-assoc-iou-bonus", "0.05",
                    "--graph-assoc-min-assignment-gain", "0.005",
                    "--graph-assoc-max-cost-delta", "0.08",
                    "--graph-assoc-force-match-cost", "0.03",
                ],
                notes="hold match count, same structure as repaired seq2 baseline with mild force cost",
            ),
            make_variant(
                "seq2_widergraph_force003",
                [2],
                [
                    "--graph-assoc-top-k", "4",
                    "--graph-assoc-max-rows", "5",
                    "--graph-assoc-max-cols", "5",
                    "--graph-assoc-row-margin", "0.06",
                    "--graph-assoc-col-margin", "0.06",
                    "--graph-assoc-min-reclaim-tracklet-len", "15",
                    "--graph-assoc-recent-owner-max-tracklet-len", "12",
                    "--graph-assoc-min-box-iou", "0.55",
                    "--graph-assoc-reclaim-bonus", "0.15",
                    "--graph-assoc-recent-owner-penalty", "0.08",
                    "--graph-assoc-iou-bonus", "0.05",
                    "--graph-assoc-min-assignment-gain", "0.005",
                    "--graph-assoc-max-cost-delta", "0.08",
                    "--graph-assoc-force-match-cost", "0.03",
                ],
                notes="larger local competition graph while preserving match count",
            ),
            make_variant(
                "seq2_stable_reclaim_force003",
                [2],
                [
                    "--graph-assoc-row-margin", "0.04",
                    "--graph-assoc-col-margin", "0.04",
                    "--graph-assoc-min-reclaim-time-since-update", "2",
                    "--graph-assoc-min-reclaim-tracklet-len", "20",
                    "--graph-assoc-recent-owner-max-tracklet-len", "8",
                    "--graph-assoc-min-box-iou", "0.58",
                    "--graph-assoc-reclaim-bonus", "0.12",
                    "--graph-assoc-recent-owner-penalty", "0.06",
                    "--graph-assoc-iou-bonus", "0.05",
                    "--graph-assoc-min-assignment-gain", "0.008",
                    "--graph-assoc-max-cost-delta", "0.06",
                    "--graph-assoc-force-match-cost", "0.03",
                ],
                notes="tighter reclaim gate, fewer but cleaner reroutes",
            ),
            make_variant(
                "seq2_softforce005",
                [2],
                [
                    "--graph-assoc-row-margin", "0.05",
                    "--graph-assoc-col-margin", "0.05",
                    "--graph-assoc-min-reclaim-tracklet-len", "15",
                    "--graph-assoc-recent-owner-max-tracklet-len", "12",
                    "--graph-assoc-min-box-iou", "0.55",
                    "--graph-assoc-reclaim-bonus", "0.15",
                    "--graph-assoc-recent-owner-penalty", "0.08",
                    "--graph-assoc-iou-bonus", "0.05",
                    "--graph-assoc-min-assignment-gain", "0.005",
                    "--graph-assoc-max-cost-delta", "0.08",
                    "--graph-assoc-force-match-cost", "0.05",
                ],
                notes="same repaired structure with stronger but still bounded forced edge cost",
            ),
        ]

        seq2_results: List[Dict[str, object]] = []
        for idx, variant in enumerate(seq2_variants, start=1):
            step = f"seq2_{idx}_{variant['name']}"
            child_run_root = run_root / "runs" / step
            child_name = f"{args.queue_name}_{step}"
            child_log = logs_dir / f"{step}.log"
            append_row(
                rows,
                step,
                str(variant["name"]),
                child_run_root,
                child_log,
                list(variant["seq_ids"]),
                json.dumps({"extra_args": variant["extra_args"]}, ensure_ascii=False),
                notes=str(variant["notes"]),
            )
            update_row(rows, step, status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            cmd = build_experiment_command(args, child_run_root, child_name, list(variant["seq_ids"]), list(variant["extra_args"]))
            rc = run_step(cmd, child_log)
            if rc != 0:
                update_row(rows, step, status="failed", finished_at=now_iso(), notes=f"{variant['notes']} | return_code={rc}")
                write_rows(summary_csv, QUEUE_FIELDS, rows)
                continue
            metrics = parse_metrics_delta(child_run_root)
            identical_count = parse_identical_count(child_run_root)
            result = {
                **variant,
                **metrics,
                "identical_count": identical_count,
            }
            seq2_results.append(result)
            update_row(
                rows,
                step,
                status="success",
                finished_at=now_iso(),
                delta_hota=metrics["delta_hota"],
                delta_assa=metrics["delta_assa"],
                delta_idf1=metrics["delta_idf1"],
                delta_mota=metrics["delta_mota"],
                delta_ids=metrics["delta_ids"],
                delta_frag=metrics["delta_frag"],
                identical_count=identical_count,
                notes=f"{variant['notes']} | seq2 complete",
            )
            write_rows(summary_csv, QUEUE_FIELDS, rows)

        ranked_seq2 = rank_variants(seq2_results)
        top_seq2 = ranked_seq2[:2]
        validated_results: List[Dict[str, object]] = []
        for idx, variant in enumerate(top_seq2, start=1):
            step = f"seq5_validate_{idx}_{variant['name']}"
            child_run_root = run_root / "runs" / step
            child_name = f"{args.queue_name}_{step}"
            child_log = logs_dir / f"{step}.log"
            append_row(
                rows,
                step,
                str(variant["name"]),
                child_run_root,
                child_log,
                [5],
                json.dumps({"extra_args": variant["extra_args"]}, ensure_ascii=False),
                notes=f"validate top seq2 candidate on MOT20-05: {variant['notes']}",
            )
            update_row(rows, step, status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            cmd = build_experiment_command(args, child_run_root, child_name, [5], list(variant["extra_args"]))
            rc = run_step(cmd, child_log)
            if rc != 0:
                update_row(rows, step, status="failed", finished_at=now_iso(), notes=f"seq5 validation failed | return_code={rc}")
                write_rows(summary_csv, QUEUE_FIELDS, rows)
                continue
            metrics = parse_metrics_delta(child_run_root)
            identical_count = parse_identical_count(child_run_root)
            combined = dict(variant)
            combined.update(
                {
                    "seq5_delta_hota": metrics["delta_hota"],
                    "seq5_delta_assa": metrics["delta_assa"],
                    "seq5_delta_idf1": metrics["delta_idf1"],
                    "seq5_delta_mota": metrics["delta_mota"],
                    "seq5_delta_ids": metrics["delta_ids"],
                    "seq5_delta_frag": metrics["delta_frag"],
                    "seq5_identical_count": identical_count,
                }
            )
            validated_results.append(combined)
            update_row(
                rows,
                step,
                status="success",
                finished_at=now_iso(),
                delta_hota=metrics["delta_hota"],
                delta_assa=metrics["delta_assa"],
                delta_idf1=metrics["delta_idf1"],
                delta_mota=metrics["delta_mota"],
                delta_ids=metrics["delta_ids"],
                delta_frag=metrics["delta_frag"],
                identical_count=identical_count,
                notes=f"seq5 validation complete for {variant['name']}",
            )
            write_rows(summary_csv, QUEUE_FIELDS, rows)

        if validated_results:
            best_validated = sorted(
                validated_results,
                key=lambda row: (
                    float(row.get("delta_hota", -999.0)) + float(row.get("seq5_delta_hota", -999.0)),
                    float(row.get("delta_idf1", -999.0)) + float(row.get("seq5_delta_idf1", -999.0)),
                ),
                reverse=True,
            )[0]

            combined_variants = [
                make_variant(
                    f"seq25_best_{best_validated['name']}",
                    [2, 5],
                    list(best_validated["extra_args"]),
                    notes="combined validation for the best seq2+seq5 validated candidate",
                )
            ]

            neighbor_args = list(best_validated["extra_args"])
            if "--graph-assoc-force-match-cost" in neighbor_args:
                pos = neighbor_args.index("--graph-assoc-force-match-cost")
                current_force = float(neighbor_args[pos + 1])
                neighbor_args[pos + 1] = f"{min(0.08, current_force + 0.02):.2f}"
            else:
                neighbor_args.extend(["--graph-assoc-force-match-cost", "0.05"])
            combined_variants.append(
                make_variant(
                    f"seq25_neighbor_{best_validated['name']}",
                    [2, 5],
                    neighbor_args,
                    notes="neighbor combined validation around the current best candidate",
                )
            )

            for idx, variant in enumerate(combined_variants, start=1):
                step = f"seq25_{idx}_{variant['name']}"
                child_run_root = run_root / "runs" / step
                child_name = f"{args.queue_name}_{step}"
                child_log = logs_dir / f"{step}.log"
                append_row(
                    rows,
                    step,
                    str(variant["name"]),
                    child_run_root,
                    child_log,
                    list(variant["seq_ids"]),
                    json.dumps({"extra_args": variant["extra_args"]}, ensure_ascii=False),
                    notes=str(variant["notes"]),
                )
                update_row(rows, step, status="running", started_at=now_iso())
                write_rows(summary_csv, QUEUE_FIELDS, rows)
                cmd = build_experiment_command(args, child_run_root, child_name, list(variant["seq_ids"]), list(variant["extra_args"]))
                rc = run_step(cmd, child_log)
                if rc != 0:
                    update_row(rows, step, status="failed", finished_at=now_iso(), notes=f"{variant['notes']} | return_code={rc}")
                    write_rows(summary_csv, QUEUE_FIELDS, rows)
                    continue
                metrics = parse_metrics_delta(child_run_root)
                identical_count = parse_identical_count(child_run_root)
                update_row(
                    rows,
                    step,
                    status="success",
                    finished_at=now_iso(),
                    delta_hota=metrics["delta_hota"],
                    delta_assa=metrics["delta_assa"],
                    delta_idf1=metrics["delta_idf1"],
                    delta_mota=metrics["delta_mota"],
                    delta_ids=metrics["delta_ids"],
                    delta_frag=metrics["delta_frag"],
                    identical_count=identical_count,
                    notes=f"{variant['notes']} | combined validation complete",
                )
                write_rows(summary_csv, QUEUE_FIELDS, rows)

        queue_plan_status(args, "completed", summary_csv, queue_log, notes="graph-assoc 5h queue completed")
        queue_registry(args, "success", summary_csv, queue_log, notes="graph-assoc 5h queue completed")
    except Exception as exc:
        queue_plan_status(args, "failed", summary_csv, queue_log, notes=str(exc))
        queue_registry(args, "failed", summary_csv, queue_log, notes=str(exc))
        raise


if __name__ == "__main__":
    main()
