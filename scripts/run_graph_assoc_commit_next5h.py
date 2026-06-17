#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
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
    "summary_csv",
    "log_path",
    "started_at",
    "finished_at",
    "rows_jsonl",
    "candidate_rows",
    "event_rows",
    "clusters",
    "accepted_clusters",
    "rejected_clusters",
    "gt_positive_clusters",
    "gt_negative_clusters",
    "train_examples",
    "val_examples",
    "best_epoch",
    "val_row_acc",
    "val_commit_precision",
    "val_commit_recall",
    "notes",
    "params_json",
]


STABLE_EXTRA_ARGS = [
    "--graph-assoc-top-k",
    "3",
    "--graph-assoc-no-col-only-blocks",
    "--graph-assoc-require-row-involved-strict-reclaim",
    "--graph-assoc-max-rows",
    "4",
    "--graph-assoc-max-cols",
    "4",
    "--graph-assoc-row-margin",
    "0.05",
    "--graph-assoc-col-margin",
    "0.05",
    "--graph-assoc-min-reclaim-time-since-update",
    "1",
    "--graph-assoc-max-reclaim-time-since-update",
    "8",
    "--graph-assoc-min-reclaim-tracklet-len",
    "15",
    "--graph-assoc-recent-owner-max-time-since-update",
    "1",
    "--graph-assoc-recent-owner-max-tracklet-len",
    "12",
    "--graph-assoc-young-active-max-time-since-update",
    "1",
    "--graph-assoc-young-active-max-tracklet-len",
    "20",
    "--graph-assoc-young-active-min-reclaim-gap",
    "2",
    "--graph-assoc-young-active-max-cost-delta",
    "0.03",
    "--graph-assoc-stale-lost-owner-min-time-since-update",
    "9",
    "--graph-assoc-stale-lost-owner-min-tracklet-len",
    "100",
    "--graph-assoc-stale-lost-owner-active-max-time-since-update",
    "1",
    "--graph-assoc-stale-lost-owner-min-introduced-edge-utility",
    "0.0",
    "--graph-assoc-min-box-iou",
    "0.55",
    "--graph-assoc-reclaim-bonus",
    "0.15",
    "--graph-assoc-recent-owner-penalty",
    "0.08",
    "--graph-assoc-iou-bonus",
    "0.05",
    "--graph-assoc-score-bonus",
    "0.02",
    "--graph-assoc-min-assignment-gain",
    "0.005",
    "--graph-assoc-max-cost-delta",
    "0.08",
    "--graph-assoc-row-involved-min-assignment-gain",
    "0.03",
    "--graph-assoc-col-only-min-assignment-gain",
    "0.07",
    "--graph-assoc-col-only-max-cost-delta",
    "0.02",
    "--graph-assoc-force-match-cost",
    "0.0",
    "--graph-assoc-protect-young-active-rows",
    "--graph-assoc-protect-stale-lost-owner-rows",
]

YOUNG_EXTRA_ARGS = [
    "--graph-assoc-top-k",
    "3",
    "--graph-assoc-no-col-only-blocks",
    "--graph-assoc-require-row-involved-strict-reclaim",
    "--graph-assoc-max-rows",
    "4",
    "--graph-assoc-max-cols",
    "4",
    "--graph-assoc-row-margin",
    "0.05",
    "--graph-assoc-col-margin",
    "0.05",
    "--graph-assoc-min-reclaim-time-since-update",
    "1",
    "--graph-assoc-max-reclaim-time-since-update",
    "8",
    "--graph-assoc-min-reclaim-tracklet-len",
    "15",
    "--graph-assoc-recent-owner-max-time-since-update",
    "1",
    "--graph-assoc-recent-owner-max-tracklet-len",
    "12",
    "--graph-assoc-young-active-max-time-since-update",
    "1",
    "--graph-assoc-young-active-max-tracklet-len",
    "20",
    "--graph-assoc-young-active-min-reclaim-gap",
    "2",
    "--graph-assoc-young-active-max-cost-delta",
    "0.03",
    "--graph-assoc-min-box-iou",
    "0.55",
    "--graph-assoc-reclaim-bonus",
    "0.15",
    "--graph-assoc-recent-owner-penalty",
    "0.08",
    "--graph-assoc-iou-bonus",
    "0.05",
    "--graph-assoc-score-bonus",
    "0.02",
    "--graph-assoc-min-assignment-gain",
    "0.005",
    "--graph-assoc-max-cost-delta",
    "0.08",
    "--graph-assoc-row-involved-min-assignment-gain",
    "0.03",
    "--graph-assoc-col-only-min-assignment-gain",
    "0.07",
    "--graph-assoc-col-only-max-cost-delta",
    "0.02",
    "--graph-assoc-force-match-cost",
    "0.0",
    "--graph-assoc-protect-young-active-rows",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Run the next five hours of graph-assoc commit data experiments.")
    parser.add_argument("--run-root", default=str(REPO_ROOT / "outputs" / f"graph_assoc_commit_next5h_{ts}"))
    parser.add_argument("--queue-name", default=f"graph_assoc_commit_next5h_{ts}")
    parser.add_argument("--python-bin", default=sys.executable)
    return parser.parse_args()


def write_rows(path: Path, fieldnames: Iterable[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def update_row(rows: List[Dict[str, object]], step: str, **updates: object) -> None:
    for row in rows:
        if str(row.get("step", "")) == str(step):
            row.update(updates)
            return
    raise KeyError(f"Missing queue step: {step}")


def append_row(
    rows: List[Dict[str, object]],
    *,
    step: str,
    name: str,
    run_root: Path,
    summary_csv: Path,
    log_path: Path,
    notes: str = "",
    params_json: str = "",
) -> None:
    rows.append(
        {
            "step": step,
            "name": name,
            "status": "pending",
            "run_root": str(run_root),
            "summary_csv": str(summary_csv),
            "log_path": str(log_path),
            "started_at": "",
            "finished_at": "",
            "rows_jsonl": "",
            "candidate_rows": "",
            "event_rows": "",
            "clusters": "",
            "accepted_clusters": "",
            "rejected_clusters": "",
            "gt_positive_clusters": "",
            "gt_negative_clusters": "",
            "train_examples": "",
            "val_examples": "",
            "best_epoch": "",
            "val_row_acc": "",
            "val_commit_precision": "",
            "val_commit_recall": "",
            "notes": notes,
            "params_json": params_json,
        }
    )


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
        "scripts/run_graph_assoc_commit_next5h.py",
        "--dataset",
        "MOT20",
        "--split",
        "val_half",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        "graph_assoc_commit_next5h",
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
        "scripts/run_graph_assoc_commit_next5h.py",
        "--dataset",
        "MOT20",
        "--split",
        "val_half",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        "graph_assoc_commit_next5h",
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


def run_step(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[started_at] {now_iso()}\n")
        handle.write(f"[cwd] {REPO_ROOT}\n")
        handle.write("[cmd] " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[finished_at] {now_iso()}\n")
        handle.write(f"[return_code] {proc.returncode}\n")
    return int(proc.returncode)


def count_jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def find_seq_candidate_jsonl(run_root: Path, seq_name: str) -> Path:
    analysis_dir = run_root / "graph_assoc_analysis"
    direct = analysis_dir / f"{seq_name}_candidates.jsonl"
    if direct.is_file():
        return direct
    partial = direct.with_suffix(".jsonl.partial")
    if partial.is_file():
        return partial
    fallback = sorted(analysis_dir.glob("*_candidates.jsonl"))
    if fallback:
        return fallback[0]
    partial_fallback = sorted(analysis_dir.glob("*_candidates.jsonl.partial"))
    if partial_fallback:
        return partial_fallback[0]
    return direct


def find_seq_events_jsonl(run_root: Path, seq_name: str) -> Path:
    analysis_dir = run_root / "graph_assoc_analysis"
    direct = analysis_dir / f"{seq_name}_events.jsonl"
    if direct.is_file():
        return direct
    partial = direct.with_suffix(".jsonl.partial")
    if partial.is_file():
        return partial
    fallback = sorted(analysis_dir.glob("*_events.jsonl"))
    if fallback:
        return fallback[0]
    partial_fallback = sorted(analysis_dir.glob("*_events.jsonl.partial"))
    if partial_fallback:
        return partial_fallback[0]
    return direct


def parse_graph_assoc_seq_summary(run_root: Path, seq_name: str) -> Dict[str, object]:
    summary_path = run_root / "graph_assoc_analysis" / f"{seq_name}_summary.csv"
    row = read_csv_rows(summary_path)
    summary_row = row[0] if row else {}
    candidate_jsonl = find_seq_candidate_jsonl(run_root, seq_name)
    event_jsonl = find_seq_events_jsonl(run_root, seq_name)
    candidate_rows = int(float(summary_row.get("candidate_count", 0) or 0)) if summary_row else 0
    if candidate_rows <= 0:
        candidate_rows = count_jsonl_rows(candidate_jsonl)
    event_rows = int(float(summary_row.get("event_count", 0) or 0)) if summary_row else 0
    if event_rows <= 0:
        event_rows = count_jsonl_rows(event_jsonl)
    return {
        "summary_path": summary_path,
        "rows_jsonl": candidate_jsonl,
        "candidate_rows": candidate_rows,
        "event_rows": event_rows,
    }


def parse_stage1_metrics(run_root: Path) -> Dict[str, object]:
    train_summary_rows = read_csv_rows(run_root / "summary.csv")
    train_summary = train_summary_rows[0] if train_summary_rows else {}
    data_summary_rows = read_csv_rows(run_root / "graph_assoc_commit_data" / "summary.csv")
    data_summary = data_summary_rows[0] if data_summary_rows else {}
    return {
        "clusters": int(float(data_summary.get("clusters", 0) or 0)),
        "accepted_clusters": int(float(data_summary.get("accepted_clusters", 0) or 0)),
        "rejected_clusters": int(float(data_summary.get("rejected_clusters", 0) or 0)),
        "gt_positive_clusters": int(float(data_summary.get("gt_positive_clusters", 0) or 0)),
        "gt_negative_clusters": int(float(data_summary.get("gt_negative_clusters", 0) or 0)),
        "train_examples": int(float(train_summary.get("train_examples", 0) or 0)),
        "val_examples": int(float(train_summary.get("val_examples", 0) or 0)),
        "best_epoch": int(float(train_summary.get("best_epoch", 0) or 0)) if str(train_summary.get("best_epoch", "")).strip() else "",
        "val_row_acc": train_summary.get("val_row_acc", ""),
        "val_commit_precision": train_summary.get("val_commit_precision", ""),
        "val_commit_recall": train_summary.get("val_commit_recall", ""),
        "status": str(train_summary.get("status", "")).strip(),
    }


def build_graphassoc_eval_cmd(
    args: argparse.Namespace,
    *,
    run_root: Path,
    experiment_name: str,
    seq_ids: List[int],
    extra_args: List[str],
) -> List[str]:
    return [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "run_botsort_graphassoc_mot20_eval.py"),
        "--run-root",
        str(run_root),
        "--experiment-name",
        experiment_name,
        "--variant-name",
        "botsort_graphassoc_mot20_canddump",
        "--seq-ids",
        *[str(v) for v in seq_ids],
        "--graph-assoc-dump-candidate-rows",
        *extra_args,
    ]


def build_stage1_cmd(
    *,
    out_dir: Path,
    manifest_csv: Path,
    epochs: int,
    hidden_dim: int,
    batch_size: int,
) -> List[str]:
    return [
        "bash",
        str(REPO_ROOT / "scripts" / "run_graph_assoc_commit_stage1.sh"),
        str(out_dir),
        str(manifest_csv),
        "MOT20-02",
        "MOT20-05",
        str(int(epochs)),
        str(int(hidden_dim)),
        str(int(batch_size)),
        "8",
        "2",
        "1",
        "8",
        "32",
        "MOT20",
        "val_half",
    ]


def write_manifest(path: Path, sources: List[Dict[str, str]]) -> None:
    fieldnames = [
        "rows_jsonl",
        "source_tag",
        "host_variant",
        "split_tag",
        "dataset",
        "data_root",
        "split",
        "split_part",
        "seq_name",
        "dataset_tag",
        "feature_version",
    ]
    write_rows(path, fieldnames, sources)


def build_manifest_sources(candidates: List[Dict[str, object]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for item in candidates:
        rows_jsonl = Path(str(item["rows_jsonl"])).resolve()
        if not rows_jsonl.is_file():
            continue
        seq_name = str(item["seq_name"])
        split_tag = "val" if seq_name == "MOT20-05" else "train"
        rows.append(
            {
                "rows_jsonl": str(rows_jsonl),
                "source_tag": str(item["source_tag"]),
                "host_variant": str(item["host_variant"]),
                "split_tag": split_tag,
                "dataset": "MOT20",
                "data_root": "/gemini/code/datasets",
                "split": "train",
                "split_part": "val_half",
                "seq_name": seq_name,
                "dataset_tag": "graph_assoc_commit",
                "feature_version": "graph_assoc_v1",
            }
        )
    return rows


def has_train_and_val_sources(sources: List[Dict[str, str]]) -> bool:
    split_tags = {str(row.get("split_tag", "")).strip() for row in sources}
    return "train" in split_tags and "val" in split_tags


def run_candidate_step(
    *,
    args: argparse.Namespace,
    rows: List[Dict[str, object]],
    queue_summary_csv: Path,
    step: str,
    name: str,
    seq_id: int,
    seq_name: str,
    host_variant: str,
    extra_args: List[str],
    run_root: Path,
    log_path: Path,
    notes: str,
) -> Dict[str, object] | None:
    summary_csv = run_root / "summary.csv"
    append_row(
        rows,
        step=step,
        name=name,
        run_root=run_root,
        summary_csv=summary_csv,
        log_path=log_path,
        notes=notes,
        params_json=json.dumps({"seq_id": seq_id, "extra_args": extra_args}, ensure_ascii=False),
    )
    update_row(rows, step, status="running", started_at=now_iso())
    write_rows(queue_summary_csv, QUEUE_FIELDS, rows)

    cmd = build_graphassoc_eval_cmd(
        args,
        run_root=run_root,
        experiment_name=name,
        seq_ids=[seq_id],
        extra_args=list(extra_args),
    )
    rc = run_step(cmd, log_path)
    if rc != 0:
        update_row(
            rows,
            step,
            status="failed",
            finished_at=now_iso(),
            notes=f"{notes} | return_code={rc}",
        )
        write_rows(queue_summary_csv, QUEUE_FIELDS, rows)
        return None

    parsed = parse_graph_assoc_seq_summary(run_root, seq_name)
    update_row(
        rows,
        step,
        status="success",
        finished_at=now_iso(),
        rows_jsonl=str(parsed["rows_jsonl"]),
        candidate_rows=parsed["candidate_rows"],
        event_rows=parsed["event_rows"],
        notes=f"{notes} | candidate dump complete",
    )
    write_rows(queue_summary_csv, QUEUE_FIELDS, rows)
    return {
        "rows_jsonl": str(parsed["rows_jsonl"]),
        "candidate_rows": int(parsed["candidate_rows"]),
        "event_rows": int(parsed["event_rows"]),
        "seq_name": seq_name,
        "source_tag": f"{host_variant}_{seq_name}",
        "host_variant": host_variant,
    }


def run_stage1_step(
    *,
    rows: List[Dict[str, object]],
    queue_summary_csv: Path,
    step: str,
    name: str,
    manifest_rows: List[Dict[str, str]],
    out_dir: Path,
    log_path: Path,
    notes: str,
    epochs: int,
    hidden_dim: int,
    batch_size: int,
) -> bool:
    summary_csv = out_dir / "summary.csv"
    manifest_csv = out_dir / "source_manifest.csv"
    append_row(
        rows,
        step=step,
        name=name,
        run_root=out_dir,
        summary_csv=summary_csv,
        log_path=log_path,
        notes=notes,
        params_json=json.dumps(
            {
                "epochs": epochs,
                "hidden_dim": hidden_dim,
                "batch_size": batch_size,
                "manifest_sources": len(manifest_rows),
            },
            ensure_ascii=False,
        ),
    )

    if not manifest_rows or not has_train_and_val_sources(manifest_rows):
        update_row(
            rows,
            step,
            status="failed",
            started_at=now_iso(),
            finished_at=now_iso(),
            notes=f"{notes} | missing train/val candidate sources",
        )
        write_rows(queue_summary_csv, QUEUE_FIELDS, rows)
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest_csv, manifest_rows)
    update_row(rows, step, status="running", started_at=now_iso(), notes=f"{notes} | manifest ready")
    write_rows(queue_summary_csv, QUEUE_FIELDS, rows)

    cmd = build_stage1_cmd(
        out_dir=out_dir,
        manifest_csv=manifest_csv,
        epochs=epochs,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
    )
    rc = run_step(cmd, log_path)
    parsed = parse_stage1_metrics(out_dir)
    ok = rc == 0 and str(parsed.get("status", "")) == "ok"
    update_row(
        rows,
        step,
        status="success" if ok else "failed",
        finished_at=now_iso(),
        clusters=parsed["clusters"],
        accepted_clusters=parsed["accepted_clusters"],
        rejected_clusters=parsed["rejected_clusters"],
        gt_positive_clusters=parsed["gt_positive_clusters"],
        gt_negative_clusters=parsed["gt_negative_clusters"],
        train_examples=parsed["train_examples"],
        val_examples=parsed["val_examples"],
        best_epoch=parsed["best_epoch"],
        val_row_acc=parsed["val_row_acc"],
        val_commit_precision=parsed["val_commit_precision"],
        val_commit_recall=parsed["val_commit_recall"],
        notes=f"{notes} | stage1 {'complete' if ok else 'failed'}",
    )
    write_rows(queue_summary_csv, QUEUE_FIELDS, rows)
    return ok


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    logs_dir = run_root / "logs"
    queue_summary_csv = run_root / "summary.csv"
    queue_log = logs_dir / "queue.log"
    rows: List[Dict[str, object]] = []
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    with queue_log.open("a", encoding="utf-8") as handle:
        handle.write(f"[queue_start] {now_iso()} queue_name={args.queue_name}\n")

    queue_plan_status(args, "running", queue_summary_csv, queue_log, notes="graph-assoc commit next5h queue started")
    queue_registry(args, "running", queue_summary_csv, queue_log, notes="graph-assoc commit next5h queue started")
    write_rows(queue_summary_csv, QUEUE_FIELDS, rows)

    try:
        successful_candidates: List[Dict[str, object]] = []

        stable_seq5 = run_candidate_step(
            args=args,
            rows=rows,
            queue_summary_csv=queue_summary_csv,
            step="stable_seq5_dump",
            name=f"{args.queue_name}_stable_seq5_dump",
            seq_id=5,
            seq_name="MOT20-05",
            host_variant="graphassoc_stalelostprotect_canddump",
            extra_args=STABLE_EXTRA_ARGS,
            run_root=run_root / "runs" / "stable_seq5_dump",
            log_path=logs_dir / "stable_seq5_dump.log",
            notes="stable candidate dump on MOT20-05 using the baseline-matching protected graph config",
        )
        if stable_seq5:
            successful_candidates.append(stable_seq5)

        stable_seq2 = run_candidate_step(
            args=args,
            rows=rows,
            queue_summary_csv=queue_summary_csv,
            step="stable_seq2_dump",
            name=f"{args.queue_name}_stable_seq2_dump",
            seq_id=2,
            seq_name="MOT20-02",
            host_variant="graphassoc_stalelostprotect_canddump",
            extra_args=STABLE_EXTRA_ARGS,
            run_root=run_root / "runs" / "stable_seq2_dump",
            log_path=logs_dir / "stable_seq2_dump.log",
            notes="stable candidate dump on MOT20-02 to pair positive and negative sequence evidence",
        )
        if stable_seq2:
            successful_candidates.append(stable_seq2)

        stable_manifest_rows = build_manifest_sources(
            [row for row in successful_candidates if str(row.get("host_variant", "")) == "graphassoc_stalelostprotect_canddump"]
        )
        run_stage1_step(
            rows=rows,
            queue_summary_csv=queue_summary_csv,
            step="stage1_stable",
            name="graph_assoc_commit_stage1_stable",
            manifest_rows=stable_manifest_rows,
            out_dir=run_root / "stage1" / "stable_2src",
            log_path=logs_dir / "stage1_stable.log",
            notes="first learned commit trainer on stable seq2+seq5 candidate sources",
            epochs=12,
            hidden_dim=128,
            batch_size=8,
        )

        young_seq5 = run_candidate_step(
            args=args,
            rows=rows,
            queue_summary_csv=queue_summary_csv,
            step="young_seq5_dump",
            name=f"{args.queue_name}_young_seq5_dump",
            seq_id=5,
            seq_name="MOT20-05",
            host_variant="graphassoc_youngactiveprotectcost03_canddump",
            extra_args=YOUNG_EXTRA_ARGS,
            run_root=run_root / "runs" / "young_seq5_dump",
            log_path=logs_dir / "young_seq5_dump.log",
            notes="looser candidate dump on MOT20-05 to widen acceptance boundary examples",
        )
        if young_seq5:
            successful_candidates.append(young_seq5)

        young_seq2 = run_candidate_step(
            args=args,
            rows=rows,
            queue_summary_csv=queue_summary_csv,
            step="young_seq2_dump",
            name=f"{args.queue_name}_young_seq2_dump",
            seq_id=2,
            seq_name="MOT20-02",
            host_variant="graphassoc_youngactiveprotectcost03_canddump",
            extra_args=YOUNG_EXTRA_ARGS,
            run_root=run_root / "runs" / "young_seq2_dump",
            log_path=logs_dir / "young_seq2_dump.log",
            notes="looser candidate dump on MOT20-02 for more diverse positive commit examples",
        )
        if young_seq2:
            successful_candidates.append(young_seq2)

        expanded_manifest_rows = build_manifest_sources(successful_candidates)
        run_stage1_step(
            rows=rows,
            queue_summary_csv=queue_summary_csv,
            step="stage1_expanded",
            name="graph_assoc_commit_stage1_expanded",
            manifest_rows=expanded_manifest_rows,
            out_dir=run_root / "stage1" / "expanded_4src",
            log_path=logs_dir / "stage1_expanded.log",
            notes="second learned commit trainer on expanded four-source candidate pool",
            epochs=14,
            hidden_dim=128,
            batch_size=8,
        )

        run_stage1_step(
            rows=rows,
            queue_summary_csv=queue_summary_csv,
            step="stage1_expanded_wide",
            name="graph_assoc_commit_stage1_expanded_wide",
            manifest_rows=expanded_manifest_rows,
            out_dir=run_root / "stage1" / "expanded_4src_wide",
            log_path=logs_dir / "stage1_expanded_wide.log",
            notes="capacity check for the learned commit model after candidate expansion",
            epochs=14,
            hidden_dim=192,
            batch_size=8,
        )

        queue_plan_status(args, "completed", queue_summary_csv, queue_log, notes="graph-assoc commit next5h queue completed")
        queue_registry(args, "success", queue_summary_csv, queue_log, notes="graph-assoc commit next5h queue completed")
        with queue_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[queue_done] {now_iso()} status=success\n")
    except Exception as exc:
        queue_plan_status(args, "failed", queue_summary_csv, queue_log, notes=str(exc))
        queue_registry(args, "failed", queue_summary_csv, queue_log, notes=str(exc))
        with queue_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[queue_done] {now_iso()} status=failed error={exc}\n")
        raise


if __name__ == "__main__":
    main()
