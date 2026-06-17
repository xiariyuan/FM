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
EVAL_SCRIPT = REPO_ROOT / "scripts" / "run_botsort_graphassoc_mot20_eval.py"
DEFAULT_GATE_CKPT = (
    REPO_ROOT / "outputs" / "20260505_081619_graphassoc_commit_gatedblend_next5h" / "train_policy" / "best.pt"
)
DEFAULT_REFERENCE_METRICS = (
    REPO_ROOT / "outputs" / "20260505_081619_graphassoc_commit_gatedblend_next5h" / "mot20_eval" / "03_policy_t003" / "metrics_delta.csv"
)

QUEUE_FIELDS = [
    "step",
    "name",
    "status",
    "run_root",
    "summary_csv",
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
    "changed_blocks",
    "forced_matches",
    "accepted_candidates",
    "rejected_candidates",
    "notes",
    "params_json",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Run structural follow-up experiments for graph-association gate.")
    parser.add_argument("--run-root", default=str(REPO_ROOT / "outputs" / f"graphassoc_gate_structural_next5h_{ts}"))
    parser.add_argument("--queue-name", default=f"graphassoc_gate_structural_next5h_{ts}")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--gate-checkpoint", default=str(DEFAULT_GATE_CKPT))
    parser.add_argument("--seq-ids", nargs="+", type=int, default=[2, 5])
    parser.add_argument("--reference-metrics", default=str(DEFAULT_REFERENCE_METRICS))
    parser.add_argument("--commit-device", default="cuda")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=[],
        help="Optional subset of variants to run. Accepts step ids like 02, 02_name, or the raw variant name.",
    )
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
    raise KeyError(f"missing queue step: {step}")


def find_row(rows: List[Dict[str, object]], step: str) -> Dict[str, object] | None:
    for row in rows:
        if str(row.get("step", "")) == str(step):
            return row
    return None


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
        "scripts/run_graphassoc_gate_structural_next5h.py",
        "--dataset",
        "MOT20",
        "--split",
        "val_half",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        "graphassoc_gate_structural",
        "--tag",
        args.queue_name,
        "--run-root",
        str(Path(args.run_root).expanduser().resolve()),
        "--summary-csv",
        str(summary_csv),
        "--log-path",
        str(log_path),
        "--checkpoint",
        str(Path(args.gate_checkpoint).expanduser().resolve()),
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
        "scripts/run_graphassoc_gate_structural_next5h.py",
        "--dataset",
        "MOT20",
        "--split",
        "val_half",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        "graphassoc_gate_structural",
        "--tag",
        args.queue_name,
        "--run-root",
        str(Path(args.run_root).expanduser().resolve()),
        "--summary-csv",
        str(summary_csv),
        "--checkpoint",
        str(Path(args.gate_checkpoint).expanduser().resolve()),
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
        handle.write("[cmd] " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[finished_at] {now_iso()}\n")
        handle.write(f"[return_code] {proc.returncode}\n")
    return int(proc.returncode)


def parse_metrics_delta(run_root: Path) -> Dict[str, object]:
    rows = read_csv_rows(run_root / "metrics_delta.csv")
    if not rows:
        raise FileNotFoundError(f"missing metrics_delta.csv under {run_root}")
    row = rows[0]
    return {
        "delta_hota": float(row.get("delta_HOTA", 0.0) or 0.0),
        "delta_assa": float(row.get("delta_AssA", 0.0) or 0.0),
        "delta_idf1": float(row.get("delta_IDF1", 0.0) or 0.0),
        "delta_mota": float(row.get("delta_MOTA", 0.0) or 0.0),
        "delta_ids": int(round(float(row.get("delta_IDs", 0.0) or 0.0))),
        "delta_frag": int(round(float(row.get("delta_Frag", 0.0) or 0.0))),
    }


def parse_runtime(run_root: Path) -> Dict[str, object]:
    rows = read_csv_rows(run_root / "runtime_per_sequence.csv")
    combined = None
    for row in rows:
        if str(row.get("seq", "")) == "MOT20-02|MOT20-05":
            combined = row
            break
    if combined is None and rows:
        combined = rows[-1]
    if combined is None:
        return {}
    return {
        "changed_blocks": int(float(combined.get("changed_blocks", 0) or 0)),
        "forced_matches": int(float(combined.get("forced_matches", 0) or 0)),
        "accepted_candidates": int(float(combined.get("learned_commit_margin_accept_count", 0) or 0)),
        "rejected_candidates": int(float(combined.get("learned_commit_margin_reject_count", 0) or 0)),
    }


def common_graph_args(args: argparse.Namespace) -> List[str]:
    return [
        "--graph-assoc-commit-checkpoint",
        str(Path(args.gate_checkpoint).expanduser().resolve()),
        "--graph-assoc-commit-device",
        str(args.commit_device),
        "--graph-assoc-commit-score-margin",
        "0.0",
        "--graph-assoc-commit-replace-rules",
        "--graph-assoc-dump-candidate-rows",
    ]


def common_gateonly_args(args: argparse.Namespace, score_margin: str) -> List[str]:
    return [
        "--graph-assoc-commit-checkpoint",
        str(Path(args.gate_checkpoint).expanduser().resolve()),
        "--graph-assoc-commit-device",
        str(args.commit_device),
        "--graph-assoc-commit-score-margin",
        score_margin,
        "--graph-assoc-commit-gate-only",
        "--graph-assoc-dump-candidate-rows",
    ]


def variant_catalog(args: argparse.Namespace) -> List[Dict[str, object]]:
    common = common_graph_args(args)
    gateonly_m015 = common_gateonly_args(args, "-0.15")
    gateonly_m025 = common_gateonly_args(args, "-0.25")
    wide_graph = [
        "--graph-assoc-top-k", "4",
        "--graph-assoc-max-rows", "5",
        "--graph-assoc-max-cols", "5",
        "--graph-assoc-row-margin", "0.06",
        "--graph-assoc-col-margin", "0.06",
        "--graph-assoc-min-reclaim-tracklet-len", "12",
        "--graph-assoc-recent-owner-max-tracklet-len", "14",
        "--graph-assoc-min-box-iou", "0.50",
        "--graph-assoc-reclaim-bonus", "0.16",
        "--graph-assoc-recent-owner-penalty", "0.08",
        "--graph-assoc-iou-bonus", "0.06",
        "--graph-assoc-score-bonus", "0.02",
        "--graph-assoc-min-assignment-gain", "0.003",
        "--graph-assoc-max-cost-delta", "0.10",
        "--graph-assoc-row-involved-min-assignment-gain", "0.003",
        "--graph-assoc-col-only-min-assignment-gain", "0.003",
        "--graph-assoc-col-only-max-cost-delta", "0.10",
        "--graph-assoc-force-match-cost", "0.0",
    ]
    default_graph = [
        "--graph-assoc-top-k", "3",
        "--graph-assoc-max-rows", "4",
        "--graph-assoc-max-cols", "4",
        "--graph-assoc-row-margin", "0.05",
        "--graph-assoc-col-margin", "0.05",
        "--graph-assoc-min-reclaim-tracklet-len", "15",
        "--graph-assoc-recent-owner-max-tracklet-len", "12",
        "--graph-assoc-min-box-iou", "0.55",
        "--graph-assoc-reclaim-bonus", "0.15",
        "--graph-assoc-recent-owner-penalty", "0.08",
        "--graph-assoc-iou-bonus", "0.05",
        "--graph-assoc-score-bonus", "0.02",
        "--graph-assoc-min-assignment-gain", "0.005",
        "--graph-assoc-max-cost-delta", "0.08",
        "--graph-assoc-row-involved-min-assignment-gain", "0.005",
        "--graph-assoc-col-only-min-assignment-gain", "0.005",
        "--graph-assoc-col-only-max-cost-delta", "0.08",
        "--graph-assoc-force-match-cost", "0.0",
    ]
    variants = [
        {
            "name": "wide_replace_weighted_neutral",
            "extra_args": wide_graph + common,
            "notes": "expand the local competition graph and let the learned dual-head gate replace hand rules",
        },
        {
            "name": "wide_replace_pos_minus_neutral_t003",
            "extra_args": wide_graph
            + common
            + [
                "--graph-assoc-commit-decision-mode",
                "positive_minus_neutral",
                "--graph-assoc-commit-threshold",
                "0.03",
            ],
            "notes": "use direct positive-minus-neutral scoring to test whether neutral weighting is over-suppressing useful rewrites",
        },
        {
            "name": "wide_replace_dualthr_p070_n070",
            "extra_args": wide_graph
            + common
            + [
                "--graph-assoc-commit-decision-mode",
                "dual_threshold",
                "--graph-assoc-commit-positive-threshold",
                "0.70",
                "--graph-assoc-commit-neutral-threshold",
                "0.70",
            ],
            "notes": "force a two-head accept rule: enough positive evidence and bounded neutral risk",
        },
        {
            "name": "default_replace_pos_minus_neutral_t003",
            "extra_args": default_graph
            + common
            + [
                "--graph-assoc-commit-decision-mode",
                "positive_minus_neutral",
                "--graph-assoc-commit-threshold",
                "0.03",
            ],
            "notes": "keep the current graph geometry but change the learned decision surface",
        },
    ]
    variants.extend(
        [
            {
                "name": "wide_replace_weighted_thr000",
                "extra_args": wide_graph
                + common
                + [
                    "--graph-assoc-commit-threshold",
                    "0.0",
                ],
                "notes": "lower the weighted dual-head accept threshold to expand from the ultra-conservative 18 accepted blocks",
            },
            {
                "name": "wide_replace_weighted_thr_minus005",
                "extra_args": wide_graph
                + common
                + [
                    "--graph-assoc-commit-threshold",
                    "-0.05",
                ],
                "notes": "stress-test a wider learned replacement policy; candidate replay predicts roughly twice the accepted block count",
            },
            {
                "name": "wide_gateonly_weighted_m015",
                "extra_args": wide_graph + gateonly_m015,
                "notes": "rules-first hybrid: hand rules form a quality floor while the learned gate only vetoes high-risk blocks",
            },
            {
                "name": "wide_gateonly_weighted_m025",
                "extra_args": wide_graph + gateonly_m025,
                "notes": "looser rules-first hybrid to recover positive-utility candidates over-rejected by the learned gate",
            },
        ]
    )
    expand_graph_med = [
        "--graph-assoc-top-k", "5",
        "--graph-assoc-max-rows", "6",
        "--graph-assoc-max-cols", "6",
        "--graph-assoc-row-margin", "0.07",
        "--graph-assoc-col-margin", "0.07",
        "--graph-assoc-min-reclaim-tracklet-len", "10",
        "--graph-assoc-recent-owner-max-tracklet-len", "16",
        "--graph-assoc-min-box-iou", "0.45",
        "--graph-assoc-reclaim-bonus", "0.17",
        "--graph-assoc-recent-owner-penalty", "0.07",
        "--graph-assoc-iou-bonus", "0.07",
        "--graph-assoc-score-bonus", "0.02",
        "--graph-assoc-min-assignment-gain", "0.002",
        "--graph-assoc-max-cost-delta", "0.12",
        "--graph-assoc-row-involved-min-assignment-gain", "0.002",
        "--graph-assoc-col-only-min-assignment-gain", "0.002",
        "--graph-assoc-col-only-max-cost-delta", "0.12",
        "--graph-assoc-force-match-cost", "0.0",
    ]
    expand_graph_aggr = [
        "--graph-assoc-top-k", "6",
        "--graph-assoc-max-rows", "7",
        "--graph-assoc-max-cols", "7",
        "--graph-assoc-row-margin", "0.08",
        "--graph-assoc-col-margin", "0.08",
        "--graph-assoc-min-reclaim-tracklet-len", "8",
        "--graph-assoc-recent-owner-max-tracklet-len", "18",
        "--graph-assoc-min-box-iou", "0.40",
        "--graph-assoc-reclaim-bonus", "0.18",
        "--graph-assoc-recent-owner-penalty", "0.06",
        "--graph-assoc-iou-bonus", "0.08",
        "--graph-assoc-score-bonus", "0.02",
        "--graph-assoc-min-assignment-gain", "0.002",
        "--graph-assoc-max-cost-delta", "0.14",
        "--graph-assoc-row-involved-min-assignment-gain", "0.002",
        "--graph-assoc-col-only-min-assignment-gain", "0.002",
        "--graph-assoc-col-only-max-cost-delta", "0.14",
        "--graph-assoc-force-match-cost", "0.0",
    ]
    variants.extend(
        [
            {
                "name": "expand_med_replace",
                "extra_args": expand_graph_med + common,
                "notes": "larger candidate graph with learned replacement to test whether coverage, not gate logic, is the main limiter",
            },
            {
                "name": "expand_med_drop_replace",
                "extra_args": expand_graph_med + ["--graph-assoc-allow-match-count-drop"] + common,
                "notes": "medium graph expansion with match-count drop allowed to recover useful rewrites that the old rule rejected",
            },
            {
                "name": "expand_aggr_drop_replace",
                "extra_args": expand_graph_aggr + ["--graph-assoc-allow-match-count-drop"] + common,
                "notes": "aggressive graph expansion with match-count drop to stress the recovery path and long-range reassociation",
            },
            {
                "name": "expand_aggr_drop_replace_protect",
                "extra_args": expand_graph_aggr
                + [
                    "--graph-assoc-allow-match-count-drop",
                    "--graph-assoc-protect-young-active-rows",
                    "--graph-assoc-young-active-max-time-since-update",
                    "1",
                    "--graph-assoc-young-active-max-tracklet-len",
                    "24",
                    "--graph-assoc-young-active-min-reclaim-gap",
                    "3",
                    "--graph-assoc-young-active-max-cost-delta",
                    "0.03",
                    "--graph-assoc-protect-stale-lost-owner-rows",
                    "--graph-assoc-stale-lost-owner-min-time-since-update",
                    "10",
                    "--graph-assoc-stale-lost-owner-min-tracklet-len",
                    "80",
                    "--graph-assoc-stale-lost-owner-active-max-time-since-update",
                    "1",
                    "--graph-assoc-stale-lost-owner-min-introduced-edge-utility",
                    "0.01",
                ]
                + common,
                "notes": "aggressive graph expansion with explicit protection for young active rows and stale lost owners",
            },
        ]
    )
    return variants


def normalize_step_token(token: str) -> str:
    text = str(token).strip()
    if not text:
        return ""
    if text.isdigit():
        return f"{int(text):02d}"
    return text


def selected_variants(args: argparse.Namespace) -> List[Dict[str, object]]:
    variants = variant_catalog(args)
    if not args.steps:
        return variants

    requested = {normalize_step_token(token) for token in args.steps if normalize_step_token(token)}
    selected: List[Dict[str, object]] = []
    for idx, variant in enumerate(variants, start=1):
        step = f"{idx:02d}"
        full_step = f"{step}_{variant['name']}"
        if step in requested or full_step in requested or str(variant["name"]) in requested:
            selected.append(variant)
    if not selected:
        raise ValueError(f"no structural variants matched --steps={args.steps}")
    return selected


def build_eval_command(args: argparse.Namespace, run_root: Path, exp_name: str, extra_args: List[str]) -> List[str]:
    return [
        args.python_bin,
        str(EVAL_SCRIPT),
        "--run-root",
        str(run_root),
        "--experiment-name",
        exp_name,
        "--variant-name",
        "botsort_graphassoc_gate_structural",
        "--seq-ids",
        *[str(v) for v in args.seq_ids],
        *extra_args,
    ]


def append_reference_row(args: argparse.Namespace, rows: List[Dict[str, object]], summary_csv: Path) -> None:
    if find_row(rows, "reference_current_gate") is not None:
        return
    ref_path = Path(args.reference_metrics).expanduser().resolve()
    if not ref_path.is_file():
        return
    metrics = parse_metrics_delta(ref_path.parent)
    runtime = parse_runtime(ref_path.parent)
    rows.append(
        {
            "step": "reference_current_gate",
            "name": "graphassoc_gate_next5h_20260424_230311",
            "status": "reference",
            "run_root": str(ref_path.parent),
            "summary_csv": str(ref_path.parent / "summary.csv"),
            "log_path": "",
            "started_at": "",
            "finished_at": "",
            "seq_ids": "|".join(str(v) for v in args.seq_ids),
            **metrics,
            **runtime,
            "notes": "current best structured validation reference; not rerun in this queue",
            "params_json": json.dumps({"reference_metrics": str(ref_path)}, ensure_ascii=False),
        }
    )
    write_rows(summary_csv, QUEUE_FIELDS, rows)


def main() -> None:
    args = parse_args()
    queue_root = Path(args.run_root).expanduser().resolve()
    logs_dir = queue_root / "logs"
    summary_csv = queue_root / "summary.csv"
    queue_log = logs_dir / "queue.log"
    rows: List[Dict[str, object]] = [dict(row) for row in read_csv_rows(summary_csv)]

    queue_root.mkdir(parents=True, exist_ok=True)
    queue_plan_status(args, "running", summary_csv, queue_log, notes="graphassoc structural gate queue started")
    queue_registry(args, "running", summary_csv, queue_log, notes="graphassoc structural gate queue started")
    append_reference_row(args, rows, summary_csv)

    try:
        for idx, variant in enumerate(variant_catalog(args), start=1):
            if args.steps:
                step_token = f"{idx:02d}"
                full_step_token = f"{step_token}_{variant['name']}"
                requested = {normalize_step_token(token) for token in args.steps if normalize_step_token(token)}
                if (
                    step_token not in requested
                    and full_step_token not in requested
                    and str(variant["name"]) not in requested
                ):
                    continue
            step = f"{idx:02d}_{variant['name']}"
            child_root = queue_root / "runs" / step
            child_log = logs_dir / f"{step}.log"
            exp_name = f"{args.queue_name}_{step}"
            row_defaults = {
                "step": step,
                "name": variant["name"],
                "status": "pending",
                "run_root": str(child_root),
                "summary_csv": str(child_root / "summary.csv"),
                "log_path": str(child_log),
                "started_at": "",
                "finished_at": "",
                "seq_ids": "|".join(str(v) for v in args.seq_ids),
                "notes": variant["notes"],
                "params_json": json.dumps({"extra_args": variant["extra_args"]}, ensure_ascii=False),
            }
            existing = find_row(rows, step)
            if existing is None:
                rows.append(row_defaults)
            else:
                for key in ("name", "run_root", "summary_csv", "log_path", "seq_ids", "notes", "params_json"):
                    existing[key] = row_defaults[key]
            if (child_root / "metrics_delta.csv").is_file():
                metrics = parse_metrics_delta(child_root)
                runtime = parse_runtime(child_root)
                existing = find_row(rows, step) or {}
                update_row(
                    rows,
                    step,
                    status="success",
                    started_at=existing.get("started_at", ""),
                    finished_at=existing.get("finished_at", "") or now_iso(),
                    **metrics,
                    **runtime,
                    notes=f"{variant['notes']} | complete",
                )
                write_rows(summary_csv, QUEUE_FIELDS, rows)
                continue
            update_row(rows, step, status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            cmd = build_eval_command(args, child_root, exp_name, list(variant["extra_args"]))
            rc = run_step(cmd, child_log)
            if rc != 0:
                update_row(rows, step, status="failed", finished_at=now_iso(), notes=f"{variant['notes']} | return_code={rc}")
                write_rows(summary_csv, QUEUE_FIELDS, rows)
                continue
            metrics = parse_metrics_delta(child_root)
            runtime = parse_runtime(child_root)
            update_row(
                rows,
                step,
                status="success",
                finished_at=now_iso(),
                **metrics,
                **runtime,
                notes=f"{variant['notes']} | complete",
            )
            write_rows(summary_csv, QUEUE_FIELDS, rows)

        queue_plan_status(args, "completed", summary_csv, queue_log, notes="graphassoc structural gate queue completed")
        queue_registry(args, "success", summary_csv, queue_log, notes="graphassoc structural gate queue completed")
    except Exception as exc:
        for row in rows:
            if str(row.get("status", "")) in {"pending", "running"}:
                row["status"] = "failed"
                row["finished_at"] = now_iso()
                row["notes"] = f"{row.get('notes', '')} | queue_exception={exc}".strip()
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        queue_plan_status(args, "failed", summary_csv, queue_log, notes=str(exc))
        queue_registry(args, "failed", summary_csv, queue_log, notes=str(exc))
        raise


if __name__ == "__main__":
    main()
