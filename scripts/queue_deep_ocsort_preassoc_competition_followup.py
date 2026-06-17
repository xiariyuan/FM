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
    "owner_max_hits",
    "max_owner_edge_deficit",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDs",
    "delta_Frag",
    "selected_matches",
    "candidate_rows",
    "decision",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recorded follow-up queue around the current best pre-association stale-competition full7 owner-block configuration."
    )
    parser.add_argument("--out-root", default="")
    parser.add_argument(
        "--reference-run-root",
        default=str(
            REPO_ROOT
            / "outputs"
            / "deep_ocsort_preassoc_competition_full7_anchor_h20_i075_age50_b010_pen005_ownerblock_oh8_odef010_20260405_1"
        ),
    )
    parser.add_argument(
        "--reuse-raw-from",
        default=str(
            REPO_ROOT
            / "outputs"
            / "deep_ocsort_preassoc_competition_full7_anchor_h20_i075_age50_b010_pen005_ownerblock_oh8_odef010_20260405_1"
        ),
    )
    parser.add_argument(
        "--seq-names",
        nargs="*",
        default=[
            "MOT17-02-FRCNN",
            "MOT17-04-FRCNN",
            "MOT17-05-FRCNN",
            "MOT17-09-FRCNN",
            "MOT17-10-FRCNN",
            "MOT17-11-FRCNN",
            "MOT17-13-FRCNN",
        ],
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
        if str(row["step"]) == step:
            row.update(updates)
            return
    raise KeyError(f"Missing queue step: {step}")


def get_row(rows: List[Dict[str, object]], step: str) -> Dict[str, object] | None:
    for row in rows:
        if str(row.get("step", "")) == step:
            return row
    return None


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
        "scripts/queue_deep_ocsort_preassoc_competition_followup.py",
        "--dataset",
        "MOT17",
        "--split",
        "val_half",
        "--tracker-family",
        "deep_ocsort_preassoc_competition",
        "--variant",
        run_root.name,
        "--tag",
        "deep_ocsort_preassoc_competition_followup",
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
            }
    raise ValueError(f"Missing competition runtime row in {runtime_compare_csv}")


def build_smoke_cmd(
    *,
    out_dir: Path,
    reuse_raw_from: Path,
    seq_names: List[str],
    owner_max_hits: int,
    max_owner_edge_deficit: float,
) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_deep_ocsort_preassoc_competition_smoke.py"),
        "--out-root",
        str(out_dir),
        "--reuse-raw-from",
        str(reuse_raw_from),
        "--seq-names",
        *seq_names,
        "--preassoc-stale-competition-min-time-since-update",
        "2",
        "--preassoc-stale-competition-max-time-since-update",
        "8",
        "--preassoc-stale-competition-min-hits",
        "20",
        "--preassoc-stale-competition-min-box-iou",
        "0.75",
        "--preassoc-stale-competition-min-edge-score",
        "0.0",
        "--preassoc-stale-competition-bias",
        "0.1",
        "--preassoc-stale-competition-iou-scale",
        "0.0",
        "--preassoc-stale-competition-require-raw-owner",
        "--preassoc-stale-competition-min-age-gap-vs-owner",
        "50",
        "--preassoc-stale-competition-owner-max-hits",
        str(owner_max_hits),
        "--preassoc-stale-competition-owner-edge-penalty",
        "0.05",
        "--preassoc-stale-competition-max-owner-edge-deficit",
        str(max_owner_edge_deficit),
        "--preassoc-stale-competition-block-owner-on-reclaim",
    ]


def main() -> None:
    args = parse_args()
    queue_name = Path(args.out_root).name if args.out_root else f"deep_ocsort_preassoc_competition_followup_{timestamp_tag()}"
    out_root = Path(args.out_root).resolve() if args.out_root else (REPO_ROOT / "outputs" / queue_name).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "summary.csv"
    decision_csv = out_root / "decision.csv"
    logs_dir = out_root / "logs"
    registry_csv = str(Path(args.registry_csv).resolve())
    reference_run_root = Path(args.reference_run_root).resolve()
    reuse_raw_from = Path(args.reuse_raw_from).resolve()

    variants = [
        {
            "step": "tight_h8_odef005",
            "variant": "owner_max_hits=8,max_owner_edge_deficit=0.05",
            "suffix": "oh8_odef005",
            "owner_max_hits": 8,
            "max_owner_edge_deficit": 0.05,
            "notes": "tighter owner-edge deficit, target ~2 selected matches",
        },
        {
            "step": "loose_h8_odef015",
            "variant": "owner_max_hits=8,max_owner_edge_deficit=0.15",
            "suffix": "oh8_odef015",
            "owner_max_hits": 8,
            "max_owner_edge_deficit": 0.15,
            "notes": "slightly looser owner-edge deficit, target ~5 selected matches",
        },
        {
            "step": "looser_h8_odef020",
            "variant": "owner_max_hits=8,max_owner_edge_deficit=0.20",
            "suffix": "oh8_odef020",
            "owner_max_hits": 8,
            "max_owner_edge_deficit": 0.20,
            "notes": "looser owner-edge deficit, target ~8 selected matches",
        },
        {
            "step": "weakowner_h5_odef020",
            "variant": "owner_max_hits=5,max_owner_edge_deficit=0.20",
            "suffix": "oh5_odef020",
            "owner_max_hits": 5,
            "max_owner_edge_deficit": 0.20,
            "notes": "same rough trigger budget via weaker-owner gate instead of tighter edge-deficit",
        },
    ]

    existing_queue_rows = [dict(row) for row in read_rows(summary_csv)]
    existing_decision_rows = [dict(row) for row in read_rows(decision_csv)]
    queue_rows: List[Dict[str, object]] = []
    decision_rows: List[Dict[str, object]] = existing_decision_rows

    reference_metrics_csv = reference_run_root / "metrics_delta.csv"
    reference_runtime_csv = reference_run_root / "runtime_compare.csv"
    reference_summary_csv = reference_run_root / "summary.csv"
    ref_metrics = read_metrics_delta(reference_metrics_csv)
    ref_runtime = read_runtime(reference_runtime_csv)
    upsert_decision_row(
        decision_rows,
        {
            "step": "reference",
            "variant": "owner_max_hits=8,max_owner_edge_deficit=0.10",
            "status": "success",
            "out_dir": str(reference_run_root),
            "summary_csv": str(reference_summary_csv),
            "metrics_delta_csv": str(reference_metrics_csv),
            "runtime_compare_csv": str(reference_runtime_csv),
            "owner_max_hits": 8,
            "max_owner_edge_deficit": 0.10,
            "delta_HOTA": ref_metrics["delta_HOTA"],
            "delta_AssA": ref_metrics["delta_AssA"],
            "delta_IDF1": ref_metrics["delta_IDF1"],
            "delta_MOTA": ref_metrics["delta_MOTA"],
            "delta_IDs": ref_metrics["delta_IDs"],
            "delta_Frag": ref_metrics["delta_Frag"],
            "selected_matches": int(ref_runtime["selected_matches"]),
            "candidate_rows": int(ref_runtime["candidate_rows"]),
            "decision": "reference_best_so_far",
            "notes": "existing full7 positive baseline for neighborhood follow-up",
        },
    )

    for variant in variants:
        child_out = out_root / variant["suffix"]
        default_row = {
            "step": variant["step"],
            "name": child_out.name,
            "status": "pending",
            "out_dir": str(child_out),
            "summary_csv": str(child_out / "summary.csv"),
            "log_path": str(logs_dir / f"{variant['step']}.log"),
            "started_at": "",
            "finished_at": "",
            "notes": variant["notes"],
        }
        existing_row = get_row(existing_queue_rows, str(variant["step"]))
        queue_rows.append(dict(existing_row) if existing_row is not None else default_row)

    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
    write_rows(decision_csv, DECISION_FIELDS, decision_rows)
    append_registry(
        summary_csv,
        out_root,
        "running",
        "preassoc competition full7 follow-up queue resumed" if existing_queue_rows else "preassoc competition full7 follow-up queue started",
        registry_csv,
    )

    try:
        best_hota = float(ref_metrics["delta_HOTA"])
        best_idf1 = float(ref_metrics["delta_IDF1"])
        best_step = "reference"
        for row in decision_rows:
            if str(row.get("step", "")) == "reference" or str(row.get("status", "")) != "success":
                continue
            row_hota = float(row.get("delta_HOTA", 0.0) or 0.0)
            row_idf1 = float(row.get("delta_IDF1", 0.0) or 0.0)
            if row_hota > best_hota + 1e-9 or (abs(row_hota - best_hota) <= 1e-9 and row_idf1 > best_idf1 + 1e-9):
                best_hota = row_hota
                best_idf1 = row_idf1
                best_step = str(row.get("step", "reference"))
        for variant in variants:
            step = str(variant["step"])
            child_out = out_root / str(variant["suffix"])
            child_summary_csv = child_out / "summary.csv"
            metrics_delta_csv = child_out / "metrics_delta.csv"
            runtime_compare_csv = child_out / "runtime_compare.csv"
            log_path = logs_dir / f"{step}.log"
            existing_row = get_row(queue_rows, step)
            if existing_row is None:
                raise KeyError(f"Missing queue row: {step}")

            child_complete = False
            if child_summary_csv.is_file() and metrics_delta_csv.is_file() and runtime_compare_csv.is_file():
                try:
                    ensure_child_success(child_summary_csv)
                    child_complete = True
                except Exception:
                    child_complete = False
            if child_complete:
                metrics = read_metrics_delta(metrics_delta_csv)
                runtime = read_runtime(runtime_compare_csv)
                decision = "candidate"
                if (
                    float(metrics["delta_HOTA"]) > best_hota + 1e-9
                    or (
                        abs(float(metrics["delta_HOTA"]) - best_hota) <= 1e-9
                        and float(metrics["delta_IDF1"]) > best_idf1 + 1e-9
                    )
                ):
                    best_hota = float(metrics["delta_HOTA"])
                    best_idf1 = float(metrics["delta_IDF1"])
                    best_step = step
                    decision = "new_best"
                upsert_decision_row(
                    decision_rows,
                    {
                        "step": step,
                        "variant": variant["variant"],
                        "status": "success",
                        "out_dir": str(child_out),
                        "summary_csv": str(child_summary_csv),
                        "metrics_delta_csv": str(metrics_delta_csv),
                        "runtime_compare_csv": str(runtime_compare_csv),
                        "owner_max_hits": int(variant["owner_max_hits"]),
                        "max_owner_edge_deficit": float(variant["max_owner_edge_deficit"]),
                        "delta_HOTA": metrics["delta_HOTA"],
                        "delta_AssA": metrics["delta_AssA"],
                        "delta_IDF1": metrics["delta_IDF1"],
                        "delta_MOTA": metrics["delta_MOTA"],
                        "delta_IDs": metrics["delta_IDs"],
                        "delta_Frag": metrics["delta_Frag"],
                        "selected_matches": int(runtime["selected_matches"]),
                        "candidate_rows": int(runtime["candidate_rows"]),
                        "decision": decision,
                        "notes": variant["notes"],
                    },
                )
                update_row(
                    queue_rows,
                    step,
                    status="success",
                    finished_at=child_finished_at(child_summary_csv),
                    out_dir=str(child_out),
                    summary_csv=str(child_summary_csv),
                    log_path=str(log_path),
                    notes=(
                        f"{variant['notes']} "
                        f"delta_HOTA={metrics['delta_HOTA']:.3f} "
                        f"delta_IDF1={metrics['delta_IDF1']:.3f} "
                        f"selected={int(runtime['selected_matches'])}"
                    ),
                )
                write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
                write_rows(decision_csv, DECISION_FIELDS, decision_rows)
                continue

            if str(existing_row.get("status", "")) == "running":
                update_row(
                    queue_rows,
                    step,
                    status="pending",
                    notes=f"{variant['notes']} | resumed_after_interrupted_queue",
                )
            update_row(queue_rows, step, status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
            cmd = build_smoke_cmd(
                out_dir=child_out,
                reuse_raw_from=reuse_raw_from,
                seq_names=[str(seq) for seq in args.seq_names],
                owner_max_hits=int(variant["owner_max_hits"]),
                max_owner_edge_deficit=float(variant["max_owner_edge_deficit"]),
            )
            return_code = run_step(cmd, log_path, cwd=REPO_ROOT)
            if return_code != 0:
                raise RuntimeError(f"Child run failed: {step}")
            ensure_child_success(child_summary_csv)
            metrics = read_metrics_delta(metrics_delta_csv)
            runtime = read_runtime(runtime_compare_csv)
            decision = "candidate"
            if (
                float(metrics["delta_HOTA"]) > best_hota + 1e-9
                or (
                    abs(float(metrics["delta_HOTA"]) - best_hota) <= 1e-9
                    and float(metrics["delta_IDF1"]) > best_idf1 + 1e-9
                )
            ):
                best_hota = float(metrics["delta_HOTA"])
                best_idf1 = float(metrics["delta_IDF1"])
                best_step = step
                decision = "new_best"
            upsert_decision_row(
                decision_rows,
                {
                    "step": step,
                    "variant": variant["variant"],
                    "status": "success",
                    "out_dir": str(child_out),
                    "summary_csv": str(child_summary_csv),
                    "metrics_delta_csv": str(metrics_delta_csv),
                    "runtime_compare_csv": str(runtime_compare_csv),
                    "owner_max_hits": int(variant["owner_max_hits"]),
                    "max_owner_edge_deficit": float(variant["max_owner_edge_deficit"]),
                    "delta_HOTA": metrics["delta_HOTA"],
                    "delta_AssA": metrics["delta_AssA"],
                    "delta_IDF1": metrics["delta_IDF1"],
                    "delta_MOTA": metrics["delta_MOTA"],
                    "delta_IDs": metrics["delta_IDs"],
                    "delta_Frag": metrics["delta_Frag"],
                    "selected_matches": int(runtime["selected_matches"]),
                    "candidate_rows": int(runtime["candidate_rows"]),
                    "decision": decision,
                    "notes": variant["notes"],
                },
            )
            update_row(
                queue_rows,
                step,
                status="success",
                finished_at=now_iso(),
                out_dir=str(child_out),
                summary_csv=str(child_summary_csv),
                log_path=str(log_path),
                notes=(
                    f"{variant['notes']} "
                    f"delta_HOTA={metrics['delta_HOTA']:.3f} "
                    f"delta_IDF1={metrics['delta_IDF1']:.3f} "
                    f"selected={int(runtime['selected_matches'])}"
                ),
            )
            write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
            write_rows(decision_csv, DECISION_FIELDS, decision_rows)

        for row in decision_rows:
            if str(row.get("step", "")) == best_step:
                row["decision"] = "best_after_queue"
            elif str(row.get("decision", "")) == "reference_best_so_far":
                row["decision"] = "reference"
        write_rows(decision_csv, DECISION_FIELDS, decision_rows)
        append_registry(
            summary_csv,
            out_root,
            "success",
            f"preassoc competition follow-up queue finished best_step={best_step} best_delta_HOTA={best_hota:.3f}",
            registry_csv,
        )
    except Exception as exc:
        finished_at = now_iso()
        for row in queue_rows:
            if str(row.get("status", "")) == "running":
                row["status"] = "failed"
                row["finished_at"] = finished_at
                row["notes"] = f"{row.get('notes', '')} | failed: {exc}".strip()
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
        write_rows(decision_csv, DECISION_FIELDS, decision_rows)
        append_registry(summary_csv, out_root, "failed", f"preassoc competition follow-up queue failed: {exc}", registry_csv)
        raise


if __name__ == "__main__":
    main()
