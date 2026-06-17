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
    parser = argparse.ArgumentParser(description="Run a five-hour graph-assoc commit margin follow-up queue.")
    parser.add_argument("--run-root", default=str(REPO_ROOT / "outputs" / f"graphassoc_commit_margin_next5h_{ts}"))
    parser.add_argument("--queue-name", default=f"graphassoc_commit_margin_next5h_{ts}")
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


def append_row(
    rows: List[Dict[str, object]],
    step: str,
    name: str,
    run_root: Path | str,
    log_path: Path | str,
    seq_ids: List[int],
    params_json: str,
    notes: str = "",
) -> None:
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


def find_row(rows: List[Dict[str, object]], step: str) -> Dict[str, object] | None:
    for row in rows:
        if str(row.get("step")) == str(step):
            return row
    return None


def update_row(rows: List[Dict[str, object]], step: str, **updates: object) -> None:
    row = find_row(rows, step)
    if row is not None:
        row.update(updates)
        return
    raise KeyError(f"Missing queue step: {step}")


def append_log_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


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
        "scripts/run_graphassoc_commit_margin_next5h.py",
        "--dataset",
        "MOT20",
        "--split",
        "val_half",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        "graphassoc_commit_margin_next5h",
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
        "scripts/run_graphassoc_commit_margin_next5h.py",
        "--dataset",
        "MOT20",
        "--split",
        "val_half",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        "graphassoc_commit_margin_next5h",
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
        handle.write("[cmd] " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[finished_at] {now_iso()}\n")
        handle.write(f"[return_code] {proc.returncode}\n")
    return int(proc.returncode)


def parse_metrics_delta(run_root: Path) -> Dict[str, float | int]:
    rows = read_csv_rows(run_root / "metrics_delta.csv")
    if not rows:
        raise FileNotFoundError(f"Missing metrics_delta.csv under {run_root}")
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


def child_finished_at(run_root: Path) -> str:
    rows = read_csv_rows(run_root / "summary.csv")
    if not rows:
        return ""
    for preferred_step in ("compare", "graph_eval", "graph_track", "reference_eval"):
        for row in rows:
            if row.get("step") == preferred_step and row.get("finished_at"):
                return str(row.get("finished_at") or "")
    finished_values = [str(row.get("finished_at") or "") for row in rows if row.get("finished_at")]
    return max(finished_values) if finished_values else ""


def try_collect_completed_result(variant: Dict[str, object], run_root: Path) -> Dict[str, object] | None:
    if not (run_root / "metrics_delta.csv").is_file():
        return None
    return collect_result(variant, run_root)


def rank_variants(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        results,
        key=lambda row: (
            float(row.get("delta_hota", -999.0)),
            float(row.get("delta_idf1", -999.0)),
            float(row.get("delta_mota", -999.0)),
            -abs(int(row.get("delta_ids", 999999))),
            -abs(int(row.get("delta_frag", 999999))),
            -int(row.get("identical_count", 999999)),
        ),
        reverse=True,
    )


def build_variant(
    *,
    name: str,
    variant_name: str,
    score_margin: str,
    seq_ids: List[int],
    notes: str,
    replace_rules: bool = True,
    gate_only: bool = False,
    existing_run_root: str = "",
) -> Dict[str, object]:
    extra_args = [
        "--variant-name",
        variant_name,
        "--graph-assoc-commit-checkpoint",
        "outputs/graph_assoc_commit_policy_balanced_policyloss_20260420_3/best.pt",
        "--graph-assoc-commit-device",
        "cpu",
        "--graph-assoc-commit-score-margin",
        score_margin,
    ]
    if replace_rules:
        extra_args.append("--graph-assoc-commit-replace-rules")
    if gate_only:
        extra_args.append("--graph-assoc-commit-gate-only")
    return {
        "name": name,
        "variant_name": variant_name,
        "score_margin": score_margin,
        "seq_ids": list(seq_ids),
        "extra_args": extra_args,
        "notes": notes,
        "existing_run_root": existing_run_root,
    }


def build_experiment_command(
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
        "--seq-ids",
        *[str(v) for v in seq_ids],
        *extra_args,
    ]


def collect_result(variant: Dict[str, object], run_root: Path) -> Dict[str, object]:
    metrics = parse_metrics_delta(run_root)
    identical_count = parse_identical_count(run_root)
    return {
        **variant,
        **metrics,
        "run_root": str(run_root),
        "identical_count": int(identical_count),
    }


def run_variant(
    *,
    args: argparse.Namespace,
    rows: List[Dict[str, object]],
    summary_csv: Path,
    logs_dir: Path,
    base_run_root: Path,
    step: str,
    variant: Dict[str, object],
    experiment_name: str,
) -> Dict[str, object] | None:
    child_run_root = base_run_root / "runs" / step
    child_log = logs_dir / f"{step}.log"
    existing_row = find_row(rows, step)
    if existing_row is None:
        append_row(
            rows,
            step,
            str(variant["name"]),
            child_run_root,
            child_log,
            list(variant["seq_ids"]),
            json.dumps(
                {
                    "variant_name": variant["variant_name"],
                    "score_margin": variant["score_margin"],
                    "extra_args": variant["extra_args"],
                },
                ensure_ascii=False,
            ),
            notes=str(variant["notes"]),
        )
    else:
        child_run_root = Path(str(existing_row.get("run_root") or child_run_root))
        child_log = Path(str(existing_row.get("log_path") or child_log))

    existing_result = try_collect_completed_result(variant, child_run_root)
    if existing_result is not None:
        update_row(
            rows,
            step,
            status="success",
            finished_at=child_finished_at(child_run_root) or str(find_row(rows, step).get("finished_at") or now_iso()),
            delta_hota=existing_result["delta_hota"],
            delta_assa=existing_result["delta_assa"],
            delta_idf1=existing_result["delta_idf1"],
            delta_mota=existing_result["delta_mota"],
            delta_ids=existing_result["delta_ids"],
            delta_frag=existing_result["delta_frag"],
            identical_count=existing_result["identical_count"],
            notes=f"{variant['notes']} | complete",
        )
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        return existing_result

    current_row = find_row(rows, step)
    started_at = str(current_row.get("started_at") or now_iso())
    update_row(rows, step, status="running", started_at=started_at)
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    cmd = build_experiment_command(
        args,
        run_root=child_run_root,
        experiment_name=experiment_name,
        seq_ids=list(variant["seq_ids"]),
        extra_args=list(variant["extra_args"]),
    )
    rc = run_step(cmd, child_log)
    if rc != 0:
        update_row(rows, step, status="failed", finished_at=now_iso(), notes=f"{variant['notes']} | return_code={rc}")
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        return None

    result = collect_result(variant, child_run_root)
    update_row(
        rows,
        step,
        status="success",
        finished_at=now_iso(),
        delta_hota=result["delta_hota"],
        delta_assa=result["delta_assa"],
        delta_idf1=result["delta_idf1"],
        delta_mota=result["delta_mota"],
        delta_ids=result["delta_ids"],
        delta_frag=result["delta_frag"],
        identical_count=result["identical_count"],
        notes=f"{variant['notes']} | complete",
    )
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    return result


def load_existing_result(variant: Dict[str, object]) -> Dict[str, object] | None:
    existing_run_root = str(variant.get("existing_run_root", "") or "").strip()
    if not existing_run_root:
        return None
    path = Path(existing_run_root).expanduser().resolve()
    if not path.is_dir():
        return None
    return collect_result(variant, path)


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    logs_dir = run_root / "logs"
    summary_csv = run_root / "summary.csv"
    queue_log = logs_dir / "queue.log"
    rows: List[Dict[str, object]] = [dict(row) for row in read_csv_rows(summary_csv)]

    seq5_sweeps = [
        build_variant(
            name="seq5_margin011",
            variant_name="botsort_graphassoc_policy_margin011",
            score_margin="0.11",
            seq_ids=[5],
            notes="prune the 6 lowest-confidence replacement edits while keeping the main 0.25 cluster",
        ),
        build_variant(
            name="seq5_margin02495",
            variant_name="botsort_graphassoc_policy_margin02495",
            score_margin="0.2495",
            seq_ids=[5],
            notes="keep only the strongest replacement edits but avoid the sharp drop at 0.25",
        ),
        build_variant(
            name="seq5_margin030",
            variant_name="botsort_graphassoc_policy_margin030",
            score_margin="0.30",
            seq_ids=[5],
            notes="stress-test a much stricter replacement gate",
        ),
    ]

    candidate_catalog = [
        build_variant(
            name="existing_margin003",
            variant_name="botsort_graphassoc_policy_margin03",
            score_margin="0.03",
            seq_ids=[5],
            notes="current best seq5 replacement baseline",
            existing_run_root="outputs/botsort_graphassoc_policy_seq5_margin03_20260421_1",
        ),
        build_variant(
            name="existing_margin025",
            variant_name="botsort_graphassoc_policy_margin25",
            score_margin="0.25",
            seq_ids=[5],
            notes="strict replacement baseline",
            existing_run_root="outputs/botsort_graphassoc_policy_seq5_margin25_20260421_1",
        ),
        build_variant(
            name="existing_gateonly003",
            variant_name="botsort_graphassoc_policy_gateonly03",
            score_margin="0.03",
            seq_ids=[5],
            notes="gate-only stability baseline",
            replace_rules=False,
            gate_only=True,
            existing_run_root="outputs/botsort_graphassoc_policy_seq5_gateonly03_20260421_1",
        ),
    ] + seq5_sweeps

    append_log_line(queue_log, f"[queue_started] {now_iso()}")
    queue_plan_status(args, "running", summary_csv, queue_log, notes="graph-assoc commit margin next5h queue started")
    queue_registry(args, "running", summary_csv, queue_log, notes="graph-assoc commit margin next5h queue started")

    try:
        new_results: List[Dict[str, object]] = []
        for idx, variant in enumerate(seq5_sweeps, start=1):
            step = f"seq5_sweep_{idx}_{variant['name']}"
            exp_name = f"{args.queue_name}_{step}"
            result = run_variant(
                args=args,
                rows=rows,
                summary_csv=summary_csv,
                logs_dir=logs_dir,
                base_run_root=run_root,
                step=step,
                variant=variant,
                experiment_name=exp_name,
            )
            if result is not None:
                new_results.append(result)

        candidate_results: List[Dict[str, object]] = []
        for variant in candidate_catalog:
            if str(variant.get("existing_run_root", "") or "").strip():
                existing = load_existing_result(variant)
                if existing is not None:
                    candidate_results.append(existing)
        candidate_results.extend(new_results)

        ranked = rank_variants(candidate_results)
        top_two = ranked[:2]

        for idx, variant in enumerate(top_two, start=1):
            validation_variant = build_variant(
                name=f"seq2_validate_{variant['name']}",
                variant_name=str(variant["variant_name"]),
                score_margin=str(variant["score_margin"]),
                seq_ids=[2],
                notes=f"seq2 validation for ranked top-{idx} seq5 candidate: {variant['name']}",
                replace_rules="gateonly" not in str(variant["name"]),
                gate_only="gateonly" in str(variant["name"]),
            )
            step = f"seq2_validate_{idx}_{variant['name']}"
            exp_name = f"{args.queue_name}_{step}"
            run_variant(
                args=args,
                rows=rows,
                summary_csv=summary_csv,
                logs_dir=logs_dir,
                base_run_root=run_root,
                step=step,
                variant=validation_variant,
                experiment_name=exp_name,
            )

        append_log_line(queue_log, f"[queue_completed] {now_iso()}")
        queue_plan_status(args, "completed", summary_csv, queue_log, notes="graph-assoc commit margin next5h queue completed")
        queue_registry(args, "success", summary_csv, queue_log, notes="graph-assoc commit margin next5h queue completed")
    except Exception as exc:
        for row in rows:
            if str(row.get("status")) in {"pending", "running"}:
                row["status"] = "failed"
                row["finished_at"] = now_iso()
                row["notes"] = f"{row.get('notes', '')} | queue_exception={exc}".strip()
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        append_log_line(queue_log, f"[queue_failed] {now_iso()} {exc}")
        queue_plan_status(args, "failed", summary_csv, queue_log, notes=str(exc))
        queue_registry(args, "failed", summary_csv, queue_log, notes=str(exc))
        raise


if __name__ == "__main__":
    main()
