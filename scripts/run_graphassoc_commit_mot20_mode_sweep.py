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
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "graph_assoc_commit_mot17_expand_20260424_150607" / "train_policy" / "best.pt"
REFERENCE_RUN_ROOT = REPO_ROOT / "outputs" / "graph_assoc_commit_mot17_expand_20260424_150607" / "mot20_eval"

QUEUE_FIELDS = [
    "step",
    "name",
    "status",
    "run_root",
    "summary_csv",
    "log_path",
    "started_at",
    "finished_at",
    "decision_mode",
    "score_margin",
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
    parser = argparse.ArgumentParser(description="Run a MOT20 val_half sweep over action-policy decision modes.")
    parser.add_argument("--run-root", default=str(REPO_ROOT / "outputs" / f"graphassoc_commit_mot20_mode_sweep_{ts}"))
    parser.add_argument("--queue-name", default=f"graphassoc_commit_mot20_mode_sweep_{ts}")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-margin", type=float, default=0.11)
    parser.add_argument("--seq-ids", nargs="+", type=int, default=[2, 5])
    parser.add_argument("--resume", action="store_true", help="reuse successful rows in an existing run_root and continue remaining steps")
    parser.add_argument(
        "--skip-existing-results",
        action="store_true",
        help="pass through to the eval script so already materialized sequence results are skipped",
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


def append_row(
    rows: List[Dict[str, object]],
    *,
    step: str,
    name: str,
    status: str,
    run_root: Path,
    summary_csv: Path,
    log_path: Path,
    decision_mode: str,
    score_margin: float,
    notes: str = "",
    params_json: str = "",
) -> None:
    rows.append(
        {
            "step": step,
            "name": name,
            "status": status,
            "run_root": str(run_root),
            "summary_csv": str(summary_csv),
            "log_path": str(log_path),
            "started_at": "",
            "finished_at": "",
            "decision_mode": decision_mode,
            "score_margin": score_margin,
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
        if str(row.get("step", "")) == str(step):
            return row
    return None


def update_row(rows: List[Dict[str, object]], step: str, **updates: object) -> None:
    row = find_row(rows, step)
    if row is None:
        raise KeyError(f"Missing queue step: {step}")
    row.update(updates)


def append_log_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def run_step(cmd: List[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[started_at] {now_iso()}\n")
        handle.write(f"[cwd] {cwd}\n")
        handle.write("[cmd] " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[finished_at] {now_iso()}\n")
        handle.write(f"[return_code] {proc.returncode}\n")
    return int(proc.returncode)


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
        "scripts/run_graphassoc_commit_mot20_mode_sweep.py",
        "--dataset",
        "MOT20",
        "--split",
        "val_half",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        "graphassoc_commit_mot20_mode_sweep",
        "--run-root",
        str(Path(args.run_root).expanduser().resolve()),
        "--summary-csv",
        str(summary_csv),
        "--log-path",
        str(log_path),
        "--notes",
        notes,
        "--extra",
        f"checkpoint={args.checkpoint}",
        f"device={args.device}",
        f"score_margin={args.score_margin}",
        f"seq_ids={'|'.join(str(v) for v in args.seq_ids)}",
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
        "scripts/run_graphassoc_commit_mot20_mode_sweep.py",
        "--dataset",
        "MOT20",
        "--split",
        "val_half",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        "graphassoc_commit_mot20_mode_sweep",
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
        "--extra",
        f"checkpoint={args.checkpoint}",
        f"device={args.device}",
        f"score_margin={args.score_margin}",
        f"seq_ids={'|'.join(str(v) for v in args.seq_ids)}",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


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


def collect_result(variant: Dict[str, object], run_root: Path) -> Dict[str, object]:
    metrics = parse_metrics_delta(run_root)
    identical_count = parse_identical_count(run_root)
    return {
        **variant,
        **metrics,
        "run_root": str(run_root),
        "identical_count": int(identical_count),
    }


def load_existing_result(variant: Dict[str, object]) -> Dict[str, object] | None:
    existing_run_root_raw = str(variant.get("existing_run_root", "") or "").strip()
    if not existing_run_root_raw:
        return None
    existing_run_root = Path(existing_run_root_raw).expanduser()
    if not existing_run_root.is_dir():
        return None
    return collect_result(variant, existing_run_root)


def build_variant(
    *,
    name: str,
    decision_mode: str,
    score_margin: float,
    seq_ids: List[int],
    notes: str,
    variant_name: str,
    checkpoint: str,
    existing_run_root: str = "",
) -> Dict[str, object]:
    extra_args = [
        "--variant-name",
        variant_name,
        "--graph-assoc-commit-checkpoint",
        str(checkpoint),
        "--graph-assoc-commit-device",
        "cuda",
        "--graph-assoc-commit-score-margin",
        f"{score_margin}",
        "--graph-assoc-commit-decision-mode",
        decision_mode,
    ]
    return {
        "name": name,
        "variant_name": variant_name,
        "decision_mode": decision_mode,
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
        *(["--skip-existing-results"] if args.skip_existing_results else []),
    ]


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
            step=step,
            name=str(variant["name"]),
            status="pending",
            run_root=child_run_root,
            summary_csv=summary_csv,
            log_path=child_log,
            decision_mode=str(variant["decision_mode"]),
            score_margin=float(variant["score_margin"]),
            notes=str(variant["notes"]),
            params_json=json.dumps(
                {
                    "variant_name": variant["variant_name"],
                    "decision_mode": variant["decision_mode"],
                    "score_margin": variant["score_margin"],
                    "extra_args": variant["extra_args"],
                },
                ensure_ascii=False,
            ),
        )
    else:
        child_run_root = Path(str(existing_row.get("run_root") or child_run_root))
        child_log = Path(str(existing_row.get("log_path") or child_log))

    if args.resume and existing_row is not None and str(existing_row.get("status", "")) in {"success", "reference"}:
        try:
            return collect_result(variant, child_run_root)
        except FileNotFoundError:
            pass

    existing_result = load_existing_result(variant)
    if existing_result is not None:
        update_row(
            rows,
            step,
            status="reference" if str(variant.get("existing_run_root", "") or "") else "success",
            finished_at=now_iso(),
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
    rc = run_step(cmd, child_log, cwd=REPO_ROOT)
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


def rank_results(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        results,
        key=lambda row: (
            float(row.get("delta_hota", -999.0)),
            float(row.get("delta_idf1", -999.0)),
            float(row.get("delta_assa", -999.0)),
            float(row.get("delta_mota", -999.0)),
            -abs(int(row.get("delta_ids", 999999))),
            -abs(int(row.get("delta_frag", 999999))),
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
    rows: List[Dict[str, object]] = [dict(row) for row in read_csv_rows(summary_csv)]

    reference_variant = build_variant(
        name="existing_action_margin",
        decision_mode="action_margin",
        score_margin=float(args.score_margin),
        seq_ids=list(args.seq_ids),
        notes="existing MOT20 action_margin reference",
        variant_name="botsort_graphassoc_setslot_action_margin_reference",
        checkpoint=str(args.checkpoint),
        existing_run_root=str(REFERENCE_RUN_ROOT),
    )
    new_variants = [
        build_variant(
            name="policy_score",
            decision_mode="policy_score",
            score_margin=float(args.score_margin),
            seq_ids=list(args.seq_ids),
            notes="compare policy_score on the provided checkpoint",
            variant_name="botsort_graphassoc_policy_score",
            checkpoint=str(args.checkpoint),
        ),
        build_variant(
            name="selection_score",
            decision_mode="selection_score",
            score_margin=float(args.score_margin),
            seq_ids=list(args.seq_ids),
            notes="compare selection_score on the provided checkpoint",
            variant_name="botsort_graphassoc_selection_score",
            checkpoint=str(args.checkpoint),
        ),
        build_variant(
            name="legacy_policy_score",
            decision_mode="legacy_policy_score",
            score_margin=float(args.score_margin),
            seq_ids=list(args.seq_ids),
            notes="compare legacy_policy_score on the provided checkpoint",
            variant_name="botsort_graphassoc_legacy_policy_score",
            checkpoint=str(args.checkpoint),
        ),
        build_variant(
            name="residual_policy_score",
            decision_mode="policy_score",
            score_margin=float(args.score_margin),
            seq_ids=list(args.seq_ids),
            notes="compare residual policy_score on the provided checkpoint",
            variant_name="botsort_graphassoc_residual_policy_score",
            checkpoint=str(args.checkpoint),
        ),
        build_variant(
            name="gain_pred",
            decision_mode="gain_pred",
            score_margin=float(args.score_margin),
            seq_ids=list(args.seq_ids),
            notes="compare gain_pred on the provided checkpoint",
            variant_name="botsort_graphassoc_gain_pred",
            checkpoint=str(args.checkpoint),
        ),
        build_variant(
            name="router_margin",
            decision_mode="router_margin",
            score_margin=float(args.score_margin),
            seq_ids=list(args.seq_ids),
            notes="compare router_margin on the provided checkpoint",
            variant_name="botsort_graphassoc_router_margin",
            checkpoint=str(args.checkpoint),
        ),
        build_variant(
            name="router_confidence",
            decision_mode="router_confidence",
            score_margin=float(args.score_margin),
            seq_ids=list(args.seq_ids),
            notes="compare router_confidence on the provided checkpoint",
            variant_name="botsort_graphassoc_router_confidence",
            checkpoint=str(args.checkpoint),
        ),
    ]

    append_log_line(queue_log, f"[queue_started] {now_iso()}")
    queue_plan_status(args, "running", summary_csv, queue_log, notes="graph-assoc MOT20 decision-mode sweep started")
    queue_registry(args, "running", summary_csv, queue_log, notes="graph-assoc MOT20 decision-mode sweep started")

    try:
        if find_row(rows, "reference_action_margin_existing") is None:
            append_row(
                rows,
                step="reference_action_margin_existing",
                name=str(reference_variant["name"]),
                status="reference",
                run_root=REFERENCE_RUN_ROOT,
                summary_csv=summary_csv,
                log_path=REFERENCE_RUN_ROOT / "logs" / "compare.log",
                decision_mode=str(reference_variant["decision_mode"]),
                score_margin=float(reference_variant["score_margin"]),
                notes=str(reference_variant["notes"]),
                params_json=json.dumps(
                    {
                        "existing_run_root": str(REFERENCE_RUN_ROOT),
                        "checkpoint": str(args.checkpoint),
                        "decision_mode": reference_variant["decision_mode"],
                        "score_margin": reference_variant["score_margin"],
                    },
                    ensure_ascii=False,
                ),
            )
        reference_result = collect_result(reference_variant, REFERENCE_RUN_ROOT)
        update_row(
            rows,
            "reference_action_margin_existing",
            delta_hota=reference_result["delta_hota"],
            delta_assa=reference_result["delta_assa"],
            delta_idf1=reference_result["delta_idf1"],
            delta_mota=reference_result["delta_mota"],
            delta_ids=reference_result["delta_ids"],
            delta_frag=reference_result["delta_frag"],
            identical_count=reference_result["identical_count"],
            finished_at=now_iso(),
            notes=f"{reference_variant['notes']} | complete",
        )
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        new_results: List[Dict[str, object]] = [reference_result]
        for idx, variant in enumerate(new_variants, start=1):
            step = f"{idx:02d}_{variant['name']}"
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

        ranked = rank_results(new_results)
        best = ranked[0] if ranked else None
        if best is not None:
            append_log_line(
                queue_log,
                "best_mode="
                + str(best["decision_mode"])
                + f" delta_HOTA={best['delta_hota']:.6f}"
                + f" delta_AssA={best['delta_assa']:.6f}"
                + f" delta_IDF1={best['delta_idf1']:.6f}",
            )

        append_log_line(queue_log, f"[queue_completed] {now_iso()}")
        queue_plan_status(args, "completed", summary_csv, queue_log, notes="graph-assoc MOT20 decision-mode sweep completed")
        queue_registry(args, "success", summary_csv, queue_log, notes="graph-assoc MOT20 decision-mode sweep completed")
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
