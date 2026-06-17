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
    "scope",
    "profile",
    "status",
    "out_dir",
    "summary_csv",
    "metrics_delta_csv",
    "runtime_compare_csv",
    "lifecycle_candidate_pairs",
    "lifecycle_matches",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDs",
    "delta_Frag",
    "decision",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-gated next-10h queue for raw baseline vs raw+lifecycle Deep-OC-SORT validation.")
    parser.add_argument("--out-root", default="")
    parser.add_argument("--start-profile", choices=["anchor", "relaxed", "wide"], default="anchor")
    parser.add_argument(
        "--quick-seqs",
        nargs="*",
        default=[
            "MOT17-05-FRCNN",
            "MOT17-11-FRCNN",
        ],
    )
    parser.add_argument(
        "--full7-seqs",
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
        "scripts/queue_deep_ocsort_lifecycle_next10h.py",
        "--dataset",
        "MOT17",
        "--split",
        "val_half",
        "--tracker-family",
        "deep_ocsort_lifecycle",
        "--variant",
        run_root.name,
        "--tag",
        "deep_ocsort_lifecycle_next10h",
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


def build_smoke_cmd(out_dir: Path, seqs: List[str], profile: Dict[str, object]) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_deep_ocsort_lifecycle_smoke.py"),
        "--out-root",
        str(out_dir),
        "--seq-names",
        *seqs,
        "--lifecycle-reclaim-min-time-since-update",
        str(profile["min_tsu"]),
        "--lifecycle-reclaim-max-time-since-update",
        str(profile["max_tsu"]),
        "--lifecycle-reclaim-min-hits",
        str(profile["min_hits"]),
        "--lifecycle-reclaim-min-box-iou",
        str(profile["min_iou"]),
        "--lifecycle-reclaim-min-box-area-ratio",
        str(profile["min_area_ratio"]),
        "--lifecycle-reclaim-max-box-area-ratio",
        str(profile["max_area_ratio"]),
    ]


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
        if str(row.get("name", "")) == "lifecycle":
            return {
                "lifecycle_candidate_pairs": float(row.get("lifecycle_reclaim_candidate_pairs", 0.0) or 0.0),
                "lifecycle_matches": float(row.get("lifecycle_reclaim_matches", 0.0) or 0.0),
            }
    raise ValueError(f"Missing lifecycle runtime row in {runtime_compare_csv}")


def main() -> None:
    args = parse_args()
    queue_name = Path(args.out_root).name if args.out_root else f"deep_ocsort_lifecycle_next10h_{timestamp_tag()}"
    out_root = Path(args.out_root).resolve() if args.out_root else (REPO_ROOT / "outputs" / queue_name).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "summary.csv"
    decision_csv = out_root / "decision.csv"
    registry_csv = str(Path(args.registry_csv).resolve())

    profiles = [
        {
            "name": "anchor",
            "min_tsu": 2,
            "max_tsu": 8,
            "min_hits": 15,
            "min_iou": 0.65,
            "min_area_ratio": 0.50,
            "max_area_ratio": 2.00,
        },
        {
            "name": "relaxed",
            "min_tsu": 1,
            "max_tsu": 10,
            "min_hits": 10,
            "min_iou": 0.60,
            "min_area_ratio": 0.45,
            "max_area_ratio": 2.20,
        },
        {
            "name": "wide",
            "min_tsu": 1,
            "max_tsu": 12,
            "min_hits": 6,
            "min_iou": 0.50,
            "min_area_ratio": 0.35,
            "max_area_ratio": 2.80,
        },
    ]
    start_idx = next(index for index, profile in enumerate(profiles) if profile["name"] == args.start_profile)
    profiles = profiles[start_idx:]

    rows: List[Dict[str, object]] = []
    decision_rows: List[Dict[str, object]] = []
    for profile in profiles:
        for scope in ("quick", "full7"):
            step = f"{profile['name']}_{scope}"
            rows.append(
                {
                    "step": step,
                    "name": f"{queue_name}_{step}",
                    "status": "pending",
                    "out_dir": str(out_root / step),
                    "summary_csv": str(out_root / step / "summary.csv"),
                    "log_path": str(out_root / "logs" / f"{step}.log"),
                    "started_at": "",
                    "finished_at": "",
                    "notes": (
                        f"{scope} lifecycle smoke profile={profile['name']} "
                        f"tsu={profile['min_tsu']}-{profile['max_tsu']} "
                        f"hits>={profile['min_hits']} iou>={profile['min_iou']:.2f}"
                    ),
                }
            )
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    write_rows(decision_csv, DECISION_FIELDS, decision_rows)
    append_registry(summary_csv, out_root, "running", "deep ocsort lifecycle next10h queue started", registry_csv)

    any_candidates = False
    any_promoted = False

    try:
        for profile in profiles:
            quick_step = f"{profile['name']}_quick"
            quick_out = out_root / quick_step
            update_row(rows, quick_step, status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            quick_row = next(item for item in rows if item["step"] == quick_step)
            return_code = run_step(
                build_smoke_cmd(quick_out, list(args.quick_seqs), profile),
                Path(str(quick_row["log_path"])),
                cwd=REPO_ROOT,
            )
            if return_code != 0:
                update_row(rows, quick_step, status="failed", finished_at=now_iso())
                write_rows(summary_csv, QUEUE_FIELDS, rows)
                raise RuntimeError(f"Step failed: {quick_step}")
            ensure_child_success(Path(str(quick_row["summary_csv"])))
            quick_metrics = read_metrics_delta(quick_out / "metrics_delta.csv")
            quick_runtime = read_runtime(quick_out / "runtime_compare.csv")
            quick_candidates = int(round(float(quick_runtime["lifecycle_candidate_pairs"])))
            quick_matches = int(round(float(quick_runtime["lifecycle_matches"])))
            decision = "promote_full7" if (quick_candidates > 0 or quick_matches > 0) else "skip_full7"
            note = (
                f"candidates={quick_candidates} matches={quick_matches} "
                f"delta_HOTA={quick_metrics['delta_HOTA']:.3f} delta_IDF1={quick_metrics['delta_IDF1']:.3f}"
            )
            update_row(rows, quick_step, status="success", finished_at=now_iso(), notes=note)
            decision_rows.append(
                {
                    "step": quick_step,
                    "scope": "quick",
                    "profile": profile["name"],
                    "status": "success",
                    "out_dir": str(quick_out),
                    "summary_csv": str(quick_out / "summary.csv"),
                    "metrics_delta_csv": str(quick_out / "metrics_delta.csv"),
                    "runtime_compare_csv": str(quick_out / "runtime_compare.csv"),
                    "lifecycle_candidate_pairs": quick_candidates,
                    "lifecycle_matches": quick_matches,
                    "delta_HOTA": quick_metrics["delta_HOTA"],
                    "delta_AssA": quick_metrics["delta_AssA"],
                    "delta_IDF1": quick_metrics["delta_IDF1"],
                    "delta_MOTA": quick_metrics["delta_MOTA"],
                    "delta_IDs": quick_metrics["delta_IDs"],
                    "delta_Frag": quick_metrics["delta_Frag"],
                    "decision": decision,
                    "notes": note,
                }
            )
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            write_rows(decision_csv, DECISION_FIELDS, decision_rows)

            full7_step = f"{profile['name']}_full7"
            if quick_candidates <= 0 and quick_matches <= 0:
                update_row(
                    rows,
                    full7_step,
                    status="skipped",
                    finished_at=now_iso(),
                    notes=f"skipped because quick produced no lifecycle candidates under profile={profile['name']}",
                )
                write_rows(summary_csv, QUEUE_FIELDS, rows)
                continue

            any_candidates = True
            update_row(rows, full7_step, status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            full7_out = out_root / full7_step
            full7_row = next(item for item in rows if item["step"] == full7_step)
            return_code = run_step(
                build_smoke_cmd(full7_out, list(args.full7_seqs), profile),
                Path(str(full7_row["log_path"])),
                cwd=REPO_ROOT,
            )
            if return_code != 0:
                update_row(rows, full7_step, status="failed", finished_at=now_iso())
                write_rows(summary_csv, QUEUE_FIELDS, rows)
                raise RuntimeError(f"Step failed: {full7_step}")
            ensure_child_success(Path(str(full7_row["summary_csv"])))
            full7_metrics = read_metrics_delta(full7_out / "metrics_delta.csv")
            full7_runtime = read_runtime(full7_out / "runtime_compare.csv")
            full7_candidates = int(round(float(full7_runtime["lifecycle_candidate_pairs"])))
            full7_matches = int(round(float(full7_runtime["lifecycle_matches"])))
            any_promoted = True
            full7_note = (
                f"candidates={full7_candidates} matches={full7_matches} "
                f"delta_HOTA={full7_metrics['delta_HOTA']:.3f} delta_IDF1={full7_metrics['delta_IDF1']:.3f}"
            )
            update_row(rows, full7_step, status="success", finished_at=now_iso(), notes=full7_note)
            decision_rows.append(
                {
                    "step": full7_step,
                    "scope": "full7",
                    "profile": profile["name"],
                    "status": "success",
                    "out_dir": str(full7_out),
                    "summary_csv": str(full7_out / "summary.csv"),
                    "metrics_delta_csv": str(full7_out / "metrics_delta.csv"),
                    "runtime_compare_csv": str(full7_out / "runtime_compare.csv"),
                    "lifecycle_candidate_pairs": full7_candidates,
                    "lifecycle_matches": full7_matches,
                    "delta_HOTA": full7_metrics["delta_HOTA"],
                    "delta_AssA": full7_metrics["delta_AssA"],
                    "delta_IDF1": full7_metrics["delta_IDF1"],
                    "delta_MOTA": full7_metrics["delta_MOTA"],
                    "delta_IDs": full7_metrics["delta_IDs"],
                    "delta_Frag": full7_metrics["delta_Frag"],
                    "decision": "validated_full7",
                    "notes": full7_note,
                }
            )
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            write_rows(decision_csv, DECISION_FIELDS, decision_rows)

        final_note = "queue complete"
        if not any_candidates:
            final_note = "queue complete: no lifecycle candidates on quick across all profiles; move candidate formation earlier than post-association"
        elif not any_promoted:
            final_note = "queue complete: quick produced candidates but no full7 promotion was executed"
        append_registry(summary_csv, out_root, "success", final_note, registry_csv)
    except Exception:
        append_registry(summary_csv, out_root, "failed", "deep ocsort lifecycle next10h queue failed", registry_csv)
        raise


if __name__ == "__main__":
    main()
