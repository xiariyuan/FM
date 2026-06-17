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
    "probe_summary_csv",
    "candidate_rows",
    "positive_gt_rows",
    "best_det_unmatched_rows",
    "best_det_matched_other_rows",
    "decision",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-gated next-10h queue for pre-association stale-continuation probes.")
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
        "scripts/queue_deep_ocsort_preassoc_next10h.py",
        "--dataset",
        "MOT17",
        "--split",
        "val_half",
        "--tracker-family",
        "deep_ocsort_preassoc_probe",
        "--variant",
        run_root.name,
        "--tag",
        "deep_ocsort_preassoc_next10h",
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


def build_probe_cmd(out_dir: Path, seqs: List[str], profile: Dict[str, object]) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_deep_ocsort_preassoc_stale_probe.py"),
        "--out-root",
        str(out_dir),
        "--seq-names",
        *seqs,
        "--preassoc-stale-probe-min-time-since-update",
        str(profile["min_tsu"]),
        "--preassoc-stale-probe-max-time-since-update",
        str(profile["max_tsu"]),
        "--preassoc-stale-probe-min-hits",
        str(profile["min_hits"]),
        "--preassoc-stale-probe-min-box-iou",
        str(profile["min_iou"]),
        "--preassoc-stale-probe-min-combined-score",
        str(profile["min_combined_score"]),
    ]


def read_probe_summary(probe_summary_csv: Path) -> Dict[str, float]:
    rows = read_rows(probe_summary_csv)
    for row in rows:
        if str(row.get("seq_name", "")) == "COMBINED":
            return {
                "candidate_rows": float(row.get("preassoc_stale_probe_candidate_rows", 0.0) or 0.0),
                "positive_gt_rows": float(row.get("preassoc_stale_probe_positive_gt_rows", 0.0) or 0.0),
                "best_det_unmatched_rows": float(row.get("preassoc_stale_probe_best_det_unmatched_rows", 0.0) or 0.0),
                "best_det_matched_other_rows": float(row.get("preassoc_stale_probe_best_det_matched_other_rows", 0.0) or 0.0),
            }
    raise ValueError(f"Missing COMBINED row in {probe_summary_csv}")


def main() -> None:
    args = parse_args()
    queue_name = Path(args.out_root).name if args.out_root else f"deep_ocsort_preassoc_next10h_{timestamp_tag()}"
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
            "min_combined_score": 0.0,
        },
        {
            "name": "relaxed",
            "min_tsu": 1,
            "max_tsu": 10,
            "min_hits": 10,
            "min_iou": 0.60,
            "min_combined_score": 0.0,
        },
        {
            "name": "wide",
            "min_tsu": 1,
            "max_tsu": 12,
            "min_hits": 6,
            "min_iou": 0.50,
            "min_combined_score": 0.0,
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
                        f"{scope} preassoc stale probe profile={profile['name']} "
                        f"tsu={profile['min_tsu']}-{profile['max_tsu']} "
                        f"hits>={profile['min_hits']} iou>={profile['min_iou']:.2f}"
                    ),
                }
            )
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    write_rows(decision_csv, DECISION_FIELDS, decision_rows)
    append_registry(summary_csv, out_root, "running", "deep ocsort preassoc next10h queue started", registry_csv)

    promoted = False
    found_positive = False

    try:
        for profile in profiles:
            quick_step = f"{profile['name']}_quick"
            quick_out = out_root / quick_step
            update_row(rows, quick_step, status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            quick_row = next(item for item in rows if item["step"] == quick_step)
            return_code = run_step(
                build_probe_cmd(quick_out, list(args.quick_seqs), profile),
                Path(str(quick_row["log_path"])),
                cwd=REPO_ROOT,
            )
            if return_code != 0:
                update_row(rows, quick_step, status="failed", finished_at=now_iso())
                write_rows(summary_csv, QUEUE_FIELDS, rows)
                raise RuntimeError(f"Step failed: {quick_step}")
            ensure_child_success(Path(str(quick_row["summary_csv"])))
            quick_probe = read_probe_summary(quick_out / "probe_summary.csv")
            positive_gt_rows = int(round(float(quick_probe["positive_gt_rows"])))
            candidate_rows = int(round(float(quick_probe["candidate_rows"])))
            found_positive = found_positive or (positive_gt_rows > 0)
            decision = "promote_full7" if positive_gt_rows > 0 else "continue_profile_sweep"
            note = (
                f"candidate_rows={candidate_rows} positive_gt_rows={positive_gt_rows} "
                f"best_det_unmatched_rows={int(round(float(quick_probe['best_det_unmatched_rows'])))}"
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
                    "probe_summary_csv": str(quick_out / "probe_summary.csv"),
                    "candidate_rows": candidate_rows,
                    "positive_gt_rows": positive_gt_rows,
                    "best_det_unmatched_rows": int(round(float(quick_probe["best_det_unmatched_rows"]))),
                    "best_det_matched_other_rows": int(round(float(quick_probe["best_det_matched_other_rows"]))),
                    "decision": decision,
                    "notes": note,
                }
            )
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            write_rows(decision_csv, DECISION_FIELDS, decision_rows)

            full7_step = f"{profile['name']}_full7"
            if positive_gt_rows <= 0:
                update_row(
                    rows,
                    full7_step,
                    status="skipped",
                    finished_at=now_iso(),
                    notes=f"skipped because quick positive_gt_rows=0 under profile={profile['name']}",
                )
                write_rows(summary_csv, QUEUE_FIELDS, rows)
                continue

            promoted = True
            update_row(rows, full7_step, status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            full7_out = out_root / full7_step
            full7_row = next(item for item in rows if item["step"] == full7_step)
            return_code = run_step(
                build_probe_cmd(full7_out, list(args.full7_seqs), profile),
                Path(str(full7_row["log_path"])),
                cwd=REPO_ROOT,
            )
            if return_code != 0:
                update_row(rows, full7_step, status="failed", finished_at=now_iso())
                write_rows(summary_csv, QUEUE_FIELDS, rows)
                raise RuntimeError(f"Step failed: {full7_step}")
            ensure_child_success(Path(str(full7_row["summary_csv"])))
            full7_probe = read_probe_summary(full7_out / "probe_summary.csv")
            full7_positive = int(round(float(full7_probe["positive_gt_rows"])))
            full7_candidate = int(round(float(full7_probe["candidate_rows"])))
            full7_note = (
                f"candidate_rows={full7_candidate} positive_gt_rows={full7_positive} "
                f"best_det_matched_other_rows={int(round(float(full7_probe['best_det_matched_other_rows'])))}"
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
                    "probe_summary_csv": str(full7_out / "probe_summary.csv"),
                    "candidate_rows": full7_candidate,
                    "positive_gt_rows": full7_positive,
                    "best_det_unmatched_rows": int(round(float(full7_probe["best_det_unmatched_rows"]))),
                    "best_det_matched_other_rows": int(round(float(full7_probe["best_det_matched_other_rows"]))),
                    "decision": "implement_preassoc_intervention",
                    "notes": full7_note,
                }
            )
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            write_rows(decision_csv, DECISION_FIELDS, decision_rows)
            skip_note = f"skipped because queue already promoted and completed on profile={profile['name']}"
            skip_time = now_iso()
            for pending_row in rows:
                if str(pending_row.get("status", "")) != "pending":
                    continue
                pending_row["status"] = "skipped"
                pending_row["finished_at"] = skip_time
                pending_row["notes"] = skip_note
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            break

        final_note = "preassoc queue complete"
        if promoted:
            final_note = "preassoc queue complete: positive GT stale rows found, next step is intervention before primary association"
        elif not found_positive:
            final_note = "preassoc queue complete: no positive GT stale rows found on quick profiles, redesign probe or move earlier in dataflow"
        append_registry(summary_csv, out_root, "success", final_note, registry_csv)
    except Exception:
        append_registry(summary_csv, out_root, "failed", "deep ocsort preassoc next10h queue failed", registry_csv)
        raise


if __name__ == "__main__":
    main()
