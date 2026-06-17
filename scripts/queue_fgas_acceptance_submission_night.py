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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue long-tail MOT17 full submission exports after acceptance-night evals.")
    parser.add_argument(
        "--wait-summary-csv",
        default=str(REPO_ROOT / "outputs" / "fgas_acceptance_night_20260401_1" / "summary.csv"),
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--out-root", default="")
    parser.add_argument(
        "--resolver-checkpoint",
        default=str(REPO_ROOT / "outputs" / "fgas_block_resolver_v3_nofreq_hard3x4_ambig_20260331_1" / "best.pt"),
    )
    parser.add_argument(
        "--hard32-gate-checkpoint",
        default=str(REPO_ROOT / "outputs" / "fgas_acceptance_gate_hard3x4_20260401_smoke" / "best.pt"),
    )
    parser.add_argument(
        "--hard64-gate-checkpoint",
        default=str(REPO_ROOT / "outputs" / "fgas_acceptance_night_20260401_1" / "train_hard64" / "best.pt"),
    )
    parser.add_argument(
        "--linear-gate-checkpoint",
        default=str(REPO_ROOT / "outputs" / "fgas_acceptance_night_20260401_1" / "train_linear" / "best.pt"),
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
        "scripts/queue_fgas_acceptance_submission_night.py",
        "--dataset",
        "MOT17",
        "--split",
        "submission_night_queue",
        "--tracker-family",
        "deep_ocsort_fgas",
        "--variant",
        run_root.name,
        "--tag",
        "fgas_acceptance_submission_night",
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def queue_finished(summary_csv: Path) -> bool:
    rows = read_rows(summary_csv)
    if not rows:
        return False
    statuses = {str(row.get("status", "")).strip() for row in rows}
    if "running" in statuses:
        return False
    if statuses <= {"pending", ""}:
        return False
    return True


def ensure_success(summary_csv: Path) -> None:
    rows = read_rows(summary_csv)
    if not rows:
        raise FileNotFoundError(f"Missing summary rows: {summary_csv}")
    statuses = {str(row.get("status", "")).strip() for row in rows}
    if statuses != {"success"}:
        raise RuntimeError(f"Unexpected status in {summary_csv}: {sorted(statuses)}")


def submit_cmd(out_dir: Path, resolver_ckpt: str, gate_ckpt: str, gate_thresh: float) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "make_mot17_full_submission_deep_ocsort_fgas.py"),
        "--out-root",
        str(out_dir),
        "--checkpoint",
        str(resolver_ckpt),
        "--profile",
        "full",
        "--fgas-profile-mode",
        "soft_acceptance",
        "--fgas-soft-lambda",
        "0.5",
        "--fgas-soft-only-changed-blocks",
        "--fgas-soft-only-changed-frontier",
        "--fgas-acceptance-gate-checkpoint",
        str(gate_ckpt),
        "--fgas-acceptance-gate-thresh",
        str(gate_thresh),
    ]


def main() -> None:
    args = parse_args()
    queue_name = Path(args.out_root).name if args.out_root else f"fgas_acceptance_submission_night_{timestamp_tag()}"
    out_root = Path(args.out_root) if args.out_root else REPO_ROOT / "outputs" / queue_name
    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "summary.csv"
    wait_summary = Path(args.wait_summary_csv)

    submit_specs = [
        ("submit_hard32_t050", args.hard32_gate_checkpoint, 0.50),
        ("submit_hard32_t060", args.hard32_gate_checkpoint, 0.60),
        ("submit_hard32_t070", args.hard32_gate_checkpoint, 0.70),
        ("submit_hard64_t050", args.hard64_gate_checkpoint, 0.50),
        ("submit_hard64_t060", args.hard64_gate_checkpoint, 0.60),
        ("submit_linear_t050", args.linear_gate_checkpoint, 0.50),
    ]

    rows: List[Dict[str, object]] = [
        {
            "step": "wait_acceptance_night",
            "name": f"{queue_name}_wait_acceptance_night",
            "status": "pending",
            "out_dir": str(wait_summary.parent),
            "summary_csv": str(wait_summary),
            "log_path": str(out_root / "logs" / "wait_acceptance_night.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"wait for {wait_summary}",
        }
    ]
    for step_name, gate_ckpt, gate_thresh in submit_specs:
        rows.append(
            {
                "step": step_name,
                "name": f"{queue_name}_{step_name}",
                "status": "pending",
                "out_dir": str(out_root / step_name),
                "summary_csv": str(out_root / step_name / "summary.csv"),
                "log_path": str(out_root / "logs" / f"{step_name}.log"),
                "started_at": "",
                "finished_at": "",
                "notes": f"full 42-file submission gate={Path(gate_ckpt).name} thresh={gate_thresh:.2f}",
            }
        )

    write_rows(summary_csv, QUEUE_FIELDS, rows)
    append_registry(summary_csv, out_root, "running", "FGAS acceptance submission queue started", args.registry_csv)

    try:
        update_row(rows, "wait_acceptance_night", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        wait_log = out_root / "logs" / "wait_acceptance_night.log"
        wait_log.parent.mkdir(parents=True, exist_ok=True)
        with wait_log.open("w", encoding="utf-8") as handle:
            handle.write(f"[started_at] {now_iso()}\n")
            handle.write(f"[wait_summary_csv] {wait_summary}\n")
            while not queue_finished(wait_summary):
                handle.write(f"[poll] {now_iso()} waiting\n")
                handle.flush()
                time.sleep(int(args.poll_seconds))
            handle.write(f"[finished_at] {now_iso()}\n")
        update_row(rows, "wait_acceptance_night", status="success", finished_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        for step_name, gate_ckpt, gate_thresh in submit_specs:
            update_row(rows, step_name, status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            row = next(item for item in rows if item["step"] == step_name)
            cmd = submit_cmd(Path(str(row["out_dir"])), args.resolver_checkpoint, gate_ckpt, gate_thresh)
            return_code = run_step(cmd, Path(str(row["log_path"])), cwd=REPO_ROOT)
            if return_code != 0:
                update_row(rows, step_name, status="failed", finished_at=now_iso())
                write_rows(summary_csv, QUEUE_FIELDS, rows)
                raise RuntimeError(f"Step failed: {step_name}")
            ensure_success(Path(str(row["summary_csv"])))
            update_row(rows, step_name, status="success", finished_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)

        append_registry(summary_csv, out_root, "success", "FGAS acceptance submission queue complete", args.registry_csv)
    except Exception:
        append_registry(summary_csv, out_root, "failed", "FGAS acceptance submission queue failed", args.registry_csv)
        raise


if __name__ == "__main__":
    main()
