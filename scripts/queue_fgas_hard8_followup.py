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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Follow-up queue for best hard8 acceptance-gate candidate.")
    parser.add_argument("--out-root", default="")
    parser.add_argument(
        "--resolver-checkpoint",
        default=str(REPO_ROOT / "outputs" / "fgas_block_resolver_v3_nofreq_hard3x4_ambig_20260331_1" / "best.pt"),
    )
    parser.add_argument(
        "--gate-checkpoint",
        default=str(REPO_ROOT / "outputs" / "fgas_acceptance_stage3_night_20260401_1" / "train_hard8_64" / "best.pt"),
    )
    parser.add_argument("--gate-thresh", type=float, default=0.50)
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
    parser.add_argument(
        "--hardslice-seqs",
        nargs="*",
        default=[
            "MOT17-05-FRCNN",
            "MOT17-10-FRCNN",
            "MOT17-13-FRCNN",
        ],
    )
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp_tag() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


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
        "scripts/queue_fgas_hard8_followup.py",
        "--dataset",
        "MOT17",
        "--split",
        "hard8_followup",
        "--tracker-family",
        "deep_ocsort_fgas",
        "--variant",
        run_root.name,
        "--tag",
        "fgas_hard8_followup",
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def ensure_success(summary_csv: Path) -> None:
    rows = read_rows(summary_csv)
    if not rows:
        raise FileNotFoundError(f"Missing summary rows: {summary_csv}")
    statuses = {str(row.get("status", "")).strip() for row in rows}
    if statuses != {"success"}:
        raise RuntimeError(f"Unexpected status in {summary_csv}: {sorted(statuses)}")


def eval_cmd(
    out_dir: Path,
    seqs: List[str],
    resolver_ckpt: str,
    gate_ckpt: str,
    gate_thresh: float,
    soft_lambda: float,
) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_deep_ocsort_fgas_smoke.py"),
        "--seq-names",
        *seqs,
        "--checkpoint",
        str(resolver_ckpt),
        "--out-root",
        str(out_dir),
        "--fgas-assignment-mode",
        "blend",
        "--fgas-blend-weight",
        "0.5",
        "--disable-controller",
        "--fgas-soft-enable",
        "--fgas-soft-only-changed-blocks",
        "--fgas-soft-only-changed-frontier",
        "--fgas-acceptance-gate-checkpoint",
        str(gate_ckpt),
        "--fgas-acceptance-gate-thresh",
        str(gate_thresh),
        "--fgas-soft-lambda",
        str(soft_lambda),
    ]


def main() -> None:
    args = parse_args()
    queue_name = Path(args.out_root).name if args.out_root else f"fgas_hard8_followup_{timestamp_tag()}"
    out_root = resolve_repo_path(args.out_root) if args.out_root else (REPO_ROOT / "outputs" / queue_name).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "summary.csv"
    resolver_checkpoint = resolve_repo_path(args.resolver_checkpoint)
    gate_checkpoint = resolve_repo_path(args.gate_checkpoint)
    registry_csv = resolve_repo_path(args.registry_csv)

    rows: List[Dict[str, object]] = []
    specs = [
        ("eval_hard8_64_t050_hardslice", list(args.hardslice_seqs), args.gate_thresh, 0.5, "best hard8 gate on hard slice"),
        ("eval_hard8_64_t050_l030_full7", list(args.full7_seqs), args.gate_thresh, 0.3, "best hard8 gate full7 lambda=0.3"),
        ("eval_hard8_64_t050_l070_full7", list(args.full7_seqs), args.gate_thresh, 0.7, "best hard8 gate full7 lambda=0.7"),
    ]
    for step_name, seqs, gate_thresh, soft_lambda, notes in specs:
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
                "notes": f"{notes} gate={gate_checkpoint.name} thresh={gate_thresh:.2f} lambda={soft_lambda:.2f}",
            }
        )

    write_rows(summary_csv, QUEUE_FIELDS, rows)
    append_registry(summary_csv, out_root, "running", "FGAS hard8 follow-up queue started", str(registry_csv))

    try:
        for step_name, seqs, gate_thresh, soft_lambda, _notes in specs:
            update_row(rows, step_name, status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            row = next(item for item in rows if item["step"] == step_name)
            return_code = run_step(
                eval_cmd(out_root / step_name, seqs, str(resolver_checkpoint), str(gate_checkpoint), float(gate_thresh), float(soft_lambda)),
                Path(str(row["log_path"])),
                cwd=REPO_ROOT,
            )
            if return_code != 0:
                update_row(rows, step_name, status="failed", finished_at=now_iso())
                write_rows(summary_csv, QUEUE_FIELDS, rows)
                raise RuntimeError(f"Step failed: {step_name}")
            ensure_success(Path(str(row["summary_csv"])))
            update_row(rows, step_name, status="success", finished_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)

        append_registry(summary_csv, out_root, "success", "FGAS hard8 follow-up queue complete", str(registry_csv))
    except Exception:
        append_registry(summary_csv, out_root, "failed", "FGAS hard8 follow-up queue failed", str(registry_csv))
        raise


if __name__ == "__main__":
    main()
