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
    parser = argparse.ArgumentParser(description="Second-stage overnight queue for FGAS acceptance-gate calibration.")
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
        "--train-jsonl",
        default=str(REPO_ROOT / "outputs" / "fgas_acceptance_train_hard3x4_20260401_smoke" / "acceptance_dataset.jsonl"),
    )
    parser.add_argument(
        "--combo-train-jsonl",
        default=str(REPO_ROOT / "outputs" / "fgas_acceptance_train_combo_20260401_smoke" / "acceptance_dataset.jsonl"),
    )
    parser.add_argument(
        "--val-jsonl",
        default=str(REPO_ROOT / "outputs" / "fgas_acceptance_val_looseB_20260401_smoke" / "acceptance_dataset.jsonl"),
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
    parser.add_argument("--enable-final-export", action="store_true")
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
        "scripts/queue_fgas_acceptance_stage2_night.py",
        "--dataset",
        "MOT17",
        "--split",
        "stage2_night_queue",
        "--tracker-family",
        "deep_ocsort_fgas",
        "--variant",
        run_root.name,
        "--tag",
        "fgas_acceptance_stage2_night",
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


def queue_succeeded(summary_csv: Path) -> bool:
    rows = read_rows(summary_csv)
    if not rows:
        return False
    statuses = {str(row.get("status", "")).strip() for row in rows}
    return statuses == {"success"}


def ensure_success(summary_csv: Path) -> None:
    rows = read_rows(summary_csv)
    if not rows:
        raise FileNotFoundError(f"Missing summary rows: {summary_csv}")
    statuses = {str(row.get("status", "")).strip() for row in rows}
    if statuses != {"success"}:
        raise RuntimeError(f"Unexpected status in {summary_csv}: {sorted(statuses)}")


def train_gate_cmd(out_dir: Path, train_jsonl: str, val_jsonl: str, hidden_dim: int, dropout: float = 0.0) -> List[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_fgas_acceptance_gate.py"),
        "--train-jsonl",
        str(train_jsonl),
        "--val-jsonl",
        str(val_jsonl),
        "--out-dir",
        str(out_dir),
        "--device",
        "cuda",
        "--epochs",
        "50",
        "--batch-size",
        "128",
        "--hidden-dim",
        str(hidden_dim),
    ]
    if float(dropout) > 0.0:
        cmd.extend(["--dropout", str(dropout)])
    return cmd


def eval_cmd(
    out_dir: Path,
    seqs: List[str],
    resolver_ckpt: str,
    gate_ckpt: str,
    gate_thresh: float,
    *,
    base_margin: float = 1.0,
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
        "--fgas-soft-row-base-margin-thresh",
        str(base_margin),
        "--fgas-acceptance-gate-checkpoint",
        str(gate_ckpt),
        "--fgas-acceptance-gate-thresh",
        str(gate_thresh),
        "--fgas-soft-lambda",
        "0.5",
    ]


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
    queue_name = Path(args.out_root).name if args.out_root else f"fgas_acceptance_stage2_night_{timestamp_tag()}"
    out_root = resolve_repo_path(args.out_root) if args.out_root else (REPO_ROOT / "outputs" / queue_name).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "summary.csv"
    wait_summary = resolve_repo_path(args.wait_summary_csv)
    resolver_checkpoint = resolve_repo_path(args.resolver_checkpoint)
    hard32_gate_checkpoint = resolve_repo_path(args.hard32_gate_checkpoint)
    hard64_gate_checkpoint = resolve_repo_path(args.hard64_gate_checkpoint)
    train_jsonl = resolve_repo_path(args.train_jsonl)
    combo_train_jsonl = resolve_repo_path(args.combo_train_jsonl)
    val_jsonl = resolve_repo_path(args.val_jsonl)
    registry_csv = resolve_repo_path(args.registry_csv)

    hard128_dir = out_root / "train_hard128"
    combo64_dir = out_root / "train_combo64"
    hard128_ckpt = hard128_dir / "best.pt"
    combo64_ckpt = combo64_dir / "best.pt"

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
        },
        {
            "step": "train_hard128",
            "name": f"{queue_name}_train_hard128",
            "status": "pending",
            "out_dir": str(hard128_dir),
            "summary_csv": str(hard128_dir / "summary.csv"),
            "log_path": str(out_root / "logs" / "train_hard128.log"),
            "started_at": "",
            "finished_at": "",
            "notes": "hard3x4 acceptance gate hidden_dim=128",
        },
        {
            "step": "train_combo64",
            "name": f"{queue_name}_train_combo64",
            "status": "pending",
            "out_dir": str(combo64_dir),
            "summary_csv": str(combo64_dir / "summary.csv"),
            "log_path": str(out_root / "logs" / "train_combo64.log"),
            "started_at": "",
            "finished_at": "",
            "notes": "combo acceptance gate hidden_dim=64",
        },
    ]

    eval_specs = [
        ("eval_hard32_t055_full7", str(hard32_gate_checkpoint), 0.55, list(args.full7_seqs), 1.0),
        ("eval_hard32_t065_full7", str(hard32_gate_checkpoint), 0.65, list(args.full7_seqs), 1.0),
        ("eval_hard64_t055_full7", str(hard64_gate_checkpoint), 0.55, list(args.full7_seqs), 1.0),
        ("eval_hard64_t065_full7", str(hard64_gate_checkpoint), 0.65, list(args.full7_seqs), 1.0),
        ("eval_hard32_t060_m010_full7", str(hard32_gate_checkpoint), 0.60, list(args.full7_seqs), 0.10),
        ("eval_hard64_t060_m010_full7", str(hard64_gate_checkpoint), 0.60, list(args.full7_seqs), 0.10),
        ("eval_hard128_t060_full7", str(hard128_ckpt.resolve()), 0.60, list(args.full7_seqs), 1.0),
        ("eval_combo64_t060_full7", str(combo64_ckpt.resolve()), 0.60, list(args.full7_seqs), 1.0),
        ("eval_hard32_t065_hardslice", str(hard32_gate_checkpoint), 0.65, list(args.hardslice_seqs), 1.0),
        ("eval_hard64_t065_hardslice", str(hard64_gate_checkpoint), 0.65, list(args.hardslice_seqs), 1.0),
        ("eval_hard128_t060_hardslice", str(hard128_ckpt.resolve()), 0.60, list(args.hardslice_seqs), 1.0),
        ("eval_combo64_t060_hardslice", str(combo64_ckpt.resolve()), 0.60, list(args.hardslice_seqs), 1.0),
    ]
    for step_name, gate_ckpt, gate_thresh, seqs, base_margin in eval_specs:
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
                "notes": f"gate={Path(gate_ckpt).name} thresh={gate_thresh:.2f} base_margin={base_margin:.2f} seqs={' '.join(seqs)}",
            }
        )

    if args.enable_final_export:
        rows.append(
            {
                "step": "submit_bestguess_hard64_t060",
                "name": f"{queue_name}_submit_bestguess_hard64_t060",
                "status": "pending",
                "out_dir": str(out_root / "submit_bestguess_hard64_t060"),
                "summary_csv": str(out_root / "submit_bestguess_hard64_t060" / "summary.csv"),
                "log_path": str(out_root / "logs" / "submit_bestguess_hard64_t060.log"),
                "started_at": "",
                "finished_at": "",
                "notes": "single final 42-file export for current best-guess candidate hard64_t060",
            }
        )

    write_rows(summary_csv, QUEUE_FIELDS, rows)
    append_registry(summary_csv, out_root, "running", "FGAS acceptance stage2 night queue started", str(registry_csv))

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
        if not queue_succeeded(wait_summary):
            raise RuntimeError(f"Upstream queue did not finish successfully: {wait_summary}")
        update_row(rows, "wait_acceptance_night", status="success", finished_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        step_cmds: List[tuple[str, List[str], Path]] = [
            ("train_hard128", train_gate_cmd(hard128_dir, str(train_jsonl), str(val_jsonl), 128), REPO_ROOT),
            ("train_combo64", train_gate_cmd(combo64_dir, str(combo_train_jsonl), str(val_jsonl), 64), REPO_ROOT),
        ]
        for step_name, gate_ckpt, gate_thresh, seqs, base_margin in eval_specs:
            step_cmds.append(
                (
                    step_name,
                    eval_cmd(out_root / step_name, seqs, str(resolver_checkpoint), gate_ckpt, gate_thresh, base_margin=base_margin),
                    REPO_ROOT,
                )
            )
        if args.enable_final_export:
            step_cmds.append(
                (
                    "submit_bestguess_hard64_t060",
                    submit_cmd(out_root / "submit_bestguess_hard64_t060", str(resolver_checkpoint), str(hard64_gate_checkpoint), 0.60),
                    REPO_ROOT,
                )
            )

        for step_name, cmd, cwd in step_cmds:
            update_row(rows, step_name, status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            row = next(item for item in rows if item["step"] == step_name)
            return_code = run_step(cmd, Path(str(row["log_path"])), cwd=cwd)
            if return_code != 0:
                update_row(rows, step_name, status="failed", finished_at=now_iso())
                write_rows(summary_csv, QUEUE_FIELDS, rows)
                raise RuntimeError(f"Step failed: {step_name}")
            ensure_success(Path(str(row["summary_csv"])))
            update_row(rows, step_name, status="success", finished_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)

        append_registry(summary_csv, out_root, "success", "FGAS acceptance stage2 night queue complete", str(registry_csv))
    except Exception:
        append_registry(summary_csv, out_root, "failed", "FGAS acceptance stage2 night queue failed", str(registry_csv))
        raise


if __name__ == "__main__":
    main()
