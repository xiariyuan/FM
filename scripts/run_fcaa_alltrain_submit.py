#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
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

SUBMIT_FIELDS = [
    "name",
    "checkpoint",
    "dataset",
    "out_dir",
    "zip_path",
    "latest_zip_txt",
    "precheck_log",
    "run_log",
    "fcaa_label",
    "fcaa_trigger_mode",
    "fcaa_trigger_margin",
    "fcaa_lambda",
    "fcaa_topk",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final FCAA all-train build + MOT17 test packaging.")
    parser.add_argument("--dataset-root", default="/gemini/code/datasets")
    parser.add_argument("--benchmark", default="MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seq-names", nargs="*", default=[])
    parser.add_argument("--pairbank-device", default="cuda")
    parser.add_argument("--optimizer", choices=["adamw", "lbfgs"], default="lbfgs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-weight", type=float, default=3.0)
    parser.add_argument("--ambiguous-oversample", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--trigger-mode", default="shared_det_top1")
    parser.add_argument("--trigger-margin", type=float, default=0.05)
    parser.add_argument("--fcaa-lambda", type=float, default=0.3)
    parser.add_argument("--fcaa-topk", type=int, default=3)
    parser.add_argument("--fcaa-label", default="freq_final_alltrain_public21")
    parser.add_argument("--submit-data-root", default="/gemini/code/datasets")
    parser.add_argument("--submit-track-profile", default="")
    parser.add_argument("--reuse-pairbank-dir", default="")
    parser.add_argument("--out-root", default="")
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


def update_row(rows: List[Dict[str, object]], step: str, **updates: object) -> None:
    for row in rows:
        if str(row["step"]) == step:
            row.update(updates)
            return
    raise KeyError(f"Missing queue step: {step}")


def run_step(cmd: List[str], log_path: Path, *, cwd: Path, env: Dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[started_at] {now_iso()}\n")
        handle.write(f"[cwd] {cwd}\n")
        handle.write("[cmd] " + " ".join(cmd) + "\n\n")
        handle.flush()
        process = subprocess.run(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
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
        "scripts/run_fcaa_alltrain_submit.py",
        "--dataset",
        "MOT17",
        "--split",
        "train_full_to_test",
        "--tracker-family",
        "botsort_fcaa",
        "--variant",
        run_root.name,
        "--tag",
        "fcaa_alltrain_submit",
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def seq_note(seq_names: List[str]) -> str:
    return "|".join(seq_names) if seq_names else "ALL_TRAIN_SEQS"


def read_single_row(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, got {len(rows)}")
    return rows[0]


def copy_step_timestamps(rows: List[Dict[str, object]], step: str, started_at: str, finished_at: str, notes: str) -> None:
    update_row(rows, step, status="success", started_at=started_at, finished_at=finished_at, notes=notes)


def main() -> None:
    args = parse_args()
    queue_name = Path(args.out_root).name if args.out_root else f"fcaa_alltrain_submit_{timestamp_tag()}"
    out_root = Path(args.out_root) if args.out_root else REPO_ROOT / "outputs" / queue_name
    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "summary.csv"

    reuse_pairbank_dir = Path(args.reuse_pairbank_dir) if args.reuse_pairbank_dir else None
    pairbank_dir = reuse_pairbank_dir if reuse_pairbank_dir else out_root / "pairbank_full"
    train_freq_dir = out_root / "train_freq_final"
    submit_dir = out_root / "submit_freq"
    submit_summary_csv = submit_dir / "summary.csv"
    seqs_note = seq_note(list(args.seq_names))

    rows: List[Dict[str, object]] = [
        {
            "step": "pairbank_full",
            "name": f"{queue_name}_pairbank_full",
            "status": "pending",
            "out_dir": str(pairbank_dir),
            "summary_csv": str(pairbank_dir / "summary.csv"),
            "log_path": str(out_root / "logs" / "pairbank_full.log"),
            "started_at": "",
            "finished_at": "",
            "notes": (
                f"subset=full grouping=shared_det_top1 seqs={seqs_note}"
                if reuse_pairbank_dir is None
                else f"reused pairbank dir={pairbank_dir}"
            ),
        },
        {
            "step": "train_freq_final",
            "name": f"{queue_name}_train_freq_final",
            "status": "pending",
            "out_dir": str(train_freq_dir),
            "summary_csv": str(train_freq_dir / "summary.csv"),
            "log_path": str(out_root / "logs" / "train_freq_final.log"),
            "started_at": "",
            "finished_at": "",
            "notes": "mode=freq all-train optimizer=lbfgs val_pairbank=train_pairbank",
        },
        {
            "step": "submit_freq",
            "name": f"{queue_name}_submit_freq",
            "status": "pending",
            "out_dir": str(submit_dir),
            "summary_csv": str(submit_summary_csv),
            "log_path": str(out_root / "logs" / "submit_freq.log"),
            "started_at": "",
            "finished_at": "",
            "notes": (
                f"dataset=MOT17 label={args.fcaa_label} trigger={args.trigger_mode} "
                f"margin={args.trigger_margin} lambda={args.fcaa_lambda} topk={args.fcaa_topk}"
            ),
        },
    ]
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    append_registry(summary_csv, out_root, "running", "fcaa all-train submit queue started", args.registry_csv)

    try:
        if reuse_pairbank_dir is not None:
            reused_summary = read_single_row(pairbank_dir / "summary.csv")
            if str(reused_summary.get("status", "")) != "success":
                raise RuntimeError(f"Reused pairbank is not successful: {pairbank_dir / 'summary.csv'}")
            copy_step_timestamps(
                rows,
                "pairbank_full",
                started_at=str(reused_summary.get("started_at", "")),
                finished_at=str(reused_summary.get("finished_at", "")),
                notes=f"reused pairbank dir={pairbank_dir}",
            )
            write_rows(summary_csv, QUEUE_FIELDS, rows)
        else:
            update_row(rows, "pairbank_full", status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            pairbank_cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "build_fcaa_pairbank.py"),
                "--dataset-root",
                args.dataset_root,
                "--benchmark",
                args.benchmark,
                "--split",
                args.split,
                "--subset",
                "full",
                "--grouping",
                "shared_det_top1",
                "--out-dir",
                str(pairbank_dir),
                "--dataset-name",
                f"{queue_name}_pairbank_full",
                "--device",
                args.pairbank_device,
                "--top-k",
                str(args.top_k),
            ]
            if args.seq_names:
                pairbank_cmd.extend(["--seq-names", *list(args.seq_names)])
            pairbank_rc = run_step(pairbank_cmd, Path(rows[0]["log_path"]), cwd=REPO_ROOT)
            if pairbank_rc != 0:
                update_row(rows, "pairbank_full", status="failed", finished_at=now_iso(), notes=f"exit_code={pairbank_rc}")
                write_rows(summary_csv, QUEUE_FIELDS, rows)
                append_registry(summary_csv, out_root, "failed", "fcaa all-train submit queue failed in pairbank step", args.registry_csv)
                return
            update_row(rows, "pairbank_full", status="success", finished_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)

        pairbank_jsonl = pairbank_dir / "pairbank.jsonl"
        update_row(rows, "train_freq_final", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        train_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_fcaa_pair_scorer.py"),
            "--pairbank-jsonl",
            str(pairbank_jsonl),
            "--val-pairbank-jsonl",
            str(pairbank_jsonl),
            "--out-dir",
            str(train_freq_dir),
            "--run-name",
            f"{queue_name}_freq_final",
            "--mode",
            "freq",
            "--optimizer",
            args.optimizer,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--positive-weight",
            str(args.positive_weight),
            "--ambiguous-oversample",
            str(args.ambiguous_oversample),
            "--seed",
            str(args.seed),
        ]
        train_rc = run_step(train_cmd, Path(rows[1]["log_path"]), cwd=REPO_ROOT)
        if train_rc != 0:
            update_row(rows, "train_freq_final", status="failed", finished_at=now_iso(), notes=f"exit_code={train_rc}")
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            append_registry(summary_csv, out_root, "failed", "fcaa all-train submit queue failed in train step", args.registry_csv)
            return
        update_row(rows, "train_freq_final", status="success", finished_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        checkpoint = train_freq_dir / "best.pt"
        submit_summary_row: Dict[str, object] = {
            "name": f"{queue_name}_submit_freq",
            "checkpoint": str(checkpoint),
            "dataset": args.benchmark,
            "out_dir": str(submit_dir),
            "zip_path": "",
            "latest_zip_txt": str(submit_dir / "latest_zip.txt"),
            "precheck_log": str(submit_dir / "precheck.log"),
            "run_log": str(submit_dir / "run.log"),
            "fcaa_label": args.fcaa_label,
            "fcaa_trigger_mode": args.trigger_mode,
            "fcaa_trigger_margin": args.trigger_margin,
            "fcaa_lambda": args.fcaa_lambda,
            "fcaa_topk": args.fcaa_topk,
            "status": "running",
            "error": "",
        }
        write_rows(submit_summary_csv, SUBMIT_FIELDS, [submit_summary_row])

        update_row(rows, "submit_freq", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        submit_cmd = [
            "bash",
            str(REPO_ROOT / "scripts" / "run_botsort_fcaa_submit.sh"),
            str(checkpoint),
            args.benchmark,
            str(submit_dir),
            args.submit_data_root,
        ]
        submit_env = dict(os.environ)
        submit_env["REPO_ROOT"] = str(REPO_ROOT)
        submit_env["FCAA_LABEL"] = str(args.fcaa_label)
        submit_env["FCAA_TRIGGER_MODE"] = str(args.trigger_mode)
        submit_env["FCAA_TRIGGER_MARGIN"] = str(args.trigger_margin)
        submit_env["FCAA_LAMBDA"] = str(args.fcaa_lambda)
        submit_env["FCAA_TOPK"] = str(args.fcaa_topk)
        if args.submit_track_profile:
            submit_env["TRACK_PROFILE"] = str(args.submit_track_profile)
        submit_rc = run_step(submit_cmd, Path(rows[2]["log_path"]), cwd=REPO_ROOT, env=submit_env)
        if submit_rc != 0:
            submit_summary_row["status"] = "failed"
            submit_summary_row["error"] = f"exit_code={submit_rc}"
            write_rows(submit_summary_csv, SUBMIT_FIELDS, [submit_summary_row])
            update_row(rows, "submit_freq", status="failed", finished_at=now_iso(), notes=f"exit_code={submit_rc}")
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            append_registry(summary_csv, out_root, "failed", "fcaa all-train submit queue failed in submit step", args.registry_csv)
            return

        latest_zip_txt = submit_dir / "latest_zip.txt"
        zip_path = latest_zip_txt.read_text(encoding="utf-8").strip() if latest_zip_txt.is_file() else ""
        submit_summary_row["zip_path"] = zip_path
        submit_summary_row["status"] = "success"
        write_rows(submit_summary_csv, SUBMIT_FIELDS, [submit_summary_row])
        update_row(rows, "submit_freq", status="success", finished_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        train_summary = read_single_row(train_freq_dir / "summary.csv")
        notes = (
            "fcaa all-train submit queue success "
            f"best_epoch={train_summary.get('best_epoch', '')} "
            f"train_groups={train_summary.get('train_groups', '')} "
            f"val_ambiguous_top1={train_summary.get('val_ambiguous_top1', '')} "
            f"zip={zip_path}"
        )
        append_registry(summary_csv, out_root, "success", notes, args.registry_csv)
    except Exception as exc:
        if submit_summary_csv.exists():
            submit_summary_row = read_single_row(submit_summary_csv)
            submit_summary_row["status"] = "failed"
            submit_summary_row["error"] = str(exc)
            write_rows(submit_summary_csv, SUBMIT_FIELDS, [submit_summary_row])
        for step in ("pairbank_full", "train_freq_final", "submit_freq"):
            for row in rows:
                if str(row["step"]) == step and str(row["status"]) == "running":
                    row["status"] = "failed"
                    row["finished_at"] = now_iso()
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        append_registry(summary_csv, out_root, "failed", f"fcaa all-train submit queue exception: {exc}", args.registry_csv)
        raise


if __name__ == "__main__":
    main()
