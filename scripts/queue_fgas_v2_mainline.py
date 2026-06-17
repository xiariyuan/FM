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

SELECTION_FIELDS = [
    "mode",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDSW",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue FGAS-v2 mainline after the current FGAS full7 run finishes.")
    parser.add_argument(
        "--wait-summary-csv",
        default=str(REPO_ROOT / "outputs" / "fgas_block_halfsplit_full7_20260331_1" / "summary.csv"),
    )
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--out-root", default="")
    parser.add_argument("--dataset-root", default="/gemini/code/datasets")
    parser.add_argument("--benchmark", default="MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--smoke-seq", default="MOT17-05-FRCNN")
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
        "scripts/queue_fgas_v2_mainline.py",
        "--dataset",
        "MOT17",
        "--split",
        "queued_after_full7",
        "--tracker-family",
        "botsort_fgas",
        "--variant",
        run_root.name,
        "--tag",
        "fgas_v2_mainline",
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


def build_runner_cmd(
    *,
    out_root: Path,
    seq_names: List[str],
    skip_train: bool,
    nofreq_checkpoint: str = "",
    full_checkpoint: str = "",
    assignment_mode: str,
    row_nomatch_weight: float,
) -> List[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_fgas_block_halfsplit.py"),
        "--out-root",
        str(out_root),
        "--dataset-root",
        "/gemini/code/datasets",
        "--benchmark",
        "MOT17",
        "--split",
        "train",
        "--seq-names",
        *seq_names,
        "--arch",
        "v2_trackdet",
        "--train-device",
        "cuda",
        "--track-device",
        "gpu",
        "--hidden-dim",
        "96",
        "--stage-embed-dim",
        "16",
        "--num-heads",
        "4",
        "--num-attn-layers",
        "2",
        "--col-bce-weight",
        "0.25",
        "--track-high-thresh",
        "0.5",
        "--proximity-thresh",
        "0.9",
        "--appearance-thresh",
        "0.25",
        "--fgas-topk",
        "5",
        "--fgas-max-rows",
        "3",
        "--fgas-max-cols",
        "3",
        "--fgas-blend-weight",
        "0.5",
        "--fgas-assignment-mode",
        str(assignment_mode),
        "--fgas-row-nomatch-weight",
        str(row_nomatch_weight),
    ]
    if skip_train:
        cmd.append("--skip-train")
        cmd.extend(["--nofreq-checkpoint", str(nofreq_checkpoint)])
        cmd.extend(["--full-checkpoint", str(full_checkpoint)])
    return cmd


def load_nofreq_delta(metrics_delta_csv: Path) -> Dict[str, float]:
    rows = read_rows(metrics_delta_csv)
    for row in rows:
        if str(row.get("name", "")) == "nofreq":
            return {
                "delta_HOTA": float(row.get("delta_HOTA", 0.0)),
                "delta_AssA": float(row.get("delta_AssA", 0.0)),
                "delta_IDF1": float(row.get("delta_IDF1", 0.0)),
                "delta_MOTA": float(row.get("delta_MOTA", 0.0)),
                "delta_IDSW": float(row.get("delta_IDSW", 0.0)),
            }
    raise ValueError(f"Missing nofreq row in {metrics_delta_csv}")


def best_mode(replace_delta: Dict[str, float], blend_delta: Dict[str, float]) -> str:
    replace_key = (
        float(replace_delta["delta_HOTA"]),
        float(replace_delta["delta_AssA"]),
        float(replace_delta["delta_IDF1"]),
        -float(replace_delta["delta_IDSW"]),
    )
    blend_key = (
        float(blend_delta["delta_HOTA"]),
        float(blend_delta["delta_AssA"]),
        float(blend_delta["delta_IDF1"]),
        -float(blend_delta["delta_IDSW"]),
    )
    return "replace" if replace_key >= blend_key else "blend"


def main() -> None:
    args = parse_args()
    queue_name = Path(args.out_root).name if args.out_root else f"fgas_v2_mainline_{timestamp_tag()}"
    out_root = Path(args.out_root) if args.out_root else REPO_ROOT / "outputs" / queue_name
    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "summary.csv"
    selection_csv = out_root / "selection.csv"
    wait_summary = Path(args.wait_summary_csv)

    rows: List[Dict[str, object]] = [
        {
            "step": "wait_current_full7",
            "name": f"{queue_name}_wait_current_full7",
            "status": "pending",
            "out_dir": str(wait_summary.parent),
            "summary_csv": str(wait_summary),
            "log_path": str(out_root / "logs" / "wait_current_full7.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"wait for {wait_summary}",
        },
        {
            "step": "smoke_replace",
            "name": f"{queue_name}_smoke_replace",
            "status": "pending",
            "out_dir": str(out_root / "smoke_replace"),
            "summary_csv": str(out_root / "smoke_replace" / "summary.csv"),
            "log_path": str(out_root / "logs" / "smoke_replace.log"),
            "started_at": "",
            "finished_at": "",
            "notes": "v2_trackdet smoke with replace assignment and row no-match damping",
        },
        {
            "step": "smoke_blend",
            "name": f"{queue_name}_smoke_blend",
            "status": "pending",
            "out_dir": str(out_root / "smoke_blend"),
            "summary_csv": str(out_root / "smoke_blend" / "summary.csv"),
            "log_path": str(out_root / "logs" / "smoke_blend.log"),
            "started_at": "",
            "finished_at": "",
            "notes": "v2_trackdet smoke with blend assignment and no row no-match damping",
        },
        {
            "step": "full7_best",
            "name": f"{queue_name}_full7_best",
            "status": "pending",
            "out_dir": str(out_root / "full7_best"),
            "summary_csv": str(out_root / "full7_best" / "summary.csv"),
            "log_path": str(out_root / "logs" / "full7_best.log"),
            "started_at": "",
            "finished_at": "",
            "notes": "best v2_trackdet smoke config promoted to full7",
        },
    ]
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    append_registry(summary_csv, out_root, "running", "fgas v2 mainline queued", args.registry_csv)

    try:
        update_row(rows, "wait_current_full7", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        wait_log = Path(next(row["log_path"] for row in rows if row["step"] == "wait_current_full7"))
        wait_log.parent.mkdir(parents=True, exist_ok=True)
        with wait_log.open("w", encoding="utf-8") as handle:
            handle.write(f"[started_at] {now_iso()}\n")
            handle.write(f"[wait_summary_csv] {wait_summary}\n")
            while not queue_finished(wait_summary):
                handle.write(f"[poll] {now_iso()} waiting\n")
                handle.flush()
                time.sleep(max(15, int(args.poll_seconds)))
            handle.write(f"[finished_at] {now_iso()}\n")
        update_row(rows, "wait_current_full7", status="success", finished_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        smoke_replace_dir = out_root / "smoke_replace"
        update_row(rows, "smoke_replace", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        smoke_replace_cmd = build_runner_cmd(
            out_root=smoke_replace_dir,
            seq_names=[str(args.smoke_seq)],
            skip_train=False,
            assignment_mode="replace",
            row_nomatch_weight=0.35,
        )
        rc = run_step(smoke_replace_cmd, Path(next(row["log_path"] for row in rows if row["step"] == "smoke_replace")), cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"smoke_replace failed with exit code {rc}")
        update_row(rows, "smoke_replace", status="success", finished_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        nofreq_ckpt = smoke_replace_dir / "train_nofreq" / "best.pt"
        full_ckpt = smoke_replace_dir / "train_full" / "best.pt"

        smoke_blend_dir = out_root / "smoke_blend"
        update_row(rows, "smoke_blend", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        smoke_blend_cmd = build_runner_cmd(
            out_root=smoke_blend_dir,
            seq_names=[str(args.smoke_seq)],
            skip_train=True,
            nofreq_checkpoint=str(nofreq_ckpt),
            full_checkpoint=str(full_ckpt),
            assignment_mode="blend",
            row_nomatch_weight=0.0,
        )
        rc = run_step(smoke_blend_cmd, Path(next(row["log_path"] for row in rows if row["step"] == "smoke_blend")), cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"smoke_blend failed with exit code {rc}")
        update_row(rows, "smoke_blend", status="success", finished_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        replace_delta = load_nofreq_delta(smoke_replace_dir / "metrics_delta.csv")
        blend_delta = load_nofreq_delta(smoke_blend_dir / "metrics_delta.csv")
        write_rows(
            selection_csv,
            SELECTION_FIELDS,
            [
                {"mode": "replace", **replace_delta},
                {"mode": "blend", **blend_delta},
            ],
        )
        chosen_mode = best_mode(replace_delta, blend_delta)
        chosen_assignment = "replace" if chosen_mode == "replace" else "blend"
        chosen_nomatch = 0.35 if chosen_mode == "replace" else 0.0

        full7_dir = out_root / "full7_best"
        update_row(
            rows,
            "full7_best",
            status="running",
            started_at=now_iso(),
            notes=f"chosen_mode={chosen_mode} selection_csv={selection_csv}",
        )
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        full7_cmd = build_runner_cmd(
            out_root=full7_dir,
            seq_names=list(args.full7_seqs),
            skip_train=True,
            nofreq_checkpoint=str(nofreq_ckpt),
            full_checkpoint=str(full_ckpt),
            assignment_mode=chosen_assignment,
            row_nomatch_weight=chosen_nomatch,
        )
        rc = run_step(full7_cmd, Path(next(row["log_path"] for row in rows if row["step"] == "full7_best")), cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"full7_best failed with exit code {rc}")
        update_row(rows, "full7_best", status="success", finished_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        append_registry(summary_csv, out_root, "success", "fgas v2 mainline queue finished", args.registry_csv)
    except Exception as exc:
        for row in rows:
            if str(row.get("status", "")) == "running":
                row["status"] = "failed"
                row["finished_at"] = now_iso()
                row["notes"] = f"{row.get('notes', '')} error={exc}".strip()
                break
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        append_registry(summary_csv, out_root, "failed", f"fgas v2 mainline queue failed: {exc}", args.registry_csv)
        raise


if __name__ == "__main__":
    main()
