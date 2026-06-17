#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


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

RESULT_FIELDS = [
    "step",
    "status",
    "conf_thresh",
    "soft_lambda",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDs",
    "delta_Frag",
    "run_root",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue a small half-val sweep for block-primitive soft-only integration.")
    parser.add_argument("--out-root", default="")
    parser.add_argument(
        "--block-primitive-checkpoint",
        default=str(REPO_ROOT / "outputs" / "fgas_block_primitive_smoke_20260401_1" / "best.pt"),
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
    parser.add_argument("--conf-thresh-values", nargs="*", type=float, default=[0.50, 0.55, 0.60])
    parser.add_argument("--soft-lambda-values", nargs="*", type=float, default=[0.50, 0.60, 0.70])
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
        "scripts/queue_blockprimitive_softonly_sweep.py",
        "--dataset",
        "MOT17",
        "--split",
        "val_half",
        "--tracker-family",
        "deep_ocsort_fgas",
        "--variant",
        run_root.name,
        "--tag",
        "blockprimitive_softonly_sweep",
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


def read_delta(metrics_delta_csv: Path) -> Dict[str, float]:
    rows = read_rows(metrics_delta_csv)
    if not rows:
        raise FileNotFoundError(f"Missing metrics delta: {metrics_delta_csv}")
    row = rows[0]
    return {
        "delta_HOTA": float(row.get("delta_HOTA", 0.0)),
        "delta_AssA": float(row.get("delta_AssA", 0.0)),
        "delta_IDF1": float(row.get("delta_IDF1", 0.0)),
        "delta_MOTA": float(row.get("delta_MOTA", 0.0)),
        "delta_IDs": float(row.get("delta_IDs", 0.0)),
        "delta_Frag": float(row.get("delta_Frag", 0.0)),
    }


def build_eval_cmd(
    *,
    out_root: Path,
    seq_names: List[str],
    block_primitive_checkpoint: Path,
    conf_thresh: float,
    soft_lambda: float,
) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_deep_ocsort_fgas_smoke.py"),
        "--seq-names",
        *seq_names,
        "--checkpoint",
        "",
        "--block-primitive-checkpoint",
        str(block_primitive_checkpoint),
        "--fgas-block-primitive-conf-thresh",
        f"{conf_thresh:.2f}",
        "--fgas-soft-enable",
        "--fgas-soft-lambda",
        f"{soft_lambda:.2f}",
        "--fgas-soft-only-changed-blocks",
        "--disable-controller",
        "--out-root",
        str(out_root),
    ]


def spec_step_name(conf_thresh: float, soft_lambda: float) -> str:
    return f"c{int(round(conf_thresh * 100)):02d}_l{int(round(soft_lambda * 100)):02d}"


def main() -> None:
    args = parse_args()
    queue_name = Path(args.out_root).name if args.out_root else f"blockprimitive_softonly_sweep_{timestamp_tag()}"
    out_root = resolve_repo_path(args.out_root) if args.out_root else (REPO_ROOT / "outputs" / queue_name).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "summary.csv"
    results_csv = out_root / "results.csv"
    registry_csv = resolve_repo_path(args.registry_csv)
    primitive_ckpt = resolve_repo_path(args.block_primitive_checkpoint)

    specs: List[Tuple[str, float, float]] = []
    for conf_thresh in list(args.conf_thresh_values):
        for soft_lambda in list(args.soft_lambda_values):
            specs.append((spec_step_name(float(conf_thresh), float(soft_lambda)), float(conf_thresh), float(soft_lambda)))

    queue_rows: List[Dict[str, object]] = []
    result_rows: List[Dict[str, object]] = []
    for step_name, conf_thresh, soft_lambda in specs:
        run_dir = out_root / step_name
        queue_rows.append(
            {
                "step": step_name,
                "name": f"{queue_name}_{step_name}",
                "status": "pending",
                "out_dir": str(run_dir),
                "summary_csv": str(run_dir / "summary.csv"),
                "log_path": str(out_root / "logs" / f"{step_name}.log"),
                "started_at": "",
                "finished_at": "",
                "notes": f"block primitive soft-only conf={conf_thresh:.2f} lambda={soft_lambda:.2f}",
            }
        )
        result_rows.append(
            {
                "step": step_name,
                "status": "pending",
                "conf_thresh": f"{conf_thresh:.2f}",
                "soft_lambda": f"{soft_lambda:.2f}",
                "delta_HOTA": "",
                "delta_AssA": "",
                "delta_IDF1": "",
                "delta_MOTA": "",
                "delta_IDs": "",
                "delta_Frag": "",
                "run_root": str(run_dir),
            }
        )

    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
    write_rows(results_csv, RESULT_FIELDS, result_rows)
    append_registry(summary_csv, out_root, "running", "block primitive soft-only sweep started", str(registry_csv))

    try:
        for step_name, conf_thresh, soft_lambda in specs:
            update_row(queue_rows, step_name, status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
            row = next(item for item in queue_rows if str(item["step"]) == step_name)
            run_dir = Path(str(row["out_dir"]))
            return_code = run_step(
                build_eval_cmd(
                    out_root=run_dir,
                    seq_names=list(args.seq_names),
                    block_primitive_checkpoint=primitive_ckpt,
                    conf_thresh=float(conf_thresh),
                    soft_lambda=float(soft_lambda),
                ),
                Path(str(row["log_path"])),
                cwd=REPO_ROOT,
            )
            if return_code != 0:
                update_row(queue_rows, step_name, status="failed", finished_at=now_iso())
                write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
                for result_row in result_rows:
                    if str(result_row["step"]) == step_name:
                        result_row["status"] = "failed"
                write_rows(results_csv, RESULT_FIELDS, result_rows)
                raise RuntimeError(f"Step failed: {step_name}")

            sub_summary_csv = Path(str(row["summary_csv"]))
            ensure_success(sub_summary_csv)
            delta = read_delta(run_dir / "metrics_delta.csv")
            update_row(queue_rows, step_name, status="success", finished_at=now_iso())
            for result_row in result_rows:
                if str(result_row["step"]) != step_name:
                    continue
                result_row.update({"status": "success", **delta})
                break
            write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
            write_rows(results_csv, RESULT_FIELDS, result_rows)

        append_registry(summary_csv, out_root, "success", "block primitive soft-only sweep complete", str(registry_csv))
    except Exception:
        append_registry(summary_csv, out_root, "failed", "block primitive soft-only sweep failed", str(registry_csv))
        raise


if __name__ == "__main__":
    main()
