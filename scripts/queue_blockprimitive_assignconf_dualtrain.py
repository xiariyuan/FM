#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import torch


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
    "variant",
    "feature_mode",
    "train_status",
    "eval_status",
    "train_dir",
    "eval_dir",
    "checkpoint",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDs",
    "delta_Frag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Night queue for assignconf block primitive retraining and half-val evaluation.")
    parser.add_argument("--out-root", default="")
    parser.add_argument(
        "--train-jsonl",
        default=str(REPO_ROOT / "outputs" / "fgas_blockbank_train_half_looseB_v3ctx_hard3x4_ambig_20260331_1" / "blockbank.jsonl"),
    )
    parser.add_argument(
        "--val-jsonl",
        default=str(REPO_ROOT / "outputs" / "fgas_blockbank_val_half_looseB_v3ctx_20260331_1" / "blockbank.jsonl"),
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
    parser.add_argument("--feature-modes", nargs="*", default=["nofreq", "full"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--stage-embed-dim", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--ambiguous-oversample", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf-thresh", type=float, default=0.60)
    parser.add_argument("--soft-lambda", type=float, default=0.60)
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
        "scripts/queue_blockprimitive_assignconf_dualtrain.py",
        "--dataset",
        "MOT17",
        "--split",
        "blockbank_to_val_half",
        "--tracker-family",
        "deep_ocsort_fgas",
        "--variant",
        run_root.name,
        "--tag",
        "blockprimitive_assignconf_dualtrain",
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def ensure_summary_success(summary_csv: Path) -> None:
    rows = read_rows(summary_csv)
    if not rows:
        raise FileNotFoundError(f"Missing summary rows: {summary_csv}")
    statuses = {str(row.get("status", "")).strip() for row in rows}
    if statuses != {"success"}:
        raise RuntimeError(f"Unexpected status in {summary_csv}: {sorted(statuses)}")


def ensure_single_success(summary_csv: Path) -> None:
    rows = read_rows(summary_csv)
    if len(rows) != 1:
        raise RuntimeError(f"Expected single-row summary in {summary_csv}, got {len(rows)}")
    if str(rows[0].get("status", "")).strip() != "success":
        raise RuntimeError(f"Unexpected status in {summary_csv}: {rows[0].get('status', '')}")


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


def build_train_cmd(args: argparse.Namespace, *, out_dir: Path, feature_mode: str) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_fgas_block_primitive.py"),
        "--train-jsonl",
        str(resolve_repo_path(args.train_jsonl)),
        "--val-jsonl",
        str(resolve_repo_path(args.val_jsonl)),
        "--out-dir",
        str(out_dir),
        "--device",
        str(args.device),
        "--feature-mode",
        str(feature_mode),
        "--epochs",
        str(int(args.epochs)),
        "--batch-size",
        str(int(args.batch_size)),
        "--lr",
        str(float(args.lr)),
        "--weight-decay",
        str(float(args.weight_decay)),
        "--hidden-dim",
        str(int(args.hidden_dim)),
        "--stage-embed-dim",
        str(int(args.stage_embed_dim)),
        "--num-heads",
        str(int(args.num_heads)),
        "--num-layers",
        str(int(args.num_layers)),
        "--dropout",
        str(float(args.dropout)),
        "--ambiguous-oversample",
        str(float(args.ambiguous_oversample)),
        "--seed",
        str(int(args.seed)),
    ]


def build_eval_cmd(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    checkpoint: Path,
) -> List[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_deep_ocsort_fgas_smoke.py"),
        "--seq-names",
        *[str(seq) for seq in args.seq_names],
        "--checkpoint",
        "",
        "--block-primitive-checkpoint",
        str(checkpoint),
        "--fgas-block-primitive-conf-thresh",
        f"{float(args.conf_thresh):.2f}",
        "--fgas-soft-enable",
        "--fgas-soft-lambda",
        f"{float(args.soft_lambda):.2f}",
        "--fgas-soft-only-changed-blocks",
        "--disable-controller",
        "--out-root",
        str(out_dir),
    ]


def main() -> None:
    args = parse_args()
    queue_name = Path(args.out_root).name if args.out_root else f"blockprimitive_assignconf_dualtrain_{timestamp_tag()}"
    out_root = resolve_repo_path(args.out_root) if args.out_root else (REPO_ROOT / "outputs" / queue_name).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "summary.csv"
    results_csv = out_root / "results.csv"
    registry_csv = resolve_repo_path(args.registry_csv)
    logs_dir = out_root / "logs"

    queue_rows: List[Dict[str, object]] = []
    result_rows: List[Dict[str, object]] = []
    feature_modes = [str(mode) for mode in args.feature_modes]
    for feature_mode in feature_modes:
        variant = f"assignconf_{feature_mode}"
        train_dir = out_root / variant / "train"
        eval_dir = out_root / variant / "eval"
        queue_rows.append(
            {
                "step": f"{variant}_train",
                "name": f"{queue_name}_{variant}_train",
                "status": "pending",
                "out_dir": str(train_dir),
                "summary_csv": str(train_dir / "summary.csv"),
                "log_path": str(logs_dir / f"{variant}_train.log"),
                "started_at": "",
                "finished_at": "",
                "notes": f"feature_mode={feature_mode} assignconf train",
            }
        )
        queue_rows.append(
            {
                "step": f"{variant}_eval",
                "name": f"{queue_name}_{variant}_eval",
                "status": "pending",
                "out_dir": str(eval_dir),
                "summary_csv": str(eval_dir / "summary.csv"),
                "log_path": str(logs_dir / f"{variant}_eval.log"),
                "started_at": "",
                "finished_at": "",
                "notes": f"feature_mode={feature_mode} half-val eval conf={float(args.conf_thresh):.2f} lambda={float(args.soft_lambda):.2f}",
            }
        )
        result_rows.append(
            {
                "variant": variant,
                "feature_mode": feature_mode,
                "train_status": "pending",
                "eval_status": "pending",
                "train_dir": str(train_dir),
                "eval_dir": str(eval_dir),
                "checkpoint": str(train_dir / "best.pt"),
                "delta_HOTA": "",
                "delta_AssA": "",
                "delta_IDF1": "",
                "delta_MOTA": "",
                "delta_IDs": "",
                "delta_Frag": "",
            }
        )

    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
    write_rows(results_csv, RESULT_FIELDS, result_rows)
    append_registry(summary_csv, out_root, "running", "assignconf dual-train queue started", str(registry_csv))

    try:
        for feature_mode in feature_modes:
            variant = f"assignconf_{feature_mode}"
            train_step = f"{variant}_train"
            eval_step = f"{variant}_eval"
            train_row = next(row for row in queue_rows if str(row["step"]) == train_step)
            eval_row = next(row for row in queue_rows if str(row["step"]) == eval_step)
            train_dir = Path(str(train_row["out_dir"]))
            eval_dir = Path(str(eval_row["out_dir"]))
            checkpoint = train_dir / "best.pt"

            update_row(queue_rows, train_step, status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
            return_code = run_step(
                build_train_cmd(args, out_dir=train_dir, feature_mode=feature_mode),
                Path(str(train_row["log_path"])),
                cwd=REPO_ROOT,
            )
            if return_code != 0:
                update_row(queue_rows, train_step, status="failed", finished_at=now_iso())
                for result_row in result_rows:
                    if str(result_row["variant"]) == variant:
                        result_row["train_status"] = "failed"
                write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
                write_rows(results_csv, RESULT_FIELDS, result_rows)
                raise RuntimeError(f"Training failed: {variant}")
            ensure_single_success(train_dir / "summary.csv")
            update_row(queue_rows, train_step, status="success", finished_at=now_iso())
            for result_row in result_rows:
                if str(result_row["variant"]) == variant:
                    result_row["train_status"] = "success"
            write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
            write_rows(results_csv, RESULT_FIELDS, result_rows)

            update_row(queue_rows, eval_step, status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
            return_code = run_step(
                build_eval_cmd(args, out_dir=eval_dir, checkpoint=checkpoint),
                Path(str(eval_row["log_path"])),
                cwd=REPO_ROOT,
            )
            if return_code != 0:
                update_row(queue_rows, eval_step, status="failed", finished_at=now_iso())
                for result_row in result_rows:
                    if str(result_row["variant"]) == variant:
                        result_row["eval_status"] = "failed"
                write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
                write_rows(results_csv, RESULT_FIELDS, result_rows)
                raise RuntimeError(f"Evaluation failed: {variant}")
            ensure_summary_success(eval_dir / "summary.csv")
            delta = read_delta(eval_dir / "metrics_delta.csv")
            update_row(queue_rows, eval_step, status="success", finished_at=now_iso())
            for result_row in result_rows:
                if str(result_row["variant"]) != variant:
                    continue
                result_row.update({"eval_status": "success", **delta})
                break
            write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
            write_rows(results_csv, RESULT_FIELDS, result_rows)

        append_registry(summary_csv, out_root, "success", "assignconf dual-train queue complete", str(registry_csv))
    except Exception:
        append_registry(summary_csv, out_root, "failed", "assignconf dual-train queue failed", str(registry_csv))
        raise


if __name__ == "__main__":
    main()
