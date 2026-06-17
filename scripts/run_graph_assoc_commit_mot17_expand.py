#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
BOT_ROOT = REPO_ROOT / "external" / "BoT-SORT-main"
DATA_ROOT = Path("/gemini/code/datasets")
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
PLAN_CSV = REPO_ROOT / "outputs" / "experiment_plan.csv"

BASE_SOURCE_MANIFEST = (
    REPO_ROOT
    / "outputs"
    / "graph_assoc_commit_next5h_20260419_000454"
    / "stage1"
    / "expanded_4src"
    / "source_manifest.csv"
)
MOT17_TRAIN_SEQS = [2, 4, 5, 9, 10, 11, 13]
MOT17_DETECTOR_EXTS = ["DPM", "FRCNN", "SDP"]

QUEUE_FIELDS = [
    "step",
    "name",
    "status",
    "out_dir",
    "summary_csv",
    "log_path",
    "started_at",
    "finished_at",
    "artifact_path",
    "artifact_path_2",
    "notes",
]

TRACK_GRAPH_ARGS = [
    "--graph-assoc-top-k",
    "3",
    "--graph-assoc-max-rows",
    "4",
    "--graph-assoc-max-cols",
    "4",
    "--graph-assoc-row-margin",
    "0.03",
    "--graph-assoc-col-margin",
    "0.03",
    "--graph-assoc-min-reclaim-time-since-update",
    "1",
    "--graph-assoc-max-reclaim-time-since-update",
    "8",
    "--graph-assoc-min-reclaim-tracklet-len",
    "20",
    "--graph-assoc-recent-owner-max-time-since-update",
    "1",
    "--graph-assoc-recent-owner-max-tracklet-len",
    "8",
    "--graph-assoc-young-active-max-time-since-update",
    "1",
    "--graph-assoc-young-active-max-tracklet-len",
    "20",
    "--graph-assoc-young-active-min-reclaim-gap",
    "2",
    "--graph-assoc-young-active-max-cost-delta",
    "-1.0",
    "--graph-assoc-stale-lost-owner-min-time-since-update",
    "9",
    "--graph-assoc-stale-lost-owner-min-tracklet-len",
    "100",
    "--graph-assoc-stale-lost-owner-active-max-time-since-update",
    "1",
    "--graph-assoc-stale-lost-owner-min-introduced-edge-utility",
    "0.0",
    "--graph-assoc-min-box-iou",
    "0.6",
    "--graph-assoc-reclaim-bonus",
    "0.08",
    "--graph-assoc-recent-owner-penalty",
    "0.05",
    "--graph-assoc-iou-bonus",
    "0.04",
    "--graph-assoc-score-bonus",
    "0.02",
    "--graph-assoc-min-assignment-gain",
    "0.01",
    "--graph-assoc-max-cost-delta",
    "0.05",
    "--graph-assoc-row-involved-min-assignment-gain",
    "0.01",
    "--graph-assoc-col-only-min-assignment-gain",
    "0.01",
    "--graph-assoc-col-only-max-cost-delta",
    "0.05",
    "--graph-assoc-force-match-cost",
    "0.0",
]

TRAIN_HYPERPARAMS = {
    "epochs": 60,
    "batch_size": 8,
    "hidden_dim": 128,
    "policy_hidden_dim": 128,
    "token_dim": 128,
    "num_slots": 4,
    "num_encoder_layers": 2,
    "num_heads": 4,
    "dropout": 0.1,
    "action_loss_weight": 1.0,
    "gain_loss_weight": 0.5,
    "policy_loss_weight": 0.5,
    "rank_loss_weight": 0.5,
    "rank_margin": 0.1,
    "ordinal_rank_loss_weight": 0.25,
    "ordinal_rank_margin": 0.08,
    "selection_metric": "action_macro_f1",
    "use_balanced_sampler": True,
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Expand graph-assoc commit training with MOT17 train data.")
    parser.add_argument("--run-root", default=str(REPO_ROOT / "outputs" / f"graph_assoc_commit_mot17_expand_{ts}"))
    parser.add_argument("--experiment-name", default=f"graph_assoc_commit_mot17_expand_{ts}")
    parser.add_argument("--variant-name", default="graph_assoc_commit_policy_setslot_mot17_expand")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    parser.add_argument("--plan-csv", default=str(PLAN_CSV))
    parser.add_argument("--base-manifest", default=str(BASE_SOURCE_MANIFEST))
    parser.add_argument("--mot17-commit-checkpoint", default=str(REPO_ROOT / "outputs" / "graph_assoc_commit_policy_setslot_actionmacro_20260424_2" / "best.pt"))
    parser.add_argument("--mot17-commit-device", default="cuda")
    parser.add_argument("--mot17-commit-score-margin", type=float, default=0.11)
    parser.add_argument("--mot20-eval-device", default="cuda")
    parser.add_argument("--mot20-eval-score-margin", type=float, default=0.11)
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


def update_row(rows: List[Dict[str, object]], step: str, **updates: object) -> None:
    for row in rows:
        if str(row.get("step", "")) == str(step):
            row.update(updates)
            return
    raise KeyError(f"Missing step row: {step}")


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


def upsert_plan(
    *,
    args: argparse.Namespace,
    status: str,
    run_root: Path,
    summary_csv: Path,
    log_path: Path,
    notes: str = "",
    checkpoint: str = "",
) -> None:
    cmd = [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "upsert_experiment_plan.py"),
        "--csv",
        str(args.plan_csv),
        "--key",
        f"run_root:{run_root}",
        "--status",
        status,
        "--kind",
        "train",
        "--script",
        "scripts/run_graph_assoc_commit_mot17_expand.py",
        "--dataset",
        "MOT17",
        "--split",
        "train",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        args.variant_name,
        "--tag",
        args.experiment_name,
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--checkpoint",
        checkpoint,
        "--log-path",
        str(log_path),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def append_registry(
    *,
    args: argparse.Namespace,
    status: str,
    run_root: Path,
    summary_csv: Path,
    log_path: Path,
    notes: str = "",
    checkpoint: str = "",
    kind: str = "train",
) -> None:
    cmd = [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(args.registry_csv),
        "--kind",
        kind,
        "--status",
        status,
        "--script",
        "scripts/run_graph_assoc_commit_mot17_expand.py",
        "--dataset",
        "MOT17",
        "--split",
        "train",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        args.variant_name,
        "--tag",
        args.experiment_name,
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--checkpoint",
        checkpoint,
        "--log-path",
        str(log_path),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def load_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing base manifest: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Empty base manifest: {path}")
    return rows


def write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "rows_jsonl",
        "source_tag",
        "host_variant",
        "split_tag",
        "dataset",
        "data_root",
        "split",
        "split_part",
        "seq_name",
        "dataset_tag",
        "feature_version",
    ]
    write_rows(path, fieldnames, rows)


def collect_dump_summary(analysis_dir: Path, out_csv: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for summary_path in sorted(analysis_dir.glob("*_summary.csv")):
        per_rows = read_csv_rows(summary_path)
        if not per_rows:
            continue
        row = per_rows[0]
        seq_name = str(row.get("seq_name", "")).strip() or summary_path.name.replace("_summary.csv", "")
        candidate_jsonl = str(row.get("candidate_jsonl", "")).strip() or str(
            analysis_dir / f"{seq_name}_candidates.jsonl"
        )
        event_jsonl = str(row.get("event_jsonl", "")).strip() or str(analysis_dir / f"{seq_name}_events.jsonl")
        rows.append(
            {
                "seq_name": seq_name,
                "summary_file": str(summary_path),
                "candidate_jsonl": candidate_jsonl,
                "event_jsonl": event_jsonl,
                "candidate_count": str(row.get("candidate_count", "")),
                "event_count": str(row.get("event_count", "")),
                "frames": str(row.get("frames", "")),
                "checkpoint_complete": str(row.get("checkpoint_complete", "")),
                "status": str(row.get("enabled", "")),
            }
        )
    if rows:
        write_rows(
            out_csv,
            [
                "seq_name",
                "summary_file",
                "candidate_jsonl",
                "event_jsonl",
                "candidate_count",
                "event_count",
                "frames",
                "checkpoint_complete",
                "status",
            ],
            rows,
        )
    return rows


def build_combined_manifest(base_manifest: Path, analysis_dir: Path, out_manifest: Path) -> List[Dict[str, str]]:
    base_rows = load_manifest(base_manifest)
    mot17_rows: List[Dict[str, str]] = []
    for seq_id in MOT17_TRAIN_SEQS:
        for ext in MOT17_DETECTOR_EXTS:
            seq_name = f"MOT17-{seq_id:02d}-{ext}"
            candidate_jsonl = analysis_dir / f"{seq_name}_candidates.jsonl"
            if not candidate_jsonl.is_file():
                raise FileNotFoundError(f"Missing candidate JSONL for {seq_name}: {candidate_jsonl}")
            mot17_rows.append(
                {
                    "rows_jsonl": str(candidate_jsonl.resolve()),
                    "source_tag": f"graphassoc_setslot_mot17train_{seq_name}",
                    "host_variant": "graphassoc_setslot_mot17train_canddump",
                    "split_tag": "train",
                    "dataset": "MOT17",
                    "data_root": "/gemini/code/datasets",
                    "split": "train",
                    "split_part": "full",
                    "seq_name": seq_name,
                    "dataset_tag": "graph_assoc_commit",
                    "feature_version": "graph_assoc_v1",
                }
            )
    combined = list(base_rows) + mot17_rows
    write_manifest(out_manifest, combined)
    return combined


def build_track_cmd(*, args: argparse.Namespace, run_root: Path, exp_name: str, analysis_dir: Path) -> List[str]:
    return [
        args.python_bin,
        "-u",
        "tools/track.py",
        str(DATA_ROOT / "MOT17"),
        "--benchmark",
        "MOT17",
        "--eval",
        "train",
        "--seq-ids",
        *[str(v) for v in MOT17_TRAIN_SEQS],
        "-f",
        "./yolox/exps/example/mot/yolox_x_mix_det.py",
        "-c",
        "./pretrained/bytetrack_x_mot17.pth.tar",
        "--with-reid",
        "--fast-reid-config",
        "fast_reid/configs/MOT17/sbs_S50.yml",
        "--fast-reid-weights",
        "pretrained/mot17_sbs_S50.pth",
        "--cmc-method",
        "file",
        "--fuse",
        "--experiment-name",
        exp_name,
        "--run-manifest-path",
        str(run_root / "run_manifest.json"),
        "--track_high_thresh",
        "0.6",
        "--track_low_thresh",
        "0.1",
        "--new_track_thresh",
        "0.7",
        "--track_buffer",
        "30",
        "--match_thresh",
        "0.7",
        "--proximity_thresh",
        "0.5",
        "--appearance_thresh",
        "0.25",
        "--graph-assoc-enable",
        "--graph-assoc-analysis-dir",
        str(analysis_dir),
        *TRACK_GRAPH_ARGS,
        "--graph-assoc-commit-checkpoint",
        str(Path(args.mot17_commit_checkpoint).resolve()),
        "--graph-assoc-commit-device",
        str(args.mot17_commit_device),
        "--graph-assoc-commit-score-margin",
        str(args.mot17_commit_score_margin),
        "--graph-assoc-commit-decision-mode",
        "action_margin",
        "--graph-assoc-commit-replace-rules",
        "--graph-assoc-dump-candidate-rows",
    ]


def build_dataset_cmd(*, args: argparse.Namespace, manifest_path: Path, out_dir: Path) -> List[str]:
    return [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "build_graph_assoc_commit_dataset.py"),
        "--source-manifest",
        str(manifest_path),
        "--out-dir",
        str(out_dir),
        "--dataset",
        "MOT17",
        "--split",
        "train",
        "--split-part",
        "full",
        "--topk",
        "8",
        "--min-detections",
        "2",
        "--min-positive-matches",
        "1",
        "--max-detections",
        "8",
        "--max-tracks",
        "32",
        "--dataset-tag",
        "graph_assoc_commit",
        "--feature-version",
        "graph_assoc_v1",
    ]


def build_train_cmd(*, args: argparse.Namespace, data_jsonl: Path, out_dir: Path, manifest_path: Path) -> List[str]:
    return [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "train_graph_assoc_commit_policy.py"),
        "--data-jsonl",
        str(data_jsonl),
        "--out-dir",
        str(out_dir),
        "--epochs",
        str(TRAIN_HYPERPARAMS["epochs"]),
        "--batch-size",
        str(TRAIN_HYPERPARAMS["batch_size"]),
        "--hidden-dim",
        str(TRAIN_HYPERPARAMS["hidden_dim"]),
        "--policy-hidden-dim",
        str(TRAIN_HYPERPARAMS["policy_hidden_dim"]),
        "--token-dim",
        str(TRAIN_HYPERPARAMS["token_dim"]),
        "--num-slots",
        str(TRAIN_HYPERPARAMS["num_slots"]),
        "--num-encoder-layers",
        str(TRAIN_HYPERPARAMS["num_encoder_layers"]),
        "--num-heads",
        str(TRAIN_HYPERPARAMS["num_heads"]),
        "--dropout",
        str(TRAIN_HYPERPARAMS["dropout"]),
        "--action-loss-weight",
        str(TRAIN_HYPERPARAMS["action_loss_weight"]),
        "--gain-loss-weight",
        str(TRAIN_HYPERPARAMS["gain_loss_weight"]),
        "--policy-loss-weight",
        str(TRAIN_HYPERPARAMS["policy_loss_weight"]),
        "--rank-loss-weight",
        str(TRAIN_HYPERPARAMS["rank_loss_weight"]),
        "--rank-margin",
        str(TRAIN_HYPERPARAMS["rank_margin"]),
        "--ordinal-rank-loss-weight",
        str(TRAIN_HYPERPARAMS["ordinal_rank_loss_weight"]),
        "--ordinal-rank-margin",
        str(TRAIN_HYPERPARAMS["ordinal_rank_margin"]),
        "--selection-metric",
        str(TRAIN_HYPERPARAMS["selection_metric"]),
        "--use-balanced-sampler",
        "--dataset-tag",
        "graph_assoc_commit",
        "--source-manifest",
        str(manifest_path),
        "--feature-version",
        "graph_assoc_v1",
    ]


def build_eval_cmd(*, args: argparse.Namespace, run_root: Path, checkpoint: Path) -> List[str]:
    return [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "run_botsort_graphassoc_mot20_eval.py"),
        "--run-root",
        str(run_root),
        "--experiment-name",
        run_root.name,
        "--variant-name",
        "botsort_graphassoc_mot20_setslot_mot17_expand",
        "--seq-ids",
        "2",
        "5",
        "--graph-assoc-commit-checkpoint",
        str(checkpoint),
        "--graph-assoc-commit-device",
        str(args.mot20_eval_device),
        "--graph-assoc-commit-score-margin",
        str(args.mot20_eval_score_margin),
        "--graph-assoc-commit-decision-mode",
        "action_margin",
    ]


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    logs_dir = run_root / "logs"
    dump_dir = run_root / "mot17_dump"
    dump_analysis_dir = dump_dir / "graph_assoc_analysis"
    dump_summary_csv = run_root / "mot17_dump_summary.csv"
    manifest_csv = run_root / "source_manifest.csv"
    data_dir = run_root / "graph_assoc_commit_data"
    train_dir = run_root / "train_policy"
    eval_dir = run_root / "mot20_eval"
    queue_summary_csv = run_root / "summary.csv"

    rows: List[Dict[str, object]] = [
        {
            "step": "mot17_dump",
            "name": args.experiment_name,
            "status": "running",
            "out_dir": str(dump_analysis_dir),
            "summary_csv": str(dump_summary_csv),
            "log_path": str(logs_dir / "mot17_dump.log"),
            "started_at": now_iso(),
            "finished_at": "",
            "artifact_path": str(dump_analysis_dir),
            "artifact_path_2": "",
            "notes": "running MOT17 train graph-association candidate dump",
        },
        {
            "step": "build_dataset",
            "name": args.experiment_name,
            "status": "pending",
            "out_dir": str(data_dir),
            "summary_csv": str(data_dir / "summary.csv"),
            "log_path": str(logs_dir / "build_dataset.log"),
            "started_at": "",
            "finished_at": "",
            "artifact_path": str(manifest_csv),
            "artifact_path_2": str(data_dir / "cluster_examples.jsonl"),
            "notes": "",
        },
        {
            "step": "train_policy",
            "name": args.experiment_name,
            "status": "pending",
            "out_dir": str(train_dir),
            "summary_csv": str(train_dir / "summary.csv"),
            "log_path": str(logs_dir / "train_policy.log"),
            "started_at": "",
            "finished_at": "",
            "artifact_path": str(data_dir / "cluster_examples.jsonl"),
            "artifact_path_2": str(train_dir / "best.pt"),
            "notes": "",
        },
        {
            "step": "mot20_eval",
            "name": args.experiment_name,
            "status": "pending",
            "out_dir": str(eval_dir),
            "summary_csv": str(eval_dir / "summary.csv"),
            "log_path": str(logs_dir / "mot20_eval.log"),
            "started_at": "",
            "finished_at": "",
            "artifact_path": str(eval_dir / "metrics_compare.csv"),
            "artifact_path_2": str(train_dir / "best.pt"),
            "notes": "",
        },
    ]

    write_rows(queue_summary_csv, QUEUE_FIELDS, rows)
    upsert_plan(
        args=args,
        status="running",
        run_root=run_root,
        summary_csv=queue_summary_csv,
        log_path=logs_dir / "mot17_dump.log",
        notes="MOT17 train expansion queue started",
    )
    append_registry(
        args=args,
        status="running",
        run_root=run_root,
        summary_csv=queue_summary_csv,
        log_path=logs_dir / "mot17_dump.log",
        notes="MOT17 train expansion queue started",
    )

    try:
        if not Path(args.base_manifest).is_file():
            raise FileNotFoundError(f"Missing base manifest: {args.base_manifest}")

        track_cmd = build_track_cmd(
            args=args,
            run_root=run_root,
            exp_name=args.experiment_name,
            analysis_dir=dump_analysis_dir,
        )
        rc = run_step(track_cmd, logs_dir / "mot17_dump.log", cwd=BOT_ROOT)
        dump_rows: List[Dict[str, str]] = []
        if rc == 0:
            dump_rows = collect_dump_summary(dump_analysis_dir, dump_summary_csv)
            if not dump_rows:
                raise RuntimeError("No MOT17 dump summaries were produced")
            update_row(
                rows,
                "mot17_dump",
                status="success",
                finished_at=now_iso(),
                artifact_path=str(dump_analysis_dir),
                artifact_path_2=str(dump_summary_csv),
                notes=f"candidate dump complete, {len(dump_rows)} sequence files",
            )
            write_rows(queue_summary_csv, QUEUE_FIELDS, rows)
        else:
            raise RuntimeError(f"MOT17 candidate dump failed with exit code {rc}")

        write_manifest_rows = build_combined_manifest(Path(args.base_manifest), dump_analysis_dir, manifest_csv)
        update_row(
            rows,
            "build_dataset",
            status="running",
            started_at=now_iso(),
            notes=f"building dataset from {len(write_manifest_rows)} sources",
        )
        write_rows(queue_summary_csv, QUEUE_FIELDS, rows)
        build_cmd = build_dataset_cmd(args=args, manifest_path=manifest_csv, out_dir=data_dir)
        rc = run_step(build_cmd, logs_dir / "build_dataset.log", cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"graph-assoc dataset build failed with exit code {rc}")
        update_row(
            rows,
            "build_dataset",
            status="success",
            finished_at=now_iso(),
            artifact_path=str(manifest_csv),
            artifact_path_2=str(data_dir / "cluster_examples.jsonl"),
            notes=f"combined manifest ready with {len(write_manifest_rows)} sources",
        )
        write_rows(queue_summary_csv, QUEUE_FIELDS, rows)

        update_row(
            rows,
            "train_policy",
            status="running",
            started_at=now_iso(),
            notes="training graph-assoc policy with expanded MOT17 data",
        )
        write_rows(queue_summary_csv, QUEUE_FIELDS, rows)
        train_cmd = build_train_cmd(args=args, data_jsonl=data_dir / "cluster_examples.jsonl", out_dir=train_dir, manifest_path=manifest_csv)
        rc = run_step(train_cmd, logs_dir / "train_policy.log", cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"graph-assoc policy training failed with exit code {rc}")
        train_summary = read_csv_rows(train_dir / "summary.csv")
        best_ckpt = train_dir / "best.pt"
        update_row(
            rows,
            "train_policy",
            status="success",
            finished_at=now_iso(),
            artifact_path=str(data_dir / "cluster_examples.jsonl"),
            artifact_path_2=str(best_ckpt),
            notes=f"training complete, best_epoch={train_summary[0].get('best_epoch', '') if train_summary else ''}",
        )
        write_rows(queue_summary_csv, QUEUE_FIELDS, rows)

        update_row(
            rows,
            "mot20_eval",
            status="running",
            started_at=now_iso(),
            notes="evaluating expanded checkpoint on MOT20 val_half",
        )
        write_rows(queue_summary_csv, QUEUE_FIELDS, rows)
        eval_cmd = build_eval_cmd(args=args, run_root=eval_dir, checkpoint=best_ckpt)
        rc = run_step(eval_cmd, logs_dir / "mot20_eval.log", cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"MOT20 eval failed with exit code {rc}")
        update_row(
            rows,
            "mot20_eval",
            status="success",
            finished_at=now_iso(),
            artifact_path=str(eval_dir / "metrics_compare.csv"),
            artifact_path_2=str(best_ckpt),
            notes="MOT20 validation complete",
        )
        write_rows(queue_summary_csv, QUEUE_FIELDS, rows)

        upsert_plan(
            args=args,
            status="completed",
            run_root=run_root,
            summary_csv=queue_summary_csv,
            log_path=logs_dir / "mot17_dump.log",
            notes="MOT17 train expansion queue completed",
            checkpoint=str(best_ckpt),
        )
        append_registry(
            args=args,
            status="success",
            run_root=run_root,
            summary_csv=queue_summary_csv,
            log_path=logs_dir / "mot17_dump.log",
            notes="MOT17 train expansion queue completed",
            checkpoint=str(best_ckpt),
        )
    except Exception as exc:  # noqa: BLE001
        failed_at = now_iso()
        for row in rows:
            if str(row.get("status", "")) in {"running", "pending"}:
                row["status"] = "failed"
                row["finished_at"] = failed_at
                row["notes"] = f"{row.get('notes', '')} | failed: {exc}".strip()
        write_rows(queue_summary_csv, QUEUE_FIELDS, rows)
        upsert_plan(
            args=args,
            status="failed",
            run_root=run_root,
            summary_csv=queue_summary_csv,
            log_path=logs_dir / "mot17_dump.log",
            notes=f"MOT17 train expansion queue failed: {exc}",
        )
        append_registry(
            args=args,
            status="failed",
            run_root=run_root,
            summary_csv=queue_summary_csv,
            log_path=logs_dir / "mot17_dump.log",
            notes=f"MOT17 train expansion queue failed: {exc}",
        )
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
