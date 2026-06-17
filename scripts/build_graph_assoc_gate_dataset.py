#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.graph_assoc_gate_runtime import (  # noqa: E402
    GRAPH_ASSOC_GATE_FEATURE_NAMES,
    build_graph_assoc_gate_feature_vector,
    infer_rules_passed_before_gate,
)
from models.graph_assoc_gate import GraphAssocDualHeadGate, GraphAssocGate  # noqa: E402
from scripts.build_graph_assoc_commit_dataset import (  # noqa: E402
    _build_example,
    _infer_seq_name,
    _read_row_jsonl,
)
from scripts.build_gt_pseudotrack_groups import (  # noqa: E402
    _history_file_name,
    _read_gt_rows,
    _read_seqinfo,
)


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
SUMMARY_FIELDS = [
    "train_jsonl",
    "val_jsonl",
    "sources",
    "rows_sources",
    "rows_total",
    "rules_pass_rows",
    "train_rows",
    "val_rows",
    "train_positive",
    "train_negative",
    "train_neutral",
    "train_harmful",
    "val_positive",
    "val_negative",
    "val_neutral",
    "val_harmful",
    "feature_dim",
    "positive_target",
    "neutral_target",
    "harmful_target",
    "neutral_weight",
    "harmful_weight",
    "positive_weight_scale",
    "hard_negative_checkpoint",
    "hard_negative_score_threshold",
    "hard_negative_weight_multiplier",
    "train_hard_negative_boosted",
    "val_hard_negative_boosted",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build graph-association learned gate dataset from candidate rows.")
    parser.add_argument("--source-manifest", default="")
    parser.add_argument("--rows-jsonl", nargs="*", default=[])
    parser.add_argument("--reuse-dataset-dir", default="", help="Reuse an existing gate dataset dir and only regenerate supervision fields")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", default="MOT20", choices=["MOT17", "MOT20"])
    parser.add_argument("--data-root", default="/gemini/code/datasets")
    parser.add_argument("--split", default="train")
    parser.add_argument("--split-part", default="val_half", choices=["full", "train_half", "val_half"])
    parser.add_argument("--val-patterns", default="")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--min-positive-matches", type=int, default=1)
    parser.add_argument("--positive-target", type=float, default=1.0)
    parser.add_argument("--neutral-target", type=float, default=0.25)
    parser.add_argument("--harmful-target", type=float, default=0.0)
    parser.add_argument("--neutral-weight", type=float, default=0.5)
    parser.add_argument("--harmful-weight", type=float, default=1.25)
    parser.add_argument("--positive-weight-scale", type=float, default=0.25)
    parser.add_argument("--hard-negative-checkpoint", default="")
    parser.add_argument("--hard-negative-score-threshold", type=float, default=-1.0)
    parser.add_argument("--hard-negative-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_single_row_csv(path: Path, fieldnames: List[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_registry(args: argparse.Namespace, summary_csv: Path, status: str, notes: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(args.registry_csv),
        "--kind",
        "analysis",
        "--status",
        status,
        "--script",
        "scripts/build_graph_assoc_gate_dataset.py",
        "--dataset",
        str(args.dataset),
        "--split",
        f"graph_assoc_gate_{args.split_part}",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        Path(args.out_dir).name,
        "--tag",
        Path(args.out_dir).name,
        "--run-root",
        str(Path(args.out_dir)),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def _parse_patterns(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [token.strip() for token in text.split(",") if token.strip()]


def _match_patterns(text: str, patterns: List[str]) -> bool:
    haystack = str(text or "")
    return any(pattern in haystack for pattern in patterns)


def _load_sources(args: argparse.Namespace) -> List[Dict[str, str]]:
    val_patterns = _parse_patterns(args.val_patterns)
    if str(args.source_manifest or "").strip():
        manifest_path = Path(args.source_manifest).resolve()
        rows: List[Dict[str, str]] = []
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                rows_jsonl = Path(str(row.get("rows_jsonl", "")).strip())
                if not rows_jsonl.is_absolute():
                    rows_jsonl = (manifest_path.parent / rows_jsonl).resolve()
                split_tag = str(row.get("split_tag", "")).strip()
                if not split_tag:
                    split_tag = "val" if _match_patterns(str(rows_jsonl), val_patterns) else "train"
                rows.append(
                    {
                        "rows_jsonl": str(rows_jsonl),
                        "source_tag": str(row.get("source_tag", "")).strip() or f"source_{idx:03d}",
                        "host_variant": str(row.get("host_variant", "")).strip() or rows_jsonl.parent.name,
                        "split_tag": str(split_tag),
                        "dataset": str(row.get("dataset", "")).strip() or str(args.dataset),
                        "data_root": str(row.get("data_root", "")).strip() or str(args.data_root),
                        "split": str(row.get("split", "")).strip() or str(args.split),
                        "split_part": str(row.get("split_part", "")).strip() or str(args.split_part),
                        "seq_name": str(row.get("seq_name", "")).strip(),
                        "feature_version": str(row.get("feature_version", "")).strip() or "graph_assoc_gate_v1",
                        "dataset_tag": str(row.get("dataset_tag", "")).strip() or "graph_assoc_gate",
                    }
                )
        return rows

    sources: List[Dict[str, str]] = []
    for idx, raw_path in enumerate(list(args.rows_jsonl or [])):
        rows_jsonl = Path(str(raw_path)).resolve()
        split_tag = "val" if _match_patterns(str(rows_jsonl), val_patterns) else "train"
        sources.append(
            {
                "rows_jsonl": str(rows_jsonl),
                "source_tag": f"source_{idx:03d}_{rows_jsonl.stem}",
                "host_variant": rows_jsonl.parent.name,
                "split_tag": split_tag,
                "dataset": str(args.dataset),
                "data_root": str(args.data_root),
                "split": str(args.split),
                "split_part": str(args.split_part),
                "seq_name": "",
                "feature_version": "graph_assoc_gate_v1",
                "dataset_tag": "graph_assoc_gate",
            }
        )
    return sources


def _rewrite_existing_dataset(
    args: argparse.Namespace,
    *,
    reuse_dir: Path,
    train_jsonl: Path,
    val_jsonl: Path,
    summary_row: Dict[str, object],
) -> Dict[str, Counter[str]]:
    split_counters: Dict[str, Counter[str]] = {"train": Counter(), "val": Counter()}
    summary_row["sources"] = 1
    hard_negative_threshold = float(args.hard_negative_score_threshold)
    hard_negative_multiplier = float(args.hard_negative_weight_multiplier)
    hard_negative_model = None
    if str(args.hard_negative_checkpoint or "").strip() and hard_negative_threshold >= 0.0 and hard_negative_multiplier > 1.0:
        payload = torch.load(Path(str(args.hard_negative_checkpoint)).resolve(), map_location="cpu")
        payload_model_type = str(payload.get("model_type", "")).strip().lower()
        state_dict = dict(payload.get("model_state", payload))
        input_dim = int(payload.get("input_dim", len(payload.get("feature_names", GRAPH_ASSOC_GATE_FEATURE_NAMES))))
        hidden_dim = int(payload.get("hidden_dim", 32))
        dropout = float(payload.get("dropout", 0.0))
        num_hidden_layers = int(payload.get("num_hidden_layers", 1))
        if payload_model_type == "dual_head" or any(key.startswith("neutral_head") for key in state_dict):
            hard_negative_model = GraphAssocDualHeadGate(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                num_hidden_layers=num_hidden_layers,
            )
        else:
            hard_negative_model = GraphAssocGate(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                num_hidden_layers=num_hidden_layers,
            )
        hard_negative_model.load_state_dict(state_dict)
        hard_negative_model.eval()

    with train_jsonl.open("w", encoding="utf-8") as train_fp, val_jsonl.open("w", encoding="utf-8") as val_fp:
        for split_tag in ("train", "val"):
            input_jsonl = reuse_dir / f"{split_tag}.jsonl"
            if not input_jsonl.is_file():
                raise FileNotFoundError(f"Missing reused dataset file: {input_jsonl}")
            with input_jsonl.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if "gt_gain" not in row:
                        raise ValueError(f"Row missing gt_gain in reused dataset: {input_jsonl}")
                    gt_gain = int(row.get("gt_gain", 0))
                    train_target, sample_weight, label, gain_class, gain_class_name = _supervision_from_gain(
                        gt_gain=gt_gain,
                        positive_target=float(args.positive_target),
                        neutral_target=float(args.neutral_target),
                        harmful_target=float(args.harmful_target),
                        neutral_weight=float(args.neutral_weight),
                        harmful_weight=float(args.harmful_weight),
                        positive_weight_scale=float(args.positive_weight_scale),
                    )
                    row["label"] = int(label)
                    row["gain_class"] = int(gain_class)
                    row["gain_class_name"] = str(gain_class_name)
                    row["train_target"] = float(train_target)
                    row["sample_weight"] = float(sample_weight)
                    row["feature_names"] = list(row.get("feature_names") or GRAPH_ASSOC_GATE_FEATURE_NAMES)
                    if hard_negative_model is not None and split_tag == "train" and gt_gain <= 0:
                        feature_tensor = torch.tensor(row["features"], dtype=torch.float32).view(1, -1)
                        with torch.inference_mode():
                            hard_negative_out = hard_negative_model(feature_tensor)
                            if isinstance(hard_negative_out, tuple):
                                hard_negative_prob = float(torch.sigmoid(hard_negative_out[0]).item())
                            else:
                                hard_negative_prob = float(torch.sigmoid(hard_negative_out).item())
                        if hard_negative_prob >= hard_negative_threshold:
                            row["sample_weight"] = float(row["sample_weight"]) * hard_negative_multiplier
                            row["hard_negative_boosted"] = 1
                            row["hard_negative_score"] = float(hard_negative_prob)
                            split_counters[split_tag]["hard_negative_boosted"] += 1
                        else:
                            row["hard_negative_boosted"] = 0
                            row["hard_negative_score"] = float(hard_negative_prob)
                    else:
                        row["hard_negative_boosted"] = 0

                    summary_row["rows_sources"] = int(summary_row.get("rows_sources", 0)) + 1
                    summary_row["rows_total"] = int(summary_row.get("rows_total", 0)) + 1
                    summary_row["rules_pass_rows"] = int(summary_row.get("rules_pass_rows", 0)) + int(
                        row.get("rules_passed_before_gate", 1)
                    )

                    split_counters[split_tag]["rows"] += 1
                    split_counters[split_tag]["positive"] += int(gt_gain > 0)
                    split_counters[split_tag]["negative"] += int(gt_gain <= 0)
                    split_counters[split_tag]["neutral"] += int(gt_gain == 0)
                    split_counters[split_tag]["harmful"] += int(gt_gain < 0)

                    out_fp = val_fp if split_tag == "val" else train_fp
                    out_fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    return split_counters


def _supervision_from_gain(
    gt_gain: int,
    *,
    positive_target: float,
    neutral_target: float,
    harmful_target: float,
    neutral_weight: float,
    harmful_weight: float,
    positive_weight_scale: float,
) -> tuple[float, float, int, int, str]:
    gt_gain = int(gt_gain)
    if gt_gain > 0:
        return (
            float(positive_target),
            1.0 + float(positive_weight_scale) * float(gt_gain),
            1,
            2,
            "positive",
        )
    if gt_gain < 0:
        return (
            float(harmful_target),
            float(harmful_weight) + float(abs(gt_gain) - 1) * float(positive_weight_scale),
            0,
            0,
            "harmful",
        )
    return float(neutral_target), float(neutral_weight), 0, 1, "neutral"


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = out_dir / "train.jsonl"
    val_jsonl = out_dir / "val.jsonl"
    summary_csv = out_dir / "summary.csv"
    summary_row: Dict[str, object] = {
        "train_jsonl": str(train_jsonl),
        "val_jsonl": str(val_jsonl),
        "sources": 0,
        "rows_sources": 0,
        "rows_total": 0,
        "rules_pass_rows": 0,
        "train_rows": 0,
        "val_rows": 0,
        "train_positive": 0,
        "train_negative": 0,
        "train_neutral": 0,
        "train_harmful": 0,
        "val_positive": 0,
        "val_negative": 0,
        "val_neutral": 0,
        "val_harmful": 0,
        "feature_dim": int(len(GRAPH_ASSOC_GATE_FEATURE_NAMES)),
        "positive_target": float(args.positive_target),
        "neutral_target": float(args.neutral_target),
        "harmful_target": float(args.harmful_target),
        "neutral_weight": float(args.neutral_weight),
        "harmful_weight": float(args.harmful_weight),
        "positive_weight_scale": float(args.positive_weight_scale),
        "hard_negative_checkpoint": str(args.hard_negative_checkpoint),
        "hard_negative_score_threshold": float(args.hard_negative_score_threshold),
        "hard_negative_weight_multiplier": float(args.hard_negative_weight_multiplier),
        "train_hard_negative_boosted": 0,
        "val_hard_negative_boosted": 0,
        "status": "running",
        "error": "",
    }
    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
    append_registry(args, summary_csv, "running", "building graph-association gate dataset")

    try:
        reuse_dataset_dir = Path(str(args.reuse_dataset_dir or "")).resolve() if str(args.reuse_dataset_dir or "").strip() else None
        if reuse_dataset_dir is not None:
            split_counters = _rewrite_existing_dataset(
                args,
                reuse_dir=reuse_dataset_dir,
                train_jsonl=train_jsonl,
                val_jsonl=val_jsonl,
                summary_row=summary_row,
            )
            summary_row.update(
                {
                    "train_rows": int(split_counters["train"]["rows"]),
                    "val_rows": int(split_counters["val"]["rows"]),
                    "train_positive": int(split_counters["train"]["positive"]),
                    "train_negative": int(split_counters["train"]["negative"]),
                    "train_neutral": int(split_counters["train"]["neutral"]),
                    "train_harmful": int(split_counters["train"]["harmful"]),
                    "train_hard_negative_boosted": int(split_counters["train"]["hard_negative_boosted"]),
                    "val_positive": int(split_counters["val"]["positive"]),
                    "val_negative": int(split_counters["val"]["negative"]),
                    "val_neutral": int(split_counters["val"]["neutral"]),
                    "val_harmful": int(split_counters["val"]["harmful"]),
                    "val_hard_negative_boosted": int(split_counters["val"]["hard_negative_boosted"]),
                    "status": "success",
                }
            )
            if int(summary_row["train_rows"]) <= 0 or int(summary_row["val_rows"]) <= 0:
                raise ValueError(f"Invalid split sizes: train={summary_row['train_rows']} val={summary_row['val_rows']}")

            write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
            append_registry(
                args,
                summary_csv,
                "success",
                f"graph-association gate dataset reused from {reuse_dataset_dir.name}",
            )
            return 0

        sources = _load_sources(args)
        if not sources:
            raise ValueError("No gate dataset sources provided.")
        summary_row["sources"] = int(len(sources))
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)

        gt_cache: Dict[tuple[str, str, str, str, str], tuple[dict[int, list[dict[str, float]]], float, float, int]] = {}
        split_counters: Dict[str, Counter[str]] = {"train": Counter(), "val": Counter()}

        with train_jsonl.open("w", encoding="utf-8") as train_fp, val_jsonl.open("w", encoding="utf-8") as val_fp:
            for source in sources:
                rows_path = Path(source["rows_jsonl"]).resolve()
                if not rows_path.is_file():
                    raise FileNotFoundError(f"Missing candidate rows jsonl: {rows_path}")
                source_rows = _read_row_jsonl(rows_path)
                summary_row["rows_sources"] = int(summary_row.get("rows_sources", 0)) + int(len(source_rows))

                for row in source_rows:
                    seq = _infer_seq_name(source, row)
                    cache_key = (
                        str(source["dataset"]),
                        str(source["data_root"]),
                        str(source["split"]),
                        str(source["split_part"]),
                        str(seq),
                    )
                    if cache_key not in gt_cache:
                        seq_dir = Path(str(source["data_root"])) / str(source["dataset"]) / str(source["split"]) / str(seq)
                        gt_path = _history_file_name(seq_dir, "gt", str(source["split_part"]))
                        gt_rows = _read_gt_rows(gt_path)
                        seqinfo = _read_seqinfo(seq_dir)
                        seq_length = int(seqinfo.get("seqLength", seqinfo.get("seqlength")))
                        seq_w = float(seqinfo.get("imWidth", seqinfo.get("imwidth")))
                        seq_h = float(seqinfo.get("imHeight", seqinfo.get("imheight")))
                        gt_cache[cache_key] = (gt_rows, seq_w, seq_h, seq_length)

                    gt_rows, seq_w, seq_h, seq_length = gt_cache[cache_key]
                    example = _build_example(
                        row=row,
                        seq=seq,
                        source=source,
                        gt_rows=gt_rows,
                        seq_w=seq_w,
                        seq_h=seq_h,
                        seq_length=seq_length,
                        split_part=str(source["split_part"]),
                        topk=int(args.topk),
                        min_positive_matches=int(args.min_positive_matches),
                    )
                    if example is None:
                        continue

                    summary_row["rows_total"] = int(summary_row.get("rows_total", 0)) + 1
                    rules_passed = bool(infer_rules_passed_before_gate(row))
                    if not rules_passed:
                        continue
                    summary_row["rules_pass_rows"] = int(summary_row.get("rules_pass_rows", 0)) + 1

                    split_tag = str(source.get("split_tag", "train") or "train").strip().lower()
                    if split_tag not in {"train", "val"}:
                        split_tag = "train"

                    gt_gain = int(example.get("gt_gain", 0))
                    train_target, sample_weight, label, gain_class, gain_class_name = _supervision_from_gain(
                        gt_gain=gt_gain,
                        positive_target=float(args.positive_target),
                        neutral_target=float(args.neutral_target),
                        harmful_target=float(args.harmful_target),
                        neutral_weight=float(args.neutral_weight),
                        harmful_weight=float(args.harmful_weight),
                        positive_weight_scale=float(args.positive_weight_scale),
                    )
                    record = {
                        "cluster_id": str(example["cluster_id"]),
                        "seq": str(example["seq"]),
                        "frame": int(example["frame"]),
                        "source_tag": str(example["source_tag"]),
                        "host_variant": str(example["host_variant"]),
                        "split_tag": str(split_tag),
                        "decision": str(example.get("decision", "")),
                        "decision_source": str(example.get("decision_source", "")),
                        "skip_reason": str(example.get("skip_reason", "")),
                        "accepted": int(example.get("accepted", 0)),
                        "rules_passed_before_gate": int(rules_passed),
                        "gt_gain": int(gt_gain),
                        "gt_decision": str(example.get("gt_decision", "")),
                        "baseline_true_matches": int(example.get("baseline_true_matches", 0)),
                        "chosen_true_matches": int(example.get("chosen_true_matches", 0)),
                        "label": int(label),
                        "gain_class": int(gain_class),
                        "gain_class_name": str(gain_class_name),
                        "train_target": float(train_target),
                        "sample_weight": float(sample_weight),
                        "feature_names": list(GRAPH_ASSOC_GATE_FEATURE_NAMES),
                        "features": build_graph_assoc_gate_feature_vector(row),
                    }

                    split_counters[split_tag]["rows"] += 1
                    split_counters[split_tag]["positive"] += int(gt_gain > 0)
                    split_counters[split_tag]["negative"] += int(gt_gain <= 0)
                    split_counters[split_tag]["neutral"] += int(gt_gain == 0)
                    split_counters[split_tag]["harmful"] += int(gt_gain < 0)

                    fp = val_fp if split_tag == "val" else train_fp
                    fp.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary_row.update(
            {
                "train_rows": int(split_counters["train"]["rows"]),
                "val_rows": int(split_counters["val"]["rows"]),
                "train_positive": int(split_counters["train"]["positive"]),
                "train_negative": int(split_counters["train"]["negative"]),
                "train_neutral": int(split_counters["train"]["neutral"]),
                "train_harmful": int(split_counters["train"]["harmful"]),
                "train_hard_negative_boosted": int(split_counters["train"]["hard_negative_boosted"]),
                "val_positive": int(split_counters["val"]["positive"]),
                "val_negative": int(split_counters["val"]["negative"]),
                "val_neutral": int(split_counters["val"]["neutral"]),
                "val_harmful": int(split_counters["val"]["harmful"]),
                "val_hard_negative_boosted": int(split_counters["val"]["hard_negative_boosted"]),
                "status": "success",
            }
        )
        if int(summary_row["train_rows"]) <= 0 or int(summary_row["val_rows"]) <= 0:
            raise ValueError(f"Invalid split sizes: train={summary_row['train_rows']} val={summary_row['val_rows']}")

        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, "success", "graph-association gate dataset ready")
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = repr(exc)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, "failed", "graph-association gate dataset failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
