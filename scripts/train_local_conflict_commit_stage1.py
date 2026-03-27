#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.local_conflict_commit import LocalConflictCommitRefiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LocalConflictCommitRefiner.")
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-sequences", type=str, default="")
    parser.add_argument("--val-sequences", type=str, default="")
    parser.add_argument("--strict-sequence-split", action="store_true")
    parser.add_argument("--min-val-examples", type=int, default=1)
    parser.add_argument("--dataset-tag", type=str, default="")
    parser.add_argument("--source-manifest", type=str, default="")
    parser.add_argument("--feature-version", type=str, default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _write_single_row_csv(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_csv_tokens(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [token.strip() for token in text.split(",") if token.strip()]


def _sequence_aliases(seq: str) -> set[str]:
    raw = str(seq or "").strip()
    if not raw:
        return set()
    name = Path(raw).name
    aliases = {raw, name}
    if name.count("-") >= 2:
        aliases.add(name.rsplit("-", 1)[0])
    return {token for token in aliases if token}


def _sequence_matches(seq: str, tokens: list[str]) -> bool:
    if not tokens:
        return False
    aliases = _sequence_aliases(seq)
    for token in tokens:
        token = str(token or "").strip()
        if not token:
            continue
        for alias in aliases:
            if alias == token or alias.startswith(token):
                return True
    return False


class ClusterCommitDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples[index]
        return {
            "cluster_id": str(row["cluster_id"]),
            "seq": str(row["seq"]),
            "frame": int(row["frame"]),
            "source_tag": str(row.get("source_tag", "")),
            "host_variant": str(row.get("host_variant", "")),
            "split_tag": str(row.get("split_tag", "")),
            "feature_version": str(row.get("feature_version", "")),
            "det_features": torch.tensor(row["det_features"], dtype=torch.float32),
            "track_features": torch.tensor(row["track_features"], dtype=torch.float32),
            "edge_features": torch.tensor(row["edge_features"], dtype=torch.float32),
            "edge_det_index": torch.tensor(row["edge_det_index"], dtype=torch.long),
            "edge_track_index": torch.tensor(row["edge_track_index"], dtype=torch.long),
            "cluster_features": torch.tensor(row["cluster_features"], dtype=torch.float32),
            "target_by_det": torch.tensor(row["target_by_det"], dtype=torch.long),
            "target_committed_matches": int(row.get("target_committed_matches", 0)),
            "trigger_pass": int(row.get("trigger_pass", 0)),
        }


def _collate_cluster_samples(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


def _load_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def _fallback_split(examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_at = max(int(len(examples) * 0.8), 1)
    train_rows = examples[:split_at]
    val_rows = examples[split_at:] or examples[-1:]
    return train_rows, val_rows


def _split_examples(
    *,
    examples: list[dict[str, Any]],
    train_sequences_arg: str,
    val_sequences_arg: str,
    strict_sequence_split: bool,
    min_val_examples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str], str]:
    train_tokens = _parse_csv_tokens(train_sequences_arg)
    val_tokens = _parse_csv_tokens(val_sequences_arg)
    all_sequences = sorted({str(row["seq"]) for row in examples})
    seq_counts = {seq: sum(1 for row in examples if str(row["seq"]) == seq) for seq in all_sequences}

    tagged_train = [row for row in examples if str(row.get("split_tag", "")) == "train"]
    tagged_val = [row for row in examples if str(row.get("split_tag", "")) == "val"]

    split_mode = ""
    if tagged_train or tagged_val:
        train_rows = list(tagged_train)
        val_rows = list(tagged_val)
        split_mode = "manifest_split_tag"
    elif train_tokens or val_tokens:
        train_rows = []
        val_rows = []
        for row in examples:
            seq = str(row["seq"])
            if _sequence_matches(seq, val_tokens):
                val_rows.append(row)
            elif train_tokens:
                if _sequence_matches(seq, train_tokens):
                    train_rows.append(row)
            else:
                train_rows.append(row)
        split_mode = "requested_sequences"
    elif len(all_sequences) > 1:
        val_sequence = min(all_sequences, key=lambda seq: (seq_counts.get(seq, 0), seq))
        train_rows = [row for row in examples if str(row["seq"]) != val_sequence]
        val_rows = [row for row in examples if str(row["seq"]) == val_sequence]
        split_mode = "auto_smallest_sequence"
    else:
        train_rows, val_rows = _fallback_split(examples)
        split_mode = "fallback_80_20"

    if not train_rows or not val_rows or len(val_rows) < int(min_val_examples):
        if strict_sequence_split:
            raise ValueError(
                "Strict sequence split failed: "
                f"train_rows={len(train_rows)} val_rows={len(val_rows)} min_val_examples={int(min_val_examples)}"
            )
        train_rows, val_rows = _fallback_split(examples)
        split_mode = "fallback_80_20"

    train_sequences = sorted({str(row["seq"]) for row in train_rows})
    val_sequences = sorted({str(row["seq"]) for row in val_rows})
    return train_rows, val_rows, train_sequences, val_sequences, split_mode


def _join_unique(rows: list[dict[str, Any]], key: str) -> str:
    values = sorted({str(row.get(key, "")).strip() for row in rows if str(row.get(key, "")).strip()})
    return ",".join(values)


def _run_epoch(
    *,
    model: LocalConflictCommitRefiner,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    totals = {
        "loss": 0.0,
        "clusters": 0.0,
        "detections": 0.0,
        "row_correct": 0.0,
        "commit_correct": 0.0,
        "pred_commits": 0.0,
        "target_commits": 0.0,
        "tp_commits": 0.0,
    }

    for batch in loader:
        losses = []
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        for sample in batch:
            det_features = sample["det_features"].to(device)
            track_features = sample["track_features"].to(device)
            edge_features = sample["edge_features"].to(device)
            edge_det_index = sample["edge_det_index"].to(device)
            edge_track_index = sample["edge_track_index"].to(device)
            cluster_features = sample["cluster_features"].to(device)
            target_by_det = sample["target_by_det"].to(device)

            outputs = model(
                det_features=det_features,
                track_features=track_features,
                edge_features=edge_features,
                edge_det_index=edge_det_index,
                edge_track_index=edge_track_index,
                cluster_features=cluster_features,
            )
            dense_logits = LocalConflictCommitRefiner.build_dense_assignment_logits(
                num_detections=det_features.shape[0],
                num_tracks=track_features.shape[0],
                edge_logits=outputs["edge_logits"],
                edge_det_index=edge_det_index,
                edge_track_index=edge_track_index,
                defer_logits=outputs["defer_logits"],
            )
            target = target_by_det.clone()
            target[target < 0] = int(track_features.shape[0])
            loss = F.cross_entropy(dense_logits, target)
            losses.append(loss)

            pred = dense_logits.argmax(dim=-1)
            totals["clusters"] += 1.0
            totals["detections"] += float(target.numel())
            totals["loss"] += float(loss.detach().item()) * float(target.numel())
            totals["row_correct"] += float((pred == target).sum().item())
            pred_commit = pred < int(track_features.shape[0])
            target_commit = target < int(track_features.shape[0])
            totals["commit_correct"] += float((pred_commit == target_commit).sum().item())
            totals["pred_commits"] += float(pred_commit.sum().item())
            totals["target_commits"] += float(target_commit.sum().item())
            totals["tp_commits"] += float((pred_commit & target_commit).sum().item())

        if losses and optimizer is not None:
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            optimizer.step()

    denom = max(totals["detections"], 1.0)
    pred_commits = max(totals["pred_commits"], 1.0)
    target_commits = max(totals["target_commits"], 1.0)
    return {
        "loss": totals["loss"] / denom,
        "row_acc": totals["row_correct"] / denom,
        "commit_acc": totals["commit_correct"] / denom,
        "commit_precision": totals["tp_commits"] / pred_commits,
        "commit_recall": totals["tp_commits"] / target_commits,
        "clusters": totals["clusters"],
        "detections": totals["detections"],
    }


def main() -> int:
    args = parse_args()
    _set_seed(int(args.seed))

    data_jsonl = Path(args.data_jsonl).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result_csv = out_dir / "result.csv"
    summary_csv = out_dir / "summary.csv"
    metrics_jsonl = out_dir / "metrics.jsonl"
    best_ckpt = out_dir / "best.pt"

    running_row = {
        "exp_name": "local_conflict_commit_stage1",
        "data_jsonl": str(data_jsonl),
        "source_manifest": str(args.source_manifest or ""),
        "dataset_tag": str(args.dataset_tag or ""),
        "feature_version": str(args.feature_version or ""),
        "checkpoint": "",
        "train_examples": "",
        "val_examples": "",
        "train_sequences": "",
        "val_sequences": "",
        "train_host_variants": "",
        "val_host_variants": "",
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "strict_sequence_split": int(bool(args.strict_sequence_split)),
        "min_val_examples": int(args.min_val_examples),
        "split_mode": "",
        "best_epoch": "",
        "train_loss": "",
        "val_loss": "",
        "train_row_acc": "",
        "val_row_acc": "",
        "val_commit_precision": "",
        "val_commit_recall": "",
        "status": "running",
        "error": "",
    }
    fieldnames = list(running_row.keys())
    _write_single_row_csv(result_csv, fieldnames, running_row)
    _write_single_row_csv(summary_csv, fieldnames, running_row)
    metrics_jsonl.write_text("", encoding="utf-8")

    try:
        examples = _load_examples(data_jsonl)
        train_rows, val_rows, train_sequences, val_sequences, split_mode = _split_examples(
            examples=examples,
            train_sequences_arg=str(args.train_sequences or ""),
            val_sequences_arg=str(args.val_sequences or ""),
            strict_sequence_split=bool(args.strict_sequence_split),
            min_val_examples=int(args.min_val_examples),
        )

        train_loader = DataLoader(
            ClusterCommitDataset(train_rows),
            batch_size=max(int(args.batch_size), 1),
            shuffle=True,
            collate_fn=_collate_cluster_samples,
        )
        val_loader = DataLoader(
            ClusterCommitDataset(val_rows),
            batch_size=max(int(args.batch_size), 1),
            shuffle=False,
            collate_fn=_collate_cluster_samples,
        )

        device = torch.device(args.device)
        model = LocalConflictCommitRefiner(hidden_dim=int(args.hidden_dim), dropout=float(args.dropout)).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )

        best_val_loss = float("inf")
        best_row: dict[str, Any] | None = None
        with metrics_jsonl.open("a", encoding="utf-8") as metrics_fp:
            for epoch in range(1, int(args.epochs) + 1):
                train_metrics = _run_epoch(model=model, loader=train_loader, optimizer=optimizer, device=device)
                with torch.inference_mode():
                    val_metrics = _run_epoch(model=model, loader=val_loader, optimizer=None, device=device)

                metrics_row = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_row_acc": train_metrics["row_acc"],
                    "train_commit_acc": train_metrics["commit_acc"],
                    "val_loss": val_metrics["loss"],
                    "val_row_acc": val_metrics["row_acc"],
                    "val_commit_acc": val_metrics["commit_acc"],
                    "val_commit_precision": val_metrics["commit_precision"],
                    "val_commit_recall": val_metrics["commit_recall"],
                    "split_mode": split_mode,
                    "train_examples": len(train_rows),
                    "val_examples": len(val_rows),
                }
                metrics_fp.write(json.dumps(metrics_row) + "\n")
                metrics_fp.flush()

                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = float(val_metrics["loss"])
                    best_row = {
                        "exp_name": "local_conflict_commit_stage1",
                        "data_jsonl": str(data_jsonl),
                        "source_manifest": str(args.source_manifest or ""),
                        "dataset_tag": str(args.dataset_tag or ""),
                        "feature_version": str(args.feature_version or ""),
                        "checkpoint": str(best_ckpt),
                        "train_examples": int(len(train_rows)),
                        "val_examples": int(len(val_rows)),
                        "train_sequences": ",".join(train_sequences),
                        "val_sequences": ",".join(val_sequences),
                        "train_host_variants": _join_unique(train_rows, "host_variant"),
                        "val_host_variants": _join_unique(val_rows, "host_variant"),
                        "epochs": int(args.epochs),
                        "batch_size": int(args.batch_size),
                        "hidden_dim": int(args.hidden_dim),
                        "dropout": float(args.dropout),
                        "strict_sequence_split": int(bool(args.strict_sequence_split)),
                        "min_val_examples": int(args.min_val_examples),
                        "split_mode": split_mode,
                        "best_epoch": int(epoch),
                        "train_loss": float(train_metrics["loss"]),
                        "val_loss": float(val_metrics["loss"]),
                        "train_row_acc": float(train_metrics["row_acc"]),
                        "val_row_acc": float(val_metrics["row_acc"]),
                        "val_commit_precision": float(val_metrics["commit_precision"]),
                        "val_commit_recall": float(val_metrics["commit_recall"]),
                        "status": "ok",
                        "error": "",
                    }
                    torch.save(
                        model.checkpoint_payload(
                            epoch=int(epoch),
                            train_metrics=train_metrics,
                            val_metrics=val_metrics,
                            data_jsonl=str(data_jsonl),
                            source_manifest=str(args.source_manifest or ""),
                            dataset_tag=str(args.dataset_tag or ""),
                            feature_version=str(args.feature_version or ""),
                            train_examples=int(len(train_rows)),
                            val_examples=int(len(val_rows)),
                            train_sequences=list(train_sequences),
                            val_sequences=list(val_sequences),
                            split_mode=split_mode,
                            train_host_variants=_join_unique(train_rows, "host_variant"),
                            val_host_variants=_join_unique(val_rows, "host_variant"),
                        ),
                        best_ckpt,
                    )
                    _write_single_row_csv(summary_csv, fieldnames, best_row)

        if best_row is None:
            best_row = dict(running_row)
            best_row.update(
                {
                    "train_examples": int(len(train_rows)),
                    "val_examples": int(len(val_rows)),
                    "train_sequences": ",".join(train_sequences),
                    "val_sequences": ",".join(val_sequences),
                    "train_host_variants": _join_unique(train_rows, "host_variant"),
                    "val_host_variants": _join_unique(val_rows, "host_variant"),
                    "split_mode": split_mode,
                    "status": "failed",
                    "error": "no_best_epoch",
                }
            )

        _write_single_row_csv(result_csv, fieldnames, best_row)
        _write_single_row_csv(summary_csv, fieldnames, best_row)
        return 0 if str(best_row.get("status", "")) == "ok" else 1
    except Exception as exc:
        failed_row = dict(running_row)
        failed_row.update(
            {
                "status": "failed",
                "error": str(exc),
            }
        )
        _write_single_row_csv(result_csv, fieldnames, failed_row)
        _write_single_row_csv(summary_csv, fieldnames, failed_row)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
