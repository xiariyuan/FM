#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.fcaa.fcaa.data.pairbank_dataset import PairGroupDataset, collate_pair_groups, group_pairbank_rows, load_pairbank_rows
from projects.fcaa.fcaa.metrics.offline_metrics import binary_auc, edit_flip_stats, flatten_scores, group_top1_accuracy
from projects.fcaa.fcaa.model.pair_scorer import FCAAPairScorer


SUMMARY_FIELDS = [
    "name",
    "pairbank_jsonl",
    "checkpoint",
    "groups",
    "ambiguous_groups",
    "auc",
    "ambiguous_auc",
    "top1",
    "ambiguous_top1",
    "wrong_to_right_rate",
    "right_to_wrong_rate",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an FCAA pair scorer on a pair-bank.")
    parser.add_argument("--pairbank-jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--name", default="fcaa_pair_eval")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def pad_group_rows(batches: list[torch.Tensor], *, padding_value: float = 0.0) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for batch in batches:
        for idx in range(batch.shape[0]):
            rows.append(batch[idx])
    if not rows:
        return torch.zeros((0, 0), dtype=torch.float32)
    return pad_sequence(rows, batch_first=True, padding_value=padding_value)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    summary_row = {
        "name": args.name,
        "pairbank_jsonl": args.pairbank_jsonl,
        "checkpoint": args.checkpoint,
        "groups": 0,
        "ambiguous_groups": 0,
        "auc": "",
        "ambiguous_auc": "",
        "top1": "",
        "ambiguous_top1": "",
        "wrong_to_right_rate": "",
        "right_to_wrong_rate": "",
        "status": "running",
        "error": "",
    }
    write_rows(summary_csv, SUMMARY_FIELDS, [summary_row])
    try:
        payload = torch.load(args.checkpoint, map_location="cpu")
        mode = str(payload["mode"])
        model = FCAAPairScorer(input_dim=int(payload["input_dim"]))
        model.load_state_dict(payload["model_state"])
        model.eval()
        rows = load_pairbank_rows(Path(args.pairbank_jsonl))
        groups = group_pairbank_rows(rows, mode)
        loader = DataLoader(PairGroupDataset(groups, ambiguous_oversample=1.0), batch_size=512, shuffle=False, num_workers=int(args.num_workers), collate_fn=collate_pair_groups)

        all_logits = []
        all_labels = []
        all_masks = []
        all_ambiguous = []
        all_baseline = []
        with torch.no_grad():
            for batch in loader:
                features = batch["features"]
                labels = batch["labels"]
                mask = batch["mask"]
                ambiguous = batch["ambiguous"]
                assert isinstance(features, torch.Tensor)
                assert isinstance(labels, torch.Tensor)
                assert isinstance(mask, torch.Tensor)
                assert isinstance(ambiguous, torch.Tensor)
                logits = model(features)
                all_logits.append(logits)
                all_labels.append(labels)
                all_masks.append(mask)
                all_ambiguous.append(ambiguous)
                all_baseline.append(features[..., 0])
        logits = pad_group_rows(all_logits, padding_value=0.0)
        labels = pad_group_rows(all_labels, padding_value=0.0)
        mask = pad_group_rows([batch.to(dtype=torch.float32) for batch in all_masks], padding_value=0.0).to(dtype=torch.bool)
        ambiguous = torch.cat(all_ambiguous, dim=0)
        baseline = pad_group_rows(all_baseline, padding_value=0.0)
        flat = flatten_scores(logits, labels, mask)
        ambiguous_scores = []
        ambiguous_labels = []
        for idx in range(logits.shape[0]):
            if not bool(ambiguous[idx].item()):
                continue
            for cand_idx in torch.nonzero(mask[idx], as_tuple=False).view(-1).tolist():
                ambiguous_scores.append(float(logits[idx, cand_idx].item()))
                ambiguous_labels.append(int(labels[idx, cand_idx].item()))
        flips = edit_flip_stats(logits, labels, mask, baseline)
        summary_row.update(
            {
                "groups": int(logits.shape[0]),
                "ambiguous_groups": int(ambiguous.sum().item()),
                "auc": binary_auc(flat["scores"], flat["labels"]),
                "ambiguous_auc": binary_auc(ambiguous_scores, ambiguous_labels),
                "top1": group_top1_accuracy(logits, labels, mask),
                "ambiguous_top1": group_top1_accuracy(logits, labels, mask, ambiguous_only=True, ambiguous=ambiguous),
                "wrong_to_right_rate": flips["wrong_to_right_rate"],
                "right_to_wrong_rate": flips["right_to_wrong_rate"],
                "status": "success",
            }
        )
        write_rows(summary_csv, SUMMARY_FIELDS, [summary_row])
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = str(exc)
        write_rows(summary_csv, SUMMARY_FIELDS, [summary_row])
        raise


if __name__ == "__main__":
    main()
