#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.fcaa.fcaa.metrics.offline_metrics import binary_auc, edit_flip_stats, flatten_scores, group_top1_accuracy
from projects.fgas.fgas.data.blockbank_io import load_blockbank_jsonl
from projects.fgas.fgas.data.pairgroup_dataset import PairGroupDataset, blockbank_to_pair_groups, collate_pair_groups
from projects.fgas.fgas.model.pair_scorer import FGASPairScorer


SUMMARY_FIELDS = [
    "name",
    "blockbank_jsonl",
    "checkpoint",
    "mode",
    "blocks",
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
    parser = argparse.ArgumentParser(description="Evaluate the detector-aware FGAS pair scorer on a blockbank.")
    parser.add_argument("--blockbank-jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--name", default="fgas_pair_eval")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def pad_group_rows(batches: List[torch.Tensor], *, padding_value: float = 0.0) -> torch.Tensor:
    rows: List[torch.Tensor] = []
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
    summary_row: Dict[str, object] = {
        "name": args.name,
        "blockbank_jsonl": args.blockbank_jsonl,
        "checkpoint": args.checkpoint,
        "mode": "",
        "blocks": 0,
        "groups": 0,
        "ambiguous_groups": 0,
        "auc": 0.0,
        "ambiguous_auc": 0.0,
        "top1": 0.0,
        "ambiguous_top1": 0.0,
        "wrong_to_right_rate": 0.0,
        "right_to_wrong_rate": 0.0,
        "status": "running",
        "error": "",
    }
    write_rows(summary_csv, SUMMARY_FIELDS, [summary_row])
    try:
        payload = torch.load(args.checkpoint, map_location="cpu")
        include_frequency = str(payload.get("mode", "nofreq")) == "freq"
        model = FGASPairScorer(
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload.get("hidden_dim", 64)),
            dropout=float(payload.get("dropout", 0.0)),
        )
        model.load_state_dict(payload["model_state"])
        model.eval()
        blocks = load_blockbank_jsonl(Path(args.blockbank_jsonl))
        groups = blockbank_to_pair_groups(blocks, include_frequency=include_frequency)
        if not groups:
            raise ValueError("No groups built from blockbank.")
        loader = DataLoader(
            PairGroupDataset(groups, ambiguous_oversample=1.0),
            batch_size=512,
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=collate_pair_groups,
        )
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
                baseline = batch["baseline"]
                assert isinstance(features, torch.Tensor)
                assert isinstance(labels, torch.Tensor)
                assert isinstance(mask, torch.Tensor)
                assert isinstance(ambiguous, torch.Tensor)
                assert isinstance(baseline, torch.Tensor)
                logits = model(features)
                all_logits.append(logits)
                all_labels.append(labels)
                all_masks.append(mask)
                all_ambiguous.append(ambiguous)
                all_baseline.append(baseline)
        logits = pad_group_rows(all_logits, padding_value=0.0)
        labels = pad_group_rows(all_labels, padding_value=0.0)
        mask = pad_group_rows([item.to(dtype=torch.float32) for item in all_masks], padding_value=0.0).to(dtype=torch.bool)
        ambiguous = torch.cat(all_ambiguous, dim=0)
        baseline = pad_group_rows(all_baseline, padding_value=0.0)
        flat = flatten_scores(logits, labels, mask)
        ambiguous_scores: List[float] = []
        ambiguous_labels: List[int] = []
        for idx in range(logits.shape[0]):
            if not bool(ambiguous[idx].item()):
                continue
            for cand_idx in torch.nonzero(mask[idx], as_tuple=False).view(-1).tolist():
                ambiguous_scores.append(float(logits[idx, cand_idx].item()))
                ambiguous_labels.append(int(labels[idx, cand_idx].item()))
        flips = edit_flip_stats(logits, labels, mask, baseline)
        summary_row.update(
            {
                "mode": str(payload.get("mode", "nofreq")),
                "blocks": int(len(blocks)),
                "groups": int(len(groups)),
                "ambiguous_groups": int(sum(1 for group in groups if group.ambiguous)),
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
