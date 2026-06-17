#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.graph_assoc_gate import GraphAssocDualHeadGate, GraphAssocGate
from models.graph_assoc_gate_runtime import GRAPH_ASSOC_GATE_FEATURE_NAMES


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
SUMMARY_FIELDS = [
    "train_jsonl",
    "val_jsonl",
    "model_type",
    "init_checkpoint",
    "init_loaded",
    "feature_dim",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "hidden_dim",
    "num_hidden_layers",
    "dropout",
    "pos_weight",
    "neutral_pos_weight",
    "neutral_loss_weight",
    "neutral_risk_weight",
    "rank_loss_weight",
    "rank_margin",
    "ordinal_rank_loss_weight",
    "ordinal_rank_margin",
    "avg_train_target",
    "avg_train_weight",
    "avg_val_target",
    "avg_val_weight",
    "best_epoch",
    "best_metric",
    "best_threshold",
    "val_accuracy",
    "val_balanced_accuracy",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_positive_recall",
    "val_mean_prob_harmful",
    "val_mean_prob_neutral",
    "val_mean_prob_positive",
    "val_prob_gap_pos_neutral",
    "val_prob_gap_neutral_harmful",
    "val_mean_decision_harmful",
    "val_mean_decision_neutral",
    "val_mean_decision_positive",
    "train_rows",
    "val_rows",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train graph-association gate model.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-hidden-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--model-type", default="single_head", choices=["single_head", "dual_head"])
    parser.add_argument("--init-checkpoint", default="", help="Optional checkpoint path to warm start from.")
    parser.add_argument("--pos-weight", type=float, default=0.0, help="<=0 enables automatic class-ratio weighting")
    parser.add_argument("--neutral-pos-weight", type=float, default=1.0, help="Positive weight for the neutral-risk head BCE; ignored for single_head")
    parser.add_argument("--neutral-loss-weight", type=float, default=1.0, help="Loss weight applied to the neutral-risk head BCE; ignored for single_head")
    parser.add_argument("--neutral-risk-weight", type=float, default=1.0, help="Dual-head runtime decision uses gain_prob - neutral_risk_weight * neutral_prob")
    parser.add_argument("--rank-loss-weight", type=float, default=0.0, help="Additional pairwise ranking loss weight for positive-vs-nonpositive ordering")
    parser.add_argument("--rank-margin", type=float, default=0.1, help="Margin used by the optional pairwise ranking loss")
    parser.add_argument("--ordinal-rank-loss-weight", type=float, default=0.0, help="Additional ordered ranking loss weight for positive > neutral > harmful separation")
    parser.add_argument("--ordinal-rank-margin", type=float, default=0.08, help="Margin used by the optional ordered ranking loss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    parser.add_argument("--registry-dataset", default="MOT20")
    parser.add_argument("--registry-split", default="graph_assoc_gate_jsonl")
    parser.add_argument("--registry-notes", default="graph-association gate training")
    return parser.parse_args()


def write_single_row_csv(path: Path, fieldnames: List[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_registry(args: argparse.Namespace, summary_csv: Path, checkpoint: Path | None, status: str, notes: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(args.registry_csv),
        "--kind",
        "train",
        "--status",
        status,
        "--script",
        "scripts/train_graph_assoc_gate.py",
        "--dataset",
        str(args.registry_dataset),
        "--split",
        str(args.registry_split),
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
        "--checkpoint",
        str(checkpoint) if checkpoint else "",
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class GateDataset(Dataset):
    def __init__(self, rows: List[Dict[str, object]], feature_names: List[str]) -> None:
        self.rows = rows
        self.feature_names = feature_names

    @staticmethod
    def from_jsonl(path: str) -> "GateDataset":
        rows: List[Dict[str, object]] = []
        feature_names: List[str] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row_feature_names = [str(name) for name in list(row.get("feature_names", []) or [])]
                if not row_feature_names:
                    row_feature_names = list(GRAPH_ASSOC_GATE_FEATURE_NAMES)
                    row["feature_names"] = list(row_feature_names)
                row_features = list(row.get("features", []) or [])
                if not feature_names:
                    feature_names = list(row_feature_names)
                if row_feature_names != feature_names:
                    raise ValueError(f"Inconsistent feature_names in {path}")
                if len(row_features) != len(feature_names):
                    raise ValueError(f"Feature length mismatch in {path}: expected {len(feature_names)} got {len(row_features)}")
                rows.append(row)
        return GateDataset(rows, feature_names or list(GRAPH_ASSOC_GATE_FEATURE_NAMES))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        gain_class = int(row.get("gain_class", 2 if int(row["label"]) > 0 else 0))
        return {
            "features": torch.tensor(row["features"], dtype=torch.float32),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "gain_class": torch.tensor(gain_class, dtype=torch.long),
            "neutral_label": torch.tensor(1.0 if gain_class == 1 else 0.0, dtype=torch.float32),
            "train_target": torch.tensor(float(row.get("train_target", row["label"])), dtype=torch.float32),
            "sample_weight": torch.tensor(float(row.get("sample_weight", 1.0)), dtype=torch.float32),
        }


def collate_gate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "features": torch.stack([row["features"] for row in batch], dim=0),
        "label": torch.stack([row["label"] for row in batch], dim=0),
        "gain_class": torch.stack([row["gain_class"] for row in batch], dim=0),
        "neutral_label": torch.stack([row["neutral_label"] for row in batch], dim=0),
        "train_target": torch.stack([row["train_target"] for row in batch], dim=0),
        "sample_weight": torch.stack([row["sample_weight"] for row in batch], dim=0),
    }


def weighted_bce_loss(logits: torch.Tensor, targets: torch.Tensor, sample_weight: torch.Tensor, pos_weight_value: float) -> torch.Tensor:
    pos_weight = torch.tensor([float(pos_weight_value)], dtype=torch.float32, device=logits.device)
    losses = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    return (losses * sample_weight).mean()


def pairwise_rank_loss(logits: torch.Tensor, labels: torch.Tensor, sample_weight: torch.Tensor, margin: float) -> torch.Tensor:
    pos_mask = labels > 0.5
    neg_mask = labels <= 0.5
    if int(pos_mask.sum().item()) <= 0 or int(neg_mask.sum().item()) <= 0:
        return logits.new_zeros(())

    pos_logits = logits[pos_mask].view(-1, 1)
    neg_logits = logits[neg_mask].view(1, -1)
    pair_margin = float(margin) - (pos_logits - neg_logits)
    pair_losses = torch.relu(pair_margin)

    pos_weight = sample_weight[pos_mask].view(-1, 1)
    neg_weight = sample_weight[neg_mask].view(1, -1)
    pair_weight = pos_weight * neg_weight
    denom = torch.clamp(pair_weight.sum(), min=1e-8)
    return (pair_losses * pair_weight).sum() / denom


def ordered_pairwise_rank_loss(
    logits: torch.Tensor,
    gain_class: torch.Tensor,
    sample_weight: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    if logits.numel() <= 1:
        return logits.new_zeros(())

    class_diff = gain_class.view(-1, 1) - gain_class.view(1, -1)
    valid_mask = class_diff > 0
    if int(valid_mask.sum().item()) <= 0:
        return logits.new_zeros(())

    logit_diff = logits.view(-1, 1) - logits.view(1, -1)
    desired_margin = class_diff.to(dtype=logits.dtype) * float(margin)
    pair_losses = torch.relu(desired_margin - logit_diff)
    pair_weight = sample_weight.view(-1, 1) * sample_weight.view(1, -1)
    weighted_mask = valid_mask.to(dtype=logits.dtype)
    denom = torch.clamp((pair_weight * weighted_mask).sum(), min=1e-8)
    return (pair_losses * pair_weight * weighted_mask).sum() / denom


def _forward_scores(
    model: nn.Module,
    features: torch.Tensor,
    *,
    model_type: str,
    neutral_risk_weight: float,
) -> Dict[str, torch.Tensor]:
    if str(model_type) == "dual_head":
        gain_logits, neutral_logits = model(features)
        gain_prob = torch.sigmoid(gain_logits)
        neutral_prob = torch.sigmoid(neutral_logits)
        decision_score = gain_prob - float(neutral_risk_weight) * neutral_prob
        return {
            "gain_logits": gain_logits,
            "neutral_logits": neutral_logits,
            "gain_prob": gain_prob,
            "neutral_prob": neutral_prob,
            "decision_score": decision_score,
        }

    logits = model(features)
    gain_prob = torch.sigmoid(logits)
    zeros = torch.zeros_like(gain_prob)
    return {
        "gain_logits": logits,
        "neutral_logits": zeros,
        "gain_prob": gain_prob,
        "neutral_prob": zeros,
        "decision_score": gain_prob,
    }


def compute_metrics_from_scores(scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> Dict[str, float]:
    preds = (scores >= float(threshold)).float()
    tp = float(((preds == 1) & (labels == 1)).sum().item())
    tn = float(((preds == 0) & (labels == 0)).sum().item())
    fp = float(((preds == 1) & (labels == 0)).sum().item())
    fn = float(((preds == 0) & (labels == 1)).sum().item())
    total = max(tp + tn + fp + fn, 1.0)
    accuracy = (tp + tn) / total
    recall = tp / max(tp + fn, 1.0)
    tnr = tn / max(tn + fp, 1.0)
    precision = tp / max(tp + fp, 1.0)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-8)
    balanced_accuracy = 0.5 * (recall + tnr)
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_recall": float(recall),
    }


def _threshold_grid(model_type: str) -> List[float]:
    if str(model_type) == "dual_head":
        return [float(step) / 100.0 for step in range(-50, 96)]
    return [float(step) / 100.0 for step in range(5, 96)]


def select_best_threshold(scores: torch.Tensor, labels: torch.Tensor, *, model_type: str) -> tuple[float, Dict[str, float]]:
    grid = _threshold_grid(model_type)
    best_threshold = float(grid[0]) if grid else 0.5
    best_metrics = compute_metrics_from_scores(scores, labels, best_threshold)
    best_key = (
        float(best_metrics["balanced_accuracy"]),
        float(best_metrics["f1"]),
        float(best_metrics["precision"]),
        -abs(best_threshold - 0.5),
    )
    for threshold in grid[1:]:
        metrics = compute_metrics_from_scores(scores, labels, threshold)
        key = (
            float(metrics["balanced_accuracy"]),
            float(metrics["f1"]),
            float(metrics["precision"]),
            -abs(threshold - 0.5),
        )
        if key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_key = key
    return best_threshold, best_metrics


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    model_type: str,
    pos_weight_value: float,
    neutral_pos_weight_value: float,
    neutral_loss_weight: float,
    neutral_risk_weight: float,
) -> Dict[str, float]:
    model.eval()
    all_gain_prob: List[torch.Tensor] = []
    all_neutral_prob: List[torch.Tensor] = []
    all_decision_score: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_gain_classes: List[torch.Tensor] = []
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            neutral_label = batch["neutral_label"].to(device)
            sample_weight = torch.ones_like(labels, device=device)
            outputs = _forward_scores(
                model,
                features,
                model_type=model_type,
                neutral_risk_weight=neutral_risk_weight,
            )
            gain_loss = weighted_bce_loss(outputs["gain_logits"], labels, sample_weight, pos_weight_value)
            loss = gain_loss
            if str(model_type) == "dual_head":
                neutral_loss = weighted_bce_loss(
                    outputs["neutral_logits"],
                    neutral_label,
                    sample_weight,
                    neutral_pos_weight_value,
                )
                loss = loss + float(neutral_loss_weight) * neutral_loss
            total_loss += float(loss.item())
            batches += 1
            all_gain_prob.append(outputs["gain_prob"].cpu())
            all_neutral_prob.append(outputs["neutral_prob"].cpu())
            all_decision_score.append(outputs["decision_score"].cpu())
            all_labels.append(labels.cpu())
            all_gain_classes.append(batch["gain_class"].cpu())
    gain_prob = torch.cat(all_gain_prob, dim=0) if all_gain_prob else torch.zeros((0,), dtype=torch.float32)
    neutral_prob = torch.cat(all_neutral_prob, dim=0) if all_neutral_prob else torch.zeros((0,), dtype=torch.float32)
    decision_score = torch.cat(all_decision_score, dim=0) if all_decision_score else torch.zeros((0,), dtype=torch.float32)
    labels = torch.cat(all_labels, dim=0) if all_labels else torch.zeros((0,), dtype=torch.float32)
    gain_class = torch.cat(all_gain_classes, dim=0) if all_gain_classes else torch.zeros((0,), dtype=torch.long)
    best_threshold, metrics = select_best_threshold(decision_score, labels, model_type=model_type)
    metrics["loss"] = float(total_loss / batches) if batches else 0.0
    metrics["best_threshold"] = float(best_threshold)
    for class_id, name in ((0, "harmful"), (1, "neutral"), (2, "positive")):
        mask = gain_class == int(class_id)
        metrics[f"mean_prob_{name}"] = float(gain_prob[mask].mean().item()) if int(mask.sum().item()) > 0 else 0.0
        metrics[f"mean_decision_{name}"] = float(decision_score[mask].mean().item()) if int(mask.sum().item()) > 0 else 0.0
        if str(model_type) == "dual_head":
            metrics[f"mean_neutral_prob_{name}"] = float(neutral_prob[mask].mean().item()) if int(mask.sum().item()) > 0 else 0.0
    metrics["prob_gap_pos_neutral"] = float(metrics["mean_prob_positive"] - metrics["mean_prob_neutral"])
    metrics["prob_gap_neutral_harmful"] = float(metrics["mean_prob_neutral"] - metrics["mean_prob_harmful"])
    return metrics


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    metrics_jsonl = out_dir / "metrics.jsonl"
    best_ckpt = out_dir / "best.pt"
    summary_row: Dict[str, object] = {
        "train_jsonl": str(Path(args.train_jsonl)),
        "val_jsonl": str(Path(args.val_jsonl)),
        "model_type": str(args.model_type),
        "init_checkpoint": str(Path(str(args.init_checkpoint)).expanduser().resolve()) if str(args.init_checkpoint or "").strip() else "",
        "init_loaded": False,
        "feature_dim": 0,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": int(args.hidden_dim),
        "num_hidden_layers": int(args.num_hidden_layers),
        "dropout": float(args.dropout),
        "pos_weight": 0.0,
        "neutral_pos_weight": float(args.neutral_pos_weight),
        "neutral_loss_weight": float(args.neutral_loss_weight),
        "neutral_risk_weight": float(args.neutral_risk_weight),
        "rank_loss_weight": float(args.rank_loss_weight),
        "rank_margin": float(args.rank_margin),
        "ordinal_rank_loss_weight": float(args.ordinal_rank_loss_weight),
        "ordinal_rank_margin": float(args.ordinal_rank_margin),
        "avg_train_target": 0.0,
        "avg_train_weight": 0.0,
        "avg_val_target": 0.0,
        "avg_val_weight": 0.0,
        "best_epoch": -1,
        "best_metric": 0.0,
        "best_threshold": 0.5,
        "val_accuracy": 0.0,
        "val_balanced_accuracy": 0.0,
        "val_precision": 0.0,
        "val_recall": 0.0,
        "val_f1": 0.0,
        "val_positive_recall": 0.0,
        "val_mean_prob_harmful": 0.0,
        "val_mean_prob_neutral": 0.0,
        "val_mean_prob_positive": 0.0,
        "val_prob_gap_pos_neutral": 0.0,
        "val_prob_gap_neutral_harmful": 0.0,
        "val_mean_decision_harmful": 0.0,
        "val_mean_decision_neutral": 0.0,
        "val_mean_decision_positive": 0.0,
        "train_rows": 0,
        "val_rows": 0,
        "status": "running",
        "error": "",
    }
    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
    append_registry(args, summary_csv, best_ckpt, "running", str(args.registry_notes))

    try:
        train_dataset = GateDataset.from_jsonl(args.train_jsonl)
        val_dataset = GateDataset.from_jsonl(args.val_jsonl)
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("Empty graph-association gate dataset.")
        if list(train_dataset.feature_names) != list(val_dataset.feature_names):
            raise ValueError("Train/val gate datasets have different feature_names.")
        feature_names = list(train_dataset.feature_names)

        summary_row["feature_dim"] = int(len(feature_names))
        summary_row["train_rows"] = int(len(train_dataset))
        summary_row["val_rows"] = int(len(val_dataset))
        summary_row["avg_train_target"] = float(sum(float(row.get("train_target", row["label"])) for row in train_dataset.rows) / float(max(len(train_dataset.rows), 1)))
        summary_row["avg_train_weight"] = float(sum(float(row.get("sample_weight", 1.0)) for row in train_dataset.rows) / float(max(len(train_dataset.rows), 1)))
        summary_row["avg_val_target"] = float(sum(float(row.get("train_target", row["label"])) for row in val_dataset.rows) / float(max(len(val_dataset.rows), 1)))
        summary_row["avg_val_weight"] = float(sum(float(row.get("sample_weight", 1.0)) for row in val_dataset.rows) / float(max(len(val_dataset.rows), 1)))
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)

        pos_count = sum(int(row["label"]) for row in train_dataset.rows)
        neg_count = len(train_dataset.rows) - pos_count
        pos_weight_value = float(args.pos_weight)
        if pos_weight_value <= 0.0:
            pos_weight_value = float(neg_count / max(pos_count, 1))
        summary_row["pos_weight"] = float(pos_weight_value)
        neutral_pos_weight_value = float(args.neutral_pos_weight)
        if str(args.model_type) == "dual_head" and neutral_pos_weight_value <= 0.0:
            neutral_pos_count = sum(int(int(row.get("gain_class", 2 if int(row["label"]) > 0 else 0)) == 1) for row in train_dataset.rows)
            neutral_neg_count = len(train_dataset.rows) - neutral_pos_count
            neutral_pos_weight_value = float(neutral_neg_count / max(neutral_pos_count, 1))
        summary_row["neutral_pos_weight"] = float(neutral_pos_weight_value)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)

        train_loader = DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=0,
            collate_fn=collate_gate,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_gate,
        )

        device = torch.device(str(args.device))
        if str(args.model_type) == "dual_head":
            model = GraphAssocDualHeadGate(
                input_dim=len(feature_names),
                hidden_dim=int(args.hidden_dim),
                dropout=float(args.dropout),
                num_hidden_layers=int(args.num_hidden_layers),
            ).to(device)
        else:
            model = GraphAssocGate(
                input_dim=len(feature_names),
                hidden_dim=int(args.hidden_dim),
                dropout=float(args.dropout),
                num_hidden_layers=int(args.num_hidden_layers),
            ).to(device)
        if str(args.init_checkpoint or "").strip():
            init_path = Path(str(args.init_checkpoint)).expanduser().resolve()
            if not init_path.is_file():
                raise FileNotFoundError(f"Missing init checkpoint: {init_path}")
            payload = torch.load(init_path, map_location="cpu")
            payload_model_type = str(payload.get("model_type", str(args.model_type)) or str(args.model_type))
            if payload_model_type != str(args.model_type):
                raise ValueError(
                    f"Init checkpoint model_type mismatch: expected {args.model_type}, got {payload_model_type}"
                )
            payload_input_dim = int(payload.get("input_dim", len(feature_names)))
            if payload_input_dim != int(len(feature_names)):
                raise ValueError(
                    f"Init checkpoint input_dim mismatch: expected {len(feature_names)}, got {payload_input_dim}"
                )
            payload_feature_names = [str(name) for name in list(payload.get("feature_names", feature_names) or [])]
            if payload_feature_names and payload_feature_names != feature_names:
                raise ValueError("Init checkpoint feature_names mismatch with current dataset.")
            current_state = model.state_dict()
            pretrained_state = dict(payload["model_state"])
            filtered_state = {
                key: value
                for key, value in pretrained_state.items()
                if key in current_state and tuple(value.shape) == tuple(current_state[key].shape)
            }
            skipped_state = sorted(set(pretrained_state.keys()) - set(filtered_state.keys()))
            load_result = model.load_state_dict(filtered_state, strict=False)
            if skipped_state or load_result.missing_keys or load_result.unexpected_keys:
                print(
                    "[init_checkpoint] partial load:",
                    f"loaded={len(filtered_state)}",
                    f"skipped={len(skipped_state)}",
                    f"missing={len(load_result.missing_keys)}",
                    f"unexpected={len(load_result.unexpected_keys)}",
                )
            summary_row["init_loaded"] = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

        has_soft_supervision = any(
            abs(float(row.get("train_target", row["label"])) - float(row["label"])) > 1e-6
            or abs(float(row.get("sample_weight", 1.0)) - 1.0) > 1e-6
            for row in train_dataset.rows
        )
        best_metric = -1.0
        with metrics_jsonl.open("w", encoding="utf-8") as handle:
            for epoch in range(1, int(args.epochs) + 1):
                model.train()
                for batch in train_loader:
                    features = batch["features"].to(device)
                    labels = batch["label"].to(device)
                    gain_class = batch["gain_class"].to(device)
                    neutral_label = batch["neutral_label"].to(device)
                    train_target = batch["train_target"].to(device)
                    sample_weight = batch["sample_weight"].to(device)
                    outputs = _forward_scores(
                        model,
                        features,
                        model_type=str(args.model_type),
                        neutral_risk_weight=float(args.neutral_risk_weight),
                    )
                    gain_loss = weighted_bce_loss(outputs["gain_logits"], train_target, sample_weight, pos_weight_value)
                    loss = gain_loss
                    if str(args.model_type) == "dual_head":
                        neutral_loss = weighted_bce_loss(
                            outputs["neutral_logits"],
                            neutral_label,
                            sample_weight,
                            neutral_pos_weight_value,
                        )
                        loss = loss + float(args.neutral_loss_weight) * neutral_loss
                    if float(args.rank_loss_weight) > 0.0:
                        rank_loss = pairwise_rank_loss(
                            outputs["decision_score"],
                            labels,
                            sample_weight,
                            margin=float(args.rank_margin),
                        )
                        loss = loss + float(args.rank_loss_weight) * rank_loss
                    if float(args.ordinal_rank_loss_weight) > 0.0:
                        ordinal_rank = ordered_pairwise_rank_loss(
                            outputs["decision_score"],
                            gain_class,
                            sample_weight,
                            margin=float(args.ordinal_rank_margin),
                        )
                        loss = loss + float(args.ordinal_rank_loss_weight) * ordinal_rank
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                val_metrics = evaluate(
                    model,
                    val_loader,
                    device,
                    model_type=str(args.model_type),
                    pos_weight_value=pos_weight_value,
                    neutral_pos_weight_value=neutral_pos_weight_value,
                    neutral_loss_weight=float(args.neutral_loss_weight),
                    neutral_risk_weight=float(args.neutral_risk_weight),
                )
                handle.write(json.dumps({"epoch": int(epoch), **val_metrics}))
                handle.write("\n")
                handle.flush()

                metric = float(val_metrics["balanced_accuracy"])
                if metric >= best_metric:
                    best_metric = metric
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "feature_names": list(feature_names),
                            "input_dim": int(len(feature_names)),
                            "hidden_dim": int(args.hidden_dim),
                            "num_hidden_layers": int(args.num_hidden_layers),
                            "dropout": float(args.dropout),
                            "model_type": str(args.model_type),
                            "train_target_mode": "soft_weighted" if has_soft_supervision else "hard",
                            "gate_threshold": float(val_metrics["best_threshold"]),
                            "decision_mode": "positive_minus_weighted_neutral"
                            if str(args.model_type) == "dual_head"
                            else "positive_probability",
                            "neutral_pos_weight": float(neutral_pos_weight_value),
                            "neutral_loss_weight": float(args.neutral_loss_weight),
                            "neutral_risk_weight": float(args.neutral_risk_weight),
                        },
                        best_ckpt,
                    )
                    summary_row.update(
                        {
                            "best_epoch": int(epoch),
                            "best_metric": float(metric),
                            "best_threshold": float(val_metrics["best_threshold"]),
                            "val_accuracy": float(val_metrics["accuracy"]),
                            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                            "val_precision": float(val_metrics["precision"]),
                            "val_recall": float(val_metrics["recall"]),
                            "val_f1": float(val_metrics["f1"]),
                            "val_positive_recall": float(val_metrics["positive_recall"]),
                            "val_mean_prob_harmful": float(val_metrics["mean_prob_harmful"]),
                            "val_mean_prob_neutral": float(val_metrics["mean_prob_neutral"]),
                            "val_mean_prob_positive": float(val_metrics["mean_prob_positive"]),
                            "val_prob_gap_pos_neutral": float(val_metrics["prob_gap_pos_neutral"]),
                            "val_prob_gap_neutral_harmful": float(val_metrics["prob_gap_neutral_harmful"]),
                            "val_mean_decision_harmful": float(val_metrics["mean_decision_harmful"]),
                            "val_mean_decision_neutral": float(val_metrics["mean_decision_neutral"]),
                            "val_mean_decision_positive": float(val_metrics["mean_decision_positive"]),
                        }
                    )
                    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)

        summary_row["status"] = "success"
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, best_ckpt, "success", str(args.registry_notes))
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = repr(exc)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, best_ckpt, "failed", str(args.registry_notes))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
