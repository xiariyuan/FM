#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.graph_assoc_commit_policy import (  # noqa: E402
    ACTION_NAMES,
    MODEL_ARCH_HIER_ROUTE,
    MODEL_ARCH_ROUTED_MOE,
    MODEL_ARCH_SET_SLOT,
    GraphAssocCommitHierarchicalRoutePolicy,
    GraphAssocCommitRoutedSetSlotPolicy,
    GraphAssocCommitSetSlotPolicy as GraphAssocCommitPolicy,
)


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
SUMMARY_FIELDS = [
    "data_jsonl",
    "source_manifest",
    "dataset_tag",
    "feature_version",
    "model_arch",
    "checkpoint",
    "init_checkpoint",
    "train_examples",
    "val_examples",
    "train_sequences",
    "val_sequences",
    "epochs",
    "batch_size",
    "hidden_dim",
    "policy_hidden_dim",
    "token_dim",
    "num_slots",
    "num_encoder_layers",
    "num_heads",
    "num_experts",
    "router_hidden_dim",
    "cluster_dim",
    "router_temperature",
    "policy_score_mode",
    "train_selection_head_only",
    "dropout",
    "action_loss_weight",
    "gain_loss_weight",
    "policy_loss_weight",
    "router_balance_loss_weight",
    "expert_aux_loss_weight",
    "rank_loss_weight",
    "rank_margin",
    "ordinal_rank_loss_weight",
    "ordinal_rank_margin",
    "risk_weight_power",
    "risk_cost_scale",
    "risk_utility_scale",
    "risk_negative_boost",
    "risk_neutral_boost",
    "risk_positive_discount",
    "risk_target_cost_weight",
    "risk_target_suppressed_weight",
    "risk_target_recent_owner_weight",
    "risk_target_introduced_weight",
    "selection_metric",
    "use_balanced_sampler",
    "class_weight_rewrite",
    "class_weight_defer",
    "class_weight_reject",
    "best_metric_name",
    "best_epoch",
    "best_metric",
    "val_action_acc",
    "val_action_macro_f1",
    "val_policy_loss",
    "val_policy_sign_acc",
    "val_rank_loss",
    "val_ordinal_rank_loss",
    "train_router_balance_loss",
    "train_router_entropy",
    "train_router_top1",
    "train_router_margin",
    "train_expert_aux_loss",
    "train_route_loss",
    "train_gate_loss",
    "train_route_acc",
    "train_gate_acc",
    "val_router_balance_loss",
    "val_router_entropy",
    "val_router_top1",
    "val_router_margin",
    "val_expert_aux_loss",
    "val_route_loss",
    "val_gate_loss",
    "val_route_acc",
    "val_gate_acc",
    "val_rewrite_precision",
    "val_rewrite_recall",
    "val_defer_precision",
    "val_defer_recall",
    "val_reject_precision",
    "val_reject_recall",
    "val_gain_mae",
    "val_gain_sign_acc",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train graph-association commit action policy.")
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--policy-hidden-dim", type=int, default=128)
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--num-slots", type=int, default=4)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=3)
    parser.add_argument("--router-hidden-dim", type=int, default=128)
    parser.add_argument("--gate-hidden-dim", type=int, default=128)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument(
        "--policy-score-mode",
        default="learned_selection",
        choices=[
            "learned_selection",
            "legacy_mixture",
            "residual_selection",
            "calibrated_residual",
            "gated_blend",
            "gated_route_blend",
            "route_residual",
            "base_policy",
            "final_action_margin",
        ],
    )
    parser.add_argument("--train-selection-head-only", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--gain-loss-weight", type=float, default=0.5)
    parser.add_argument("--policy-loss-weight", type=float, default=0.5)
    parser.add_argument("--router-balance-loss-weight", type=float, default=0.02)
    parser.add_argument("--expert-aux-loss-weight", type=float, default=0.20)
    parser.add_argument("--route-loss-weight", type=float, default=0.50)
    parser.add_argument("--gate-loss-weight", type=float, default=0.25)
    parser.add_argument("--rank-loss-weight", type=float, default=0.0)
    parser.add_argument("--rank-margin", type=float, default=0.1)
    parser.add_argument("--ordinal-rank-loss-weight", type=float, default=0.0)
    parser.add_argument("--ordinal-rank-margin", type=float, default=0.08)
    parser.add_argument("--risk-weight-power", type=float, default=1.0)
    parser.add_argument("--risk-cost-scale", type=float, default=0.10)
    parser.add_argument("--risk-utility-scale", type=float, default=0.05)
    parser.add_argument("--risk-negative-boost", type=float, default=1.5)
    parser.add_argument("--risk-neutral-boost", type=float, default=0.75)
    parser.add_argument("--risk-positive-discount", type=float, default=0.35)
    parser.add_argument("--risk-target-cost-weight", type=float, default=0.35)
    parser.add_argument("--risk-target-suppressed-weight", type=float, default=0.15)
    parser.add_argument("--risk-target-recent-owner-weight", type=float, default=0.10)
    parser.add_argument("--risk-target-introduced-weight", type=float, default=0.08)
    parser.add_argument(
        "--selection-metric",
        default="policy_sign_acc",
        choices=["policy_sign_acc", "action_macro_f1", "gain_sign_acc"],
    )
    parser.add_argument(
        "--model-arch",
        default=MODEL_ARCH_SET_SLOT,
        choices=[MODEL_ARCH_SET_SLOT, MODEL_ARCH_ROUTED_MOE, MODEL_ARCH_HIER_ROUTE],
    )
    parser.add_argument("--use-balanced-sampler", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-sequences", type=str, default="")
    parser.add_argument("--val-sequences", type=str, default="")
    parser.add_argument("--strict-sequence-split", action="store_true")
    parser.add_argument("--min-val-examples", type=int, default=1)
    parser.add_argument("--dataset-tag", type=str, default="graph_assoc_commit_policy")
    parser.add_argument("--source-manifest", type=str, default="")
    parser.add_argument("--feature-version", type=str, default="graph_assoc_v1")
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def _write_single_row_csv(path: Path, fieldnames: List[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _append_registry(args: argparse.Namespace, summary_csv: Path, checkpoint: Path | None, status: str, notes: str) -> None:
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
        "scripts/train_graph_assoc_commit_policy.py",
        "--dataset",
        str(args.dataset_tag or ""),
        "--split",
        "graph_assoc_commit_policy",
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


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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
        split_at = max(int(len(examples) * 0.8), 1)
        train_rows = examples[:split_at]
        val_rows = examples[split_at:] or examples[-1:]
        split_mode = "fallback_80_20"

    if not train_rows or not val_rows or len(val_rows) < int(min_val_examples):
        if strict_sequence_split:
            raise ValueError(
                "Strict sequence split failed: "
                f"train_rows={len(train_rows)} val_rows={len(val_rows)} min_val_examples={int(min_val_examples)}"
            )
        split_at = max(int(len(examples) * 0.8), 1)
        train_rows = examples[:split_at]
        val_rows = examples[split_at:] or examples[-1:]
        split_mode = "fallback_80_20"

    train_sequences = sorted({str(row["seq"]) for row in train_rows})
    val_sequences = sorted({str(row["seq"]) for row in val_rows})
    return train_rows, val_rows, train_sequences, val_sequences, split_mode


class CommitPolicyDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        action_label = int(row.get("action_label", 1 if int(row.get("gt_gain", 0)) == 0 else (0 if int(row.get("gt_gain", 0)) > 0 else 2)))
        route_label = 0 if int(row.get("gt_gain", 0)) > 0 else (1 if int(row.get("gt_gain", 0)) == 0 else 2)
        gate_label = int(bool(row.get("rules_passed_before_learned_gate", row.get("learned_commit_replace_rules", 0))))
        return {
            "cluster_id": str(row.get("cluster_id", "")),
            "seq": str(row.get("seq", "")),
            "frame": int(row.get("frame", 0)),
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
            "action_label": torch.tensor(action_label, dtype=torch.long),
            "route_label": torch.tensor(route_label, dtype=torch.long),
            "gate_label": torch.tensor(gate_label, dtype=torch.long),
            "gain_target": torch.tensor(float(row.get("gt_gain", 0.0)), dtype=torch.float32),
            "risk_weight": torch.tensor(float(row.get("risk_weight", 1.0)), dtype=torch.float32),
            "risk_adjusted_gain_target": torch.tensor(
                float(row.get("risk_adjusted_gain_target", row.get("gt_gain", 0.0))),
                dtype=torch.float32,
            ),
            "utility_gain": float(row.get("utility_gain", 0.0)),
            "cost_delta": float(row.get("cost_delta", 0.0)),
            "baseline_cost": float(row.get("baseline_cost", 0.0)),
            "chosen_cost": float(row.get("chosen_cost", 0.0)),
            "introduced_row_count": int(row.get("introduced_row_count", len(row.get("introduced_rows", [])))),
            "suppressed_row_count": int(row.get("suppressed_row_count", len(row.get("suppressed_rows", [])))),
            "recent_owner_row_count": int(row.get("recent_owner_row_count", len(row.get("recent_owner_rows", [])))),
            "gt_gain": int(row.get("gt_gain", 0)),
            "gt_decision": str(row.get("gt_decision", "")),
        }


def _collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


def _load_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "gt_gain" not in row:
                raise ValueError(f"Missing gt_gain in {path}")
            if "action_label" not in row:
                gt_gain = int(row.get("gt_gain", 0))
                row["action_label"] = 0 if gt_gain > 0 else (2 if gt_gain < 0 else 1)
                row["action_name"] = "rewrite" if gt_gain > 0 else ("reject" if gt_gain < 0 else "defer")
            examples.append(row)
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def _risk_weight_from_row(
    row: dict[str, Any],
    *,
    cost_scale: float,
    utility_scale: float,
    power: float,
    negative_boost: float,
    neutral_boost: float,
    positive_discount: float,
) -> float:
    gt_gain = int(row.get("gt_gain", 0))
    cost_delta = max(float(row.get("cost_delta", 0.0)), 0.0)
    utility_gain = float(row.get("utility_gain", 0.0))
    introduced_count = int(row.get("introduced_row_count", len(row.get("introduced_rows", []))))
    suppressed_count = int(row.get("suppressed_row_count", len(row.get("suppressed_rows", []))))
    recent_owner_count = int(row.get("recent_owner_row_count", len(row.get("recent_owner_rows", []))))
    risk_cost = cost_delta / max(float(cost_scale), 1e-6)
    weak_gain = max(-utility_gain, 0.0) / max(float(utility_scale), 1e-6)
    structural_risk = 0.35 * float(suppressed_count) + 0.20 * float(recent_owner_count) + 0.15 * float(introduced_count)
    risk_raw = max(risk_cost + weak_gain + structural_risk, 0.0)
    risk_mag = min(risk_raw, 4.0)
    if gt_gain < 0:
        factor = 1.0 + float(negative_boost) * (risk_mag ** max(float(power), 0.0))
    elif gt_gain == 0:
        factor = 1.0 + float(neutral_boost) * (risk_mag ** max(float(power), 0.0))
    else:
        factor = 1.0 / (1.0 + float(positive_discount) * (risk_mag ** max(float(power), 0.0)))
    return float(min(max(factor, 0.25), 6.0))


def _risk_adjusted_target_from_row(
    row: dict[str, Any],
    *,
    cost_weight: float,
    suppressed_weight: float,
    recent_owner_weight: float,
    introduced_weight: float,
) -> float:
    gt_gain = float(row.get("gt_gain", 0.0))
    cost_delta = max(float(row.get("cost_delta", 0.0)), 0.0)
    suppressed_count = int(row.get("suppressed_row_count", len(row.get("suppressed_rows", []))))
    recent_owner_count = int(row.get("recent_owner_row_count", len(row.get("recent_owner_rows", []))))
    introduced_count = int(row.get("introduced_row_count", len(row.get("introduced_rows", []))))
    penalty = (
        float(cost_weight) * (cost_delta / 0.10)
        + float(suppressed_weight) * float(suppressed_count)
        + float(recent_owner_weight) * float(recent_owner_count)
        + float(introduced_weight) * float(introduced_count)
    )
    if gt_gain <= 0.0:
        return float(gt_gain - penalty)
    return float(gt_gain - min(penalty, 0.75))


def _attach_risk_targets(
    rows: list[dict[str, Any]],
    *,
    cost_scale: float,
    utility_scale: float,
    power: float,
    negative_boost: float,
    neutral_boost: float,
    positive_discount: float,
    target_cost_weight: float,
    target_suppressed_weight: float,
    target_recent_owner_weight: float,
    target_introduced_weight: float,
) -> list[dict[str, Any]]:
    updated_rows: list[dict[str, Any]] = []
    for base_row in rows:
        row = dict(base_row)
        row["risk_weight"] = _risk_weight_from_row(
            row,
            cost_scale=cost_scale,
            utility_scale=utility_scale,
            power=power,
            negative_boost=negative_boost,
            neutral_boost=neutral_boost,
            positive_discount=positive_discount,
        )
        row["risk_adjusted_gain_target"] = _risk_adjusted_target_from_row(
            row,
            cost_weight=target_cost_weight,
            suppressed_weight=target_suppressed_weight,
            recent_owner_weight=target_recent_owner_weight,
            introduced_weight=target_introduced_weight,
        )
        updated_rows.append(row)
    return updated_rows


def _load_state_dict_shape_compatible(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    model_state = model.state_dict()
    compatible_state: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            skipped.append(key)
            continue
        if tuple(target.shape) == tuple(value.shape):
            compatible_state[key] = value
            continue
        if key.endswith("cluster_mlp.0.weight") and value.ndim == 2 and target.ndim == 2 and value.shape[0] == target.shape[0] and value.shape[1] < target.shape[1]:
            padded = target.clone()
            padded.zero_()
            padded[:, : int(value.shape[1])] = value.to(dtype=target.dtype)
            compatible_state[key] = padded
            continue
        if key.endswith("cluster_token_proj.0.weight") and value.ndim == 2 and target.ndim == 2 and value.shape[0] == target.shape[0] and value.shape[1] < target.shape[1]:
            padded = target.clone()
            padded.zero_()
            padded[:, : int(value.shape[1])] = value.to(dtype=target.dtype)
            compatible_state[key] = padded
            continue
        skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    return list(missing), list(unexpected) + skipped


def _safe_f1(precision: float, recall: float) -> float:
    precision = float(precision)
    recall = float(recall)
    denom = precision + recall
    if denom <= 1e-12:
        return 0.0
    return 2.0 * precision * recall / denom


def _confusion_counts(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    mat = torch.zeros((int(num_classes), int(num_classes)), dtype=torch.float32)
    for p, t in zip(preds.view(-1).tolist(), targets.view(-1).tolist()):
        if 0 <= int(t) < int(num_classes) and 0 <= int(p) < int(num_classes):
            mat[int(t), int(p)] += 1.0
    return mat


def _macro_f1_from_confusion(confusion: torch.Tensor) -> tuple[float, list[dict[str, float]]]:
    details: list[dict[str, float]] = []
    f1s: list[float] = []
    for idx in range(int(confusion.shape[0])):
        tp = float(confusion[idx, idx].item())
        fp = float(confusion[:, idx].sum().item() - tp)
        fn = float(confusion[idx, :].sum().item() - tp)
        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        f1 = _safe_f1(precision, recall)
        details.append({"precision": precision, "recall": recall, "f1": f1})
        f1s.append(f1)
    return float(sum(f1s) / max(len(f1s), 1)), details


def _pairwise_rank_loss(logits: torch.Tensor, labels: torch.Tensor, sample_weight: torch.Tensor, margin: float) -> torch.Tensor:
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


def _ordered_pairwise_rank_loss(
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


def _run_epoch(
    *,
    model: GraphAssocCommitPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    class_weights: torch.Tensor,
    action_loss_weight: float,
    gain_loss_weight: float,
    policy_loss_weight: float,
    router_balance_loss_weight: float,
    expert_aux_loss_weight: float,
    route_loss_weight: float,
    gate_loss_weight: float,
    rank_loss_weight: float,
    rank_margin: float,
    ordinal_rank_loss_weight: float,
    ordinal_rank_margin: float,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    totals = {
        "loss": 0.0,
        "action_loss": 0.0,
        "gain_loss": 0.0,
        "policy_loss": 0.0,
        "rank_loss": 0.0,
        "ordinal_rank_loss": 0.0,
        "router_balance_loss": 0.0,
        "router_entropy": 0.0,
        "router_top1": 0.0,
        "router_margin": 0.0,
        "expert_aux_loss": 0.0,
        "route_loss": 0.0,
        "gate_loss": 0.0,
        "route_correct": 0.0,
        "gate_correct": 0.0,
        "router_batches": 0.0,
        "samples": 0.0,
        "rewrite_correct": 0.0,
        "defer_correct": 0.0,
        "reject_correct": 0.0,
        "pred_rewrite": 0.0,
        "pred_defer": 0.0,
        "pred_reject": 0.0,
        "tgt_rewrite": 0.0,
        "tgt_defer": 0.0,
        "tgt_reject": 0.0,
        "gain_abs_err": 0.0,
        "gain_sign_correct": 0.0,
        "policy_abs_err": 0.0,
        "policy_sign_correct": 0.0,
    }

    for batch in loader:
        losses = []
        batch_router_probs: list[torch.Tensor] = []
        batch_policy_scores: list[torch.Tensor] = []
        batch_gain_targets: list[torch.Tensor] = []
        batch_gain_classes: list[torch.Tensor] = []
        batch_sample_weights: list[torch.Tensor] = []
        balance_loss: torch.Tensor | None = None
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        for sample in batch:
            det_features = sample["det_features"].to(device)
            track_features = sample["track_features"].to(device)
            edge_features = sample["edge_features"].to(device)
            edge_det_index = sample["edge_det_index"].to(device)
            edge_track_index = sample["edge_track_index"].to(device)
            cluster_features = sample["cluster_features"].to(device)
            action_label = sample["action_label"].to(device)
            route_label = sample["route_label"].to(device)
            gate_label = sample["gate_label"].to(device)
            gain_target = sample["gain_target"].to(device)
            risk_weight = sample["risk_weight"].to(device)
            risk_adjusted_gain_target = sample["risk_adjusted_gain_target"].to(device)

            outputs = model(
                det_features=det_features,
                track_features=track_features,
                edge_features=edge_features,
                edge_det_index=edge_det_index,
                edge_track_index=edge_track_index,
                cluster_features=cluster_features,
            )
            action_logits = outputs["action_logits"].view(1, -1)
            gain_pred = outputs["gain_pred"].view(())
            policy_score = outputs["policy_score"].view(())
            action_loss = F.cross_entropy(action_logits, action_label.view(1), weight=class_weights) * risk_weight.view(())
            gain_loss = F.smooth_l1_loss(gain_pred, gain_target.view(()), reduction="none") * risk_weight.view(())
            policy_loss = (
                F.smooth_l1_loss(policy_score, risk_adjusted_gain_target.view(()), reduction="none") * risk_weight.view(())
            )
            loss = (
                float(action_loss_weight) * action_loss
                + float(gain_loss_weight) * gain_loss
                + float(policy_loss_weight) * policy_loss
            )
            router_probs = outputs.get("router_probs")
            if isinstance(router_probs, torch.Tensor) and router_probs.numel() > 0:
                batch_router_probs.append(router_probs.view(-1))
                totals["router_entropy"] += float(
                    float(outputs.get("router_entropy", torch.tensor(0.0, device=device)).detach().item())
                )
                totals["router_top1"] += float(router_probs.max().detach().item())
                totals["router_margin"] += float(
                    float(outputs.get("router_margin", torch.tensor(0.0, device=device)).detach().item())
                )
            route_logits = outputs.get("route_logits")
            gate_logit = outputs.get("gate_logit")
            if isinstance(route_logits, torch.Tensor) and route_logits.numel() >= len(ACTION_NAMES):
                route_loss = F.cross_entropy(route_logits.view(1, -1), route_label.view(1)) * risk_weight.view(())
                totals["route_loss"] += float(route_loss.detach().item())
                loss = loss + float(route_loss_weight) * route_loss
            else:
                route_loss = None
            if isinstance(gate_logit, torch.Tensor) and gate_logit.numel() > 0:
                gate_target = gate_label.to(dtype=torch.float32).view(())
                gate_loss = F.binary_cross_entropy_with_logits(gate_logit.view(()), gate_target, reduction="none") * risk_weight.view(())
                totals["gate_loss"] += float(gate_loss.detach().item())
                loss = loss + float(gate_loss_weight) * gate_loss
            else:
                gate_loss = None
            expert_action_logits = outputs.get("expert_action_logits")
            expert_gain_pred = outputs.get("expert_gain_pred")
            if isinstance(expert_action_logits, torch.Tensor) and isinstance(expert_gain_pred, torch.Tensor):
                expert_targets = action_label.view(1).expand(int(expert_action_logits.shape[0]))
                expert_action_loss = (
                    F.cross_entropy(
                        expert_action_logits.view(int(expert_action_logits.shape[0]), -1),
                        expert_targets,
                        weight=class_weights,
                    )
                    * risk_weight.view(())
                )
                expert_gain_target = gain_target.view(()).expand_as(expert_gain_pred.view(-1))
                expert_gain_loss = F.smooth_l1_loss(expert_gain_pred.view(-1), expert_gain_target, reduction="none").mean() * risk_weight.view(())
                sample_aux_loss = float(action_loss_weight) * expert_action_loss + float(gain_loss_weight) * expert_gain_loss
                totals["expert_aux_loss"] += float(sample_aux_loss.detach().item())
                loss = loss + float(expert_aux_loss_weight) * sample_aux_loss
            losses.append(loss)
            batch_policy_scores.append(policy_score)
            batch_gain_targets.append(risk_adjusted_gain_target.view(()))
            gain_class = torch.where(
                gain_target.view(()) > 0.0,
                torch.tensor(2, dtype=torch.long, device=device),
                torch.where(
                    gain_target.view(()) < 0.0,
                    torch.tensor(0, dtype=torch.long, device=device),
                    torch.tensor(1, dtype=torch.long, device=device),
                ),
            )
            batch_gain_classes.append(gain_class)
            batch_sample_weights.append(class_weights[int(action_label.item())] * risk_weight.view(()))

            pred_action = int(action_logits.argmax(dim=-1).item())
            pred_gain_sign = 0 if float(gain_pred.item()) == 0.0 else (1 if float(gain_pred.item()) > 0.0 else -1)
            pred_policy_sign = 0 if float(policy_score.item()) == 0.0 else (1 if float(policy_score.item()) > 0.0 else -1)
            tgt_gain = int(sample["gt_gain"])
            tgt_sign = 0 if tgt_gain == 0 else (1 if tgt_gain > 0 else -1)

            totals["samples"] += 1.0
            totals["loss"] += float(loss.detach().item())
            totals["action_loss"] += float(action_loss.detach().item())
            totals["gain_loss"] += float(gain_loss.detach().item())
            totals["policy_loss"] += float(policy_loss.detach().item())
            totals["rewrite_correct"] += float(int(pred_action == 0 and int(action_label.item()) == 0))
            totals["defer_correct"] += float(int(pred_action == 1 and int(action_label.item()) == 1))
            totals["reject_correct"] += float(int(pred_action == 2 and int(action_label.item()) == 2))
            totals["pred_rewrite"] += float(int(pred_action == 0))
            totals["pred_defer"] += float(int(pred_action == 1))
            totals["pred_reject"] += float(int(pred_action == 2))
            totals["tgt_rewrite"] += float(int(action_label.item()) == 0)
            totals["tgt_defer"] += float(int(action_label.item()) == 1)
            totals["tgt_reject"] += float(int(action_label.item()) == 2)
            totals["gain_abs_err"] += abs(float(gain_pred.item()) - float(gain_target.item()))
            totals["gain_sign_correct"] += float(int(pred_gain_sign == tgt_sign))
            totals["policy_abs_err"] += abs(float(policy_score.item()) - float(gain_target.item()))
            totals["policy_sign_correct"] += float(int(pred_policy_sign == tgt_sign))
            if isinstance(route_logits, torch.Tensor) and route_logits.numel() >= len(ACTION_NAMES):
                pred_route = int(route_logits.argmax(dim=-1).item())
                totals["route_correct"] += float(int(pred_route == int(route_label.item())))
            if isinstance(gate_logit, torch.Tensor) and gate_logit.numel() > 0:
                pred_gate = int(torch.sigmoid(gate_logit.view(())).item() >= 0.5)
                totals["gate_correct"] += float(int(pred_gate == int(gate_label.item())))

        if batch_policy_scores and float(rank_loss_weight) > 0.0:
            policy_scores = torch.stack(batch_policy_scores).view(-1)
            gain_targets = torch.stack(batch_gain_targets).view(-1)
            sample_weights = torch.stack(batch_sample_weights).view(-1)
            rank_loss = _pairwise_rank_loss(
                policy_scores,
                gain_targets > 0.0,
                sample_weights,
                margin=float(rank_margin),
            )
            totals["rank_loss"] += float(rank_loss.detach().item())

        if batch_policy_scores and float(ordinal_rank_loss_weight) > 0.0:
            policy_scores = torch.stack(batch_policy_scores).view(-1)
            gain_classes = torch.stack(batch_gain_classes).view(-1)
            sample_weights = torch.stack(batch_sample_weights).view(-1)
            ordinal_rank_loss = _ordered_pairwise_rank_loss(
                policy_scores,
                gain_classes,
                sample_weights,
                margin=float(ordinal_rank_margin),
            )
            totals["ordinal_rank_loss"] += float(ordinal_rank_loss.detach().item())

        if batch_router_probs and float(router_balance_loss_weight) > 0.0:
            router_probs = torch.stack(batch_router_probs, dim=0).mean(dim=0)
            balance_loss = torch.sum(router_probs * torch.log(torch.clamp(router_probs * float(router_probs.numel()), min=1e-8)))
            totals["router_balance_loss"] += float(balance_loss.detach().item())
            totals["router_batches"] += 1.0
        elif batch_router_probs:
            totals["router_batches"] += 1.0

        if losses and optimizer is not None:
            batch_loss = torch.stack(losses).mean()
            if balance_loss is not None and float(router_balance_loss_weight) > 0.0:
                batch_loss = batch_loss + float(router_balance_loss_weight) * balance_loss
            if batch_policy_scores and float(rank_loss_weight) > 0.0:
                batch_loss = batch_loss + float(rank_loss_weight) * rank_loss
            if batch_policy_scores and float(ordinal_rank_loss_weight) > 0.0:
                batch_loss = batch_loss + float(ordinal_rank_loss_weight) * ordinal_rank_loss
            batch_loss.backward()
            optimizer.step()

    denom = max(totals["samples"], 1.0)
    pred_rewrite = max(totals["pred_rewrite"], 1.0)
    pred_defer = max(totals["pred_defer"], 1.0)
    pred_reject = max(totals["pred_reject"], 1.0)
    tgt_rewrite = max(totals["tgt_rewrite"], 1.0)
    tgt_defer = max(totals["tgt_defer"], 1.0)
    tgt_reject = max(totals["tgt_reject"], 1.0)
    router_batches = max(totals["router_batches"], 1.0)
    return {
        "loss": totals["loss"] / denom,
        "action_loss": totals["action_loss"] / denom,
        "gain_loss": totals["gain_loss"] / denom,
        "policy_loss": totals["policy_loss"] / denom,
        "rank_loss": totals["rank_loss"] / denom,
        "ordinal_rank_loss": totals["ordinal_rank_loss"] / denom,
        "router_balance_loss": totals["router_balance_loss"] / router_batches,
        "router_entropy": totals["router_entropy"] / denom,
        "router_top1": totals["router_top1"] / denom,
        "router_margin": totals["router_margin"] / denom,
        "expert_aux_loss": totals["expert_aux_loss"] / denom,
        "route_loss": totals["route_loss"] / denom,
        "gate_loss": totals["gate_loss"] / denom,
        "route_acc": totals["route_correct"] / denom,
        "gate_acc": totals["gate_correct"] / denom,
        "action_acc": (
            totals["rewrite_correct"] + totals["defer_correct"] + totals["reject_correct"]
        )
        / denom,
        "rewrite_precision": totals["rewrite_correct"] / pred_rewrite,
        "rewrite_recall": totals["rewrite_correct"] / tgt_rewrite,
        "defer_precision": totals["defer_correct"] / pred_defer,
        "defer_recall": totals["defer_correct"] / tgt_defer,
        "reject_precision": totals["reject_correct"] / pred_reject,
        "reject_recall": totals["reject_correct"] / tgt_reject,
        "gain_mae": totals["gain_abs_err"] / denom,
        "gain_sign_acc": totals["gain_sign_correct"] / denom,
        "policy_mae": totals["policy_abs_err"] / denom,
        "policy_sign_acc": totals["policy_sign_correct"] / denom,
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
        "exp_name": "graph_assoc_commit_policy",
        "data_jsonl": str(data_jsonl),
        "source_manifest": str(args.source_manifest or ""),
        "dataset_tag": str(args.dataset_tag or ""),
        "feature_version": str(args.feature_version or ""),
        "model_arch": str(args.model_arch or MODEL_ARCH_SET_SLOT),
        "checkpoint": "",
        "init_checkpoint": str(args.init_checkpoint or ""),
        "train_examples": "",
        "val_examples": "",
        "train_sequences": "",
        "val_sequences": "",
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "policy_hidden_dim": int(args.policy_hidden_dim),
        "token_dim": int(args.token_dim),
        "num_slots": int(args.num_slots),
        "num_encoder_layers": int(args.num_encoder_layers),
        "num_heads": int(args.num_heads),
        "num_experts": int(args.num_experts),
        "router_hidden_dim": int(args.router_hidden_dim),
        "cluster_dim": "",
        "gate_hidden_dim": int(args.gate_hidden_dim),
        "router_temperature": float(args.router_temperature),
        "policy_score_mode": "",
        "train_selection_head_only": int(bool(args.train_selection_head_only)),
        "dropout": float(args.dropout),
        "action_loss_weight": float(args.action_loss_weight),
        "gain_loss_weight": float(args.gain_loss_weight),
        "policy_loss_weight": float(args.policy_loss_weight),
        "router_balance_loss_weight": float(args.router_balance_loss_weight),
        "expert_aux_loss_weight": float(args.expert_aux_loss_weight),
        "route_loss_weight": float(args.route_loss_weight),
        "gate_loss_weight": float(args.gate_loss_weight),
        "rank_loss_weight": float(args.rank_loss_weight),
        "rank_margin": float(args.rank_margin),
        "ordinal_rank_loss_weight": float(args.ordinal_rank_loss_weight),
        "ordinal_rank_margin": float(args.ordinal_rank_margin),
        "risk_weight_power": float(args.risk_weight_power),
        "risk_cost_scale": float(args.risk_cost_scale),
        "risk_utility_scale": float(args.risk_utility_scale),
        "risk_negative_boost": float(args.risk_negative_boost),
        "risk_neutral_boost": float(args.risk_neutral_boost),
        "risk_positive_discount": float(args.risk_positive_discount),
        "risk_target_cost_weight": float(args.risk_target_cost_weight),
        "risk_target_suppressed_weight": float(args.risk_target_suppressed_weight),
        "risk_target_recent_owner_weight": float(args.risk_target_recent_owner_weight),
        "risk_target_introduced_weight": float(args.risk_target_introduced_weight),
        "selection_metric": str(args.selection_metric),
        "use_balanced_sampler": int(bool(args.use_balanced_sampler)),
        "class_weight_rewrite": "",
        "class_weight_defer": "",
        "class_weight_reject": "",
        "best_metric_name": "",
        "best_epoch": "",
        "best_metric": "",
        "val_action_acc": "",
        "val_action_macro_f1": "",
        "val_policy_loss": "",
        "val_policy_sign_acc": "",
        "val_rank_loss": "",
        "val_ordinal_rank_loss": "",
        "train_router_balance_loss": "",
        "train_router_entropy": "",
        "train_router_top1": "",
        "train_router_margin": "",
        "train_expert_aux_loss": "",
        "train_route_loss": "",
        "train_gate_loss": "",
        "train_route_acc": "",
        "train_gate_acc": "",
        "val_router_balance_loss": "",
        "val_router_entropy": "",
        "val_router_top1": "",
        "val_router_margin": "",
        "val_expert_aux_loss": "",
        "val_route_loss": "",
        "val_gate_loss": "",
        "val_route_acc": "",
        "val_gate_acc": "",
        "val_rewrite_precision": "",
        "val_rewrite_recall": "",
        "val_defer_precision": "",
        "val_defer_recall": "",
        "val_reject_precision": "",
        "val_reject_recall": "",
        "val_gain_mae": "",
        "val_gain_sign_acc": "",
        "status": "running",
        "error": "",
    }
    fieldnames = list(running_row.keys())
    _write_single_row_csv(result_csv, fieldnames, running_row)
    _write_single_row_csv(summary_csv, fieldnames, running_row)
    metrics_jsonl.write_text("", encoding="utf-8")

    try:
        examples = _load_examples(data_jsonl)
        examples = _attach_risk_targets(
            examples,
            cost_scale=float(args.risk_cost_scale),
            utility_scale=float(args.risk_utility_scale),
            power=float(args.risk_weight_power),
            negative_boost=float(args.risk_negative_boost),
            neutral_boost=float(args.risk_neutral_boost),
            positive_discount=float(args.risk_positive_discount),
            target_cost_weight=float(args.risk_target_cost_weight),
            target_suppressed_weight=float(args.risk_target_suppressed_weight),
            target_recent_owner_weight=float(args.risk_target_recent_owner_weight),
            target_introduced_weight=float(args.risk_target_introduced_weight),
        )
        train_rows, val_rows, train_sequences, val_sequences, split_mode = _split_examples(
            examples=examples,
            train_sequences_arg=str(args.train_sequences or ""),
            val_sequences_arg=str(args.val_sequences or ""),
            strict_sequence_split=bool(args.strict_sequence_split),
            min_val_examples=int(args.min_val_examples),
        )

        train_counts = torch.zeros((len(ACTION_NAMES),), dtype=torch.float32)
        for row in train_rows:
            label = int(row.get("action_label", 1))
            if 0 <= label < len(ACTION_NAMES):
                train_counts[label] += 1.0
        cluster_dim = int(len(train_rows[0].get("cluster_features", []))) if train_rows else 0

        train_loader = DataLoader(
            CommitPolicyDataset(train_rows),
            batch_size=max(int(args.batch_size), 1),
            shuffle=not bool(args.use_balanced_sampler),
            sampler=(
                WeightedRandomSampler(
                    weights=[
                        1.0 / max(
                            float(train_counts[int(row.get("action_label", 1))].item()),
                            1.0,
                        )
                        for row in train_rows
                    ],
                    num_samples=len(train_rows),
                    replacement=True,
                )
                if bool(args.use_balanced_sampler) and len(train_rows) > 0
                else None
            ),
            collate_fn=_collate,
        )
        val_loader = DataLoader(
            CommitPolicyDataset(val_rows),
            batch_size=max(int(args.batch_size), 1),
            shuffle=False,
            collate_fn=_collate,
        )

        total = float(train_counts.sum().item()) if train_counts.sum().item() > 0 else 1.0
        class_weights = torch.tensor(
            [total / max(float(count.item()) * len(ACTION_NAMES), 1.0) for count in train_counts],
            dtype=torch.float32,
        )
        class_weights = class_weights / max(float(class_weights.mean().item()), 1e-8)

        device = torch.device(args.device)
        model_kwargs = {
            "hidden_dim": int(args.hidden_dim),
            "policy_hidden_dim": int(args.policy_hidden_dim),
            "dropout": float(args.dropout),
            "token_dim": int(args.token_dim),
            "num_slots": int(args.num_slots),
            "num_encoder_layers": int(args.num_encoder_layers),
            "num_heads": int(args.num_heads),
            "cluster_dim": int(cluster_dim),
        }
        if str(args.model_arch) == MODEL_ARCH_ROUTED_MOE:
            model_cls = GraphAssocCommitRoutedSetSlotPolicy
            running_row["policy_score_mode"] = str(args.policy_score_mode)
            model_kwargs.update(
                {
                    "num_experts": int(args.num_experts),
                    "router_hidden_dim": int(args.router_hidden_dim),
                    "cluster_dim": int(cluster_dim),
                    "router_temperature": float(args.router_temperature),
                    "policy_score_mode": str(args.policy_score_mode),
                }
            )
        elif str(args.model_arch) == MODEL_ARCH_HIER_ROUTE:
            model_cls = GraphAssocCommitHierarchicalRoutePolicy
            running_row["policy_score_mode"] = str(args.policy_score_mode)
            model_kwargs.update(
                {
                    "num_experts": int(args.num_experts),
                    "router_hidden_dim": int(args.router_hidden_dim),
                    "gate_hidden_dim": int(args.gate_hidden_dim),
                    "cluster_dim": int(cluster_dim),
                    "route_temperature": float(args.router_temperature),
                    "policy_score_mode": str(args.policy_score_mode),
                }
            )
        else:
            model_cls = GraphAssocCommitPolicy
        model = model_cls(**model_kwargs).to(device)
        if str(args.init_checkpoint or "").strip():
            init_ckpt = Path(str(args.init_checkpoint)).expanduser().resolve()
            if not init_ckpt.is_file():
                raise FileNotFoundError(f"Missing init checkpoint: {init_ckpt}")
            init_payload = torch.load(init_ckpt, map_location=device)
            init_state = init_payload.get("model_state", init_payload)
            _missing, _unexpected = _load_state_dict_shape_compatible(model, init_state)
        if bool(args.train_selection_head_only):
            for name, param in model.named_parameters():
                param.requires_grad = bool(name.startswith("selection_"))
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        class_weights = class_weights.to(device)

        best_metric: tuple[float, float, float, float, float] | None = None
        best_row: dict[str, Any] | None = None

        with metrics_jsonl.open("a", encoding="utf-8") as metrics_fp:
            for epoch in range(1, int(args.epochs) + 1):
                train_metrics = _run_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    class_weights=class_weights,
                    action_loss_weight=float(args.action_loss_weight),
                    gain_loss_weight=float(args.gain_loss_weight),
                    policy_loss_weight=float(args.policy_loss_weight),
                    router_balance_loss_weight=float(args.router_balance_loss_weight),
                    expert_aux_loss_weight=float(args.expert_aux_loss_weight),
                    route_loss_weight=float(args.route_loss_weight),
                    gate_loss_weight=float(args.gate_loss_weight),
                    rank_loss_weight=float(args.rank_loss_weight),
                    rank_margin=float(args.rank_margin),
                    ordinal_rank_loss_weight=float(args.ordinal_rank_loss_weight),
                    ordinal_rank_margin=float(args.ordinal_rank_margin),
                )
                with torch.inference_mode():
                    val_metrics = _run_epoch(
                        model=model,
                        loader=val_loader,
                        optimizer=None,
                        device=device,
                        class_weights=class_weights,
                        action_loss_weight=float(args.action_loss_weight),
                        gain_loss_weight=float(args.gain_loss_weight),
                        policy_loss_weight=float(args.policy_loss_weight),
                        router_balance_loss_weight=float(args.router_balance_loss_weight),
                        expert_aux_loss_weight=float(args.expert_aux_loss_weight),
                        route_loss_weight=float(args.route_loss_weight),
                        gate_loss_weight=float(args.gate_loss_weight),
                        rank_loss_weight=float(args.rank_loss_weight),
                        rank_margin=float(args.rank_margin),
                        ordinal_rank_loss_weight=float(args.ordinal_rank_loss_weight),
                        ordinal_rank_margin=float(args.ordinal_rank_margin),
                    )

                val_confusion = torch.zeros((len(ACTION_NAMES), len(ACTION_NAMES)), dtype=torch.float32)
                model.eval()
                with torch.inference_mode():
                    for batch in val_loader:
                        for sample in batch:
                            outputs = model(
                                det_features=sample["det_features"].to(device),
                                track_features=sample["track_features"].to(device),
                                edge_features=sample["edge_features"].to(device),
                                edge_det_index=sample["edge_det_index"].to(device),
                                edge_track_index=sample["edge_track_index"].to(device),
                                cluster_features=sample["cluster_features"].to(device),
                            )
                            pred_action = outputs["action_logits"].argmax(dim=-1).view(-1).cpu()
                            val_confusion += _confusion_counts(pred_action, sample["action_label"].view(1), len(ACTION_NAMES))
                val_macro_f1, per_class = _macro_f1_from_confusion(val_confusion)
                metrics_row = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_action_loss": train_metrics["action_loss"],
                    "train_gain_loss": train_metrics["gain_loss"],
                    "train_rank_loss": train_metrics["rank_loss"],
                    "train_ordinal_rank_loss": train_metrics["ordinal_rank_loss"],
                    "train_router_balance_loss": train_metrics["router_balance_loss"],
                    "train_router_entropy": train_metrics["router_entropy"],
                    "train_router_top1": train_metrics["router_top1"],
                    "train_router_margin": train_metrics["router_margin"],
                    "train_expert_aux_loss": train_metrics["expert_aux_loss"],
                    "train_action_acc": train_metrics["action_acc"],
                    "val_loss": val_metrics["loss"],
                    "val_action_loss": val_metrics["action_loss"],
                    "val_gain_loss": val_metrics["gain_loss"],
                    "val_rank_loss": val_metrics["rank_loss"],
                    "val_ordinal_rank_loss": val_metrics["ordinal_rank_loss"],
                    "val_router_balance_loss": val_metrics["router_balance_loss"],
                    "val_router_entropy": val_metrics["router_entropy"],
                    "val_router_top1": val_metrics["router_top1"],
                    "val_router_margin": val_metrics["router_margin"],
                    "val_expert_aux_loss": val_metrics["expert_aux_loss"],
                    "val_action_acc": val_metrics["action_acc"],
                    "val_action_macro_f1": val_macro_f1,
                    "val_policy_loss": val_metrics["policy_loss"],
                    "val_policy_sign_acc": val_metrics["policy_sign_acc"],
                    "val_rewrite_precision": per_class[0]["precision"],
                    "val_rewrite_recall": per_class[0]["recall"],
                    "val_defer_precision": per_class[1]["precision"],
                    "val_defer_recall": per_class[1]["recall"],
                    "val_reject_precision": per_class[2]["precision"],
                    "val_reject_recall": per_class[2]["recall"],
                    "val_gain_mae": val_metrics["gain_mae"],
                    "val_gain_sign_acc": val_metrics["gain_sign_acc"],
                    "split_mode": split_mode,
                    "train_examples": len(train_rows),
                    "val_examples": len(val_rows),
                }
                metrics_fp.write(json.dumps(metrics_row, ensure_ascii=False) + "\n")
                metrics_fp.flush()

                if str(args.selection_metric) == "action_macro_f1":
                    selection_key = (
                        float(val_macro_f1),
                        float(val_metrics["gain_sign_acc"]),
                        float(val_metrics["policy_sign_acc"]),
                        float(val_metrics["action_acc"]),
                        float(-val_metrics["policy_mae"]),
                    )
                    best_metric_name = "val_action_macro_f1"
                    best_metric_value = float(val_macro_f1)
                elif str(args.selection_metric) == "gain_sign_acc":
                    selection_key = (
                        float(val_metrics["gain_sign_acc"]),
                        float(val_macro_f1),
                        float(val_metrics["policy_sign_acc"]),
                        float(val_metrics["action_acc"]),
                        float(-val_metrics["gain_mae"]),
                    )
                    best_metric_name = "val_gain_sign_acc"
                    best_metric_value = float(val_metrics["gain_sign_acc"])
                else:
                    selection_key = (
                        float(val_metrics["policy_sign_acc"]),
                        float(val_macro_f1),
                        float(val_metrics["gain_sign_acc"]),
                        float(val_metrics["action_acc"]),
                        float(-val_metrics["policy_mae"]),
                    )
                    best_metric_name = "val_policy_sign_acc"
                    best_metric_value = float(val_metrics["policy_sign_acc"])
                if best_metric is None or selection_key > best_metric:
                    best_metric = selection_key
                    best_row = {
                        "exp_name": "graph_assoc_commit_policy",
                        "data_jsonl": str(data_jsonl),
                        "source_manifest": str(args.source_manifest or ""),
                        "dataset_tag": str(args.dataset_tag or ""),
                        "feature_version": str(args.feature_version or ""),
                        "model_arch": str(args.model_arch or MODEL_ARCH_SET_SLOT),
                        "checkpoint": str(best_ckpt),
                        "init_checkpoint": str(args.init_checkpoint or ""),
                        "train_examples": int(len(train_rows)),
                        "val_examples": int(len(val_rows)),
                        "train_sequences": ",".join(train_sequences),
                        "val_sequences": ",".join(val_sequences),
                        "epochs": int(args.epochs),
                        "batch_size": int(args.batch_size),
                        "hidden_dim": int(args.hidden_dim),
                        "policy_hidden_dim": int(args.policy_hidden_dim),
                        "token_dim": int(args.token_dim),
                        "num_slots": int(args.num_slots),
                        "num_encoder_layers": int(args.num_encoder_layers),
                        "num_heads": int(args.num_heads),
                        "num_experts": int(args.num_experts),
                        "router_hidden_dim": int(args.router_hidden_dim),
                        "cluster_dim": int(cluster_dim),
                        "router_temperature": float(args.router_temperature),
                        "policy_score_mode": str(args.policy_score_mode),
                        "dropout": float(args.dropout),
                        "action_loss_weight": float(args.action_loss_weight),
                        "gain_loss_weight": float(args.gain_loss_weight),
                        "policy_loss_weight": float(args.policy_loss_weight),
                        "router_balance_loss_weight": float(args.router_balance_loss_weight),
                        "expert_aux_loss_weight": float(args.expert_aux_loss_weight),
                        "rank_loss_weight": float(args.rank_loss_weight),
                        "rank_margin": float(args.rank_margin),
                        "ordinal_rank_loss_weight": float(args.ordinal_rank_loss_weight),
                        "ordinal_rank_margin": float(args.ordinal_rank_margin),
                        "risk_weight_power": float(args.risk_weight_power),
                        "risk_cost_scale": float(args.risk_cost_scale),
                        "risk_utility_scale": float(args.risk_utility_scale),
                        "risk_negative_boost": float(args.risk_negative_boost),
                        "risk_neutral_boost": float(args.risk_neutral_boost),
                        "risk_positive_discount": float(args.risk_positive_discount),
                        "risk_target_cost_weight": float(args.risk_target_cost_weight),
                        "risk_target_suppressed_weight": float(args.risk_target_suppressed_weight),
                        "risk_target_recent_owner_weight": float(args.risk_target_recent_owner_weight),
                        "risk_target_introduced_weight": float(args.risk_target_introduced_weight),
                        "selection_metric": str(args.selection_metric),
                        "use_balanced_sampler": int(bool(args.use_balanced_sampler)),
                        "class_weight_rewrite": float(class_weights[0].item()),
                        "class_weight_defer": float(class_weights[1].item()),
                        "class_weight_reject": float(class_weights[2].item()),
                        "best_metric_name": str(best_metric_name),
                        "best_epoch": int(epoch),
                        "best_metric": float(best_metric_value),
                        "val_action_acc": float(val_metrics["action_acc"]),
                        "val_action_macro_f1": float(val_macro_f1),
                        "val_policy_loss": float(val_metrics["policy_loss"]),
                        "val_policy_sign_acc": float(val_metrics["policy_sign_acc"]),
                        "val_rank_loss": float(val_metrics["rank_loss"]),
                        "val_ordinal_rank_loss": float(val_metrics["ordinal_rank_loss"]),
                        "train_router_balance_loss": float(train_metrics["router_balance_loss"]),
                        "train_router_entropy": float(train_metrics["router_entropy"]),
                        "train_router_top1": float(train_metrics["router_top1"]),
                        "train_router_margin": float(train_metrics["router_margin"]),
                        "train_expert_aux_loss": float(train_metrics["expert_aux_loss"]),
                        "val_router_balance_loss": float(val_metrics["router_balance_loss"]),
                        "val_router_entropy": float(val_metrics["router_entropy"]),
                        "val_router_top1": float(val_metrics["router_top1"]),
                        "val_router_margin": float(val_metrics["router_margin"]),
                        "val_expert_aux_loss": float(val_metrics["expert_aux_loss"]),
                        "val_rewrite_precision": float(per_class[0]["precision"]),
                        "val_rewrite_recall": float(per_class[0]["recall"]),
                        "val_defer_precision": float(per_class[1]["precision"]),
                        "val_defer_recall": float(per_class[1]["recall"]),
                        "val_reject_precision": float(per_class[2]["precision"]),
                        "val_reject_recall": float(per_class[2]["recall"]),
                        "val_gain_mae": float(val_metrics["gain_mae"]),
                        "val_gain_sign_acc": float(val_metrics["gain_sign_acc"]),
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
                            val_action_macro_f1=float(val_macro_f1),
                            val_policy_sign_acc=float(val_metrics["policy_sign_acc"]),
                            val_rank_loss=float(val_metrics["rank_loss"]),
                            val_ordinal_rank_loss=float(val_metrics["ordinal_rank_loss"]),
                            train_router_balance_loss=float(train_metrics["router_balance_loss"]),
                            train_router_entropy=float(train_metrics["router_entropy"]),
                            train_router_top1=float(train_metrics["router_top1"]),
                            train_router_margin=float(train_metrics["router_margin"]),
                            train_expert_aux_loss=float(train_metrics["expert_aux_loss"]),
                            val_router_balance_loss=float(val_metrics["router_balance_loss"]),
                            val_router_entropy=float(val_metrics["router_entropy"]),
                            val_router_top1=float(val_metrics["router_top1"]),
                            val_router_margin=float(val_metrics["router_margin"]),
                            val_expert_aux_loss=float(val_metrics["expert_aux_loss"]),
                            best_metric=float(best_metric_value),
                            selection_metric=str(args.selection_metric),
                            rank_loss_weight=float(args.rank_loss_weight),
                            rank_margin=float(args.rank_margin),
                            ordinal_rank_loss_weight=float(args.ordinal_rank_loss_weight),
                            ordinal_rank_margin=float(args.ordinal_rank_margin),
                            risk_weight_power=float(args.risk_weight_power),
                            risk_cost_scale=float(args.risk_cost_scale),
                            risk_utility_scale=float(args.risk_utility_scale),
                            risk_negative_boost=float(args.risk_negative_boost),
                            risk_neutral_boost=float(args.risk_neutral_boost),
                            risk_positive_discount=float(args.risk_positive_discount),
                            risk_target_cost_weight=float(args.risk_target_cost_weight),
                            risk_target_suppressed_weight=float(args.risk_target_suppressed_weight),
                            risk_target_recent_owner_weight=float(args.risk_target_recent_owner_weight),
                            risk_target_introduced_weight=float(args.risk_target_introduced_weight),
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
                            "model_arch": str(args.model_arch or MODEL_ARCH_SET_SLOT),
                            "best_metric_name": str({
                                "action_macro_f1": "val_action_macro_f1",
                                "gain_sign_acc": "val_gain_sign_acc",
                                "policy_sign_acc": "val_policy_sign_acc",
                    }.get(str(args.selection_metric), "val_policy_sign_acc")),
                    "status": "failed",
                    "error": "no_best_epoch",
                }
            )

        _write_single_row_csv(result_csv, fieldnames, best_row)
        _write_single_row_csv(summary_csv, fieldnames, best_row)
        _append_registry(
            args,
            summary_csv,
            best_ckpt if best_ckpt.is_file() else None,
            "success" if str(best_row.get("status", "")) == "ok" else "failed",
            "graph-assoc action policy training",
        )
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
        _append_registry(args, summary_csv, None, "failed", f"graph-assoc action policy training failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
