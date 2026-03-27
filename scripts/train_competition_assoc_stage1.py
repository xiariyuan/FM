#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.competition_assoc import (
    CANDIDATE_FEATURES,
    OBSERVED_GROUP_FEATURES,
    CompetitionAssociationController,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a first-stage competition association controller.")
    ap.add_argument("--cases-csv", required=True)
    ap.add_argument("--group-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--candidate-loss-weight", type=float, default=0.5)
    ap.add_argument("--bridge-loss-weight", type=float, default=0.2)
    ap.add_argument("--val-mod", type=int, default=5)
    ap.add_argument("--val-rem", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


@dataclass
class Sample:
    group_id: str
    seq: str
    frame: int
    group_features: np.ndarray
    candidate_features: np.ndarray
    valid_mask: np.ndarray
    action_target: int
    candidate_target: int
    bridge_target: float


class CompetitionAssocDataset(Dataset):
    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        return {
            "group_id": s.group_id,
            "seq": s.seq,
            "frame": s.frame,
            "group_features": torch.from_numpy(s.group_features),
            "candidate_features": torch.from_numpy(s.candidate_features),
            "valid_mask": torch.from_numpy(s.valid_mask),
            "action_target": torch.tensor(s.action_target, dtype=torch.long),
            "candidate_target": torch.tensor(s.candidate_target, dtype=torch.long),
            "bridge_target": torch.tensor(s.bridge_target, dtype=torch.float32),
        }


def _write_single_row_csv(path: Path, fieldnames: list[str], row: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_case_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {str(row["group_id"]): dict(row) for row in reader}


def _safe_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def _safe_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except Exception:
        return int(default)


def _build_samples(case_rows: dict[str, dict[str, str]], group_jsonl: Path) -> tuple[list[Sample], list[Sample]]:
    train_samples: list[Sample] = []
    val_samples: list[Sample] = []

    with group_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            group = json.loads(line)
            group_id = str(group["group_id"])
            case = case_rows.get(group_id)
            if case is None:
                continue

            candidates = sorted(group.get("candidates", []), key=lambda x: int(x.get("track_rank", 0)))
            k = len(candidates)
            if k == 0:
                continue
            group_feat = np.asarray([_safe_float(case, name) for name in OBSERVED_GROUP_FEATURES], dtype=np.float32)
            cand_feat = np.zeros((k, len(CANDIDATE_FEATURES)), dtype=np.float32)
            valid_mask = np.zeros((k,), dtype=np.bool_)
            for idx, cand in enumerate(candidates):
                rank = max(int(cand.get("track_rank", idx + 1)), 1)
                row = {
                    "base_score": float(cand.get("base_score", 0.0)),
                    "refined_score": float(cand.get("refined_score", 0.0)),
                    "motion_score": float(cand.get("motion_score", 0.0)),
                    "track_gap": float(cand.get("track_gap", 0.0)),
                    "track_hist_len": float(cand.get("track_hist_len", 0.0)),
                    "track_rank_frac": float(rank) / float(max(k, 1)),
                }
                cand_feat[idx] = np.asarray([_safe_float(row, name) for name in CANDIDATE_FEATURES], dtype=np.float32)
                valid_mask[idx] = bool(int(cand.get("valid_train_row", 1)) > 0)

            sample = Sample(
                group_id=group_id,
                seq=str(case["seq"]),
                frame=_safe_int(case, "frame"),
                group_features=group_feat,
                candidate_features=cand_feat,
                valid_mask=valid_mask,
                action_target=_safe_int(case, "action_target_id"),
                candidate_target=max(_safe_int(case, "target_candidate_rank", -1) - 1, -1),
                bridge_target=float(_safe_int(case, "continuity_bridge")),
            )
            if sample.frame % 5 == 0:
                val_samples.append(sample)
            else:
                train_samples.append(sample)

    return train_samples, val_samples


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_k = max(item["candidate_features"].shape[0] for item in batch)
    cand_dim = batch[0]["candidate_features"].shape[1]
    group_feats = torch.stack([item["group_features"] for item in batch], dim=0)
    cand_feats = torch.zeros((len(batch), max_k, cand_dim), dtype=torch.float32)
    valid_mask = torch.zeros((len(batch), max_k), dtype=torch.bool)
    candidate_target = torch.full((len(batch),), -1, dtype=torch.long)

    for i, item in enumerate(batch):
        k = item["candidate_features"].shape[0]
        cand_feats[i, :k] = item["candidate_features"]
        valid_mask[i, :k] = item["valid_mask"]
        tgt = int(item["candidate_target"].item())
        if tgt >= 0 and tgt < k:
            candidate_target[i] = tgt

    return {
        "group_id": [item["group_id"] for item in batch],
        "seq": [item["seq"] for item in batch],
        "frame": torch.tensor([int(item["frame"]) for item in batch], dtype=torch.long),
        "group_features": group_feats,
        "candidate_features": cand_feats,
        "valid_mask": valid_mask,
        "action_target": torch.stack([item["action_target"] for item in batch], dim=0),
        "candidate_target": candidate_target,
        "bridge_target": torch.stack([item["bridge_target"] for item in batch], dim=0),
    }


def _class_weights(samples: list[Sample]) -> torch.Tensor:
    counts = np.zeros((3,), dtype=np.float32)
    for s in samples:
        counts[int(s.action_target)] += 1.0
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _run_epoch(
    *,
    model: CompetitionAssociationController,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    action_weights: torch.Tensor,
    candidate_loss_weight: float,
    bridge_loss_weight: float,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    totals = {
        "loss": 0.0,
        "groups": 0.0,
        "action_correct": 0.0,
        "action_total": 0.0,
        "rerank_total": 0.0,
        "rerank_candidate_correct": 0.0,
        "rerank_action_correct": 0.0,
        "bridge_total": 0.0,
        "bridge_tp": 0.0,
        "bridge_pred_pos": 0.0,
        "bridge_gt_pos": 0.0,
        "base_rerank_top1_correct": 0.0,
    }

    for batch in loader:
        group_features = batch["group_features"].to(device=device, dtype=torch.float32)
        candidate_features = batch["candidate_features"].to(device=device, dtype=torch.float32)
        valid_mask = batch["valid_mask"].to(device=device)
        action_target = batch["action_target"].to(device=device)
        candidate_target = batch["candidate_target"].to(device=device)
        bridge_target = batch["bridge_target"].to(device=device)

        outputs = model(
            group_features=group_features,
            candidate_features=candidate_features,
            valid_mask=valid_mask,
        )
        tensor_checks = {
            "group_features": group_features,
            "candidate_features": candidate_features,
            "action_logits": outputs["action_logits"],
            "candidate_logits": outputs["candidate_logits"],
            "continuity_logit": outputs["continuity_logit"],
        }
        for name, tensor in tensor_checks.items():
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"non-finite tensor detected: {name}")
        action_loss = F.cross_entropy(outputs["action_logits"], action_target, weight=action_weights.to(device))
        rerank_mask = candidate_target >= 0
        if bool(rerank_mask.any()):
            candidate_loss = F.cross_entropy(outputs["candidate_logits"][rerank_mask], candidate_target[rerank_mask])
        else:
            candidate_loss = outputs["candidate_logits"].sum() * 0.0
        bridge_loss = F.binary_cross_entropy_with_logits(outputs["continuity_logit"], bridge_target)
        loss = action_loss + float(candidate_loss_weight) * candidate_loss + float(bridge_loss_weight) * bridge_loss
        if not torch.isfinite(loss):
            raise RuntimeError(
                "non-finite loss detected: "
                f"action_loss={float(action_loss.detach().cpu().item())} "
                f"candidate_loss={float(candidate_loss.detach().cpu().item())} "
                f"bridge_loss={float(bridge_loss.detach().cpu().item())}"
            )

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            action_pred = outputs["action_logits"].argmax(dim=-1)
            candidate_pred = outputs["candidate_logits"].argmax(dim=-1)
            bridge_prob = torch.sigmoid(outputs["continuity_logit"])
            bridge_pred = bridge_prob >= 0.5

            totals["loss"] += float(loss.item()) * float(group_features.shape[0])
            totals["groups"] += float(group_features.shape[0])
            totals["action_correct"] += float((action_pred == action_target).sum().item())
            totals["action_total"] += float(action_target.numel())

            if bool(rerank_mask.any()):
                totals["rerank_total"] += float(rerank_mask.sum().item())
                totals["rerank_candidate_correct"] += float((candidate_pred[rerank_mask] == candidate_target[rerank_mask]).sum().item())
                totals["rerank_action_correct"] += float((action_pred[rerank_mask] == action_target[rerank_mask]).sum().item())
                totals["base_rerank_top1_correct"] += float((candidate_target[rerank_mask] == 0).sum().item())

            totals["bridge_total"] += float(bridge_target.numel())
            totals["bridge_tp"] += float(((bridge_pred == 1) & (bridge_target == 1)).sum().item())
            totals["bridge_pred_pos"] += float((bridge_pred == 1).sum().item())
            totals["bridge_gt_pos"] += float((bridge_target == 1).sum().item())

    groups = max(totals["groups"], 1.0)
    action_total = max(totals["action_total"], 1.0)
    rerank_total = max(totals["rerank_total"], 1.0)
    bridge_pred = max(totals["bridge_pred_pos"], 1.0)
    bridge_gt = max(totals["bridge_gt_pos"], 1.0)
    bridge_precision = totals["bridge_tp"] / bridge_pred
    bridge_recall = totals["bridge_tp"] / bridge_gt
    bridge_f1 = 0.0
    if bridge_precision + bridge_recall > 0:
        bridge_f1 = 2.0 * bridge_precision * bridge_recall / (bridge_precision + bridge_recall)

    return {
        "loss": totals["loss"] / groups,
        "action_acc": totals["action_correct"] / action_total,
        "rerank_candidate_acc": totals["rerank_candidate_correct"] / rerank_total,
        "rerank_action_acc": totals["rerank_action_correct"] / rerank_total,
        "base_rerank_top1_acc": totals["base_rerank_top1_correct"] / rerank_total,
        "bridge_precision": bridge_precision,
        "bridge_recall": bridge_recall,
        "bridge_f1": bridge_f1,
        "groups": groups,
        "rerank_total": totals["rerank_total"],
    }


def main() -> int:
    args = parse_args()
    _set_seed(int(args.seed))

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result_csv = out_dir / "result.csv"
    summary_csv = out_dir / "summary.csv"
    metrics_jsonl = out_dir / "metrics.jsonl"

    fieldnames = [
        "exp_name",
        "cases_csv",
        "group_jsonl",
        "out_dir",
        "epochs",
        "batch_size",
        "best_epoch",
        "best_score",
        "val_action_acc",
        "val_rerank_candidate_acc",
        "val_rerank_action_acc",
        "val_base_rerank_top1_acc",
        "val_bridge_f1",
        "train_groups",
        "val_groups",
        "val_rerank_total",
        "status",
    ]
    running_row = {
        "exp_name": "competition_assoc_stage1",
        "cases_csv": str(Path(args.cases_csv).resolve()),
        "group_jsonl": str(Path(args.group_jsonl).resolve()),
        "out_dir": str(out_dir),
        "epochs": str(args.epochs),
        "batch_size": str(args.batch_size),
        "best_epoch": "",
        "best_score": "",
        "val_action_acc": "",
        "val_rerank_candidate_acc": "",
        "val_rerank_action_acc": "",
        "val_base_rerank_top1_acc": "",
        "val_bridge_f1": "",
        "train_groups": "",
        "val_groups": "",
        "val_rerank_total": "",
        "status": "running",
    }
    for path in (result_csv, summary_csv):
        _write_single_row_csv(path, fieldnames, running_row)

    train_samples: list[Sample] = []
    val_samples: list[Sample] = []
    try:
        case_rows = _load_case_rows(Path(args.cases_csv).resolve())
        train_samples, val_samples = _build_samples(case_rows, Path(args.group_jsonl).resolve())
        if not train_samples or not val_samples:
            raise RuntimeError(f"Need non-empty train/val samples, got train={len(train_samples)} val={len(val_samples)}")

        train_ds = CompetitionAssocDataset(train_samples)
        val_ds = CompetitionAssocDataset(val_samples)
        train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, collate_fn=_collate)
        val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=_collate)

        device = torch.device(args.device)
        model = CompetitionAssociationController(
            group_dim=len(OBSERVED_GROUP_FEATURES),
            candidate_dim=len(CANDIDATE_FEATURES),
            hidden_dim=int(args.hidden_dim),
            dropout=float(args.dropout),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        action_weights = _class_weights(train_samples)

        best_score = -1e9
        best_epoch = -1
        best_metrics: dict[str, float] = {}
        best_path = out_dir / "best.pt"

        with metrics_jsonl.open("w", encoding="utf-8") as metrics_fp:
            for epoch in range(int(args.epochs)):
                train_metrics = _run_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    action_weights=action_weights,
                    candidate_loss_weight=float(args.candidate_loss_weight),
                    bridge_loss_weight=float(args.bridge_loss_weight),
                )
                val_metrics = _run_epoch(
                    model=model,
                    loader=val_loader,
                    optimizer=None,
                    device=device,
                    action_weights=action_weights,
                    candidate_loss_weight=float(args.candidate_loss_weight),
                    bridge_loss_weight=float(args.bridge_loss_weight),
                )
                score = float(val_metrics["rerank_candidate_acc"]) + 0.25 * float(val_metrics["action_acc"]) + 0.10 * float(val_metrics["bridge_f1"])
                row = {
                    "epoch": epoch,
                    "score": score,
                    "train": train_metrics,
                    "val": val_metrics,
                }
                metrics_fp.write(json.dumps(row, sort_keys=True) + "\n")
                metrics_fp.flush()
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_metrics = dict(val_metrics)
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": {
                                "group_dim": len(OBSERVED_GROUP_FEATURES),
                                "candidate_dim": len(CANDIDATE_FEATURES),
                                "hidden_dim": int(args.hidden_dim),
                                "dropout": float(args.dropout),
                            },
                            "epoch": int(epoch),
                            "score": float(score),
                            "val_metrics": val_metrics,
                        },
                        best_path,
                    )

        final_row = dict(running_row)
        final_row.update(
            {
                "best_epoch": str(best_epoch),
                "best_score": f"{best_score:.6f}",
                "val_action_acc": f"{best_metrics.get('action_acc', 0.0):.6f}",
                "val_rerank_candidate_acc": f"{best_metrics.get('rerank_candidate_acc', 0.0):.6f}",
                "val_rerank_action_acc": f"{best_metrics.get('rerank_action_acc', 0.0):.6f}",
                "val_base_rerank_top1_acc": f"{best_metrics.get('base_rerank_top1_acc', 0.0):.6f}",
                "val_bridge_f1": f"{best_metrics.get('bridge_f1', 0.0):.6f}",
                "train_groups": str(len(train_samples)),
                "val_groups": str(len(val_samples)),
                "val_rerank_total": str(int(best_metrics.get('rerank_total', 0.0))),
                "status": "ok",
            }
        )
        for path in (result_csv, summary_csv):
            _write_single_row_csv(path, fieldnames, final_row)
        print(json.dumps(final_row, indent=2))
        return 0
    except Exception as exc:
        failed_row = dict(running_row)
        failed_row.update(
            {
                "train_groups": str(len(train_samples)) if train_samples else "",
                "val_groups": str(len(val_samples)) if val_samples else "",
                "status": "failed",
            }
        )
        for path in (result_csv, summary_csv):
            _write_single_row_csv(path, fieldnames, failed_row)
        print(f"[competition-stage1] failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
