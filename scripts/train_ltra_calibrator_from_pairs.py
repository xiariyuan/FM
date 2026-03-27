#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
BOT_ROOT = REPO_ROOT / "external/BoT-SORT-main"
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from tracker.laplace_assoc import LEARNED_FEATURE_NAMES  # noqa: E402


@dataclass
class Candidate:
    features: np.ndarray
    spatial: float
    laplace: float
    motion: float
    label: int


@dataclass
class Group:
    seq: str
    frame: int
    det_index: int
    assoc_stage: str
    candidates: List[Candidate]
    positive_index: int


def parse_args():
    parser = argparse.ArgumentParser(description="Train a grouped alpha/r calibrator from LTRA pair logs.")
    parser.add_argument("--pair-csv", nargs="+", required=True, help="One or more *_pairs.csv files")
    parser.add_argument("--out-npz", required=True, help="Output .npz for runtime calibrator")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--hidden-dim", type=int, default=16, help="Hidden dim (<=0 => linear)")
    parser.add_argument("--epochs", type=int, default=12, help="Epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Groups per batch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--min-history", type=int, default=3, help="Normalize history_len by this value")
    parser.add_argument("--assoc-stage", type=str, default="primary", help="Only use groups from this assoc stage")
    parser.add_argument("--train-seqs", nargs="*", default=None, help="Optional sequence prefixes for train split")
    parser.add_argument("--val-seqs", nargs="*", default=None, help="Optional sequence prefixes for val split")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Random validation fraction if --val-seqs is unset")
    parser.add_argument("--max-hard-negatives", type=int, default=4, help="Max hardest negatives per group")
    parser.add_argument("--max-random-negatives", type=int, default=2, help="Extra random negatives per group")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for candidate-set CE")
    parser.add_argument("--rank-margin", type=float, default=0.05, help="Margin for ranking loss")
    parser.add_argument("--trust-margin", type=float, default=0.03, help="Margin for trust supervision")
    parser.add_argument("--loss-bg-weight", type=float, default=0.25, help="Weight for background loss")
    parser.add_argument("--loss-rank-weight", type=float, default=0.25, help="Weight for ranking loss")
    parser.add_argument("--loss-trust-weight", type=float, default=0.10, help="Weight for trust auxiliary loss")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--patience", type=int, default=2, help="Early-stop patience on validation loss")
    parser.add_argument(
        "--allow-legacy-schema",
        action="store_true",
        help="Allow training from older pair-log CSVs that are missing current learned-LTRA columns",
    )
    return parser.parse_args()


def _to_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row[key])
    except Exception:
        return float(default)


def _to_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row[key]))
    except Exception:
        return int(default)


def _matches_prefix(seq_name: str, prefixes: Sequence[str] | None) -> bool:
    if not prefixes:
        return True
    return any(seq_name.startswith(prefix) for prefix in prefixes)


def extract_features(row: Dict[str, str], min_history: int) -> np.ndarray:
    spatial = _to_float(row, "spatial_sim", 0.0)
    laplace = _to_float(row, "laplace_sim", 0.0)
    motion = _to_float(row, "motion_sim", 0.0)
    agreement = _to_float(row, "agreement", 0.0)
    stability = _to_float(row, "stability", 1.0)
    coherence = _to_float(row, "coherence", 1.0)
    det_score = _to_float(row, "det_score", 1.0)
    gap = _to_int(row, "gap", 0)
    history_len = _to_int(row, "history_len", 0)
    absdiff = abs(spatial - laplace)
    min_sim = min(spatial, laplace)
    prod_sim = _to_float(row, "prod_sim", spatial * laplace)
    gap_log1p = math.log1p(max(0, gap))
    hist_norm = min(1.0, max(0.0, float(history_len) / float(max(int(min_history), 1))))
    amb_spa = _to_float(row, "amb_spa", 0.0)
    amb_lap = _to_float(row, "amb_lap", 0.0)
    amb_mot = _to_float(row, "amb_mot", 0.0)
    return np.asarray(
        [
            spatial,
            laplace,
            motion,
            absdiff,
            min_sim,
            prod_sim,
            agreement,
            stability,
            coherence,
            det_score,
            gap_log1p,
            hist_norm,
            amb_spa,
            amb_lap,
            amb_mot,
        ],
        dtype=np.float32,
    )


def _hardness(candidate: Candidate) -> float:
    return max(candidate.spatial, candidate.laplace, candidate.motion)


def _build_groups(rows: Iterable[Dict[str, str]], min_history: int, assoc_stage: str) -> List[Group]:
    grouped: Dict[tuple, List[Dict[str, str]]] = {}
    for row in rows:
        if row.get("assoc_stage", "primary") != assoc_stage:
            continue
        key = (row["seq"], _to_int(row, "frame"), _to_int(row, "det_index"), row.get("assoc_stage", "primary"))
        grouped.setdefault(key, []).append(row)

    groups: List[Group] = []
    for (seq, frame, det_index, stage), items in grouped.items():
        candidates: List[Candidate] = []
        positive_indices = []
        for item in items:
            features = extract_features(item, min_history=min_history)
            cand = Candidate(
                features=features,
                spatial=float(features[0]),
                laplace=float(features[1]),
                motion=float(features[2]),
                label=_to_int(item, "is_true_match", 0),
            )
            if cand.label == 1:
                positive_indices.append(len(candidates))
            candidates.append(cand)

        if len(candidates) == 0:
            continue
        if len(positive_indices) > 1:
            # Skip ambiguous supervision groups.
            continue
        positive_index = positive_indices[0] if positive_indices else -1
        groups.append(
            Group(
                seq=seq,
                frame=frame,
                det_index=det_index,
                assoc_stage=stage,
                candidates=candidates,
                positive_index=positive_index,
            )
        )
    return groups


def _sample_group(group: Group, max_hard_negatives: int, max_random_negatives: int) -> Group:
    if group.positive_index < 0:
        negatives = group.candidates
        if len(negatives) <= max_hard_negatives + max_random_negatives or (max_hard_negatives + max_random_negatives) <= 0:
            return group
        ordered = sorted(negatives, key=_hardness, reverse=True)
        keep = ordered[:max_hard_negatives]
        rest = ordered[max_hard_negatives:]
        if rest and max_random_negatives > 0:
            keep.extend(random.sample(rest, k=min(max_random_negatives, len(rest))))
        return Group(group.seq, group.frame, group.det_index, group.assoc_stage, keep, -1)

    pos = group.candidates[group.positive_index]
    negatives = [c for idx, c in enumerate(group.candidates) if idx != group.positive_index]
    if len(negatives) <= max_hard_negatives + max_random_negatives or (max_hard_negatives + max_random_negatives) <= 0:
        return Group(group.seq, group.frame, group.det_index, group.assoc_stage, [pos] + negatives, 0)

    negatives = sorted(negatives, key=_hardness, reverse=True)
    keep = negatives[:max_hard_negatives]
    rest = negatives[max_hard_negatives:]
    if rest and max_random_negatives > 0:
        keep.extend(random.sample(rest, k=min(max_random_negatives, len(rest))))
    return Group(group.seq, group.frame, group.det_index, group.assoc_stage, [pos] + keep, 0)


REQUIRED_PAIR_LOG_COLUMNS = {
    "seq",
    "frame",
    "assoc_stage",
    "det_index",
    "gap",
    "history_len",
    "chosen",
    "is_true_match",
    "track_gt_id",
    "det_gt_id",
    "spatial_sim",
    "laplace_sim",
    "motion_sim",
    "agreement",
    "stability",
    "coherence",
    "det_score",
    "amb_spa",
    "amb_lap",
    "amb_mot",
}


def load_rows(csv_paths: Sequence[str], allow_legacy_schema: bool) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for csv_path in csv_paths:
        with Path(csv_path).open("r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            missing = sorted(REQUIRED_PAIR_LOG_COLUMNS.difference(fieldnames))
            if missing and not allow_legacy_schema:
                raise ValueError(
                    "Pair-log schema is stale and would silently zero-fill learned features. "
                    f"Missing columns in {csv_path}: {missing}. "
                    "Regenerate logs with the current runtime logger, or pass --allow-legacy-schema only for smoke tests."
                )
            rows.extend(reader)
    return rows


def split_groups(groups: List[Group], train_prefixes: Sequence[str] | None, val_prefixes: Sequence[str] | None, val_frac: float):
    if val_prefixes:
        train_groups = [g for g in groups if _matches_prefix(g.seq, train_prefixes) and not _matches_prefix(g.seq, val_prefixes)]
        val_groups = [g for g in groups if _matches_prefix(g.seq, val_prefixes)]
    elif train_prefixes:
        train_groups = [g for g in groups if _matches_prefix(g.seq, train_prefixes)]
        val_groups = [g for g in groups if not _matches_prefix(g.seq, train_prefixes)]
    else:
        shuffled = groups[:]
        random.shuffle(shuffled)
        cut = max(1, int(len(shuffled) * float(val_frac)))
        val_groups = shuffled[:cut]
        train_groups = shuffled[cut:]

    if not train_groups:
        raise ValueError("No training groups after split.")
    if not val_groups:
        val_groups = train_groups[: max(1, min(64, len(train_groups) // 10 or 1))]
    return train_groups, val_groups


class AlphaRCalibratorTorch(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.use_mlp = hidden_dim > 0
        if self.use_mlp:
            self.fc1 = nn.Linear(in_dim, hidden_dim)
            self.alpha_head = nn.Linear(hidden_dim, 1)
            self.r_head = nn.Linear(hidden_dim, 1)
        else:
            self.fc1 = None
            self.alpha_head = nn.Linear(in_dim, 1)
            self.r_head = nn.Linear(in_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.alpha_head.bias.fill_(math.log(0.35 / (1.0 - 0.35)))
            self.r_head.bias.fill_(0.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = F.gelu(self.fc1(x)) if self.use_mlp else x
        alpha = torch.sigmoid(self.alpha_head(h)).squeeze(-1)
        r = torch.sigmoid(self.r_head(h)).squeeze(-1)
        return {"alpha": alpha, "r": r}


def _build_normalizer(groups: Sequence[Group]) -> tuple[np.ndarray, np.ndarray]:
    feats = np.concatenate([np.stack([c.features for c in g.candidates], axis=0) for g in groups], axis=0)
    mean = feats.mean(axis=0).astype(np.float32)
    std = np.clip(feats.std(axis=0).astype(np.float32), 1e-6, None)
    return mean, std


def _normalize_group(group: Group, mean: np.ndarray, std: np.ndarray) -> Group:
    candidates = []
    for cand in group.candidates:
        feat = ((cand.features - mean) / std).astype(np.float32)
        candidates.append(Candidate(feat, cand.spatial, cand.laplace, cand.motion, cand.label))
    return Group(group.seq, group.frame, group.det_index, group.assoc_stage, candidates, group.positive_index)


def _prepare_groups(groups: Sequence[Group], mean: np.ndarray, std: np.ndarray, max_hard_negatives: int, max_random_negatives: int) -> List[Group]:
    prepared = []
    for group in groups:
        sampled = _sample_group(group, max_hard_negatives=max_hard_negatives, max_random_negatives=max_random_negatives)
        prepared.append(_normalize_group(sampled, mean=mean, std=std))
    return prepared


def _batched_groups(groups: Sequence[Group], batch_size: int):
    idx = list(range(len(groups)))
    random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield [groups[i] for i in idx[start : start + batch_size]]


def _forward_batch(model: AlphaRCalibratorTorch, batch_groups: Sequence[Group], device: torch.device):
    features = np.concatenate([np.stack([cand.features for cand in g.candidates], axis=0) for g in batch_groups], axis=0)
    raw_spatial = np.concatenate([[cand.spatial for cand in g.candidates] for g in batch_groups], axis=0).astype(np.float32)
    raw_laplace = np.concatenate([[cand.laplace for cand in g.candidates] for g in batch_groups], axis=0).astype(np.float32)
    raw_motion = np.concatenate([[cand.motion for cand in g.candidates] for g in batch_groups], axis=0).astype(np.float32)
    xb = torch.from_numpy(features).to(device=device, dtype=torch.float32)
    out = model(xb)
    alpha = out["alpha"]
    r = out["r"]
    spatial = torch.from_numpy(raw_spatial).to(device=device, dtype=torch.float32)
    laplace = torch.from_numpy(raw_laplace).to(device=device, dtype=torch.float32)
    motion = torch.from_numpy(raw_motion).to(device=device, dtype=torch.float32)
    s_app = (1.0 - alpha) * spatial + alpha * laplace
    s_fuse = (r * s_app + (1.0 - r) * motion).clamp(min=1e-4, max=1.0 - 1e-4)
    return xb, alpha, r, s_app, s_fuse


def _batch_losses(
    batch_groups: Sequence[Group],
    alpha: torch.Tensor,
    r: torch.Tensor,
    s_app: torch.Tensor,
    s_fuse: torch.Tensor,
    temperature: float,
    rank_margin: float,
    trust_margin: float,
):
    ce_terms = []
    bg_terms = []
    rank_terms = []
    trust_terms = []
    top1_correct = 0
    top1_total = 0
    pos_groups = 0
    bg_groups = 0

    offset = 0
    for group in batch_groups:
        n = len(group.candidates)
        group_slice = slice(offset, offset + n)
        fuse_g = s_fuse[group_slice]
        app_g = s_app[group_slice]
        motion_g = torch.tensor([cand.motion for cand in group.candidates], device=s_fuse.device, dtype=s_fuse.dtype)
        r_g = r[group_slice]
        logits = torch.logit(fuse_g, eps=1e-4) / max(float(temperature), 1e-4)

        if group.positive_index >= 0:
            target = torch.tensor([group.positive_index], device=s_fuse.device, dtype=torch.long)
            ce_terms.append(F.cross_entropy(logits.unsqueeze(0), target))
            pos_groups += 1
            pred = int(torch.argmax(fuse_g).item())
            top1_correct += int(pred == group.positive_index)
            top1_total += 1

            neg_mask = torch.ones((n,), device=s_fuse.device, dtype=torch.bool)
            neg_mask[group.positive_index] = False
            if neg_mask.any():
                z_pos = logits[group.positive_index]
                z_neg = logits[neg_mask].max()
                rank_terms.append(F.relu(torch.tensor(rank_margin, device=s_fuse.device, dtype=s_fuse.dtype) - z_pos + z_neg))

                delta_app = app_g[group.positive_index] - app_g[neg_mask].max()
                delta_mot = motion_g[group.positive_index] - motion_g[neg_mask].max()
                if delta_app > delta_mot + trust_margin:
                    trust_target = torch.tensor([1.0], device=s_fuse.device, dtype=s_fuse.dtype)
                    trust_terms.append(F.binary_cross_entropy(r_g[group.positive_index].view(1), trust_target))
                elif delta_mot > delta_app + trust_margin:
                    trust_target = torch.tensor([0.0], device=s_fuse.device, dtype=s_fuse.dtype)
                    trust_terms.append(F.binary_cross_entropy(r_g[group.positive_index].view(1), trust_target))
        else:
            bg_groups += 1
            bg_terms.append(-torch.log((1.0 - fuse_g.max()).clamp(min=1e-4)))

        offset += n

    zero = s_fuse.new_tensor(0.0)
    ce = torch.stack(ce_terms).mean() if ce_terms else zero
    bg = torch.stack(bg_terms).mean() if bg_terms else zero
    rank = torch.stack(rank_terms).mean() if rank_terms else zero
    trust = torch.stack(trust_terms).mean() if trust_terms else zero
    metrics = {
        "ce": ce,
        "bg": bg,
        "rank": rank,
        "trust": trust,
        "top1_acc": float(top1_correct) / float(max(top1_total, 1)),
        "pos_groups": pos_groups,
        "bg_groups": bg_groups,
    }
    return metrics


def evaluate(model, groups, device, args):
    model.eval()
    losses = []
    accs = []
    with torch.no_grad():
        for start in range(0, len(groups), max(1, int(args.batch_size))):
            batch_groups = groups[start : start + int(args.batch_size)]
            _, alpha, r, s_app, s_fuse = _forward_batch(model, batch_groups, device)
            metrics = _batch_losses(
                batch_groups,
                alpha=alpha,
                r=r,
                s_app=s_app,
                s_fuse=s_fuse,
                temperature=args.temperature,
                rank_margin=args.rank_margin,
                trust_margin=args.trust_margin,
            )
            loss = (
                metrics["ce"]
                + float(args.loss_bg_weight) * metrics["bg"]
                + float(args.loss_rank_weight) * metrics["rank"]
                + float(args.loss_trust_weight) * metrics["trust"]
            )
            losses.append(float(loss.item()))
            accs.append(metrics["top1_acc"])
    return {
        "loss": float(np.mean(losses)) if losses else float("inf"),
        "top1_acc": float(np.mean(accs)) if accs else 0.0,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_rows(args.pair_csv, allow_legacy_schema=bool(args.allow_legacy_schema))
    groups = _build_groups(rows, min_history=args.min_history, assoc_stage=args.assoc_stage)
    if not groups:
        raise ValueError("No valid groups found in the provided pair logs.")
    print(f"[data] groups loaded: {len(groups)}")

    train_groups, val_groups = split_groups(groups, args.train_seqs, args.val_seqs, args.val_frac)
    print(f"[split] train={len(train_groups)} val={len(val_groups)}")

    mean, std = _build_normalizer(train_groups)
    train_groups = _prepare_groups(
        train_groups,
        mean=mean,
        std=std,
        max_hard_negatives=args.max_hard_negatives,
        max_random_negatives=args.max_random_negatives,
    )
    val_groups = _prepare_groups(
        val_groups,
        mean=mean,
        std=std,
        max_hard_negatives=args.max_hard_negatives,
        max_random_negatives=args.max_random_negatives,
    )

    device = torch.device("cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = AlphaRCalibratorTorch(in_dim=len(LEARNED_FEATURE_NAMES), hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best = {"loss": float("inf"), "state": None, "epoch": 0}
    bad_epochs = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = []
        for batch_groups in _batched_groups(train_groups, batch_size=max(1, int(args.batch_size))):
            _, alpha, r, s_app, s_fuse = _forward_batch(model, batch_groups, device)
            metrics = _batch_losses(
                batch_groups,
                alpha=alpha,
                r=r,
                s_app=s_app,
                s_fuse=s_fuse,
                temperature=args.temperature,
                rank_margin=args.rank_margin,
                trust_margin=args.trust_margin,
            )
            loss = (
                metrics["ce"]
                + float(args.loss_bg_weight) * metrics["bg"]
                + float(args.loss_rank_weight) * metrics["rank"]
                + float(args.loss_trust_weight) * metrics["trust"]
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running.append(float(loss.detach().cpu().item()))

        val_metrics = evaluate(model, val_groups, device, args)
        train_loss = float(np.mean(running)) if running else float("inf")
        print(f"[epoch {epoch:02d}] train_loss={train_loss:.6f} val_loss={val_metrics['loss']:.6f} val_top1={val_metrics['top1_acc']:.4f}")

        if val_metrics["loss"] < best["loss"]:
            best["loss"] = val_metrics["loss"]
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["epoch"] = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                print(f"[early-stop] epoch={epoch} best_epoch={best['epoch']}")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    weights = {
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "feature_names": np.asarray(LEARNED_FEATURE_NAMES, dtype=object),
        "temperature": np.asarray([float(args.temperature)], dtype=np.float32),
    }
    if model.use_mlp:
        weights["W1"] = state["fc1.weight"].T
        weights["b1"] = state["fc1.bias"]
    weights["W_alpha"] = state["alpha_head.weight"].reshape(-1)
    weights["b_alpha"] = state["alpha_head.bias"].reshape(-1)
    weights["W_r"] = state["r_head.weight"].reshape(-1)
    weights["b_r"] = state["r_head.bias"].reshape(-1)
    np.savez(out_path, **weights)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
