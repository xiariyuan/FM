#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
BOT_ROOT = REPO_ROOT / "external/BoT-SORT-main"
import sys

if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from tracker.laplace_assoc import PAIR_FEATURE_NAMES, TRACK_FEATURE_NAMES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a learnable Laplace pole-bank LTRA checkpoint from GT pseudo-track NPZ groups.")
    parser.add_argument("--train-npz", nargs="+", required=True)
    parser.add_argument("--val-npz", nargs="*", default=[])
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-groups", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pair-hidden", type=int, default=16)
    parser.add_argument("--track-hidden", type=int, default=8)
    parser.add_argument("--tau-init", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--tau-min", type=float, default=0.5)
    parser.add_argument("--tau-max", type=float, default=32.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--trust-margin", type=float, default=0.03)
    parser.add_argument("--loss-bg-weight", type=float, default=0.25)
    parser.add_argument("--loss-rank-weight", type=float, default=0.25)
    parser.add_argument("--loss-trust-weight", type=float, default=0.10)
    parser.add_argument("--loss-pole-weight", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=2)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.exp(x) - 1.0)


@dataclass
class DatasetTensors:
    path: str
    det_feat: np.ndarray
    hist_feat: np.ndarray
    hist_mask: np.ndarray
    track_feat: np.ndarray
    ctx_feat: np.ndarray
    group_id: np.ndarray
    label: np.ndarray
    batch_slices: list[tuple[int, int]]


def load_npz(path: str) -> DatasetTensors:
    z = np.load(path, allow_pickle=True)
    required = ["det_feat", "hist_feat", "hist_mask", "track_feat", "ctx_feat", "group_id", "label"]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise ValueError(f"Missing required NPZ keys in {path}: {missing}")
    group_id = np.asarray(z["group_id"], dtype=np.int64)
    return DatasetTensors(
        path=str(path),
        det_feat=np.asarray(z["det_feat"]),
        hist_feat=np.asarray(z["hist_feat"]),
        hist_mask=np.asarray(z["hist_mask"], dtype=np.float32),
        track_feat=np.asarray(z["track_feat"], dtype=np.float32),
        ctx_feat=np.asarray(z["ctx_feat"], dtype=np.float32),
        group_id=group_id,
        label=np.asarray(z["label"], dtype=np.float32),
        batch_slices=build_batch_slices(group_id, batch_groups=None),
    )


def build_batch_slices(group_ids: np.ndarray, batch_groups: int | None) -> list[tuple[int, int]]:
    if group_ids.size == 0:
        return []
    change = np.flatnonzero(group_ids[1:] != group_ids[:-1]) + 1
    starts = np.concatenate(([0], change)).astype(np.int64)
    ends = np.concatenate((change, [group_ids.shape[0]])).astype(np.int64)
    if batch_groups is None or batch_groups <= 0:
        return [(int(starts[0]), int(ends[-1]))]
    batch_slices = []
    for batch_start in range(0, starts.shape[0], int(batch_groups)):
        group_start = int(starts[batch_start])
        group_end_idx = min(batch_start + int(batch_groups) - 1, ends.shape[0] - 1)
        group_end = int(ends[group_end_idx])
        batch_slices.append((group_start, group_end))
    return batch_slices


def prepare_dataset(data: DatasetTensors, batch_groups: int) -> DatasetTensors:
    data.batch_slices = build_batch_slices(data.group_id, batch_groups)
    return data


def load_datasets(paths: Sequence[str], batch_groups: int, split_name: str) -> list[DatasetTensors]:
    datasets = []
    print(f"[load] {split_name} shards={len(paths)}", flush=True)
    for shard_idx, path in enumerate(paths, start=1):
        data = prepare_dataset(load_npz(path), batch_groups)
        datasets.append(data)
        print(
            f"[load] {split_name} {shard_idx}/{len(paths)} path={path} "
            f"candidates={data.group_id.shape[0]} batches={len(data.batch_slices)}",
            flush=True,
        )
    return datasets


def _group_layout(group_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if group_ids.numel() == 0:
        empty = group_ids.new_zeros((0,), dtype=torch.long)
        return empty, empty, empty, empty, empty
    change = torch.nonzero(group_ids[1:] != group_ids[:-1], as_tuple=False).flatten() + 1
    starts = torch.cat([group_ids.new_zeros((1,), dtype=torch.long), change.to(dtype=torch.long)])
    ends = torch.cat([change.to(dtype=torch.long), group_ids.new_tensor([group_ids.numel()], dtype=torch.long)])
    lengths = ends - starts
    rows = torch.repeat_interleave(torch.arange(lengths.numel(), device=group_ids.device, dtype=torch.long), lengths)
    cols = torch.arange(group_ids.numel(), device=group_ids.device, dtype=torch.long) - torch.repeat_interleave(starts, lengths)
    return starts, ends, lengths, rows, cols


def _group_matrix(values: torch.Tensor, group_ids: torch.Tensor, fill_value: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    starts, _, lengths, rows, cols = _group_layout(group_ids)
    if lengths.numel() == 0:
        empty = values.new_zeros((0, 0))
        valid = torch.zeros((0, 0), dtype=torch.bool, device=values.device)
        return empty, valid, lengths
    max_len = int(lengths.max().item())
    mat = values.new_full((lengths.numel(), max_len), float(fill_value))
    valid = torch.zeros((lengths.numel(), max_len), dtype=torch.bool, device=values.device)
    mat[rows, cols] = values
    valid[rows, cols] = True
    return mat, valid, lengths


def group_margin(values: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    mat, valid, lengths = _group_matrix(values, group_ids, fill_value=-1e9)
    if lengths.numel() == 0:
        return torch.zeros_like(values)
    top1 = mat.max(dim=1).values
    if mat.shape[1] >= 2:
        top2 = mat.topk(k=2, dim=1).values[:, 1]
        margin_per_group = torch.where(lengths > 1, top1 - top2, top1)
    else:
        margin_per_group = top1
    margin_per_group = margin_per_group.clamp(min=0.0, max=1.0)
    return torch.repeat_interleave(margin_per_group, lengths)


class PoleBankLTRA(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        pair_hidden: int = 16,
        track_hidden: int = 8,
        tau_init: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
        tau_min: float = 0.5,
        tau_max: float = 32.0,
        use_pair_mlp: bool = True,
        use_track_mlp: bool = True,
    ) -> None:
        super().__init__()
        tau_init = [float(v) for v in tau_init]
        if not tau_init:
            raise ValueError("tau_init must be non-empty")
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.use_pair_mlp = bool(use_pair_mlp and pair_hidden > 0)
        self.use_track_mlp = bool(use_track_mlp and track_hidden > 0)

        delta_init = []
        prev = self.tau_min
        for tau in tau_init:
            delta_init.append(max(float(tau) - prev, 1e-3))
            prev = float(tau)
        self.raw_tau_delta = nn.Parameter(inv_softplus(torch.tensor(delta_init, dtype=torch.float32)))

        if self.use_track_mlp:
            self.track_fc1 = nn.Linear(len(TRACK_FEATURE_NAMES), int(track_hidden))
            self.track_fc2 = nn.Linear(int(track_hidden), len(tau_init))
        else:
            self.track_fc1 = None
            self.track_fc2 = nn.Linear(len(TRACK_FEATURE_NAMES), len(tau_init))

        if self.use_pair_mlp:
            self.pair_fc1 = nn.Linear(len(PAIR_FEATURE_NAMES), int(pair_hidden))
            self.alpha_head = nn.Linear(int(pair_hidden), 1)
            self.r_head = nn.Linear(int(pair_hidden), 1)
        else:
            self.pair_fc1 = None
            self.alpha_head = nn.Linear(len(PAIR_FEATURE_NAMES), 1)
            self.r_head = nn.Linear(len(PAIR_FEATURE_NAMES), 1)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.alpha_head.bias.fill_(math.log(0.35 / (1.0 - 0.35)))
            self.r_head.bias.fill_(0.0)

    def tau_values(self) -> torch.Tensor:
        delta = F.softplus(self.raw_tau_delta)
        tau = self.tau_min + torch.cumsum(delta, dim=0)
        return tau.clamp(max=self.tau_max)

    def _track_gate_logits(self, track_features: torch.Tensor) -> torch.Tensor:
        if self.use_track_mlp:
            h = F.gelu(self.track_fc1(track_features))
            return self.track_fc2(h)
        return self.track_fc2(track_features)

    def _pair_hidden(self, pair_features: torch.Tensor) -> torch.Tensor:
        if self.use_pair_mlp:
            return F.gelu(self.pair_fc1(pair_features))
        return pair_features


def build_model_inputs(
    model: PoleBankLTRA,
    det_feat: torch.Tensor,
    hist_feat: torch.Tensor,
    hist_mask: torch.Tensor,
    track_feat_raw: torch.Tensor,
    ctx_feat_raw: torch.Tensor,
    group_ids: torch.Tensor,
    pair_mean: torch.Tensor,
    pair_std: torch.Tensor,
    track_mean: torch.Tensor,
    track_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    det_feat = F.normalize(det_feat, dim=-1)
    hist_feat = F.normalize(hist_feat, dim=-1)
    tau_values = model.tau_values()

    ages = torch.arange(hist_feat.shape[1] - 1, -1, -1, device=hist_feat.device, dtype=hist_feat.dtype)
    weights = torch.exp(-ages[None, None, :] / tau_values[:, None, None].clamp(min=1e-4))
    weights = weights * hist_mask.unsqueeze(0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    protos = torch.einsum("knl,nld->knd", weights, hist_feat).permute(1, 0, 2)
    protos = F.normalize(protos, dim=-1)
    s_k = (protos * det_feat[:, None, :]).sum(dim=-1).clamp(min=-1.0, max=1.0)
    s_k = 0.5 * (s_k + 1.0)

    last_feat = hist_feat[:, -1, :]
    coherence = (protos * last_feat[:, None, :]).sum(dim=-1).clamp(min=-1.0, max=1.0)
    coherence = 0.5 * (coherence + 1.0)
    coherence_mean = coherence.mean(dim=-1)
    stability = track_feat_raw[:, 2]

    track_features_raw = torch.stack(
        [
            track_feat_raw[:, 0],
            track_feat_raw[:, 1],
            stability,
            coherence_mean,
        ],
        dim=-1,
    )
    track_features = (track_features_raw - track_mean) / track_std.clamp(min=1e-6)
    pi = torch.softmax(model._track_gate_logits(track_features), dim=-1)
    s_lap = (pi * s_k).sum(dim=-1)

    spatial_sim = ctx_feat_raw[:, 0]
    motion_sim = ctx_feat_raw[:, 1]
    det_score = ctx_feat_raw[:, 2]
    amb_spa = ctx_feat_raw[:, 3]
    amb_mot = ctx_feat_raw[:, 4]
    amb_lap = group_margin(s_lap, group_ids)

    absdiff = torch.abs(spatial_sim - s_lap)
    min_sim = torch.minimum(spatial_sim, s_lap)
    prod_sim = (spatial_sim * s_lap).clamp(min=0.0, max=1.0)
    agreement = (1.0 - absdiff).clamp(min=0.0, max=1.0)

    pair_features_raw = torch.stack(
        [
            spatial_sim,
            s_lap,
            motion_sim,
            absdiff,
            min_sim,
            prod_sim,
            agreement,
            stability,
            coherence_mean,
            det_score,
            track_feat_raw[:, 0],
            track_feat_raw[:, 1],
            amb_spa,
            amb_lap,
            amb_mot,
        ],
        dim=-1,
    )
    pair_features = (pair_features_raw - pair_mean) / pair_std.clamp(min=1e-6)
    hidden = model._pair_hidden(pair_features)
    alpha = torch.sigmoid(model.alpha_head(hidden)).squeeze(-1)
    r = torch.sigmoid(model.r_head(hidden)).squeeze(-1)

    s_app = (1.0 - alpha) * spatial_sim + alpha * s_lap
    s_fuse = (r * s_app + (1.0 - r) * motion_sim).clamp(min=1e-4, max=1.0 - 1e-4)
    return {
        "tau_values": tau_values,
        "pi": pi,
        "s_k": s_k,
        "s_lap": s_lap,
        "coherence": coherence_mean,
        "stability": stability,
        "track_features_raw": track_features_raw,
        "pair_features_raw": pair_features_raw,
        "alpha": alpha,
        "r": r,
        "s_app": s_app,
        "s_fuse": s_fuse,
        "motion": motion_sim,
    }


def _dataset_feature_stats(
    model: PoleBankLTRA,
    datasets: Sequence[DatasetTensors],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pair_sum = None
    pair_sumsq = None
    track_sum = None
    track_sumsq = None
    pair_count = 0
    track_count = 0
    model.eval()
    print(f"[stats] start shards={len(datasets)}", flush=True)
    with torch.no_grad():
        for shard_idx, data in enumerate(datasets, start=1):
            print(f"[stats] shard {shard_idx}/{len(datasets)} {data.path}", flush=True)
            for start, end in data.batch_slices:
                sl = slice(start, end)
                group_ids = torch.from_numpy(data.group_id[sl]).to(device)
                det_feat = torch.from_numpy(data.det_feat[sl]).to(device=device, dtype=torch.float32)
                hist_feat = torch.from_numpy(data.hist_feat[sl]).to(device=device, dtype=torch.float32)
                hist_mask = torch.from_numpy(data.hist_mask[sl]).to(device)
                track_feat_raw = torch.from_numpy(data.track_feat[sl]).to(device)
                ctx_feat_raw = torch.from_numpy(data.ctx_feat[sl]).to(device)

                pair_mean = torch.zeros((len(PAIR_FEATURE_NAMES),), device=device)
                pair_std = torch.ones((len(PAIR_FEATURE_NAMES),), device=device)
                track_mean = torch.zeros((len(TRACK_FEATURE_NAMES),), device=device)
                track_std = torch.ones((len(TRACK_FEATURE_NAMES),), device=device)
                out = build_model_inputs(
                    model=model,
                    det_feat=det_feat,
                    hist_feat=hist_feat,
                    hist_mask=hist_mask,
                    track_feat_raw=track_feat_raw,
                    ctx_feat_raw=ctx_feat_raw,
                    group_ids=group_ids,
                    pair_mean=pair_mean,
                    pair_std=pair_std,
                    track_mean=track_mean,
                    track_std=track_std,
                )
                pair_raw = out["pair_features_raw"].cpu().numpy().astype(np.float64)
                track_raw = out["track_features_raw"].cpu().numpy().astype(np.float64)
                if pair_sum is None:
                    pair_sum = pair_raw.sum(axis=0)
                    pair_sumsq = np.square(pair_raw).sum(axis=0)
                    track_sum = track_raw.sum(axis=0)
                    track_sumsq = np.square(track_raw).sum(axis=0)
                else:
                    pair_sum += pair_raw.sum(axis=0)
                    pair_sumsq += np.square(pair_raw).sum(axis=0)
                    track_sum += track_raw.sum(axis=0)
                    track_sumsq += np.square(track_raw).sum(axis=0)
                pair_count += pair_raw.shape[0]
                track_count += track_raw.shape[0]
            print(
                f"[stats] shard {shard_idx}/{len(datasets)} done pair_count={pair_count} track_count={track_count}",
                flush=True,
            )

    if pair_count == 0 or track_count == 0:
        raise ValueError("No training candidates were found while computing feature statistics.")
    pair_mean = (pair_sum / float(pair_count)).astype(np.float32)
    pair_var = np.maximum(pair_sumsq / float(pair_count) - np.square(pair_mean.astype(np.float64)), 1e-12)
    pair_std = np.sqrt(pair_var).astype(np.float32)
    track_mean = (track_sum / float(track_count)).astype(np.float32)
    track_var = np.maximum(track_sumsq / float(track_count) - np.square(track_mean.astype(np.float64)), 1e-12)
    track_std = np.sqrt(track_var).astype(np.float32)
    print(f"[stats] done pair_count={pair_count} track_count={track_count}", flush=True)
    return pair_mean, pair_std, track_mean, track_std


def batch_losses(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float,
    rank_margin: float,
    trust_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    s_fuse = outputs["s_fuse"]
    s_app = outputs["s_app"]
    motion = outputs["motion"]
    r = outputs["r"]
    fuse_mat, valid, lengths = _group_matrix(s_fuse, group_ids, fill_value=1e-4)
    label_mat, _, _ = _group_matrix(labels, group_ids, fill_value=0.0)
    app_mat, _, _ = _group_matrix(s_app, group_ids, fill_value=-1e9)
    mot_mat, _, _ = _group_matrix(motion, group_ids, fill_value=-1e9)
    r_mat, _, _ = _group_matrix(r, group_ids, fill_value=0.0)

    logits = torch.logit(fuse_mat.clamp(min=1e-4, max=1.0 - 1e-4), eps=1e-4) / max(float(temperature), 1e-4)
    logits = logits.masked_fill(~valid, -1e9)

    pos_mask = label_mat > 0.5
    has_pos = pos_mask.any(dim=1)
    bg_mask = ~has_pos

    zero = s_fuse.new_tensor(0.0)

    if has_pos.any():
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        ce = (-log_probs[pos_mask]).mean()

        pred_idx = logits.argmax(dim=1)
        pos_idx = pos_mask.float().argmax(dim=1)
        top1 = float((pred_idx[has_pos] == pos_idx[has_pos]).float().mean().item())

        neg_logits = logits.masked_fill(pos_mask, -1e9)
        neg_app = app_mat.masked_fill(pos_mask | ~valid, -1e9)
        neg_mot = mot_mat.masked_fill(pos_mask | ~valid, -1e9)

        pos_logits = logits[pos_mask]
        pos_app = app_mat[pos_mask]
        pos_mot = mot_mat[pos_mask]
        pos_r = r_mat[pos_mask]

        neg_has_any = (~pos_mask & valid).any(dim=1)
        pos_has_neg = neg_has_any[has_pos]

        if pos_has_neg.any():
            z_pos = pos_logits[pos_has_neg]
            z_neg = neg_logits.max(dim=1).values[has_pos][pos_has_neg]
            rank = F.relu(
                torch.tensor(rank_margin, device=logits.device, dtype=logits.dtype) - z_pos + z_neg
            ).mean()

            app_gap = pos_app[pos_has_neg] - neg_app.max(dim=1).values[has_pos][pos_has_neg]
            mot_gap = pos_mot[pos_has_neg] - neg_mot.max(dim=1).values[has_pos][pos_has_neg]
            pos_r_active = pos_r[pos_has_neg]
            trust_pos = app_gap > (mot_gap + trust_margin)
            trust_neg = mot_gap > (app_gap + trust_margin)
            trust_targets = []
            trust_preds = []
            if trust_pos.any():
                trust_targets.append(torch.ones_like(pos_r_active[trust_pos]))
                trust_preds.append(pos_r_active[trust_pos])
            if trust_neg.any():
                trust_targets.append(torch.zeros_like(pos_r_active[trust_neg]))
                trust_preds.append(pos_r_active[trust_neg])
            if trust_targets:
                trust = F.binary_cross_entropy(torch.cat(trust_preds), torch.cat(trust_targets))
            else:
                trust = zero
        else:
            rank = zero
            trust = zero
    else:
        ce = zero
        rank = zero
        trust = zero
        top1 = 0.0

    if bg_mask.any():
        bg = (-torch.log((1.0 - fuse_mat.max(dim=1).values[bg_mask]).clamp(min=1e-4))).mean()
    else:
        bg = zero

    return ce, bg, rank, trust, top1


def evaluate(
    model: PoleBankLTRA,
    datasets: Sequence[DatasetTensors],
    device: torch.device,
    pair_mean: torch.Tensor,
    pair_std: torch.Tensor,
    track_mean: torch.Tensor,
    track_std: torch.Tensor,
    temperature: float,
    rank_margin: float,
    trust_margin: float,
    loss_bg_weight: float,
    loss_rank_weight: float,
    loss_trust_weight: float,
    loss_pole_weight: float,
    tau_min: float,
    tau_max: float,
) -> tuple[float, float]:
    if not datasets:
        return float("nan"), float("nan")
    model.eval()
    losses = []
    top1s = []
    with torch.no_grad():
        for data in datasets:
            for start, end in data.batch_slices:
                sl = slice(start, end)
                group_ids = torch.from_numpy(data.group_id[sl]).to(device)
                labels = torch.from_numpy(data.label[sl]).to(device)
                outputs = build_model_inputs(
                    model=model,
                    det_feat=torch.from_numpy(data.det_feat[sl]).to(device=device, dtype=torch.float32),
                    hist_feat=torch.from_numpy(data.hist_feat[sl]).to(device=device, dtype=torch.float32),
                    hist_mask=torch.from_numpy(data.hist_mask[sl]).to(device),
                    track_feat_raw=torch.from_numpy(data.track_feat[sl]).to(device),
                    ctx_feat_raw=torch.from_numpy(data.ctx_feat[sl]).to(device),
                    group_ids=group_ids,
                    pair_mean=pair_mean,
                    pair_std=pair_std,
                    track_mean=track_mean,
                    track_std=track_std,
                )
                ce, bg, rank, trust, top1 = batch_losses(
                    outputs=outputs,
                    labels=labels,
                    group_ids=group_ids,
                    temperature=temperature,
                    rank_margin=rank_margin,
                    trust_margin=trust_margin,
                )
                tau_values = outputs["tau_values"]
                pole_reg = (
                    F.relu(torch.tensor(float(tau_min), device=device) - tau_values).pow(2).mean()
                    + F.relu(tau_values - torch.tensor(float(tau_max), device=device)).pow(2).mean()
                )
                loss = ce + loss_bg_weight * bg + loss_rank_weight * rank + loss_trust_weight * trust + loss_pole_weight * pole_reg
                losses.append(float(loss.item()))
                top1s.append(float(top1))
    return float(np.mean(losses)) if losses else float("inf"), float(np.mean(top1s)) if top1s else 0.0


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = torch.device("cuda" if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")

    train_paths = [str(Path(p)) for p in args.train_npz]
    val_paths = [str(Path(p)) for p in args.val_npz] if args.val_npz else []
    train_datasets = load_datasets(train_paths, int(args.batch_groups), "train")
    val_datasets = load_datasets(val_paths, int(args.batch_groups), "val") if val_paths else []
    first_train = train_datasets[0]

    model = PoleBankLTRA(
        emb_dim=int(first_train.det_feat.shape[1]),
        pair_hidden=int(args.pair_hidden),
        track_hidden=int(args.track_hidden),
        tau_init=args.tau_init,
        tau_min=float(args.tau_min),
        tau_max=float(args.tau_max),
        use_pair_mlp=int(args.pair_hidden) > 0,
        use_track_mlp=int(args.track_hidden) > 0,
    ).to(device)

    pair_mean_np, pair_std_np, track_mean_np, track_std_np = _dataset_feature_stats(
        model=model,
        datasets=train_datasets,
        device=device,
    )
    pair_mean = torch.from_numpy(pair_mean_np).to(device)
    pair_std = torch.from_numpy(pair_std_np).to(device)
    track_mean = torch.from_numpy(track_mean_np).to(device)
    track_std = torch.from_numpy(track_std_np).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_losses = []
        epoch_top1 = []

        epoch_train_sets = train_datasets[:]
        random.shuffle(epoch_train_sets)
        for train in epoch_train_sets:
            batch_slices = train.batch_slices[:]
            random.shuffle(batch_slices)
            for start, end in batch_slices:
                sl = slice(start, end)
                group_ids = torch.from_numpy(train.group_id[sl]).to(device)
                labels = torch.from_numpy(train.label[sl]).to(device)
                outputs = build_model_inputs(
                    model=model,
                    det_feat=torch.from_numpy(train.det_feat[sl]).to(device=device, dtype=torch.float32),
                    hist_feat=torch.from_numpy(train.hist_feat[sl]).to(device=device, dtype=torch.float32),
                    hist_mask=torch.from_numpy(train.hist_mask[sl]).to(device),
                    track_feat_raw=torch.from_numpy(train.track_feat[sl]).to(device),
                    ctx_feat_raw=torch.from_numpy(train.ctx_feat[sl]).to(device),
                    group_ids=group_ids,
                    pair_mean=pair_mean,
                    pair_std=pair_std,
                    track_mean=track_mean,
                    track_std=track_std,
                )
                ce, bg, rank, trust, top1 = batch_losses(
                    outputs=outputs,
                    labels=labels,
                    group_ids=group_ids,
                    temperature=float(args.temperature),
                    rank_margin=float(args.rank_margin),
                    trust_margin=float(args.trust_margin),
                )
                tau_values = outputs["tau_values"]
                pole_reg = (
                    F.relu(torch.tensor(float(args.tau_min), device=device) - tau_values).pow(2).mean()
                    + F.relu(tau_values - torch.tensor(float(args.tau_max), device=device)).pow(2).mean()
                )
                loss = (
                    ce
                    + float(args.loss_bg_weight) * bg
                    + float(args.loss_rank_weight) * rank
                    + float(args.loss_trust_weight) * trust
                    + float(args.loss_pole_weight) * pole_reg
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_losses.append(float(loss.item()))
                epoch_top1.append(float(top1))

        if val_datasets:
            val_loss, val_top1 = evaluate(
                model=model,
                datasets=val_datasets,
                device=device,
                pair_mean=pair_mean,
                pair_std=pair_std,
                track_mean=track_mean,
                track_std=track_std,
                temperature=float(args.temperature),
                rank_margin=float(args.rank_margin),
                trust_margin=float(args.trust_margin),
                loss_bg_weight=float(args.loss_bg_weight),
                loss_rank_weight=float(args.loss_rank_weight),
                loss_trust_weight=float(args.loss_trust_weight),
                loss_pole_weight=float(args.loss_pole_weight),
                tau_min=float(args.tau_min),
                tau_max=float(args.tau_max),
            )
        else:
            val_loss, val_top1 = float("nan"), float("nan")
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        train_top1 = float(np.mean(epoch_top1)) if epoch_top1 else 0.0
        if val_datasets:
            print(
                f"[epoch {epoch:02d}] train_loss={train_loss:.6f} train_top1={train_top1:.4f} "
                f"val_loss={val_loss:.6f} val_top1={val_top1:.4f} "
                f"taus={model.tau_values().detach().cpu().numpy().round(4).tolist()}"
            )

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(args.patience):
                    print("[early-stop]")
                    break
        else:
            print(
                f"[epoch {epoch:02d}] train_loss={train_loss:.6f} train_top1={train_top1:.4f} "
                f"taus={model.tau_values().detach().cpu().numpy().round(4).tolist()}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    tau_values = model.tau_values().detach().cpu().numpy().astype(np.float32)
    save_dict = {
        "feature_mean": pair_mean_np.astype(np.float32),
        "feature_std": pair_std_np.astype(np.float32),
        "feature_names": np.asarray(PAIR_FEATURE_NAMES, dtype=object),
        "pair_feature_mean": pair_mean_np.astype(np.float32),
        "pair_feature_std": pair_std_np.astype(np.float32),
        "pair_feature_names": np.asarray(PAIR_FEATURE_NAMES, dtype=object),
        "track_feature_mean": track_mean_np.astype(np.float32),
        "track_feature_std": track_std_np.astype(np.float32),
        "track_feature_names": np.asarray(TRACK_FEATURE_NAMES, dtype=object),
        "tau_values": tau_values,
        "W_track2": state["track_fc2.weight"].T,
        "b_track2": state["track_fc2.bias"],
        "W_alpha": state["alpha_head.weight"].reshape(-1),
        "b_alpha": state["alpha_head.bias"].reshape(-1),
        "W_r": state["r_head.weight"].reshape(-1),
        "b_r": state["r_head.bias"].reshape(-1),
        "temperature": np.asarray([float(args.temperature)], dtype=np.float32),
    }
    if "track_fc1.weight" in state:
        save_dict["W_track1"] = state["track_fc1.weight"].T
        save_dict["b_track1"] = state["track_fc1.bias"]
    if "pair_fc1.weight" in state:
        save_dict["W1"] = state["pair_fc1.weight"].T
        save_dict["b1"] = state["pair_fc1.bias"]
    np.savez(out_path, **save_dict)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
