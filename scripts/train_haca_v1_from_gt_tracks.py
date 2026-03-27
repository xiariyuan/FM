#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


HACA_PAIR_TOKEN_NAMES = (
    "anchor_sim",
    "spatial_sim",
    "motion_sim",
    "temp_sim",
    "hist_last_sim",
    "hist_max_sim",
    "hist_std_sim",
    "gap_log1p",
    "hist_norm",
    "stability",
    "coherence",
    "anchor_z",
    "anchor_margin",
    "anchor_rank",
    "det_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HACA from GT pseudo-track NPZ groups.")
    parser.add_argument("--version", type=str, default="haca_v1", choices=["haca_v1", "haca_v2"])
    parser.add_argument("--train-npz", nargs="+", required=True)
    parser.add_argument("--val-npz", nargs="*", default=[])
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-groups", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--hist-hidden", type=int, default=16)
    parser.add_argument("--pair-hidden", type=int, default=64)
    parser.add_argument("--hist-gate-hidden", type=int, default=8)
    parser.add_argument("--anchor-alpha", type=float, default=0.35)
    parser.add_argument("--delta-scale", type=float, default=1.5)
    parser.add_argument("--laplace-decay-scales", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--max-history", type=int, default=0, help="0 means infer from NPZ hist_feat width")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--safe-margin", type=float, default=0.10)
    parser.add_argument("--loss-bg-weight", type=float, default=0.25)
    parser.add_argument("--loss-margin-weight", type=float, default=0.25)
    parser.add_argument("--loss-safe-weight", type=float, default=0.10)
    parser.add_argument("--loss-res-weight", type=float, default=0.02)
    parser.add_argument("--loss-shift-weight", type=float, default=0.0)
    parser.add_argument("--shift-batch-prob", type=float, default=0.0)
    parser.add_argument("--corrupt-feat-noise", type=float, default=0.03)
    parser.add_argument("--corrupt-score-noise", type=float, default=0.08)
    parser.add_argument("--corrupt-history-min-ratio", type=float, default=0.35)
    parser.add_argument("--ood-scale", type=float, default=6.0)
    parser.add_argument("--ood-quantile", type=float, default=0.95)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--disable-set-encoder", action="store_true")
    parser.add_argument("--disable-background", action="store_true")
    parser.add_argument("--disable-hist-gate", action="store_true")
    parser.add_argument("--disable-ood-gate", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_batch_slices(group_ids: np.ndarray, batch_groups: int | None) -> list[tuple[int, int]]:
    if group_ids.size == 0:
        return []
    change = np.flatnonzero(group_ids[1:] != group_ids[:-1]) + 1
    starts = np.concatenate(([0], change)).astype(np.int64)
    ends = np.concatenate((change, [group_ids.shape[0]])).astype(np.int64)
    if batch_groups is None or batch_groups <= 0:
        return [(int(starts[0]), int(ends[-1]))]

    batch_slices: list[tuple[int, int]] = []
    for batch_start in range(0, starts.shape[0], int(batch_groups)):
        group_start = int(starts[batch_start])
        group_end_idx = min(batch_start + int(batch_groups) - 1, ends.shape[0] - 1)
        group_end = int(ends[group_end_idx])
        batch_slices.append((group_start, group_end))
    return batch_slices


def load_npz(path: str, batch_groups: int) -> DatasetTensors:
    z = np.load(path, allow_pickle=True)
    required = ["det_feat", "hist_feat", "hist_mask", "track_feat", "ctx_feat", "group_id", "label"]
    missing = [key for key in required if key not in z.files]
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
        batch_slices=build_batch_slices(group_id, batch_groups),
    )


def load_datasets(paths: Sequence[str], batch_groups: int, split_name: str) -> list[DatasetTensors]:
    datasets: list[DatasetTensors] = []
    print(f"[load] {split_name} shards={len(paths)}", flush=True)
    for shard_idx, path in enumerate(paths, start=1):
        data = load_npz(path, batch_groups=batch_groups)
        datasets.append(data)
        group_count = int(np.unique(data.group_id).shape[0]) if data.group_id.size > 0 else 0
        print(
            f"[load] {split_name} {shard_idx}/{len(paths)} path={path} "
            f"candidates={data.group_id.shape[0]} groups={group_count} batch_slices={len(data.batch_slices)}",
            flush=True,
        )
    return datasets


def group_segments(group_ids: torch.Tensor) -> list[tuple[int, int]]:
    if group_ids.numel() == 0:
        return []
    group_ids = group_ids.reshape(-1)
    change = torch.nonzero(group_ids[1:] != group_ids[:-1], as_tuple=False).flatten() + 1
    starts = torch.cat([group_ids.new_zeros((1,), dtype=torch.long), change.to(dtype=torch.long)])
    ends = torch.cat([change.to(dtype=torch.long), group_ids.new_tensor([group_ids.numel()], dtype=torch.long)])
    return [(int(s.item()), int(e.item())) for s, e in zip(starts, ends)]


class HACAV1(nn.Module):
    def __init__(
        self,
        version: str = "haca_v1",
        hist_hidden: int = 16,
        pair_hidden: int = 64,
        hist_gate_hidden: int = 8,
        anchor_alpha: float = 0.35,
        delta_scale: float = 1.5,
        decay_scales: Sequence[float] = (1.0, 2.0, 4.0),
        min_history: int = 3,
        use_set_encoder: bool = True,
        use_background: bool = True,
        use_hist_gate: bool = False,
        use_ood_gate: bool = False,
        ood_scale: float = 6.0,
    ) -> None:
        super().__init__()
        self.version = str(version)
        self.anchor_alpha = float(anchor_alpha)
        self.delta_scale = float(delta_scale)
        self.min_history = int(min_history)
        self.use_set_encoder = bool(use_set_encoder)
        self.use_background = bool(use_background)
        self.use_hist_gate = bool(use_hist_gate)
        self.use_ood_gate = bool(use_ood_gate)
        self.ood_scale = float(ood_scale)
        self.register_buffer("decay_scales", torch.tensor(list(decay_scales), dtype=torch.float32))
        self.register_buffer("token_mean", torch.zeros(len(HACA_PAIR_TOKEN_NAMES), dtype=torch.float32))
        self.register_buffer("token_std", torch.ones(len(HACA_PAIR_TOKEN_NAMES), dtype=torch.float32))
        self.register_buffer("ood_threshold", torch.tensor(float("inf"), dtype=torch.float32))

        self.hist_fc1 = nn.Linear(3, int(hist_hidden))
        self.hist_fc2 = nn.Linear(int(hist_hidden), int(hist_hidden))
        self.hist_attn = nn.Linear(int(hist_hidden), 1)

        self.pair_fc = nn.Linear(len(HACA_PAIR_TOKEN_NAMES), int(pair_hidden))
        self.set_fc = nn.Linear(int(pair_hidden) * 3, int(pair_hidden)) if self.use_set_encoder else None
        self.hist_gate_fc1 = nn.Linear(4, int(hist_gate_hidden)) if self.use_hist_gate else None
        self.hist_gate_fc2 = nn.Linear(int(hist_gate_hidden), 1) if self.use_hist_gate else None
        self.delta_head = nn.Linear(int(pair_hidden), 1)
        self.beta_head = nn.Linear(int(pair_hidden), 1)
        self.bg_head = nn.Linear(int(pair_hidden), 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.7)
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.beta_head.bias.fill_(-2.0)
            self.bg_head.bias.fill_(-2.0)
            if self.hist_gate_fc2 is not None:
                self.hist_gate_fc2.bias.fill_(-0.5)

    def set_token_stats(self, mean: np.ndarray, std: np.ndarray, threshold: float) -> None:
        with torch.no_grad():
            self.token_mean.copy_(torch.as_tensor(mean, dtype=self.token_mean.dtype, device=self.token_mean.device))
            self.token_std.copy_(torch.as_tensor(std, dtype=self.token_std.dtype, device=self.token_std.device))
            self.ood_threshold.fill_(float(threshold))

    def _temporal_summary(
        self,
        det_feat: torch.Tensor,
        hist_feat: torch.Tensor,
        hist_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        det_feat = F.normalize(det_feat, dim=-1)
        hist_feat = F.normalize(hist_feat, dim=-1)
        sims = (hist_feat * det_feat[:, None, :]).sum(dim=-1).clamp(min=-1.0, max=1.0)
        sims = 0.5 * (sims + 1.0)

        n, length = hist_mask.shape
        device = hist_mask.device
        mask_bool = hist_mask > 0.5
        counts = hist_mask.sum(dim=1).clamp(min=1.0)

        ages = torch.arange(length - 1, -1, -1, device=device, dtype=hist_feat.dtype).view(1, length).expand(n, length)
        age_log = torch.log1p(ages)
        delta = torch.zeros_like(sims)
        if length > 1:
            delta[:, 1:] = sims[:, 1:] - sims[:, :-1]

        step_feat = torch.stack([sims, age_log, delta], dim=-1)
        h = F.gelu(self.hist_fc1(step_feat))
        h = F.gelu(self.hist_fc2(h))
        attn_logits = self.hist_attn(h).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask_bool, -1e9)
        attn = torch.softmax(attn_logits, dim=1)
        attn = attn * hist_mask
        attn = attn / attn.sum(dim=1, keepdim=True).clamp(min=1e-6)

        temp_sim = (attn * sims).sum(dim=1)
        last_sim = sims[:, -1]
        masked_sims = sims.masked_fill(~mask_bool, -1e9)
        max_sim = masked_sims.max(dim=1).values.clamp(min=0.0, max=1.0)
        mean_sim = (sims * hist_mask).sum(dim=1) / counts
        std_sim = torch.sqrt((((sims - mean_sim[:, None]) ** 2) * hist_mask).sum(dim=1) / counts).clamp(min=0.0)
        return temp_sim, last_sim, max_sim, std_sim, sims

    def _masked_track_stats(
        self,
        hist_feat: torch.Tensor,
        hist_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hist_feat = F.normalize(hist_feat, dim=-1)
        counts = hist_mask.sum(dim=1)
        stability = []
        coherence = []
        hist_norm = []
        tau = self.decay_scales.to(device=hist_feat.device, dtype=hist_feat.dtype)

        for row in range(hist_feat.shape[0]):
            count = int(round(float(counts[row].item())))
            hist_norm.append(min(1.0, float(count) / float(max(self.min_history, 1))))
            if count <= 0:
                stability.append(1.0)
                coherence.append(1.0)
                continue

            valid_hist = hist_feat[row, -count:]
            last = valid_hist[-1]
            if count < 3:
                stability.append(1.0)
            else:
                delta2 = valid_hist[2:] - 2.0 * valid_hist[1:-1] + valid_hist[:-2]
                curvature = torch.sqrt(torch.mean(delta2 ** 2, dim=1)).mean()
                stability.append(float(torch.exp(-curvature).item()))

            if count < max(self.min_history, 1):
                coherence.append(1.0)
            else:
                ages = torch.arange(count - 1, -1, -1, device=hist_feat.device, dtype=hist_feat.dtype)
                proto_sims = []
                for scale in tau:
                    weight = torch.exp(-ages / scale.clamp(min=1e-4))
                    weight = weight / weight.sum().clamp(min=1e-6)
                    proto = F.normalize((weight[:, None] * valid_hist).sum(dim=0), dim=0)
                    proto_sim = ((proto * last).sum().clamp(min=-1.0, max=1.0) + 1.0) * 0.5
                    proto_sims.append(proto_sim)
                coherence.append(float(torch.stack(proto_sims).mean().item()))

        return (
            torch.tensor(stability, device=hist_feat.device, dtype=hist_feat.dtype),
            torch.tensor(coherence, device=hist_feat.device, dtype=hist_feat.dtype),
            torch.tensor(hist_norm, device=hist_feat.device, dtype=hist_feat.dtype),
        )

    def _heuristic_anchor(
        self,
        det_feat: torch.Tensor,
        hist_feat: torch.Tensor,
        hist_mask: torch.Tensor,
        track_feat: torch.Tensor,
        ctx_feat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        det_feat = F.normalize(det_feat, dim=-1)
        hist_feat = F.normalize(hist_feat, dim=-1)
        length = hist_feat.shape[1]
        hist_len = hist_mask.sum(dim=1)
        ages = torch.arange(length - 1, -1, -1, device=hist_feat.device, dtype=hist_feat.dtype)

        tau = self.decay_scales.to(device=hist_feat.device, dtype=hist_feat.dtype)
        weights = torch.exp(-ages[None, None, :] / tau[:, None, None].clamp(min=1e-4))
        weights = weights * hist_mask.unsqueeze(0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        protos = torch.einsum("knl,nld->knd", weights, hist_feat).permute(1, 0, 2)
        protos = F.normalize(protos, dim=-1)
        s_k = (protos * det_feat[:, None, :]).sum(dim=-1).clamp(min=-1.0, max=1.0)
        s_k = 0.5 * (s_k + 1.0)
        laplace_sim = s_k.mean(dim=1)
        last_feat = hist_feat[:, -1, :]
        last_sim = ((last_feat * det_feat).sum(dim=-1).clamp(min=-1.0, max=1.0) + 1.0) * 0.5
        laplace_sim = torch.where(hist_len < float(max(self.min_history, 1)), last_sim, laplace_sim)

        spatial_sim = ctx_feat[:, 0].clamp(min=0.0, max=1.0)
        motion_sim = ctx_feat[:, 1].clamp(min=0.0, max=1.0)
        det_score = ctx_feat[:, 2].clamp(min=0.0, max=1.0)
        stability, coherence, hist_norm = self._masked_track_stats(hist_feat=hist_feat, hist_mask=hist_mask)
        stability = stability.clamp(min=0.0, max=1.0)
        coherence = coherence.clamp(min=0.0, max=1.0)
        hist_norm = hist_norm.clamp(min=0.0, max=1.0)
        agreement = (1.0 - torch.abs(spatial_sim - laplace_sim)).clamp(min=0.0, max=1.0)

        pair_rel = 0.35 * stability + 0.35 * coherence + 0.30 * agreement
        pair_rel = pair_rel * (0.5 + 0.5 * det_score)
        pair_rel = pair_rel.clamp(min=0.0, max=1.0)
        appearance_sim = (1.0 - self.anchor_alpha) * spatial_sim + self.anchor_alpha * laplace_sim
        anchor_sim = (pair_rel * appearance_sim + (1.0 - pair_rel) * motion_sim).clamp(min=1e-4, max=1.0 - 1e-4)
        return {
            "anchor_sim": anchor_sim,
            "laplace_sim": laplace_sim,
            "appearance_sim": appearance_sim,
            "pair_rel": pair_rel,
            "spatial_sim": spatial_sim,
            "motion_sim": motion_sim,
            "det_score": det_score,
            "stability": stability,
            "coherence": coherence,
            "hist_norm": hist_norm,
        }

    def _hist_gate(
        self,
        gap_log: torch.Tensor,
        hist_norm: torch.Tensor,
        stability: torch.Tensor,
        coherence: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_hist_gate or self.hist_gate_fc1 is None or self.hist_gate_fc2 is None:
            return torch.ones_like(gap_log)
        feat = torch.stack([gap_log, hist_norm, stability, coherence], dim=-1)
        hidden = F.gelu(self.hist_gate_fc1(feat))
        return torch.sigmoid(self.hist_gate_fc2(hidden).squeeze(-1))

    def forward(
        self,
        det_feat: torch.Tensor,
        hist_feat: torch.Tensor,
        hist_mask: torch.Tensor,
        track_feat: torch.Tensor,
        ctx_feat: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        temp_sim, last_sim, max_sim, std_sim, _ = self._temporal_summary(det_feat, hist_feat, hist_mask)
        anchor = self._heuristic_anchor(det_feat, hist_feat, hist_mask, track_feat, ctx_feat)

        anchor_sim = anchor["anchor_sim"]
        spatial_sim = anchor["spatial_sim"]
        motion_sim = anchor["motion_sim"]
        det_score = anchor["det_score"]
        stability = anchor["stability"]
        coherence = anchor["coherence"]
        hist_norm = anchor["hist_norm"]

        anchor_z = torch.zeros_like(anchor_sim)
        anchor_margin = torch.zeros_like(anchor_sim)
        anchor_rank = torch.zeros_like(anchor_sim)
        segments = group_segments(group_ids)
        for start, end in segments:
            scores = anchor_sim[start:end]
            group_mean = scores.mean()
            group_std = scores.std(unbiased=False).clamp(min=1e-6)
            anchor_z[start:end] = (scores - group_mean) / group_std
            if end - start == 1:
                anchor_margin[start:end] = scores.clamp(min=0.0, max=1.0)
                anchor_rank[start:end] = 0.0
                continue
            order = torch.argsort(scores, descending=True)
            ranks = torch.empty_like(order)
            ranks[order] = torch.arange(order.numel(), device=scores.device, dtype=order.dtype)
            anchor_rank[start:end] = ranks.float() / float(order.numel())

            top_vals, top_idx = torch.topk(scores, k=2)
            best_idx = int(top_idx[0].item())
            best_val = top_vals[0]
            second_val = top_vals[1]
            idx = torch.arange(scores.numel(), device=scores.device)
            max_other = torch.where(idx == best_idx, second_val, best_val)
            anchor_margin[start:end] = scores - max_other

        pair_tokens = torch.stack(
            [
                anchor_sim,
                spatial_sim,
                motion_sim,
                temp_sim,
                last_sim,
                max_sim,
                std_sim,
                track_feat[:, 0],
                hist_norm,
                stability,
                coherence,
                anchor_z,
                anchor_margin,
                anchor_rank,
                det_score,
            ],
            dim=-1,
        )
        pair_embed = F.gelu(self.pair_fc(pair_tokens))

        set_embed = pair_embed.clone()
        if self.use_set_encoder and self.set_fc is not None:
            set_chunks = []
            for start, end in segments:
                group_embed = pair_embed[start:end]
                mean_embed = group_embed.mean(dim=0, keepdim=True).expand_as(group_embed)
                max_embed = group_embed.max(dim=0, keepdim=True).values.expand_as(group_embed)
                set_in = torch.cat([group_embed, mean_embed, max_embed], dim=-1)
                set_chunks.append(F.gelu(self.set_fc(set_in)))
            set_embed = torch.cat(set_chunks, dim=0) if set_chunks else set_embed

        delta = self.delta_head(set_embed).squeeze(-1)
        beta_pred = torch.sigmoid(self.beta_head(set_embed).squeeze(-1))
        beta_hist = self._hist_gate(
            gap_log=track_feat[:, 0],
            hist_norm=hist_norm,
            stability=stability,
            coherence=coherence,
        )
        beta_ood = torch.ones_like(beta_pred)
        beta = (beta_pred * beta_hist * beta_ood).clamp(min=0.0, max=1.0)
        bg_prob = torch.zeros_like(anchor_sim)
        if self.use_background:
            for start, end in segments:
                group_embed = set_embed[start:end]
                group_bg = torch.sigmoid(self.bg_head(group_embed.mean(dim=0, keepdim=True))).reshape(())
                bg_prob[start:end] = group_bg

        anchor_logit = torch.logit(anchor_sim.clamp(min=1e-4, max=1.0 - 1e-4), eps=1e-4)
        refined_sim = torch.sigmoid(anchor_logit + self.delta_scale * torch.tanh(delta))
        final_sim = ((1.0 - beta) * anchor_sim + beta * refined_sim).clamp(min=1e-4, max=1.0 - 1e-4)
        if self.use_background:
            final_sim = ((1.0 - bg_prob) * final_sim).clamp(min=1e-4, max=1.0 - 1e-4)

        return {
            "anchor_sim": anchor_sim,
            "laplace_sim": anchor["laplace_sim"],
            "appearance_sim": anchor["appearance_sim"],
            "pair_rel": anchor["pair_rel"],
            "spatial_sim": spatial_sim,
            "motion_sim": motion_sim,
            "temp_sim": temp_sim,
            "last_sim": last_sim,
            "max_sim": max_sim,
            "std_sim": std_sim,
            "pair_tokens": pair_tokens,
            "delta": delta,
            "beta": beta,
            "beta_pred": beta_pred,
            "beta_hist": beta_hist,
            "beta_ood": beta_ood,
            "beta_eff": beta,
            "bg_prob": bg_prob,
            "refined_sim": refined_sim,
            "final_sim": final_sim,
        }


def batch_losses(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float,
    margin: float,
    safe_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    zero = outputs["final_sim"].new_tensor(0.0)
    list_terms = []
    margin_terms = []
    bg_terms = []
    safe_terms = []
    res_terms = []
    top1_hits = []

    final_sim = outputs["final_sim"]
    anchor_sim = outputs["anchor_sim"]
    beta = outputs["beta_eff"]
    delta = outputs["delta"]
    bg_prob = outputs["bg_prob"]

    for start, end in group_segments(group_ids):
        labels_g = labels[start:end]
        final_g = final_sim[start:end]
        anchor_g = anchor_sim[start:end]
        beta_g = beta[start:end]
        delta_g = delta[start:end]
        bg_g = bg_prob[start]
        is_bg = bool(torch.all(labels_g < 0.5))
        bg_target = bg_g.new_tensor(1.0 if is_bg else 0.0)
        bg_terms.append(F.binary_cross_entropy(bg_g.view(1), bg_target.view(1)))
        res_terms.append(torch.mean(torch.abs(beta_g * torch.tanh(delta_g))))

        logits = torch.logit(final_g.clamp(min=1e-4, max=1.0 - 1e-4), eps=1e-4) / max(float(temperature), 1e-4)
        if is_bg:
            bg_terms.append(-torch.log((1.0 - final_g.max()).clamp(min=1e-4)))
            continue

        pos_idx = int(torch.argmax(labels_g).item())
        list_terms.append(F.cross_entropy(logits.view(1, -1), torch.tensor([pos_idx], device=logits.device)))
        top1_hits.append(float(torch.argmax(logits).item() == pos_idx))

        if end - start > 1:
            neg_mask = torch.ones_like(labels_g, dtype=torch.bool)
            neg_mask[pos_idx] = False
            z_pos = logits[pos_idx]
            z_neg = logits[neg_mask].max()
            margin_terms.append(F.relu(z_pos.new_tensor(float(margin)) - z_pos + z_neg))

            anchor_pred = int(torch.argmax(anchor_g).item())
            if anchor_pred == pos_idx:
                anchor_neg = anchor_g[neg_mask].max()
                if float(anchor_g[pos_idx] - anchor_neg) > float(safe_margin):
                    safe_terms.append(torch.mean((final_g - anchor_g) ** 2))
        else:
            anchor_pred = 0
            if anchor_pred == pos_idx and float(anchor_g[0]) > float(safe_margin):
                safe_terms.append(torch.mean((final_g - anchor_g) ** 2))

    list_loss = torch.stack(list_terms).mean() if list_terms else zero
    margin_loss = torch.stack(margin_terms).mean() if margin_terms else zero
    bg_loss = torch.stack(bg_terms).mean() if bg_terms else zero
    safe_loss = torch.stack(safe_terms).mean() if safe_terms else zero
    res_loss = torch.stack(res_terms).mean() if res_terms else zero
    top1 = float(np.mean(top1_hits)) if top1_hits else 0.0
    return list_loss, margin_loss, bg_loss, safe_loss, res_loss, top1


def apply_shift_corruption(
    det_feat: torch.Tensor,
    hist_feat: torch.Tensor,
    hist_mask: torch.Tensor,
    track_feat: torch.Tensor,
    ctx_feat: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    det_feat_c = det_feat.clone()
    hist_feat_c = hist_feat.clone()
    hist_mask_c = hist_mask.clone()
    track_feat_c = track_feat.clone()
    ctx_feat_c = ctx_feat.clone()

    if args.corrupt_feat_noise > 0:
        det_feat_c = det_feat_c + torch.randn_like(det_feat_c) * float(args.corrupt_feat_noise)
        hist_feat_c = hist_feat_c + torch.randn_like(hist_feat_c) * float(args.corrupt_feat_noise) * 0.75

    if args.corrupt_score_noise > 0:
        score_scale = 1.0 + (torch.rand_like(ctx_feat_c[:, :3]) * 2.0 - 1.0) * float(args.corrupt_score_noise)
        score_bias = torch.randn_like(ctx_feat_c[:, :3]) * float(args.corrupt_score_noise) * 0.5
        ctx_feat_c[:, :3] = (ctx_feat_c[:, :3] * score_scale + score_bias).clamp(min=0.0, max=1.0)
        track_feat_c[:, 0] = (track_feat_c[:, 0] + torch.rand_like(track_feat_c[:, 0]) * float(args.corrupt_score_noise)).clamp(min=0.0)

    min_ratio = float(max(0.05, min(args.corrupt_history_min_ratio, 1.0)))
    counts = torch.round(hist_mask_c.sum(dim=1)).to(dtype=torch.long)
    for row_idx, count in enumerate(counts.tolist()):
        if count <= 1:
            continue
        if random.random() > 0.7:
            continue
        min_keep = max(1, int(round(float(count) * min_ratio)))
        keep = random.randint(min_keep, count)
        hist_mask_c[row_idx].zero_()
        hist_mask_c[row_idx, -keep:] = 1.0

    return det_feat_c, hist_feat_c, hist_mask_c, track_feat_c, ctx_feat_c


def shift_fallback_loss(outputs_corr: dict[str, torch.Tensor]) -> torch.Tensor:
    beta_target = torch.zeros_like(outputs_corr["beta_eff"])
    beta_loss = F.binary_cross_entropy(outputs_corr["beta_eff"], beta_target)
    anchor_align = torch.mean(torch.abs(outputs_corr["final_sim"] - outputs_corr["anchor_sim"]))
    return beta_loss + anchor_align


def run_dataset(
    model: HACAV1,
    datasets: Sequence[DatasetTensors],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    losses: list[float] = []
    top1s: list[float] = []

    for shard_idx, data in enumerate(datasets, start=1):
        if is_train:
            random.shuffle(data.batch_slices)
        for batch_idx, (start, end) in enumerate(data.batch_slices, start=1):
            sl = slice(start, end)
            det_feat = torch.from_numpy(data.det_feat[sl]).to(device=device, dtype=torch.float32)
            hist_feat = torch.from_numpy(data.hist_feat[sl]).to(device=device, dtype=torch.float32)
            hist_mask = torch.from_numpy(data.hist_mask[sl]).to(device=device, dtype=torch.float32)
            track_feat = torch.from_numpy(data.track_feat[sl]).to(device=device, dtype=torch.float32)
            ctx_feat = torch.from_numpy(data.ctx_feat[sl]).to(device=device, dtype=torch.float32)
            group_ids = torch.from_numpy(data.group_id[sl]).to(device=device, dtype=torch.long)
            labels = torch.from_numpy(data.label[sl]).to(device=device, dtype=torch.float32)

            with torch.set_grad_enabled(is_train):
                outputs = model(
                    det_feat=det_feat,
                    hist_feat=hist_feat,
                    hist_mask=hist_mask,
                    track_feat=track_feat,
                    ctx_feat=ctx_feat,
                    group_ids=group_ids,
                )
                list_loss, margin_loss, bg_loss, safe_loss, res_loss, top1 = batch_losses(
                    outputs=outputs,
                    labels=labels,
                    group_ids=group_ids,
                    temperature=args.temperature,
                    margin=args.margin,
                    safe_margin=args.safe_margin,
                )
                loss = (
                    list_loss
                    + args.loss_margin_weight * margin_loss
                    + args.loss_bg_weight * bg_loss
                    + args.loss_safe_weight * safe_loss
                    + args.loss_res_weight * res_loss
                )
                if (
                    is_train
                    and args.loss_shift_weight > 0.0
                    and random.random() < float(max(0.0, min(args.shift_batch_prob, 1.0)))
                ):
                    det_feat_c, hist_feat_c, hist_mask_c, track_feat_c, ctx_feat_c = apply_shift_corruption(
                        det_feat=det_feat,
                        hist_feat=hist_feat,
                        hist_mask=hist_mask,
                        track_feat=track_feat,
                        ctx_feat=ctx_feat,
                        args=args,
                    )
                    outputs_corr = model(
                        det_feat=det_feat_c,
                        hist_feat=hist_feat_c,
                        hist_mask=hist_mask_c,
                        track_feat=track_feat_c,
                        ctx_feat=ctx_feat_c,
                        group_ids=group_ids,
                    )
                    loss = loss + args.loss_shift_weight * shift_fallback_loss(outputs_corr)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
                optimizer.step()

            losses.append(float(loss.item()))
            top1s.append(float(top1))

        print(
            f"[{'train' if is_train else 'eval'}] shard {shard_idx}/{len(datasets)} "
            f"path={data.path} avg_loss={np.mean(losses):.6f} avg_top1={np.mean(top1s) if top1s else 0.0:.4f}",
            flush=True,
        )
    return float(np.mean(losses)) if losses else float("inf"), float(np.mean(top1s)) if top1s else 0.0


def fit_token_stats(
    model: HACAV1,
    datasets: Sequence[DatasetTensors],
    device: torch.device,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    total_count = 0
    sum_token = None
    sumsq_token = None

    model.eval()
    with torch.no_grad():
        for data in datasets:
            for start, end in data.batch_slices:
                sl = slice(start, end)
                outputs = model(
                    det_feat=torch.from_numpy(data.det_feat[sl]).to(device=device, dtype=torch.float32),
                    hist_feat=torch.from_numpy(data.hist_feat[sl]).to(device=device, dtype=torch.float32),
                    hist_mask=torch.from_numpy(data.hist_mask[sl]).to(device=device, dtype=torch.float32),
                    track_feat=torch.from_numpy(data.track_feat[sl]).to(device=device, dtype=torch.float32),
                    ctx_feat=torch.from_numpy(data.ctx_feat[sl]).to(device=device, dtype=torch.float32),
                    group_ids=torch.from_numpy(data.group_id[sl]).to(device=device, dtype=torch.long),
                )
                tokens = outputs["pair_tokens"].detach().cpu().numpy().astype(np.float64)
                if tokens.size == 0:
                    continue
                if sum_token is None:
                    sum_token = tokens.sum(axis=0)
                    sumsq_token = np.square(tokens).sum(axis=0)
                else:
                    sum_token += tokens.sum(axis=0)
                    sumsq_token += np.square(tokens).sum(axis=0)
                total_count += tokens.shape[0]

    if total_count <= 0 or sum_token is None or sumsq_token is None:
        dim = len(HACA_PAIR_TOKEN_NAMES)
        return np.zeros((dim,), dtype=np.float32), np.ones((dim,), dtype=np.float32), float("inf")

    mean = sum_token / float(total_count)
    var = np.maximum((sumsq_token / float(total_count)) - np.square(mean), 1e-6)
    std = np.sqrt(var)

    score_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for data in datasets:
            for start, end in data.batch_slices:
                sl = slice(start, end)
                outputs = model(
                    det_feat=torch.from_numpy(data.det_feat[sl]).to(device=device, dtype=torch.float32),
                    hist_feat=torch.from_numpy(data.hist_feat[sl]).to(device=device, dtype=torch.float32),
                    hist_mask=torch.from_numpy(data.hist_mask[sl]).to(device=device, dtype=torch.float32),
                    track_feat=torch.from_numpy(data.track_feat[sl]).to(device=device, dtype=torch.float32),
                    ctx_feat=torch.from_numpy(data.ctx_feat[sl]).to(device=device, dtype=torch.float32),
                    group_ids=torch.from_numpy(data.group_id[sl]).to(device=device, dtype=torch.long),
                )
                tokens = outputs["pair_tokens"].detach().cpu().numpy().astype(np.float32)
                if tokens.size == 0:
                    continue
                z = (tokens - mean.astype(np.float32)[None, :]) / np.clip(std.astype(np.float32)[None, :], 1e-6, None)
                score_chunks.append(np.sqrt(np.mean(np.square(z), axis=1)).astype(np.float32))

    if not score_chunks:
        threshold = float("inf")
    else:
        threshold = float(np.quantile(np.concatenate(score_chunks, axis=0), min(max(float(quantile), 0.5), 0.999)))

    return mean.astype(np.float32), std.astype(np.float32), threshold


def export_checkpoint(model: HACAV1, out_path: Path, max_history: int) -> None:
    state = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in model.state_dict().items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        version=np.asarray([model.version], dtype=object),
        pair_token_names=np.asarray(HACA_PAIR_TOKEN_NAMES, dtype=object),
        anchor_alpha=np.asarray([model.anchor_alpha], dtype=np.float32),
        delta_scale=np.asarray([model.delta_scale], dtype=np.float32),
        min_history=np.asarray([model.min_history], dtype=np.int32),
        max_history=np.asarray([int(max_history)], dtype=np.int32),
        decay_scales=model.decay_scales.detach().cpu().numpy().astype(np.float32),
        use_set_encoder=np.asarray([1 if model.use_set_encoder else 0], dtype=np.int32),
        use_background=np.asarray([1 if model.use_background else 0], dtype=np.int32),
        use_hist_gate=np.asarray([1 if model.use_hist_gate else 0], dtype=np.int32),
        use_ood_gate=np.asarray([1 if model.use_ood_gate else 0], dtype=np.int32),
        ood_scale=np.asarray([model.ood_scale], dtype=np.float32),
        token_mean=model.token_mean.detach().cpu().numpy().astype(np.float32),
        token_std=model.token_std.detach().cpu().numpy().astype(np.float32),
        ood_threshold=model.ood_threshold.detach().cpu().numpy().astype(np.float32),
        W_hist1=state["hist_fc1.weight"].T,
        b_hist1=state["hist_fc1.bias"],
        W_hist2=state["hist_fc2.weight"].T,
        b_hist2=state["hist_fc2.bias"],
        W_hist_attn=state["hist_attn.weight"].T,
        b_hist_attn=state["hist_attn.bias"],
        W_pair=state["pair_fc.weight"].T,
        b_pair=state["pair_fc.bias"],
        W_set=state["set_fc.weight"].T if model.set_fc is not None else np.zeros((0, 0), dtype=np.float32),
        b_set=state["set_fc.bias"] if model.set_fc is not None else np.zeros((0,), dtype=np.float32),
        W_hist_gate1=state["hist_gate_fc1.weight"].T if model.hist_gate_fc1 is not None else np.zeros((0, 0), dtype=np.float32),
        b_hist_gate1=state["hist_gate_fc1.bias"] if model.hist_gate_fc1 is not None else np.zeros((0,), dtype=np.float32),
        W_hist_gate2=state["hist_gate_fc2.weight"].T if model.hist_gate_fc2 is not None else np.zeros((0, 0), dtype=np.float32),
        b_hist_gate2=state["hist_gate_fc2.bias"] if model.hist_gate_fc2 is not None else np.zeros((0,), dtype=np.float32),
        W_delta=state["delta_head.weight"].T,
        b_delta=state["delta_head.bias"],
        W_beta=state["beta_head.weight"].T,
        b_beta=state["beta_head.bias"],
        W_bg=state["bg_head.weight"].T,
        b_bg=state["bg_head.bias"],
    )
    print(f"[saved] {out_path}", flush=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    train_sets = load_datasets(args.train_npz, batch_groups=args.batch_groups, split_name="train")
    val_sets = load_datasets(args.val_npz, batch_groups=args.batch_groups, split_name="val") if args.val_npz else []
    if not train_sets:
        raise ValueError("No training shards were provided.")

    inferred_max_history = int(train_sets[0].hist_feat.shape[1])
    max_history = int(args.max_history) if int(args.max_history) > 0 else inferred_max_history

    use_hist_gate = args.version == "haca_v2" and not args.disable_hist_gate
    use_ood_gate = args.version == "haca_v2" and not args.disable_ood_gate

    model = HACAV1(
        version=args.version,
        hist_hidden=args.hist_hidden,
        pair_hidden=args.pair_hidden,
        hist_gate_hidden=args.hist_gate_hidden,
        anchor_alpha=args.anchor_alpha,
        delta_scale=args.delta_scale,
        decay_scales=args.laplace_decay_scales,
        min_history=args.min_history,
        use_set_encoder=not args.disable_set_encoder,
        use_background=not args.disable_background,
        use_hist_gate=use_hist_gate,
        use_ood_gate=use_ood_gate,
        ood_scale=args.ood_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metric = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_top1 = run_dataset(model, train_sets, device, optimizer, args)
        if val_sets:
            with torch.no_grad():
                val_loss, val_top1 = run_dataset(model, val_sets, device, None, args)
            metric = val_loss
        else:
            val_loss, val_top1 = train_loss, train_top1
            metric = train_loss

        print(
            f"[epoch {epoch:02d}] train_loss={train_loss:.6f} train_top1={train_top1:.4f} "
            f"val_loss={val_loss:.6f} val_top1={val_top1:.4f}",
            flush=True,
        )

        if metric < best_metric:
            best_metric = metric
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                print("[early-stop]", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if model.use_ood_gate:
        token_mean, token_std, ood_threshold = fit_token_stats(
            model=model,
            datasets=train_sets,
            device=device,
            quantile=args.ood_quantile,
        )
        model.set_token_stats(token_mean, token_std, ood_threshold)
        print(
            f"[token-stats] threshold={ood_threshold:.6f} mean_norm={float(np.linalg.norm(token_mean)):.6f} "
            f"std_mean={float(np.mean(token_std)):.6f}",
            flush=True,
        )
    export_checkpoint(model=model, out_path=Path(args.out_npz), max_history=max_history)


if __name__ == "__main__":
    main()
