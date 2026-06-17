#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from train_haca_v1_from_gt_tracks import (
    DatasetTensors,
    HACA_PAIR_TOKEN_NAMES,
    HACAV1,
    apply_shift_corruption,
    group_segments,
    infer_max_history,
    iter_loaded_datasets,
    set_seed,
)


DUEL_EXTRA_DIMS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HACA-v3 / ATCR from GT pseudo-track NPZ groups.")
    parser.add_argument("--version", type=str, default="haca_v3")
    parser.add_argument("--base-npz", required=True, help="Frozen HACA-v2 checkpoint used as the ATCR base.")
    parser.add_argument("--train-npz", nargs="+", required=True)
    parser.add_argument("--val-npz", nargs="*", default=[])
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=12)
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
    parser.add_argument("--max-history", type=int, default=0)
    parser.add_argument("--comp-hidden", type=int, default=64)
    parser.add_argument("--comp-topk", type=int, default=3)
    parser.add_argument(
        "--positive-injection-prob",
        type=float,
        default=1.0,
        help="During training only, probability of forcing the positive candidate into the selected top-k when it falls outside. Set <1.0 to reduce the train/inference gap.",
    )
    parser.add_argument("--comp-margin-quantile", type=float, default=0.35)
    parser.add_argument("--comp-margin-temperature", type=float, default=0.03)
    parser.add_argument("--comp-delta-scale", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--duel-margin", type=float, default=0.20)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--safe-margin", type=float, default=0.10)
    parser.add_argument("--loss-bg-weight", type=float, default=0.0)
    parser.add_argument("--loss-margin-weight", type=float, default=0.0)
    parser.add_argument("--loss-duel-weight", type=float, default=0.50)
    parser.add_argument("--loss-safe-weight", type=float, default=0.20)
    parser.add_argument("--loss-res-weight", type=float, default=0.0)
    parser.add_argument("--loss-shift-weight", type=float, default=0.0)
    parser.add_argument("--shift-batch-prob", type=float, default=0.0)
    parser.add_argument("--ood-scale", type=float, default=0.0)
    parser.add_argument("--ood-quantile", type=float, default=0.95)
    parser.add_argument("--corrupt-feat-noise", type=float, default=0.0)
    parser.add_argument("--corrupt-score-noise", type=float, default=0.0)
    parser.add_argument("--corrupt-history-min-ratio", type=float, default=0.35)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--disable-set-encoder", action="store_true")
    parser.add_argument("--disable-background", action="store_true")
    parser.add_argument("--disable-hist-gate", action="store_true")
    parser.add_argument("--disable-ood-gate", action="store_true")
    return parser.parse_args()


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)


def _load_base_model(base_npz: str, device: torch.device) -> tuple[HACAV1, dict[str, np.ndarray], int]:
    z = np.load(base_npz, allow_pickle=True)
    data = {key: z[key] for key in z.files}

    version = str(np.asarray(data["version"]).reshape(-1)[0]) if "version" in data else "haca_v2"
    hist_hidden = int(np.asarray(data["W_hist1"]).shape[1])
    pair_hidden = int(np.asarray(data["W_pair"]).shape[1])
    use_set_encoder = bool(int(np.asarray(data["use_set_encoder"], dtype=np.int32).reshape(-1)[0]))
    use_background = bool(int(np.asarray(data["use_background"], dtype=np.int32).reshape(-1)[0]))
    use_hist_gate = bool(int(np.asarray(data["use_hist_gate"], dtype=np.int32).reshape(-1)[0])) if "use_hist_gate" in data else False
    use_ood_gate = bool(int(np.asarray(data["use_ood_gate"], dtype=np.int32).reshape(-1)[0])) if "use_ood_gate" in data else False
    hist_gate_hidden = int(np.asarray(data["W_hist_gate1"]).shape[1]) if use_hist_gate and np.asarray(data["W_hist_gate1"]).size > 0 else 8

    model = HACAV1(
        version=version,
        hist_hidden=hist_hidden,
        pair_hidden=pair_hidden,
        hist_gate_hidden=hist_gate_hidden,
        anchor_alpha=float(np.asarray(data["anchor_alpha"], dtype=np.float32).reshape(-1)[0]),
        delta_scale=float(np.asarray(data["delta_scale"], dtype=np.float32).reshape(-1)[0]),
        decay_scales=np.asarray(data["decay_scales"], dtype=np.float32).reshape(-1).tolist(),
        min_history=int(np.asarray(data["min_history"], dtype=np.int32).reshape(-1)[0]),
        use_set_encoder=use_set_encoder,
        use_background=use_background,
        use_hist_gate=use_hist_gate,
        use_ood_gate=use_ood_gate,
        ood_scale=float(np.asarray(data["ood_scale"], dtype=np.float32).reshape(-1)[0]) if "ood_scale" in data else 0.0,
    ).to(device)

    state = model.state_dict()
    state["hist_fc1.weight"] = torch.from_numpy(np.asarray(data["W_hist1"], dtype=np.float32).T)
    state["hist_fc1.bias"] = torch.from_numpy(np.asarray(data["b_hist1"], dtype=np.float32))
    state["hist_fc2.weight"] = torch.from_numpy(np.asarray(data["W_hist2"], dtype=np.float32).T)
    state["hist_fc2.bias"] = torch.from_numpy(np.asarray(data["b_hist2"], dtype=np.float32))
    state["hist_attn.weight"] = torch.from_numpy(np.asarray(data["W_hist_attn"], dtype=np.float32).T)
    state["hist_attn.bias"] = torch.from_numpy(np.asarray(data["b_hist_attn"], dtype=np.float32))
    state["pair_fc.weight"] = torch.from_numpy(np.asarray(data["W_pair"], dtype=np.float32).T)
    state["pair_fc.bias"] = torch.from_numpy(np.asarray(data["b_pair"], dtype=np.float32))
    if model.set_fc is not None:
        state["set_fc.weight"] = torch.from_numpy(np.asarray(data["W_set"], dtype=np.float32).T)
        state["set_fc.bias"] = torch.from_numpy(np.asarray(data["b_set"], dtype=np.float32))
    if model.hist_gate_fc1 is not None:
        state["hist_gate_fc1.weight"] = torch.from_numpy(np.asarray(data["W_hist_gate1"], dtype=np.float32).T)
        state["hist_gate_fc1.bias"] = torch.from_numpy(np.asarray(data["b_hist_gate1"], dtype=np.float32))
        state["hist_gate_fc2.weight"] = torch.from_numpy(np.asarray(data["W_hist_gate2"], dtype=np.float32).T)
        state["hist_gate_fc2.bias"] = torch.from_numpy(np.asarray(data["b_hist_gate2"], dtype=np.float32))
    state["delta_head.weight"] = torch.from_numpy(np.asarray(data["W_delta"], dtype=np.float32).T)
    state["delta_head.bias"] = torch.from_numpy(np.asarray(data["b_delta"], dtype=np.float32))
    state["beta_head.weight"] = torch.from_numpy(np.asarray(data["W_beta"], dtype=np.float32).T)
    state["beta_head.bias"] = torch.from_numpy(np.asarray(data["b_beta"], dtype=np.float32))
    state["bg_head.weight"] = torch.from_numpy(np.asarray(data["W_bg"], dtype=np.float32).T)
    state["bg_head.bias"] = torch.from_numpy(np.asarray(data["b_bg"], dtype=np.float32))
    model.load_state_dict(state, strict=True)

    if "token_mean" in data:
        model.token_mean.copy_(torch.from_numpy(np.asarray(data["token_mean"], dtype=np.float32)).to(device))
    if "token_std" in data:
        model.token_std.copy_(torch.from_numpy(np.asarray(data["token_std"], dtype=np.float32)).to(device))
    if "ood_threshold" in data:
        model.ood_threshold.copy_(torch.from_numpy(np.asarray(data["ood_threshold"], dtype=np.float32)).to(device))

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    max_history = int(np.asarray(data["max_history"], dtype=np.int32).reshape(-1)[0]) if "max_history" in data else 0
    return model, data, max_history


class ATCRModel(nn.Module):
    def __init__(
        self,
        base_model: HACAV1,
        comp_hidden: int,
        comp_topk: int,
        comp_margin_threshold: float,
        comp_margin_temperature: float,
        comp_delta_scale: float,
        positive_injection_prob: float,
    ) -> None:
        super().__init__()
        self.base = base_model
        self.comp_hidden = int(comp_hidden)
        self.comp_topk = int(comp_topk)
        self.comp_margin_threshold = float(comp_margin_threshold)
        self.comp_margin_temperature = float(comp_margin_temperature)
        self.comp_delta_scale = float(comp_delta_scale)
        self.positive_injection_prob = float(np.clip(positive_injection_prob, 0.0, 1.0))

        duel_dim = len(HACA_PAIR_TOKEN_NAMES) * 3 + DUEL_EXTRA_DIMS
        comp_in_dim = len(HACA_PAIR_TOKEN_NAMES) + self.comp_hidden + DUEL_EXTRA_DIMS

        self.duel_fc1 = nn.Linear(duel_dim, self.comp_hidden)
        self.duel_fc2 = nn.Linear(self.comp_hidden, self.comp_hidden)
        self.attn_fc = nn.Linear(self.comp_hidden, 1)
        self.comp_fc1 = nn.Linear(comp_in_dim, self.comp_hidden)
        self.comp_fc2 = nn.Linear(self.comp_hidden, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.7)
                nn.init.zeros_(module.bias)

    def _base_outputs(
        self,
        det_feat: torch.Tensor,
        hist_feat: torch.Tensor,
        hist_mask: torch.Tensor,
        track_feat: torch.Tensor,
        ctx_feat: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            base = self.base(
                det_feat=det_feat,
                hist_feat=hist_feat,
                hist_mask=hist_mask,
                track_feat=track_feat,
                ctx_feat=ctx_feat,
                group_ids=group_ids,
            )
        base = {key: value.detach() for key, value in base.items()}
        base_prebg = (((1.0 - base["beta_eff"]) * base["anchor_sim"]) + base["beta_eff"] * base["refined_sim"]).clamp(min=1e-4, max=1.0 - 1e-4)
        bg_prob = base["bg_prob"] if self.base.use_background else torch.zeros_like(base_prebg)
        base_final = ((1.0 - bg_prob) * base_prebg).clamp(min=1e-4, max=1.0 - 1e-4)
        base["base_prebg_sim"] = base_prebg
        base["base_final_sim"] = base_final
        base["base_logit"] = torch.logit(base_prebg, eps=1e-4)
        return base

    def _select_topk(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor | None,
        force_positive: bool,
    ) -> torch.Tensor:
        k = min(max(int(self.comp_topk), 1), int(logits.numel()))
        if k <= 0:
            return torch.empty((0,), device=logits.device, dtype=torch.long)
        topk = torch.topk(logits, k=k).indices
        if force_positive and labels is not None and labels.numel() > 0 and bool(torch.any(labels > 0.5)):
            pos_idx = int(torch.argmax(labels).item())
            if not bool(torch.any(topk == pos_idx)):
                if topk.numel() == 1:
                    topk = torch.tensor([pos_idx], device=logits.device, dtype=torch.long)
                else:
                    topk = torch.cat([topk[:-1], topk.new_tensor([pos_idx])], dim=0)
        if topk.numel() > 1:
            topk = topk[torch.argsort(logits[topk], descending=True)]
        return topk

    def forward(
        self,
        det_feat: torch.Tensor,
        hist_feat: torch.Tensor,
        hist_mask: torch.Tensor,
        track_feat: torch.Tensor,
        ctx_feat: torch.Tensor,
        group_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        base = self._base_outputs(det_feat, hist_feat, hist_mask, track_feat, ctx_feat, group_ids)
        pair_tokens = base["pair_tokens"]
        base_prebg = base["base_prebg_sim"]
        base_final = base["base_final_sim"]
        base_logits = base["base_logit"]
        bg_prob = base["bg_prob"] if self.base.use_background else torch.zeros_like(base_prebg)

        comp_logits = base_logits.clone()
        comp_prebg = base_prebg.clone()
        final_sim = base_final.clone()

        comp_active = torch.zeros_like(base_prebg)
        comp_margin = torch.zeros_like(base_prebg)
        comp_entropy = torch.zeros_like(base_prebg)
        comp_residual = torch.zeros_like(base_prebg)
        comp_topk_mask = torch.zeros_like(base_prebg)
        comp_rank_before = torch.zeros_like(base_prebg)
        comp_rank_after = torch.zeros_like(base_prebg)
        group_ambiguous = torch.zeros_like(base_prebg)

        for start, end in group_segments(group_ids):
            logits_g = base_logits[start:end]
            scores_g = base_prebg[start:end]
            labels_g = labels[start:end] if labels is not None else None
            force_positive = bool(self.training and random.random() < self.positive_injection_prob)
            topk_idx = self._select_topk(logits_g, labels_g, force_positive=force_positive)
            abs_idx = torch.arange(start, end, device=logits_g.device, dtype=torch.long)
            topk_abs = abs_idx[topk_idx]
            if topk_idx.numel() == 0:
                continue

            order_before = torch.argsort(logits_g, descending=True)
            ranks_before = torch.empty_like(order_before)
            ranks_before[order_before] = torch.arange(order_before.numel(), device=logits_g.device, dtype=order_before.dtype)
            comp_rank_before[start:end] = ranks_before.float() / float(max(order_before.numel(), 1))

            top_scores = scores_g[topk_idx]
            margin = top_scores[0] if top_scores.numel() == 1 else top_scores[0] - top_scores[1]
            probs = torch.softmax(logits_g[topk_idx], dim=0)
            entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-12)))
            trust = torch.mean(base["beta_hist"][topk_abs] * base["beta_ood"][topk_abs])
            bg_scalar = bg_prob[start]
            activation = ((1.0 - bg_scalar) * trust * torch.sigmoid((margin.new_tensor(self.comp_margin_threshold) - margin) / max(self.comp_margin_temperature, 1e-4))).detach()
            ambiguous = False
            if labels_g is not None and bool(torch.any(labels_g > 0.5)):
                pos_idx = int(torch.argmax(labels_g).item())
                ambiguous = bool(int(torch.argmax(logits_g).item()) != pos_idx or float(margin.item()) < float(self.comp_margin_threshold))
            elif topk_idx.numel() > 1:
                ambiguous = bool(float(margin.item()) < float(self.comp_margin_threshold))
            group_ambiguous[start:end] = 1.0 if ambiguous else 0.0

            comp_active[start:end] = activation
            comp_margin[start:end] = margin.detach()
            comp_entropy[start:end] = entropy.detach()
            comp_topk_mask[topk_abs] = 1.0

            if topk_idx.numel() <= 1 or float(activation.item()) <= 1e-6:
                order_after = torch.argsort(comp_logits[start:end], descending=True)
                ranks_after = torch.empty_like(order_after)
                ranks_after[order_after] = torch.arange(order_after.numel(), device=logits_g.device, dtype=order_after.dtype)
                comp_rank_after[start:end] = ranks_after.float() / float(max(order_after.numel(), 1))
                continue

            residuals = []
            for idx_i in topk_idx.tolist():
                rivals = [idx_k for idx_k in topk_idx.tolist() if idx_k != idx_i]
                if not rivals:
                    residuals.append(logits_g.new_tensor(0.0))
                    continue

                xi = pair_tokens[start + idx_i]
                xk = pair_tokens[start + torch.as_tensor(rivals, device=pair_tokens.device, dtype=torch.long)]
                xi_rep = xi.unsqueeze(0).expand(len(rivals), -1)
                zdiff = (logits_g[idx_i].expand(len(rivals)) - logits_g[torch.as_tensor(rivals, device=logits_g.device, dtype=torch.long)]).unsqueeze(-1)
                margin_col = torch.full((len(rivals), 1), float(margin.item()), device=pair_tokens.device, dtype=pair_tokens.dtype)
                entropy_col = torch.full((len(rivals), 1), float(entropy.item()), device=pair_tokens.device, dtype=pair_tokens.dtype)
                duel_in = torch.cat([xi_rep, xk, xi_rep - xk, zdiff, margin_col, entropy_col], dim=-1)
                duel_h = _gelu(self.duel_fc1(duel_in))
                duel_h = _gelu(self.duel_fc2(duel_h))
                attn = torch.softmax(self.attn_fc(duel_h).squeeze(-1), dim=0)
                ctx = torch.sum(attn[:, None] * duel_h, dim=0)
                comp_in = torch.cat(
                    [
                        xi,
                        ctx,
                        logits_g[idx_i].reshape(1),
                        margin.reshape(1),
                        entropy.reshape(1),
                    ],
                    dim=0,
                ).unsqueeze(0)
                comp_h = _gelu(self.comp_fc1(comp_in))
                residuals.append(self.comp_fc2(comp_h).reshape(()))

            residuals_t = torch.stack(residuals)
            residuals_t = residuals_t - residuals_t.mean()
            comp_residual[topk_abs] = residuals_t
            comp_logits[topk_abs] = logits_g[topk_idx] + activation * self.comp_delta_scale * torch.tanh(residuals_t)
            comp_prebg[start:end] = torch.sigmoid(comp_logits[start:end]).clamp(min=1e-4, max=1.0 - 1e-4)

            order_after = torch.argsort(comp_logits[start:end], descending=True)
            ranks_after = torch.empty_like(order_after)
            ranks_after[order_after] = torch.arange(order_after.numel(), device=logits_g.device, dtype=order_after.dtype)
            comp_rank_after[start:end] = ranks_after.float() / float(max(order_after.numel(), 1))

        final_sim = ((1.0 - bg_prob) * comp_prebg).clamp(min=1e-4, max=1.0 - 1e-4)
        final_logits = torch.logit(final_sim, eps=1e-4)

        return {
            "pair_tokens": pair_tokens,
            "base_prebg_sim": base_prebg,
            "base_final_sim": base_final,
            "base_logit": base["base_logit"],
            "comp_prebg_sim": comp_prebg,
            "final_sim": final_sim,
            "final_logit": final_logits,
            "comp_active": comp_active,
            "comp_margin": comp_margin,
            "comp_entropy": comp_entropy,
            "comp_residual": comp_residual,
            "comp_topk_mask": comp_topk_mask,
            "comp_rank_before": comp_rank_before,
            "comp_rank_after": comp_rank_after,
            "group_ambiguous": group_ambiguous,
        }


def fit_margin_threshold(
    model: ATCRModel,
    datasets: Sequence[DatasetTensors] | None,
    device: torch.device,
    quantile: float,
    npz_paths: Sequence[str] | None = None,
    batch_groups: int = 0,
) -> float:
    margins: list[float] = []
    if datasets is None:
        if npz_paths is None:
            raise ValueError("npz_paths are required when datasets is None")
        dataset_iter = iter_loaded_datasets(npz_paths, batch_groups=batch_groups, split_name="margin")
    else:
        dataset_iter = enumerate(datasets, start=1)

    model.eval()
    with torch.no_grad():
        for _, data in dataset_iter:
            for start, end in data.batch_slices:
                sl = slice(start, end)
                labels = torch.from_numpy(data.label[sl]).to(device=device, dtype=torch.float32)
                outputs = model(
                    det_feat=torch.from_numpy(data.det_feat[sl]).to(device=device, dtype=torch.float32),
                    hist_feat=torch.from_numpy(data.hist_feat[sl]).to(device=device, dtype=torch.float32),
                    hist_mask=torch.from_numpy(data.hist_mask[sl]).to(device=device, dtype=torch.float32),
                    track_feat=torch.from_numpy(data.track_feat[sl]).to(device=device, dtype=torch.float32),
                    ctx_feat=torch.from_numpy(data.ctx_feat[sl]).to(device=device, dtype=torch.float32),
                    group_ids=torch.from_numpy(data.group_id[sl]).to(device=device, dtype=torch.long),
                    labels=labels,
                )
                base_scores = outputs["base_prebg_sim"]
                group_ids = torch.from_numpy(data.group_id[sl]).to(device=device, dtype=torch.long)
                for seg_start, seg_end in group_segments(group_ids):
                    labels_g = labels[seg_start:seg_end]
                    if not bool(torch.any(labels_g > 0.5)):
                        continue
                    scores_g = base_scores[seg_start:seg_end]
                    if scores_g.numel() <= 1:
                        margins.append(float(scores_g[0].item()))
                    else:
                        vals = torch.topk(scores_g, k=2).values
                        margins.append(float((vals[0] - vals[1]).item()))
    if not margins:
        return 0.10
    return float(np.quantile(np.asarray(margins, dtype=np.float32), min(max(float(quantile), 0.05), 0.95)))


def batch_losses(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float,
    duel_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    zero = outputs["final_sim"].new_tensor(0.0)
    list_terms = []
    duel_terms = []
    safe_terms = []
    top1_hits = []

    final_logits = outputs["final_logit"] / max(float(temperature), 1e-4)
    base_logits = outputs["base_logit"] / max(float(temperature), 1e-4)
    final_sim = outputs["final_sim"]
    base_sim = outputs["base_final_sim"]
    ambiguous = outputs["group_ambiguous"]

    for start, end in group_segments(group_ids):
        labels_g = labels[start:end]
        logits_g = final_logits[start:end]
        base_g = base_logits[start:end]
        if not bool(torch.any(labels_g > 0.5)):
            safe_terms.append(torch.mean((logits_g - base_g) ** 2))
            continue

        pos_idx = int(torch.argmax(labels_g).item())
        list_terms.append(F.cross_entropy(logits_g.view(1, -1), torch.tensor([pos_idx], device=logits_g.device)))
        top1_hits.append(float(torch.argmax(final_sim[start:end]).item() == pos_idx))

        if bool(ambiguous[start].item()) and (end - start) > 1:
            neg_mask = torch.ones_like(labels_g, dtype=torch.bool)
            neg_mask[pos_idx] = False
            duel_terms.append(F.relu(logits_g.new_tensor(float(duel_margin)) - logits_g[pos_idx] + logits_g[neg_mask].max()))
        else:
            safe_terms.append(torch.mean((logits_g - base_g) ** 2))

    list_loss = torch.stack(list_terms).mean() if list_terms else zero
    duel_loss = torch.stack(duel_terms).mean() if duel_terms else zero
    safe_loss = torch.stack(safe_terms).mean() if safe_terms else zero
    top1 = float(np.mean(top1_hits)) if top1_hits else 0.0
    return list_loss, duel_loss, safe_loss, top1


def atcr_shift_fallback_loss(outputs_corr: dict[str, torch.Tensor]) -> torch.Tensor:
    align_final = torch.mean(torch.abs(outputs_corr["final_sim"] - outputs_corr["base_final_sim"]))
    align_prebg = torch.mean(torch.abs(outputs_corr["comp_prebg_sim"] - outputs_corr["base_prebg_sim"]))
    active_weight = outputs_corr["comp_topk_mask"] * outputs_corr["comp_active"]
    active_norm = active_weight.sum().clamp(min=1.0)
    sparse_res = torch.sum(active_weight * torch.abs(torch.tanh(outputs_corr["comp_residual"]))) / active_norm
    return align_final + align_prebg + sparse_res


def run_dataset(
    model: ATCRModel,
    datasets: Sequence[DatasetTensors] | None,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    npz_paths: Sequence[str] | None = None,
    split_name: str | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    losses: list[float] = []
    top1s: list[float] = []
    if datasets is None:
        if npz_paths is None or split_name is None:
            raise ValueError("npz_paths and split_name are required when datasets is None")
        dataset_iter = iter_loaded_datasets(npz_paths, batch_groups=args.batch_groups, split_name=split_name)
        total_shards = len(npz_paths)
    else:
        dataset_iter = enumerate(datasets, start=1)
        total_shards = len(datasets)

    for shard_idx, data in dataset_iter:
        if is_train:
            random.shuffle(data.batch_slices)
        for start, end in data.batch_slices:
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
                    labels=labels,
                )
                list_loss, duel_loss, safe_loss, top1 = batch_losses(
                    outputs=outputs,
                    labels=labels,
                    group_ids=group_ids,
                    temperature=args.temperature,
                    duel_margin=args.duel_margin,
                )
                loss = list_loss + args.loss_duel_weight * duel_loss + args.loss_safe_weight * safe_loss

                if (
                    is_train
                    and args.loss_shift_weight > 0.0
                    and args.shift_batch_prob > 0.0
                    and random.random() < float(args.shift_batch_prob)
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
                        labels=labels,
                    )
                    shift_loss = atcr_shift_fallback_loss(outputs_corr)
                    loss = loss + args.loss_shift_weight * shift_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
                optimizer.step()

            losses.append(float(loss.item()))
            top1s.append(float(top1))

        print(
            f"[{'train' if is_train else 'eval'}] shard {shard_idx}/{total_shards} "
            f"path={data.path} avg_loss={np.mean(losses):.6f} avg_top1={np.mean(top1s) if top1s else 0.0:.4f}",
            flush=True,
        )
    return float(np.mean(losses)) if losses else float("inf"), float(np.mean(top1s)) if top1s else 0.0


def export_checkpoint(
    model: ATCRModel,
    out_path: Path,
    base_data: dict[str, np.ndarray],
) -> None:
    state = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in model.state_dict().items() if not key.startswith("base.")}
    payload = {key: value for key, value in base_data.items()}
    payload["version"] = np.asarray(["haca_v3"], dtype=object)
    payload["comp_topk"] = np.asarray([model.comp_topk], dtype=np.int32)
    payload["comp_margin_threshold"] = np.asarray([model.comp_margin_threshold], dtype=np.float32)
    payload["comp_margin_temperature"] = np.asarray([model.comp_margin_temperature], dtype=np.float32)
    payload["comp_delta_scale"] = np.asarray([model.comp_delta_scale], dtype=np.float32)
    payload["W_duel1"] = state["duel_fc1.weight"].T
    payload["b_duel1"] = state["duel_fc1.bias"]
    payload["W_duel2"] = state["duel_fc2.weight"].T
    payload["b_duel2"] = state["duel_fc2.bias"]
    payload["W_attn"] = state["attn_fc.weight"].T
    payload["b_attn"] = state["attn_fc.bias"]
    payload["W_comp1"] = state["comp_fc1.weight"].T
    payload["b_comp1"] = state["comp_fc1.bias"]
    payload["W_comp2"] = state["comp_fc2.weight"].T
    payload["b_comp2"] = state["comp_fc2.bias"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
    print(f"[saved] {out_path}", flush=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    train_paths = list(args.train_npz)
    val_paths = list(args.val_npz)
    if not train_paths:
        raise ValueError("No training shards were provided.")
    infer_max_history(train_paths)

    base_model, base_data, _ = _load_base_model(args.base_npz, device=device)
    print(
        "[base] "
        f"version={base_model.version} "
        f"use_background={int(base_model.use_background)} "
        f"use_hist_gate={int(base_model.use_hist_gate)} "
        f"use_ood_gate={int(base_model.use_ood_gate)} "
        f"ood_scale={base_model.ood_scale:.2f}",
        flush=True,
    )
    margin_threshold = fit_margin_threshold(
        model=ATCRModel(
            base_model=base_model,
            comp_hidden=args.comp_hidden,
            comp_topk=args.comp_topk,
            comp_margin_threshold=0.10,
            comp_margin_temperature=args.comp_margin_temperature,
            comp_delta_scale=args.comp_delta_scale,
            positive_injection_prob=args.positive_injection_prob,
        ).to(device),
        datasets=None,
        device=device,
        quantile=args.comp_margin_quantile,
        npz_paths=train_paths,
        batch_groups=args.batch_groups,
    )
    print(f"[margin-threshold] quantile={args.comp_margin_quantile:.3f} threshold={margin_threshold:.6f}", flush=True)

    model = ATCRModel(
        base_model=base_model,
        comp_hidden=args.comp_hidden,
        comp_topk=args.comp_topk,
        comp_margin_threshold=margin_threshold,
        comp_margin_temperature=args.comp_margin_temperature,
        comp_delta_scale=args.comp_delta_scale,
        positive_injection_prob=args.positive_injection_prob,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_metric = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_top1 = run_dataset(
            model,
            None,
            device,
            optimizer,
            args,
            npz_paths=train_paths,
            split_name="train",
        )
        if val_paths:
            with torch.no_grad():
                val_loss, val_top1 = run_dataset(
                    model,
                    None,
                    device,
                    None,
                    args,
                    npz_paths=val_paths,
                    split_name="val",
                )
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
    export_checkpoint(model=model, out_path=Path(args.out_npz), base_data=base_data)


if __name__ == "__main__":
    main()
