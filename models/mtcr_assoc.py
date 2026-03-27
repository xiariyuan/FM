from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


MTCR_PAIR_TOKEN_NAMES = (
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


def _last_valid_indices(track_masks: torch.Tensor) -> torch.Tensor:
    valid = (~track_masks).to(dtype=torch.long)
    time_idx = torch.arange(track_masks.shape[1], device=track_masks.device, dtype=torch.long).view(1, -1)
    last = (valid * (time_idx + 1)).max(dim=1).values - 1
    return last.clamp(min=0)


class MTCRAssociationAdapter(nn.Module):
    """
    Multi-Time-constant Competitive Residual (MTCR)

    A safe ambiguity-aware association plug-in:
    - starts from an existing anchor score matrix
    - extracts temporal evidence from the full detection-to-history similarity sequence
    - only re-ranks top-k rivals inside each detection-centered candidate set
    - uses bounded residuals, so default initialization is a no-op
    """

    def __init__(
        self,
        hist_hidden: int = 16,
        comp_hidden: int = 64,
        topk: int = 3,
        margin_threshold: float = 0.10,
        margin_temperature: float = 0.03,
        delta_scale: float = 1.0,
        min_history: int = 3,
        decay_scales: Iterable[float] = (1.0, 2.0, 4.0),
        gate_cap: float = 0.20,
        bg_scale: float = 0.75,
        gate_init_logit: float = -2.0,
        bg_init_logit: float = -4.0,
        hidden_init_gain: float = 0.25,
        output_init_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.hist_hidden = int(max(hist_hidden, 8))
        self.comp_hidden = int(max(comp_hidden, 16))
        self.topk = int(max(topk, 1))
        self.margin_threshold = float(margin_threshold)
        self.margin_temperature = float(max(margin_temperature, 1e-4))
        self.delta_scale = float(delta_scale)
        self.min_history = int(max(min_history, 1))
        self.gate_cap = float(max(gate_cap, 1e-3))
        self.bg_scale = float(min(max(bg_scale, 0.0), 1.0))
        self.gate_init_logit = float(gate_init_logit)
        self.bg_init_logit = float(bg_init_logit)
        self.hidden_init_gain = float(hidden_init_gain)
        self.output_init_scale = float(output_init_scale)
        self.register_buffer(
            "decay_scales",
            torch.tensor([float(x) for x in decay_scales], dtype=torch.float32),
            persistent=True,
        )

        self.hist_fc1 = nn.Linear(3, self.hist_hidden)
        self.hist_fc2 = nn.Linear(self.hist_hidden, self.hist_hidden)
        self.hist_attn = nn.Linear(self.hist_hidden, 1)

        duel_dim = len(MTCR_PAIR_TOKEN_NAMES) * 3 + 3
        comp_in_dim = len(MTCR_PAIR_TOKEN_NAMES) + self.comp_hidden + 3
        group_in_dim = len(MTCR_PAIR_TOKEN_NAMES) + 6
        self.duel_fc1 = nn.Linear(duel_dim, self.comp_hidden)
        self.duel_fc2 = nn.Linear(self.comp_hidden, self.comp_hidden)
        self.attn_fc = nn.Linear(self.comp_hidden, 1)
        self.comp_fc1 = nn.Linear(comp_in_dim, self.comp_hidden)
        self.comp_fc2 = nn.Linear(self.comp_hidden, 1)
        self.group_fc1 = nn.Linear(group_in_dim, self.comp_hidden)
        self.group_fc2 = nn.Linear(self.comp_hidden, self.comp_hidden)
        self.gate_head = nn.Linear(self.comp_hidden, 1)
        self.bg_head = nn.Linear(self.comp_hidden, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Temporal summary stays learnable. Competition head starts near no-op, but not at a dead symmetric zero point.
        for layer in (self.hist_fc1, self.hist_fc2, self.hist_attn):
            nn.init.xavier_uniform_(layer.weight, gain=0.7)
            nn.init.zeros_(layer.bias)
        for layer in (self.duel_fc1, self.duel_fc2, self.attn_fc, self.comp_fc1, self.group_fc1, self.group_fc2):
            nn.init.xavier_uniform_(layer.weight, gain=self.hidden_init_gain)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.comp_fc2.weight, mean=0.0, std=self.output_init_scale)
        nn.init.zeros_(self.comp_fc2.bias)
        nn.init.normal_(self.gate_head.weight, mean=0.0, std=self.output_init_scale)
        nn.init.normal_(self.bg_head.weight, mean=0.0, std=self.output_init_scale)
        nn.init.constant_(self.gate_head.bias, self.gate_init_logit)
        nn.init.constant_(self.bg_head.bias, self.bg_init_logit)

    @classmethod
    def from_npz(cls, path: str | Path) -> "MTCRAssociationAdapter":
        z = np.load(str(path), allow_pickle=True)
        files = {key: z[key] for key in z.files}
        topk = int(np.asarray(files.get("comp_topk", np.asarray([3], dtype=np.int32))).reshape(-1)[0])
        margin_threshold = float(np.asarray(files.get("comp_margin_threshold", np.asarray([0.10], dtype=np.float32))).reshape(-1)[0])
        margin_temperature = float(np.asarray(files.get("comp_margin_temperature", np.asarray([0.03], dtype=np.float32))).reshape(-1)[0])
        delta_scale = float(np.asarray(files.get("comp_delta_scale", np.asarray([1.0], dtype=np.float32))).reshape(-1)[0])
        hist_hidden = int(np.asarray(files.get("hist_hidden", np.asarray([16], dtype=np.int32))).reshape(-1)[0])
        comp_hidden = int(np.asarray(files.get("comp_hidden", np.asarray([64], dtype=np.int32))).reshape(-1)[0])
        min_history = int(np.asarray(files.get("min_history", np.asarray([3], dtype=np.int32))).reshape(-1)[0])
        decay_scales = np.asarray(files.get("decay_scales", np.asarray([1.0, 2.0, 4.0], dtype=np.float32))).reshape(-1).tolist()
        gate_cap = float(np.asarray(files.get("comp_gate_cap", np.asarray([0.20], dtype=np.float32))).reshape(-1)[0])
        bg_scale = float(np.asarray(files.get("comp_bg_scale", np.asarray([0.75], dtype=np.float32))).reshape(-1)[0])
        gate_init_logit = float(np.asarray(files.get("comp_gate_init_logit", np.asarray([-2.0], dtype=np.float32))).reshape(-1)[0])
        bg_init_logit = float(np.asarray(files.get("comp_bg_init_logit", np.asarray([-4.0], dtype=np.float32))).reshape(-1)[0])

        model = cls(
            hist_hidden=hist_hidden,
            comp_hidden=comp_hidden,
            topk=topk,
            margin_threshold=margin_threshold,
            margin_temperature=margin_temperature,
            delta_scale=delta_scale,
            min_history=min_history,
            decay_scales=decay_scales,
            gate_cap=gate_cap,
            bg_scale=bg_scale,
            gate_init_logit=gate_init_logit,
            bg_init_logit=bg_init_logit,
        )
        state = model.state_dict()

        def _maybe(key_weight: str, key_bias: str, state_weight: str, state_bias: str) -> None:
            if key_weight in files and key_bias in files:
                state[state_weight] = torch.from_numpy(np.asarray(files[key_weight], dtype=np.float32).T)
                state[state_bias] = torch.from_numpy(np.asarray(files[key_bias], dtype=np.float32))

        _maybe("W_hist1", "b_hist1", "hist_fc1.weight", "hist_fc1.bias")
        _maybe("W_hist2", "b_hist2", "hist_fc2.weight", "hist_fc2.bias")
        _maybe("W_hist_attn", "b_hist_attn", "hist_attn.weight", "hist_attn.bias")
        _maybe("W_duel1", "b_duel1", "duel_fc1.weight", "duel_fc1.bias")
        _maybe("W_duel2", "b_duel2", "duel_fc2.weight", "duel_fc2.bias")
        _maybe("W_attn", "b_attn", "attn_fc.weight", "attn_fc.bias")
        _maybe("W_comp1", "b_comp1", "comp_fc1.weight", "comp_fc1.bias")
        _maybe("W_comp2", "b_comp2", "comp_fc2.weight", "comp_fc2.bias")
        _maybe("W_group1", "b_group1", "group_fc1.weight", "group_fc1.bias")
        _maybe("W_group2", "b_group2", "group_fc2.weight", "group_fc2.bias")
        _maybe("W_gate", "b_gate", "gate_head.weight", "gate_head.bias")
        _maybe("W_bg", "b_bg", "bg_head.weight", "bg_head.bias")
        model.load_state_dict(state, strict=True)
        return model

    def _masked_track_stats(
        self,
        track_history_features: torch.Tensor,
        track_history_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = F.normalize(track_history_features, dim=-1)
        valid = (~track_history_masks).to(dtype=feats.dtype)
        counts = valid.sum(dim=1).clamp(min=1.0)
        last_idx = _last_valid_indices(track_history_masks)
        last_feat = feats[torch.arange(feats.shape[0], device=feats.device), last_idx]

        stability = []
        coherence = []
        hist_norm = []
        scales = self.decay_scales.to(device=feats.device, dtype=feats.dtype)
        for row in range(feats.shape[0]):
            count = int(round(float(counts[row].item())))
            hist_norm.append(min(1.0, float(count) / float(max(self.min_history, 1))))
            if count <= 0:
                stability.append(1.0)
                coherence.append(1.0)
                continue

            valid_hist = feats[row, -count:]
            if count < 3:
                stability.append(1.0)
            else:
                delta2 = valid_hist[2:] - 2.0 * valid_hist[1:-1] + valid_hist[:-2]
                curvature = torch.sqrt(torch.mean(delta2 ** 2, dim=1)).mean()
                stability.append(float(torch.exp(-curvature).item()))

            if count < self.min_history:
                coherence.append(1.0)
            else:
                ages = torch.arange(count - 1, -1, -1, device=feats.device, dtype=feats.dtype)
                proto_sims = []
                for scale in scales:
                    weight = torch.exp(-ages / scale.clamp(min=1e-4))
                    weight = weight / weight.sum().clamp(min=1e-6)
                    proto = F.normalize((weight[:, None] * valid_hist).sum(dim=0), dim=0)
                    proto_sims.append(((proto * last_feat[row]).sum().clamp(min=-1.0, max=1.0) + 1.0) * 0.5)
                coherence.append(float(torch.stack(proto_sims).mean().item()))

        return (
            torch.tensor(stability, device=feats.device, dtype=feats.dtype),
            torch.tensor(coherence, device=feats.device, dtype=feats.dtype),
            torch.tensor(hist_norm, device=feats.device, dtype=feats.dtype),
        )

    def _temporal_summary(
        self,
        det_features: torch.Tensor,
        track_history_features: torch.Tensor,
        track_history_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        det = F.normalize(det_features, dim=-1)
        hist = F.normalize(track_history_features, dim=-1)
        sims = torch.einsum("nd,mtd->nmt", det, hist).clamp(min=-1.0, max=1.0)
        sims = 0.5 * (sims + 1.0)

        num_dets, num_tracks, window = sims.shape
        valid = (~track_history_masks).to(dtype=sims.dtype)
        valid_full = valid.unsqueeze(0).expand(num_dets, -1, -1)
        last_idx = _last_valid_indices(track_history_masks)

        time_idx = torch.arange(window, device=sims.device, dtype=sims.dtype).view(1, 1, window)
        age = (last_idx.to(dtype=sims.dtype).view(1, num_tracks, 1) - time_idx).clamp(min=0.0)
        age_log = torch.log1p(age).expand(num_dets, -1, -1)

        delta = torch.zeros_like(sims)
        if window > 1:
            delta[..., 1:] = sims[..., 1:] - sims[..., :-1]

        step_feat = torch.stack([sims, age_log, delta], dim=-1)
        h = F.gelu(self.hist_fc1(step_feat))
        h = F.gelu(self.hist_fc2(h))
        attn_logits = self.hist_attn(h).squeeze(-1)
        attn_logits = attn_logits.masked_fill(valid_full <= 0.0, -1e9)
        attn = torch.softmax(attn_logits, dim=-1)
        attn = attn * valid_full
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        temp_sim = (attn * sims).sum(dim=-1)
        last_sim = sims.gather(dim=-1, index=last_idx.view(1, num_tracks, 1).expand(num_dets, -1, 1)).squeeze(-1)
        max_sim = sims.masked_fill(valid_full <= 0.0, -1e9).max(dim=-1).values.clamp(min=0.0, max=1.0)
        counts = valid.sum(dim=1).clamp(min=1.0)
        mean_sim = (sims * valid_full).sum(dim=-1) / counts.view(1, num_tracks)
        std_sim = torch.sqrt((((sims - mean_sim.unsqueeze(-1)) ** 2) * valid_full).sum(dim=-1) / counts.view(1, num_tracks)).clamp(min=0.0)
        return temp_sim, last_sim, max_sim, std_sim

    def _build_pair_tokens(
        self,
        anchor_scores: torch.Tensor,
        det_features: torch.Tensor,
        track_history_features: torch.Tensor,
        track_history_masks: torch.Tensor,
        motion_scores: torch.Tensor,
        det_scores: Optional[torch.Tensor],
        track_gaps: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        anchor = anchor_scores.clamp(min=1e-4, max=1.0 - 1e-4)
        spatial = anchor
        motion = motion_scores if motion_scores is not None else anchor_scores
        motion = motion.clamp(min=0.0, max=1.0)

        temp_sim, last_sim, max_sim, std_sim = self._temporal_summary(
            det_features=det_features,
            track_history_features=track_history_features,
            track_history_masks=track_history_masks,
        )
        stability, coherence, hist_norm = self._masked_track_stats(
            track_history_features=track_history_features,
            track_history_masks=track_history_masks,
        )
        stability = stability.clamp(min=0.0, max=1.0)
        coherence = coherence.clamp(min=0.0, max=1.0)
        hist_norm = hist_norm.clamp(min=0.0, max=1.0)

        num_dets, num_tracks = anchor.shape
        det_conf = (
            det_scores.view(-1, 1).to(device=anchor.device, dtype=anchor.dtype).clamp(min=0.0, max=1.0)
            if det_scores is not None
            else torch.ones((num_dets, 1), device=anchor.device, dtype=anchor.dtype)
        )
        gap_log = (
            torch.log1p(track_gaps.to(device=anchor.device, dtype=anchor.dtype).clamp(min=0.0)).view(1, num_tracks).expand(num_dets, -1)
            if track_gaps is not None
            else torch.zeros((num_dets, num_tracks), device=anchor.device, dtype=anchor.dtype)
        )
        hist_norm_full = hist_norm.view(1, num_tracks).expand(num_dets, -1)
        stability_full = stability.view(1, num_tracks).expand(num_dets, -1)
        coherence_full = coherence.view(1, num_tracks).expand(num_dets, -1)

        anchor_z = torch.zeros_like(anchor)
        anchor_margin = torch.zeros_like(anchor)
        anchor_rank = torch.zeros_like(anchor)
        for det_idx in range(num_dets):
            row = anchor[det_idx]
            mean = row.mean()
            std = row.std(unbiased=False).clamp(min=1e-6)
            anchor_z[det_idx] = (row - mean) / std
            if num_tracks == 1:
                anchor_margin[det_idx] = row
                anchor_rank[det_idx] = 0.0
                continue
            order = torch.argsort(row, descending=True)
            ranks = torch.empty_like(order)
            ranks[order] = torch.arange(order.numel(), device=row.device, dtype=order.dtype)
            anchor_rank[det_idx] = ranks.float() / float(max(order.numel(), 1))
            top_vals, top_idx = torch.topk(row, k=min(2, row.numel()))
            best_idx = int(top_idx[0].item())
            best_val = top_vals[0]
            second_val = top_vals[1] if top_vals.numel() > 1 else top_vals[0]
            idx = torch.arange(row.numel(), device=row.device)
            max_other = torch.where(idx == best_idx, second_val, best_val)
            anchor_margin[det_idx] = row - max_other

        pair_tokens = torch.stack(
            [
                anchor,
                spatial,
                motion,
                temp_sim,
                last_sim,
                max_sim,
                std_sim,
                gap_log,
                hist_norm_full,
                stability_full,
                coherence_full,
                anchor_z,
                anchor_margin,
                anchor_rank,
                det_conf.expand(-1, num_tracks),
            ],
            dim=-1,
        )
        aux = {
            "temp_sim": temp_sim,
            "last_sim": last_sim,
            "max_sim": max_sim,
            "std_sim": std_sim,
            "stability": stability,
            "coherence": coherence,
            "hist_norm": hist_norm,
            "anchor_margin": anchor_margin,
            "anchor_rank": anchor_rank,
            "det_conf": det_conf,
        }
        return pair_tokens, aux

    def forward(
        self,
        anchor_scores: torch.Tensor,
        det_features: torch.Tensor,
        track_history_features: torch.Tensor,
        track_history_masks: torch.Tensor,
        motion_scores: Optional[torch.Tensor] = None,
        det_scores: Optional[torch.Tensor] = None,
        track_gaps: Optional[torch.Tensor] = None,
        force_include_indices: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if anchor_scores.numel() == 0:
            return {
                "final_scores": anchor_scores,
                "pair_tokens": anchor_scores.new_zeros((0, 0, len(MTCR_PAIR_TOKEN_NAMES))),
                "comp_active": anchor_scores.new_zeros((0, 0)),
                "comp_margin": anchor_scores.new_zeros((0, 0)),
                "comp_entropy": anchor_scores.new_zeros((0, 0)),
                "comp_residual": anchor_scores.new_zeros((0, 0)),
                "group_gate": anchor_scores.new_zeros((0, 1)),
                "bg_prob": anchor_scores.new_zeros((0, 1)),
            }

        work_dtype = torch.float32 if anchor_scores.dtype in (torch.float16, torch.bfloat16) else anchor_scores.dtype
        scores = anchor_scores.to(dtype=work_dtype).clamp(min=1e-4, max=1.0 - 1e-4)
        det = det_features.to(device=scores.device, dtype=work_dtype)
        hist = track_history_features.to(device=scores.device, dtype=work_dtype)
        masks = track_history_masks.to(device=scores.device)
        motion = motion_scores.to(device=scores.device, dtype=work_dtype) if motion_scores is not None else scores
        det_conf = det_scores.to(device=scores.device, dtype=work_dtype) if det_scores is not None else None
        gaps = track_gaps.to(device=scores.device, dtype=work_dtype) if track_gaps is not None else None
        force_idx = (
            force_include_indices.to(device=scores.device, dtype=torch.long).view(-1)
            if force_include_indices is not None
            else None
        )

        pair_tokens, _ = self._build_pair_tokens(
            anchor_scores=scores,
            det_features=det,
            track_history_features=hist,
            track_history_masks=masks,
            motion_scores=motion,
            det_scores=det_conf,
            track_gaps=gaps,
        )

        logits = torch.logit(scores, eps=1e-4)
        final_logits = logits.clone()
        comp_active = torch.zeros_like(scores)
        comp_margin = torch.zeros_like(scores)
        comp_entropy = torch.zeros_like(scores)
        comp_residual = torch.zeros_like(scores)
        group_gate_out = torch.zeros((scores.shape[0], 1), device=scores.device, dtype=work_dtype)
        bg_prob_out = torch.zeros((scores.shape[0], 1), device=scores.device, dtype=work_dtype)

        for det_idx in range(scores.shape[0]):
            row_scores = scores[det_idx]
            row_logits = logits[det_idx]
            if row_scores.numel() <= 1:
                comp_margin[det_idx] = row_scores
                continue

            k = min(self.topk, int(row_scores.numel()))
            topk_idx = torch.topk(row_scores, k=k).indices
            if force_idx is not None and det_idx < force_idx.numel():
                gt_idx = int(force_idx[det_idx].item())
                if 0 <= gt_idx < row_scores.numel() and not bool(torch.any(topk_idx == gt_idx)):
                    topk_idx = topk_idx.clone()
                    topk_idx[-1] = gt_idx
                    topk_idx = topk_idx[torch.argsort(row_scores[topk_idx], descending=True)]
            top_scores = row_scores[topk_idx]
            margin = top_scores[0] - top_scores[1] if top_scores.numel() > 1 else top_scores[0]
            probs = torch.softmax(row_logits[topk_idx], dim=0)
            entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-12)))

            hist_trust = pair_tokens[det_idx, topk_idx, 8:11].mean()
            det_trust = pair_tokens[det_idx, topk_idx, 14].mean()
            top1_score = top_scores[0]
            top2_score = top_scores[1] if top_scores.numel() > 1 else top_scores[0]
            group_mean = pair_tokens[det_idx, topk_idx].mean(dim=0)
            group_in = torch.cat(
                [
                    group_mean,
                    torch.stack(
                        [
                            top1_score,
                            top2_score,
                            margin,
                            entropy,
                            hist_trust,
                            det_trust,
                        ],
                        dim=0,
                    ),
                ],
                dim=0,
            ).unsqueeze(0)
            group_h = F.gelu(self.group_fc1(group_in))
            group_h = F.gelu(self.group_fc2(group_h))
            group_gate = self.gate_cap * torch.sigmoid(self.gate_head(group_h).reshape(()))
            bg_prob = torch.sigmoid(self.bg_head(group_h).reshape(()))
            base_activation = hist_trust * det_trust * torch.sigmoid((margin.new_tensor(self.margin_threshold) - margin) / self.margin_temperature)
            activation = base_activation * group_gate * (1.0 - bg_prob)
            comp_active[det_idx] = activation
            comp_margin[det_idx] = margin
            comp_entropy[det_idx] = entropy
            group_gate_out[det_idx, 0] = group_gate
            bg_prob_out[det_idx, 0] = bg_prob

            if topk_idx.numel() <= 1 or float(activation.item()) <= 1e-6:
                continue

            residuals = []
            for idx_i in topk_idx.tolist():
                rivals = [idx_k for idx_k in topk_idx.tolist() if idx_k != idx_i]
                if not rivals:
                    residuals.append(row_logits.new_tensor(0.0))
                    continue
                xi = pair_tokens[det_idx, idx_i]
                rivals_t = torch.as_tensor(rivals, device=row_logits.device, dtype=torch.long)
                xk = pair_tokens[det_idx, rivals_t]
                xi_rep = xi.unsqueeze(0).expand(len(rivals), -1)
                zdiff = (row_logits[idx_i].expand(len(rivals)) - row_logits[rivals_t]).unsqueeze(-1)
                margin_col = torch.full((len(rivals), 1), float(margin.item()), device=row_logits.device, dtype=work_dtype)
                entropy_col = torch.full((len(rivals), 1), float(entropy.item()), device=row_logits.device, dtype=work_dtype)
                duel_in = torch.cat([xi_rep, xk, xi_rep - xk, zdiff, margin_col, entropy_col], dim=-1)
                duel_h = F.gelu(self.duel_fc1(duel_in))
                duel_h = F.gelu(self.duel_fc2(duel_h))
                attn = torch.softmax(self.attn_fc(duel_h).squeeze(-1), dim=0)
                ctx = torch.sum(attn[:, None] * duel_h, dim=0)
                comp_in = torch.cat(
                    [
                        xi,
                        ctx,
                        row_logits[idx_i].reshape(1),
                        margin.reshape(1),
                        entropy.reshape(1),
                    ],
                    dim=0,
                ).unsqueeze(0)
                comp_h = F.gelu(self.comp_fc1(comp_in))
                residuals.append(self.comp_fc2(comp_h).reshape(()))

            residuals_t = torch.stack(residuals)
            residuals_t = residuals_t - residuals_t.mean()
            comp_residual[det_idx, topk_idx] = residuals_t
            final_logits[det_idx, topk_idx] = row_logits[topk_idx] + activation * self.delta_scale * torch.tanh(residuals_t)

        final_scores = torch.sigmoid(final_logits)
        final_scores = final_scores * (1.0 - self.bg_scale * bg_prob_out.expand_as(final_scores))
        final_scores = final_scores.clamp(min=1e-4, max=1.0 - 1e-4)
        return {
            "final_scores": final_scores.to(dtype=anchor_scores.dtype),
            "pair_tokens": pair_tokens.to(dtype=anchor_scores.dtype),
            "comp_active": comp_active.to(dtype=anchor_scores.dtype),
            "comp_margin": comp_margin.to(dtype=anchor_scores.dtype),
            "comp_entropy": comp_entropy.to(dtype=anchor_scores.dtype),
            "comp_residual": comp_residual.to(dtype=anchor_scores.dtype),
            "group_gate": group_gate_out.to(dtype=anchor_scores.dtype),
            "bg_prob": bg_prob_out.to(dtype=anchor_scores.dtype),
        }
