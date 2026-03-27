from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn


RUNTIME_REPLAY_FEATURE_NAMES = (
    "anchor_score",
    "base_score",
    "refined_score",
    "motion_score",
    "det_score",
    "log1p_track_gap",
    "log1p_track_hist_len",
    "base_margin",
    "refined_margin",
    "rank_margin",
    "rank_entropy",
    "rank_frac",
    "dx_norm",
    "dy_norm",
    "log_w_ratio",
    "log_h_ratio",
    "log_area_ratio",
    "det_track_iou",
)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    weight = mask.to(dtype=x.dtype)
    denom = weight.sum(dim=dim, keepdim=keepdim).clamp(min=1e-6)
    return (x * weight).sum(dim=dim, keepdim=keepdim) / denom


def _masked_fill_logits(logits: torch.Tensor, valid_mask: torch.Tensor, fill: float = -1e9) -> torch.Tensor:
    return logits.masked_fill(~valid_mask, fill)


def _masked_std(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mean = _masked_mean(x, mask, dim=dim, keepdim=True)
    centered = (x - mean) ** 2
    return torch.sqrt(_masked_mean(centered, mask, dim=dim, keepdim=False).clamp(min=0.0))


def _sequence_last_valid(mask: torch.Tensor) -> torch.Tensor:
    valid = (~mask).to(dtype=torch.long)
    idx = torch.arange(mask.shape[-1], device=mask.device, dtype=torch.long).view(1, 1, -1)
    last = (valid * (idx + 1)).max(dim=-1).values - 1
    return last.clamp(min=0)


def _group_rank_margin(anchor_scores: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    masked_scores = _masked_fill_logits(anchor_scores, valid_mask, fill=-1.0)
    top2 = torch.topk(masked_scores, k=min(2, anchor_scores.shape[1]), dim=1).values
    if top2.shape[1] == 1:
        margin = top2[:, 0]
    else:
        margin = top2[:, 0] - top2[:, 1]

    anchor_logits = torch.logit(anchor_scores.clamp(min=1e-4, max=1.0 - 1e-4), eps=1e-4)
    masked_logits = _masked_fill_logits(anchor_logits, valid_mask, fill=-1e9)
    prob = torch.softmax(masked_logits, dim=1)
    prob = prob * valid_mask.to(dtype=prob.dtype)
    prob = prob / prob.sum(dim=1, keepdim=True).clamp(min=1e-6)
    entropy = -(prob * torch.log(prob.clamp(min=1e-8))).sum(dim=1)
    return margin, entropy


class ScalarEvidenceTower(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiScaleTemporalEvidenceTower(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int,
        proj_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.det_proj = nn.Linear(feat_dim, hidden_dim)
        self.hist_proj = nn.Linear(feat_dim, hidden_dim)
        self.aux_proj = nn.Linear(4, hidden_dim)
        self.branch_k3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.branch_k5 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.branch_dilated = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.out_proj = nn.Linear(hidden_dim * 3, proj_dim)
        self.attn_q = nn.Linear(hidden_dim, proj_dim)
        self.attn_k = nn.Linear(proj_dim, proj_dim)
        self.attn_v = nn.Linear(proj_dim, proj_dim)
        self.conf_head = nn.Linear(proj_dim, 1)
        self.unc_head = nn.Linear(proj_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        det_features: torch.Tensor,
        hist_features: torch.Tensor,
        hist_masks: torch.Tensor,
        hist_times: Optional[torch.Tensor] = None,
        det_times: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        batch, cand, steps, feat_dim = hist_features.shape
        det_norm = F.normalize(det_features, dim=-1)
        hist_norm = F.normalize(hist_features, dim=-1)
        det_expand = det_norm.unsqueeze(1).unsqueeze(2).expand(batch, cand, steps, feat_dim)
        sim = (det_expand * hist_norm).sum(dim=-1).clamp(min=-1.0, max=1.0)
        sim = 0.5 * (sim + 1.0)

        delta = torch.zeros_like(sim)
        if steps > 1:
            delta[..., 1:] = sim[..., 1:] - sim[..., :-1]

        last_valid = _sequence_last_valid(hist_masks)
        if hist_times is not None and det_times is not None:
            age = det_times.to(device=hist_features.device, dtype=torch.float32).view(batch, 1, 1) - hist_times.to(
                device=hist_features.device,
                dtype=torch.float32,
            )
            age = age.clamp(min=0.0)
            age = torch.log1p(age)
        else:
            time_idx = torch.arange(steps, device=hist_features.device, dtype=torch.float32).view(1, 1, steps)
            age = (last_valid.to(dtype=torch.float32).unsqueeze(-1) - time_idx).clamp(min=0.0)
            age = torch.log1p(age)
        valid = (~hist_masks).to(dtype=hist_features.dtype)
        valid_ratio = valid.sum(dim=-1, keepdim=True) / float(max(steps, 1))
        aux = torch.stack([sim, delta, age, valid.expand_as(sim)], dim=-1)

        det_token = self.det_proj(det_features).unsqueeze(1).unsqueeze(2).expand(batch, cand, steps, -1)
        hist_token = self.hist_proj(hist_features)
        aux_token = self.aux_proj(aux)
        x = F.gelu(det_token + hist_token + aux_token)
        x = self.dropout(x)
        x = x.view(batch * cand, steps, -1).transpose(1, 2)

        b3 = F.gelu(self.branch_k3(x))
        b5 = F.gelu(self.branch_k5(x))
        bd = F.gelu(self.branch_dilated(x))
        multi = torch.cat([b3, b5, bd], dim=1).transpose(1, 2)
        step_tokens = self.out_proj(multi)
        step_tokens = self.dropout(step_tokens)

        query = self.attn_q(self.det_proj(det_features)).unsqueeze(1).expand(batch, cand, -1).reshape(batch * cand, 1, -1)
        keys = self.attn_k(step_tokens)
        values = self.attn_v(step_tokens)
        attn = (query * keys).sum(dim=-1) / math.sqrt(max(keys.shape[-1], 1))
        mask = hist_masks.view(batch * cand, steps)
        attn = attn.masked_fill(mask, -1e9)
        attn = torch.softmax(attn, dim=-1)
        attn = attn * (~mask).to(dtype=attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        pooled = torch.sum(attn.unsqueeze(-1) * values, dim=1)

        temp_conf = torch.sigmoid(self.conf_head(pooled)).view(batch, cand)
        temp_unc = F.softplus(self.unc_head(pooled)).view(batch, cand)
        pooled = pooled.view(batch, cand, -1)
        sim_mean = _masked_mean(sim, ~hist_masks, dim=-1, keepdim=False)
        sim_std = _masked_std(sim, ~hist_masks, dim=-1)
        return {
            "embedding": pooled,
            "temp_conf": temp_conf,
            "temp_uncertainty": temp_unc,
            "sim_mean": sim_mean,
            "sim_std": sim_std,
            "valid_ratio": valid_ratio.squeeze(-1),
        }


class RuntimeReplayAssociationAdapter(nn.Module):
    """
    Full runtime-replay reranker:
    - scalar evidence tower
    - temporal evidence tower on raw detection/history tensors
    - candidate competition tower with set attention + duel aggregation
    - ambiguity-triggered safety controller with bounded residual output
    """

    def __init__(
        self,
        scalar_dim: int = len(RUNTIME_REPLAY_FEATURE_NAMES),
        feat_dim: int = 256,
        scalar_hidden: int = 96,
        scalar_out: int = 96,
        temporal_hidden: int = 96,
        temporal_out: int = 96,
        token_dim: int = 160,
        duel_dim: int = 160,
        group_hidden: int = 128,
        num_heads: int = 4,
        topk: int = 5,
        delta_scale: float = 1.0,
        margin_threshold: float = 0.10,
        margin_temperature: float = 0.03,
        gate_cap: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.scalar_dim = int(scalar_dim)
        self.feat_dim = int(feat_dim)
        self.scalar_hidden = int(scalar_hidden)
        self.scalar_out = int(scalar_out)
        self.temporal_hidden = int(temporal_hidden)
        self.temporal_out = int(temporal_out)
        self.token_dim = int(token_dim)
        self.duel_dim = int(duel_dim)
        self.group_hidden = int(group_hidden)
        self.num_heads = int(num_heads)
        self.topk = int(max(topk, 1))
        self.delta_scale = float(delta_scale)
        self.margin_threshold = float(margin_threshold)
        self.margin_temperature = float(max(margin_temperature, 1e-4))
        self.gate_cap = float(max(gate_cap, 1e-3))

        self.scalar_tower = ScalarEvidenceTower(
            in_dim=self.scalar_dim,
            hidden_dim=self.scalar_hidden,
            out_dim=self.scalar_out,
            dropout=dropout,
        )
        self.temporal_tower = MultiScaleTemporalEvidenceTower(
            feat_dim=self.feat_dim,
            hidden_dim=self.temporal_hidden,
            proj_dim=self.temporal_out,
            dropout=dropout,
        )
        self.meta_proj = nn.Sequential(
            nn.Linear(6, self.scalar_out),
            nn.GELU(),
            nn.LayerNorm(self.scalar_out),
        )
        self.token_proj = nn.Linear(self.scalar_out + self.temporal_out + self.scalar_out, self.token_dim)
        self.self_attn = nn.MultiheadAttention(self.token_dim, self.num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.token_dim * 2, self.token_dim),
        )
        self.ln1 = nn.LayerNorm(self.token_dim)
        self.ln2 = nn.LayerNorm(self.token_dim)

        duel_in_dim = self.token_dim * 3 + 4
        self.duel_mlp = nn.Sequential(
            nn.Linear(duel_in_dim, self.duel_dim),
            nn.GELU(),
            nn.LayerNorm(self.duel_dim),
            nn.Linear(self.duel_dim, self.duel_dim),
            nn.GELU(),
            nn.LayerNorm(self.duel_dim),
        )
        self.duel_attn = nn.Linear(self.duel_dim, 1)
        self.comp_head = nn.Sequential(
            nn.Linear(self.token_dim + self.duel_dim + 5, self.group_hidden),
            nn.GELU(),
            nn.LayerNorm(self.group_hidden),
            nn.Linear(self.group_hidden, 1),
        )
        self.beta_head = nn.Sequential(
            nn.Linear(self.token_dim + self.duel_dim + 4, self.group_hidden),
            nn.GELU(),
            nn.LayerNorm(self.group_hidden),
            nn.Linear(self.group_hidden, 1),
        )
        self.group_head = nn.Sequential(
            nn.Linear(self.token_dim + 8, self.group_hidden),
            nn.GELU(),
            nn.LayerNorm(self.group_hidden),
        )
        self.gate_head = nn.Linear(self.group_hidden, 1)
        self.null_head = nn.Linear(self.group_hidden, 1)
        self.delta_head = nn.Linear(self.group_hidden, 1)
        self._init_safe_noop()

    def _init_safe_noop(self) -> None:
        # Safe plugin default: start near the anchor path and require training evidence
        # before the reranker or null branch can noticeably change decisions.
        comp_last = self.comp_head[-1]
        beta_last = self.beta_head[-1]
        nn.init.zeros_(comp_last.weight)
        nn.init.zeros_(comp_last.bias)
        nn.init.zeros_(beta_last.weight)
        nn.init.constant_(beta_last.bias, -4.0)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -4.0)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.constant_(self.delta_head.bias, -4.0)
        nn.init.zeros_(self.null_head.weight)
        nn.init.constant_(self.null_head.bias, -8.0)

    def get_config(self) -> dict[str, Any]:
        return {
            "scalar_dim": self.scalar_dim,
            "feat_dim": self.feat_dim,
            "scalar_hidden": self.scalar_hidden,
            "scalar_out": self.scalar_out,
            "temporal_hidden": self.temporal_hidden,
            "temporal_out": self.temporal_out,
            "token_dim": self.token_dim,
            "duel_dim": self.duel_dim,
            "group_hidden": self.group_hidden,
            "num_heads": self.num_heads,
            "topk": self.topk,
            "delta_scale": self.delta_scale,
            "margin_threshold": self.margin_threshold,
            "margin_temperature": self.margin_temperature,
            "gate_cap": self.gate_cap,
        }

    @classmethod
    def from_checkpoint(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "RuntimeReplayAssociationAdapter":
        ckpt = torch.load(str(path), map_location=map_location)
        config = dict(ckpt.get("config", {}))
        model = cls(**config)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        return model

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            "config": self.get_config(),
            "state_dict": self.state_dict(),
            "feature_names": list(RUNTIME_REPLAY_FEATURE_NAMES),
        }

    def _build_meta_features(self, anchor_scores: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        margin, entropy = _group_rank_margin(anchor_scores, valid_mask)
        rank_frac = torch.zeros_like(anchor_scores)
        logits = _masked_fill_logits(anchor_scores, valid_mask, fill=-1.0)
        order = torch.argsort(logits, dim=1, descending=True)
        rank_vals = torch.arange(anchor_scores.shape[1], device=anchor_scores.device, dtype=torch.float32)
        rank_grid = rank_vals.unsqueeze(0).expand_as(anchor_scores)
        rank_frac.scatter_(1, order, rank_grid / float(max(anchor_scores.shape[1] - 1, 1)))
        anchor_logits = torch.logit(anchor_scores.clamp(min=1e-4, max=1.0 - 1e-4), eps=1e-4)
        meta = torch.stack(
            [
                anchor_scores,
                anchor_logits,
                rank_frac,
                margin.unsqueeze(1).expand_as(anchor_scores),
                entropy.unsqueeze(1).expand_as(anchor_scores),
                valid_mask.to(dtype=anchor_scores.dtype),
            ],
            dim=-1,
        )
        return meta, margin, entropy

    def forward(
        self,
        anchor_scores: torch.Tensor,
        scalar_features: torch.Tensor,
        det_features: torch.Tensor,
        hist_features: torch.Tensor,
        hist_masks: torch.Tensor,
        hist_times: Optional[torch.Tensor] = None,
        det_times: Optional[torch.Tensor] = None,
        det_scores: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if anchor_scores.dim() == 1:
            anchor_scores = anchor_scores.unsqueeze(0)
        if scalar_features.dim() == 2:
            scalar_features = scalar_features.unsqueeze(0)
        if det_features.dim() == 1:
            det_features = det_features.unsqueeze(0)
        if hist_features.dim() == 3:
            hist_features = hist_features.unsqueeze(0)
        if hist_masks.dim() == 2:
            hist_masks = hist_masks.unsqueeze(0)
        if hist_times is not None and hist_times.dim() == 2:
            hist_times = hist_times.unsqueeze(0)

        batch, cand = anchor_scores.shape
        if valid_mask is None:
            valid_mask = torch.ones((batch, cand), device=anchor_scores.device, dtype=torch.bool)
        else:
            valid_mask = valid_mask.to(device=anchor_scores.device, dtype=torch.bool)

        scalar_embed = self.scalar_tower(scalar_features)
        temp_out = self.temporal_tower(
            det_features=det_features,
            hist_features=hist_features,
            hist_masks=hist_masks,
            hist_times=hist_times,
            det_times=det_times,
        )
        temp_embed = temp_out["embedding"]

        meta, margin, entropy = self._build_meta_features(anchor_scores, valid_mask)
        meta_embed = self.meta_proj(meta)

        token = self.token_proj(torch.cat([scalar_embed, temp_embed, meta_embed], dim=-1))
        attn_out, _ = self.self_attn(token, token, token, key_padding_mask=~valid_mask)
        token = self.ln1(token + attn_out)
        token = self.ln2(token + self.ffn(token))

        ti = token.unsqueeze(2).expand(-1, -1, cand, -1)
        tk = token.unsqueeze(1).expand(-1, cand, -1, -1)
        anchor_logits = torch.logit(anchor_scores.clamp(min=1e-4, max=1.0 - 1e-4), eps=1e-4)
        logit_delta = anchor_logits.unsqueeze(2) - anchor_logits.unsqueeze(1)
        score_delta = anchor_scores.unsqueeze(2) - anchor_scores.unsqueeze(1)
        pair_group = torch.stack(
            [
                logit_delta,
                margin.unsqueeze(1).unsqueeze(2).expand(-1, cand, cand),
                entropy.unsqueeze(1).unsqueeze(2).expand(-1, cand, cand),
            ],
            dim=-1,
        )
        duel_input = torch.cat([ti, tk, ti - tk, score_delta.unsqueeze(-1), pair_group], dim=-1)
        duel_embed = self.duel_mlp(duel_input)
        duel_attn = self.duel_attn(duel_embed).squeeze(-1)
        pair_mask = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
        eye = torch.eye(cand, device=anchor_scores.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = pair_mask & (~eye)
        duel_attn = duel_attn.masked_fill(~pair_mask, -1e9)
        duel_weight = torch.softmax(duel_attn, dim=-1)
        duel_weight = duel_weight * pair_mask.to(dtype=duel_weight.dtype)
        duel_weight = duel_weight / duel_weight.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        duel_context = torch.sum(duel_weight.unsqueeze(-1) * duel_embed, dim=2)

        hist_valid = (~hist_masks).to(dtype=anchor_scores.dtype)
        hist_count = hist_valid.sum(dim=-1)
        hist_ratio = hist_count / float(max(hist_masks.shape[-1], 1))
        temp_conf = temp_out["temp_conf"]
        temp_unc = temp_out["temp_uncertainty"]
        comp_input = torch.cat(
            [
                token,
                duel_context,
                anchor_scores.unsqueeze(-1),
                anchor_logits.unsqueeze(-1),
                temp_conf.unsqueeze(-1),
                temp_unc.unsqueeze(-1),
                hist_ratio.unsqueeze(-1),
            ],
            dim=-1,
        )
        residual = self.comp_head(comp_input).squeeze(-1)
        beta = torch.sigmoid(
            self.beta_head(
                torch.cat(
                    [
                        token,
                        duel_context,
                        anchor_scores.unsqueeze(-1),
                        temp_conf.unsqueeze(-1),
                        hist_ratio.unsqueeze(-1),
                        temp_out["sim_mean"].unsqueeze(-1),
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
        )

        token_mean = _masked_mean(token, valid_mask.unsqueeze(-1), dim=1, keepdim=False)
        candidate_count = valid_mask.sum(dim=1).to(dtype=anchor_scores.dtype)
        mean_hist_ratio = _masked_mean(hist_ratio, valid_mask, dim=1, keepdim=False)
        mean_temp_conf = _masked_mean(temp_conf, valid_mask, dim=1, keepdim=False)
        mean_temp_unc = _masked_mean(temp_unc, valid_mask, dim=1, keepdim=False)
        if det_scores is None:
            det_scores = torch.ones((batch,), device=anchor_scores.device, dtype=anchor_scores.dtype)
        else:
            det_scores = det_scores.to(device=anchor_scores.device, dtype=anchor_scores.dtype).view(-1)
        group_features = torch.stack(
            [
                margin,
                entropy,
                candidate_count / float(max(cand, 1)),
                mean_hist_ratio,
                mean_temp_conf,
                mean_temp_unc,
                det_scores,
                temp_out["sim_std"].masked_fill(~valid_mask, 0.0).max(dim=1).values,
            ],
            dim=-1,
        )
        group_embed = self.group_head(torch.cat([token_mean, group_features], dim=-1))
        ambiguity_prior = torch.sigmoid((self.margin_threshold - margin) / self.margin_temperature)
        learned_gate = torch.sigmoid(self.gate_head(group_embed).squeeze(-1))
        group_activation = (self.gate_cap * ambiguity_prior * learned_gate * mean_hist_ratio.clamp(min=0.0, max=1.0)).clamp(0.0, self.gate_cap)
        null_logit = self.null_head(group_embed).squeeze(-1)
        delta_cap = self.delta_scale * torch.sigmoid(self.delta_head(group_embed).squeeze(-1))

        residual = residual.masked_fill(~valid_mask, 0.0)
        residual_center = residual - _masked_mean(residual, valid_mask, dim=1, keepdim=True)
        final_logits = anchor_logits + group_activation.unsqueeze(-1) * beta * delta_cap.unsqueeze(-1) * torch.tanh(residual_center)
        final_logits = _masked_fill_logits(final_logits, valid_mask, fill=-20.0)
        null_prob = torch.sigmoid(null_logit)
        final_scores = (1.0 - null_prob.unsqueeze(-1)) * torch.sigmoid(final_logits)
        final_scores = final_scores * valid_mask.to(dtype=final_scores.dtype)
        candidate_logprob = torch.log(final_scores.clamp(min=1e-8))
        candidate_logprob = _masked_fill_logits(candidate_logprob, valid_mask, fill=-1e9)
        joint_logits = torch.cat([candidate_logprob, torch.log(null_prob.clamp(min=1e-8)).unsqueeze(-1)], dim=1)

        return {
            "anchor_logits": anchor_logits,
            "candidate_logits": final_logits,
            "candidate_scores": final_scores,
            "candidate_logprob": candidate_logprob,
            "joint_logits": joint_logits,
            "null_logit": null_logit,
            "null_prob": null_prob,
            "group_activation": group_activation,
            "delta_cap": delta_cap,
            "candidate_beta": beta,
            "residual": residual,
            "residual_centered": residual_center,
            "margin": margin,
            "entropy": entropy,
            "temp_conf": temp_conf,
            "temp_uncertainty": temp_unc,
            "hist_ratio": hist_ratio,
            "valid_mask": valid_mask,
        }


def save_runtime_replay_checkpoint(
    model: RuntimeReplayAssociationAdapter,
    path: str | Path,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    payload = model.to_checkpoint()
    if extra:
        payload.update(extra)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
