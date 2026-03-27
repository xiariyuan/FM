import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from tracker.laplace_assoc import _normalize_rows, _track_history, laplace_fuse_distance


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


ATCR_DUEL_EXTRA_DIMS = 3


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _gelu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.clip(ex.sum(axis=axis, keepdims=True), 1e-12, None)


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.matmul(x, weight) + bias.reshape((1,) * (x.ndim - 1) + (-1,))


def _history_similarity_features(det_feat: np.ndarray, hist_feat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    det_feat = _normalize_rows(det_feat)[0]
    hist_feat = _normalize_rows(hist_feat)
    sims = np.clip((np.matmul(hist_feat, det_feat) + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
    if sims.size == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    ages = np.arange(sims.shape[0] - 1, -1, -1, dtype=np.float32)
    age_log = np.log1p(ages)
    delta = np.zeros_like(sims)
    if sims.shape[0] > 1:
        delta[1:] = sims[1:] - sims[:-1]
    return sims, age_log, delta, hist_feat


@dataclass
class HACAV1Checkpoint:
    version: str
    anchor_alpha: float
    delta_scale: float
    min_history: int
    max_history: int
    decay_scales: np.ndarray
    use_set_encoder: bool
    use_background: bool
    use_hist_gate: bool
    use_ood_gate: bool
    ood_scale: float
    W_hist1: np.ndarray
    b_hist1: np.ndarray
    W_hist2: np.ndarray
    b_hist2: np.ndarray
    W_hist_attn: np.ndarray
    b_hist_attn: np.ndarray
    W_pair: np.ndarray
    b_pair: np.ndarray
    W_set: Optional[np.ndarray]
    b_set: Optional[np.ndarray]
    W_hist_gate1: Optional[np.ndarray]
    b_hist_gate1: Optional[np.ndarray]
    W_hist_gate2: Optional[np.ndarray]
    b_hist_gate2: Optional[np.ndarray]
    W_delta: np.ndarray
    b_delta: np.ndarray
    W_beta: np.ndarray
    b_beta: np.ndarray
    W_bg: np.ndarray
    b_bg: np.ndarray
    token_mean: Optional[np.ndarray]
    token_std: Optional[np.ndarray]
    ood_threshold: float
    comp_topk: int = 0
    comp_margin_threshold: float = 0.0
    comp_margin_temperature: float = 0.03
    comp_delta_scale: float = 1.0
    W_duel1: Optional[np.ndarray] = None
    b_duel1: Optional[np.ndarray] = None
    W_duel2: Optional[np.ndarray] = None
    b_duel2: Optional[np.ndarray] = None
    W_attn: Optional[np.ndarray] = None
    b_attn: Optional[np.ndarray] = None
    W_comp1: Optional[np.ndarray] = None
    b_comp1: Optional[np.ndarray] = None
    W_comp2: Optional[np.ndarray] = None
    b_comp2: Optional[np.ndarray] = None

    @classmethod
    def from_npz(cls, path: str) -> "HACAV1Checkpoint":
        if not path:
            raise ValueError("Missing HACA checkpoint path")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing HACA checkpoint file: {path}")
        z = np.load(path, allow_pickle=True)

        version = str(np.asarray(z["version"]).reshape(-1)[0]) if "version" in z.files else "haca_v1"
        if version not in {"haca_v1", "haca_v2", "haca_v3"}:
            raise ValueError(f"Unsupported HACA checkpoint version: {version}")

        def req(name: str) -> np.ndarray:
            if name not in z.files:
                raise ValueError(f"Missing required HACA checkpoint tensor: {name}")
            return np.asarray(z[name], dtype=np.float32)

        def opt(name: str) -> Optional[np.ndarray]:
            if name not in z.files:
                return None
            value = np.asarray(z[name], dtype=np.float32)
            return None if value.size == 0 else value

        return cls(
            version=version,
            anchor_alpha=float(np.asarray(z["anchor_alpha"], dtype=np.float32).reshape(-1)[0]),
            delta_scale=float(np.asarray(z["delta_scale"], dtype=np.float32).reshape(-1)[0]),
            min_history=int(np.asarray(z["min_history"], dtype=np.int32).reshape(-1)[0]),
            max_history=int(np.asarray(z["max_history"], dtype=np.int32).reshape(-1)[0]),
            decay_scales=np.asarray(z["decay_scales"], dtype=np.float32).reshape(-1),
            use_set_encoder=bool(int(np.asarray(z["use_set_encoder"], dtype=np.int32).reshape(-1)[0])),
            use_background=bool(int(np.asarray(z["use_background"], dtype=np.int32).reshape(-1)[0])),
            use_hist_gate=bool(int(np.asarray(z["use_hist_gate"], dtype=np.int32).reshape(-1)[0])) if "use_hist_gate" in z.files else False,
            use_ood_gate=bool(int(np.asarray(z["use_ood_gate"], dtype=np.int32).reshape(-1)[0])) if "use_ood_gate" in z.files else False,
            ood_scale=float(np.asarray(z["ood_scale"], dtype=np.float32).reshape(-1)[0]) if "ood_scale" in z.files else 0.0,
            W_hist1=req("W_hist1"),
            b_hist1=req("b_hist1"),
            W_hist2=req("W_hist2"),
            b_hist2=req("b_hist2"),
            W_hist_attn=req("W_hist_attn"),
            b_hist_attn=req("b_hist_attn"),
            W_pair=req("W_pair"),
            b_pair=req("b_pair"),
            W_set=opt("W_set"),
            b_set=opt("b_set"),
            W_hist_gate1=opt("W_hist_gate1"),
            b_hist_gate1=opt("b_hist_gate1"),
            W_hist_gate2=opt("W_hist_gate2"),
            b_hist_gate2=opt("b_hist_gate2"),
            W_delta=req("W_delta"),
            b_delta=req("b_delta"),
            W_beta=req("W_beta"),
            b_beta=req("b_beta"),
            W_bg=req("W_bg"),
            b_bg=req("b_bg"),
            token_mean=opt("token_mean"),
            token_std=opt("token_std"),
            ood_threshold=float(np.asarray(z["ood_threshold"], dtype=np.float32).reshape(-1)[0]) if "ood_threshold" in z.files else float("inf"),
            comp_topk=int(np.asarray(z["comp_topk"], dtype=np.int32).reshape(-1)[0]) if "comp_topk" in z.files else 0,
            comp_margin_threshold=float(np.asarray(z["comp_margin_threshold"], dtype=np.float32).reshape(-1)[0]) if "comp_margin_threshold" in z.files else 0.0,
            comp_margin_temperature=float(np.asarray(z["comp_margin_temperature"], dtype=np.float32).reshape(-1)[0]) if "comp_margin_temperature" in z.files else 0.03,
            comp_delta_scale=float(np.asarray(z["comp_delta_scale"], dtype=np.float32).reshape(-1)[0]) if "comp_delta_scale" in z.files else 1.0,
            W_duel1=opt("W_duel1"),
            b_duel1=opt("b_duel1"),
            W_duel2=opt("W_duel2"),
            b_duel2=opt("b_duel2"),
            W_attn=opt("W_attn"),
            b_attn=opt("b_attn"),
            W_comp1=opt("W_comp1"),
            b_comp1=opt("b_comp1"),
            W_comp2=opt("W_comp2"),
            b_comp2=opt("b_comp2"),
        )

    def temporal_summary(self, det_feat: np.ndarray, hist_feat: np.ndarray) -> tuple[float, float, float, float, np.ndarray]:
        sims, age_log, delta, hist_feat = _history_similarity_features(det_feat, hist_feat)
        if sims.size == 0:
            return 0.0, 0.0, 0.0, 0.0, np.zeros((0,), dtype=np.float32)

        step_feat = np.stack([sims, age_log, delta], axis=-1).astype(np.float32)
        h = _gelu(_linear(step_feat, self.W_hist1, self.b_hist1))
        h = _gelu(_linear(h, self.W_hist2, self.b_hist2))
        attn_logits = _linear(h, self.W_hist_attn, self.b_hist_attn).reshape(-1)
        attn = _softmax(attn_logits, axis=0).astype(np.float32)

        temp_sim = float(np.sum(attn * sims))
        last_sim = float(sims[-1])
        max_sim = float(np.max(sims))
        std_sim = float(np.std(sims))
        return temp_sim, last_sim, max_sim, std_sim, attn

    def hist_gate(self, hist_state: np.ndarray) -> np.ndarray:
        hist_state = np.asarray(hist_state, dtype=np.float32)
        if (
            not self.use_hist_gate
            or self.W_hist_gate1 is None
            or self.b_hist_gate1 is None
            or self.W_hist_gate2 is None
            or self.b_hist_gate2 is None
            or hist_state.size == 0
        ):
            return np.ones((hist_state.shape[0],), dtype=np.float32)
        hidden = _gelu(_linear(hist_state, self.W_hist_gate1, self.b_hist_gate1))
        return _sigmoid(_linear(hidden, self.W_hist_gate2, self.b_hist_gate2).reshape(-1)).astype(np.float32)

    def ood_gate(self, pair_tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pair_tokens = np.asarray(pair_tokens, dtype=np.float32)
        if (
            not self.use_ood_gate
            or self.token_mean is None
            or self.token_std is None
            or pair_tokens.size == 0
        ):
            return np.ones((pair_tokens.shape[0],), dtype=np.float32), np.zeros((pair_tokens.shape[0],), dtype=np.float32)
        token_mean = np.asarray(self.token_mean, dtype=np.float32).reshape(1, -1)
        token_std = np.clip(np.asarray(self.token_std, dtype=np.float32).reshape(1, -1), 1e-6, None)
        z = (pair_tokens - token_mean) / token_std
        score = np.sqrt(np.mean(np.square(z), axis=1)).astype(np.float32)
        gate = np.exp(-float(self.ood_scale) * np.clip(score - float(self.ood_threshold), a_min=0.0, a_max=None)).astype(np.float32)
        return gate, score

    @property
    def has_competition_head(self) -> bool:
        return (
            self.version == "haca_v3"
            and self.comp_topk > 0
            and self.W_duel1 is not None
            and self.b_duel1 is not None
            and self.W_duel2 is not None
            and self.b_duel2 is not None
            and self.W_attn is not None
            and self.b_attn is not None
            and self.W_comp1 is not None
            and self.b_comp1 is not None
            and self.W_comp2 is not None
            and self.b_comp2 is not None
        )

    def competitive_refine(
        self,
        pair_tokens: np.ndarray,
        base_sim: np.ndarray,
        beta_hist: np.ndarray,
        beta_ood: np.ndarray,
        bg_prob: float,
    ) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
        pair_tokens = np.asarray(pair_tokens, dtype=np.float32)
        base_sim = np.asarray(base_sim, dtype=np.float32).reshape(-1)
        beta_hist = np.asarray(beta_hist, dtype=np.float32).reshape(-1)
        beta_ood = np.asarray(beta_ood, dtype=np.float32).reshape(-1)

        group_size = int(base_sim.shape[0])
        if group_size == 0:
            empty = np.zeros((0,), dtype=np.float32)
            return empty, {
                "active": empty,
                "margin": 0.0,
                "entropy": 0.0,
                "topk_mask": empty,
                "residual": empty,
                "rank_before": empty,
                "rank_after": empty,
            }

        logits0 = np.log(np.clip(base_sim, 1e-4, 1.0 - 1e-4) / np.clip(1.0 - base_sim, 1e-4, 1.0))
        k = min(max(int(self.comp_topk), 1), group_size)
        top_idx = np.argsort(-logits0)[:k]
        top_scores = base_sim[top_idx]
        margin = float(top_scores[0]) if top_scores.size == 1 else float(top_scores[0] - top_scores[1])
        probs = _softmax(logits0[top_idx], axis=0)
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
        trust = float(np.mean(beta_hist[top_idx] * beta_ood[top_idx]))
        activation = float((1.0 - float(np.clip(bg_prob, 0.0, 1.0))) * trust * _sigmoid((float(self.comp_margin_threshold) - margin) / max(float(self.comp_margin_temperature), 1e-4)))

        refined_logits = logits0.copy()
        residual_full = np.zeros((group_size,), dtype=np.float32)
        topk_mask = np.zeros((group_size,), dtype=np.float32)
        topk_mask[top_idx] = 1.0
        active_full = np.full((group_size,), activation, dtype=np.float32)

        order_before = np.argsort(-logits0)
        rank_before = np.empty_like(order_before)
        rank_before[order_before] = np.arange(order_before.shape[0], dtype=np.int64)
        rank_before = (rank_before.astype(np.float32) / float(max(group_size, 1))).astype(np.float32)

        if self.has_competition_head and activation > 1e-6 and top_idx.size > 1:
            residuals = []
            duel_hidden_size = int(self.W_duel2.shape[1])
            ctx_zeros = np.zeros((duel_hidden_size,), dtype=np.float32)
            for idx_i in top_idx.tolist():
                rivals = [idx_k for idx_k in top_idx.tolist() if idx_k != idx_i]
                if not rivals:
                    residuals.append(0.0)
                    continue

                xi = pair_tokens[idx_i]
                xk = pair_tokens[rivals]
                xi_rep = np.repeat(xi[None, :], len(rivals), axis=0)
                margin_col = np.full((len(rivals), 1), margin, dtype=np.float32)
                entropy_col = np.full((len(rivals), 1), entropy, dtype=np.float32)
                zdiff = (logits0[idx_i] - logits0[rivals]).astype(np.float32).reshape(-1, 1)
                duel_in = np.concatenate([xi_rep, xk, xi_rep - xk, zdiff, margin_col, entropy_col], axis=-1).astype(np.float32)

                duel_h = _gelu(_linear(duel_in, self.W_duel1, self.b_duel1))
                duel_h = _gelu(_linear(duel_h, self.W_duel2, self.b_duel2))
                attn = _softmax(_linear(duel_h, self.W_attn, self.b_attn).reshape(-1), axis=0).astype(np.float32)
                ctx = (attn[:, None] * duel_h).sum(axis=0) if duel_h.size > 0 else ctx_zeros

                comp_in = np.concatenate(
                    [
                        xi,
                        ctx.astype(np.float32),
                        np.asarray([logits0[idx_i], margin, entropy], dtype=np.float32),
                    ],
                    axis=0,
                )[None, :]
                comp_h = _gelu(_linear(comp_in, self.W_comp1, self.b_comp1))
                residuals.append(float(_linear(comp_h, self.W_comp2, self.b_comp2).reshape(-1)[0]))

            residuals = np.asarray(residuals, dtype=np.float32)
            residuals = residuals - residuals.mean()
            refined_logits[top_idx] = logits0[top_idx] + activation * float(self.comp_delta_scale) * np.tanh(residuals)
            residual_full[top_idx] = residuals

        refined_sim = _sigmoid(refined_logits).astype(np.float32)
        order_after = np.argsort(-refined_logits)
        rank_after = np.empty_like(order_after)
        rank_after[order_after] = np.arange(order_after.shape[0], dtype=np.int64)
        rank_after = (rank_after.astype(np.float32) / float(max(group_size, 1))).astype(np.float32)

        return refined_sim, {
            "active": active_full,
            "margin": margin,
            "entropy": entropy,
            "topk_mask": topk_mask,
            "residual": residual_full,
            "rank_before": rank_before,
            "rank_after": rank_after,
        }


def haca_fuse_distance(
    tracks,
    detections,
    spatial_cost,
    motion_cost,
    checkpoint: HACAV1Checkpoint,
    det_scores=None,
    decay_scales=None,
    min_history=None,
    proto_mode="multi",
    use_det_score=True,
    track_gaps=None,
    valid_mask=None,
    use_set_encoder: Optional[bool] = None,
    use_background: Optional[bool] = None,
    delta_scale: Optional[float] = None,
    return_debug: bool = False,
):
    spatial_cost = np.asarray(spatial_cost, dtype=np.float32)
    motion_cost = np.asarray(motion_cost, dtype=np.float32)
    if spatial_cost.size == 0:
        if return_debug:
            return spatial_cost, None
        return spatial_cost

    use_set_encoder = checkpoint.use_set_encoder if use_set_encoder is None else bool(use_set_encoder)
    use_background = checkpoint.use_background if use_background is None else bool(use_background)
    delta_scale = float(checkpoint.delta_scale if delta_scale is None else delta_scale)

    decay_scales = checkpoint.decay_scales if decay_scales is None else decay_scales
    min_history = checkpoint.min_history if min_history is None else min_history

    anchor_out = laplace_fuse_distance(
        tracks=tracks,
        detections=detections,
        spatial_cost=spatial_cost,
        motion_cost=motion_cost,
        det_scores=det_scores,
        decay_scales=decay_scales,
        appearance_alpha=checkpoint.anchor_alpha,
        min_history=min_history,
        proto_mode=proto_mode,
        use_reliability=True,
        use_det_score=use_det_score,
        track_gaps=track_gaps,
        valid_mask=valid_mask,
        calibrator=None,
        use_pole_bank=False,
        return_debug=True,
    )
    anchor_cost, anchor_debug = anchor_out
    if anchor_debug is None:
        if return_debug:
            return anchor_cost, None
        return anchor_cost

    if valid_mask is None:
        valid_mask = np.ones_like(anchor_cost, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    det_feats = []
    for det in detections:
        feat = getattr(det, "curr_feat", None)
        if feat is None:
            if return_debug:
                return anchor_cost, anchor_debug
            return anchor_cost
        det_feats.append(_normalize_rows(np.asarray(feat, dtype=np.float32))[0])
    det_feats = np.stack(det_feats, axis=0).astype(np.float32)

    if det_scores is None:
        det_scores = np.ones((len(detections),), dtype=np.float32)
    det_scores = np.asarray(det_scores, dtype=np.float32).reshape(-1)
    if track_gaps is None:
        track_gaps = np.zeros((len(tracks),), dtype=np.float32)
    track_gaps = np.asarray(track_gaps, dtype=np.float32).reshape(-1)

    anchor_sim = np.asarray(anchor_debug["fused_sim"], dtype=np.float32)
    spatial_sim = np.asarray(anchor_debug["spatial_sim"], dtype=np.float32)
    motion_sim = np.asarray(anchor_debug["motion_sim"], dtype=np.float32)
    stability = np.asarray(anchor_debug["stability"], dtype=np.float32).reshape(-1)
    coherence = np.asarray(anchor_debug["coherence"], dtype=np.float32).reshape(-1)

    final_sim = anchor_sim.copy()
    learned_delta = np.zeros_like(anchor_sim, dtype=np.float32)
    learned_beta = np.zeros_like(anchor_sim, dtype=np.float32)
    learned_beta_pred = np.zeros_like(anchor_sim, dtype=np.float32)
    learned_beta_hist = np.zeros_like(anchor_sim, dtype=np.float32)
    learned_beta_ood = np.zeros_like(anchor_sim, dtype=np.float32)
    learned_ood_score = np.zeros_like(anchor_sim, dtype=np.float32)
    background_gate = np.zeros((len(detections),), dtype=np.float32)
    temporal_sim = np.zeros_like(anchor_sim, dtype=np.float32)
    history_last = np.zeros_like(anchor_sim, dtype=np.float32)
    history_max = np.zeros_like(anchor_sim, dtype=np.float32)
    history_std = np.zeros_like(anchor_sim, dtype=np.float32)
    anchor_z = np.zeros_like(anchor_sim, dtype=np.float32)
    anchor_margin = np.zeros_like(anchor_sim, dtype=np.float32)
    anchor_rank = np.zeros_like(anchor_sim, dtype=np.float32)
    comp_active = np.zeros_like(anchor_sim, dtype=np.float32)
    comp_margin = np.zeros((len(detections),), dtype=np.float32)
    comp_entropy = np.zeros((len(detections),), dtype=np.float32)
    comp_topk = np.zeros_like(anchor_sim, dtype=np.float32)
    comp_residual = np.zeros_like(anchor_sim, dtype=np.float32)
    comp_rank_before = np.zeros_like(anchor_sim, dtype=np.float32)
    comp_rank_after = np.zeros_like(anchor_sim, dtype=np.float32)

    for det_idx, det_feat in enumerate(det_feats):
        rows = np.where(valid_mask[:, det_idx])[0]
        if rows.size == 0:
            continue

        pair_tokens = []
        pair_embed = []
        for row in rows.tolist():
            hist = _track_history(tracks[row])
            if checkpoint.max_history > 0 and hist.shape[0] > checkpoint.max_history:
                hist = hist[-int(checkpoint.max_history) :]
            temp_sim, last_sim, max_sim, std_sim, _ = checkpoint.temporal_summary(det_feat=det_feat, hist_feat=hist)
            temporal_sim[row, det_idx] = temp_sim
            history_last[row, det_idx] = last_sim
            history_max[row, det_idx] = max_sim
            history_std[row, det_idx] = std_sim

        anchor_vals = anchor_sim[rows, det_idx].astype(np.float32)
        mean_val = float(np.mean(anchor_vals))
        std_val = float(np.std(anchor_vals))
        std_val = max(std_val, 1e-6)
        order = np.argsort(-anchor_vals)
        rank_pos = np.empty_like(order)
        rank_pos[order] = np.arange(order.shape[0], dtype=np.int64)
        for local_idx, row in enumerate(rows.tolist()):
            if rows.size == 1:
                margin = float(np.clip(anchor_vals[local_idx], 0.0, 1.0))
            else:
                other = np.delete(anchor_vals, local_idx)
                margin = float(anchor_vals[local_idx] - np.max(other))
            gap_log = float(np.log1p(max(float(track_gaps[row]), 0.0)))
            hist = _track_history(tracks[row])
            hist_len = int(hist.shape[0])
            hist_norm = float(min(1.0, float(hist_len) / float(max(int(min_history), 1))))
            anchor_z[row, det_idx] = float((anchor_vals[local_idx] - mean_val) / std_val)
            anchor_margin[row, det_idx] = margin
            anchor_rank[row, det_idx] = float(rank_pos[local_idx]) / float(max(rows.size, 1))
            token = np.asarray(
                [
                    float(anchor_sim[row, det_idx]),
                    float(spatial_sim[row, det_idx]),
                    float(motion_sim[row, det_idx]),
                    float(temporal_sim[row, det_idx]),
                    float(history_last[row, det_idx]),
                    float(history_max[row, det_idx]),
                    float(history_std[row, det_idx]),
                    gap_log,
                    hist_norm,
                    float(stability[row]),
                    float(coherence[row]),
                    float(anchor_z[row, det_idx]),
                    float(anchor_margin[row, det_idx]),
                    float(anchor_rank[row, det_idx]),
                    float(np.clip(det_scores[det_idx], 0.0, 1.0)),
                ],
                dtype=np.float32,
            )
            pair_tokens.append(token)

        pair_tokens_arr = np.stack(pair_tokens, axis=0).astype(np.float32)
        pair_embed_arr = _gelu(_linear(pair_tokens_arr, checkpoint.W_pair, checkpoint.b_pair))
        if use_set_encoder and checkpoint.W_set is not None and checkpoint.b_set is not None:
            mean_embed = pair_embed_arr.mean(axis=0, keepdims=True).repeat(pair_embed_arr.shape[0], axis=0)
            max_embed = pair_embed_arr.max(axis=0, keepdims=True).repeat(pair_embed_arr.shape[0], axis=0)
            set_in = np.concatenate([pair_embed_arr, mean_embed, max_embed], axis=-1)
            pair_embed_arr = _gelu(_linear(set_in, checkpoint.W_set, checkpoint.b_set))

        group_context = pair_embed_arr.mean(axis=0, keepdims=True)
        bg = float(_sigmoid(_linear(group_context, checkpoint.W_bg, checkpoint.b_bg)).reshape(-1)[0]) if use_background else 0.0
        background_gate[det_idx] = bg

        delta = _linear(pair_embed_arr, checkpoint.W_delta, checkpoint.b_delta).reshape(-1)
        beta_pred = _sigmoid(_linear(pair_embed_arr, checkpoint.W_beta, checkpoint.b_beta).reshape(-1))
        hist_state = pair_tokens_arr[:, 7:11]
        beta_hist = checkpoint.hist_gate(hist_state)
        beta_ood, ood_score = checkpoint.ood_gate(pair_tokens_arr)
        beta = np.clip(beta_pred * beta_hist * beta_ood, 0.0, 1.0).astype(np.float32)
        anchor_group = anchor_vals
        anchor_logit = np.log(np.clip(anchor_group, 1e-4, 1.0 - 1e-4) / np.clip(1.0 - anchor_group, 1e-4, 1.0))
        s_tilde = _sigmoid(anchor_logit + delta_scale * np.tanh(delta))
        s_prebg = (((1.0 - beta) * anchor_group) + beta * s_tilde).astype(np.float32)

        comp_debug = None
        if checkpoint.version == "haca_v3" and checkpoint.has_competition_head:
            s_prebg, comp_debug = checkpoint.competitive_refine(
                pair_tokens=pair_tokens_arr,
                base_sim=s_prebg,
                beta_hist=beta_hist,
                beta_ood=beta_ood,
                bg_prob=bg,
            )

        s_final = ((1.0 - bg) * s_prebg).astype(np.float32)
        s_final = np.clip(s_final, 0.0, 1.0)

        for local_idx, row in enumerate(rows.tolist()):
            final_sim[row, det_idx] = s_final[local_idx]
            learned_delta[row, det_idx] = float(delta[local_idx])
            learned_beta[row, det_idx] = float(beta[local_idx])
            learned_beta_pred[row, det_idx] = float(beta_pred[local_idx])
            learned_beta_hist[row, det_idx] = float(beta_hist[local_idx])
            learned_beta_ood[row, det_idx] = float(beta_ood[local_idx])
            learned_ood_score[row, det_idx] = float(ood_score[local_idx])
            if comp_debug is not None:
                comp_active[row, det_idx] = float(comp_debug["active"][local_idx])
                comp_topk[row, det_idx] = float(comp_debug["topk_mask"][local_idx])
                comp_residual[row, det_idx] = float(comp_debug["residual"][local_idx])
                comp_rank_before[row, det_idx] = float(comp_debug["rank_before"][local_idx])
                comp_rank_after[row, det_idx] = float(comp_debug["rank_after"][local_idx])
                comp_margin[det_idx] = float(comp_debug["margin"])
                comp_entropy[det_idx] = float(comp_debug["entropy"])

    result = np.ones_like(anchor_cost, dtype=np.float32)
    result[valid_mask] = np.clip(1.0 - final_sim[valid_mask], 0.0, 1.0)

    if not return_debug:
        return result

    debug = dict(anchor_debug)
    debug.update(
        {
            "anchor_sim": anchor_sim,
            "fused_sim": final_sim,
            "learned_r": learned_beta,
            "final_sim": final_sim,
            "haca_delta": learned_delta,
            "haca_beta": learned_beta,
            "haca_beta_pred": learned_beta_pred,
            "haca_beta_hist": learned_beta_hist,
            "haca_beta_ood": learned_beta_ood,
            "haca_ood_score": learned_ood_score,
            "haca_background": background_gate,
            "haca_temp_sim": temporal_sim,
            "haca_hist_last": history_last,
            "haca_hist_max": history_max,
            "haca_hist_std": history_std,
            "haca_anchor_z": anchor_z,
            "haca_anchor_margin": anchor_margin,
            "haca_anchor_rank": anchor_rank,
            "haca_pair_token_names": HACA_PAIR_TOKEN_NAMES,
            "haca_comp_active": comp_active,
            "haca_comp_margin": comp_margin,
            "haca_comp_entropy": comp_entropy,
            "haca_comp_topk": comp_topk,
            "haca_comp_residual": comp_residual,
            "haca_comp_rank_before": comp_rank_before,
            "haca_comp_rank_after": comp_rank_after,
        }
    )
    return result, debug
