import numpy as np


PAIR_FEATURE_NAMES = (
    "spatial_sim",
    "laplace_sim",
    "motion_sim",
    "absdiff",
    "min_sim",
    "prod_sim",
    "agreement",
    "stability",
    "coherence",
    "det_score",
    "gap_log1p",
    "hist_norm",
    "amb_spa",
    "amb_lap",
    "amb_mot",
)

TRACK_FEATURE_NAMES = (
    "gap_log1p",
    "hist_norm",
    "stability",
    "coherence",
)


def _normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return x / denom


def _track_history(track):
    if hasattr(track, "laplace_history") and len(track.laplace_history) > 0:
        hist = np.asarray(list(track.laplace_history), dtype=np.float32)
        if hist.ndim == 1:
            hist = hist[None, :]
        return _normalize_rows(hist)

    feats = list(getattr(track, "features", []))
    if len(feats) == 0:
        base = getattr(track, "smooth_feat", None)
        if base is None:
            base = getattr(track, "curr_feat", None)
        if base is None:
            return np.zeros((0, 0), dtype=np.float32)
        feats = [base]

    hist = np.asarray(feats, dtype=np.float32)
    if hist.ndim == 1:
        hist = hist[None, :]
    return _normalize_rows(hist)


def _top2_margin(sim_mat, valid_mask, axis):
    sim_mat = np.asarray(sim_mat, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if axis == 1:
        length = sim_mat.shape[0]
        get_vals = lambda idx: sim_mat[idx][valid_mask[idx]]
    elif axis == 0:
        length = sim_mat.shape[1]
        get_vals = lambda idx: sim_mat[:, idx][valid_mask[:, idx]]
    else:
        raise ValueError(f"Unsupported axis for _top2_margin: {axis}")

    margins = np.zeros((length,), dtype=np.float32)
    for idx in range(length):
        vals = get_vals(idx)
        if vals.size == 0:
            margins[idx] = 0.0
        elif vals.size == 1:
            margins[idx] = float(np.clip(vals[0], 0.0, 1.0))
        else:
            top2 = np.partition(vals, -2)[-2:]
            margins[idx] = float(np.clip(top2.max() - top2.min(), 0.0, 1.0))
    return margins


def _pair_ambiguity(sim_mat, valid_mask):
    row_margin = _top2_margin(sim_mat, valid_mask, axis=1).reshape(-1, 1)
    col_margin = _top2_margin(sim_mat, valid_mask, axis=0).reshape(1, -1)
    return np.minimum(row_margin, col_margin).astype(np.float32)


def _build_exp_prototype(hist, tau):
    hist = _normalize_rows(hist)
    length = int(hist.shape[0])
    age = np.arange(length - 1, -1, -1, dtype=np.float32)
    tau = max(float(tau), 1e-3)
    w = np.exp(-age / tau).astype(np.float32)
    w = w / np.clip(w.sum(), 1e-12, None)
    proto = (w[:, None] * hist).sum(axis=0)
    return _normalize_rows(proto)[0]


def _track_stability(hist):
    hist = _normalize_rows(hist)
    length = int(hist.shape[0])
    if length < 3:
        return 1.0
    delta2 = hist[2:] - 2.0 * hist[1:-1] + hist[:-2]
    curvature = np.sqrt(np.mean(delta2 ** 2, axis=1)).mean()
    return float(np.exp(-curvature))


def _track_gate_features(length, stability, coherence, gap, min_history):
    hist_norm = min(1.0, max(0.0, float(length) / float(max(int(min_history), 1))))
    return np.asarray(
        [
            np.log1p(max(0.0, float(gap))),
            hist_norm,
            float(stability),
            float(coherence),
        ],
        dtype=np.float32,
    )


def _build_track_pole_bank(track, tau_values, gap, min_history, calibrator=None):
    hist = _track_history(track)
    if hist.shape[0] == 0:
        return None, None, 0.0, 0.0, 0, None

    last = hist[-1]
    tau_values = np.asarray(tau_values, dtype=np.float32).reshape(-1)
    if tau_values.size == 0:
        raise ValueError("tau_values must be non-empty for pole-bank inference")

    if hist.shape[0] < max(int(min_history), 1):
        protos = np.repeat(last[None, :], tau_values.size, axis=0).astype(np.float32)
        stability = 1.0
        coherence = 1.0
    else:
        protos = []
        proto_sims = []
        for tau in tau_values:
            proto = _build_exp_prototype(hist, tau=float(tau))
            protos.append(proto)
            proto_sims.append(((proto * last).sum() + 1.0) * 0.5)
        protos = np.stack(protos, axis=0).astype(np.float32)
        stability = _track_stability(hist)
        coherence = float(np.mean(proto_sims))

    length = int(hist.shape[0])
    track_feats = _track_gate_features(length, stability, coherence, gap, min_history)

    if calibrator is not None and calibrator.has_pole_bank:
        calibrator.validate_track_feature_names(TRACK_FEATURE_NAMES)
        pi = calibrator.predict_track_pi(track_feats[None, :])[0]
    else:
        pi = np.ones((protos.shape[0],), dtype=np.float32) / float(protos.shape[0])

    return protos, pi.astype(np.float32), stability, coherence, length, track_feats.astype(np.float32)


def _build_track_prototypes(track, decay_scales, min_history):
    hist = _track_history(track)
    if hist.shape[0] == 0:
        return None, 0.0, 0.0, 0

    last = hist[-1]
    if hist.shape[0] < max(int(min_history), 1):
        return last[None, :], 1.0, 1.0, int(hist.shape[0])

    age = np.arange(hist.shape[0] - 1, -1, -1, dtype=np.float32)
    protos = []
    proto_sims = []
    for scale in decay_scales:
        scale = max(float(scale), 1e-3)
        w = np.exp(-age / scale).astype(np.float32)
        w = w / np.clip(w.sum(), 1e-12, None)
        proto = (w[:, None] * hist).sum(axis=0)
        proto = _normalize_rows(proto)[0]
        protos.append(proto)
        proto_sims.append(((proto * last).sum() + 1.0) * 0.5)

    protos = np.stack(protos, axis=0).astype(np.float32)
    coherence = float(np.mean(proto_sims))
    stability = _track_stability(hist)
    return protos, stability, coherence, int(hist.shape[0])


def laplace_fuse_cost(
    tracks,
    detections,
    track_indices,
    detection_indices,
    spatial_cost,
    gating_threshold,
    decay_scales=(1.0, 2.0, 4.0),
    appearance_alpha=0.35,
    min_history=3,
    calibrator=None,
    use_pole_bank=True,
    return_debug=False,
):
    spatial_cost = np.asarray(spatial_cost, dtype=np.float32)
    if spatial_cost.size == 0:
        if return_debug:
            return spatial_cost, None
        return spatial_cost

    selected_dets = [detections[i] for i in detection_indices]
    det_feats = _normalize_rows(np.asarray([det.feature for det in selected_dets], dtype=np.float32))
    det_scores = np.asarray([det.confidence for det in selected_dets], dtype=np.float32)

    laplace_sim = np.zeros_like(spatial_cost, dtype=np.float32)
    stability = np.ones((len(track_indices),), dtype=np.float32)
    coherence = np.ones((len(track_indices),), dtype=np.float32)
    history_len = np.ones((len(track_indices),), dtype=np.float32)
    motion_sim = np.zeros_like(spatial_cost, dtype=np.float32)
    track_gaps = np.asarray([max(0, int(getattr(tracks[idx], "time_since_update", 1))) for idx in track_indices], dtype=np.float32)
    pole_weights = []
    track_gate_features = []

    measurements = np.asarray([det.to_xyah() for det in selected_dets], dtype=np.float32)
    gating_threshold = float(max(gating_threshold, 1e-6))
    valid_mask = np.ones_like(spatial_cost, dtype=bool)

    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        if calibrator is not None and calibrator.has_pole_bank and use_pole_bank:
            protos, pi, stab, coh, length, track_feats = _build_track_pole_bank(
                track=track,
                tau_values=np.asarray(calibrator.tau_values, dtype=np.float32),
                gap=float(track_gaps[row]),
                min_history=min_history,
                calibrator=calibrator,
            )
            stability[row] = stab
            coherence[row] = coh
            history_len[row] = float(length)
            pole_weights.append(pi)
            track_gate_features.append(track_feats)
            if protos is None:
                laplace_sim[row] = np.clip(1.0 - spatial_cost[row], 0.0, 1.0)
            else:
                sim = np.matmul(protos, det_feats.T)
                sim = np.clip((sim + 1.0) * 0.5, 0.0, 1.0)
                laplace_sim[row] = (pi[:, None] * sim).sum(axis=0)
        else:
            protos, stab, coh, length = _build_track_prototypes(track, decay_scales, min_history)
            stability[row] = stab
            coherence[row] = coh
            history_len[row] = float(length)
            pole_weights.append(None)
            track_gate_features.append(None)
            if protos is None:
                laplace_sim[row] = np.clip(1.0 - spatial_cost[row], 0.0, 1.0)
            else:
                sim = np.matmul(protos, det_feats.T)
                sim = np.clip((sim + 1.0) * 0.5, 0.0, 1.0)
                laplace_sim[row] = sim.mean(axis=0)

        gating_distance = track.kf.gating_distance(
            track.mean,
            track.covariance,
            measurements,
            only_position=False,
        )
        motion_sim[row] = np.clip(1.0 - gating_distance / gating_threshold, 0.0, 1.0)

    spatial_sim = np.clip(1.0 - spatial_cost, 0.0, 1.0)
    absdiff = np.abs(spatial_sim - laplace_sim)
    agreement = np.clip(1.0 - absdiff, 0.0, 1.0)
    prod_sim = np.clip(spatial_sim * laplace_sim, 0.0, 1.0)

    pair_rel = (
        0.30 * stability[:, None]
        + 0.30 * coherence[:, None]
        + 0.25 * agreement
        + 0.15 * np.clip(det_scores.reshape(1, -1), 0.0, 1.0)
    )
    pair_rel = np.clip(pair_rel, 0.0, 1.0)

    if calibrator is None:
        appearance_alpha = float(np.clip(appearance_alpha, 0.0, 1.0))
        appearance_sim = (1.0 - appearance_alpha) * spatial_sim + appearance_alpha * laplace_sim
        fused_sim = pair_rel * appearance_sim + (1.0 - pair_rel) * motion_sim
        learned_alpha = None
        learned_r = pair_rel
        result = np.clip(1.0 - fused_sim, 0.0, 1.0)
    else:
        gap_log = np.log1p(np.clip(track_gaps, 0.0, None)).astype(np.float32).reshape(-1, 1)
        gap_full = gap_log.repeat(spatial_sim.shape[1], axis=1)
        hist_factor = np.clip(history_len / float(max(int(min_history), 1)), 0.0, 1.0).astype(np.float32).reshape(-1, 1)
        hist_full = hist_factor.repeat(spatial_sim.shape[1], axis=1)
        stab_full = stability.reshape(-1, 1).repeat(spatial_sim.shape[1], axis=1)
        coh_full = coherence.reshape(-1, 1).repeat(spatial_sim.shape[1], axis=1)
        det_full = np.clip(det_scores, 0.0, 1.0).reshape(1, -1).repeat(spatial_sim.shape[0], axis=0)

        amb_spa = _pair_ambiguity(spatial_sim, valid_mask)
        amb_lap = _pair_ambiguity(laplace_sim, valid_mask)
        amb_mot = _pair_ambiguity(motion_sim, valid_mask)

        pair_features = np.stack(
            [
                spatial_sim,
                laplace_sim,
                motion_sim,
                absdiff,
                np.minimum(spatial_sim, laplace_sim),
                prod_sim,
                agreement,
                stab_full,
                coh_full,
                det_full,
                gap_full,
                hist_full,
                amb_spa,
                amb_lap,
                amb_mot,
            ],
            axis=-1,
        ).astype(np.float32)

        calibrator.validate_pair_feature_names(PAIR_FEATURE_NAMES)
        learned_alpha, learned_r = calibrator.predict_alpha_r(pair_features)
        learned_alpha = np.clip(learned_alpha, 0.0, 1.0)
        learned_r = np.clip(learned_r, 0.0, 1.0)

        appearance_sim = (1.0 - learned_alpha) * spatial_sim + learned_alpha * laplace_sim
        fused_sim = learned_r * appearance_sim + (1.0 - learned_r) * motion_sim
        result = np.clip(1.0 - fused_sim, 0.0, 1.0)

    if not return_debug:
        return result

    if calibrator is None:
        amb_spa = _pair_ambiguity(spatial_sim, valid_mask)
        amb_lap = _pair_ambiguity(laplace_sim, valid_mask)
        amb_mot = _pair_ambiguity(motion_sim, valid_mask)

    debug = {
        "pair_rel": pair_rel,
        "learned_alpha": learned_alpha,
        "learned_r": learned_r,
        "appearance_sim": appearance_sim,
        "fused_sim": fused_sim,
        "motion_sim": motion_sim,
        "spatial_sim": spatial_sim,
        "laplace_sim": laplace_sim,
        "agreement": agreement,
        "stability": stability,
        "coherence": coherence,
        "prod_sim": prod_sim,
        "amb_spa": amb_spa,
        "amb_lap": amb_lap,
        "amb_mot": amb_mot,
        "feature_names": PAIR_FEATURE_NAMES,
        "pair_feature_names": PAIR_FEATURE_NAMES,
        "track_feature_names": TRACK_FEATURE_NAMES,
        "pole_weights": pole_weights,
        "track_gate_features": track_gate_features,
        "tau_values": np.asarray(calibrator.tau_values, dtype=np.float32) if (calibrator is not None and calibrator.has_pole_bank) else None,
    }
    return result, debug
