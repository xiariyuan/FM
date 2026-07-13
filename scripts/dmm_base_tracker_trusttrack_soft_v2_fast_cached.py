#!/usr/bin/env python3
"""Fast mmap-backed TrustTrack-soft-v2 launcher.

This preserves the soft-v2 algorithm but replaces the expensive Python
track-by-detection loops and millions of per-pair dict entries with vectorized
NumPy matrices plus lazy pair lookup. It can also memory-map dump arrays from a
cache made by cache_detection_dump.py via TRUST_DUMP_CACHE_DIR.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

import dmm_base_tracker_trusttrack_soft_v2 as soft


class PairLookup:
    """Lazy mapping from (track_id, det_global_idx) to a matrix value."""

    def __init__(self, track_index, det_index, values, valid=None, default=None):
        self.track_index = track_index
        self.det_index = det_index
        self.values = values
        self.valid = valid
        self.default = default

    def get(self, key, default=None):
        try:
            ti = self.track_index[int(key[0])]
            di = self.det_index[int(key[1])]
        except (KeyError, TypeError, ValueError):
            return self.default if default is None else default
        if self.valid is not None and not bool(self.valid[ti, di]):
            return self.default if default is None else default
        value = self.values[ti, di]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def __len__(self):
        if self.valid is None:
            return int(self.values.size)
        return int(np.count_nonzero(self.valid))


class MetaLookup:
    """Lazy metadata dictionary for the actual matched pair only."""

    def __init__(self, track_index, det_index, arrays):
        self.track_index = track_index
        self.det_index = det_index
        self.arrays = arrays

    def get(self, key, default=None):
        try:
            ti = self.track_index[int(key[0])]
            di = self.det_index[int(key[1])]
        except (KeyError, TypeError, ValueError):
            return {} if default is None else default
        out = {}
        for name, arr in self.arrays.items():
            if arr.ndim == 1:
                value = arr[ti]
            else:
                value = arr[ti, di]
            out[name] = value.item() if isinstance(value, np.generic) else value
        return out

    def clear(self):
        # soft.main calls clear() after every frame. Matrices are released when
        # the next frame replaces the global lookup object.
        return None


def _row_margins(matrix: np.ndarray) -> np.ndarray:
    t, d = matrix.shape
    if d <= 1:
        return np.ones_like(matrix, dtype=np.float32)
    top_idx = np.argmax(matrix, axis=1)
    top1 = matrix[np.arange(t), top_idx]
    temp = matrix.copy()
    temp[np.arange(t), top_idx] = -np.inf
    top2 = np.max(temp, axis=1)
    other = np.broadcast_to(top1[:, None], matrix.shape).copy()
    other[np.arange(t), top_idx] = top2
    return (matrix - other).astype(np.float32, copy=False)


def _col_margins(matrix: np.ndarray) -> np.ndarray:
    t, d = matrix.shape
    if t <= 1:
        return np.ones_like(matrix, dtype=np.float32)
    top_idx = np.argmax(matrix, axis=0)
    top1 = matrix[top_idx, np.arange(d)]
    temp = matrix.copy()
    temp[top_idx, np.arange(d)] = -np.inf
    top2 = np.max(temp, axis=0)
    other = np.broadcast_to(top1[None, :], matrix.shape).copy()
    other[top_idx, np.arange(d)] = top2
    return (matrix - other).astype(np.float32, copy=False)


def _vectorized_cues(tracks, boxes: np.ndarray, feats: np.ndarray):
    n_t = len(tracks)
    n_d = len(boxes)
    if n_t == 0 or n_d == 0:
        empty = np.zeros((n_t, n_d), dtype=np.float32)
        return {name: empty.copy() for name in ('app', 'motion', 'iou', 'shape')}

    tboxes = np.stack([np.asarray(t.tlbr, dtype=np.float32) for t in tracks], axis=0)
    dboxes = np.asarray(boxes, dtype=np.float32)

    # Appearance: one BLAS matrix multiplication for the whole frame.
    det_feats = np.asarray(feats, dtype=np.float32)
    det_feats /= np.maximum(np.linalg.norm(det_feats, axis=1, keepdims=True), 1e-12)
    track_feats = np.zeros((n_t, det_feats.shape[1]), dtype=np.float32)
    valid_track_feat = np.zeros((n_t,), dtype=bool)
    for i, track in enumerate(tracks):
        feat = getattr(track, 'smooth_feat', None)
        if feat is not None:
            feat = np.asarray(feat, dtype=np.float32)
            track_feats[i] = feat / max(float(np.linalg.norm(feat)), 1e-12)
            valid_track_feat[i] = True
    app = track_feats @ det_feats.T
    app[~valid_track_feat, :] = 0.0

    tw = np.maximum(tboxes[:, 2] - tboxes[:, 0], 1e-6)
    th = np.maximum(tboxes[:, 3] - tboxes[:, 1], 1e-6)
    dw = np.maximum(dboxes[:, 2] - dboxes[:, 0], 1e-6)
    dh = np.maximum(dboxes[:, 3] - dboxes[:, 1], 1e-6)
    tcx = (tboxes[:, 0] + tboxes[:, 2]) * 0.5
    tcy = (tboxes[:, 1] + tboxes[:, 3]) * 0.5
    dcx = (dboxes[:, 0] + dboxes[:, 2]) * 0.5
    dcy = (dboxes[:, 1] + dboxes[:, 3]) * 0.5

    scale = np.maximum(np.sqrt(tw * th), 1.0)
    dist = np.sqrt((tcx[:, None] - dcx[None, :]) ** 2 + (tcy[:, None] - dcy[None, :]) ** 2)
    motion = np.exp(-dist / scale[:, None]).astype(np.float32)

    xx1 = np.maximum(tboxes[:, None, 0], dboxes[None, :, 0])
    yy1 = np.maximum(tboxes[:, None, 1], dboxes[None, :, 1])
    xx2 = np.minimum(tboxes[:, None, 2], dboxes[None, :, 2])
    yy2 = np.minimum(tboxes[:, None, 3], dboxes[None, :, 3])
    inter = np.maximum(xx2 - xx1, 0.0) * np.maximum(yy2 - yy1, 0.0)
    tarea = tw * th
    darea = dw * dh
    iou = inter / np.maximum(tarea[:, None] + darea[None, :] - inter, 1e-9)
    iou = iou.astype(np.float32)

    taspect = tw / th
    daspect = dw / dh
    shape_penalty = np.abs(np.log(dh[None, :] / th[:, None])) + np.abs(
        np.log(daspect[None, :] / taspect[:, None])
    )
    shape = np.exp(-shape_penalty).astype(np.float32)

    return {'app': app, 'motion': motion, 'iou': iou, 'shape': shape}


def compute_soft_map_fast(tracker, boxes, scores, feats, det_ids, cfg):
    tracks = [
        t for t in getattr(tracker, 'tracked_stracks', [])
        if bool(getattr(t, 'is_activated', True))
    ]
    n_t = len(tracks)
    soft._TRUST_PAIR_META = MetaLookup({}, {}, {})
    if not cfg.enable or n_t == 0 or len(boxes) == 0:
        empty = PairLookup({}, {}, np.zeros((0, 0), dtype=np.float32), default=None)
        return empty, empty

    keep = np.where(np.asarray(scores, dtype=np.float32) >= float(cfg.min_det_score))[0]
    if keep.size == 0:
        empty = PairLookup({}, {}, np.zeros((0, 0), dtype=np.float32), default=None)
        return empty, empty

    boxes_f = np.asarray(boxes[keep], dtype=np.float32)
    feats_f = np.asarray(feats[keep], dtype=np.float32)
    det_ids_f = np.asarray(det_ids[keep], dtype=np.int64)
    cues = _vectorized_cues(tracks, boxes_f, feats_f)

    row = {name: _row_margins(mat) for name, mat in cues.items()}
    col = {name: _col_margins(mat) for name, mat in cues.items()}
    pair = {name: np.minimum(row[name], col[name]) for name in cues}
    trust = np.maximum.reduce([pair[name] for name in ('app', 'motion', 'iou', 'shape')])
    collapse = 1.0 - trust

    track_age = np.asarray([int(getattr(t, 'tracklet_len', 0)) for t in tracks], dtype=np.int32)
    age_ok = track_age[:, None] >= int(cfg.min_track_age)
    risk = np.zeros_like(trust, dtype=np.float32)
    reason_code = np.zeros_like(trust, dtype=np.uint8)

    if float(cfg.soft_start) < 999.0:
        mask = collapse >= float(cfg.soft_start)
        denom = max(1e-6, float(cfg.soft_extreme) - float(cfg.soft_start))
        collapse_risk = np.clip((collapse - float(cfg.soft_start)) / denom, 0.0, 1.0)
        risk = np.maximum(risk, np.where(mask, collapse_risk, 0.0))
        reason_code |= mask.astype(np.uint8) * 1

    if float(cfg.app_col_thresh) > -998.0:
        mask = col['app'] <= float(cfg.app_col_thresh)
        denom = max(1e-6, float(cfg.app_col_thresh) - float(cfg.app_col_extreme))
        app_risk = np.clip((float(cfg.app_col_thresh) - col['app']) / denom, 0.0, 1.0)
        risk = np.maximum(risk, np.where(mask, app_risk, 0.0))
        reason_code |= mask.astype(np.uint8) * 2

    if float(cfg.motion_col_thresh) > -998.0:
        mask = col['motion'] <= float(cfg.motion_col_thresh)
        denom = max(1e-6, float(cfg.motion_col_thresh) - float(cfg.motion_col_extreme))
        motion_risk = np.clip((float(cfg.motion_col_thresh) - col['motion']) / denom, 0.0, 1.0)
        risk = np.maximum(risk, np.where(mask, motion_risk, 0.0))
        reason_code |= mask.astype(np.uint8) * 4

    valid = age_ok & (reason_code != 0)
    alpha = float(cfg.soft_alpha) + (float(cfg.extreme_alpha) - float(cfg.soft_alpha)) * risk
    alpha = np.clip(alpha, 0.0, 0.9999).astype(np.float32)

    track_index = {int(getattr(t, 'track_id', -1)): i for i, t in enumerate(tracks)}
    det_index = {int(gid): i for i, gid in enumerate(det_ids_f)}

    reason_text = np.empty(reason_code.shape, dtype=object)
    reason_text[:] = ''
    reason_text[reason_code == 1] = 'collapse'
    reason_text[reason_code == 2] = 'app_col'
    reason_text[reason_code == 3] = 'collapse|app_col'
    reason_text[reason_code == 4] = 'motion_col'
    reason_text[reason_code == 5] = 'collapse|motion_col'
    reason_text[reason_code == 6] = 'app_col|motion_col'
    reason_text[reason_code == 7] = 'collapse|app_col|motion_col'

    meta_arrays = {
        'trust': trust,
        'collapse': collapse,
        'track_age': track_age,
        'app_row_margin': row['app'],
        'app_col_margin': col['app'],
        'motion_row_margin': row['motion'],
        'motion_col_margin': col['motion'],
        'iou_row_margin': row['iou'],
        'iou_col_margin': col['iou'],
        'shape_row_margin': row['shape'],
        'shape_col_margin': col['shape'],
        'app_pair_margin': pair['app'],
        'motion_pair_margin': pair['motion'],
        'iou_pair_margin': pair['iou'],
        'shape_pair_margin': pair['shape'],
    }
    soft._TRUST_PAIR_META = MetaLookup(track_index, det_index, meta_arrays)
    alpha_lookup = PairLookup(track_index, det_index, alpha, valid=valid, default=None)
    reason_lookup = PairLookup(track_index, det_index, reason_text, valid=valid, default='')

    soft._TRUST_STATS['candidate_pairs'] += int(n_t * len(boxes_f))
    soft._TRUST_STATS['soft_pairs_predicted'] += int(np.count_nonzero(valid))
    return reason_lookup, alpha_lookup


# Patch the algorithm's expensive pre-association risk calculation only.
soft.compute_soft_map = compute_soft_map_fast

# Optional mmap cache loader.
_ORIGINAL_LOAD_DUMP = soft.spda.base.load_dump


def _load_cached_or_original(path: Path):
    cache_value = os.environ.get('TRUST_DUMP_CACHE_DIR', '').strip()
    if not cache_value:
        return _ORIGINAL_LOAD_DUMP(path)
    cache_dir = Path(cache_value)
    if not (cache_dir / 'manifest.json').exists():
        raise FileNotFoundError(f'missing cache manifest: {cache_dir / "manifest.json"}')
    result = {}
    for key in ('detections', 'features', 'frame_offsets', 'columns', 'image_files', 'image_wh'):
        array_path = cache_dir / f'{key}.npy'
        if not array_path.exists():
            raise FileNotFoundError(f'missing cached array: {array_path}')
        result[key] = np.load(array_path, mmap_mode='r', allow_pickle=True)
    return result


soft.spda.base.load_dump = _load_cached_or_original


if __name__ == '__main__':
    soft.main()
