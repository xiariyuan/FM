from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ['seq', 'canonical_rank', 'u', 'v', 'boundary_frame', 'transaction_type']
WINDOWS = [5, 20]
HORIZONS = [5, 20]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_mapping(values: list[str], expected: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if '=' not in value:
            raise RuntimeError(f'track-result must be SEQ=PATH: {value}')
        sequence, raw_path = value.split('=', 1)
        if sequence in result:
            raise RuntimeError(f'duplicate track-result sequence: {sequence}')
        result[sequence] = Path(raw_path)
    missing = sorted(set(expected) - set(result))
    extra = sorted(set(result) - set(expected))
    if missing or extra:
        raise RuntimeError(f'track-result mismatch: missing={missing}, extra={extra}')
    return result


def rounded(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=[np.number]).columns:
        result[column] = result[column].round(12)
    return result


def load_tracks(path: Path) -> dict[int, dict[str, np.ndarray]]:
    columns = ['frame', 'track_id', 'x', 'y', 'w', 'h', 'score', 'a', 'b', 'c']
    raw = pd.read_csv(path, header=None, names=columns)
    raw['frame'] = raw.frame.astype(int)
    raw['track_id'] = raw.track_id.astype(int)
    if raw.duplicated(['frame', 'track_id']).any():
        raise RuntimeError(f'duplicate frame-track row: {path}')
    output: dict[int, dict[str, np.ndarray]] = {}
    for track_id, group in raw.groupby('track_id', sort=False):
        group = group.sort_values('frame')
        width = group.w.to_numpy(float)
        height = group.h.to_numpy(float)
        x = group.x.to_numpy(float)
        y = group.y.to_numpy(float)
        output[int(track_id)] = {
            'frame': group.frame.to_numpy(int),
            'cx': x + width / 2.0,
            'cy': y + height / 2.0,
            'bottom': y + height,
            'w': width,
            'h': height,
            'lw': np.log(np.maximum(width, 1.0)),
            'lh': np.log(np.maximum(height, 1.0)),
            'scale': np.sqrt(np.maximum(width * height, 1.0)),
            'score': group.score.to_numpy(float),
        }
    return output


def take(track: dict[str, np.ndarray], start: int, end: int | None = None) -> dict[str, np.ndarray]:
    frames = track['frame']
    left = int(np.searchsorted(frames, start, side='left'))
    right = len(frames) if end is None else int(np.searchsorted(frames, end, side='left'))
    return {key: value[left:right] for key, value in track.items()}


def before(track: dict[str, np.ndarray], start: int) -> dict[str, np.ndarray]:
    right = int(np.searchsorted(track['frame'], start, side='left'))
    return {key: value[:right] for key, value in track.items()}


def head(track: dict[str, np.ndarray], size: int) -> dict[str, np.ndarray]:
    return {key: value[:size] for key, value in track.items()}


def tail(track: dict[str, np.ndarray], size: int) -> dict[str, np.ndarray]:
    return {key: value[-size:] for key, value in track.items()}


def fit_state(track: dict[str, np.ndarray]) -> dict[str, float] | None:
    if len(track['frame']) < 2 or np.ptp(track['frame']) == 0:
        return None
    frames = track['frame'].astype(float)
    reference = float(frames[-1])
    dt = frames - reference
    state: dict[str, float] = {'reference': reference}
    for name in ['cx', 'cy', 'bottom', 'lw', 'lh']:
        slope, intercept = np.polyfit(dt, track[name].astype(float), 1)
        state[f'{name}_0'] = float(intercept)
        state[f'{name}_v'] = float(slope)
    return state


def state_error(state: dict[str, float] | None, target: dict[str, np.ndarray]) -> np.ndarray:
    if state is None or len(target['frame']) == 0:
        return np.array([], dtype=float)
    dt = target['frame'].astype(float) - state['reference']
    cx = state['cx_0'] + state['cx_v'] * dt
    cy = state['cy_0'] + state['cy_v'] * dt
    bottom = state['bottom_0'] + state['bottom_v'] * dt
    lw = state['lw_0'] + state['lw_v'] * dt
    lh = state['lh_0'] + state['lh_v'] * dt
    scale = np.maximum(target['scale'], 1.0)
    center = np.hypot(cx - target['cx'], cy - target['cy']) / scale
    bottom_error = np.abs(bottom - target['bottom']) / scale
    size = np.abs(lw - target['lw']) + np.abs(lh - target['lh'])
    return center + 0.25 * bottom_error + 0.10 * size


def error_summary(values: np.ndarray) -> tuple[float, float, float]:
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(values)), float(np.quantile(values, 0.75)), float(np.max(values))


def velocity_features(
    pre_state: dict[str, float] | None,
    post_state: dict[str, float] | None,
    scale: float,
) -> dict[str, float]:
    if pre_state is None or post_state is None:
        return {
            'pre_speed_norm': np.nan,
            'post_speed_norm': np.nan,
            'velocity_delta_norm': np.nan,
            'velocity_direction_change': np.nan,
            'log_size_velocity_delta': np.nan,
        }
    pre = np.array([pre_state['cx_v'], pre_state['cy_v']], dtype=float)
    post = np.array([post_state['cx_v'], post_state['cy_v']], dtype=float)
    pre_norm = float(np.linalg.norm(pre))
    post_norm = float(np.linalg.norm(post))
    denominator = max(pre_norm * post_norm, 1e-8)
    cosine = float(np.clip(np.dot(pre, post) / denominator, -1.0, 1.0))
    return {
        'pre_speed_norm': pre_norm / max(scale, 1.0),
        'post_speed_norm': post_norm / max(scale, 1.0),
        'velocity_delta_norm': float(np.linalg.norm(post - pre)) / max(scale, 1.0),
        'velocity_direction_change': 1.0 - cosine,
        'log_size_velocity_delta': float(
            np.hypot(
                post_state['lw_v'] - pre_state['lw_v'],
                post_state['lh_v'] - pre_state['lh_v'],
            )
        ),
    }


def box_iou(
    cx_a: np.ndarray,
    cy_a: np.ndarray,
    w_a: np.ndarray,
    h_a: np.ndarray,
    cx_b: np.ndarray,
    cy_b: np.ndarray,
    w_b: np.ndarray,
    h_b: np.ndarray,
) -> np.ndarray:
    left = np.maximum(cx_a - w_a / 2.0, cx_b - w_b / 2.0)
    top = np.maximum(cy_a - h_a / 2.0, cy_b - h_b / 2.0)
    right = np.minimum(cx_a + w_a / 2.0, cx_b + w_b / 2.0)
    bottom = np.minimum(cy_a + h_a / 2.0, cy_b + h_b / 2.0)
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    union = w_a * h_a + w_b * h_b - intersection
    return intersection / np.maximum(union, 1e-8)


def pair_overlap(donor: dict[str, np.ndarray], receiver: dict[str, np.ndarray]) -> dict[str, float]:
    common, donor_indices, receiver_indices = np.intersect1d(
        donor['frame'], receiver['frame'], assume_unique=True, return_indices=True
    )
    if len(common) == 0:
        return {
            'overlap_rows_raw': 0.0,
            'overlap_iou_mean': np.nan,
            'overlap_iou_max': np.nan,
            'overlap_iou_ge_030_fraction': np.nan,
            'overlap_iou_ge_050_fraction': np.nan,
            'overlap_center_distance_mean': np.nan,
            'overlap_center_distance_min': np.nan,
            'overlap_score_absdiff_mean': np.nan,
        }
    iou = box_iou(
        donor['cx'][donor_indices], donor['cy'][donor_indices], donor['w'][donor_indices], donor['h'][donor_indices],
        receiver['cx'][receiver_indices], receiver['cy'][receiver_indices], receiver['w'][receiver_indices], receiver['h'][receiver_indices],
    )
    scale = np.sqrt(np.maximum(donor['scale'][donor_indices] * receiver['scale'][receiver_indices], 1.0))
    distance = np.hypot(
        donor['cx'][donor_indices] - receiver['cx'][receiver_indices],
        donor['cy'][donor_indices] - receiver['cy'][receiver_indices],
    ) / scale
    return {
        'overlap_rows_raw': float(len(common)),
        'overlap_iou_mean': float(np.mean(iou)),
        'overlap_iou_max': float(np.max(iou)),
        'overlap_iou_ge_030_fraction': float(np.mean(iou >= 0.30)),
        'overlap_iou_ge_050_fraction': float(np.mean(iou >= 0.50)),
        'overlap_center_distance_mean': float(np.mean(distance)),
        'overlap_center_distance_min': float(np.min(distance)),
        'overlap_score_absdiff_mean': float(
            np.mean(np.abs(donor['score'][donor_indices] - receiver['score'][receiver_indices]))
        ),
    }


def boundary_pair_features(
    donor_pre: dict[str, np.ndarray], receiver_post: dict[str, np.ndarray]
) -> dict[str, float]:
    if len(donor_pre['frame']) == 0 or len(receiver_post['frame']) == 0:
        return {
            'donor_receiver_boundary_center_norm': np.nan,
            'donor_receiver_boundary_bottom_norm': np.nan,
            'donor_receiver_boundary_log_size': np.nan,
            'donor_receiver_boundary_score_absdiff': np.nan,
        }
    di = -1
    ri = 0
    scale = max(float(np.sqrt(donor_pre['scale'][di] * receiver_post['scale'][ri])), 1.0)
    return {
        'donor_receiver_boundary_center_norm': float(
            np.hypot(
                donor_pre['cx'][di] - receiver_post['cx'][ri],
                donor_pre['cy'][di] - receiver_post['cy'][ri],
            ) / scale
        ),
        'donor_receiver_boundary_bottom_norm': float(
            abs(donor_pre['bottom'][di] - receiver_post['bottom'][ri]) / scale
        ),
        'donor_receiver_boundary_log_size': float(
            abs(donor_pre['lw'][di] - receiver_post['lw'][ri])
            + abs(donor_pre['lh'][di] - receiver_post['lh'][ri])
        ),
        'donor_receiver_boundary_score_absdiff': float(
            abs(donor_pre['score'][di] - receiver_post['score'][ri])
        ),
    }


def build_event_features(event: pd.Series, tracks: dict[int, dict[str, np.ndarray]]) -> dict[str, object]:
    donor_id = int(event.donor_anchor)
    receiver_id = int(event.receiver_anchor)
    start = int(event.effective_start_frame)
    if donor_id not in tracks or receiver_id not in tracks:
        raise RuntimeError(f'missing anchor track: donor={donor_id}, receiver={receiver_id}')
    donor = tracks[donor_id]
    receiver = tracks[receiver_id]
    donor_pre = before(donor, start)
    receiver_pre = before(receiver, start)
    receiver_post = take(receiver, start)
    if len(receiver_post['frame']) != int(event.changed_rows):
        key = tuple(event[column] for column in KEYS)
        raise RuntimeError(
            f'changed-row mismatch {key}: got={len(receiver_post["frame"])}, expected={event.changed_rows}'
        )
    record: dict[str, object] = {column: event[column] for column in KEYS}
    record.update(
        {
            'donor_total_rows_raw': float(len(donor['frame'])),
            'receiver_total_rows_raw': float(len(receiver['frame'])),
            'donor_span_raw': float(donor['frame'][-1] - donor['frame'][0] + 1),
            'receiver_span_raw': float(receiver['frame'][-1] - receiver['frame'][0] + 1),
            'donor_history_rows_raw': float(len(donor_pre['frame'])),
            'receiver_history_rows_raw': float(len(receiver_pre['frame'])),
            'donor_future_rows_raw': float(len(donor['frame']) - len(donor_pre['frame'])),
            'receiver_future_rows_raw': float(len(receiver_post['frame'])),
            'donor_history_fraction_raw': float(len(donor_pre['frame']) / max(len(donor['frame']), 1)),
            'receiver_history_fraction_raw': float(len(receiver_pre['frame']) / max(len(receiver['frame']), 1)),
            'donor_gap_to_start_raw': float(start - donor_pre['frame'][-1]) if len(donor_pre['frame']) else np.nan,
            'receiver_gap_to_start_raw': float(start - receiver_pre['frame'][-1]) if len(receiver_pre['frame']) else np.nan,
            'donor_ephemeral_log_ratio': float(
                np.log1p(len(receiver_post['frame'])) - np.log1p(len(donor['frame']))
            ),
            'history_imbalance_log_raw': float(
                np.log1p(len(donor_pre['frame'])) - np.log1p(len(receiver_pre['frame']))
            ),
            'future_to_donor_rows_ratio': float(
                len(receiver_post['frame']) / max(len(donor['frame']), 1)
            ),
        }
    )
    record.update(pair_overlap(donor, receiver))
    record.update(boundary_pair_features(donor_pre, receiver_post))

    if len(receiver_pre['frame']) and len(receiver_post['frame']):
        scale = max(float(np.sqrt(receiver_pre['scale'][-1] * receiver_post['scale'][0])), 1.0)
        record['receiver_boundary_center_jump_norm'] = float(
            np.hypot(
                receiver_pre['cx'][-1] - receiver_post['cx'][0],
                receiver_pre['cy'][-1] - receiver_post['cy'][0],
            ) / scale
        )
        record['receiver_boundary_bottom_jump_norm'] = float(
            abs(receiver_pre['bottom'][-1] - receiver_post['bottom'][0]) / scale
        )
        record['receiver_boundary_log_size_jump'] = float(
            abs(receiver_pre['lw'][-1] - receiver_post['lw'][0])
            + abs(receiver_pre['lh'][-1] - receiver_post['lh'][0])
        )
        record['receiver_boundary_score_jump'] = float(
            abs(receiver_pre['score'][-1] - receiver_post['score'][0])
        )
    else:
        record['receiver_boundary_center_jump_norm'] = np.nan
        record['receiver_boundary_bottom_jump_norm'] = np.nan
        record['receiver_boundary_log_size_jump'] = np.nan
        record['receiver_boundary_score_jump'] = np.nan

    for window in WINDOWS:
        pre_window = tail(receiver_pre, window)
        post_window = head(receiver_post, window)
        pre_state = fit_state(pre_window)
        post_state = fit_state(post_window)
        median_scale = float(np.median(receiver_post['scale'][: max(1, min(window, len(receiver_post['scale'])))]))
        for name, value in velocity_features(pre_state, post_state, median_scale).items():
            record[f'receiver_w{window}_{name}'] = value

        backtest_count = min(5, max(0, len(receiver_pre['frame']) // 3))
        if len(receiver_pre['frame']) - backtest_count >= 2 and backtest_count >= 1:
            train = {key: value[: len(value) - backtest_count] for key, value in receiver_pre.items()}
            target = {key: value[len(value) - backtest_count :] for key, value in receiver_pre.items()}
            backtest = state_error(fit_state(tail(train, window)), target)
            backtest_mean = float(np.mean(backtest)) if len(backtest) else np.nan
        else:
            backtest_mean = np.nan
        record[f'receiver_w{window}_pre_backtest_error_mean'] = backtest_mean

        donor_state = fit_state(tail(donor_pre, window))
        for horizon in HORIZONS:
            target = head(receiver_post, horizon)
            self_errors = state_error(pre_state, target)
            donor_errors = state_error(donor_state, target)
            self_mean, self_q75, self_max = error_summary(self_errors)
            donor_mean, donor_q75, donor_max = error_summary(donor_errors)
            prefix = f'w{window}_h{horizon}'
            record[f'receiver_self_error_{prefix}_mean'] = self_mean
            record[f'receiver_self_error_{prefix}_q75'] = self_q75
            record[f'receiver_self_error_{prefix}_max'] = self_max
            record[f'donor_transfer_error_{prefix}_mean'] = donor_mean
            record[f'donor_transfer_error_{prefix}_q75'] = donor_q75
            record[f'donor_transfer_error_{prefix}_max'] = donor_max
            record[f'receiver_change_ratio_{prefix}'] = float(
                self_mean / max(backtest_mean, 0.05)
            ) if np.isfinite(self_mean) and np.isfinite(backtest_mean) else np.nan
            record[f'donor_minus_receiver_error_{prefix}'] = (
                donor_mean - self_mean
                if np.isfinite(donor_mean) and np.isfinite(self_mean)
                else np.nan
            )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--executability', required=True)
    parser.add_argument('--track-result', action='append', required=True, help='SEQ=PATH')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    events = pd.read_csv(args.executability)
    missing = [
        column
        for column in KEYS + ['donor_anchor', 'receiver_anchor', 'effective_start_frame', 'changed_rows']
        if column not in events.columns
    ]
    if missing:
        raise RuntimeError(f'executability missing columns: {missing}')
    if events.duplicated(KEYS).any():
        raise RuntimeError('duplicate event keys')
    sequences = sorted(events.seq.astype(str).unique().tolist())
    track_paths = parse_mapping(args.track_result, sequences)
    rows: list[dict[str, object]] = []
    for sequence in sequences:
        tracks = load_tracks(track_paths[sequence])
        for _, event in events[events.seq == sequence].iterrows():
            rows.append(build_event_features(event, tracks))
    features = pd.DataFrame(rows).sort_values(KEYS).reset_index(drop=True)
    if len(features) != len(events) or features.duplicated(KEYS).any():
        raise RuntimeError('feature output key integrity failure')
    feature_names = [column for column in features.columns if column not in KEYS]
    forbidden_tokens = ['gt', 'idtp', 'utility', 'trackeval', 'matched']
    forbidden = [
        column for column in feature_names if any(token in column.lower() for token in forbidden_tokens)
    ]
    if forbidden:
        raise RuntimeError(f'forbidden feature names: {forbidden}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    rounded(features).to_csv(out_dir / 'receiver_change_mechanism_features.csv', index=False)
    pd.DataFrame({'feature': feature_names}).to_csv(out_dir / 'feature_list.csv', index=False)
    report = {
        'protocol': {
            'scope': 'Deployable receiver change-point, ephemeral-anchor, and pair-overlap features.',
            'inputs': 'Frozen executable event metadata and raw tracker rows only.',
            'windows': WINDOWS,
            'horizons': HORIZONS,
            'ground_truth_inputs': 0,
            'utility_inputs': 0,
            'trackeval_calls': 0,
            'p15_locked_labels_read': 0,
            'p15_locked_trackeval_calls': 0,
            'p15_remaining_locked_rows_untouched': 156,
        },
        'dataset': {
            'events': int(len(features)),
            'sequences': sequences,
            'features': int(len(feature_names)),
            'duplicate_keys': int(features.duplicated(KEYS).sum()),
            'maximum_missing_fraction': float(features[feature_names].isna().mean().max()),
            'forbidden_features': forbidden,
        },
    }
    (out_dir / 'report.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    manifest = {
        path.name: sha256(path)
        for path in sorted(out_dir.iterdir())
        if path.is_file() and path.name != 'prediction_manifest.json'
    }
    (out_dir / 'prediction_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
