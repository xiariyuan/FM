from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ['seq', 'canonical_rank', 'u', 'v', 'boundary_frame', 'transaction_type']
HISTORY_SIZES = [5, 20]
HORIZONS: list[int | None] = [30, 60, 120, None]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_mapping(
    values: list[str], label: str, expected_sequences: list[str]
) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values:
        if '=' not in value:
            raise RuntimeError(f'{label} entries must be SEQ=PATH: {value}')
        seq, raw_path = value.split('=', 1)
        if seq in mapping:
            raise RuntimeError(f'duplicate {label} sequence: {seq}')
        mapping[seq] = Path(raw_path)
    missing = sorted(set(expected_sequences) - set(mapping))
    if missing:
        raise RuntimeError(f'{label} missing sequences: {missing}')
    extra = sorted(set(mapping) - set(expected_sequences))
    if extra:
        raise RuntimeError(f'{label} has unexpected sequences: {extra}')
    return mapping


def fit_state(group: pd.DataFrame, size: int) -> dict[str, float | int]:
    window = group.tail(size).copy()
    if window.empty:
        raise RuntimeError('motion history is empty')
    time = window.frame.to_numpy(float)
    last_frame = float(time[-1])
    dt = time - last_frame
    cx = (window.x + window.w / 2).to_numpy(float)
    cy = (window.y + window.h / 2).to_numpy(float)
    bottom = (window.y + window.h).to_numpy(float)
    log_w = np.log(window.w.clip(lower=1).to_numpy(float))
    log_h = np.log(window.h.clip(lower=1).to_numpy(float))

    def fit(values: np.ndarray) -> tuple[float, float]:
        if len(values) < 2 or np.ptp(dt) == 0:
            return float(values[-1]), 0.0
        slope, intercept = np.polyfit(dt, values, 1)
        return float(intercept), float(slope)

    result: dict[str, float | int] = {}
    for name, values in [('cx', cx), ('cy', cy), ('bottom', bottom), ('lw', log_w), ('lh', log_h)]:
        result[f'{name}_0'], result[f'{name}_v'] = fit(values)
    result['last_frame'] = int(last_frame)
    result['rows'] = int(len(window))
    return result


def predict_state(
    state: dict[str, float | int], frames: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = frames - float(state['last_frame'])
    cx = float(state['cx_0']) + float(state['cx_v']) * dt
    cy = float(state['cy_0']) + float(state['cy_v']) * dt
    bottom = float(state['bottom_0']) + float(state['bottom_v']) * dt
    width = np.exp(np.clip(float(state['lw_0']) + float(state['lw_v']) * dt, -5, 10))
    height = np.exp(np.clip(float(state['lh_0']) + float(state['lh_v']) * dt, -5, 10))
    return cx, cy, bottom, width, height


def add_stats(prefix: str, values: np.ndarray, record: dict[str, object]) -> None:
    values = np.asarray(values, float)
    functions = [
        ('mean', np.mean),
        ('q25', lambda x: np.quantile(x, 0.25)),
        ('median', np.median),
        ('min', np.min),
        ('max', np.max),
        ('std', np.std),
    ]
    for name, function in functions:
        record[f'{prefix}_{name}'] = float(function(values)) if len(values) else np.nan


def load_tracks(path: Path) -> pd.DataFrame:
    columns = ['frame', 'track_id', 'x', 'y', 'w', 'h', 'score', 'a', 'b', 'c']
    frame = pd.read_csv(path, header=None, names=columns)
    frame['frame'] = frame.frame.astype(int)
    frame['track_id'] = frame.track_id.astype(int)
    if frame.duplicated(['frame', 'track_id']).any():
        raise RuntimeError(f'duplicate frame-track rows: {path}')
    return frame


def build_features(
    executability: pd.DataFrame,
    changed: pd.DataFrame,
    track_paths: dict[str, Path],
) -> pd.DataFrame:
    outputs: list[dict[str, object]] = []
    sequences = sorted(executability.seq.astype(str).unique().tolist())
    if set(track_paths) != set(sequences):
        raise RuntimeError('track-result mappings do not match executability sequences')
    for seq in sequences:
        tracks = load_tracks(track_paths[seq])
        by_track = {
            int(track_id): group.sort_values('frame')
            for track_id, group in tracks.groupby('track_id', sort=False)
        }
        seq_events = executability[executability.seq == seq]
        changed_groups = {
            key: group
            for key, group in changed[changed.seq == seq].groupby(KEYS, sort=False)
        }
        for _, event in seq_events.iterrows():
            key = tuple(event[column] for column in KEYS)
            if key not in changed_groups:
                raise RuntimeError(f'missing changed rows: {key}')
            changed_rows = changed_groups[key].copy()
            source = int(changed_rows.edited_label.iloc[0])
            if changed_rows.edited_label.astype(int).nunique() != 1:
                raise RuntimeError(f'actual source label is not unique: {key}')
            receivers = sorted(changed_rows.baseline_label.astype(int).unique())
            start = int(event.effective_start_frame)
            future = changed_rows.merge(
                tracks[['frame', 'track_id', 'x', 'y', 'w', 'h']],
                left_on=['frame', 'baseline_label'],
                right_on=['frame', 'track_id'],
                how='left',
                validate='one_to_one',
            )
            if future[['x', 'y', 'w', 'h']].isna().any().any():
                raise RuntimeError(f'missing future geometry: {key}')
            if source not in by_track:
                raise RuntimeError(f'missing actual source track {source}: {key}')
            missing_receivers = [track_id for track_id in receivers if track_id not in by_track]
            if missing_receivers:
                raise RuntimeError(f'missing receiver tracks {missing_receivers}: {key}')

            nominal_source = int(event.u if event.transaction_type == 'u_to_v' else event.v)
            nominal_receiver = int(event.v if event.transaction_type == 'u_to_v' else event.u)
            record: dict[str, object] = {column: event[column] for column in KEYS}
            record['actual_source_label'] = source
            record['receiver_family_size'] = len(receivers)
            record['aggregate_anchor_used'] = int(
                source != nominal_source or set(receivers) != {nominal_receiver}
            )

            frames = future.frame.to_numpy(float)
            actual_cx = (future.x + future.w / 2).to_numpy(float)
            actual_cy = (future.y + future.h / 2).to_numpy(float)
            actual_bottom = (future.y + future.h).to_numpy(float)
            actual_w = future.w.to_numpy(float)
            actual_h = future.h.to_numpy(float)
            scale = np.sqrt(np.maximum(actual_w * actual_h, 1.0))

            for history_size in HISTORY_SIZES:
                source_history = by_track[source][by_track[source].frame < start]
                if source_history.empty:
                    raise RuntimeError(f'empty source history: {key}')
                source_state = fit_state(source_history, history_size)
                record[f'source_gap_{history_size}'] = start - int(source_state['last_frame'])
                record[f'source_speed_{history_size}'] = float(
                    np.hypot(float(source_state['cx_v']), float(source_state['cy_v']))
                )
                receiver_states: dict[int, dict[str, float | int]] = {}
                for receiver in receivers:
                    receiver_history = by_track[receiver][by_track[receiver].frame < start]
                    if receiver_history.empty:
                        raise RuntimeError(f'empty receiver history {receiver}: {key}')
                    receiver_states[receiver] = fit_state(receiver_history, history_size)
                record[f'receiver_gap_mean_{history_size}'] = float(
                    np.mean([start - int(state['last_frame']) for state in receiver_states.values()])
                )
                record[f'receiver_speed_mean_{history_size}'] = float(
                    np.mean([
                        np.hypot(float(state['cx_v']), float(state['cy_v']))
                        for state in receiver_states.values()
                    ])
                )

                source_cx, source_cy, source_bottom_pred, source_w, source_h = predict_state(
                    source_state, frames
                )
                source_center = np.hypot(actual_cx - source_cx, actual_cy - source_cy) / scale
                source_bottom = np.abs(actual_bottom - source_bottom_pred) / np.maximum(actual_h, 1)
                source_size = (
                    np.abs(np.log(np.maximum(actual_w, 1) / np.maximum(source_w, 1)))
                    + np.abs(np.log(np.maximum(actual_h, 1) / np.maximum(source_h, 1)))
                )

                receiver_center = np.zeros(len(future))
                receiver_bottom = np.zeros(len(future))
                receiver_size = np.zeros(len(future))
                receiver_labels = future.baseline_label.to_numpy(int)
                for receiver, receiver_state in receiver_states.items():
                    indices = np.flatnonzero(receiver_labels == receiver)
                    pred_cx, pred_cy, pred_bottom, pred_w, pred_h = predict_state(
                        receiver_state, frames[indices]
                    )
                    receiver_center[indices] = (
                        np.hypot(actual_cx[indices] - pred_cx, actual_cy[indices] - pred_cy)
                        / scale[indices]
                    )
                    receiver_bottom[indices] = (
                        np.abs(actual_bottom[indices] - pred_bottom) / np.maximum(actual_h[indices], 1)
                    )
                    receiver_size[indices] = (
                        np.abs(np.log(np.maximum(actual_w[indices], 1) / np.maximum(pred_w, 1)))
                        + np.abs(np.log(np.maximum(actual_h[indices], 1) / np.maximum(pred_h, 1)))
                    )

                source_total = source_center + 0.25 * source_bottom + 0.10 * source_size
                receiver_total = receiver_center + 0.25 * receiver_bottom + 0.10 * receiver_size
                motion_margin = receiver_total - source_total
                for horizon in HORIZONS:
                    suffix = 'full' if horizon is None else f'h{horizon}'
                    mask = (
                        np.ones(len(future), bool)
                        if horizon is None
                        else future.frames_after_handoff.to_numpy() <= horizon
                    )
                    for name, values in [
                        ('source_center_resid', source_center),
                        ('receiver_center_resid', receiver_center),
                        ('source_bottom_resid', source_bottom),
                        ('receiver_bottom_resid', receiver_bottom),
                        ('source_size_resid', source_size),
                        ('receiver_size_resid', receiver_size),
                        ('motion_margin', motion_margin),
                    ]:
                        add_stats(f'{name}_{history_size}_{suffix}', values[mask], record)
                    record[f'motion_source_win_fraction_{history_size}_{suffix}'] = float(
                        np.mean(motion_margin[mask] > 0)
                    )

                for horizon in [60, None]:
                    suffix = 'full' if horizon is None else f'h{horizon}'
                    mask = (
                        np.ones(len(future), bool)
                        if horizon is None
                        else future.frames_after_handoff.to_numpy() <= horizon
                    )
                    future_frames = frames[mask]
                    future_cx = actual_cx[mask]
                    future_cy = actual_cy[mask]
                    if len(future_frames) >= 2 and np.ptp(future_frames) > 0:
                        vx = np.polyfit(future_frames - future_frames[0], future_cx, 1)[0]
                        vy = np.polyfit(future_frames - future_frames[0], future_cy, 1)[0]
                        record[f'future_source_velocity_gap_{history_size}_{suffix}'] = float(
                            np.hypot(vx - float(source_state['cx_v']), vy - float(source_state['cy_v']))
                        )
                        receiver_velocity = np.mean(
                            [[float(state['cx_v']), float(state['cy_v'])] for state in receiver_states.values()],
                            axis=0,
                        )
                        record[f'future_receiver_velocity_gap_{history_size}_{suffix}'] = float(
                            np.hypot(vx - receiver_velocity[0], vy - receiver_velocity[1])
                        )
                        record[f'velocity_margin_{history_size}_{suffix}'] = (
                            float(record[f'future_receiver_velocity_gap_{history_size}_{suffix}'])
                            - float(record[f'future_source_velocity_gap_{history_size}_{suffix}'])
                        )
                    else:
                        record[f'future_source_velocity_gap_{history_size}_{suffix}'] = np.nan
                        record[f'future_receiver_velocity_gap_{history_size}_{suffix}'] = np.nan
                        record[f'velocity_margin_{history_size}_{suffix}'] = np.nan
            outputs.append(record)
    return pd.DataFrame(outputs).sort_values(KEYS).reset_index(drop=True)


def compact_features() -> list[str]:
    features = ['source_gap_20', 'receiver_gap_mean_20', 'source_speed_20', 'receiver_speed_mean_20']
    for horizon in ['h30', 'h60', 'h120', 'full']:
        features.extend([
            f'source_center_resid_20_{horizon}_mean',
            f'receiver_center_resid_20_{horizon}_mean',
            f'motion_margin_20_{horizon}_mean',
            f'motion_margin_20_{horizon}_q25',
            f'motion_margin_20_{horizon}_min',
            f'motion_source_win_fraction_20_{horizon}',
        ])
    features.extend([
        'future_source_velocity_gap_20_h60',
        'future_receiver_velocity_gap_20_h60',
        'velocity_margin_20_h60',
        'future_source_velocity_gap_20_full',
        'future_receiver_velocity_gap_20_full',
        'velocity_margin_20_full',
        'aggregate_anchor_used',
        'receiver_family_size',
    ])
    return list(dict.fromkeys(features))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--executability', required=True)
    parser.add_argument('--changed-rows', required=True)
    parser.add_argument('--track-result', action='append', required=True, help='SEQ=PATH')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    executability_path = Path(args.executability)
    changed_path = Path(args.changed_rows)
    executability = pd.read_csv(executability_path)
    executability = executability[executability.accepted == 1].copy()
    sequences = sorted(executability.seq.astype(str).unique().tolist())
    track_paths = parse_mapping(args.track_result, 'track-result', sequences)
    changed = pd.read_csv(
        changed_path,
        usecols=KEYS + ['frame', 'frames_after_handoff', 'baseline_label', 'edited_label'],
    )
    features = build_features(executability, changed, track_paths)
    if len(features) != len(executability):
        raise RuntimeError(f'feature/event row mismatch: {len(features)} != {len(executability)}')
    if features.duplicated(KEYS).any():
        raise RuntimeError('duplicate event keys in motion features')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    feature_path = out_dir / 'actual_anchor_motion_features.csv'
    compact_path = out_dir / 'compact_feature_list.csv'
    features.to_csv(feature_path, index=False)
    pd.DataFrame({'feature': compact_features()}).to_csv(compact_path, index=False)
    report = {
        'dataset': {
            'events': int(len(features)),
            'sequences': sorted(features.seq.unique().tolist()),
            'feature_columns': int(len(features.columns) - len(KEYS)),
            'aggregate_anchor_events': int(features.aggregate_anchor_used.sum()),
            'maximum_missing_fraction': float(features.isna().mean().max()),
        },
        'protocol': {
            'scope': 'Train-only actual-anchor future motion transfer features.',
            'actual_source': 'Unique edited_label from the executor geometry diff.',
            'receiver_family': 'Per-row baseline_label from the executor geometry diff.',
            'history_sizes': HISTORY_SIZES,
            'future_horizons': ['h30', 'h60', 'h120', 'full'],
            'motion_error': 'center residual + 0.25*bottom residual + 0.10*log-size residual.',
            'forbidden_inputs': [
                'matched_gt_id', 'matched_gt_iou', 'dominant_gt', 'row_class',
                'utility labels', 'TrackEval metrics', 'locked artifacts',
            ],
            'locked_labels_read': 0,
            'locked_trackeval_calls': 0,
        },
    }
    report_path = out_dir / 'report.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    manifest = {
        'actual_anchor_motion_features.csv': sha256_file(feature_path),
        'compact_feature_list.csv': sha256_file(compact_path),
        'report.json': sha256_file(report_path),
    }
    (out_dir / 'prediction_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
