from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import pandas as pd
from numpy.lib import format as npformat

KEYS = ['seq', 'canonical_rank', 'u', 'v', 'boundary_frame', 'transaction_type']
ALLOWED_CHANGED_COLUMNS = KEYS + [
    'frame', 'frames_after_handoff', 'baseline_label', 'edited_label'
]
TRACK_COLUMNS = ['frame', 'track_id', 'x', 'y', 'w', 'h', 'score', 'unused1', 'unused2', 'unused3']
HISTORY_SIZES = [5, 20, 60]
HORIZONS: list[int | None] = [30, 60, 120, 300, None]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rounded(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=[np.number]).columns:
        result[column] = result[column].round(12)
    return result


def mmap_npz_member(npz_path: Path, member: str) -> np.memmap:
    """Memory-map an uncompressed .npy member inside a ZIP_STORED .npz file."""
    with ZipFile(npz_path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != ZIP_STORED:
            raise RuntimeError(f'{npz_path}:{member} is compressed; zero-copy mmap is unsafe')
        local_header_offset = int(info.header_offset)

    with npz_path.open('rb') as handle:
        handle.seek(local_header_offset)
        raw = handle.read(30)
        if len(raw) != 30:
            raise RuntimeError(f'cannot read local ZIP header: {npz_path}:{member}')
        fields = struct.unpack('<IHHHHHIIIHH', raw)
        signature = fields[0]
        name_length = fields[-2]
        extra_length = fields[-1]
        if signature != 0x04034B50:
            raise RuntimeError(f'invalid local ZIP signature: {npz_path}:{member}')
        member_start = local_header_offset + 30 + name_length + extra_length
        handle.seek(member_start)
        version = npformat.read_magic(handle)
        shape, fortran_order, dtype = npformat._read_array_header(handle, version)
        data_offset = handle.tell()

    return np.memmap(
        npz_path,
        dtype=dtype,
        mode='r',
        offset=data_offset,
        shape=shape,
        order='F' if fortran_order else 'C',
    )


def l2_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def l2_vector(values: np.ndarray) -> np.ndarray | None:
    if len(values) == 0:
        return None
    mean = l2_rows(values).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-12:
        return None
    return mean / norm


def cosine(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return float('nan')
    return float(np.dot(left, right))


def pair_iou(track_boxes: np.ndarray, detection_boxes: np.ndarray) -> np.ndarray:
    if len(track_boxes) == 0 or len(detection_boxes) == 0:
        return np.zeros((len(track_boxes), len(detection_boxes)), dtype=np.float32)
    left = np.maximum(track_boxes[:, None, :2], detection_boxes[None, :, :2])
    right = np.minimum(track_boxes[:, None, 2:], detection_boxes[None, :, 2:])
    wh = np.maximum(0.0, right - left)
    intersection = wh[..., 0] * wh[..., 1]
    track_area = np.maximum(0.0, track_boxes[:, 2] - track_boxes[:, 0]) * np.maximum(
        0.0, track_boxes[:, 3] - track_boxes[:, 1]
    )
    detection_area = np.maximum(0.0, detection_boxes[:, 2] - detection_boxes[:, 0]) * np.maximum(
        0.0, detection_boxes[:, 3] - detection_boxes[:, 1]
    )
    union = track_area[:, None] + detection_area[None, :] - intersection
    return intersection / np.maximum(union, 1e-12)


def load_events(executability_path: str) -> pd.DataFrame:
    frame = pd.read_csv(executability_path)
    required = KEYS + ['accepted', 'changed_rows', 'effective_start_frame']
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f'executability missing columns: {missing}')
    frame = frame[frame.accepted == 1].copy()
    if frame[KEYS].duplicated().any():
        raise RuntimeError('duplicate accepted directional event keys')
    if frame.effective_start_frame.isna().any():
        raise RuntimeError('accepted events contain missing effective_start_frame')
    return frame[required].sort_values(KEYS).reset_index(drop=True)


def load_changed_rows(path: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    missing = [column for column in ALLOWED_CHANGED_COLUMNS if column not in header.columns]
    if missing:
        raise RuntimeError(f'changed-row table missing allowed columns: {missing}')
    frame = pd.read_csv(path, usecols=ALLOWED_CHANGED_COLUMNS)
    if frame.duplicated(KEYS + ['frame', 'baseline_label']).any():
        raise RuntimeError('duplicate changed-row geometry keys')
    return frame.sort_values(KEYS + ['frame']).reset_index(drop=True)


def load_track_rows(
    path: Path,
    relevant_ids: set[int],
) -> tuple[pd.DataFrame, dict[int, int]]:
    full = pd.read_csv(path, header=None, names=TRACK_COLUMNS)
    full['frame'] = full.frame.astype(int)
    full['track_id'] = full.track_id.astype(int)
    if full.duplicated(['frame', 'track_id']).any():
        raise RuntimeError(f'duplicate baseline frame-track keys: {path}')
    full = full.sort_values(['track_id', 'frame']).reset_index(drop=True)
    fifth_frame: dict[int, int] = {}
    for track_id, group in full.groupby('track_id', sort=False):
        if len(group) >= 5:
            fifth_frame[int(track_id)] = int(group.iloc[4].frame)

    frame = full[full.track_id.isin(relevant_ids)].copy()
    frame['x2'] = frame.x + frame.w
    frame['y2'] = frame.y + frame.h
    return frame.reset_index(drop=True), fifth_frame


def load_tracklet_start_bank(
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    feature_path = cache_dir / 'tracklet_reid_features.npz'
    index_path = cache_dir / 'tracklet_reid_index.csv'
    if not feature_path.exists() or not index_path.exists():
        raise RuntimeError(f'incomplete tracklet cache: {cache_dir}')
    with np.load(feature_path, allow_pickle=False) as archive:
        track_ids = archive['track_id'].astype(int)
        start = l2_rows(np.asarray(archive['start'], dtype=np.float32))
    index = pd.read_csv(index_path)
    required = ['track_id', 'num_start_samples', 'has_feature']
    missing = [column for column in required if column not in index.columns]
    if missing:
        raise RuntimeError(f'tracklet index missing columns {missing}: {index_path}')
    if not np.array_equal(track_ids, index.track_id.astype(int).to_numpy()):
        raise RuntimeError(f'tracklet feature/index order mismatch: {cache_dir}')
    valid = (index.num_start_samples >= 5) & (index.has_feature == 1)
    return track_ids[valid.to_numpy()], start[valid.to_numpy()]


def collect_needed_rows(
    events: pd.DataFrame,
    changed: pd.DataFrame,
    tracks: pd.DataFrame,
) -> pd.DataFrame:
    by_track = {int(track_id): group for track_id, group in tracks.groupby('track_id', sort=False)}
    changed_groups = {key: group for key, group in changed.groupby(KEYS, sort=False)}
    needed: list[pd.DataFrame] = []
    for _, event in events.iterrows():
        key = tuple(event[column] for column in KEYS)
        event_changed = changed_groups.get(key)
        if event_changed is None:
            raise RuntimeError(f'missing changed rows while collecting histories: {key}')
        source_labels = sorted(event_changed.edited_label.astype(int).unique().tolist())
        if len(source_labels) != 1:
            raise RuntimeError(f'event has non-unique actual source label: {key} -> {source_labels}')
        receiver_labels = sorted(event_changed.baseline_label.astype(int).unique().tolist())
        start = int(event.effective_start_frame)
        for track_id in source_labels + receiver_labels:
            history = by_track.get(track_id)
            if history is None:
                continue
            history = history[history.frame < start].tail(max(HISTORY_SIZES))
            needed.append(history)
    changed_keys = changed[['frame', 'baseline_label']].rename(columns={'baseline_label': 'track_id'})
    future = tracks.merge(changed_keys.drop_duplicates(), on=['frame', 'track_id'], how='inner')
    needed.append(future)
    result = pd.concat(needed, ignore_index=True).drop_duplicates(['frame', 'track_id'])
    return result.sort_values(['frame', 'track_id']).reset_index(drop=True)


def match_rows_to_dump(
    needed: pd.DataFrame,
    detections: np.ndarray,
    offsets: np.ndarray,
    column_map: dict[str, int],
    min_iou: float,
) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    max_frame = len(offsets) - 1
    for frame, group in needed.groupby('frame', sort=True):
        frame = int(frame)
        if frame < 1 or frame > max_frame:
            for _, row in group.iterrows():
                records.append({
                    'frame': frame,
                    'track_id': int(row.track_id),
                    'feature_index': -1,
                    'match_iou': 0.0,
                    'detection_score': float('nan'),
                })
            continue
        lo = int(offsets[frame - 1])
        hi = int(offsets[frame])
        block = np.asarray(detections[lo:hi], dtype=np.float32)
        valid = block[:, column_map['has_reid']] > 0.5
        valid_positions = np.flatnonzero(valid)
        detection_boxes = block[valid][:, [
            column_map['x1'], column_map['y1'], column_map['x2'], column_map['y2']
        ]]
        track_boxes = group[['x', 'y', 'x2', 'y2']].to_numpy(np.float32)
        iou = pair_iou(track_boxes, detection_boxes)
        if iou.shape[1] == 0:
            best_local = np.full(len(group), -1, dtype=int)
            best_iou = np.zeros(len(group), dtype=np.float32)
        else:
            best_local = np.argmax(iou, axis=1)
            best_iou = iou[np.arange(len(group)), best_local]
        for position, (_, row) in enumerate(group.iterrows()):
            accepted = bool(best_local[position] >= 0 and best_iou[position] >= min_iou)
            if accepted:
                detection_position = int(valid_positions[best_local[position]])
                feature_index = lo + detection_position
                score = float(block[detection_position, column_map['score']])
            else:
                feature_index = -1
                score = float('nan')
            records.append({
                'frame': frame,
                'track_id': int(row.track_id),
                'feature_index': int(feature_index),
                'match_iou': float(best_iou[position]) if best_local[position] >= 0 else 0.0,
                'detection_score': score,
            })
    return pd.DataFrame(records).sort_values(['frame', 'track_id']).reset_index(drop=True)


def matched_feature_rows(
    rows: pd.DataFrame,
    match_map: pd.DataFrame,
    features: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    merged = rows.merge(match_map, on=['frame', 'track_id'], how='left', validate='one_to_one')
    valid = merged.feature_index.fillna(-1).astype(int) >= 0
    indices = merged.loc[valid, 'feature_index'].astype(int).to_numpy()
    values = np.asarray(features[indices], dtype=np.float32) if len(indices) else np.zeros((0, features.shape[1]), dtype=np.float32)
    return values, merged


def aggregate_values(prefix: str, values: np.ndarray, output: dict[str, float | int]) -> None:
    if len(values) == 0:
        for name in ['mean', 'q25', 'median', 'min', 'max', 'std']:
            output[f'{prefix}_{name}'] = float('nan')
        return
    output[f'{prefix}_mean'] = float(np.mean(values))
    output[f'{prefix}_q25'] = float(np.quantile(values, 0.25))
    output[f'{prefix}_median'] = float(np.median(values))
    output[f'{prefix}_min'] = float(np.min(values))
    output[f'{prefix}_max'] = float(np.max(values))
    output[f'{prefix}_std'] = float(np.std(values))


def build_sequence_features(
    seq_events: pd.DataFrame,
    seq_changed: pd.DataFrame,
    track_path: Path,
    dump_path: Path,
    tracklet_cache_dir: Path,
    min_iou: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    relevant_ids = (
        set(seq_events.u.astype(int))
        | set(seq_events.v.astype(int))
        | set(seq_changed.baseline_label.astype(int))
        | set(seq_changed.edited_label.astype(int))
    )
    tracks, fifth_frame = load_track_rows(track_path, relevant_ids)
    tracklet_ids, tracklet_start = load_tracklet_start_bank(tracklet_cache_dir)
    tracklet_position = {int(track_id): index for index, track_id in enumerate(tracklet_ids)}
    needed = collect_needed_rows(seq_events, seq_changed, tracks)

    detections = mmap_npz_member(dump_path, 'detections.npy')
    features = mmap_npz_member(dump_path, 'features.npy')
    offsets = mmap_npz_member(dump_path, 'frame_offsets.npy')
    with np.load(dump_path, allow_pickle=False) as archive:
        columns = [str(value) for value in archive['columns'].tolist()]
    column_map = {column: index for index, column in enumerate(columns)}
    required_dump = ['x1', 'y1', 'x2', 'y2', 'score', 'has_reid']
    missing_dump = [column for column in required_dump if column not in column_map]
    if missing_dump:
        raise RuntimeError(f'dump missing columns {missing_dump}: {dump_path}')

    match_map = match_rows_to_dump(needed, detections, offsets, column_map, min_iou)
    by_track = {int(track_id): group for track_id, group in tracks.groupby('track_id', sort=False)}
    changed_groups = {key: group for key, group in seq_changed.groupby(KEYS, sort=False)}
    outputs: list[dict[str, float | int | str]] = []

    for _, event in seq_events.iterrows():
        key = tuple(event[column] for column in KEYS)
        future_rows = changed_groups.get(key)
        if future_rows is None:
            raise RuntimeError(f'missing changed rows for accepted event: {key}')
        nominal_source = int(event.u if event.transaction_type == 'u_to_v' else event.v)
        nominal_receiver = int(event.v if event.transaction_type == 'u_to_v' else event.u)
        source_labels = sorted(future_rows.edited_label.astype(int).unique().tolist())
        if len(source_labels) != 1:
            raise RuntimeError(f'event has non-unique actual source label: {key} -> {source_labels}')
        actual_source = int(source_labels[0])
        actual_receivers = sorted(future_rows.baseline_label.astype(int).unique().tolist())

        start = int(event.effective_start_frame)
        source_history = by_track.get(actual_source)
        if source_history is None:
            source_history = tracks.iloc[0:0].copy()
        source_history = source_history[source_history.frame < start].tail(max(HISTORY_SIZES))
        source_values, source_matches = matched_feature_rows(source_history, match_map, features)

        receiver_histories: dict[int, pd.DataFrame] = {}
        receiver_matches: dict[int, pd.DataFrame] = {}
        for receiver_label in actual_receivers:
            history = by_track.get(receiver_label)
            if history is None:
                history = tracks.iloc[0:0].copy()
            history = history[history.frame < start].tail(max(HISTORY_SIZES))
            receiver_histories[receiver_label] = history
            _, matched = matched_feature_rows(history, match_map, features)
            receiver_matches[receiver_label] = matched
        future_geometry = future_rows[['frame', 'baseline_label', 'frames_after_handoff']].rename(
            columns={'baseline_label': 'track_id'}
        )
        future_geometry = future_geometry.merge(
            tracks[['frame', 'track_id', 'x', 'y', 'x2', 'y2']],
            on=['frame', 'track_id'],
            how='left',
            validate='one_to_one',
        )
        if future_geometry[['x', 'y', 'x2', 'y2']].isna().any().any():
            raise RuntimeError(f'missing receiver geometry for changed rows: {key}')
        future_values, future_matches = matched_feature_rows(future_geometry, match_map, features)

        record: dict[str, float | int | str] = {column: event[column] for column in KEYS}
        record['effective_start_frame'] = start
        record['nominal_source_label'] = nominal_source
        record['nominal_receiver_label'] = nominal_receiver
        record['actual_source_label'] = actual_source
        record['actual_receiver_label_count'] = int(len(actual_receivers))
        record['actual_source_matches_nominal'] = int(actual_source == nominal_source)
        record['receiver_rows_matching_nominal_fraction'] = float(
            np.mean(future_rows.baseline_label.astype(int).to_numpy() == nominal_receiver)
        )
        record['aggregate_anchor_used'] = int(
            actual_source != nominal_source
            or set(actual_receivers) != {nominal_receiver}
        )
        record['future_rows_total'] = int(len(future_rows))

        excluded_labels = {actual_source, *actual_receivers}
        third_party_ids = [
            int(track_id)
            for track_id in tracklet_ids
            if int(track_id) not in excluded_labels
            and int(track_id) in fifth_frame
            and fifth_frame[int(track_id)] < start
        ]
        third_positions = [tracklet_position[track_id] for track_id in third_party_ids]
        third_party_prototypes = (
            tracklet_start[np.asarray(third_positions, dtype=int)]
            if third_positions
            else np.zeros((0, tracklet_start.shape[1]), dtype=np.float32)
        )
        record['third_party_candidate_count'] = int(len(third_party_ids))

        source_prototypes: dict[int, np.ndarray | None] = {}
        receiver_prototypes: dict[int, dict[int, np.ndarray | None]] = {
            size: {} for size in HISTORY_SIZES
        }
        source_valid = source_matches[source_matches.feature_index.fillna(-1).astype(int) >= 0]
        for size in HISTORY_SIZES:
            source_tail = source_valid.tail(size)
            source_indices = source_tail.feature_index.astype(int).to_numpy()
            source_prototypes[size] = (
                l2_vector(np.asarray(features[source_indices], dtype=np.float32))
                if len(source_indices) else None
            )
            record[f'source_history_available_{size}'] = int(min(size, len(source_history)))
            record[f'source_history_matched_{size}'] = int(len(source_indices))
            record[f'source_history_coverage_{size}'] = float(
                len(source_indices) / max(1, min(size, len(source_history)))
            )

            receiver_available: list[int] = []
            receiver_matched: list[int] = []
            receiver_coverage: list[float] = []
            pre_cosines: list[float] = []
            for receiver_label in actual_receivers:
                history = receiver_histories[receiver_label]
                valid = receiver_matches[receiver_label]
                valid = valid[valid.feature_index.fillna(-1).astype(int) >= 0].tail(size)
                indices = valid.feature_index.astype(int).to_numpy()
                prototype = (
                    l2_vector(np.asarray(features[indices], dtype=np.float32))
                    if len(indices) else None
                )
                receiver_prototypes[size][receiver_label] = prototype
                available = int(min(size, len(history)))
                receiver_available.append(available)
                receiver_matched.append(int(len(indices)))
                receiver_coverage.append(float(len(indices) / max(1, available)))
                value = cosine(source_prototypes[size], prototype)
                if np.isfinite(value):
                    pre_cosines.append(value)
            record[f'receiver_family_history_available_sum_{size}'] = int(sum(receiver_available))
            record[f'receiver_family_history_matched_sum_{size}'] = int(sum(receiver_matched))
            record[f'receiver_family_history_coverage_mean_{size}'] = float(np.mean(receiver_coverage))
            record[f'receiver_family_history_coverage_min_{size}'] = float(np.min(receiver_coverage))
            aggregate_values(
                f'pre_source_receiver_cos_{size}',
                np.asarray(pre_cosines, dtype=float),
                record,
            )
        record['source_history_drift_5_60'] = 1.0 - cosine(
            source_prototypes[5], source_prototypes[60]
        )
        receiver_drifts = [
            1.0 - cosine(receiver_prototypes[5][label], receiver_prototypes[60][label])
            for label in actual_receivers
        ]
        receiver_drifts = [value for value in receiver_drifts if np.isfinite(value)]
        aggregate_values(
            'receiver_family_history_drift_5_60',
            np.asarray(receiver_drifts, dtype=float),
            record,
        )

        future_valid = future_matches[future_matches.feature_index.fillna(-1).astype(int) >= 0].copy()
        if len(future_valid):
            future_indices = future_valid.feature_index.astype(int).to_numpy()
            normalized_future = l2_rows(np.asarray(features[future_indices], dtype=np.float32))
            future_valid['_feature_row'] = list(normalized_future)
            if len(third_party_prototypes):
                third_similarity = normalized_future @ third_party_prototypes.T
                partition = np.partition(
                    third_similarity,
                    kth=max(0, third_similarity.shape[1] - 2),
                    axis=1,
                )
                third_top1 = partition[:, -1]
                third_top2 = (
                    partition[:, -2]
                    if third_similarity.shape[1] >= 2
                    else np.full(len(partition), np.nan, dtype=np.float32)
                )
            else:
                third_top1 = np.full(len(normalized_future), np.nan, dtype=np.float32)
                third_top2 = np.full(len(normalized_future), np.nan, dtype=np.float32)
            future_valid['_third_party_top1'] = third_top1
            future_valid['_third_party_top2'] = third_top2
            future_valid['_third_party_gap'] = third_top1 - third_top2
        else:
            future_valid['_feature_row'] = pd.Series(dtype=object)
            future_valid['_third_party_top1'] = pd.Series(dtype=float)
            future_valid['_third_party_top2'] = pd.Series(dtype=float)
            future_valid['_third_party_gap'] = pd.Series(dtype=float)

        for horizon in HORIZONS:
            suffix = 'full' if horizon is None else f'h{horizon}'
            total_subset = future_matches if horizon is None else future_matches[
                future_matches.frames_after_handoff <= horizon
            ]
            valid_subset = future_valid if horizon is None else future_valid[
                future_valid.frames_after_handoff <= horizon
            ]
            record[f'future_rows_{suffix}'] = int(len(total_subset))
            record[f'future_reid_rows_{suffix}'] = int(len(valid_subset))
            record[f'future_reid_coverage_{suffix}'] = float(len(valid_subset) / max(1, len(total_subset)))
            if not len(valid_subset):
                for history_size in HISTORY_SIZES:
                    for measure in ['donor_cos', 'receiver_cos', 'margin']:
                        aggregate_values(f'{measure}_{history_size}_{suffix}', np.asarray([]), record)
                    record[f'margin_positive_frac_{history_size}_{suffix}'] = float('nan')
                record[f'future_internal_cos_{suffix}'] = float('nan')
                continue
            values = np.stack(valid_subset._feature_row.to_list()).astype(np.float32)
            future_prototype = l2_vector(values)
            record[f'future_internal_cos_{suffix}'] = float(np.mean(values @ future_prototype)) if future_prototype is not None else float('nan')
            for history_size in HISTORY_SIZES:
                source_proto = source_prototypes[history_size]
                source_cos = values @ source_proto if source_proto is not None else np.full(len(values), np.nan)
                receiver_cos = np.full(len(values), np.nan, dtype=np.float32)
                receiver_labels = valid_subset.track_id.astype(int).to_numpy()
                for row_index, receiver_label in enumerate(receiver_labels):
                    receiver_proto = receiver_prototypes[history_size].get(int(receiver_label))
                    if receiver_proto is not None:
                        receiver_cos[row_index] = float(np.dot(values[row_index], receiver_proto))
                valid_margin = np.isfinite(source_cos) & np.isfinite(receiver_cos)
                margin = source_cos[valid_margin] - receiver_cos[valid_margin]
                aggregate_values(
                    f'source_cos_{history_size}_{suffix}',
                    source_cos[np.isfinite(source_cos)],
                    record,
                )
                aggregate_values(
                    f'receiver_cos_{history_size}_{suffix}',
                    receiver_cos[np.isfinite(receiver_cos)],
                    record,
                )
                aggregate_values(f'margin_{history_size}_{suffix}', margin, record)
                record[f'margin_rows_{history_size}_{suffix}'] = int(len(margin))
                record[f'margin_coverage_{history_size}_{suffix}'] = float(
                    len(margin) / max(1, len(valid_subset))
                )
                record[f'margin_positive_frac_{history_size}_{suffix}'] = float(np.mean(margin > 0.0)) if len(margin) else float('nan')

                if history_size == 20:
                    third_top1 = valid_subset._third_party_top1.to_numpy(float)
                    third_top2 = valid_subset._third_party_top2.to_numpy(float)
                    third_gap = valid_subset._third_party_gap.to_numpy(float)
                    valid_third = np.isfinite(third_top1)
                    pair_valid = (
                        np.isfinite(source_cos)
                        & np.isfinite(receiver_cos)
                        & valid_third
                    )
                    pair_best = np.maximum(source_cos[pair_valid], receiver_cos[pair_valid])
                    source_vs_third = source_cos[pair_valid] - third_top1[pair_valid]
                    receiver_vs_third = receiver_cos[pair_valid] - third_top1[pair_valid]
                    pair_vs_third = pair_best - third_top1[pair_valid]
                    aggregate_values(
                        f'third_party_top1_{suffix}',
                        third_top1[valid_third],
                        record,
                    )
                    aggregate_values(
                        f'third_party_gap_{suffix}',
                        third_gap[np.isfinite(third_gap)],
                        record,
                    )
                    aggregate_values(
                        f'source_vs_third_{suffix}',
                        source_vs_third,
                        record,
                    )
                    aggregate_values(
                        f'receiver_vs_third_{suffix}',
                        receiver_vs_third,
                        record,
                    )
                    aggregate_values(
                        f'pair_vs_third_{suffix}',
                        pair_vs_third,
                        record,
                    )
                    record[f'third_party_rows_{suffix}'] = int(pair_valid.sum())
                    record[f'third_party_coverage_{suffix}'] = float(
                        pair_valid.sum() / max(1, len(valid_subset))
                    )
                    record[f'third_party_win_fraction_{suffix}'] = (
                        float(np.mean(pair_vs_third < 0.0))
                        if len(pair_vs_third)
                        else float('nan')
                    )
        outputs.append(record)

    feature_frame = pd.DataFrame(outputs).sort_values(KEYS).reset_index(drop=True)
    match_rate = float((match_map.feature_index >= 0).mean()) if len(match_map) else 0.0
    sequence_report = {
        'seq': str(seq_events.seq.iloc[0]),
        'events': int(len(seq_events)),
        'relevant_track_ids': int(len(relevant_ids)),
        'needed_baseline_rows': int(len(needed)),
        'matched_baseline_rows': int((match_map.feature_index >= 0).sum()),
        'baseline_row_match_rate': match_rate,
        'mean_future_full_coverage': float(feature_frame.future_reid_coverage_full.mean()),
        'minimum_future_full_coverage': float(feature_frame.future_reid_coverage_full.min()),
        'events_with_zero_future_reid': int((feature_frame.future_reid_rows_full == 0).sum()),
        'events_with_missing_source20': int((feature_frame.source_history_matched_20 == 0).sum()),
        'events_with_missing_receiver_family20': int((feature_frame.receiver_family_history_matched_sum_20 == 0).sum()),
        'events_using_aggregate_anchor': int(feature_frame.aggregate_anchor_used.sum()),
        'mean_third_party_candidates': float(feature_frame.third_party_candidate_count.mean()),
        'minimum_third_party_candidates': int(feature_frame.third_party_candidate_count.min()),
        'events_with_zero_third_party_candidates': int((feature_frame.third_party_candidate_count == 0).sum()),
    }
    return feature_frame, sequence_report


def parse_sequence_mapping(values: list[str], label: str) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values:
        if '=' not in value:
            raise RuntimeError(f'{label} entries must be SEQ=PATH: {value}')
        sequence, path = value.split('=', 1)
        sequence = sequence.strip()
        if sequence in mapping:
            raise RuntimeError(f'duplicate {label} sequence: {sequence}')
        mapping[sequence] = Path(path)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--executability', required=True)
    parser.add_argument('--changed-rows', required=True)
    parser.add_argument('--track-result', action='append', required=True, help='SEQ=PATH')
    parser.add_argument('--dump-npz', action='append', required=True, help='SEQ=PATH')
    parser.add_argument('--tracklet-cache', action='append', required=True, help='SEQ=DIR')
    parser.add_argument('--min-iou', type=float, default=0.5)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    events = load_events(args.executability)
    changed = load_changed_rows(args.changed_rows)
    track_paths = parse_sequence_mapping(args.track_result, 'track-result')
    dump_paths = parse_sequence_mapping(args.dump_npz, 'dump-npz')
    tracklet_cache_dirs = parse_sequence_mapping(args.tracklet_cache, 'tracklet-cache')
    sequences = sorted(events.seq.unique())
    if (
        set(track_paths) != set(sequences)
        or set(dump_paths) != set(sequences)
        or set(tracklet_cache_dirs) != set(sequences)
    ):
        raise RuntimeError('track/dump/tracklet-cache mappings must exactly cover accepted event sequences')
    if len(changed.merge(events[KEYS], on=KEYS, how='inner')) != len(changed):
        raise RuntimeError('changed-row table includes rows outside accepted event set')

    output_frames = []
    sequence_reports = []
    for sequence in sequences:
        frame, report = build_sequence_features(
            events[events.seq == sequence].reset_index(drop=True),
            changed[changed.seq == sequence].reset_index(drop=True),
            track_paths[sequence],
            dump_paths[sequence],
            tracklet_cache_dirs[sequence],
            args.min_iou,
        )
        output_frames.append(frame)
        sequence_reports.append(report)

    features = pd.concat(output_frames, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    if len(features) != len(events) or features[KEYS].duplicated().any():
        raise RuntimeError('boundary-local feature event cardinality mismatch')
    forbidden = [
        column for column in features.columns
        if any(token in column.lower() for token in ['gt_', 'row_class', 'delta_hota', 'delta_assa', 'idtp_delta'])
    ]
    if forbidden:
        raise RuntimeError(f'forbidden label columns in output: {forbidden}')

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    rounded(features).to_csv(out / 'boundary_local_appearance_features.csv', index=False)
    rounded(pd.DataFrame(sequence_reports)).to_csv(out / 'sequence_match_summary.csv', index=False)
    report = {
        'protocol': {
            'scope': 'Train-only boundary-local appearance features for accepted directional events.',
            'allowed_changed_row_columns': ALLOWED_CHANGED_COLUMNS,
            'forbidden_inputs': ['matched_gt_id', 'matched_gt_iou', 'dominant_gt', 'row_class', 'utility labels', 'TrackEval metrics', 'locked artifacts'],
            'history_sizes': HISTORY_SIZES,
            'future_horizons': ['h30', 'h60', 'h120', 'h300', 'full'],
            'minimum_detection_iou': args.min_iou,
            'feature_source': 'Memory-mapped YOLOX FastReID detection features; event-pre source/receiver histories, event-post changed receiver rows, and third-party tracklet start prototypes whose fifth baseline row predates the event.',
            'locked_rows_read': 0,
            'locked_labels_read': 0,
        },
        'dataset': {
            'events': int(len(features)),
            'sequences': sequences,
            'feature_columns': int(len(features.columns) - len(KEYS)),
            'changed_rows': int(len(changed)),
        },
        'sequence_reports': sequence_reports,
        'overall': {
            'mean_future_full_coverage': float(features.future_reid_coverage_full.mean()),
            'minimum_future_full_coverage': float(features.future_reid_coverage_full.min()),
            'events_with_zero_future_reid': int((features.future_reid_rows_full == 0).sum()),
            'events_with_missing_source20': int((features.source_history_matched_20 == 0).sum()),
            'events_with_missing_receiver_family20': int((features.receiver_family_history_matched_sum_20 == 0).sum()),
            'events_using_aggregate_anchor': int(features.aggregate_anchor_used.sum()),
            'mean_third_party_candidates': float(features.third_party_candidate_count.mean()),
            'minimum_third_party_candidates': int(features.third_party_candidate_count.min()),
            'events_with_zero_third_party_candidates': int((features.third_party_candidate_count == 0).sum()),
        },
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    manifest = {
        path.name: sha256(path)
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != 'prediction_manifest.json'
    }
    (out / 'prediction_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
