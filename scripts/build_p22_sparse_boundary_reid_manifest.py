from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


KEYS = ['seq', 'canonical_rank', 'u', 'v', 'boundary_frame', 'transaction_type']
WINDOW_ROWS = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_mapping(values: list[str], label: str, sequences: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if '=' not in value:
            raise RuntimeError(f'{label} entries must be SEQ=PATH: {value}')
        sequence, raw_path = value.split('=', 1)
        if sequence in result:
            raise RuntimeError(f'duplicate {label} sequence: {sequence}')
        result[sequence] = Path(raw_path)
    missing = sorted(set(sequences) - set(result))
    extra = sorted(set(result) - set(sequences))
    if missing or extra:
        raise RuntimeError(f'{label} mapping mismatch: missing={missing}, extra={extra}')
    return result


def load_tracks(path: Path) -> pd.DataFrame:
    columns = ['frame', 'track_id', 'x', 'y', 'w', 'h', 'score', 'a', 'b', 'c']
    frame = pd.read_csv(path, header=None, names=columns)
    frame['frame'] = frame.frame.astype(int)
    frame['track_id'] = frame.track_id.astype(int)
    if frame.duplicated(['frame', 'track_id']).any():
        raise RuntimeError(f'duplicate frame-track rows: {path}')
    if (frame[['w', 'h']] <= 0).any().any():
        raise RuntimeError(f'non-positive tracker boxes: {path}')
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--executability', required=True)
    parser.add_argument('--track-result', action='append', required=True)
    parser.add_argument('--image-dir', action='append', required=True)
    parser.add_argument('--reid-config', required=True)
    parser.add_argument('--reid-weights', required=True)
    parser.add_argument('--runtime-status', default='manifest_only')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    events = pd.read_csv(args.executability)
    required = KEYS + [
        'effective_start_frame',
        'donor_anchor',
        'receiver_anchor',
    ]
    missing = [column for column in required if column not in events.columns]
    if missing:
        raise RuntimeError(f'executability columns missing: {missing}')
    if len(events) != 11705 or events.duplicated(KEYS).any():
        raise RuntimeError('event-bank key integrity failure')
    sequences = sorted(events.seq.unique().tolist())
    track_paths = parse_mapping(args.track_result, 'track-result', sequences)
    image_dirs = parse_mapping(args.image_dir, 'image-dir', sequences)
    config_path = Path(args.reid_config)
    weights_path = Path(args.reid_weights)
    if not config_path.is_file() or not weights_path.is_file():
        raise RuntimeError('ReID config or weights missing')

    crop_records: dict[tuple[str, int, int], dict[str, object]] = {}
    index_rows: list[dict[str, object]] = []
    per_sequence: list[dict[str, object]] = []
    for sequence in sequences:
        tracks = load_tracks(track_paths[sequence])
        by_track = {
            int(track_id): group.sort_values('frame')
            for track_id, group in tracks.groupby('track_id', sort=False)
        }
        image_dir = image_dirs[sequence]
        if not image_dir.is_dir():
            raise RuntimeError(f'image directory missing: {image_dir}')
        checked_frames: dict[int, str] = {}
        before = len(crop_records)
        sequence_events = events[events.seq == sequence].sort_values(KEYS)
        for event in sequence_events.itertuples(index=False):
            start = int(event.effective_start_frame)
            donor = int(event.donor_anchor)
            receiver = int(event.receiver_anchor)
            if donor not in by_track or receiver not in by_track:
                raise RuntimeError(f'event anchor missing from tracks: {sequence}')
            windows = {
                'donor_history': by_track[donor][by_track[donor].frame < start].tail(
                    WINDOW_ROWS
                ),
                'receiver_history': by_track[receiver][
                    by_track[receiver].frame < start
                ].tail(WINDOW_ROWS),
                'receiver_future': by_track[receiver][
                    by_track[receiver].frame >= start
                ].head(WINDOW_ROWS),
            }
            if any(window.empty for window in windows.values()):
                key = tuple(getattr(event, column) for column in KEYS)
                raise RuntimeError(f'empty sparse ReID window: {key}')
            event_key = {column: getattr(event, column) for column in KEYS}
            for role, window in windows.items():
                for position, row in enumerate(window.itertuples(index=False), start=1):
                    frame_number = int(row.frame)
                    track_id = int(row.track_id)
                    crop_key = (sequence, frame_number, track_id)
                    if frame_number not in checked_frames:
                        image_path = image_dir / f'{frame_number:06d}.jpg'
                        if not image_path.is_file():
                            raise RuntimeError(f'image missing: {image_path}')
                        checked_frames[frame_number] = str(image_path)
                    record = {
                        'seq': sequence,
                        'frame': frame_number,
                        'track_id': track_id,
                        'x1': float(row.x),
                        'y1': float(row.y),
                        'x2': float(row.x + row.w),
                        'y2': float(row.y + row.h),
                        'track_score': float(row.score),
                        'image_path': checked_frames[frame_number],
                    }
                    previous = crop_records.get(crop_key)
                    if previous is not None:
                        geometry = ['x1', 'y1', 'x2', 'y2', 'track_score']
                        if any(abs(float(previous[c]) - float(record[c])) > 1e-9 for c in geometry):
                            raise RuntimeError(f'inconsistent duplicate crop geometry: {crop_key}')
                    else:
                        crop_records[crop_key] = record
                    index_rows.append(
                        {
                            **event_key,
                            'effective_start_frame': start,
                            'donor_anchor': donor,
                            'receiver_anchor': receiver,
                            'role': role,
                            'role_position': position,
                            'crop_seq': sequence,
                            'crop_frame': frame_number,
                            'crop_track_id': track_id,
                        }
                    )
        per_sequence.append(
            {
                'seq': sequence,
                'events': int(len(sequence_events)),
                'unique_crops': int(len(crop_records) - before),
                'unique_image_frames': int(len(checked_frames)),
            }
        )

    crops = pd.DataFrame(crop_records.values()).sort_values(
        ['seq', 'frame', 'track_id']
    )
    crops.insert(0, 'crop_id', range(1, len(crops) + 1))
    crop_ids = crops.set_index(['seq', 'frame', 'track_id']).crop_id
    event_index = pd.DataFrame(index_rows)
    event_index['crop_id'] = [
        int(crop_ids.loc[(sequence, frame, track_id)])
        for sequence, frame, track_id in event_index[
            ['crop_seq', 'crop_frame', 'crop_track_id']
        ].itertuples(index=False, name=None)
    ]
    event_index = event_index.sort_values(
        KEYS + ['role', 'role_position', 'crop_id']
    )
    if event_index.duplicated(KEYS + ['role', 'role_position']).any():
        raise RuntimeError('duplicate event-role-position rows')
    role_sizes = event_index.groupby(KEYS + ['role'], sort=True).size()
    expected_role_groups = len(events) * 3
    if len(role_sizes) != expected_role_groups:
        raise RuntimeError(
            f'event-role group mismatch: actual={len(role_sizes)}, '
            f'expected={expected_role_groups}'
        )
    if int(role_sizes.min()) < 1 or int(role_sizes.max()) > WINDOW_ROWS:
        raise RuntimeError('event-role window size is outside the fixed 1..5 range')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops.to_csv(out_dir / 'sparse_crop_manifest.csv', index=False)
    event_index.to_csv(out_dir / 'event_window_index.csv', index=False)
    pd.DataFrame(per_sequence).to_csv(
        out_dir / 'sequence_crop_summary.csv', index=False
    )
    report = {
        'protocol': {
            'scope': 'Sparse appearance-observation manifest for P22/P23 boundary events.',
            'window_rows_per_role': WINDOW_ROWS,
            'roles': ['donor_history', 'receiver_history', 'receiver_future'],
            'selection': 'Last five pre-boundary rows for donor and receiver; first five receiver rows at or after the effective start.',
            'deduplication_key': ['seq', 'frame', 'track_id'],
            'feature_extraction_performed': False,
            'runtime_status': args.runtime_status,
            'ground_truth_inputs': 0,
            'utility_inputs': 0,
            'trackeval_calls': 0,
            'p15_locked_labels_read': 0,
            'p15_locked_trackeval_calls': 0,
            'p15_remaining_locked_rows_untouched': 156,
        },
        'dataset': {
            'events': int(len(events)),
            'sequences': sequences,
            'event_window_rows': int(len(event_index)),
            'maximum_event_window_rows': int(len(events) * 3 * WINDOW_ROWS),
            'minimum_rows_per_event_role': int(role_sizes.min()),
            'maximum_rows_per_event_role': int(role_sizes.max()),
            'unique_crops': int(len(crops)),
            'unique_image_frames': int(
                crops[['seq', 'frame']].drop_duplicates().shape[0]
            ),
            'duplicate_crop_keys': int(crops.duplicated(['seq', 'frame', 'track_id']).sum()),
            'duplicate_event_role_positions': int(
                event_index.duplicated(KEYS + ['role', 'role_position']).sum()
            ),
        },
        'reid_assets': {
            'config_path': str(config_path),
            'config_sha256': sha256(config_path),
            'weights_path': str(weights_path),
            'weights_sha256': sha256(weights_path),
        },
        'decision': {
            'manifest_ready': True,
            'appearance_features_ready': False,
            'deployment_allowed': False,
            'locked_manifest_created': False,
            'p15_policy_changed': False,
            'next_stage': 'Run deterministic sparse FastReID extraction when the host inference service is available, then audit appearance-conditioned rescue under strict sequence-LOSO.',
        },
    }
    report_path = out_dir / 'report.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    manifest = {
        'schema_version': 1,
        'files': {
            path.name: sha256(path)
            for path in sorted(out_dir.iterdir())
            if path.is_file() and path.name != 'prediction_manifest.json'
        },
    }
    (out_dir / 'prediction_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n'
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
