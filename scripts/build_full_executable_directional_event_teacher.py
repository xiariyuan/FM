from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from build_directional_local_counterfactual_labels import (
    HORIZONS,
    compute_horizon_metrics,
    load_gt,
    match_records_to_gt,
)
from build_generic_directional_local_counterfactual_bank import (
    EVENT_KEYS,
    enumerate_candidates,
    load_baseline_records,
    rounded,
    sha256,
)


def parse_mapping(values: list[str], label: str) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values:
        if '=' not in value:
            raise RuntimeError(f'{label} entries must be SEQ=PATH: {value}')
        sequence, raw_path = value.split('=', 1)
        if sequence in mapping:
            raise RuntimeError(f'duplicate {label} sequence: {sequence}')
        mapping[sequence] = Path(raw_path)
    return mapping


def build_event_teacher(
    records: list[dict[str, Any]],
    events: list[dict[str, Any]],
    gt_path: Path,
    iou_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    matched_gt_ids, gt_ious, match_summary = match_records_to_gt(
        records,
        load_gt(gt_path),
        iou_threshold,
    )
    label_indices: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        label_indices[int(record['label'])].append(index)

    rows: list[dict[str, Any]] = []
    for event in events:
        start_frame = int(event['effective_start_frame'])
        donor_anchor = int(event['donor_anchor'])
        receiver_anchor = int(event['receiver_anchor'])
        target_indices = list(event['target_indices'])
        donor_history = [
            index
            for index in label_indices[donor_anchor]
            if int(records[index]['frame']) < start_frame
        ]
        receiver_history = {
            receiver_anchor: [
                index
                for index in label_indices[receiver_anchor]
                if int(records[index]['frame']) < start_frame
            ]
        }
        future_by_anchor = {receiver_anchor: target_indices}
        output = {
            key: value
            for key, value in event.items()
            if key not in {'target_indices', 'reject_reason'}
        }
        output['receiver_anchor_count'] = 1
        output['receiver_anchor_counts'] = json.dumps(
            {str(receiver_anchor): len(target_indices)}, sort_keys=True
        )
        output['target_rows_with_receiver_anchor'] = len(target_indices)
        for horizon_name, horizon in HORIZONS:
            metrics, _ = compute_horizon_metrics(
                records,
                matched_gt_ids,
                gt_ious,
                donor_history,
                receiver_history,
                future_by_anchor,
                start_frame,
                horizon,
            )
            output.update(
                {f'{horizon_name}_{key}': value for key, value in metrics.items()}
            )
        rows.append(output)
    frame = pd.DataFrame(rows)
    if frame.duplicated(EVENT_KEYS).any():
        raise RuntimeError('duplicate event keys in full executable teacher')
    return frame, match_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', action='append', required=True, help='SEQ=PATH')
    parser.add_argument('--gt', action='append', required=True, help='SEQ=PATH')
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    baseline_paths = parse_mapping(args.baseline, 'baseline')
    gt_paths = parse_mapping(args.gt, 'gt')
    if set(baseline_paths) != set(gt_paths):
        raise RuntimeError('baseline and GT mappings must cover the same sequences')
    sequences = sorted(baseline_paths)

    candidate_parts: list[pd.DataFrame] = []
    executability_parts: list[pd.DataFrame] = []
    teacher_parts: list[pd.DataFrame] = []
    matching_parts: list[pd.DataFrame] = []
    sequence_rows: list[dict[str, Any]] = []

    for sequence in sequences:
        records = load_baseline_records(baseline_paths[sequence])
        events, candidate_audit = enumerate_candidates(sequence, records)
        # The complete executable event set is now frozen. GT is first read below.
        teacher, matching = build_event_teacher(
            records,
            events,
            gt_paths[sequence],
            args.iou_threshold,
        )
        executability = teacher[
            EVENT_KEYS
            + [
                'accepted',
                'changed_rows',
                'effective_start_frame',
                'donor_raw_tid',
                'receiver_raw_tid',
                'donor_anchor',
                'receiver_anchor',
            ]
        ].copy()
        executability['reject_reason'] = ''
        candidate_frame = pd.DataFrame(candidate_audit)
        candidate_frame['selected_for_teacher'] = candidate_frame.accepted.astype(int)

        candidate_parts.append(candidate_frame)
        executability_parts.append(executability)
        teacher_parts.append(teacher)
        matching_parts.append(pd.DataFrame([{'seq': sequence, **matching}]))
        sequence_rows.append(
            {
                'seq': sequence,
                'tracker_rows': len(records),
                'track_ids': len({int(record['raw_tid']) for record in records}),
                'ordered_candidate_directions': len(candidate_audit),
                'executable_events': len(teacher),
                'changed_rows_represented': int(teacher.changed_rows.sum()),
                'positive_events': int((teacher.full_idtp_delta_norm > 0.0).sum()),
                'negative_events': int((teacher.full_idtp_delta_norm < 0.0).sum()),
                'zero_events': int((teacher.full_idtp_delta_norm == 0.0).sum()),
                'positive_fraction': float(
                    (teacher.full_idtp_delta_norm > 0.0).mean()
                ),
                'gt_match_rate': float(
                    matching['matched_tracker_rows']
                    / max(1, matching['tracker_rows'])
                ),
                'baseline_sha256': sha256(baseline_paths[sequence]),
                'gt_sha256': sha256(gt_paths[sequence]),
            }
        )

    candidate_frame = pd.concat(candidate_parts, ignore_index=True)
    executability_frame = pd.concat(executability_parts, ignore_index=True)
    teacher_frame = pd.concat(teacher_parts, ignore_index=True)
    matching_frame = pd.concat(matching_parts, ignore_index=True)
    sequence_frame = pd.DataFrame(sequence_rows)
    if teacher_frame.duplicated(EVENT_KEYS).any():
        raise RuntimeError('duplicate full-bank event keys')
    if len(teacher_frame) != len(executability_frame):
        raise RuntimeError('teacher/executability cardinality mismatch')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    rounded(candidate_frame).to_csv(out_dir / 'candidate_audit.csv', index=False)
    rounded(executability_frame).to_csv(out_dir / 'executability.csv', index=False)
    rounded(teacher_frame).to_csv(out_dir / 'event_local_labels.csv', index=False)
    rounded(matching_frame).to_csv(
        out_dir / 'sequence_gt_matching_summary.csv', index=False
    )
    rounded(sequence_frame).to_csv(out_dir / 'sequence_summary.csv', index=False)

    report = {
        'protocol': {
            'scope': 'Full executable directional event-level counterfactual teacher bank.',
            'candidate_freeze': 'All executable directions from every overlapping raw-track pair at its final common frame are retained. No geometry ranking or teacher-based filtering is applied.',
            'leakage_rule': 'Candidate enumeration and executable-event freezing use tracker rows only. GT is first read after the complete event set is frozen.',
            'teacher_scope': 'Event-level multi-horizon local association and local IDTP targets. No row-level label file is emitted.',
            'global_trackeval_calls': 0,
            'p15_locked_labels_read': 0,
            'p15_locked_trackeval_calls': 0,
            'p15_remaining_locked_rows_untouched': 156,
            'minimum_gt_iou': args.iou_threshold,
        },
        'dataset': {
            'sequences': sequences,
            'sequence_count': len(sequences),
            'ordered_candidate_directions': len(candidate_frame),
            'executable_events': len(teacher_frame),
            'changed_rows_represented': int(teacher_frame.changed_rows.sum()),
            'positive_events': int((teacher_frame.full_idtp_delta_norm > 0.0).sum()),
            'negative_events': int((teacher_frame.full_idtp_delta_norm < 0.0).sum()),
            'zero_events': int((teacher_frame.full_idtp_delta_norm == 0.0).sum()),
            'positive_fraction': float(
                (teacher_frame.full_idtp_delta_norm > 0.0).mean()
            ),
        },
        'sequence_reports': sequence_rows,
    }
    report_path = out_dir / 'report.json'
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    manifest = {
        path.name: sha256(path)
        for path in sorted(out_dir.iterdir())
        if path.is_file() and path.name != 'manifest.json'
    }
    (out_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
