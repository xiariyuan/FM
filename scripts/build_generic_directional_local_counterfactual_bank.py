from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_directional_local_counterfactual_labels import (
    HORIZONS,
    compute_horizon_metrics,
    load_baseline_records,
    load_gt,
    match_records_to_gt,
    rounded,
    sha256,
)
from eval_directional_identity_handoff import (
    build_directional_sequence_index,
    plan_directional_handoff,
)


EVENT_KEYS = [
    'seq',
    'canonical_rank',
    'u',
    'v',
    'boundary_frame',
    'transaction_type',
]


def record_box(record: dict[str, Any]) -> np.ndarray:
    parts = record['parts']
    return np.asarray(
        [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])],
        dtype=float,
    )


def aggregate_boundary_box(
    records: list[dict[str, Any]], indices: list[int]
) -> np.ndarray:
    boxes = np.stack([record_box(records[index]) for index in indices])
    return np.median(boxes, axis=0)


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x1 = max(lx, rx)
    y1 = max(ly, ry)
    x2 = min(lx + lw, rx + rw)
    y2 = min(ly + lh, ry + rh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = lw * lh + rw * rh - intersection
    return float(intersection / max(union, 1e-12))


def boundary_features(
    records: list[dict[str, Any]],
    sequence_index: dict[str, Any],
    donor: int,
    receiver: int,
    boundary: int,
    handoff: int,
    target_indices: list[int],
) -> dict[str, float | int]:
    raw_frame_indices = sequence_index['raw_frame_indices']
    raw_frames = sequence_index['raw_frames']
    donor_box = aggregate_boundary_box(
        records, raw_frame_indices[(donor, boundary)]
    )
    receiver_box = aggregate_boundary_box(
        records, raw_frame_indices[(receiver, boundary)]
    )
    donor_center = donor_box[:2] + donor_box[2:] / 2.0
    receiver_center = receiver_box[:2] + receiver_box[2:] / 2.0
    geometric_scale = math.sqrt(
        max(1.0, math.sqrt(donor_box[2] * donor_box[3] * receiver_box[2] * receiver_box[3]))
    )
    donor_frames = raw_frames[donor]
    receiver_frames = raw_frames[receiver]
    common_frames = sorted(set(donor_frames) & set(receiver_frames))
    future_frames = sorted({int(records[index]['frame']) for index in target_indices})
    donor_scores = [
        float(records[index]['parts'][6])
        for index in raw_frame_indices[(donor, boundary)]
    ]
    receiver_scores = [
        float(records[index]['parts'][6])
        for index in raw_frame_indices[(receiver, boundary)]
    ]
    return {
        'candidate_policy_version': 1,
        'pair_common_frames': len(common_frames),
        'pair_common_span': common_frames[-1] - common_frames[0] + 1,
        'donor_first_frame': min(donor_frames),
        'donor_last_frame': max(donor_frames),
        'receiver_first_frame': min(receiver_frames),
        'receiver_last_frame': max(receiver_frames),
        'donor_track_rows': len(donor_frames),
        'receiver_track_rows': len(receiver_frames),
        'donor_history_rows': sum(frame < handoff for frame in donor_frames),
        'receiver_history_rows': sum(frame < handoff for frame in receiver_frames),
        'handoff_gap': handoff - boundary,
        'future_receiver_rows': len(target_indices),
        'future_receiver_unique_frames': len(future_frames),
        'future_receiver_span': future_frames[-1] - future_frames[0] + 1,
        'future_receiver_density': len(future_frames)
        / max(1, future_frames[-1] - future_frames[0] + 1),
        'boundary_center_distance_norm': float(
            np.linalg.norm(donor_center - receiver_center) / geometric_scale
        ),
        'boundary_bottom_gap_norm': float(
            abs((donor_box[1] + donor_box[3]) - (receiver_box[1] + receiver_box[3]))
            / max(1.0, math.sqrt(donor_box[3] * receiver_box[3]))
        ),
        'boundary_iou': box_iou(donor_box, receiver_box),
        'boundary_log_width_ratio_abs': float(
            abs(math.log(max(donor_box[2], 1.0) / max(receiver_box[2], 1.0)))
        ),
        'boundary_log_height_ratio_abs': float(
            abs(math.log(max(donor_box[3], 1.0) / max(receiver_box[3], 1.0)))
        ),
        'boundary_donor_score': float(np.mean(donor_scores)),
        'boundary_receiver_score': float(np.mean(receiver_scores)),
    }


def enumerate_candidates(
    sequence: str,
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sequence_index = build_directional_sequence_index(records)
    raw_frames = {
        int(track_id): set(frames)
        for track_id, frames in sequence_index['raw_frames'].items()
    }
    accepted: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    track_ids = sorted(raw_frames)

    for left_index, u in enumerate(track_ids):
        for v in track_ids[left_index + 1 :]:
            common = sorted(raw_frames[u] & raw_frames[v])
            if not common:
                continue
            boundary = common[-1]
            for transaction_type, donor, receiver in [
                ('u_to_v', u, v),
                ('v_to_u', v, u),
            ]:
                event = {
                    'seq': sequence,
                    'u': u,
                    'v': v,
                    'boundary_frame': boundary,
                }
                plan = plan_directional_handoff(
                    sequence_index, event, transaction_type
                )
                metadata = plan['metadata']
                row = {
                    'seq': sequence,
                    'u': u,
                    'v': v,
                    'boundary_frame': boundary,
                    'transaction_type': transaction_type,
                    'donor_raw_tid': donor,
                    'receiver_raw_tid': receiver,
                    'accepted': int(plan['accepted']),
                    'reject_reason': str(plan['reason']),
                    'effective_start_frame': metadata.get('handoff_frame'),
                    'changed_rows': int(metadata.get('changed_rows', 0)),
                    'pair_common_frames': len(common),
                    'pair_common_span': common[-1] - common[0] + 1,
                }
                audit.append(row)
                if not plan['accepted']:
                    continue
                target_indices = list(plan['target_indices'])
                handoff = int(metadata['handoff_frame'])
                accepted.append(
                    {
                        **row,
                        'donor_anchor': int(metadata['donor_anchor']),
                        'receiver_anchor': int(metadata['receiver_anchor']),
                        'target_indices': target_indices,
                        **boundary_features(
                            records,
                            sequence_index,
                            donor,
                            receiver,
                            boundary,
                            handoff,
                            target_indices,
                        ),
                    }
                )

    accepted.sort(
        key=lambda row: (
            int(row['boundary_frame']),
            int(row['u']),
            int(row['v']),
            str(row['transaction_type']),
        )
    )
    for rank, row in enumerate(accepted, start=1):
        row['canonical_rank'] = rank
    rank_lookup = {
        (
            row['u'],
            row['v'],
            row['boundary_frame'],
            row['transaction_type'],
        ): row['canonical_rank']
        for row in accepted
    }
    for row in audit:
        row['canonical_rank'] = rank_lookup.get(
            (
                row['u'],
                row['v'],
                row['boundary_frame'],
                row['transaction_type'],
            ),
            0,
        )
    audit.sort(
        key=lambda row: (
            int(row['boundary_frame']),
            int(row['u']),
            int(row['v']),
            str(row['transaction_type']),
        )
    )
    return accepted, audit


def build_teacher_bank(
    sequence: str,
    records: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    gt_path: Path,
    iou_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    matched_gt_ids, gt_ious, match_summary = match_records_to_gt(
        records,
        load_gt(gt_path),
        iou_threshold,
    )
    label_indices: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        label_indices[int(record['label'])].append(index)

    event_rows: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []
    feature_exclusions = {'target_indices', 'reject_reason'}

    for event in accepted:
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
            if key not in feature_exclusions
        }
        output['receiver_anchor_count'] = 1
        output['receiver_anchor_counts'] = json.dumps(
            {str(receiver_anchor): len(target_indices)}, sort_keys=True
        )
        output['target_rows_with_receiver_anchor'] = len(target_indices)
        full_identities: dict[str, Any] | None = None
        for horizon_name, horizon in HORIZONS:
            metrics, identities = compute_horizon_metrics(
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
            if horizon_name == 'full':
                full_identities = identities
        event_rows.append(output)

        if full_identities is None:
            raise RuntimeError('full-horizon identities were not produced')
        donor_gt = int(full_identities['donor_identity'])
        aggregate_receiver_gt = int(full_identities['receiver_identity'])
        receiver_identity_by_anchor = {
            int(anchor): int(identity)
            for anchor, identity in full_identities[
                'receiver_identity_by_anchor'
            ].items()
        }
        for index in target_indices:
            gt_id = int(matched_gt_ids[index])
            baseline_anchor = int(records[index]['label'])
            receiver_gt = receiver_identity_by_anchor.get(baseline_anchor, -1)
            if gt_id < 0:
                row_class = 'unmatched'
            elif donor_gt >= 0 and donor_gt == receiver_gt and gt_id == donor_gt:
                row_class = 'shared_history_identity'
            elif donor_gt >= 0 and gt_id == donor_gt:
                row_class = 'benefit'
            elif receiver_gt >= 0 and gt_id == receiver_gt:
                row_class = 'harm'
            else:
                row_class = 'other_gt'
            changed_rows.append(
                {
                    **{key: event[key] for key in EVENT_KEYS},
                    'frame': int(records[index]['frame']),
                    'frames_after_handoff': int(records[index]['frame'])
                    - start_frame,
                    'baseline_label': baseline_anchor,
                    'edited_label': donor_anchor,
                    'matched_gt_id': gt_id,
                    'matched_gt_iou': float(gt_ious[index]),
                    'donor_dominant_gt_full': donor_gt,
                    'receiver_dominant_gt_full': receiver_gt,
                    'aggregate_receiver_dominant_gt_full': aggregate_receiver_gt,
                    'row_class': row_class,
                }
            )

    return pd.DataFrame(event_rows), pd.DataFrame(changed_rows), match_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--gt', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    gt_path = Path(args.gt)
    records = load_baseline_records(baseline_path)
    accepted, candidate_audit = enumerate_candidates(args.sequence, records)
    event_labels, changed_rows, gt_summary = build_teacher_bank(
        args.sequence,
        records,
        accepted,
        gt_path,
        args.iou_threshold,
    )
    if event_labels.duplicated(EVENT_KEYS).any():
        raise RuntimeError('duplicate accepted event keys')
    if len(event_labels) != len(accepted):
        raise RuntimeError('accepted event/teacher cardinality mismatch')
    if len(changed_rows) != int(event_labels.changed_rows.sum()):
        raise RuntimeError('changed-row cardinality mismatch')

    executability_columns = EVENT_KEYS + [
        'accepted',
        'changed_rows',
        'effective_start_frame',
        'reject_reason',
        'donor_raw_tid',
        'receiver_raw_tid',
        'donor_anchor',
        'receiver_anchor',
    ]
    executability = event_labels.copy()
    executability['reject_reason'] = ''
    executability = executability[executability_columns]
    positive = int((event_labels.full_idtp_delta_norm > 0.0).sum())
    negative = int((event_labels.full_idtp_delta_norm < 0.0).sum())
    zero = int((event_labels.full_idtp_delta_norm == 0.0).sum())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    rounded(pd.DataFrame(candidate_audit)).to_csv(
        out_dir / 'candidate_audit.csv', index=False
    )
    rounded(executability).to_csv(out_dir / 'executability.csv', index=False)
    rounded(event_labels).to_csv(out_dir / 'event_local_labels.csv', index=False)
    rounded(changed_rows).to_csv(out_dir / 'changed_row_labels.csv', index=False)
    rounded(pd.DataFrame([{'seq': args.sequence, **gt_summary}])).to_csv(
        out_dir / 'sequence_gt_matching_summary.csv', index=False
    )

    report = {
        'protocol': {
            'scope': 'Independent-domain directional local counterfactual teacher bank.',
            'candidate_policy': 'For every unordered raw-track pair with temporal overlap, use the final common frame and audit both directions with the frozen directional planner. Keep every executable direction; no GT or utility value participates in candidate construction or filtering.',
            'teacher': 'GT is used only after candidate construction to compute dense local association and local IDTP counterfactual targets.',
            'global_trackeval_calls': 0,
            'p15_locked_labels_read': 0,
            'p15_locked_trackeval_calls': 0,
            'p15_remaining_locked_rows_untouched': 156,
            'minimum_gt_iou': args.iou_threshold,
        },
        'inputs': {
            'baseline': str(baseline_path),
            'baseline_sha256': sha256(baseline_path),
            'gt': str(gt_path),
            'gt_sha256': sha256(gt_path),
        },
        'dataset': {
            'sequence': args.sequence,
            'tracker_rows': len(records),
            'track_ids': len({int(record['raw_tid']) for record in records}),
            'ordered_candidate_directions': len(candidate_audit),
            'accepted_events': len(event_labels),
            'changed_rows_labeled': len(changed_rows),
            'gt_matched_tracker_rows': int(gt_summary['matched_tracker_rows']),
            'gt_match_rate': float(
                gt_summary['matched_tracker_rows'] / max(1, gt_summary['tracker_rows'])
            ),
        },
        'full_idtp_delta_norm': {
            'positive_events': positive,
            'negative_events': negative,
            'zero_events': zero,
            'positive_fraction': float(positive / max(1, len(event_labels))),
            'mean': float(event_labels.full_idtp_delta_norm.mean()),
            'minimum': float(event_labels.full_idtp_delta_norm.min()),
            'maximum': float(event_labels.full_idtp_delta_norm.max()),
        },
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
