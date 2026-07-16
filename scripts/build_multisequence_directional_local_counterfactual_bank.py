from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_generic_directional_local_counterfactual_bank import (
    EVENT_KEYS,
    build_teacher_bank,
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


def priority(row: dict[str, object]) -> float:
    return float(
        float(row['boundary_center_distance_norm'])
        + 0.25 * float(row['boundary_bottom_gap_norm'])
        + 0.10
        * (
            float(row['boundary_log_width_ratio_abs'])
            + float(row['boundary_log_height_ratio_abs'])
        )
        - 0.25 * float(row['boundary_iou'])
    )


def select_events(
    accepted: list[dict[str, object]],
    track_spacing: int,
    max_events: int,
) -> list[dict[str, object]]:
    ranked = []
    for row in accepted:
        item = dict(row)
        item['enumeration_rank'] = int(item['canonical_rank'])
        item['candidate_priority'] = priority(item)
        ranked.append(item)
    ranked.sort(
        key=lambda row: (
            float(row['candidate_priority']),
            int(row['boundary_frame']),
            int(row['u']),
            int(row['v']),
            str(row['transaction_type']),
        )
    )
    selected: list[dict[str, object]] = []
    last_boundary_by_track: dict[int, int] = {}
    for row in ranked:
        boundary = int(row['boundary_frame'])
        u = int(row['u'])
        v = int(row['v'])
        if abs(boundary - last_boundary_by_track.get(u, -10**9)) < track_spacing:
            continue
        if abs(boundary - last_boundary_by_track.get(v, -10**9)) < track_spacing:
            continue
        selected.append(row)
        last_boundary_by_track[u] = boundary
        last_boundary_by_track[v] = boundary
        if max_events > 0 and len(selected) >= max_events:
            break
    selected.sort(
        key=lambda row: (
            int(row['boundary_frame']),
            int(row['u']),
            int(row['v']),
            str(row['transaction_type']),
        )
    )
    for rank, row in enumerate(selected, start=1):
        row['canonical_rank'] = rank
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', action='append', required=True, help='SEQ=PATH')
    parser.add_argument('--gt', action='append', required=True, help='SEQ=PATH')
    parser.add_argument('--track-spacing', type=int, default=30)
    parser.add_argument('--max-events-per-sequence', type=int, default=500)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    baseline_paths = parse_mapping(args.baseline, 'baseline')
    gt_paths = parse_mapping(args.gt, 'gt')
    if set(baseline_paths) != set(gt_paths):
        raise RuntimeError('baseline and GT mappings must cover the same sequences')
    sequences = sorted(baseline_paths)

    candidate_parts = []
    executability_parts = []
    event_parts = []
    changed_parts = []
    match_parts = []
    sequence_reports = []

    for sequence in sequences:
        records = load_baseline_records(baseline_paths[sequence])
        accepted_all, candidate_audit = enumerate_candidates(sequence, records)
        selected = select_events(
            accepted_all,
            args.track_spacing,
            args.max_events_per_sequence,
        )
        selected_lookup = {
            (
                int(row['u']),
                int(row['v']),
                int(row['boundary_frame']),
                str(row['transaction_type']),
            ): row
            for row in selected
        }
        accepted_lookup = {
            (
                int(row['u']),
                int(row['v']),
                int(row['boundary_frame']),
                str(row['transaction_type']),
            ): row
            for row in accepted_all
        }
        for audit_row in candidate_audit:
            key = (
                int(audit_row['u']),
                int(audit_row['v']),
                int(audit_row['boundary_frame']),
                str(audit_row['transaction_type']),
            )
            accepted_row = accepted_lookup.get(key)
            selected_row = selected_lookup.get(key)
            audit_row['enumeration_rank'] = (
                int(accepted_row['canonical_rank']) if accepted_row else 0
            )
            audit_row['candidate_priority'] = (
                priority(accepted_row) if accepted_row else float('nan')
            )
            audit_row['selected_for_teacher'] = int(selected_row is not None)
            audit_row['canonical_rank'] = (
                int(selected_row['canonical_rank']) if selected_row else 0
            )

        event_labels, changed_rows, match_summary = build_teacher_bank(
            sequence,
            records,
            selected,
            gt_paths[sequence],
            args.iou_threshold,
        )
        executability = event_labels[
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

        candidate_parts.append(pd.DataFrame(candidate_audit))
        executability_parts.append(executability)
        event_parts.append(event_labels)
        changed_parts.append(changed_rows)
        match_parts.append(pd.DataFrame([{'seq': sequence, **match_summary}]))
        sequence_reports.append(
            {
                'seq': sequence,
                'tracker_rows': len(records),
                'track_ids': len({int(record['raw_tid']) for record in records}),
                'ordered_candidate_directions': len(candidate_audit),
                'executable_events_before_canonicalization': len(accepted_all),
                'selected_events': len(event_labels),
                'selected_changed_rows': len(changed_rows),
                'positive_events': int((event_labels.full_idtp_delta_norm > 0.0).sum()),
                'negative_events': int((event_labels.full_idtp_delta_norm < 0.0).sum()),
                'zero_events': int((event_labels.full_idtp_delta_norm == 0.0).sum()),
                'positive_fraction': float(
                    (event_labels.full_idtp_delta_norm > 0.0).mean()
                ),
                'gt_match_rate': float(
                    match_summary['matched_tracker_rows']
                    / max(1, match_summary['tracker_rows'])
                ),
                'baseline_sha256': sha256(baseline_paths[sequence]),
                'gt_sha256': sha256(gt_paths[sequence]),
            }
        )

    candidate_frame = pd.concat(candidate_parts, ignore_index=True)
    executability_frame = pd.concat(executability_parts, ignore_index=True)
    event_frame = pd.concat(event_parts, ignore_index=True)
    changed_frame = pd.concat(changed_parts, ignore_index=True)
    match_frame = pd.concat(match_parts, ignore_index=True)
    if event_frame.duplicated(EVENT_KEYS).any():
        raise RuntimeError('duplicate event keys in multisequence bank')
    if len(changed_frame) != int(event_frame.changed_rows.sum()):
        raise RuntimeError('multisequence changed-row cardinality mismatch')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    rounded(candidate_frame).to_csv(out_dir / 'candidate_audit.csv', index=False)
    rounded(executability_frame).to_csv(out_dir / 'executability.csv', index=False)
    rounded(event_frame).to_csv(out_dir / 'event_local_labels.csv', index=False)
    rounded(changed_frame).to_csv(out_dir / 'changed_row_labels.csv', index=False)
    rounded(match_frame).to_csv(
        out_dir / 'sequence_gt_matching_summary.csv', index=False
    )
    rounded(pd.DataFrame(sequence_reports)).to_csv(
        out_dir / 'sequence_summary.csv', index=False
    )

    report = {
        'protocol': {
            'scope': 'Seven-sequence independent-domain directional local teacher bank.',
            'candidate_enumeration': 'For every unordered raw-track pair with temporal overlap, audit both directions at the final common frame using the frozen directional planner.',
            'candidate_priority': 'boundary_center_distance_norm + 0.25*boundary_bottom_gap_norm + 0.10*(absolute log width ratio + absolute log height ratio) - 0.25*boundary_iou; lower is preferred.',
            'canonicalization': {
                'per_track_boundary_spacing_frames': args.track_spacing,
                'maximum_events_per_sequence': args.max_events_per_sequence,
            },
            'leakage_rule': 'Candidate enumeration, priority, and canonicalization use tracker geometry only. GT is read only after the selected event set is frozen.',
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
            'executable_events_before_canonicalization': int(
                sum(row['executable_events_before_canonicalization'] for row in sequence_reports)
            ),
            'selected_events': len(event_frame),
            'changed_rows_labeled': len(changed_frame),
            'positive_events': int((event_frame.full_idtp_delta_norm > 0.0).sum()),
            'negative_events': int((event_frame.full_idtp_delta_norm < 0.0).sum()),
            'zero_events': int((event_frame.full_idtp_delta_norm == 0.0).sum()),
            'positive_fraction': float(
                (event_frame.full_idtp_delta_norm > 0.0).mean()
            ),
        },
        'sequence_reports': sequence_reports,
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
