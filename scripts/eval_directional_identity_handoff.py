from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from eval_canonical_segment_transaction_replay import apply_local_transactions


def _copy_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(record, parts=list(record['parts'])) for record in records]


def build_directional_sequence_index(records: list[dict[str, Any]]) -> dict[str, Any]:
    raw_frame_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
    raw_frames: dict[int, list[int]] = defaultdict(list)
    frame_label_counts: dict[int, Counter[int]] = defaultdict(Counter)
    max_frame = 0
    for index, record in enumerate(records):
        frame = int(record['frame'])
        raw_tid = int(record['raw_tid'])
        label = int(record['label'])
        max_frame = max(max_frame, frame)
        raw_frame_indices[(raw_tid, frame)].append(index)
        raw_frames[raw_tid].append(frame)
        frame_label_counts[frame][label] += 1
    for raw_tid in raw_frames:
        raw_frames[raw_tid] = sorted(set(raw_frames[raw_tid]))
    return {
        'records': records,
        'raw_frame_indices': raw_frame_indices,
        'raw_frames': raw_frames,
        'frame_label_counts': frame_label_counts,
        'max_frame': max_frame,
    }


def plan_directional_handoff(
    sequence_index: dict[str, Any],
    event: dict[str, Any],
    direction: str,
) -> dict[str, Any]:
    if direction not in {'u_to_v', 'v_to_u'}:
        raise ValueError(direction)

    records = sequence_index['records']
    raw_frame_indices = sequence_index['raw_frame_indices']
    raw_frames = sequence_index['raw_frames']
    frame_label_counts = sequence_index['frame_label_counts']
    max_frame = int(sequence_index['max_frame'])

    u = int(event['u'])
    v = int(event['v'])
    boundary = int(event['boundary_frame'])
    donor, receiver = (u, v) if direction == 'u_to_v' else (v, u)

    donor_at_boundary = raw_frame_indices.get((donor, boundary), [])
    receiver_at_boundary = raw_frame_indices.get((receiver, boundary), [])
    base = {
        **event,
        'transaction_type': direction,
        'boundary_frame': boundary,
        'donor_raw_tid': donor,
        'receiver_raw_tid': receiver,
    }
    if not donor_at_boundary or not receiver_at_boundary:
        return {'accepted': False, 'reason': 'missing_pair_at_boundary', 'metadata': base, 'target_indices': []}

    donor_labels = {int(records[index]['label']) for index in donor_at_boundary}
    receiver_labels = {int(records[index]['label']) for index in receiver_at_boundary}
    if len(donor_labels) != 1 or len(receiver_labels) != 1:
        return {'accepted': False, 'reason': 'nonunique_boundary_label_state', 'metadata': base, 'target_indices': []}
    donor_anchor = next(iter(donor_labels))
    receiver_anchor = next(iter(receiver_labels))
    if donor_anchor == receiver_anchor:
        return {'accepted': False, 'reason': 'same_anchor_label', 'metadata': base, 'target_indices': []}

    donor_frame_set = set(raw_frames.get(donor, []))
    receiver_future_frames = [frame for frame in raw_frames.get(receiver, []) if frame > boundary]
    handoff_frame = next((frame for frame in receiver_future_frames if frame not in donor_frame_set), None)
    if handoff_frame is None:
        return {'accepted': False, 'reason': 'no_post_boundary_handoff_frame', 'metadata': base, 'target_indices': []}

    receiver_continuation_frames = [frame for frame in receiver_future_frames if frame >= handoff_frame]
    donor_reappearance = [frame for frame in receiver_continuation_frames if frame in donor_frame_set]
    if donor_reappearance:
        metadata = {**base, 'handoff_frame': handoff_frame, 'first_collision_frame': min(donor_reappearance)}
        return {
            'accepted': False,
            'reason': 'donor_reappears_during_receiver_continuation',
            'metadata': metadata,
            'target_indices': [],
        }

    target_indices: list[int] = []
    for frame in receiver_continuation_frames:
        target_indices.extend(raw_frame_indices.get((receiver, frame), []))
    if not target_indices:
        metadata = {**base, 'handoff_frame': handoff_frame}
        return {'accepted': False, 'reason': 'no_receiver_rows_after_handoff', 'metadata': metadata, 'target_indices': []}

    # Baseline is already globally collision-free. Because only receiver rows change,
    # a new duplicate can arise only if donor_anchor already exists at a target frame
    # outside the receiver rows being relabeled.
    for frame in receiver_continuation_frames:
        receiver_indices = raw_frame_indices.get((receiver, frame), [])
        receiver_old_donor_anchor_count = sum(
            int(records[index]['label']) == donor_anchor for index in receiver_indices
        )
        other_donor_anchor_count = frame_label_counts[frame][donor_anchor] - receiver_old_donor_anchor_count
        if other_donor_anchor_count > 0:
            metadata = {**base, 'handoff_frame': handoff_frame, 'collision_frame': frame}
            return {
                'accepted': False,
                'reason': 'duplicate_id_after_directional_handoff',
                'metadata': metadata,
                'target_indices': [],
            }

    changed = sum(int(records[index]['label']) != donor_anchor for index in target_indices)
    if changed == 0:
        metadata = {**base, 'handoff_frame': handoff_frame}
        return {'accepted': False, 'reason': 'no_effect', 'metadata': metadata, 'target_indices': []}

    metadata = {
        **base,
        'handoff_frame': handoff_frame,
        'end_frame': max_frame,
        'donor_anchor': donor_anchor,
        'receiver_anchor': receiver_anchor,
        'changed_rows': changed,
    }
    return {
        'accepted': True,
        'reason': '',
        'metadata': metadata,
        'target_indices': target_indices,
        'new_label': donor_anchor,
    }


def apply_directional_handoff(
    baseline_records: list[dict[str, Any]],
    event: dict[str, Any],
    direction: str,
    sequence_index: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    index = sequence_index or build_directional_sequence_index(baseline_records)
    plan = plan_directional_handoff(index, event, direction)
    if not plan['accepted']:
        rejected = [{**plan['metadata'], 'reason': plan['reason']}]
        return _copy_records(baseline_records), [], rejected, 0

    records = _copy_records(baseline_records)
    for target_index in plan['target_indices']:
        records[target_index]['label'] = int(plan['new_label'])
    accepted = [dict(plan['metadata'])]
    return records, accepted, [], int(plan['metadata']['changed_rows'])


def apply_typed_transactions(
    baseline_records: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    records = _copy_records(baseline_records)
    accepted_all: list[dict[str, Any]] = []
    rejected_all: list[dict[str, Any]] = []
    changed_total = 0

    for rank, event in enumerate(events, 1):
        transaction_type = str(event.get('transaction_type', 'perm_swap'))
        if transaction_type == 'perm_swap':
            modified, accepted, rejected, changed = apply_local_transactions(records, [event], 'perm')
        elif transaction_type in {'u_to_v', 'v_to_u'}:
            modified, accepted, rejected, changed = apply_directional_handoff(records, event, transaction_type)
        else:
            raise ValueError(transaction_type)
        if accepted:
            records = modified
            accepted_all.extend([{**item, 'typed_rank': rank} for item in accepted])
            changed_total += changed
        else:
            rejected_all.extend([{**item, 'typed_rank': rank} for item in rejected])

    return records, accepted_all, rejected_all, changed_total
