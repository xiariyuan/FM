#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path


def as_int(v, default=0):
    try: return int(float(v))
    except Exception: return default


def as_float(v, default=0.0):
    try: return float(v)
    except Exception: return default


def true_tid_from_event(row):
    rank = as_int(row.get('rank_true_before'), -1)
    if rank < 0: return -1
    tids = [x for x in str(row.get('candidate_tids', '')).split('|') if x != '']
    if rank >= len(tids): return -1
    return as_int(tids[rank], -1)


def event_ok(row, policy):
    contains = as_int(row.get('contains_true_before_topk'), 0) == 1
    target = true_tid_from_event(row)
    chosen = as_int(row.get('chosen_tid'), -1)
    if not contains or target < 0 or chosen < 0 or target == chosen:
        return False
    bad = as_int(row.get('is_bad_commit_before'), 0) == 1
    low = as_int(row.get('low_margin_005'), 0) == 1
    idsw = as_int(row.get('is_gt_idsw'), 0) == 1
    sw = as_int(row.get('is_track_switch'), 0) == 1
    gap = as_int(row.get('true_last_seen_gap'), -1)
    if policy == 'strict_safe_reclaim':
        return bad and low and 0 <= gap <= 6
    if policy == 'oracle_safe_reclaim':
        return bad and 0 <= gap <= 10
    if policy == 'idsw_safe_reclaim':
        return idsw and 0 <= gap <= 10
    if policy == 'track_switch_safe_reclaim':
        return sw and 0 <= gap <= 10
    if policy == 'low005_wrong_reclaim':
        return bad and low
    raise ValueError(policy)


def read_mot(path: Path):
    rows = []
    by_frame_id = defaultdict(list)
    with path.open('r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line: continue
            p = line.split(',')
            if len(p) < 6: continue
            fr = as_int(p[0], -1); tid = as_int(p[1], -1)
            rows.append({'idx': idx, 'frame': fr, 'track_id': tid, 'parts': p})
            by_frame_id[(fr, tid)].append(idx)
    return rows, by_frame_id


def build_segments(events, max_event_gap, min_segment_events):
    grouped = defaultdict(list)
    for r in events:
        chosen = as_int(r.get('chosen_tid'), -1)
        target = true_tid_from_event(r)
        det_gt = as_int(r.get('det_gt'), -1)
        fr = as_int(r.get('frame'), -1)
        if fr < 0 or chosen < 0 or target < 0: continue
        grouped[(chosen, target, det_gt)].append(r)
    segments = []
    for key, rr in grouped.items():
        rr.sort(key=lambda x: as_int(x.get('frame'), -1))
        cur = []
        prev = None
        for r in rr:
            fr = as_int(r.get('frame'), -1)
            if prev is None or fr - prev <= max_event_gap:
                cur.append(r)
            else:
                if len(cur) >= min_segment_events:
                    segments.append((key, cur))
                cur = [r]
            prev = fr
        if len(cur) >= min_segment_events:
            segments.append((key, cur))
    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track-file', required=True)
    ap.add_argument('--events-csv', required=True)
    ap.add_argument('--out-file', required=True)
    ap.add_argument('--segments-csv', required=True)
    ap.add_argument('--summary-json', required=True)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--max-event-gap', type=int, default=3)
    ap.add_argument('--min-segment-events', type=int, default=2)
    ap.add_argument('--fill-full-span', action='store_true')
    ap.add_argument('--skip-collisions', action='store_true')
    args = ap.parse_args()

    rows, by_frame_id = read_mot(Path(args.track_file))
    occupied = defaultdict(set)
    for r in rows:
        occupied[r['frame']].add(r['track_id'])

    raw_events = []
    with open(args.events_csv, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if event_ok(r, args.policy):
                raw_events.append(r)
    segments = build_segments(raw_events, args.max_event_gap, args.min_segment_events)

    stats = defaultdict(int)
    stats['raw_eligible_events'] = len(raw_events)
    stats['candidate_segments'] = len(segments)
    seg_rows = []
    planned = []
    used_rows = set()
    for sid, (key, evs) in enumerate(segments):
        chosen, target, det_gt = key
        frames = sorted({as_int(e.get('frame'), -1) for e in evs})
        if args.fill_full_span and frames:
            candidate_frames = list(range(min(frames), max(frames) + 1))
        else:
            candidate_frames = frames
        # Only frames where chosen has an output row can be reassigned.
        apply_frames = [fr for fr in candidate_frames if (fr, chosen) in by_frame_id]
        collision_frames = [fr for fr in apply_frames if target in occupied.get(fr, set())]
        safe = len(collision_frames) == 0
        applied = False
        if apply_frames and (safe or not args.skip_collisions):
            # avoid modifying the same output row twice
            idxs = []
            for fr in apply_frames:
                idxs.extend(by_frame_id.get((fr, chosen), []))
            idxs = [i for i in idxs if i not in used_rows]
            if idxs:
                planned.extend((i, chosen, target, sid) for i in idxs)
                used_rows.update(idxs)
                applied = True
        stats['segments_safe'] += int(safe)
        stats['segments_collision'] += int(not safe)
        stats['segments_applied'] += int(applied)
        stats['events_in_candidate_segments'] += len(evs)
        stats['events_in_applied_segments'] += len(evs) if applied else 0
        stats['frames_in_applied_segments'] += len(apply_frames) if applied else 0
        for e in evs:
            if applied:
                stats['applied_event_is_gt_idsw'] += as_int(e.get('is_gt_idsw'), 0)
                stats['applied_event_is_track_switch'] += as_int(e.get('is_track_switch'), 0)
                stats['applied_event_is_bad_commit'] += as_int(e.get('is_bad_commit_before'), 0)
        seg_rows.append({
            'segment_id': sid, 'chosen_tid': chosen, 'target_tid': target, 'det_gt': det_gt,
            'start_frame': min(frames) if frames else -1, 'end_frame': max(frames) if frames else -1,
            'event_count': len(evs), 'apply_frame_count': len(apply_frames),
            'safe_no_collision': int(safe), 'collision_frame_count': len(collision_frames),
            'applied': int(applied), 'policy': args.policy,
            'frames': '|'.join(map(str, frames[:100])),
            'collision_frames': '|'.join(map(str, collision_frames[:100])),
        })
    for idx, chosen, target, sid in planned:
        rows[idx]['parts'][1] = str(target)
    out = Path(args.out_file); out.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (as_int(r['parts'][0]), as_int(r['parts'][1]), as_float(r['parts'][2]), as_float(r['parts'][3])))
    with out.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(','.join(r['parts']) + '\n')
    stats['applied_rows'] = len(planned)
    stats['policy'] = args.policy
    stats['max_event_gap'] = args.max_event_gap
    stats['min_segment_events'] = args.min_segment_events
    stats['fill_full_span'] = bool(args.fill_full_span)
    stats['skip_collisions'] = bool(args.skip_collisions)
    Path(args.summary_json).write_text(json.dumps(dict(stats), indent=2, sort_keys=True) + '\n', encoding='utf-8')
    with open(args.segments_csv, 'w', newline='', encoding='utf-8') as f:
        fields = list(seg_rows[0].keys()) if seg_rows else ['segment_id']
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(seg_rows)
    print(json.dumps(dict(stats), indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
