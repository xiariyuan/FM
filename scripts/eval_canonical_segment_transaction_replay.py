from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from eval_assa_swap_merge_fusion import (
    build_merge_map,
    read_track_rows,
    transpose_current_labels,
    write_rows,
)

SEQS = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_events(scores: pd.DataFrame, score_col: str, cluster_radius: int, spacing: int):
    z = scores.copy()
    z['u'] = z[['track_a', 'track_b']].min(axis=1).astype(int)
    z['v'] = z[['track_a', 'track_b']].max(axis=1).astype(int)
    z = z[z.u != z.v].sort_values(['u', 'v', 'boundary_frame', score_col], ascending=[True, True, True, False])

    clustered = []
    for (u, v), g in z.groupby(['u', 'v'], sort=True):
        current = []
        last_frame = None
        for r in g.to_dict('records'):
            fr = int(r['boundary_frame'])
            if current and fr - int(last_frame) > cluster_radius:
                best = max(current, key=lambda x: (float(x[score_col]), -int(x['boundary_frame'])))
                best = dict(best); best['cluster_size'] = len(current)
                best['cluster_min_frame'] = min(int(x['boundary_frame']) for x in current)
                best['cluster_max_frame'] = max(int(x['boundary_frame']) for x in current)
                best['u'] = int(u); best['v'] = int(v)
                clustered.append(best)
                current = []
            current.append(r); last_frame = fr
        if current:
            best = max(current, key=lambda x: (float(x[score_col]), -int(x['boundary_frame'])))
            best = dict(best); best['cluster_size'] = len(current)
            best['cluster_min_frame'] = min(int(x['boundary_frame']) for x in current)
            best['cluster_max_frame'] = max(int(x['boundary_frame']) for x in current)
            best['u'] = int(u); best['v'] = int(v)
            clustered.append(best)

    clustered.sort(key=lambda x: (-float(x[score_col]), int(x['boundary_frame']), int(x['u']), int(x['v'])))
    selected = []
    last_by_track = {}
    for e in clustered:
        fr = int(e['boundary_frame']); u = int(e['u']); v = int(e['v'])
        if abs(fr - last_by_track.get(u, -10**9)) < spacing:
            continue
        if abs(fr - last_by_track.get(v, -10**9)) < spacing:
            continue
        selected.append(e); last_by_track[u] = fr; last_by_track[v] = fr
    return clustered, selected


def reconstruct_best_with_provenance(source_file: Path, merge_links: pd.DataFrame,
                                     aggressive_events: pd.DataFrame):
    rows, by_frame, spans = read_track_rows(source_file)
    merge_map, selected_links = build_merge_map(merge_links, spans)
    events_by_frame = defaultdict(list)
    for e in aggressive_events.to_dict('records'):
        events_by_frame[int(e['frame'])].append(e)

    frames = defaultdict(list)
    for p in rows:
        frames[int(float(p[0]))].append(p)
    root_perm = {}
    records = []
    for fr in sorted(frames):
        for e in sorted(events_by_frame.get(fr, []), key=lambda x: (int(x['track_a']), int(x['track_b']))):
            a = int(e['track_a']); b = int(e['track_b'])
            left_root = merge_map.get(a, a); right_root = merge_map.get(b, b)
            left = root_perm.get(left_root, left_root); right = root_perm.get(right_root, right_root)
            if left == right:
                raise RuntimeError(f'aggressive baseline event collapsed at {fr}: {a},{b}')
            transpose_current_labels(root_perm, left, right)
        for source_index, p in enumerate(frames[fr]):
            raw_tid = int(float(p[1])); root = merge_map.get(raw_tid, raw_tid)
            label = int(root_perm.get(root, root))
            q = list(p); q[1] = str(label)
            records.append({'frame': fr, 'raw_tid': raw_tid, 'label': label,
                            'parts': q, 'source_index': source_index})
    records.sort(key=lambda r: (r['frame'], r['label']))
    return records, len(selected_links)


def records_to_rows(records):
    out = []
    for r in records:
        q = list(r['parts']); q[1] = str(int(r['label'])); out.append(q)
    out.sort(key=lambda p: (int(float(p[0])), int(float(p[1]))))
    return out


def event_end(event, mode: str, max_frame: int, next_by_pair: dict[tuple[int, int, int], int]):
    start = int(event['boundary_frame'])
    if mode == 'perm':
        return max_frame
    if mode == 'h30':
        return min(max_frame, start + 29)
    if mode == 'h60':
        return min(max_frame, start + 59)
    if mode == 'next_pair':
        return min(max_frame, next_by_pair.get((int(event['u']), int(event['v']), start), max_frame) - 1)
    raise ValueError(mode)


def apply_local_transactions(baseline_records, events, mode: str):
    records = [dict(r, parts=list(r['parts'])) for r in baseline_records]
    frame_indices = defaultdict(list); raw_frame_indices = defaultdict(list)
    max_frame = 0
    for i, r in enumerate(records):
        fr = int(r['frame']); raw = int(r['raw_tid']); max_frame = max(max_frame, fr)
        frame_indices[fr].append(i); raw_frame_indices[(raw, fr)].append(i)

    by_pair = defaultdict(list)
    for e in events:
        by_pair[(int(e['u']), int(e['v']))].append(int(e['boundary_frame']))
    next_by_pair = {}
    for (u, v), frames in by_pair.items():
        frames = sorted(set(frames))
        for i, fr in enumerate(frames[:-1]):
            next_by_pair[(u, v, fr)] = frames[i + 1]

    accepted = []; rejected = []; touched = set()
    event_rows_changed = []
    for rank, e in enumerate(events, 1):
        u = int(e['u']); v = int(e['v']); start = int(e['boundary_frame'])
        end = event_end(e, mode, max_frame, next_by_pair)
        if end < start:
            rejected.append({**e, 'rank': rank, 'reason': 'empty_interval', 'end_frame': end})
            continue
        targets = []
        overlap = False
        for fr in range(start, end + 1):
            for raw in (u, v):
                for idx in raw_frame_indices.get((raw, fr), []):
                    if idx in touched:
                        overlap = True
                    targets.append(idx)
        if overlap:
            rejected.append({**e, 'rank': rank, 'reason': 'overlapping_prior_transaction', 'end_frame': end})
            continue
        if not targets:
            rejected.append({**e, 'rank': rank, 'reason': 'no_rows_in_interval', 'end_frame': end})
            continue

        snapshot = {idx: int(records[idx]['label']) for idx in targets}
        changed = 0
        # Per frame: swap the two segment labels if both tracks exist. If only one is
        # present, use the other track's event-start anchor label. Collision audit below
        # decides whether this one-sided continuation is safe.
        start_u = [int(records[i]['label']) for i in raw_frame_indices.get((u, start), [])]
        start_v = [int(records[i]['label']) for i in raw_frame_indices.get((v, start), [])]
        if not start_u or not start_v:
            rejected.append({**e, 'rank': rank, 'reason': 'missing_pair_at_start', 'end_frame': end})
            continue
        anchor_u, anchor_v = start_u[0], start_v[0]
        if anchor_u == anchor_v:
            rejected.append({**e, 'rank': rank, 'reason': 'same_anchor_label', 'end_frame': end})
            continue

        for fr in range(start, end + 1):
            ui = raw_frame_indices.get((u, fr), []); vi = raw_frame_indices.get((v, fr), [])
            if ui and vi:
                ul = [int(records[i]['label']) for i in ui]
                vl = [int(records[i]['label']) for i in vi]
                if len(set(ul)) != 1 or len(set(vl)) != 1:
                    for idx, old in snapshot.items(): records[idx]['label'] = old
                    rejected.append({**e, 'rank': rank, 'reason': 'nonunique_raw_label_state', 'end_frame': end})
                    break
                for i in ui: records[i]['label'] = vl[0]
                for i in vi: records[i]['label'] = ul[0]
            elif ui:
                for i in ui: records[i]['label'] = anchor_v
            elif vi:
                for i in vi: records[i]['label'] = anchor_u
        else:
            bad_frame = None
            for fr in range(start, end + 1):
                labels = [int(records[i]['label']) for i in frame_indices.get(fr, [])]
                if len(labels) != len(set(labels)):
                    bad_frame = fr; break
            if bad_frame is not None:
                for idx, old in snapshot.items(): records[idx]['label'] = old
                rejected.append({**e, 'rank': rank, 'reason': 'duplicate_id_after_local_transaction',
                                 'collision_frame': bad_frame, 'end_frame': end})
                continue
            changed = sum(int(records[idx]['label']) != old for idx, old in snapshot.items())
            if changed == 0:
                for idx, old in snapshot.items(): records[idx]['label'] = old
                rejected.append({**e, 'rank': rank, 'reason': 'no_effect', 'end_frame': end})
                continue
            touched.update(targets)
            accepted.append({**e, 'rank': rank, 'start_frame': start, 'end_frame': end,
                             'anchor_u': anchor_u, 'anchor_v': anchor_v, 'changed_rows': changed})
            event_rows_changed.append(changed)
            continue
        # A loop break reaches here after state restoration.
        continue

    # Global duplicate check.
    for fr, idxs in frame_indices.items():
        labels = [int(records[i]['label']) for i in idxs]
        if len(labels) != len(set(labels)):
            raise RuntimeError(f'duplicate IDs remain at frame {fr}')
    return records, accepted, rejected, sum(event_rows_changed)


def evaluate(pdir: Path, name: str):
    cmd = [
        sys.executable, 'scripts/eval_motstyle_trackeval.py',
        '--benchmark-name', 'MOT20', '--split-to-eval', 'train',
        '--gt-root', 'datasets/MOT20/train', '--results-dir', str(pdir / 'track_results'),
        '--tracker-name', name, '--work-dir', str(pdir / 'eval_work'), '--seqs', *SEQS,
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (pdir / 'eval.log').write_text(proc.stdout)
    detail = pdir / 'eval_work' / 'eval' / name / 'pedestrian_detailed.csv'
    result = {'returncode': proc.returncode}
    if detail.exists():
        rows = list(csv.DictReader(detail.open()))
        for seq in SEQS + ['COMBINED']:
            r = next((x for x in rows if x['seq'] == seq), None)
            if r:
                result[seq] = {
                    'HOTA': float(r['HOTA___AUC']) * 100,
                    'DetA': float(r['DetA___AUC']) * 100,
                    'AssA': float(r['AssA___AUC']) * 100,
                    'IDF1': float(r['IDF1']) * 100 if float(r['IDF1']) < 2 else float(r['IDF1']),
                    'IDSW': int(float(r['IDSW'])),
                }
        result['simple_avg_HOTA'] = sum(result[s]['HOTA'] for s in SEQS) / 4
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-root', required=True)
    ap.add_argument('--best-root', required=True)
    ap.add_argument('--merge-links', required=True)
    ap.add_argument('--aggressive-events', required=True)
    ap.add_argument('--pair-scores', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--score-col', default='oof_pair_reciprocal_ensemble')
    ap.add_argument('--cluster-radius', type=int, default=3)
    ap.add_argument('--spacing', type=int, default=30)
    ap.add_argument('--budgets', nargs='*', type=int, default=[25, 50, 100])
    ap.add_argument('--modes', nargs='*', default=['perm', 'h30', 'h60', 'next_pair'])
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    links = pd.read_csv(args.merge_links)
    aggressive = pd.read_csv(args.aggressive_events)
    pair_scores = pd.read_csv(args.pair_scores)
    if 'seq' in pair_scores.columns:
        pair_scores = pair_scores[pair_scores.seq == 'MOT20-02'].copy()
    clustered, ranked = canonical_events(pair_scores, args.score_col, args.cluster_radius, args.spacing)
    pd.DataFrame(clustered).to_csv(out / 'canonical_clustered_events.csv', index=False)
    pd.DataFrame(ranked).to_csv(out / 'canonical_ranked_events.csv', index=False)

    baseline_by_seq = {}; reconstruction = []
    for seq in SEQS:
        records, selected_links = reconstruct_best_with_provenance(
            Path(args.source_root) / f'{seq}.txt',
            links[links.seq == seq], aggressive[aggressive.seq == seq],
        )
        baseline_by_seq[seq] = records
        tmp = out / '_baseline_reconstruction' / f'{seq}.txt'
        write_rows(tmp, records_to_rows(records))
        ref = Path(args.best_root) / f'{seq}.txt'
        exact = sha256(tmp) == sha256(ref)
        reconstruction.append({'seq': seq, 'selected_links': selected_links,
                               'exact_best_match': exact, 'sha': sha256(tmp), 'ref_sha': sha256(ref)})
    if not all(x['exact_best_match'] for x in reconstruction):
        raise RuntimeError(f'baseline reconstruction mismatch: {reconstruction}')
    (out / 'baseline_reconstruction.json').write_text(json.dumps(reconstruction, indent=2) + '\n')

    summaries = []
    for budget in args.budgets:
        selected = ranked[:budget]
        for mode in args.modes:
            name = f'b{budget}_{mode}'
            pdir = out / name
            modified, accepted, rejected, changed = apply_local_transactions(
                baseline_by_seq['MOT20-02'], selected, mode,
            )
            for seq in SEQS:
                target = pdir / 'track_results' / f'{seq}.txt'; target.parent.mkdir(parents=True, exist_ok=True)
                if seq == 'MOT20-02':
                    write_rows(target, records_to_rows(modified))
                else:
                    shutil.copy2(Path(args.best_root) / f'{seq}.txt', target)
            pd.DataFrame(selected).to_csv(pdir / 'requested_events.csv', index=False)
            pd.DataFrame(accepted).to_csv(pdir / 'accepted_events.csv', index=False)
            if rejected:
                pd.DataFrame(rejected).to_csv(pdir / 'rejected_events.csv', index=False)
            eval_result = evaluate(pdir, name)
            summary = {
                'name': name, 'budget': budget, 'mode': mode,
                'requested': len(selected), 'accepted': len(accepted), 'rejected': len(rejected),
                'changed_rows': changed,
                'score_col': args.score_col, 'cluster_radius': args.cluster_radius,
                'spacing': args.spacing, 'eval': eval_result,
            }
            (pdir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            summaries.append(summary)
            print(json.dumps({
                'name': name, 'requested': len(selected), 'accepted': len(accepted),
                'rejected': len(rejected), 'changed_rows': changed,
                'M02': eval_result.get('MOT20-02'),
                'COMBINED': eval_result.get('COMBINED'),
                'mean': eval_result.get('simple_avg_HOTA'),
            }, indent=2), flush=True)
    (out / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')

if __name__ == '__main__':
    main()
