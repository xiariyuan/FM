from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

SEQS = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']
POLICIES = {
    'conservative4': {
        'horizon': 'perm',
        'score_col': 'assa_swap_perm_risk_et_l0p25',
        'min_spacing': 30,
        'aggregation': 'max',
    },
    'aggressive15': {
        'horizon': 'perm',
        'score_col': 'assa_swap_perm_risk_et_l1p0',
        'min_spacing': 30,
        'aggregation': 'max',
    },
}


def read_track_rows(path: Path):
    rows = []
    by_frame = defaultdict(list)
    spans = {}
    with path.open() as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 6:
                continue
            fr = int(float(p[0])); tid = int(float(p[1]))
            rows.append(p); by_frame[fr].append(tid)
            spans.setdefault(tid, [fr, fr])
            spans[tid][0] = min(spans[tid][0], fr)
            spans[tid][1] = max(spans[tid][1], fr)
    return rows, dict(by_frame), spans


def build_merge_map(links: pd.DataFrame, spans: dict[int, list[int]]):
    parent = {}
    used_source = set(); used_target = set(); selected = []

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    records = links.to_dict('records')
    records.sort(key=lambda r: (-float(r['fusion_score']), int(float(r['gap']))))
    for e in records:
        a = int(float(e['track_a'])); b = int(float(e['track_b']))
        if a not in spans or b not in spans or spans[a][1] >= spans[b][0]:
            continue
        if a in used_source or b in used_target:
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        parent[rb] = ra
        used_source.add(a); used_target.add(b); selected.append(e)
    return {tid: find(tid) for tid in spans}, selected


def select_swap_events(scores: pd.DataFrame, diagnostics: pd.DataFrame, policy: dict):
    col = policy['score_col']; spacing = policy['min_spacing']; agg = policy['aggregation']
    selected = []
    for seq in SEQS:
        z = diagnostics[
            (diagnostics.horizon == policy['horizon']) &
            (diagnostics.score_col == col) &
            (diagnostics.min_spacing == spacing) &
            (diagnostics.aggregation == agg) &
            (diagnostics.heldout_seq == seq)
        ]
        if len(z) != 1:
            raise RuntimeError(f'missing diagnostic row for {seq} {policy}')
        cutoff = float(z.iloc[0].learned_cutoff)
        g = scores[scores.seq == seq].sort_values(
            [col, 'candidate_ioa', 'frame'], ascending=[False, False, True]
        )
        path = []; last = {}
        for r in g.itertuples(index=False):
            a = int(r.track_a); b = int(r.track_b); fr = int(r.frame)
            if abs(fr - last.get(a, -10**9)) < spacing or abs(fr - last.get(b, -10**9)) < spacing:
                continue
            path.append(r); last[a] = fr; last[b] = fr
        for r in path:
            if float(getattr(r, col + '_seqpct')) >= cutoff:
                d = r._asdict(); d['learned_cutoff'] = cutoff
                selected.append(d)
    return selected


def transpose_current_labels(perm: dict[int, int], left: int, right: int):
    """Compose the current label map with a transposition on emitted labels."""
    keys = set(perm) | {left, right}
    old = {k: perm.get(k, k) for k in keys}
    for k, value in old.items():
        if value == left:
            perm[k] = right
        elif value == right:
            perm[k] = left
        else:
            perm[k] = value


def simulate_suffix_variant(rows, by_frame, events, merge_map, variant):
    """Apply time-ordered suffix swaps.

    swap_only:          S_f(t)
    merge_then_swap:    S_f(M(t))
    swap_then_merge:    M(S_f(t))
    """
    events_by_frame = defaultdict(list)
    for event in events:
        events_by_frame[int(event['frame'])].append(event)

    raw_perm = {}
    root_perm = {}
    accepted = []; rejected = []
    output = []; changed = 0
    event_effect_rows = defaultdict(int)

    def raw_current(tid):
        return raw_perm.get(tid, tid)

    def root_current(root):
        return root_perm.get(root, root)

    def output_label(tid):
        if variant == 'swap_only':
            return raw_current(tid)
        if variant == 'merge_then_swap':
            return root_current(merge_map.get(tid, tid))
        if variant == 'swap_then_merge':
            swapped = raw_current(tid)
            return merge_map.get(swapped, swapped)
        raise ValueError(variant)

    def future_collision(start_frame):
        for fr in sorted(f for f in by_frame if f >= start_frame):
            labels = [output_label(tid) for tid in by_frame[fr]]
            if len(labels) != len(set(labels)):
                return fr
        return None

    frames = defaultdict(list)
    for p in rows:
        frames[int(float(p[0]))].append(p)

    for fr in sorted(frames):
        for event in sorted(events_by_frame.get(fr, []), key=lambda x: (int(x['track_a']), int(x['track_b']))):
            a = int(event['track_a']); b = int(event['track_b'])
            raw_snapshot = dict(raw_perm); root_snapshot = dict(root_perm)
            if variant in {'swap_only', 'swap_then_merge'}:
                left = raw_current(a); right = raw_current(b)
                if left == right:
                    rejected.append({**event, 'reason': 'same_current_raw_label', 'label_a': left, 'label_b': right})
                    continue
                transpose_current_labels(raw_perm, left, right)
            else:
                left = root_current(merge_map.get(a, a)); right = root_current(merge_map.get(b, b))
                if left == right:
                    rejected.append({**event, 'reason': 'same_current_merge_root', 'label_a': left, 'label_b': right})
                    continue
                transpose_current_labels(root_perm, left, right)

            collision_frame = future_collision(fr)
            if collision_frame is not None:
                raw_perm.clear(); raw_perm.update(raw_snapshot)
                root_perm.clear(); root_perm.update(root_snapshot)
                rejected.append({
                    **event, 'reason': 'duplicate_id_after_suffix_composition',
                    'collision_frame': collision_frame, 'label_a': left, 'label_b': right,
                })
                continue
            accepted.append({**event, 'label_a_before': left, 'label_b_before': right})

        emitted = []
        for src in frames[fr]:
            q = list(src); tid = int(float(q[1])); new = int(output_label(tid))
            changed += int(new != tid); q[1] = str(new); emitted.append(q)
            for event in accepted:
                if int(event['frame']) <= fr and tid in {int(event['track_a']), int(event['track_b'])} and new != tid:
                    key = (str(event['seq']), int(event['frame']), int(event['track_a']), int(event['track_b']))
                    event_effect_rows[key] += 1
        ids = [int(float(q[1])) for q in emitted]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f'duplicate IDs remain in {variant} at frame {fr}')
        output.extend(emitted)

    # A suffix transaction must change at least one row after its event.
    ineffective = []
    for event in accepted:
        key = (str(event['seq']), int(event['frame']), int(event['track_a']), int(event['track_b']))
        if event_effect_rows.get(key, 0) == 0:
            ineffective.append(key)
    if ineffective:
        raise RuntimeError(f'accepted suffix events had no row-level effect: {ineffective[:10]}')

    output.sort(key=lambda p: (int(float(p[0])), int(float(p[1]))))
    return output, accepted, rejected, changed, event_effect_rows


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for p in rows:
            f.write(','.join(p) + '\n')


def file_sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


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
    ap.add_argument('--merge-links', required=True)
    ap.add_argument('--merge-reference-root', required=True)
    ap.add_argument('--scores', required=True)
    ap.add_argument('--diagnostics', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--policies', nargs='*', default=list(POLICIES))
    ap.add_argument('--variants', nargs='*', default=['swap_only', 'merge_then_swap', 'swap_then_merge'])
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.scores); diagnostics = pd.read_csv(args.diagnostics)
    links = pd.read_csv(args.merge_links); summaries = []

    # Verify the merge map exactly reconstructs the stored best result; no redundant TrackEval.
    merge_stats = []
    for seq in SEQS:
        rows, _, spans = read_track_rows(Path(args.source_root) / f'{seq}.txt')
        merge_map, selected_links = build_merge_map(links[links.seq == seq], spans)
        reconstructed = []
        for p in rows:
            q = list(p); tid = int(float(q[1])); q[1] = str(merge_map.get(tid, tid)); reconstructed.append(q)
        reconstructed.sort(key=lambda p: (int(float(p[0])), int(float(p[1]))))
        tmp = out / '_merge_reconstruction' / f'{seq}.txt'; write_rows(tmp, reconstructed)
        ref = Path(args.merge_reference_root) / f'{seq}.txt'
        merge_stats.append({
            'seq': seq, 'selected_links': len(selected_links),
            'exact_reference_match': file_sha256(tmp) == file_sha256(ref),
        })
    if not all(x['exact_reference_match'] for x in merge_stats):
        raise RuntimeError(f'merge reconstruction mismatch: {merge_stats}')
    merge_summary_path = Path(args.merge_reference_root).parent / 'summary.json'
    merge_reference_summary = json.loads(merge_summary_path.read_text())
    baseline = {
        'variant': 'merge_only_reference', 'by_seq': merge_stats,
        'eval': merge_reference_summary['eval'],
    }
    (out / 'merge_only_reference.json').write_text(json.dumps(baseline, indent=2) + '\n')
    summaries.append(baseline)

    for policy_name in args.policies:
        policy = POLICIES[policy_name]
        events = select_swap_events(scores, diagnostics, policy)
        pd.DataFrame(events).to_csv(out / f'{policy_name}_selected_events.csv', index=False)
        for variant in args.variants:
            pdir = out / f'{policy_name}_{variant}'
            all_accepted = []; all_rejected = []; by_seq_stats = []
            for seq in SEQS:
                rows, by_frame, spans = read_track_rows(Path(args.source_root) / f'{seq}.txt')
                merge_map, selected_links = build_merge_map(links[links.seq == seq], spans)
                seq_events = [e for e in events if e['seq'] == seq]
                output, accepted, rejected, changed, effects = simulate_suffix_variant(
                    rows, by_frame, seq_events, merge_map, variant,
                )
                write_rows(pdir / 'track_results' / f'{seq}.txt', output)
                all_accepted += accepted; all_rejected += rejected
                by_seq_stats.append({
                    'seq': seq, 'requested_swaps': len(seq_events),
                    'accepted_swaps': len(accepted), 'rejected_swaps': len(rejected),
                    'selected_merge_links': len(selected_links), 'changed_rows': changed,
                    'event_effect_rows': int(sum(effects.values())),
                    'assa_proxy_delta_requested': float(sum(float(e['assa_swap_delta_perm_proxy']) for e in seq_events)),
                    'assa_proxy_delta_accepted': float(sum(float(e['assa_swap_delta_perm_proxy']) for e in accepted)),
                })
            eval_result = evaluate(pdir, f'{policy_name}_{variant}')
            summary = {
                'policy_name': policy_name, 'policy': policy, 'variant': variant,
                'requested_swaps': len(events), 'accepted_swaps': len(all_accepted),
                'rejected_swaps': len(all_rejected), 'by_seq': by_seq_stats, 'eval': eval_result,
            }
            (pdir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            pd.DataFrame(all_accepted).to_csv(pdir / 'accepted_swaps.csv', index=False)
            if all_rejected:
                pd.DataFrame(all_rejected).to_csv(pdir / 'rejected_swaps.csv', index=False)
            summaries.append(summary)
            print(json.dumps({
                'policy': policy_name, 'variant': variant,
                'requested': len(events), 'accepted': len(all_accepted), 'rejected': len(all_rejected),
                'changed_rows': sum(x['changed_rows'] for x in by_seq_stats),
                'combined': eval_result.get('COMBINED'), 'm02': eval_result.get('MOT20-02'),
                'avg': eval_result.get('simple_avg_HOTA'),
            }, indent=2), flush=True)
    (out / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')

if __name__ == '__main__':
    main()
