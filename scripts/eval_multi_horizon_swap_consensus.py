from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from eval_assa_swap_merge_fusion import (
    SEQS,
    build_merge_map,
    read_track_rows,
    write_rows,
)

BASE_POLICIES = {
    'h30_safe5': {
        'horizon': 'h30', 'duration': 30,
        'score_col': 'assa_swap_h30_risk_hgb_l0p5',
        'min_spacing': 30, 'aggregation': 'q75',
    },
    'h60_safe6': {
        'horizon': 'h60', 'duration': 60,
        'score_col': 'assa_swap_h60_risk_et_l1p0',
        'min_spacing': 20, 'aggregation': 'q75',
    },
    'perm_aggressive15': {
        'horizon': 'perm', 'duration': None,
        'score_col': 'assa_swap_perm_risk_et_l1p0',
        'min_spacing': 30, 'aggregation': 'max',
    },
    'perm_safe15': {
        'horizon': 'perm', 'duration': None,
        'score_col': 'assa_swap_perm_risk_et_l0p5',
        'min_spacing': 30, 'aggregation': 'q75',
    },
}


def select_events(scores: pd.DataFrame, diagnostics: pd.DataFrame, name: str, policy: dict):
    h = policy['horizon']; col = policy['score_col']
    spacing = policy['min_spacing']; aggregation = policy['aggregation']
    selected = []
    for seq in SEQS:
        row = diagnostics[
            (diagnostics.horizon == h) &
            (diagnostics.score_col == col) &
            (diagnostics.min_spacing == spacing) &
            (diagnostics.aggregation == aggregation) &
            (diagnostics.heldout_seq == seq)
        ]
        if len(row) != 1:
            raise RuntimeError(f'missing diagnostic row: {name}, {seq}, rows={len(row)}')
        cutoff = float(row.iloc[0].learned_cutoff)
        ranked = scores[scores.seq == seq].sort_values(
            [col, 'candidate_ioa', 'frame'], ascending=[False, False, True]
        )
        path = []; last = {}
        for r in ranked.itertuples(index=False):
            a = int(r.track_a); b = int(r.track_b); frame = int(r.frame)
            if abs(frame - last.get(a, -10**9)) < spacing:
                continue
            if abs(frame - last.get(b, -10**9)) < spacing:
                continue
            path.append(r); last[a] = frame; last[b] = frame
        for r in path:
            if float(getattr(r, col + '_seqpct')) < cutoff:
                continue
            d = r._asdict()
            d.update({
                'base_policy': name,
                'selected_horizon': h,
                'duration': policy['duration'],
                'selection_score': float(getattr(r, col)),
                'selection_seqpct': float(getattr(r, col + '_seqpct')),
                'learned_cutoff': cutoff,
                'selection_delta_proxy': float(getattr(r, f'assa_swap_delta_{h}_proxy')),
            })
            selected.append(d)
    return selected


def event_key(e):
    a = int(e['track_a']); b = int(e['track_b'])
    return (str(e['seq']), int(e['frame']), min(a, b), max(a, b))


def make_consensus(base: dict[str, list[dict]], perm_source: str, minimum_support: int, action: str):
    sources = ['h30_safe5', 'h60_safe6', perm_source]
    by_key = defaultdict(dict)
    for source in sources:
        for e in base[source]:
            by_key[event_key(e)][source] = e
    out = []
    for key, support in sorted(by_key.items()):
        if len(support) < minimum_support or perm_source not in support:
            continue
        if action == 'permanent':
            chosen = dict(support[perm_source]); duration = None
        elif action == 'shortest':
            if 'h30_safe5' in support:
                chosen = dict(support['h30_safe5']); duration = 30
            elif 'h60_safe6' in support:
                chosen = dict(support['h60_safe6']); duration = 60
            else:
                continue
        elif action == 'longest_finite':
            if 'h60_safe6' in support:
                chosen = dict(support['h60_safe6']); duration = 60
            elif 'h30_safe5' in support:
                chosen = dict(support['h30_safe5']); duration = 30
            else:
                continue
        else:
            raise ValueError(action)
        chosen['duration'] = duration
        chosen['consensus_support'] = len(support)
        chosen['support_policies'] = '|'.join(sorted(support))
        chosen['consensus_action'] = action
        chosen['consensus_perm_source'] = perm_source
        out.append(chosen)
    return out


def transpose_values(perm: dict[int, int], left: int, right: int):
    keys = set(perm) | {left, right}
    old = {k: perm.get(k, k) for k in keys}
    for key, value in old.items():
        if value == left:
            perm[key] = right
        elif value == right:
            perm[key] = left
        else:
            perm[key] = value


def simulate_merge_then_transactions(rows, events, merge_map):
    starts = defaultdict(list); ends = defaultdict(list)
    for e in events:
        start = int(e['frame']); starts[start].append(e)
        duration = e.get('duration')
        if duration is not None and not pd.isna(duration):
            ends[start + int(duration) + 1].append(e)

    frames = defaultdict(list)
    for p in rows:
        frames[int(float(p[0]))].append(p)

    perm = {}; active = {}; active_roots = set()
    accepted = []; rejected = []; output = []; changed = 0
    changed_by_event = defaultdict(int)

    def current(root):
        return perm.get(root, root)

    for frame in sorted(frames):
        # Revert finite transactions before processing the first frame outside the interval.
        for e in sorted(ends.get(frame, []), key=lambda x: event_key(x)):
            key = event_key(e)
            state = active.pop(key, None)
            if state is None:
                continue
            transpose_values(perm, state['left_label'], state['right_label'])
            active_roots.discard(state['root_a']); active_roots.discard(state['root_b'])

        for e in sorted(starts.get(frame, []), key=lambda x: event_key(x)):
            a = int(e['track_a']); b = int(e['track_b'])
            root_a = merge_map.get(a, a); root_b = merge_map.get(b, b)
            if root_a == root_b:
                rejected.append({**e, 'reason': 'same_merge_root', 'root_a': root_a, 'root_b': root_b})
                continue
            if root_a in active_roots or root_b in active_roots:
                rejected.append({**e, 'reason': 'overlapping_active_transaction', 'root_a': root_a, 'root_b': root_b})
                continue
            left = current(root_a); right = current(root_b)
            if left == right:
                rejected.append({**e, 'reason': 'same_current_label', 'root_a': root_a, 'root_b': root_b})
                continue
            transpose_values(perm, left, right)
            state = {'root_a': root_a, 'root_b': root_b, 'left_label': left, 'right_label': right}
            if e.get('duration') is not None and not pd.isna(e.get('duration')):
                active[event_key(e)] = state
                active_roots.add(root_a); active_roots.add(root_b)
            accepted.append({**e, **state})

        emitted = []
        for p in frames[frame]:
            q = list(p); tid = int(float(q[1])); root = merge_map.get(tid, tid)
            new = current(root); q[1] = str(new); emitted.append(q)
            changed += int(new != tid)
            for e in accepted:
                start = int(e['frame']); duration = e.get('duration')
                end = None if duration is None or pd.isna(duration) else start + int(duration)
                if frame < start or (end is not None and frame > end):
                    continue
                if root in {int(e['root_a']), int(e['root_b'])}:
                    changed_by_event[event_key(e)] += int(new != root)
        ids = [int(float(q[1])) for q in emitted]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f'duplicate IDs at frame {frame}')
        output.extend(emitted)

    ineffective = [event_key(e) for e in accepted if changed_by_event.get(event_key(e), 0) == 0]
    if ineffective:
        raise RuntimeError(f'accepted events changed no merged labels: {ineffective}')
    output.sort(key=lambda p: (int(float(p[0])), int(float(p[1]))))
    return output, accepted, rejected, changed, changed_by_event


def evaluate(pdir: Path, name: str):
    command = [
        sys.executable, 'scripts/eval_motstyle_trackeval.py',
        '--benchmark-name', 'MOT20', '--split-to-eval', 'train',
        '--gt-root', 'datasets/MOT20/train', '--results-dir', str(pdir / 'track_results'),
        '--tracker-name', name, '--work-dir', str(pdir / 'eval_work'), '--seqs', *SEQS,
    ]
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (pdir / 'eval.log').write_text(proc.stdout)
    detailed = pdir / 'eval_work' / 'eval' / name / 'pedestrian_detailed.csv'
    result = {'returncode': proc.returncode}
    if detailed.exists():
        rows = list(csv.DictReader(detailed.open()))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--merge-links', required=True)
    parser.add_argument('--scores', required=True)
    parser.add_argument('--diagnostics', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.scores); diagnostics = pd.read_csv(args.diagnostics)
    links = pd.read_csv(args.merge_links)
    base = {name: select_events(scores, diagnostics, name, policy) for name, policy in BASE_POLICIES.items()}

    policies = {
        'h30_safe5': base['h30_safe5'],
        'h60_safe6': base['h60_safe6'],
        'consensus3_perm_aggressive': make_consensus(base, 'perm_aggressive15', 3, 'permanent'),
        'consensus2_perm_aggressive': make_consensus(base, 'perm_aggressive15', 2, 'permanent'),
        'consensus2_shortest_aggressive': make_consensus(base, 'perm_aggressive15', 2, 'shortest'),
        'consensus2_longest_aggressive': make_consensus(base, 'perm_aggressive15', 2, 'longest_finite'),
        'consensus2_perm_safe': make_consensus(base, 'perm_safe15', 2, 'permanent'),
    }

    summaries = []
    for policy_name, events in policies.items():
        pd.DataFrame(events).to_csv(out / f'{policy_name}_selected_events.csv', index=False)
        pdir = out / policy_name; all_accepted = []; all_rejected = []; by_seq = []
        for seq in SEQS:
            rows, _, spans = read_track_rows(Path(args.source_root) / f'{seq}.txt')
            merge_map, selected_links = build_merge_map(links[links.seq == seq], spans)
            seq_events = [e for e in events if e['seq'] == seq]
            output, accepted, rejected, changed, effects = simulate_merge_then_transactions(
                rows, seq_events, merge_map,
            )
            write_rows(pdir / 'track_results' / f'{seq}.txt', output)
            all_accepted.extend(accepted); all_rejected.extend(rejected)
            by_seq.append({
                'seq': seq, 'requested': len(seq_events), 'accepted': len(accepted),
                'rejected': len(rejected), 'changed_rows': changed,
                'event_effect_rows': int(sum(effects.values())),
                'selected_merge_links': len(selected_links),
            })
        result = evaluate(pdir, policy_name)
        summary = {
            'policy_name': policy_name, 'requested': len(events),
            'accepted': len(all_accepted), 'rejected': len(all_rejected),
            'by_seq': by_seq, 'eval': result,
        }
        (pdir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        pd.DataFrame(all_accepted).to_csv(pdir / 'accepted_events.csv', index=False)
        if all_rejected:
            pd.DataFrame(all_rejected).to_csv(pdir / 'rejected_events.csv', index=False)
        summaries.append(summary)
        print(json.dumps({
            'policy': policy_name, 'requested': len(events), 'accepted': len(all_accepted),
            'rejected': len(all_rejected), 'combined': result.get('COMBINED'),
            'avg': result.get('simple_avg_HOTA'), 'm02': result.get('MOT20-02'),
            'm05': result.get('MOT20-05'),
        }, indent=2), flush=True)
    (out / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')

if __name__ == '__main__':
    main()
