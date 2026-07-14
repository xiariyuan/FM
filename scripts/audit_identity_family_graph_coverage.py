from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

SEQS = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']


class UnionFind:
    def __init__(self, nodes):
        self.parent = {x: x for x in nodes}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def load_track_frames(root: Path, seq: str):
    frames = defaultdict(set)
    with (root / f'{seq}.txt').open() as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 6:
                continue
            frames[int(float(p[1]))].add(int(float(p[0])))
    return dict(frames)


def component_stats(nodes, edges, weights, dominant_tid):
    uf = UnionFind(nodes)
    for a, b in edges:
        if a in uf.parent and b in uf.parent:
            uf.union(a, b)
    comps = defaultdict(list)
    for n in nodes:
        comps[uf.find(n)].append(n)
    if dominant_tid not in uf.parent:
        reached = set()
    else:
        root = uf.find(dominant_tid)
        reached = set(comps[root])
    return {
        'components': len(comps),
        'largest_component_tracks': max((len(x) for x in comps.values()), default=0),
        'dominant_component_tracks': len(reached),
        'dominant_component_rows': int(sum(weights.get(t, 0) for t in reached)),
        'connected_all': int(len(reached) == len(nodes) and len(nodes) > 0),
        'reached': reached,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matches', required=True)
    ap.add_argument('--track-root', required=True)
    ap.add_argument('--candidates', required=True)
    ap.add_argument('--selected-links', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    matches = pd.read_csv(args.matches, usecols=['seq', 'frame', 'track_id', 'gt_id'])
    candidates = pd.read_csv(args.candidates, usecols=[
        'seq', 'track_a', 'track_b', 'gap', 'same_gt', 'assa_merge_positive',
        'assa_merge_delta_proxy', 'endpoint_reid_aligned', 'appearance_max',
    ])
    selected = pd.read_csv(args.selected_links, usecols=['seq', 'track_a', 'track_b', 'origin'])

    family_rows = []
    edge_rows = []
    for seq in SEQS:
        m = matches[matches.seq == seq].copy()
        c = candidates[candidates.seq == seq].copy()
        s = selected[selected.seq == seq].copy()
        track_frames = load_track_frames(Path(args.track_root), seq)

        pair_counts = m.groupby(['gt_id', 'track_id']).size().rename('rows').reset_index()
        track_totals = pair_counts.groupby('track_id').rows.sum().to_dict()
        idx = pair_counts.groupby('track_id').rows.idxmax()
        dominant = pair_counts.loc[idx, ['track_id', 'gt_id', 'rows']].set_index('track_id')
        dom_gt = dominant.gt_id.to_dict()
        dom_rows = dominant.rows.to_dict()
        purity = {t: dom_rows[t] / max(1, track_totals[t]) for t in track_totals}

        cand_edges = defaultdict(list)
        utility_edges = defaultdict(list)
        reid_edges = defaultdict(list)
        for r in c.itertuples(index=False):
            a, b = int(r.track_a), int(r.track_b)
            shared = set(pair_counts[pair_counts.track_id == a].gt_id) & set(pair_counts[pair_counts.track_id == b].gt_id)
            for gt in shared:
                cand_edges[int(gt)].append((a, b))
                if int(r.assa_merge_positive) == 1:
                    utility_edges[int(gt)].append((a, b))
                if int(r.endpoint_reid_aligned) == 1 and float(r.appearance_max) >= 0.65:
                    reid_edges[int(gt)].append((a, b))
            edge_rows.append({
                'seq': seq, 'track_a': a, 'track_b': b, 'gap': int(r.gap),
                'same_gt': int(r.same_gt), 'assa_merge_positive': int(r.assa_merge_positive),
                'assa_merge_delta_proxy': float(r.assa_merge_delta_proxy),
                'endpoint_reid_aligned': int(r.endpoint_reid_aligned),
                'appearance_max': float(r.appearance_max),
                'dominant_gt_a': int(dom_gt.get(a, -1)), 'dominant_gt_b': int(dom_gt.get(b, -1)),
                'purity_a': float(purity.get(a, 0)), 'purity_b': float(purity.get(b, 0)),
            })

        selected_edges = defaultdict(list)
        for r in s.itertuples(index=False):
            a, b = int(r.track_a), int(r.track_b)
            shared = set(pair_counts[pair_counts.track_id == a].gt_id) & set(pair_counts[pair_counts.track_id == b].gt_id)
            for gt in shared:
                selected_edges[int(gt)].append((a, b))

        for gt, g in pair_counts.groupby('gt_id'):
            weights = {int(r.track_id): int(r.rows) for r in g.itertuples(index=False)}
            tids = sorted(weights)
            if len(tids) < 2:
                continue
            dominant_tid = max(tids, key=lambda t: weights[t])
            matched_rows = sum(weights.values())
            dominant_rows = weights[dominant_tid]
            debt_rows = matched_rows - dominant_rows
            impure_rows = sum(weights[t] for t in tids if dom_gt.get(t) != gt or purity.get(t, 0) < 0.9)
            pure_tids = [t for t in tids if dom_gt.get(t) == gt and purity.get(t, 0) >= 0.9]
            missing_track_rows = sum(weights[t] for t in tids if t not in track_frames)

            all_stats = component_stats(tids, cand_edges[int(gt)], weights, dominant_tid)
            util_stats = component_stats(tids, utility_edges[int(gt)], weights, dominant_tid)
            reid_stats = component_stats(tids, reid_edges[int(gt)], weights, dominant_tid)
            sel_stats = component_stats(tids, selected_edges[int(gt)], weights, dominant_tid)

            candidate_recovered_debt = max(0, all_stats['dominant_component_rows'] - dominant_rows)
            utility_recovered_debt = max(0, util_stats['dominant_component_rows'] - dominant_rows)
            selected_recovered_debt = max(0, sel_stats['dominant_component_rows'] - dominant_rows)

            # Whole-track merges cannot safely combine tracks with any shared detection frame.
            incompatible_pairs = 0
            family_pairs = 0
            for i, a in enumerate(tids):
                for b in tids[i + 1:]:
                    family_pairs += 1
                    if track_frames.get(a, set()) & track_frames.get(b, set()):
                        incompatible_pairs += 1

            if candidate_recovered_debt < debt_rows:
                bottleneck = 'candidate_generation'
            elif utility_recovered_debt < debt_rows:
                bottleneck = 'utility_model_or_representation'
            elif selected_recovered_debt < debt_rows:
                bottleneck = 'selection_budget'
            else:
                bottleneck = 'covered'
            if impure_rows > 0 or incompatible_pairs > 0:
                bottleneck = 'track_segmentation_required' if selected_recovered_debt < debt_rows else bottleneck

            family_rows.append({
                'seq': seq, 'gt_id': int(gt), 'matched_rows': matched_rows,
                'tracker_ids': len(tids), 'dominant_track_id': dominant_tid,
                'dominant_rows': dominant_rows, 'debt_rows': debt_rows,
                'impure_or_mixed_rows': int(impure_rows),
                'pure_tracker_ids_p90': len(pure_tids),
                'family_pairs': family_pairs, 'incompatible_detection_pairs': incompatible_pairs,
                'candidate_edges': len(cand_edges[int(gt)]),
                'candidate_components': all_stats['components'],
                'candidate_recovered_debt_rows': candidate_recovered_debt,
                'candidate_debt_recall': candidate_recovered_debt / max(1, debt_rows),
                'reid065_edges': len(reid_edges[int(gt)]),
                'reid065_recovered_debt_rows': max(0, reid_stats['dominant_component_rows'] - dominant_rows),
                'utility_positive_edges': len(utility_edges[int(gt)]),
                'utility_recovered_debt_rows': utility_recovered_debt,
                'utility_debt_recall': utility_recovered_debt / max(1, debt_rows),
                'selected_edges': len(selected_edges[int(gt)]),
                'selected_recovered_debt_rows': selected_recovered_debt,
                'selected_debt_recall': selected_recovered_debt / max(1, debt_rows),
                'candidate_connected_all': all_stats['connected_all'],
                'utility_connected_all': util_stats['connected_all'],
                'selected_connected_all': sel_stats['connected_all'],
                'missing_track_rows': missing_track_rows,
                'bottleneck': bottleneck,
            })

    fam = pd.DataFrame(family_rows)
    edges = pd.DataFrame(edge_rows)
    fam.to_csv(out / 'family_graph_coverage.csv', index=False)
    edges.to_csv(out / 'candidate_edge_gt_audit.csv', index=False)

    summaries = {}
    for scope, d in {
        'all_fragmented': fam,
        'debt_ge100': fam[fam.debt_rows >= 100],
        'top50pct_debt': fam[fam.debt_rows >= fam.debt_rows.quantile(.5)],
    }.items():
        total_debt = int(d.debt_rows.sum())
        summaries[scope] = {
            'families': int(len(d)),
            'debt_rows': total_debt,
            'impure_or_mixed_rows': int(d.impure_or_mixed_rows.sum()),
            'candidate_recovered_debt_rows': int(d.candidate_recovered_debt_rows.sum()),
            'candidate_debt_recall': float(d.candidate_recovered_debt_rows.sum() / max(1, total_debt)),
            'utility_recovered_debt_rows': int(d.utility_recovered_debt_rows.sum()),
            'utility_debt_recall': float(d.utility_recovered_debt_rows.sum() / max(1, total_debt)),
            'selected_recovered_debt_rows': int(d.selected_recovered_debt_rows.sum()),
            'selected_debt_recall': float(d.selected_recovered_debt_rows.sum() / max(1, total_debt)),
            'candidate_connected_all_families': int(d.candidate_connected_all.sum()),
            'utility_connected_all_families': int(d.utility_connected_all.sum()),
            'selected_connected_all_families': int(d.selected_connected_all.sum()),
            'bottleneck_counts': {str(k): int(v) for k, v in d.bottleneck.value_counts().items()},
        }
    summaries['by_seq_debt_ge100'] = {}
    for seq, d in fam[fam.debt_rows >= 100].groupby('seq'):
        total = int(d.debt_rows.sum())
        summaries['by_seq_debt_ge100'][seq] = {
            'families': int(len(d)), 'debt_rows': total,
            'candidate_recall': float(d.candidate_recovered_debt_rows.sum() / max(1, total)),
            'utility_recall': float(d.utility_recovered_debt_rows.sum() / max(1, total)),
            'selected_recall': float(d.selected_recovered_debt_rows.sum() / max(1, total)),
            'impure_row_fraction': float(d.impure_or_mixed_rows.sum() / max(1, d.matched_rows.sum())),
            'bottlenecks': {str(k): int(v) for k, v in d.bottleneck.value_counts().items()},
        }
    (out / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    print(json.dumps(summaries, indent=2))
    print('\nTop uncovered high-debt families:')
    cols = ['seq','gt_id','debt_rows','tracker_ids','impure_or_mixed_rows','candidate_debt_recall','utility_debt_recall','selected_debt_recall','incompatible_detection_pairs','bottleneck']
    print(fam.sort_values(['debt_rows','candidate_debt_recall'], ascending=[False,True])[cols].head(40).to_string(index=False))

if __name__ == '__main__':
    main()
