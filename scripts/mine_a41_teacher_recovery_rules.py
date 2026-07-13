#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Set, Tuple


def af(v, d=0.0):
    try:
        if v is None or v == '':
            return d
        return float(v)
    except Exception:
        return d


def ai(v, d=0):
    try:
        if v is None or v == '':
            return d
        return int(float(v))
    except Exception:
        return d


def read_csv(path: Path) -> List[dict]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    fields, seen = [], set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    if not fields:
        fields = ['name']
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def key(row: dict) -> Tuple[str, str, str]:
    return (row.get('seq', ''), str(int(float(row.get('track_a', 0) or 0))), str(int(float(row.get('track_b', 0) or 0))))


def key_str(k: Tuple[str, str, str]) -> str:
    return f'{k[0]}:{k[1]}->{k[2]}'


def parse_tags(s: str) -> Set[str]:
    return {x for x in str(s or '').split('|') if x}


def enrich_teacher_only(features: List[dict], a41_keys: Set[Tuple[str, str, str]], p12_keys: Set[Tuple[str, str, str]], p14_keys: Set[Tuple[str, str, str]]) -> List[dict]:
    out = []
    feat_by_key = {key(r): r for r in features}
    teacher_union = p12_keys | p14_keys
    for k in sorted(teacher_union - a41_keys):
        r = dict(feat_by_key.get(k, {}))
        if not r:
            continue
        r['edge_key'] = key_str(k)
        r['teacher_p12'] = int(k in p12_keys)
        r['teacher_p14'] = int(k in p14_keys)
        r['teacher_union'] = 1
        r['a41_aggressive'] = int(k in a41_keys)
        r['teacher_only_subset'] = 'P12_only' if k in p12_keys and k not in p14_keys else ('P14_only' if k in p14_keys and k not in p12_keys else 'P12_P14_only')
        r['label_true'] = int(str(r.get('same_gt')) == '1')
        # no-GT helper fields only; same_gt retained only as diagnostic label.
        stags = parse_tags(r.get('source_debt_tags'))
        ttags = parse_tags(r.get('target_debt_tags'))
        tags = stags | ttags
        r['has_weak_boundary'] = int('weak_end_boundary' in tags or 'weak_start_boundary' in tags)
        r['has_short_tracklet'] = int('short_tracklet' in tags or 'very_short_tracklet' in tags)
        r['has_internal_gaps'] = int('internal_gaps' in tags)
        r['has_low_avg_score'] = int('low_avg_score' in tags)
        r['max_rank'] = max(ai(r.get('out_rank_by_aflink_score'), 999), ai(r.get('in_rank_by_aflink_score'), 999))
        r['min_bidir_margin'] = min(af(r.get('out_margin_to_second_aflink_score')), af(r.get('in_margin_to_second_aflink_score')))
        r['max_unidir_margin'] = max(af(r.get('out_margin_to_second_aflink_score')), af(r.get('in_margin_to_second_aflink_score')))
        r['good_geometry'] = int(ai(r.get('geometry_risk')) <= 1)
        r['good_motion'] = int(ai(r.get('motion_risk')) <= 1)
        out.append(r)
    return out


def eval_rule(rows: List[dict], name: str, pred) -> dict:
    sel = [r for r in rows if pred(r)]
    tp = sum(ai(r.get('label_true')) for r in sel)
    by_subset = defaultdict(lambda: {'selected': 0, 'tp': 0})
    for r in sel:
        b = by_subset[r.get('teacher_only_subset')]
        b['selected'] += 1
        b['tp'] += ai(r.get('label_true'))
    return {
        'rule': name,
        'selected': len(sel),
        'tp': tp,
        'fp': len(sel) - tp,
        'precision': tp / len(sel) if sel else 0.0,
        'teacher_only_recall': tp / max(1, sum(ai(r.get('label_true')) for r in rows)),
        'selected_keys': '|'.join(r.get('edge_key', '') for r in sel),
        'by_subset': json.dumps(by_subset, sort_keys=True),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--a41-links', required=True)
    ap.add_argument('--teacher-p12', required=True)
    ap.add_argument('--teacher-p14', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = read_csv(Path(args.features))
    a41 = read_csv(Path(args.a41_links))
    p12 = read_csv(Path(args.teacher_p12))
    p14 = read_csv(Path(args.teacher_p14))
    a41_keys = {key(r) for r in a41}
    p12_keys = {key(r) for r in p12}
    p14_keys = {key(r) for r in p14}

    teacher_only = enrich_teacher_only(features, a41_keys, p12_keys, p14_keys)
    write_csv(out_dir / 'teacher_only_edges_enriched.csv', teacher_only)

    # General audit tables.
    subset_rows = []
    for subset, rows in sorted(defaultdict(list, { }).items()):
        pass
    by_subset = defaultdict(list)
    for r in teacher_only:
        by_subset[r['teacher_only_subset']].append(r)
    for subset, rows in sorted(by_subset.items()):
        tp = sum(ai(r.get('label_true')) for r in rows)
        c = Counter()
        for r in rows:
            c[f"edge_type:{r.get('edge_type')}"] += 1
            for tag in (str(r.get('source_debt_tags', '')) + '|' + str(r.get('target_debt_tags', ''))).split('|'):
                if tag:
                    c[f'debt:{tag}'] += 1
        subset_rows.append({
            'subset': subset,
            'count': len(rows),
            'tp': tp,
            'precision': tp / len(rows) if rows else 0.0,
            'aflink_mean': sum(af(r.get('aflink_score')) for r in rows) / len(rows) if rows else 0.0,
            'appearance_max_mean': sum(af(r.get('appearance_max')) for r in rows) / len(rows) if rows else 0.0,
            'risk_total_mean': sum(af(r.get('risk_total')) for r in rows) / len(rows) if rows else 0.0,
            'top_counts': json.dumps(c.most_common(20), sort_keys=True),
        })
    write_csv(out_dir / 'teacher_only_subset_summary.csv', subset_rows)

    # Rule scorecard. Conditions are no-GT deployable; precision uses train same_gt only for diagnostics.
    rules = []
    # hand-written policies around observed teacher-only structure.
    rules.append(eval_rule(teacher_only, 'teacher_any_all', lambda r: True))
    rules.append(eval_rule(teacher_only, 'p12_only_all', lambda r: ai(r.get('teacher_p12')) == 1))
    rules.append(eval_rule(teacher_only, 'p14_only_all', lambda r: ai(r.get('teacher_p14')) == 1))
    # Score/risk grids focused on small teacher-only pool.
    for src in ['union', 'p12', 'p14']:
        for min_score in [0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
            for min_app in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
                for max_risk in [2, 3, 4, 5, 7]:
                    for max_rank in [1, 2]:
                        def pred(r, src=src, min_score=min_score, min_app=min_app, max_risk=max_risk, max_rank=max_rank):
                            if src == 'p12' and ai(r.get('teacher_p12')) != 1:
                                return False
                            if src == 'p14' and ai(r.get('teacher_p14')) != 1:
                                return False
                            return (
                                af(r.get('aflink_score')) >= min_score and
                                af(r.get('appearance_max')) >= min_app and
                                ai(r.get('risk_total')) <= max_risk and
                                ai(r.get('max_rank'), 999) <= max_rank and
                                ai(r.get('geometry_risk')) <= 1 and
                                ai(r.get('motion_risk')) <= 2
                            )
                        rules.append(eval_rule(teacher_only, f'{src}_score{min_score:.2f}_app{min_app:.2f}_risk{max_risk}_rank{max_rank}', pred))
    # Type-specific rules.
    for edge_type in ['weak_boundary_recovery', 'fragmented_tracklet_recovery', 'short_gap_continuation', 'long_gap_reappearance']:
        for min_score in [0.15, 0.20, 0.30, 0.40, 0.50]:
            for max_risk in [2, 3, 4, 5]:
                def pred(r, edge_type=edge_type, min_score=min_score, max_risk=max_risk):
                    return (
                        r.get('edge_type') == edge_type and
                        af(r.get('aflink_score')) >= min_score and
                        ai(r.get('risk_total')) <= max_risk and
                        ai(r.get('geometry_risk')) <= 1 and
                        ai(r.get('motion_risk')) <= 2
                    )
                rules.append(eval_rule(teacher_only, f'type_{edge_type}_score{min_score:.2f}_risk{max_risk}', pred))

    # Keep meaningful rules and sort by practical objective.
    rules = [r for r in rules if r['selected'] > 0]
    for r in rules:
        r['objective_precision_first'] = r['tp'] + 0.5 * r['precision'] - 0.25 * r['fp']
        r['objective_tp_first'] = r['tp'] - 0.35 * r['fp']
    rules_sorted = sorted(rules, key=lambda r: (r['objective_tp_first'], r['precision'], r['selected']), reverse=True)
    write_csv(out_dir / 'teacher_only_rule_scorecard.csv', rules_sorted)
    write_csv(out_dir / 'teacher_only_rule_scorecard_precision_first.csv', sorted(rules, key=lambda r: (r['precision'], r['tp'], r['selected']), reverse=True))

    # Pick a few policies for A41_05b. These are no-GT conditions, selected using train diagnostics.
    def find_rule(name):
        for r in rules_sorted:
            if r['rule'] == name:
                return r
        return None
    candidates = []
    # Prefer rules that recover >=3 TP with reasonable FP, then precision-first variants.
    good = [r for r in rules_sorted if r['tp'] >= 3 and r['precision'] >= 0.35]
    precision = [r for r in sorted(rules, key=lambda r: (r['precision'], r['tp'], r['selected']), reverse=True) if r['selected'] >= 2 and r['precision'] >= 0.45]
    for label, rs in [('tp_oriented', good[:5]), ('precision_oriented', precision[:5])]:
        for r in rs:
            candidates.append({k: r[k] for k in ['rule', 'selected', 'tp', 'fp', 'precision', 'selected_keys']})
            candidates[-1]['policy_group'] = label
    # Always include teacher baselines for bounded ablations.
    for name in ['p12_only_all', 'p14_only_all', 'teacher_any_all']:
        r = next((x for x in rules if x['rule'] == name), None)
        if r:
            candidates.append({k: r[k] for k in ['rule', 'selected', 'tp', 'fp', 'precision', 'selected_keys']})
            candidates[-1]['policy_group'] = 'teacher_baseline'

    policy_json = {
        'a41_base': 'aggressive_p60',
        'teacher_only_edges': len(teacher_only),
        'teacher_only_tp': sum(ai(r.get('label_true')) for r in teacher_only),
        'policy_candidates': candidates,
        'decision': 'A41_05a_RULE_MINING_DONE_READY_FOR_HYBRID_EVAL',
        'leakage_note': 'Rules use same_gt only for train diagnostic selection. Deployable policies must use rule predicates or explicit teacher membership and no-GT features only.',
    }
    (out_dir / 'teacher_recovery_policy_candidates.json').write_text(json.dumps(policy_json, indent=2, sort_keys=True) + '\n')
    md = ['# A41_05a Teacher Recovery Rule Mining', '', '```json', json.dumps(policy_json, indent=2, sort_keys=True), '```', '']
    (out_dir / 'decision.md').write_text('\n'.join(md) + '\n')
    print(json.dumps(policy_json, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
