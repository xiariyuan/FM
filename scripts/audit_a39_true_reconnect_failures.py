#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def ai(v, d=0):
    try:
        return int(float(v))
    except Exception:
        return d


def af(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    fields=[]; seen=set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields or ['anchor_id'], extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def safe_div(a,b):
    return float(a)/float(b) if b else 0.0


def classify_failure(anchor, path):
    if ai(path.get('label_safe_to_rewrite')) == 1:
        return 'SAFE_PATH'
    if path.get('run_status','').startswith('timeout'):
        return 'TIMEOUT_OR_TOO_SLOW'
    planned = ai(path.get('planned_rows'))
    selected = ai(path.get('selected_fragment_count'))
    high = ai(path.get('high_reid_fragment_count'))
    bridge = ai(path.get('bridge_fragment_count'))
    gap = ai(path.get('gap_row_count'))
    wrong = ai(path.get('diag_wrong_rows'))
    skip = ai(path.get('skipped_collision_rows'))
    idsw = ai(path.get('IDSW'), 443)
    hota = af(path.get('HOTA'), 68.430)
    idf1 = af(path.get('IDF1'), 74.413)
    if planned <= 0 or selected <= 0:
        return 'NO_PATH_BUILT'
    if high <= 0:
        return 'NO_HIGH_REID_FRAGMENT'
    if wrong > 1:
        return 'WRONG_ROWS'
    if skip > 0:
        return 'COLLISION_SKIP'
    if planned < 10:
        return 'TOO_FEW_ROWS'
    if selected == 1 and high == 1 and bridge == 0:
        # Direct path exists but did not meet safety label. Explain likely direct mode weakness.
        if af(path.get('max_high_sim_to_anchor')) < 0.80:
            return 'DIRECT_ANCHOR_SIM_TOO_WEAK'
        if af(path.get('max_high_sim_to_post')) < 0.90:
            return 'DIRECT_POST_SIM_TOO_WEAK'
        if af(path.get('max_high_sim_to_pre')) < 0.40:
            return 'DIRECT_PRE_SIM_TOO_WEAK'
        return 'DIRECT_METRIC_OR_UNKNOWN_FAIL'
    if selected >= 2 and bridge <= 0:
        return 'BRIDGE_FRAGMENT_MISSING'
    if gap > 10:
        return 'GAP_TOO_LARGE'
    if idsw > 443:
        return 'METRIC_IDSW_WORSE'
    if hota < 68.430 or idf1 < 74.413:
        return 'METRIC_HOTA_IDF1_DROP'
    return 'UNKNOWN_FAIL'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--anchor-examples', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_04_learned_path_gate_dataset/anchor_examples.csv')
    ap.add_argument('--path-examples', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_04d_rule_mode_union_v2_broader_stress_and_combined/path_transaction_examples_stress_v4.csv')
    ap.add_argument('--out-dir', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_04e_safe_path_mining_and_positive_gap_audit')
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    anchors=read_csv(Path(args.anchor_examples))
    paths=read_csv(Path(args.path_examples))
    path_by_anchor={p['anchor_id']:p for p in paths if p.get('anchor_id')}
    true=[a for a in anchors if ai(a.get('label_true_reconnect'))==1]
    rows=[]
    for a in sorted(true, key=lambda r: -af(r.get('sim'))):
        p=path_by_anchor.get(a['anchor_id'], {})
        r={
            'anchor_id': a['anchor_id'],
            'tunnel_id': a.get('tunnel_id'),
            'pre_track': a.get('pre_track'),
            'post_track': a.get('post_track'),
            'pre_gt': a.get('pre_gt'),
            'post_gt': a.get('post_gt'),
            'sim': a.get('sim'),
            'row_rank': a.get('row_rank'),
            'col_rank': a.get('col_rank'),
            'row_margin': a.get('row_margin'),
            'col_margin': a.get('col_margin'),
            'lifecycle_suspension': a.get('feature_lifecycle_suspension'),
            'top11': a.get('feature_top11'),
            'pre_track_pre_rows': a.get('pre_track_pre_rows'),
            'pre_track_core_rows': a.get('pre_track_core_rows'),
            'pre_track_post_rows': a.get('pre_track_post_rows'),
            'post_track_pre_rows': a.get('post_track_pre_rows'),
            'post_track_core_rows': a.get('post_track_core_rows'),
            'post_track_post_rows': a.get('post_track_post_rows'),
            'height_ratio': a.get('height_ratio'),
            'center_delta_norm': a.get('center_delta_norm'),
            'oracle_core_exit_rows': a.get('oracle_core_exit_rows'),
            'path_run_available': int(bool(p)),
            'planned_rows': p.get('planned_rows',''),
            'applied_rows': p.get('applied_rows',''),
            'selected_fragment_count': p.get('selected_fragment_count',''),
            'high_reid_fragment_count': p.get('high_reid_fragment_count',''),
            'bridge_fragment_count': p.get('bridge_fragment_count',''),
            'gap_row_count': p.get('gap_row_count',''),
            'skipped_collision_rows': p.get('skipped_collision_rows',''),
            'diag_same_rows': p.get('diag_same_rows',''),
            'diag_wrong_rows': p.get('diag_wrong_rows',''),
            'max_high_sim_to_anchor': p.get('max_high_sim_to_anchor',''),
            'max_high_sim_to_post': p.get('max_high_sim_to_post',''),
            'max_high_sim_to_pre': p.get('max_high_sim_to_pre',''),
            'same_pre_rank_by_high_sim': p.get('same_pre_rank_by_high_sim',''),
            'direct_margin_to_second': p.get('direct_margin_to_second',''),
            'rule_mode_union_v4': p.get('rule_mode_union_v4',''),
            'label_safe_to_rewrite': p.get('label_safe_to_rewrite','0'),
            'HOTA': p.get('HOTA',''),
            'IDF1': p.get('IDF1',''),
            'IDSW': p.get('IDSW',''),
            'reject_reason': p.get('reject_reason','not_run'),
        }
        r['failure_reason']=classify_failure(a,p)
        rows.append(r)
    write_csv(out/'true_reconnect_failure_audit.csv', rows)
    by_reason=defaultdict(lambda: Counter())
    for r in rows:
        reason=r['failure_reason']
        by_reason[reason]['anchor_count']+=1
        by_reason[reason]['oracle_rows']+=ai(r.get('oracle_core_exit_rows'))
        by_reason[reason]['planned_rows']+=ai(r.get('planned_rows'))
        by_reason[reason]['safe_count']+=ai(r.get('label_safe_to_rewrite'))
    summary=[]
    for reason,c in sorted(by_reason.items(), key=lambda kv: (-kv[1]['anchor_count'], kv[0])):
        summary.append({'failure_reason':reason, **dict(c)})
    write_csv(out/'true_reconnect_failure_summary.csv', summary)
    payload={'true_reconnect_count':len(rows),'safe_count':sum(ai(r.get('label_safe_to_rewrite')) for r in rows),'failure_summary':summary}
    (out/'true_reconnect_failure_summary.json').write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n')
    md=['# A39_04e1 True Reconnect Failure Audit','','## Summary','','```json',json.dumps(payload, indent=2, sort_keys=True),'```','','## True reconnect anchors','','| anchor | sim | rank | life | planned | selected | high | bridge | wrong | skip | label | failure |','|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        md.append(f"| {r['anchor_id']} | {af(r.get('sim')):.3f} | {r.get('row_rank')}/{r.get('col_rank')} | {r.get('lifecycle_suspension')} | {r.get('planned_rows')} | {r.get('selected_fragment_count')} | {r.get('high_reid_fragment_count')} | {r.get('bridge_fragment_count')} | {r.get('diag_wrong_rows')} | {r.get('skipped_collision_rows')} | {r.get('label_safe_to_rewrite')} | {r.get('failure_reason')} |")
    (out/'true_reconnect_failure_summary.md').write_text('\n'.join(md)+'\n')
    print('\n'.join(md))

if __name__=='__main__':
    main()
