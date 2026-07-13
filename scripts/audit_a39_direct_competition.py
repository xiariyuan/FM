#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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


def is_bridge_mode(r):
    max_gap=af(r.get('max_gap_dist'))
    return int(
        ai(r.get('planned_rows'))>=30 and
        ai(r.get('selected_fragment_count'))>=2 and
        ai(r.get('high_reid_fragment_count'))>=1 and
        ai(r.get('bridge_fragment_count'))>=1 and
        ai(r.get('gap_row_count'))<=10 and
        (max_gap==0 or max_gap<=0.12) and
        ai(r.get('skipped_collision_rows'))==0
    )


def is_direct_strict(r):
    return int(
        ai(r.get('planned_rows'))>=30 and
        ai(r.get('selected_fragment_count'))==1 and
        ai(r.get('high_reid_fragment_count'))==1 and
        ai(r.get('bridge_fragment_count'))==0 and
        ai(r.get('gap_row_count'))==0 and
        ai(r.get('skipped_collision_rows'))==0 and
        af(r.get('max_high_sim_to_anchor'))>=0.80 and
        af(r.get('max_high_sim_to_post'))>=0.90
    )


def eval_rule(rows, name, fn):
    tp=fp=tn=fnn=0; fps=[]; acc=[]
    for r in rows:
        pred=int(fn(r)); lab=ai(r.get('label_safe_to_rewrite'))
        if pred and lab:
            tp+=1; acc.append(r)
        elif pred and not lab:
            fp+=1; fps.append(r); acc.append(r)
        elif not pred and not lab:
            tn+=1
        else:
            fnn+=1
    return {'rule_name':name,'tp':tp,'fp':fp,'tn':tn,'fn':fnn,'precision':tp/(tp+fp) if tp+fp else 0,'recall':tp/(tp+fnn) if tp+fnn else 0,'accepted_count':tp+fp,'safe_count':tp+fnn,'accepted_anchors':'|'.join(x['anchor_id'] for x in acc),'fp_anchors':'|'.join(x['anchor_id'] for x in fps)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stress-enriched', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_04c_candidate_expansion_false_topN_and_rule_stress/path_transaction_examples_stress_enriched.csv')
    ap.add_argument('--stress', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_04c_candidate_expansion_false_topN_and_rule_stress/path_builder_summary.csv')
    ap.add_argument('--out-dir', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_04d_rule_mode_union_v2_broader_stress_and_combined')
    ap.add_argument('--direct-margin', type=float, default=0.05)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows=read_csv(Path(args.stress_enriched)) or read_csv(Path(args.stress))
    # If high sim features are missing, set zeros.
    for r in rows:
        for k in ['max_high_sim_to_anchor','max_high_sim_to_post','max_high_sim_to_pre','max_high_rows']:
            r.setdefault(k,'0')
        r['is_direct_like']=int(ai(r.get('planned_rows'))>=30 and ai(r.get('selected_fragment_count'))==1 and ai(r.get('high_reid_fragment_count'))==1 and ai(r.get('bridge_fragment_count'))==0 and ai(r.get('gap_row_count'))==0 and ai(r.get('skipped_collision_rows'))==0)
    direct=[r for r in rows if ai(r.get('is_direct_like'))]
    by_pre=defaultdict(list); by_post=defaultdict(list)
    for r in direct:
        by_pre[(r.get('tunnel_id'),r.get('pre_track'))].append(r)
        by_post[(r.get('tunnel_id'),r.get('post_track'))].append(r)
    enriched=[]
    for r in rows:
        rec=dict(r)
        if ai(rec.get('is_direct_like')):
            gp=sorted(by_pre[(rec.get('tunnel_id'),rec.get('pre_track'))], key=lambda x: af(x.get('max_high_sim_to_anchor')), reverse=True)
            rec['same_pre_direct_candidate_count']=len(gp)
            rank=[x['anchor_id'] for x in gp].index(rec['anchor_id'])+1
            rec['same_pre_rank_by_high_sim']=rank
            best=af(gp[0].get('max_high_sim_to_anchor')) if gp else 0
            second=af(gp[1].get('max_high_sim_to_anchor')) if len(gp)>1 else 0
            current=af(rec.get('max_high_sim_to_anchor'))
            rec['same_pre_best_high_sim']=best
            rec['same_pre_second_high_sim']=second
            rec['direct_margin_to_second']=current-second if rank==1 else current-best
            gpost=sorted(by_post[(rec.get('tunnel_id'),rec.get('post_track'))], key=lambda x: af(x.get('max_high_sim_to_anchor')), reverse=True)
            rec['same_post_direct_candidate_count']=len(gpost)
            rec['same_post_rank_by_high_sim']=[x['anchor_id'] for x in gpost].index(rec['anchor_id'])+1
        else:
            rec['same_pre_direct_candidate_count']=0
            rec['same_pre_rank_by_high_sim']=0
            rec['same_pre_best_high_sim']=0
            rec['same_pre_second_high_sim']=0
            rec['direct_margin_to_second']=0
            rec['same_post_direct_candidate_count']=0
            rec['same_post_rank_by_high_sim']=0
        rec['rule_bridge_mode_v2']=is_bridge_mode(rec)
        rec['rule_direct_strict']=is_direct_strict(rec)
        rec['rule_direct_competition_v3']=int(is_direct_strict(rec) and ai(rec.get('same_pre_rank_by_high_sim'))==1 and af(rec.get('direct_margin_to_second'))>=args.direct_margin)
        rec['rule_mode_union_v3']=int(rec['rule_bridge_mode_v2'] or rec['rule_direct_competition_v3'])
        enriched.append(rec)
    write_csv(out/'direct_candidate_competition.csv', [r for r in enriched if ai(r.get('is_direct_like'))])
    write_csv(out/'path_transaction_examples_stress_v2.csv', enriched)
    report=[
        eval_rule(enriched,'rule_bridge_mode_v2',lambda r: ai(r.get('rule_bridge_mode_v2'))),
        eval_rule(enriched,'rule_direct_strict',lambda r: ai(r.get('rule_direct_strict'))),
        eval_rule(enriched,'rule_direct_competition_v3',lambda r: ai(r.get('rule_direct_competition_v3'))),
        eval_rule(enriched,'rule_mode_union_v3',lambda r: ai(r.get('rule_mode_union_v3'))),
    ]
    write_csv(out/'rule_stress_report_v2.csv', report)
    fps=[]; safe=[]
    for r in enriched:
        if ai(r.get('label_safe_to_rewrite')):
            safe.append(r)
        for rule in ['rule_bridge_mode_v2','rule_direct_strict','rule_direct_competition_v3','rule_mode_union_v3']:
            if ai(r.get(rule)) and not ai(r.get('label_safe_to_rewrite')):
                q=dict(r); q['rule_name']=rule; fps.append(q)
    write_csv(out/'false_positive_cases_v2.csv', fps)
    write_csv(out/'safe_path_cases_v2.csv', safe)
    # group summary for direct candidates
    group_rows=[]
    for key,g in by_pre.items():
        gs=sorted(g, key=lambda x: af(x.get('max_high_sim_to_anchor')), reverse=True)
        group_rows.append({
            'tunnel_id':key[0], 'pre_track':key[1], 'direct_candidate_count':len(gs),
            'anchors':'|'.join(x['anchor_id'] for x in gs),
            'labels':'|'.join(x.get('label_safe_to_rewrite','') for x in gs),
            'high_sims':'|'.join(f"{af(x.get('max_high_sim_to_anchor')):.3f}" for x in gs),
            'best_anchor':gs[0]['anchor_id'], 'best_high_sim':af(gs[0].get('max_high_sim_to_anchor')),
            'second_high_sim':af(gs[1].get('max_high_sim_to_anchor')) if len(gs)>1 else 0,
            'best_margin_to_second':af(gs[0].get('max_high_sim_to_anchor'))-(af(gs[1].get('max_high_sim_to_anchor')) if len(gs)>1 else 0),
        })
    write_csv(out/'direct_competition_summary.csv', group_rows)
    summary={'path_examples':len(enriched),'direct_like_examples':len(direct),'safe_path_examples':sum(ai(r.get('label_safe_to_rewrite')) for r in enriched),'unsafe_path_examples':len(enriched)-sum(ai(r.get('label_safe_to_rewrite')) for r in enriched),'false_positive_cases_v2':len(fps),'rule_report_v2':report}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    md=['# A39_04d Direct Competition Audit','', '## Summary','', '```json',json.dumps(summary,indent=2,sort_keys=True),'```','','## Rule report v2','','| rule | tp | fp | tn | fn | precision | recall | accepted | safe |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in report:
        md.append(f"| {r['rule_name']} | {r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['accepted_count']} | {r['safe_count']} |")
    md+=['','## Direct-like candidates','','| anchor | label | gt_same | planned | wrong | high_anchor | high_post | same_pre_count | rank | margin |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted([x for x in enriched if ai(x.get('is_direct_like'))], key=lambda x: (x.get('tunnel_id'), x.get('pre_track'), -af(x.get('max_high_sim_to_anchor')))):
        md.append(f"| {r['anchor_id']} | {r.get('label_safe_to_rewrite')} | {r.get('gt_same')} | {r.get('planned_rows')} | {r.get('diag_wrong_rows')} | {af(r.get('max_high_sim_to_anchor')):.3f} | {af(r.get('max_high_sim_to_post')):.3f} | {r.get('same_pre_direct_candidate_count')} | {r.get('same_pre_rank_by_high_sim')} | {af(r.get('direct_margin_to_second')):.3f} |")
    if fps:
        md+=['','## False positives v2','','| rule | anchor | reason |','|---|---|---|']
        for r in fps:
            md.append(f"| {r.get('rule_name')} | {r.get('anchor_id')} | {r.get('reject_reason')} |")
    else:
        md+=['','No false positives under v2/v3 rules.']
    (out/'direct_competition_decision.md').write_text('\n'.join(md)+'\n')
    print('\n'.join(md))

if __name__=='__main__':
    main()
