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
        w=csv.DictWriter(f, fieldnames=fields or ['transaction_id'], extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def get_completed_new(path_scan_dir: Path):
    # Prefer the scored aggregate emitted by run_a39_path_missing_feature_scan.py.
    # The per-case feature_summary_nogt.json is intentionally written before
    # direct competition rank/margin enrichment, so it is an intermediate state.
    scored_csv = path_scan_dir / 'path_features_new_scanned.csv'
    if scored_csv.exists():
        rows = read_csv(scored_csv)
        return {r['transaction_id']: r for r in rows if r.get('transaction_id')}
    out={}
    for p in sorted((path_scan_dir/'cases').glob('*/feature_summary_nogt.json')):
        try:
            r=json.load(open(p))
            out[r['transaction_id']]=r
        except Exception:
            pass
    return out


def direct_like_feature(r):
    return int(
        ai(r.get('planned_rows')) >= 30 and
        ai(r.get('selected_fragment_count')) == 1 and
        ai(r.get('high_reid_fragment_count')) == 1 and
        ai(r.get('bridge_fragment_count')) == 0 and
        ai(r.get('gap_row_count')) == 0 and
        ai(r.get('skipped_collision_rows')) == 0
    )


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage-report', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_06d_full_seq_no_gt_candidate_scan_seq02/transaction_stage_report_full_nogt_cached.csv')
    ap.add_argument('--scan-dir', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_06d_full_seq_no_gt_candidate_scan_seq02/A39_06d2c_path_missing_feature_scan')
    ap.add_argument('--out-dir', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_06d_full_seq_no_gt_candidate_scan_seq02/A39_06d2c_path_missing_feature_scan')
    args=ap.parse_args()
    stage=read_csv(Path(args.stage_report))
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    completed_new=get_completed_new(Path(args.scan_dir))

    missing_init=[r for r in stage if r.get('stage')=='feature_missing_cached' and r.get('candidate_family')=='path_index']
    cached_path=[r for r in stage if r.get('candidate_family')=='path' and r.get('stage') in {'accepted','scorer_rejected'}]
    completed_ids=set(completed_new)
    remaining=[r for r in missing_init if r['transaction_id'] not in completed_ids]

    # Feature-aware rows for all built path candidates: cached + completed_new.
    built=[]
    for r in cached_path:
        q=dict(r); q['feature_source']='cached_stage_report'; built.append(q)
    for r in completed_new.values():
        q=dict(r); q['feature_source']='d2c_new_scan'; built.append(q)

    # Remaining priority manifest.
    # Higher priority means more likely to produce a strict direct/bridge candidate.
    priority=[]
    risky_tunnels={'104','22','202','12','173','91','90'}
    for r in remaining:
        sim=af(r.get('sim'))
        rr=ai(r.get('row_rank'),999)
        cr=ai(r.get('col_rank'),999)
        center=af(r.get('center_delta_norm'),999)
        height=af(r.get('height_ratio'),0)
        post_col=af(r.get('post_collision_ratio'),1)
        index_mode=r.get('index_mode','')
        bridge_or_direct=int(index_mode=='bridge_or_direct_index')
        high_sim=int(sim>=0.55)
        mid_sim=int(sim>=0.45)
        low_rank=int(rr<=2 or cr<=2)
        geom_ok=int(center<=0.40 and 0.60<=height<=1.60)
        collision_ok=int(post_col<=0.0)
        risky=int(str(r.get('tunnel_id')) in risky_tunnels)
        # Priority bucket, not a decision rule.
        if bridge_or_direct:
            bucket='P0_bridge_or_direct_remaining'
            score=1000 + 100*high_sim + 10*low_rank + sim
        elif high_sim and geom_ok and collision_ok:
            bucket='P1_high_sim_geom_clean'
            score=800 + 10*low_rank + sim
        elif mid_sim and low_rank and geom_ok:
            bucket='P2_mid_sim_low_rank_geom'
            score=650 + sim
        elif risky and (low_rank or mid_sim):
            bucket='P3_risky_tunnel_candidate'
            score=500 + sim
        else:
            bucket='P4_low_priority_broad_recall'
            score=sim
        q=dict(r)
        q.update({
            'priority_bucket':bucket,
            'priority_score':score,
            'risk_tunnel':risky,
            'high_sim_ge_055':high_sim,
            'mid_sim_ge_045':mid_sim,
            'low_rank_row_or_col_le_2':low_rank,
            'geom_ok_center_height':geom_ok,
            'collision_ok_post_ratio_0':collision_ok,
            'completed_feature':0,
        })
        priority.append(q)
    priority=sorted(priority, key=lambda r:(r['priority_score'], af(r.get('sim'))), reverse=True)
    for i,r in enumerate(priority, start=1):
        r['priority_rank']=i
    write_csv(out/'path_missing_priority_manifest.csv', priority)

    # Bridge/direct missing status.
    bridge_rows=[]
    init_by_id={r['transaction_id']:r for r in missing_init}
    all_bridge_ids=[r['transaction_id'] for r in missing_init if r.get('index_mode')=='bridge_or_direct_index']
    for tid in sorted(all_bridge_ids):
        sr=init_by_id[tid]
        fr=completed_new.get(tid)
        bridge_rows.append({
            'transaction_id':tid,
            'tunnel_id':sr.get('tunnel_id'),
            'pre_track':sr.get('pre_track'),
            'post_track':sr.get('post_track'),
            'index_mode':sr.get('index_mode'),
            'pair_sim':sr.get('sim'),
            'row_rank':sr.get('row_rank'),
            'col_rank':sr.get('col_rank'),
            'completed_feature':int(fr is not None),
            'mode_after_feature':fr.get('mode') if fr else '',
            'stage_after_score':fr.get('stage') if fr else 'feature_missing',
            'nogt_accept':fr.get('nogt_accept') if fr else '',
            'nogt_reason':fr.get('nogt_reason') if fr else '',
            'planned_rows':fr.get('planned_rows') if fr else '',
            'high_reid_fragment_count':fr.get('high_reid_fragment_count') if fr else '',
            'bridge_fragment_count':fr.get('bridge_fragment_count') if fr else '',
        })
    write_csv(out/'bridge_or_direct_missing_status.csv', bridge_rows)

    # Direct competition group status over initial index + completed/cached features.
    missing_by_group=defaultdict(list)
    for r in missing_init:
        missing_by_group[(str(r.get('tunnel_id')), str(r.get('pre_track')))].append(r)
    built_by_group=defaultdict(list)
    for r in built:
        if r.get('mode')=='direct' or direct_like_feature(r):
            built_by_group[(str(r.get('tunnel_id')), str(r.get('pre_track')))].append(r)
    group_rows=[]
    for key, indexed in sorted(missing_by_group.items(), key=lambda kv:(ai(kv[0][0]), ai(kv[0][1]))):
        built_g=built_by_group.get(key, [])
        built_sorted=sorted(built_g, key=lambda x:af(x.get('max_high_sim_to_anchor')), reverse=True)
        accepted_now=[r for r in built_g if ai(r.get('nogt_accept'))]
        feature_built_missing=sum(1 for r in indexed if r['transaction_id'] in completed_ids)
        feature_missing=sum(1 for r in indexed if r['transaction_id'] not in completed_ids)
        # Known accepted cached direct in group counts as built; missing_init may not include it.
        row={
            'tunnel_id':key[0],
            'pre_track':key[1],
            'indexed_missing_candidates':len(indexed),
            'feature_built_from_missing':feature_built_missing,
            'feature_missing_candidates':feature_missing,
            'direct_like_built_candidates':len(built_g),
            'group_complete_for_missing':int(feature_missing==0),
            'any_current_accept':int(bool(accepted_now)),
            'current_accept_ids':'|'.join(r.get('transaction_id','') for r in accepted_now),
            'current_best_candidate':built_sorted[0].get('transaction_id','') if built_sorted else '',
            'current_best_high_sim':af(built_sorted[0].get('max_high_sim_to_anchor')) if built_sorted else 0,
            'current_second_high_sim':af(built_sorted[1].get('max_high_sim_to_anchor')) if len(built_sorted)>1 else 0,
            'current_best_margin_to_second':(af(built_sorted[0].get('max_high_sim_to_anchor'))-(af(built_sorted[1].get('max_high_sim_to_anchor')) if len(built_sorted)>1 else 0)) if built_sorted else 0,
            'remaining_ids':'|'.join(r['transaction_id'] for r in indexed if r['transaction_id'] not in completed_ids),
            'indexed_ids':'|'.join(r['transaction_id'] for r in indexed),
            'built_direct_ids':'|'.join(r.get('transaction_id','') for r in built_sorted),
        }
        # Risk score for scanning group.
        max_pair_sim=max((af(r.get('sim')) for r in indexed if r['transaction_id'] not in completed_ids), default=0)
        min_rank=min((min(ai(r.get('row_rank'),999), ai(r.get('col_rank'),999)) for r in indexed if r['transaction_id'] not in completed_ids), default=999)
        row['max_remaining_pair_sim']=max_pair_sim
        row['min_remaining_rank']=min_rank
        row['priority_group_score']=100*row['feature_missing_candidates'] + 20*int(max_pair_sim>=0.55) + 10*int(min_rank<=2) + max_pair_sim
        group_rows.append(row)
    group_rows=sorted(group_rows, key=lambda r:r['priority_group_score'], reverse=True)
    write_csv(out/'path_direct_competition_group_status.csv', group_rows)

    by_tunnel=[]
    for tid,rs in defaultdict(list, { }).items():
        pass
    ctr=defaultdict(list)
    for r in remaining:
        ctr[str(r.get('tunnel_id'))].append(r)
    for tid,rs in ctr.items():
        by_tunnel.append({
            'tunnel_id':tid,
            'remaining_count':len(rs),
            'bridge_or_direct_remaining':sum(1 for r in rs if r.get('index_mode')=='bridge_or_direct_index'),
            'max_pair_sim':max(af(r.get('sim')) for r in rs),
            'min_rank':min(min(ai(r.get('row_rank'),999), ai(r.get('col_rank'),999)) for r in rs),
            'ids':'|'.join(r['transaction_id'] for r in rs[:50]),
        })
    by_tunnel=sorted(by_tunnel, key=lambda r:(r['remaining_count'], r['max_pair_sim']), reverse=True)
    write_csv(out/'remaining_path_missing_by_tunnel.csv', by_tunnel)

    summary={
        'initial_missing_pool':len(missing_init),
        'completed_new_features':len(completed_new),
        'remaining_path_missing':len(remaining),
        'accepted_new':[r.get('transaction_id') for r in completed_new.values() if ai(r.get('nogt_accept'))],
        'completed_stage_counts':dict(Counter(r.get('stage') for r in completed_new.values())),
        'completed_mode_counts':dict(Counter(r.get('mode') for r in completed_new.values())),
        'remaining_priority_bucket_counts':dict(Counter(r.get('priority_bucket') for r in priority)),
        'bridge_or_direct_total':len(all_bridge_ids),
        'bridge_or_direct_completed':sum(1 for r in bridge_rows if r['completed_feature']),
        'bridge_or_direct_remaining':sum(1 for r in bridge_rows if not r['completed_feature']),
        'groups_with_current_accept':sum(1 for r in group_rows if r['any_current_accept']),
        'groups_incomplete':sum(1 for r in group_rows if not r['group_complete_for_missing']),
        'top_remaining_tunnels':by_tunnel[:10],
    }
    (out/'path_priority_audit_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')

    md=['# A39_06d2c0 Path Missing Priority + Group Audit','', '## Summary','', '```json',json.dumps(summary, indent=2, sort_keys=True),'```','','## Decision','', 'Continue path missing feature scan, but prioritize P0/P1/P2 buckets and complete direct competition groups before accepting any new direct transaction.']
    (out/'decision_d2c0_priority_audit.md').write_text('\n'.join(md)+'\n')
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__=='__main__':
    main()
