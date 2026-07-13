#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


def ai(v, d=0):
    try: return int(float(v))
    except Exception: return d


def af(v, d=0.0):
    try: return float(v)
    except Exception: return d


def read_csv(path: Path):
    if not path.exists(): return []
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


def bridge_accept(r):
    reasons=[]
    if ai(r.get('planned_rows')) < 30: reasons.append('planned_lt_30')
    if ai(r.get('selected_fragment_count')) < 2: reasons.append('selected_lt_2')
    if ai(r.get('high_reid_fragment_count')) < 1: reasons.append('high_reid_lt_1')
    if ai(r.get('bridge_fragment_count')) < 1: reasons.append('bridge_lt_1')
    if ai(r.get('gap_row_count')) > 10: reasons.append('gap_gt_10')
    if af(r.get('max_gap_dist')) > 0.12: reasons.append('max_gap_dist_gt_012')
    if ai(r.get('skipped_collision_rows')) != 0: reasons.append('collision_skipped')
    return int(not reasons), '|'.join(reasons) if reasons else 'bridge_mode_v2_nogt'


def direct_accept(r):
    reasons=[]
    if ai(r.get('planned_rows')) < 30: reasons.append('planned_lt_30')
    if ai(r.get('selected_fragment_count')) != 1: reasons.append('selected_ne_1')
    if ai(r.get('high_reid_fragment_count')) != 1: reasons.append('high_reid_ne_1')
    if ai(r.get('bridge_fragment_count')) != 0: reasons.append('bridge_ne_0')
    if ai(r.get('gap_row_count')) != 0: reasons.append('gap_ne_0')
    if ai(r.get('skipped_collision_rows')) != 0: reasons.append('collision_skipped')
    if af(r.get('max_high_sim_to_anchor')) < 0.80: reasons.append('anchor_sim_lt_080')
    if af(r.get('max_high_sim_to_post')) < 0.90: reasons.append('post_sim_lt_090')
    if af(r.get('max_high_sim_to_pre')) < 0.40: reasons.append('pre_sim_lt_040')
    if ai(r.get('same_pre_rank_by_high_sim'),999) != 1: reasons.append('same_pre_rank_ne_1')
    if af(r.get('direct_margin_to_second')) < 0.05: reasons.append('direct_margin_lt_005')
    return int(not reasons), '|'.join(reasons) if reasons else 'direct_competition_v4_nogt'


def infer_mode(r):
    if ai(r.get('planned_rows')) >= 30 and ai(r.get('selected_fragment_count')) >= 2 and ai(r.get('high_reid_fragment_count')) >= 1 and ai(r.get('bridge_fragment_count')) >= 1:
        return 'bridge'
    if ai(r.get('planned_rows')) >= 30 and ai(r.get('selected_fragment_count')) == 1 and ai(r.get('high_reid_fragment_count')) == 1 and ai(r.get('bridge_fragment_count')) == 0 and ai(r.get('gap_row_count')) == 0:
        return 'direct'
    return 'path_other'


def run_builder(job, run_dir: Path, args):
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir/'rewrite_summary.json').exists():
        return 'reused_existing'
    cmd=[
        sys.executable, 'scripts/simulate_a39_path_bridge_rewrite.py',
        '--track-file', args.track_file,
        '--gt-file', args.gt_file,
        '--tunnels-csv', args.tunnels_csv,
        '--img-dir', args.img_dir,
        '--fast-reid-config', args.fast_reid_config,
        '--fast-reid-weights', args.fast_reid_weights,
        '--out-dir', str(run_dir),
        '--tunnel-id', str(ai(job.get('tunnel_id'))),
        '--pre-anchor', str(ai(job.get('pre_track'))),
        '--post-anchor', str(ai(job.get('post_track'))),
        '--target-id', str(ai(job.get('pre_track'))),
        '--device', args.device,
    ]
    # Important: intentionally do NOT pass --anchor-gt; GT is diagnostics only in builder and not used for scoring.
    with (run_dir/'path_builder_stdout.log').open('w', encoding='utf-8') as stdout, (run_dir/'path_builder_stderr.log').open('w', encoding='utf-8') as stderr:
        ret=subprocess.run(cmd, stdout=stdout, stderr=stderr, text=True, check=False)
    return 'ok' if ret.returncode == 0 else f'path_builder_failed_{ret.returncode}'


def summarize(job, run_dir: Path, status: str):
    tid=job['transaction_id']
    base={
        'transaction_id':tid,
        'candidate_family':'path',
        'index_mode':job.get('index_mode',''),
        'tunnel_id':job.get('tunnel_id',''),
        'pre_track':job.get('pre_track',''),
        'post_track':job.get('post_track',''),
        'pair_sim':job.get('sim',''),
        'row_rank':job.get('row_rank',''),
        'col_rank':job.get('col_rank',''),
        'row_margin':job.get('row_margin',''),
        'col_margin':job.get('col_margin',''),
        'center_delta_norm':job.get('center_delta_norm',''),
        'height_ratio':job.get('height_ratio',''),
        'post_collision_ratio':job.get('post_collision_ratio',''),
        'source_run_dir':str(run_dir),
        'track_result_path':str(run_dir/'track_results'/'MOT20-02.txt'),
        'run_status':status,
    }
    if not (run_dir/'rewrite_summary.json').exists():
        base.update({'mode':'path_other','stage':'feature_failed','nogt_accept':0,'nogt_reason':status})
        return base
    rw=json.loads((run_dir/'rewrite_summary.json').read_text(encoding='utf-8'))
    sel=read_csv(run_dir/'selected_fragments.csv')
    gaps=read_csv(run_dir/'gap_rows.csv')
    high=[r for r in sel if r.get('selected_stage')=='high_reid']
    bridge=[r for r in sel if r.get('selected_stage')=='bridge_fragment']
    rec=dict(base)
    rec.update({
        'selected_fragment_count': rw.get('selected_fragment_count', len(sel)),
        'high_reid_fragment_count': len(high),
        'bridge_fragment_count': len(bridge),
        'gap_row_count': rw.get('gap_row_count', len(gaps)),
        'planned_rows': rw.get('planned_rows', 0),
        'applied_rows': rw.get('applied_rows', 0),
        'skipped_collision_rows': rw.get('skipped_collision_rows', 0),
        'max_gap_dist': max([af(r.get('dist')) for r in gaps], default=0.0),
        'max_high_sim_to_anchor': max([af(r.get('sim_to_anchor')) for r in high], default=0.0),
        'max_high_sim_to_post': max([af(r.get('sim_to_post')) for r in high], default=0.0),
        'max_high_sim_to_pre': max([af(r.get('sim_to_pre')) for r in high], default=0.0),
        'selected_fragments': '|'.join(r.get('fragment_key','') for r in sel),
    })
    rec['mode']=infer_mode(rec)
    rec['stage']='feature_built_pending_score'
    return rec


def enrich_direct_competition(rows):
    # Compute competition over all feature-built path candidates available now.
    direct_like=[]
    for r in rows:
        r.setdefault('same_pre_rank_by_high_sim','0')
        r.setdefault('direct_margin_to_second','0')
        if (ai(r.get('planned_rows'))>=30 and ai(r.get('selected_fragment_count'))==1 and ai(r.get('high_reid_fragment_count'))==1 and ai(r.get('bridge_fragment_count'))==0 and ai(r.get('gap_row_count'))==0 and ai(r.get('skipped_collision_rows'))==0):
            direct_like.append(r)
    by_pre=defaultdict(list)
    for r in direct_like:
        by_pre[(str(r.get('tunnel_id')), str(r.get('pre_track')))].append(r)
    for key, group in by_pre.items():
        group=sorted(group, key=lambda x: af(x.get('max_high_sim_to_anchor')), reverse=True)
        best=af(group[0].get('max_high_sim_to_anchor')) if group else 0.0
        second=af(group[1].get('max_high_sim_to_anchor')) if len(group)>1 else 0.0
        for i,r in enumerate(group, start=1):
            current=af(r.get('max_high_sim_to_anchor'))
            r['same_pre_direct_candidate_count']=len(group)
            r['same_pre_rank_by_high_sim']=i
            r['same_pre_best_high_sim']=best
            r['same_pre_second_high_sim']=second
            r['direct_margin_to_second']=current-second if i==1 else current-best
    return rows


def score_paths(rows):
    rows=enrich_direct_competition(rows)
    for r in rows:
        if r.get('stage')=='feature_failed':
            continue
        mode=r.get('mode') or infer_mode(r)
        r['mode']=mode
        if mode=='bridge': acc,reason=bridge_accept(r)
        elif mode=='direct': acc,reason=direct_accept(r)
        else: acc,reason=(0,'unsupported_path_mode')
        r['nogt_accept']=acc; r['nogt_reason']=reason
        r['stage']='accepted' if acc else 'scorer_rejected'
    return rows


def load_cached_path_features(stage_report: Path):
    rows=[]
    for r in read_csv(stage_report):
        if r.get('candidate_family')=='path' and r.get('stage') in {'accepted','scorer_rejected'}:
            q={k:v for k,v in r.items()}
            rows.append(q)
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage-report', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_06d_full_seq_no_gt_candidate_scan_seq02/transaction_stage_report_full_nogt_cached.csv')
    ap.add_argument('--out-dir', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_06d_full_seq_no_gt_candidate_scan_seq02/A39_06d2c_path_missing_feature_scan')
    ap.add_argument('--track-file', default='outputs/spot_runtime_gate_20260628/A37_identity_escrow_association/A37_00b_candidate_logger_seq02/track_results/MOT20-02.txt')
    ap.add_argument('--gt-file', default='datasets/MOT20/train/MOT20-02/gt/gt.txt')
    ap.add_argument('--tunnels-csv', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_00_tunnel_discovery_seq02_ioa050_dur3/tunnel_candidates.csv')
    ap.add_argument('--img-dir', default='datasets/MOT20/train/MOT20-02/img1')
    ap.add_argument('--fast-reid-config', default='external/BoT-SORT-main/fast_reid/configs/MOT20/sbs_S50.yml')
    ap.add_argument('--fast-reid-weights', default='external/BoT-SORT-main/pretrained/mot20_sbs_S50.pth')
    ap.add_argument('--mode', choices=['manifest','smoke','full'], default='smoke')
    ap.add_argument('--top-k', type=int, default=20)
    ap.add_argument('--exclude-completed-cases', action='store_true')
    ap.add_argument('--device', default='cuda')
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    stage_rows=read_csv(Path(args.stage_report))
    jobs=[r for r in stage_rows if r.get('stage')=='feature_missing_cached' and r.get('candidate_family')=='path_index']
    # Prioritize bridge_or_direct and high pair sim / rank, but do not filter out anything.
    def pri(r):
        return (1 if r.get('index_mode')=='bridge_or_direct_index' else 0, af(r.get('sim')), -ai(r.get('row_rank'),999), -ai(r.get('col_rank'),999))
    jobs=sorted(jobs, key=pri, reverse=True)
    for i,r in enumerate(jobs, start=1):
        r['job_rank']=i
    write_csv(out/'path_feature_job_manifest.csv', jobs)
    if args.mode=='manifest':
        print(json.dumps({'job_count':len(jobs),'out_dir':str(out)}, indent=2))
        return
    if args.exclude_completed_cases:
        jobs=[r for r in jobs if not (out/'cases'/r['transaction_id']/'rewrite_summary.json').exists() and not (out/'cases'/r['transaction_id']/'feature_failed.json').exists()]
    if args.mode=='smoke':
        jobs=jobs[:args.top_k]
    # full runs all remaining jobs.
    scanned=[]
    for idx,job in enumerate(jobs, start=1):
        tid=job['transaction_id']; run_dir=out/'cases'/tid
        status=run_builder(job, run_dir, args)
        rec=summarize(job, run_dir, status)
        scanned.append(rec)
        if rec.get('stage')=='feature_failed':
            (run_dir/'feature_failed.json').write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
        (run_dir/'feature_summary_nogt.json').write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
        print(f'[{idx}/{len(jobs)}] {tid} {status} mode={rec.get("mode")} planned={rec.get("planned_rows")} high={rec.get("high_reid_fragment_count")} bridge={rec.get("bridge_fragment_count")}')
    # Aggregate all completed new feature summaries.
    new=[]
    for p in sorted((out/'cases').glob('*/feature_summary_nogt.json')):
        try: new.append(json.load(open(p)))
        except Exception as e: print('bad feature',p,e)
    all_features=load_cached_path_features(Path(args.stage_report))+new
    all_features=score_paths(all_features)
    # Split outputs.
    new_ids=set(r['transaction_id'] for r in new)
    new_scored=[r for r in all_features if r.get('transaction_id') in new_ids]
    accepted=[r for r in all_features if ai(r.get('nogt_accept'))]
    new_accepted=[r for r in new_scored if ai(r.get('nogt_accept'))]
    rejected_new=[r for r in new_scored if not ai(r.get('nogt_accept'))]
    write_csv(out/'path_features_new_scanned.csv', new_scored)
    write_csv(out/'path_features_all_scored_with_cached.csv', all_features)
    write_csv(out/'accepted_path_manifest_all.csv', accepted)
    write_csv(out/'accepted_path_manifest_new.csv', new_accepted)
    write_csv(out/'rejected_path_manifest_new.csv', rejected_new)
    completed_ids=set(r['transaction_id'] for r in new)
    remaining=[r for r in [x for x in stage_rows if x.get('stage')=='feature_missing_cached' and x.get('candidate_family')=='path_index'] if r['transaction_id'] not in completed_ids]
    summary={'initial_missing_pool':len([x for x in stage_rows if x.get('stage')=='feature_missing_cached' and x.get('candidate_family')=='path_index']),'completed_new_features':len(new),'remaining_path_missing':len(remaining),'new_accepted':[r['transaction_id'] for r in new_accepted],'accepted_all':[r['transaction_id'] for r in accepted],'stage_counts_new':dict(Counter(r.get('stage') for r in new_scored)),'mode_counts_new':dict(Counter(r.get('mode') for r in new_scored))}
    (out/'path_scan_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__':
    main()
