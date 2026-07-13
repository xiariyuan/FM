#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict, deque
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


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


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
    if ai(r.get('same_pre_rank_by_high_sim'), 999) != 1: reasons.append('same_pre_rank_ne_1')
    if af(r.get('direct_margin_to_second')) < 0.05: reasons.append('direct_margin_lt_005')
    return int(not reasons), '|'.join(reasons) if reasons else 'direct_competition_v4_nogt'


def swap_accept(r, m_pre=0.15, m_swap=0.08, m_post=0.05, m_boundary=0.60, min_rows=5, min_frames=20):
    # Strict no-GT: intentionally does not read wrong_after_swap_rows or TrackEval metrics.
    reasons=[]
    if ai(r.get('changed_rows')) <= 0: reasons.append('no_swap_rows')
    if ai(r.get('duplicate_covered_by_other')): reasons.append('duplicate_covered')
    if ai(r.get('pred_segments_count')) != 1: reasons.append('fragmented_or_multi_segment')
    if ai(r.get('pred_swap_frames')) < min_frames: reasons.append('too_few_swap_frames')
    if not ai(r.get('proto_a_available')) or not ai(r.get('proto_b_available')): reasons.append('missing_proto')
    for region in ['a_pre_A','b_pre_B','a_swap_B','b_swap_A','a_post_B','b_post_A']:
        if ai(r.get(f'{region}_feat_rows')) < min_rows:
            reasons.append(f'{region}_too_few_feat_rows')
    for region in ['a_pre_A','b_pre_B']:
        if af(r.get(f'{region}_margin_mean')) < m_pre:
            reasons.append(f'{region}_margin_low')
    for region in ['a_swap_B','b_swap_A']:
        if af(r.get(f'{region}_margin_mean')) < m_swap:
            reasons.append(f'{region}_margin_low')
    for region in ['a_post_B','b_post_A']:
        if af(r.get(f'{region}_margin_mean')) < m_post:
            reasons.append(f'{region}_margin_low')
    if af(r.get('boundary_min_sim')) < m_boundary:
        reasons.append('boundary_sim_low')
    return int(not reasons), '|'.join(reasons) if reasons else 'persistent_handoff_proxy_v1_nogt'


def score_row(r):
    mode=r.get('mode')
    if mode == 'bridge': return bridge_accept(r)
    if mode == 'direct': return direct_accept(r)
    if mode == 'swap_persistent_handoff': return swap_accept(r)
    return 0, 'unsupported_mode'


def merge_transactions(base_track: Path, accepted, out: Path):
    base_parts=[l.split(',') for l in base_track.read_text().strip().splitlines()]
    def sig(p): return tuple([p[0]]+p[2:])
    base_map=defaultdict(deque)
    for i,p in enumerate(base_parts): base_map[sig(p)].append(i)
    changes={}; counts=Counter(); conflicts=[]; missing=[]
    for r in accepted:
        p=Path(r.get('track_result_path',''))
        if not p.exists():
            missing.append({'transaction_id':r['transaction_id'],'path':str(p)})
            continue
        local={k:deque(v) for k,v in base_map.items()}
        for line in p.read_text().strip().splitlines():
            rp=line.split(','); s=sig(rp)
            if not local[s]:
                continue
            i=local[s].popleft(); bp=base_parts[i]
            if bp[1] != rp[1]:
                if i in changes and changes[i][1] != rp[1]:
                    conflicts.append((i, changes[i], r['transaction_id'], rp[1]))
                else:
                    changes[i]=(r['transaction_id'], rp[1], bp[1], bp[0]); counts[r['transaction_id']] += 1
    if conflicts:
        raise RuntimeError(f'conflicts: {conflicts[:5]}')
    combined=[p[:] for p in base_parts]
    for i,(tid,new_id,old_id,fr) in changes.items(): combined[i][1]=new_id
    td=out/'track_results'; td.mkdir(parents=True, exist_ok=True)
    (td/'MOT20-02.txt').write_text('\n'.join(','.join(p) for p in combined)+'\n', encoding='utf-8')
    audit=[{'idx':i,'frame':fr,'old_id':old_id,'new_id':new_id,'source_transaction':tid} for i,(tid,new_id,old_id,fr) in sorted(changes.items())]
    write_csv(out/'combined_change_audit.csv', audit)
    return {'changed_rows':len(audit),'by_transaction':dict(counts),'missing':missing}


def parse_metrics(path: Path):
    lines=[x.strip() for x in path.read_text().splitlines() if x.strip()]
    return dict(zip(lines[0].split(), lines[1].split())) if len(lines)>=2 else {}


def run_trackeval(out: Path, tracker_name: str):
    eval_root=out/'eval_mot20_02'; data=eval_root/'trackers'/tracker_name/'data'; seq=eval_root/'seqmaps'
    data.mkdir(parents=True, exist_ok=True); seq.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out/'track_results'/'MOT20-02.txt', data/'MOT20-02.txt')
    (seq/'MOT20_train.txt').write_text('name\nMOT20-02\n', encoding='utf-8')
    cmd=[sys.executable,'TrackEval/scripts/run_mot_challenge.py','--GT_FOLDER','datasets/MOT20/train','--TRACKERS_FOLDER',str(eval_root/'trackers'),'--OUTPUT_FOLDER',str(eval_root/'eval'),'--TRACKERS_TO_EVAL',tracker_name,'--BENCHMARK','MOT20','--SPLIT_TO_EVAL','train','--SEQMAP_FILE',str(seq/'MOT20_train.txt'),'--SKIP_SPLIT_FOL','True','--DO_PREPROC','True','--TRACKER_SUB_FOLDER','data','--OUTPUT_SUB_FOLDER','','--PRINT_ONLY_COMBINED','True','--METRICS','HOTA','CLEAR','Identity']
    with (out/'eval_stdout.log').open('w') as stdout, (out/'eval_stderr.log').open('w') as stderr:
        subprocess.run(cmd, stdout=stdout, stderr=stderr, text=True, check=False)
    for p in (eval_root/'eval').glob('*/pedestrian_summary.txt'):
        m=parse_metrics(p)
        if m: return m
    return {}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidates', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_06c_end_to_end_no_gt_transaction_replay_seq02/transaction_candidates_nogt.csv')
    ap.add_argument('--base-track', default='outputs/spot_runtime_gate_20260628/A37_identity_escrow_association/A37_00b_candidate_logger_seq02/track_results/MOT20-02.txt')
    ap.add_argument('--out-dir', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_06c_end_to_end_no_gt_transaction_replay_seq02')
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows=read_csv(Path(args.candidates))
    scored=[]; dup=[]
    for r in rows:
        acc, reason=score_row(r)
        q=dict(r); q['nogt_accept']=acc; q['nogt_reason']=reason
        scored.append(q)
        if 'duplicate_covered' in reason:
            dup.append(q)
    accepted=[r for r in scored if ai(r.get('nogt_accept'))]
    rejected=[r for r in scored if not ai(r.get('nogt_accept')) and r not in dup]
    write_csv(out/'transaction_features_nogt.csv', scored)
    write_csv(out/'accepted_transaction_manifest_nogt.csv', accepted)
    write_csv(out/'rejected_transaction_manifest_nogt.csv', rejected)
    write_csv(out/'duplicate_covered_transactions_nogt.csv', dup)
    combined_out=out/'combined_rule_v2_nogt_stage1'
    summary=merge_transactions(Path(args.base_track), accepted, combined_out)
    metrics=run_trackeval(combined_out, 'A39_06c_rule_v2_nogt_stage1')
    summary['metrics']=metrics
    (combined_out/'combined_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    final={'candidate_count':len(rows),'accepted_count':len(accepted),'rejected_count':len(rejected),'duplicate_count':len(dup),'accepted_transactions':[r['transaction_id'] for r in accepted],'combined_summary':summary}
    (out/'summary.json').write_text(json.dumps(final, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    md=['# A39_06c Stage-1 no-GT transaction scorer replay','', '## Summary','', '```json',json.dumps(final, indent=2, sort_keys=True),'```','','## Accepted transactions','']
    for r in accepted:
        md.append(f"- {r['transaction_id']} ({r['mode']}) via {r['nogt_reason']}")
    md += ['', '## Combined metrics', '', f"HOTA={metrics.get('HOTA')} IDF1={metrics.get('IDF1')} IDSW={metrics.get('IDSW')} MOTA={metrics.get('MOTA')} Frag={metrics.get('Frag')}"]
    (out/'decision_stage1.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    print('\n'.join(md))

if __name__=='__main__':
    main()
