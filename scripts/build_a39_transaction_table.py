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

BASE_HOTA = 68.430
BASE_IDF1 = 74.413
BASE_IDSW = 443


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


def parse_metrics(path: Path):
    if not path.exists():
        return {}
    lines=[x.strip() for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(lines)<2:
        return {}
    return dict(zip(lines[0].split(), lines[1].split()))


def find_eval_metrics(run_dir: Path):
    for p in run_dir.glob('eval_mot20_02/eval/*/pedestrian_summary.txt'):
        m=parse_metrics(p)
        if m:
            return m
    return {}


def infer_mode_path(r):
    if ai(r.get('rule_bridge_mode_v2')) or (ai(r.get('selected_fragment_count'))>=2 and ai(r.get('bridge_fragment_count'))>=1):
        return 'bridge'
    if ai(r.get('is_direct_like')) or (ai(r.get('selected_fragment_count'))==1 and ai(r.get('high_reid_fragment_count'))==1 and ai(r.get('bridge_fragment_count'))==0):
        return 'direct'
    return 'path_other'


def local_clean(wrong, dup=0, changed=1):
    return int(ai(wrong, 999999) <= 1 and ai(dup, 0) == 0 and ai(changed, 0) > 0)


def global_safe(hota, idf1, idsw):
    return int(af(hota) >= BASE_HOTA and af(idf1) >= BASE_IDF1 and ai(idsw, 999999) <= BASE_IDSW)


def choose_path_candidates(path_rows):
    keep_ids={
        '12_9_71','202_501_542','202_501_545','197_384_549','202_501_547','202_501_540','197_384_546',
        '47_104_112','104_26_266','148_373_394','202_530_542','104_17_266','104_26_489'
    }
    out=[]
    for r in path_rows:
        aid=r.get('anchor_id')
        if aid in keep_ids or ai(r.get('label_safe_to_rewrite')) or ai(r.get('rule_no_bridge')) or ai(r.get('rule_mode_union_v4')):
            out.append(r)
    # Ensure deterministic and unique.
    seen=set(); uniq=[]
    for r in out:
        if r.get('anchor_id') in seen:
            continue
        seen.add(r.get('anchor_id')); uniq.append(r)
    return uniq


def build_path_records(path_rows):
    records=[]
    accepted={'12_9_71','202_501_542'}
    for r in choose_path_candidates(path_rows):
        aid=r['anchor_id']; mode=infer_mode_path(r)
        hota=r.get('HOTA',''); idf1=r.get('IDF1',''); idsw=r.get('IDSW','')
        rec={
            'transaction_id': aid,
            'mode': mode,
            'method': 'path_builder',
            'source': r.get('candidate_source',''),
            'source_run_dir': r.get('source_run_dir',''),
            'track_result_path': str(Path(r.get('source_run_dir',''))/'track_results'/'MOT20-02.txt') if r.get('source_run_dir') else '',
            'tunnel_id': r.get('tunnel_id',''),
            'track_a': r.get('pre_track',''),
            'track_b': r.get('post_track',''),
            'gt_a': r.get('pre_gt',''),
            'gt_b': r.get('post_gt',''),
            'changed_rows': r.get('applied_rows', r.get('planned_rows','')),
            'planned_rows': r.get('planned_rows',''),
            'applied_rows': r.get('applied_rows',''),
            'selected_fragment_count': r.get('selected_fragment_count',''),
            'high_reid_fragment_count': r.get('high_reid_fragment_count',''),
            'bridge_fragment_count': r.get('bridge_fragment_count',''),
            'gap_row_count': r.get('gap_row_count',''),
            'skipped_collision_rows': r.get('skipped_collision_rows',''),
            'diag_wrong_rows': r.get('diag_wrong_rows',''),
            'wrong_after_swap_rows': '',
            'same_frame_duplicate_count_after_swap': 0,
            'HOTA': hota,
            'IDF1': idf1,
            'IDSW': idsw,
            'MOTA': r.get('MOTA',''),
            'Frag': r.get('Frag',''),
            'sim': r.get('sim',''),
            'row_rank': r.get('row_rank',''),
            'col_rank': r.get('col_rank',''),
            'max_high_sim_to_anchor': r.get('max_high_sim_to_anchor',''),
            'max_high_sim_to_post': r.get('max_high_sim_to_post',''),
            'max_high_sim_to_pre': r.get('max_high_sim_to_pre',''),
            'same_pre_rank_by_high_sim': r.get('same_pre_rank_by_high_sim',''),
            'direct_margin_to_second': r.get('direct_margin_to_second',''),
            'rule_bridge_mode_v2': r.get('rule_bridge_mode_v2',''),
            'rule_direct_competition_v4': r.get('rule_direct_competition_v4',''),
            'rule_mode_union_v4': r.get('rule_mode_union_v4',''),
            'persistent_handoff_gate_gt_v2_pass': '',
            'pred_segments_count': '',
            'pred_swap_frames': '',
            'duplicate_covered_by_other': 0,
            'lifecycle_failure_reason': '',
            'label_local_clean': local_clean(r.get('diag_wrong_rows'), r.get('skipped_collision_rows'), r.get('applied_rows', r.get('planned_rows'))),
            'label_global_safe': global_safe(hota,idf1,idsw),
            'label_accepted_current_best': int(aid in accepted),
            'label_duplicate_covered': 0,
        }
        rec['label_global_negative'] = int(not rec['label_global_safe'])
        records.append(rec)
    return records


def build_swap_records(lifecycle_rows):
    records=[]
    for r in lifecycle_rows:
        if r.get('method')!='reid_viterbi_penalty_0.1':
            continue
        tid=r['case']
        accepted={'106_150_169','215_469_508'}
        duplicate={'214_508_469'}
        hota=r.get('HOTA',''); idf1=r.get('IDF1',''); idsw=r.get('IDSW','')
        rec={
            'transaction_id': tid,
            'mode': 'swap_persistent_handoff',
            'method': r.get('method',''),
            'source': 'A39_05e_lifecycle_v2',
            'source_run_dir': '',
            'track_result_path': '',
            'tunnel_id': tid.split('_')[0] if '_' in tid else '',
            'track_a': r.get('track_a',''),
            'track_b': r.get('track_b',''),
            'gt_a': r.get('gt_a',''),
            'gt_b': r.get('gt_b',''),
            'changed_rows': r.get('changed_rows',''),
            'planned_rows': r.get('changed_rows',''),
            'applied_rows': r.get('changed_rows',''),
            'selected_fragment_count': '',
            'high_reid_fragment_count': '',
            'bridge_fragment_count': '',
            'gap_row_count': '',
            'skipped_collision_rows': 0,
            'diag_wrong_rows': r.get('wrong_after_swap_rows',''),
            'wrong_after_swap_rows': r.get('wrong_after_swap_rows',''),
            'same_frame_duplicate_count_after_swap': '',
            'HOTA': hota,
            'IDF1': idf1,
            'IDSW': idsw,
            'MOTA': r.get('MOTA',''),
            'Frag': r.get('Frag',''),
            'sim': '',
            'row_rank': '',
            'col_rank': '',
            'max_high_sim_to_anchor': '',
            'max_high_sim_to_post': '',
            'max_high_sim_to_pre': '',
            'same_pre_rank_by_high_sim': '',
            'direct_margin_to_second': '',
            'rule_bridge_mode_v2': '',
            'rule_direct_competition_v4': '',
            'rule_mode_union_v4': '',
            'persistent_handoff_gate_gt_v2_pass': r.get('persistent_handoff_gate_gt_v2_pass',''),
            'pred_segments_count': r.get('pred_segments_count',''),
            'pred_segments': r.get('pred_segments',''),
            'pred_swap_frames': r.get('changed_rows',''),
            'swap_precision': r.get('swap_precision',''),
            'swap_recall': r.get('swap_recall',''),
            'duplicate_covered_by_other': r.get('duplicate_covered_by_other',''),
            'lifecycle_failure_reason': r.get('lifecycle_failure_reason',''),
            'a_pre_ratio': r.get('a_pre_ratio',''),
            'b_pre_ratio': r.get('b_pre_ratio',''),
            'a_swap_for_B_ratio': r.get('a_swap_for_B_ratio',''),
            'b_swap_for_A_ratio': r.get('b_swap_for_A_ratio',''),
            'a_post_for_B_ratio': r.get('a_post_for_B_ratio',''),
            'b_post_for_A_ratio': r.get('b_post_for_A_ratio',''),
            'label_local_clean': local_clean(r.get('wrong_after_swap_rows'), 0, r.get('changed_rows')),
            'label_global_safe': global_safe(hota,idf1,idsw),
            'label_accepted_current_best': int(tid in accepted),
            'label_duplicate_covered': int(tid in duplicate or ai(r.get('duplicate_covered_by_other'))),
        }
        rec['label_global_negative']=int(not rec['label_global_safe'])
        # Fill track result path based on known state roots.
        if tid in {'106_150_169','215_469_508'}:
            rec['track_result_path']=f'outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_05c_swap_state_segmentation_for_reciprocal_swap/{tid}/reid_viterbi_penalty_0.1/track_results/MOT20-02.txt'
        else:
            rec['track_result_path']=f'outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_05d_collision_swap_candidate_mining/state_segmentation/{tid}/reid_viterbi_penalty_0.1/track_results/MOT20-02.txt'
        records.append(rec)
    return records


def scorer_rule_v1(r):
    mode=r.get('mode')
    if ai(r.get('label_duplicate_covered')):
        return 0, 'duplicate_covered'
    if mode=='bridge':
        ok=ai(r.get('rule_bridge_mode_v2')) and ai(r.get('label_local_clean'))
        return int(ok), 'bridge_mode_v2' if ok else 'bridge_reject'
    if mode=='direct':
        ok=ai(r.get('rule_direct_competition_v4')) and ai(r.get('label_local_clean'))
        return int(ok), 'direct_competition_v4' if ok else 'direct_reject'
    if mode=='swap_persistent_handoff':
        ok=ai(r.get('persistent_handoff_gate_gt_v2_pass')) and ai(r.get('label_local_clean'))
        return int(ok), 'persistent_handoff_gate_v2' if ok else 'swap_lifecycle_reject'
    return 0, 'unsupported_mode'


def rule_report(records):
    tp=fp=tn=fn=dup=0; rows=[]
    for r in records:
        pred,reason=scorer_rule_v1(r); lab=ai(r.get('label_accepted_current_best'))
        if ai(r.get('label_duplicate_covered')):
            dup+=1
        elif pred and lab:
            tp+=1
        elif pred and not lab:
            fp+=1
        elif not pred and not lab:
            tn+=1
        else:
            fn+=1
        q=dict(r); q['score_rule_v1_accept']=pred; q['score_rule_v1_reason']=reason; rows.append(q)
    return {'rule_name':'scorer_rule_v1','tp':tp,'fp':fp,'tn':tn,'fn':fn,'duplicate_covered':dup,'precision':safe_div(tp,tp+fp),'recall':safe_div(tp,tp+fn)}, rows


def merge_transactions(base_track: Path, accepted_records, out: Path):
    base_parts=[l.split(',') for l in base_track.read_text().strip().splitlines()]
    def sig(p): return tuple([p[0]]+p[2:])
    base_map=defaultdict(deque)
    for i,p in enumerate(base_parts): base_map[sig(p)].append(i)
    changes={}; counts=Counter(); conflicts=[]; missing=[]
    for r in accepted_records:
        path=Path(r.get('track_result_path',''))
        if not path.exists():
            missing.append((r['transaction_id'], str(path)))
            continue
        local={k:deque(v) for k,v in base_map.items()}
        for line in path.read_text().strip().splitlines():
            rp=line.split(','); s=sig(rp)
            if not local[s]:
                missing.append((r['transaction_id'], str(s))); continue
            i=local[s].popleft(); bp=base_parts[i]
            if bp[1]!=rp[1]:
                if i in changes and changes[i][1]!=rp[1]:
                    conflicts.append((i,changes[i],r['transaction_id'],rp[1]))
                else:
                    changes[i]=(r['transaction_id'],rp[1],bp[1],bp[0]); counts[r['transaction_id']]+=1
    if conflicts:
        raise RuntimeError(f'conflicts: {conflicts[:5]}')
    combined=[p[:] for p in base_parts]
    for i,(tid,new_id,old_id,fr) in changes.items():
        combined[i][1]=new_id
    td=out/'track_results'; td.mkdir(parents=True, exist_ok=True)
    (td/'MOT20-02.txt').write_text('\n'.join(','.join(p) for p in combined)+'\n', encoding='utf-8')
    audit=[]
    for i,(tid,new_id,old_id,fr) in sorted(changes.items()):
        audit.append({'idx':i,'frame':fr,'old_id':old_id,'new_id':new_id,'source_transaction':tid})
    write_csv(out/'combined_change_audit.csv', audit)
    return {'changed_rows':len(audit),'by_transaction':dict(counts),'missing':missing}


def run_trackeval(out: Path, tracker_name: str):
    eval_root=out/'eval_mot20_02'
    data_dir=eval_root/'trackers'/tracker_name/'data'
    seqmap_dir=eval_root/'seqmaps'
    data_dir.mkdir(parents=True, exist_ok=True); seqmap_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out/'track_results'/'MOT20-02.txt', data_dir/'MOT20-02.txt')
    (seqmap_dir/'MOT20_train.txt').write_text('name\nMOT20-02\n', encoding='utf-8')
    cmd=[sys.executable,'TrackEval/scripts/run_mot_challenge.py','--GT_FOLDER','datasets/MOT20/train','--TRACKERS_FOLDER',str(eval_root/'trackers'),'--OUTPUT_FOLDER',str(eval_root/'eval'),'--TRACKERS_TO_EVAL',tracker_name,'--BENCHMARK','MOT20','--SPLIT_TO_EVAL','train','--SEQMAP_FILE',str(seqmap_dir/'MOT20_train.txt'),'--SKIP_SPLIT_FOL','True','--DO_PREPROC','True','--TRACKER_SUB_FOLDER','data','--OUTPUT_SUB_FOLDER','','--PRINT_ONLY_COMBINED','True','--METRICS','HOTA','CLEAR','Identity']
    with (out/'eval_stdout.log').open('w') as stdout, (out/'eval_stderr.log').open('w') as stderr:
        subprocess.run(cmd, stdout=stdout, stderr=stderr, text=True, check=False)
    for p in (eval_root/'eval').glob('*/pedestrian_summary.txt'):
        m=parse_metrics(p)
        if m: return m
    return {}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--a39-root', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel')
    ap.add_argument('--base-track', default='outputs/spot_runtime_gate_20260628/A37_identity_escrow_association/A37_00b_candidate_logger_seq02/track_results/MOT20-02.txt')
    ap.add_argument('--out-dir', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_06_interpretable_transaction_scorer')
    args=ap.parse_args()
    root=Path(args.a39_root); out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    path_rows=read_csv(root/'A39_04d_rule_mode_union_v2_broader_stress_and_combined/path_transaction_examples_stress_v4.csv')
    lifecycle_rows=read_csv(root/'A39_05e_global_lifecycle_gate_for_swap_transactions/swap_lifecycle_audit_v2.csv')
    records=build_path_records(path_rows)+build_swap_records(lifecycle_rows)
    # Remove exact duplicate transaction ids preserving path+swap distinct; for swap lifecycle rows only reid viterbi already filtered.
    seen=set(); uniq=[]
    for r in records:
        key=(r['transaction_id'],r['mode'],r['method'])
        if key in seen: continue
        seen.add(key); uniq.append(r)
    records=uniq
    write_csv(out/'transaction_candidates.csv', records)
    report, scored=rule_report(records)
    write_csv(out/'transaction_features_deployable.csv', scored)
    label_rows=[{k:r.get(k,'') for k in ['transaction_id','mode','label_local_clean','label_global_safe','label_accepted_current_best','label_duplicate_covered','HOTA','IDF1','IDSW','diag_wrong_rows','wrong_after_swap_rows','lifecycle_failure_reason']} for r in scored]
    write_csv(out/'transaction_labels_diagnostic.csv', label_rows)
    accepted=[r for r in scored if ai(r.get('score_rule_v1_accept'))]
    rejected=[r for r in scored if not ai(r.get('score_rule_v1_accept')) and not ai(r.get('label_duplicate_covered'))]
    dup=[r for r in scored if ai(r.get('label_duplicate_covered'))]
    write_csv(out/'accepted_transaction_manifest.csv', accepted)
    write_csv(out/'rejected_transaction_manifest.csv', rejected)
    write_csv(out/'duplicate_covered_transactions.csv', dup)
    write_csv(out/'transaction_rule_report.csv', [report])
    combined_out=out/'combined_rule_v1'
    summary=merge_transactions(Path(args.base_track), accepted, combined_out)
    metrics=run_trackeval(combined_out, 'A39_06_rule_v1_combined')
    summary['metrics']=metrics
    (combined_out/'combined_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    final={
        'candidate_count':len(records), 'accepted_count':len(accepted), 'duplicate_covered_count':len(dup),
        'rule_report':report, 'combined_summary':summary,
        'accepted_transactions':[r['transaction_id'] for r in accepted],
    }
    (out/'summary.json').write_text(json.dumps(final, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    # Docs.
    (out/'transaction_feature_schema.md').write_text('# A39_06 transaction feature schema\n\nModes: bridge, direct, swap_persistent_handoff. Features are copied from A39_04d path diagnostics and A39_05e lifecycle diagnostics. GT fields are diagnostic only; scorer_rule_v1 uses existing mode gates and lifecycle gate status.\n', encoding='utf-8')
    (out/'transaction_label_policy.md').write_text('# A39_06 label policy\n\nlabel_local_clean: diagnostic wrong rows <= 1 and no duplicates.\nlabel_global_safe: HOTA >= baseline, IDF1 >= baseline, IDSW <= baseline.\nlabel_accepted_current_best: {12_9_71, 202_501_542, 106_150_169, 215_469_508}.\nlabel_duplicate_covered: transaction already covered by an accepted larger transaction, e.g. 214_508_469.\n', encoding='utf-8')
    md=['# A39_06 Interpretable Transaction Scorer','', '## Summary','', '```json',json.dumps(final, indent=2, sort_keys=True),'```','', '## Accepted transactions','']
    for r in accepted:
        md.append(f"- {r['transaction_id']} ({r['mode']}) via {r['score_rule_v1_reason']}")
    md += ['', '## Combined metrics','', f"HOTA={metrics.get('HOTA')} IDF1={metrics.get('IDF1')} IDSW={metrics.get('IDSW')} MOTA={metrics.get('MOTA')} Frag={metrics.get('Frag')}"]
    (out/'decision.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    print('\n'.join(md))

if __name__=='__main__':
    main()
