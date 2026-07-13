#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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


def infer_path_mode(r):
    sc=ai(r.get('selected_fragment_count'))
    high=ai(r.get('high_reid_fragment_count'))
    br=ai(r.get('bridge_fragment_count'))
    gap=ai(r.get('gap_row_count'))
    planned=ai(r.get('planned_rows'))
    if planned >= 30 and sc >= 2 and high >= 1 and br >= 1:
        return 'bridge'
    if planned >= 30 and sc == 1 and high == 1 and br == 0 and gap == 0:
        return 'direct'
    return 'path_other'


def path_candidate_reason(r):
    reasons=[]
    if ai(r.get('planned_rows')) <= 0:
        reasons.append('no_planned_rows')
    if ai(r.get('selected_fragment_count')) <= 0:
        reasons.append('no_selected_fragments')
    if not r.get('source_run_dir'):
        reasons.append('missing_run_dir')
    return '|'.join(reasons) if reasons else 'nogt_path_candidate'


def build_path_candidates(path_rows):
    rows=[]; seen=set()
    # Stage-1 constrained pool: all already-built path transactions with nonzero planned/selected rows.
    # This does not use GT labels for scoring; it reuses path-builder outputs already available on disk.
    for r in path_rows:
        tid=r.get('anchor_id','')
        if not tid or tid in seen:
            continue
        if ai(r.get('planned_rows')) <= 0 and tid not in {'12_9_71','202_501_542'}:
            continue
        seen.add(tid)
        mode=infer_path_mode(r)
        rec={
            'transaction_id': tid,
            'mode': mode,
            'candidate_family': 'path',
            'candidate_pool_stage': 'stage1_existing_path_builder_outputs_no_gt_scoring',
            'candidate_reason': path_candidate_reason(r),
            'track_result_path': str(Path(r.get('source_run_dir',''))/'track_results'/'MOT20-02.txt') if r.get('source_run_dir') else '',
            'source_run_dir': r.get('source_run_dir',''),
            'tunnel_id': r.get('tunnel_id',''),
            'track_a': r.get('pre_track',''),
            'track_b': r.get('post_track',''),
            'planned_rows': r.get('planned_rows',''),
            'applied_rows': r.get('applied_rows',''),
            'selected_fragment_count': r.get('selected_fragment_count',''),
            'high_reid_fragment_count': r.get('high_reid_fragment_count',''),
            'bridge_fragment_count': r.get('bridge_fragment_count',''),
            'gap_row_count': r.get('gap_row_count',''),
            'skipped_collision_rows': r.get('skipped_collision_rows',''),
            'max_gap_dist': r.get('max_gap_dist',''),
            'max_bridge_score_selected': r.get('max_bridge_score_selected',''),
            'min_bridge_sim_selected': r.get('min_bridge_sim_selected',''),
            'sim': r.get('sim',''),
            'row_rank': r.get('row_rank',''),
            'col_rank': r.get('col_rank',''),
            'row_margin': r.get('row_margin',''),
            'col_margin': r.get('col_margin',''),
            'max_high_sim_to_anchor': r.get('max_high_sim_to_anchor',''),
            'max_high_sim_to_post': r.get('max_high_sim_to_post',''),
            'max_high_sim_to_pre': r.get('max_high_sim_to_pre',''),
            'same_pre_direct_candidate_count': r.get('same_pre_direct_candidate_count',''),
            'same_pre_rank_by_high_sim': r.get('same_pre_rank_by_high_sim',''),
            'direct_margin_to_second': r.get('direct_margin_to_second',''),
        }
        rows.append(rec)
    return rows


def build_swap_candidates(proxy_rows):
    rows=[]
    for r in proxy_rows:
        tid=r.get('case','')
        if not tid:
            continue
        rec={
            'transaction_id': tid,
            'mode': 'swap_persistent_handoff',
            'candidate_family': 'swap',
            'candidate_pool_stage': 'stage1_existing_overlap_proxy_features_no_gt_scoring',
            'candidate_reason': 'nogt_swap_proxy_candidate',
            'track_result_path': str(Path(r.get('state_summary_path','')).parent/'track_results'/'MOT20-02.txt') if r.get('state_summary_path') else '',
            'source_run_dir': str(Path(r.get('state_summary_path','')).parent) if r.get('state_summary_path') else '',
            'tunnel_id': tid.split('_')[0] if '_' in tid else '',
            'track_a': r.get('track_a',''),
            'track_b': r.get('track_b',''),
            'pred_segments': r.get('pred_segments',''),
            'pred_segments_count': r.get('pred_segments_count',''),
            'pred_swap_frames': r.get('pred_swap_frames',''),
            'changed_rows': r.get('changed_rows',''),
            'proto_a_available': r.get('proto_a_available',''),
            'proto_b_available': r.get('proto_b_available',''),
            'a_pre_A_rows': r.get('a_pre_A_rows',''),
            'b_pre_B_rows': r.get('b_pre_B_rows',''),
            'a_swap_B_rows': r.get('a_swap_B_rows',''),
            'b_swap_A_rows': r.get('b_swap_A_rows',''),
            'a_post_B_rows': r.get('a_post_B_rows',''),
            'b_post_A_rows': r.get('b_post_A_rows',''),
            'a_pre_A_feat_rows': r.get('a_pre_A_feat_rows',''),
            'b_pre_B_feat_rows': r.get('b_pre_B_feat_rows',''),
            'a_swap_B_feat_rows': r.get('a_swap_B_feat_rows',''),
            'b_swap_A_feat_rows': r.get('b_swap_A_feat_rows',''),
            'a_post_B_feat_rows': r.get('a_post_B_feat_rows',''),
            'b_post_A_feat_rows': r.get('b_post_A_feat_rows',''),
            'a_pre_A_margin_mean': r.get('a_pre_A_margin_mean',''),
            'b_pre_B_margin_mean': r.get('b_pre_B_margin_mean',''),
            'a_swap_B_margin_mean': r.get('a_swap_B_margin_mean',''),
            'b_swap_A_margin_mean': r.get('b_swap_A_margin_mean',''),
            'a_post_B_margin_mean': r.get('a_post_B_margin_mean',''),
            'b_post_A_margin_mean': r.get('b_post_A_margin_mean',''),
            'boundary_min_sim': r.get('boundary_min_sim',''),
            'duplicate_covered_by_other': r.get('duplicate_covered_by_other',''),
            'covered_rows_by_other': r.get('covered_rows_by_other',''),
        }
        rows.append(rec)
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--path-transactions', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_04d_rule_mode_union_v2_broader_stress_and_combined/path_transaction_examples_stress_v4.csv')
    ap.add_argument('--swap-proxy-scored', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_06b_deployable_lifecycle_proxy_for_swap_transactions/swap_lifecycle_proxy_scored_cases.csv')
    ap.add_argument('--out-dir', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_06c_end_to_end_no_gt_transaction_replay_seq02')
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    path_rows=build_path_candidates(read_csv(Path(args.path_transactions)))
    swap_rows=build_swap_candidates(read_csv(Path(args.swap_proxy_scored)))
    all_rows=path_rows+swap_rows
    write_csv(out/'bridge_candidates_nogt.csv', [r for r in path_rows if r['mode']=='bridge'])
    write_csv(out/'direct_candidates_nogt.csv', [r for r in path_rows if r['mode']=='direct'])
    write_csv(out/'swap_candidates_nogt.csv', swap_rows)
    write_csv(out/'transaction_candidates_nogt.csv', all_rows)
    leakage = '''# A39_06c Stage-1 leakage audit\n\n## Status\n\nThis is a constrained no-GT scoring replay, not yet a full sequence-wide no-GT candidate scan.\n\nCandidate pools are reused from existing path-builder outputs and existing swap proxy feature outputs. The scorer in the next step must not use GT, TrackEval metrics, diagnostic wrong rows, or oracle fields for accept/reject.\n\n## Allowed fields for scoring\n\n- frame/track/bbox-derived path-builder features\n- selected_fragment_count, high_reid_fragment_count, bridge_fragment_count, gap_row_count\n- skipped_collision_rows\n- ReID similarities, ranks, margins\n- swap ReID lifecycle proxy margins\n- boundary_min_sim\n- duplicate-covered overlap against already accepted transaction rows\n\n## Forbidden fields for scoring\n\n- gt_id / pre_gt / post_gt / gt_same\n- true_reconnect / false_reconnect\n- diag_wrong_rows / wrong_after_swap_rows\n- HOTA / IDF1 / IDSW / MOTA / Frag\n- oracle_core_exit_rows / oracle_collision_rows\n- any field whose value requires GT matching\n\n## Important correction from A39_06b\n\nA39_06b's proxy gate implementation still checked wrong_after_swap_rows. A39_06c scorer removes that check entirely. Diagnostic wrong rows may appear only in final evaluation summaries, not in accept/reject logic.\n\n## Stage-1 limitation\n\nThe candidate pool itself is constrained by previous diagnostics and cached outputs. If this replay passes, the next required step is Stage-2 full-seq no-GT candidate scan.\n'''
    (out/'leakage_audit.md').write_text(leakage, encoding='utf-8')
    print(f'wrote {len(all_rows)} candidates: path={len(path_rows)} swap={len(swap_rows)} out={out}')

if __name__=='__main__':
    main()
