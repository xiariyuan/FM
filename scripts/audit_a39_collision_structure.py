#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


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
        w=csv.DictWriter(f, fieldnames=fields or ['anchor_id'], extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def read_track(path: Path):
    rows=[]; by_frame=defaultdict(list); by_frame_tid=defaultdict(list)
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            p=line.strip().split(',')
            if len(p)<6:
                continue
            fr=ai(p[0],-1); tid=ai(p[1],-1); x=af(p[2]); y=af(p[3]); w=af(p[4]); h=af(p[5]); score=af(p[6],1.0) if len(p)>6 else 1.0
            if fr<0 or tid<0 or w<=0 or h<=0:
                continue
            r={'idx':len(rows),'frame':fr,'track_id':tid,'x':x,'y':y,'w':w,'h':h,'score':score,'box':np.array([x,y,x+w,y+h],dtype=np.float32)}
            r['cx']=x+w/2.0; r['bottom_y']=y+h; r['height']=h
            rows.append(r); by_frame[fr].append(r); by_frame_tid[(fr,tid)].append(r)
    return rows, by_frame, by_frame_tid


def read_gt(path: Path):
    by_frame=defaultdict(list)
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            p=line.strip().split(',')
            if len(p)<6:
                continue
            fr=ai(p[0],-1); gid=ai(p[1],-1); x=af(p[2]); y=af(p[3]); w=af(p[4]); h=af(p[5])
            mark=ai(p[6],1) if len(p)>6 else 1
            cls=ai(p[7],1) if len(p)>7 else 1
            if fr<0 or gid<0 or w<=0 or h<=0 or mark<=0 or cls!=1:
                continue
            by_frame[fr].append({'frame':fr,'gt_id':gid,'box':np.array([x,y,x+w,y+h],dtype=np.float32)})
    return by_frame


def iou_box(a, b):
    lt=np.maximum(a[:2], b[:2]); rb=np.minimum(a[2:], b[2:]); wh=np.clip(rb-lt,0,None); inter=float(wh[0]*wh[1])
    aa=max(float((a[2]-a[0])*(a[3]-a[1])),1e-6); bb=max(float((b[2]-b[0])*(b[3]-b[1])),1e-6)
    return inter/max(aa+bb-inter,1e-6)


def pair_iou(a, b):
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    A=np.stack([x['box'] for x in a]); B=np.stack([x['box'] for x in b])
    lt=np.maximum(A[:,None,:2], B[None,:,:2]); rb=np.minimum(A[:,None,2:], B[None,:,2:])
    wh=np.clip(rb-lt,0,None); inter=wh[...,0]*wh[...,1]
    aa=np.clip((A[:,2]-A[:,0])*(A[:,3]-A[:,1]),1e-6,None); bb=np.clip((B[:,2]-B[:,0])*(B[:,3]-B[:,1]),1e-6,None)
    return inter/np.clip(aa[:,None]+bb[None,:]-inter,1e-6,None)


def match_rows(rows_by_frame, gt_by_frame, iou_thr):
    row_gt={}; row_iou={}
    for fr in sorted(set(rows_by_frame)|set(gt_by_frame)):
        rr=rows_by_frame.get(fr,[]); gg=gt_by_frame.get(fr,[])
        if not rr or not gg:
            continue
        I=pair_iou(rr,gg); ri,ci=linear_sum_assignment(1-I)
        for r,c in zip(ri,ci):
            val=float(I[r,c])
            if val>=iou_thr:
                row_gt[rr[r]['idx']]=int(gg[c]['gt_id']); row_iou[rr[r]['idx']]=val
    return row_gt,row_iou


def relation(anchor_gt, cand_gt, target_gt):
    cand_known=cand_gt>=0; target_known=target_gt>=0
    cand_anchor=cand_known and cand_gt==anchor_gt
    target_anchor=target_known and target_gt==anchor_gt
    if cand_anchor and target_anchor:
        return 'same_gt_duplicate'
    if cand_anchor and target_known and not target_anchor:
        return 'candidate_anchor_target_other'
    if target_anchor and cand_known and not cand_anchor:
        return 'target_anchor_candidate_other'
    if cand_known and target_known and cand_gt==target_gt:
        return 'same_non_anchor_gt'
    if cand_known and target_known and cand_gt!=target_gt:
        return 'different_gt_conflict'
    if cand_anchor and not target_known:
        return 'candidate_anchor_target_unknown'
    if target_anchor and not cand_known:
        return 'target_anchor_candidate_unknown'
    return 'unknown_collision'


def classify_anchor(counter, total):
    if total<=0:
        return 'NO_COLLISION_ROWS'
    ratios={k:safe_div(v,total) for k,v in counter.items()}
    if ratios.get('same_gt_duplicate',0)>=0.8:
        return 'SAME_GT_DUPLICATE_DOMINANT'
    if ratios.get('candidate_anchor_target_other',0)>=0.8:
        return 'CANDIDATE_ANCHOR_TARGET_OTHER_DOMINANT'
    if ratios.get('different_gt_conflict',0)>=0.8:
        return 'DIFFERENT_GT_CONFLICT_DOMINANT'
    if ratios.get('candidate_anchor_target_unknown',0)+ratios.get('unknown_collision',0)>=0.5:
        return 'UNKNOWN_HEAVY'
    return 'MIXED_COLLISION'


def main():
    ap=argparse.ArgumentParser(description='A39_05a collision structure audit for collision-blocked true reconnects.')
    ap.add_argument('--track-file', default='outputs/spot_runtime_gate_20260628/A37_identity_escrow_association/A37_00b_candidate_logger_seq02/track_results/MOT20-02.txt')
    ap.add_argument('--gt-file', default='datasets/MOT20/train/MOT20-02/gt/gt.txt')
    ap.add_argument('--top-fragment-audit', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_04e_safe_path_mining_and_positive_gap_audit/true_reconnect_top1_fragment_summary.csv')
    ap.add_argument('--out-dir', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_05a_collision_structure_audit_true_reconnects')
    ap.add_argument('--focus-anchors', default='123_199_218,215_469_508,106_150_169,188_520_521')
    ap.add_argument('--audit-all-collision-blocked', action='store_true')
    ap.add_argument('--iou-thr', type=float, default=0.5)
    args=ap.parse_args()

    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    track_rows, rows_by_frame, by_frame_tid=read_track(Path(args.track_file))
    gt_by_frame=read_gt(Path(args.gt_file))
    row_gt,row_iou=match_rows(rows_by_frame, gt_by_frame, args.iou_thr)
    top_rows=read_csv(Path(args.top_fragment_audit))
    focus=set(x.strip() for x in args.focus_anchors.split(',') if x.strip())
    cases=[]
    for r in top_rows:
        if r.get('failure_reason')=='SAFE_PATH':
            continue
        if ai(r.get('gt_same_as_anchor'))!=1:
            continue
        if af(r.get('collision_ratio_if_rewrite'))<=0:
            continue
        if args.audit_all_collision_blocked or r.get('anchor_id') in focus:
            cases.append(r)
    row_out=[]; anchor_out=[]
    for c in cases:
        aid=c['anchor_id']; anchor_gt=ai(c.get('gt_id'),-1); target_id=ai(c.get('pre_track'),-1); cand_id=ai(c.get('track_id'),-1)
        f0=ai(c.get('frame_start')); f1=ai(c.get('frame_end'))
        rel_counter=Counter(); cand_gt_counter=Counter(); target_gt_counter=Counter(); frames=set(); collision_pairs=0
        for fr in range(f0,f1+1):
            cand_rows=by_frame_tid.get((fr,cand_id),[]); target_rows=by_frame_tid.get((fr,target_id),[])
            if not cand_rows or not target_rows:
                continue
            frames.add(fr)
            for cr in cand_rows:
                for tr in target_rows:
                    cand_gt=row_gt.get(cr['idx'],-1); target_gt=row_gt.get(tr['idx'],-1)
                    cand_iou=row_iou.get(cr['idx'],0.0); target_iou=row_iou.get(tr['idx'],0.0)
                    rel=relation(anchor_gt,cand_gt,target_gt)
                    rel_counter[rel]+=1; cand_gt_counter[cand_gt]+=1; target_gt_counter[target_gt]+=1; collision_pairs+=1
                    row_out.append({
                        'anchor_id':aid,'frame':fr,'anchor_gt':anchor_gt,
                        'candidate_track_id':cand_id,'target_track_id':target_id,
                        'candidate_idx':cr['idx'],'target_idx':tr['idx'],
                        'candidate_gt':cand_gt,'target_gt':target_gt,'relation':rel,
                        'candidate_is_anchor_gt':int(cand_gt==anchor_gt),'target_is_anchor_gt':int(target_gt==anchor_gt),
                        'candidate_gt_iou':cand_iou,'target_gt_iou':target_iou,
                        'candidate_target_box_iou':iou_box(cr['box'], tr['box']),
                        'candidate_score':cr['score'],'target_score':tr['score'],
                        'candidate_cx':cr['cx'],'target_cx':tr['cx'],
                        'candidate_bottom_y':cr['bottom_y'],'target_bottom_y':tr['bottom_y'],
                        'candidate_height':cr['height'],'target_height':tr['height'],
                    })
        total=collision_pairs
        cand_major,cand_major_n=(cand_gt_counter.most_common(1)[0] if cand_gt_counter else (-1,0))
        targ_major,targ_major_n=(target_gt_counter.most_common(1)[0] if target_gt_counter else (-1,0))
        rec={
            'anchor_id':aid,'tunnel_id':c.get('tunnel_id'),'anchor_gt':anchor_gt,'target_id':target_id,'candidate_track_id':cand_id,
            'fragment_key':c.get('fragment_key'),'fragment_rows':ai(c.get('rows')),'frame_start':f0,'frame_end':f1,
            'max_sim':af(c.get('max_sim')),'sim_to_anchor':af(c.get('sim_to_anchor')),'sim_to_pre':af(c.get('sim_to_pre')),'sim_to_post':af(c.get('sim_to_post')),
            'collision_ratio_if_rewrite':af(c.get('collision_ratio_if_rewrite')),'collision_rows_if_rewrite':ai(c.get('collision_rows_if_rewrite')),
            'collision_frames':len(frames),'collision_pairs':total,
            'same_gt_duplicate_rows':rel_counter.get('same_gt_duplicate',0),
            'candidate_anchor_target_other_rows':rel_counter.get('candidate_anchor_target_other',0),
            'target_anchor_candidate_other_rows':rel_counter.get('target_anchor_candidate_other',0),
            'different_gt_conflict_rows':rel_counter.get('different_gt_conflict',0),
            'unknown_collision_rows':sum(v for k,v in rel_counter.items() if 'unknown' in k),
            'same_gt_duplicate_ratio':safe_div(rel_counter.get('same_gt_duplicate',0),total),
            'candidate_anchor_target_other_ratio':safe_div(rel_counter.get('candidate_anchor_target_other',0),total),
            'different_gt_conflict_ratio':safe_div(rel_counter.get('different_gt_conflict',0),total),
            'candidate_collision_major_gt':cand_major,'candidate_collision_major_purity':safe_div(cand_major_n,total),
            'target_collision_major_gt':targ_major,'target_collision_major_purity':safe_div(targ_major_n,total),
            'relation_counts_json':json.dumps(dict(rel_counter), sort_keys=True),
        }
        rec['collision_class']=classify_anchor(rel_counter,total)
        anchor_out.append(rec)
    write_csv(out/'collision_structure_by_row.csv', row_out)
    write_csv(out/'collision_structure_by_anchor.csv', anchor_out)
    summary_counter=Counter(r['collision_class'] for r in anchor_out)
    relation_total=Counter()
    for r in anchor_out:
        relation_total.update(json.loads(r['relation_counts_json']))
    payload={'case_count':len(anchor_out),'row_count':len(row_out),'class_summary':dict(summary_counter),'relation_total':dict(relation_total)}
    (out/'summary.json').write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n')
    md=['# A39_05a Collision Structure Audit','','## Summary','','```json',json.dumps(payload, indent=2, sort_keys=True),'```','','## Anchor collision classes','','| anchor | class | collision_pairs | same_dup | cand_anchor_target_other | diff_gt | unknown | cand_major_gt | target_major_gt |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in anchor_out:
        md.append(f"| {r['anchor_id']} | {r['collision_class']} | {r['collision_pairs']} | {r['same_gt_duplicate_rows']} | {r['candidate_anchor_target_other_rows']} | {r['different_gt_conflict_rows']} | {r['unknown_collision_rows']} | {r['candidate_collision_major_gt']} | {r['target_collision_major_gt']} |")
    (out/'decision.md').write_text('\n'.join(md)+'\n')
    print('\n'.join(md))

if __name__=='__main__':
    main()
