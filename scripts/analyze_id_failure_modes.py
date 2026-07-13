#!/usr/bin/env python3
"""Decompose MOT identity errors using framewise GT/tracker matching.

Offline diagnostic only. It matches valid pedestrian GT and tracker boxes per
frame at IoU >= threshold, then analyzes GT-to-tracker ID transitions:
- contiguous active switches;
- short/medium/long-gap re-identification changes;
- whether the old/new tracker ID is simultaneously attached to another GT;
- tracker-ID purity across the sequence.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

Box = Tuple[float, float, float, float]


def xywh_to_xyxy(x, y, w, h):
    return (float(x), float(y), float(x) + float(w), float(y) + float(h))


def iou_matrix(a: Sequence[Box], b: Sequence[Box]) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    aa=np.asarray(a,dtype=np.float64); bb=np.asarray(b,dtype=np.float64)
    xx1=np.maximum(aa[:,None,0],bb[None,:,0]); yy1=np.maximum(aa[:,None,1],bb[None,:,1])
    xx2=np.minimum(aa[:,None,2],bb[None,:,2]); yy2=np.minimum(aa[:,None,3],bb[None,:,3])
    inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
    area_a=np.maximum(0,aa[:,2]-aa[:,0])*np.maximum(0,aa[:,3]-aa[:,1])
    area_b=np.maximum(0,bb[:,2]-bb[:,0])*np.maximum(0,bb[:,3]-bb[:,1])
    return inter/np.maximum(area_a[:,None]+area_b[None,:]-inter,1e-12)


def load_gt(path: Path):
    by=defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        p=line.split(','); fr=int(float(p[0])); gid=int(float(p[1])); x,y,w,h=map(float,p[2:6])
        mark=int(float(p[6])) if len(p)>6 else 1; cls=int(float(p[7])) if len(p)>7 else 1
        if mark!=0 and cls==1: by[fr].append((gid,xywh_to_xyxy(x,y,w,h)))
    return by


def load_tracker(path: Path):
    by=defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        p=line.split(','); fr=int(float(p[0])); tid=int(float(p[1])); x,y,w,h=map(float,p[2:6])
        by[fr].append((tid,xywh_to_xyxy(x,y,w,h)))
    return by


def gap_bucket(gap: int) -> str:
    if gap == 0: return 'contiguous'
    if gap <= 2: return 'micro_gap_1_2'
    if gap <= 5: return 'short_gap_3_5'
    if gap <= 10: return 'short_gap_6_10'
    if gap <= 30: return 'medium_gap_11_30'
    return 'long_gap_gt30'


def analyze(gt_by, tr_by, threshold: float):
    gt_history=defaultdict(list)  # gid -> (frame, tid, iou)
    tracker_gt_counts=defaultdict(Counter)
    frame_gt_to_tid={}; frame_tid_to_gt={}
    matched=0
    for fr in sorted(set(gt_by)|set(tr_by)):
        gs=gt_by.get(fr,[]); ts=tr_by.get(fr,[])
        ious=iou_matrix([x[1] for x in gs],[x[1] for x in ts])
        g2t={}; t2g={}
        if ious.size:
            rr,cc=linear_sum_assignment(-ious)
            for r,c in zip(rr.tolist(),cc.tolist()):
                ov=float(ious[r,c])
                if ov+np.finfo(float).eps < threshold: continue
                gid=gs[r][0]; tid=ts[c][0]
                g2t[gid]=tid; t2g[tid]=gid
                gt_history[gid].append((fr,tid,ov))
                tracker_gt_counts[tid][gid]+=1
                matched+=1
        frame_gt_to_tid[fr]=g2t; frame_tid_to_gt[fr]=t2g

    transitions=[]; bucket_counts=Counter(); type_counts=Counter()
    for gid,hist in gt_history.items():
        hist.sort()
        prev=None
        for cur in hist:
            if prev is not None and cur[1] != prev[1]:
                gap=cur[0]-prev[0]-1; bucket=gap_bucket(gap)
                old_tid=prev[1]; new_tid=cur[1]
                old_now=frame_tid_to_gt.get(cur[0],{}).get(old_tid)
                new_prev=frame_tid_to_gt.get(prev[0],{}).get(new_tid)
                if old_now is not None and old_now != gid and new_prev is not None and new_prev != gid:
                    kind='two_way_swap'
                elif old_now is not None and old_now != gid:
                    kind='old_id_taken_by_other'
                elif new_prev is not None and new_prev != gid:
                    kind='new_id_came_from_other'
                else:
                    kind='rebirth_or_unmatched_handoff'
                bucket_counts[bucket]+=1; type_counts[kind]+=1
                transitions.append({
                    'gt_id':gid,'prev_frame':prev[0],'frame':cur[0],'gap':gap,'bucket':bucket,
                    'old_tracker_id':old_tid,'new_tracker_id':new_tid,'kind':kind,
                    'prev_iou':prev[2],'current_iou':cur[2],
                    'old_tracker_current_gt':old_now,'new_tracker_previous_gt':new_prev,
                })
            prev=cur

    purity=[]
    for tid,cnt in tracker_gt_counts.items():
        total=sum(cnt.values()); gid,n=cnt.most_common(1)[0]
        purity.append((tid,total,gid,n/total,len(cnt)))
    weighted_purity=sum(total*p for _,total,_,p,_ in purity)/max(1,sum(total for _,total,_,_,_ in purity))
    impure=[x for x in purity if x[4]>1]
    impure.sort(key=lambda x:(x[3],-x[1]))

    return {
        'matched_pairs':matched,
        'gt_ids_with_matches':len(gt_history),
        'id_change_events':len(transitions),
        'gap_buckets':dict(bucket_counts),
        'transition_types':dict(type_counts),
        'tracker_ids_with_gt_matches':len(purity),
        'tracker_ids_matching_multiple_gt':len(impure),
        'weighted_tracker_id_purity':weighted_purity,
        'worst_impure_tracker_ids':[
            {'tracker_id':tid,'matched_frames':total,'dominant_gt':gid,'purity':p,'num_gt_ids':ng}
            for tid,total,gid,p,ng in impure[:30]
        ],
        'transitions':transitions,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--result-txt',required=True); ap.add_argument('--gt-txt',required=True)
    ap.add_argument('--out-json',required=True); ap.add_argument('--transitions-csv',default='')
    ap.add_argument('--iou-threshold',type=float,default=0.5)
    args=ap.parse_args()
    result=analyze(load_gt(Path(args.gt_txt)),load_tracker(Path(args.result_txt)),args.iou_threshold)
    out=Path(args.out_json); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    if args.transitions_csv:
        import csv
        p=Path(args.transitions_csv); p.parent.mkdir(parents=True,exist_ok=True)
        rows=result['transitions']; fields=sorted({k for r in rows for k in r}) if rows else []
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary={k:v for k,v in result.items() if k not in ('transitions','worst_impure_tracker_ids')}
    summary['worst_impure_tracker_ids']=result['worst_impure_tracker_ids'][:10]
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__': main()
