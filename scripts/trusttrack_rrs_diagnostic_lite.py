#!/usr/bin/env python3
"""Lightweight cue-collapse diagnostic with a frame limit.

This is a small, non-invasive diagnostic script. It reconstructs track states
from an existing MOT result file and detection dump, computes row/col/pair
margins for appearance, motion, IoU, and shape cues, and summarizes whether cue
collapse predicts bad matches. It does not change tracker outputs.
"""
from __future__ import annotations
import argparse, csv, math, os
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, field
import numpy as np

CUES = ('app', 'motion', 'iou', 'shape')


def iou(a, b):
    xx1=max(a[0],b[0]); yy1=max(a[1],b[1]); xx2=min(a[2],b[2]); yy2=min(a[3],b[3])
    inter=max(0.0,xx2-xx1)*max(0.0,yy2-yy1)
    aa=max(1e-9,(a[2]-a[0])*(a[3]-a[1])); bb=max(1e-9,(b[2]-b[0])*(b[3]-b[1]))
    return inter/max(1e-9, aa+bb-inter)


def center(b):
    return ((b[0]+b[2])*0.5, (b[1]+b[3])*0.5)


def wh(b):
    return (max(1e-6,b[2]-b[0]), max(1e-6,b[3]-b[1]))


def load_results(path, max_frame):
    by=defaultdict(list)
    for line in open(path):
        v=line.strip().split(',')
        if len(v)<6: continue
        fr=int(float(v[0]))
        if max_frame and fr>max_frame: continue
        tid=int(float(v[1])); x=float(v[2]); y=float(v[3]); w=float(v[4]); h=float(v[5])
        by[fr].append((tid,(x,y,x+w,y+h)))
    return by


def load_gt(seq, root, max_frame):
    by=defaultdict(list)
    for line in open(os.path.join(root, seq, 'gt', 'gt.txt')):
        v=line.strip().split(',')
        if len(v)<6: continue
        fr=int(float(v[0]))
        if max_frame and fr>max_frame: continue
        gid=int(float(v[1])); x=float(v[2]); y=float(v[3]); w=float(v[4]); h=float(v[5])
        mark=int(float(v[6])) if len(v)>6 else 1
        cls=int(float(v[7])) if len(v)>7 else 1
        if mark==1 and cls==1:
            by[fr].append((gid,(x,y,x+w,y+h)))
    return by


def best_gt(fr, box, gt):
    best=(-1,0.0)
    for gid,gbox in gt.get(fr,[]):
        ov=iou(box,gbox)
        if ov>best[1]: best=(gid,ov)
    return best if best[1]>=0.5 else (-1,best[1])


def majority_gt(hist):
    vals=[x for x in hist if x!=-1]
    return Counter(vals).most_common(1)[0][0] if vals else -1


@dataclass
class State:
    box: tuple
    feat: np.ndarray
    gt_hist: deque = field(default_factory=lambda: deque(maxlen=50))


def cue_scores(state, boxes, feats):
    n=len(boxes)
    out={c:np.zeros(n,dtype=np.float32) for c in CUES}
    if n==0: return out
    out['app']=feats @ state.feat
    pcx,pcy=center(state.box); pw,ph=wh(state.box); scale=max(1.0, math.sqrt(pw*ph))
    aspect_prev=pw/ph
    for i,b in enumerate(boxes):
        cx,cy=center(b); dw,dh=wh(b); aspect=dw/dh
        out['motion'][i]=math.exp(-math.hypot(cx-pcx,cy-pcy)/scale)
        out['iou'][i]=iou(state.box,b)
        out['shape'][i]=math.exp(-(abs(math.log(dh/ph))+abs(math.log(max(aspect,1e-6)/max(aspect_prev,1e-6)))))
    return out


def row_margin(scores, j):
    if len(scores)<=1: return 1.0
    chosen=float(scores[j]); other=max(float(scores[k]) for k in range(len(scores)) if k!=j)
    return chosen-other


def col_margin(cue_by_track, cue, j, tid):
    chosen=float(cue_by_track[tid][cue][j])
    vals=[float(v[cue][j]) for t,v in cue_by_track.items() if t!=tid and len(v[cue])>j]
    if not vals: return 1.0
    return chosen-max(vals)


def run(args):
    dump=np.load(args.dump_npz, allow_pickle=True)
    det=dump['detections']; feats=dump['features'].astype(np.float32)
    cols={str(x):i for i,x in enumerate(dump['columns'].tolist())}
    feats=feats/np.maximum(np.linalg.norm(feats,axis=1,keepdims=True),1e-12)
    det_by=defaultdict(list)
    for idx,r in enumerate(det):
        fr=int(r[cols['frame']])
        if args.max_frame and fr>args.max_frame: continue
        if float(r[cols['score']])>=args.min_det_score:
            det_by[fr].append(idx)
    res=load_results(args.result_txt,args.max_frame)
    gt=load_gt(args.seq,args.gt_root,args.max_frame)
    states={}; rows=[]
    for fr in sorted(res.keys()):
        idxs=det_by.get(fr,[])
        boxes=[]; scores=[]; gids=[]
        for idx in idxs:
            r=det[idx]
            boxes.append((float(r[cols['x1']]),float(r[cols['y1']]),float(r[cols['x2']]),float(r[cols['y2']])))
            scores.append(float(r[cols['score']]))
            gids.append(int(r[cols['global_det_idx']]))
        fmat=feats[idxs] if idxs else np.zeros((0,feats.shape[1]),dtype=np.float32)
        cue_by={tid:cue_scores(st,boxes,fmat) for tid,st in states.items()}
        updates=[]
        for tid,obox in res[fr]:
            if not boxes:
                continue
            ovs=[iou(obox,b) for b in boxes]
            j=int(np.argmax(ovs)); ov=float(ovs[j])
            if ov<args.match_iou:
                continue
            dg,di=best_gt(fr,boxes[j],gt)
            if tid not in states or not cue_by:
                updates.append((tid,boxes[j],fmat[j],dg)); continue
            prev_gt=majority_gt(states[tid].gt_hist)
            pair={}
            for c in CUES:
                rm=row_margin(cue_by[tid][c],j)
                cm=col_margin(cue_by,c,j,tid)
                pair[c]=min(rm,cm)
            trust=max(pair.values())
            collapse=1.0-trust
            cls='det_fp' if dg==-1 else ('unknown' if prev_gt==-1 else ('correct' if dg==prev_gt else 'wrong_id'))
            row={'frame':fr,'track_id':tid,'det_score':scores[j],'det_gt':dg,'det_iou':di,'prev_track_gt':prev_gt,'match_cls':cls,'trust':trust,'cue_collapse':collapse}
            for c in CUES:
                row[f'{c}_pair_margin']=pair[c]
            rows.append(row)
            updates.append((tid,boxes[j],fmat[j],dg))
        for tid,b,feat,dg in updates:
            nf=feat.copy()
            if tid in states:
                nf=0.9*states[tid].feat+0.1*feat
                nf=nf/max(float(np.linalg.norm(nf)),1e-12)
                states[tid].box=b; states[tid].feat=nf; states[tid].gt_hist.append(dg)
            else:
                states[tid]=State(b,nf); states[tid].gt_hist.append(dg)
    os.makedirs(os.path.dirname(args.out_csv),exist_ok=True)
    fields=['frame','track_id','det_score','det_gt','det_iou','prev_track_gt','match_cls','trust','cue_collapse']+[f'{c}_pair_margin' for c in CUES]
    with open(args.out_csv,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print('wrote',args.out_csv,'rows',len(rows),'class_counts',dict(Counter(r['match_cls'] for r in rows)))
    vals=np.array([float(r['cue_collapse']) for r in rows],dtype=float)
    if len(vals):
        qs=np.quantile(vals,[0,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1])
        for lo,hi in zip(qs[:-1],qs[1:]):
            sub=[r for r in rows if float(r['cue_collapse'])>=lo and float(r['cue_collapse'])<=hi]
            if sub:
                bad=sum(r['match_cls'] in ('wrong_id','det_fp') for r in sub)
                print('collapse_bin',round(float(lo),4),round(float(hi),4),'n',len(sub),'wrongfp_rate',round(bad/len(sub),4))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seq',required=True)
    ap.add_argument('--dump-npz',required=True)
    ap.add_argument('--result-txt',required=True)
    ap.add_argument('--out-csv',required=True)
    ap.add_argument('--gt-root',default='/gemini/code/datasets/MOT20/train')
    ap.add_argument('--max-frame',type=int,default=800)
    ap.add_argument('--min-det-score',type=float,default=0.1)
    ap.add_argument('--match-iou',type=float,default=0.8)
    args=ap.parse_args(); run(args)
if __name__=='__main__': main()
