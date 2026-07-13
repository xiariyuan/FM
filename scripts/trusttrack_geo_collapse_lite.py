#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, math, os
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, field
import numpy as np

CUES=('motion','iou','shape')

def iou(a,b):
    xx1=max(a[0],b[0]); yy1=max(a[1],b[1]); xx2=min(a[2],b[2]); yy2=min(a[3],b[3])
    inter=max(0.0,xx2-xx1)*max(0.0,yy2-yy1)
    aa=max(1e-9,(a[2]-a[0])*(a[3]-a[1])); bb=max(1e-9,(b[2]-b[0])*(b[3]-b[1]))
    return inter/max(1e-9,aa+bb-inter)

def center(b): return ((b[0]+b[2])*0.5,(b[1]+b[3])*0.5)
def wh(b): return (max(1e-6,b[2]-b[0]),max(1e-6,b[3]-b[1]))

def load_results(path,max_frame):
    by=defaultdict(list)
    for line in open(path):
        v=line.strip().split(',')
        if len(v)<6: continue
        fr=int(float(v[0]))
        if max_frame and fr>max_frame: continue
        tid=int(float(v[1])); x=float(v[2]); y=float(v[3]); w=float(v[4]); h=float(v[5])
        by[fr].append((tid,(x,y,x+w,y+h)))
    return by

def load_gt(seq,root,max_frame):
    by=defaultdict(list)
    for line in open(os.path.join(root,seq,'gt','gt.txt')):
        v=line.strip().split(',')
        if len(v)<6: continue
        fr=int(float(v[0]))
        if max_frame and fr>max_frame: continue
        gid=int(float(v[1])); x=float(v[2]); y=float(v[3]); w=float(v[4]); h=float(v[5])
        mark=int(float(v[6])) if len(v)>6 else 1; cls=int(float(v[7])) if len(v)>7 else 1
        if mark==1 and cls==1: by[fr].append((gid,(x,y,x+w,y+h)))
    return by

def best_gt(fr,box,gt):
    best=(-1,0.0)
    for gid,gbox in gt.get(fr,[]):
        ov=iou(box,gbox)
        if ov>best[1]: best=(gid,ov)
    return best if best[1]>=0.5 else (-1,best[1])

def majority(hist):
    vals=[x for x in hist if x!=-1]
    return Counter(vals).most_common(1)[0][0] if vals else -1

@dataclass
class State:
    box: tuple
    gt_hist: deque = field(default_factory=lambda: deque(maxlen=50))

def scores(st,boxes):
    out={c:np.zeros(len(boxes),dtype=np.float32) for c in CUES}
    pcx,pcy=center(st.box); pw,ph=wh(st.box); scale=max(1.0,math.sqrt(pw*ph)); asp0=pw/ph
    for i,b in enumerate(boxes):
        cx,cy=center(b); dw,dh=wh(b); asp=dw/dh
        out['motion'][i]=math.exp(-math.hypot(cx-pcx,cy-pcy)/scale)
        out['iou'][i]=iou(st.box,b)
        out['shape'][i]=math.exp(-(abs(math.log(dh/ph))+abs(math.log(max(asp,1e-6)/max(asp0,1e-6)))))
    return out

def row_margin(arr,j):
    if len(arr)<=1: return 1.0
    chosen=float(arr[j]); other=max(float(arr[k]) for k in range(len(arr)) if k!=j)
    return chosen-other

def col_margin(by,c,j,tid):
    chosen=float(by[tid][c][j])
    vals=[float(v[c][j]) for t,v in by.items() if t!=tid and len(v[c])>j]
    return 1.0 if not vals else chosen-max(vals)

def run(args):
    dump=np.load(args.dump_npz,allow_pickle=True)
    det=dump['detections']; cols={str(x):i for i,x in enumerate(dump['columns'].tolist())}
    det_by=defaultdict(list)
    for idx,r in enumerate(det):
        fr=int(r[cols['frame']])
        if args.max_frame and fr>args.max_frame: continue
        if float(r[cols['score']])>=args.min_det_score: det_by[fr].append(idx)
    res=load_results(args.result_txt,args.max_frame); gt=load_gt(args.seq,args.gt_root,args.max_frame)
    states={}; rows=[]
    for fr in sorted(res.keys()):
        boxes=[]; ds=[]
        for idx in det_by.get(fr,[]):
            r=det[idx]
            boxes.append((float(r[cols['x1']]),float(r[cols['y1']]),float(r[cols['x2']]),float(r[cols['y2']])))
            ds.append(float(r[cols['score']]))
        cue_by={tid:scores(st,boxes) for tid,st in states.items()}
        updates=[]
        for tid,obox in res[fr]:
            if not boxes: continue
            ovs=[iou(obox,b) for b in boxes]; j=int(np.argmax(ovs)); ov=float(ovs[j])
            if ov<args.match_iou: continue
            dg,di=best_gt(fr,boxes[j],gt)
            if tid in states and cue_by:
                prev=majority(states[tid].gt_hist)
                pair={c:min(row_margin(cue_by[tid][c],j),col_margin(cue_by,c,j,tid)) for c in CUES}
                trust=max(pair.values()); collapse=1.0-trust
                cls='det_fp' if dg==-1 else ('unknown' if prev==-1 else ('correct' if dg==prev else 'wrong_id'))
                rows.append({'frame':fr,'track_id':tid,'det_score':ds[j],'match_cls':cls,'det_gt':dg,'prev_gt':prev,'trust':trust,'cue_collapse':collapse,**{f'{c}_pair_margin':pair[c] for c in CUES}})
            updates.append((tid,boxes[j],dg))
        for tid,b,dg in updates:
            if tid not in states: states[tid]=State(b)
            else: states[tid].box=b
            states[tid].gt_hist.append(dg)
    os.makedirs(os.path.dirname(args.out_csv),exist_ok=True)
    fields=['frame','track_id','det_score','match_cls','det_gt','prev_gt','trust','cue_collapse']+[f'{c}_pair_margin' for c in CUES]
    with open(args.out_csv,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print('wrote',args.out_csv,'rows',len(rows),'counts',dict(Counter(r['match_cls'] for r in rows)))
    vals=np.array([r['cue_collapse'] for r in rows],float)
    if len(vals):
        qs=np.quantile(vals,[0,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1])
        for lo,hi in zip(qs[:-1],qs[1:]):
            sub=[r for r in rows if r['cue_collapse']>=lo and r['cue_collapse']<=hi]
            bad=sum(r['match_cls'] in ('wrong_id','det_fp') for r in sub)
            print('collapse_bin',round(float(lo),4),round(float(hi),4),'n',len(sub),'wrongfp_rate',round(bad/max(1,len(sub)),4))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seq',required=True); ap.add_argument('--dump-npz',required=True); ap.add_argument('--result-txt',required=True); ap.add_argument('--out-csv',required=True)
    ap.add_argument('--gt-root',default='/gemini/code/datasets/MOT20/train')
    ap.add_argument('--max-frame',type=int,default=800); ap.add_argument('--min-det-score',type=float,default=0.1); ap.add_argument('--match-iou',type=float,default=0.8)
    run(ap.parse_args())
if __name__=='__main__': main()
