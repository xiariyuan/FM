#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

def fnum(x,d=0.0):
    try: return float(x)
    except: return d

def inum(x,d=0):
    try: return int(float(x))
    except: return d

def read_mot(path, is_gt=False):
    by=defaultdict(list)
    with open(path,encoding='utf-8',errors='ignore') as fh:
        for line in fh:
            if not line.strip(): continue
            p=line.strip().split(',')
            if len(p)<6: continue
            fr=inum(p[0]); tid=inum(p[1]); x=fnum(p[2]); y=fnum(p[3]); w=fnum(p[4]); h=fnum(p[5])
            if w<=0 or h<=0: continue
            if is_gt:
                mark=inum(p[6],1) if len(p)>6 else 1; cls=inum(p[7],1) if len(p)>7 else 1
                if mark==0 or cls!=1: continue
            by[fr].append({'frame':fr,'id':tid,'box':np.array([x,y,x+w,y+h],dtype=np.float32)})
    return by

def iou_mat(a,b):
    if not a or not b: return np.zeros((len(a),len(b)),dtype=np.float32)
    A=np.stack([x['box'] for x in a]); B=np.stack([x['box'] for x in b])
    lt=np.maximum(A[:,None,:2],B[None,:,:2]); rb=np.minimum(A[:,None,2:],B[None,:,2:])
    wh=np.clip(rb-lt,0,None); inter=wh[...,0]*wh[...,1]
    aa=np.clip((A[:,2]-A[:,0])*(A[:,3]-A[:,1]),1e-6,None); bb=np.clip((B[:,2]-B[:,0])*(B[:,3]-B[:,1]),1e-6,None)
    return inter/np.clip(aa[:,None]+bb[None,:]-inter,1e-6,None)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gt-file',required=True); ap.add_argument('--track-file',required=True); ap.add_argument('--out-csv',required=True); ap.add_argument('--iou-thr',type=float,default=0.5)
    args=ap.parse_args()
    gt=read_mot(args.gt_file,True); tr=read_mot(args.track_file,False)
    prev={}; rows=[]; total_matches=0
    for fr in sorted(set(gt)|set(tr)):
        g=gt.get(fr,[]); d=tr.get(fr,[])
        if not g or not d: continue
        M=iou_mat(g,d); rr,cc=linear_sum_assignment(-M)
        for gi,di in zip(rr,cc):
            if float(M[gi,di])<args.iou_thr: continue
            gid=g[gi]['id']; tid=d[di]['id']; total_matches+=1
            old=prev.get(gid)
            is_switch= int(old is not None and old!=tid)
            if is_switch:
                rows.append({'frame':fr,'chosen_tid':tid,'det_gt':gid,'is_gt_idsw':1,'is_track_switch':1,'is_bad_commit_before':1,'low_margin_005':0,'prev_tid':old})
            prev[gid]=tid
    out=Path(args.out_csv); out.parent.mkdir(parents=True,exist_ok=True)
    fields=['frame','chosen_tid','det_gt','is_gt_idsw','is_track_switch','is_bad_commit_before','low_margin_005','prev_tid']
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print({'events':len(rows),'total_matches':total_matches,'out':str(out)})
if __name__=='__main__': main()
