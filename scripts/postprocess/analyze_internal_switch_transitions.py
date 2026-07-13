#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def ff(x, d=0.0):
    try: return float(x)
    except Exception: return d

def ii(x, d=0):
    try: return int(float(x))
    except Exception: return d

def iou(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    ax2,ay2=ax+aw,ay+ah; bx2,by2=bx+bw,by+bh
    iw=max(0,min(ax2,bx2)-max(ax,bx)); ih=max(0,min(ay2,by2)-max(ay,by))
    inter=iw*ih
    return inter/max(1e-9,aw*ah+bw*bh-inter) if inter>0 else 0.0

def center(r): return (r['x']+r['w']/2,r['y']+r['h']/2)
def area(r): return max(1e-9,r['w']*r['h'])

def load_gt(path):
    by=defaultdict(list)
    with path.open() as f:
        for line in f:
            p=line.strip().split(',')
            if len(p)<6: continue
            fr=ii(p[0]); gid=ii(p[1]); x,y,w,h=map(ff,p[2:6])
            mark=ii(p[6],1) if len(p)>6 else 1; cls=ii(p[7],1) if len(p)>7 else 1
            if mark==1 and cls==1: by[fr].append((gid,(x,y,w,h)))
    return by

def load_pred(path):
    tracks=defaultdict(list)
    with path.open() as f:
        for line in f:
            p=line.strip().split(',')
            if len(p)<6: continue
            r={'frame':ii(p[0]),'tid':ii(p[1]),'x':ff(p[2]),'y':ff(p[3]),'w':ff(p[4]),'h':ff(p[5]),'score':ff(p[6],1.0) if len(p)>6 else 1.0}
            tracks[r['tid']].append(r)
    for rs in tracks.values(): rs.sort(key=lambda r:r['frame'])
    return tracks

def annotate(rs, gt, thr):
    for r in rs:
        best=-1; bv=0
        for gid,gbox in gt.get(r['frame'],[]):
            v=iou((r['x'],r['y'],r['w'],r['h']),gbox)
            if v>bv: bv=v; best=gid
        r['gt']=best if bv>=thr else -1; r['iou']=bv

def pct(xs, p):
    if not xs: return 0.0
    xs=sorted(xs); i=int(round((len(xs)-1)*p/100)); return float(xs[i])
def stat(xs):
    if not xs: return {'n':0}
    return {'n':len(xs),'mean':sum(xs)/len(xs),'p50':pct(xs,50),'p75':pct(xs,75),'p90':pct(xs,90),'p95':pct(xs,95),'max':max(xs)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tracks-dir',required=True); ap.add_argument('--gt-root',required=True); ap.add_argument('--out-csv',required=True); ap.add_argument('--summary-json',default=''); ap.add_argument('--iou-thr',type=float,default=0.5); args=ap.parse_args()
    rows=[]
    for txt in sorted(Path(args.tracks_dir).glob('MOT20-*.txt')):
        seq=txt.stem; gt=load_gt(Path(args.gt_root)/seq/'gt'/'gt.txt'); tracks=load_pred(txt)
        for tid,rs in tracks.items():
            annotate(rs,gt,args.iou_thr)
            for a,b in zip(rs,rs[1:]):
                if a['gt']<0 or b['gt']<0: continue
                ca,cb=center(a),center(b); dt=max(1,b['frame']-a['frame']); dist=math.hypot(cb[0]-ca[0],cb[1]-ca[1]); ar=max(area(a),area(b))/max(1e-9,min(area(a),area(b))); hr=max(a['h'],b['h'])/max(1e-9,min(a['h'],b['h']))
                rows.append({'seq':seq,'track_id':tid,'frame_a':a['frame'],'frame_b':b['frame'],'same_gt':int(a['gt']==b['gt']),'gt_a':a['gt'],'gt_b':b['gt'],'gap':b['frame']-a['frame'],'missing_gap':max(0,b['frame']-a['frame']-1),'center_step':dist/dt,'center_distance':dist,'height_ratio':hr,'area_ratio':ar,'score_min':min(a['score'],b['score']),'score_drop':abs(a['score']-b['score']),'iou_a':a['iou'],'iou_b':b['iou']})
    out=Path(args.out_csv); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    pos=[r for r in rows if r['same_gt']==0]; neg=[r for r in rows if r['same_gt']==1]
    summary={'rows':len(rows),'switch_transitions':len(pos),'same_transitions':len(neg),'switch_rate':len(pos)/len(rows) if rows else 0,'metrics':{}}
    for k in ['gap','missing_gap','center_step','center_distance','height_ratio','area_ratio','score_min','score_drop']:
        summary['metrics'][k]={'switch':stat([float(r[k]) for r in pos]),'same':stat([float(r[k]) for r in neg])}
    # simple threshold recall / precision grid for switch detection
    grid=[]
    for gap_thr in [1,2,3,5,8,10,15,20,30]:
        for step_thr in [0,20,40,60,80,120,160,200]:
            sel=[r for r in rows if r['missing_gap']>=gap_thr and r['center_step']>=step_thr]
            tp=sum(1 for r in sel if r['same_gt']==0)
            grid.append({'missing_gap_ge':gap_thr,'center_step_ge':step_thr,'n':len(sel),'tp_switch':tp,'precision':tp/len(sel) if sel else 0,'recall':tp/len(pos) if pos else 0})
    summary['grid_top_precision']=sorted(grid,key=lambda x:(-x['precision'],-x['tp_switch']))[:30]
    summary['grid_top_f1']=sorted(grid,key=lambda x:-(2*x['precision']*x['recall']/(x['precision']+x['recall']) if x['precision']+x['recall']>0 else 0))[:30]
    if args.summary_json: Path(args.summary_json).write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()
