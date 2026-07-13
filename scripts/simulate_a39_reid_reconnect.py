#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict, Counter
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'external/BoT-SORT-main'))
from fast_reid.fast_reid_interfece import FastReIDInterface  # noqa

def ai(v,d=0):
    try: return int(float(v))
    except Exception: return d

def af(v,d=0.0):
    try: return float(v)
    except Exception: return d

def read_track(path):
    rows=[]; by=defaultdict(list)
    with open(path,encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            p=line.strip().split(',')
            if len(p)<6: continue
            fr=ai(p[0],-1); tid=ai(p[1],-1); x=af(p[2]); y=af(p[3]); w=af(p[4]); h=af(p[5]); score=af(p[6],1.0) if len(p)>6 else 1.0
            if fr<0 or tid<0 or w<=0 or h<=0: continue
            r={'idx':len(rows),'frame':fr,'track_id':tid,'box':np.array([x,y,x+w,y+h],dtype=np.float32),'parts':p,'score':score}
            rows.append(r); by[(fr,tid)].append(r['idx'])
    return rows,by

def read_tunnels(path):
    out=[]
    with open(path,newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            tracks=[ai(x,-1) for x in str(r.get('tracks','')).split('|') if x!='']; tracks=[x for x in tracks if x>=0]
            out.append({'tunnel_id':ai(r.get('tunnel_id'),len(out)),'start':ai(r.get('start')),'end':ai(r.get('end')),'tracks':set(tracks)})
    return out

def img_path(img_dir,fr): return Path(img_dir)/f'{fr:06d}.jpg'
def choose(rr,n):
    rr=sorted(rr,key=lambda r:(-r.get('score',1.0),r['frame']))[:n]
    return sorted(rr,key=lambda r:r['frame'])
def extract(groups,img_dir,encoder,max_crops):
    selected={k:choose(v,max_crops) for k,v in groups.items() if v}
    byfr=defaultdict(list)
    for k,rr in selected.items():
        for r in rr: byfr[r['frame']].append((k,r))
    feats=defaultdict(list)
    for fr,items in sorted(byfr.items()):
        img=cv2.imread(str(img_path(img_dir,fr)))
        if img is None: continue
        dets=np.stack([r['box'] for _,r in items]).astype(np.float32)
        fs=encoder.inference(img,dets)
        for (k,r),f in zip(items,fs): feats[k].append(f.astype(np.float32))
    proto={}
    for k,fs in feats.items():
        v=np.mean(np.stack(fs),axis=0); proto[k]=v/max(float(np.linalg.norm(v)),1e-12)
    return proto,{k:len(v) for k,v in selected.items()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--track-file',required=True); ap.add_argument('--tunnels-csv',required=True); ap.add_argument('--img-dir',required=True)
    ap.add_argument('--fast-reid-config',required=True); ap.add_argument('--fast-reid-weights',required=True)
    ap.add_argument('--out-file',required=True); ap.add_argument('--summary-json',required=True); ap.add_argument('--pairs-csv',required=True)
    ap.add_argument('--pre-window',type=int,default=10); ap.add_argument('--post-window',type=int,default=10); ap.add_argument('--apply-window',type=int,default=10)
    ap.add_argument('--max-crops-per-track',type=int,default=5); ap.add_argument('--min-group-rows',type=int,default=2)
    ap.add_argument('--min-sim',type=float,default=0.6); ap.add_argument('--min-row-margin',type=float,default=0.05); ap.add_argument('--min-col-margin',type=float,default=0.00)
    ap.add_argument('--device',default='cuda')
    args=ap.parse_args()
    rows,by_frame_tid=read_track(args.track_file); tunnels=read_tunnels(args.tunnels_csv)
    groups=defaultdict(list)
    for t in tunnels:
        for r in rows:
            if r['track_id'] not in t['tracks']: continue
            side=None
            if t['start']-args.pre_window <= r['frame'] < t['start']: side='pre'
            elif t['end'] < r['frame'] <= t['end']+args.post_window: side='post'
            if side: groups[(t['tunnel_id'],side,r['track_id'])].append(r)
    groups={k:v for k,v in groups.items() if len(v)>=args.min_group_rows}
    encoder=FastReIDInterface(args.fast_reid_config,args.fast_reid_weights,args.device,batch_size=32)
    proto,crop_counts=extract(groups,args.img_dir,encoder,args.max_crops_per_track)
    planned=[]; pair_rows=[]; stats=Counter()
    for t in tunnels:
        tid=t['tunnel_id']
        pre=[k for k in proto if k[0]==tid and k[1]=='pre']
        post=[k for k in proto if k[0]==tid and k[1]=='post']
        if not pre or not post: continue
        P=np.stack([proto[k] for k in pre]); Q=np.stack([proto[k] for k in post]); S=P@Q.T
        row_sorted=np.sort(S,axis=1)[:,::-1]
        row_margin=row_sorted[:,0]-(row_sorted[:,1] if S.shape[1]>1 else -1)
        col_sorted=np.sort(S,axis=0)[::-1,:]
        col_margin=col_sorted[0,:]-(col_sorted[1,:] if S.shape[0]>1 else -1)
        ri,ci=linear_sum_assignment(-S)
        stats['tunnels_with_prepost']+=1
        for i,j in zip(ri,ci):
            sim=float(S[i,j]); rm=float(row_margin[i]); cm=float(col_margin[j])
            pre_id=pre[i][2]; post_id=post[j][2]
            accept=(sim>=args.min_sim and rm>=args.min_row_margin and cm>=args.min_col_margin and pre_id!=post_id)
            pair_rows.append({'tunnel_id':tid,'pre_track':pre_id,'post_track':post_id,'sim':sim,'row_margin':rm,'col_margin':cm,'accept':int(accept),'pre_crops':crop_counts.get(pre[i],0),'post_crops':crop_counts.get(post[j],0)})
            if accept:
                stats['accepted_pairs']+=1
                f0=t['end']+1; f1=t['end']+args.apply_window
                for fr in range(f0,f1+1):
                    for idx in by_frame_tid.get((fr,post_id),[]): planned.append((idx,pre_id,tid,post_id,sim))
    final={r['idx']:r['track_id'] for r in rows}
    # apply with collision skip: if target id already exists in that frame, skip that row.
    by_frame_current=defaultdict(set)
    for r in rows: by_frame_current[r['frame']].add(r['track_id'])
    for idx,target,tid,post_id,sim in planned:
        fr=rows[idx]['frame']; orig=final[idx]
        if orig==target: continue
        if target in by_frame_current[fr]:
            stats['skipped_collision_rows']+=1; continue
        by_frame_current[fr].discard(orig); by_frame_current[fr].add(target); final[idx]=target; stats['applied_rows']+=1
    stats['planned_rows']=len(planned); stats['groups']=len(groups); stats['features']=len(proto)
    # final exact uniqueness repair using moderate temp ids
    seen=defaultdict(list)
    for idx,fid in final.items(): seen[(rows[idx]['frame'],fid)].append(idx)
    temp_base=1000000
    for (fr,fid),idxs in seen.items():
        if len(idxs)>1:
            for idx in idxs[1:]: final[idx]=temp_base+idx; stats['final_temp_rows']+=1
    for r in rows: r['parts'][1]=str(final[r['idx']])
    out=Path(args.out_file); out.parent.mkdir(parents=True,exist_ok=True)
    for r in sorted(rows,key=lambda x:(ai(x['parts'][0]),ai(x['parts'][1]),af(x['parts'][2]),af(x['parts'][3]))):
        out.write_text('',encoding='utf-8') if False else None
    with out.open('w',encoding='utf-8') as f:
        for r in sorted(rows,key=lambda x:(ai(x['parts'][0]),ai(x['parts'][1]),af(x['parts'][2]),af(x['parts'][3]))): f.write(','.join(r['parts'])+'\n')
    with open(args.pairs_csv,'w',newline='',encoding='utf-8') as f:
        fields=['tunnel_id','pre_track','post_track','sim','row_margin','col_margin','accept','pre_crops','post_crops']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(pair_rows)
    stats['min_sim']=args.min_sim; stats['min_row_margin']=args.min_row_margin; stats['min_col_margin']=args.min_col_margin; stats['apply_window']=args.apply_window
    Path(args.summary_json).write_text(json.dumps(dict(stats),indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(stats),indent=2,sort_keys=True))
if __name__=='__main__': main()
