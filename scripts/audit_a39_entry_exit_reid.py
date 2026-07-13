#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, os, sys
from collections import defaultdict, Counter
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'external/BoT-SORT-main'))
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
            r={'idx':len(rows),'frame':fr,'track_id':tid,'box':np.array([x,y,x+w,y+h],dtype=np.float32),'score':score}
            rows.append(r); by[fr].append(r)
    return rows, by

def read_gt(path):
    by=defaultdict(list)
    with open(path,encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            p=line.strip().split(',')
            if len(p)<6: continue
            fr=ai(p[0],-1); gid=ai(p[1],-1); x=af(p[2]); y=af(p[3]); w=af(p[4]); h=af(p[5])
            mark=ai(p[6],1) if len(p)>6 else 1; cls=ai(p[7],1) if len(p)>7 else 1
            if fr<0 or gid<0 or w<=0 or h<=0 or mark<=0 or cls!=1: continue
            by[fr].append({'gt_id':gid,'box':np.array([x,y,x+w,y+h],dtype=np.float32)})
    return by

def pair_iou(rr,gg):
    if not rr or not gg: return np.zeros((len(rr),len(gg)),dtype=np.float32)
    A=np.stack([x['box'] for x in rr]); B=np.stack([x['box'] for x in gg])
    lt=np.maximum(A[:,None,:2],B[None,:,:2]); rb=np.minimum(A[:,None,2:],B[None,:,2:])
    wh=np.clip(rb-lt,0,None); inter=wh[...,0]*wh[...,1]
    aa=np.clip((A[:,2]-A[:,0])*(A[:,3]-A[:,1]),1e-6,None); bb=np.clip((B[:,2]-B[:,0])*(B[:,3]-B[:,1]),1e-6,None)
    return inter/np.clip(aa[:,None]+bb[None,:]-inter,1e-6,None)

def match_row_gt(rows_by, gt_by, thr):
    row_gt={}; row_iou={}
    for fr in sorted(set(rows_by)|set(gt_by)):
        rr=rows_by.get(fr,[]); gg=gt_by.get(fr,[])
        if not rr or not gg: continue
        I=pair_iou(rr,gg); ri,ci=linear_sum_assignment(1-I)
        for r,c in zip(ri,ci):
            val=float(I[r,c])
            if val>=thr:
                row_gt[rr[r]['idx']]=int(gg[c]['gt_id']); row_iou[rr[r]['idx']]=val
    return row_gt,row_iou

def read_tunnels(path):
    out=[]
    with open(path,newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            tracks=[ai(x,-1) for x in str(r.get('tracks','')).split('|') if x!='']
            tracks=[x for x in tracks if x>=0]
            out.append({'tunnel_id':ai(r.get('tunnel_id'),len(out)),'start':ai(r.get('start')),'end':ai(r.get('end')),'tracks':set(tracks),'duration':ai(r.get('duration'))})
    return out

def img_path(img_dir, frame):
    return Path(img_dir)/f'{frame:06d}.jpg'

def choose_rows(rows, max_per_track):
    if len(rows)<=max_per_track: return rows
    rows=sorted(rows,key=lambda r:(-r.get('score',1.0), r['frame']))
    return sorted(rows[:max_per_track],key=lambda r:r['frame'])

def extract_group_features(groups, img_dir, encoder, max_per_track):
    # groups: key -> list rows. Returns key->normalized mean feat.
    selected={k:choose_rows(v,max_per_track) for k,v in groups.items() if v}
    frame_items=defaultdict(list)
    for k,rr in selected.items():
        for r in rr:
            frame_items[r['frame']].append((k,r))
    feats=defaultdict(list)
    for fr,items in sorted(frame_items.items()):
        img=cv2.imread(str(img_path(img_dir,fr)))
        if img is None:
            continue
        dets=np.stack([r['box'] for _,r in items],axis=0).astype(np.float32)
        out=encoder.inference(img,dets)
        for (k,r),feat in zip(items,out):
            feats[k].append(feat.astype(np.float32))
    proto={}
    for k,fs in feats.items():
        if not fs: continue
        v=np.mean(np.stack(fs,axis=0),axis=0)
        n=max(float(np.linalg.norm(v)),1e-12); proto[k]=v/n
    return proto, {k:len(v) for k,v in selected.items()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--track-file',required=True)
    ap.add_argument('--gt-file',required=True)
    ap.add_argument('--tunnels-csv',required=True)
    ap.add_argument('--img-dir',required=True)
    ap.add_argument('--fast-reid-config',required=True)
    ap.add_argument('--fast-reid-weights',required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--pre-window',type=int,default=10)
    ap.add_argument('--post-window',type=int,default=10)
    ap.add_argument('--max-crops-per-track',type=int,default=5)
    ap.add_argument('--iou-thr',type=float,default=0.5)
    ap.add_argument('--device',default='cuda')
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows,rows_by=read_track(args.track_file); gt_by=read_gt(args.gt_file); tunnels=read_tunnels(args.tunnels_csv)
    row_gt,row_iou=match_row_gt(rows_by,gt_by,args.iou_thr)
    # Build pre/post groups for tunnel-track side.
    groups=defaultdict(list); group_gt=defaultdict(Counter)
    for t in tunnels:
        start,end=t['start'],t['end']
        for r in rows:
            if r['track_id'] not in t['tracks']: continue
            side=None
            if start-args.pre_window <= r['frame'] < start: side='pre'
            elif end < r['frame'] <= end+args.post_window: side='post'
            if side is None: continue
            k=(t['tunnel_id'],side,r['track_id'])
            groups[k].append(r)
            gid=row_gt.get(r['idx'],-1)
            if gid>=0: group_gt[k][gid]+=1
    encoder=FastReIDInterface(args.fast_reid_config,args.fast_reid_weights,args.device,batch_size=32)
    proto,crop_counts=extract_group_features(groups,args.img_dir,encoder,args.max_crops_per_track)
    group_rows=[]
    for k,rr in groups.items():
        gid=-1; cnt=0; purity=0.0
        if group_gt[k]:
            gid,cnt=group_gt[k].most_common(1)[0]
            purity=cnt/max(1,sum(group_gt[k].values()))
        group_rows.append({'tunnel_id':k[0],'side':k[1],'track_id':k[2],'rows':len(rr),'crops':crop_counts.get(k,0),'major_gt':gid,'gt_count':cnt,'gt_purity':purity,'has_feat':int(k in proto)})
    # evaluate entry->exit matching where GT appears on both sides.
    pair_rows=[]; summary=Counter()
    for t in tunnels:
        tid=t['tunnel_id']
        pre=[g for g in group_rows if g['tunnel_id']==tid and g['side']=='pre' and g['has_feat'] and g['major_gt']>=0 and g['gt_purity']>=0.6]
        post=[g for g in group_rows if g['tunnel_id']==tid and g['side']=='post' and g['has_feat'] and g['major_gt']>=0 and g['gt_purity']>=0.6]
        if not pre or not post: continue
        summary['tunnels_with_prepost']+=1
        P=np.stack([proto[(tid,'pre',g['track_id'])] for g in pre]); Q=np.stack([proto[(tid,'post',g['track_id'])] for g in post])
        S=P@Q.T
        for i,g in enumerate(pre):
            summary['pre_groups_eval']+=1
            same=[j for j,h in enumerate(post) if h['major_gt']==g['major_gt']]
            if not same: continue
            summary['pre_groups_with_true_post']+=1
            best=int(np.argmax(S[i])); true_best=max(same,key=lambda j:S[i,j])
            sorted_scores=np.sort(S[i])[::-1]
            margin=float(sorted_scores[0]-sorted_scores[1]) if len(sorted_scores)>1 else 1.0
            ok=int(post[best]['major_gt']==g['major_gt'])
            summary['top1_correct']+=ok
            pair_rows.append({'tunnel_id':tid,'pre_track':g['track_id'],'pre_gt':g['major_gt'],'post_best_track':post[best]['track_id'],'post_best_gt':post[best]['major_gt'],'true_post_track':post[true_best]['track_id'],'sim_best':float(S[i,best]),'sim_true':float(S[i,true_best]),'margin':margin,'correct':ok,'pre_rows':g['rows'],'post_rows_best':post[best]['rows']})
        # Hungarian one-to-one accuracy over intersection GTs.
        common=sorted(set(g['major_gt'] for g in pre)&set(h['major_gt'] for h in post))
        if common:
            ri,ci=linear_sum_assignment(-S)
            for i,j in zip(ri,ci):
                if pre[i]['major_gt'] in common:
                    summary['hungarian_pairs']+=1
                    summary['hungarian_correct']+=int(pre[i]['major_gt']==post[j]['major_gt'])
    def rate(a,b): return float(a)/float(b) if b else 0.0
    payload={'summary_counts':dict(summary),'rates':{'top1_accuracy':rate(summary['top1_correct'],summary['pre_groups_with_true_post']),'hungarian_accuracy':rate(summary['hungarian_correct'],summary['hungarian_pairs'])},'params':vars(args)}
    with open(out/'entry_exit_groups.csv','w',newline='',encoding='utf-8') as f:
        fields=['tunnel_id','side','track_id','rows','crops','major_gt','gt_count','gt_purity','has_feat']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(group_rows)
    with open(out/'entry_exit_reid_pairs.csv','w',newline='',encoding='utf-8') as f:
        fields=['tunnel_id','pre_track','pre_gt','post_best_track','post_best_gt','true_post_track','sim_best','sim_true','margin','correct','pre_rows','post_rows_best']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(pair_rows)
    (out/'entry_exit_reid_summary.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    md=['# A39_02a entry-exit ReID feasibility','', '## Counts','```json',json.dumps(dict(summary),indent=2,sort_keys=True),'```','', '## Rates']
    for k,v in payload['rates'].items(): md.append(f'- {k}: {v:.4f}')
    (out/'entry_exit_reid_summary.md').write_text('\n'.join(md)+'\n')
    print('\n'.join(md))
if __name__=='__main__': main()
