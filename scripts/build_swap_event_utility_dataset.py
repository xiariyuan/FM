from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']


def cosine(a,b):
    if a is None or b is None:return 0.0
    na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b))
    if na<1e-8 or nb<1e-8:return 0.0
    return float(np.dot(a,b)/(na*nb))


def load_tracks(path:Path):
    by_tid=defaultdict(list)
    with path.open() as f:
        for line in f:
            p=line.strip().split(',')
            if len(p)<7:continue
            fr=int(float(p[0]));tid=int(float(p[1]));x,y,w,h,score=map(float,p[2:7])
            by_tid[tid].append((fr,x,y,w,h,score))
    out={}
    for tid,rows in by_tid.items():
        rows.sort();out[tid]={r[0]:r for r in rows}
    return out


def center(r):return np.array([r[1]+.5*r[3],r[2]+.5*r[4]],dtype=float)

def size(r):return np.array([r[3],r[4]],dtype=float)


def latest(rows:Dict[int,tuple],frame:int,lo:int):
    cand=[rows[f] for f in range(frame,lo-1,-1) if f in rows]
    return cand[0] if cand else None


def earliest(rows:Dict[int,tuple],frame:int,hi:int):
    cand=[rows[f] for f in range(frame,hi+1) if f in rows]
    return cand[0] if cand else None


def velocity(rows:Dict[int,tuple],end_frame:int,window:int,backward:bool=True):
    fs=sorted(f for f in rows if end_frame-window<=f<=end_frame) if backward else sorted(f for f in rows if end_frame<=f<=end_frame+window)
    if len(fs)<2:return np.zeros(2),0
    a=rows[fs[0]];b=rows[fs[-1]];dt=max(1,fs[-1]-fs[0])
    return (center(b)-center(a))/dt,len(fs)


def local_stats(rows:Dict[int,tuple],lo:int,hi:int):
    rs=[rows[f] for f in range(lo,hi+1) if f in rows]
    if not rs:return {'n':0,'score_mean':0,'score_min':0,'height_mean':0,'area_mean':0}
    sc=np.array([r[5] for r in rs]);he=np.array([r[4] for r in rs]);ar=np.array([r[3]*r[4] for r in rs])
    return {'n':len(rs),'score_mean':float(sc.mean()),'score_min':float(sc.min()),'height_mean':float(he.mean()),'area_mean':float(ar.mean())}


def load_reid(root:Path,seq:str):
    p=root/seq/'tracklet_reid_features.npz'
    if not p.exists():return {}
    z=np.load(p);out={}
    for i,tid in enumerate(z['track_id'].astype(int)):
        out[int(tid)]={k:z[k][i].astype(np.float32) for k in ['start','end','global_mean','high_score']}
    return out


def load_track_observables(path:Path,seq:str):
    d=pd.read_csv(path);d=d[d.seq==seq]
    drop={'dominant_gt','dominant_gt_rows','track_gt_purity','track_unique_gt','repair_debt_rows','repair_debt_ratio','family_debt_rows','family_debt_ratio','family_unique_tracker_ids','is_dominant_tid_for_gt','matched_rows'}
    cols=[c for c in d.columns if c not in drop|{'seq','track_id'} and pd.api.types.is_numeric_dtype(d[c])]
    return {int(r.track_id):{c:float(getattr(r,c)) if pd.notna(getattr(r,c)) else 0.0 for c in cols} for r in d.itertuples(index=False)},cols


def load_debt(path:Path,seq:str):
    d=pd.read_csv(path);d=d[d.seq==seq]
    n=max(1,len(d))
    return {int(r.track_id):{'debt_score':float(r.score),'debt_rank_pct':1-(float(r.rank)-1)/n} for r in d.itertuples(index=False)}


def load_matches(path:Path,seq:str):
    d=pd.read_csv(path,usecols=['seq','frame','track_id','gt_id','iou']);d=d[d.seq==seq]
    by_track=defaultdict(dict)
    for r in d.itertuples(index=False):by_track[int(r.track_id)][int(r.frame)]=(int(r.gt_id),float(r.iou))
    return by_track


def previous_gt(m:Dict[int,tuple],frame:int,lookback:int):
    for f in range(frame-1,max(0,frame-lookback)-1,-1):
        if f in m:return m[f][0],f,m[f][1]
    return None


def utility(matches,ta,tb,frame,horizon,lookback):
    pa=previous_gt(matches.get(ta,{}),frame,lookback);pb=previous_gt(matches.get(tb,{}),frame,lookback)
    if pa is None or pb is None or pa[0]==pb[0]:return None
    ga,gb=pa[0],pb[0];keep=swap=obs=0
    for f in range(frame,frame+horizon+1):
        ma=matches.get(ta,{}).get(f);mb=matches.get(tb,{}).get(f)
        if ma:
            obs+=1;keep+=int(ma[0]==ga);swap+=int(ma[0]==gb)
        if mb:
            obs+=1;keep+=int(mb[0]==gb);swap+=int(mb[0]==ga)
    return {'prev_gt_a':ga,'prev_gt_b':gb,'prev_match_frame_a':pa[1],'prev_match_frame_b':pb[1],'prev_iou_a':pa[2],'prev_iou_b':pb[2],f'keep_support_{horizon}':keep,f'swap_support_{horizon}':swap,f'observed_support_{horizon}':obs,f'swap_utility_{horizon}':swap-keep}


def candidate_frames(group:pd.DataFrame,stride:int,min_peak:float):
    frames=group.frame.to_numpy(int);ioa=group.ioa_min_area.to_numpy(float);out=[]
    start=0
    for i in range(1,len(group)+1):
        if i==len(group) or frames[i]-frames[i-1]>1:
            run_frames=frames[start:i];run_ioa=ioa[start:i]
            if len(run_frames) and run_ioa.max()>=min_peak:
                run_start=int(run_frames[0]);run_end=int(run_frames[-1]);duration=run_end-run_start+1
                # Peak per temporal stride bin plus global episode peak.
                chosen=set()
                for lo in range(run_start,run_end+1,stride):
                    mask=(run_frames>=lo)&(run_frames<lo+stride)
                    if mask.any():
                        idx=np.where(mask)[0][np.argmax(run_ioa[mask])];chosen.add(int(run_frames[idx]))
                chosen.add(int(run_frames[np.argmax(run_ioa)]))
                frame_to_ioa={int(f):float(v) for f,v in zip(run_frames,run_ioa)}
                for fr in sorted(chosen):
                    out.append({'frame':fr,'episode_start':run_start,'episode_end':run_end,'episode_duration':duration,'episode_frames':len(run_frames),'episode_ioa_max':float(run_ioa.max()),'episode_ioa_mean':float(run_ioa.mean()),'episode_ioa_std':float(run_ioa.std()),'episode_ioa_start':float(run_ioa[0]),'episode_ioa_end':float(run_ioa[-1]),'candidate_ioa':frame_to_ioa[fr],'candidate_episode_frac':(fr-run_start)/max(1,duration-1)})
            start=i
    return out


def pair_features(seq,ta,tb,c,tracks,reid,obs,debt):
    f=int(c['frame']);ra=tracks.get(ta,{});rb=tracks.get(tb,{})
    a0=latest(ra,f-1,f-8);b0=latest(rb,f-1,f-8);a1=earliest(ra,f,f+8);b1=earliest(rb,f,f+8)
    if any(x is None for x in [a0,b0,a1,b1]):return None
    va,na=velocity(ra,a0[0],5,True);vb,nb=velocity(rb,b0[0],5,True)
    dta=max(1,a1[0]-a0[0]);dtb=max(1,b1[0]-b0[0]);preda=center(a0)+va*dta;predb=center(b0)+vb*dtb
    hscale=max(1,.25*(a0[4]+b0[4]+a1[4]+b1[4]))
    keep_motion=(np.linalg.norm(preda-center(a1))+np.linalg.norm(predb-center(b1)))/hscale
    swap_motion=(np.linalg.norm(preda-center(b1))+np.linalg.norm(predb-center(a1)))/hscale
    keep_size=(np.abs(np.log((size(a1)+1)/(size(a0)+1))).sum()+np.abs(np.log((size(b1)+1)/(size(b0)+1))).sum())
    swap_size=(np.abs(np.log((size(b1)+1)/(size(a0)+1))).sum()+np.abs(np.log((size(a1)+1)/(size(b0)+1))).sum())
    pre_rel=center(a0)-center(b0);post_rel=center(a1)-center(b1)
    pre_a=local_stats(ra,f-5,f-1);pre_b=local_stats(rb,f-5,f-1);post_a=local_stats(ra,f,f+5);post_b=local_stats(rb,f,f+5)
    out=dict(c);out.update({'seq':seq,'track_a':ta,'track_b':tb,'pre_frame_a':a0[0],'pre_frame_b':b0[0],'post_frame_a':a1[0],'post_frame_b':b1[0],
      'keep_motion_cost':float(keep_motion),'swap_motion_cost':float(swap_motion),'swap_motion_advantage':float(keep_motion-swap_motion),
      'keep_size_cost':float(keep_size),'swap_size_cost':float(swap_size),'swap_size_advantage':float(keep_size-swap_size),
      'pre_center_distance':float(np.linalg.norm(pre_rel)/hscale),'post_center_distance':float(np.linalg.norm(post_rel)/hscale),
      'relative_motion_cos':float(np.dot(va,vb)/(max(1e-8,np.linalg.norm(va)*np.linalg.norm(vb)))),'pre_velocity_a':float(np.linalg.norm(va)/hscale),'pre_velocity_b':float(np.linalg.norm(vb)/hscale),
      'pre_dx':float(pre_rel[0]/hscale),'pre_dy':float(pre_rel[1]/hscale),'post_dx':float(post_rel[0]/hscale),'post_dy':float(post_rel[1]/hscale),
      'x_order_flip':int(pre_rel[0]*post_rel[0]<0),'y_order_flip':int(pre_rel[1]*post_rel[1]<0),
      'score_drop_a':pre_a['score_mean']-post_a['score_mean'],'score_drop_b':pre_b['score_mean']-post_b['score_mean'],'score_min_pair':min(pre_a['score_min'],pre_b['score_min'],post_a['score_min'],post_b['score_min']),
      'height_ratio_pre':float(max(a0[4],b0[4])/max(1,min(a0[4],b0[4]))),'height_ratio_post':float(max(a1[4],b1[4])/max(1,min(a1[4],b1[4]))),
      'track_pos_frac_a':f/max(1,max(ra)-min(ra)) if ra else 0,'track_pos_frac_b':f/max(1,max(rb)-min(rb)) if rb else 0,
    })
    fa=reid.get(ta);fb=reid.get(tb)
    for ka,kb,name in [('global_mean','global_mean','reid_global_cos'),('start','start','reid_start_cos'),('end','end','reid_end_cos'),('high_score','high_score','reid_high_cos'),('start','end','reid_a_start_b_end'),('end','start','reid_a_end_b_start')]:
        out[name]=cosine(fa.get(ka) if fa else None,fb.get(kb) if fb else None)
    for prefix,tid in [('a',ta),('b',tb)]:
        for k,v in debt.get(tid,{'debt_score':0,'debt_rank_pct':0}).items():out[f'{prefix}_{k}']=v
        vals=obs.get(tid,{})
        for k in ['track_len','track_span','track_density','score_mean','score_std','score_min','height_cv','speed_mean','speed_std','speed_max','accel_mean','accel_max','appearance_drift','overlap_event_count','overlap_partner_count','overlap_ioa_mean','overlap_ioa_max','overlap_frame_fraction']:
            out[f'{prefix}_{k}']=vals.get(k,0.0)
    for k in ['debt_score','debt_rank_pct','appearance_drift','overlap_partner_count','overlap_frame_fraction','speed_max','score_std']:
        av=out.get('a_'+k,0);bv=out.get('b_'+k,0);out['pair_max_'+k]=max(av,bv);out['pair_min_'+k]=min(av,bv);out['pair_absdiff_'+k]=abs(av-bv)
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--overlap-csv',required=True);ap.add_argument('--matches-csv',required=True);ap.add_argument('--track-root',required=True);ap.add_argument('--reid-root',required=True);ap.add_argument('--track-observables',required=True);ap.add_argument('--debt-oof',required=True);ap.add_argument('--out-dir',required=True);ap.add_argument('--stride',type=int,default=5);ap.add_argument('--min-peak-ioa',type=float,default=.65);ap.add_argument('--lookback',type=int,default=6);args=ap.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    overlap=pd.read_csv(args.overlap_csv);all_rows=[];reports=[]
    for seq in SEQS:
        print('[swap-dataset]',seq,flush=True)
        ov=overlap[overlap.seq==seq].sort_values(['track_i','track_j','frame'])
        tracks=load_tracks(Path(args.track_root)/f'{seq}.txt');matches=load_matches(Path(args.matches_csv),seq);reid=load_reid(Path(args.reid_root),seq);obs,_=load_track_observables(Path(args.track_observables),seq);debt=load_debt(Path(args.debt_oof),seq)
        candidates=0;labeled=0;positive10=positive30=positive60=0
        for (ta,tb),g in ov.groupby(['track_i','track_j'],sort=False):
            for c in candidate_frames(g,args.stride,args.min_peak_ioa):
                candidates+=1;feat=pair_features(seq,int(ta),int(tb),c,tracks,reid,obs,debt)
                if feat is None:continue
                labels={}
                ok=True
                for h in [10,30,60]:
                    u=utility(matches,int(ta),int(tb),int(c['frame']),h,args.lookback)
                    if u is None:ok=False;break
                    labels.update(u)
                if not ok:continue
                feat.update(labels);feat['swap_positive_10']=int(feat['swap_utility_10']>0);feat['swap_positive_30']=int(feat['swap_utility_30']>0);feat['swap_positive_60']=int(feat['swap_utility_60']>0);feat['swap_strong_30']=int(feat['swap_utility_30']>=4);feat['swap_strong_60']=int(feat['swap_utility_60']>=6)
                all_rows.append(feat);labeled+=1;positive10+=feat['swap_positive_10'];positive30+=feat['swap_positive_30'];positive60+=feat['swap_positive_60']
        reports.append({'seq':seq,'overlap_rows':len(ov),'candidate_events':candidates,'labeled_events':labeled,'positive10':positive10,'positive30':positive30,'positive60':positive60})
        print(reports[-1],flush=True)
    df=pd.DataFrame(all_rows)
    # Sequence percentiles for all numeric observable columns only.
    label_prefixes=('prev_gt_','prev_match_','prev_iou_','keep_support_','swap_support_','observed_support_','swap_utility_','swap_positive_','swap_strong_')
    idcols={'seq','track_a','track_b','frame'}
    num=[c for c in df.columns if c not in idcols and not c.startswith(label_prefixes) and pd.api.types.is_numeric_dtype(df[c])]
    pct={}
    for c in num:pct[c+'_seqpct']=df.groupby('seq')[c].rank(pct=True,method='average')
    if pct:df=pd.concat([df,pd.DataFrame(pct)],axis=1)
    df.to_csv(out/'swap_event_utility_dataset.csv',index=False)
    summary={'rows':len(df),'columns':len(df.columns),'stride':args.stride,'min_peak_ioa':args.min_peak_ioa,'lookback':args.lookback,'reports':reports,'utility_quantiles':{str(h):df[f'swap_utility_{h}'].quantile([0,.01,.1,.5,.9,.99,1]).to_dict() for h in [10,30,60]}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
