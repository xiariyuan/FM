from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']


def load_matches(path,seq):
 d=pd.read_csv(path,usecols=['seq','frame','track_id','gt_id','iou']);d=d[d.seq==seq]
 by=defaultdict(dict)
 for r in d.itertuples(index=False):by[int(r.track_id)][int(r.frame)]=int(r.gt_id)
 return by


def load_tracks(path):
 by=defaultdict(dict)
 with open(path) as f:
  for line in f:
   p=line.strip().split(',')
   if len(p)>=7:
    by[int(float(p[1]))][int(float(p[0]))]=tuple(map(float,p[2:7]))
 return by


def load_obs(path,seq):
 d=pd.read_csv(path);d=d[d.seq==seq]
 return {int(r.track_id):r for r in d.itertuples(index=False)}


def load_debt(path,seq):
 d=pd.read_csv(path);d=d[d.seq==seq]
 return {int(r.track_id):r for r in d.itertuples(index=False)}


def load_reid(path,seq):
 p=path/seq/'tracklet_reid_features.npz'
 if not p.exists():return {}
 z=np.load(p);out={}
 for i,t in enumerate(z['track_id'].astype(int)):
  out[int(t)]={k:z[k][i] for k in ['start','end','global_mean','high_score']}
 return out


def cosine(a,b):
 if a is None or b is None:return 0.
 na=np.linalg.norm(a);nb=np.linalg.norm(b)
 return float(np.dot(a,b)/max(1e-8,na*nb))


def build_gt_prefix(matches):
 out={}
 for tid,frames in matches.items():
  maxf=max(frames);arr=np.full(maxf+2,-1,dtype=np.int32)
  for f,g in frames.items():arr[f]=g
  # previous valid GT
  prev=np.full_like(arr,-1)
  cur=-1
  for i in range(len(arr)):
   if arr[i]>=0:cur=arr[i]
   prev[i]=cur
  out[tid]=prev
 return out


def utility_fast(gt_prev,a,b,frame,h):
 if a not in gt_prev or b not in gt_prev:return None
 pa=gt_prev[a][frame-1] if frame-1<len(gt_prev[a]) else -1
 pb=gt_prev[b][frame-1] if frame-1<len(gt_prev[b]) else -1
 if pa<0 or pb<0 or pa==pb:return None
 keep=swap=0
 end=min(frame+h,len(gt_prev[a])-1,len(gt_prev[b])-1)
 for f in range(frame,end+1):
  ga=gt_prev[a][f];gb=gt_prev[b][f]
  if ga>=0:
   keep+=ga==pa;swap+=ga==pb
  if gb>=0:
   keep+=gb==pb;swap+=gb==pa
 return swap-keep,pa,pb,keep,swap


def build_candidates(overlap,seq):
 out=[]
 d=overlap[overlap.seq==seq].sort_values(['track_i','track_j','frame'])
 for (a,b),g in d.groupby(['track_i','track_j']):
  if g.ioa_min_area.max()<.65:continue
  idx=g.ioa_min_area.idxmax();r=g.loc[idx]
  out.append((int(a),int(b),int(r.frame),float(r.ioa_min_area),int(g.frame.min()),int(g.frame.max())))
 return out


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--overlap-csv',required=True);ap.add_argument('--matches-csv',required=True);ap.add_argument('--track-root',required=True);ap.add_argument('--reid-root',required=True);ap.add_argument('--obs-csv',required=True);ap.add_argument('--debt-csv',required=True);ap.add_argument('--out-dir',required=True);args=ap.parse_args()
 out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
 overlap=pd.read_csv(args.overlap_csv);rows=[];reports=[]
 for seq in SEQS:
  print('processing',seq,flush=True)
  matches=load_matches(args.matches_csv,seq);gt_prev=build_gt_prefix(matches);tracks=load_tracks(Path(args.track_root)/f'{seq}.txt');obs=load_obs(args.obs_csv,seq);debt=load_debt(args.debt_csv,seq);reid=load_reid(Path(args.reid_root),seq)
  cand=build_candidates(overlap,seq);pos=[0,0,0];
  for a,b,fr,ioa,es,ee in cand:
   base={'seq':seq,'track_a':a,'track_b':b,'frame':fr,'candidate_ioa':ioa,'episode_start':es,'episode_end':ee,'episode_duration':ee-es+1}
   for h in [10,30,60]:
    u=utility_fast(gt_prev,a,b,fr,h)
    if u is None:break
    base[f'swap_utility_{h}']=u[0];base[f'keep_support_{h}']=u[3];base[f'swap_support_{h}']=u[4]
    pos[h//30 if h in [30,60] else 0]+=int(u[0]>0)
   else:
    oa=obs.get(a);ob=obs.get(b);da=debt.get(a);db=debt.get(b)
    base['a_debt']=float(getattr(da,'score',0));base['b_debt']=float(getattr(db,'score',0));base['a_overlap']=float(getattr(oa,'overlap_event_count',0));base['b_overlap']=float(getattr(ob,'overlap_event_count',0))
    base['a_len']=float(getattr(oa,'track_len',0));base['b_len']=float(getattr(ob,'track_len',0));base['a_purity']=float(getattr(oa,'track_gt_purity',0));base['b_purity']=float(getattr(ob,'track_gt_purity',0))
    ra=reid.get(a);rb=reid.get(b)
    base['reid_global_cos']=cosine(ra.get('global_mean') if ra else None,rb.get('global_mean') if rb else None)
    base['reid_end_cos']=cosine(ra.get('end') if ra else None,rb.get('end') if rb else None)
    base['reid_cross_cos']=cosine(ra.get('end') if ra else None,rb.get('start') if rb else None)
    rows.append(base)
  reports.append({'seq':seq,'candidates':len(cand),'rows_total':len(rows)})
  print(reports[-1],flush=True)
 df=pd.DataFrame(rows)
 for c in df.select_dtypes(include='number').columns:
  if not c.startswith(('swap_','keep_','frame','track_')):
   df[c+'_seqpct']=df.groupby('seq')[c].rank(pct=True)
 summary={'rows':len(df),'columns':len(df.columns),'reports':reports,'positive30':int((df.swap_utility_30>0).sum()),'positive60':int((df.swap_utility_60>0).sum())}
 df.to_csv(out/'swap_event_utility_dataset.csv',index=False)
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
