from __future__ import annotations
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']

def tracker_counts(root:Path):
 out={}
 for seq in SEQS:
  c=Counter()
  with (root/f'{seq}.txt').open() as f:
   for line in f:
    p=line.strip().split(',')
    if len(p)>=2:c[int(float(p[1]))]+=1
  out[seq]=c
 return out

def gt_counts(root:Path):
 out={}
 for seq in SEQS:
  c=Counter()
  with (root/seq/'gt'/'gt.txt').open() as f:
   for line in f:
    p=line.strip().split(',')
    if len(p)<8:continue
    mark=int(float(p[6]));cls=int(float(p[7]))
    if mark>0 and cls==1:c[int(float(p[1]))]+=1
  out[seq]=c
 return out

def matched_counts(path:Path):
 out=defaultdict(lambda:defaultdict(Counter))
 with path.open(newline='') as f:
  for r in csv.DictReader(f):
   seq=r['seq']
   if seq in SEQS:out[seq][int(r['track_id'])][int(r['gt_id'])]+=1
 return out

def contrib(gtc:Counter,tc:int,m:Counter)->float:
 return float(sum((n*n)/max(1,gtc[g]+tc-n) for g,n in m.items()))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidates',required=True);ap.add_argument('--track-root',required=True);ap.add_argument('--gt-root',required=True);ap.add_argument('--matches-csv',required=True);ap.add_argument('--out-dir',required=True);args=ap.parse_args()
 out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(args.candidates);tc=tracker_counts(Path(args.track_root));gc=gt_counts(Path(args.gt_root));mc=matched_counts(Path(args.matches_csv))
 cache={}
 for seq in SEQS:
  for tid,n in tc[seq].items():cache[(seq,tid)]=contrib(gc[seq],n,mc[seq].get(tid,Counter()))
 deltas=[];after_vals=[];before_vals=[];shared=[];combined_dom=[]
 for r in df.itertuples(index=False):
  seq=str(r.seq);a=int(r.track_a);b=int(r.track_b);ma=mc[seq].get(a,Counter());mb=mc[seq].get(b,Counter())
  mm=ma+mb;before=cache.get((seq,a),0.0)+cache.get((seq,b),0.0);after=contrib(gc[seq],tc[seq].get(a,0)+tc[seq].get(b,0),mm)
  delta=after-before;deltas.append(delta);after_vals.append(after);before_vals.append(before)
  shared.append(int(bool(set(ma)&set(mb))))
  combined_dom.append(mm.most_common(1)[0][0] if mm else -1)
 df['assa_merge_before_proxy']=before_vals;df['assa_merge_after_proxy']=after_vals;df['assa_merge_delta_proxy']=deltas
 df['assa_merge_positive']= (df.assa_merge_delta_proxy>0).astype(int)
 df['assa_merge_delta_per_row']=df.assa_merge_delta_proxy/np.maximum(1,df.len_a+df.len_b)
 df['shares_any_gt']=shared;df['combined_dominant_gt']=combined_dom
 df.to_csv(out/'train_candidates_with_utility.csv',index=False)
 byseq=[]
 for seq in SEQS:
  s=df[df.seq==seq]
  byseq.append({'seq':seq,'rows':len(s),'positive':int(s.assa_merge_positive.sum()),'positive_rate':float(s.assa_merge_positive.mean()),'delta_sum':float(s.assa_merge_delta_proxy.sum()),'delta_positive_sum':float(s.loc[s.assa_merge_delta_proxy>0,'assa_merge_delta_proxy'].sum()),'delta_negative_sum':float(s.loc[s.assa_merge_delta_proxy<0,'assa_merge_delta_proxy'].sum()),'same_gt_positive':int(((s.same_gt==1)&(s.assa_merge_positive==1)).sum()),'same_gt_negative_utility':int(((s.same_gt==1)&(s.assa_merge_positive==0)).sum())})
 report={'rows':len(df),'positive':int(df.assa_merge_positive.sum()),'positive_rate':float(df.assa_merge_positive.mean()),'same_gt':int(df.same_gt.sum()),'same_gt_positive_utility':int(((df.same_gt==1)&(df.assa_merge_positive==1)).sum()),'same_gt_nonpositive_utility':int(((df.same_gt==1)&(df.assa_merge_positive==0)).sum()),'different_gt_positive_utility':int(((df.same_gt==0)&(df.assa_merge_positive==1)).sum()),'by_seq':byseq,'formula':'sum_g m_gt^2/(gt_count + tracker_count - m_gt); delta after merging minus before'}
 (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
 print('\npositive utility quantiles',df.loc[df.assa_merge_delta_proxy>0,'assa_merge_delta_proxy'].quantile([0,.1,.25,.5,.75,.9,.99,1]).to_dict())
 print('negative utility quantiles',df.loc[df.assa_merge_delta_proxy<0,'assa_merge_delta_proxy'].quantile([0,.01,.1,.25,.5,.75,.9,1]).to_dict())
if __name__=='__main__':main()
