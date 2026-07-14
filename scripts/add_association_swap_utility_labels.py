from __future__ import annotations
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
HORIZONS=[10,30,60,None]

def gt_counts(root:Path):
 out={}
 for seq in SEQS:
  c=Counter()
  with (root/seq/'gt'/'gt.txt').open() as f:
   for line in f:
    p=line.strip().split(',')
    if len(p)>=8 and int(float(p[6]))>0 and int(float(p[7]))==1:c[int(float(p[1]))]+=1
  out[seq]=c
 return out

def track_data(root:Path):
 counts={};frames={};maxframe={}
 for seq in SEQS:
  c=Counter();fr=defaultdict(list);mf=0
  with (root/f'{seq}.txt').open() as f:
   for line in f:
    p=line.strip().split(',')
    if len(p)<2:continue
    frame=int(float(p[0]));tid=int(float(p[1]));c[tid]+=1;fr[tid].append(frame);mf=max(mf,frame)
  counts[seq]=c;frames[seq]={t:np.asarray(v,dtype=np.int32) for t,v in fr.items()};maxframe[seq]=mf
 return counts,frames,maxframe

def match_data(path:Path):
 total=defaultdict(lambda:defaultdict(Counter));events=defaultdict(lambda:defaultdict(list))
 with path.open(newline='') as f:
  for r in csv.DictReader(f):
   seq=r['seq']
   if seq not in SEQS:continue
   t=int(r['track_id']);g=int(r['gt_id']);fr=int(r['frame'])
   total[seq][t][g]+=1;events[seq][t].append((fr,g))
 for seq in events:
  for t in events[seq]:events[seq][t].sort()
 return total,events

def contrib(gtc:Counter,tc:int,m:Counter)->float:
 return float(sum((n*n)/max(1,gtc[g]+tc-n) for g,n in m.items() if n>0))
def interval_count(arr,start,end):
 if arr is None or len(arr)==0:return 0
 return int(np.searchsorted(arr,end,side='right')-np.searchsorted(arr,start,side='left'))
def interval_matches(rows,start,end):
 c=Counter()
 for fr,g in rows:
  if fr<start:continue
  if fr>end:break
  c[g]+=1
 return c
def counter_sub_add(base,sub,add):
 out=Counter(base)
 for k,v in sub.items():
  out[k]-=v
  if out[k]<=0:out.pop(k,None)
 out.update(add)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset-root',required=True);ap.add_argument('--track-root',required=True);ap.add_argument('--gt-root',required=True);ap.add_argument('--matches-csv',required=True);ap.add_argument('--out-dir',required=True);args=ap.parse_args()
 out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
 parts=[pd.read_csv(Path(args.dataset_root)/f'{s}.csv') for s in SEQS];df=pd.concat(parts,ignore_index=True)
 gtc=gt_counts(Path(args.gt_root));tc,tframes,maxframe=track_data(Path(args.track_root));mc,me=match_data(Path(args.matches_csv))
 base_cache={(s,t):contrib(gtc[s],n,mc[s].get(t,Counter())) for s in SEQS for t,n in tc[s].items()}
 results={h:[] for h in HORIZONS}
 before=[]
 for i,r in enumerate(df.itertuples(index=False),1):
  seq=str(r.seq);a=int(r.track_a);b=int(r.track_b);start=int(r.frame)
  old=base_cache.get((seq,a),0.0)+base_cache.get((seq,b),0.0);before.append(old)
  ma=mc[seq].get(a,Counter());mb=mc[seq].get(b,Counter())
  for h in HORIZONS:
   end=maxframe[seq] if h is None else min(maxframe[seq],start+h)
   mia=interval_matches(me[seq].get(a,[]),start,end);mib=interval_matches(me[seq].get(b,[]),start,end)
   na=tc[seq].get(a,0)-interval_count(tframes[seq].get(a),start,end)+interval_count(tframes[seq].get(b),start,end)
   nb=tc[seq].get(b,0)-interval_count(tframes[seq].get(b),start,end)+interval_count(tframes[seq].get(a),start,end)
   new_ma=counter_sub_add(ma,mia,mib);new_mb=counter_sub_add(mb,mib,mia)
   new=contrib(gtc[seq],na,new_ma)+contrib(gtc[seq],nb,new_mb)
   results[h].append((new,new-old,na,nb,sum(mia.values()),sum(mib.values())))
  if i%1000==0:print('labeled',i,'/',len(df),flush=True)
 df['assa_swap_before_proxy']=before
 for h in HORIZONS:
  tag='perm' if h is None else f'h{h}'
  vals=results[h]
  df[f'assa_swap_after_{tag}_proxy']=[x[0] for x in vals]
  df[f'assa_swap_delta_{tag}_proxy']=[x[1] for x in vals]
  df[f'assa_swap_positive_{tag}']=(df[f'assa_swap_delta_{tag}_proxy']>0).astype(int)
  df[f'swap_tracker_count_a_{tag}']=[x[2] for x in vals]
  df[f'swap_tracker_count_b_{tag}']=[x[3] for x in vals]
  df[f'swap_matched_rows_a_{tag}']=[x[4] for x in vals]
  df[f'swap_matched_rows_b_{tag}']=[x[5] for x in vals]
 df.to_csv(out/'swap_event_association_utility.csv',index=False)
 report={'rows':len(df),'by_horizon':{}}
 for h in HORIZONS:
  tag='perm' if h is None else f'h{h}';c=f'assa_swap_delta_{tag}_proxy';s=df[c]
  report['by_horizon'][tag]={'positive':int((s>0).sum()),'positive_rate':float((s>0).mean()),'positive_sum':float(s[s>0].sum()),'negative_sum':float(s[s<0].sum()),'quantiles':{str(k):float(v) for k,v in s.quantile([0,.01,.1,.5,.9,.99,1]).items()}}
 (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
