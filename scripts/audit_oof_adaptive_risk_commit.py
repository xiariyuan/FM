from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
SCORES=[
 'meta_safe_benefit',
 'meta_risk_score_l0p25',
 'meta_risk_score_l0p5',
 'meta_risk_score_l1p0',
 'meta_risk_score_l2p0',
]

def add_ranks(df,col):
 x=df.sort_values(['seq','track_a',col,'gap','track_b'],ascending=[True,True,False,True,True]).copy()
 x['out_rank']=x.groupby(['seq','track_a']).cumcount()+1
 x=x.sort_values(['seq','track_b',col,'gap','track_a'],ascending=[True,True,False,True,True])
 x['in_rank']=x.groupby(['seq','track_b']).cumcount()+1
 x['max_rank_policy']=x[['out_rank','in_rank']].max(axis=1)
 return x

def conflict_path(g,col,max_rank):
 g=g[(g.max_rank_policy<=max_rank)&(g.endpoint_reid_aligned==1)&(g.appearance_max>=.65)&(g.max_debt_pct>=.5)]
 g=g.sort_values([col,'gap'],ascending=[False,True])
 used_s=set();used_t=set();parent={};out=[]
 def find(x):
  parent.setdefault(x,x)
  while parent[x]!=x:
   parent[x]=parent[parent[x]];x=parent[x]
  return x
 for r in g.itertuples(index=False):
  a=int(r.track_a);b=int(r.track_b)
  if a in used_s or b in used_t:continue
  ra,rb=find(a),find(b)
  if ra==rb:continue
  parent[rb]=ra;used_s.add(a);used_t.add(b);out.append(r)
 return out

def optimal_cutoff(path,col):
 if not path:return 1.000001,0,0.0
 delta=np.array([float(r.assa_merge_delta_proxy) for r in path])
 cum=np.cumsum(delta)
 k=int(np.argmax(cum))+1
 if cum[k-1]<=0:return 1.000001,0,0.0
 cutoff=min(float(getattr(r,col+'_seqpct')) for r in path[:k])
 return cutoff,k,float(cum[k-1])

def apply_cutoff(path,col,cutoff):
 sel=[r for r in path if float(getattr(r,col+'_seqpct'))>=cutoff]
 d=np.array([float(r.assa_merge_delta_proxy) for r in sel],dtype=float)
 return {'selected':len(sel),'positive':int((d>0).sum()) if len(d) else 0,'precision':float((d>0).mean()) if len(d) else 0.0,'utility_sum':float(d.sum()) if len(d) else 0.0,'positive_utility':float(d[d>0].sum()) if len(d) else 0.0,'negative_utility':float(d[d<0].sum()) if len(d) else 0.0,'worst':float(d.min()) if len(d) else 0.0}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scores',required=True);ap.add_argument('--out-dir',required=True);args=ap.parse_args()
 use=['seq','track_a','track_b','gap','endpoint_reid_aligned','appearance_max','max_debt_pct','assa_merge_delta_proxy','assa_merge_positive']
 for c in SCORES:use += [c,c+'_seqpct']
 df=pd.read_csv(args.scores,usecols=lambda c:c in set(use))
 out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for col in SCORES:
  ranked=add_ranks(df,col)
  for max_rank in [1,2,3]:
   paths={s:conflict_path(ranked[ranked.seq==s],col,max_rank) for s in SEQS}
   opts={s:optimal_cutoff(paths[s],col) for s in SEQS}
   for agg_name,q in [('median',.5),('q75',.75),('max',1.0)]:
    for held in SEQS:
     train_cut=[opts[s][0] for s in SEQS if s!=held and opts[s][1]>0]
     cutoff=float(np.quantile(train_cut,q)) if train_cut else 1.000001
     diag=apply_cutoff(paths[held],col,cutoff)
     rows.append({'score_col':col,'max_rank':max_rank,'aggregation':agg_name,'heldout_seq':held,'learned_cutoff':cutoff,'train_opt_cutoffs':json.dumps({s:opts[s][0] for s in SEQS if s!=held}),'train_opt_k':json.dumps({s:opts[s][1] for s in SEQS if s!=held}),'train_opt_utility':json.dumps({s:opts[s][2] for s in SEQS if s!=held}),**diag})
 res=pd.DataFrame(rows)
 res.to_csv(out/'heldout_policy_diagnostics.csv',index=False)
 agg=res.groupby(['score_col','max_rank','aggregation']).agg(selected=('selected','sum'),positive=('positive','sum'),utility_sum=('utility_sum','sum'),positive_utility=('positive_utility','sum'),negative_utility=('negative_utility','sum'),worst=('worst','min'),mean_precision=('precision','mean'),positive_sequences=('utility_sum',lambda s:int((s>0).sum())),negative_sequences=('utility_sum',lambda s:int((s<0).sum()))).reset_index()
 agg['precision']=agg.positive/agg.selected.clip(lower=1)
 agg.to_csv(out/'policy_summary.csv',index=False)
 print('=== best total utility ===')
 print(agg.sort_values(['utility_sum','positive_sequences','precision'],ascending=False).head(30).to_string(index=False))
 print('\n=== safest positive ===')
 print(agg[(agg.utility_sum>0)&(agg.selected>0)].sort_values(['negative_sequences','worst','utility_sum'],ascending=[True,False,False]).head(30).to_string(index=False))
 print('\n=== per heldout for top candidates ===')
 top=agg.sort_values(['utility_sum','positive_sequences'],ascending=False).head(8)[['score_col','max_rank','aggregation']]
 print(res.merge(top,on=['score_col','max_rank','aggregation']).sort_values(['score_col','max_rank','aggregation','heldout_seq']).to_string(index=False))
 (out/'summary.json').write_text(json.dumps({'rows':len(res),'policies':len(agg),'best':agg.sort_values(['utility_sum','positive_sequences'],ascending=False).head(20).to_dict('records')},indent=2)+'\n')
if __name__=='__main__':main()
