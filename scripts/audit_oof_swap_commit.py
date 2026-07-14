from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
SCORES=[
 'swap_p_et','swap_p_hgb','swap_safe_et','swap_safe_hgb','swap_signed_log_pred',
 'swap_risk_score_et_l0p25','swap_risk_score_et_l0p5','swap_risk_score_et_l1p0','swap_risk_score_et_l2p0',
 'swap_risk_score_hgb_l0p25','swap_risk_score_hgb_l0p5','swap_risk_score_hgb_l1p0','swap_risk_score_hgb_l2p0'
]

def greedy_events(g,col,min_spacing):
 g=g.sort_values([col,'candidate_ioa','frame'],ascending=[False,False,True])
 selected=[];last_by_track={}
 for r in g.itertuples(index=False):
  a=int(r.track_a);b=int(r.track_b);fr=int(r.frame)
  if abs(fr-last_by_track.get(a,-10**9))<min_spacing:continue
  if abs(fr-last_by_track.get(b,-10**9))<min_spacing:continue
  selected.append(r);last_by_track[a]=fr;last_by_track[b]=fr
 return selected

def optimal_cutoff(path,col):
 if not path:return 1.000001,0,0.0
 d=np.array([float(r.swap_utility_30) for r in path]);cum=np.cumsum(d);k=int(np.argmax(cum))+1
 if cum[k-1]<=0:return 1.000001,0,0.0
 cutoff=min(float(getattr(r,col+'_seqpct')) for r in path[:k])
 return cutoff,k,float(cum[k-1])

def apply_cutoff(path,col,cutoff):
 sel=[r for r in path if float(getattr(r,col+'_seqpct'))>=cutoff]
 d=np.array([float(r.swap_utility_30) for r in sel])
 return {'selected':len(sel),'positive':int((d>0).sum()) if len(d) else 0,'precision':float((d>0).mean()) if len(d) else 0.0,'utility_sum':float(d.sum()) if len(d) else 0.0,'positive_utility':float(d[d>0].sum()) if len(d) else 0.0,'negative_utility':float(d[d<0].sum()) if len(d) else 0.0,'worst':float(d.min()) if len(d) else 0.0}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scores',required=True);ap.add_argument('--out-dir',required=True);args=ap.parse_args()
 use=['seq','track_a','track_b','frame','candidate_ioa','swap_utility_30']
 for c in SCORES:use += [c,c+'_seqpct']
 df=pd.read_csv(args.scores,usecols=lambda c:c in set(use));out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True);rows=[]
 for col in SCORES:
  for spacing in [10,20,30,60]:
   paths={s:greedy_events(df[df.seq==s],col,spacing) for s in SEQS};opts={s:optimal_cutoff(paths[s],col) for s in SEQS}
   for agg,q in [('median',.5),('q75',.75),('max',1.0)]:
    for held in SEQS:
     cuts=[opts[s][0] for s in SEQS if s!=held and opts[s][1]>0]
     cutoff=float(np.quantile(cuts,q)) if cuts else 1.000001
     diag=apply_cutoff(paths[held],col,cutoff)
     rows.append({'score_col':col,'min_spacing':spacing,'aggregation':agg,'heldout_seq':held,'learned_cutoff':cutoff,'train_opt_cutoffs':json.dumps({s:opts[s][0] for s in SEQS if s!=held}),'train_opt_k':json.dumps({s:opts[s][1] for s in SEQS if s!=held}),'train_opt_utility':json.dumps({s:opts[s][2] for s in SEQS if s!=held}),**diag})
 res=pd.DataFrame(rows);res.to_csv(out/'heldout_diagnostics.csv',index=False)
 agg=res.groupby(['score_col','min_spacing','aggregation']).agg(selected=('selected','sum'),positive=('positive','sum'),utility_sum=('utility_sum','sum'),positive_utility=('positive_utility','sum'),negative_utility=('negative_utility','sum'),worst=('worst','min'),positive_sequences=('utility_sum',lambda s:int((s>0).sum())),negative_sequences=('utility_sum',lambda s:int((s<0).sum())),mean_precision=('precision','mean')).reset_index();agg['precision']=agg.positive/agg.selected.clip(lower=1);agg.to_csv(out/'policy_summary.csv',index=False)
 print('=== best utility ===');print(agg.sort_values(['utility_sum','positive_sequences','worst'],ascending=False).head(40).to_string(index=False))
 safe=agg[(agg.utility_sum>0)&(agg.negative_sequences==0)&(agg.selected>0)]
 print('\n=== positive in every active heldout ===');print(safe.sort_values(['utility_sum','worst','precision'],ascending=False).head(40).to_string(index=False))
 top=agg.sort_values(['utility_sum','positive_sequences'],ascending=False).head(12)[['score_col','min_spacing','aggregation']]
 print('\n=== top heldout details ===');print(res.merge(top,on=['score_col','min_spacing','aggregation']).sort_values(['score_col','min_spacing','aggregation','heldout_seq']).to_string(index=False))
 (out/'summary.json').write_text(json.dumps({'rows':len(res),'policies':len(agg),'best':agg.sort_values(['utility_sum','positive_sequences'],ascending=False).head(30).to_dict('records')},indent=2)+'\n')
if __name__=='__main__':main()
