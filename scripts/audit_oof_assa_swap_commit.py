from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
HORIZONS=['h10','h30','h60','perm']

def score_columns(df,h):
 p=f'assa_swap_{h}_'
 cols=[]
 for c in df.columns:
  if c.startswith(p) and not c.endswith('_seqpct') and any(x in c for x in ['_p_et','_p_hgb','_safe_et','_safe_hgb','_risk_et_l','_risk_hgb_l','_signed_log_pred']):
   if c+'_seqpct' in df.columns:cols.append(c)
 return sorted(set(cols))

def greedy(g,col,spacing):
 g=g.sort_values([col,'candidate_ioa','frame'],ascending=[False,False,True])
 selected=[];last={}
 for r in g.itertuples(index=False):
  a=int(r.track_a);b=int(r.track_b);fr=int(r.frame)
  if abs(fr-last.get(a,-10**9))<spacing or abs(fr-last.get(b,-10**9))<spacing:continue
  selected.append(r);last[a]=fr;last[b]=fr
 return selected

def optimal_cutoff(path,col,delta_col):
 if not path:return 1.000001,0,0.0
 d=np.asarray([float(getattr(r,delta_col)) for r in path]);cum=np.cumsum(d);k=int(np.argmax(cum))+1
 if cum[k-1]<=0:return 1.000001,0,0.0
 cutoff=min(float(getattr(r,col+'_seqpct')) for r in path[:k])
 return cutoff,k,float(cum[k-1])
def apply(path,col,delta_col,cutoff):
 sel=[r for r in path if float(getattr(r,col+'_seqpct'))>=cutoff]
 d=np.asarray([float(getattr(r,delta_col)) for r in sel])
 return {'selected':len(sel),'positive':int((d>0).sum()) if len(d) else 0,'precision':float((d>0).mean()) if len(d) else 0.0,'delta_sum':float(d.sum()) if len(d) else 0.0,'positive_delta':float(d[d>0].sum()) if len(d) else 0.0,'negative_delta':float(d[d<0].sum()) if len(d) else 0.0,'worst':float(d.min()) if len(d) else 0.0}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scores',required=True);ap.add_argument('--out-dir',required=True);args=ap.parse_args()
 df=pd.read_csv(args.scores);out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True);rows=[]
 for h in HORIZONS:
  delta_col=f'assa_swap_delta_{h}_proxy'
  for col in score_columns(df,h):
   for spacing in [10,20,30,60,120]:
    paths={s:greedy(df[df.seq==s],col,spacing) for s in SEQS};opts={s:optimal_cutoff(paths[s],col,delta_col) for s in SEQS}
    for agg,q in [('median',.5),('q75',.75),('max',1.0)]:
     for held in SEQS:
      cuts=[opts[s][0] for s in SEQS if s!=held and opts[s][1]>0]
      cutoff=float(np.quantile(cuts,q)) if cuts else 1.000001
      z=apply(paths[held],col,delta_col,cutoff)
      rows.append({'horizon':h,'score_col':col,'min_spacing':spacing,'aggregation':agg,'heldout_seq':held,'learned_cutoff':cutoff,'train_opt_cutoffs':json.dumps({s:opts[s][0] for s in SEQS if s!=held}),'train_opt_k':json.dumps({s:opts[s][1] for s in SEQS if s!=held}),'train_opt_delta':json.dumps({s:opts[s][2] for s in SEQS if s!=held}),**z})
 res=pd.DataFrame(rows);res.to_csv(out/'heldout_diagnostics.csv',index=False)
 agg=res.groupby(['horizon','score_col','min_spacing','aggregation']).agg(selected=('selected','sum'),positive=('positive','sum'),delta_sum=('delta_sum','sum'),positive_delta=('positive_delta','sum'),negative_delta=('negative_delta','sum'),worst=('worst','min'),positive_sequences=('delta_sum',lambda s:int((s>0).sum())),negative_sequences=('delta_sum',lambda s:int((s<0).sum())),mean_precision=('precision','mean')).reset_index();agg['precision']=agg.positive/agg.selected.clip(lower=1);agg.to_csv(out/'policy_summary.csv',index=False)
 print('=== best overall ===')
 print(agg.sort_values(['delta_sum','positive_sequences','worst'],ascending=False).head(50).to_string(index=False))
 safe=agg[(agg.delta_sum>0)&(agg.negative_sequences==0)&(agg.selected>0)]
 print('\n=== nonnegative heldout policies ===')
 print(safe.sort_values(['delta_sum','worst','precision'],ascending=False).head(50).to_string(index=False))
 top=agg.sort_values(['delta_sum','positive_sequences','worst'],ascending=False).head(20)[['horizon','score_col','min_spacing','aggregation']]
 details=res.merge(top,on=['horizon','score_col','min_spacing','aggregation']).sort_values(['horizon','score_col','min_spacing','aggregation','heldout_seq'])
 details.to_csv(out/'top_policy_details.csv',index=False)
 (out/'summary.json').write_text(json.dumps({'diagnostics':len(res),'policies':len(agg),'best':agg.sort_values(['delta_sum','positive_sequences','worst'],ascending=False).head(50).to_dict('records')},indent=2)+'\n')
if __name__=='__main__':main()
