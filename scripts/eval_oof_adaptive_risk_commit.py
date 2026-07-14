from __future__ import annotations
import argparse,csv,json,subprocess,sys
from collections import defaultdict
from pathlib import Path
import pandas as pd

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
POLICIES=[
 {'name':'l1_r2_q75','score_col':'meta_risk_score_l1p0','max_rank':2,'aggregation':'q75'},
 {'name':'l1_r1_max','score_col':'meta_risk_score_l1p0','max_rank':1,'aggregation':'max'},
 {'name':'l2_r1_q75','score_col':'meta_risk_score_l2p0','max_rank':1,'aggregation':'q75'},
 {'name':'l025_r1_max','score_col':'meta_risk_score_l0p25','max_rank':1,'aggregation':'max'},
]

def add_ranks(df,col):
 x=df.sort_values(['seq','track_a',col,'gap','track_b'],ascending=[True,True,False,True,True]).copy()
 x['out_rank']=x.groupby(['seq','track_a']).cumcount()+1
 x=x.sort_values(['seq','track_b',col,'gap','track_a'],ascending=[True,True,False,True,True])
 x['in_rank']=x.groupby(['seq','track_b']).cumcount()+1
 x['max_rank_policy']=x[['out_rank','in_rank']].max(axis=1)
 return x

def read_rows(path):
 rows=[];spans={}
 with open(path) as f:
  for line in f:
   p=line.strip().split(',')
   if len(p)<6:continue
   fr=int(float(p[0]));tid=int(float(p[1]));rows.append(p)
   spans.setdefault(tid,[fr,fr]);spans[tid]=[min(spans[tid][0],fr),max(spans[tid][1],fr)]
 return rows,spans

def apply(src,out,cand,score_col):
 rows,spans=read_rows(src);parent={};used_s=set();used_t=set();sel=[]
 def find(x):
  parent.setdefault(x,x)
  while parent[x]!=x:
   parent[x]=parent[parent[x]];x=parent[x]
  return x
 for r in cand.sort_values([score_col,'gap'],ascending=[False,True]).itertuples(index=False):
  a=int(r.track_a);b=int(r.track_b)
  if a not in spans or b not in spans or spans[a][1]>=spans[b][0]:continue
  if a in used_s or b in used_t:continue
  ra,rb=find(a),find(b)
  if ra==rb:continue
  parent[rb]=ra;used_s.add(a);used_t.add(b);sel.append(r._asdict())
 involved={int(r[k]) for r in sel for k in ['track_a','track_b']}
 idmap={tid:find(tid) for tid in involved}
 frames=defaultdict(list);outrows=[];changed=0
 for p in rows:
  q=list(p);tid=int(float(q[1]));new=idmap.get(tid,tid);changed+=int(new!=tid);q[1]=str(new);frames[int(float(q[0]))].append(new);outrows.append(q)
 dup=[fr for fr,ids in frames.items() if len(ids)!=len(set(ids))]
 if dup:raise RuntimeError(f'duplicate IDs {src}: {dup[:5]}')
 out.parent.mkdir(parents=True,exist_ok=True)
 outrows.sort(key=lambda p:(int(float(p[0])),int(float(p[1])),float(p[2]),float(p[3])))
 with open(out,'w') as f:
  for p in outrows:f.write(','.join(p)+'\n')
 return sel,{'links':len(sel),'changed_rows':changed,'utility_sum':sum(float(r['assa_merge_delta_proxy']) for r in sel),'positive':sum(float(r['assa_merge_delta_proxy'])>0 for r in sel)}
def evaluate(pdir,name):
 cmd=[sys.executable,'scripts/eval_motstyle_trackeval.py','--benchmark-name','MOT20','--split-to-eval','train','--gt-root','datasets/MOT20/train','--results-dir',str(pdir/'track_results'),'--tracker-name',name,'--work-dir',str(pdir/'eval_work'),'--seqs',*SEQS]
 p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(pdir/'eval.log').write_text(p.stdout)
 detail=pdir/'eval_work/eval'/name/'pedestrian_detailed.csv';res={'returncode':p.returncode}
 if detail.exists():
  rows=list(csv.DictReader(open(detail)))
  for seq in SEQS+['COMBINED']:
   r=next((x for x in rows if x['seq']==seq),None)
   if r:res[seq]={'HOTA':float(r['HOTA___AUC'])*100,'DetA':float(r['DetA___AUC'])*100,'AssA':float(r['AssA___AUC'])*100,'IDF1':float(r['IDF1'])*100 if float(r['IDF1'])<2 else float(r['IDF1']),'IDSW':int(float(r['IDSW']))}
  res['simple_avg_HOTA']=sum(res[s]['HOTA'] for s in SEQS)/4
 return res
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scores',required=True);ap.add_argument('--diagnostics',required=True);ap.add_argument('--source-dir',required=True);ap.add_argument('--out-dir',required=True);args=ap.parse_args()
 use=['seq','track_a','track_b','gap','endpoint_reid_aligned','appearance_max','max_debt_pct','assa_merge_delta_proxy','assa_merge_positive']
 for p in POLICIES:use += [p['score_col'],p['score_col']+'_seqpct']
 df=pd.read_csv(args.scores,usecols=lambda c:c in set(use));diag=pd.read_csv(args.diagnostics);out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True);summaries=[]
 for pol in POLICIES:
  ranked=add_ranks(df,pol['score_col']);pdir=out/pol['name'];tdir=pdir/'track_results';tdir.mkdir(parents=True,exist_ok=True);allsel=[];byseq=[]
  for seq in SEQS:
   row=diag[(diag.score_col==pol['score_col'])&(diag.max_rank==pol['max_rank'])&(diag.aggregation==pol['aggregation'])&(diag.heldout_seq==seq)].iloc[0]
   cutoff=float(row.learned_cutoff)
   cand=ranked[(ranked.seq==seq)&(ranked.max_rank_policy<=pol['max_rank'])&(ranked.endpoint_reid_aligned==1)&(ranked.appearance_max>=.65)&(ranked.max_debt_pct>=.5)&(ranked[pol['score_col']+'_seqpct']>=cutoff)]
   sel,st=apply(Path(args.source_dir)/f'{seq}.txt',tdir/f'{seq}.txt',cand,pol['score_col']);st.update({'seq':seq,'cutoff':cutoff,'candidates':len(cand)});byseq.append(st);allsel+=sel
  fields=[]
  for r in allsel:
   for k in r:
    if k not in fields:fields.append(k)
  with open(pdir/'selected_links.csv','w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(allsel)
  ev=evaluate(pdir,'adaptive_'+pol['name'])
  summary={'policy':pol,'selected_links':len(allsel),'utility_sum':sum(float(r['assa_merge_delta_proxy']) for r in allsel),'positive':sum(float(r['assa_merge_delta_proxy'])>0 for r in allsel),'by_seq':byseq,'eval':ev}
  (pdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');summaries.append(summary)
  print(json.dumps({'policy':pol['name'],'selected':summary['selected_links'],'utility':summary['utility_sum'],'positive':summary['positive'],'combined':ev.get('COMBINED'),'m02':ev.get('MOT20-02'),'avg':ev.get('simple_avg_HOTA')},indent=2),flush=True)
 (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
if __name__=='__main__':main()
