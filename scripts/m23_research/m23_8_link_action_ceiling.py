from __future__ import annotations

# Research artifact for the MOT20 M23 sequence-LOSO audit.
import csv,importlib.util,json,subprocess,sys
from collections import Counter,defaultdict
from pathlib import Path
import pandas as pd

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
PARENT=Path('outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results')
SCORES=Path('outputs/spot_runtime_gate_20260628/A42_long_gap_global_association/A42_02b_ranking_model_train_eval/a42_train_oof_scores.csv')
OUT=Path('outputs/mot20_m23_20260718/link_action_ceiling_v1')

def load_m23():
 s=importlib.util.spec_from_file_location('m23','scripts/audit_m23_mot20_expanded_evidence_oracle.py');m=importlib.util.module_from_spec(s);sys.modules['m23']=m;s.loader.exec_module(m);return m

def read_rows(path):
 rows=[]
 for line in path.open():
  p=line.rstrip('\n').split(',')
  if len(p)>=6:rows.append(p)
 return rows

def main():
 m=load_m23();df=pd.read_csv(SCORES);td=OUT/'track_results';td.mkdir(parents=True,exist_ok=True);plan=[]
 for seq in SEQS:
  baseline=m.load_baseline(PARENT/f'{seq}.txt');gt=m.load_gt(Path('datasets/MOT20/train')/seq/'gt/gt.txt');counts=defaultdict(Counter);spans={}
  for fr in sorted(baseline):
   for c in baseline[fr]:
    t=int(c.original_id);spans.setdefault(t,[fr,fr]);spans[t]=[min(spans[t][0],fr),max(spans[t][1],fr)]
   kept,valid,_=m.valid_and_distractor_filtered(baseline[fr],gt.get(fr,[]))
   for ci,gi,iou in m.match_candidates(kept,valid):counts[int(kept[ci].original_id)][int(valid[gi].gt_id)]+=1
  support={t:(c.most_common(1)[0][0] if c else 0) for t,c in counts.items()}
  # Candidate graph limits which IDs may be linked; per GT choose maximum-support non-overlapping path.
  g=df[df.seq.eq(seq)];adj=defaultdict(set)
  for r in g.itertuples():
   a=int(r.track_a);b=int(r.track_b)
   if a in spans and b in spans and spans[a][1]<spans[b][0] and support.get(a,0)>0 and support.get(a)==support.get(b):adj[a].add(b)
  idmap={}
  for gid in sorted(set(support.values())-{0}):
   ids=sorted([t for t,x in support.items() if x==gid],key=lambda t:(spans[t][0],spans[t][1],t))
   # DAG longest path, edge only if present in A42 candidate graph. Weight is matched support of destination.
   best={t:(counts[t][gid],(t,)) for t in ids}
   for b in ids:
    options=[(best[a][0]+counts[b][gid],best[a][1]+(b,)) for a in ids if b in adj.get(a,set()) and spans[a][1]<spans[b][0]]
    if options:best[b]=max([best[b],*options],key=lambda x:(x[0],len(x[1]),tuple(-z for z in x[1])))
   path=max(best.values(),key=lambda x:(x[0],len(x[1]),tuple(-z for z in x[1])))[1]
   if len(path)>1:
    root=path[0]
    for t in path:idmap[t]=root
    for a,b in zip(path,path[1:]):plan.append({'seq':seq,'gt_id':gid,'track_a':a,'track_b':b,'support_a':counts[a][gid],'support_b':counts[b][gid]})
  rows=read_rows(PARENT/f'{seq}.txt');out=[];frames=defaultdict(list)
  for p in rows:
   q=list(p);old=int(float(q[1]));new=idmap.get(old,old);q[1]=str(new);frames[int(float(q[0]))].append(new);out.append(q)
  dup=[f for f,x in frames.items() if len(x)!=len(set(x))]
  if dup:raise RuntimeError(f'{seq}: duplicate {dup[:5]}')
  out.sort(key=lambda p:(int(float(p[0])),int(float(p[1])),float(p[2]),float(p[3])))
  with (td/f'{seq}.txt').open('w') as f:
   for p in out:f.write(','.join(p)+'\n')
 with (OUT/'oracle_links.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(plan[0]));w.writeheader();w.writerows(plan)
 name='link_action_ceiling';cmd=[sys.executable,'scripts/eval_motstyle_trackeval.py','--benchmark-name','MOT20','--split-to-eval','train','--gt-root','datasets/MOT20/train','--results-dir',str(td),'--tracker-name',name,'--work-dir',str(OUT/'eval_work'),'--seqs',*SEQS]
 p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(OUT/'eval.log').write_text(p.stdout)
 if p.returncode:raise RuntimeError(p.stdout[-4000:])
 rr=list(csv.DictReader((OUT/'eval_work/eval'/name/'pedestrian_detailed.csv').open()));res={}
 for s in SEQS+['COMBINED']:
  r=next(x for x in rr if x['seq']==s);res[s]={'HOTA':100*float(r['HOTA___AUC']),'DetA':100*float(r['DetA___AUC']),'AssA':100*float(r['AssA___AUC']),'IDSW':int(float(r['IDSW']))}
 report={'oracle':True,'deployment_allowed':False,'candidate_graph':'all A42 edges with current-parent modal-GT equality','links':len(plan),'by_seq_links':dict(Counter(r['seq'] for r in plan)),'eval':res};(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
