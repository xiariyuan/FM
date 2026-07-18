from __future__ import annotations

# Research artifact for the MOT20 M23 sequence-LOSO audit.
import argparse,csv,json,math,subprocess,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
PARENT=Path('outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results')
DATA=Path('outputs/mot20_m23_20260718/micrograph_chunk30_v1')
PRED=Path('outputs/mot20_m23_20260718/micrograph_chunk30_loso_v1')
OUT=Path('outputs/mot20_m23_20260718/micrograph_chunk30_policy_explore_v1')
SYN_BASE=10_000_000;STRIDE=1_000_000;CHUNK_SPAN=30;GAP_BREAK=1

def read_parent(path):
 rows=[]
 for li,line in enumerate(path.open()):
  p=line.rstrip('\n').split(',')
  if len(p)<6:continue
  rows.append({'line':li,'frame':int(float(p[0])),'track_id':int(float(p[1])),'fields':p})
 return rows

def line_chunks(rows,meta):
 by=defaultdict(list)
 for i,r in enumerate(rows):by[r['track_id']].append(i)
 mapping={};cid=0
 for tid,ids in sorted(by.items()):
  ids.sort(key=lambda i:(rows[i]['frame'],i));start=0;ordinal=0
  for k in range(1,len(ids)+1):
   br=k==len(ids) or rows[ids[k]]['frame']-rows[ids[k-1]]['frame']>GAP_BREAK or rows[ids[k]]['frame']-rows[ids[start]]['frame']>=CHUNK_SPAN
   if not br:continue
   row=meta.iloc[cid]
   if int(row.source_track_id)!=tid or int(row.source_ordinal)!=ordinal or int(row.first_frame)!=rows[ids[start]]['frame'] or int(row.last_frame)!=rows[ids[k-1]]['frame']:
    raise RuntimeError(f'chunk reconstruction mismatch at {cid}')
   for q in ids[start:k]:mapping[q]=cid
   cid+=1;ordinal+=1;start=k
 if cid!=len(meta) or len(mapping)!=len(rows):raise RuntimeError(f'chunk count mismatch {cid} {len(meta)}')
 return mapping

def policy_matching(edges,n,tau,bonus):
 p=np.clip(edges.pred_same_prob.to_numpy(float),1e-6,1-1e-6);benefit=np.log(p/(1-p))-math.log(tau/(1-tau))+bonus*edges.source_adjacent.to_numpy(float);keep=benefit>0
 e=edges.loc[keep].copy();b=benefit[keep];utility=1/(1+np.exp(-b));rr=e.src_chunk.to_numpy(int);cc=e.dst_chunk.to_numpy(int);cost=1.000001-utility
 rows=np.concatenate([rr,np.arange(n,dtype=int)]);cols=np.concatenate([cc,n+np.arange(n,dtype=int)]);data=np.concatenate([cost,np.ones(n,float)])
 mat=coo_matrix((data,(rows,cols)),shape=(n,2*n)).tocsr();ri,ci=min_weight_full_bipartite_matching(mat);chosen=ci<n;selected_pairs=set(zip(ri[chosen].tolist(),ci[chosen].tolist()));sel=e[[ (int(a),int(b)) in selected_pairs for a,b in zip(e.src_chunk,e.dst_chunk)]].copy();sel['benefit']=[benefit[idx] for idx in sel.index]
 return sel

def chains(selected,n):
 succ={int(r.src_chunk):int(r.dst_chunk) for r in selected.itertuples()};pred={v:k for k,v in succ.items()}
 if len(succ)!=len(pred):raise RuntimeError('matching not one-to-one')
 roots=[i for i in range(n) if i not in pred];chain={};
 for root in roots:
  cur=root;seen=set()
  while cur not in seen:
   seen.add(cur);chain[cur]=root
   if cur not in succ:break
   cur=succ[cur]
  else:raise RuntimeError('cycle')
 if len(chain)!=n:raise RuntimeError(f'unassigned chunks {n-len(chain)}')
 return chain

def write_tracker(seq,meta,selected,path):
 rows=read_parent(PARENT/f'{seq}.txt');mapping=line_chunks(rows,meta);chain=chains(selected,len(meta));base=SYN_BASE+SEQS.index(seq)*STRIDE;out=[];seen=set()
 for i,r in enumerate(rows):
  cid=mapping[i];nid=base+chain[cid]
  if nid>=(1<<24):raise RuntimeError('ID out of float32 exact range')
  q=list(r['fields']);q[1]=str(nid);key=(r['frame'],nid)
  if key in seen:raise RuntimeError(f'{seq}: duplicate {key}')
  seen.add(key);out.append(q)
 out.sort(key=lambda p:(int(float(p[0])),int(float(p[1])),float(p[2]),float(p[3])))
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w') as f:
  for p in out:f.write(','.join(p)+'\n')
 return {'rows':len(rows),'chains':len(set(chain.values())),'selected_edges':len(selected),'cross_edges':int((selected.same_source==0).sum()),'source_adjacent_edges':int(selected.source_adjacent.sum()),'diagnostic_same_gt':int(selected.same_gt.sum()),'diagnostic_precision':float(selected.same_gt.mean()) if len(selected) else None,'diagnostic_cross_same_gt':int(selected.loc[selected.same_source==0,'same_gt'].sum()),'diagnostic_cross_precision':float(selected.loc[selected.same_source==0,'same_gt'].mean()) if (selected.same_source==0).any() else None}

def evaluate(root,name):
 cmd=[sys.executable,'scripts/eval_motstyle_trackeval.py','--benchmark-name','MOT20','--split-to-eval','train','--gt-root','datasets/MOT20/train','--results-dir',str(root/'track_results'),'--tracker-name',name,'--work-dir',str(root/'eval_work'),'--seqs',*SEQS]
 p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(root/'eval.log').write_text(p.stdout)
 if p.returncode:raise RuntimeError(p.stdout[-4000:])
 rr=list(csv.DictReader((root/'eval_work/eval'/name/'pedestrian_detailed.csv').open()));res={}
 for s in SEQS+['COMBINED']:
  r=next(x for x in rr if x['seq']==s);res[s]={'HOTA':100*float(r['HOTA___AUC']),'DetA':100*float(r['DetA___AUC']),'AssA':100*float(r['AssA___AUC']),'IDSW':int(float(r['IDSW']))}
 return res

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--policy',action='append',required=True,help='NAME,TAU,BONUS');ap.add_argument('--no-eval',action='store_true');args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);summ=[]
 cache={}
 for seq in SEQS:
  cache[seq]=(pd.read_parquet(DATA/seq/'microtracklets.parquet'),pd.read_parquet(PRED/f'{seq}_edge_predictions.parquet'))
 for spec in args.policy:
  name,t,b=spec.split(',');tau=float(t);bonus=float(b);root=OUT/name;by=[]
  for seq in SEQS:
   meta,edges=cache[seq];sel=policy_matching(edges,len(meta),tau,bonus);sd=root/'selected_edges';sd.mkdir(parents=True,exist_ok=True);sel.to_parquet(sd/f'{seq}.parquet',index=False);rec=write_tracker(seq,meta,sel,root/'track_results'/f'{seq}.txt');rec['seq']=seq;by.append(rec)
  report={'name':name,'tau':tau,'source_adjacent_bonus':bonus,'by_seq':by,'deployment_allowed':True,'gt_used_in_inference':False}
  if not args.no_eval:report['eval']=evaluate(root,name)
  (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');summ.append(report);print(json.dumps({'name':name,'by_seq':by,'combined':report.get('eval',{}).get('COMBINED')},indent=2),flush=True)
 (OUT/'summary.json').write_text(json.dumps(summ,indent=2)+'\n')
if __name__=='__main__':main()
