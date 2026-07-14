from __future__ import annotations
import argparse,csv,json,subprocess,sys
from collections import defaultdict
from pathlib import Path
import pandas as pd

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
POLICY={'score_col':'swap_risk_score_hgb_l0p25','min_spacing':10,'aggregation':'q75'}

def read_rows(path):
 rows=[]
 with open(path) as f:
  for line in f:
   p=line.strip().split(',')
   if len(p)>=6:rows.append(p)
 return rows

def select_events(scores,diag):
 col=POLICY['score_col'];selected=[]
 for seq in SEQS:
  row=diag[(diag.score_col==col)&(diag.min_spacing==POLICY['min_spacing'])&(diag.aggregation==POLICY['aggregation'])&(diag.heldout_seq==seq)].iloc[0]
  cutoff=float(row.learned_cutoff)
  g=scores[(scores.seq==seq)&(scores[col+'_seqpct']>=cutoff)].sort_values([col,'candidate_ioa','frame'],ascending=[False,False,True])
  last={}
  for r in g.itertuples(index=False):
   a=int(r.track_a);b=int(r.track_b);fr=int(r.frame)
   if abs(fr-last.get(a,-10**9))<POLICY['min_spacing'] or abs(fr-last.get(b,-10**9))<POLICY['min_spacing']:continue
   d=r._asdict();d['learned_cutoff']=cutoff;selected.append(d);last[a]=fr;last[b]=fr
 return selected

def source_to_emitted_map(source_rows,target_rows):
 if len(source_rows)!=len(target_rows):raise RuntimeError('row count mismatch')
 out={}
 for s,t in zip(source_rows,target_rows):
  if s[0]!=t[0] or s[2:]!=t[2:]:raise RuntimeError('non-id rows differ')
  out[(int(float(s[0])),int(float(s[1])))]=int(float(t[1]))
 return out

def apply_swaps(source_internal_path,input_path,out_path,events):
 internal=read_rows(source_internal_path);rows=read_rows(input_path);internal_map=source_to_emitted_map(internal,rows)
 byframe=defaultdict(list)
 for e in events:byframe[int(e['frame'])].append(e)
 mapping={};out=[];applied=[];skipped=[];rows_by=defaultdict(list)
 for r in rows:rows_by[int(float(r[0]))].append(r)
 for fr in sorted(rows_by):
  for e in sorted(byframe.get(fr,[]),key=lambda x:(int(x['track_a']),int(x['track_b']))):
   ia=int(e['track_a']);ib=int(e['track_b'])
   la=internal_map.get((fr,ia));lb=internal_map.get((fr,ib))
   if la is None or lb is None or la==lb:
    skipped.append({**e,'reason':'missing_or_same_emitted_label','label_a':la,'label_b':lb});continue
   mapping.setdefault(la,la);mapping.setdefault(lb,lb)
   mapping[la],mapping[lb]=mapping[lb],mapping[la]
   applied.append({**e,'label_a_before':la,'label_b_before':lb})
  emitted=[]
  for src in rows_by[fr]:
   r=list(src);tid=int(float(r[1]));mapping.setdefault(tid,tid);r[1]=str(mapping[tid]);emitted.append(r)
  ids=[int(float(r[1])) for r in emitted]
  if len(ids)!=len(set(ids)):raise RuntimeError(f'duplicate IDs at frame {fr}')
  out.extend(emitted)
 out_path.parent.mkdir(parents=True,exist_ok=True)
 with open(out_path,'w') as f:
  for r in out:f.write(','.join(r)+'\n')
 return applied,skipped,sum(a[1]!=b[1] for a,b in zip(rows,out))
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
 ap=argparse.ArgumentParser();ap.add_argument('--scores',required=True);ap.add_argument('--diagnostics',required=True);ap.add_argument('--internal-root',required=True);ap.add_argument('--input-roots',nargs='+',required=True);ap.add_argument('--input-names',nargs='+',required=True);ap.add_argument('--out-dir',required=True);args=ap.parse_args()
 if len(args.input_roots)!=len(args.input_names):raise ValueError('root/name mismatch')
 scores=pd.read_csv(args.scores);diag=pd.read_csv(args.diagnostics);events=select_events(scores,diag);out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(events).to_csv(out/'selected_swap_events.csv',index=False)
 summaries=[]
 for root,name in zip(args.input_roots,args.input_names):
  pdir=out/name;tdir=pdir/'track_results';tdir.mkdir(parents=True,exist_ok=True);applied=[];skipped=[];byseq=[]
  for seq in SEQS:
   ev=[e for e in events if e['seq']==seq]
   a,s,changed=apply_swaps(Path(args.internal_root)/f'{seq}.txt',Path(root)/f'{seq}.txt',tdir/f'{seq}.txt',ev)
   applied+=a;skipped+=s;byseq.append({'seq':seq,'requested':len(ev),'applied':len(a),'skipped':len(s),'changed_rows':changed,'utility30':sum(float(x['swap_utility_30']) for x in a)})
  evres=evaluate(pdir,'swap_'+name);summary={'name':name,'policy':POLICY,'requested':len(events),'applied':len(applied),'skipped':len(skipped),'by_seq':byseq,'eval':evres}
  (pdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');pd.DataFrame(applied).to_csv(pdir/'applied_events.csv',index=False)
  if skipped:pd.DataFrame(skipped).to_csv(pdir/'skipped_events.csv',index=False)
  summaries.append(summary);print(json.dumps(summary,indent=2),flush=True)
 (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
if __name__=='__main__':main()
