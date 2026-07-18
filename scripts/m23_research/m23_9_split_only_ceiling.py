from __future__ import annotations

# Research artifact for the MOT20 M23 sequence-LOSO audit.
import csv, importlib.util, json, subprocess, sys
from collections import defaultdict
from pathlib import Path

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
PARENT=Path('outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results')
OUT=Path('outputs/mot20_m23_20260718/internal_split_only_ceiling_v1')
SYN_BASE=14_000_000

def load_m23():
 s=importlib.util.spec_from_file_location('m23','scripts/audit_m23_mot20_expanded_evidence_oracle.py');m=importlib.util.module_from_spec(s);sys.modules['m23']=m;s.loader.exec_module(m);return m

def read_lines(path):
 rows=[]
 for i,line in enumerate(path.open()):
  p=line.rstrip('\n').split(',')
  if len(p)>=6:rows.append((i,p))
 return rows

def fill_zeros(states):
 # Fill only zero runs bounded by the same positive identity. Other zeros stay attached to preceding run.
 out=list(states);n=len(out);i=0
 while i<n:
  if out[i]!=0:i+=1;continue
  j=i
  while j<n and out[j]==0:j+=1
  left=out[i-1] if i>0 else 0;right=out[j] if j<n else 0
  if left>0 and left==right:
   for k in range(i,j):out[k]=left
  elif left>0:
   for k in range(i,j):out[k]=left
  elif right>0:
   for k in range(i,j):out[k]=right
  i=j
 return out

def main():
 m23=load_m23();td=OUT/'track_results';td.mkdir(parents=True,exist_ok=True);plan=[];seq_counts={}
 for si,seq in enumerate(SEQS):
  rows=read_lines(PARENT/f'{seq}.txt');baseline=m23.load_baseline(PARENT/f'{seq}.txt');gt=m23.load_gt(Path('datasets/MOT20/train')/seq/'gt/gt.txt')
  gt_by_line={i:0 for i,_ in rows}
  for f in sorted(baseline):
   kept,valid,_=m23.valid_and_distractor_filtered(baseline[f],gt.get(f,[]))
   for ci,gi,iou in m23.match_candidates(kept,valid):
    line=int(kept[ci].uid & ((1<<32)-1));gt_by_line[line]=int(valid[gi].gt_id)
  by_tid=defaultdict(list)
  for i,p in rows:by_tid[int(float(p[1]))].append((int(float(p[0])),i,p))
  new_by_line={};next_id=SYN_BASE+si*500_000;splits=0;runs=0;changed=0
  for tid,items in by_tid.items():
   items.sort();states=fill_zeros([gt_by_line[i] for _,i,_ in items]);start=0;run_index=0
   for k in range(1,len(items)+1):
    boundary=k==len(items) or (states[k]>0 and states[k-1]>0 and states[k]!=states[k-1])
    if not boundary:continue
    identity=tid if run_index==0 else next_id
    if run_index>0:next_id+=1;splits+=1
    for q in range(start,k):
     _,line,_=items[q];new_by_line[line]=identity;changed+=identity!=tid
    plan.append({'seq':seq,'source_track_id':tid,'run_index':run_index,'first_frame':items[start][0],'last_frame':items[k-1][0],'rows':k-start,'diagnostic_gt':states[start],'new_id':identity})
    runs+=1;run_index+=1;start=k
  out=[];seen=set()
  for i,p in rows:
   q=list(p);q[1]=str(new_by_line[i]);key=(int(float(q[0])),int(float(q[1])))
   if key in seen:raise RuntimeError(f'{seq}: duplicate {key}')
   seen.add(key);out.append(q)
  out.sort(key=lambda p:(int(float(p[0])),int(float(p[1])),float(p[2]),float(p[3])))
  with (td/f'{seq}.txt').open('w') as f:
   for p in out:f.write(','.join(p)+'\n')
  seq_counts[seq]={'source_tracks':len(by_tid),'runs':runs,'splits':splits,'changed_rows':changed,'rows':len(rows)}
 with (OUT/'split_plan.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(plan[0]));w.writeheader();w.writerows(plan)
 name='internal_split_only_ceiling';cmd=[sys.executable,'scripts/eval_motstyle_trackeval.py','--benchmark-name','MOT20','--split-to-eval','train','--gt-root','datasets/MOT20/train','--results-dir',str(td),'--tracker-name',name,'--work-dir',str(OUT/'eval_work'),'--seqs',*SEQS]
 p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(OUT/'eval.log').write_text(p.stdout)
 if p.returncode:raise RuntimeError(p.stdout[-4000:])
 rr=list(csv.DictReader((OUT/'eval_work/eval'/name/'pedestrian_detailed.csv').open()));res={}
 for s in SEQS+['COMBINED']:
  r=next(x for x in rr if x['seq']==s);res[s]={'HOTA':100*float(r['HOTA___AUC']),'DetA':100*float(r['DetA___AUC']),'AssA':100*float(r['AssA___AUC']),'IDSW':int(float(r['IDSW']))}
 report={'oracle':True,'deployment_allowed':False,'protocol':'split current track IDs only at persistent GT identity changes; no cross-track relinking; preserve all rows and boxes','counts':seq_counts,'eval':res};(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
