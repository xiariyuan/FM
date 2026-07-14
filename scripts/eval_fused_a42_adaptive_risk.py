from __future__ import annotations
import argparse,csv,json,subprocess,sys
from collections import defaultdict
from pathlib import Path

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']

def read_csv(path):
 with open(path,newline='') as f:return list(csv.DictReader(f))
def read_tracks(path):
 rows=[];spans={}
 with open(path) as f:
  for line in f:
   p=line.strip().split(',')
   if len(p)<6:continue
   fr=int(float(p[0]));tid=int(float(p[1]));rows.append(p)
   spans.setdefault(tid,[fr,fr]);spans[tid]=[min(spans[tid][0],fr),max(spans[tid][1],fr)]
 return rows,spans
def apply(src,out,edges):
 rows,spans=read_tracks(src);parent={};used_s=set();used_t=set();sel=[]
 def find(x):
  parent.setdefault(x,x)
  while parent[x]!=x:
   parent[x]=parent[parent[x]];x=parent[x]
  return x
 for e in sorted(edges,key=lambda r:(-float(r['fusion_score']),int(float(r['gap'])))):
  a=int(float(e['track_a']));b=int(float(e['track_b']))
  if a not in spans or b not in spans or spans[a][1]>=spans[b][0]:continue
  if a in used_s or b in used_t:continue
  ra,rb=find(a),find(b)
  if ra==rb:continue
  parent[rb]=ra;used_s.add(a);used_t.add(b);sel.append(e)
 involved={int(float(r[k])) for r in sel for k in ['track_a','track_b']};idmap={t:find(t) for t in involved}
 frames=defaultdict(list);outrows=[];changed=0
 for p in rows:
  q=list(p);tid=int(float(q[1]));new=idmap.get(tid,tid);changed+=int(new!=tid);q[1]=str(new);frames[int(float(q[0]))].append(new);outrows.append(q)
 dup=[fr for fr,ids in frames.items() if len(ids)!=len(set(ids))]
 if dup:raise RuntimeError(f'duplicate IDs {src}: {dup[:5]}')
 out.parent.mkdir(parents=True,exist_ok=True);outrows.sort(key=lambda p:(int(float(p[0])),int(float(p[1]))))
 with open(out,'w') as f:
  for p in outrows:f.write(','.join(p)+'\n')
 return sel,{'links':len(sel),'changed_rows':changed,'a42':sum(r['origin']=='a42' for r in sel),'adaptive':sum(r['origin']=='adaptive' for r in sel)}
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
 ap=argparse.ArgumentParser();ap.add_argument('--a42',required=True);ap.add_argument('--adaptive-root',required=True);ap.add_argument('--source-dir',required=True);ap.add_argument('--out-dir',required=True);args=ap.parse_args()
 a42=read_csv(args.a42);out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True);summaries=[]
 for policy in ['l1_r1_max','l1_r2_q75']:
  adaptive=read_csv(Path(args.adaptive_root)/policy/'selected_links.csv')
  for priority in ['a42','adaptive']:
   name=f'{policy}_{priority}_priority';edges=[]
   for r in a42:
    x=dict(r);x['origin']='a42';base=float(r['a42_model_score']);x['fusion_score']=base+(2 if priority=='a42' else 0);edges.append(x)
   for r in adaptive:
    x=dict(r);x['origin']='adaptive';base=float(r['meta_risk_score_l1p0']);x['fusion_score']=base+(2 if priority=='adaptive' else 0);edges.append(x)
   pdir=out/name;tdir=pdir/'track_results';tdir.mkdir(parents=True,exist_ok=True);allsel=[];byseq=[]
   for seq in SEQS:
    sel,st=apply(Path(args.source_dir)/f'{seq}.txt',tdir/f'{seq}.txt',[e for e in edges if e['seq']==seq]);st['seq']=seq;byseq.append(st);allsel+=sel
   fields=[]
   for r in allsel:
    for k in r:
     if k not in fields:fields.append(k)
   with open(pdir/'selected_links.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(allsel)
   ev=evaluate(pdir,'fused_'+name);summary={'name':name,'selected':len(allsel),'a42':sum(r['origin']=='a42' for r in allsel),'adaptive':sum(r['origin']=='adaptive' for r in allsel),'by_seq':byseq,'eval':ev}
   (pdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');summaries.append(summary)
   print(json.dumps({'name':name,'selected':summary['selected'],'origins':[summary['a42'],summary['adaptive']],'combined':ev.get('COMBINED'),'m02':ev.get('MOT20-02'),'avg':ev.get('simple_avg_HOTA')},indent=2),flush=True)
 (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
if __name__=='__main__':main()
