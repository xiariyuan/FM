from __future__ import annotations

# Research artifact for the MOT20 M23 sequence-LOSO audit.
import argparse,csv,json,shutil,subprocess,sys
from collections import defaultdict
from pathlib import Path
import pandas as pd

PARENT=Path('outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results')
LINKS=Path('outputs/mot20_m23_20260718/global_match_explore_v1/global_match_t80/selected_links.csv')
BASE={'MOT20-01':{'HOTA':78.81905,'AssA':76.06038,'IDSW':41},'MOT20-02':{'HOTA':71.57491,'AssA':63.63815,'IDSW':286},'MOT20-03':{'HOTA':80.571616,'AssA':79.99768,'IDSW':139},'MOT20-05':{'HOTA':79.49144,'AssA':77.14945,'IDSW':434}}

def read_rows(path):
 rows=[];frames=defaultdict(list)
 for line in path.open():
  p=line.rstrip('\n').split(',');
  if len(p)<6:continue
  rows.append(p);frames[int(float(p[0]))].append(int(float(p[1])))
 return rows

def write_one(rows,a,b,path):
 out=[];frameids=defaultdict(list);changed=0
 for p in rows:
  q=list(p);tid=int(float(q[1]));new=a if tid==b else tid;q[1]=str(new);changed+=new!=tid
  frameids[int(float(q[0]))].append(new);out.append(q)
 dup=[f for f,x in frameids.items() if len(x)!=len(set(x))]
 if dup: raise RuntimeError(f'duplicate IDs {dup[:5]}')
 out.sort(key=lambda p:(int(float(p[0])),int(float(p[1])),float(p[2]),float(p[3])))
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w') as f:
  for p in out:f.write(','.join(p)+'\n')
 return changed

def eval_one(seq,track,work,name):
 cmd=[sys.executable,'scripts/eval_motstyle_trackeval.py','--benchmark-name','MOT20','--split-to-eval','train','--gt-root','datasets/MOT20/train','--results-dir',str(track.parent),'--tracker-name',name,'--work-dir',str(work),'--seqs',seq]
 p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(p.stdout[-3000:])
 detail=work/'eval'/name/'pedestrian_detailed.csv';r=next(csv.DictReader(detail.open()))
 return {'HOTA':100*float(r['HOTA___AUC']),'AssA':100*float(r['AssA___AUC']),'IDSW':int(float(r['IDSW']))}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seq',required=True);ap.add_argument('--out-dir',required=True);ap.add_argument('--start',type=int,default=0);ap.add_argument('--end',type=int,default=None);args=ap.parse_args();seq=args.seq;root=Path(args.out_dir);root.mkdir(parents=True,exist_ok=True)
 links=pd.read_csv(LINKS);links=links[links.seq.eq(seq)].reset_index(drop=True);rows=read_rows(PARENT/f'{seq}.txt');results=[]
 end=len(links) if args.end is None else min(args.end,len(links))
 for i in range(args.start,end):
  r=links.iloc[i]
  case=root/f'case_{i:04d}';td=case/'track_results';track=td/f'{seq}.txt';changed=write_one(rows,int(r.track_a),int(r.track_b),track)
  ev=eval_one(seq,track,case/'eval_work',f'edge_{seq}_{i:04d}')
  rec={**r.to_dict(),'case_index':i,'changed_rows':changed,**ev,'delta_HOTA':ev['HOTA']-BASE[seq]['HOTA'],'delta_AssA':ev['AssA']-BASE[seq]['AssA'],'delta_IDSW':ev['IDSW']-BASE[seq]['IDSW']}
  results.append(rec);print(json.dumps({'seq':seq,'i':i,'edge':[int(r.track_a),int(r.track_b)],'delta_HOTA':rec['delta_HOTA'],'delta_IDSW':rec['delta_IDSW']}),flush=True)
  shutil.rmtree(case)
 pd.DataFrame(results).to_csv(root/f'edge_utility_{args.start:04d}_{end:04d}.csv',index=False)
 (root/f'summary_{args.start:04d}_{end:04d}.json').write_text(json.dumps({'seq':seq,'start':args.start,'end':end,'rows':len(results),'positive':sum(x['delta_HOTA']>0 for x in results),'utility_sum':sum(x['delta_HOTA'] for x in results),'max':max((x['delta_HOTA'] for x in results),default=0),'min':min((x['delta_HOTA'] for x in results),default=0)},indent=2)+'\n')
if __name__=='__main__':main()
