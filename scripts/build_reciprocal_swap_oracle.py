from __future__ import annotations
import argparse,csv,json,subprocess,sys
from collections import defaultdict
from pathlib import Path

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']

def load_switches(path):
 by=defaultdict(list)
 with open(path,newline='') as f:
  for r in csv.DictReader(f):
   by[r['seq']].append({
    'frame':int(r['frame']),'track_id':int(r['track_id']),
    'prev_gt_id':int(r['prev_gt_id']),'new_gt_id':int(r['new_gt_id']),
    'gap_since_prev_match':int(r['gap_since_prev_match']),
    'overlap_max_ioa_1':float(r.get('overlap_max_ioa_1') or 0),
    'overlap_max_ioa_3':float(r.get('overlap_max_ioa_3') or 0),
   })
 return by

def mine(events,max_frame_delta=1):
 ev=sorted(events,key=lambda x:(x['frame'],x['track_id']))
 out=[];seen=set()
 for i,a in enumerate(ev):
  for b in ev[i+1:]:
   if b['frame']-a['frame']>max_frame_delta:break
   if a['track_id']==b['track_id']:continue
   if a['prev_gt_id']==b['new_gt_id'] and a['new_gt_id']==b['prev_gt_id'] and a['prev_gt_id']!=a['new_gt_id']:
    frame=max(a['frame'],b['frame']);pair=tuple(sorted((a['track_id'],b['track_id'])))
    key=(frame,pair)
    if key in seen:continue
    seen.add(key)
    out.append({
     'frame':frame,'track_a':pair[0],'track_b':pair[1],
     'event_frame_a':a['frame'],'event_frame_b':b['frame'],
     'gt_before_a':a['prev_gt_id'] if a['track_id']==pair[0] else b['prev_gt_id'],
     'gt_after_a':a['new_gt_id'] if a['track_id']==pair[0] else b['new_gt_id'],
     'gt_before_b':b['prev_gt_id'] if b['track_id']==pair[1] else a['prev_gt_id'],
     'gt_after_b':b['new_gt_id'] if b['track_id']==pair[1] else a['new_gt_id'],
     'overlap_max_ioa_1':max(a['overlap_max_ioa_1'],b['overlap_max_ioa_1']),
     'overlap_max_ioa_3':max(a['overlap_max_ioa_3'],b['overlap_max_ioa_3']),
     'gap_max':max(a['gap_since_prev_match'],b['gap_since_prev_match']),
    })
 # Collapse duplicate oscillation observations for same pair within 2 frames, keep first.
 ded=[];last={}
 for e in sorted(out,key=lambda x:(x['frame'],x['track_a'],x['track_b'])):
  p=(e['track_a'],e['track_b'])
  if p in last and e['frame']-last[p]<=2:continue
  ded.append(e);last[p]=e['frame']
 return ded

def load_rows(path):
 rows=[]
 with open(path) as f:
  for line in f:
   p=line.strip().split(',')
   if len(p)>=6:rows.append(p)
 return rows

def apply(rows,events):
 byframe=defaultdict(list)
 for e in events:byframe[e['frame']].append(e)
 mapping={};out=[];changed=0;applied=[]
 rows_by=defaultdict(list)
 for r in rows:rows_by[int(float(r[0]))].append(r)
 for fr in sorted(rows_by):
  for e in sorted(byframe.get(fr,[]),key=lambda x:(x['track_a'],x['track_b'])):
   a=e['track_a'];b=e['track_b'];mapping.setdefault(a,a);mapping.setdefault(b,b)
   mapping[a],mapping[b]=mapping[b],mapping[a];applied.append(e)
  emitted=[]
  for src in rows_by[fr]:
   r=list(src);tid=int(float(r[1]));mapping.setdefault(tid,tid);new=mapping[tid];changed+=int(new!=tid);r[1]=str(new);emitted.append(r)
  ids=[int(float(r[1])) for r in emitted]
  if len(ids)!=len(set(ids)):raise RuntimeError(f'duplicate ids frame {fr}')
  out.extend(emitted)
 return out,changed,applied

def write(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with open(path,'w') as f:
  for r in rows:f.write(','.join(r)+'\n')
def evaluate(out,name):
 cmd=[sys.executable,'scripts/eval_motstyle_trackeval.py','--benchmark-name','MOT20','--split-to-eval','train','--gt-root','datasets/MOT20/train','--results-dir',str(out/'track_results'),'--tracker-name',name,'--work-dir',str(out/'eval_work'),'--seqs',*SEQS]
 p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'eval.log').write_text(p.stdout)
 detail=out/'eval_work/eval'/name/'pedestrian_detailed.csv';res={'returncode':p.returncode}
 if detail.exists():
  rows=list(csv.DictReader(open(detail)))
  for seq in SEQS+['COMBINED']:
   r=next((x for x in rows if x['seq']==seq),None)
   if r:res[seq]={'HOTA':float(r['HOTA___AUC'])*100,'DetA':float(r['DetA___AUC'])*100,'AssA':float(r['AssA___AUC'])*100,'IDF1':float(r['IDF1'])*100 if float(r['IDF1'])<2 else float(r['IDF1']),'IDSW':int(float(r['IDSW']))}
  res['simple_avg_HOTA']=sum(res[s]['HOTA'] for s in SEQS)/4
 return res
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--switches',required=True);ap.add_argument('--track-root',required=True);ap.add_argument('--out-dir',required=True);ap.add_argument('--max-frame-delta',type=int,default=1);args=ap.parse_args()
 out=Path(args.out_dir);tdir=out/'track_results';tdir.mkdir(parents=True,exist_ok=True);by=load_switches(args.switches);all_events=[];reports=[]
 for seq in SEQS:
  ev=mine(by[seq],args.max_frame_delta)
  rows=load_rows(Path(args.track_root)/f'{seq}.txt');repaired,changed,applied=apply(rows,ev);write(tdir/f'{seq}.txt',repaired)
  for e in ev:e['seq']=seq
  all_events+=ev
  reports.append({'seq':seq,'switch_rows':len(by[seq]),'reciprocal_events':len(ev),'changed_rows':changed,'mean_ioa1':sum(e['overlap_max_ioa_1'] for e in ev)/max(1,len(ev))})
 fields=[]
 for r in all_events:
  for k in r:
   if k not in fields:fields.append(k)
 with open(out/'reciprocal_swap_events.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(all_events)
 ev=evaluate(out,'reciprocal_swap_oracle');summary={'max_frame_delta':args.max_frame_delta,'reports':reports,'events':len(all_events),'eval':ev}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
