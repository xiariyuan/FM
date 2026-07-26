from __future__ import annotations
import csv,hashlib,importlib.util,json,math,subprocess,sys
from collections import defaultdict
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import linear_sum_assignment
REPO=Path(__file__).resolve().parents[2]
SEQS=('MOT20-01','MOT20-02','MOT20-03','MOT20-05')
ROOT=REPO/'outputs/mot20_m28_20260726/m28_a5_m23_70a_m28_union_capacity'
BASE46=REPO/'outputs/mot20_m23_20260718/m23_46_pure_hgb_topk_combined_oof_v1/track_results'
BASE70=REPO/'outputs/mot20_m23_20260718/m23_70a_causal_branch_capacity_v1/capacity'
ACTIONS={
'MOT20-01':REPO/'outputs/mot20_m28_20260726/m28_a3_strong_host_m23_46_m01_v1/teacher_capacity/selected_actions.csv',
'MOT20-02':REPO/'outputs/mot20_m28_20260726/m28_a4_strong_host_multisequence/MOT20-02/teacher_capacity/selected_actions.csv',
'MOT20-03':REPO/'outputs/mot20_m28_20260726/m28_a4_strong_host_multisequence/MOT20-03/teacher_capacity/selected_actions.csv',
'MOT20-05':REPO/'outputs/mot20_m28_20260726/m28_a4_strong_host_multisequence/MOT20-05/teacher_capacity/selected_actions.csv'}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,REPO/p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def read(path):
 rows=[];by=defaultdict(list)
 for i,line in enumerate(path.open()):
  x=line.rstrip().split(',');f=int(float(x[0]));tid=int(float(x[1]));a,b,w,h=map(float,x[2:6]);r={'index':i,'fields':x,'frame':f,'id':tid,'box':np.array([a,b,a+w,b+h],np.float64),'score':float(x[6])};rows.append(r);by[f].append(i)
 return rows,by
def iou(a,b):
 x1=np.maximum(a[:,None,0],b[None,:,0]);y1=np.maximum(a[:,None,1],b[None,:,1]);x2=np.minimum(a[:,None,2],b[None,:,2]);y2=np.minimum(a[:,None,3],b[None,:,3]);inter=np.maximum(0,x2-x1)*np.maximum(0,y2-y1);aa=(a[:,2]-a[:,0])*(a[:,3]-a[:,1]);bb=(b[:,2]-b[:,0])*(b[:,3]-b[:,1]);return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)
def align(base,teacher):
 A,ba=read(base);B,bb=read(teacher);mapping=np.full(len(A),-1,np.int64);min_iou=1.;score_bad=0
 if set(ba)!=set(bb):raise RuntimeError('frame sets differ')
 for f in sorted(ba):
  ia,ib=ba[f],bb[f]
  if len(ia)!=len(ib):raise RuntimeError(f'frame count mismatch {f}')
  xa=np.stack([A[i]['box'] for i in ia]);xb=np.stack([B[i]['box'] for i in ib]);ov=iou(xa,xb);r,c=linear_sum_assignment(-ov);vals=ov[r,c];min_iou=min(min_iou,float(vals.min()))
  if np.any(vals<.999999):raise RuntimeError(f'imperfect payload alignment {f} {vals.min()}')
  for u,v in zip(r,c):
   mapping[ia[u]]=ib[v];score_bad+=abs(A[ia[u]]['score']-B[ib[v]]['score'])>1e-6
 if np.any(mapping<0) or score_bad:raise RuntimeError('incomplete alignment or score mismatch')
 return A,B,mapping,min_iou
def write(path,prepared,ids):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w') as f:
  for fields,tid in zip(prepared.parent_rows,ids):
   x=list(fields);x[1]=str(int(tid));f.write(','.join(x)+'\n')
def official(results,name,out):
 cmd=[sys.executable,'scripts/eval_motstyle_trackeval.py','--benchmark-name','MOT20','--split-to-eval','train','--gt-root','datasets/MOT20/train','--results-dir',str(results),'--tracker-name',name,'--work-dir',str(out),'--keep-workdir','--seqs',*SEQS];p=subprocess.run(cmd,cwd=REPO,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out.parent/f'{name}.log').write_text(p.stdout)
 if p.returncode:raise RuntimeError(p.stdout[-10000:])
 q=out/'eval'/name/'pedestrian_summary.txt';lines=[x.strip() for x in q.read_text().splitlines() if x.strip()];v=dict(zip(lines[0].split(),lines[1].split()));return {k:float(v[k]) for k in ('HOTA','DetA','AssA','IDF1')}|{'IDSW':int(float(v['IDSW'])),'Dets':int(float(v['Dets']))}
def main():
 ROOT.mkdir(parents=True,exist_ok=True);m37=load('m28a5_exact','scripts/m23_research/m23_37_fast_exact_hota_teacher.py');reports=[];baseline_dir=ROOT/'baseline_track_results';union_dir=ROOT/'union_track_results';baseline_dir.mkdir(exist_ok=True);union_dir.mkdir(exist_ok=True)
 freeze={'experiment_id':'M28-A5','teacher_only':True,'deployable':False,'mot20_test_reads':0,'test_submission':False,'mapping':'per-frame Hungarian exact bbox/score alignment from M23-46 to M23-70A','action_order':'existing M28 selected_actions.csv order; accept only positive exact-HOTA step on M23-70A','inputs':{}}
 for seq in SEQS:
  p46=BASE46/f'{seq}.txt';p70=BASE70/seq/'track_results'/f'{seq}.txt';act=ACTIONS[seq];freeze['inputs'][seq]={'m23_46_sha256':sha(p46),'m23_70a_sha256':sha(p70),'m28_actions_sha256':sha(act)}
  A,B,map_idx,min_iou=align(p46,p70);actions=pd.read_csv(act);prepared=m37.PreparedExactHOTA(seq,p70,ROOT/seq/'exact_cache');current=prepared.parent_row_ids.copy();base_metrics=prepared.evaluate_row_ids_incremental(current);base_ids=np.asarray([r['id'] for r in A],np.int64);base_frames=np.asarray([r['frame'] for r in A],np.int32);teacher_frames=np.asarray([int(float(x[0])) for x in prepared.parent_rows],np.int32);accepted=[];invalid=0;nonpositive=0
  for order,a in enumerate(actions.itertuples(index=False)):
   y,o=int(a.young_track_id),int(a.old_track_id);yb=np.flatnonzero(base_ids==y);ob=np.flatnonzero(base_ids==o)
   if not len(yb) or not len(ob):invalid+=1;continue
   # latest old row in M23-46 timeline defines the current target identity in M23-70A.
   ob_last=ob[np.lexsort((ob,base_frames[ob]))[-1]];target_idx=int(map_idx[ob_last]);target_id=int(current[target_idx]);yt=map_idx[yb];proposal=current.copy();changed=yt[proposal[yt]!=target_id]
   if not len(changed):nonpositive+=1;continue
   proposal[yt]=target_id;valid=True
   for f in np.unique(teacher_frames[yt]):
    z=proposal[teacher_frames==f]
    if len(z)!=len(np.unique(z)):valid=False;break
   if not valid:invalid+=1;continue
   metrics=prepared.evaluate_row_ids_incremental(proposal);gain=float(metrics['HOTA']-base_metrics['HOTA']) if not accepted else float(metrics['HOTA']-accepted[-1]['HOTA_after'])
   current_metrics=prepared.evaluate_row_ids_incremental(current) if not accepted else None
   # recompute step against actual current state, not original baseline.
   before=float(current_metrics['HOTA']) if current_metrics is not None else float(accepted[-1]['HOTA_after'])
   gain=float(metrics['HOTA']-before)
   if gain<=0:nonpositive+=1;continue
   current=proposal;accepted.append({'order':order,'event_index':int(a.event_index),'young_track_id':y,'old_track_id':o,'target_m23_70a_id':target_id,'changed_rows':int(len(changed)),'source_step_delta_HOTA':float(a.step_delta_HOTA),'union_step_delta_HOTA':gain,'HOTA_after':float(metrics['HOTA'])})
  final_metrics=prepared.evaluate_row_ids_incremental(current);out=ROOT/seq;out.mkdir(parents=True,exist_ok=True);pd.DataFrame(accepted).to_csv(out/'accepted_union_actions.csv',index=False);path=union_dir/f'{seq}.txt';write(path,prepared,current);(baseline_dir/f'{seq}.txt').write_bytes(p70.read_bytes());rep={'seq':seq,'baseline_HOTA':float(base_metrics['HOTA']),'union_HOTA':float(final_metrics['HOTA']),'delta_HOTA':float(final_metrics['HOTA']-base_metrics['HOTA']),'input_actions':len(actions),'accepted_actions':len(accepted),'invalid_actions':invalid,'nonpositive_actions':nonpositive,'min_alignment_iou':min_iou,'tracker_sha256':sha(path)};reports.append(rep);(out/'report.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n');print(json.dumps(rep,sort_keys=True),flush=True)
 (ROOT/'freeze_manifest.json').write_text(json.dumps(freeze,indent=2,sort_keys=True)+'\n');base_off=official(baseline_dir,'m28_a5_m23_70a_baseline',ROOT/'baseline_official_eval');union_off=official(union_dir,'m28_a5_union_teacher',ROOT/'union_official_eval');delta={k:union_off[k]-base_off[k] for k in base_off};gate=bool(union_off['HOTA']>=80.30 and all(r['delta_HOTA']>=0 for r in reports) and union_off['IDSW']<=base_off['IDSW']);report={'experiment_id':'M28-A5','status':'completed','decision':'PASS_M28_A5_AUTHORIZE_JOINT_STUDENT' if gate else 'FAIL_M28_A5_CLOSE_UNION','teacher_only':True,'deployable':False,'mot20_test_reads':0,'test_submission':False,'baseline_official':base_off,'union_official':union_off,'delta':delta,'folds':reports,'gate':{'minimum_combined_HOTA':80.30,'every_sequence_nonnegative':all(r['delta_HOTA']>=0 for r in reports),'combined_IDSW_nonincrease':union_off['IDSW']<=base_off['IDSW'],'pass':gate}};(ROOT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');pd.DataFrame(reports+[{'seq':'COMBINED','baseline_HOTA':base_off['HOTA'],'union_HOTA':union_off['HOTA'],'delta_HOTA':delta['HOTA'],'input_actions':sum(r['input_actions'] for r in reports),'accepted_actions':sum(r['accepted_actions'] for r in reports),'invalid_actions':sum(r['invalid_actions'] for r in reports),'nonpositive_actions':sum(r['nonpositive_actions'] for r in reports)}]).to_csv(ROOT/'summary.csv',index=False);print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__':main()
