from __future__ import annotations
import argparse, collections, hashlib, importlib.util, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[2]
BASE_ROOT=REPO/'outputs/mot20_m23_20260718/m23_46_pure_hgb_topk_combined_oof_v1/track_results'
DUMP_ROOT=REPO/'outputs/alink_train_inputs/phase0_root'
DEFAULT_ROOT=REPO/'outputs/mot20_m28_20260726/m28_b1_fast_validation'
TOPK=8;MAX_GAP=120;MIN_OLD_ROWS=4;MIN_OLD_REID=2;MIN_YOUNG_REID=2

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
B0=load('m28b1fast_b0',REPO/'scripts/m28_research/m28_b0_m23_46_deferred_identity.py')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def unit(x):
 z=np.asarray(x,np.float32);n=float(np.linalg.norm(z));return None if n<=1e-12 or not np.isfinite(z).all() else z/n

def slope(q,key):
 if len(q)<2:return 0.0
 x=np.asarray([r['frame'] for r in q],np.float64);y=np.asarray([r[key] for r in q],np.float64)
 if x[-1]==x[0]:return 0.0
 xm=x.mean();den=float(np.sum((x-xm)*(x-xm)))
 return 0.0 if den<=1e-12 else float(np.sum((x-xm)*y)/den)

class State:
 __slots__=('rows','last','last5','features','feature_sum','prototype','reid_count','vx','vy')
 def __init__(self):
  self.rows=0;self.last=None;self.last5=collections.deque(maxlen=5);self.features=collections.deque(maxlen=8);self.feature_sum=np.zeros(2048,np.float32);self.prototype=None;self.reid_count=0;self.vx=0.0;self.vy=0.0
 def add(self,row,gid,feature_memmap):
  self.rows+=1;self.last=row;self.last5.append(row);self.vx=slope(self.last5,'cx');self.vy=slope(self.last5,'cy')
  if gid>=0:
   z=unit(feature_memmap[int(gid)])
   if z is not None:
    if len(self.features)==self.features.maxlen:self.feature_sum-=self.features[0]
    self.features.append(z);self.feature_sum+=z;self.reid_count+=1
    self.prototype=unit(self.feature_sum/len(self.features))

def freeze(seq,out_root):
 out=Path(out_root)/seq/'frozen_candidates'
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True)
 baseline=BASE_ROOT/f'{seq}.txt';dump=DUMP_ROOT/seq/'dump_yolox_reid.npz'
 B0.SEQ=seq;B0.BASELINE=baseline;B0.DUMP=dump
 rows=B0.parse(baseline);phase,miou,features=B0.map_features(rows)
 by={}
 for i,r in enumerate(rows):by.setdefault(int(r['track_id']),[]).append(i)
 for tid in by:by[tid].sort(key=lambda i:(int(rows[i]['frame']),i))
 # Event identity/order exactly follows the original sorted track-id implementation.
 events=[];young_proto={}
 for tid,inds in sorted(by.items()):
  if len(inds)<2:continue
  birth=int(rows[inds[0]]['frame']);decision=int(rows[inds[1]]['frame'])
  gids=[int(phase[i]) for i in inds if int(rows[i]['frame'])<=decision and int(phase[i])>=0]
  p,count=B0.proto(gids,features,2,False)
  if p is None or count<MIN_YOUNG_REID:continue
  ei=len(events);events.append({'event_index':ei,'birth_frame':birth,'decision_frame':decision,'young_track_id':int(tid),'candidate_count':0,'young_rows':len(inds),'gt_opened':False})
  young_proto[ei]=(p,count,rows[inds[0]])
 # Sweep all tracker observations strictly before each birth; no future track-lifecycle read.
 global_rows=sorted(range(len(rows)),key=lambda i:(int(rows[i]['frame']),i));pointer=0;states={}
 candidates=[];audits=[]
 for e in sorted(events,key=lambda x:(int(x['birth_frame']),int(x['event_index']))):
  birth=int(e['birth_frame']);young=int(e['young_track_id'])
  while pointer<len(global_rows) and int(rows[global_rows[pointer]]['frame'])<birth:
   i=global_rows[pointer];r=rows[i];tid=int(r['track_id']);st=states.get(tid)
   if st is None:st=states[tid]=State()
   st.add(r,int(phase[i]),features);pointer+=1
  p,ycount,young_row=young_proto[int(e['event_index'])]
  old_ids=[];protos=[];gaps=[];motions=[];hrs=[];old_rows=[];old_reids=[];last_frames=[]
  for old,st in states.items():
   if old==young or st.rows<MIN_OLD_ROWS or st.reid_count<MIN_OLD_REID or st.prototype is None:continue
   gap=birth-int(st.last['frame'])
   if gap<1 or gap>MAX_GAP:continue
   dt=float(gap);px=float(st.last['cx'])+st.vx*dt;py=float(st.last['cy'])+st.vy*dt
   scale=max(.5*(float(st.last['h'])+float(young_row['h'])),1.0)
   motion=math.hypot(float(young_row['cx'])-px,float(young_row['cy'])-py)/scale
   hr=max(float(young_row['h']),1e-3)/max(float(st.last['h']),1e-3)
   old_ids.append(int(old));protos.append(st.prototype);gaps.append(gap);motions.append(motion);hrs.append(hr);old_rows.append(st.rows);old_reids.append(min(st.reid_count,8));last_frames.append(int(st.last['frame']))
  selected=[]
  if old_ids:
   P=np.stack(protos).astype(np.float32,copy=False);app=P@np.asarray(p,np.float32)
   gaps_a=np.asarray(gaps,np.float64);mot_a=np.asarray(motions,np.float64);hr_a=np.asarray(hrs,np.float64);ids_a=np.asarray(old_ids,np.int64)
   score=app.astype(np.float64)-.35*mot_a-.03*np.log1p(gaps_a)-.08*np.abs(np.log(hr_a))
   order=np.lexsort((ids_a,mot_a,-app.astype(np.float64),-score))[:TOPK]
   for rank,j in enumerate(order,1):
    selected.append({'event_index':int(e['event_index']),'birth_frame':birth,'decision_frame':int(e['decision_frame']),'young_track_id':young,'old_track_id':int(ids_a[j]),'gap':int(gaps_a[j]),'old_rows_past':int(old_rows[j]),'old_reid_count_past':int(old_reids[j]),'young_reid_count':int(ycount),'appearance_cos':float(app[j]),'motion_error':float(mot_a[j]),'height_ratio':float(hr_a[j]),'candidate_score':float(score[j]),'old_last_observed_frame':int(last_frames[j]),'young_max_feature_frame':int(e['decision_frame']),'old_max_feature_frame':int(last_frames[j]),'gt_opened':False,'candidate_rank':rank})
  e['candidate_count']=len(selected);candidates.extend(selected)
  audits.append({'event_index':int(e['event_index']),'birth_frame':birth,'decision_frame':int(e['decision_frame']),'max_old_observed_frame':max((x['old_last_observed_frame'] for x in selected),default=-1),'max_old_feature_frame':max((x['old_max_feature_frame'] for x in selected),default=-1),'max_young_feature_frame':int(e['decision_frame'])})
 C=pd.DataFrame(candidates).sort_values(['event_index','candidate_rank']).reset_index(drop=True);E=pd.DataFrame(events).sort_values('event_index').reset_index(drop=True);A=pd.DataFrame(audits).sort_values('event_index').reset_index(drop=True)
 if C.empty:raise RuntimeError('no candidates')
 if not ((C.old_last_observed_frame<C.birth_frame)&(C.old_max_feature_frame<C.birth_frame)&(C.young_max_feature_frame<=C.decision_frame)).all():raise RuntimeError('causality invariant failed')
 C.to_parquet(out/'candidates.parquet',index=False);E.to_parquet(out/'events.parquet',index=False);A.to_csv(out/'causality_audit.csv',index=False)
 manifest={'experiment_id':'M28-B1','stage':'m23_46_candidates_frozen','seq':seq,'host':'M23-46 strict sequence-LOSO tracker','gt_opened':False,'candidate_generation':'young first two observations; old rows/features strictly before young birth','future_row_reads':0,'events':len(E),'events_with_candidates':int((E.candidate_count>0).sum()),'candidate_actions':len(C),'topk':TOPK,'max_gap':MAX_GAP,'mapped_rows':int((phase>=0).sum()),'tracker_rows':len(rows),'mapping_rate':float((phase>=0).mean()),'median_mapping_iou':float(np.median(miou[phase>=0])),'baseline_sha256':sha(baseline),'dump_sha256':sha(dump),'candidates_sha256':sha(out/'candidates.parquet'),'events_sha256':sha(out/'events.parquet'),'script_sha256':sha(Path(__file__)),'mot20_test_reads':0,'implementation':'causal frame sweep with rolling exact last-8 ReID prototype; no full-sequence last-frame use'}
 (out/'freeze_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(json.dumps(manifest,indent=2,sort_keys=True))

def validate(seq,fast_root,reference_root):
 a=pd.read_parquet(Path(fast_root)/seq/'frozen_candidates/candidates.parquet').sort_values(['event_index','candidate_rank']).reset_index(drop=True)
 b=pd.read_parquet(Path(reference_root)/seq/'frozen_candidates/candidates.parquet').sort_values(['event_index','candidate_rank']).reset_index(drop=True)
 keys=['event_index','birth_frame','decision_frame','young_track_id','old_track_id','candidate_rank']
 same_keys=a[keys].equals(b[keys]);result={'seq':seq,'rows_fast':len(a),'rows_reference':len(b),'same_candidate_keys':bool(same_keys)}
 for c in ['appearance_cos','motion_error','height_ratio','candidate_score']:
  result[c+'_max_abs']=float(np.max(np.abs(a[c].to_numpy(float)-b[c].to_numpy(float)))) if len(a)==len(b) else None
 result['pass']=bool(same_keys and len(a)==len(b) and result['candidate_score_max_abs']<1e-5)
 print(json.dumps(result,indent=2,sort_keys=True))
 if not result['pass']:raise RuntimeError(result)

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','validate']);p.add_argument('--seq',required=True);p.add_argument('--out-root',default=str(DEFAULT_ROOT));p.add_argument('--reference-root',default='outputs/mot20_m28_20260726/m28_b1_m23_46_multisequence');a=p.parse_args()
 if a.stage=='freeze':freeze(a.seq,a.out_root)
 else:validate(a.seq,a.out_root,a.reference_root)
if __name__=='__main__':main()
