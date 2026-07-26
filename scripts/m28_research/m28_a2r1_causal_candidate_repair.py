from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[2]
SEQ='MOT20-01'
ROOT=REPO/'outputs/mot20_m28_20260726/m28_a2r1_causal_candidate_repair_m01'
BASELINE=REPO/'outputs/mot20_m27_20260726/m27_a0_exact_idsw_source_attribution_m01/frozen_runtime/baseline_online.txt'
UPDATES=REPO/'outputs/mot20_m27_20260726/m27_a0_exact_idsw_source_attribution_m01/frozen_runtime/association_updates.parquet'
DUMP=REPO/'outputs/alink_train_inputs/phase0_root/MOT20-01/dump_yolox_reid.npz'
TOPK=8;MAX_GAP=120;MIN_ROWS=4;MIN_REID=2

def load(name:str,path:Path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
M28=load('m28a2r1_base',REPO/'scripts/m28_research/m28_a0_deferred_identity_inheritance.py')

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write_json(path:Path,payload:Any):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')

def freeze_candidates():
 out=ROOT/'frozen_candidates'
 if out.exists(): raise FileExistsError(out)
 out.mkdir(parents=True)
 rows=M28.parse_tracker(BASELINE);updates=pd.read_parquet(UPDATES)
 events=updates[(updates.method.astype(str)=='update')&(updates.stage.astype(str)=='unconfirmed')].copy()
 events.sort_values(['frame','track_id'],inplace=True);events.drop_duplicates(['frame','track_id'],inplace=True);events.reset_index(drop=True,inplace=True);events.insert(0,'event_index',np.arange(len(events),dtype=int))
 dump=np.load(DUMP,allow_pickle=True);features=np.asarray(dump['features'])
 by_rows=defaultdict(list);by_updates=defaultdict(list)
 for r in rows:by_rows[int(r['track_id'])].append(r)
 for tid in by_rows:by_rows[tid].sort(key=lambda r:(int(r['frame']),int(r['row_index'])))
 for u in updates.itertuples(index=False):
  gid=int(u.det_global_idx)
  if gid>=0:by_updates[int(u.track_id)].append((int(u.frame),gid))
 for tid in by_updates:by_updates[tid].sort()
 candidates=[];event_rows=[];audit=[]
 for e in events.itertuples(index=False):
  ei=int(e.event_index);frame=int(e.frame);young=int(e.track_id)
  young_hist=[gid for f,gid in by_updates.get(young,[]) if f<=frame]
  young_proto,young_count=M28.prototype(young_hist,features,2,False)
  if young_proto is None:continue
  young_past_rows=[r for r in by_rows.get(young,[]) if int(r['frame'])<=frame]
  if not young_past_rows:continue
  event_row=young_past_rows[-1]
  records=[]
  for old,all_rows in by_rows.items():
   if old==young:continue
   past=[r for r in all_rows if int(r['frame'])<frame]
   if len(past)<MIN_ROWS:continue
   last=past[-1];gap=frame-int(last['frame'])
   if gap<1 or gap>MAX_GAP:continue
   old_hist=[gid for f,gid in by_updates.get(old,[]) if f<frame]
   old_proto,old_count=M28.prototype(old_hist,features,8,True)
   if old_proto is None or old_count<MIN_REID:continue
   vx,vy=M28.velocity(past)
   appearance=float(old_proto@young_proto);dt=float(gap)
   px=float(last['cx'])+vx*dt;py=float(last['cy'])+vy*dt
   scale=max(.5*(float(last['h'])+float(event_row['h'])),1.0)
   motion=math.hypot(float(event_row['cx'])-px,float(event_row['cy'])-py)/scale
   hr=max(float(event_row['h']),1e-3)/max(float(last['h']),1e-3)
   score=appearance-.35*motion-.03*math.log1p(gap)-.08*abs(math.log(hr))
   records.append({'event_index':ei,'frame':frame,'young_track_id':young,'old_track_id':int(old),'gap':gap,'old_rows_past':len(past),'old_reid_count_past':int(old_count),'young_reid_count':int(young_count),'appearance_cos':appearance,'motion_error':motion,'height_ratio':hr,'candidate_score':score,'old_last_observed_frame':int(last['frame']),'old_max_feature_frame':max((f for f,_ in by_updates.get(old,[]) if f<frame),default=-1),'young_max_feature_frame':max((f for f,_ in by_updates.get(young,[]) if f<=frame),default=-1),'gt_opened':False})
  records.sort(key=lambda x:(-x['candidate_score'],-x['appearance_cos'],x['motion_error'],x['old_track_id']))
  selected=records[:TOPK]
  for rank,r in enumerate(selected,1):r['candidate_rank']=rank;candidates.append(r)
  event_rows.append({'event_index':ei,'frame':frame,'young_track_id':young,'candidate_count':len(selected),'gt_opened':False})
  audit.append({'event_index':ei,'frame':frame,'max_old_observed_frame':max((r['old_last_observed_frame'] for r in selected),default=-1),'max_old_feature_frame':max((r['old_max_feature_frame'] for r in selected),default=-1),'max_young_feature_frame':max((r['young_max_feature_frame'] for r in selected),default=-1)})
 C=pd.DataFrame(candidates);E=pd.DataFrame(event_rows);A=pd.DataFrame(audit)
 if C.empty:raise RuntimeError('no causal candidates')
 if not ((C.old_last_observed_frame<C.frame)&(C.old_max_feature_frame<C.frame)&(C.young_max_feature_frame<=C.frame)).all():raise RuntimeError('future feature/read invariant failed')
 C.to_parquet(out/'candidates.parquet',index=False);E.to_parquet(out/'events.parquet',index=False);A.to_csv(out/'causality_audit.csv',index=False)
 manifest={'experiment_id':'M28-A2-R1','stage':'causal_candidates_frozen','seq':SEQ,'gt_opened':False,'candidate_generation':'only rows/features with frame <= decision frame; old identity strictly < decision frame','future_row_reads':0,'events':len(E),'candidate_actions':len(C),'topk':TOPK,'max_gap':MAX_GAP,'baseline_sha256':sha(BASELINE),'updates_sha256':sha(UPDATES),'dump_sha256':sha(DUMP),'candidates_sha256':sha(out/'candidates.parquet'),'events_sha256':sha(out/'events.parquet'),'script_sha256':sha(Path(__file__)),'mot20_test_reads':0}
 write_json(out/'freeze_manifest.json',manifest);print(json.dumps(manifest,indent=2,sort_keys=True));print(C.groupby('event_index').agg(n=('old_track_id','size'),best=('candidate_score','max'),min_gap=('gap','min')).head(80).to_string())

def label_capacity():
 frozen=ROOT/'frozen_candidates';out=ROOT/'teacher_capacity'
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);C=pd.read_parquet(frozen/'candidates.parquet')
 m37=load('m28a2r1_exact',REPO/'scripts/m23_research/m23_37_fast_exact_hota_teacher.py');prepared=m37.PreparedExactHOTA(SEQ,BASELINE,out/'exact_cache')
 base_ids=prepared.parent_row_ids.copy();base=prepared.evaluate_row_ids_incremental(base_ids);frames=np.asarray([int(float(x[0])) for x in prepared.parent_rows],np.int32)
 labels=[]
 for k,c in enumerate(C.itertuples(index=False),1):
  ids=base_ids.copy();mask=(ids==int(c.young_track_id))&(frames>=int(c.frame));ids[mask]=int(c.old_track_id)
  duplicate=any(len(ids[frames==f])!=len(np.unique(ids[frames==f])) for f in np.unique(frames[mask])) if mask.any() else True
  if duplicate:
   labels.append({**c._asdict(),'status':'invalid_future_identity_conflict','modified_rows':int(mask.sum()),'delta_HOTA':math.nan});continue
  met=prepared.evaluate_row_ids_incremental(ids);labels.append({**c._asdict(),'status':'success','modified_rows':int(mask.sum()),'HOTA':float(met['HOTA']),'DetA':float(met['DetA']),'AssA':float(met['AssA']),'delta_HOTA':float(met['HOTA']-base['HOTA']),'delta_DetA':float(met['DetA']-base['DetA']),'delta_AssA':float(met['AssA']-base['AssA'])})
  if k%100==0:print('labeled',k,'/',len(C),flush=True)
 L=pd.DataFrame(labels);L.to_parquet(out/'exact_labels.parquet',index=False);S=L[L.status=='success'].copy().sort_values(['delta_HOTA','candidate_score'],ascending=[False,False])
 current=base_ids.copy();cur=dict(base);used_y=set();used_o=set();selected=[]
 for c in S[S.delta_HOTA>0].itertuples(index=False):
  y,o=int(c.young_track_id),int(c.old_track_id)
  if y in used_y or o in used_o:continue
  prop=current.copy();mask=(prop==y)&(frames>=int(c.frame));prop[mask]=o
  if not mask.any() or any(len(prop[frames==f])!=len(np.unique(prop[frames==f])) for f in np.unique(frames[mask])):continue
  met=prepared.evaluate_row_ids_incremental(prop);gain=float(met['HOTA']-cur['HOTA'])
  if gain<=0:continue
  current=prop;cur=met;used_y.add(y);used_o.add(o);selected.append({'event_index':int(c.event_index),'frame':int(c.frame),'young_track_id':y,'old_track_id':o,'candidate_rank':int(c.candidate_rank),'individual_delta_HOTA':float(c.delta_HOTA),'step_delta_HOTA':gain,'modified_rows':int(mask.sum())})
 Sel=pd.DataFrame(selected);Sel.to_csv(out/'selected_actions.csv',index=False);tracker=out/'track_results'/f'{SEQ}.txt';M28.write_tracker(tracker,prepared,current);official=M28.official_eval(tracker.parent,'m28_a2r1_causal_teacher',out/'official_eval')
 positive=int((S.delta_HOTA>0).sum());best=float(S.delta_HOTA.max()) if len(S) else 0.;delta=float(cur['HOTA']-base['HOTA']);orig=json.loads((REPO/'outputs/mot20_m28_20260726/m28_a0_deferred_identity_inheritance_m01_v1/report.json').read_text())
 gate=bool(positive>=8 and best>=.10 and delta>=.50 and official['IDSW']<=49)
 report={'experiment_id':'M28-A2-R1','status':'completed','decision':'PASS_CAUSAL_REPAIR_EXTEND_ALL4' if gate else 'FAIL_CAUSAL_REPAIR_CLOSE_M28','teacher_only':True,'deployable':False,'gt_opened_after_candidate_freeze':True,'candidate_actions':len(C),'successful_actions':len(S),'invalid_future_identity_conflicts':int((L.status!='success').sum()),'positive_actions':positive,'selected_actions':len(Sel),'baseline_metrics':{k:float(base[k]) for k in ['HOTA','DetA','AssA']},'teacher_metrics':{k:float(cur[k]) for k in ['HOTA','DetA','AssA']},'delta_HOTA':delta,'official_trackeval':official,'original_noncausal_delta_HOTA':float(orig['combined_delta_HOTA']),'capacity_retention':delta/max(float(orig['combined_delta_HOTA']),1e-12),'gate':{'minimum_positive_actions':8,'minimum_best_single_delta_HOTA':.10,'minimum_delta_HOTA':.50,'IDSW_nonincrease':True,'pass':gate},'future_row_reads_in_candidate_generation':0,'mot20_test_reads':0,'test_submission':False,'labels_sha256':sha(out/'exact_labels.parquet'),'tracker_sha256':sha(tracker)}
 write_json(ROOT/'report.json',report);pd.DataFrame([report|{'official_trackeval':json.dumps(official),'gate':json.dumps(report['gate'])}]).to_csv(ROOT/'summary.csv',index=False);print(json.dumps(report,indent=2,sort_keys=True));print('\nTOP\n',S.head(30).to_string(index=False));print('\nSELECTED\n',Sel.to_string(index=False))

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze-candidates','label-capacity']);a=p.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
 freeze_candidates() if a.stage=='freeze-candidates' else label_capacity()
if __name__=='__main__':main()
