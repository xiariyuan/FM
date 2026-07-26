from __future__ import annotations
import argparse,bisect,hashlib,importlib.util,json,math,sys
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from scipy.optimize import linear_sum_assignment

REPO=Path(__file__).resolve().parents[2]
SEQ='MOT20-01'
ROOT=REPO/'outputs/mot20_m28_20260726/m28_b0_m23_46_deferred_identity_m01'
BASELINE=REPO/'outputs/mot20_m23_20260718/m23_46_pure_hgb_topk_combined_oof_v1/track_results/MOT20-01.txt'
DUMP=REPO/'outputs/alink_train_inputs/phase0_root/MOT20-01/dump_yolox_reid.npz'
TOPK=8;MAX_GAP=120;MIN_OLD_ROWS=4;MIN_OLD_REID=2;MIN_YOUNG_REID=2;MAP_IOU=.5

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
M10=load('m28b0_m10',REPO/'scripts/m23_research/m23_10_build_micrograph.py')
M28=load('m28b0_m28',REPO/'scripts/m28_research/m28_a0_deferred_identity_inheritance.py')

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write_json(p:Path,x:Any):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

def parse(path:Path):
 rows=[]
 for li,line in enumerate(path.open()):
  a=line.rstrip().split(',')
  if len(a)<7:continue
  f=int(float(a[0]));tid=int(float(a[1]));x,y,w,h=map(float,a[2:6]);rows.append({'row_index':len(rows),'line_index':li,'frame':f,'track_id':tid,'x':x,'y':y,'w':w,'h':h,'x1':x,'y1':y,'x2':x+w,'y2':y+h,'cx':x+.5*w,'cy':y+.5*h,'fields':a})
 return rows

def unit(x):
 x=np.asarray(x,np.float32);n=float(np.linalg.norm(x));return None if n<=1e-12 or not np.isfinite(x).all() else x/n

def proto(gids,features,limit,from_end):
 q=gids[-limit:] if from_end else gids[:limit];v=[]
 for gid in q:
  if gid<0:continue
  z=unit(features[int(gid)])
  if z is not None:v.append(z)
 if not v:return None,0
 return unit(np.mean(np.stack(v),0)),len(v)

def velocity(rr):
 q=rr[-5:]
 if len(q)<2:return 0.,0.
 f=np.asarray([r['frame'] for r in q],float)
 if len(np.unique(f))<2:return 0.,0.
 return float(np.polyfit(f,np.asarray([r['cx'] for r in q]),1)[0]),float(np.polyfit(f,np.asarray([r['cy'] for r in q]),1)[0])

def map_features(rows):
 det=M10.member(DUMP,'detections.npy');cols=M10.member(DUMP,'columns.npy',True).tolist();ci={x:i for i,x in enumerate(cols)};offs=M10.member(DUMP,'frame_offsets.npy');features=M10.mmap_member(DUMP,'features.npy');phase=np.full(len(rows),-1,np.int64);miou=np.zeros(len(rows),np.float32);by={}
 for i,r in enumerate(rows):by.setdefault(int(r['frame']),[]).append(i)
 for f,inds0 in by.items():
  inds=np.asarray(inds0,np.int64);st=int(offs[f-1]);en=int(offs[f]) if f<len(offs) else len(det);d=det[st:en]
  if not len(d):continue
  tb=np.asarray([[rows[i]['x1'],rows[i]['y1'],rows[i]['x2'],rows[i]['y2']] for i in inds],np.float32);db=d[:,[ci['x1'],ci['y1'],ci['x2'],ci['y2']]].astype(np.float32);ov=M10.iou_matrix(tb,db);ri,cj=linear_sum_assignment(-ov)
  for a,b in zip(ri,cj):
   if ov[a,b]>=MAP_IOU and d[b,ci['has_reid']]>.5:phase[inds[a]]=st+b;miou[inds[a]]=ov[a,b]
 return phase,miou,features

def freeze_candidates():
 out=ROOT/'frozen_candidates'
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);rows=parse(BASELINE);phase,miou,features=map_features(rows)
 by={}
 for i,r in enumerate(rows):by.setdefault(int(r['track_id']),[]).append(i)
 for tid in by:by[tid].sort(key=lambda i:(int(rows[i]['frame']),i))
 meta={}
 for tid,inds in by.items():meta[tid]={'inds':inds,'frames':[int(rows[i]['frame']) for i in inds],'gids':[int(phase[i]) for i in inds]}
 events=[];candidates=[];audit=[]
 for tid,m in sorted(meta.items()):
  if len(m['inds'])<2:continue
  birth=int(m['frames'][0]);decision=int(m['frames'][1]);young_gids=[g for f,g in zip(m['frames'],m['gids']) if f<=decision and g>=0]
  young_proto,ycount=proto(young_gids,features,2,False)
  if young_proto is None or ycount<MIN_YOUNG_REID:continue
  young_row=rows[m['inds'][0]];records=[]
  for old,om in meta.items():
   if old==tid:continue
   ri=bisect.bisect_left(om['frames'],birth)
   if ri<MIN_OLD_ROWS:continue
   last_i=om['inds'][ri-1];last=rows[last_i];gap=birth-int(last['frame'])
   if gap<1 or gap>MAX_GAP:continue
   old_gids=[g for f,g in zip(om['frames'][:ri],om['gids'][:ri]) if f<birth and g>=0]
   old_proto,ocount=proto(old_gids,features,8,True)
   if old_proto is None or ocount<MIN_OLD_REID:continue
   rr=[rows[i] for i in om['inds'][:ri]];vx,vy=velocity(rr);appearance=float(old_proto@young_proto);dt=float(gap);px=float(last['cx'])+vx*dt;py=float(last['cy'])+vy*dt;scale=max(.5*(float(last['h'])+float(young_row['h'])),1.);motion=math.hypot(float(young_row['cx'])-px,float(young_row['cy'])-py)/scale;hr=max(float(young_row['h']),1e-3)/max(float(last['h']),1e-3);score=appearance-.35*motion-.03*math.log1p(gap)-.08*abs(math.log(hr))
   records.append({'event_index':len(events),'birth_frame':birth,'decision_frame':decision,'young_track_id':int(tid),'old_track_id':int(old),'gap':gap,'old_rows_past':ri,'old_reid_count_past':int(ocount),'young_reid_count':int(ycount),'appearance_cos':appearance,'motion_error':motion,'height_ratio':hr,'candidate_score':score,'old_last_observed_frame':int(last['frame']),'young_max_feature_frame':max((f for f,g in zip(m['frames'],m['gids']) if f<=decision and g>=0),default=-1),'old_max_feature_frame':max((f for f,g in zip(om['frames'][:ri],om['gids'][:ri]) if f<birth and g>=0),default=-1),'gt_opened':False})
  records.sort(key=lambda x:(-x['candidate_score'],-x['appearance_cos'],x['motion_error'],x['old_track_id']));sel=records[:TOPK]
  for rank,r in enumerate(sel,1):r['candidate_rank']=rank;candidates.append(r)
  events.append({'event_index':len(events),'birth_frame':birth,'decision_frame':decision,'young_track_id':int(tid),'candidate_count':len(sel),'young_rows':len(m['inds']),'gt_opened':False});audit.append({'event_index':len(events)-1,'birth_frame':birth,'decision_frame':decision,'max_old_observed_frame':max((r['old_last_observed_frame'] for r in sel),default=-1),'max_old_feature_frame':max((r['old_max_feature_frame'] for r in sel),default=-1),'max_young_feature_frame':max((r['young_max_feature_frame'] for r in sel),default=-1)})
 C=pd.DataFrame(candidates);E=pd.DataFrame(events);A=pd.DataFrame(audit)
 if C.empty:raise RuntimeError('no candidates')
 if not ((C.old_last_observed_frame<C.birth_frame)&(C.old_max_feature_frame<C.birth_frame)&(C.young_max_feature_frame<=C.decision_frame)).all():raise RuntimeError('causal invariant failed')
 C.to_parquet(out/'candidates.parquet',index=False);E.to_parquet(out/'events.parquet',index=False);A.to_csv(out/'causality_audit.csv',index=False)
 manifest={'experiment_id':'M28-B0','stage':'m23_46_candidates_frozen','seq':SEQ,'host':'M23-46 strict sequence-LOSO tracker','gt_opened':False,'candidate_generation':'young first two observations; old rows/features strictly before young birth','future_row_reads':0,'events':len(E),'events_with_candidates':int(C.event_index.nunique()),'candidate_actions':len(C),'topk':TOPK,'max_gap':MAX_GAP,'mapped_rows':int((phase>=0).sum()),'tracker_rows':len(rows),'mapping_rate':float((phase>=0).mean()),'median_mapping_iou':float(np.median(miou[phase>=0])),'baseline_sha256':sha(BASELINE),'dump_sha256':sha(DUMP),'candidates_sha256':sha(out/'candidates.parquet'),'events_sha256':sha(out/'events.parquet'),'script_sha256':sha(Path(__file__)),'mot20_test_reads':0};write_json(out/'freeze_manifest.json',manifest);print(json.dumps(manifest,indent=2,sort_keys=True));print(C.groupby('event_index').agg(n=('old_track_id','size'),best=('candidate_score','max'),min_gap=('gap','min')).to_string())

def dominant(matches):
 counts=matches.groupby(['tracker_id','gt_id']).size().reset_index(name='count');tot=counts.groupby('tracker_id')['count'].sum().to_dict();counts['purity']=[float(r.count)/max(int(tot[int(r.tracker_id)]),1) for r in counts.itertuples(index=False)];best=counts.sort_values(['tracker_id','count','gt_id'],ascending=[True,False,True]).groupby('tracker_id',as_index=False).first();return dict(zip(best.tracker_id.astype(int),best.gt_id.astype(int))),dict(zip(best.tracker_id.astype(int),best.purity.astype(float)))

def teacher():
 frozen=ROOT/'frozen_candidates';out=ROOT/'teacher_capacity'
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);C=pd.read_parquet(frozen/'candidates.parquet');m27=load('m28b0_m27',REPO/'scripts/m27_research/m27_a0_exact_idsw_source_attribution.py');m37=load('m28b0_m37',REPO/'scripts/m23_research/m23_37_fast_exact_hota_teacher.py');prepared=m37.PreparedExactHOTA(SEQ,BASELINE,out/'exact_cache');tracker_map,_=m27.original_tracker_mapping(prepared);switches,matches=m27.reconstruct_clear_idsw(prepared.data,tracker_map);dmap,purity=dominant(matches)
 C['young_gt']=C.young_track_id.map(dmap).fillna(-1).astype(int);C['old_gt']=C.old_track_id.map(dmap).fillna(-2).astype(int);C['young_purity']=C.young_track_id.map(purity).fillna(0.);C['old_purity']=C.old_track_id.map(purity).fillna(0.);C['same_dominant_gt']=((C.young_gt>0)&(C.young_gt==C.old_gt)).astype(int);screen=C[C.same_dominant_gt==1].copy();base_ids=prepared.parent_row_ids.copy();frames=np.asarray([int(float(x[0])) for x in prepared.parent_rows],np.int32);base=prepared.evaluate_row_ids_incremental(base_ids);labels=[]
 for c in screen.itertuples(index=False):
  ids=base_ids.copy();mask=(ids==int(c.young_track_id));ids[mask]=int(c.old_track_id);valid=bool(mask.any())
  if valid:
   for f in np.unique(frames[mask]):
    q=ids[frames==f]
    if len(q)!=len(np.unique(q)):valid=False;break
  if not valid:labels.append({**c._asdict(),'status':'invalid_future_identity_conflict','modified_rows':int(mask.sum()),'delta_HOTA':math.nan});continue
  met=prepared.evaluate_row_ids_incremental(ids);labels.append({**c._asdict(),'status':'success','modified_rows':int(mask.sum()),'HOTA':float(met['HOTA']),'DetA':float(met['DetA']),'AssA':float(met['AssA']),'delta_HOTA':float(met['HOTA']-base['HOTA']),'delta_DetA':float(met['DetA']-base['DetA']),'delta_AssA':float(met['AssA']-base['AssA'])})
 L=pd.DataFrame(labels);L.to_parquet(out/'identity_consistent_exact_labels.parquet',index=False);S=L[L.status=='success'].sort_values(['delta_HOTA','candidate_score'],ascending=[False,False]);cur_ids=base_ids.copy();cur=dict(base);used_y=set();used_o=set();selected=[]
 for c in S[S.delta_HOTA>0].itertuples(index=False):
  y,o=int(c.young_track_id),int(c.old_track_id)
  if y in used_y or o in used_o:continue
  prop=cur_ids.copy();mask=prop==y;prop[mask]=o
  if not mask.any():continue
  valid=True
  for f in np.unique(frames[mask]):
   q=prop[frames==f]
   if len(q)!=len(np.unique(q)):valid=False;break
  if not valid:continue
  met=prepared.evaluate_row_ids_incremental(prop);gain=float(met['HOTA']-cur['HOTA'])
  if gain<=0:continue
  cur_ids=prop;cur=met;used_y.add(y);used_o.add(o);selected.append({'event_index':int(c.event_index),'birth_frame':int(c.birth_frame),'decision_frame':int(c.decision_frame),'young_track_id':y,'old_track_id':o,'candidate_rank':int(c.candidate_rank),'individual_delta_HOTA':float(c.delta_HOTA),'step_delta_HOTA':gain,'modified_rows':int(mask.sum()),'young_purity':float(c.young_purity),'old_purity':float(c.old_purity)})
 Sel=pd.DataFrame(selected);Sel.to_csv(out/'selected_actions.csv',index=False);tracker=out/'track_results'/f'{SEQ}.txt';M28.write_tracker(tracker,prepared,cur_ids);M28.SEQ=SEQ;official=M28.official_eval(tracker.parent,'m28_b0_m23_46_teacher',out/'official_eval');delta=float(cur['HOTA']-base['HOTA']);positive=int((S.delta_HOTA>0).sum());best=float(S.delta_HOTA.max()) if len(S) else 0.;gate=bool(positive>=5 and delta>=.25 and best>=.05 and official['IDSW']<=len(switches))
 report={'experiment_id':'M28-B0','status':'completed','decision':'PASS_M28_B0_EXTEND_ALL4' if gate else 'FAIL_M28_B0_DO_NOT_STACK_ON_M23_46','teacher_only':True,'deployable':False,'strictly_causal_candidate_generation':True,'gt_opened_after_candidate_freeze':True,'host':'M23-46','candidate_actions':len(C),'identity_consistent_actions':len(screen),'valid_identity_consistent_actions':len(S),'invalid_future_identity_conflicts':int((L.status!='success').sum()),'positive_actions':positive,'selected_actions':len(Sel),'baseline_metrics':{k:float(base[k]) for k in ['HOTA','DetA','AssA']},'teacher_metrics':{k:float(cur[k]) for k in ['HOTA','DetA','AssA']},'delta_HOTA':delta,'best_single_delta_HOTA':best,'official_trackeval':official,'official_clear_idsw_baseline':len(switches),'gate':{'minimum_positive_actions':5,'minimum_delta_HOTA':.25,'minimum_best_single_delta_HOTA':.05,'IDSW_nonincrease':True,'pass':gate},'future_row_reads':0,'mot20_test_reads':0,'test_submission':False,'tracker_sha256':sha(tracker)};write_json(ROOT/'report.json',report);pd.DataFrame([{'baseline_HOTA':base['HOTA'],'teacher_HOTA':cur['HOTA'],'delta_HOTA':delta,'positive_actions':positive,'selected_actions':len(Sel),'official_IDSW':official['IDSW'],'gate_pass':int(gate)}]).to_csv(ROOT/'summary.csv',index=False);print(json.dumps(report,indent=2,sort_keys=True));print('\nTOP\n',S.head(30).to_string(index=False));print('\nSELECTED\n',Sel.to_string(index=False))

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze-candidates','teacher']);a=p.parse_args();ROOT.mkdir(parents=True,exist_ok=True);freeze_candidates() if a.stage=='freeze-candidates' else teacher()
if __name__=='__main__':main()
