from __future__ import annotations
import argparse,bisect,hashlib,importlib.util,json,math,sys
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd

REPO=Path(__file__).resolve().parents[2]
OLD_ROOT=REPO/'outputs/mot20_m28_20260726/m28_a2_multisequence_capacity'

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
M28=load('m28opt_base',REPO/'scripts/m28_research/m28_a0_deferred_identity_inheritance.py')
M10=load('m28opt_m10',REPO/'scripts/m23_research/m23_10_build_micrograph.py')
TOPK=8;MAX_GAP=120;MIN_ROWS=4;MIN_REID=2

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write_json(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

def freeze(seq,out_root):
 out=out_root/'frozen_candidates'
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True)
 baseline=OLD_ROOT/seq/'frozen_runtime/baseline_online.txt';updates_path=OLD_ROOT/seq/'frozen_runtime/association_updates.parquet';dump=REPO/'outputs/alink_train_inputs/phase0_root'/seq/'dump_yolox_reid.npz'
 rows=M28.parse_tracker(baseline);updates=pd.read_parquet(updates_path);events=updates[(updates.method.astype(str)=='update')&(updates.stage.astype(str)=='unconfirmed')].copy();events.sort_values(['frame','track_id'],inplace=True);events.drop_duplicates(['frame','track_id'],inplace=True);events.reset_index(drop=True,inplace=True);events.insert(0,'event_index',np.arange(len(events),dtype=int));features=M10.mmap_member(dump,'features.npy')
 by_rows={};by_updates={}
 for r in rows:by_rows.setdefault(int(r['track_id']),[]).append(r)
 for tid in by_rows:by_rows[tid].sort(key=lambda r:(int(r['frame']),int(r['row_index'])))
 for u in updates.itertuples(index=False):
  gid=int(u.det_global_idx)
  if gid>=0:by_updates.setdefault(int(u.track_id),[]).append((int(u.frame),gid))
 for tid in by_updates:by_updates[tid].sort()
 meta={}
 for tid,rr in by_rows.items():
  uu=by_updates.get(tid,[]);meta[tid]={'rows':rr,'row_frames':[int(x['frame']) for x in rr],'updates':uu,'update_frames':[int(x[0]) for x in uu],'gids':[int(x[1]) for x in uu]}
 candidates=[];event_rows=[];audit=[]
 for n,e in enumerate(events.itertuples(index=False),1):
  ei=int(e.event_index);frame=int(e.frame);young=int(e.track_id);ym=meta.get(young)
  if ym is None:continue
  yui=bisect.bisect_right(ym['update_frames'],frame);young_proto,young_count=M28.prototype(ym['gids'][:yui],features,2,False)
  yri=bisect.bisect_right(ym['row_frames'],frame)
  if young_proto is None or yri<=0:continue
  event_row=ym['rows'][yri-1];records=[]
  for old,om in meta.items():
   if old==young:continue
   ri=bisect.bisect_left(om['row_frames'],frame)
   if ri<MIN_ROWS:continue
   last=om['rows'][ri-1];gap=frame-int(last['frame'])
   if gap<1 or gap>MAX_GAP:continue
   ui=bisect.bisect_left(om['update_frames'],frame);old_proto,old_count=M28.prototype(om['gids'][:ui],features,8,True)
   if old_proto is None or old_count<MIN_REID:continue
   vx,vy=M28.velocity(om['rows'][:ri]);appearance=float(old_proto@young_proto);dt=float(gap);px=float(last['cx'])+vx*dt;py=float(last['cy'])+vy*dt;scale=max(.5*(float(last['h'])+float(event_row['h'])),1.0);motion=math.hypot(float(event_row['cx'])-px,float(event_row['cy'])-py)/scale;hr=max(float(event_row['h']),1e-3)/max(float(last['h']),1e-3);score=appearance-.35*motion-.03*math.log1p(gap)-.08*abs(math.log(hr))
   records.append({'event_index':ei,'frame':frame,'young_track_id':young,'old_track_id':int(old),'gap':gap,'old_rows_past':ri,'old_reid_count_past':int(old_count),'young_reid_count':int(young_count),'appearance_cos':appearance,'motion_error':motion,'height_ratio':hr,'candidate_score':score,'old_last_observed_frame':int(last['frame']),'old_max_feature_frame':int(om['update_frames'][ui-1]) if ui else -1,'young_max_feature_frame':int(ym['update_frames'][yui-1]) if yui else -1,'gt_opened':False})
  records.sort(key=lambda x:(-x['candidate_score'],-x['appearance_cos'],x['motion_error'],x['old_track_id']));sel=records[:TOPK]
  for rank,r in enumerate(sel,1):r['candidate_rank']=rank;candidates.append(r)
  event_rows.append({'event_index':ei,'frame':frame,'young_track_id':young,'candidate_count':len(sel),'gt_opened':False});audit.append({'event_index':ei,'frame':frame,'max_old_observed_frame':max((r['old_last_observed_frame'] for r in sel),default=-1),'max_old_feature_frame':max((r['old_max_feature_frame'] for r in sel),default=-1),'max_young_feature_frame':max((r['young_max_feature_frame'] for r in sel),default=-1)})
  if n%200==0:print(seq,'events',n,'/',len(events),flush=True)
 C=pd.DataFrame(candidates);E=pd.DataFrame(event_rows);A=pd.DataFrame(audit)
 if C.empty:raise RuntimeError('no candidates')
 if not ((C.old_last_observed_frame<C.frame)&(C.old_max_feature_frame<C.frame)&(C.young_max_feature_frame<=C.frame)).all():raise RuntimeError('causal invariant')
 C.to_parquet(out/'candidates.parquet',index=False);E.to_parquet(out/'events.parquet',index=False);A.to_csv(out/'causality_audit.csv',index=False);manifest={'experiment_id':'M28-A2-R1','implementation_repair':'optimized prefix bisection and mmap; scientific semantics unchanged','seq':seq,'stage':'causal_candidates_frozen','gt_opened':False,'future_row_reads':0,'events':len(E),'candidate_actions':len(C),'topk':TOPK,'max_gap':MAX_GAP,'baseline_sha256':sha(baseline),'updates_sha256':sha(updates_path),'dump_sha256':sha(dump),'candidates_sha256':sha(out/'candidates.parquet'),'events_sha256':sha(out/'events.parquet'),'script_sha256':sha(Path(__file__)),'mot20_test_reads':0};write_json(out/'freeze_manifest.json',manifest);print(json.dumps(manifest,indent=2,sort_keys=True))

def main():
 p=argparse.ArgumentParser();p.add_argument('--seq',required=True);p.add_argument('--out-root',required=True);a=p.parse_args();freeze(a.seq,Path(a.out_root))
if __name__=='__main__':main()
