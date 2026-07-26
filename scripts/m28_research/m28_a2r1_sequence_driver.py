from __future__ import annotations
import argparse, importlib.util, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'outputs/mot20_m28_20260726/m28_a2r1_causal_multisequence'
OLD_ROOT=REPO/'outputs/mot20_m28_20260726/m28_a2_multisequence_capacity'
SEQS=('MOT20-01','MOT20-02','MOT20-03','MOT20-05')

def load(name,relative):
 p=REPO/relative;s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def root(seq):return ROOT/seq

def freeze(seq):
 mod=load(f'm28r1_freeze_{seq[-2:]}','scripts/m28_research/m28_a2r1_causal_candidate_repair.py')
 mod.SEQ=seq;mod.ROOT=root(seq);mod.BASELINE=OLD_ROOT/seq/'frozen_runtime/baseline_online.txt';mod.UPDATES=OLD_ROOT/seq/'frozen_runtime/association_updates.parquet';mod.DUMP=REPO/'outputs/alink_train_inputs/phase0_root'/seq/'dump_yolox_reid.npz';mod.freeze_candidates()

def dominant_identity(matches):
 counts=matches.groupby(['tracker_id','gt_id']).size().reset_index(name='count');tot=counts.groupby('tracker_id')['count'].sum().to_dict();counts['purity']=[float(r.count)/max(int(tot[int(r.tracker_id)]),1) for r in counts.itertuples(index=False)];best=counts.sort_values(['tracker_id','count','gt_id'],ascending=[True,False,True]).groupby('tracker_id',as_index=False).first();return dict(zip(best.tracker_id.astype(int),best.gt_id.astype(int))),dict(zip(best.tracker_id.astype(int),best.purity.astype(float)))

def teacher(seq):
 r=root(seq);out=r/'teacher_capacity'
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);C=pd.read_parquet(r/'frozen_candidates/candidates.parquet')
 m27=load(f'm27r1_{seq[-2:]}','scripts/m27_research/m27_a0_exact_idsw_source_attribution.py');m37=load(f'm37r1_{seq[-2:]}','scripts/m23_research/m23_37_fast_exact_hota_teacher.py');m28=load(f'm28r1_teacher_{seq[-2:]}','scripts/m28_research/m28_a0_deferred_identity_inheritance.py')
 baseline=OLD_ROOT/seq/'frozen_runtime/baseline_online.txt';prepared=m37.PreparedExactHOTA(seq,baseline,out/'exact_cache');tracker_map,_=m27.original_tracker_mapping(prepared);switches,matches=m27.reconstruct_clear_idsw(prepared.data,tracker_map);dmap,purity=dominant_identity(matches)
 C['young_gt']=C.young_track_id.map(dmap).fillna(-1).astype(int);C['old_gt']=C.old_track_id.map(dmap).fillna(-2).astype(int);C['young_purity']=C.young_track_id.map(purity).fillna(0.0);C['old_purity']=C.old_track_id.map(purity).fillna(0.0);C['same_dominant_gt']=((C.young_gt>0)&(C.young_gt==C.old_gt)).astype(int);screen=C[C.same_dominant_gt==1].copy()
 base_ids=prepared.parent_row_ids.copy();frames=np.asarray([int(float(f[0])) for f in prepared.parent_rows],np.int32);base=prepared.evaluate_row_ids_incremental(base_ids);labels=[]
 for k,c in enumerate(screen.itertuples(index=False),1):
  ids=base_ids.copy();mask=(ids==int(c.young_track_id))&(frames>=int(c.frame));ids[mask]=int(c.old_track_id);valid=bool(mask.any())
  if valid:
   for f in np.unique(frames[mask]):
    q=ids[frames==f]
    if len(q)!=len(np.unique(q)):valid=False;break
  if not valid:labels.append({**c._asdict(),'status':'invalid_future_identity_conflict','modified_rows':int(mask.sum()),'delta_HOTA':math.nan});continue
  met=prepared.evaluate_row_ids_incremental(ids);labels.append({**c._asdict(),'status':'success','modified_rows':int(mask.sum()),'HOTA':float(met['HOTA']),'DetA':float(met['DetA']),'AssA':float(met['AssA']),'delta_HOTA':float(met['HOTA']-base['HOTA']),'delta_DetA':float(met['DetA']-base['DetA']),'delta_AssA':float(met['AssA']-base['AssA'])})
  if k%100==0:print(seq,'exact',k,'/',len(screen),flush=True)
 L=pd.DataFrame(labels);L.to_parquet(out/'identity_consistent_exact_labels.parquet',index=False);S=L[L.status=='success'].sort_values(['delta_HOTA','candidate_score'],ascending=[False,False]);current=base_ids.copy();cur=dict(base);used_y=set();used_o=set();selected=[]
 for c in S[S.delta_HOTA>0].itertuples(index=False):
  y,o=int(c.young_track_id),int(c.old_track_id)
  if y in used_y or o in used_o:continue
  prop=current.copy();mask=(prop==y)&(frames>=int(c.frame));prop[mask]=o
  if not mask.any():continue
  valid=True
  for f in np.unique(frames[mask]):
   q=prop[frames==f]
   if len(q)!=len(np.unique(q)):valid=False;break
  if not valid:continue
  met=prepared.evaluate_row_ids_incremental(prop);gain=float(met['HOTA']-cur['HOTA'])
  if gain<=0:continue
  current=prop;cur=met;used_y.add(y);used_o.add(o);selected.append({'event_index':int(c.event_index),'frame':int(c.frame),'young_track_id':y,'old_track_id':o,'candidate_rank':int(c.candidate_rank),'individual_delta_HOTA':float(c.delta_HOTA),'step_delta_HOTA':gain,'modified_rows':int(mask.sum()),'young_purity':float(c.young_purity),'old_purity':float(c.old_purity)})
 Sel=pd.DataFrame(selected);Sel.to_csv(out/'selected_actions.csv',index=False);tracker=out/'track_results'/f'{seq}.txt';m28.write_tracker(tracker,prepared,current);m28.SEQ=seq;official=m28.official_eval(tracker.parent,f'm28_a2r1_{seq}_teacher',out/'official_eval')
 updates=pd.read_parquet(OLD_ROOT/seq/'frozen_runtime/association_updates.parquet');group={(int(f),int(t)):g for (f,t),g in updates.groupby(['frame','track_id'],sort=False)};unconf=[]
 for sw in switches.itertuples(index=False):
  g=group.get((int(sw.frame),int(sw.tracker_id)),pd.DataFrame())
  if len(g) and ((g.method.astype(str)=='update')&(g.stage.astype(str)=='unconfirmed')).any():unconf.append(sw)
 covered=sum(bool(((C.frame==int(sw.frame))&(C.young_track_id==int(sw.tracker_id))&(C.old_track_id==int(sw.previous_tracker_id))).any()) for sw in unconf)
 old=json.loads((OLD_ROOT/seq/'report.json').read_text()) if (OLD_ROOT/seq/'report.json').exists() else {}
 rep={'experiment_id':'M28-A2-R1','seq':seq,'status':'completed','teacher_only':True,'deployable':False,'gt_opened_after_candidate_freeze':True,'future_row_reads_in_candidate_generation':0,'candidate_actions':len(C),'identity_consistent_actions':len(screen),'valid_identity_consistent_actions':len(S),'invalid_future_identity_conflicts':int((L.status!='success').sum()),'positive_identity_consistent_actions':int((S.delta_HOTA>0).sum()),'selected_actions':len(Sel),'baseline_metrics':{k:float(base[k]) for k in ['HOTA','DetA','AssA']},'combined_metrics':{k:float(cur[k]) for k in ['HOTA','DetA','AssA']},'combined_delta_HOTA':float(cur['HOTA']-base['HOTA']),'official_trackeval':official,'official_clear_idsw_baseline':len(switches),'unconfirmed_source_idsw':len(unconf),'correct_predecessor_top8_coverage':covered,'correct_predecessor_top8_rate':covered/max(len(unconf),1),'old_noncausal_delta_HOTA':old.get('combined_delta_HOTA'),'capacity_retention':float(cur['HOTA']-base['HOTA'])/max(float(old.get('combined_delta_HOTA',1e-12)),1e-12),'candidate_manifest':json.loads((r/'frozen_candidates/freeze_manifest.json').read_text()),'mot20_test_reads':0,'test_submission':False}
 (r/'report.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n');pd.DataFrame([{'seq':seq,'baseline_HOTA':base['HOTA'],'candidate_actions':len(C),'identity_consistent_actions':len(screen),'valid_actions':len(S),'positive_actions':int((S.delta_HOTA>0).sum()),'selected_actions':len(Sel),'teacher_HOTA':cur['HOTA'],'delta_HOTA':rep['combined_delta_HOTA'],'official_IDSW':official['IDSW'],'unconfirmed_IDSW':len(unconf),'predecessor_coverage':covered,'predecessor_rate':rep['correct_predecessor_top8_rate'],'capacity_retention':rep['capacity_retention']}]).to_csv(r/'summary.csv',index=False);print(json.dumps(rep,indent=2,sort_keys=True));print('\nTOP\n',S.head(30).to_string(index=False));print('\nSELECTED\n',Sel.head(80).to_string(index=False))

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze-candidates','teacher']);p.add_argument('--seq',required=True,choices=SEQS);a=p.parse_args();root(a.seq).mkdir(parents=True,exist_ok=True);freeze(a.seq) if a.stage=='freeze-candidates' else teacher(a.seq)
if __name__=='__main__':main()
