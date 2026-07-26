from __future__ import annotations
import hashlib,importlib.util,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np,pandas as pd
REPO=Path(__file__).resolve().parents[1]
SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
BASE=REPO/'outputs/mot20_m23_20260718/m23_46_pure_hgb_topk_combined_oof_v1/track_results'
OUT=REPO/'outputs/mot20_m29_20260726/m29_a0_m23_46_identity_episode_attribution'
def load(name,p):
 s=importlib.util.spec_from_file_location(name,REPO/p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def parse_frames(path):
 by=defaultdict(list)
 for line in Path(path).open():
  a=line.rstrip().split(',')
  if len(a)>=2:by[int(float(a[1]))].append(int(float(a[0])))
 return {k:sorted(set(v)) for k,v in by.items()}
def main():
 OUT.mkdir(parents=True,exist_ok=True);m27=load('m29_m27','scripts/m27_research/m27_a0_exact_idsw_source_attribution.py');m37=load('m29_m37','scripts/m23_research/m23_37_fast_exact_hota_teacher.py')
 rows=[];summ=[]
 for seq in SEQS:
  track=BASE/f'{seq}.txt';prepared=m37.PreparedExactHOTA(seq,track,OUT/seq/'exact_cache');tracker_map,_=m27.original_tracker_mapping(prepared);switches,_=m27.reconstruct_clear_idsw(prepared.data,tracker_map);frames_by=parse_frames(track)
  for sw in switches.itertuples(index=False):
   frame=int(sw.frame);tid=int(sw.tracker_id);fs=frames_by.get(tid,[]);pos=int(np.searchsorted(fs,frame,'left'))
   if pos>=len(fs) or fs[pos]!=frame:continue
   start=pos
   while start>0 and fs[start]-fs[start-1]==1:start-=1
   episode_age=pos-start+1;full_age=pos+1;previous_same=fs[pos-1] if pos>0 else None;same_gap=None if previous_same is None else frame-previous_same
   if episode_age<=3:cat='episode_age_1_3'
   elif episode_age<=8:cat='episode_age_4_8'
   else:cat='episode_age_9plus'
   rows.append({'seq':seq,'frame':frame,'gt_id':int(sw.gt_id),'tracker_id':tid,'previous_tracker_id':int(sw.previous_tracker_id),'gap_since_last_gt_match':int(sw.gap_since_last_match),'similarity':float(sw.similarity),'full_track_age_rows':full_age,'episode_start_frame':int(fs[start]),'episode_age_rows':episode_age,'same_tracker_previous_frame':previous_same,'same_tracker_gap':same_gap,'category':cat,'new_full_track':int(full_age<=3),'reentry_episode':int(start>0 and episode_age<=3)})
  d=pd.DataFrame([x for x in rows if x['seq']==seq]);n=len(d);early=int((d.episode_age_rows<=3).sum());summ.append({'seq':seq,'official_reconstructed_IDSW':n,'episode_age_1_3':early,'episode_age_1_3_rate':early/max(n,1),'new_full_track':int(d.new_full_track.sum()),'reentry_episode':int(d.reentry_episode.sum()),'tracker_sha256':sha(track)})
 D=pd.DataFrame(rows);S=pd.DataFrame(summ);D.to_parquet(OUT/'idsw_episode_events.parquet',index=False);S.to_csv(OUT/'per_sequence_summary.csv',index=False)
 total=len(D);early=int((D.episode_age_rows<=3).sum());gate={'combined_episode_age_1_3_rate_at_least_0p45':early/max(total,1)>=.45,'every_sequence_rate_at_least_0p25':bool((S.episode_age_1_3_rate>=.25).all())};gate['pass']=all(gate.values())
 report={'experiment_id':'M29-A0','status':'completed','decision':'PASS_M29_A0_AUTHORIZE_DEFERRED_EPISODE_IDENTITY_CAPACITY' if gate['pass'] else 'FAIL_M29_A0_EPISODE_BIRTH_NOT_DOMINANT','diagnostic_only':True,'mot20_test_reads':0,'official_reconstructed_IDSW':total,'episode_age_1_3':early,'episode_age_1_3_rate':early/max(total,1),'new_full_track':int(D.new_full_track.sum()),'reentry_episode':int(D.reentry_episode.sum()),'category_counts':D.category.value_counts().to_dict(),'per_sequence':summ,'gate':gate,'events_sha256':sha(OUT/'idsw_episode_events.parquet')}
 (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2,sort_keys=True));print('\nCROSSTAB\n',pd.crosstab(D.seq,D.category).to_string())
if __name__=='__main__':main()
