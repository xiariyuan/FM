from __future__ import annotations
import argparse,hashlib,importlib.util,io,json,math,struct,subprocess,sys,zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd
from scipy.optimize import linear_sum_assignment
REPO=Path(__file__).resolve().parents[2]
SEQ='MOT20-01'
BASELINE=REPO/'outputs/mot20_m23_20260718/m23_46_pure_hgb_topk_combined_oof_v1/track_results/MOT20-01.txt'
DUMP=REPO/'outputs/alink_train_inputs/phase0_root/MOT20-01/dump_yolox_reid.npz'
ROOT=REPO/'outputs/mot20_m28_20260726/m28_a3_strong_host_m23_46_m01_v1'
TOPK=8;MAX_GAP=120;MIN_ROWS=4;MIN_REID=2;IOU_THR=.5

def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write_json(path:Path,payload:Any):path.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
def load(name,relative):
 p=REPO/relative;s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def member(path,name,allow_pickle=False):
 with zipfile.ZipFile(path) as z:return np.load(io.BytesIO(z.read(name)),allow_pickle=allow_pickle)
def mmap_member(path,name):
 with zipfile.ZipFile(path) as z:
  info=z.getinfo(name)
  if info.compress_type!=zipfile.ZIP_STORED:raise RuntimeError('compressed feature member')
 with path.open('rb') as f:
  f.seek(info.header_offset+26);fn,ex=struct.unpack('<HH',f.read(4));off=info.header_offset+30+fn+ex;f.seek(off);v=np.lib.format.read_magic(f);shape,fort,dtype=np.lib.format._read_array_header(f,v);arr=f.tell()
 return np.memmap(path,dtype=dtype,mode='r',offset=arr,shape=shape,order='F' if fort else 'C')
def iou(a,b):
 if not len(a) or not len(b):return np.zeros((len(a),len(b)),np.float32)
 x1=np.maximum(a[:,None,0],b[None,:,0]);y1=np.maximum(a[:,None,1],b[None,:,1]);x2=np.minimum(a[:,None,2],b[None,:,2]);y2=np.minimum(a[:,None,3],b[None,:,3]);inter=np.maximum(0,x2-x1)*np.maximum(0,y2-y1);aa=np.maximum(0,a[:,2]-a[:,0])*np.maximum(0,a[:,3]-a[:,1]);bb=np.maximum(0,b[:,2]-b[:,0])*np.maximum(0,b[:,3]-b[:,1]);return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)
def parse_tracker(path):
 rows=[]
 for idx,line in enumerate(path.open()):
  p=line.rstrip().split(',')
  if len(p)<7:continue
  f=int(float(p[0]));tid=int(float(p[1]));x,y,w,h=map(float,p[2:6]);rows.append({'row_index':idx,'fields':p,'frame':f,'track_id':tid,'x':x,'y':y,'w':w,'h':h,'x1':x,'y1':y,'x2':x+w,'y2':y+h,'cx':x+.5*w,'cy':y+.5*h})
 return rows
def map_phase(rows):
 det=member(DUMP,'detections.npy');cols=member(DUMP,'columns.npy',True).tolist();ci={x:i for i,x in enumerate(cols)};offs=member(DUMP,'frame_offsets.npy');feat=mmap_member(DUMP,'features.npy');phase=np.full(len(rows),-1,np.int64);match_iou=np.zeros(len(rows),np.float32);by=defaultdict(list)
 for i,r in enumerate(rows):by[int(r['frame'])].append(i)
 for f,inds0 in by.items():
  inds=np.asarray(inds0,np.int64);st=int(offs[f-1]);en=int(offs[f]) if f<len(offs) else len(det);d=det[st:en]
  if not len(d):continue
  tb=np.asarray([[rows[i][k] for k in ('x1','y1','x2','y2')] for i in inds],np.float32);db=d[:,[ci['x1'],ci['y1'],ci['x2'],ci['y2']]].astype(np.float32);ov=iou(tb,db);ri,cj=linear_sum_assignment(-ov)
  for a,b in zip(ri,cj):
   if ov[a,b]>=IOU_THR and d[b,ci['has_reid']]>.5:phase[inds[a]]=st+b;match_iou[inds[a]]=ov[a,b]
 return phase,match_iou,feat
def unit(x):
 x=np.asarray(x,np.float32);n=float(np.linalg.norm(x));return None if n<=1e-12 or not np.isfinite(n) else x/n
def proto(gids,feat,limit,from_end):
 q=gids[-limit:] if from_end else gids[:limit];v=[]
 for g in q:
  if g<0:continue
  z=unit(feat[int(g)])
  if z is not None:v.append(z)
 if not v:return None,0
 return unit(np.mean(np.stack(v),0)),len(v)
def velocity(rows):
 q=rows[-5:]
 if len(q)<2:return 0.,0.
 f=np.asarray([r['frame'] for r in q],float)
 if len(np.unique(f))<2:return 0.,0.
 return float(np.polyfit(f,[r['cx'] for r in q],1)[0]),float(np.polyfit(f,[r['cy'] for r in q],1)[0])
def freeze():
 out=ROOT/'frozen_candidates'
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True)
 rows=parse_tracker(BASELINE);phase,miou,feat=map_phase(rows);by=defaultdict(list)
 for i,r in enumerate(rows):by[int(r['track_id'])].append(i)
 summaries={};events=[];cands=[];event_idx=0
 for tid,inds in sorted(by.items()):
  inds.sort(key=lambda i:(rows[i]['frame'],i));track_rows=[rows[i] for i in inds];mapped=[i for i in inds if phase[i]>=0]
  if len(track_rows)<2 or len(mapped)<2:continue
  confirm_line=mapped[1];confirm_frame=int(rows[confirm_line]['frame']);young_first=int(track_rows[0]['frame']);young_gids=[int(phase[i]) for i in mapped if int(rows[i]['frame'])<=confirm_frame];yp,yc=proto(young_gids,feat,2,False)
  if yp is None:continue
  events.append({'event_index':event_idx,'confirmation_frame':confirm_frame,'birth_frame':young_first,'young_track_id':tid,'young_rows':len(track_rows),'young_reid_count':yc,'gt_opened':False})
  summaries[tid]={'inds':inds,'rows':track_rows,'first_frame':young_first,'last_frame':int(track_rows[-1]['frame']),'frames':{int(x['frame']) for x in track_rows},'mapped':mapped}
  event_idx+=1
 # complete summaries for all tracks, including those without events
 for tid,inds in by.items():
  if tid in summaries:continue
  inds.sort(key=lambda i:(rows[i]['frame'],i));tr=[rows[i] for i in inds];summaries[tid]={'inds':inds,'rows':tr,'first_frame':int(tr[0]['frame']),'last_frame':int(tr[-1]['frame']),'frames':{int(x['frame']) for x in tr},'mapped':[i for i in inds if phase[i]>=0]}
 for event in events:
  yi=int(event['young_track_id']);birth=int(event['birth_frame']);confirm=int(event['confirmation_frame']);young=summaries[yi];ygids=[int(phase[i]) for i in young['mapped'] if int(rows[i]['frame'])<=confirm];yp,yc=proto(ygids,feat,2,False);event_row=next(x for x in young['rows'] if int(x['frame'])==confirm);records=[]
  for old_id,old in summaries.items():
   if old_id==yi or int(old['last_frame'])>=birth or len(old['rows'])<MIN_ROWS:continue
   gap=birth-int(old['last_frame'])
   if gap<1 or gap>MAX_GAP or old['frames'] & young['frames']:continue
   ogids=[int(phase[i]) for i in old['mapped']];op,oc=proto(ogids,feat,8,True)
   if op is None or oc<MIN_REID:continue
   vx,vy=velocity(old['rows']);dt=float(confirm-int(old['last_frame']));predx=float(old['rows'][-1]['cx'])+vx*dt;predy=float(old['rows'][-1]['cy'])+vy*dt;scale=max(.5*(float(old['rows'][-1]['h'])+float(event_row['h'])),1.0);motion=math.hypot(float(event_row['cx'])-predx,float(event_row['cy'])-predy)/scale;hr=max(float(event_row['h']),1e-3)/max(float(old['rows'][-1]['h']),1e-3);app=float(op@yp);score=app-.35*motion-.03*math.log1p(gap)-.08*abs(math.log(hr))
   records.append({'event_index':int(event['event_index']),'confirmation_frame':confirm,'birth_frame':birth,'young_track_id':yi,'old_track_id':int(old_id),'gap':gap,'old_rows':len(old['rows']),'old_reid_count':oc,'young_reid_count':yc,'appearance_cos':app,'motion_error':motion,'height_ratio':hr,'candidate_score':score,'gt_opened':False})
  records.sort(key=lambda x:(-x['candidate_score'],-x['appearance_cos'],x['motion_error'],x['old_track_id']))
  for rank,r in enumerate(records[:TOPK],1):r['candidate_rank']=rank;cands.append(r)
 ef=pd.DataFrame(events);cf=pd.DataFrame(cands)
 if cf.empty:raise RuntimeError('no candidates')
 ef.to_parquet(out/'events.parquet',index=False);cf.to_parquet(out/'candidates.parquet',index=False)
 manifest={'experiment_id':'M28-A3','stage':'candidates_frozen','seq':SEQ,'host':'strict M23-46','gt_opened':False,'mot20_test_reads':0,'baseline_sha256':sha256(BASELINE),'input_sha256':sha256(DUMP),'events':len(ef),'events_with_candidates':int(cf.event_index.nunique()),'candidate_actions':len(cf),'candidate_topk':TOPK,'max_gap':MAX_GAP,'mapped_rows':int((phase>=0).sum()),'tracker_rows':len(rows),'events_sha256':sha256(out/'events.parquet'),'candidates_sha256':sha256(out/'candidates.parquet'),'script_sha256':sha256(Path(__file__)),'forbidden_gt_columns':[]}
 write_json(out/'freeze_manifest.json',manifest);print(json.dumps(manifest,indent=2,sort_keys=True));print(cf.groupby('event_index').agg(n=('old_track_id','size'),best=('candidate_score','max'),rank1_app=('appearance_cos','max')).head(100).to_string())
def dominant(matches):
 c=matches.groupby(['tracker_id','gt_id']).size().reset_index(name='count');tot=c.groupby('tracker_id')['count'].sum().to_dict();c['purity']=[float(r.count)/max(int(tot[int(r.tracker_id)]),1) for r in c.itertuples(index=False)];b=c.sort_values(['tracker_id','count','gt_id'],ascending=[True,False,True]).groupby('tracker_id',as_index=False).first();return dict(zip(b.tracker_id.astype(int),b.gt_id.astype(int))),dict(zip(b.tracker_id.astype(int),b.purity.astype(float)))
def official(trackdir,name,out):
 cmd=[sys.executable,'scripts/eval_motstyle_trackeval.py','--benchmark-name','MOT20','--split-to-eval','train','--gt-root','datasets/MOT20/train','--results-dir',str(trackdir),'--tracker-name',name,'--work-dir',str(out),'--keep-workdir','--seqs',SEQ];p=subprocess.run(cmd,cwd=REPO,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out.parent/'official.log').write_text(p.stdout)
 if p.returncode:raise RuntimeError(p.stdout[-8000:])
 q=out/'eval'/name/'pedestrian_summary.txt';lines=[x.strip() for x in q.read_text().splitlines() if x.strip()];v=dict(zip(lines[0].split(),lines[1].split()));return {k:float(v[k]) for k in ('HOTA','DetA','AssA','IDF1')}|{'IDSW':int(float(v['IDSW'])),'Dets':int(float(v['Dets']))}
def teacher():
 frozen=ROOT/'frozen_candidates';out=ROOT/'teacher_capacity'
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);cands=pd.read_parquet(frozen/'candidates.parquet');m37=load('m28a3_exact','scripts/m23_research/m23_37_fast_exact_hota_teacher.py');m27=load('m28a3_ids','scripts/m27_research/m27_a0_exact_idsw_source_attribution.py');m28=load('m28a3_writer','scripts/m28_research/m28_a0_deferred_identity_inheritance.py');prepared=m37.PreparedExactHOTA(SEQ,BASELINE,out/'exact_cache');tmap,_=m27.original_tracker_mapping(prepared);_,matches=m27.reconstruct_clear_idsw(prepared.data,tmap);dmap,purity=dominant(matches);cands['young_gt']=cands.young_track_id.map(dmap).fillna(-1).astype(int);cands['old_gt']=cands.old_track_id.map(dmap).fillna(-2).astype(int);cands['young_purity']=cands.young_track_id.map(purity).fillna(0.0);cands['old_purity']=cands.old_track_id.map(purity).fillna(0.0);cands['same_dominant_gt']=((cands.young_gt>0)&(cands.young_gt==cands.old_gt)).astype(int);screen=cands[cands.same_dominant_gt==1].copy();base_ids=prepared.parent_row_ids.copy();base_metrics=prepared.evaluate_row_ids_incremental(base_ids);labels=[]
 for i,c in enumerate(screen.itertuples(index=False),1):
  ids=base_ids.copy();mask=ids==int(c.young_track_id);ids[mask]=int(c.old_track_id);valid=bool(mask.any())
  if valid:
   frames=np.asarray([int(float(x[0])) for x in prepared.parent_rows]);
   for f in np.unique(frames[mask]):
    z=ids[frames==f]
    if len(z)!=len(np.unique(z)):valid=False;break
  if not valid:labels.append({**c._asdict(),'status':'invalid','modified_rows':int(mask.sum()),'delta_HOTA':math.nan});continue
  met=prepared.evaluate_row_ids_incremental(ids);labels.append({**c._asdict(),'status':'success','modified_rows':int(mask.sum()),'HOTA':float(met['HOTA']),'DetA':float(met['DetA']),'AssA':float(met['AssA']),'delta_HOTA':float(met['HOTA']-base_metrics['HOTA']),'delta_DetA':float(met['DetA']-base_metrics['DetA']),'delta_AssA':float(met['AssA']-base_metrics['AssA'])})
  if i%50==0:print('exact',i,'/',len(screen),flush=True)
 lf=pd.DataFrame(labels);lf.to_parquet(out/'identity_consistent_exact_labels.parquet',index=False);success=lf[lf.status=='success'].sort_values(['delta_HOTA','candidate_score'],ascending=[False,False]);current=base_ids.copy();cm=dict(base_metrics);used_y=set();used_o=set();selected=[];frames=np.asarray([int(float(x[0])) for x in prepared.parent_rows])
 for c in success[success.delta_HOTA>0].itertuples(index=False):
  y,o=int(c.young_track_id),int(c.old_track_id)
  if y in used_y or o in used_o:continue
  proposal=current.copy();mask=proposal==y
  if not mask.any():continue
  proposal[mask]=o;valid=True
  for f in np.unique(frames[mask]):
   z=proposal[frames==f]
   if len(z)!=len(np.unique(z)):valid=False;break
  if not valid:continue
  met=prepared.evaluate_row_ids_incremental(proposal);gain=float(met['HOTA']-cm['HOTA'])
  if gain<=0:continue
  current=proposal;cm=met;used_y.add(y);used_o.add(o);selected.append({'event_index':int(c.event_index),'confirmation_frame':int(c.confirmation_frame),'birth_frame':int(c.birth_frame),'young_track_id':y,'old_track_id':o,'candidate_rank':int(c.candidate_rank),'individual_delta_HOTA':float(c.delta_HOTA),'step_delta_HOTA':gain,'modified_rows':int(mask.sum())})
 sf=pd.DataFrame(selected);sf.to_csv(out/'selected_actions.csv',index=False);path=out/'track_results'/f'{SEQ}.txt';m28.write_tracker(path,prepared,current);m28.SEQ=SEQ;off=official(path.parent,'m28_a3_m23_46_teacher',out/'official_eval');delta=float(cm['HOTA']-base_metrics['HOTA']);positive=int((success.delta_HOTA>0).sum());gate=bool(positive>=5 and delta>=.30 and off['IDSW']<=46);report={'experiment_id':'M28-A3','seq':SEQ,'status':'completed','decision':'PASS_M28_A3_EXPAND_STRONG_HOST' if gate else 'FAIL_M28_A3_CLOSE_STRONG_HOST','teacher_only':True,'deployable':False,'gt_opened_after_candidate_freeze':True,'mot20_test_reads':0,'baseline_metrics':{k:float(base_metrics[k]) for k in ('HOTA','DetA','AssA')},'candidate_actions':len(cands),'identity_consistent_actions':len(screen),'positive_actions':positive,'selected_actions':len(selected),'combined_metrics':{k:float(cm[k]) for k in ('HOTA','DetA','AssA')},'combined_delta_HOTA':delta,'official_trackeval':off,'gate':{'minimum_positive_actions':5,'minimum_delta_HOTA':.30,'IDSW_nonincrease_vs_M23_46_M01_46':True,'pass':gate},'candidate_manifest_sha256':sha256(frozen/'freeze_manifest.json'),'tracker_sha256':sha256(path)};write_json(ROOT/'report.json',report);pd.DataFrame([{'baseline_HOTA':base_metrics['HOTA'],'teacher_HOTA':cm['HOTA'],'delta_HOTA':delta,'positive_actions':positive,'selected_actions':len(selected),'official_IDSW':off['IDSW'],'gate_pass':int(gate),'decision':report['decision']}]).to_csv(ROOT/'summary.csv',index=False);print(json.dumps(report,indent=2,sort_keys=True));print('\nTOP\n',success.head(30).to_string(index=False));print('\nSELECTED\n',sf.to_string(index=False))
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','teacher'));a=p.parse_args();ROOT.mkdir(parents=True,exist_ok=True);freeze() if a.stage=='freeze' else teacher()
if __name__=='__main__':main()
