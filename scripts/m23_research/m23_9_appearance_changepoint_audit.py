from __future__ import annotations

# Research artifact for the MOT20 M23 sequence-LOSO audit.
import csv,importlib.util,io,json,math,struct,sys,zipfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score,average_precision_score

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
PARENT=Path('outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results')
PHASE=Path('outputs/alink_train_inputs/phase0_root')
OUT=Path('outputs/mot20_m23_20260718/appearance_changepoint_audit_v1')
DIM=128;SEED=2309;IOU_THR=.5;WINDOW=5

def load_module():
 s=importlib.util.spec_from_file_location('m23','scripts/audit_m23_mot20_expanded_evidence_oracle.py');m=importlib.util.module_from_spec(s);sys.modules['m23']=m;s.loader.exec_module(m);return m

def read_member(path,name,allow_pickle=False):
 with zipfile.ZipFile(path) as z:return np.load(io.BytesIO(z.read(name)),allow_pickle=allow_pickle)

def mmap_member(path,member):
 with zipfile.ZipFile(path) as z:
  info=z.getinfo(member)
  if info.compress_type!=zipfile.ZIP_STORED:raise RuntimeError('compressed feature member')
 with path.open('rb') as f:
  f.seek(info.header_offset+26);fn,ex=struct.unpack('<HH',f.read(4));off=info.header_offset+30+fn+ex;f.seek(off);ver=np.lib.format.read_magic(f);shape,fort,dtype=np.lib.format._read_array_header(f,ver);arr_off=f.tell()
 return np.memmap(path,dtype=dtype,mode='r',offset=arr_off,shape=shape,order='F' if fort else 'C')

def iou_matrix(a,b):
 if len(a)==0 or len(b)==0:return np.zeros((len(a),len(b)),np.float32)
 xx1=np.maximum(a[:,None,0],b[None,:,0]);yy1=np.maximum(a[:,None,1],b[None,:,1]);xx2=np.minimum(a[:,None,2],b[None,:,2]);yy2=np.minimum(a[:,None,3],b[None,:,3]);inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1);aa=np.maximum(0,a[:,2]-a[:,0])*np.maximum(0,a[:,3]-a[:,1]);bb=np.maximum(0,b[:,2]-b[:,0])*np.maximum(0,b[:,3]-b[:,1]);return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)

def read_tracker(path):
 rows=[]
 for li,line in enumerate(path.open()):
  p=line.strip().split(',');
  if len(p)<6:continue
  f=int(float(p[0]));t=int(float(p[1]));x,y,w,h=map(float,p[2:6]);rows.append((li,f,t,x,y,x+w,y+h))
 return pd.DataFrame(rows,columns=['line','frame','track_id','x1','y1','x2','y2'])

def project_indices(features,unique_idx,proj):
 out=np.empty((len(unique_idx),proj.shape[1]),np.float32);chunk=2048
 for st in range(0,len(unique_idx),chunk):
  ids=unique_idx[st:st+chunk];x=np.asarray(features[ids],np.float32);x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);y=x@proj;y/=np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-12);out[st:st+len(ids)]=y
 return out

def modal(values):
 x=[int(v) for v in values if int(v)>0]
 if not x:return (0,0)
 c=Counter(x);g,n=c.most_common(1)[0]
 if list(c.values()).count(n)>1:return (0,n)
 return g,n

def main():
 OUT.mkdir(parents=True,exist_ok=True);m23=load_module();rng=np.random.default_rng(SEED);proj=rng.normal(size=(2048,DIM)).astype(np.float32)/math.sqrt(DIM);all_rows=[];summ=[]
 for seq in SEQS:
  tr=read_tracker(PARENT/f'{seq}.txt');n=len(tr);phase_path=PHASE/seq/'dump_yolox_reid.npz';det=read_member(phase_path,'detections.npy');cols=read_member(phase_path,'columns.npy',True).tolist();ci={x:i for i,x in enumerate(cols)};offs=read_member(phase_path,'frame_offsets.npy');features=mmap_member(phase_path,'features.npy')
  phase_idx=np.full(n,-1,np.int64);match_iou=np.zeros(n,np.float32)
  frame_groups=tr.groupby('frame').groups
  for f,idxs0 in frame_groups.items():
   idxs=np.asarray(list(idxs0),np.int64);st=int(offs[int(f)-1]);en=int(offs[int(f)]) if int(f)<len(offs) else len(det);d=det[st:en];
   if len(d)==0:continue
   tb=tr.loc[idxs,['x1','y1','x2','y2']].to_numpy(np.float32);db=d[:,[ci['x1'],ci['y1'],ci['x2'],ci['y2']]].astype(np.float32);ov=iou_matrix(tb,db);ri,cj=linear_sum_assignment(-ov)
   for r,c in zip(ri,cj):
    if ov[r,c]>=IOU_THR and d[c,ci['has_reid']]>.5:phase_idx[idxs[r]]=st+c;match_iou[idxs[r]]=ov[r,c]
  # GT row labels, diagnostic only.
  gt_id=np.zeros(n,np.int32);baseline=m23.load_baseline(PARENT/f'{seq}.txt');gt=m23.load_gt(Path('datasets/MOT20/train')/seq/'gt/gt.txt')
  for f in sorted(baseline):
   kept,valid,_=m23.valid_and_distractor_filtered(baseline[f],gt.get(f,[]))
   for ai,gi,ov in m23.match_candidates(kept,valid):
    line=int(kept[ai].uid & ((1<<32)-1));gt_id[line]=int(valid[gi].gt_id)
  unique=np.unique(phase_idx[phase_idx>=0]);emb=project_indices(features,unique,proj);pos=np.searchsorted(unique,phase_idx.clip(min=0));mapped=phase_idx>=0
  seq_rows=[]
  for tid,inds0 in tr.groupby('track_id').groups.items():
   inds=np.asarray(sorted(inds0,key=lambda i:int(tr.at[i,'frame'])),np.int64);valid_positions=np.flatnonzero(mapped[inds])
   if len(valid_positions)<4:continue
   for z in range(len(valid_positions)-1):
    left_pos=valid_positions[z];right_pos=valid_positions[z+1];left=inds[max(0,left_pos-WINDOW+1):left_pos+1];right=inds[right_pos:min(len(inds),right_pos+WINDOW)]
    left=left[mapped[left]];right=right[mapped[right]]
    if len(left)<2 or len(right)<2:continue
    gl,nl=modal(gt_id[left]);gr,nr=modal(gt_id[right])
    if gl<=0 or gr<=0:continue
    le=emb[pos[left]].mean(axis=0);re=emb[pos[right]].mean(axis=0);le/=max(np.linalg.norm(le),1e-12);re/=max(np.linalg.norm(re),1e-12)
    i1=inds[left_pos];i2=inds[right_pos];row={'seq':seq,'track_id':int(tid),'left_frame':int(tr.at[i1,'frame']),'right_frame':int(tr.at[i2,'frame']),'frame_gap':int(tr.at[i2,'frame']-tr.at[i1,'frame']),'window_cos':float(le@re),'pair_cos':float(emb[pos[i1]]@emb[pos[i2]]),'left_gt':gl,'right_gt':gr,'boundary':int(gl!=gr),'left_support':nl,'right_support':nr,'left_rows':len(left),'right_rows':len(right),'left_match_iou':float(match_iou[i1]),'right_match_iou':float(match_iou[i2])};seq_rows.append(row)
  sdf=pd.DataFrame(seq_rows);sdf.to_csv(OUT/f'{seq}_transitions.csv',index=False);all_rows+=seq_rows
  if len(sdf) and sdf.boundary.nunique()==2:
   auc=roc_auc_score(sdf.boundary,-sdf.window_cos);ap=average_precision_score(sdf.boundary,-sdf.window_cos);pauc=roc_auc_score(sdf.boundary,-sdf.pair_cos)
  else:auc=ap=pauc=None
  rec={'seq':seq,'tracker_rows':n,'mapped_rows':int(mapped.sum()),'mapping_rate':float(mapped.mean()),'transitions':len(sdf),'boundaries':int(sdf.boundary.sum()) if len(sdf) else 0,'window_drop_auc':auc,'window_drop_ap':ap,'pair_drop_auc':pauc};summ.append(rec);print(json.dumps(rec),flush=True)
 pd.DataFrame(all_rows).to_csv(OUT/'all_transitions.csv',index=False);d=pd.DataFrame(all_rows)
 report={'protocol':{'features':'Phase-0 FastReID, frame-IoU Hungarian mapping, fixed 128D Gaussian projection seed 2309','window':WINDOW,'gt_use':'diagnostic labels only','deployment_allowed':False},'by_seq':summ}
 if len(d) and d.boundary.nunique()==2:report['combined']={'transitions':len(d),'boundaries':int(d.boundary.sum()),'window_drop_auc':float(roc_auc_score(d.boundary,-d.window_cos)),'window_drop_ap':float(average_precision_score(d.boundary,-d.window_cos)),'pair_drop_auc':float(roc_auc_score(d.boundary,-d.pair_cos))}
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
