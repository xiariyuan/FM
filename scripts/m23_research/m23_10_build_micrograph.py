from __future__ import annotations

# Research artifact for the MOT20 M23 sequence-LOSO audit.
import argparse,bisect,hashlib,importlib.util,io,json,math,struct,sys,zipfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
PARENT=Path('outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results')
PHASE=Path('outputs/alink_train_inputs/phase0_root')
OUT=Path('outputs/mot20_m23_20260718/micrograph_chunk30_v1')
CHUNK_SPAN=30;GAP_BREAK=1;MAX_GAP=300;TOPK=16;IOU_THR=.5;DIM=128;SEED=2310

def load_m23():
 s=importlib.util.spec_from_file_location('m23','scripts/audit_m23_mot20_expanded_evidence_oracle.py');m=importlib.util.module_from_spec(s);sys.modules['m23']=m;s.loader.exec_module(m);return m

def member(path,name,allow_pickle=False):
 with zipfile.ZipFile(path) as z:return np.load(io.BytesIO(z.read(name)),allow_pickle=allow_pickle)

def mmap_member(path,name):
 with zipfile.ZipFile(path) as z:
  inf=z.getinfo(name)
  if inf.compress_type!=zipfile.ZIP_STORED:raise RuntimeError('compressed feature member')
 with path.open('rb') as f:
  f.seek(inf.header_offset+26);fn,ex=struct.unpack('<HH',f.read(4));off=inf.header_offset+30+fn+ex;f.seek(off);v=np.lib.format.read_magic(f);shape,fort,dtype=np.lib.format._read_array_header(f,v);arr=f.tell()
 return np.memmap(path,dtype=dtype,mode='r',offset=arr,shape=shape,order='F' if fort else 'C')

def iou_matrix(a,b):
 if not len(a) or not len(b):return np.zeros((len(a),len(b)),np.float32)
 x1=np.maximum(a[:,None,0],b[None,:,0]);y1=np.maximum(a[:,None,1],b[None,:,1]);x2=np.minimum(a[:,None,2],b[None,:,2]);y2=np.minimum(a[:,None,3],b[None,:,3]);inter=np.maximum(0,x2-x1)*np.maximum(0,y2-y1);aa=np.maximum(0,a[:,2]-a[:,0])*np.maximum(0,a[:,3]-a[:,1]);bb=np.maximum(0,b[:,2]-b[:,0])*np.maximum(0,b[:,3]-b[:,1]);return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)

def read_tracker(path):
 rows=[]
 for li,line in enumerate(path.open()):
  p=line.rstrip('\n').split(',')
  if len(p)<6:continue
  f=int(float(p[0]));t=int(float(p[1]));x,y,w,h=map(float,p[2:6]);rows.append({'line':li,'frame':f,'track_id':t,'x1':x,'y1':y,'x2':x+w,'y2':y+h})
 return rows

def map_phase(rows,seq,proj):
 p=PHASE/seq/'dump_yolox_reid.npz';det=member(p,'detections.npy');cols=member(p,'columns.npy',True).tolist();ci={x:i for i,x in enumerate(cols)};offs=member(p,'frame_offsets.npy');feat=mmap_member(p,'features.npy');phase=np.full(len(rows),-1,np.int64);miou=np.zeros(len(rows),np.float32);by=defaultdict(list)
 for i,r in enumerate(rows):by[r['frame']].append(i)
 for f,ids0 in by.items():
  ids=np.asarray(ids0,np.int64);st=int(offs[f-1]);en=int(offs[f]) if f<len(offs) else len(det);d=det[st:en]
  if not len(d):continue
  tb=np.asarray([[rows[i][k] for k in ['x1','y1','x2','y2']] for i in ids],np.float32);db=d[:,[ci['x1'],ci['y1'],ci['x2'],ci['y2']]].astype(np.float32);ov=iou_matrix(tb,db);ri,cj=linear_sum_assignment(-ov)
  for a,b in zip(ri,cj):
   if ov[a,b]>=IOU_THR and d[b,ci['has_reid']]>.5:phase[ids[a]]=st+b;miou[ids[a]]=ov[a,b]
 unique=np.unique(phase[phase>=0]);emb=np.empty((len(unique),DIM),np.float32)
 for st in range(0,len(unique),2048):
  ids=unique[st:st+2048];x=np.asarray(feat[ids],np.float32);x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);y=x@proj;y/=np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-12);emb[st:st+len(ids)]=y
 pos=np.searchsorted(unique,phase.clip(min=0));return phase,miou,emb,pos

def gt_labels(rows,seq,m23):
 out=np.zeros(len(rows),np.int32);b=m23.load_baseline(PARENT/f'{seq}.txt');gt=m23.load_gt(Path('datasets/MOT20/train')/seq/'gt/gt.txt')
 for f in sorted(b):
  kept,valid,_=m23.valid_and_distractor_filtered(b[f],gt.get(f,[]))
  for ci,gi,ov in m23.match_candidates(kept,valid):out[int(kept[ci].uid & ((1<<32)-1))]=int(valid[gi].gt_id)
 return out

def slope(frames,values):
 if len(frames)<2 or frames[-1]==frames[0]:return 0.0
 x=np.asarray(frames,float);y=np.asarray(values,float);x=x-x.mean();return float((x@y)/max(x@x,1e-12))

def build_chunks(rows,phase,miou,emb,pos,gt):
 by=defaultdict(list)
 for i,r in enumerate(rows):by[r['track_id']].append(i)
 chunks=[];protos=[];cid=0
 for tid,ids in sorted(by.items()):
  ids.sort(key=lambda i:(rows[i]['frame'],i));start=0;ordinal=0
  for k in range(1,len(ids)+1):
   br=k==len(ids) or rows[ids[k]]['frame']-rows[ids[k-1]]['frame']>GAP_BREAK or rows[ids[k]]['frame']-rows[ids[start]]['frame']>=CHUNK_SPAN
   if not br:continue
   q=ids[start:k];mapped=[i for i in q if phase[i]>=0];proto=np.zeros(DIM,np.float32);cons=0.0
   if mapped:
    em=emb[pos[mapped]];proto=em.mean(axis=0);proto/=max(np.linalg.norm(proto),1e-12);cons=float(np.mean(em@proto))
   c=Counter(int(gt[i]) for i in q if int(gt[i])>0);modal,count=(c.most_common(1)[0] if c else (0,0));first,last=rows[q[0]],rows[q[-1]];head=q[:min(5,len(q))];tail=q[max(0,len(q)-5):]
   hf=[rows[i]['frame'] for i in head];tf=[rows[i]['frame'] for i in tail];hcx=[.5*(rows[i]['x1']+rows[i]['x2']) for i in head];hcy=[.5*(rows[i]['y1']+rows[i]['y2']) for i in head];tcx=[.5*(rows[i]['x1']+rows[i]['x2']) for i in tail];tcy=[.5*(rows[i]['y1']+rows[i]['y2']) for i in tail]
   row={'chunk_id':cid,'source_track_id':tid,'source_ordinal':ordinal,'first_frame':first['frame'],'last_frame':last['frame'],'span_frames':last['frame']-first['frame']+1,'rows':len(q),'first_line':q[0],'last_line':q[-1],'mapped_rows':len(mapped),'mapping_rate':len(mapped)/len(q),'mean_match_iou':float(np.mean(miou[mapped])) if mapped else 0.0,'appearance_consistency':cons,'first_cx':hcx[0],'first_cy':hcy[0],'last_cx':tcx[-1],'last_cy':tcy[-1],'first_h':first['y2']-first['y1'],'last_h':last['y2']-last['y1'],'start_vx':slope(hf,hcx),'start_vy':slope(hf,hcy),'end_vx':slope(tf,tcx),'end_vy':slope(tf,tcy),'modal_gt':modal,'modal_count':count,'modal_purity':count/max(sum(c.values()),1),'matched_gt_rows':sum(c.values())}
   chunks.append(row);protos.append(proto);cid+=1;ordinal+=1;start=k
 return pd.DataFrame(chunks),np.asarray(protos,np.float32)

def edge_features(a,b,cos):
 gap=int(b.first_frame-a.last_frame-1);h=max(.5*(a.last_h+b.first_h),1.0);dt=max(b.first_frame-a.last_frame,1);predx=a.last_cx+a.end_vx*dt;predy=a.last_cy+a.end_vy*dt;backx=b.first_cx-b.start_vx*dt;backy=b.first_cy-b.start_vy*dt;ferr=math.hypot(b.first_cx-predx,b.first_cy-predy)/h;berr=math.hypot(a.last_cx-backx,a.last_cy-backy)/h;disp=math.hypot(b.first_cx-a.last_cx,b.first_cy-a.last_cy)/h;va=np.asarray([a.end_vx,a.end_vy]);vb=np.asarray([b.start_vx,b.start_vy]);vn=float(va@vb/max(np.linalg.norm(va)*np.linalg.norm(vb),1e-8));same_source=int(a.source_track_id==b.source_track_id);adj=int(same_source and b.source_ordinal==a.source_ordinal+1)
 return {'src_chunk':int(a.chunk_id),'dst_chunk':int(b.chunk_id),'src_track':int(a.source_track_id),'dst_track':int(b.source_track_id),'gap':gap,'log_gap':math.log1p(max(gap,0)),'appearance_cos':float(cos),'same_source':same_source,'source_adjacent':adj,'forward_motion_error':ferr,'backward_motion_error':berr,'motion_error_min':min(ferr,berr),'motion_error_mean':.5*(ferr+berr),'endpoint_displacement':disp,'velocity_cos':vn,'log_height_ratio':math.log(max(b.first_h,1e-3)/max(a.last_h,1e-3)),'src_rows':int(a.rows),'dst_rows':int(b.rows),'src_mapping_rate':float(a.mapping_rate),'dst_mapping_rate':float(b.mapping_rate),'mapping_rate_min':min(a.mapping_rate,b.mapping_rate),'src_consistency':float(a.appearance_consistency),'dst_consistency':float(b.appearance_consistency),'consistency_min':min(a.appearance_consistency,b.appearance_consistency),'src_match_iou':float(a.mean_match_iou),'dst_match_iou':float(b.mean_match_iou),'same_gt':int(a.modal_gt>0 and a.modal_gt==b.modal_gt),'src_modal_gt':int(a.modal_gt),'dst_modal_gt':int(b.modal_gt),'src_purity':float(a.modal_purity),'dst_purity':float(b.modal_purity),'label_confidence':min(float(a.modal_purity),float(b.modal_purity))*min(float(a.mapping_rate),float(b.mapping_rate))}

def build_edges(meta,proto):
 if not np.array_equal(meta.chunk_id.to_numpy(int),np.arange(len(meta),dtype=int)):
  raise RuntimeError('chunk_id must equal metadata row index')
 rows=[];pairs=set();starts=meta.first_frame.to_numpy(int);order=np.argsort(starts);sorted_starts=starts[order];by_source=defaultdict(list)
 for r in meta.itertuples():by_source[int(r.source_track_id)].append(r)
 forced=set()
 for tid,items in by_source.items():
  items=sorted(items,key=lambda r:r.source_ordinal)
  for a,b in zip(items,items[1:]):
   if b.first_frame>a.last_frame:forced.add((int(a.chunk_id),int(b.chunk_id)))
 for a in meta.itertuples():
  lo=np.searchsorted(sorted_starts,a.last_frame+1,'left');hi=np.searchsorted(sorted_starts,a.last_frame+MAX_GAP+2,'left');cand=order[lo:hi]
  if len(cand):
   sims=proto[cand]@proto[int(a.chunk_id)];k=min(TOPK,len(cand));top=np.argpartition(-sims,k-1)[:k];top=top[np.argsort(-sims[top])]
   for q in top:
    j=int(cand[q]);pair=(int(a.chunk_id),j)
    if pair not in pairs:
     rows.append(edge_features(a,meta.iloc[j],float(sims[q])));pairs.add(pair)
 for i,j in sorted(forced):
  if (i,j) not in pairs:
   rows.append(edge_features(meta.iloc[i],meta.iloc[j],float(proto[i]@proto[j])));pairs.add((i,j))
 edges=pd.DataFrame(rows).reset_index(drop=True)
 edges['out_rank']=edges.groupby('src_chunk').appearance_cos.rank(method='first',ascending=False).astype(int);edges['in_rank']=edges.groupby('dst_chunk').appearance_cos.rank(method='first',ascending=False).astype(int);edges['max_rank']=edges[['out_rank','in_rank']].max(axis=1)
 best_out=edges.groupby('src_chunk').appearance_cos.transform('max');best_in=edges.groupby('dst_chunk').appearance_cos.transform('max');edges['out_margin']=best_out-edges.appearance_cos;edges['in_margin']=best_in-edges.appearance_cos;edges['max_margin']=edges[['out_margin','in_margin']].max(axis=1)
 return edges

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--seq',action='append',choices=SEQS);args=parser.parse_args();sequences=args.seq or SEQS
 OUT.mkdir(parents=True,exist_ok=True);m23=load_m23();rng=np.random.default_rng(SEED);proj=rng.normal(size=(2048,DIM)).astype(np.float32)/math.sqrt(DIM);report=[]
 for seq in sequences:
  rows=read_tracker(PARENT/f'{seq}.txt');phase,miou,emb,pos=map_phase(rows,seq,proj);gt=gt_labels(rows,seq,m23);meta,proto=build_chunks(rows,phase,miou,emb,pos,gt);edges=build_edges(meta,proto);sd=OUT/seq;sd.mkdir(parents=True,exist_ok=True);meta.to_parquet(sd/'microtracklets.parquet',index=False);np.save(sd/'prototypes.f16.npy',proto.astype(np.float16));edges.to_parquet(sd/'candidate_edges.parquet',index=False)
  clean=edges[(edges.src_purity>=.7)&(edges.dst_purity>=.7)&(edges.src_modal_gt>0)&(edges.dst_modal_gt>0)];rec={'seq':seq,'tracker_rows':len(rows),'mapped_rows':int((phase>=0).sum()),'chunks':len(meta),'edges':len(edges),'forced_source_edges':int(edges.source_adjacent.sum()),'clean_edges':len(clean),'clean_positive':int(clean.same_gt.sum()),'clean_positive_rate':float(clean.same_gt.mean()) if len(clean) else 0.0,'cross_edges':int((edges.same_source==0).sum()),'cross_positive':int(edges.loc[edges.same_source==0,'same_gt'].sum())};report.append(rec);print(json.dumps(rec),flush=True)
 (OUT/'report.json').write_text(json.dumps({'protocol':{'chunk_span':CHUNK_SPAN,'gap_break':GAP_BREAK,'max_gap':MAX_GAP,'appearance_topk':TOPK,'projected_dim':DIM,'projection_seed':SEED,'feature_gt_use':'labels and diagnostic purity only; excluded from model features'},'sequences':report},indent=2)+'\n')
if __name__=='__main__':main()
