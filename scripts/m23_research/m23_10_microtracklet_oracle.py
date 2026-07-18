from __future__ import annotations

# Research artifact for the MOT20 M23 sequence-LOSO audit.
import bisect,csv,hashlib,importlib.util,io,json,math,struct,subprocess,sys,zipfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
PARENT=Path('outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results')
PHASE=Path('outputs/alink_train_inputs/phase0_root')
OUT=Path('outputs/mot20_m23_20260718/microtracklet_oracle_chunk30_v1')
CHUNK_SPAN=30;GAP_BREAK=1;IOU_THR=.5;DIM=128;SEED=2310;SYN_BASE=10_000_000;STRIDE=1_000_000

def load_m23():
 s=importlib.util.spec_from_file_location('m23','scripts/audit_m23_mot20_expanded_evidence_oracle.py');m=importlib.util.module_from_spec(s);sys.modules['m23']=m;s.loader.exec_module(m);return m

def member(path,name,allow_pickle=False):
 with zipfile.ZipFile(path) as z:return np.load(io.BytesIO(z.read(name)),allow_pickle=allow_pickle)

def mmap_member(path,name):
 with zipfile.ZipFile(path) as z:
  inf=z.getinfo(name)
  if inf.compress_type!=zipfile.ZIP_STORED:raise RuntimeError('compressed')
 with path.open('rb') as f:
  f.seek(inf.header_offset+26);fn,ex=struct.unpack('<HH',f.read(4));off=inf.header_offset+30+fn+ex;f.seek(off);v=np.lib.format.read_magic(f);shape,fort,dtype=np.lib.format._read_array_header(f,v);arr=f.tell()
 return np.memmap(path,dtype=dtype,mode='r',offset=arr,shape=shape,order='F' if fort else 'C')

def iou(a,b):
 if not len(a) or not len(b):return np.zeros((len(a),len(b)),np.float32)
 x1=np.maximum(a[:,None,0],b[None,:,0]);y1=np.maximum(a[:,None,1],b[None,:,1]);x2=np.minimum(a[:,None,2],b[None,:,2]);y2=np.minimum(a[:,None,3],b[None,:,3]);inter=np.maximum(0,x2-x1)*np.maximum(0,y2-y1);aa=np.maximum(0,a[:,2]-a[:,0])*np.maximum(0,a[:,3]-a[:,1]);bb=np.maximum(0,b[:,2]-b[:,0])*np.maximum(0,b[:,3]-b[:,1]);return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)

def read_tracker(path):
 rows=[]
 for li,line in enumerate(path.open()):
  p=line.rstrip('\n').split(',')
  if len(p)<6:continue
  f=int(float(p[0]));t=int(float(p[1]));x,y,w,h=map(float,p[2:6]);rows.append({'line':li,'frame':f,'track_id':t,'x1':x,'y1':y,'x2':x+w,'y2':y+h,'fields':p})
 return rows

def map_phase(rows,seq,proj):
 p=PHASE/seq/'dump_yolox_reid.npz';det=member(p,'detections.npy');cols=member(p,'columns.npy',True).tolist();ci={x:i for i,x in enumerate(cols)};offs=member(p,'frame_offsets.npy');feat=mmap_member(p,'features.npy');phase=np.full(len(rows),-1,np.int64);miou=np.zeros(len(rows),np.float32);by=defaultdict(list)
 for i,r in enumerate(rows):by[r['frame']].append(i)
 for f,ids0 in by.items():
  ids=np.asarray(ids0,np.int64);st=int(offs[f-1]);en=int(offs[f]) if f<len(offs) else len(det);d=det[st:en]
  if not len(d):continue
  tb=np.asarray([[rows[i][k] for k in ['x1','y1','x2','y2']] for i in ids],np.float32);db=d[:,[ci['x1'],ci['y1'],ci['x2'],ci['y2']]].astype(np.float32);ov=iou(tb,db);ri,cj=linear_sum_assignment(-ov)
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

def chunks(rows,phase,miou,emb,pos,gt):
 by=defaultdict(list)
 for i,r in enumerate(rows):by[r['track_id']].append(i)
 out=[];cid=0
 for tid,ids in sorted(by.items()):
  ids.sort(key=lambda i:(rows[i]['frame'],i));start=0
  for k in range(1,len(ids)+1):
   br=k==len(ids) or rows[ids[k]]['frame']-rows[ids[k-1]]['frame']>GAP_BREAK or rows[ids[k]]['frame']-rows[ids[start]]['frame']>=CHUNK_SPAN
   if not br:continue
   q=ids[start:k];mapped=[i for i in q if phase[i]>=0];proto=np.zeros(DIM,np.float32)
   if mapped:
    proto=emb[pos[mapped]].mean(axis=0);proto/=max(np.linalg.norm(proto),1e-12)
   c=Counter(int(gt[i]) for i in q if int(gt[i])>0);modal,count=(c.most_common(1)[0] if c else (0,0));first,last=rows[q[0]],rows[q[-1]]
   out.append({'chunk_id':cid,'source_track_id':tid,'source_chunk_index':start,'first_frame':first['frame'],'last_frame':last['frame'],'rows':len(q),'line_indices':q,'mapped_rows':len(mapped),'mapping_rate':len(mapped)/len(q),'mean_match_iou':float(np.mean(miou[mapped])) if mapped else 0.0,'modal_gt':modal,'modal_count':count,'modal_purity':count/max(sum(c.values()),1),'proto':proto,'first_cx':.5*(first['x1']+first['x2']),'first_cy':.5*(first['y1']+first['y2']),'last_cx':.5*(last['x1']+last['x2']),'last_cy':.5*(last['y1']+last['y2']),'first_h':first['y2']-first['y1'],'last_h':last['y2']-last['y1']});cid+=1;start=k
 return out

def wis(items):
 x=sorted(items,key=lambda r:(r['last_frame'],r['first_frame'],r['chunk_id']));ends=[r['last_frame'] for r in x];pred=[bisect.bisect_left(ends,r['first_frame'])-1 for r in x];dp=[(0,0,0,())]
 for i,r in enumerate(x,1):
  b=dp[pred[i-1]+1];take=(b[0]+r['modal_count'],b[1]+r['rows'],b[2]-1,b[3]+(i-1,));skip=dp[i-1];dp.append(take if take[:3]>skip[:3] else skip)
 return [x[i] for i in dp[-1][3]]

def main():
 OUT.mkdir(parents=True,exist_ok=True);td=OUT/'track_results';td.mkdir(parents=True,exist_ok=True);m23=load_m23();rng=np.random.default_rng(SEED);proj=rng.normal(size=(2048,DIM)).astype(np.float32)/math.sqrt(DIM);allmeta=[];summary=[]
 for si,seq in enumerate(SEQS):
  rows=read_tracker(PARENT/f'{seq}.txt');phase,miou,emb,pos=map_phase(rows,seq,proj);gt=gt_labels(rows,seq,m23);cs=chunks(rows,phase,miou,emb,pos,gt);groups=defaultdict(list)
  for c in cs:
   if c['modal_gt']>0:groups[c['modal_gt']].append(c)
  selected=[]
  for gid,x in groups.items():selected+=wis(x)
  selids={c['chunk_id'] for c in selected};line_id={};overlap_pairs=0
  for c in selected:
   nid=SYN_BASE+si*STRIDE+c['modal_gt']
   for li in c['line_indices']:line_id[li]=nid
  out=[];seen=set();changed=0
  for r in rows:
   q=list(r['fields']);new=line_id.get(r['line'],r['track_id']);changed+=new!=r['track_id'];q[1]=str(new);key=(r['frame'],new)
   if key in seen:raise RuntimeError(f'{seq}: duplicate {key}')
   seen.add(key);out.append(q)
  out.sort(key=lambda p:(int(float(p[0])),int(float(p[1])),float(p[2]),float(p[3])))
  with (td/f'{seq}.txt').open('w') as f:
   for p in out:f.write(','.join(p)+'\n')
  for c in cs:
   allmeta.append({k:v for k,v in c.items() if k not in ['line_indices','proto']}|{'seq':seq,'selected_oracle_chain':c['chunk_id'] in selids})
  rec={'seq':seq,'rows':len(rows),'mapped_rows':int((phase>=0).sum()),'chunks':len(cs),'positive_chunks':sum(c['modal_gt']>0 for c in cs),'selected_chunks':len(selected),'selected_modal_rows':sum(c['modal_count'] for c in selected),'all_modal_rows':sum(c['modal_count'] for c in cs if c['modal_gt']>0),'selected_recall':sum(c['modal_count'] for c in selected)/max(sum(c['modal_count'] for c in cs if c['modal_gt']>0),1),'changed_rows':changed};summary.append(rec);print(json.dumps(rec),flush=True)
 pd.DataFrame(allmeta).to_csv(OUT/'microtracklets.csv',index=False)
 name='microtracklet_oracle_chunk30';cmd=[sys.executable,'scripts/eval_motstyle_trackeval.py','--benchmark-name','MOT20','--split-to-eval','train','--gt-root','datasets/MOT20/train','--results-dir',str(td),'--tracker-name',name,'--work-dir',str(OUT/'eval_work'),'--seqs',*SEQS];p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(OUT/'eval.log').write_text(p.stdout)
 if p.returncode:raise RuntimeError(p.stdout[-4000:])
 rr=list(csv.DictReader((OUT/'eval_work/eval'/name/'pedestrian_detailed.csv').open()));res={}
 for s in SEQS+['COMBINED']:
  r=next(x for x in rr if x['seq']==s);res[s]={'HOTA':100*float(r['HOTA___AUC']),'DetA':100*float(r['DetA___AUC']),'AssA':100*float(r['AssA___AUC']),'IDSW':int(float(r['IDSW']))}
 report={'oracle':True,'deployment_allowed':False,'protocol':{'chunk_span_frames':CHUNK_SPAN,'gap_break':GAP_BREAK,'mapping_iou':IOU_THR,'projected_dim':DIM,'projection_seed':SEED,'chain':'per-GT WIS maximizing modal matches; preserve rows and boxes'},'summary':summary,'eval':res};(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
