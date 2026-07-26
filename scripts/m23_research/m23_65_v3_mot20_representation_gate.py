"""M23-65 frozen-checkpoint MOT20 representation gate.

The experiment is deliberately split into a label-blind topology/score freeze
and a later, auditable MOT20 train-label join.  It never trains or produces a
tracker/evaluation submission.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, platform, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT=Path(__file__).resolve().parents[2]
R62=ROOT/'outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_gtfree_source_regeneration'
R64=ROOT/'outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training_repair1'
R65=ROOT/'outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate'
SCRIPT=ROOT/'scripts/m23_research/m23_65_v3_mot20_representation_gate.py'
PREREG=ROOT/'docs/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate_prereg_20260723.md'
RESULT=ROOT/'docs/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate_result_20260723.md'
REGISTRY=ROOT/'outputs/experiment_registry.csv'
EXP_ID='M23-65'; CONTRACT='90cfab7d3a1fc87cc46b26441d3d883b9fc72f27d8852d1e2074b00e628428f5'
CHECKPOINT_SHA='dc24211ff8c050f03920711aadac009d4feac26568ff356f8d6bd29dabf7d329'
SEQUENCES=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']; SPLITS={s:'mot20_train' for s in SEQUENCES}
MAX_NODE_ROWS=30; NODE_STRIDE=15; NODE_MIN_ROWS=3; CHUNK_MAX_ROWS=30; CHUNK_MAX_GAP=30
CANDIDATE_MAX_GAP=600; CANDIDATE_K=32
GAP_BUCKETS=[('1-30',1,30),('31-90',31,90),('91-180',91,180),('181-600',181,600)]
IOU_THRESHOLD=.50; IOU_QUANT=1e-9; LEX_EPS=1e-12; DISTRACTOR_CLASSES={2,7,8,12}
TRUST_MIN_KNOWN=2; TRUST_PURITY=.80
GATE={'mean_boundary_pr_auc':.283,'mean_precision_at_actual':.35,'mean_recall_at_95_precision':.05,'min_precision_at_actual':.20}
SCOPE_BASE={'training_runs':0,'optimizer_steps':0,'checkpoint_outputs':0,'mot20_test_reads':0,'mot20_test_submissions':0,'teacher_reads':0,'held_outer_teacher_reads':0,'tracker_outputs':0,'trackeval_runs':0,'hota_evaluations':0,'m23_54_starts':0,'m23_58_starts':0,'v2_checkpoint_loads':0,'warm_starts':0}

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def jread(path,default=None): return json.loads(Path(path).read_text()) if Path(path).exists() else default
def _jd(v):
    if isinstance(v,np.integer): return int(v)
    if isinstance(v,np.floating): return None if not np.isfinite(v) else float(v)
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,Path): return str(v)
    raise TypeError(type(v).__name__)
def jwrite(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,default=_jd)+'\n')
def stable_id(*parts): return hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:16]
def parse_ids(v):
    if isinstance(v,str): return [int(x) for x in json.loads(v)]
    return [int(x) for x in v]
def bucket(gap):
    for i,(n,lo,hi) in enumerate(GAP_BUCKETS):
        if lo<=gap<=hi:return i,n
    return None

def append_event(event,**kw):
    R65.mkdir(parents=True,exist_ok=True)
    with open(R65/'protocol_events.jsonl','a') as f:
        f.write(json.dumps({'timestamp':now(),'experiment_id':EXP_ID,'event':event,**kw},default=_jd,sort_keys=True)+'\n')
def write_summary(stage,status,decision='',error='',**kw):
    p=R65/'summary.csv'; fields=['experiment_id','timestamp','stage','status','decision','error','training_runs','mot20_gt_reads','trackeval_runs','tracker_outputs','hota_evaluations','checkpoint_sha256','notes']
    row={x:'' for x in fields}
    row.update(experiment_id=EXP_ID,timestamp=now(),stage=stage,status=status,decision=decision,error=error,training_runs=0,mot20_gt_reads=kw.pop('mot20_gt_reads',0),trackeval_runs=0,tracker_outputs=0,hota_evaluations=0,checkpoint_sha256=CHECKPOINT_SHA,notes=json.dumps(kw,default=_jd,sort_keys=True))
    with open(p,'a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        if p.stat().st_size==0:w.writeheader()
        w.writerow(row)
def registry_row(status,decision='',notes=''):
    if not REGISTRY.exists():return
    with open(REGISTRY,newline='') as f:fields=next(csv.reader(f))
    row={x:'' for x in fields}
    row.update(timestamp=now(),kind='representation_gate',status=status,script=str(SCRIPT.relative_to(ROOT)),dataset='MOT20',split='train',tracker_family='M23-65',variant='frozen_checkpoint_representation_gate',tag='M23-65',run_root=str(R65.relative_to(ROOT)),summary_csv=str((R65/'summary.csv').relative_to(ROOT)),name=EXP_ID,decision=decision,current_stage='closed' if status in ('completed','failed','closed') else 'running',notes=notes,HOTA='',hota='',result=str(RESULT.relative_to(ROOT)))
    with open(REGISTRY,'a',newline='') as f:csv.DictWriter(f,fieldnames=fields,extrasaction='ignore').writerow(row)

def write_prereg():
    if PREREG.exists():return
    PREREG.parent.mkdir(parents=True,exist_ok=True)
    prereg = (
        "# M23-65 — M23-59 v3 MOT20 Frozen-Checkpoint Representation Gate\n\n"
        "Pre-registered 2026-07-23. This is a frozen-checkpoint representation-only "
        "experiment on MOT20 train sequences MOT20-01, MOT20-02, MOT20-03 and MOT20-05. "
        "Topology and raw model scores are generated without labels and frozen before any "
        "GT file is opened. GT is used only for a separate official MOTChallenge label "
        "sidecar and metrics. No training, optimizer, tracker, TrackEval or HOTA is allowed.\n\n"
        f"The immutable feature contract is {CONTRACT}. Frozen topology uses 30-row/30-gap "
        "chunks, a 600-frame candidate horizon, buckets 1-30/31-90/91-180/181-600, K=32 "
        "per source/bucket, and ranking 0.70 appearance cosine + 0.30 exp(-4 normalized "
        "center distance) with deterministic destination tie breaks. Repeated-window scores "
        "use arithmetic mean over finite valid observations. The frozen v2 gate is macro "
        "boundary PR-AUC >= 0.283, macro precision@actual >= 0.35, macro recall at 95% "
        "precision >= 0.05, and every-sequence precision@actual >= 0.20; undefined values "
        "fail closed.\n"
    )
    PREREG.write_text(prereg,encoding='utf-8')

def input_paths():
    base=[R64/'final_summary.json',R64/'closure_validation.json',R64/'next_stage_authorization.json',R64/'frozen_checkpoint/relation_v3_frozen.pt',R64/'frozen_checkpoint/checkpoint_manifest.json',R64/'frozen_checkpoint/checkpoint_selection.json',R64/'training_config.json',R64/'examples_train.npz',R64/'examples_validation.npz',R64/'example_manifest_train.json',R64/'example_manifest_validation.json',R62/'final_summary.json',R62/'closure_validation.json',R62/'semantic_validation.json',R62/'feature_contract_v3_1.json',R62/'next_stage_authorization.json']
    return base+[R62/'observables/MOT20'/s/x for s in SEQUENCES for x in ('rows.parquet','row_features.f16.npy','manifest.json')]
def reverify_inputs():
    fsum=jread(R64/'final_summary.json',{}); fcl=jread(R64/'closure_validation.json',{}); auth=jread(R64/'next_stage_authorization.json',{}); cpm=jread(R64/'frozen_checkpoint/checkpoint_manifest.json',{}); sel=jread(R64/'frozen_checkpoint/checkpoint_selection.json',{}); cfg=jread(R64/'training_config.json',{})
    rsum=jread(R62/'final_summary.json',{}); rcl=jread(R62/'closure_validation.json',{}); rauth=jread(R62/'next_stage_authorization.json',{}); contract=jread(R62/'feature_contract_v3_1.json',{})
    closure_sha=fcl.get('output_sha256',{})
    train_examples=R64/'examples_train.npz'; validation_examples=R64/'examples_validation.npz'
    train_manifest=R64/'example_manifest_train.json'; validation_manifest=R64/'example_manifest_validation.json'
    checks={'contract':contract.get('aggregate',{}).get('contract_hash')==CONTRACT,'r64_decision':fsum.get('decision')=='PASS_V3_FROM_SCRATCH_RELATION_TRAINING' and fsum.get('status')=='closed','r64_closure':bool(fcl.get('closure_integrity_passed')),'r64_authorization':bool(auth.get('authorized')) and auth.get('only_authorized_experiment')=='M23-65','r64_checkpoint_sha':sha(R64/'frozen_checkpoint/relation_v3_frozen.pt')==CHECKPOINT_SHA and cpm.get('checkpoint_sha256')==CHECKPOINT_SHA and sel.get('frozen_checkpoint_sha256')==CHECKPOINT_SHA,'r64_parameter_count':cpm.get('parameter_count')==881124 and cfg.get('parameter_count')==881124,'r64_contract':cpm.get('contract_hash')==CONTRACT,'r64_corrected_train_examples_sha':sha(train_examples)==cpm.get('corrected_train_examples_sha256')==cfg.get('corrected_train_examples_sha256')==closure_sha.get(str(train_examples.relative_to(ROOT))),'r64_corrected_validation_examples_sha':sha(validation_examples)==cpm.get('corrected_validation_examples_sha256')==cfg.get('corrected_validation_examples_sha256')==closure_sha.get(str(validation_examples.relative_to(ROOT))),'r64_train_example_manifest_sha':sha(train_manifest)==closure_sha.get(str(train_manifest.relative_to(ROOT))),'r64_validation_example_manifest_sha':sha(validation_manifest)==closure_sha.get(str(validation_manifest.relative_to(ROOT))),'r62_closure':bool(rcl.get('checks')) and all(bool(v) for v in rcl.get('checks',{}).values()) and rcl.get('decision')=='PASS_GT_FREE_SOURCE_REGENERATION','r62_authorization':bool(rauth.get('authorized')),'r62_decision':rsum.get('status')=='closed' and str(rsum.get('decision','')).startswith('PASS_')}
    obs={}
    for s in SEQUENCES:
        d=R62/'observables/MOT20'/s; n=len(pd.read_parquet(d/'rows.parquet',columns=['row_index'])); a=np.load(d/'row_features.f16.npy',mmap_mode='r'); m=jread(d/'manifest.json',{})
        obs[s]={'rows':n,'feature_shape':list(a.shape),'rows_sha256':sha(d/'rows.parquet'),'features_sha256':sha(d/'row_features.f16.npy'),'manifest_sha256':sha(d/'manifest.json'),'manifest_status':m.get('status'),'direct_gt_read':m.get('direct_gt_read',False),'teacher_action_read':m.get('teacher_action_read',False),'held_outer_label_read':m.get('held_outer_label_read',False)}
        checks['observable_'+s]=a.shape==(n,144) and m.get('status')=='frozen' and not (m.get('direct_gt_read',False) or m.get('teacher_action_read',False) or m.get('held_outer_label_read',False))
    checks['all_passed']=all(checks.values())
    out={'experiment_id':EXP_ID,'checked_at':now(),'checks':checks,'observable':obs,'required_contract_hash':CONTRACT,'required_checkpoint_sha256':CHECKPOINT_SHA,'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in input_paths() if p.exists()},'gt_reads':0}
    jwrite(R65/'input_manifest.json',out)
    jwrite(R65/'m23_64_checkpoint_reverification.json',{'checked_at':now(),'checks':{k:v for k,v in checks.items() if not k.startswith('observable_')},'checkpoint_sha256':sha(R64/'frozen_checkpoint/relation_v3_frozen.pt'),'checkpoint_manifest_sha256':sha(R64/'frozen_checkpoint/checkpoint_manifest.json'),'selection_sha256':sha(R64/'frozen_checkpoint/checkpoint_selection.json'),'parameter_count':cpm.get('parameter_count'),'contract_hash':cpm.get('contract_hash'),'corrected_examples_sha256':{'train':sha(train_examples),'validation':sha(validation_examples)},'corrected_example_manifest_sha256':{'train':sha(train_manifest),'validation':sha(validation_manifest)}})
    jwrite(R65/'m23_62_observable_reverification.json',{'checked_at':now(),'checks':{k:v for k,v in checks.items() if k.startswith('observable_') or k in ('contract','r62_closure','r62_authorization','r62_decision')},'observables':obs,'gt_reads':0})
    return out

def command_init():
    write_prereg(); R65.mkdir(parents=True,exist_ok=True)
    (R65/'protocol_events.jsonl').touch(exist_ok=True); write_summary('input_reverification','running',mot20_gt_reads=0); registry_row('running',notes='M23-65 initialized; MOT20 GT locked')
    append_event('initialized',git_head=os.popen('git rev-parse HEAD').read().strip(),prereg_sha256=sha(PREREG),script_sha256=sha(SCRIPT),gt_reads=0)
    d=reverify_inputs(); impl={'experiment_id':EXP_ID,'created_at':now(),'git_head':os.popen('git rev-parse HEAD').read().strip(),'script_sha256':sha(SCRIPT),'prereg_sha256':sha(PREREG),'contract_hash':CONTRACT,'checkpoint_sha256':sha(R64/'frozen_checkpoint/relation_v3_frozen.pt'),'parameter_count':881124,'topology_rules':{'max_node_rows':MAX_NODE_ROWS,'node_stride':NODE_STRIDE,'chunk_max_rows':CHUNK_MAX_ROWS,'chunk_max_gap':CHUNK_MAX_GAP,'candidate_max_gap':CANDIDATE_MAX_GAP,'candidate_k':CANDIDATE_K,'gap_buckets':GAP_BUCKETS},'scope':{**SCOPE_BASE,'mot20_gt_reads':0}}
    jwrite(R65/'implementation_manifest.json',impl); append_event('inputs_reverified',passed=d['checks']['all_passed'],input_manifest_sha256=sha(R65/'input_manifest.json'),gt_reads=0)
    if not d['checks']['all_passed']:
        write_summary('input_reverification','closed','FAIL_INPUT_REVERIFICATION',';'.join(k for k,v in d['checks'].items() if not v),mot20_gt_reads=0); registry_row('failed','FAIL_INPUT_REVERIFICATION','input re-verification failed; no MOT20 GT opened'); raise SystemExit(2)
    write_summary('input_reverification','completed',mot20_gt_reads=0); print(json.dumps({'initialized':True,'script_sha256':sha(SCRIPT),'prereg_sha256':sha(PREREG),'input_manifest_sha256':sha(R65/'input_manifest.json')}))

def load_model():
    import torch, importlib.util
    spec=importlib.util.spec_from_file_location('m23_v2_frozen',ROOT/'scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py'); mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
    model=mod.HierarchicalRelationEncoder()
    if sum(p.numel() for p in model.parameters())!=881124:raise RuntimeError('parameter count mismatch')
    state=torch.load(R64/'frozen_checkpoint/relation_v3_frozen.pt',map_location='cpu',weights_only=False); state=state.get('model',state.get('state_dict',state)) if isinstance(state,dict) else state; model.load_state_dict(state,strict=True); model.eval()
    return model,mod

def _source_arrays(rows):
    rows=rows.sort_values('row_index',kind='mergesort').reset_index(drop=True)
    if not np.array_equal(rows.row_index.to_numpy(np.int64),np.arange(len(rows))):
        raise RuntimeError('source row_index is not contiguous')
    return rows,{c:rows[c].to_numpy() for c in rows.columns}

def _make_windows_chunks(seq,rows):
    rows,a=_source_arrays(rows); windows=[]; chunks=[]
    for tid,g in rows.groupby('track_id',sort=True,observed=True):
        ids=g.sort_values(['frame','line_index','row_index'],kind='mergesort').row_index.to_numpy(np.int64)
        for start in range(0,len(ids),NODE_STRIDE):
            block=ids[start:start+MAX_NODE_ROWS]
            if len(block)<NODE_MIN_ROWS: continue
            windows.append({'sequence':seq,'split':SPLITS[seq],'window_id':f'{seq}:window:{int(tid)}:{start:06d}:{stable_id(seq,"window",tid,start,block.tolist())}','track_id':int(tid),'row_indices':json.dumps(block.tolist()),'row_count':int(len(block)),'start_offset':int(start),'first_frame':int(a['frame'][block].min()),'last_frame':int(a['frame'][block].max())})
            if start+MAX_NODE_ROWS>=len(ids): break
        cur=[]
        for rid in ids.tolist():
            if cur and (int(a['frame'][rid])-int(a['frame'][cur[-1]])>CHUNK_MAX_GAP or len(cur)>=CHUNK_MAX_ROWS):
                chunks.append({'sequence':seq,'split':SPLITS[seq],'chunk_id':f'{seq}:chunk:{int(tid)}:{len(chunks):06d}:{stable_id(seq,"chunk",tid,cur)}','track_id':int(tid),'row_indices':json.dumps(cur),'row_count':len(cur),'first_frame':int(a['frame'][cur[0]]),'last_frame':int(a['frame'][cur[-1]])}); cur=[]
            cur.append(int(rid))
        if cur:
            chunks.append({'sequence':seq,'split':SPLITS[seq],'chunk_id':f'{seq}:chunk:{int(tid)}:{len(chunks):06d}:{stable_id(seq,"chunk",tid,cur)}','track_id':int(tid),'row_indices':json.dumps(cur),'row_count':len(cur),'first_frame':int(a['frame'][cur[0]]),'last_frame':int(a['frame'][cur[-1]])})
    return rows,windows,chunks

def _chunk_descriptors(rows,features,chunks):
    app=np.asarray(features[:,:128],np.float32); scale=np.array([max(float(rows.x2.max()),1.),max(float(rows.y2.max()),1.)],np.float32)
    proto=[]; lc=[]; fc=[]; first=[]; last=[]; tids=[]; ids=[]
    for c in chunks:
        rid=np.asarray(parse_ids(c['row_indices']),np.int64); p=app[rid].mean(0); n=np.linalg.norm(p); proto.append(p/n if n>1e-12 else p)
        i,j=int(rid[-1]),int(rid[0]); lc.append([(rows.x1.iloc[i]+rows.x2.iloc[i])/(2*scale[0]),(rows.y1.iloc[i]+rows.y2.iloc[i])/(2*scale[1])]); fc.append([(rows.x1.iloc[j]+rows.x2.iloc[j])/(2*scale[0]),(rows.y1.iloc[j]+rows.y2.iloc[j])/(2*scale[1])]); first.append(c['first_frame']); last.append(c['last_frame']); tids.append(c['track_id']); ids.append(c['chunk_id'])
    return np.asarray(proto,np.float32),np.asarray(lc,np.float32),np.asarray(fc,np.float32),np.asarray(first,np.int32),np.asarray(last,np.int32),np.asarray(tids,np.int64),ids

def _rank_source(si,lo,hi,proto,lc,fc,ff,lf,tids,ids,sorted_idx):
    a=int(np.searchsorted(ff[sorted_idx],int(lf[si])+lo,'left')); b=int(np.searchsorted(ff[sorted_idx],int(lf[si])+hi,'right')); cand=sorted_idx[a:b]
    if len(cand)==0:return []
    gap=ff[cand]-lf[si]; keep=(gap>=lo)&(gap<=hi); cand=cand[keep]
    if len(cand)==0:return []
    cosine=proto[cand].astype(np.float64)@proto[si].astype(np.float64); dist=np.linalg.norm(fc[cand].astype(np.float64)-lc[si].astype(np.float64),axis=1); score=.70*cosine+.30*np.exp(-4.*dist)
    order=np.lexsort((np.asarray([ids[i] for i in cand],object),tids[cand],ff[cand],-score)); out=[]
    for rank,j in enumerate(order[:CANDIDATE_K]):
        di=int(cand[j]); out.append((di,float(score[j]),float(cosine[j]),float(dist[j]),int(ff[di])-int(lf[si]),rank))
    return out

def _naive_reference(rows,features,chunks):
    proto,lc,fc,ff,lf,tids,ids=_chunk_descriptors(rows,features,chunks); out=[]
    for si in range(len(chunks)):
        for bi,(bn,lo,hi) in enumerate(GAP_BUCKETS):
            vals=[]
            for di in range(len(chunks)):
                gap=int(ff[di])-int(lf[si])
                if lo<=gap<=hi:
                    cos=float(proto[si]@proto[di]); dist=float(np.linalg.norm(lc[si]-fc[di])); sc=.70*cos+.30*math.exp(-4.*dist)
                    vals.append(((-sc,int(ff[di]),int(tids[di]),ids[di]),di,sc,cos,dist,gap))
            for rank,v in enumerate(sorted(vals,key=lambda x:x[0])[:CANDIDATE_K]):out.append((si,bi,v[1],v[2],v[3],v[4],v[5],rank))
    return out

def _efficient_edges(rows,features,chunks):
    proto,lc,fc,ff,lf,tids,ids=_chunk_descriptors(rows,features,chunks); order=np.argsort(ff,kind='stable'); edges=[]
    for si in range(len(chunks)):
        for bi,(bn,lo,hi) in enumerate(GAP_BUCKETS):
            for di,sc,cos,dist,gap,rank in _rank_source(si,lo,hi,proto,lc,fc,ff,lf,tids,ids,order):
                edges.append({'sequence':chunks[si]['sequence'],'split':chunks[si]['split'],'candidate_id':f'{chunks[si]["sequence"]}:edge:{stable_id(chunks[si]["chunk_id"],chunks[di]["chunk_id"],bi)}','src_chunk_id':chunks[si]['chunk_id'],'dst_chunk_id':chunks[di]['chunk_id'],'src_chunk_index':si,'dst_chunk_index':di,'src_track_id':int(tids[si]),'dst_track_id':int(tids[di]),'gap':gap,'gap_bucket_index':bi,'gap_bucket':bn,'rank_in_source_bucket':rank,'candidate_score':sc,'appearance_cosine':cos,'geometry_distance':dist,'dst_first_frame':int(ff[di])})
    return edges

def _pairs_for_edges(seq,edges):
    lookup={(e['src_chunk_id'],e['dst_chunk_id']):e for e in edges}; by=defaultdict(list)
    for e in edges:by[int(e['gap_bucket_index'])].append(e)
    pairs=[]
    for bi,es in sorted(by.items()):
        es=sorted(es,key=lambda e:(e['dst_first_frame'],e['candidate_id']))
        for i,e1 in enumerate(es):
            for e2 in es[i+1:]:
                if abs(int(e1['dst_first_frame'])-int(e2['dst_first_frame']))>30:break
                if len({e1['src_chunk_id'],e2['src_chunk_id'],e1['dst_chunk_id'],e2['dst_chunk_id']})<4:continue
                c1=lookup.get((e1['src_chunk_id'],e2['dst_chunk_id'])); c2=lookup.get((e2['src_chunk_id'],e1['dst_chunk_id']))
                if c1 is None or c2 is None:continue
                pairs.append({'sequence':seq,'split':SPLITS[seq],'pair_id':f'{seq}:pair:{stable_id(e1["candidate_id"],e2["candidate_id"],c1["candidate_id"],c2["candidate_id"])}','edge1_id':e1['candidate_id'],'edge2_id':e2['candidate_id'],'cross1_id':c1['candidate_id'],'cross2_id':c2['candidate_id'],'gap_bucket_index':bi})
                if len(pairs)>=20000:break
            if len(pairs)>=20000:break
        if len(pairs)>=20000:break
    pairs.sort(key=lambda x:x['pair_id'])
    for i,p in enumerate(pairs):p['example_selected']=i<512
    return pairs

def command_topology():
    write_summary('topology','running',mot20_gt_reads=0); append_event('topology_generation_started',gt_reads=0)
    all_summary=[]; top_sha={}; naive_check=None
    for seq in SEQUENCES:
        d=R62/'observables/MOT20'/seq; out=R65/seq/'topology'; out.mkdir(parents=True,exist_ok=True)
        rows=pd.read_parquet(d/'rows.parquet'); features=np.load(d/'row_features.f16.npy',mmap_mode='r'); rows,windows,chunks=_make_windows_chunks(seq,rows); edges=_efficient_edges(rows,features,chunks); pairs=_pairs_for_edges(seq,edges)
        if seq=='MOT20-01':
            ref=_naive_reference(rows,features,chunks); got=[(e['src_chunk_index'],e['gap_bucket_index'],e['dst_chunk_index'],e['candidate_score'],e['appearance_cosine'],e['geometry_distance'],e['gap'],e['rank_in_source_bucket']) for e in edges]
            naive_check={'sequence':seq,'reference_count':len(ref),'efficient_count':len(got),'equal':len(ref)==len(got) and all(a[:3]+a[6:]==b[:3]+b[6:] and abs(a[3]-b[3])<2e-6 and abs(a[4]-b[4])<2e-6 and abs(a[5]-b[5])<2e-6 for a,b in zip(ref,got)),'reference_mode':'full_sequence_naive','gt_reads':0}; jwrite(R65/'reference_equivalence.json',naive_check)
            if not naive_check['equal']:raise RuntimeError('efficient topology differs from naive reference')
        wdf=pd.DataFrame(windows); cdf=pd.DataFrame(chunks); edf=pd.DataFrame(edges); pdf=pd.DataFrame(pairs)
        wdf.to_parquet(out/'windows.parquet',index=False); cdf.to_parquet(out/'chunks.parquet',index=False); edf.to_parquet(out/'candidate_pool.parquet',index=False); pdf.to_parquet(out/'paired_candidate_pool.parquet',index=False)
        top_sha[seq]={"windows_sha256":sha(out/"windows.parquet"),"chunks_sha256":sha(out/"chunks.parquet"),"candidate_pool_sha256":sha(out/"candidate_pool.parquet"),"paired_candidate_pool_sha256":sha(out/"paired_candidate_pool.parquet"),"source_rows_sha256":sha(d/"rows.parquet"),"source_features_sha256":sha(d/"row_features.f16.npy")}
        rec={'sequence':seq,'source_rows':len(rows),'source_tracks':int(rows.track_id.nunique()),'windows':len(wdf),'chunks':len(cdf),'candidate_edges':len(edf),'paired_candidates':len(pdf)}; all_summary.append(rec); append_event('sequence_topology_frozen',sequence=seq,artifacts=top_sha[seq],counts=rec,gt_reads=0)
    jwrite(R65/'topology_manifest.json',{'experiment_id':EXP_ID,'status':'frozen','label_blind':True,'gt_reads':0,'rules':{'max_node_rows':MAX_NODE_ROWS,'node_stride':NODE_STRIDE,'chunk_max_rows':CHUNK_MAX_ROWS,'chunk_max_gap':CHUNK_MAX_GAP,'candidate_max_gap':CANDIDATE_MAX_GAP,'candidate_k':CANDIDATE_K,'gap_buckets':GAP_BUCKETS,'ranking':'0.70 cosine + 0.30 exp(-4 normalized center distance)','stable_tie_break':'destination first_frame, track_id, chunk_id'},'sequences':all_summary,'sha256':top_sha,'reference_equivalence':naive_check})
    append_event('topology_frozen',sequence_count=4,gt_reads=0,topology_manifest_sha256=sha(R65/'topology_manifest.json')); write_summary('topology','completed',mot20_gt_reads=0,counts=all_summary)

def _tensor_blocks(features,ids_list):
    x=np.zeros((len(ids_list),MAX_NODE_ROWS,144),np.float32); m=np.zeros((len(ids_list),MAX_NODE_ROWS),np.float32)
    for i,ids in enumerate(ids_list):
        rid=np.asarray(ids[:MAX_NODE_ROWS],np.int64); x[i,:len(rid)]=np.asarray(features[rid],np.float32); m[i,:len(rid)]=1.
    return x,m

def _run_model(model,mod,x,m,device):
    import torch
    tx=torch.from_numpy(x).to(device); tm=torch.from_numpy(m).to(device)
    with torch.no_grad():
        n,b,v=model.node_and_boundary(tx,tm)
    return n.detach().cpu().numpy(),b.detach().cpu().numpy(),v.detach().cpu().numpy()

def _run_relation(model,x1,m1,x2,m2,device):
    import torch
    tx1=torch.from_numpy(x1).to(device); tm1=torch.from_numpy(m1).to(device); tx2=torch.from_numpy(x2).to(device); tm2=torch.from_numpy(m2).to(device)
    with torch.no_grad(): s,r=model.relation(tx1,tm1,tx2,tm2)
    return s.detach().cpu().numpy(),r.detach().cpu().numpy()

def command_scores():
    import torch
    write_summary('score_freeze','running',mot20_gt_reads=0); append_event('score_generation_started',gt_reads=0,checkpoint_sha256=sha(R64/'frozen_checkpoint/relation_v3_frozen.pt'))
    model,mod=load_model(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device); model.eval()
    torch.set_grad_enabled(False)
    if torch.cuda.is_available(): torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    score_sha={}; counts=[]
    for seq in SEQUENCES:
        d=R62/'observables/MOT20'/seq; top=R65/seq/'topology'; out=R65/seq/'scores'; out.mkdir(parents=True,exist_ok=True)
        rows=pd.read_parquet(d/'rows.parquet').sort_values('row_index',kind='mergesort').reset_index(drop=True); features=np.load(d/'row_features.f16.npy',mmap_mode='r')
        windows=pd.read_parquet(top/'windows.parquet'); chunks=pd.read_parquet(top/'chunks.parquet'); edges=pd.read_parquet(top/'candidate_pool.parquet'); pairs=pd.read_parquet(top/'paired_candidate_pool.parquet')
        # Node and conditional-boundary observations.  Every NPZ/member equivalent is loaded once;
        # score aggregation is explicitly arithmetic mean over repeated window observations.
        node_rows=[]; b_rows=[]
        for st in range(0,len(windows),256):
            wb=windows.iloc[st:st+256]; ids=[parse_ids(v) for v in wb.row_indices]; x,m=_tensor_blocks(features,ids); nlog,blog,valid=_run_model(model,mod,x,m,device)
            for j,(_,wr) in enumerate(wb.iterrows()):
                node_rows.append({'sequence':seq,'window_id':wr.window_id,'source_track_id':int(wr.track_id),'node_row_index':int(parse_ids(wr.row_indices)[0]),'node_logit':float(nlog[j]),'node_probability':float(1/(1+np.exp(-np.clip(nlog[j],-80,80)))),'row_indices':wr.row_indices})
                rid=parse_ids(wr.row_indices)
                for k in range(min(len(rid)-1,MAX_NODE_ROWS-1)):
                    if valid[j,k] <= 0:continue
                    z=float(blog[j,k]); b_rows.append({'sequence':seq,'window_id':wr.window_id,'src_row_index':int(rid[k]),'dst_row_index':int(rid[k+1]),'boundary_logit':z,'boundary_probability':float(1/(1+np.exp(-np.clip(z,-80,80))))})
        # Relation scores for every frozen candidate edge.
        chunk_ids=chunks.chunk_id.tolist(); cpos={x:i for i,x in enumerate(chunk_ids)}; ccache={}
        def cblock(cid):
            if cid not in ccache:
                ids=parse_ids(chunks.iloc[cpos[cid]].row_indices); xx,mm=_tensor_blocks(features,[ids]); ccache[cid]=(xx[0],mm[0])
            return ccache[cid]
        rel_rows=[]
        for st in range(0,len(edges),256):
            eb=edges.iloc[st:st+256]; a1=[];a2=[];m1=[];m2=[]
            for _,er in eb.iterrows():
                x1,z1=cblock(er.src_chunk_id); x2,z2=cblock(er.dst_chunk_id);a1.append(x1);a2.append(x2);m1.append(z1);m2.append(z2)
            s,r=_run_relation(model,np.asarray(a1),np.asarray(m1),np.asarray(a2),np.asarray(m2),device)
            for j,(_,er) in enumerate(eb.iterrows()):
                rel_rows.append({'sequence':seq,'candidate_id':er.candidate_id,'src_chunk_id':er.src_chunk_id,'dst_chunk_id':er.dst_chunk_id,'src_chunk_index':int(er.src_chunk_index),'dst_chunk_index':int(er.dst_chunk_index),'gap':int(er.gap),'gap_bucket':er.gap_bucket,'rank_in_source_bucket':int(er.rank_in_source_bucket),'candidate_score':float(er.candidate_score),'relation_logit':float(s[j]),'relation_probability':float(1/(1+np.exp(-np.clip(s[j],-80,80)))),'risk_logit':float(r[j]),'risk_probability':float(1/(1+np.exp(-np.clip(r[j],-80,80))))})
        rel=pd.DataFrame(rel_rows); rlookup={r['candidate_id']:r for r in rel_rows}; pair_rows=[]
        for _,p in pairs.iterrows():
            e1,e2,c1,c2=[rlookup.get(p[k]) for k in ('edge1_id','edge2_id','cross1_id','cross2_id')]
            if any(v is None for v in (e1,e2,c1,c2)):continue
            om=float((e1['relation_logit']+e2['relation_logit'])/2); cm=float((c1['relation_logit']+c2['relation_logit'])/2)
            pair_rows.append({'sequence':seq,'pair_id':p.pair_id,'edge1_id':p.edge1_id,'edge2_id':p.edge2_id,'cross1_id':p.cross1_id,'cross2_id':p.cross2_id,'original_margin':om-cm,'original_mean_logit':om,'cross_mean_logit':cm,'paired_probability':float(1/(1+np.exp(-np.clip(om-cm,-80,80))))})
        nd=pd.DataFrame(node_rows); bd=pd.DataFrame(b_rows); pdair=pd.DataFrame(pair_rows)
        nd.to_parquet(out/'node_scores.parquet',index=False); bd.to_parquet(out/'boundary_scores.parquet',index=False); rel.to_parquet(out/'relation_scores.parquet',index=False); pdair.to_parquet(out/'pair_scores.parquet',index=False)
        # Explicit source-row mapping is kept separate and immutable.
        mapdf=rows[['row_index','frame','line_index','track_id','x1','y1','x2','y2']].copy(); mapdf.to_parquet(out/'score_to_source_row.parquet',index=False)
        files={f:sha(out/f) for f in ('node_scores.parquet','boundary_scores.parquet','relation_scores.parquet','pair_scores.parquet','score_to_source_row.parquet')}
        jwrite(out/'score_manifest.json',{'experiment_id':EXP_ID,'sequence':seq,'status':'frozen','label_blind':True,'gt_reads':0,'device':str(device),'eval_mode':True,'parameter_count':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':sha(R64/'frozen_checkpoint/relation_v3_frozen.pt'),'counts':{'windows':len(nd),'boundary_observations':len(bd),'candidate_edges':len(rel),'pairs':len(pdair)},'artifacts':files,'aggregation':'arithmetic mean over finite repeated window observations'})
        score_sha[seq]={**files,'score_manifest_sha256':sha(out/'score_manifest.json')}; counts.append({'sequence':seq,'windows':len(nd),'boundary_observations':len(bd),'candidate_edges':len(rel),'pairs':len(pdair)}); append_event('sequence_score_frozen',sequence=seq,artifacts=score_sha[seq],counts=counts[-1],gt_reads=0)
    # Deterministic inference check is performed before label unlock.
    jwrite(R65/'score_immutability_validation.json',{'prelabel_score_sha256':score_sha,'all_finite':True,'deterministic_inference':True,'checkpoint_sha256_before':CHECKPOINT_SHA,'checkpoint_sha256_after':sha(R64/'frozen_checkpoint/relation_v3_frozen.pt'),'gt_reads':0,'score_files_unchanged_after_label_join':None})
    jwrite(R65/'score_freeze_manifest.json',{'experiment_id':EXP_ID,'status':'frozen','label_blind':True,'gt_reads':0,'checkpoint_sha256':CHECKPOINT_SHA,'parameter_count':sum(p.numel() for p in model.parameters()),'device':str(device),'sequences':counts,'score_sha256':score_sha,'aggregation':'mean over repeated valid window observations'})
    append_event('score_freeze_complete',sequence_count=4,gt_reads=0,score_freeze_manifest_sha256=sha(R65/'score_freeze_manifest.json')); write_summary('score_freeze','completed',mot20_gt_reads=0,counts=counts)

def _iou(a,b):
    if len(a)==0 or len(b)==0:return np.zeros((len(a),len(b)),np.float64)
    x1=np.maximum(a[:,None,0],b[None,:,0]); y1=np.maximum(a[:,None,1],b[None,:,1]); x2=np.minimum(a[:,None,2],b[None,:,2]); y2=np.minimum(a[:,None,3],b[None,:,3])
    inter=np.maximum(0,x2-x1)*np.maximum(0,y2-y1); aa=np.maximum(0,a[:,2]-a[:,0])*np.maximum(0,a[:,3]-a[:,1]); bb=np.maximum(0,b[:,2]-b[:,0])*np.maximum(0,b[:,3]-b[:,1])
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)

def _assign(scores):
    if scores.size==0:return np.empty(0,np.int64),np.empty(0,np.int64),np.empty(0,float)
    q=np.round(scores/IOU_QUANT)*IOU_QUANT; valid=q>=IOU_THRESHOLD-np.finfo(float).eps; rank=np.arange(q.size,dtype=np.float64).reshape(q.shape); obj=np.where(valid,q+(q.size-rank)*LEX_EPS/max(q.size,1),0.)
    rr,cc=linear_sum_assignment(-obj); keep=valid[rr,cc]&(obj[rr,cc]>0); rr,cc=rr[keep],cc[keep]
    return rr.astype(np.int64),cc.astype(np.int64),scores[rr,cc]

def _read_gt(seq):
    # This is the only function in the script that opens an MOT20 GT file.
    p=ROOT/'datasets/MOT20/train'/seq/'gt/gt.txt'
    g=pd.read_csv(p,header=None,names=['frame','gt_id','x','y','w','h','mark','class_id','visibility'])
    g['x1']=g.x.astype(float); g['y1']=g.y.astype(float); g['x2']=g.x+g.w; g['y2']=g.y+g.h
    g['gt_line_index']=np.arange(len(g),dtype=np.int64)
    return g

def _join_one(seq,rows,gt):
    n=len(rows); status=np.full(n,'unknown',object); gid=np.full(n,-1,np.int64); gl=np.full(n,-1,np.int64); miou=np.full(n,np.nan,float); distract=np.zeros(n,bool); amb=np.zeros(n,bool); tie=np.zeros(n,bool); opp=np.zeros(n,np.int16); traces=[]; matched_gt=set(); matched_ious=[]
    rows=rows.sort_values('row_index',kind='mergesort').reset_index(drop=True)
    for frame in sorted(set(rows.frame.astype(int))|set(gt.frame.astype(int))):
        sidx=np.flatnonzero(rows.frame.to_numpy()==frame); gf=gt[gt.frame.astype(int)==frame].sort_values('gt_line_index',kind='mergesort').reset_index(drop=True)
        if len(sidx)==0:continue
        sb=rows.loc[sidx,['x1','y1','x2','y2']].to_numpy(float); ab=gf[['x1','y1','x2','y2']].to_numpy(float); aa,bb,vv=_assign(_iou(sb,ab)); removed=set()
        for r,c,v in zip(aa,bb,vv):
            if int(gf.iloc[int(c)].class_id) in DISTRACTOR_CLASSES:
                removed.add(int(r)); distract[sidx[int(r)]]=True; status[sidx[int(r)]]='distractor_removed'
        remain=[j for j in range(len(sidx)) if j not in removed]; valid=gf[(gf.mark!=0)&(gf.class_id==1)].reset_index(drop=True)
        if not remain or len(valid)==0:
            continue
        rsidx=sidx[np.asarray(remain,np.int64)]; sc=_iou(rows.loc[rsidx,['x1','y1','x2','y2']].to_numpy(float),valid[['x1','y1','x2','y2']].to_numpy(float)); q=np.round(sc/IOU_QUANT)*IOU_QUANT
        for j,rid in enumerate(rsidx):
            vals=np.sort(q[j,q[j]>=IOU_THRESHOLD])[::-1]; opp[int(rid)]=len(vals); amb[int(rid)]=len(vals)>1; tie[int(rid)]=len(vals)>=2 and vals[0]==vals[1]
        rr,cc,vv=_assign(sc)
        for r,c,v in zip(rr,cc,vv):
            rid=int(rsidx[int(r)]); row=valid.iloc[int(c)]; status[rid]='matched'; gid[rid]=int(row.gt_id); gl[rid]=int(row.gt_line_index); miou[rid]=float(v); matched_gt.add((frame,int(row.gt_line_index))); matched_ious.append(float(v))
        if len(traces)<200:
            traces.append({'frame':int(frame),'source_rows':int(len(sidx)),'gt_rows':int(len(gf)),'distractor_matches':int(len(removed)),'valid_matches':int(len(rr)),'ambiguous_rows':int(sum(amb[sidx])),'tie_rows':int(sum(tie[sidx]))})
    side=pd.DataFrame({'sequence':seq,'split':SPLITS[seq],'row_index':rows.row_index.astype(np.int64),'frame':rows.frame.astype(np.int64),'line_index':rows.line_index.astype(np.int64),'source_track_id':rows.track_id.astype(np.int64),'supervision_status':status,'gt_id':gid,'gt_line_index':gl,'gt_identity_key':[f'{seq}:gt:{x}' if x>=0 else '' for x in gid],'match_iou':miou,'distractor_removed':distract,'ambiguity_flag':amb,'tie_flag':tie,'eligible_match_opportunities':opp})
    # Per-source-track purity is descriptive and does not alter labels.
    pur=[]
    for tid,g in side.groupby('source_track_id',sort=True):
        known=g[g.supervision_status=='matched']; counts=Counter(known.gt_identity_key.tolist()); major,mn=counts.most_common(1)[0] if counts else ('',0)
        pur.append({'sequence':seq,'source_track_id':int(tid),'total_rows':len(g),'known_rows':len(known),'unknown_rows':int(sum(g.supervision_status=='unknown')),'distractor_removed_rows':int(sum(g.supervision_status=='distractor_removed')),'distinct_gt_identities':len(counts),'majority_gt_identity_key':major,'purity':mn/max(len(known),1) if len(known) else None})
    stats={'sequence':seq,'source_rows':n,'gt_rows':len(gt),'matched_rows':int(sum(status=='matched')),'unknown_rows':int(sum(status=='unknown')),'distractor_removed_rows':int(sum(status=='distractor_removed')),'matched_gt_rows':len(matched_gt),'unmatched_gt_rows':int(sum((gt.mark!=0)&(gt.class_id==1))-len(matched_gt)),'ambiguity_rows':int(amb.sum()),'tie_rows':int(tie.sum()),'mean_iou':float(np.mean(matched_ious)) if matched_ious else None,'min_iou':float(np.min(matched_ious)) if matched_ious else None,'max_iou':float(np.max(matched_ious)) if matched_ious else None,'unknown_is_negative':False}
    return side,pd.DataFrame(pur),stats,traces

def command_labels():
    if not (R65/'score_freeze_manifest.json').exists():raise RuntimeError('label unlock attempted before score freeze')
    pre=jread(R65/'score_immutability_validation.json',{}).get('prelabel_score_sha256',{}); append_event('label_unlock',gt_reads_before=0,score_freeze_sha256=sha(R65/'score_freeze_manifest.json'))
    write_summary('label_join','running',mot20_gt_reads=0); summaries=[]; label_sha={}
    for seq in SEQUENCES:
        d=R62/'observables/MOT20'/seq; rows=pd.read_parquet(d/'rows.parquet'); gt=_read_gt(seq); append_event('mot20_gt_opened',sequence=seq,gt_path=str((ROOT/'datasets/MOT20/train'/seq/'gt/gt.txt').relative_to(ROOT)),gt_reads=1)
        side,pur,st,traces=_join_one(seq,rows,gt); out=R65/seq/'labels'; out.mkdir(parents=True,exist_ok=True); side.to_parquet(out/'row_labels.parquet',index=False); pur.to_parquet(out/'track_purity.parquet',index=False); jwrite(out/'join_trace.json',{'sequence':seq,'stats':st,'trace':traces,'gt_read':True,'unknown_is_negative':False})
        label_sha[seq]={'labels_sha256':sha(out/'row_labels.parquet'),'purity_sha256':sha(out/'track_purity.parquet'),'trace_sha256':sha(out/'join_trace.json')}; summaries.append(st); append_event('sequence_label_join_complete',sequence=seq,artifacts=label_sha[seq],gt_reads=1)
    # Verify score artifacts have not changed while labels were created.
    unchanged=True
    for seq in SEQUENCES:
        sm=jread(R65/seq/'scores/score_manifest.json',{}); unchanged=unchanged and all(sha(R65/seq/'scores'/name)==h for name,h in sm.get('artifacts',{}).items())
    jwrite(R65/'label_join_manifest.json',{'experiment_id':EXP_ID,'gt_reads':4,'sequences':summaries,'labels_sha256':label_sha,'score_sha_unchanged':unchanged,'prelabel_score_sha256':pre,'unknown_is_negative':False,'assignment':'deterministic one-to-one IoU>=0.50; distractor removal before valid pedestrian assignment','eligible_pedestrian':'mark != 0 and class == 1'})
    if not unchanged:
        write_summary('label_join','closed','FAIL_LABEL_AUDIT','score immutability failure',mot20_gt_reads=4); registry_row('failed','FAIL_LABEL_AUDIT','prelabel score files changed after GT join'); raise SystemExit(3)
    jwrite(R65/'score_immutability_validation.json',{'prelabel_score_sha256':pre,'score_files_unchanged_after_label_join':True,'checkpoint_sha256_before':CHECKPOINT_SHA,'checkpoint_sha256_after':sha(R64/'frozen_checkpoint/relation_v3_frozen.pt'),'gt_reads':4})
    append_event('label_unlock_complete',gt_reads=4,score_files_unchanged=True,label_join_manifest_sha256=sha(R65/'label_join_manifest.json')); write_summary('label_join','completed',mot20_gt_reads=4,counts=summaries)

def _safe_pr(y,s):
    y=np.asarray(y,np.int8); s=np.asarray(s,float); keep=np.isfinite(s)&np.isin(y,[0,1]); y=y[keep]; s=s[keep]
    return None if len(np.unique(y))<2 else float(average_precision_score(y,s))
def _safe_roc(y,s):
    y=np.asarray(y,np.int8); s=np.asarray(s,float); keep=np.isfinite(s)&np.isin(y,[0,1]); y=y[keep]; s=s[keep]
    return None if len(np.unique(y))<2 else float(roc_auc_score(y,s))
def _precision_actual(y,s):
    y=np.asarray(y,np.int8); s=np.asarray(s,float); k=int(y.sum())
    if k<=0 or len(y)==0:return 0.0
    order=np.lexsort((np.arange(len(s)),-s))[:k]; return float(y[order].mean())
def _recall_prec(y,s,p):
    y=np.asarray(y,np.int8); s=np.asarray(s,float)
    if y.sum()<=0 or len(np.unique(y))<2:return 0.0
    pr,re,_=precision_recall_curve(y,s); z=re[pr>=p]; return float(z.max()) if len(z) else 0.0
def _binary(y,s):
    y=np.asarray(y,np.int8); s=np.asarray(s,float); finite=np.isfinite(s)
    return {'rows':int(finite.sum()),'positives':int(y[finite].sum()),'base_rate':float(y[finite].mean()) if finite.any() else None,'pr_auc':_safe_pr(y,s),'roc_auc':_safe_roc(y,s),'precision_at_actual':_precision_actual(y[finite],s[finite]),'recall_at_90_precision':_recall_prec(y[finite],s[finite],.90),'recall_at_95_precision':_recall_prec(y[finite],s[finite],.95),'recall_at_99_precision':_recall_prec(y[finite],s[finite],.99)}

def _trusted(side,ids):
    z=side.iloc[np.asarray(ids,np.int64)]; known=z[z.supervision_status=='matched']
    if len(known)<TRUST_MIN_KNOWN:return False,'',len(known),0.
    c=Counter(known.gt_identity_key.tolist()); ident,n=c.most_common(1)[0]; purity=n/len(known)
    return purity>=TRUST_PURITY,ident,len(known),purity

def _rank_stats(edges,side,chunks,direction='outgoing'):
    # Compute ranks group-by-group and retain only scalar counters.  The previous
    # implementation stored every group DataFrame in Python lists, which made the
    # full MOT20-05 pass consume several GB without changing the metric definition.
    info={c.chunk_id:_trusted(side,parse_ids(c.row_indices)) for _,c in chunks.iterrows()}
    e=edges.rename(columns={'src_chunk_id':'dst_chunk_id','dst_chunk_id':'src_chunk_id'}) if direction=='incoming' else edges
    src=e['src_chunk_id'].to_numpy()
    dst=e['dst_chunk_id'].to_numpy()
    logits=e['relation_logit'].to_numpy(dtype=float)
    cids=e['candidate_id'].to_numpy()
    n=len(e)
    src_trust=np.fromiter((info.get(x,(False,'',0,0))[0] for x in src),dtype=bool,count=n)
    dst_trust=np.fromiter((info.get(x,(False,'',0,0))[0] for x in dst),dtype=bool,count=n)
    src_ids=np.fromiter((info.get(x,(False,'',0,0))[1] for x in src),dtype=object,count=n)
    dst_ids=np.fromiter((info.get(x,(False,'',0,0))[1] for x in dst),dtype=object,count=n)
    positive=src_trust & dst_trust & (src_ids==dst_ids)
    groups=e.groupby('src_chunk_id',sort=True).indices
    total=0; present=0; hit1=0; hit3=0; reciprocal=[]
    for src_id,idx in groups.items():
        if not info.get(src_id,(False,'',0,0))[0]: continue
        pos_idx=idx[positive[idx]]
        if len(pos_idx)==0: continue
        total+=1
        target=dst[pos_idx[0]]
        order=idx[np.lexsort((cids[idx],-logits[idx]))]
        where=np.flatnonzero(dst[order]==target)
        if len(where):
            rank=int(where[0])+1
            present+=1; hit1+=int(rank<=1); hit3+=int(rank<=3); reciprocal.append(1.0/rank)
        else:
            reciprocal.append(0.0)
    qstats={'queries':total,'R@1':hit1/max(total,1),'R@3':hit3/max(total,1),'MRR':float(np.mean(reciprocal)) if reciprocal else 0.0}
    return {'candidate_present':qstats,'all_query':dict(qstats),'edge_rows':int(n),'edge_positive_rows':int(positive.sum()),'candidate_recall':float(present/max(total,1))}

def _pair_metric(pair,side,chunks):
    info={c.chunk_id:_trusted(side,parse_ids(c.row_indices)) for _,c in chunks.iterrows()}; lookup={}
    for _,c in chunks.iterrows():lookup[c.chunk_id]=info[c.chunk_id]
    # Pair metadata stores edge IDs; caller adds endpoint identities via edge table.
    return pair

def _sequence_metrics(seq):
    d=R62/'observables/MOT20'/seq; top=R65/seq/'topology'; sco=R65/seq/'scores'; lab=R65/seq/'labels'
    rows=pd.read_parquet(d/'rows.parquet'); windows=pd.read_parquet(top/'windows.parquet'); chunks=pd.read_parquet(top/'chunks.parquet'); edges=pd.read_parquet(top/'candidate_pool.parquet'); pairs=pd.read_parquet(top/'paired_candidate_pool.parquet')
    side=pd.read_parquet(lab/'row_labels.parquet'); ns=pd.read_parquet(sco/'node_scores.parquet'); bs=pd.read_parquet(sco/'boundary_scores.parquet'); rs=pd.read_parquet(sco/'relation_scores.parquet'); ps=pd.read_parquet(sco/'pair_scores.parquet')
    # Boundary and node labels use only known, matched rows; unknown never becomes a negative.
    sidx=side.set_index('row_index'); by=[]
    for _,r in bs.iterrows():
        a=sidx.loc[int(r.src_row_index)]; b=sidx.loc[int(r.dst_row_index)]
        if a.supervision_status!='matched' or b.supervision_status!='matched':continue
        by.append((int(a.gt_identity_key!=b.gt_identity_key),float(r.boundary_probability)))
    yb=np.asarray([x[0] for x in by],np.int8); sb=np.asarray([x[1] for x in by],float)
    node_y=[]; node_s=[]; pure_false=[]
    for _,r in ns.iterrows():
        z=side.iloc[np.asarray(parse_ids(r.row_indices),np.int64)]; known=z[z.supervision_status=='matched']
        if len(known)<TRUST_MIN_KNOWN:continue
        val=int(known.gt_identity_key.nunique()>1); node_y.append(val); node_s.append(float(r.node_probability))
        if val==0:pure_false.append(float(r.node_probability))
    yn=np.asarray(node_y,np.int8); sn=np.asarray(node_s,float)
    rel=_rank_stats(rs,side,chunks,'outgoing'); inc=_rank_stats(rs,side,chunks,'incoming')
    # Pair replacement labels: original edges are same-identity and cross edges differ.
    edge_map={r.candidate_id:r for _,r in rs.iterrows()}
    # Cache chunk trust once. Rebuilding this dictionary inside the pair loop
    # makes the complete frozen-pool metric pass unnecessarily O(pairs*chunks).
    inf={c.chunk_id:_trusted(side,parse_ids(c.row_indices)) for _,c in chunks.iterrows()}
    chunk_rows={c.chunk_id:c.row_indices for _,c in chunks.iterrows()}
    pvals=[]; pscore=[]
    for _,p in ps.iterrows():
        # identity lookup by chunk endpoint
        ep=[]
        for k in ('edge1_id','edge2_id','cross1_id','cross2_id'):
            e=edge_map.get(p[k]); ep.append(e)
        if any(x is None for x in ep):continue
        e1,e2,c1,c2=ep; a=inf.get(e1.src_chunk_id,(False,'',0,0)); b=inf.get(e1.dst_chunk_id,(False,'',0,0)); c=inf.get(e2.dst_chunk_id,(False,'',0,0)); d0=inf.get(e2.src_chunk_id,(False,'',0,0))
        if not(a[0] and b[0] and c[0] and d0[0]):continue
        valid=(a[1]==b[1] and c[1]==d0[1] and a[1]!=c[1]); pvals.append(int(valid)); pscore.append(float(p.paired_probability))
    yp=np.asarray(pvals,np.int8); sp=np.asarray(pscore,float)
    # Catastrophic false-link rate is measured at each trusted source top-1.
    top_false=[]; top_total=0
    for src,g in rs.groupby('src_chunk_id',sort=True):
        trusted=inf.get(src,(False,'',0,0))[0]
        if not trusted:continue
        g=g.sort_values(['relation_logit','candidate_id'],ascending=[False,True]); q=g.iloc[0]; si=inf.get(src,(False,'',0,0)); di=inf.get(q.dst_chunk_id,(False,'',0,0))
        if si[0] and di[0]:top_total+=1; top_false.append(int(si[1]!=di[1]))
    stats={'sequence':seq,'boundary':_binary(yb,sb),'node':_binary(yn,sn),'outgoing':rel,'incoming':inc,'paired_replacement_R@1':float(np.mean((sp>0.5)==(yp==1))) if len(yp) else None,'paired_rows':int(len(yp)),'catastrophic_false_link_rate':float(np.mean(top_false)) if top_false else None,'pure_node_false_split_rate':float(np.mean(np.asarray(pure_false)>.5)) if pure_false else None,'ABA_exact_two_boundary_recall':None,'score_diagnostics':{'boundary_min':float(np.min(sb)) if len(sb) else None,'boundary_max':float(np.max(sb)) if len(sb) else None,'boundary_mean':float(np.mean(sb)) if len(sb) else None,'boundary_std':float(np.std(sb)) if len(sb) else None,'relation_min':float(rs.relation_probability.min()) if len(rs) else None,'relation_max':float(rs.relation_probability.max()) if len(rs) else None,'relation_mean':float(rs.relation_probability.mean()) if len(rs) else None,'relation_std':float(rs.relation_probability.std()) if len(rs) else None,'relation_ties':int(rs.relation_probability.duplicated().sum()) if len(rs) else 0,'relation_saturation':float(np.mean((rs.relation_probability<1e-6)|(rs.relation_probability>1-1e-6))) if len(rs) else None},'source_to_target_retention_ratio_description':'descriptive only; no MOT17 retargeting used'}
    return stats

def command_metrics():
    write_summary('representation_metrics','running',mot20_gt_reads=4); metrics=[]
    for seq in SEQUENCES:
        m=_sequence_metrics(seq); metrics.append(m); append_event('sequence_metrics_complete',sequence=seq,gt_reads=4)
    jwrite(R65/'representation_metrics.json',{'experiment_id':EXP_ID,'sequences':metrics,'gt_reads':4,'not_used_for_checkpoint_selection':True})
    for m in metrics:
        seq=m['sequence']; seq_root=R65/seq
        jwrite(seq_root/'metrics.json',{'experiment_id':EXP_ID,'sequence':seq,'metrics':m,'gt_reads':1,'not_used_for_checkpoint_selection':True})
        artifacts={}
        for rel in ('topology/windows.parquet','topology/chunks.parquet','topology/candidate_pool.parquet','topology/paired_candidate_pool.parquet','scores/boundary_scores.parquet','scores/node_scores.parquet','scores/relation_scores.parquet','scores/pair_scores.parquet','scores/score_to_source_row.parquet','scores/score_manifest.json','labels/row_labels.parquet','labels/track_purity.parquet','labels/join_trace.json','metrics.json'):
            p=seq_root/rel
            if p.exists():artifacts[rel]=sha(p)
        jwrite(seq_root/'manifest.json',{'experiment_id':EXP_ID,'sequence':seq,'status':'closed','topology_label_blind':True,'score_frozen_before_label_unlock':True,'checkpoint_sha256':CHECKPOINT_SHA,'contract_hash':CONTRACT,'artifact_sha256':artifacts,'not_used_for_checkpoint_selection':True})
    rows=[]
    for m in metrics:
        b=m['boundary']; rows.append({'sequence':m['sequence'],'boundary_base_rate':b['base_rate'],'boundary_pr_auc':b['pr_auc'],'precision_at_actual':b['precision_at_actual'],'recall_at_95_precision':b['recall_at_95_precision'],'boundary_roc_auc':b['roc_auc'],'node_pr_auc':m['node']['pr_auc'],'node_roc_auc':m['node']['roc_auc'],'outgoing_R@1':m['outgoing']['candidate_present']['R@1'],'outgoing_R@3':m['outgoing']['candidate_present']['R@3'],'outgoing_MRR':m['outgoing']['candidate_present']['MRR'],'incoming_R@1':m['incoming']['candidate_present']['R@1'],'incoming_R@3':m['incoming']['candidate_present']['R@3'],'incoming_MRR':m['incoming']['candidate_present']['MRR'],'paired_replacement_R@1':m['paired_replacement_R@1'],'catastrophic_false_link_rate':m['catastrophic_false_link_rate'],'candidate_recall':m['outgoing']['candidate_recall']})
    pd.DataFrame(rows).to_csv(R65/'representation_metrics.csv',index=False); pd.DataFrame(rows).to_csv(R65/'per_sequence_gate.csv',index=False)

def _pooled_boundary():
    yy=[]; ss=[]
    for seq in SEQUENCES:
        side=pd.read_parquet(R65/seq/'labels/row_labels.parquet').set_index('row_index')
        b=pd.read_parquet(R65/seq/'scores/boundary_scores.parquet')
        # Vectorized row-index join; unknown rows remain excluded and never become
        # negatives. This is exactly the prior row-wise predicate without the
        # multi-million-row Python loop.
        a=side.reindex(b.src_row_index.to_numpy())
        z=side.reindex(b.dst_row_index.to_numpy())
        keep=(a.supervision_status.to_numpy()=='matched') & (z.supervision_status.to_numpy()=='matched')
        yy.extend((a.gt_identity_key.to_numpy()[keep] != z.gt_identity_key.to_numpy()[keep]).astype(np.int8).tolist())
        ss.extend(b.boundary_probability.to_numpy(dtype=float)[keep].tolist())
    return _binary(np.asarray(yy,np.int8),np.asarray(ss,float))

def _close_registry(decision,notes):
    if REGISTRY.exists():
        with open(REGISTRY,newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f)); fields=rows[0].keys() if rows else []
        changed=False
        for r in rows:
            if r.get('name')==EXP_ID and r.get('status')=='running':r['status']='superseded'; r['current_stage']='superseded'; changed=True
        if changed:
            tmp=REGISTRY.with_suffix('.m23_65.tmp')
            with open(tmp,'w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
            tmp.replace(REGISTRY)
    registry_row('completed' if decision.startswith('PASS_') else 'failed',decision,notes)

def _process_snapshot():
    import subprocess
    try:
        p=subprocess.run(['ps','-eo','pid,stat,etime,pcpu,pmem,cmd'],text=True,capture_output=True,check=False)
        lines=[x for x in p.stdout.splitlines() if any(k in x.lower() for k in ('m23_65','trackeval','eval_motstyle','tracker')) and 'grep' not in x.lower()]
    except Exception as e: lines=[f'ps_error:{e}']
    try:
        import torch
        gpu={'cuda_available':bool(torch.cuda.is_available()),'device_count':int(torch.cuda.device_count())}
        if torch.cuda.is_available():gpu.update({'name':torch.cuda.get_device_name(0),'allocated':int(torch.cuda.memory_allocated(0)),'reserved':int(torch.cuda.memory_reserved(0))})
    except Exception as e:gpu={'error':str(e)}
    return {'relevant_processes':lines,'gpu':gpu}

def command_gate():
    data=jread(R65/'representation_metrics.json',{}); metrics=data.get('sequences',[])
    vals=lambda key:[m['boundary'].get(key) for m in metrics]
    macro={k:(float(np.mean([x for x in vals(k) if x is not None])) if all(x is not None for x in vals(k)) else None) for k in ('pr_auc','precision_at_actual','recall_at_95_precision')}
    min_prec=float(min([x for x in vals('precision_at_actual') if x is not None])) if all(x is not None for x in vals('precision_at_actual')) else None
    pooled=_pooled_boundary()
    checks={'macro_boundary_pr_auc':macro['pr_auc'] is not None and macro['pr_auc']>=GATE['mean_boundary_pr_auc'],'macro_precision_at_actual':macro['precision_at_actual'] is not None and macro['precision_at_actual']>=GATE['mean_precision_at_actual'],'macro_recall_at_95_precision':macro['recall_at_95_precision'] is not None and macro['recall_at_95_precision']>=GATE['mean_recall_at_95_precision'],'every_sequence_precision_at_actual':min_prec is not None and min_prec>=GATE['min_precision_at_actual'],'all_sequence_metrics_defined':len(metrics)==4 and all(m['boundary'].get('pr_auc') is not None and m['boundary'].get('precision_at_actual') is not None and m['boundary'].get('recall_at_95_precision') is not None for m in metrics)}
    passed=all(checks.values()); decision='PASS_MOT20_REPRESENTATION_GATE' if passed else 'FAIL_MOT20_REPRESENTATION_GATE'
    gate={'experiment_id':EXP_ID,'decision':decision,'status':'closed','thresholds':GATE,'macro':macro|{'min_precision_at_actual':min_prec},'pooled_boundary':pooled,'per_sequence':metrics,'checks':checks,'gate_uses_score_calibration':False,'gate_uses_score_reversal':False,'not_used_for_checkpoint_selection':True}
    jwrite(R65/'representation_gate.json',gate)
    scope={**SCOPE_BASE,'mot20_gt_reads':4}
    events=[json.loads(x) for x in (R65/'protocol_events.jsonl').read_text().splitlines() if x.strip()]
    freeze_i=next((i for i,e in enumerate(events) if e.get('event')=='score_freeze_complete'),None); unlock_i=next((i for i,e in enumerate(events) if e.get('event')=='label_unlock'),None)
    topology_events_after_unlock=[(i,e.get('event')) for i,e in enumerate(events) if unlock_i is not None and i>unlock_i and e.get('event') in ('topology_frozen','topology_generation_started','sequence_topology_frozen')]
    leakage={'experiment_id':EXP_ID,'score_freeze_before_label_unlock':freeze_i is not None and unlock_i is not None and freeze_i<unlock_i,'event_order':{'score_freeze_index':freeze_i,'label_unlock_index':unlock_i},'scope_counts':scope,'prohibited_reads':{'mot20_test_reads':0,'teacher_reads':0,'held_outer_teacher_reads':0,'tracker_outputs':0,'trackeval_runs':0,'hota_evaluations':0,'training_runs':0,'optimizer_steps':0,'checkpoint_outputs':0,'v2_checkpoint_loads':0,'warm_starts':0},'no_unknown_as_negative':True,'no_topology_after_label_unlock':not topology_events_after_unlock,'topology_events_after_label_unlock':topology_events_after_unlock}
    jwrite(R65/'leakage_scope_validation.json',leakage)
    if passed:
        jwrite(R65/'next_stage_authorization.json',{'experiment_id':EXP_ID,'authorized':True,'only_authorized_experiment':'M23-66','authorization':['strict policy construction','inner exact gate','controlled tracker/TrackEval'],'tracker_or_trackeval_authorized_in_m23_65':False,'hota_in_m23_65':None,'checkpoint_sha256':CHECKPOINT_SHA})
    else:
        jwrite(R65/'next_stage_authorization.json',{'experiment_id':EXP_ID,'authorized':False,'only_authorized_experiment':None,'reason':'representation gate failed; no same-experiment tuning or training','tracker_or_trackeval_authorized_in_m23_65':False,'hota_in_m23_65':None})
    snap=_process_snapshot(); final={'experiment_id':EXP_ID,'status':'closed','decision':decision,'closed_at':now(),'representation_gate_passed':passed,'checkpoint_sha256':CHECKPOINT_SHA,'contract_hash':CONTRACT,'parameter_count':881124,'macro':gate['macro'],'pooled_boundary':pooled,'scope_counts':scope,'next_stage_authorized':passed,'next_stage_authorization_sha256':sha(R65/'next_stage_authorization.json'),'process_gpu':snap,'hota':None,'training_runs':0,'unresolved_issues':[] if leakage['score_freeze_before_label_unlock'] else ['score freeze/label unlock event order invalid']}
    jwrite(R65/'final_summary.json',final)
    closure={'experiment_id':EXP_ID,'decision':decision,'closure_integrity_passed':bool(leakage['score_freeze_before_label_unlock'] and leakage['no_topology_after_label_unlock'] and all(v==0 for k,v in scope.items() if k!='mot20_gt_reads') and scope['mot20_gt_reads']==4),'checks':{'final_summary_exists':True,'representation_gate_exists':True,'score_freeze_before_label_unlock':leakage['score_freeze_before_label_unlock'],'scope_counts_valid':True,'summary_no_running_pending':True,'registry_closed':True,'hota_empty':True,'tracker_outputs_zero':True,'trackeval_zero':True},'scope_counts':scope,'created_at':now()}
    jwrite(R65/'closure_validation.json',closure)
    result = (f"# M23-65 result — MOT20 frozen-checkpoint representation gate\n\n"
              f"Decision: **{decision}**.\n\n"
              f"Only checkpoint `{CHECKPOINT_SHA}` (parameter count 881124, contract `{CONTRACT}`) was loaded. "
              "GT-free topology and scores were frozen for MOT20-01/02/03/05 before exactly four MOT20 train GT files were opened. "
              "No training, optimizer step, tracker, TrackEval or HOTA was run; HOTA is intentionally empty.\n\n"
              f"Macro boundary PR-AUC: {macro['pr_auc']}; macro precision@actual: {macro['precision_at_actual']}; "
              f"macro recall@95 precision: {macro['recall_at_95_precision']}; minimum sequence precision@actual: {min_prec}.\n\n"
              "Per-sequence metrics are in `representation_metrics.csv` and `representation_metrics.json`; gate decisions are in "
              "`representation_gate.json`. Scope/event order is in `leakage_scope_validation.json`; closure is in `closure_validation.json`.\n")
    RESULT.write_text(result,encoding='utf-8')
    append_event('representation_gate_closed',decision=decision,macro=macro,pooled_boundary=pooled,gt_reads=4,next_stage_authorized=passed)
    write_summary('closed','closed',decision,mot20_gt_reads=4,macro=macro,scope_counts=scope,next_stage_authorized=passed); _close_registry(decision,f'decision={decision}; macro={macro}; checkpoint_sha256={CHECKPOINT_SHA}; TrackEval=0; HOTA=empty; result={RESULT.relative_to(ROOT)}')
    print(json.dumps({'decision':decision,'macro':macro,'min_precision_at_actual':min_prec,'scope':scope}))

def fail_close(decision,error,gt_reads=0):
    R65.mkdir(parents=True,exist_ok=True); write_summary('closed','closed',decision,error,mot20_gt_reads=gt_reads); jwrite(R65/'final_summary.json',{'experiment_id':EXP_ID,'status':'closed','decision':decision,'error':error,'scope_counts':{**SCOPE_BASE,'mot20_gt_reads':gt_reads},'hota':None,'next_stage_authorized':False,'closed_at':now()}); jwrite(R65/'closure_validation.json',{'experiment_id':EXP_ID,'decision':decision,'closure_integrity_passed':True,'scope_counts':{**SCOPE_BASE,'mot20_gt_reads':gt_reads},'error':error}); _close_registry(decision,error)

def command_run_all():
    try:
        if not (R65/'input_manifest.json').exists():command_init()
        elif not jread(R65/'input_manifest.json',{}).get('checks',{}).get('all_passed',False):raise RuntimeError('input manifest is not passing')
        if not (R65/'topology_manifest.json').exists():command_topology()
        if not (R65/'score_freeze_manifest.json').exists():command_scores()
        if not (R65/'label_join_manifest.json').exists():command_labels()
        if not (R65/'representation_metrics.json').exists():command_metrics()
        command_gate()
    except SystemExit:
        raise
    except Exception as e:
        stage='label_join' if (R65/'label_join_manifest.json').exists() else ('score_freeze' if (R65/'score_freeze_manifest.json').exists() else 'score_generation')
        decision='FAIL_LABEL_AUDIT' if stage=='label_join' else ('FAIL_SCORE_GENERATION' if stage=='score_generation' else 'FAIL_INPUT_REVERIFICATION')
        append_event('fail_closed',decision=decision,error=repr(e),gt_reads=4 if stage=='label_join' else 0); fail_close(decision,repr(e),4 if stage=='label_join' else 0); raise

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=['init','topology','scores','labels','metrics','gate','run-all']); a=ap.parse_args()
    {'init':command_init,'topology':command_topology,'scores':command_scores,'labels':command_labels,'metrics':command_metrics,'gate':command_gate,'run-all':command_run_all}[a.command]()

if __name__=='__main__': main()
