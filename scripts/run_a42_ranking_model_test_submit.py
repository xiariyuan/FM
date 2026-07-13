#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, shutil, subprocess, sys, zipfile, hashlib
from pathlib import Path
from typing import Dict, Tuple, Set
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

TEST_SEQS=['MOT20-04','MOT20-06','MOT20-07','MOT20-08']
FEATURE_COLS=[
'gap','len_a','len_b','duration_a','duration_b','avg_score_a','avg_score_b','last_score_a','first_score_b',
'center_distance','center_distance_per_frame','predicted_distance','predicted_distance_per_frame','velocity_cosine',
'height_ratio','area_ratio','bottom_y_gap','x_bucket_delta','bottom_bucket_delta','zone_exact_match','zone_near_match',
'source_debt_count','target_debt_count','edge_debt_score','geometry_risk','motion_risk',
'cos_end_start','cos_end_global','cos_global_start','cos_global_global','cos_high_high','cos_start_start','cos_end_end',
'appearance_mean','appearance_max','appearance_min','appearance_std','appearance_gap_consistency','a42_appearance_score','a42_rule_score_v1',
'out_rank_by_appearance_max','out_margin_to_second_appearance_max','out_group_size','in_rank_by_appearance_max','in_margin_to_second_appearance_max','in_group_size',
'out_rank_by_a42_rule_score_v1','out_margin_to_second_a42_rule_score_v1','in_rank_by_a42_rule_score_v1','in_margin_to_second_a42_rule_score_v1',
'out_rank_by_cos_global_global','out_margin_to_second_cos_global_global','in_rank_by_cos_global_global','in_margin_to_second_cos_global_global',
'out_rank_by_cos_high_high','out_margin_to_second_cos_high_high','in_rank_by_cos_high_high','in_margin_to_second_cos_high_high',
'max_rank_by_a42_rule_score_v1','min_margin_by_a42_rule_score_v1','max_rank_by_appearance_max','min_margin_by_appearance_max']

def ai(v,d=0):
    try:
        if v is None or v=='': return d
        return int(float(v))
    except Exception: return d

def af(v,d=0.0):
    try:
        if v is None or v=='': return d
        return float(v)
    except Exception: return d

def read_rows(path:Path):
    with path.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))
def write_rows(path:Path, rows):
    rows=list(rows); fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); fields.append(k)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or ['seq'],extrasaction='ignore'); w.writeheader(); w.writerows(rows)
def load_manifest(path:Path, train:bool):
    ids=['seq','track_a','track_b','gap','gap_bucket']+(['same_gt'] if train else [])
    df=pd.read_csv(path,usecols=ids+FEATURE_COLS)
    for c in FEATURE_COLS+(['same_gt'] if train else []):
        df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0.0)
    df['track_a']=df['track_a'].astype(str); df['track_b']=df['track_b'].astype(str)
    df['log_gap']=np.log1p(df['gap']); df['inv_gap']=1.0/np.maximum(1.0,df['gap'].to_numpy(dtype=float)); df['score_minus_app']=df['a42_rule_score_v1']-df['appearance_max']
    return df
def Xmat(df): return df[FEATURE_COLS+['log_gap','inv_gap','score_minus_app']].to_numpy(dtype=np.float32)
def hard_mask(df):
    return ((df['same_gt'].to_numpy()==1)|(df['appearance_max'].to_numpy()>=0.55)|(df['a42_rule_score_v1'].to_numpy()>=0.45)|(df['max_rank_by_a42_rule_score_v1'].to_numpy()<=20)|(df['cos_global_global'].to_numpy()>=0.55))
def train_model(X,y):
    pos=max(1,int(y.sum())); neg=max(1,int(len(y)-y.sum())); pos_weight=min(80.0,max(10.0,0.35*neg/pos))
    sw=np.where(y==1,pos_weight,1.0).astype(np.float32)
    clf=HistGradientBoostingClassifier(max_iter=120,learning_rate=0.06,max_leaf_nodes=31,l2_regularization=0.15,min_samples_leaf=25,random_state=20260703,early_stopping=True,validation_fraction=0.12,n_iter_no_change=12)
    clf.fit(X,y,sample_weight=sw); return clf, pos_weight
def add_rank(df):
    df=df.copy(); df['_orig']=np.arange(len(df)); df['a42_model_score']=pd.to_numeric(df['a42_model_score'],errors='coerce').fillna(0)
    df=df.sort_values(['seq','track_a','a42_model_score','_orig'],ascending=[True,True,False,True])
    df['out_rank_by_a42_model_score']=df.groupby(['seq','track_a']).cumcount()+1
    df=df.sort_values(['seq','track_b','a42_model_score','_orig'],ascending=[True,True,False,True])
    df['in_rank_by_a42_model_score']=df.groupby(['seq','track_b']).cumcount()+1
    df['max_rank_by_a42_model_score']=np.maximum(df['out_rank_by_a42_model_score'],df['in_rank_by_a42_model_score'])
    return df.sort_values('_orig').drop(columns=['_orig'])
def key_from(row): return (row['seq'],str(ai(row['track_a'])),str(ai(row['track_b'])))
def load_keys(path:Path)->Set[Tuple[str,str,str]]: return {key_from(r) for r in read_rows(path)}
def source_file(src:Path,seq:str):
    p=src/seq/f'{seq}.txt'
    if p.exists(): return p
    p=src/f'{seq}.txt'
    if p.exists(): return p
    raise FileNotFoundError(seq)
def read_mot(path:Path):
    out=[]
    with path.open('r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts=line.split(',')
            if len(parts)>=6: out.append((int(float(parts[0])),int(float(parts[1])),parts))
    return out
def find(parent:Dict[int,int],x:int):
    parent.setdefault(x,x)
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
def md5(path:Path):
    h=hashlib.md5()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def line_count(path:Path):
    with path.open('r',encoding='utf-8') as f: return sum(1 for _ in f)
def link_edges(edge_rows,src_dir:Path,linked_dir:Path):
    linked_dir.mkdir(parents=True,exist_ok=True); by={s:[] for s in TEST_SEQS}
    for r in edge_rows: by.setdefault(r['seq'],[]).append(r)
    selected=[]; by_seq=[]; audit=[]
    for seq in TEST_SEQS:
        rows=sorted(by.get(seq,[]),key=lambda r:af(r.get('edge_weight')),reverse=True)
        parent={}; used_s=set(); used_t=set(); final=[]
        for r in rows:
            a=ai(r['track_a']); b=ai(r['track_b'])
            if a in used_s or b in used_t: continue
            ra,rb=find(parent,a),find(parent,b)
            if ra==rb: continue
            parent[rb]=ra; used_s.add(a); used_t.add(b); final.append(r)
        ids=set()
        for r in final: ids.add(ai(r['track_a'])); ids.add(ai(r['track_b']))
        idmap={tid:find(parent,tid) for tid in ids}
        src=source_file(src_dir,seq); out=linked_dir/f'{seq}.txt'
        mot=[]
        for _,tid,parts in read_mot(src):
            pp=list(parts); pp[1]=str(idmap.get(tid,tid)); mot.append(pp)
        mot.sort(key=lambda p:(int(float(p[0])),int(float(p[1])),float(p[2]),float(p[3])))
        with out.open('w',encoding='utf-8') as f:
            for pp in mot: f.write(','.join(pp)+'\n')
        selected.extend(final)
        by_seq.append({'seq':seq,'candidate_edges':len(rows),'accepted_links':len(final),'base_links':sum(ai(r.get('is_base')) for r in final),'recovery_links':sum(ai(r.get('is_recovery')) for r in final)})
        audit.append({'seq':seq,'source_file':str(src),'source_rows':line_count(src),'source_md5':md5(src),'linked_file':str(out),'linked_rows':line_count(out),'linked_md5':md5(out),'row_count_ok':int(line_count(src)==line_count(out))})
    return selected,by_seq,audit
def interpolate(raw:Path,track:Path,out:Path):
    cmd=[sys.executable,'scripts/postprocess/linear_interpolate_mot.py','--input-dir',str(raw),'--output-dir',str(track),'--max-gap','30','--summary-json',str(out/'interp_summary.json'),'--summary-csv',str(out/'interp_summary.csv')]
    p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT); (out/'interp_stdout.log').write_text(p.stdout,encoding='utf-8'); return p.returncode
def package_validate(track:Path,package:Path,zip_path:Path,out:Path):
    package.mkdir(parents=True,exist_ok=True)
    for seq in TEST_SEQS: shutil.copy2(track/f'{seq}.txt',package/f'{seq}.txt')
    if zip_path.exists(): zip_path.unlink()
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
        for seq in TEST_SEQS: zf.write(package/f'{seq}.txt',arcname=f'{seq}.txt')
    logs={}
    for name,cmd in {'results_dir':[sys.executable,'scripts/check_mot20_submission.py','--results-dir',str(package),'--profile','mot20_test_4'],'zip_path':[sys.executable,'scripts/check_mot20_submission.py','--zip-path',str(zip_path),'--profile','mot20_test_4']}.items():
        p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT); log=out/f'validation_{name}.txt'; log.write_text(p.stdout,encoding='utf-8'); logs[name]={'returncode':p.returncode,'log_path':str(log)}
    return logs
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train-manifest',required=True); ap.add_argument('--test-manifest',required=True); ap.add_argument('--base-links',required=True); ap.add_argument('--source-by-seq-dir',required=True); ap.add_argument('--out-dir',required=True); ap.add_argument('--top-recovery',type=int,default=150); ap.add_argument('--max-rank',type=int,default=3); ap.add_argument('--min-app',type=float,default=0.55)
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    print('[load train]'); train=load_manifest(Path(args.train_manifest),True); y=train['same_gt'].to_numpy(dtype=np.int8); hm=hard_mask(train)
    print('[train model]',len(train),int(hm.sum()),int(y.sum())); clf,pos_weight=train_model(Xmat(train.loc[hm]),y[hm]); del train
    print('[load test]'); test=load_manifest(Path(args.test_manifest),False); scores=clf.predict_proba(Xmat(test))[:,1].astype(np.float32)
    sdf=test[['seq','track_a','track_b','gap','gap_bucket','appearance_max','geometry_risk','motion_risk']].copy(); sdf['a42_model_score']=scores; del test
    print('[rank]'); sdf=add_rank(sdf)
    score_cols=['seq','track_a','track_b','gap','gap_bucket','a42_model_score','appearance_max','geometry_risk','motion_risk','out_rank_by_a42_model_score','in_rank_by_a42_model_score','max_rank_by_a42_model_score']
    sdf[score_cols].to_csv(out/'a42_test_model_scores.csv',index=False)
    base_keys=load_keys(Path(args.base_links)); keys=list(zip(sdf['seq'],sdf['track_a'].astype(str),sdf['track_b'].astype(str))); is_base=np.array([k in base_keys for k in keys],dtype=bool)
    cand=(~is_base)&(sdf['max_rank_by_a42_model_score'].to_numpy()<=args.max_rank)&(sdf['appearance_max'].to_numpy()>=args.min_app)&(sdf['geometry_risk'].to_numpy()<=1)&(sdf['motion_risk'].to_numpy()<=2)
    idx=np.where(cand)[0]
    if len(idx)>args.top_recovery: idx=idx[np.argsort(-sdf['a42_model_score'].to_numpy()[idx])[:args.top_recovery]]
    selected_mask=is_base.copy(); selected_mask[idx]=True
    edge_rows=[]
    for _,r in sdf.loc[selected_mask].iterrows():
        k=(r['seq'],str(r['track_a']),str(r['track_b'])); base=k in base_keys; score=float(r['a42_model_score'])
        edge_rows.append({'seq':r['seq'],'track_a':str(r['track_a']),'track_b':str(r['track_b']),'gap':int(r['gap']),'gap_bucket':r['gap_bucket'],'a42_model_score':score,'appearance_max':float(r['appearance_max']),'max_rank_by_a42_model_score':int(r['max_rank_by_a42_model_score']),'is_base':int(base),'is_recovery':int(not base),'edge_weight':(2.0+score) if base else score})
    raw=out/'raw_linked_results'; track=out/'track_results'
    selected,by_seq,audit=link_edges(edge_rows,Path(args.source_by_seq_dir),raw); write_rows(out/'accepted_links.csv',selected)
    interp_rc=interpolate(raw,track,out)
    zip_path=out/f'MOT20_A42_02b_top{args.top_recovery}_rank{args.max_rank}_app{args.min_app:.2f}_submission.zip'
    validation=package_validate(track,out/'package_root',zip_path,out)
    summary={'policy':f'top{args.top_recovery}_rank{args.max_rank}_app{args.min_app:.2f}','train_pos_weight':pos_weight,'base_links':len(base_keys),'recovery_candidates_before_topk':int(cand.sum()),'accepted_links_total':len(selected),'accepted_recovery_links':sum(ai(r['is_recovery']) for r in selected),'by_seq':by_seq,'input_output_audit':audit,'interp_returncode':interp_rc,'zip_path':str(zip_path),'validation':validation,'decision':'PASS_FORMAT_READY' if all(a['row_count_ok'] for a in audit) and interp_rc==0 and all(v['returncode']==0 for v in validation.values()) else 'CHECK_FAILED'}
    (out/'submission_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()
