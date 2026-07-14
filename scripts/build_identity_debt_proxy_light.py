from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

SEQS = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na=float(np.linalg.norm(a)); nb=float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float('nan')
    return float(np.dot(a,b)/(na*nb))


def load_tracks(path: Path) -> Dict[int,List[dict]]:
    tracks=defaultdict(list)
    with path.open() as f:
        for line in f:
            p=line.strip().split(',')
            if len(p)<6: continue
            fr=int(float(p[0])); tid=int(float(p[1]))
            x,y,w,h=map(float,p[2:6]); score=float(p[6]) if len(p)>6 else 1.0
            tracks[tid].append({'frame':fr,'x':x,'y':y,'w':w,'h':h,'score':score,'cx':x+w/2,'cy':y+h/2})
    for tid in tracks: tracks[tid].sort(key=lambda r:r['frame'])
    return tracks


def geometry(rows: List[dict], seq_len: int) -> dict:
    frames=np.asarray([r['frame'] for r in rows],dtype=np.int64)
    scores=np.asarray([r['score'] for r in rows],dtype=float)
    ws=np.asarray([r['w'] for r in rows],dtype=float)
    hs=np.asarray([r['h'] for r in rows],dtype=float)
    areas=np.maximum(ws*hs,1e-6); aspects=ws/np.maximum(hs,1e-6)
    diffs=np.diff(frames); gaps=np.maximum(diffs-1,0)
    speeds=[]; acc=[]; prev=None
    for a,b in zip(rows[:-1],rows[1:]):
        dt=max(1,b['frame']-a['frame']); norm=max(1.0,(a['h']+b['h'])/2)
        v=math.hypot(b['cx']-a['cx'],b['cy']-a['cy'])/dt/norm
        speeds.append(v)
        if prev is not None: acc.append(abs(v-prev))
        prev=v
    span=int(frames[-1]-frames[0]+1)
    return {
        'track_len':len(rows),'track_span':span,'track_density':len(rows)/max(1,span),
        'start_frac':frames[0]/max(1,seq_len),'end_frac':frames[-1]/max(1,seq_len),'span_frac':span/max(1,seq_len),
        'internal_gap_count':int((gaps>0).sum()),'internal_gap_frames':int(gaps.sum()),'max_internal_gap':int(gaps.max()) if len(gaps) else 0,
        'score_mean':float(scores.mean()),'score_std':float(scores.std()),'score_min':float(scores.min()),'score_p10':float(np.quantile(scores,.1)),
        'log_area_mean':float(np.log(areas).mean()),'log_area_std':float(np.log(areas).std()),
        'aspect_mean':float(aspects.mean()),'aspect_std':float(aspects.std()),
        'height_mean':float(hs.mean()),'height_cv':float(hs.std()/max(1e-6,hs.mean())),
        'speed_mean':float(np.mean(speeds)) if speeds else 0.0,'speed_std':float(np.std(speeds)) if speeds else 0.0,
        'speed_max':float(np.max(speeds)) if speeds else 0.0,'accel_mean':float(np.mean(acc)) if acc else 0.0,'accel_max':float(np.max(acc)) if acc else 0.0,
    }


def load_reid(root: Path, seq: str) -> Dict[int,dict]:
    d=root/seq
    z=np.load(d/'tracklet_reid_features.npz')
    idx=list(csv.DictReader((d/'tracklet_reid_index.csv').open()))
    track_ids=np.asarray(z['track_id'],dtype=np.int64)
    starts=np.asarray(z['start'],dtype=np.float32)
    ends=np.asarray(z['end'],dtype=np.float32)
    globals_=np.asarray(z['global_mean'],dtype=np.float32)
    highs=np.asarray(z['high_score'],dtype=np.float32)
    z.close()
    out={}
    for i,tid in enumerate(track_ids):
        meta=idx[i]; tid=int(tid)
        out[tid]={
            'start':starts[i],'end':ends[i],
            'global':globals_[i],'high':highs[i],
            'index_len':int(meta['track_len']),'index_start':int(meta['start_frame']),'index_end':int(meta['end_frame']),
            'samples':int(meta['total_available_features']),
        }
    return out


def load_labels(path: Path) -> Dict[str,Dict[int,dict]]:
    counts=defaultdict(lambda:defaultdict(Counter))
    with path.open(newline='') as f:
        for r in csv.DictReader(f):
            seq=r['seq']
            if seq in SEQS: counts[seq][int(r['track_id'])][int(r['gt_id'])]+=1
    out={}
    for seq,by_tid in counts.items():
        by_gt=defaultdict(Counter)
        for tid,c in by_tid.items():
            for gt,n in c.items(): by_gt[gt][tid]+=n
        dom_gt={tid:c.most_common(1)[0][0] for tid,c in by_tid.items()}
        dom_tid={gt:c.most_common(1)[0][0] for gt,c in by_gt.items()}
        gt_total={gt:sum(c.values()) for gt,c in by_gt.items()}
        gt_dom={gt:c[dom_tid[gt]] for gt,c in by_gt.items()}
        o={}
        for tid,c in by_tid.items():
            total=sum(c.values()); dgt=dom_gt[tid]; dominant=c[dgt]
            repair=sum(n for gt,n in c.items() if not (gt==dgt and dom_tid[gt]==tid))
            o[tid]={
                'matched_rows':total,'dominant_gt':dgt,'dominant_gt_rows':dominant,
                'track_gt_purity':dominant/max(1,total),'track_unique_gt':len(c),
                'repair_debt_rows':repair,'repair_debt_ratio':repair/max(1,total),
                'family_debt_rows':gt_total[dgt]-gt_dom[dgt],
                'family_debt_ratio':(gt_total[dgt]-gt_dom[dgt])/max(1,gt_total[dgt]),
                'family_unique_tracker_ids':len(by_gt[dgt]),
                'is_dominant_tid_for_gt':int(dom_tid[dgt]==tid),
            }
        out[seq]=o
    return out


def add_overlap(dataset: Dict[tuple,dict], path: Path) -> None:
    agg=defaultdict(lambda:{'events':0,'frames':set(),'partners':set(),'sum_ioa':0.0,'max_ioa':0.0})
    with path.open(newline='') as f:
        for r in csv.DictReader(f):
            seq=r['seq']; fr=int(r['frame']); a=int(r['track_i']); b=int(r['track_j']); ioa=float(r['ioa_min_area'])
            for tid,other in ((a,b),(b,a)):
                key=(seq,tid)
                if key not in dataset: continue
                x=agg[key]; x['events']+=1; x['frames'].add(fr); x['partners'].add(other); x['sum_ioa']+=ioa; x['max_ioa']=max(x['max_ioa'],ioa)
    for key,row in dataset.items():
        x=agg[key]
        row['overlap_event_count']=x['events']; row['overlap_frame_count']=len(x['frames']); row['overlap_partner_count']=len(x['partners'])
        row['overlap_ioa_mean']=x['sum_ioa']/max(1,x['events']); row['overlap_ioa_max']=x['max_ioa']
        row['overlap_frame_fraction']=len(x['frames'])/max(1,row['track_len'])
        row['overlap_events_per_frame']=x['events']/max(1,row['track_len'])
        row['overlap_partners_per_100f']=100*len(x['partners'])/max(1,row['track_len'])


def selection_metrics(df: pd.DataFrame, score: np.ndarray, frac: float) -> dict:
    order=np.argsort(-score,kind='mergesort'); n=max(1,round(len(df)*frac)); sel=df.iloc[order[:n]]
    fam=df.groupby('dominant_gt',as_index=False).family_debt_rows.max()
    selected_fams=set(sel.dominant_gt.astype(int))
    return {
        'fraction':frac,'selected_tracks':n,
        'track_debt_recall':float(sel.repair_debt_rows.sum()/max(1,df.repair_debt_rows.sum())),
        'family_debt_recall':float(fam[fam.dominant_gt.isin(selected_fams)].family_debt_rows.sum()/max(1,fam.family_debt_rows.sum())),
        'precision_family_debt_ge100':float((sel.family_debt_rows>=100).mean()),
        'unique_families':len(selected_fams),
    }


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: List[str], target: str, model_name: str) -> tuple[np.ndarray,dict]:
    imp=SimpleImputer(strategy='median')
    Xtr=imp.fit_transform(train[features]); Xte=imp.transform(test[features])
    y=np.log1p(train[target].to_numpy(float))
    seq_counts=train.groupby('seq').size().to_dict()
    w=np.asarray([1.0/seq_counts[s] for s in train.seq],dtype=float); w*=len(w)/w.sum()
    if model_name=='ridge':
        sc=StandardScaler(); Xtr2=sc.fit_transform(Xtr); Xte2=sc.transform(Xte)
        model=Ridge(alpha=20.0); model.fit(Xtr2,y,sample_weight=w); pred=np.expm1(model.predict(Xte2)); importance=np.abs(model.coef_)
    elif model_name=='rf':
        model=RandomForestRegressor(n_estimators=300,min_samples_leaf=5,max_features=.7,n_jobs=-1,random_state=42)
        model.fit(Xtr,y,sample_weight=w); pred=np.expm1(model.predict(Xte)); importance=model.feature_importances_
    elif model_name=='hgb':
        model=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=12,l2_regularization=2.0,random_state=42)
        model.fit(Xtr,y,sample_weight=w); pred=np.expm1(model.predict(Xte)); importance=np.zeros(len(features))
    else: raise ValueError(model_name)
    return np.maximum(pred,0),dict(zip(features,map(float,importance)))


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--track-root',required=True); ap.add_argument('--feature-root',required=True)
    ap.add_argument('--matches-csv',required=True); ap.add_argument('--overlap-csv',required=True); ap.add_argument('--out-dir',required=True)
    args=ap.parse_args()
    track_root=Path(args.track_root); feature_root=Path(args.feature_root); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    labels=load_labels(Path(args.matches_csv)); dataset={}
    for seq in SEQS:
        tracks=load_tracks(track_root/f'{seq}.txt'); seq_len=max(r['frame'] for rs in tracks.values() for r in rs); reid=load_reid(feature_root,seq)
        for tid,rows in tracks.items():
            row={'seq':seq,'track_id':tid}; row.update(geometry(rows,seq_len))
            f=reid.get(tid)
            if f:
                start_ok=f['index_start']==rows[0]['frame']; end_ok=f['index_end']==rows[-1]['frame']; reliable=start_ok and end_ok
                row.update({'has_reid':1,'reid_samples':f['samples'],'reid_start_match':int(start_ok),'reid_end_match':int(end_ok),'reid_reliable':int(reliable),'reid_index_len_ratio':f['index_len']/max(1,len(rows))})
                vals={
                    'start_end_cos':cosine(f['start'],f['end']),'start_global_cos':cosine(f['start'],f['global']),
                    'end_global_cos':cosine(f['end'],f['global']),'high_global_cos':cosine(f['high'],f['global']),
                    'start_high_cos':cosine(f['start'],f['high']),'end_high_cos':cosine(f['end'],f['high']),
                }
                for k,v in vals.items(): row[k]=v if reliable else float('nan')
            else:
                row.update({'has_reid':0,'reid_samples':0,'reid_start_match':0,'reid_end_match':0,'reid_reliable':0,'reid_index_len_ratio':float('nan')})
                for k in ['start_end_cos','start_global_cos','end_global_cos','high_global_cos','start_high_cos','end_high_cos']: row[k]=float('nan')
            row['appearance_drift']=1-row['start_end_cos'] if math.isfinite(row['start_end_cos']) else float('nan')
            row.update(labels.get(seq,{}).get(tid,{'matched_rows':0,'dominant_gt':-1,'dominant_gt_rows':0,'track_gt_purity':0.0,'track_unique_gt':0,'repair_debt_rows':0,'repair_debt_ratio':0.0,'family_debt_rows':0,'family_debt_ratio':0.0,'family_unique_tracker_ids':0,'is_dominant_tid_for_gt':0}))
            dataset[(seq,tid)]=row
    add_overlap(dataset,Path(args.overlap_csv))
    df=pd.DataFrame(dataset.values()).sort_values(['seq','track_id']).reset_index(drop=True)
    for c in ['track_len','track_span','score_std','speed_max','appearance_drift','overlap_event_count','overlap_frame_count','overlap_partner_count','overlap_frame_fraction','overlap_events_per_frame']:
        df[c+'_pct']=df.groupby('seq')[c].rank(pct=True,method='average').fillna(0.0)
    label_cols={'matched_rows','dominant_gt','dominant_gt_rows','track_gt_purity','track_unique_gt','repair_debt_rows','repair_debt_ratio','family_debt_rows','family_debt_ratio','family_unique_tracker_ids','is_dominant_tid_for_gt'}
    id_cols={'seq','track_id'}
    features=[c for c in df.columns if c not in label_cols|id_cols and pd.api.types.is_numeric_dtype(df[c])]
    df.to_csv(out/'dataset.csv',index=False)
    all_results=[]; pred_rows=[]; importances=defaultdict(float)
    baselines=['overlap_event_count','overlap_partner_count','track_len','appearance_drift']
    for holdout in SEQS:
        train=df[df.seq!=holdout].copy(); test=df[df.seq==holdout].copy().reset_index(drop=True)
        scores={b:test[b].fillna(test[b].median()).to_numpy(float) for b in baselines}
        for target in ['repair_debt_rows','family_debt_rows']:
            for model in ['ridge','rf','hgb']:
                name=f'{model}_{target}'
                pred,imp=fit_predict(train,test,features,target,model); scores[name]=pred
                if model=='rf':
                    for k,v in imp.items(): importances[k]+=v/len(SEQS)/2
        for name,score in scores.items():
            yfam=(test.family_debt_rows>=100).astype(int).to_numpy()
            rho=spearmanr(score,test.family_debt_rows).statistic if len(np.unique(score))>1 else float('nan')
            ap=average_precision_score(yfam,score); auc=roc_auc_score(yfam,score)
            rec={'holdout':holdout,'method':name,'spearman_family_debt':float(rho),'ap_family_debt_ge100':float(ap),'auc_family_debt_ge100':float(auc),'topk':[selection_metrics(test,score,f) for f in [.1,.2,.3,.5]]}
            all_results.append(rec)
            pp=test[['seq','track_id','dominant_gt','repair_debt_rows','family_debt_rows']].copy(); pp['method']=name; pp['score']=score; pp['rank']=pd.Series(score).rank(ascending=False,method='first').astype(int); pred_rows.append(pp)
    pd.concat(pred_rows).to_csv(out/'loo_predictions.csv',index=False)
    impdf=pd.DataFrame([{'feature':k,'importance':v} for k,v in importances.items()]).sort_values('importance',ascending=False); impdf.to_csv(out/'rf_feature_importance.csv',index=False)
    agg={}
    for method in sorted({r['method'] for r in all_results}):
        rr=[r for r in all_results if r['method']==method]
        agg[method]={'mean_spearman':float(np.nanmean([r['spearman_family_debt'] for r in rr])),'mean_ap':float(np.mean([r['ap_family_debt_ge100'] for r in rr])),'mean_auc':float(np.mean([r['auc_family_debt_ge100'] for r in rr])),'topk':{}}
        for frac in [.1,.2,.3,.5]:
            vals=[next(x for x in r['topk'] if x['fraction']==frac) for r in rr]
            agg[method]['topk'][str(frac)]={k:float(np.mean([v[k] for v in vals])) for k in ['track_debt_recall','family_debt_recall','precision_family_debt_ge100']}
    report={'protocol':{'features':'tracker-observable only','labels':'GT used only for offline debt targets','validation':'leave-one-sequence-out','reid_policy':'per-track features used only when start/end frames align'},'rows':len(df),'rows_by_seq':df.groupby('seq').size().to_dict(),'features':features,'fold_results':all_results,'aggregate':agg,'top20_rf_features':impdf.head(20).to_dict(orient='records')}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    m02=[r for r in all_results if r['holdout']=='MOT20-02']
    print(json.dumps({'rows':len(df),'features':len(features),'m02':m02,'aggregate':agg,'top_features':impdf.head(15).to_dict(orient='records')},indent=2))

if __name__=='__main__': main()
