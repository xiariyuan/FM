#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

FEATURES=['row_count','duration','num_gaps','missing_gap_frames','max_gap','interpolated_count','interpolated_fraction','avg_score','min_score','max_score','median_score','score_std','avg_area','median_area','area_std','avg_height','median_height','height_std','avg_width','avg_bottom_y','avg_center_speed','max_center_speed','center_speed_std']

def read_rows(path, condition):
    rows=[]
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['condition']!=condition: continue
            rr=dict(r); rr['is_bad']=1 if rr['quality_label']=='bad_track' else 0
            for k in FEATURES: rr[k]=float(rr[k])
            rows.append(rr)
    return rows

def xy(rows): return np.array([[r[k] for k in FEATURES] for r in rows],float), np.array([r['is_bad'] for r in rows],int)
def model_factory(kind):
    if kind=='logistic': return Pipeline([('scaler',StandardScaler()),('clf',LogisticRegression(max_iter=2000,class_weight='balanced'))])
    if kind=='rf': return RandomForestClassifier(n_estimators=400,min_samples_leaf=5,class_weight='balanced_subsample',random_state=7,n_jobs=-1)
    return HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,max_leaf_nodes=15,l2_regularization=0.1,random_state=7)
def proba(m,X): return m.predict_proba(X)[:,1]
def metrics(y,p,ths=(0.5,0.7,0.8,0.9,0.95,0.98,0.99)):
    out={}
    order=np.argsort(-p)
    for k in [10,20,50,100,200]:
        sel=order[:min(k,len(y))]; out[f'top{k}_n']=len(sel); out[f'top{k}_tp']=int(y[sel].sum()); out[f'top{k}_precision']=float(y[sel].mean()) if len(sel) else 0.0
    for t in ths:
        pred=p>=t; n=int(pred.sum()); tp=int(y[pred].sum()) if n else 0
        out[f'n_ge_{t}']=n; out[f'tp_ge_{t}']=tp; out[f'precision_ge_{t}']=float(tp/n) if n else 0.0
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tracks',required=True); ap.add_argument('--condition',default='interp_gap30'); ap.add_argument('--out-dir',required=True); ap.add_argument('--model-kind',default='hgb',choices=['hgb','rf','logistic']); args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=read_rows(args.tracks,args.condition); seqs=sorted(set(r['seq'] for r in rows)); oof=[]; folds=[]
    for seq in seqs:
        tr=[r for r in rows if r['seq']!=seq]; va=[r for r in rows if r['seq']==seq]
        Xtr,ytr=xy(tr); Xva,yva=xy(va); m=model_factory(args.model_kind); m.fit(Xtr,ytr); pv=proba(m,Xva)
        rep={'val_seq':seq,'train_rows':len(tr),'val_rows':len(va),'train_bad':int(ytr.sum()),'val_bad':int(yva.sum()),'val_bad_rate':float(yva.mean())}
        if len(set(yva))>1:
            rep['roc_auc']=float(roc_auc_score(yva,pv)); rep['pr_auc']=float(average_precision_score(yva,pv))
        rep.update(metrics(yva,pv)); folds.append(rep)
        for r,s in zip(va,pv):
            rr=dict(r); rr['bad_score']=float(s); oof.append(rr)
    X,y=xy(rows); final=model_factory(args.model_kind); final.fit(X,y)
    with (out/'track_quality_filter.pkl').open('wb') as f: pickle.dump({'model':final,'features':FEATURES,'condition':args.condition,'model_kind':args.model_kind},f)
    with (out/'oof_predictions.csv').open('w',newline='') as f:
        fields=list(oof[0].keys()); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(oof)
    y_o=np.array([r['is_bad'] for r in oof],int); p_o=np.array([r['bad_score'] for r in oof],float)
    summary={'condition':args.condition,'model_kind':args.model_kind,'rows':len(rows),'bad':int(y.sum()),'bad_rate':float(y.mean()),'folds':folds}
    if len(set(y_o))>1:
        summary['oof_roc_auc']=float(roc_auc_score(y_o,p_o)); summary['oof_pr_auc']=float(average_precision_score(y_o,p_o))
    summary.update(metrics(y_o,p_o))
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    md=['# Track Quality Filter OOF Summary','',f"condition: {args.condition}",f"model: {args.model_kind}",f"rows: {len(rows)}",f"bad: {int(y.sum())}",f"bad_rate: {float(y.mean()):.4f}",f"oof_pr_auc: {summary.get('oof_pr_auc')}",f"oof_roc_auc: {summary.get('oof_roc_auc')}",'','## OOF','| metric | value |','|---|---:|']
    for k in ['top10_precision','top10_tp','top20_precision','top20_tp','top50_precision','top50_tp','top100_precision','top100_tp','n_ge_0.9','tp_ge_0.9','precision_ge_0.9','n_ge_0.95','tp_ge_0.95','precision_ge_0.95','n_ge_0.98','tp_ge_0.98','precision_ge_0.98','n_ge_0.99','tp_ge_0.99','precision_ge_0.99']:
        md.append(f"| {k} | {summary.get(k)} |")
    md += ['','## Folds','| val_seq | rows | bad | pr_auc | top20_p | top20_tp | n>=0.98 | p>=0.98 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in folds: md.append(f"| {r['val_seq']} | {r['val_rows']} | {r['val_bad']} | {r.get('pr_auc')} | {r.get('top20_precision')} | {r.get('top20_tp')} | {r.get('n_ge_0.98')} | {r.get('precision_ge_0.98')} |")
    (out/'summary.md').write_text('\n'.join(md)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()
