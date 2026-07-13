#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, pickle
from pathlib import Path
from collections import defaultdict
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

BASE_FEATURES = [
    'gap','center_distance','center_distance_per_frame','predicted_distance','predicted_distance_per_frame',
    'velocity_cosine','height_ratio','area_ratio','bottom_y_gap','len_a','len_b','duration_a','duration_b',
    'avg_score_a','avg_score_b','last_score_a','first_score_b'
]
REID_FEATURES = [
    'cos_end_start','cos_end_global','cos_global_start','cos_global_global','cos_high_high',
    'cos_start_start','cos_end_end','appearance_mean','appearance_max','appearance_min',
    'appearance_std','appearance_gap_consistency','has_reid'
]
FEATURES = BASE_FEATURES + REID_FEATURES

def read_rows(path: Path):
    rows=[]
    with path.open() as f:
        for r in csv.DictReader(f):
            rr=dict(r)
            rr['same_gt']=int(float(rr['same_gt']))
            for k in FEATURES:
                rr[k]=float(rr.get(k, 0.0) or 0.0)
            rows.append(rr)
    return rows

def make_xy(rows):
    X=np.array([[r[k] for k in FEATURES] for r in rows], dtype=float)
    y=np.array([r['same_gt'] for r in rows], dtype=int)
    return X,y

def model_factory(kind):
    if kind=='logistic':
        return Pipeline([('scaler',StandardScaler()),('clf',LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0))])
    if kind=='rf':
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=8, class_weight='balanced_subsample', random_state=7, n_jobs=-1)
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.1, random_state=7)

def proba(model,X):
    if hasattr(model,'predict_proba'):
        return model.predict_proba(X)[:,1]
    s=model.decision_function(X)
    return 1/(1+np.exp(-s))

def topk_metrics(y, p, ks=(5,10,20,50,100)):
    order=np.argsort(-p)
    out={}
    for k in ks:
        kk=min(k,len(y)); sel=order[:kk]
        out[f'precision_at_{k}']=float(y[sel].mean()) if kk else 0.0
        out[f'tp_at_{k}']=int(y[sel].sum()) if kk else 0
    return out

def threshold_metrics(y,p,ths=(0.5,0.7,0.8,0.9,0.95,0.98,0.99)):
    out={}
    for t in ths:
        pred=p>=t
        n=int(pred.sum()); tp=int(y[pred].sum()) if n else 0
        out[f'n_ge_{t}']=n; out[f'tp_ge_{t}']=tp; out[f'precision_ge_{t}']=float(tp/n) if n else 0.0
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pairs',required=True); ap.add_argument('--out-dir',required=True); ap.add_argument('--model-kind',default='hgb',choices=['hgb','logistic','rf']); args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows=read_rows(Path(args.pairs)); seqs=sorted(set(r['seq'] for r in rows))
    fold_reports=[]; oof=[]
    for seq in seqs:
        tr=[r for r in rows if r['seq']!=seq]; va=[r for r in rows if r['seq']==seq]
        Xtr,ytr=make_xy(tr); Xva,yva=make_xy(va)
        m=model_factory(args.model_kind); m.fit(Xtr,ytr); pv=proba(m,Xva)
        rep={'val_seq':seq,'train_rows':len(tr),'val_rows':len(va),'train_pos':int(ytr.sum()),'val_pos':int(yva.sum()),'val_positive_rate':float(yva.mean()) if len(yva) else 0.0}
        try: rep['roc_auc']=float(roc_auc_score(yva,pv)) if len(set(yva))>1 else None
        except Exception: rep['roc_auc']=None
        try: rep['pr_auc']=float(average_precision_score(yva,pv)) if len(set(yva))>1 else None
        except Exception: rep['pr_auc']=None
        rep.update(topk_metrics(yva,pv)); rep.update(threshold_metrics(yva,pv))
        fold_reports.append(rep)
        for r,score in zip(va,pv):
            rr={k:r[k] for k in r.keys()}; rr['aflink_score']=float(score); oof.append(rr)
    # final model on all rows
    X,y=make_xy(rows); final=model_factory(args.model_kind); final.fit(X,y)
    with (out/'aflink_lite_model.pkl').open('wb') as f: pickle.dump({'model':final,'features':FEATURES,'model_kind':args.model_kind}, f)
    with (out/'fold_report.json').open('w') as f: json.dump(fold_reports,f,indent=2,sort_keys=True)
    with (out/'oof_predictions.csv').open('w', newline='') as f:
        fields=list(oof[0].keys()) if oof else []
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(oof)
    # aggregate
    y_o=np.array([r['same_gt'] for r in oof],dtype=int); p_o=np.array([r['aflink_score'] for r in oof],dtype=float)
    summary={'model_kind':args.model_kind,'rows':len(rows),'positives':int(y.sum()),'positive_rate':float(y.mean()),'folds':fold_reports}
    if len(set(y_o))>1:
        summary['oof_roc_auc']=float(roc_auc_score(y_o,p_o)); summary['oof_pr_auc']=float(average_precision_score(y_o,p_o))
    summary.update(topk_metrics(y_o,p_o,ks=(20,50,100,200,500)))
    summary.update(threshold_metrics(y_o,p_o))
    with (out/'summary.json').open('w') as f: json.dump(summary,f,indent=2,sort_keys=True)
    md=['# AFLink-lite OOF Training Summary','',f"model_kind: {args.model_kind}",f"rows: {len(rows)}",f"positives: {int(y.sum())}",f"positive_rate: {float(y.mean()):.4f}",f"oof_pr_auc: {summary.get('oof_pr_auc')}",f"oof_roc_auc: {summary.get('oof_roc_auc')}",'','## OOF top-k / thresholds','| metric | value |','|---|---:|']
    for k in ['precision_at_20','tp_at_20','precision_at_50','tp_at_50','precision_at_100','tp_at_100','precision_at_200','tp_at_200','precision_at_500','tp_at_500','n_ge_0.9','tp_ge_0.9','precision_ge_0.9','n_ge_0.95','tp_ge_0.95','precision_ge_0.95','n_ge_0.98','tp_ge_0.98','precision_ge_0.98','n_ge_0.99','tp_ge_0.99','precision_ge_0.99']:
        md.append(f"| {k} | {summary.get(k)} |")
    md += ['','## Folds','| val_seq | val_rows | val_pos | pr_auc | p@20 | tp@20 | p@50 | tp@50 | n>=0.98 | p>=0.98 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in fold_reports:
        md.append(f"| {r['val_seq']} | {r['val_rows']} | {r['val_pos']} | {r.get('pr_auc')} | {r.get('precision_at_20')} | {r.get('tp_at_20')} | {r.get('precision_at_50')} | {r.get('tp_at_50')} | {r.get('n_ge_0.98')} | {r.get('precision_ge_0.98')} |")
    (out/'summary.md').write_text('\n'.join(md)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()
