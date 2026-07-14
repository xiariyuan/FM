from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
ID_COLS={'seq','track_a','track_b','frame'}
LABEL_PREFIXES=(
    'prev_gt_','prev_match_','keep_support_','swap_support_','observed_support_',
    'swap_utility_','swap_positive_','swap_strong_'
)


def load_dataset(root: Path) -> pd.DataFrame:
    parts=[]
    for seq in SEQS:
        p=root/f'{seq}.csv'
        x=pd.read_csv(p)
        parts.append(x)
    return pd.concat(parts,ignore_index=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols=[]
    for c in df.columns:
        if c in ID_COLS or c.startswith(LABEL_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def array(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return df[features].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)


def fit_pos_models(X,y,seed):
    pos=max(1,int(y.sum()));neg=max(1,len(y)-pos)
    et=ExtraTreesClassifier(
        n_estimators=500,max_features=.75,min_samples_leaf=2,n_jobs=-1,
        class_weight={0:1.0,1:min(120.0,neg/pos)},random_state=seed
    )
    et.fit(X,y)
    hgb=HistGradientBoostingClassifier(
        max_iter=220,learning_rate=.045,max_leaf_nodes=21,min_samples_leaf=12,
        l2_regularization=3.0,early_stopping=True,validation_fraction=.15,
        n_iter_no_change=20,random_state=seed
    )
    sw=np.where(y==1,min(100.0,.6*neg/pos),1.0)
    hgb.fit(X,y,sample_weight=sw)
    return et,hgb


def fit_risk_model(X,y,seed):
    pos=max(1,int(y.sum()));neg=max(1,len(y)-pos)
    et=ExtraTreesClassifier(
        n_estimators=450,max_features=.75,min_samples_leaf=2,n_jobs=-1,
        class_weight={0:1.0,1:min(80.0,neg/pos)},random_state=seed
    )
    et.fit(X,y)
    return et


def rank_metrics(frame: pd.DataFrame, score: np.ndarray) -> dict:
    y=(frame.swap_utility_30.to_numpy(float)>0).astype(np.int8)
    delta=frame.swap_utility_30.to_numpy(float)
    out={
        'ap':float(average_precision_score(y,score)),
        'auc':float(roc_auc_score(y,score)),
        'topk':[]
    }
    order=np.argsort(-score,kind='mergesort')
    for k in [1,2,3,5,10,15,20,30,50,75,100]:
        n=min(k,len(order));idx=order[:n];d=delta[idx]
        out['topk'].append({
            'k':k,'selected':n,'positive':int((d>0).sum()),
            'precision':float((d>0).mean()) if n else 0.0,
            'utility_sum':float(d.sum()),
            'positive_utility':float(d[d>0].sum()),
            'negative_utility':float(d[d<0].sum()),
            'worst':float(d.min()) if n else 0.0,
        })
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset-root',required=True)
    ap.add_argument('--out-dir',required=True)
    args=ap.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    df=load_dataset(Path(args.dataset_root))
    features=feature_columns(df)
    X=array(df,features)
    y_pos=(df.swap_utility_30.to_numpy(float)>0).astype(np.int8)
    y_risk=(df.swap_utility_30.to_numpy(float)<=-4).astype(np.int8)
    signed=np.sign(df.swap_utility_30.to_numpy(float))*np.log1p(np.minimum(np.abs(df.swap_utility_30.to_numpy(float)),100.0))
    p_et=np.zeros(len(df));p_hgb=np.zeros(len(df));p_risk=np.zeros(len(df));pred_signed=np.zeros(len(df));folds=[]
    for i,seq in enumerate(SEQS):
        tr=(df.seq!=seq).to_numpy();va=(df.seq==seq).to_numpy()
        et,hgb=fit_pos_models(X[tr],y_pos[tr],100+i)
        risk=fit_risk_model(X[tr],y_risk[tr],200+i)
        reg=HistGradientBoostingRegressor(
            max_iter=220,learning_rate=.04,max_leaf_nodes=21,min_samples_leaf=12,
            l2_regularization=3.0,early_stopping=True,validation_fraction=.15,
            n_iter_no_change=20,random_state=300+i
        )
        reg.fit(X[tr],signed[tr],sample_weight=1+np.minimum(5,np.abs(signed[tr])))
        p_et[va]=et.predict_proba(X[va])[:,1]
        p_hgb[va]=hgb.predict_proba(X[va])[:,1]
        p_risk[va]=risk.predict_proba(X[va])[:,1]
        pred_signed[va]=reg.predict(X[va])
        fold=df[va]
        folds.append({
            'seq':seq,'rows':int(va.sum()),'positive':int(y_pos[va].sum()),'risk':int(y_risk[va].sum()),
            'et':rank_metrics(fold,p_et[va]),'hgb':rank_metrics(fold,p_hgb[va]),
            'risk_auc':float(roc_auc_score(y_risk[va],p_risk[va])) if len(np.unique(y_risk[va]))>1 else None,
        })
    df['swap_p_et']=p_et
    df['swap_p_hgb']=p_hgb
    df['swap_p_risk4']=p_risk
    df['swap_signed_log_pred']=pred_signed
    df['swap_safe_et']=p_et*(1-p_risk)
    df['swap_safe_hgb']=p_hgb*(1-p_risk)
    for lam in [.25,.5,1.0,2.0]:
        tag=str(lam).replace('.','p')
        df[f'swap_risk_score_et_l{tag}']=p_et-lam*p_risk
        df[f'swap_risk_score_hgb_l{tag}']=p_hgb-lam*p_risk
    score_cols=[c for c in df.columns if c.startswith('swap_p_') or c.startswith('swap_safe_') or c.startswith('swap_risk_score_') or c=='swap_signed_log_pred']
    for c in score_cols:
        df[c+'_seqpct']=df.groupby('seq')[c].rank(pct=True,method='average')
    df.to_csv(out/'oof_swap_scores.csv',index=False)
    et,hgb=fit_pos_models(X,y_pos,100)
    risk=fit_risk_model(X,y_risk,200)
    reg=HistGradientBoostingRegressor(max_iter=220,learning_rate=.04,max_leaf_nodes=21,min_samples_leaf=12,l2_regularization=3.0,random_state=300)
    reg.fit(X,signed,sample_weight=1+np.minimum(5,np.abs(signed)))
    joblib.dump({'features':features,'positive_et':et,'positive_hgb':hgb,'risk4':risk,'signed_log':reg},out/'swap_transaction_ranker.joblib',compress=3)
    summary={
        'rows':len(df),'features':len(features),'positive':int(y_pos.sum()),'risk4':int(y_risk.sum()),
        'mean_et_ap':float(np.mean([f['et']['ap'] for f in folds])),
        'mean_et_auc':float(np.mean([f['et']['auc'] for f in folds])),
        'mean_hgb_ap':float(np.mean([f['hgb']['ap'] for f in folds])),
        'mean_hgb_auc':float(np.mean([f['hgb']['auc'] for f in folds])),
        'mean_risk_auc':float(np.mean([f['risk_auc'] for f in folds if f['risk_auc'] is not None])),
        'folds':folds,
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
