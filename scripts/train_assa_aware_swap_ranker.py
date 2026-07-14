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
HORIZONS=['h10','h30','h60','perm']
RISK_THRESH={'h10':-5.0,'h30':-15.0,'h60':-30.0,'perm':-100.0}
ID_COLS={'seq','track_a','track_b','frame'}
LEAK_PREFIXES=(
    'prev_gt_','prev_match_','keep_support_','swap_support_','observed_support_',
    'swap_utility_','swap_positive_','swap_strong_',
    'assa_swap_','swap_tracker_count_','swap_matched_rows_'
)
# The original local utility and matched-GT descriptors are labels/audit data, never features.
LEAK_EXACT={'prev_gt_a','prev_gt_b','prev_match_iou_a','prev_match_iou_b','prev_match_frame_a','prev_match_frame_b'}


def feature_columns(df:pd.DataFrame)->list[str]:
    out=[]
    for c in df.columns:
        if c in ID_COLS or c in LEAK_EXACT or c.startswith(LEAK_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def matrix(df,features):
    return df[features].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)


def fit_benefit(X,y,seed):
    pos=max(1,int(y.sum()));neg=max(1,len(y)-pos);w=min(120.0,neg/pos)
    et=ExtraTreesClassifier(n_estimators=600,max_features=.75,min_samples_leaf=2,n_jobs=-1,class_weight={0:1,1:w},random_state=seed)
    et.fit(X,y)
    hgb=HistGradientBoostingClassifier(max_iter=260,learning_rate=.04,max_leaf_nodes=19,min_samples_leaf=12,l2_regularization=4.0,early_stopping=True,validation_fraction=.15,n_iter_no_change=25,random_state=seed)
    hgb.fit(X,y,sample_weight=np.where(y==1,min(100.0,.65*w),1.0))
    return et,hgb


def fit_risk(X,y,seed):
    pos=max(1,int(y.sum()));neg=max(1,len(y)-pos);w=min(100.0,neg/pos)
    et=ExtraTreesClassifier(n_estimators=500,max_features=.75,min_samples_leaf=2,n_jobs=-1,class_weight={0:1,1:w},random_state=seed)
    et.fit(X,y)
    return et


def metrics(y,score):
    return {'ap':float(average_precision_score(y,score)),'auc':float(roc_auc_score(y,score))}


def topk(delta,score):
    order=np.argsort(-score,kind='mergesort');rows=[]
    for k in [1,2,3,5,10,15,20,30,50,75,100]:
        idx=order[:min(k,len(order))];d=delta[idx]
        rows.append({'k':k,'selected':len(idx),'positive':int((d>0).sum()),'precision':float((d>0).mean()) if len(d) else 0.0,'delta_sum':float(d.sum()),'positive_delta':float(d[d>0].sum()),'negative_delta':float(d[d<0].sum()),'worst':float(d.min()) if len(d) else 0.0})
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--out-dir',required=True);args=ap.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.dataset);features=feature_columns(df);X=matrix(df,features)
    summary={'rows':len(df),'features':len(features),'horizons':{},'folds':[]};models={};all_score_cols=[]
    for hi,h in enumerate(HORIZONS):
        delta=df[f'assa_swap_delta_{h}_proxy'].to_numpy(float)
        y=(delta>0).astype(np.int8);risk=(delta<=RISK_THRESH[h]).astype(np.int8)
        p_et=np.zeros(len(df));p_hgb=np.zeros(len(df));p_risk=np.zeros(len(df));pred=np.zeros(len(df));fold_rows=[]
        for si,seq in enumerate(SEQS):
            tr=(df.seq!=seq).to_numpy();va=(df.seq==seq).to_numpy()
            et,hgb=fit_benefit(X[tr],y[tr],1000+100*hi+si);rm=fit_risk(X[tr],risk[tr],2000+100*hi+si)
            reg=HistGradientBoostingRegressor(max_iter=260,learning_rate=.04,max_leaf_nodes=19,min_samples_leaf=12,l2_regularization=4.0,early_stopping=True,validation_fraction=.15,n_iter_no_change=25,random_state=3000+100*hi+si)
            target=np.sign(delta[tr])*np.log1p(np.minimum(np.abs(delta[tr]),500.0))
            reg.fit(X[tr],target,sample_weight=1+np.minimum(6,np.abs(target)))
            p_et[va]=et.predict_proba(X[va])[:,1];p_hgb[va]=hgb.predict_proba(X[va])[:,1];p_risk[va]=rm.predict_proba(X[va])[:,1];pred[va]=reg.predict(X[va])
            fold_rows.append({'seq':seq,'rows':int(va.sum()),'positive':int(y[va].sum()),'risk':int(risk[va].sum()),'et':metrics(y[va],p_et[va]),'hgb':metrics(y[va],p_hgb[va]),'risk_auc':float(roc_auc_score(risk[va],p_risk[va])) if len(np.unique(risk[va]))>1 else None,'topk_et':topk(delta[va],p_et[va]),'topk_hgb':topk(delta[va],p_hgb[va])})
        base=f'assa_swap_{h}'
        df[base+'_p_et']=p_et;df[base+'_p_hgb']=p_hgb;df[base+'_p_risk']=p_risk;df[base+'_signed_log_pred']=pred
        df[base+'_safe_et']=p_et*(1-p_risk);df[base+'_safe_hgb']=p_hgb*(1-p_risk)
        for lam in [.1,.25,.5,1.0,2.0]:
            tag=str(lam).replace('.','p')
            df[f'{base}_risk_et_l{tag}']=p_et-lam*p_risk
            df[f'{base}_risk_hgb_l{tag}']=p_hgb-lam*p_risk
        cols=[c for c in df.columns if c.startswith(base+'_p_') or c.startswith(base+'_safe_') or c.startswith(base+'_risk_') or c==base+'_signed_log_pred']
        for c in cols:
            df[c+'_seqpct']=df.groupby('seq')[c].rank(pct=True,method='average')
        all_score_cols+=cols
        et,hgb=fit_benefit(X,y,1000+100*hi);rm=fit_risk(X,risk,2000+100*hi)
        reg=HistGradientBoostingRegressor(max_iter=260,learning_rate=.04,max_leaf_nodes=19,min_samples_leaf=12,l2_regularization=4.0,random_state=3000+100*hi)
        target=np.sign(delta)*np.log1p(np.minimum(np.abs(delta),500.0));reg.fit(X,target,sample_weight=1+np.minimum(6,np.abs(target)))
        models[h]={'benefit_et':et,'benefit_hgb':hgb,'risk':rm,'signed_log':reg,'risk_threshold':RISK_THRESH[h]}
        summary['horizons'][h]={'positive':int(y.sum()),'risk':int(risk.sum()),'mean_et_ap':float(np.mean([x['et']['ap'] for x in fold_rows])),'mean_et_auc':float(np.mean([x['et']['auc'] for x in fold_rows])),'mean_hgb_ap':float(np.mean([x['hgb']['ap'] for x in fold_rows])),'mean_hgb_auc':float(np.mean([x['hgb']['auc'] for x in fold_rows])),'mean_risk_auc':float(np.mean([x['risk_auc'] for x in fold_rows if x['risk_auc'] is not None]))}
        summary['folds']+= [{'horizon':h,**x} for x in fold_rows]
    df.to_csv(out/'oof_assa_swap_scores.csv',index=False)
    joblib.dump({'features':features,'models':models},out/'assa_aware_swap_ranker.joblib',compress=3)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['horizons'],indent=2),flush=True)

if __name__=='__main__':main()
