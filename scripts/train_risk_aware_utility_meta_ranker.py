from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
FEATURES=[
 'utility_prob_hgb_norm','utility_prob_et_norm','utility_prob_hgb_norm_seqpct','utility_prob_et_norm_seqpct',
 'appearance_consensus','observable_link_prior','out_prior_rankpct','in_prior_rankpct',
 'appearance_max','appearance_mean','appearance_std','cos_end_start','cos_global_global','cos_high_high',
 'appearance_max_seqpct','appearance_mean_seqpct','cos_end_start_seqpct','cos_global_global_seqpct','cos_high_high_seqpct',
 'out_appearance_max_rankpct','in_appearance_max_rankpct','out_cos_end_start_rankpct','in_cos_end_start_rankpct',
 'predicted_step_seqpct','center_step_seqpct','height_log_abs_seqpct','area_log_abs_seqpct','gap_seqpct',
 'max_debt_pct','min_debt_pct','source_overlap_ioa_max','target_overlap_ioa_max','endpoint_reid_aligned'
]


def array(df):
 return df[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)


def fit_cls(X,y,seed):
 pos=max(1,int(y.sum()));neg=max(1,len(y)-pos)
 m=ExtraTreesClassifier(n_estimators=350,min_samples_leaf=3,max_features=.8,n_jobs=-1,class_weight={0:1.0,1:min(180.0,neg/pos)},random_state=seed)
 m.fit(X,y);return m


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scores',required=True);ap.add_argument('--out-dir',required=True);args=ap.parse_args()
 out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(args.scores);X=array(df);delta=df.assa_merge_delta_proxy.to_numpy(float)
 y_pos=(delta>0).astype(np.int8);y_cat10=(delta<=-10).astype(np.int8);y_cat50=(delta<=-50).astype(np.int8)
 y_signed=np.sign(delta)*np.log1p(np.minimum(np.abs(delta),200.0))
 ppos=np.zeros(len(df));pc10=np.zeros(len(df));pc50=np.zeros(len(df));preg=np.zeros(len(df));folds=[]
 for i,seq in enumerate(SEQS):
  tr=(df.seq!=seq).to_numpy();va=(df.seq==seq).to_numpy()
  mp=fit_cls(X[tr],y_pos[tr],100+i);m10=fit_cls(X[tr],y_cat10[tr],200+i);m50=fit_cls(X[tr],y_cat50[tr],300+i)
  reg=HistGradientBoostingRegressor(max_iter=180,learning_rate=.045,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=3.0,random_state=400+i,early_stopping=True,validation_fraction=.12,n_iter_no_change=15)
  reg.fit(X[tr],y_signed[tr],sample_weight=1+np.minimum(5,np.abs(y_signed[tr])))
  ppos[va]=mp.predict_proba(X[va])[:,1];pc10[va]=m10.predict_proba(X[va])[:,1];pc50[va]=m50.predict_proba(X[va])[:,1];preg[va]=reg.predict(X[va])
  folds.append({'seq':seq,'rows':int(va.sum()),'positive':int(y_pos[va].sum()),'cat10':int(y_cat10[va].sum()),'cat50':int(y_cat50[va].sum()),'pos_ap':float(average_precision_score(y_pos[va],ppos[va])),'pos_auc':float(roc_auc_score(y_pos[va],ppos[va])),'cat10_auc':float(roc_auc_score(y_cat10[va],pc10[va])),'cat50_auc':float(roc_auc_score(y_cat50[va],pc50[va]))})
 df['meta_p_benefit']=ppos;df['meta_p_cat10']=pc10;df['meta_p_cat50']=pc50;df['meta_signed_log_utility']=preg
 for lam in [.25,.5,1.0,2.0]:
  df[f'meta_risk_score_l{str(lam).replace(".","p")}']=ppos-lam*pc10-2*lam*pc50
 df['meta_safe_benefit']=ppos*(1-pc10)*(1-pc50)
 for c in ['meta_p_benefit','meta_p_cat10','meta_p_cat50','meta_signed_log_utility','meta_safe_benefit','meta_risk_score_l0p25','meta_risk_score_l0p5','meta_risk_score_l1p0','meta_risk_score_l2p0']:
  df[c+'_seqpct']=df.groupby('seq')[c].rank(pct=True,method='average')
 df.to_csv(out/'oof_risk_aware_scores.csv',index=False)
 mp=fit_cls(X,y_pos,100);m10=fit_cls(X,y_cat10,200);m50=fit_cls(X,y_cat50,300)
 reg=HistGradientBoostingRegressor(max_iter=180,learning_rate=.045,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=3.0,random_state=400)
 reg.fit(X,y_signed,sample_weight=1+np.minimum(5,np.abs(y_signed)))
 joblib.dump({'features':FEATURES,'benefit':mp,'cat10':m10,'cat50':m50,'signed_log_utility':reg},out/'risk_aware_utility_meta_ranker.joblib',compress=3)
 report={'features':FEATURES,'folds':folds,'mean_pos_ap':float(np.mean([x['pos_ap'] for x in folds])),'mean_pos_auc':float(np.mean([x['pos_auc'] for x in folds])),'mean_cat10_auc':float(np.mean([x['cat10_auc'] for x in folds])),'mean_cat50_auc':float(np.mean([x['cat50_auc'] for x in folds]))}
 (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
