from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
LABELS={
 'dominant_gt_a','dominant_gt_b','purity_a','purity_b','same_gt',
 'assa_merge_before_proxy','assa_merge_after_proxy','assa_merge_delta_proxy',
 'assa_merge_positive','assa_merge_delta_per_row','shares_any_gt','combined_dominant_gt'
}
IDS={'split','seq','track_a','track_b'}
HIGH=['appearance_max','appearance_mean','cos_end_start','cos_global_global','cos_high_high','cos_end_global','cos_global_start','max_debt_pct','min_debt_pct']
LOW=['predicted_step','center_step','height_log_abs','area_log_abs','gap','appearance_std']
STABLE_RAW=[
 'log_gap','density_a','density_b','score_mean_a','score_mean_b','score_min_a','score_min_b',
 'height_ratio','height_log_abs','area_ratio','area_log_abs','velocity_cosine',
 'source_debt_pct','target_debt_pct','max_debt_pct','min_debt_pct',
 'source_overlap_ioa_max','target_overlap_ioa_max','source_appearance_drift','target_appearance_drift',
 'endpoint_reid_aligned','cos_end_start','cos_global_global','cos_high_high','cos_end_global',
 'cos_global_start','appearance_mean','appearance_max','appearance_min','appearance_std'
]


def add_normalized_features(df:pd.DataFrame)->pd.DataFrame:
 x=df.copy()
 numeric=[c for c in x.columns if c not in LABELS|IDS and pd.api.types.is_numeric_dtype(x[c])]
 # Percentiles within each sequence are deployable and remove scene scale.
 for c in numeric:
  x[c+'_seqpct']=x.groupby('seq')[c].rank(pct=True,method='average').fillna(.5)
 # Relative candidate ranks for each source and target.
 for c in HIGH+LOW:
  if c not in x: continue
  asc=c in LOW
  for side,key in [('out','track_a'),('in','track_b')]:
   group=x.groupby(['seq',key])[c]
   x[f'{side}_{c}_rankpct']=group.rank(pct=True,ascending=asc,method='average').fillna(1.0)
   best=group.transform('min' if asc else 'max')
   x[f'{side}_{c}_gapbest']=(x[c]-best).abs()
 # A scene-normalized hand score used only as a feature, not as a label.
 x['appearance_consensus']=(x['appearance_max']+x['appearance_mean']+x['cos_end_start']+x['cos_global_global'])/4
 x['geometry_consistency']=-(x['predicted_step_seqpct']+x['center_step_seqpct']+x['height_log_abs_seqpct']+x['area_log_abs_seqpct'])/4
 x['observable_link_prior']=x['appearance_max_seqpct']+.5*x['appearance_mean_seqpct']+.5*x['max_debt_pct']+.25*x['geometry_consistency']
 for side,key in [('out','track_a'),('in','track_b')]:
  x[f'{side}_prior_rankpct']=x.groupby(['seq',key])['observable_link_prior'].rank(pct=True,ascending=False,method='average')
 return x


def feature_columns(df:pd.DataFrame)->list[str]:
 engineered=[c for c in df.columns if c.endswith('_seqpct') or c.endswith('_rankpct') or c.endswith('_gapbest')]
 extras=['appearance_consensus','geometry_consistency','observable_link_prior']
 return [c for c in STABLE_RAW+engineered+extras if c in df and c not in LABELS]


def arr(df,features):
 return df[features].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)


def fit_models(X,y,seed):
 pos=max(1,int(y.sum()));neg=max(1,len(y)-pos)
 hgb=HistGradientBoostingClassifier(max_iter=180,learning_rate=.045,max_leaf_nodes=31,min_samples_leaf=25,l2_regularization=3.0,random_state=seed,early_stopping=True,validation_fraction=.12,n_iter_no_change=15)
 sw=np.where(y==1,min(180.0,.45*neg/pos),1.0)
 hgb.fit(X,y,sample_weight=sw)
 et=ExtraTreesClassifier(n_estimators=350,min_samples_leaf=3,max_features=.75,n_jobs=-1,class_weight={0:1.0,1:min(160.0,neg/pos)},random_state=seed)
 et.fit(X,y)
 return hgb,et


def ranking_metrics(frame,score):
 y=frame.assa_merge_positive.to_numpy(np.int8);delta=frame.assa_merge_delta_proxy.to_numpy(float)
 out={'ap':float(average_precision_score(y,score)),'auc':float(roc_auc_score(y,score)),'topk':[]}
 order=np.argsort(-score,kind='mergesort')
 for k in [1,2,3,5,10,20,30,50]:
  n=min(k,len(order));idx=order[:n];d=delta[idx]
  out['topk'].append({'k':k,'positive':int((d>0).sum()),'positive_rate':float((d>0).mean()),'utility_sum':float(d.sum()),'positive_utility_sum':float(d[d>0].sum()),'negative_utility_sum':float(d[d<0].sum())})
 return out


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--out-dir',required=True);args=ap.parse_args()
 out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
 raw=pd.read_csv(args.dataset);df=add_normalized_features(raw);features=feature_columns(df);X=arr(df,features);y=df.assa_merge_positive.to_numpy(np.int8)
 ph=np.zeros(len(df));pe=np.zeros(len(df));folds=[]
 for i,seq in enumerate(SEQS):
  tr=(df.seq!=seq).to_numpy();va=(df.seq==seq).to_numpy();hgb,et=fit_models(X[tr],y[tr],42+i)
  sh=hgb.predict_proba(X[va])[:,1];se=et.predict_proba(X[va])[:,1];ph[va]=sh;pe[va]=se
  folds.append({'seq':seq,'rows':int(va.sum()),'positive':int(y[va].sum()),'hgb':ranking_metrics(df[va],sh),'extra_trees':ranking_metrics(df[va],se)})
 df['utility_prob_hgb_norm']=ph;df['utility_prob_et_norm']=pe
 for c in ['utility_prob_hgb_norm','utility_prob_et_norm']:
  df[c+'_seqpct']=df.groupby('seq')[c].rank(pct=True,method='average')
 keep=list(raw.columns)+[c for c in df.columns if c not in raw.columns]
 df[keep].to_csv(out/'oof_normalized_utility_scores.csv',index=False)
 hgb,et=fit_models(X,y,42);joblib.dump({'features':features,'hgb':hgb,'extra_trees':et,'normalization':'sequence percentiles and source/target candidate ranks','train_sequences':SEQS},out/'normalized_utility_ranker.joblib',compress=3)
 summary={'feature_count':len(features),'features':features,'folds':folds,'mean_hgb_ap':float(np.mean([f['hgb']['ap'] for f in folds])),'mean_hgb_auc':float(np.mean([f['hgb']['auc'] for f in folds])),'mean_et_ap':float(np.mean([f['extra_trees']['ap'] for f in folds])),'mean_et_auc':float(np.mean([f['extra_trees']['auc'] for f in folds]))}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({'feature_count':len(features),'means':{k:summary[k] for k in ['mean_hgb_ap','mean_hgb_auc','mean_et_ap','mean_et_auc']},'folds':folds},indent=2))
if __name__=='__main__':main()
