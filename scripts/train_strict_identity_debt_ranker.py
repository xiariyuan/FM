from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score

SEQS = ['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
LABELS = {
    'matched_rows','dominant_gt','dominant_gt_rows','track_gt_purity','track_unique_gt',
    'repair_debt_rows','repair_debt_ratio','family_debt_rows','family_debt_ratio',
    'family_unique_tracker_ids','is_dominant_tid_for_gt'
}
IDS = {'seq','track_id'}
# Artifacts caused by reusing features from an earlier track version. They are not part of the method.
BANNED = {
    'reid_index_len_ratio','reid_start_match','reid_end_match','reid_reliable',
    'has_reid','reid_samples'
}


def metrics(df: pd.DataFrame, score: np.ndarray) -> dict:
    y=(df.family_debt_rows>=100).astype(int).to_numpy()
    out={
        'spearman_family_debt':float(spearmanr(score,df.family_debt_rows).statistic),
        'ap_family_debt_ge100':float(average_precision_score(y,score)),
        'auc_family_debt_ge100':float(roc_auc_score(y,score)),
        'topk':[]
    }
    fam=df.groupby('dominant_gt',as_index=False).family_debt_rows.max()
    for frac in [.1,.2,.3,.5]:
        n=max(1,round(len(df)*frac)); order=np.argsort(-score,kind='mergesort'); sel=df.iloc[order[:n]]
        sf=set(sel.dominant_gt.astype(int))
        out['topk'].append({
            'fraction':frac,'selected_tracks':n,
            'track_debt_recall':float(sel.repair_debt_rows.sum()/max(1,df.repair_debt_rows.sum())),
            'family_debt_recall':float(fam[fam.dominant_gt.isin(sf)].family_debt_rows.sum()/max(1,fam.family_debt_rows.sum())),
            'precision_family_debt_ge100':float((sel.family_debt_rows>=100).mean()),
            'unique_families':len(sf),
        })
    return out


def fit_model(train: pd.DataFrame, features: list[str]) -> tuple[SimpleImputer,RandomForestRegressor]:
    imp=SimpleImputer(strategy='median')
    X=imp.fit_transform(train[features])
    y=np.log1p(train.family_debt_rows.to_numpy(float))
    seq_counts=train.groupby('seq').size().to_dict()
    w=np.asarray([1.0/seq_counts[s] for s in train.seq],dtype=float); w*=len(w)/w.sum()
    model=RandomForestRegressor(
        n_estimators=600,min_samples_leaf=5,max_features=.7,n_jobs=-1,
        random_state=42,oob_score=True,bootstrap=True
    )
    model.fit(X,y,sample_weight=w)
    return imp,model


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',required=True)
    ap.add_argument('--out-dir',required=True)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.dataset)
    features=[c for c in df.columns if c not in LABELS|IDS|BANNED and pd.api.types.is_numeric_dtype(df[c])]
    folds=[]; pred_parts=[]
    for holdout in SEQS:
        train=df[df.seq!=holdout].copy(); test=df[df.seq==holdout].copy().reset_index(drop=True)
        imp,model=fit_model(train,features)
        score=np.maximum(0,np.expm1(model.predict(imp.transform(test[features]))))
        rec={'holdout':holdout}; rec.update(metrics(test,score)); folds.append(rec)
        p=test[['seq','track_id','dominant_gt','repair_debt_rows','family_debt_rows']].copy()
        p['score']=score; p['rank']=pd.Series(score).rank(ascending=False,method='first').astype(int)
        pred_parts.append(p)
    imp,model=fit_model(df,features)
    bundle={
        'feature_columns':features,'imputer':imp,'model':model,
        'target':'log1p(family_debt_rows)','train_sequences':SEQS,
        'banned_columns':sorted(BANNED),
        'feature_policy':'tracker-observable; no GT; no cross-version ReID alignment metadata'
    }
    joblib.dump(bundle,out/'strict_family_debt_rf.joblib',compress=3)
    importance=pd.DataFrame({'feature':features,'importance':model.feature_importances_}).sort_values('importance',ascending=False)
    importance.to_csv(out/'feature_importance.csv',index=False)
    pd.concat(pred_parts,ignore_index=True).to_csv(out/'loo_predictions.csv',index=False)
    aggregate={
        'mean_spearman':float(np.mean([r['spearman_family_debt'] for r in folds])),
        'mean_ap':float(np.mean([r['ap_family_debt_ge100'] for r in folds])),
        'mean_auc':float(np.mean([r['auc_family_debt_ge100'] for r in folds])),
        'topk':{}
    }
    for frac in [.1,.2,.3,.5]:
        vals=[next(x for x in r['topk'] if x['fraction']==frac) for r in folds]
        aggregate['topk'][str(frac)]={k:float(np.mean([v[k] for v in vals])) for k in ['track_debt_recall','family_debt_recall','precision_family_debt_ge100']}
    report={
        'protocol':bundle['feature_policy'],'feature_count':len(features),'features':features,
        'folds':folds,'aggregate':aggregate,'oob_score_full_train':float(model.oob_score_),
        'top20_features':importance.head(20).to_dict(orient='records'),
        'model_path':str(out/'strict_family_debt_rf.joblib')
    }
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'feature_count':len(features),'aggregate':aggregate,'m02':next(r for r in folds if r['holdout']=='MOT20-02'),'top15':report['top20_features'][:15]},indent=2))

if __name__=='__main__': main()
