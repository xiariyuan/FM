#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

TRAIN_SEQS = ['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
ID_COLS = ['seq','track_a','track_b','gap','gap_bucket','same_gt']
FEATURE_COLS = [
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
    'max_rank_by_a42_rule_score_v1','min_margin_by_a42_rule_score_v1','max_rank_by_appearance_max','min_margin_by_appearance_max'
]


def ai(v, d=0):
    try:
        if v is None or v == '':
            return d
        return int(float(v))
    except Exception:
        return d


def af(v, d=0.0):
    try:
        if v is None or v == '':
            return d
        return float(v)
    except Exception:
        return d


def key_from_row(r) -> Tuple[str, str, str]:
    return (str(r['seq']), str(int(float(r['track_a']))), str(int(float(r['track_b']))))


def read_csv_rows(path: Path) -> List[dict]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    if not fields:
        fields = ['seq']
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def load_manifest(path: Path) -> pd.DataFrame:
    usecols = ID_COLS + FEATURE_COLS
    df = pd.read_csv(path, usecols=usecols)
    for c in FEATURE_COLS + ['same_gt']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    df['track_a'] = df['track_a'].astype(str)
    df['track_b'] = df['track_b'].astype(str)
    # derived transforms
    df['log_gap'] = np.log1p(df['gap'])
    df['inv_gap'] = 1.0 / np.maximum(1.0, df['gap'].to_numpy(dtype=float))
    df['score_minus_app'] = df['a42_rule_score_v1'] - df['appearance_max']
    return df


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = FEATURE_COLS + ['log_gap','inv_gap','score_minus_app']
    return df[cols].to_numpy(dtype=np.float32)


def make_hard_mask(df: pd.DataFrame) -> np.ndarray:
    return (
        (df['same_gt'].to_numpy() == 1) |
        (df['appearance_max'].to_numpy() >= 0.55) |
        (df['a42_rule_score_v1'].to_numpy() >= 0.45) |
        (df['max_rank_by_a42_rule_score_v1'].to_numpy() <= 20) |
        (df['cos_global_global'].to_numpy() >= 0.55)
    )


def train_model(X: np.ndarray, y: np.ndarray, seed: int) -> HistGradientBoostingClassifier:
    pos = max(1, int(y.sum()))
    neg = max(1, int(len(y) - y.sum()))
    pos_weight = min(80.0, max(10.0, 0.35 * neg / pos))
    sw = np.where(y == 1, pos_weight, 1.0).astype(np.float32)
    clf = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.15,
        min_samples_leaf=25,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=12,
    )
    clf.fit(X, y, sample_weight=sw)
    return clf


def add_model_rank_fields(score_df: pd.DataFrame) -> pd.DataFrame:
    df = score_df.copy()
    df['a42_model_score'] = pd.to_numeric(df['a42_model_score'], errors='coerce').fillna(0.0)
    # ranks descending, method first deterministic after sorting.
    df['_orig'] = np.arange(len(df))
    df = df.sort_values(['seq','track_a','a42_model_score','_orig'], ascending=[True, True, False, True])
    df['out_rank_by_a42_model_score'] = df.groupby(['seq','track_a']).cumcount() + 1
    best = df.groupby(['seq','track_a'])['a42_model_score'].transform('max')
    second = df.groupby(['seq','track_a'])['a42_model_score'].transform(lambda s: s.nlargest(2).iloc[-1] if len(s) > 1 else 0.0)
    df['out_best_a42_model_score'] = best
    df['out_second_a42_model_score'] = second
    df['out_margin_to_second_a42_model_score'] = np.where(df['out_rank_by_a42_model_score'] == 1, df['a42_model_score'] - second, df['a42_model_score'] - best)
    df = df.sort_values(['seq','track_b','a42_model_score','_orig'], ascending=[True, True, False, True])
    df['in_rank_by_a42_model_score'] = df.groupby(['seq','track_b']).cumcount() + 1
    best = df.groupby(['seq','track_b'])['a42_model_score'].transform('max')
    second = df.groupby(['seq','track_b'])['a42_model_score'].transform(lambda s: s.nlargest(2).iloc[-1] if len(s) > 1 else 0.0)
    df['in_best_a42_model_score'] = best
    df['in_second_a42_model_score'] = second
    df['in_margin_to_second_a42_model_score'] = np.where(df['in_rank_by_a42_model_score'] == 1, df['a42_model_score'] - second, df['a42_model_score'] - best)
    df['max_rank_by_a42_model_score'] = np.maximum(df['out_rank_by_a42_model_score'], df['in_rank_by_a42_model_score'])
    df['min_margin_by_a42_model_score'] = np.minimum(df['out_margin_to_second_a42_model_score'], df['in_margin_to_second_a42_model_score'])
    df = df.sort_values('_orig').drop(columns=['_orig'])
    return df


def source_file(source_dir: Path, seq: str) -> Path:
    p = source_dir / f'{seq}.txt'
    if p.exists(): return p
    p = source_dir / seq / f'{seq}.txt'
    if p.exists(): return p
    raise FileNotFoundError(seq)


def read_mot(path: Path):
    rows=[]
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts=line.split(',')
            if len(parts) >= 6:
                rows.append((int(float(parts[0])), int(float(parts[1])), parts))
    return rows


def find(parent: Dict[int,int], x:int) -> int:
    parent.setdefault(x,x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x


def link_edges(edge_rows: List[dict], source_dir: Path, linked_dir: Path):
    linked_dir.mkdir(parents=True, exist_ok=True)
    by={seq:[] for seq in TRAIN_SEQS}
    for r in edge_rows:
        by.setdefault(r['seq'], []).append(r)
    selected_all=[]; by_seq=[]
    for seq in TRAIN_SEQS:
        rows=sorted(by.get(seq,[]), key=lambda r: af(r.get('edge_weight')), reverse=True)
        parent={}; used_s=set(); used_t=set(); final=[]
        for r in rows:
            a=ai(r['track_a']); b=ai(r['track_b'])
            if a in used_s or b in used_t: continue
            ra,rb=find(parent,a),find(parent,b)
            if ra == rb: continue
            parent[rb]=ra; used_s.add(a); used_t.add(b); final.append(r)
        ids=set()
        for r in final:
            ids.add(ai(r['track_a'])); ids.add(ai(r['track_b']))
        idmap={tid:find(parent,tid) for tid in ids}
        src=source_file(source_dir, seq); out=linked_dir/f'{seq}.txt'
        mot=[]
        for _,tid,parts in read_mot(src):
            pp=list(parts); pp[1]=str(idmap.get(tid, tid)); mot.append(pp)
        mot.sort(key=lambda p:(int(float(p[0])), int(float(p[1])), float(p[2]), float(p[3])))
        with out.open('w', encoding='utf-8') as f:
            for pp in mot: f.write(','.join(pp)+'\n')
        tp=sum(ai(r.get('same_gt')) for r in final)
        by_seq.append({'seq':seq,'candidate_edges':len(rows),'accepted_links':len(final),'tp_train_label':tp,'precision_train_label':tp/len(final) if final else 0.0})
        selected_all.extend(final)
    return selected_all, by_seq


def interpolate_and_eval(policy_dir: Path, tracker_name: str) -> dict:
    raw = policy_dir/'raw_linked_results'
    track = policy_dir/'track_results'
    cmd=[sys.executable,'scripts/postprocess/linear_interpolate_mot.py','--input-dir',str(raw),'--output-dir',str(track),'--max-gap','30','--summary-json',str(policy_dir/'interp_summary.json'),'--summary-csv',str(policy_dir/'interp_summary.csv')]
    p=subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (policy_dir/'interp_stdout.log').write_text(p.stdout, encoding='utf-8')
    eval_root=policy_dir/'eval_mot20_all_train'
    data=eval_root/'trackers'/tracker_name/'data'; data.mkdir(parents=True, exist_ok=True)
    for seq in TRAIN_SEQS:
        shutil.copy2(track/f'{seq}.txt', data/f'{seq}.txt')
    seqmap=eval_root/'seqmaps'/'MOT20_train.txt'; seqmap.parent.mkdir(parents=True, exist_ok=True); seqmap.write_text('name\n'+'\n'.join(TRAIN_SEQS)+'\n')
    ecmd=[sys.executable,'TrackEval/scripts/run_mot_challenge.py','--GT_FOLDER','datasets/MOT20/train','--TRACKERS_FOLDER',str(eval_root/'trackers'),'--OUTPUT_FOLDER',str(eval_root/'eval'),'--TRACKERS_TO_EVAL',tracker_name,'--BENCHMARK','MOT20','--SPLIT_TO_EVAL','train','--SEQMAP_FILE',str(seqmap),'--SKIP_SPLIT_FOL','True','--DO_PREPROC','True','--TRACKER_SUB_FOLDER','data','--OUTPUT_SUB_FOLDER','','--PRINT_ONLY_COMBINED','True','--METRICS','HOTA','CLEAR','Identity']
    q=subprocess.run(ecmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (policy_dir/'trackeval_stdout.log').write_text(q.stdout, encoding='utf-8')
    summary=eval_root/'eval'/tracker_name/'pedestrian_summary.txt'
    metrics={}
    if summary.exists():
        lines=[x.strip() for x in summary.read_text().splitlines() if x.strip()]
        if len(lines)>=2: metrics=dict(zip(lines[0].split(), lines[1].split()))
    metrics['interp_returncode']=p.returncode; metrics['trackeval_returncode']=q.returncode; metrics['summary_path']=str(summary)
    return metrics


def load_base_keys(path: Path) -> Set[Tuple[str,str,str]]:
    keys=set()
    for r in read_csv_rows(path):
        keys.add((r['seq'], str(ai(r['track_a'])), str(ai(r['track_b']))))
    return keys


def make_policy_edges(score_df: pd.DataFrame, base_keys: Set[Tuple[str,str,str]], policy: dict) -> List[dict]:
    df=score_df
    keys = list(zip(df['seq'], df['track_a'].astype(str), df['track_b'].astype(str)))
    is_base = np.array([k in base_keys for k in keys], dtype=bool)
    m = np.zeros(len(df), dtype=bool)
    m |= is_base
    cand = (~is_base)
    cand &= df['a42_model_score'].to_numpy() >= policy.get('min_score', 0.0)
    cand &= df['max_rank_by_a42_model_score'].to_numpy() <= policy.get('max_rank', 999)
    cand &= df['appearance_max'].to_numpy() >= policy.get('min_app', 0.0)
    cand &= df['geometry_risk'].to_numpy() <= policy.get('max_geom', 999)
    cand &= df['motion_risk'].to_numpy() <= policy.get('max_motion', 999)
    if policy.get('gap_scope') == '1_150': cand &= df['gap'].to_numpy() <= 150
    if policy.get('gap_scope') == '1_60': cand &= df['gap'].to_numpy() <= 60
    # top N by score among remaining recovery candidates.
    idx = np.where(cand)[0]
    if policy.get('top_recovery', 0) and len(idx) > policy['top_recovery']:
        order = np.argsort(-df['a42_model_score'].to_numpy()[idx])[:policy['top_recovery']]
        idx = idx[order]
    m[idx] = True
    out=[]
    sub=df.loc[m]
    for _,r in sub.iterrows():
        k=(r['seq'], str(r['track_a']), str(r['track_b']))
        base = k in base_keys
        weight = (2.0 + float(r['a42_model_score'])) if base else float(r['a42_model_score'])
        out.append({
            'seq': r['seq'], 'track_a': str(r['track_a']), 'track_b': str(r['track_b']), 'gap': int(r['gap']), 'gap_bucket': r['gap_bucket'],
            'same_gt': int(r['same_gt']), 'a42_model_score': float(r['a42_model_score']), 'appearance_max': float(r['appearance_max']),
            'max_rank_by_a42_model_score': int(r['max_rank_by_a42_model_score']), 'is_base': int(base), 'is_recovery': int(not base),
            'edge_weight': weight,
        })
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-manifest', required=True)
    ap.add_argument('--base-links', required=True)
    ap.add_argument('--source-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df=load_manifest(Path(args.train_manifest))
    y=df['same_gt'].to_numpy(dtype=np.int8)
    X=feature_matrix(df)
    hard=make_hard_mask(df)
    oof=np.zeros(len(df), dtype=np.float32)
    fold_reports=[]
    for i,seq in enumerate(TRAIN_SEQS):
        train_mask=(df['seq'].to_numpy()!=seq) & hard
        val_mask=(df['seq'].to_numpy()==seq)
        clf=train_model(X[train_mask], y[train_mask], seed=42+i)
        oof[val_mask]=clf.predict_proba(X[val_mask])[:,1].astype(np.float32)
        ap_score=average_precision_score(y[val_mask], oof[val_mask]) if y[val_mask].sum()>0 else 0.0
        try:
            auc=roc_auc_score(y[val_mask], oof[val_mask]) if len(np.unique(y[val_mask]))>1 else 0.0
        except Exception:
            auc=0.0
        fold_reports.append({'seq':seq,'train_rows':int(train_mask.sum()),'val_rows':int(val_mask.sum()),'val_true':int(y[val_mask].sum()),'avg_precision':float(ap_score),'roc_auc':float(auc),'n_iter':int(getattr(clf,'n_iter_',0))})
    score_df=df[['seq','track_a','track_b','gap','gap_bucket','same_gt','appearance_max','geometry_risk','motion_risk']].copy()
    score_df['a42_model_score']=oof
    score_df=add_model_rank_fields(score_df)
    # Save compact OOF scores.
    score_cols=['seq','track_a','track_b','gap','gap_bucket','same_gt','a42_model_score','appearance_max','geometry_risk','motion_risk','out_rank_by_a42_model_score','in_rank_by_a42_model_score','max_rank_by_a42_model_score','out_margin_to_second_a42_model_score','in_margin_to_second_a42_model_score','min_margin_by_a42_model_score']
    score_df[score_cols].to_csv(out/'a42_train_oof_scores.csv', index=False)
    base_keys=load_base_keys(Path(args.base_links))
    policies=[]
    for top in [25,50,75,100,150,200,300]:
        policies.append({'name':f'base_plus_model_top{top}_rank2_app060_allgap','top_recovery':top,'max_rank':2,'min_app':0.60,'max_geom':1,'max_motion':2,'gap_scope':'1_300','min_score':0.0})
    for top in [50,100,150]:
        policies.append({'name':f'base_plus_model_top{top}_rank3_app055_allgap','top_recovery':top,'max_rank':3,'min_app':0.55,'max_geom':1,'max_motion':2,'gap_scope':'1_300','min_score':0.0})
    summaries=[]
    for pol in policies:
        pdir=out/pol['name']; pdir.mkdir(parents=True, exist_ok=True)
        edges=make_policy_edges(score_df, base_keys, pol)
        selected, by_seq=link_edges(edges, Path(args.source_dir), pdir/'raw_linked_results')
        write_csv(pdir/'accepted_links.csv', selected)
        metrics=interpolate_and_eval(pdir, 'A42_02b_'+pol['name'])
        tp=sum(ai(r['same_gt']) for r in selected); rec=sum(ai(r['is_recovery']) for r in selected); rectp=sum(ai(r['same_gt']) for r in selected if ai(r['is_recovery']))
        summary={'policy':pol,'accepted_links_total':len(selected),'accepted_recovery_links':rec,'tp_train_label':tp,'recovery_tp_train_label':rectp,'precision_train_label':tp/len(selected) if selected else 0.0,'by_seq':by_seq,'metrics':metrics,'decision':'PASS_EVAL_DONE' if metrics.get('trackeval_returncode')==0 else 'EVAL_FAILED'}
        (pdir/'link_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
        summaries.append(summary)
    # compact output
    with (out/'a42_02b_policy_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        fields=['policy','accepted_links_total','accepted_recovery_links','tp_train_label','recovery_tp_train_label','precision_train_label','HOTA','IDF1','MOTA','AssA','DetA','IDSW','Frag','decision','summary_path']
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for s in summaries:
            m=s['metrics']; w.writerow({'policy':s['policy']['name'],'accepted_links_total':s['accepted_links_total'],'accepted_recovery_links':s['accepted_recovery_links'],'tp_train_label':s['tp_train_label'],'recovery_tp_train_label':s['recovery_tp_train_label'],'precision_train_label':s['precision_train_label'],**{k:m.get(k,'') for k in ['HOTA','IDF1','MOTA','AssA','DetA','IDSW','Frag']},'decision':s['decision'],'summary_path':m.get('summary_path','')})
    best=sorted(summaries, key=lambda s:(float(s['metrics'].get('HOTA') or 0), float(s['metrics'].get('IDF1') or 0)), reverse=True)[0]
    final={'fold_reports':fold_reports,'policies':summaries,'best_policy':best['policy']['name'],'decision':'A42_02b_RANKING_MODEL_TRAIN_EVAL_DONE'}
    (out/'a42_02b_summary.json').write_text(json.dumps(final, indent=2, sort_keys=True)+'\n')
    (out/'decision.md').write_text('# A42_02b Ranking Model Train Eval\n\n```json\n'+json.dumps({'fold_reports':fold_reports,'best_policy':best['policy']['name'],'best_metrics':best['metrics'],'decision':final['decision']}, indent=2, sort_keys=True)+'\n```\n')
    print(json.dumps({'fold_reports':fold_reports,'best_policy':best['policy']['name'],'best_metrics':best['metrics'],'decision':final['decision']}, indent=2, sort_keys=True))
if __name__=='__main__': main()
