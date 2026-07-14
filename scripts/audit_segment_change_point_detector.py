from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ID_COLS = {'seq', 'track_id', 'boundary_frame', 'prev_frame'}
LABEL_PREFIXES = ('label_switch_', 'distance_to_switch_')


def nms_events(df: pd.DataFrame, score_col: str, radius: int) -> pd.DataFrame:
    selected = []
    for tid, g in df.sort_values(score_col, ascending=False).groupby('track_id', sort=False):
        kept = []
        for row in g.itertuples(index=False):
            frame = int(row.boundary_frame)
            if any(abs(frame - x) <= radius for x in kept):
                continue
            kept.append(frame); selected.append(row._asdict())
    return pd.DataFrame(selected).sort_values(score_col, ascending=False).reset_index(drop=True)


def event_metrics(candidates: pd.DataFrame, selected: pd.DataFrame, exact_events: pd.DataFrame,
                  score_col: str, tolerance: int, budgets: list[int]):
    truth = {(int(r.track_id), int(r.boundary_frame)) for r in exact_events.itertuples(index=False)}
    out = []
    for budget in budgets:
        pred = [(int(r.track_id), int(r.boundary_frame)) for r in selected.head(budget).itertuples(index=False)]
        hit_truth = set(); hit_pred = 0
        for tid, frame in pred:
            matches = [(t, f) for t, f in truth if t == tid and abs(f - frame) <= tolerance]
            if matches:
                nearest = min(matches, key=lambda x: abs(x[1] - frame))
                hit_truth.add(nearest); hit_pred += 1
        out.append({
            'budget': budget, 'selected': min(budget, len(pred)),
            'true_events': len(truth), 'matched_true_events': len(hit_truth),
            'event_recall': len(hit_truth) / max(1, len(truth)),
            'event_precision_upper': hit_pred / max(1, len(pred)),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-csv', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--persistence', type=int, default=5)
    ap.add_argument('--label-window', type=int, default=3)
    ap.add_argument('--candidate-ioa', type=float, default=.1)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--nms-radius', type=int, default=10)
    ap.add_argument('--debt-csv')
    ap.add_argument('--min-debt-pct', type=float, default=0.0)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features_csv)
    if args.debt_csv:
        debt = pd.read_csv(args.debt_csv)
        if 'seq' in debt.columns and 'seq' in df.columns:
            seq_values = set(df.seq.astype(str).unique())
            debt = debt[debt.seq.astype(str).isin(seq_values)].copy()
        debt = debt[['track_id','score','rank']].drop_duplicates('track_id').rename(columns={'score':'debt_score','rank':'debt_rank'})
        debt['debt_pct'] = debt.debt_score.rank(pct=True, method='average')
        df = df.merge(debt, on='track_id', how='left')
        df['debt_score'] = df.debt_score.fillna(0.0)
        df['debt_rank'] = df.debt_rank.fillna(df.debt_rank.max() + 1 if df.debt_rank.notna().any() else 1)
        df['debt_pct'] = df.debt_pct.fillna(0.0)
    exact_col = f'label_switch_p{args.persistence}'
    dist_col = f'distance_to_switch_p{args.persistence}'
    exact_events = df[df[exact_col] == 1][['track_id', 'boundary_frame']].drop_duplicates()
    pool = df[(df.overlap_max_ioa >= args.candidate_ioa) & (df.get('debt_pct', pd.Series(1.0, index=df.index)) >= args.min_debt_pct)].copy().reset_index(drop=True)
    pool['target'] = (pool[dist_col] <= args.label_window).astype(int)

    features = [c for c in pool.columns
                if c not in ID_COLS | {'target'}
                and not c.startswith(LABEL_PREFIXES)
                and pd.api.types.is_numeric_dtype(pool[c])]
    groups = pool.track_id.to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(args.folds, len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    oof_et = np.zeros(len(pool), dtype=float)
    oof_hgb = np.zeros(len(pool), dtype=float)
    fold_reports = []
    importance = np.zeros(len(features), dtype=float)

    for fold, (tr, te) in enumerate(splitter.split(pool[features], pool.target, groups), 1):
        train, test = pool.iloc[tr], pool.iloc[te]
        imp = SimpleImputer(strategy='median')
        xtr = imp.fit_transform(train[features]); xte = imp.transform(test[features])
        ytr = train.target.to_numpy(int); yte = test.target.to_numpy(int)
        et = ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=4, max_features=.8,
            class_weight='balanced', n_jobs=-1, random_state=100 + fold,
        )
        et.fit(xtr, ytr); pet = et.predict_proba(xte)[:, 1]
        importance += et.feature_importances_ / n_splits

        pos_weight = max(1.0, (len(ytr) - ytr.sum()) / max(1, ytr.sum()))
        sample_weight = np.where(ytr == 1, pos_weight, 1.0)
        hgb = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=.06, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=2.0, random_state=200 + fold,
        )
        hgb.fit(xtr, ytr, sample_weight=sample_weight); phgb = hgb.predict_proba(xte)[:, 1]
        oof_et[te] = pet; oof_hgb[te] = phgb
        fold_reports.append({
            'fold': fold, 'train_rows': len(tr), 'test_rows': len(te),
            'train_tracks': int(train.track_id.nunique()), 'test_tracks': int(test.track_id.nunique()),
            'test_positive_rows': int(yte.sum()),
            'et_ap': float(average_precision_score(yte, pet)),
            'et_auc': float(roc_auc_score(yte, pet)),
            'hgb_ap': float(average_precision_score(yte, phgb)),
            'hgb_auc': float(roc_auc_score(yte, phgb)),
        })

    pool['oof_et'] = oof_et; pool['oof_hgb'] = oof_hgb
    pool['oof_ensemble'] = .5 * (oof_et + oof_hgb)
    # Observable nonlearned baseline for calibration context.
    pool['heuristic_score'] = (
        pool.appearance_change.rank(pct=True) +
        pool.appearance_margin.rank(pct=True) +
        pool.prediction_error_norm.rank(pct=True) +
        pool.overlap_max_ioa.rank(pct=True)
    ) / 4.0

    aggregate = {'rows': len(pool), 'positive_rows': int(pool.target.sum()),
                 'prevalence': float(pool.target.mean()), 'exact_events': len(exact_events),
                 'candidate_exact_event_recall': float(
                     len(pool.merge(exact_events, on=['track_id','boundary_frame'])) / max(1, len(exact_events)))
                 }
    policy_reports = []
    for score_col in ['heuristic_score', 'oof_et', 'oof_hgb', 'oof_ensemble']:
        aggregate[score_col] = {
            'row_ap': float(average_precision_score(pool.target, pool[score_col])),
            'row_auc': float(roc_auc_score(pool.target, pool[score_col])),
            'spearman_to_distance': float(spearmanr(pool[score_col], -np.minimum(pool[dist_col], 1000)).statistic),
        }
        selected = nms_events(pool, score_col, args.nms_radius)
        metrics = event_metrics(pool, selected, exact_events, score_col, args.label_window,
                                [50, 100, 200, 500, 1000, 2000])
        for row in metrics:
            policy_reports.append({'score': score_col, **row})

    pool.to_csv(out / 'm02_grouped_oof_change_scores.csv', index=False)
    pd.DataFrame(fold_reports).to_csv(out / 'fold_metrics.csv', index=False)
    pd.DataFrame(policy_reports).to_csv(out / 'event_budget_metrics.csv', index=False)
    pd.DataFrame({'feature': features, 'importance': importance}).sort_values(
        'importance', ascending=False).to_csv(out / 'feature_importance.csv', index=False)
    report = {
        'protocol': {
            'scope': 'MOT20-02 diagnostic pilot only; not a deployable cross-sequence validation',
            'validation': 'GroupKFold by tracker ID',
            'labels': f'persistent GT switch p{args.persistence} within +/-{args.label_window} frames',
            'candidate_rule': f'overlap_max_ioa >= {args.candidate_ioa}; debt_pct >= {args.min_debt_pct}',
            'features': features,
        },
        'folds': fold_reports, 'aggregate': aggregate,
        'best_event_policies': pd.DataFrame(policy_reports).sort_values(
            ['budget','event_recall','event_precision_upper'], ascending=[True,False,False]
        ).groupby('budget').head(2).to_dict(orient='records'),
        'top_features': pd.DataFrame({'feature': features, 'importance': importance}).sort_values(
            'importance', ascending=False).head(20).to_dict(orient='records'),
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
