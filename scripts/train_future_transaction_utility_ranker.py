from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold


FORBID_EXACT = {
    'canonical_rank', 'rank', 'u', 'v', 'track_a', 'track_b', 'track_id', 'seq',
    'target', 'label_pair_class', 'a_gt_old', 'a_gt_new', 'b_gt_old', 'b_gt_new',
    'a_gt_old_rows', 'a_gt_new_rows', 'b_gt_old_rows', 'b_gt_new_rows',
    'a_gt_old_purity', 'a_gt_new_purity', 'b_gt_old_purity', 'b_gt_new_purity',
    'label_a_changed', 'label_b_changed', 'label_reciprocal_swap',
    'label_old_carrier', 'label_new_source', 'label_pair_related',
    'cluster_min_frame', 'cluster_max_frame',
}
FORBID_SUBSTRINGS = ('delta_', 'changed_rows', 'end_frame', 'reject_reason', 'track_sha')


def rank_percentile(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method='average', pct=True).to_numpy(dtype=float)


def choose_features(df: pd.DataFrame, variant: str) -> list[str]:
    out = []
    for c in df.columns:
        if c in FORBID_EXACT or any(s in c for s in FORBID_SUBSTRINGS):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if variant == 'raw' and (
            c.startswith('oof_') or c in {'unary_only_score', 'proposal_rank', 'debt_rank'}
        ):
            continue
        out.append(c)
    return out


def metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict:
    positive = (y > 0).astype(int)
    rho = spearmanr(y, pred, nan_policy='omit').statistic
    result = {
        'rmse': float(mean_squared_error(y, pred) ** 0.5),
        'mae': float(mean_absolute_error(y, pred)),
        'spearman': float(rho) if np.isfinite(rho) else None,
        'positive_ap': float(average_precision_score(positive, prob)),
        'positive_auc': float(roc_auc_score(positive, prob)),
    }
    order = np.argsort(-pred)
    for k in [5, 10, 20, 30]:
        idx = order[:k]
        result[f'top{k}_positive_precision'] = float(positive[idx].mean())
        result[f'top{k}_actual_utility_sum'] = float(y[idx].sum())
        result[f'top{k}_actual_utility_mean'] = float(y[idx].mean())
    result['all_positive_oracle_sum'] = float(y[y > 0].sum())
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--utility-csv', nargs='+', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--variant', choices=['raw', 'enriched'], default='raw')
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(args.features)
    utility_parts = []
    for p in args.utility_csv:
        z = pd.read_csv(p)
        z = z[z['mode'] == 'perm'].copy()
        utility_parts.append(z)
    u = pd.concat(utility_parts, ignore_index=True).sort_values('rank')
    if u['rank'].duplicated().any():
        raise RuntimeError('duplicate utility ranks')
    label_cols = ['rank', 'u', 'v', 'boundary_frame', 'delta_HOTA', 'delta_AssA', 'delta_IDF1', 'delta_IDSW']
    d = f.merge(
        u[label_cols],
        left_on=['canonical_rank', 'u', 'v', 'boundary_frame'],
        right_on=['rank', 'u', 'v', 'boundary_frame'],
        how='inner', validate='one_to_one',
    )
    if len(d) != 100:
        raise RuntimeError(f'expected 100 joined rows, got {len(d)}')
    d['pair_group'] = d['u'].astype(str) + '_' + d['v'].astype(str)
    features = choose_features(d, args.variant)
    X = d[features]
    y = d['delta_HOTA'].to_numpy(dtype=float)
    positive = (y > 0).astype(int)
    groups = d['pair_group'].to_numpy()

    pred_et = np.zeros(len(d)); pred_hgb = np.zeros(len(d)); prob_et = np.zeros(len(d))
    fold_id = np.full(len(d), -1, dtype=int)
    splitter = GroupKFold(n_splits=args.folds)
    for fold, (tr, te) in enumerate(splitter.split(X, y, groups)):
        imp = SimpleImputer(strategy='median')
        xtr = imp.fit_transform(X.iloc[tr]); xte = imp.transform(X.iloc[te])
        et = ExtraTreesRegressor(
            n_estimators=600, max_depth=8, min_samples_leaf=3,
            max_features=0.65, random_state=1100 + fold, n_jobs=-1,
        )
        hgb = HistGradientBoostingRegressor(
            max_iter=180, learning_rate=0.04, max_leaf_nodes=7,
            min_samples_leaf=8, l2_regularization=1.0, random_state=1200 + fold,
        )
        clf = ExtraTreesClassifier(
            n_estimators=600, max_depth=8, min_samples_leaf=3,
            max_features=0.65, class_weight='balanced',
            random_state=1300 + fold, n_jobs=-1,
        )
        et.fit(xtr, y[tr]); hgb.fit(xtr, y[tr]); clf.fit(xtr, positive[tr])
        pred_et[te] = et.predict(xte)
        pred_hgb[te] = hgb.predict(xte)
        prob_et[te] = clf.predict_proba(xte)[:, 1]
        fold_id[te] = fold

    rank_et = rank_percentile(pred_et)
    rank_hgb = rank_percentile(pred_hgb)
    rank_prob = rank_percentile(prob_et)
    ensemble_rank = 0.4 * rank_et + 0.4 * rank_hgb + 0.2 * rank_prob
    ensemble_utility = 0.5 * pred_et + 0.5 * pred_hgb

    d['fold'] = fold_id
    d['oof_utility_et'] = pred_et
    d['oof_utility_hgb'] = pred_hgb
    d['oof_positive_prob_et'] = prob_et
    d['oof_rank_et'] = rank_et
    d['oof_rank_hgb'] = rank_hgb
    d['oof_rank_positive'] = rank_prob
    d['oof_utility_ensemble'] = ensemble_utility
    d['oof_transaction_rank'] = ensemble_rank
    d['actual_positive'] = positive
    d = d.sort_values('oof_transaction_rank', ascending=False)
    d.to_csv(out / 'top100_oof_transaction_scores.csv', index=False)

    report = {
        'protocol': {
            'scope': 'MOT20-02 diagnostic only',
            'validation': f'{args.folds}-fold GroupKFold by unordered track pair',
            'variant': args.variant,
            'feature_count': len(features),
            'feature_policy': 'numeric observable tracker/ReID/future-video features; GT, labels, IDs, canonical rank, counterfactual outcomes and post-action statistics excluded',
            'primary_rank': '0.4 rank(ET reg) + 0.4 rank(HGB reg) + 0.2 rank(ET positive classifier)',
        },
        'dataset': {
            'events': len(d), 'unique_pair_groups': int(d.pair_group.nunique()),
            'positive_events': int(positive.sum()), 'negative_or_zero_events': int((positive == 0).sum()),
            'mean_delta_HOTA': float(y.mean()), 'positive_utility_sum': float(y[y > 0].sum()),
        },
        'models': {
            'extra_trees_regression': metrics(y, pred_et, prob_et),
            'hist_gradient_boosting_regression': metrics(y, pred_hgb, prob_et),
            'ensemble_utility': metrics(y, ensemble_utility, prob_et),
            'ensemble_rank': metrics(y, ensemble_rank, prob_et),
        },
        'top_ranked': d[[
            'canonical_rank', 'u', 'v', 'boundary_frame', 'pair_group',
            'delta_HOTA', 'delta_AssA', 'delta_IDSW',
            'oof_utility_et', 'oof_utility_hgb', 'oof_positive_prob_et',
            'oof_utility_ensemble', 'oof_transaction_rank',
        ]].head(30).to_dict('records'),
        'features': features,
    }

    # Full-data ET feature importance is descriptive only and never used for OOF selection.
    imp = SimpleImputer(strategy='median')
    xa = imp.fit_transform(X)
    final_et = ExtraTreesRegressor(
        n_estimators=1000, max_depth=8, min_samples_leaf=3,
        max_features=0.65, random_state=20260714, n_jobs=-1,
    )
    final_et.fit(xa, y)
    importance = pd.DataFrame({'feature': features, 'importance': final_et.feature_importances_})
    importance = importance.sort_values('importance', ascending=False)
    importance.to_csv(out / 'descriptive_feature_importance.csv', index=False)
    report['descriptive_top_importance'] = importance.head(30).to_dict('records')
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({
        'dataset': report['dataset'],
        'model_metrics': report['models'],
        'top10': report['top_ranked'][:10],
        'top_importance': report['descriptive_top_importance'][:15],
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
