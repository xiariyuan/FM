from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ID_COLS = {
    'seq','track_id','track_a','track_b','boundary_frame','prev_frame',
    'proposal_rank','partner_rank','partner_frame_delta','label_pair_class',
}
FORBIDDEN_PREFIXES = ('label_', 'a_gt_', 'b_gt_', 'distance_to_switch_')
FORBIDDEN_EXACT = {'oof_et','oof_hgb','oof_ensemble','heuristic_score','target'}


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float('nan')


def event_budget_metrics(df: pd.DataFrame, score_col: str, target_col: str, budgets):
    # One partner decision per boundary proposal.
    best = df.sort_values(score_col, ascending=False).groupby(
        ['track_a','boundary_frame'], as_index=False, sort=False
    ).head(1).sort_values(score_col, ascending=False)
    true_events = set(map(tuple, df[df[target_col] == 1][['track_a','boundary_frame']].drop_duplicates().values.tolist()))
    rows = []
    for budget in budgets:
        selected = best.head(budget)
        pred_events = list(map(tuple, selected[['track_a','boundary_frame']].values.tolist()))
        hits = sum(x in true_events for x in pred_events)
        rows.append({
            'score': score_col, 'target': target_col, 'budget': budget,
            'selected': len(selected), 'true_events_in_bank': len(true_events),
            'hits': int(hits), 'event_recall_in_bank': hits / max(1, len(true_events)),
            'event_precision': hits / max(1, len(selected)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair-bank', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.pair_bank)
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    features = [c for c in numeric if c not in ID_COLS | FORBIDDEN_EXACT
                and not c.startswith(FORBIDDEN_PREFIXES)]
    # Explicitly keep observable unary proposal scores, but not their GT labels.
    for c in ['oof_hgb','oof_et','oof_ensemble','heuristic_score']:
        if c in df.columns and c not in features:
            features.append(c)

    targets = {
        'reciprocal': 'label_reciprocal_swap',
        'related': 'label_pair_related',
    }
    groups = df.track_a.to_numpy()
    splitter = GroupKFold(n_splits=min(args.folds, df.track_a.nunique()))
    fold_reports = []; all_importance = []

    for target_name, target_col in targets.items():
        y = df[target_col].to_numpy(int)
        oof_et = np.zeros(len(df)); oof_hgb = np.zeros(len(df))
        importance = np.zeros(len(features))
        for fold, (tr, te) in enumerate(splitter.split(df[features], y, groups), 1):
            train, test = df.iloc[tr], df.iloc[te]
            imp = SimpleImputer(strategy='median')
            xtr = imp.fit_transform(train[features]); xte = imp.transform(test[features])
            ytr, yte = y[tr], y[te]
            et = ExtraTreesClassifier(
                n_estimators=400, min_samples_leaf=3, max_features=.75,
                class_weight='balanced', n_jobs=-1, random_state=1000 + fold,
            )
            et.fit(xtr, ytr); pet = et.predict_proba(xte)[:, 1]
            importance += et.feature_importances_ / splitter.n_splits
            pos_weight = max(1.0, (len(ytr) - ytr.sum()) / max(1, ytr.sum()))
            hgb = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=.05, max_leaf_nodes=15,
                min_samples_leaf=12, l2_regularization=3.0,
                random_state=2000 + fold,
            )
            hgb.fit(xtr, ytr, sample_weight=np.where(ytr == 1, pos_weight, 1.0))
            phgb = hgb.predict_proba(xte)[:, 1]
            oof_et[te] = pet; oof_hgb[te] = phgb
            fold_reports.append({
                'target': target_name, 'fold': fold, 'train_rows': len(tr),
                'test_rows': len(te), 'test_positive': int(yte.sum()),
                'et_ap': float(average_precision_score(yte, pet)),
                'et_auc': safe_auc(yte, pet),
                'hgb_ap': float(average_precision_score(yte, phgb)),
                'hgb_auc': safe_auc(yte, phgb),
            })
        df[f'oof_pair_{target_name}_et'] = oof_et
        df[f'oof_pair_{target_name}_hgb'] = oof_hgb
        df[f'oof_pair_{target_name}_ensemble'] = .5 * (oof_et + oof_hgb)
        all_importance.extend({'target': target_name, 'feature': f, 'importance': float(v)}
                              for f, v in zip(features, importance))

    # Nonlearned pair-state baselines.
    df['pair_swap_margin_filled'] = df.pair_swap_margin.fillna(-10.0)
    df['pair_related_similarity'] = df[['old_handoff_score','new_source_score']].max(axis=1).fillna(-10.0)
    df['unary_only_score'] = df.oof_hgb if 'oof_hgb' in df else -df.proposal_rank

    aggregate = {}
    budget_rows = []
    for target_name, target_col in targets.items():
        score_cols = [
            'unary_only_score',
            'pair_swap_margin_filled' if target_name == 'reciprocal' else 'pair_related_similarity',
            f'oof_pair_{target_name}_et', f'oof_pair_{target_name}_hgb',
            f'oof_pair_{target_name}_ensemble',
        ]
        y = df[target_col].to_numpy(int)
        aggregate[target_name] = {
            'rows': len(df), 'positives': int(y.sum()), 'prevalence': float(y.mean()),
            'unique_positive_events': int(df[df[target_col] == 1][['track_a','boundary_frame']].drop_duplicates().shape[0]),
            'scores': {},
        }
        for score in score_cols:
            aggregate[target_name]['scores'][score] = {
                'ap': float(average_precision_score(y, df[score])),
                'auc': safe_auc(y, df[score]),
            }
            budget_rows.extend(event_budget_metrics(df, score, target_col,
                                                    [25,50,100,200,500,1000]))

    df.to_csv(out / 'm02_pair_state_oof_scores.csv', index=False)
    pd.DataFrame(fold_reports).to_csv(out / 'fold_metrics.csv', index=False)
    pd.DataFrame(budget_rows).to_csv(out / 'event_budget_metrics.csv', index=False)
    importance_df = pd.DataFrame(all_importance).sort_values(['target','importance'], ascending=[True,False])
    importance_df.to_csv(out / 'feature_importance.csv', index=False)
    report = {
        'protocol': {
            'scope': 'MOT20-02 diagnostic pilot only',
            'validation': 'GroupKFold by changed-track track_a',
            'candidate_bank': 'top-5000 unary NMS proposals x top-3 observable overlap partners',
            'GT_policy': 'GT labels excluded from features; diagnostic OOF unary score is retained and is not yet cross-sequence deployable.',
            'feature_count': len(features), 'features': features,
        },
        'folds': fold_reports,
        'aggregate': aggregate,
        'best_budget_rows': pd.DataFrame(budget_rows).sort_values(
            ['target','budget','event_precision','event_recall_in_bank'], ascending=[True,True,False,False]
        ).groupby(['target','budget']).head(3).to_dict(orient='records'),
        'top_features': importance_df.groupby('target').head(20).to_dict(orient='records'),
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
