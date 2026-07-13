#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import pandas as pd
import numpy as np

NUM_COLS = [
    'aflink_score','debt_adjusted_edge_score','out_rank_by_aflink_score','in_rank_by_aflink_score',
    'out_margin_to_second_aflink_score','in_margin_to_second_aflink_score','edge_debt_score',
    'risk_total','geometry_risk','motion_risk','competition_risk','appearance_max','appearance_mean',
    'predicted_distance_per_frame','height_ratio','area_ratio','bottom_y_gap'
]


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    if 'same_gt' in df.columns:
        df['same_gt_num'] = pd.to_numeric(df['same_gt'], errors='coerce').fillna(0).astype(int)
    else:
        df['same_gt_num'] = 0
    return df


def mask_for(df: pd.DataFrame, cfg: dict) -> pd.Series:
    m = pd.Series(True, index=df.index)
    m &= df['aflink_score'] >= cfg.get('min_aflink', -1)
    m &= df['debt_adjusted_edge_score'] >= cfg.get('min_adjusted', -999)
    m &= df['out_rank_by_aflink_score'] <= cfg.get('max_out_rank', 999)
    m &= df['in_rank_by_aflink_score'] <= cfg.get('max_in_rank', 999)
    m &= np.minimum(df['out_margin_to_second_aflink_score'], df['in_margin_to_second_aflink_score']) >= cfg.get('min_bidir_margin', -999)
    m &= df['edge_debt_score'] >= cfg.get('min_debt', 0)
    m &= df['risk_total'] <= cfg.get('max_risk', 999)
    m &= df['geometry_risk'] <= cfg.get('max_geometry_risk', 999)
    m &= df['motion_risk'] <= cfg.get('max_motion_risk', 999)
    m &= df['competition_risk'] <= cfg.get('max_competition_risk', 999)
    if 'min_appearance_max' in cfg:
        m &= df['appearance_max'] >= cfg['min_appearance_max']
    if 'max_predicted_pf' in cfg:
        m &= df['predicted_distance_per_frame'] <= cfg['max_predicted_pf']
    if 'edge_set' in cfg:
        es = cfg['edge_set']
        if es == 'no_highrisk':
            m &= df['edge_type'] != 'high_risk_geometry'
        elif es == 'boundary_or_fragment':
            m &= df['edge_type'].isin(['weak_boundary_recovery','fragmented_tracklet_recovery'])
        elif es == 'stable_gap':
            m &= df['edge_type'].isin(['short_gap_continuation','long_gap_reappearance'])
    return m


def eval_policy(train: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> dict:
    mt = mask_for(train, cfg)
    ms = mask_for(test, cfg)
    selected = int(mt.sum())
    tp = int(train.loc[mt, 'same_gt_num'].sum()) if selected else 0
    pos = int(train['same_gt_num'].sum())
    test_selected = int(ms.sum())
    row = {k:v for k,v in cfg.items() if not k.startswith('_')}
    row.update({
        'train_selected': selected,
        'train_tp': tp,
        'train_precision': tp / selected if selected else 0.0,
        'train_recall': tp / pos if pos else 0.0,
        'test_selected': test_selected,
    })
    # per seq selected counts for sanity
    for seq, n in test.loc[ms].groupby('seq').size().to_dict().items():
        row[f'test_selected_{seq}'] = int(n)
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields or ['policy'], extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--out-dir', required=True)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    train=load_df(args.train); test=load_df(args.test)
    policies=[]
    # Simple score baselines.
    for thr in [0.95,0.90,0.85,0.80,0.70,0.60,0.50,0.40,0.30,0.20,0.15]:
        policies.append({'policy':f'score_ge_{thr:.2f}', 'min_aflink':thr})
    # Strict rank/margin policies.
    for thr in [0.90,0.85,0.80,0.70,0.60,0.50,0.40,0.30,0.20,0.15]:
        for margin in [0.02,0.05,0.10,0.15,0.20]:
            policies.append({'policy':f'rank11_thr{thr:.2f}_m{margin:.2f}_risk2', 'min_aflink':thr, 'max_out_rank':1, 'max_in_rank':1, 'min_bidir_margin':margin, 'max_risk':2, 'max_geometry_risk':1, 'max_motion_risk':1, 'max_competition_risk':2})
            policies.append({'policy':f'debt_rank11_thr{thr:.2f}_m{margin:.2f}_risk2', 'min_aflink':thr, 'max_out_rank':1, 'max_in_rank':1, 'min_bidir_margin':margin, 'min_debt':1, 'max_risk':2, 'max_geometry_risk':1, 'max_motion_risk':1, 'max_competition_risk':2})
            policies.append({'policy':f'debt2_rank11_thr{thr:.2f}_m{margin:.2f}_risk2', 'min_aflink':thr, 'max_out_rank':1, 'max_in_rank':1, 'min_bidir_margin':margin, 'min_debt':2, 'max_risk':2, 'max_geometry_risk':1, 'max_motion_risk':1, 'max_competition_risk':2})
    # Boundary/fragment focused debt policies.
    for thr in [0.15,0.20,0.30,0.40,0.50,0.60,0.70]:
        for margin in [0.03,0.05,0.10,0.15]:
            policies.append({'policy':f'bf_debt_rank11_thr{thr:.2f}_m{margin:.2f}', 'edge_set':'boundary_or_fragment', 'min_aflink':thr, 'max_out_rank':1, 'max_in_rank':1, 'min_bidir_margin':margin, 'min_debt':1, 'max_risk':2, 'max_geometry_risk':1, 'max_motion_risk':1, 'max_competition_risk':2})
            policies.append({'policy':f'bf_debt2_rank11_thr{thr:.2f}_m{margin:.2f}', 'edge_set':'boundary_or_fragment', 'min_aflink':thr, 'max_out_rank':1, 'max_in_rank':1, 'min_bidir_margin':margin, 'min_debt':2, 'max_risk':2, 'max_geometry_risk':1, 'max_motion_risk':1, 'max_competition_risk':2})
    # Stable gap policies, useful for lower-risk continuation.
    for thr in [0.10,0.15,0.20,0.30,0.40,0.50]:
        for margin in [0.02,0.05,0.10]:
            policies.append({'policy':f'stable_rank11_thr{thr:.2f}_m{margin:.2f}', 'edge_set':'stable_gap', 'min_aflink':thr, 'max_out_rank':1, 'max_in_rank':1, 'min_bidir_margin':margin, 'max_risk':2, 'max_geometry_risk':1, 'max_motion_risk':1, 'max_competition_risk':2})
    # Appearance rescue policies independent of model score, but rank constrained.
    for app in [0.80,0.85,0.90]:
        for margin in [0.05,0.10,0.15]:
            policies.append({'policy':f'app{app:.2f}_rank11_m{margin:.2f}_score005', 'min_aflink':0.05, 'min_appearance_max':app, 'max_out_rank':1, 'max_in_rank':1, 'min_bidir_margin':margin, 'min_debt':1, 'max_risk':2, 'max_geometry_risk':1, 'max_motion_risk':1, 'max_competition_risk':2})
    rows=[eval_policy(train,test,cfg) for cfg in policies]
    rows=sorted(rows, key=lambda r:(r['train_precision'], r['train_tp'], r['test_selected']), reverse=True)
    write_csv(out/'a41_gate_fast_policy_candidates.csv', rows)
    usable=[r for r in rows if r['train_selected']>=5]
    strict=[r for r in usable if r['train_precision']>=0.80 and r['train_selected']>=10]
    balanced=[r for r in usable if r['train_precision']>=0.70 and r['train_selected']>=20]
    aggressive=[r for r in usable if r['train_precision']>=0.60 and r['train_selected']>=50]
    for name,rs in [('strict_p80',strict),('balanced_p70',balanced),('aggressive_p60',aggressive)]:
        write_csv(out/f'{name}.csv', sorted(rs, key=lambda r:(r['train_tp'], r['test_selected'], r['train_precision']), reverse=True)[:100])
    best_strict=(sorted(strict, key=lambda r:(r['train_tp'], r['test_selected'], r['train_precision']), reverse=True)[:1] or [None])[0]
    best_balanced=(sorted(balanced, key=lambda r:(r['train_tp'], r['test_selected'], r['train_precision']), reverse=True)[:1] or [None])[0]
    best_aggressive=(sorted(aggressive, key=lambda r:(r['train_tp'], r['test_selected'], r['train_precision']), reverse=True)[:1] or [None])[0]
    summary={
        'train_edges':len(train), 'test_edges':len(test), 'train_positive_edges':int(train['same_gt_num'].sum()), 'policies':len(policies),
        'best_strict_p80':best_strict, 'best_balanced_p70':best_balanced, 'best_aggressive_p60':best_aggressive,
        'decision':'A41_01b_FAST_GATE_CALIBRATION_DONE'
    }
    (out/'gate_fast_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    (out/'decision.md').write_text('# A41_01b Fast Gate Calibration\n\n```json\n'+json.dumps(summary, indent=2, sort_keys=True)+'\n```\n\nNext: use selected balanced/strict policy in A41_02 one-to-one solver.\n')
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
