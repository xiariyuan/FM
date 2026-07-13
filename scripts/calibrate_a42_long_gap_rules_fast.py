#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import pandas as pd

USECOLS = [
    'seq','track_a','track_b','gap','gap_bucket','same_gt',
    'appearance_max','a42_rule_score_v1','cos_global_global','cos_high_high',
    'max_rank_by_a42_rule_score_v1','min_margin_by_a42_rule_score_v1',
    'max_rank_by_appearance_max','min_margin_by_appearance_max',
    'geometry_risk','motion_risk','edge_debt_score','zone_near_match','zone_exact_match',
    'center_distance_per_frame','predicted_distance_per_frame'
]
NUMERIC = [c for c in USECOLS if c not in {'seq','track_a','track_b','gap_bucket'}]

def load(path: str, train: bool):
    cols = [c for c in USECOLS if train or c != 'same_gt']
    df = pd.read_csv(path, usecols=cols)
    for c in df.columns:
        if c in NUMERIC:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    return df

def write_csv(path: Path, rows: list[dict]):
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields or ['rule'], extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

def build_arrays(df: pd.DataFrame, train: bool):
    arr={
        'gap': df['gap'].to_numpy(np.int32),
        'app': df['appearance_max'].to_numpy(np.float32),
        'score': df['a42_rule_score_v1'].to_numpy(np.float32),
        'gg': df['cos_global_global'].to_numpy(np.float32),
        'hh': df['cos_high_high'].to_numpy(np.float32),
        'rank_score': df['max_rank_by_a42_rule_score_v1'].to_numpy(np.float32),
        'margin_score': df['min_margin_by_a42_rule_score_v1'].to_numpy(np.float32),
        'rank_app': df['max_rank_by_appearance_max'].to_numpy(np.float32),
        'margin_app': df['min_margin_by_appearance_max'].to_numpy(np.float32),
        'geom': df['geometry_risk'].to_numpy(np.float32),
        'motion': df['motion_risk'].to_numpy(np.float32),
        'debt': df['edge_debt_score'].to_numpy(np.float32),
        'zone': df['zone_near_match'].to_numpy(np.float32),
        'bucket': df['gap_bucket'].astype(str).to_numpy(),
    }
    if train:
        arr['label'] = df['same_gt'].to_numpy(np.int8)
    return arr

def gap_mask(a, scope):
    g=a['gap']
    if scope == '1_60': return (g>=1)&(g<=60)
    if scope == '61_150': return (g>=61)&(g<=150)
    if scope == '151_300': return (g>=151)&(g<=300)
    if scope == '1_150': return (g>=1)&(g<=150)
    if scope == '1_300': return (g>=1)&(g<=300)
    raise KeyError(scope)

def mask_for(a, cfg):
    rank = a['rank_score'] if cfg['rank_col']=='score' else a['rank_app']
    margin = a['margin_score'] if cfg['rank_col']=='score' else a['margin_app']
    m = gap_mask(a, cfg['gap_scope'])
    m &= a['app'] >= cfg['min_app']
    m &= a['score'] >= cfg['min_score']
    m &= rank <= cfg['max_rank']
    m &= margin >= cfg['min_margin']
    m &= a['geom'] <= cfg['max_geom']
    m &= a['motion'] <= cfg['max_motion']
    if cfg['min_debt'] > 0:
        m &= a['debt'] >= cfg['min_debt']
    if cfg['zone_near']:
        m &= a['zone'] >= 1
    if cfg['min_gg'] is not None:
        m &= a['gg'] >= cfg['min_gg']
    if cfg['min_hh'] is not None:
        m &= a['hh'] >= cfg['min_hh']
    return m

def eval_train(a, cfg):
    m=mask_for(a,cfg)
    sel=int(m.sum())
    if sel == 0: return None, m
    tp=int(a['label'][m].sum())
    row={k:v for k,v in cfg.items()}
    row.update({'selected':sel,'tp':tp,'fp':sel-tp,'precision':tp/sel if sel else 0.0,'recall_all_true':tp/max(1,int(a['label'].sum()))})
    for b in ['1_60','61_150','151_300']:
        mb=m & (a['bucket']==b)
        s=int(mb.sum()); t=int(a['label'][mb].sum()) if s else 0
        row[f'{b}_selected']=s; row[f'{b}_tp']=t; row[f'{b}_precision']=t/s if s else 0.0
    row['rule']=f"{cfg['gap_scope']}_{cfg['rank_col']}_app{cfg['min_app']:.2f}_score{cfg['min_score']:.2f}_rank{cfg['max_rank']}_m{cfg['min_margin']:.2f}_g{cfg['max_geom']}_mo{cfg['max_motion']}_d{cfg['min_debt']}_z{int(cfg['zone_near'])}_gg{cfg['min_gg'] if cfg['min_gg'] is not None else 'none'}_hh{cfg['min_hh'] if cfg['min_hh'] is not None else 'none'}"
    row['objective_balanced'] = tp - 0.45*(sel-tp) + 0.15*row['precision']
    row['objective_precision'] = 10*row['precision'] + 0.05*tp - 0.02*(sel-tp)
    return row, m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--out-dir', required=True)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print('[load] train')
    train=load(args.train, True)
    print('[load] test')
    test=load(args.test, False)
    ta=build_arrays(train, True); xa=build_arrays(test, False)
    configs=[]
    for gap_scope in ['1_60','1_150','1_300','61_150','151_300']:
      for min_app in [0.75,0.80,0.85,0.90]:
       for min_score in [0.50,0.55,0.60,0.65,0.70]:
        for max_rank in [1,2,3]:
         for min_margin in [-0.05,0.0,0.02,0.05,0.10]:
          for max_geom,max_motion in [(0,0),(0,1),(1,1),(1,2)]:
           for rank_col in ['score','app']:
            for zone_near in [False, True]:
             # keep min_debt/gg/hh sparse to avoid exploding grid
             for min_debt in [0,1]:
              for min_gg,min_hh in [(None,None),(0.60,None),(0.65,None),(None,0.65)]:
               cfg={'gap_scope':gap_scope,'min_app':min_app,'min_score':min_score,'max_rank':max_rank,'min_margin':min_margin,'max_geom':max_geom,'max_motion':max_motion,'rank_col':rank_col,'zone_near':zone_near,'min_debt':min_debt,'min_gg':min_gg,'min_hh':min_hh}
               row,_=eval_train(ta,cfg)
               if row and row['selected']>=2:
                   configs.append(row)
    configs=sorted(configs, key=lambda r:(r['objective_balanced'], r['precision'], r['tp']), reverse=True)
    # Evaluate test selection only for top candidates/views to save time.
    chosen=[]; seen=set()
    views={
        'balanced_top': configs[:300],
        'precision_ge_070': [r for r in configs if r['precision']>=0.70 and r['selected']>=5][:200],
        'precision_ge_060': [r for r in configs if r['precision']>=0.60 and r['selected']>=10][:200],
        'precision_ge_050': [r for r in configs if r['precision']>=0.50 and r['selected']>=20][:200],
        'tp_ge_100': [r for r in configs if r['tp']>=100][:200],
        'long_gap_tp': [r for r in configs if (r['61_150_tp']+r['151_300_tp'])>=5][:200],
    }
    for rs in views.values():
        for r in rs:
            if r['rule'] not in seen:
                chosen.append(r); seen.add(r['rule'])
    for r in chosen:
        cfg={k:r[k] for k in ['gap_scope','min_app','min_score','max_rank','min_margin','max_geom','max_motion','rank_col','zone_near','min_debt','min_gg','min_hh']}
        mt=mask_for(xa,cfg)
        r['test_selected']=int(mt.sum())
        for b in ['1_60','61_150','151_300']:
            r[f'{b}_test_selected']=int((mt & (xa['bucket']==b)).sum())
    write_csv(out/'a42_rule_scorecard_all_train.csv', configs)
    for name,rs in views.items():
        write_csv(out/f'{name}.csv', rs)
    write_csv(out/'a42_rule_policy_candidates.csv', chosen)
    best={name:(rs[0] if rs else None) for name,rs in views.items()}
    summary={'train_rows':len(train),'test_rows':len(test),'train_true':int(ta['label'].sum()),'rules_total_train':len(configs),'policy_candidates':len(chosen),'best':best,'decision':'A42_02a_FAST_RULE_CALIBRATION_DONE'}
    (out/'a42_02a_fast_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (out/'decision.md').write_text('# A42_02a Fast Rule Calibration\n\n```json\n'+json.dumps(summary,indent=2,sort_keys=True)[:14000]+'\n```\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()
