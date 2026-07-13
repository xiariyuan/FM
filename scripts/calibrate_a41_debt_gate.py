#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import pandas as pd
import numpy as np


def write_csv(path, rows):
    rows=list(rows)
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or ['name'],extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def load_df(path):
    df=pd.read_csv(path)
    for c in ['aflink_score','debt_adjusted_edge_score','out_rank_by_aflink_score','in_rank_by_aflink_score','out_margin_to_second_aflink_score','in_margin_to_second_aflink_score','edge_debt_score','risk_total','geometry_risk','motion_risk','competition_risk']:
        if c in df.columns:
            df[c]=pd.to_numeric(df[c], errors='coerce').fillna(0)
    if 'same_gt' in df.columns:
        df['same_gt_num']=pd.to_numeric(df['same_gt'], errors='coerce').fillna(0).astype(int)
    else:
        df['same_gt_num']=0
    return df


def mask_for(df,cfg):
    m=(df['aflink_score']>=cfg['min_aflink'])
    m &= (df['debt_adjusted_edge_score']>=cfg['min_adjusted'])
    m &= (df['out_rank_by_aflink_score']<=cfg['max_rank'])
    m &= (df['in_rank_by_aflink_score']<=cfg['max_rank'])
    m &= (np.minimum(df['out_margin_to_second_aflink_score'], df['in_margin_to_second_aflink_score'])>=cfg['min_margin'])
    m &= (df['edge_debt_score']>=cfg['min_debt'])
    m &= (df['risk_total']<=cfg['max_risk'])
    m &= (df['geometry_risk']<=1)
    m &= (df['motion_risk']<=1)
    m &= (df['competition_risk']<=2)
    edge_set=cfg['edge_set']
    if edge_set=='no_highrisk':
        m &= (df['edge_type']!='high_risk_geometry')
    elif edge_set=='boundary_or_fragment':
        m &= df['edge_type'].isin(['weak_boundary_recovery','fragmented_tracklet_recovery'])
    elif edge_set=='stable_gap':
        m &= df['edge_type'].isin(['short_gap_continuation','long_gap_reappearance'])
    return m


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--out-dir', required=True)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    train=load_df(args.train); test=load_df(args.test)
    pos=int(train['same_gt_num'].sum())
    configs=[]
    grids={
        'min_aflink':[0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95],
        'min_adjusted':[0.0,0.20,0.50,0.70,0.90],
        'max_rank':[1,2],
        'min_margin':[0.02,0.03,0.05,0.08,0.10,0.15,0.20],
        'min_debt':[1,2,3,4],
        'max_risk':[0,1,2,3],
        'edge_set':['all','no_highrisk','boundary_or_fragment','stable_gap'],
    }
    for min_aflink in grids['min_aflink']:
     base1=train['aflink_score']>=min_aflink
     if not base1.any(): continue
     for min_adjusted in grids['min_adjusted']:
      for max_rank in grids['max_rank']:
       for min_margin in grids['min_margin']:
        for min_debt in grids['min_debt']:
         for max_risk in grids['max_risk']:
          for edge_set in grids['edge_set']:
           cfg={'min_aflink':min_aflink,'min_adjusted':min_adjusted,'max_rank':max_rank,'min_margin':min_margin,'min_debt':min_debt,'max_risk':max_risk,'edge_set':edge_set}
           mt=mask_for(train,cfg)
           selected=int(mt.sum())
           if selected < 5: continue
           tp=int(train.loc[mt,'same_gt_num'].sum())
           precision=tp/selected if selected else 0
           recall=tp/pos if pos else 0
           mtest=mask_for(test,cfg)
           test_selected=int(mtest.sum())
           configs.append({**cfg,'selected':selected,'tp':tp,'precision':precision,'recall':recall,'test_selected':test_selected,'objective_balanced':tp + 0.02*test_selected - 20*max(0,0.70-precision),'objective_strict':tp + 0.01*test_selected - 30*max(0,0.80-precision)})
    configs=sorted(configs, key=lambda r:(r['precision'],r['tp'],r['test_selected']), reverse=True)
    write_csv(out/'a41_gate_sweep_all.csv', configs)
    views={
      'precision_ge_080':[r for r in configs if r['precision']>=0.80 and r['selected']>=10],
      'precision_ge_070':[r for r in configs if r['precision']>=0.70 and r['selected']>=20],
      'precision_ge_060':[r for r in configs if r['precision']>=0.60 and r['selected']>=50],
    }
    for name,rs in views.items():
        rs=sorted(rs,key=lambda r:(r['tp'],r['test_selected'],r['precision']), reverse=True)
        write_csv(out/f'{name}.csv', rs[:200])
    best_strict=(sorted(views['precision_ge_080'], key=lambda r:(r['tp'],r['test_selected'],r['precision']), reverse=True)[:1] or [None])[0]
    best_balanced=(sorted(views['precision_ge_070'], key=lambda r:(r['tp'],r['test_selected'],r['precision']), reverse=True)[:1] or [None])[0]
    best_aggressive=(sorted(views['precision_ge_060'], key=lambda r:(r['tp'],r['test_selected'],r['precision']), reverse=True)[:1] or [None])[0]
    summary={'train_edges':len(train),'test_edges':len(test),'train_positive_edges':pos,'total_configs':len(configs),'best_strict_p80':best_strict,'best_balanced_p70':best_balanced,'best_aggressive_p60':best_aggressive,'decision':'A41_01b_GATE_CALIBRATION_DONE'}
    (out/'gate_calibration_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (out/'decision.md').write_text('# A41_01b Gate Calibration\n\n```json\n'+json.dumps(summary,indent=2,sort_keys=True)+'\n```\n\nNext: use balanced/strict gate as input policy for A41_02 global one-to-one solver.\n')
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__': main()
