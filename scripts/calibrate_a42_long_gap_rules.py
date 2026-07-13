#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import pandas as pd
import numpy as np

USECOLS=['seq','track_a','track_b','gap','gap_bucket','same_gt','appearance_max','a42_rule_score_v1','max_rank_by_a42_rule_score_v1','min_margin_by_a42_rule_score_v1','max_rank_by_appearance_max','min_margin_by_appearance_max','geometry_risk','motion_risk','edge_debt_score','zone_near_match','zone_exact_match','has_reid','cos_global_global','cos_high_high','center_distance_per_frame','predicted_distance_per_frame']

def load(path, train=True):
    cols=[c for c in USECOLS if c!='same_gt' or train]
    df=pd.read_csv(path,usecols=cols)
    for c in df.columns:
        if c not in ['seq','track_a','track_b','gap_bucket']:
            df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0)
    if train and 'same_gt' in df.columns:
        df['same_gt']=df['same_gt'].astype(np.int8)
    return df

def mask_gap(df,scope):
    g=df['gap']
    if scope=='all' or scope=='1_300': return (g>=1)&(g<=300)
    if scope=='1_60': return (g>=1)&(g<=60)
    if scope=='61_150': return (g>=61)&(g<=150)
    if scope=='151_300': return (g>=151)&(g<=300)
    if scope=='1_150': return (g>=1)&(g<=150)
    return pd.Series(False,index=df.index)

def eval_rule(df, test, cfg):
    m=mask_gap(df,cfg['gap_scope'])
    m &= df['appearance_max']>=cfg['min_app']
    m &= df['a42_rule_score_v1']>=cfg['min_score']
    m &= df['max_rank_by_a42_rule_score_v1']<=cfg['max_rank']
    m &= df['min_margin_by_a42_rule_score_v1']>=cfg['min_margin']
    m &= df['geometry_risk']<=cfg['max_geom']
    m &= df['motion_risk']<=cfg['max_motion']
    m &= df['edge_debt_score']>=cfg['min_debt']
    if cfg['zone_near']:
        m &= df['zone_near_match']>=1
    if cfg['min_gg'] is not None:
        m &= df['cos_global_global']>=cfg['min_gg']
    sel=int(m.sum())
    if sel==0: return None
    tp=int(df.loc[m,'same_gt'].sum())
    mt=mask_gap(test,cfg['gap_scope'])
    mt &= test['appearance_max']>=cfg['min_app']
    mt &= test['a42_rule_score_v1']>=cfg['min_score']
    mt &= test['max_rank_by_a42_rule_score_v1']<=cfg['max_rank']
    mt &= test['min_margin_by_a42_rule_score_v1']>=cfg['min_margin']
    mt &= test['geometry_risk']<=cfg['max_geom']
    mt &= test['motion_risk']<=cfg['max_motion']
    mt &= test['edge_debt_score']>=cfg['min_debt']
    if cfg['zone_near']:
        mt &= test['zone_near_match']>=1
    if cfg['min_gg'] is not None:
        mt &= test['cos_global_global']>=cfg['min_gg']
    test_sel=int(mt.sum())
    row={k:v for k,v in cfg.items()}
    row.update({'selected':sel,'tp':tp,'fp':sel-tp,'precision':tp/sel if sel else 0.0,'recall_of_all_true':tp/max(1,int(df['same_gt'].sum())),'test_selected':test_sel})
    # per bucket true/select diagnostics
    for b in ['1_60','61_150','151_300']:
        mb=m & (df['gap_bucket']==b)
        s=int(mb.sum()); t=int(df.loc[mb,'same_gt'].sum()) if s else 0
        row[f'{b}_selected']=s; row[f'{b}_tp']=t; row[f'{b}_precision']=t/s if s else 0.0
        tb=int((mt & (test['gap_bucket']==b)).sum()); row[f'{b}_test_selected']=tb
    row['rule']=f"{cfg['gap_scope']}_app{cfg['min_app']:.2f}_score{cfg['min_score']:.2f}_rank{cfg['max_rank']}_m{cfg['min_margin']:.2f}_g{cfg['max_geom']}_mo{cfg['max_motion']}_d{cfg['min_debt']}_z{int(cfg['zone_near'])}_gg{cfg['min_gg'] if cfg['min_gg'] is not None else 'none'}"
    return row

def write_csv(path, rows):
    rows=list(rows); fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); fields.append(k)
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or ['rule'],extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',required=True)
    ap.add_argument('--test',required=True)
    ap.add_argument('--out-dir',required=True)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    df=load(args.train,True); test=load(args.test,False)
    configs=[]
    for gap_scope in ['1_60','1_150','1_300','61_150','151_300']:
      for min_app in [0.70,0.75,0.80,0.85,0.90]:
       for min_score in [0.45,0.50,0.55,0.60,0.65,0.70]:
        for max_rank in [1,2,3,5]:
         for min_margin in [-0.05,0.0,0.02,0.05,0.10]:
          for max_geom in [0,1]:
           for max_motion in [0,1,2]:
            for min_debt in [0,1]:
             for zone_near in [False,True]:
              for min_gg in [None,0.55,0.60,0.65,0.70]:
               cfg={'gap_scope':gap_scope,'min_app':min_app,'min_score':min_score,'max_rank':max_rank,'min_margin':min_margin,'max_geom':max_geom,'max_motion':max_motion,'min_debt':min_debt,'zone_near':zone_near,'min_gg':min_gg}
               row=eval_rule(df,test,cfg)
               if row and row['selected']>=2:
                row['objective_submit']=row['tp'] - 0.40*row['fp'] + 0.005*row['test_selected']
                row['objective_precision']=row['precision']*10 + row['tp']*0.1 - row['fp']*0.05
                configs.append(row)
    configs=sorted(configs,key=lambda r:(r['objective_submit'],r['precision'],r['tp']),reverse=True)
    write_csv(out/'a42_rule_scorecard_all.csv',configs)
    # useful views
    views={
      'precision_ge_070':[r for r in configs if r['precision']>=0.70 and r['selected']>=5],
      'precision_ge_060':[r for r in configs if r['precision']>=0.60 and r['selected']>=10],
      'precision_ge_050':[r for r in configs if r['precision']>=0.50 and r['selected']>=20],
      'tp_ge_100':[r for r in configs if r['tp']>=100],
      'long_gap_tp':[r for r in configs if r['61_150_tp']+r['151_300_tp']>=10],
    }
    for name,rs in views.items():
        write_csv(out/f'{name}.csv',rs[:200])
    best={name:(rs[0] if rs else None) for name,rs in views.items()}
    summary={'train_rows':len(df),'test_rows':len(test),'train_true':int(df['same_gt'].sum()),'rules':len(configs),'best':best,'decision':'A42_02a_RULE_CALIBRATION_DONE'}
    (out/'a42_02a_rule_calibration_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (out/'decision.md').write_text('# A42_02a Rule Calibration\n\n```json\n'+json.dumps(summary,indent=2,sort_keys=True)[:12000]+'\n```\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()
