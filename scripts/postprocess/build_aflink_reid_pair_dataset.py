#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

REID_FEATURES = ['cos_end_start','cos_end_global','cos_global_start','cos_global_global','cos_high_high','cos_start_start','cos_end_end','appearance_mean','appearance_max','appearance_min','appearance_std','appearance_gap_consistency']

def stat(xs):
    if not xs: return {'n':0,'mean':0,'std':0,'min':0,'p05':0,'p25':0,'p50':0,'p75':0,'p95':0,'max':0}
    a=np.asarray(xs,dtype=float)
    return {'n':int(len(a)),'mean':float(a.mean()),'std':float(a.std()),'min':float(a.min()),'p05':float(np.percentile(a,5)),'p25':float(np.percentile(a,25)),'p50':float(np.percentile(a,50)),'p75':float(np.percentile(a,75)),'p95':float(np.percentile(a,95)),'max':float(a.max())}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--features-npz', required=True)
    ap.add_argument('--features-index', required=True)
    ap.add_argument('--out-dir', required=True)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    data=np.load(args.features_npz)
    # Materialize compressed npz arrays once. Accessing data[name][i] repeatedly
    # would repeatedly decompress the array and can be extremely slow.
    mats={name:data[name].astype(np.float32) for name in ['start','end','global_mean','high_score']}
    keys=[]
    with open(args.features_index) as f:
        for r in csv.DictReader(f): keys.append((r['seq'], int(r['track_id'])))
    fmap={}
    for i,k in enumerate(keys):
        fmap[k]={name:mats[name][i] for name in ['start','end','global_mean','high_score']}
    def cos(a,b): return float(np.dot(a,b))
    rows=[]; valid=missing=0; pos=neg=0
    with open(args.pairs) as f:
        reader=csv.DictReader(f)
        for r in reader:
            seq=r['seq']; a_id=int(float(r['track_a'])); b_id=int(float(r['track_b']))
            fa=fmap.get((seq,a_id)); fb=fmap.get((seq,b_id))
            rr=dict(r)
            if int(float(rr.get('same_gt',0)))==1: pos+=1
            else: neg+=1
            if fa is None or fb is None:
                missing+=1; rr['has_reid']=0
                for k in REID_FEATURES: rr[k]=0.0
            else:
                valid+=1; rr['has_reid']=1
                vals=[cos(fa['end'],fb['start']),cos(fa['end'],fb['global_mean']),cos(fa['global_mean'],fb['start']),cos(fa['global_mean'],fb['global_mean']),cos(fa['high_score'],fb['high_score']),cos(fa['start'],fb['start']),cos(fa['end'],fb['end'])]
                for k,v in zip(REID_FEATURES[:7], vals): rr[k]=v
                rr['appearance_mean']=float(np.mean(vals)); rr['appearance_max']=float(np.max(vals)); rr['appearance_min']=float(np.min(vals)); rr['appearance_std']=float(np.std(vals)); rr['appearance_gap_consistency']=float(vals[0]-vals[3])
            rows.append(rr)
    out_csv=out/'aflink_pair_candidates_with_reid.csv'
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    metrics=['cos_end_start','cos_global_global','cos_high_high','appearance_mean','appearance_max','appearance_min']
    summary={'rows':len(rows),'valid_reid':valid,'missing_reid':missing,'positive':pos,'negative':neg,'positive_rate':pos/len(rows) if rows else 0.0,'metrics':{}}
    for m in metrics:
        p=[float(r[m]) for r in rows if int(float(r['same_gt']))==1 and int(float(r['has_reid']))==1]
        n=[float(r[m]) for r in rows if int(float(r['same_gt']))==0 and int(float(r['has_reid']))==1]
        summary['metrics'][m]={'positive':stat(p),'negative':stat(n),'gap_mean':float(np.mean(p)-np.mean(n)) if p and n else 0.0}
    thr=[]
    for m in ['cos_end_start','appearance_mean','appearance_max','cos_high_high']:
        for t in [0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
            sel=[r for r in rows if int(float(r['has_reid']))==1 and float(r[m])>=t]
            tp=sum(int(float(r['same_gt'])) for r in sel)
            thr.append({'metric':m,'threshold':t,'n':len(sel),'tp':tp,'precision':tp/len(sel) if sel else 0.0})
    summary['threshold_report']=thr
    (out/'pair_dataset_reid_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    md=['# A23_02 Pair Dataset With ReID','','| metric | value |','|---|---:|',f"| rows | {summary['rows']} |",f"| valid_reid | {summary['valid_reid']} |",f"| missing_reid | {summary['missing_reid']} |",f"| positive | {summary['positive']} |",f"| negative | {summary['negative']} |",f"| positive_rate | {summary['positive_rate']:.6f} |",'','## Cosine distributions','| feature | pos_mean | neg_mean | gap | pos_p50 | neg_p50 | pos_p75 | neg_p75 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,s in summary['metrics'].items(): md.append(f"| {m} | {s['positive']['mean']:.4f} | {s['negative']['mean']:.4f} | {s['gap_mean']:.4f} | {s['positive']['p50']:.4f} | {s['negative']['p50']:.4f} | {s['positive']['p75']:.4f} | {s['negative']['p75']:.4f} |")
    md+=['','## Threshold report top rows','| metric | threshold | n | tp | precision |','|---|---:|---:|---:|---:|']
    for r in sorted(thr, key=lambda x:(-x['precision'],-x['tp']))[:30]: md.append(f"| {r['metric']} | {r['threshold']} | {r['n']} | {r['tp']} | {r['precision']:.4f} |")
    (out/'pair_dataset_reid_summary.md').write_text('\n'.join(md)+'\n')
    print(f"done rows={len(rows)} valid={valid} pos={pos} out={out_csv}")
if __name__=='__main__': main()
