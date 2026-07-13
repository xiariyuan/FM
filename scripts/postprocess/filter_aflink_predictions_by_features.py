#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def ff(x, d=0.0):
    try: return float(x)
    except Exception: return d


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--summary', default='')
    ap.add_argument('--min-score', type=float, default=0.2)
    ap.add_argument('--min-velocity-cosine', type=float, default=-2.0)
    ap.add_argument('--max-pred-dist-pf', type=float, default=1e9)
    ap.add_argument('--max-center-dist-pf', type=float, default=1e9)
    ap.add_argument('--max-height-ratio', type=float, default=1e9)
    ap.add_argument('--max-area-ratio', type=float, default=1e9)
    ap.add_argument('--min-cos-global-global', type=float, default=-2.0)
    ap.add_argument('--min-cos-high-high', type=float, default=-2.0)
    args=ap.parse_args()
    rows=[]; kept=[]
    with open(args.input, encoding='utf-8') as f:
        reader=csv.DictReader(f)
        fields=reader.fieldnames or []
        for r in reader:
            rows.append(r)
            s=ff(r.get('aflink_score', r.get('score', 0)))
            ok=(s>=args.min_score and
                ff(r.get('velocity_cosine'))>=args.min_velocity_cosine and
                ff(r.get('predicted_distance_per_frame'))<=args.max_pred_dist_pf and
                ff(r.get('center_distance_per_frame'))<=args.max_center_dist_pf and
                ff(r.get('height_ratio'))<=args.max_height_ratio and
                ff(r.get('area_ratio'))<=args.max_area_ratio and
                ff(r.get('cos_global_global'))>=args.min_cos_global_global and
                ff(r.get('cos_high_high'))>=args.min_cos_high_high)
            if ok:
                rr=dict(r)
                if 'aflink_score' not in rr and 'score' in rr: rr['aflink_score']=rr['score']
                kept.append(rr)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if kept:
        fields=list(kept[0].keys())
        with open(args.output,'w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(kept)
    else:
        with open(args.output,'w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    tp=sum(int(float(r.get('same_gt',0) or 0)) for r in kept)
    summary={'input_rows':len(rows),'kept_rows':len(kept),'tp_train_label':tp,'precision_train_label':tp/len(kept) if kept else 0,'params':vars(args)}
    if args.summary: Path(args.summary).write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()
