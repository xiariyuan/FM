#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path


def xywh_to_xyxy(b):
    x,y,w,h=b; return [x,y,x+w,y+h]

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1)
    inter=iw*ih
    if inter<=0: return 0.0
    aa=max(0,ax2-ax1)*max(0,ay2-ay1); bb=max(0,bx2-bx1)*max(0,by2-by1)
    return inter/max(1e-9,aa+bb-inter)

def greedy(gts,dets,thr):
    used=set(); tp=0; ious=[]
    for d in dets:
        db=xywh_to_xyxy(d['bbox']); best=-1; bv=0.0
        for gi,g in enumerate(gts):
            if gi in used: continue
            v=iou(db, xywh_to_xyxy(g['bbox']))
            if v>bv: best,bv=gi,v
        if best>=0 and bv>=thr:
            used.add(best); tp+=1; ious.append(bv)
    fp=len(dets)-tp; fn=len(gts)-tp
    return tp,fp,fn,ious

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--preds', required=True)
    ap.add_argument('--gt', default='external/BoT-SORT-main/datasets/MIX_CH_MOT17_MOT20/annotations/val_mot20_quick.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--thresholds', default='0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.12,0.15,0.20,0.25,0.30,0.40,0.50')
    args=ap.parse_args()
    preds=json.loads(Path(args.preds).read_text())
    gtj=json.loads(Path(args.gt).read_text())
    gt_by_img=defaultdict(list)
    img_ids=[]
    for im in gtj['images']:
        img_ids.append(int(im['id']))
    for a in gtj['annotations']:
        if int(a.get('iscrowd',0)): continue
        gt_by_img[int(a['image_id'])].append(a)
    pred_by_img=defaultdict(list)
    for d in preds:
        pred_by_img[int(d['image_id'])].append(d)
    total_gt=sum(len(v) for v in gt_by_img.values())
    rows=[]
    for s in [float(x) for x in args.thresholds.split(',') if x.strip()]:
        stats={'score':s,'gt':total_gt,'detections':0}
        for t in [0.5,0.75]:
            TP=FP=FN=0; all_ious=[]
            for img_id in img_ids:
                gts=gt_by_img.get(img_id,[])
                dets=[d for d in pred_by_img.get(img_id,[]) if d['score']>=s]
                dets.sort(key=lambda x:x['score'], reverse=True)
                if t==0.5:
                    stats['detections'] += len(dets)
                tp,fp,fn,ivs=greedy(gts,dets,t)
                TP+=tp; FP+=fp; FN+=fn; all_ious.extend(ivs)
            stats[f'tp{int(t*100)}']=TP; stats[f'fp{int(t*100)}']=FP; stats[f'fn{int(t*100)}']=FN
            stats[f'precision{int(t*100)}']=TP/max(1,TP+FP)
            stats[f'recall{int(t*100)}']=TP/max(1,total_gt)
            stats[f'f1_{int(t*100)}']=2*stats[f'precision{int(t*100)}']*stats[f'recall{int(t*100)}']/max(1e-9,stats[f'precision{int(t*100)}']+stats[f'recall{int(t*100)}'])
            stats[f'mean_iou{int(t*100)}']=sum(all_ious)/len(all_ious) if all_ious else 0.0
        rows.append(stats)
    out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    # Print compact table and Pareto-like useful settings.
    print('score det tp50 fp50 fn50 prec50 rec50 f1_50 tp75 fp75 fn75 prec75 rec75 f1_75')
    for r in rows:
        print(f"{r['score']:.3f} {r['detections']} {r['tp50']} {r['fp50']} {r['fn50']} {r['precision50']:.4f} {r['recall50']:.4f} {r['f1_50']:.4f} {r['tp75']} {r['fp75']} {r['fn75']} {r['precision75']:.4f} {r['recall75']:.4f} {r['f1_75']:.4f}")
    base=rows[0]
    print('\nCandidates: FP50 reduction with recall50 drop <= 0.2%, 0.5%, 1.0%')
    for lim in [0.002,0.005,0.01]:
        cands=[r for r in rows if base['recall50']-r['recall50'] <= lim]
        if not cands: continue
        best=max(cands, key=lambda r: base['fp50']-r['fp50'])
        print(f"limit={lim:.3f}: score={best['score']:.3f}, fp50_delta={best['fp50']-base['fp50']}, fn50_delta={best['fn50']-base['fn50']}, prec50={best['precision50']:.4f}, rec50={best['recall50']:.4f}, f1_50={best['f1_50']:.4f}, fp75_delta={best['fp75']-base['fp75']}, fn75_delta={best['fn75']-base['fn75']}")
    best_f1=max(rows, key=lambda r:r['f1_50'])
    print(f"best_f1_50: score={best_f1['score']:.3f}, f1={best_f1['f1_50']:.4f}, prec={best_f1['precision50']:.4f}, rec={best_f1['recall50']:.4f}, fp={best_f1['fp50']}, fn={best_f1['fn50']}")
if __name__=='__main__': main()
