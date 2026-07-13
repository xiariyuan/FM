#!/usr/bin/env python3
"""Offline diagnostic for TrustTrack reliability-reversal signals.

This does not modify tracker behavior. It reconstructs online track states from
an existing MOT result file, matches result boxes back to detection-dump boxes,
computes cue margins, and checks whether high reliability-reversal scores are
associated with wrong/unstable identity events.
"""
from __future__ import annotations
import argparse, csv, math, os
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, field
import numpy as np

CUES = ('app', 'motion', 'iou', 'shape')


def iou_one(a, b):
    xx1=max(a[0],b[0]); yy1=max(a[1],b[1]); xx2=min(a[2],b[2]); yy2=min(a[3],b[3])
    inter=max(0.0, xx2-xx1)*max(0.0, yy2-yy1)
    aa=max(1e-9, (a[2]-a[0])*(a[3]-a[1])); bb=max(1e-9, (b[2]-b[0])*(b[3]-b[1]))
    return inter/max(1e-9, aa+bb-inter)


def center(box):
    return ((box[0]+box[2])*0.5, (box[1]+box[3])*0.5)


def wh(box):
    return (max(1e-6, box[2]-box[0]), max(1e-6, box[3]-box[1]))


def norm_feat(x):
    x = x.astype(np.float32, copy=False)
    n = float(np.linalg.norm(x))
    return x / max(n, 1e-12)


def load_gt(seq, root):
    path=os.path.join(root, seq, 'gt', 'gt.txt')
    by=defaultdict(list)
    for line in open(path):
        v=line.strip().split(',')
        if len(v)<6: continue
        fr=int(float(v[0])); gid=int(float(v[1])); x=float(v[2]); y=float(v[3]); w=float(v[4]); h=float(v[5])
        mark=int(float(v[6])) if len(v)>6 else 1
        cls=int(float(v[7])) if len(v)>7 else 1
        if mark==1 and cls==1:
            by[fr].append((gid, (x,y,x+w,y+h)))
    return by


def best_gt(fr, box, gt, thr=0.5):
    best_gid=-1; best_iou=0.0
    for gid,gbox in gt.get(fr, []):
        ov=iou_one(box,gbox)
        if ov>best_iou:
            best_gid=gid; best_iou=ov
    if best_iou >= thr:
        return best_gid, best_iou
    return -1, best_iou


def load_results(path):
    by=defaultdict(list)
    for line in open(path):
        v=line.strip().split(',')
        if len(v)<6: continue
        fr=int(float(v[0])); tid=int(float(v[1])); x=float(v[2]); y=float(v[3]); w=float(v[4]); h=float(v[5])
        by[fr].append((tid, (x,y,x+w,y+h)))
    return by


@dataclass
class TrackState:
    box: tuple
    feat: np.ndarray
    last_frame: int
    hist: dict = field(default_factory=lambda: {c:0.0 for c in CUES})
    gt_hist: deque = field(default_factory=lambda: deque(maxlen=50))


def score_cues(state: TrackState, det_boxes, det_feats):
    n=len(det_boxes)
    out={c:np.zeros(n, dtype=np.float32) for c in CUES}
    if n == 0:
        return out
    # appearance cosine, features assumed normalized.
    out['app'] = det_feats @ state.feat
    pcx,pcy=center(state.box); pw,ph=wh(state.box)
    prev_scale=max(1.0, math.sqrt(pw*ph))
    for k,box in enumerate(det_boxes):
        dcx,dcy=center(box); dw,dh=wh(box)
        dist=math.hypot(dcx-pcx, dcy-pcy) / prev_scale
        out['motion'][k]=math.exp(-dist)
        out['iou'][k]=iou_one(state.box, box)
        aspect_prev=pw/ph; aspect_det=dw/dh
        size_pen=abs(math.log(max(dh,1e-6)/max(ph,1e-6))) + abs(math.log(max(aspect_det,1e-6)/max(aspect_prev,1e-6)))
        out['shape'][k]=math.exp(-size_pen)
    return out


def signed_row_margin(scores, chosen_idx):
    chosen=float(scores[chosen_idx])
    if len(scores)<=1:
        return 1.0, chosen, -1.0
    mask=np.ones(len(scores), dtype=bool); mask[chosen_idx]=False
    other=float(np.max(scores[mask]))
    return chosen-other, chosen, other


def signed_col_margin(track_scores, chosen_tid):
    chosen=float(track_scores.get(chosen_tid, -1.0))
    others=[v for k,v in track_scores.items() if k != chosen_tid]
    if not others:
        return 1.0, chosen, -1.0
    other=float(max(others))
    return chosen-other, chosen, other


def majority_gt(hist):
    vals=[x for x in hist if x != -1]
    if not vals: return -1
    return Counter(vals).most_common(1)[0][0]


def run(args):
    dump=np.load(args.dump_npz, allow_pickle=True)
    det=dump['detections']; feats=dump['features'].astype(np.float32)
    cols={str(x):i for i,x in enumerate(dump['columns'].tolist())}
    # Normalize features once.
    norms=np.linalg.norm(feats, axis=1, keepdims=True)
    feats=feats/np.maximum(norms, 1e-12)
    frame_to_detidx=defaultdict(list)
    for idx,r in enumerate(det):
        if float(r[cols['score']]) >= args.min_det_score:
            frame_to_detidx[int(r[cols['frame']])].append(idx)
    res_by=load_results(args.result_txt)
    gt=load_gt(args.seq, args.gt_root)
    states={}
    rows=[]
    for fr in sorted(res_by.keys()):
        det_indices=frame_to_detidx.get(fr, [])
        det_boxes=[]; det_scores=[]; det_gids=[]
        for idx in det_indices:
            r=det[idx]
            box=(float(r[cols['x1']]),float(r[cols['y1']]),float(r[cols['x2']]),float(r[cols['y2']]))
            det_boxes.append(box); det_scores.append(float(r[cols['score']])); det_gids.append(int(r[cols['global_det_idx']]))
        det_feats=feats[det_indices] if det_indices else np.zeros((0, feats.shape[1]), dtype=np.float32)
        # Match output boxes back to detection boxes.
        matched=[]
        used=set()
        for tid,obox in res_by[fr]:
            if det_boxes:
                ovs=[iou_one(obox,db) for db in det_boxes]
                j=int(np.argmax(ovs)); ov=float(ovs[j])
                if ov >= args.match_iou and j not in used:
                    used.add(j); matched.append((tid, obox, j, ov))
                else:
                    matched.append((tid, obox, None, ov if det_boxes else 0.0))
            else:
                matched.append((tid, obox, None, 0.0))
        # Precompute per-track per-cue scores for current detections using previous states.
        cue_scores_by_tid={}
        active_prev_tids=[tid for tid in states.keys()]
        for tid in active_prev_tids:
            cue_scores_by_tid[tid]=score_cues(states[tid], det_boxes, det_feats)
        # Create events before updating states.
        updates=[]
        for tid, obox, j, match_iou in matched:
            if j is None:
                # no detection feature; still update with previous if possible skipped.
                dg,di=best_gt(fr, obox, gt)
                updates.append((tid, obox, None, dg))
                continue
            dg,di=best_gt(fr, det_boxes[j], gt)
            if tid not in states or not cue_scores_by_tid:
                updates.append((tid, det_boxes[j], det_feats[j], dg))
                continue
            st=states[tid]
            prev_gt=majority_gt(st.gt_hist)
            row_m={}; col_m={}; pair_m={}; chosen_scores={}; other_scores={}
            for c in CUES:
                scores=cue_scores_by_tid[tid][c]
                rm, cs, ro=signed_row_margin(scores, j)
                # column scores for this det across all previous track states.
                ts={otid: float(cue_scores_by_tid[otid][c][j]) for otid in active_prev_tids if otid in cue_scores_by_tid and len(cue_scores_by_tid[otid][c])>j}
                cm, _, co=signed_col_margin(ts, tid)
                row_m[c]=rm; col_m[c]=cm; pair_m[c]=min(rm, cm)
                chosen_scores[c]=cs; other_scores[c]=max(ro, co)
            hist_best=max(CUES, key=lambda c: st.hist.get(c,0.0))
            curr_best=max(CUES, key=lambda c: pair_m[c])
            hist_best_margin=pair_m[hist_best]
            curr_best_margin=pair_m[curr_best]
            rrs=max(0.0, curr_best_margin - hist_best_margin) if curr_best != hist_best else 0.0
            collapse=1.0 - max(pair_m.values())
            if dg == -1:
                cls='det_fp'
            elif prev_gt == -1:
                cls='unknown'
            elif dg == prev_gt:
                cls='correct'
            else:
                cls='wrong_id'
            row={
                'seq':args.seq,'frame':fr,'track_id':tid,'det_global_idx':det_gids[j],
                'det_score':det_scores[j], 'match_iou_to_dump':match_iou,
                'det_gt':dg,'det_iou':di,'prev_track_gt':prev_gt,'match_cls':cls,
                'hist_best_cue':hist_best,'curr_best_cue':curr_best,'rrs':rrs,'cue_collapse':collapse,
            }
            for c in CUES:
                row[f'{c}_row_margin']=row_m[c]
                row[f'{c}_col_margin']=col_m[c]
                row[f'{c}_pair_margin']=pair_m[c]
                row[f'{c}_hist']=st.hist.get(c,0.0)
                row[f'{c}_chosen_score']=chosen_scores[c]
                row[f'{c}_other_score']=other_scores[c]
            rows.append(row)
            updates.append((tid, det_boxes[j], det_feats[j], dg))
        # Update states after all events in frame.
        for tid, box, feat, dg in updates:
            if feat is None:
                continue
            if tid in states:
                st=states[tid]
                # Update feature EMA.
                new_feat=0.9*st.feat + 0.1*feat
                new_feat=new_feat/max(float(np.linalg.norm(new_feat)),1e-12)
                st.feat=new_feat; st.box=box; st.last_frame=fr; st.gt_hist.append(dg)
                # Hist cue reliability from current event if available.
                # Use positive part of pair margins from most recent row for this tid/frame.
                if rows and rows[-1].get('frame')==fr and rows[-1].get('track_id')==tid:
                    for c in CUES:
                        st.hist[c]=0.9*st.hist.get(c,0.0)+0.1*max(0.0,float(rows[-1][f'{c}_pair_margin']))
            else:
                states[tid]=TrackState(box=box, feat=feat.copy(), last_frame=fr)
                states[tid].gt_hist.append(dg)
    # Future switch labels.
    by_tid=defaultdict(list)
    for idx,r in enumerate(rows): by_tid[int(r['track_id'])].append(idx)
    for tid,idxs in by_tid.items():
        idxs.sort(key=lambda k: int(rows[k]['frame']))
        for pos,k in enumerate(idxs):
            fr=int(rows[k]['frame']); dg=int(rows[k]['det_gt'])
            for K in args.future_k:
                fut=0
                for kk in idxs[pos+1:]:
                    if int(rows[kk]['frame']) - fr > K: break
                    ndg=int(rows[kk]['det_gt'])
                    if dg != -1 and ndg != -1 and ndg != dg:
                        fut=1; break
                rows[k][f'future_switch_{K}']=fut
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    fieldnames=[]
    for r in rows:
        for k in r.keys():
            if k not in fieldnames: fieldnames.append(k)
    with open(args.out_csv,'w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    print('wrote', args.out_csv, 'rows', len(rows))
    # Summary bins by RRS.
    def bad(r): return r['match_cls'] in ('wrong_id','det_fp') or any(int(r.get(f'future_switch_{K}',0)) for K in args.future_k)
    vals=[float(r['rrs']) for r in rows]
    print('class_counts', dict(Counter(r['match_cls'] for r in rows)))
    if vals:
        qs=np.quantile(vals, [0,0.5,0.75,0.9,0.95,0.99,1.0])
        print('rrs_quantiles', [round(float(x),5) for x in qs])
        for lo,hi in zip(qs[:-1], qs[1:]):
            sub=[r for r in rows if float(r['rrs'])>=lo and float(r['rrs'])<=hi]
            if sub:
                print('bin', round(float(lo),5), round(float(hi),5), 'n',len(sub), 'bad_rate', round(sum(bad(r) for r in sub)/len(sub),4), 'wrong_or_fp', sum(r['match_cls'] in ('wrong_id','det_fp') for r in sub))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seq', required=True)
    ap.add_argument('--dump-npz', required=True)
    ap.add_argument('--result-txt', required=True)
    ap.add_argument('--gt-root', default='/gemini/code/datasets/MOT20/train')
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--min-det-score', type=float, default=0.1)
    ap.add_argument('--match-iou', type=float, default=0.8)
    ap.add_argument('--future-k', type=int, nargs='+', default=[1,3,5,10])
    args=ap.parse_args(); run(args)
if __name__=='__main__': main()
