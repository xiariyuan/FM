#!/usr/bin/env python3
"""TrustTrack-soft-v2: cue-collapse-aware adaptive ReID memory update.

Minimal intervention version:
- Does NOT change Hungarian matching, Kalman update, lifecycle, or detection output.
- Before each tracker.update(), computes local cue margins between active tracks
  and current detections.
- If a matched track-detection pair has high cue ambiguity, temporarily raises
  the feature EMA alpha so the long-term identity memory updates conservatively.

This tests the hypothesis: ambiguity should modulate memory update strength,
not fully suppress identity adaptation.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import dmm_base_tracker_shadow_pda as spda
import dmm_base_tracker_shadow_pda_v3 as v3

# Global soft-update maps consumed by the monkeypatched DMMTrack.update.
_TRUST_SOFT_REASON_MAP: Dict[Tuple[int, int], str] = {}
_TRUST_SOFT_ALPHA_MAP: Dict[Tuple[int, int], float] = {}
_TRUST_PAIR_META: Dict[Tuple[int, int], dict] = {}
_TRUST_MATCH_ROWS = []
_TRUST_STATS = {
    'frames': 0,
    'candidate_pairs': 0,
    'soft_pairs_predicted': 0,
    'feature_updates_soft': 0,
    'feature_updates_normal': 0,
    'soft_alpha_sum': 0.0,
}
_ORIG_DMMTRACK_UPDATE = spda.base.DMMTrack.update
_PATCHED = False


def _iou(a, b) -> float:
    xx1=max(float(a[0]),float(b[0])); yy1=max(float(a[1]),float(b[1]))
    xx2=min(float(a[2]),float(b[2])); yy2=min(float(a[3]),float(b[3]))
    inter=max(0.0,xx2-xx1)*max(0.0,yy2-yy1)
    aa=max(1e-9,(float(a[2])-float(a[0]))*(float(a[3])-float(a[1])))
    bb=max(1e-9,(float(b[2])-float(b[0]))*(float(b[3])-float(b[1])))
    return inter/max(1e-9,aa+bb-inter)


def _center(b):
    return ((float(b[0])+float(b[2]))*0.5, (float(b[1])+float(b[3]))*0.5)


def _wh(b):
    return (max(1e-6,float(b[2])-float(b[0])), max(1e-6,float(b[3])-float(b[1])))


def _unit(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,dtype=np.float32)
    return x/max(float(np.linalg.norm(x)),1e-12)


def _patched_update(self, new_track, frame_id):
    gid = int(getattr(new_track, 'det_global_idx', -1))
    tid = int(getattr(self, 'track_id', -1))
    key = (tid, gid)
    soft_alpha = _TRUST_SOFT_ALPHA_MAP.get(key)
    soft_reason = _TRUST_SOFT_REASON_MAP.get(key, '')
    meta = dict(_TRUST_PAIR_META.get(key, {}))
    meta.update({
        'frame': int(frame_id),
        'track_id': tid,
        'det_global_idx': gid,
        'det_score': float(getattr(new_track, 'score', 0.0)),
        'tracklet_len_before': int(getattr(self, 'tracklet_len', 0)),
        'soft_applied': int(soft_alpha is not None),
        'soft_alpha': float(soft_alpha) if soft_alpha is not None else float(getattr(self, 'alpha', 0.9)),
        'soft_reason': soft_reason,
    })
    _TRUST_MATCH_ROWS.append(meta)
    if soft_alpha is not None and getattr(new_track, 'curr_feat', None) is not None:
        old_alpha = float(getattr(self, 'alpha', 0.9))
        self.alpha = float(soft_alpha)
        try:
            _ORIG_DMMTRACK_UPDATE(self, new_track, frame_id)
        finally:
            self.alpha = old_alpha
        self.trust_soft_last_reason = soft_reason
        self.trust_soft_last_alpha = float(soft_alpha)
        _TRUST_STATS['feature_updates_soft'] += 1
        _TRUST_STATS['soft_alpha_sum'] += float(soft_alpha)
    else:
        _ORIG_DMMTRACK_UPDATE(self, new_track, frame_id)
        _TRUST_STATS['feature_updates_normal'] += 1


def _ensure_patch():
    global _PATCHED
    if not _PATCHED:
        spda.base.DMMTrack.update = _patched_update
        _PATCHED = True


def _cue_matrices(tracks, boxes: np.ndarray, feats: np.ndarray):
    nT=len(tracks); nD=len(boxes)
    mats={c:np.zeros((nT,nD),dtype=np.float32) for c in ('app','motion','iou','shape')}
    if nT==0 or nD==0:
        return mats
    feat_unit = feats.astype(np.float32, copy=False)
    norms=np.linalg.norm(feat_unit,axis=1,keepdims=True)
    feat_unit=feat_unit/np.maximum(norms,1e-12)
    for ti,t in enumerate(tracks):
        tbox=np.asarray(t.tlbr,dtype=np.float32)
        tw,th=_wh(tbox); tcx,tcy=_center(tbox); scale=max(1.0,math.sqrt(tw*th)); asp0=tw/th
        sf=getattr(t,'smooth_feat',None)
        if sf is not None and feat_unit.size:
            mats['app'][ti,:]=feat_unit @ _unit(sf)
        else:
            mats['app'][ti,:]=0.0
        for di,dbox in enumerate(boxes):
            dw,dh=_wh(dbox); dcx,dcy=_center(dbox); asp=dw/dh
            mats['motion'][ti,di]=math.exp(-math.hypot(dcx-tcx,dcy-tcy)/scale)
            mats['iou'][ti,di]=_iou(tbox,dbox)
            mats['shape'][ti,di]=math.exp(-(abs(math.log(dh/th))+abs(math.log(max(asp,1e-6)/max(asp0,1e-6)))))
    return mats


def _margin(arr: np.ndarray, axis: str, i: int, j: int) -> float:
    chosen=float(arr[i,j])
    if axis=='row':
        vals=np.delete(arr[i,:], j)
    else:
        vals=np.delete(arr[:,j], i)
    if vals.size==0:
        return 1.0
    return chosen-float(np.max(vals))


def compute_soft_map(tracker, boxes: np.ndarray, scores: np.ndarray, feats: np.ndarray, det_ids: np.ndarray, cfg):
    tracks=[t for t in getattr(tracker,'tracked_stracks',[]) if bool(getattr(t,'is_activated',True))]
    nT=len(tracks); nD=len(boxes)
    global _TRUST_PAIR_META
    reason_map={}; alpha_map={}
    _TRUST_PAIR_META = {}
    if not cfg.enable or nT==0 or nD==0:
        return reason_map, alpha_map
    keep=np.where(np.asarray(scores,dtype=np.float32) >= float(cfg.min_det_score))[0]
    if keep.size == 0:
        return reason_map, alpha_map
    boxes_f=boxes[keep]
    feats_f=feats[keep]
    det_ids_f=det_ids[keep]
    mats=_cue_matrices(tracks,boxes_f,feats_f)
    nDf=len(boxes_f)
    for ti,t in enumerate(tracks):
        tid=int(getattr(t,'track_id',-1))
        track_age=int(getattr(t,'tracklet_len',0))
        if tid < 0:
            continue
        for di in range(nDf):
            pair={}; row_m={}; col_m={}
            for c,mat in mats.items():
                rm=_margin(mat,'row',ti,di)
                cm=_margin(mat,'col',ti,di)
                row_m[c]=rm; col_m[c]=cm; pair[c]=min(rm,cm)
            trust=max(pair.values())
            collapse=1.0-trust
            key=(tid,int(det_ids_f[di]))
            _TRUST_PAIR_META[key]={
                'trust': float(trust),
                'collapse': float(collapse),
                'track_age': track_age,
                'app_row_margin': float(row_m.get('app', 1.0)),
                'app_col_margin': float(col_m.get('app', 1.0)),
                'motion_row_margin': float(row_m.get('motion', 1.0)),
                'motion_col_margin': float(col_m.get('motion', 1.0)),
                'iou_row_margin': float(row_m.get('iou', 1.0)),
                'iou_col_margin': float(col_m.get('iou', 1.0)),
                'shape_row_margin': float(row_m.get('shape', 1.0)),
                'shape_col_margin': float(col_m.get('shape', 1.0)),
                'app_pair_margin': float(pair.get('app', 1.0)),
                'motion_pair_margin': float(pair.get('motion', 1.0)),
                'iou_pair_margin': float(pair.get('iou', 1.0)),
                'shape_pair_margin': float(pair.get('shape', 1.0)),
            }
            if track_age < int(cfg.min_track_age):
                continue
            reasons=[]
            risk=0.0
            if collapse >= float(cfg.soft_start):
                reasons.append('collapse')
                denom=max(1e-6,float(cfg.soft_extreme)-float(cfg.soft_start))
                risk=max(risk,min(1.0,max(0.0,(collapse-float(cfg.soft_start))/denom)))
            if col_m.get('app',1.0) <= float(cfg.app_col_thresh):
                reasons.append('app_col')
                denom=max(1e-6,float(cfg.app_col_thresh)-float(cfg.app_col_extreme))
                risk=max(risk,min(1.0,max(0.0,(float(cfg.app_col_thresh)-col_m['app'])/denom)))
            if col_m.get('motion',1.0) <= float(cfg.motion_col_thresh):
                reasons.append('motion_col')
                denom=max(1e-6,float(cfg.motion_col_thresh)-float(cfg.motion_col_extreme))
                risk=max(risk,min(1.0,max(0.0,(float(cfg.motion_col_thresh)-col_m['motion'])/denom)))
            if reasons:
                alpha=float(cfg.soft_alpha)+(float(cfg.extreme_alpha)-float(cfg.soft_alpha))*risk
                reason_map[key]='|'.join(reasons)
                alpha_map[key]=float(min(max(alpha,0.0),0.9999))
    _TRUST_STATS['candidate_pairs'] += int(nT*nDf)
    _TRUST_STATS['soft_pairs_predicted'] += len(alpha_map)
    return reason_map, alpha_map


class TrustConfig:
    def __init__(self, enable=False, soft_start=0.80, soft_extreme=1.00,
                 soft_alpha=0.98, extreme_alpha=0.995,
                 app_col_thresh=-999.0, app_col_extreme=-1.0,
                 motion_col_thresh=-999.0, motion_col_extreme=-1.0,
                 min_det_score=0.6, min_track_age=5):
        self.enable=bool(enable)
        self.soft_start=float(soft_start)
        self.soft_extreme=float(soft_extreme)
        self.soft_alpha=float(soft_alpha)
        self.extreme_alpha=float(extreme_alpha)
        self.app_col_thresh=float(app_col_thresh)
        self.app_col_extreme=float(app_col_extreme)
        self.motion_col_thresh=float(motion_col_thresh)
        self.motion_col_extreme=float(motion_col_extreme)
        self.min_det_score=float(min_det_score)
        self.min_track_age=int(min_track_age)


def parse_args():
    trust_parser=argparse.ArgumentParser(add_help=False)
    trust_parser.add_argument('--trust-soft-enable', action='store_true')
    trust_parser.add_argument('--trust-soft-start', type=float, default=0.80)
    trust_parser.add_argument('--trust-soft-extreme', type=float, default=1.00)
    trust_parser.add_argument('--trust-soft-alpha', type=float, default=0.98)
    trust_parser.add_argument('--trust-extreme-alpha', type=float, default=0.995)
    trust_parser.add_argument('--trust-app-col-thresh', type=float, default=-999.0)
    trust_parser.add_argument('--trust-app-col-extreme', type=float, default=-1.0)
    trust_parser.add_argument('--trust-motion-col-thresh', type=float, default=-999.0)
    trust_parser.add_argument('--trust-motion-col-extreme', type=float, default=-1.0)
    trust_parser.add_argument('--trust-min-det-score', type=float, default=0.60)
    trust_parser.add_argument('--trust-min-track-age', type=int, default=5)
    trust_parser.add_argument('--trust-pair-log-csv', type=str, default='')
    trust_args, remaining = trust_parser.parse_known_args()
    old=sys.argv
    try:
        sys.argv=[sys.argv[0]]+remaining
        args, v2cfg = v3.parse_args()
    finally:
        sys.argv=old
    trust=TrustConfig(
        enable=trust_args.trust_soft_enable,
        soft_start=trust_args.trust_soft_start,
        soft_extreme=trust_args.trust_soft_extreme,
        soft_alpha=trust_args.trust_soft_alpha,
        extreme_alpha=trust_args.trust_extreme_alpha,
        app_col_thresh=trust_args.trust_app_col_thresh,
        app_col_extreme=trust_args.trust_app_col_extreme,
        motion_col_thresh=trust_args.trust_motion_col_thresh,
        motion_col_extreme=trust_args.trust_motion_col_extreme,
        min_det_score=trust_args.trust_min_det_score,
        min_track_age=trust_args.trust_min_track_age,
    )
    trust.pair_log_csv = str(trust_args.trust_pair_log_csv or '')
    return args, v2cfg, trust


def main():
    args, v2cfg, trust = parse_args()
    _ensure_patch()
    dump=spda.base.load_dump(Path(args.dump_npz))
    det=np.asarray(dump['detections'],dtype=np.float32)
    # Keep the full feature dump in its original dtype (usually float16).
    # Converting the whole sequence to float32 upfront is extremely slow and
    # memory-heavy; cast only the current-frame slice below.
    feat=np.asarray(dump['features'])
    offsets=np.asarray(dump['frame_offsets'],dtype=np.int64)
    columns=[str(x) for x in dump['columns'].tolist()]
    col={name:i for i,name in enumerate(columns)}
    n_frames=len(offsets)-1
    if args.limit_frames and args.limit_frames>0:
        n_frames=min(n_frames,int(args.limit_frames))
    tracker=v3.make_tracker(args,v2cfg)
    mot_rows=[]; frames_with_outputs=0
    global _TRUST_SOFT_REASON_MAP, _TRUST_SOFT_ALPHA_MAP
    for frame in range(1,n_frames+1):
        start=int(offsets[frame-1]); end=int(offsets[frame])
        rows=det[start:end]; feats=np.asarray(feat[start:end], dtype=np.float32)
        if rows.size:
            boxes=rows[:,[col['x1'],col['y1'],col['x2'],col['y2']]].astype(np.float32)
            frame_scores=rows[:,col['score']].astype(np.float32)
            det_ids=rows[:,col['global_det_idx']].astype(np.int64)
        else:
            boxes=np.zeros((0,4),dtype=np.float32); frame_scores=np.zeros((0,),dtype=np.float32); det_ids=np.zeros((0,),dtype=np.int64)
            feats=np.zeros((0,feat.shape[1] if feat.ndim==2 else 0),dtype=np.float32)
        _TRUST_STATS['frames'] += 1
        _TRUST_SOFT_REASON_MAP, _TRUST_SOFT_ALPHA_MAP = compute_soft_map(tracker, boxes, frame_scores, feats, det_ids, trust)
        online_targets=tracker.update(boxes, frame_scores, feats, det_ids)
        _TRUST_SOFT_REASON_MAP = {}
        _TRUST_SOFT_ALPHA_MAP = {}
        _TRUST_PAIR_META.clear()
        frame_count=0
        for track in online_targets:
            x,y,w,h=[float(v) for v in track.tlwh]
            if w*h < float(args.min_box_area) or w <= 0 or h <= 0:
                continue
            if w/max(h,1e-12) > float(args.aspect_ratio_thresh):
                continue
            mot_rows.append((frame,int(track.track_id),x,y,w,h,float(track.score)))
            frame_count+=1
        if frame_count>0:
            frames_with_outputs+=1
    spda.base.write_mot_results(Path(args.out), mot_rows)
    summary={
        'seq':args.seq,
        'frames':int(n_frames),
        'rows':len(mot_rows),
        'frames_with_outputs':frames_with_outputs,
        'unique_tracks':len({r[1] for r in mot_rows}),
        'config':vars(args),
        'v3':asdict(v2cfg),
        'trust_soft':trust.__dict__,
        'trust_stats':dict(_TRUST_STATS),
        'trust_soft_mean_alpha': float(_TRUST_STATS['soft_alpha_sum']) / max(1, int(_TRUST_STATS['feature_updates_soft'])),
        'osr':asdict(tracker.osr),
        'shadow':asdict(tracker.shadow),
        'osr_stats':tracker.osr_stats,
        'shadow_stats':tracker.shadow_stats,
        'active_shadow_states':len(tracker.shadows),
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True,exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    if getattr(trust, 'pair_log_csv', ''):
        log_path=Path(trust.pair_log_csv)
        log_path.parent.mkdir(parents=True,exist_ok=True)
        fields=sorted({k for row in _TRUST_MATCH_ROWS for k in row.keys()})
        with log_path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(_TRUST_MATCH_ROWS)
    if args.debug_csv:
        spda.base.write_debug(Path(args.debug_csv), tracker.debug_rows)
    if args.shadow_events_csv:
        out_csv=Path(args.shadow_events_csv); out_csv.parent.mkdir(parents=True,exist_ok=True)
        fieldnames=sorted({k for row in tracker.shadow_event_rows for k in row.keys()})
        with out_csv.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fieldnames); writer.writeheader(); writer.writerows(tracker.shadow_event_rows)
    print(json.dumps(summary,indent=2,sort_keys=True),flush=True)

if __name__=='__main__':
    main()
