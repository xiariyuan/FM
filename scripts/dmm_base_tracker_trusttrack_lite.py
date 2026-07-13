#!/usr/bin/env python3
"""TrustTrack-lite: cue-collapse-aware ReID memory gate.

Minimal intervention version:
- Does NOT change Hungarian matching, Kalman update, lifecycle, or detection output.
- Before each tracker.update(), computes local cue margins between active tracks
  and current detections.
- If a matched track-detection pair has high cue collapse / low column margin,
  skips only the ReID feature update for that match.

This tests the hypothesis: high-ambiguity matches may be acceptable for short-term
box continuity, but should not contaminate long-term identity memory.
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

# Global freeze map consumed by the monkeypatched DMMTrack.update.
_TRUST_FREEZE_MAP: Dict[Tuple[int, int], str] = {}
_TRUST_PAIR_META: Dict[Tuple[int, int], dict] = {}
_TRUST_MATCH_ROWS = []
_TRUST_STATS = {
    'frames': 0,
    'candidate_pairs': 0,
    'freeze_pairs_predicted': 0,
    'feature_updates_frozen': 0,
    'feature_updates_normal': 0,
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
    meta = dict(_TRUST_PAIR_META.get(key, {}))
    meta.update({
        'frame': int(frame_id),
        'track_id': tid,
        'det_global_idx': gid,
        'det_score': float(getattr(new_track, 'score', 0.0)),
        'frozen': int(key in _TRUST_FREEZE_MAP),
        'freeze_reason': _TRUST_FREEZE_MAP.get(key, ''),
    })
    _TRUST_MATCH_ROWS.append(meta)
    if key in _TRUST_FREEZE_MAP and getattr(new_track, 'curr_feat', None) is not None:
        saved_feat = np.asarray(new_track.curr_feat, dtype=np.float32)
        saved_feat = saved_feat / max(float(np.linalg.norm(saved_feat)), 1e-12)
        # Preserve all geometric/Kalman/lifecycle updates, but suppress the
        # original feature update so smooth_feat/history remain clean.
        new_track.curr_feat = None
        try:
            _ORIG_DMMTRACK_UPDATE(self, new_track, frame_id)
        finally:
            new_track.curr_feat = saved_feat
        # Crucial: current evidence is still refreshed. Only the long-term
        # identity prototype (smooth_feat + history deque) is frozen.
        self.curr_feat = saved_feat.copy()
        self.trust_freeze_last_reason = _TRUST_FREEZE_MAP.get(key, '')
        _TRUST_STATS['feature_updates_frozen'] += 1
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


def compute_freeze_map(tracker, boxes: np.ndarray, scores: np.ndarray, feats: np.ndarray, det_ids: np.ndarray, cfg) -> Dict[Tuple[int,int], str]:
    tracks=[t for t in getattr(tracker,'tracked_stracks',[]) if bool(getattr(t,'is_activated',True))]
    nT=len(tracks); nD=len(boxes)
    global _TRUST_PAIR_META
    fmap={}
    _TRUST_PAIR_META = {}
    if not cfg.enable or nT==0 or nD==0:
        return fmap
    # Crucial speed guard: TrustTrack-lite memory gate is meant for matched
    # primary/mid detections, not every ultra-low candidate. Filter before the
    # expensive appearance matrix.
    keep=np.where(np.asarray(scores,dtype=np.float32) >= float(cfg.min_det_score))[0]
    if keep.size == 0:
        return fmap
    boxes_f=boxes[keep]
    feats_f=feats[keep]
    det_ids_f=det_ids[keep]
    mats=_cue_matrices(tracks,boxes_f,feats_f)
    nDf=len(boxes_f)
    for ti,t in enumerate(tracks):
        tid=int(getattr(t,'track_id',-1))
        if tid < 0:
            continue
        for di in range(nDf):
            pair={}
            row_m={}; col_m={}
            for c,mat in mats.items():
                rm=_margin(mat,'row',ti,di)
                cm=_margin(mat,'col',ti,di)
                row_m[c]=rm; col_m[c]=cm; pair[c]=min(rm,cm)
            trust=max(pair.values())
            collapse=1.0-trust
            reasons=[]
            if collapse >= float(cfg.collapse_thresh):
                reasons.append('collapse')
            if col_m.get('app',1.0) <= float(cfg.app_col_thresh):
                reasons.append('app_col')
            if col_m.get('motion',1.0) <= float(cfg.motion_col_thresh):
                reasons.append('motion_col')
            key=(tid,int(det_ids_f[di]))
            _TRUST_PAIR_META[key]={
                'trust': float(trust),
                'collapse': float(collapse),
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
            if reasons:
                fmap[key]='|'.join(reasons)
    _TRUST_STATS['candidate_pairs'] += int(nT*nDf)
    _TRUST_STATS['freeze_pairs_predicted'] += len(fmap)
    return fmap


class TrustConfig:
    def __init__(self, enable=False, collapse_thresh=0.80, app_col_thresh=0.08, motion_col_thresh=0.08, min_det_score=0.1):
        self.enable=bool(enable)
        self.collapse_thresh=float(collapse_thresh)
        self.app_col_thresh=float(app_col_thresh)
        self.motion_col_thresh=float(motion_col_thresh)
        self.min_det_score=float(min_det_score)


def parse_args():
    trust_parser=argparse.ArgumentParser(add_help=False)
    trust_parser.add_argument('--trust-lite-enable', action='store_true')
    trust_parser.add_argument('--trust-collapse-thresh', type=float, default=0.80)
    trust_parser.add_argument('--trust-app-col-thresh', type=float, default=0.08)
    trust_parser.add_argument('--trust-motion-col-thresh', type=float, default=0.08)
    trust_parser.add_argument('--trust-min-det-score', type=float, default=0.10)
    trust_parser.add_argument('--trust-pair-log-csv', type=str, default='')
    trust_args, remaining = trust_parser.parse_known_args()
    old=sys.argv
    try:
        sys.argv=[sys.argv[0]]+remaining
        args, v2cfg = v3.parse_args()
    finally:
        sys.argv=old
    trust=TrustConfig(
        enable=trust_args.trust_lite_enable,
        collapse_thresh=trust_args.trust_collapse_thresh,
        app_col_thresh=trust_args.trust_app_col_thresh,
        motion_col_thresh=trust_args.trust_motion_col_thresh,
        min_det_score=trust_args.trust_min_det_score,
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
    global _TRUST_FREEZE_MAP
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
        _TRUST_FREEZE_MAP = compute_freeze_map(tracker, boxes, frame_scores, feats, det_ids, trust)
        online_targets=tracker.update(boxes, frame_scores, feats, det_ids)
        _TRUST_FREEZE_MAP = {}
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
        'trust_lite':trust.__dict__,
        'trust_stats':dict(_TRUST_STATS),
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
