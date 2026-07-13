#!/usr/bin/env python3
"""TrustTrack-soft: cue-collapse-aware adaptive ReID memory update.

Minimal intervention version:
- Does NOT change Hungarian matching, Kalman update, lifecycle, or detection output.
- Computes local cue margins between active tracks and current detections.
- For ambiguous matched pairs, temporarily raises the ReID EMA alpha so the
  identity memory updates conservatively instead of being fully frozen.
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
_TRUST_SOFT_ALPHA_MAP: Dict[Tuple[int, int], float] = {}
_TRUST_SOFT_REASON_MAP: Dict[Tuple[int, int], str] = {}
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

    boxes=np.asarray(boxes,dtype=np.float32)
    feats=np.asarray(feats,dtype=np.float32)
    feat_norms=np.linalg.norm(feats,axis=1,keepdims=True)
    feat_unit=feats/np.maximum(feat_norms,1e-12)

    track_boxes=np.stack([np.asarray(t.tlbr,dtype=np.float32) for t in tracks],axis=0)
    track_feats=[]
    for t in tracks:
        sf=getattr(t,'smooth_feat',None)
        if sf is None:
            track_feats.append(np.zeros((feats.shape[1],),dtype=np.float32))
        else:
            track_feats.append(_unit(sf))
    track_feats=np.stack(track_feats,axis=0)
    mats['app']=(track_feats @ feat_unit.T).astype(np.float32,copy=False)

    tx1,ty1,tx2,ty2=[track_boxes[:,k] for k in range(4)]
    dx1,dy1,dx2,dy2=[boxes[:,k] for k in range(4)]
    tw=np.maximum(1e-6,tx2-tx1); th=np.maximum(1e-6,ty2-ty1)
    dw=np.maximum(1e-6,dx2-dx1); dh=np.maximum(1e-6,dy2-dy1)
    tcx=(tx1+tx2)*0.5; tcy=(ty1+ty2)*0.5
    dcx=(dx1+dx2)*0.5; dcy=(dy1+dy2)*0.5
    scale=np.maximum(1.0,np.sqrt(tw*th))
    dist=np.sqrt((tcx[:,None]-dcx[None,:])**2+(tcy[:,None]-dcy[None,:])**2)/scale[:,None]
    mats['motion']=np.exp(-dist).astype(np.float32)

    xx1=np.maximum(tx1[:,None],dx1[None,:]); yy1=np.maximum(ty1[:,None],dy1[None,:])
    xx2=np.minimum(tx2[:,None],dx2[None,:]); yy2=np.minimum(ty2[:,None],dy2[None,:])
    inter=np.maximum(0.0,xx2-xx1)*np.maximum(0.0,yy2-yy1)
    union=(tw*th)[:,None]+(dw*dh)[None,:]-inter
    mats['iou']=(inter/np.maximum(union,1e-12)).astype(np.float32)

    tasp=tw/th; dasp=dw/dh
    size_term=np.abs(np.log(dh[None,:]/th[:,None]))
    aspect_term=np.abs(np.log(np.maximum(dasp[None,:],1e-6)/np.maximum(tasp[:,None],1e-6)))
    mats['shape']=np.exp(-(size_term+aspect_term)).astype(np.float32)
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


def _exclusive_margin_matrix(mat: np.ndarray, axis: int) -> np.ndarray:
    """Chosen score minus best competitor for every matrix entry."""
    mat=np.asarray(mat,dtype=np.float32)
    n0,n1=mat.shape
    if axis==1:
        if n1<=1:
            return np.ones_like(mat,dtype=np.float32)
        top_idx=np.argmax(mat,axis=1)
        top_val=mat[np.arange(n0),top_idx]
        second=np.partition(mat,n1-2,axis=1)[:,n1-2]
        competitor=np.broadcast_to(top_val[:,None],mat.shape).copy()
        competitor[np.arange(n0),top_idx]=second
    elif axis==0:
        if n0<=1:
            return np.ones_like(mat,dtype=np.float32)
        top_idx=np.argmax(mat,axis=0)
        top_val=mat[top_idx,np.arange(n1)]
        second=np.partition(mat,n0-2,axis=0)[n0-2,:]
        competitor=np.broadcast_to(top_val[None,:],mat.shape).copy()
        competitor[top_idx,np.arange(n1)]=second
    else:
        raise ValueError(axis)
    return mat-competitor

def compute_soft_map(tracker, boxes: np.ndarray, scores: np.ndarray, feats: np.ndarray, det_ids: np.ndarray, cfg):
    tracks=[t for t in getattr(tracker,'tracked_stracks',[]) if bool(getattr(t,'is_activated',True))]
    nT=len(tracks); nD=len(boxes)
    global _TRUST_PAIR_META
    alpha_map={}; reason_map={}
    _TRUST_PAIR_META = {}
    if not cfg.enable or nT==0 or nD==0:
        return alpha_map, reason_map
    keep=np.where(np.asarray(scores,dtype=np.float32) >= float(cfg.min_det_score))[0]
    if keep.size == 0:
        return alpha_map, reason_map
    boxes_f=np.asarray(boxes[keep],dtype=np.float32)
    feats_f=np.asarray(feats[keep],dtype=np.float32)
    det_ids_f=np.asarray(det_ids[keep],dtype=np.int64)
    mats=_cue_matrices(tracks,boxes_f,feats_f)
    nDf=len(boxes_f)

    row_m={c:_exclusive_margin_matrix(mat,axis=1) for c,mat in mats.items()}
    col_m={c:_exclusive_margin_matrix(mat,axis=0) for c,mat in mats.items()}
    pair_m={c:np.minimum(row_m[c],col_m[c]) for c in mats}
    trust=np.maximum.reduce([pair_m[c] for c in ('app','motion','iou','shape')])
    collapse=1.0-trust

    ages=np.asarray([int(getattr(t,'tracklet_len',0)) for t in tracks],dtype=np.int32)
    eligible=ages[:,None] >= int(cfg.min_track_age)
    risk=np.zeros_like(collapse,dtype=np.float32)
    reason_bits=np.zeros_like(collapse,dtype=np.uint8)

    collapse_mask=collapse >= float(cfg.soft_start)
    if np.any(collapse_mask):
        denom=max(1e-6,float(cfg.soft_extreme)-float(cfg.soft_start))
        collapse_risk=np.clip((collapse-float(cfg.soft_start))/denom,0.0,1.0)
        risk=np.maximum(risk,collapse_risk)
        reason_bits[collapse_mask] |= 1

    if float(cfg.app_col_thresh) > -998.0:
        app_mask=col_m['app'] <= float(cfg.app_col_thresh)
        denom=max(1e-6,float(cfg.app_col_thresh)-float(cfg.app_col_extreme))
        app_risk=np.clip((float(cfg.app_col_thresh)-col_m['app'])/denom,0.0,1.0)
        risk=np.maximum(risk,app_risk)
        reason_bits[app_mask] |= 2

    if float(cfg.motion_col_thresh) > -998.0:
        motion_mask=col_m['motion'] <= float(cfg.motion_col_thresh)
        denom=max(1e-6,float(cfg.motion_col_thresh)-float(cfg.motion_col_extreme))
        motion_risk=np.clip((float(cfg.motion_col_thresh)-col_m['motion'])/denom,0.0,1.0)
        risk=np.maximum(risk,motion_risk)
        reason_bits[motion_mask] |= 4

    trigger=eligible & (reason_bits != 0)
    trigger_indices=np.argwhere(trigger)
    for ti,di in trigger_indices.tolist():
        tid=int(getattr(tracks[ti],'track_id',-1))
        if tid<0:
            continue
        key=(tid,int(det_ids_f[di]))
        alpha=float(cfg.soft_alpha)+(float(cfg.extreme_alpha)-float(cfg.soft_alpha))*float(risk[ti,di])
        alpha_map[key]=float(min(max(alpha,0.0),0.9999))
        bits=int(reason_bits[ti,di])
        reasons=[]
        if bits&1: reasons.append('collapse')
        if bits&2: reasons.append('app_col')
        if bits&4: reasons.append('motion_col')
        reason_map[key]='|'.join(reasons)
        # Meta is only required for the small set of interventions. This keeps
        # full-sequence diagnostics cheap while preserving exact auditability.
        _TRUST_PAIR_META[key]={
            'trust':float(trust[ti,di]),'collapse':float(collapse[ti,di]),
            'track_age':int(ages[ti]),
            'app_row_margin':float(row_m['app'][ti,di]),'app_col_margin':float(col_m['app'][ti,di]),
            'motion_row_margin':float(row_m['motion'][ti,di]),'motion_col_margin':float(col_m['motion'][ti,di]),
            'iou_row_margin':float(row_m['iou'][ti,di]),'iou_col_margin':float(col_m['iou'][ti,di]),
            'shape_row_margin':float(row_m['shape'][ti,di]),'shape_col_margin':float(col_m['shape'][ti,di]),
            'app_pair_margin':float(pair_m['app'][ti,di]),'motion_pair_margin':float(pair_m['motion'][ti,di]),
            'iou_pair_margin':float(pair_m['iou'][ti,di]),'shape_pair_margin':float(pair_m['shape'][ti,di]),
        }
    _TRUST_STATS['candidate_pairs'] += int(nT*nDf)
    _TRUST_STATS['soft_pairs_predicted'] += int(len(alpha_map))
    return alpha_map, reason_map

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
    global _TRUST_SOFT_ALPHA_MAP, _TRUST_SOFT_REASON_MAP
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
        _TRUST_SOFT_ALPHA_MAP, _TRUST_SOFT_REASON_MAP = compute_soft_map(tracker, boxes, frame_scores, feats, det_ids, trust)
        online_targets=tracker.update(boxes, frame_scores, feats, det_ids)
        _TRUST_SOFT_ALPHA_MAP = {}
        _TRUST_SOFT_REASON_MAP = {}
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
        'trust_soft_mean_alpha': (float(_TRUST_STATS['soft_alpha_sum']) / max(1, int(_TRUST_STATS['feature_updates_soft']))),
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
