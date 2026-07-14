from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_segment_change_point_features import best_iou, l2_vec, load_small_arrays, npy_member_memmap
from build_segment_pair_state_bank import read_tracks


def cosine(a, b):
    if a is None or b is None:
        return np.nan
    return float(np.dot(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True)
    ap.add_argument('--events', required=True)
    ap.add_argument('--top-k', type=int, default=100)
    ap.add_argument('--track-file', required=True)
    ap.add_argument('--dump-npz', required=True)
    ap.add_argument('--overlap-events', required=True)
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--min-iou', type=float, default=.5)
    ap.add_argument('--pre-window', type=int, default=10)
    args = ap.parse_args()

    events = pd.read_csv(args.events).head(args.top_k).copy().reset_index(drop=True)
    events.insert(0, 'canonical_rank', np.arange(1, len(events) + 1, dtype=int))
    tracks = read_tracks(Path(args.track_file))
    npz = Path(args.dump_npz)
    det = npy_member_memmap(npz, 'detections.npy')
    feat = npy_member_memmap(npz, 'features.npy')
    offsets, col = load_small_arrays(npz)

    ov = pd.read_csv(args.overlap_events)
    ov = ov[ov.seq == args.seq].copy()
    pair_ov = defaultdict(list)
    for r in ov.itertuples(index=False):
        u, v = sorted((int(r.track_i), int(r.track_j)))
        pair_ov[(u, v)].append((int(r.frame), float(r.ioa_min_area)))
    for key in pair_ov:
        pair_ov[key].sort()

    feature_cache = {}
    quality_cache = {}

    def row_feature(tid: int, frame: int):
        key = (tid, frame)
        if key in feature_cache:
            return feature_cache[key]
        row = tracks.get(tid, {}).get(frame)
        if row is None or frame < 1 or frame >= len(offsets):
            feature_cache[key] = None; quality_cache[key] = 0.0; return None
        lo, hi = int(offsets[frame - 1]), int(offsets[frame])
        if hi <= lo:
            feature_cache[key] = None; quality_cache[key] = 0.0; return None
        boxes = np.asarray(det[lo:hi][:, [col['x1'], col['y1'], col['x2'], col['y2']]], dtype=np.float32)
        j, iou = best_iou(row['box'], boxes)
        if j < 0 or iou < args.min_iou:
            feature_cache[key] = None; quality_cache[key] = float(iou); return None
        x = np.asarray(feat[lo + j], dtype=np.float32)
        if float(np.linalg.norm(x)) < 1e-8:
            feature_cache[key] = None; quality_cache[key] = float(iou); return None
        x = l2_vec(x)
        feature_cache[key] = x; quality_cache[key] = float(iou)
        return x

    proto_cache = {}
    def proto(tid: int, start: int, end: int):
        key = (tid, start, end)
        if key in proto_cache:
            return proto_cache[key]
        frames = [fr for fr in tracks.get(tid, {}) if start <= fr <= end]
        frames.sort()
        vecs = [row_feature(tid, fr) for fr in frames]
        valid = [(fr, x) for fr, x in zip(frames, vecs) if x is not None]
        if not valid:
            out = (None, 0, len(frames), 0.0)
        else:
            p = l2_vec(np.mean(np.stack([x for _, x in valid]), axis=0))
            q = float(np.mean([quality_cache[(tid, fr)] for fr, _ in valid]))
            out = (p, len(valid), len(frames), q)
        proto_cache[key] = out
        return out

    horizons = [30, 60, 120, 300]
    chunks = [(0,29),(30,59),(60,119),(120,299)]
    rows = []
    for ix, e in events.iterrows():
        u, v = int(e.u), int(e.v); t = int(e.boundary_frame)
        uf = sorted(tracks.get(u, {})); vf = sorted(tracks.get(v, {}))
        u_start, u_end = (min(uf), max(uf)) if uf else (np.nan, np.nan)
        v_start, v_end = (min(vf), max(vf)) if vf else (np.nan, np.nan)
        u_pre, u_pre_n, u_pre_total, u_pre_q = proto(u, t-args.pre_window, t-1)
        v_pre, v_pre_n, v_pre_total, v_pre_q = proto(v, t-args.pre_window, t-1)
        row = e.to_dict()
        row.update({
            'u_track_start': u_start, 'u_track_end': u_end,
            'v_track_start': v_start, 'v_track_end': v_end,
            'u_age_at_event': t-u_start if uf else np.nan,
            'v_age_at_event': t-v_start if vf else np.nan,
            'u_future_life': u_end-t+1 if uf else 0,
            'v_future_life': v_end-t+1 if vf else 0,
            'pair_min_future_life': min(u_end-t+1, v_end-t+1) if uf and vf else 0,
            'pair_max_future_life': max(u_end-t+1, v_end-t+1) if uf and vf else 0,
            'pair_end_gap_abs': abs(u_end-v_end) if uf and vf else np.nan,
            'u_pre_rows': u_pre_n, 'v_pre_rows': v_pre_n,
            'u_pre_match_iou': u_pre_q, 'v_pre_match_iou': v_pre_q,
        })

        ovs = [(fr, val) for fr, val in pair_ov.get((u,v), []) if fr >= t]
        row['future_overlap_total_count'] = len(ovs)
        row['future_overlap_last_offset'] = max([fr-t for fr,_ in ovs], default=-1)
        row['future_overlap_max_ioa'] = max([x for _,x in ovs], default=0.0)
        row['future_overlap_mean_ioa'] = float(np.mean([x for _,x in ovs])) if ovs else 0.0

        swap_margins = []
        for h in horizons:
            end = t+h-1
            up, un, ut, uq = proto(u, t, end)
            vp, vn, vt, vq = proto(v, t, end)
            keep = cosine(u_pre, up) + cosine(v_pre, vp)
            swap = cosine(u_pre, vp) + cosine(v_pre, up)
            margin = swap-keep if np.isfinite(keep) and np.isfinite(swap) else np.nan
            swap_margins.append(margin)
            u_present = sum(t <= fr <= end for fr in uf)
            v_present = sum(t <= fr <= end for fr in vf)
            copresent = len(set(fr for fr in uf if t <= fr <= end) & set(fr for fr in vf if t <= fr <= end))
            hov = [(fr,x) for fr,x in ovs if fr <= end]
            row.update({
                f'u_present_h{h}': u_present,
                f'v_present_h{h}': v_present,
                f'copresent_h{h}': copresent,
                f'copresent_frac_h{h}': copresent/max(1,h),
                f'u_reid_rows_h{h}': un, f'v_reid_rows_h{h}': vn,
                f'u_reid_coverage_h{h}': un/max(1,ut), f'v_reid_coverage_h{h}': vn/max(1,vt),
                f'u_match_iou_h{h}': uq, f'v_match_iou_h{h}': vq,
                f'keep_score_h{h}': keep, f'swap_score_h{h}': swap,
                f'swap_margin_h{h}': margin,
                f'overlap_count_h{h}': len(hov),
                f'overlap_max_h{h}': max([x for _,x in hov], default=0.0),
                f'overlap_mean_h{h}': float(np.mean([x for _,x in hov])) if hov else 0.0,
            })
        for j,(lo,hi) in enumerate(chunks):
            up, un, ut, uq = proto(u, t+lo, t+hi)
            vp, vn, vt, vq = proto(v, t+lo, t+hi)
            keep = cosine(u_pre, up) + cosine(v_pre, vp)
            swap = cosine(u_pre, vp) + cosine(v_pre, up)
            row[f'chunk{j}_keep_score'] = keep
            row[f'chunk{j}_swap_score'] = swap
            row[f'chunk{j}_swap_margin'] = swap-keep if np.isfinite(keep) and np.isfinite(swap) else np.nan
            row[f'chunk{j}_valid_both'] = int(up is not None and vp is not None)
        valid_m = [x for x in swap_margins if np.isfinite(x)]
        row['swap_margin_future_mean'] = float(np.mean(valid_m)) if valid_m else np.nan
        row['swap_margin_future_min'] = float(np.min(valid_m)) if valid_m else np.nan
        row['swap_margin_future_max'] = float(np.max(valid_m)) if valid_m else np.nan
        row['swap_margin_positive_horizons'] = int(sum(x > 0 for x in valid_m))
        row['swap_margin_non_decreasing'] = int(all(a <= b for a,b in zip(valid_m,valid_m[1:]))) if len(valid_m)>=2 else 0
        rows.append(row)
        if len(rows)%20==0:
            print(json.dumps({'done':len(rows),'feature_cache':len(feature_cache)}),flush=True)

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    summary = {
        'seq':args.seq,'events':len(rows),'top_k':args.top_k,
        'feature_cache_rows':len(feature_cache),
        'observable_policy':'All features use tracker outputs, detector/ReID dumps, and future video frames only; no GT columns are created.',
    }
    out.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
