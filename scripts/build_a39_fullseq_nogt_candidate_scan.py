#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def ai(v, d=0):
    try:
        return int(float(v))
    except Exception:
        return d


def af(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def read_csv(path: Path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    fields=[]; seen=set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields or ['candidate_id'], extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def read_track(path: Path):
    by_tid=defaultdict(list); by_frame_tid=defaultdict(list)
    total=0
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            p=line.rstrip('\n').split(',')
            fr=ai(p[0], -1); tid=ai(p[1], -1)
            if fr < 0 or tid < 0:
                continue
            by_tid[tid].append(fr)
            by_frame_tid[(fr, tid)].append(total)
            total += 1
    for tid in by_tid:
        by_tid[tid]=sorted(by_tid[tid])
    return by_tid, by_frame_tid


def count_range(frames, lo, hi):
    if lo > hi:
        return 0
    return sum(1 for f in frames if lo <= f <= hi)


def overlap_frames(by_tid, a, b, lo, hi):
    sa=set(f for f in by_tid.get(a, []) if lo <= f <= hi)
    sb=set(f for f in by_tid.get(b, []) if lo <= f <= hi)
    return sorted(sa & sb)


def segments(frames):
    if not frames:
        return []
    out=[]; s=e=frames[0]
    for fr in frames[1:]:
        if fr == e+1:
            e=fr
        else:
            out.append((s,e)); s=e=fr
    out.append((s,e)); return out


def seg_str(segs):
    return '|'.join(f'{a}-{b}' if a!=b else str(a) for a,b in segs)


def max_seg_len(segs):
    return max((b-a+1 for a,b in segs), default=0)


def allowed_pair_fields(r):
    allowed=['tunnel_id','pre_track','post_track','sim','row_rank','col_rank','row_margin','col_margin','row_candidate_count','col_candidate_count','hungarian','pre_track_pre_rows','pre_track_core_rows','pre_track_post_rows','post_track_pre_rows','post_track_core_rows','post_track_post_rows','pre_bottom_rank','post_bottom_rank','bottom_rank_delta','pre_height_rank','post_height_rank','height_rank_delta','center_delta_norm','bottom_delta','height_ratio','post_rows_forecast','post_collision_rows','post_collision_ratio']
    return {k:r.get(k,'') for k in allowed}


def build_path_index(pair_rows, args):
    out=[]; summary=Counter()
    for raw in pair_rows:
        r=allowed_pair_fields(raw)
        tid=f"{r['tunnel_id']}_{r['pre_track']}_{r['post_track']}"
        pre=ai(r['pre_track']); post=ai(r['post_track'])
        if pre == post:
            summary['self_pair_skip'] += 1
            continue
        if ai(r.get('pre_track_pre_rows')) < args.path_min_pre_rows or ai(r.get('post_track_post_rows')) < args.path_min_post_rows:
            summary['prepost_skip'] += 1
            continue
        if af(r.get('post_collision_ratio'), 1.0) > args.path_max_post_collision_ratio:
            summary['post_collision_skip'] += 1
            continue
        # Bridge-like is high-confidence pair-level reconnect.
        bridge_like = (
            af(r.get('sim')) >= args.bridge_min_sim and
            ai(r.get('row_rank'), 999) <= args.bridge_max_rank and
            ai(r.get('col_rank'), 999) <= args.bridge_max_rank and
            af(r.get('row_margin')) >= args.bridge_min_margin and
            af(r.get('col_margin')) >= args.bridge_min_margin
        )
        # Direct-like should be broad; later path-builder/direct scorer does the strict work.
        direct_like = (
            (ai(r.get('row_rank'), 999) <= args.direct_max_rank or ai(r.get('col_rank'), 999) <= args.direct_max_rank) and
            af(r.get('center_delta_norm'), 999.0) <= args.direct_max_center_delta_norm and
            af(r.get('height_ratio'), 0.0) >= args.direct_min_height_ratio and
            af(r.get('height_ratio'), 999.0) <= args.direct_max_height_ratio
        )
        if not bridge_like and not direct_like:
            summary['rank_geom_skip'] += 1
            continue
        mode='bridge_index' if bridge_like else 'direct_index'
        if bridge_like and direct_like:
            mode='bridge_or_direct_index'
        rec={'candidate_id':tid,'transaction_id':tid,'candidate_family':'path_index','index_mode':mode,**r}
        out.append(rec)
        summary[mode] += 1
    return out, dict(summary)


def parse_tracks(s):
    return [ai(x) for x in str(s).split('|') if str(x).strip()!='']


def build_swap_index(tunnel_rows, by_tid, args):
    out=[]; summary=Counter(); sanity_tunnels=set()
    for tr in tunnel_rows:
        tunnel_id=ai(tr.get('tunnel_id'), -1)
        start=ai(tr.get('start')); end=ai(tr.get('end')); duration=ai(tr.get('duration'))
        tracks=parse_tracks(tr.get('tracks',''))
        summary['total_tunnels'] += 1
        if duration < args.swap_min_tunnel_duration:
            summary['duration_skip_tunnels'] += 1
            continue
        if len(tracks) < 2:
            summary['single_track_skip_tunnels'] += 1
            continue
        summary['eligible_tunnels'] += 1
        lo=start-args.swap_tunnel_pad; hi=end+args.swap_tunnel_pad
        for a in tracks:
            for b in tracks:
                if a == b:
                    continue
                summary['ordered_pairs_total'] += 1
                ov=overlap_frames(by_tid, a, b, lo, hi)
                if len(ov) < args.swap_min_overlap_frames:
                    summary['overlap_skip'] += 1
                    continue
                segs=segments(ov)
                if max_seg_len(segs) < args.swap_min_overlap_frames:
                    summary['max_contiguous_overlap_skip'] += 1
                    continue
                fs=min(ov); fe=max(ov)
                a_pre=count_range(by_tid.get(a, []), fs-args.swap_window, fs-1)
                b_pre=count_range(by_tid.get(b, []), fs-args.swap_window, fs-1)
                a_post=count_range(by_tid.get(a, []), fe+1, fe+args.swap_window)
                b_post=count_range(by_tid.get(b, []), fe+1, fe+args.swap_window)
                if min(a_pre,b_pre) < args.swap_min_pre_rows:
                    summary['pre_rows_skip'] += 1
                    continue
                if min(a_post,b_post) < args.swap_min_post_rows:
                    summary['post_rows_skip'] += 1
                    continue
                cid=f'{tunnel_id}_{a}_{b}'
                rec={
                    'candidate_id':cid,
                    'transaction_id':cid,
                    'candidate_family':'swap_overlap_index',
                    'tunnel_id':tunnel_id,
                    'track_a':a,
                    'track_b':b,
                    'tunnel_start':start,
                    'tunnel_end':end,
                    'tunnel_duration':duration,
                    'scan_start':lo,
                    'scan_end':hi,
                    'overlap_start':fs,
                    'overlap_end':fe,
                    'overlap_frames':len(ov),
                    'overlap_segments':seg_str(segs),
                    'overlap_segment_count':len(segs),
                    'max_contiguous_overlap':max_seg_len(segs),
                    'track_a_pre_rows':a_pre,
                    'track_b_pre_rows':b_pre,
                    'track_a_post_rows':a_post,
                    'track_b_post_rows':b_post,
                    'track_count_in_tunnel':len(tracks),
                    'max_ioa':tr.get('max_ioa',''),
                    'mean_ioa':tr.get('mean_ioa',''),
                }
                out.append(rec)
                summary['swap_overlap_candidates'] += 1
                if cid in {'106_150_169','106_169_150','215_469_508','215_508_469','188_520_521','91_190_200','123_199_218','90_184_169','214_508_469'}:
                    sanity_tunnels.add(cid)
    return out, dict(summary), sorted(sanity_tunnels)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--track-file', default='outputs/spot_runtime_gate_20260628/A37_identity_escrow_association/A37_00b_candidate_logger_seq02/track_results/MOT20-02.txt')
    ap.add_argument('--tunnels-csv', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_00_tunnel_discovery_seq02_ioa050_dur3/tunnel_candidates.csv')
    ap.add_argument('--pair-matrix', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_03a_v2_assignment_dryrun_seq02/full_pair_matrix.csv')
    ap.add_argument('--out-dir', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_06d_full_seq_no_gt_candidate_scan_seq02')
    ap.add_argument('--path-min-pre-rows', type=int, default=5)
    ap.add_argument('--path-min-post-rows', type=int, default=5)
    ap.add_argument('--path-max-post-collision-ratio', type=float, default=0.25)
    ap.add_argument('--bridge-min-sim', type=float, default=0.55)
    ap.add_argument('--bridge-max-rank', type=int, default=3)
    ap.add_argument('--bridge-min-margin', type=float, default=0.03)
    ap.add_argument('--direct-max-rank', type=int, default=5)
    ap.add_argument('--direct-max-center-delta-norm', type=float, default=0.80)
    ap.add_argument('--direct-min-height-ratio', type=float, default=0.50)
    ap.add_argument('--direct-max-height-ratio', type=float, default=1.80)
    ap.add_argument('--swap-min-tunnel-duration', type=int, default=3)
    ap.add_argument('--swap-tunnel-pad', type=int, default=10)
    ap.add_argument('--swap-min-overlap-frames', type=int, default=20)
    ap.add_argument('--swap-window', type=int, default=50)
    ap.add_argument('--swap-min-pre-rows', type=int, default=5)
    ap.add_argument('--swap-min-post-rows', type=int, default=5)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    by_tid,_=read_track(Path(args.track_file))
    pair_rows=read_csv(Path(args.pair_matrix))
    tunnel_rows=read_csv(Path(args.tunnels_csv))
    path_index,path_summary=build_path_index(pair_rows,args)
    swap_index,swap_summary,sanity=build_swap_index(tunnel_rows,by_tid,args)
    write_csv(out/'bridge_direct_candidate_index.csv', path_index)
    write_csv(out/'swap_overlap_candidate_index.csv', swap_index)
    all_index=path_index+swap_index
    write_csv(out/'transaction_candidate_index_fullseq_nogt.csv', all_index)
    known=['12_9_71','202_501_542','106_150_169','215_469_508','188_520_521','91_190_200','123_199_218','90_184_169','214_508_469','202_501_545','197_384_549']
    indexed=set(r['transaction_id'] for r in all_index)
    known_status=[{'transaction_id':k,'indexed':int(k in indexed)} for k in known]
    write_csv(out/'known_transaction_index_status.csv', known_status)
    summary={'path_summary':path_summary,'swap_summary':swap_summary,'total_index_candidates':len(all_index),'path_index_candidates':len(path_index),'swap_index_candidates':len(swap_index),'known_index_status':known_status,'swap_sanity_hits':sanity}
    (out/'candidate_index_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    leakage='''# A39_06d1 full-seq no-GT candidate index leakage audit\n\n## Status\n\nThis is a lightweight full-sequence candidate index scan. It does not run heavy path-builder or ReID feature extraction. It only indexes candidate transactions using no-GT fields.\n\n## Pair matrix fields allowed\n\nThe script whitelists only:\n\n- tunnel_id, pre_track, post_track\n- sim, row_rank, col_rank, row_margin, col_margin\n- row_candidate_count, col_candidate_count, hungarian\n- pre/post/core row counts\n- bottom/height/center geometry deltas\n- post_collision_rows, post_collision_ratio\n\n## Pair matrix fields explicitly not used\n\n- pre_gt, post_gt, gt_known, gt_same\n- self_continuation, true_reconnect, false_reconnect\n- oracle_core_exit_rows, oracle_collision_rows, oracle_collision_ratio\n- eval_accept_* fields\n\n## Track/tunnel fields allowed\n\n- tunnel_id, start, end, duration, tracks, max_ioa, mean_ioa\n- track frame presence, track overlap, pre/post row counts\n\n## GT / evaluation leakage\n\nNo GT, row_gt, diagnostic wrong rows, TrackEval metrics, or oracle fields are used to decide candidate inclusion.\n\nKnown transaction index status is reported only as a final sanity check; it does not affect indexing.\n'''
    (out/'leakage_audit.md').write_text(leakage, encoding='utf-8')
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
