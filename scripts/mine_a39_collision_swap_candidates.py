#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
        w=csv.DictWriter(f, fieldnames=fields or ['anchor_id'], extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def read_track_frames(path: Path):
    by_tid={}
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            p=line.strip().split(',')
            if len(p)<2:
                continue
            fr=ai(p[0],-1); tid=ai(p[1],-1)
            if fr<0 or tid<0:
                continue
            by_tid.setdefault(tid, []).append(fr)
    for tid in by_tid:
        by_tid[tid].sort()
    return by_tid


def count_range(frames, lo, hi):
    return sum(1 for x in frames if lo <= x <= hi)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--collision-anchor-csv', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_05a_collision_structure_audit_true_reconnects/all_collision_blocked/collision_structure_by_anchor.csv')
    ap.add_argument('--track-file', default='outputs/spot_runtime_gate_20260628/A37_identity_escrow_association/A37_00b_candidate_logger_seq02/track_results/MOT20-02.txt')
    ap.add_argument('--out-dir', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_05d_collision_swap_candidate_mining')
    ap.add_argument('--proto-window', type=int, default=50)
    ap.add_argument('--min-ratio', type=float, default=0.75)
    ap.add_argument('--min-purity', type=float, default=0.75)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows=read_csv(Path(args.collision_anchor_csv))
    by_tid=read_track_frames(Path(args.track_file))
    done={'106_150_169','106_169_150','215_469_508','215_508_469'}
    manifest=[]
    for r in rows:
        anchor_id=r['anchor_id']
        fs=ai(r.get('frame_start')); fe=ai(r.get('frame_end'))
        track_a=ai(r.get('target_id'))
        track_b=ai(r.get('candidate_track_id'))
        gt_a=ai(r.get('anchor_gt'), -1)
        gt_b=ai(r.get('target_collision_major_gt'), -1)
        candidate_gt=ai(r.get('candidate_collision_major_gt'), -1)
        cand_anchor_ratio=af(r.get('candidate_anchor_target_other_ratio'))
        candidate_purity=af(r.get('candidate_collision_major_purity'))
        target_purity=af(r.get('target_collision_major_purity'))
        include=int(
            gt_a>=0 and gt_b>=0 and candidate_gt==gt_a and
            cand_anchor_ratio >= args.min_ratio and
            candidate_purity >= args.min_purity and
            target_purity >= args.min_purity and
            r.get('collision_class') in {'CANDIDATE_ANCHOR_TARGET_OTHER_DOMINANT','MIXED_COLLISION'}
        )
        a_frames=by_tid.get(track_a, [])
        b_frames=by_tid.get(track_b, [])
        rec={
            'anchor_id':anchor_id,
            'case_arg':f'{anchor_id}:{r.get("tunnel_id")}:{track_a}:{track_b}:{fs}:{fe}:{gt_a}:{gt_b}',
            'tunnel_id':r.get('tunnel_id'),
            'track_a_target_id':track_a,
            'track_b_candidate_id':track_b,
            'gt_a_anchor':gt_a,
            'gt_b_target_major':gt_b,
            'frame_start':fs,
            'frame_end':fe,
            'duration':max(0, fe-fs+1),
            'collision_class':r.get('collision_class'),
            'candidate_anchor_target_other_ratio':cand_anchor_ratio,
            'candidate_collision_major_gt':candidate_gt,
            'candidate_collision_major_purity':candidate_purity,
            'target_collision_major_gt':gt_b,
            'target_collision_major_purity':target_purity,
            'collision_pairs':r.get('collision_pairs'),
            'same_gt_duplicate_rows':r.get('same_gt_duplicate_rows'),
            'candidate_anchor_target_other_rows':r.get('candidate_anchor_target_other_rows'),
            'different_gt_conflict_rows':r.get('different_gt_conflict_rows'),
            'unknown_collision_rows':r.get('unknown_collision_rows'),
            'track_a_pre_rows':count_range(a_frames, fs-args.proto_window, fs-1),
            'track_b_pre_rows':count_range(b_frames, fs-args.proto_window, fs-1),
            'track_a_inside_rows':count_range(a_frames, fs, fe),
            'track_b_inside_rows':count_range(b_frames, fs, fe),
            'track_a_post_rows':count_range(a_frames, fe+1, fe+args.proto_window),
            'track_b_post_rows':count_range(b_frames, fe+1, fe+args.proto_window),
            'proto_a_available':int(count_range(a_frames, fs-args.proto_window, fs-1) >= 5),
            'proto_b_available':int(count_range(b_frames, fs-args.proto_window, fs-1) >= 5),
            'already_covered_by_reciprocal_success':int(anchor_id in done),
            'include_candidate':include,
        }
        if rec['proto_a_available'] and rec['proto_b_available']:
            rec['prototype_status']='PRE_PRE_AVAILABLE'
        elif rec['proto_a_available'] and not rec['proto_b_available']:
            rec['prototype_status']='B_PRE_MISSING'
        elif rec['proto_b_available'] and not rec['proto_a_available']:
            rec['prototype_status']='A_PRE_MISSING'
        else:
            rec['prototype_status']='PRE_BOTH_MISSING'
        manifest.append(rec)
    manifest=sorted(manifest, key=lambda x:(-x['include_candidate'], x['already_covered_by_reciprocal_success'], -af(x['candidate_anchor_target_other_ratio']), -ai(x['collision_pairs'])))
    write_csv(out/'swap_candidate_manifest.csv', manifest)
    runnable=[r for r in manifest if r['include_candidate'] and not r['already_covered_by_reciprocal_success'] and r['proto_a_available'] and r['proto_b_available']]
    write_csv(out/'swap_candidate_manifest_runnable.csv', runnable)
    payload={'candidate_count':len(manifest),'included':sum(int(r['include_candidate']) for r in manifest),'runnable_new':len(runnable),'case_args':[r['case_arg'] for r in runnable]}
    (out/'candidate_summary.json').write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(payload, indent=2, sort_keys=True))

if __name__=='__main__':
    main()
