from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_segment_change_point_features import (
    best_iou,
    l2_vec,
    load_small_arrays,
    npy_member_memmap,
)


def read_tracks(path: Path):
    by_tid = defaultdict(dict)
    with path.open() as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 6:
                continue
            fr = int(float(p[0])); tid = int(float(p[1]))
            x, y, w, h = map(float, p[2:6]); score = float(p[6]) if len(p) > 6 else 1.0
            if w <= 1 or h <= 1:
                continue
            by_tid[tid][fr] = {
                'frame': fr, 'track_id': tid,
                'box': np.asarray([x, y, x + w, y + h], dtype=np.float32),
                'cx': x + w / 2, 'cy': y + h / 2,
                'bottom': y + h, 'h': h, 'w': w, 'area': w * h, 'score': score,
            }
    return dict(by_tid)


def nms_proposals(df: pd.DataFrame, score_col: str, radius: int, limit: int):
    selected = []
    for tid, g in df.sort_values(score_col, ascending=False).groupby('track_id', sort=False):
        kept = []
        for row in g.itertuples(index=False):
            fr = int(row.boundary_frame)
            if any(abs(fr - old) <= radius for old in kept):
                continue
            kept.append(fr); selected.append(row._asdict())
    selected.sort(key=lambda r: (-float(r[score_col]), int(r['track_id']), int(r['boundary_frame'])))
    return selected[:limit]


def load_overlap_partners(path: Path, seq: str):
    df = pd.read_csv(path)
    df = df[df.seq == seq]
    by_frame = defaultdict(list)
    for r in df.itertuples(index=False):
        by_frame[int(r.frame)].append((int(r.track_i), int(r.track_j), float(r.ioa_min_area)))
    return by_frame


def top_partners(by_frame, tid: int, frame: int, k: int, frame_pad: int):
    best = {}
    for delta in range(-frame_pad, frame_pad + 1):
        for a, b, ioa in by_frame.get(frame + delta, []):
            if a == tid:
                other = b
            elif b == tid:
                other = a
            else:
                continue
            key = int(other); value = (float(ioa), -abs(delta), -int(other))
            if key not in best or value > best[key][0]:
                best[key] = (value, delta)
    ranked = sorted(best.items(), key=lambda x: (-x[1][0][0], -x[1][0][1], x[0]))
    return [(int(t), float(v[0][0]), int(v[1])) for t, v in ranked[:k]]


def majority_label(label_map, tid: int, frames):
    labels = [label_map.get((fr, tid)) for fr in frames]
    labels = [x for x in labels if x is not None]
    if not labels:
        return None, 0, 0.0
    c = Counter(labels); label, count = c.most_common(1)[0]
    return int(label), len(labels), count / len(labels)


def cosine(a, b):
    if a is None or b is None:
        return np.nan
    return float(np.dot(a, b))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or ['seq'], extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True)
    ap.add_argument('--proposal-scores', required=True)
    ap.add_argument('--proposal-score-col', default='oof_hgb')
    ap.add_argument('--proposal-limit', type=int, default=5000)
    ap.add_argument('--proposal-nms-radius', type=int, default=10)
    ap.add_argument('--top-k-partners', type=int, default=3)
    ap.add_argument('--partner-frame-pad', type=int, default=1)
    ap.add_argument('--window', type=int, default=5)
    ap.add_argument('--min-iou', type=float, default=.5)
    ap.add_argument('--track-file', required=True)
    ap.add_argument('--dump-npz', required=True)
    ap.add_argument('--overlap-events', required=True)
    ap.add_argument('--matches', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    seq = args.seq; out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    proposal_df = pd.read_csv(args.proposal_scores)
    proposal_df = proposal_df[proposal_df.seq == seq].copy() if 'seq' in proposal_df else proposal_df
    proposals = nms_proposals(proposal_df, args.proposal_score_col,
                              args.proposal_nms_radius, args.proposal_limit)
    tracks = read_tracks(Path(args.track_file))
    overlap = load_overlap_partners(Path(args.overlap_events), seq)

    matches = pd.read_csv(args.matches, usecols=['seq', 'frame', 'track_id', 'gt_id'])
    matches = matches[matches.seq == seq].drop_duplicates(['frame', 'track_id'], keep='last')
    labels = {(int(r.frame), int(r.track_id)): int(r.gt_id) for r in matches.itertuples(index=False)}

    npz = Path(args.dump_npz)
    det = npy_member_memmap(npz, 'detections.npy')
    feat = npy_member_memmap(npz, 'features.npy')
    offsets, col = load_small_arrays(npz)
    feature_cache = {}
    proto_cache = {}
    match_quality_cache = {}

    def row_feature(tid: int, frame: int):
        key = (tid, frame)
        if key in feature_cache:
            return feature_cache[key]
        row = tracks.get(tid, {}).get(frame)
        if row is None or frame < 1 or frame >= len(offsets):
            feature_cache[key] = None; match_quality_cache[key] = 0.0; return None
        lo, hi = int(offsets[frame - 1]), int(offsets[frame])
        if hi <= lo:
            feature_cache[key] = None; match_quality_cache[key] = 0.0; return None
        boxes = np.asarray(det[lo:hi][:, [col['x1'], col['y1'], col['x2'], col['y2']]], dtype=np.float32)
        j, value = best_iou(row['box'], boxes)
        if j < 0 or value < args.min_iou:
            feature_cache[key] = None; match_quality_cache[key] = value; return None
        vector = np.asarray(feat[lo + j], dtype=np.float32)
        if float(np.linalg.norm(vector)) < 1e-8:
            feature_cache[key] = None; match_quality_cache[key] = value; return None
        vector = l2_vec(vector)
        feature_cache[key] = vector; match_quality_cache[key] = value
        return vector

    def proto(tid: int, frame: int, side: str):
        key = (tid, frame, side, args.window)
        if key in proto_cache:
            return proto_cache[key]
        track = tracks.get(tid, {})
        if side == 'left':
            frames = sorted([fr for fr in track if frame - args.window <= fr < frame])
        else:
            frames = sorted([fr for fr in track if frame <= fr < frame + args.window])
        vectors = [row_feature(tid, fr) for fr in frames]
        vectors = [x for x in vectors if x is not None]
        value = l2_vec(np.mean(np.stack(vectors), axis=0)) if vectors else None
        quality = float(np.mean([match_quality_cache[(tid, fr)] for fr in frames
                                 if (tid, fr) in match_quality_cache and feature_cache.get((tid, fr)) is not None])) if vectors else 0.0
        proto_cache[key] = (value, len(vectors), quality)
        return proto_cache[key]

    rows = []; proposals_with_partner = 0
    for pi, proposal in enumerate(proposals, 1):
        a = int(proposal['track_id']); frame = int(proposal['boundary_frame'])
        partners = top_partners(overlap, a, frame, args.top_k_partners, args.partner_frame_pad)
        if partners:
            proposals_with_partner += 1
        a_left, a_left_n, a_left_q = proto(a, frame, 'left')
        a_right, a_right_n, a_right_q = proto(a, frame, 'right')
        a_old, a_old_n, a_old_purity = majority_label(labels, a, range(frame - args.window, frame))
        a_new, a_new_n, a_new_purity = majority_label(labels, a, range(frame, frame + args.window))
        for partner_rank, (b, partner_ioa, partner_delta) in enumerate(partners, 1):
            b_left, b_left_n, b_left_q = proto(b, frame, 'left')
            b_right, b_right_n, b_right_q = proto(b, frame, 'right')
            b_old, b_old_n, b_old_purity = majority_label(labels, b, range(frame - args.window, frame))
            b_new, b_new_n, b_new_purity = majority_label(labels, b, range(frame, frame + args.window))

            a_changed = a_old is not None and a_new is not None and a_old != a_new
            b_changed = b_old is not None and b_new is not None and b_old != b_new
            reciprocal = a_changed and b_changed and b_old == a_new and b_new == a_old
            old_carrier = a_changed and a_old in {b_old, b_new}
            new_source = a_changed and a_new in {b_old, b_new}
            if not a_changed:
                pair_class = 'keep'
            elif reciprocal:
                pair_class = 'reciprocal_swap'
            elif old_carrier and new_source:
                pair_class = 'related_both_nonreciprocal'
            elif old_carrier:
                pair_class = 'old_identity_carrier'
            elif new_source:
                pair_class = 'new_identity_source'
            else:
                pair_class = 'unrelated_switch'

            self_keep = cosine(a_left, a_right) + cosine(b_left, b_right)
            reciprocal_score = cosine(a_left, b_right) + cosine(b_left, a_right)
            old_handoff_score = cosine(a_left, b_right)
            new_source_score = cosine(b_left, a_right)
            same_pre = cosine(a_left, b_left); same_post = cosine(a_right, b_right)
            valid_pair_reid = int(all(x is not None for x in [a_left, a_right, b_left, b_right]))
            if not valid_pair_reid:
                self_keep = reciprocal_score = old_handoff_score = new_source_score = np.nan

            ar = tracks.get(a, {}).get(frame); br = tracks.get(b, {}).get(frame)
            if ar is None:
                ar = tracks.get(a, {}).get(frame - 1)
            if br is None:
                br = tracks.get(b, {}).get(frame - 1)
            center_dist_norm = bottom_dist_norm = height_log_ratio_abs = area_log_ratio_abs = np.nan
            if ar is not None and br is not None:
                scale = max(ar['h'], br['h'], 1.0)
                center_dist_norm = float(np.hypot(ar['cx'] - br['cx'], ar['cy'] - br['cy']) / scale)
                bottom_dist_norm = float(abs(ar['bottom'] - br['bottom']) / scale)
                height_log_ratio_abs = float(abs(np.log(max(ar['h'], 1e-6) / max(br['h'], 1e-6))))
                area_log_ratio_abs = float(abs(np.log(max(ar['area'], 1e-6) / max(br['area'], 1e-6))))

            base = {k: v for k, v in proposal.items()
                    if not k.startswith('label_switch_') and not k.startswith('distance_to_switch_')}
            rows.append({
                **base,
                'proposal_rank': pi, 'track_a': a, 'track_b': b,
                'partner_rank': partner_rank, 'partner_ioa': partner_ioa,
                'partner_frame_delta': partner_delta,
                'a_left_feat_rows': a_left_n, 'a_right_feat_rows': a_right_n,
                'b_left_feat_rows': b_left_n, 'b_right_feat_rows': b_right_n,
                'a_left_match_iou': a_left_q, 'a_right_match_iou': a_right_q,
                'b_left_match_iou': b_left_q, 'b_right_match_iou': b_right_q,
                'valid_pair_reid': valid_pair_reid,
                'sim_a_left_a_right': cosine(a_left, a_right),
                'sim_b_left_b_right': cosine(b_left, b_right),
                'sim_a_left_b_left': same_pre,
                'sim_a_right_b_right': same_post,
                'sim_a_left_b_right': cosine(a_left, b_right),
                'sim_b_left_a_right': cosine(b_left, a_right),
                'pair_keep_score': self_keep,
                'pair_reciprocal_score': reciprocal_score,
                'pair_swap_margin': reciprocal_score - self_keep if valid_pair_reid else np.nan,
                'old_handoff_score': old_handoff_score,
                'new_source_score': new_source_score,
                'center_dist_norm': center_dist_norm,
                'bottom_dist_norm': bottom_dist_norm,
                'height_log_ratio_abs_pair': height_log_ratio_abs,
                'area_log_ratio_abs_pair': area_log_ratio_abs,
                # Diagnostic GT columns below are never candidate features.
                'a_gt_old': a_old, 'a_gt_new': a_new,
                'b_gt_old': b_old, 'b_gt_new': b_new,
                'a_gt_old_rows': a_old_n, 'a_gt_new_rows': a_new_n,
                'b_gt_old_rows': b_old_n, 'b_gt_new_rows': b_new_n,
                'a_gt_old_purity': a_old_purity, 'a_gt_new_purity': a_new_purity,
                'b_gt_old_purity': b_old_purity, 'b_gt_new_purity': b_new_purity,
                'label_a_changed': int(a_changed), 'label_b_changed': int(b_changed),
                'label_reciprocal_swap': int(reciprocal),
                'label_old_carrier': int(old_carrier),
                'label_new_source': int(new_source),
                'label_pair_related': int(old_carrier or new_source),
                'label_pair_class': pair_class,
            })
        if pi % 500 == 0:
            print(json.dumps({'proposals_done': pi, 'pair_rows': len(rows),
                              'feature_cache': len(feature_cache)}), flush=True)

    write_csv(out / f'{seq}_segment_pair_state_bank.csv', rows)
    bank = pd.DataFrame(rows)
    summary = {
        'seq': seq, 'proposal_score_col': args.proposal_score_col,
        'proposal_limit': args.proposal_limit, 'proposals_selected': len(proposals),
        'proposals_with_partner': proposals_with_partner,
        'proposal_partner_coverage': proposals_with_partner / max(1, len(proposals)),
        'top_k_partners': args.top_k_partners, 'pair_rows': len(rows),
        'valid_pair_reid_rows': int(bank.valid_pair_reid.sum()) if len(bank) else 0,
        'pair_reid_coverage': float(bank.valid_pair_reid.mean()) if len(bank) else 0.0,
        'class_counts': {str(k): int(v) for k, v in bank.label_pair_class.value_counts().items()} if len(bank) else {},
        'reciprocal_pairs': int(bank.label_reciprocal_swap.sum()) if len(bank) else 0,
        'related_pairs': int(bank.label_pair_related.sum()) if len(bank) else 0,
        'unique_reciprocal_events': int(bank[bank.label_reciprocal_swap == 1][['track_a','boundary_frame']].drop_duplicates().shape[0]) if len(bank) else 0,
        'unique_related_events': int(bank[bank.label_pair_related == 1][['track_a','boundary_frame']].drop_duplicates().shape[0]) if len(bank) else 0,
        'leakage_policy': 'Candidate proposals use diagnostic OOF unary scores and observable overlap only. GT columns are emitted strictly for M02 pilot labels and must be excluded from models.',
    }
    (out / f'{seq}_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
