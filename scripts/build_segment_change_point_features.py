from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def npy_member_memmap(npz_path: Path, member: str) -> np.memmap:
    """Memory-map a ZIP_STORED .npy member inside an uncompressed .npz."""
    with zipfile.ZipFile(npz_path) as zf:
        info = zf.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError(f'{member} is compressed and cannot be directly memory-mapped')
        header_offset = int(info.header_offset)
    with npz_path.open('rb') as f:
        f.seek(header_offset)
        local = f.read(30)
        if len(local) != 30 or local[:4] != b'PK\x03\x04':
            raise RuntimeError(f'invalid local ZIP header for {member}')
        fields = struct.unpack('<IHHHHHIIIHH', local)
        name_len, extra_len = fields[-2], fields[-1]
        f.seek(header_offset + 30 + name_len + extra_len)
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        data_offset = f.tell()
    return np.memmap(npz_path, dtype=dtype, mode='r', offset=data_offset, shape=shape,
                     order='F' if fortran else 'C')


def load_small_arrays(npz_path: Path):
    with np.load(npz_path, allow_pickle=True) as z:
        offsets = np.asarray(z['frame_offsets'], dtype=np.int64)
        columns = [str(x) for x in z['columns'].tolist()]
    return offsets, {name: i for i, name in enumerate(columns)}


def l2_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def l2_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def best_iou(box: np.ndarray, boxes: np.ndarray):
    if boxes.size == 0:
        return -1, 0.0
    lt = np.maximum(boxes[:, :2], box[:2]); rb = np.minimum(boxes[:, 2:], box[2:])
    wh = np.maximum(0.0, rb - lt); inter = wh[:, 0] * wh[:, 1]
    area_a = max(1e-12, float((box[2] - box[0]) * (box[3] - box[1])))
    area_b = np.maximum(1e-12, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    iou = inter / np.maximum(area_a + area_b - inter, 1e-12)
    j = int(np.argmax(iou))
    return j, float(iou[j])


def pair_ioa(a: np.ndarray, b: np.ndarray) -> float:
    lt = np.maximum(a[:2], b[:2]); rb = np.minimum(a[2:], b[2:])
    wh = np.maximum(0.0, rb - lt); inter = float(wh[0] * wh[1])
    aa = max(1e-12, float((a[2] - a[0]) * (a[3] - a[1])))
    bb = max(1e-12, float((b[2] - b[0]) * (b[3] - b[1])))
    return inter / min(aa, bb)


def read_tracks(path: Path):
    tracks = defaultdict(list); by_frame = defaultdict(list)
    with path.open() as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 6:
                continue
            frame = int(float(p[0])); tid = int(float(p[1]))
            x, y, w, h = map(float, p[2:6]); score = float(p[6]) if len(p) > 6 else 1.0
            if w <= 1 or h <= 1:
                continue
            row = {
                'frame': frame, 'track_id': tid,
                'box': np.asarray([x, y, x + w, y + h], dtype=np.float32),
                'cx': x + w / 2, 'cy': y + h / 2,
                'h': h, 'area': w * h, 'score': score,
            }
            tracks[tid].append(row); by_frame[frame].append(row)
    for rows in tracks.values():
        rows.sort(key=lambda r: r['frame'])
    return tracks, by_frame


def persistent_switches(matches: pd.DataFrame, persistence: int, tolerance: int = 2):
    result = defaultdict(set)
    for tid, g in matches.groupby('track_id'):
        g = g.sort_values('frame').drop_duplicates('frame', keep='last')
        frames = g.frame.astype(int).to_numpy(); labels = g.gt_id.astype(int).to_numpy()
        for j in range(persistence, len(g) - persistence + 1):
            left = labels[j - persistence:j]; right = labels[j:j + persistence]
            if len(set(left.tolist())) != 1 or len(set(right.tolist())) != 1 or left[-1] == right[0]:
                continue
            if frames[j - 1] - frames[j - persistence] > persistence - 1 + tolerance:
                continue
            if frames[j + persistence - 1] - frames[j] > persistence - 1 + tolerance:
                continue
            if frames[j] - frames[j - 1] > 1 + tolerance:
                continue
            result[int(tid)].add(int(frames[j]))
    return result


def velocity(rows, lo, hi):
    if hi - lo < 2:
        return np.zeros(2, dtype=np.float32)
    a, b = rows[lo], rows[hi - 1]; dt = max(1, b['frame'] - a['frame'])
    return np.asarray([(b['cx'] - a['cx']) / dt, (b['cy'] - a['cy']) / dt], dtype=np.float32)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ['seq']
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True)
    ap.add_argument('--track-file', required=True)
    ap.add_argument('--dump-npz', required=True)
    ap.add_argument('--matches', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--window', type=int, default=5)
    ap.add_argument('--min-iou', type=float, default=.5)
    ap.add_argument('--max-frame-gap', type=int, default=10)
    ap.add_argument('--track-index-start', type=int, default=1,
                    help='1-based inclusive index in sorted tracker-ID order')
    ap.add_argument('--track-index-end', type=int, default=0,
                    help='1-based inclusive index; 0 means all remaining tracks')
    ap.add_argument('--output-stem', default='',
                    help='Output filename stem; defaults to sequence name')
    args = ap.parse_args()

    seq = args.seq; out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tracks, by_frame = read_tracks(Path(args.track_file))
    npz_path = Path(args.dump_npz)
    detections = npy_member_memmap(npz_path, 'detections.npy')
    features = npy_member_memmap(npz_path, 'features.npy')
    offsets, col = load_small_arrays(npz_path)
    print(json.dumps({'detections_shape': detections.shape, 'features_shape': features.shape,
                      'feature_dtype': str(features.dtype)}, indent=2), flush=True)

    matches = pd.read_csv(args.matches, usecols=['seq', 'frame', 'track_id', 'gt_id'])
    matches = matches[matches.seq == seq]
    switch_sets = {p: persistent_switches(matches, p) for p in [1, 3, 5, 10]}

    all_track_items = sorted(tracks.items())
    start_index = max(1, int(args.track_index_start))
    end_index = int(args.track_index_end) if int(args.track_index_end) > 0 else len(all_track_items)
    if start_index > end_index or start_index > len(all_track_items):
        raise ValueError(f'invalid track shard {start_index}:{end_index} for {len(all_track_items)} tracks')
    end_index = min(end_index, len(all_track_items))
    selected_items = all_track_items[start_index - 1:end_index]
    selected_tids = {int(tid) for tid, _ in selected_items}
    output_stem = args.output_stem or seq

    total_rows = matched_rows = low_iou_rows = zero_feature_rows = 0
    records = []
    for local_index, (tid, rows) in enumerate(selected_items, 1):
        track_index = start_index + local_index - 1
        n = len(rows); total_rows += n
        det_idx = np.full(n, -1, dtype=np.int64); match_iou = np.zeros(n, dtype=np.float32)
        for i, row in enumerate(rows):
            frame = row['frame']
            if frame < 1 or frame >= len(offsets):
                continue
            lo, hi = int(offsets[frame - 1]), int(offsets[frame])
            if hi <= lo:
                continue
            boxes = np.asarray(detections[lo:hi][:, [col['x1'], col['y1'], col['x2'], col['y2']]], dtype=np.float32)
            j, value = best_iou(row['box'], boxes)
            if j < 0 or value < args.min_iou:
                low_iou_rows += 1; continue
            idx = lo + j
            feature = np.asarray(features[idx], dtype=np.float32)
            if float(np.linalg.norm(feature)) < 1e-8:
                zero_feature_rows += 1; continue
            det_idx[i] = idx; match_iou[i] = value; matched_rows += 1

        valid_pos = np.flatnonzero(det_idx >= 0)
        if len(valid_pos) < 2 * args.window:
            continue
        feat = l2_rows(np.asarray(features[det_idx[valid_pos]], dtype=np.float32))
        rank = {int(pos): k for k, pos in enumerate(valid_pos)}
        prefix = np.vstack([np.zeros((1, feat.shape[1]), dtype=np.float32),
                            np.cumsum(feat, axis=0, dtype=np.float32)])

        for i in range(1, n):
            if rows[i]['frame'] - rows[i - 1]['frame'] > args.max_frame_gap:
                continue
            left_pos = valid_pos[valid_pos < i]
            right_pos = valid_pos[valid_pos >= i]
            if len(left_pos) < args.window or len(right_pos) < args.window:
                continue
            lp, rp = left_pos[-args.window:], right_pos[:args.window]
            l0, l1 = rank[int(lp[0])], rank[int(lp[-1])] + 1
            r0, r1 = rank[int(rp[0])], rank[int(rp[-1])] + 1
            left_proto = l2_vec(prefix[l1] - prefix[l0]); right_proto = l2_vec(prefix[r1] - prefix[r0])
            left_x, right_x = feat[l0:l1], feat[r0:r1]
            proto_cos = float(np.dot(left_proto, right_proto))
            left_self = float(np.mean(left_x @ left_proto)); right_self = float(np.mean(right_x @ right_proto))
            left_cross = float(np.mean(left_x @ right_proto)); right_cross = float(np.mean(right_x @ left_proto))
            appearance_margin = 0.5 * (left_self + right_self - left_cross - right_cross)
            adjacent_cos = float(np.dot(feat[l1 - 1], feat[r0]))

            pre_lo = max(0, i - args.window); post_hi = min(n, i + args.window)
            v_pre = velocity(rows, pre_lo, i); v_post = velocity(rows, i, post_hi)
            speed_pre, speed_post = float(np.linalg.norm(v_pre)), float(np.linalg.norm(v_post))
            velocity_cos = float(np.dot(v_pre, v_post) / max(speed_pre * speed_post, 1e-12)) if speed_pre > 0 and speed_post > 0 else 0.0
            dt = max(1, rows[i]['frame'] - rows[i - 1]['frame'])
            pred_x = rows[i - 1]['cx'] + v_pre[0] * dt; pred_y = rows[i - 1]['cy'] + v_pre[1] * dt
            pred_error = math.hypot(rows[i]['cx'] - pred_x, rows[i]['cy'] - pred_y) / max(rows[i]['h'], rows[i - 1]['h'], 1.0)

            max_ioa = 0.0; partner_hits = 0
            for row in [rows[i - 1], rows[i]]:
                for other in by_frame[row['frame']]:
                    if other['track_id'] == tid:
                        continue
                    value = pair_ioa(row['box'], other['box'])
                    if value > 0:
                        partner_hits += 1; max_ioa = max(max_ioa, value)

            frame = rows[i]['frame']
            rec = {
                'seq': seq, 'track_id': int(tid), 'boundary_frame': int(frame),
                'prev_frame': int(rows[i - 1]['frame']), 'frame_gap': int(frame - rows[i - 1]['frame']),
                'track_length': n, 'boundary_position_ratio': i / max(1, n - 1),
                'left_match_iou_mean': float(np.mean(match_iou[lp])),
                'right_match_iou_mean': float(np.mean(match_iou[rp])),
                'proto_cos': proto_cos, 'appearance_change': 1.0 - proto_cos,
                'adjacent_cos': adjacent_cos, 'adjacent_change': 1.0 - adjacent_cos,
                'left_self_cos': left_self, 'right_self_cos': right_self,
                'left_cross_cos': left_cross, 'right_cross_cos': right_cross,
                'appearance_margin': appearance_margin,
                'velocity_cos': velocity_cos, 'speed_pre': speed_pre, 'speed_post': speed_post,
                'speed_log_ratio_abs': abs(math.log(max(speed_post, 1e-6) / max(speed_pre, 1e-6))),
                'prediction_error_norm': pred_error,
                'height_log_ratio_abs': abs(math.log(max(rows[i]['h'], 1e-6) / max(rows[i - 1]['h'], 1e-6))),
                'area_log_ratio_abs': abs(math.log(max(rows[i]['area'], 1e-6) / max(rows[i - 1]['area'], 1e-6))),
                'score_abs_jump': abs(rows[i]['score'] - rows[i - 1]['score']),
                'overlap_max_ioa': max_ioa, 'overlap_partner_hits': partner_hits,
                'frame_density': len(by_frame[frame]),
            }
            for p in [1, 3, 5, 10]:
                switches = switch_sets[p].get(int(tid), set())
                rec[f'label_switch_p{p}'] = int(frame in switches)
                rec[f'distance_to_switch_p{p}'] = min([abs(frame - x) for x in switches] or [10**9])
            records.append(rec)
        if local_index % 50 == 0 or local_index == len(selected_items):
            print(json.dumps({'tracks_done_in_shard': local_index, 'tracks_in_shard': len(selected_items),
                              'global_track_index': track_index, 'tracks_total': len(all_track_items),
                              'records': len(records), 'matched_rows': matched_rows}), flush=True)

    write_csv(out / f'{output_stem}_segment_change_features.csv', records)
    df = pd.DataFrame(records)
    summary = {
        'seq': seq, 'track_rows': total_rows, 'matched_reid_rows': matched_rows,
        'reid_match_coverage': matched_rows / max(1, total_rows),
        'low_iou_rows': low_iou_rows, 'zero_feature_rows': zero_feature_rows,
        'feature_boundaries': len(records), 'window': args.window, 'min_iou': args.min_iou,
        'track_index_start': start_index, 'track_index_end': end_index,
        'tracks_in_shard': len(selected_items), 'tracks_total': len(all_track_items),
        'selected_track_ids': sorted(selected_tids),
        'persistent_switches': {str(p): int(sum(len(v) for tid, v in switch_sets[p].items()
                                                 if int(tid) in selected_tids))
                                for p in [1,3,5,10]},
        'positive_boundaries': {str(p): int(df[f'label_switch_p{p}'].sum()) if len(df) else 0 for p in [1,3,5,10]},
    }
    (out / f'{output_stem}_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
