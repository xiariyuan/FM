#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


def l2norm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n < 1e-12] = 1.0
    return x / n


def mean_norm(feats: List[np.ndarray], dim: int = 2048) -> np.ndarray:
    if not feats:
        return np.zeros((dim,), dtype=np.float32)
    arr = np.stack(feats, axis=0).astype(np.float32)
    m = arr.mean(axis=0)
    n = np.linalg.norm(m)
    return (m / max(n, 1e-12)).astype(np.float32)


def load_tracks(path: Path) -> Dict[int, List[dict]]:
    tracks: Dict[int, List[dict]] = defaultdict(list)
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(',')
            if len(p) < 6:
                continue
            try:
                fr = int(float(p[0])); tid = int(float(p[1]))
                x, y, w, h = map(float, p[2:6])
                score = float(p[6]) if len(p) > 6 else 1.0
            except Exception:
                continue
            if w <= 2 or h <= 2:
                continue
            tracks[tid].append({'frame': fr, 'tid': tid, 'x': x, 'y': y, 'w': w, 'h': h, 'score': score})
    for tid in list(tracks):
        tracks[tid].sort(key=lambda r: r['frame'])
    return tracks


def parse_seqinfo(seq_dir: Path) -> tuple[str, str, int]:
    cfg = configparser.ConfigParser()
    cfg.read(seq_dir / 'seqinfo.ini')
    im_dir = cfg['Sequence'].get('imDir', 'img1') if cfg.has_section('Sequence') else 'img1'
    im_ext = cfg['Sequence'].get('imExt', '.jpg') if cfg.has_section('Sequence') else '.jpg'
    seq_len = int(cfg['Sequence'].get('seqLength', '0')) if cfg.has_section('Sequence') else 0
    return im_dir, im_ext, seq_len


def image_path(seq_dir: Path, frame: int) -> Path:
    im_dir, im_ext, _ = parse_seqinfo(seq_dir)
    return seq_dir / im_dir / f'{frame:06d}{im_ext}'


def choose_samples(rows: List[dict], start_k: int, end_k: int, global_k: int, high_k: int) -> dict:
    rows = sorted(rows, key=lambda r: r['frame'])
    start = rows[:start_k]
    end = rows[-end_k:] if end_k > 0 else []
    if global_k > 0 and len(rows) > global_k:
        idxs = np.linspace(0, len(rows) - 1, global_k).round().astype(int).tolist()
        glob = [rows[i] for i in sorted(set(idxs))]
    else:
        glob = list(rows)
    high = sorted(rows, key=lambda r: (-r['score'], r['frame']))[:high_k]
    # Unique samples to avoid repeated ReID inference.
    all_rows = {}
    for group, rs in [('start', start), ('end', end), ('global', glob), ('high', high)]:
        for r in rs:
            key = (int(r['tid']), int(r['frame']), round(float(r['x']), 2), round(float(r['y']), 2), round(float(r['w']), 2), round(float(r['h']), 2))
            if key not in all_rows:
                rr = dict(r)
                rr['sample_groups'] = set()
                all_rows[key] = rr
            all_rows[key]['sample_groups'].add(group)
    return {'start': start, 'end': end, 'global': glob, 'high': high, 'all': list(all_rows.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracks-dir', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--bot-sort-root', default='external/BoT-SORT-main')
    ap.add_argument('--fast-reid-config', default='fast_reid/configs/MOT20/sbs_S50.yml')
    ap.add_argument('--fast-reid-weights', default='pretrained/mot20_sbs_S50.pth')
    ap.add_argument('--device', default='gpu')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--start-k', type=int, default=5)
    ap.add_argument('--end-k', type=int, default=5)
    ap.add_argument('--global-k', type=int, default=20)
    ap.add_argument('--high-k', type=int, default=10)
    args = ap.parse_args()

    bot_root = Path(args.bot_sort_root).resolve()
    sys.path.insert(0, str(bot_root))
    from fast_reid.fast_reid_interfece import FastReIDInterface

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir = Path(args.tracks_dir)
    data_root = Path(args.data_root)

    encoder = FastReIDInterface(
        str(bot_root / args.fast_reid_config),
        str(bot_root / args.fast_reid_weights),
        args.device,
        batch_size=args.batch_size,
    )

    feature_map: Dict[tuple, np.ndarray] = {}
    track_sample_meta = []
    feature_dim = 2048
    total_samples = 0
    failed_images = 0
    seqs = sorted(p.stem for p in tracks_dir.glob('MOT20-*.txt'))

    # Plan samples per sequence/track first.
    planned = {}
    for seq in seqs:
        tracks = load_tracks(tracks_dir / f'{seq}.txt')
        planned[seq] = {}
        for tid, rows in tracks.items():
            planned[seq][tid] = choose_samples(rows, args.start_k, args.end_k, args.global_k, args.high_k)
            planned[seq][tid]['orig_rows'] = rows
            total_samples += len(planned[seq][tid]['all'])

    # Extract features frame by frame for IO efficiency.
    for seq in seqs:
        seq_dir = data_root / seq
        by_frame = defaultdict(list)
        for tid, groups in planned[seq].items():
            for r in groups['all']:
                by_frame[int(r['frame'])].append(r)
        total_frames = len(by_frame)
        print(f'[A23_01] seq={seq} frames_with_samples={total_frames} tracks={len(planned[seq])}', flush=True)
        for frame_idx, (frame, rows) in enumerate(sorted(by_frame.items()), start=1):
            if frame_idx == 1 or frame_idx % 100 == 0 or frame_idx == total_frames:
                print(f'[A23_01] seq={seq} frame_index={frame_idx}/{total_frames} frame={frame} samples={len(rows)} extracted={len(feature_map)}', flush=True)
            img_path = image_path(seq_dir, frame)
            img = cv2.imread(str(img_path))
            if img is None:
                failed_images += 1
                continue
            dets = []
            keys = []
            H, W = img.shape[:2]
            for r in rows:
                x1 = max(0.0, float(r['x'])); y1 = max(0.0, float(r['y']))
                x2 = min(float(W - 1), float(r['x']) + float(r['w'])); y2 = min(float(H - 1), float(r['y']) + float(r['h']))
                if x2 <= x1 + 1 or y2 <= y1 + 1:
                    continue
                dets.append([x1, y1, x2, y2])
                keys.append((seq, int(r['tid']), int(r['frame']), round(float(r['x']), 2), round(float(r['y']), 2), round(float(r['w']), 2), round(float(r['h']), 2)))
            if not dets:
                continue
            feats = encoder.inference(img, np.asarray(dets, dtype=np.float32))
            feats = np.asarray(feats, dtype=np.float32)
            feats = l2norm(feats)
            feature_dim = feats.shape[1] if feats.ndim == 2 and feats.shape[0] else feature_dim
            for k, feat in zip(keys, feats):
                feature_map[k] = feat.astype(np.float32)

    print(f'[A23_01] extraction_done features={len(feature_map)} failed_images={failed_images}', flush=True)
    # Aggregate per tracklet.
    start_feats = []
    end_feats = []
    global_feats = []
    high_feats = []
    rows_index = []
    seq_arr = []
    tid_arr = []
    missing_tracks = 0

    for seq in seqs:
        print(f'[A23_01] aggregate seq={seq}', flush=True)
        for tid, groups in sorted(planned[seq].items()):
            buckets = {'start': [], 'end': [], 'global': [], 'high': []}
            all_count = 0
            for group_name in ['start', 'end', 'global', 'high']:
                for r in groups[group_name]:
                    key = (seq, int(tid), int(r['frame']), round(float(r['x']), 2), round(float(r['y']), 2), round(float(r['w']), 2), round(float(r['h']), 2))
                    if key in feature_map:
                        buckets[group_name].append(feature_map[key])
                        all_count += 1
            sf = mean_norm(buckets['start'], feature_dim)
            ef = mean_norm(buckets['end'], feature_dim)
            gf = mean_norm(buckets['global'], feature_dim)
            hf = mean_norm(buckets['high'], feature_dim)
            if not any(len(v) for v in buckets.values()):
                missing_tracks += 1
            start_feats.append(sf); end_feats.append(ef); global_feats.append(gf); high_feats.append(hf)
            seq_arr.append(seq); tid_arr.append(int(tid))
            all_rows = planned[seq][tid]['all']
            rows_orig = groups.get('orig_rows', [])
            rows_index.append({
                'seq': seq,
                'track_id': int(tid),
                'track_len': len(rows_orig),
                'start_frame': int(rows_orig[0]['frame']),
                'end_frame': int(rows_orig[-1]['frame']),
                'avg_score': float(np.mean([r['score'] for r in rows_orig])) if rows_orig else 0.0,
                'num_start_samples': len(buckets['start']),
                'num_end_samples': len(buckets['end']),
                'num_global_samples': len(buckets['global']),
                'num_high_samples': len(buckets['high']),
                'total_available_features': sum(len(v) for v in buckets.values()),
                'has_feature': int(any(len(v) for v in buckets.values())),
            })

    np.savez_compressed(
        out_dir / 'tracklet_reid_features.npz',
        start=np.stack(start_feats, axis=0).astype(np.float32),
        end=np.stack(end_feats, axis=0).astype(np.float32),
        global_mean=np.stack(global_feats, axis=0).astype(np.float32),
        high_score=np.stack(high_feats, axis=0).astype(np.float32),
        seq=np.asarray(seq_arr),
        track_id=np.asarray(tid_arr, dtype=np.int64),
    )
    with (out_dir / 'tracklet_reid_index.csv').open('w', newline='', encoding='utf-8') as f:
        fields = list(rows_index[0].keys()) if rows_index else ['seq', 'track_id']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows_index)

    norms = {}
    for name, arr in [('start', start_feats), ('end', end_feats), ('global_mean', global_feats), ('high_score', high_feats)]:
        a = np.stack(arr, axis=0) if arr else np.zeros((0, feature_dim), dtype=np.float32)
        n = np.linalg.norm(a, axis=1) if len(a) else np.array([])
        norms[name] = {'mean_norm': float(n.mean()) if len(n) else 0.0, 'min_norm': float(n.min()) if len(n) else 0.0, 'max_norm': float(n.max()) if len(n) else 0.0}

    summary = {
        'seqs': seqs,
        'tracks': len(rows_index),
        'total_unique_samples_planned': total_samples,
        'features_extracted': len(feature_map),
        'missing_tracks': missing_tracks,
        'failed_images': failed_images,
        'feature_dim': feature_dim,
        'norms': norms,
        'output_npz': str(out_dir / 'tracklet_reid_features.npz'),
        'output_index': str(out_dir / 'tracklet_reid_index.csv'),
    }
    (out_dir / 'feature_extract_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    md = ['# A23_01 Tracklet ReID Feature Extraction', '', '| metric | value |', '|---|---:|']
    for k, v in summary.items():
        if k != 'norms':
            md.append(f'| {k} | {v} |')
    md.append('')
    md.append('## Norms')
    md.append('| feature | mean_norm | min_norm | max_norm |')
    md.append('|---|---:|---:|---:|')
    for name, s in norms.items():
        md.append(f"| {name} | {s['mean_norm']:.6f} | {s['min_norm']:.6f} | {s['max_norm']:.6f} |")
    (out_dir / 'feature_extract_summary.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
