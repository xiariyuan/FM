#!/usr/bin/env python3
"""Conservative offline tracklet linker for MOTChallenge results.

This script links fragmented track ids after online tracking. It is intentionally
non-invasive: it does not add boxes or change box geometry; it only remaps later
track ids to earlier track ids when a high-confidence A->B tracklet edge passes
strict appearance, motion, size, rank-margin and conflict gates.

It reuses Phase0 dump ReID features by matching a small number of sampled output
boxes to same-frame dumped detections via IoU, avoiding extra ReID inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class Row:
    frame: int
    tid: int
    x: float
    y: float
    w: float
    h: float
    score: float
    parts: List[str]

    @property
    def tlbr(self) -> np.ndarray:
        return np.asarray([self.x, self.y, self.x + self.w, self.y + self.h], dtype=np.float32)

    @property
    def center(self) -> np.ndarray:
        return np.asarray([self.x + self.w / 2.0, self.y + self.h / 2.0], dtype=np.float32)

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


@dataclass
class Tracklet:
    seq: str
    tid: int
    rows: List[Row]
    start_frame: int
    end_frame: int
    length: int
    first_box: np.ndarray
    last_box: np.ndarray
    first_center: np.ndarray
    last_center: np.ndarray
    avg_score: float
    velocity: np.ndarray
    start_feat: np.ndarray
    end_feat: np.ndarray
    global_feat: np.ndarray
    high_feat: np.ndarray
    feature_count: int
    sample_match_count: int


@dataclass
class Edge:
    seq: str
    track_a: int
    track_b: int
    gap: int
    score: float
    app_sim: float
    motion_score: float
    size_score: float
    direction_score: float
    gap_score: float
    center_step: float
    area_ratio: float
    height_ratio: float
    source_rank: int = 999999
    target_rank: int = 999999
    source_margin: float = 0.0
    target_margin: float = 0.0
    selected: int = 0


def safe_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x: str, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def l2norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def mean_norm(feats: List[np.ndarray], dim: int = 2048) -> np.ndarray:
    feats = [np.asarray(f, dtype=np.float32) for f in feats if f is not None and np.asarray(f).size]
    if not feats:
        return np.zeros((dim,), dtype=np.float32)
    return l2norm(np.stack(feats, axis=0).mean(axis=0))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def iou_one(a: np.ndarray, b: np.ndarray) -> float:
    xx1 = max(float(a[0]), float(b[0])); yy1 = max(float(a[1]), float(b[1]))
    xx2 = min(float(a[2]), float(b[2])); yy2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, xx2 - xx1); ih = max(0.0, yy2 - yy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return float(inter / max(area_a + area_b - inter, 1e-12))


def load_mot(path: Path) -> Dict[int, List[Row]]:
    tracks: Dict[int, List[Row]] = defaultdict(list)
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(',')
            if len(p) < 6:
                continue
            fr = safe_int(p[0]); tid = safe_int(p[1])
            x, y, w, h = [safe_float(v) for v in p[2:6]]
            score = safe_float(p[6], 1.0) if len(p) > 6 else 1.0
            if fr <= 0 or tid <= 0 or w <= 1.0 or h <= 1.0:
                continue
            tracks[tid].append(Row(fr, tid, x, y, w, h, score, list(p)))
    for tid in list(tracks):
        tracks[tid].sort(key=lambda r: r.frame)
    return tracks


def sample_rows(rows: List[Row], start_k: int, end_k: int, global_k: int, high_k: int) -> Dict[str, List[Row]]:
    rows = sorted(rows, key=lambda r: r.frame)
    out: Dict[str, List[Row]] = {
        'start': rows[:start_k],
        'end': rows[-end_k:] if end_k > 0 else [],
        'high': sorted(rows, key=lambda r: (-r.score, r.frame))[:high_k],
    }
    if global_k > 0 and len(rows) > global_k:
        idxs = np.linspace(0, len(rows) - 1, global_k).round().astype(int).tolist()
        out['global'] = [rows[i] for i in sorted(set(idxs))]
    else:
        out['global'] = list(rows)
    return out


def collect_sample_frames(tracks: Dict[int, List[Row]], args) -> set[int]:
    frames: set[int] = set()
    for _tid, rows in tracks.items():
        if len(rows) < int(args.min_track_len):
            continue
        samples = sample_rows(rows, args.start_k, args.end_k, args.global_k, args.high_k)
        for rs in samples.values():
            for r in rs:
                frames.add(int(r.frame))
    return frames


def load_phase0_by_frame(npz_path: Path, wanted_frames: Optional[set[int]] = None) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    z = np.load(npz_path, allow_pickle=True)
    det = np.asarray(z['detections'], dtype=np.float32)
    feat = np.asarray(z['features'])
    offsets = np.asarray(z['frame_offsets'], dtype=np.int64)
    columns = [str(x) for x in z['columns'].tolist()]
    col = {name: i for i, name in enumerate(columns)}
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    dim = int(feat.shape[1]) if feat.ndim == 2 else 0
    wanted = wanted_frames if wanted_frames is not None else set(range(1, len(offsets)))
    for frame in sorted(wanted):
        if frame < 1 or frame >= len(offsets):
            continue
        start = int(offsets[frame - 1]); end = int(offsets[frame])
        if end <= start:
            out[frame] = (np.zeros((0, 4), dtype=np.float32), np.zeros((0, dim), dtype=feat.dtype if hasattr(feat, 'dtype') else np.float32))
            continue
        boxes = det[start:end][:, [col['x1'], col['y1'], col['x2'], col['y2']]].astype(np.float32)
        feats = feat[start:end]
        out[frame] = (boxes, feats)
    return out


def match_feature(row: Row, by_frame: Dict[int, Tuple[np.ndarray, np.ndarray]], min_iou: float) -> Optional[np.ndarray]:
    boxes, feats = by_frame.get(row.frame, (None, None))
    if boxes is None or feats is None or boxes.shape[0] == 0:
        return None
    rb = row.tlbr
    # vectorized IoU
    xx1 = np.maximum(boxes[:, 0], rb[0]); yy1 = np.maximum(boxes[:, 1], rb[1])
    xx2 = np.minimum(boxes[:, 2], rb[2]); yy2 = np.minimum(boxes[:, 3], rb[3])
    iw = np.maximum(0.0, xx2 - xx1); ih = np.maximum(0.0, yy2 - yy1)
    inter = iw * ih
    area_r = max(0.0, float((rb[2] - rb[0]) * (rb[3] - rb[1])))
    area_b = np.maximum(0.0, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    ious = inter / np.maximum(area_r + area_b - inter, 1e-12)
    idx = int(np.argmax(ious))
    if float(ious[idx]) < min_iou:
        return None
    f = feats[idx]
    if float(np.linalg.norm(f)) < 1e-8:
        return None
    return l2norm(f)


def estimate_velocity(rows: List[Row], k: int) -> np.ndarray:
    if len(rows) < 2:
        return np.zeros((2,), dtype=np.float32)
    tail = rows[-k:] if len(rows) > k else rows
    if len(tail) < 2:
        return np.zeros((2,), dtype=np.float32)
    dt = max(1, int(tail[-1].frame - tail[0].frame))
    return ((tail[-1].center - tail[0].center) / float(dt)).astype(np.float32)


def build_tracklets(seq: str, tracks: Dict[int, List[Row]], by_frame: Dict[int, Tuple[np.ndarray, np.ndarray]], args) -> Dict[int, Tracklet]:
    out: Dict[int, Tracklet] = {}
    for tid, rows in tracks.items():
        if len(rows) < int(args.min_track_len):
            continue
        samples = sample_rows(rows, args.start_k, args.end_k, args.global_k, args.high_k)
        buckets: Dict[str, List[np.ndarray]] = {k: [] for k in samples}
        matched = 0
        for name, rs in samples.items():
            seen = set()
            for r in rs:
                key = (r.frame, round(r.x, 2), round(r.y, 2), round(r.w, 2), round(r.h, 2))
                if key in seen:
                    continue
                seen.add(key)
                feat = match_feature(r, by_frame, args.min_sample_iou)
                if feat is not None:
                    buckets[name].append(feat)
                    matched += 1
        dim = 2048
        for _boxes, feats in by_frame.values():
            if feats.ndim == 2 and feats.shape[1] > 0:
                dim = int(feats.shape[1]); break
        start_feat = mean_norm(buckets.get('start', []), dim)
        end_feat = mean_norm(buckets.get('end', []), dim)
        global_feat = mean_norm(buckets.get('global', []), dim)
        high_feat = mean_norm(buckets.get('high', []), dim)
        feats_available = int(any(float(np.linalg.norm(v)) > 1e-8 for v in [start_feat, end_feat, global_feat, high_feat]))
        out[tid] = Tracklet(
            seq=seq,
            tid=tid,
            rows=rows,
            start_frame=int(rows[0].frame),
            end_frame=int(rows[-1].frame),
            length=len(rows),
            first_box=rows[0].tlbr,
            last_box=rows[-1].tlbr,
            first_center=rows[0].center,
            last_center=rows[-1].center,
            avg_score=float(np.mean([r.score for r in rows])),
            velocity=estimate_velocity(rows, args.velocity_k),
            start_feat=start_feat,
            end_feat=end_feat,
            global_feat=global_feat,
            high_feat=high_feat,
            feature_count=feats_available,
            sample_match_count=int(matched),
        )
    return out


def score_edge(a: Tracklet, b: Tracklet, args) -> Optional[Edge]:
    gap = int(b.start_frame - a.end_frame)
    if gap <= 0 or gap > int(args.max_gap):
        return None
    if a.length < int(args.min_source_len) or b.length < int(args.min_target_len):
        return None
    # appearance: clean endpoint + global + high-score support
    end_start = cosine(a.end_feat, b.start_feat)
    global_sim = cosine(a.global_feat, b.global_feat)
    high_sim = cosine(a.high_feat, b.high_feat)
    app_sim = 0.55 * end_start + 0.30 * global_sim + 0.15 * high_sim
    if app_sim < float(args.min_app):
        return None

    pred = a.last_center + a.velocity * float(gap)
    dist = float(np.linalg.norm(pred - b.first_center))
    step = dist / max(1.0, float(gap))
    if step > float(args.max_center_step):
        return None
    scale = math.sqrt(max(float(a.rows[-1].area), float(b.rows[0].area), 1.0))
    motion_score = math.exp(-dist / max(scale * float(args.motion_scale), 1e-6))

    area_a = max(float(a.rows[-1].area), 1.0); area_b = max(float(b.rows[0].area), 1.0)
    area_ratio = max(area_a, area_b) / max(1.0, min(area_a, area_b))
    if area_ratio > float(args.max_area_ratio):
        return None
    ha = max(float(a.rows[-1].h), 1.0); hb = max(float(b.rows[0].h), 1.0)
    height_ratio = max(ha, hb) / max(1.0, min(ha, hb))
    if height_ratio > float(args.max_height_ratio):
        return None
    size_score = math.exp(-abs(math.log(area_ratio)))

    direction_score = 0.5
    disp = b.first_center - a.last_center
    if float(np.linalg.norm(a.velocity)) > 1e-6 and float(np.linalg.norm(disp)) > 1e-6:
        direction_score = max(0.0, float(np.dot(a.velocity, disp) / (np.linalg.norm(a.velocity) * np.linalg.norm(disp))))
    gap_score = math.exp(-float(gap) / max(float(args.gap_decay), 1.0))
    score = (
        float(args.w_app) * app_sim +
        float(args.w_motion) * motion_score +
        float(args.w_size) * size_score +
        float(args.w_direction) * direction_score +
        float(args.w_gap) * gap_score
    )
    if score < float(args.min_score):
        return None
    return Edge(a.seq, a.tid, b.tid, gap, float(score), float(app_sim), float(motion_score), float(size_score), float(direction_score), float(gap_score), float(step), float(area_ratio), float(height_ratio))


def add_ranks_and_filter(edges: List[Edge], args) -> List[Edge]:
    by_a: Dict[int, List[Edge]] = defaultdict(list)
    by_b: Dict[int, List[Edge]] = defaultdict(list)
    for e in edges:
        by_a[e.track_a].append(e)
        by_b[e.track_b].append(e)
    for group in by_a.values():
        group.sort(key=lambda e: -e.score)
        best = group[0].score if group else 0.0
        second = group[1].score if len(group) > 1 else -1.0
        for i, e in enumerate(group, start=1):
            e.source_rank = i
            if i == 1:
                e.source_margin = float(best - second) if second >= 0 else 1.0
    for group in by_b.values():
        group.sort(key=lambda e: -e.score)
        best = group[0].score if group else 0.0
        second = group[1].score if len(group) > 1 else -1.0
        for i, e in enumerate(group, start=1):
            e.target_rank = i
            if i == 1:
                e.target_margin = float(best - second) if second >= 0 else 1.0
    out = []
    for e in edges:
        if args.require_mutual_top1 and (e.source_rank != 1 or e.target_rank != 1):
            continue
        if e.source_margin < float(args.min_margin) or e.target_margin < float(args.min_margin):
            continue
        out.append(e)
    return out


def find(parent: Dict[int, int], x: int) -> int:
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def select_edges(edges: List[Edge], max_links: int) -> List[Edge]:
    edges = sorted(edges, key=lambda e: (-e.score, e.gap, e.track_a, e.track_b))
    used_succ = set()
    used_pred = set()
    parent: Dict[int, int] = {}
    selected: List[Edge] = []
    for e in edges:
        if len(selected) >= max_links:
            break
        if e.track_a in used_succ or e.track_b in used_pred:
            continue
        ra = find(parent, e.track_a); rb = find(parent, e.track_b)
        if ra == rb:
            continue
        parent[rb] = ra
        used_succ.add(e.track_a); used_pred.add(e.track_b)
        e.selected = 1
        selected.append(e)
    return selected


def apply_links(input_file: Path, output_file: Path, selected: List[Edge]) -> Dict[str, int]:
    parent: Dict[int, int] = {}
    for e in selected:
        ra = find(parent, e.track_a); rb = find(parent, e.track_b)
        parent[rb] = ra
    # Build full map for involved ids.
    involved = set()
    for e in selected:
        involved.add(e.track_a); involved.add(e.track_b)
    id_map = {tid: find(parent, tid) for tid in involved}
    rows = []
    remapped = 0
    with input_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(',')
            tid = safe_int(p[1]) if len(p) > 1 else 0
            if tid in id_map:
                root = id_map[tid]
                if root != tid:
                    p[1] = str(root)
                    remapped += 1
            rows.append(p)
    rows.sort(key=lambda p: (safe_int(p[0]), safe_int(p[1]), safe_float(p[2]), safe_float(p[3])))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', encoding='utf-8') as f:
        for p in rows:
            f.write(','.join(p) + '\n')
    return {'rows': len(rows), 'remapped_rows': remapped, 'links': len(selected)}


def write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def process_seq(seq: str, args) -> Dict[str, object]:
    input_file = Path(args.input_dir) / f'{seq}.txt'
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    phase0_npz = Path(args.phase0_root) / seq / 'dump_yolox_reid.npz'
    if not phase0_npz.is_file():
        raise FileNotFoundError(phase0_npz)
    tracks = load_mot(input_file)
    wanted_frames = collect_sample_frames(tracks, args)
    print(f'[safe_link] seq={seq} tracks={len(tracks)} sample_frames={len(wanted_frames)}', flush=True)
    by_frame = load_phase0_by_frame(phase0_npz, wanted_frames=wanted_frames)
    tracklets = build_tracklets(seq, tracks, by_frame, args)
    tids = sorted(tracklets)
    starts = sorted(tids, key=lambda tid: tracklets[tid].start_frame)
    edges: List[Edge] = []
    # Candidate construction: A end < B start <= A end + max_gap.
    for a_id in tids:
        a = tracklets[a_id]
        for b_id in starts:
            if b_id == a_id:
                continue
            b = tracklets[b_id]
            if b.start_frame <= a.end_frame:
                continue
            if b.start_frame - a.end_frame > args.max_gap:
                if b.start_frame > a.end_frame:
                    break
            e = score_edge(a, b, args)
            if e is not None:
                edges.append(e)
    gated = add_ranks_and_filter(edges, args)
    selected = select_edges(gated, int(args.max_links_per_seq))
    out_file = Path(args.output_dir) / f'{seq}.txt'
    apply_stats = apply_links(input_file, out_file, selected)

    edge_rows = [asdict(e) for e in edges]
    gated_rows = [asdict(e) for e in gated]
    selected_rows = [asdict(e) for e in selected]
    detail_dir = Path(args.detail_dir) if args.detail_dir else Path(args.output_dir).parent / 'safe_link_details'
    if args.write_details:
        write_csv(detail_dir / f'{seq}_candidate_edges.csv', edge_rows, list(asdict(edges[0]).keys()) if edges else list(Edge(seq,0,0,0,0,0,0,0,0,0,0,0,0).__dict__.keys()))
        write_csv(detail_dir / f'{seq}_gated_edges.csv', gated_rows, list(asdict(gated[0]).keys()) if gated else list(Edge(seq,0,0,0,0,0,0,0,0,0,0,0,0).__dict__.keys()))
        write_csv(detail_dir / f'{seq}_selected_links.csv', selected_rows, list(asdict(selected[0]).keys()) if selected else list(Edge(seq,0,0,0,0,0,0,0,0,0,0,0,0).__dict__.keys()))
    return {
        'seq': seq,
        'tracks_input': len(tracks),
        'tracklets_considered': len(tracklets),
        'candidate_edges': len(edges),
        'gated_edges': len(gated),
        'selected_links': len(selected),
        **apply_stats,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Safe conservative tracklet linker using Phase0 ReID features.')
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--phase0-root', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--seqs', nargs='*', default=[])
    ap.add_argument('--detail-dir', default='')
    ap.add_argument('--write-details', action='store_true')
    ap.add_argument('--max-gap', type=int, default=60)
    ap.add_argument('--min-track-len', type=int, default=2)
    ap.add_argument('--min-source-len', type=int, default=10)
    ap.add_argument('--min-target-len', type=int, default=5)
    ap.add_argument('--start-k', type=int, default=5)
    ap.add_argument('--end-k', type=int, default=5)
    ap.add_argument('--global-k', type=int, default=20)
    ap.add_argument('--high-k', type=int, default=10)
    ap.add_argument('--velocity-k', type=int, default=5)
    ap.add_argument('--min-sample-iou', type=float, default=0.45)
    ap.add_argument('--min-app', type=float, default=0.72)
    ap.add_argument('--min-score', type=float, default=0.72)
    ap.add_argument('--min-margin', type=float, default=0.03)
    ap.add_argument('--max-center-step', type=float, default=80.0)
    ap.add_argument('--max-area-ratio', type=float, default=2.5)
    ap.add_argument('--max-height-ratio', type=float, default=1.8)
    ap.add_argument('--motion-scale', type=float, default=5.0)
    ap.add_argument('--gap-decay', type=float, default=40.0)
    ap.add_argument('--w-app', type=float, default=0.60)
    ap.add_argument('--w-motion', type=float, default=0.15)
    ap.add_argument('--w-size', type=float, default=0.10)
    ap.add_argument('--w-direction', type=float, default=0.10)
    ap.add_argument('--w-gap', type=float, default=0.05)
    ap.add_argument('--max-links-per-seq', type=int, default=999999)
    ap.add_argument('--require-mutual-top1', action='store_true', default=True)
    ap.add_argument('--summary-json', default='')
    ap.add_argument('--summary-csv', default='')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    if args.seqs:
        seqs = args.seqs
    else:
        seqs = sorted(p.stem for p in in_dir.glob('MOT20-*.txt'))
    summaries = []
    for seq in seqs:
        print(f'[safe_link] seq={seq}', flush=True)
        summaries.append(process_seq(seq, args))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summaries, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if args.summary_csv:
        fields = list(summaries[0].keys()) if summaries else ['seq']
        write_csv(Path(args.summary_csv), summaries, fields)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
