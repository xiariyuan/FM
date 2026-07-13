#!/usr/bin/env python3
"""A-Link v2: wide-candidate conservative tracklet linker.

This post-processer only remaps track ids. It never adds/removes/changes boxes.
It builds wide temporal candidates, scores them with ReID/motion/size/gap cues,
assigns tier1/tier2/tier3 decisions, and greedily selects non-conflicting links.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    first_center: np.ndarray
    last_center: np.ndarray
    avg_score: float
    velocity: np.ndarray
    start_feat: np.ndarray
    end_feat: np.ndarray
    global_feat: np.ndarray
    high_feat: np.ndarray
    bank: np.ndarray
    feature_count: int
    sample_match_count: int


@dataclass
class Edge:
    seq: str
    track_a: int
    track_b: int
    gap: int
    edge_score: float
    app_sim: float
    bank_max: float
    bank_topk: float
    end_start_sim: float
    global_sim: float
    high_sim: float
    motion_score: float
    size_score: float
    direction_score: float
    gap_score: float
    quality_score: float
    center_step: float
    pred_dist: float
    area_ratio: float
    height_ratio: float
    source_rank: int = 999999
    target_rank: int = 999999
    source_margin: float = 0.0
    target_margin: float = 0.0
    tier: str = "reject"
    selected: int = 0


def sf(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def si(x, default: int = 0) -> int:
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


def read_mot(path: Path) -> Dict[int, List[Row]]:
    tracks: Dict[int, List[Row]] = defaultdict(list)
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(',')
            if len(p) < 6:
                continue
            fr = si(p[0]); tid = si(p[1])
            x, y, w, h = [sf(v) for v in p[2:6]]
            score = sf(p[6], 1.0) if len(p) > 6 else 1.0
            if fr <= 0 or tid <= 0 or w <= 1.0 or h <= 1.0:
                continue
            tracks[tid].append(Row(fr, tid, x, y, w, h, score, list(p)))
    for tid in list(tracks):
        tracks[tid].sort(key=lambda r: r.frame)
    return tracks


def load_phase0_by_frame(npz_path: Path) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    z = np.load(npz_path, allow_pickle=True)
    det = np.asarray(z['detections'], dtype=np.float32)
    feat = np.asarray(z['features'], dtype=np.float32)
    offsets = np.asarray(z['frame_offsets'], dtype=np.int64)
    columns = [str(x) for x in z['columns'].tolist()]
    c = {name: i for i, name in enumerate(columns)}
    dim = int(feat.shape[1]) if feat.ndim == 2 else 0
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for frame in range(1, len(offsets)):
        a = int(offsets[frame - 1]); b = int(offsets[frame])
        if b <= a:
            out[frame] = (np.zeros((0, 4), dtype=np.float32), np.zeros((0, dim), dtype=np.float32))
        else:
            boxes = det[a:b][:, [c['x1'], c['y1'], c['x2'], c['y2']]].astype(np.float32)
            feats = feat[a:b].astype(np.float32)
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            feats = feats / np.maximum(norms, 1e-12)
            out[frame] = (boxes, feats.astype(np.float32))
    return out


def match_feature(row: Row, by_frame: Dict[int, Tuple[np.ndarray, np.ndarray]], min_iou: float) -> Optional[np.ndarray]:
    boxes, feats = by_frame.get(row.frame, (None, None))
    if boxes is None or feats is None or boxes.shape[0] == 0:
        return None
    rb = row.tlbr
    xx1 = np.maximum(boxes[:, 0], rb[0]); yy1 = np.maximum(boxes[:, 1], rb[1])
    xx2 = np.minimum(boxes[:, 2], rb[2]); yy2 = np.minimum(boxes[:, 3], rb[3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_r = max(0.0, float((rb[2] - rb[0]) * (rb[3] - rb[1])))
    area_b = np.maximum(0.0, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    iou = inter / np.maximum(area_r + area_b - inter, 1e-12)
    idx = int(np.argmax(iou))
    if float(iou[idx]) < float(min_iou):
        return None
    f = feats[idx]
    if float(np.linalg.norm(f)) < 1e-8:
        return None
    return l2norm(f)


def sample_rows(rows: List[Row], start_k: int, end_k: int, global_k: int, high_k: int) -> Dict[str, List[Row]]:
    rows = sorted(rows, key=lambda r: r.frame)
    out = {
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


def estimate_velocity(rows: List[Row], k: int) -> np.ndarray:
    if len(rows) < 2:
        return np.zeros((2,), dtype=np.float32)
    sub = rows[-k:]
    if len(sub) < 2:
        return np.zeros((2,), dtype=np.float32)
    dt = max(1, int(sub[-1].frame - sub[0].frame))
    return ((sub[-1].center - sub[0].center) / float(dt)).astype(np.float32)


def build_tracklets(seq: str, tracks: Dict[int, List[Row]], by_frame: Dict[int, Tuple[np.ndarray, np.ndarray]], args) -> Dict[int, Tracklet]:
    dim = 2048
    for _boxes, feats in by_frame.values():
        if feats.ndim == 2 and feats.shape[1] > 0:
            dim = int(feats.shape[1]); break
    out: Dict[int, Tracklet] = {}
    for tid, rows in tracks.items():
        if len(rows) < int(args.min_track_len):
            continue
        samples = sample_rows(rows, args.start_k, args.end_k, args.global_k, args.high_k)
        buckets: Dict[str, List[np.ndarray]] = {k: [] for k in samples}
        bank_feats: List[np.ndarray] = []
        seen_rows = set()
        matched = 0
        for name, rs in samples.items():
            for r in rs:
                key = (r.frame, round(r.x, 2), round(r.y, 2), round(r.w, 2), round(r.h, 2))
                feat = match_feature(r, by_frame, args.min_sample_iou)
                if feat is not None:
                    buckets[name].append(feat)
                    matched += 1
                    if key not in seen_rows:
                        bank_feats.append(feat); seen_rows.add(key)
        if len(bank_feats) > args.max_bank_feats:
            idxs = np.linspace(0, len(bank_feats) - 1, args.max_bank_feats).round().astype(int).tolist()
            bank_feats = [bank_feats[i] for i in sorted(set(idxs))]
        bank = np.stack(bank_feats, axis=0).astype(np.float32) if bank_feats else np.zeros((0, dim), dtype=np.float32)
        out[tid] = Tracklet(
            seq=seq, tid=tid, rows=rows, start_frame=rows[0].frame, end_frame=rows[-1].frame,
            length=len(rows), first_center=rows[0].center, last_center=rows[-1].center,
            avg_score=float(np.mean([r.score for r in rows])), velocity=estimate_velocity(rows, args.velocity_k),
            start_feat=mean_norm(buckets.get('start', []), dim), end_feat=mean_norm(buckets.get('end', []), dim),
            global_feat=mean_norm(buckets.get('global', []), dim), high_feat=mean_norm(buckets.get('high', []), dim),
            bank=bank, feature_count=int(bank.shape[0] > 0), sample_match_count=int(matched)
        )
    return out


def bank_sims(a: Tracklet, b: Tracklet, topk: int) -> Tuple[float, float]:
    if a.bank.shape[0] == 0 or b.bank.shape[0] == 0:
        return 0.0, 0.0
    sims = np.matmul(a.bank.astype(np.float32), b.bank.astype(np.float32).T).reshape(-1)
    if sims.size == 0:
        return 0.0, 0.0
    m = float(np.max(sims)); k = min(int(topk), sims.size)
    top = np.partition(sims, -k)[-k:] if k > 0 else sims
    return m, float(np.mean(top))


def edge_features(a: Tracklet, b: Tracklet, args) -> Optional[Edge]:
    gap = int(b.start_frame - a.end_frame)
    if gap <= 0 or gap > int(args.max_gap):
        return None
    if a.length < int(args.min_source_len) or b.length < int(args.min_target_len):
        return None
    pred = a.last_center + a.velocity * float(gap)
    pred_dist = float(np.linalg.norm(pred - b.first_center))
    center_step = pred_dist / max(1.0, float(gap))
    if center_step > float(args.candidate_max_center_step):
        return None
    area_a = max(float(a.rows[-1].area), 1.0); area_b = max(float(b.rows[0].area), 1.0)
    area_ratio = max(area_a, area_b) / max(1.0, min(area_a, area_b))
    if area_ratio > float(args.candidate_max_area_ratio):
        return None
    h_a = max(float(a.rows[-1].h), 1.0); h_b = max(float(b.rows[0].h), 1.0)
    height_ratio = max(h_a, h_b) / max(1.0, min(h_a, h_b))
    if height_ratio > float(args.candidate_max_height_ratio):
        return None

    bank_max, bank_topk = bank_sims(a, b, args.bank_topk)
    end_start = cosine(a.end_feat, b.start_feat)
    global_sim = cosine(a.global_feat, b.global_feat)
    high_sim = cosine(a.high_feat, b.high_feat)
    app_sim = (args.app_w_bankmax * bank_max + args.app_w_banktopk * bank_topk +
               args.app_w_endstart * end_start + args.app_w_global * global_sim + args.app_w_high * high_sim)
    scale = math.sqrt(max(area_a, area_b, 1.0))
    motion_score = math.exp(-pred_dist / max(scale * float(args.motion_scale), 1e-6))
    size_score = math.exp(-abs(math.log(max(area_ratio, 1e-6)))) * math.exp(-0.5 * abs(math.log(max(height_ratio, 1e-6))))
    disp = b.first_center - a.last_center
    direction_score = 0.5
    if float(np.linalg.norm(a.velocity)) > 1e-6 and float(np.linalg.norm(disp)) > 1e-6:
        direction_score = max(0.0, float(np.dot(a.velocity, disp) / (np.linalg.norm(a.velocity) * np.linalg.norm(disp))))
    gap_score = math.exp(-float(gap) / max(float(args.gap_decay), 1.0))
    q_len = min(1.0, math.log1p(min(a.length, b.length)) / math.log1p(float(args.quality_len_norm)))
    q_score = min(1.0, max(0.0, (a.avg_score + b.avg_score) / 2.0))
    q_feat = 1.0 if a.feature_count and b.feature_count else 0.0
    quality_score = 0.45 * q_len + 0.35 * q_score + 0.20 * q_feat
    edge_score = (args.w_app * app_sim + args.w_motion * motion_score + args.w_size * size_score +
                  args.w_direction * direction_score + args.w_gap * gap_score + args.w_quality * quality_score)
    return Edge(a.seq, a.tid, b.tid, gap, float(edge_score), float(app_sim), float(bank_max), float(bank_topk),
                float(end_start), float(global_sim), float(high_sim), float(motion_score), float(size_score),
                float(direction_score), float(gap_score), float(quality_score), float(center_step), float(pred_dist),
                float(area_ratio), float(height_ratio))


def add_competition(edges: List[Edge]) -> None:
    by_a: Dict[int, List[Edge]] = defaultdict(list); by_b: Dict[int, List[Edge]] = defaultdict(list)
    for e in edges:
        by_a[e.track_a].append(e); by_b[e.track_b].append(e)
    for group in by_a.values():
        group.sort(key=lambda e: -e.edge_score)
        second = group[1].edge_score if len(group) > 1 else -1.0
        for i, e in enumerate(group, 1):
            e.source_rank = i
            if i == 1:
                e.source_margin = float(e.edge_score - second) if second >= 0 else 1.0
    for group in by_b.values():
        group.sort(key=lambda e: -e.edge_score)
        second = group[1].edge_score if len(group) > 1 else -1.0
        for i, e in enumerate(group, 1):
            e.target_rank = i
            if i == 1:
                e.target_margin = float(e.edge_score - second) if second >= 0 else 1.0


def assign_tiers(edges: List[Edge], args) -> List[Edge]:
    out = []
    for e in edges:
        e.tier = 'reject'
        margin = min(e.source_margin, e.target_margin)
        mutual = e.source_rank == 1 and e.target_rank == 1
        if (mutual and margin >= args.tier1_margin and e.app_sim >= args.tier1_app and e.edge_score >= args.tier1_score and
            e.center_step <= args.tier1_center_step and e.area_ratio <= args.tier1_area_ratio and e.height_ratio <= args.tier1_height_ratio):
            e.tier = 'tier1'
        elif (e.source_rank <= args.tier2_max_rank and e.target_rank <= args.tier2_max_rank and margin >= args.tier2_margin and
              e.app_sim >= args.tier2_app and e.edge_score >= args.tier2_score and e.center_step <= args.tier2_center_step and
              e.area_ratio <= args.tier2_area_ratio and e.height_ratio <= args.tier2_height_ratio):
            e.tier = 'tier2'
        elif (e.source_rank <= args.tier3_max_rank and e.target_rank <= args.tier3_max_rank and margin >= args.tier3_margin and
              e.app_sim >= args.tier3_app and e.edge_score >= args.tier3_score and e.center_step <= args.tier3_center_step and
              e.area_ratio <= args.tier3_area_ratio and e.height_ratio <= args.tier3_height_ratio and
              e.motion_score >= args.tier3_motion and e.size_score >= args.tier3_size and e.gap <= args.tier3_max_gap):
            e.tier = 'tier3'
        if e.tier != 'reject':
            out.append(e)
    return out


def find(parent: Dict[int, int], x: int) -> int:
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x


def select_edges(edges: List[Edge], max_links: int) -> List[Edge]:
    tier_rank = {'tier1': 0, 'tier2': 1, 'tier3': 2, 'reject': 9}
    edges = sorted(edges, key=lambda e: (tier_rank.get(e.tier, 9), -e.edge_score, e.gap, e.track_a, e.track_b))
    used_a, used_b, parent, out = set(), set(), {}, []
    for e in edges:
        if len(out) >= int(max_links):
            break
        if e.track_a in used_a or e.track_b in used_b:
            continue
        ra = find(parent, e.track_a); rb = find(parent, e.track_b)
        if ra == rb:
            continue
        parent[rb] = ra; used_a.add(e.track_a); used_b.add(e.track_b); e.selected = 1; out.append(e)
    return out


def apply_links(input_file: Path, output_file: Path, selected: List[Edge]) -> Dict[str, int]:
    parent: Dict[int, int] = {}
    for e in selected:
        ra = find(parent, e.track_a); rb = find(parent, e.track_b); parent[rb] = ra
    involved = set()
    for e in selected:
        involved.add(e.track_a); involved.add(e.track_b)
    id_map = {tid: find(parent, tid) for tid in involved}
    rows, remapped = [], 0
    with input_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(','); tid = si(p[1]) if len(p) > 1 else 0
            if tid in id_map and id_map[tid] != tid:
                p[1] = str(id_map[tid]); remapped += 1
            rows.append(p)
    rows.sort(key=lambda p: (si(p[0]), si(p[1]), sf(p[2]), sf(p[3])))
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
    phase0_npz = Path(args.phase0_root) / seq / 'dump_yolox_reid.npz'
    tracks = read_mot(input_file)
    by_frame = load_phase0_by_frame(phase0_npz)
    tracklets = build_tracklets(seq, tracks, by_frame, args)
    tids = sorted(tracklets)
    starts = sorted(tids, key=lambda tid: tracklets[tid].start_frame)
    edges: List[Edge] = []
    for a_id in tids:
        a = tracklets[a_id]
        for b_id in starts:
            if b_id == a_id:
                continue
            b = tracklets[b_id]
            if b.start_frame <= a.end_frame:
                continue
            if b.start_frame - a.end_frame > int(args.max_gap):
                break
            e = edge_features(a, b, args)
            if e is not None:
                edges.append(e)
    add_competition(edges)
    tiered = assign_tiers(edges, args)
    selected = select_edges(tiered, args.max_links_per_seq)
    out_file = Path(args.output_dir) / f'{seq}.txt'
    apply_stats = apply_links(input_file, out_file, selected)
    detail_dir = Path(args.detail_dir) if args.detail_dir else Path(args.output_dir).parent / 'alink_v2_details'
    fields = list(asdict(Edge(seq, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).keys())
    if args.write_details:
        if not args.skip_candidate_details:
            write_csv(detail_dir / f'{seq}_candidate_edges.csv', [asdict(e) for e in edges], fields)
        write_csv(detail_dir / f'{seq}_tiered_edges.csv', [asdict(e) for e in tiered], fields)
        write_csv(detail_dir / f'{seq}_selected_links.csv', [asdict(e) for e in selected], fields)
    tier_counts = defaultdict(int)
    for e in tiered:
        tier_counts[e.tier] += 1
    return {'seq': seq, 'tracks_input': len(tracks), 'tracklets_considered': len(tracklets), 'candidate_edges': len(edges),
            'tiered_edges': len(tiered), 'selected_links': len(selected), 'tier1': int(tier_counts.get('tier1', 0)),
            'tier2': int(tier_counts.get('tier2', 0)), 'tier3': int(tier_counts.get('tier3', 0)), **apply_stats}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True); ap.add_argument('--phase0-root', required=True); ap.add_argument('--output-dir', required=True)
    ap.add_argument('--seqs', nargs='*', default=[]); ap.add_argument('--detail-dir', default=''); ap.add_argument('--write-details', action='store_true')
    ap.add_argument('--skip-candidate-details', action='store_true'); ap.add_argument('--summary-json', default=''); ap.add_argument('--summary-csv', default='')
    ap.add_argument('--max-gap', type=int, default=300); ap.add_argument('--min-track-len', type=int, default=2); ap.add_argument('--min-source-len', type=int, default=3); ap.add_argument('--min-target-len', type=int, default=2)
    ap.add_argument('--candidate-max-center-step', type=float, default=220.0); ap.add_argument('--candidate-max-area-ratio', type=float, default=10.0); ap.add_argument('--candidate-max-height-ratio', type=float, default=4.0)
    ap.add_argument('--start-k', type=int, default=6); ap.add_argument('--end-k', type=int, default=6); ap.add_argument('--global-k', type=int, default=24); ap.add_argument('--high-k', type=int, default=12); ap.add_argument('--max-bank-feats', type=int, default=32); ap.add_argument('--bank-topk', type=int, default=5); ap.add_argument('--velocity-k', type=int, default=5); ap.add_argument('--min-sample-iou', type=float, default=0.45)
    ap.add_argument('--app-w-bankmax', type=float, default=0.35); ap.add_argument('--app-w-banktopk', type=float, default=0.25); ap.add_argument('--app-w-endstart', type=float, default=0.20); ap.add_argument('--app-w-global', type=float, default=0.15); ap.add_argument('--app-w-high', type=float, default=0.05)
    ap.add_argument('--motion-scale', type=float, default=5.0); ap.add_argument('--gap-decay', type=float, default=80.0); ap.add_argument('--quality-len-norm', type=int, default=30)
    ap.add_argument('--w-app', type=float, default=0.45); ap.add_argument('--w-motion', type=float, default=0.18); ap.add_argument('--w-size', type=float, default=0.12); ap.add_argument('--w-direction', type=float, default=0.10); ap.add_argument('--w-gap', type=float, default=0.07); ap.add_argument('--w-quality', type=float, default=0.08)
    ap.add_argument('--tier1-app', type=float, default=0.70); ap.add_argument('--tier1-score', type=float, default=0.70); ap.add_argument('--tier1-margin', type=float, default=0.01); ap.add_argument('--tier1-center-step', type=float, default=100.0); ap.add_argument('--tier1-area-ratio', type=float, default=4.0); ap.add_argument('--tier1-height-ratio', type=float, default=2.5)
    ap.add_argument('--tier2-app', type=float, default=0.58); ap.add_argument('--tier2-score', type=float, default=0.62); ap.add_argument('--tier2-margin', type=float, default=0.0); ap.add_argument('--tier2-max-rank', type=int, default=2); ap.add_argument('--tier2-center-step', type=float, default=80.0); ap.add_argument('--tier2-area-ratio', type=float, default=3.5); ap.add_argument('--tier2-height-ratio', type=float, default=2.2)
    ap.add_argument('--tier3-app', type=float, default=0.48); ap.add_argument('--tier3-score', type=float, default=0.58); ap.add_argument('--tier3-margin', type=float, default=0.0); ap.add_argument('--tier3-max-rank', type=int, default=3); ap.add_argument('--tier3-center-step', type=float, default=45.0); ap.add_argument('--tier3-area-ratio', type=float, default=3.0); ap.add_argument('--tier3-height-ratio', type=float, default=2.0); ap.add_argument('--tier3-motion', type=float, default=0.35); ap.add_argument('--tier3-size', type=float, default=0.40); ap.add_argument('--tier3-max-gap', type=int, default=80)
    ap.add_argument('--max-links-per-seq', type=int, default=999999)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    seqs = args.seqs or sorted(p.stem for p in Path(args.input_dir).glob('MOT20-*.txt'))
    summaries = []
    for seq in seqs:
        print(f'[alink_v2] seq={seq}', flush=True)
        summaries.append(process_seq(seq, args))
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summaries, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if args.summary_csv:
        fields = list(summaries[0].keys()) if summaries else ['seq']
        write_csv(Path(args.summary_csv), summaries, fields)
    print(json.dumps(summaries, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
