#!/usr/bin/env python3
"""Oracle upper bound for offline tracklet linking on MOT train.

Uses GT only to label each predicted tracklet by its majority matched GT identity,
then remaps non-overlapping tracklets with the same GT id into one output id.
This does not add/remove/change boxes; it estimates the ceiling of tracklet-level
ID linking under current detections/boxes.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class MotRow:
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


@dataclass
class TrackletInfo:
    seq: str
    tid: int
    start_frame: int
    end_frame: int
    length: int
    matched_count: int
    majority_gt: int
    majority_count: int
    purity: float
    match_frac: float
    eligible: int


@dataclass
class LinkEdge:
    seq: str
    source_tid: int
    target_tid: int
    gt_id: int
    gap: int
    source_purity: float
    target_purity: float
    source_match_frac: float
    target_match_frac: float
    selected: int = 0


def si(x: str, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def sf(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def load_pred(path: Path) -> Tuple[List[MotRow], Dict[int, List[MotRow]]]:
    rows: List[MotRow] = []
    tracks: Dict[int, List[MotRow]] = defaultdict(list)
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
            r = MotRow(fr, tid, x, y, w, h, score, list(p))
            rows.append(r); tracks[tid].append(r)
    for tid in list(tracks):
        tracks[tid].sort(key=lambda r: r.frame)
    return rows, tracks


def load_gt(path: Path) -> Dict[int, List[MotRow]]:
    by_frame: Dict[int, List[MotRow]] = defaultdict(list)
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(',')
            if len(p) < 6:
                continue
            fr = si(p[0]); gid = si(p[1])
            x, y, w, h = [sf(v) for v in p[2:6]]
            mark = si(p[6], 1) if len(p) > 6 else 1
            cls = si(p[7], 1) if len(p) > 7 else 1
            # TrackEval preprocessing for MOT Challenge primarily evaluates marked pedestrians.
            if mark <= 0 or cls != 1 or w <= 1.0 or h <= 1.0:
                continue
            by_frame[fr].append(MotRow(fr, gid, x, y, w, h, 1.0, list(p)))
    return by_frame


def pair_iou(preds: List[MotRow], gts: List[MotRow]) -> np.ndarray:
    if not preds or not gts:
        return np.zeros((len(preds), len(gts)), dtype=np.float32)
    p = np.stack([r.tlbr for r in preds], axis=0).astype(np.float32)
    g = np.stack([r.tlbr for r in gts], axis=0).astype(np.float32)
    xx1 = np.maximum(p[:, None, 0], g[None, :, 0])
    yy1 = np.maximum(p[:, None, 1], g[None, :, 1])
    xx2 = np.minimum(p[:, None, 2], g[None, :, 2])
    yy2 = np.minimum(p[:, None, 3], g[None, :, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    area_p = np.maximum(0, p[:, 2] - p[:, 0]) * np.maximum(0, p[:, 3] - p[:, 1])
    area_g = np.maximum(0, g[:, 2] - g[:, 0]) * np.maximum(0, g[:, 3] - g[:, 1])
    return inter / np.maximum(area_p[:, None] + area_g[None, :] - inter, 1e-12)


def match_track_rows_to_gt(rows: List[MotRow], gt_by_frame: Dict[int, List[MotRow]], iou_thr: float) -> Dict[Tuple[int, int, int], int]:
    """Return mapping key=(frame, tid, row_index_within_frame_tid_order) -> gt_id.

    The key is internal and generated deterministically while scanning rows grouped by frame.
    """
    by_frame: Dict[int, List[Tuple[int, MotRow]]] = defaultdict(list)
    for idx, r in enumerate(rows):
        by_frame[r.frame].append((idx, r))
    matched: Dict[int, int] = {}
    for fr, items in by_frame.items():
        preds = [r for _idx, r in items]
        gts = gt_by_frame.get(fr, [])
        if not preds or not gts:
            continue
        ious = pair_iou(preds, gts)
        cand = []
        pp, gg = np.where(ious >= iou_thr)
        for pi, gi in zip(pp.tolist(), gg.tolist()):
            cand.append((float(ious[pi, gi]), pi, gi))
        cand.sort(reverse=True)
        used_p, used_g = set(), set()
        for _iou, pi, gi in cand:
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi); used_g.add(gi)
            orig_idx = items[pi][0]
            matched[orig_idx] = gts[gi].tid
    return matched


def build_tracklet_infos(seq: str, rows: List[MotRow], tracks: Dict[int, List[MotRow]], matched_by_row_index: Dict[int, int], args) -> Dict[int, TrackletInfo]:
    row_to_idx = {id(r): idx for idx, r in enumerate(rows)}
    infos: Dict[int, TrackletInfo] = {}
    for tid, tr in tracks.items():
        gt_ids = []
        for r in tr:
            idx = row_to_idx[id(r)]
            if idx in matched_by_row_index:
                gt_ids.append(matched_by_row_index[idx])
        matched_count = len(gt_ids)
        if gt_ids:
            gt, count = Counter(gt_ids).most_common(1)[0]
        else:
            gt, count = -1, 0
        purity = count / matched_count if matched_count else 0.0
        match_frac = count / len(tr) if tr else 0.0
        eligible = int(
            len(tr) >= args.min_track_len and
            gt > 0 and
            purity >= args.min_purity and
            match_frac >= args.min_match_frac and
            count >= args.min_majority_count
        )
        infos[tid] = TrackletInfo(
            seq=seq,
            tid=tid,
            start_frame=int(tr[0].frame),
            end_frame=int(tr[-1].frame),
            length=len(tr),
            matched_count=matched_count,
            majority_gt=int(gt),
            majority_count=int(count),
            purity=float(purity),
            match_frac=float(match_frac),
            eligible=eligible,
        )
    return infos


def find(parent: Dict[int, int], x: int) -> int:
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def build_oracle_links(seq: str, infos: Dict[int, TrackletInfo], args) -> List[LinkEdge]:
    by_gt: Dict[int, List[TrackletInfo]] = defaultdict(list)
    for info in infos.values():
        if info.eligible:
            by_gt[info.majority_gt].append(info)
    candidates: List[LinkEdge] = []
    for gt, arr in by_gt.items():
        arr.sort(key=lambda x: (x.start_frame, x.end_frame, x.tid))
        for i, a in enumerate(arr):
            for b in arr[i + 1:]:
                if b.start_frame <= a.end_frame:
                    continue
                gap = b.start_frame - a.end_frame
                if gap > args.max_gap:
                    break
                candidates.append(LinkEdge(seq, a.tid, b.tid, gt, gap, a.purity, b.purity, a.match_frac, b.match_frac, selected=0))
    # Oracle but still structurally constrained: one successor and one predecessor, no cycles.
    candidates.sort(key=lambda e: (e.gap, -min(e.source_purity, e.target_purity), e.source_tid, e.target_tid))
    parent: Dict[int, int] = {}
    used_succ, used_pred = set(), set()
    selected: List[LinkEdge] = []
    for e in candidates:
        if e.source_tid in used_succ or e.target_tid in used_pred:
            continue
        ra = find(parent, e.source_tid); rb = find(parent, e.target_tid)
        if ra == rb:
            continue
        parent[rb] = ra
        used_succ.add(e.source_tid); used_pred.add(e.target_tid)
        e.selected = 1
        selected.append(e)
    return selected


def apply_links(input_file: Path, output_file: Path, selected: List[LinkEdge]) -> dict:
    parent: Dict[int, int] = {}
    for e in selected:
        ra = find(parent, e.source_tid); rb = find(parent, e.target_tid)
        parent[rb] = ra
    involved = set()
    for e in selected:
        involved.add(e.source_tid); involved.add(e.target_tid)
    id_map = {tid: find(parent, tid) for tid in involved}
    rows = []
    remapped = 0
    with input_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(',')
            tid = si(p[1]) if len(p) > 1 else 0
            if tid in id_map and id_map[tid] != tid:
                p[1] = str(id_map[tid]); remapped += 1
            rows.append(p)
    rows.sort(key=lambda p: (si(p[0]), si(p[1]), sf(p[2]), sf(p[3])))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', encoding='utf-8') as f:
        for p in rows:
            f.write(','.join(p) + '\n')
    return {'rows': len(rows), 'remapped_rows': remapped}


def write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def process_seq(seq: str, args) -> dict:
    input_file = Path(args.input_dir) / f'{seq}.txt'
    gt_file = Path(args.gt_root) / seq / 'gt' / 'gt.txt'
    rows, tracks = load_pred(input_file)
    gt_by_frame = load_gt(gt_file)
    matched = match_track_rows_to_gt(rows, gt_by_frame, args.iou_thr)
    infos = build_tracklet_infos(seq, rows, tracks, matched, args)
    links = build_oracle_links(seq, infos, args)
    out_file = Path(args.output_dir) / f'{seq}.txt'
    apply_stats = apply_links(input_file, out_file, links)
    detail_dir = Path(args.detail_dir) if args.detail_dir else Path(args.output_dir).parent / 'oracle_details'
    if args.write_details:
        write_csv(detail_dir / f'{seq}_tracklet_labels.csv', [asdict(v) for v in infos.values()], list(asdict(next(iter(infos.values()))).keys()) if infos else ['seq'])
        write_csv(detail_dir / f'{seq}_selected_links.csv', [asdict(e) for e in links], list(asdict(links[0]).keys()) if links else list(LinkEdge(seq,0,0,0,0,0,0,0,0).__dict__.keys()))
    eligible = sum(v.eligible for v in infos.values())
    return {
        'seq': seq,
        'input_tracks': len(tracks),
        'eligible_tracklets': int(eligible),
        'matched_rows': len(matched),
        'selected_links': len(links),
        **apply_stats,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--gt-root', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--seqs', nargs='*', default=[])
    ap.add_argument('--iou-thr', type=float, default=0.5)
    ap.add_argument('--max-gap', type=int, default=300)
    ap.add_argument('--min-track-len', type=int, default=2)
    ap.add_argument('--min-purity', type=float, default=0.60)
    ap.add_argument('--min-match-frac', type=float, default=0.20)
    ap.add_argument('--min-majority-count', type=int, default=2)
    ap.add_argument('--detail-dir', default='')
    ap.add_argument('--write-details', action='store_true')
    ap.add_argument('--summary-json', default='')
    ap.add_argument('--summary-csv', default='')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    seqs = args.seqs or sorted(p.stem for p in in_dir.glob('MOT20-*.txt'))
    summaries = []
    for seq in seqs:
        print(f'[oracle_link] seq={seq}', flush=True)
        summaries.append(process_seq(seq, args))
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summaries, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if args.summary_csv:
        fields = list(summaries[0].keys()) if summaries else ['seq']
        write_csv(Path(args.summary_csv), summaries, fields)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
