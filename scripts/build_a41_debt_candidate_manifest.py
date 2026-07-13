#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def af(v, d=0.0):
    try:
        if v is None or v == '':
            return d
        return float(v)
    except Exception:
        return d


def ai(v, d=0):
    try:
        if v is None or v == '':
            return d
        return int(float(v))
    except Exception:
        return d


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    if not fields:
        fields = ['seq']
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def make_tracklet_index(rows: List[dict]) -> Dict[Tuple[str, str], dict]:
    idx = {}
    for r in rows:
        seq = str(r.get('seq', ''))
        tid = str(r.get('track_id', ''))
        if seq and tid:
            idx[(seq, tid)] = r
    return idx


def tracklet_debt_tags(t: dict, side: str) -> List[str]:
    # Uses only tracker-output statistics. Train-only GT diagnostic columns are not used.
    tags: List[str] = []
    row_count = af(t.get('row_count'))
    duration = af(t.get('duration'), row_count)
    avg_score = af(t.get('avg_score'), 1.0)
    first_score = af(t.get('first_score'), 1.0)
    last_score = af(t.get('last_score'), 1.0)
    vx_start, vy_start = af(t.get('vx_start')), af(t.get('vy_start'))
    vx_end, vy_end = af(t.get('vx_end')), af(t.get('vy_end'))
    if row_count < 30:
        tags.append('short_tracklet')
    if row_count < 10:
        tags.append('very_short_tracklet')
    if avg_score < 0.75:
        tags.append('low_avg_score')
    if side == 'source' and last_score < 0.35:
        tags.append('weak_end_boundary')
    if side == 'target' and first_score < 0.35:
        tags.append('weak_start_boundary')
    if duration > 0 and row_count / duration < 0.85:
        tags.append('internal_gaps')
    if abs(vx_start - vx_end) + abs(vy_start - vy_end) > 20:
        tags.append('motion_instability')
    return tags


def edge_type(row: dict, source_tags: List[str], target_tags: List[str]) -> str:
    gap = af(row.get('gap'))
    predicted_pf = af(row.get('predicted_distance_per_frame'))
    height_ratio = af(row.get('height_ratio'), 1.0)
    area_ratio = af(row.get('area_ratio'), 1.0)
    tags = set(source_tags + target_tags)
    if 'weak_end_boundary' in tags or 'weak_start_boundary' in tags:
        return 'weak_boundary_recovery'
    if 'short_tracklet' in tags or 'very_short_tracklet' in tags:
        return 'fragmented_tracklet_recovery'
    if predicted_pf > 25 or not (0.55 <= height_ratio <= 1.8) or not (0.30 <= area_ratio <= 3.2):
        return 'high_risk_geometry'
    if gap <= 30:
        return 'short_gap_continuation'
    if gap <= 150:
        return 'long_gap_reappearance'
    return 'very_long_gap_candidate'


def add_group_rank_fields(rows: List[dict], score_col: str, prefix: str, key_col: str) -> None:
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r.get('seq'), r.get(key_col))].append((i, af(r.get(score_col))))
    for _key, items in groups.items():
        items = sorted(items, key=lambda x: x[1], reverse=True)
        for rank, (idx, score) in enumerate(items, start=1):
            second = items[1][1] if rank == 1 and len(items) > 1 else (items[0][1] if rank != 1 else 0.0)
            margin = score - second if rank == 1 else score - items[0][1]
            rows[idx][f'{prefix}_rank_by_{score_col}'] = rank
            rows[idx][f'{prefix}_best_{score_col}'] = items[0][1]
            rows[idx][f'{prefix}_second_{score_col}'] = items[1][1] if len(items) > 1 else 0.0
            rows[idx][f'{prefix}_margin_to_second_{score_col}'] = margin
            rows[idx][f'{prefix}_group_size'] = len(items)


def build_group_status(rows: List[dict], score_col='aflink_score') -> List[dict]:
    out = []
    for group_type, key_col in [('source', 'track_a'), ('target', 'track_b')]:
        groups = defaultdict(list)
        for r in rows:
            groups[(r.get('seq'), r.get(key_col))].append(r)
        for (seq, tid), rs in groups.items():
            rs = sorted(rs, key=lambda r: af(r.get(score_col)), reverse=True)
            best = rs[0]
            out.append({
                'seq': seq,
                'group_type': group_type,
                'track_id': tid,
                'candidate_count': len(rs),
                'best_edge': f"{best.get('track_a')}->{best.get('track_b')}",
                'best_score': af(best.get(score_col)),
                'second_score': af(rs[1].get(score_col)) if len(rs) > 1 else 0.0,
                'margin_to_second': af(best.get(score_col)) - (af(rs[1].get(score_col)) if len(rs) > 1 else 0.0),
                'best_debt_adjusted_edge_score': af(best.get('debt_adjusted_edge_score')),
                'best_edge_type': best.get('edge_type'),
                'best_source_debt_tags': best.get('source_debt_tags'),
                'best_target_debt_tags': best.get('target_debt_tags'),
                'solver_ready_edges': sum(ai(r.get('solver_ready_v1')) for r in rs),
                'old_thr015_edges': sum(ai(r.get('old_thr015_pool')) for r in rs),
            })
    return out


def risk_and_score(row: dict) -> None:
    aflink = af(row.get('aflink_score'))
    appearance_max = af(row.get('appearance_max'))
    appearance_std = af(row.get('appearance_std'))
    appearance_gap_consistency = af(row.get('appearance_gap_consistency'))
    height_ratio = af(row.get('height_ratio'), 1.0)
    area_ratio = af(row.get('area_ratio'), 1.0)
    bottom_y_gap = abs(af(row.get('bottom_y_gap')))
    predicted_pf = af(row.get('predicted_distance_per_frame'))
    center_pf = af(row.get('center_distance_per_frame'))
    velocity_cos = af(row.get('velocity_cosine'))

    geometry_risk = 0
    if not (0.65 <= height_ratio <= 1.55):
        geometry_risk += 1
    if not (0.45 <= area_ratio <= 2.40):
        geometry_risk += 1
    if bottom_y_gap > 260:
        geometry_risk += 1

    motion_risk = 0
    if predicted_pf > 18:
        motion_risk += 1
    if center_pf > 18:
        motion_risk += 1
    if velocity_cos < -0.35:
        motion_risk += 1

    out_rank = ai(row.get('out_rank_by_aflink_score'), 999)
    in_rank = ai(row.get('in_rank_by_aflink_score'), 999)
    out_margin = af(row.get('out_margin_to_second_aflink_score'))
    in_margin = af(row.get('in_margin_to_second_aflink_score'))
    competition_risk = 0
    if out_rank > 1:
        competition_risk += 1
    if in_rank > 1:
        competition_risk += 1
    if out_margin < 0.02:
        competition_risk += 1
    if in_margin < 0.02:
        competition_risk += 1

    appearance_instability_risk = 0
    if appearance_std > 0.18:
        appearance_instability_risk += 1
    if appearance_gap_consistency < 0.05:
        appearance_instability_risk += 1

    source_debt = ai(row.get('source_debt_count'))
    target_debt = ai(row.get('target_debt_count'))
    edge_debt = source_debt + target_debt
    debt_bonus = min(0.20, 0.035 * edge_debt)
    rank_bonus = 0.0
    if out_rank == 1 and in_rank == 1:
        rank_bonus = 0.08
    elif out_rank <= 2 and in_rank <= 2:
        rank_bonus = 0.035
    elif out_rank == 1 or in_rank == 1:
        rank_bonus = 0.02
    margin_bonus = max(0.0, min(0.08, 0.35 * min(max(out_margin, 0.0), max(in_margin, 0.0))))
    risk_penalty = 0.045 * geometry_risk + 0.040 * motion_risk + 0.055 * competition_risk + 0.030 * appearance_instability_risk
    adjusted = aflink + debt_bonus + rank_bonus + margin_bonus - risk_penalty

    row['geometry_risk'] = geometry_risk
    row['motion_risk'] = motion_risk
    row['competition_risk'] = competition_risk
    row['appearance_instability_risk'] = appearance_instability_risk
    row['risk_total'] = geometry_risk + motion_risk + competition_risk + appearance_instability_risk
    row['debt_adjusted_edge_score'] = adjusted
    row['old_thr015_pool'] = int(aflink >= 0.15)
    row['old_thr020_pool'] = int(aflink >= 0.20)
    row['old_thr030_pool'] = int(aflink >= 0.30)
    row['debt_gated_pool'] = int(edge_debt >= 1 and aflink >= 0.08)
    row['rank_margin_gated_pool'] = int(out_rank == 1 and in_rank == 1 and min(out_margin, in_margin) >= 0.02 and aflink >= 0.05)
    # Solver-ready is deliberately more conservative than old threshold apply.
    row['solver_ready_v1'] = int(
        edge_debt >= 1
        and out_rank <= 2
        and in_rank <= 2
        and geometry_risk <= 1
        and motion_risk <= 1
        and competition_risk <= 2
        and (
            aflink >= 0.15
            or (aflink >= 0.08 and appearance_max >= 0.72)
            or adjusted >= 0.18
        )
    )
    row['solver_ready_strict_v1'] = int(
        row['solver_ready_v1']
        and out_rank == 1
        and in_rank == 1
        and min(out_margin, in_margin) >= 0.03
        and aflink >= 0.15
        and row['risk_total'] <= 2
    )


def build_manifest(pair_path: Path, tracklet_path: Path, output_csv: Path, group_csv: Path, split_name: str) -> dict:
    pairs = read_csv(pair_path)
    tracklets = read_csv(tracklet_path)
    tidx = make_tracklet_index(tracklets)
    rows = []
    missing_tracklet = 0
    for r in pairs:
        seq = str(r.get('seq'))
        a = str(r.get('track_a'))
        b = str(r.get('track_b'))
        ta = tidx.get((seq, a), {})
        tb = tidx.get((seq, b), {})
        if not ta or not tb:
            missing_tracklet += 1
        stags = tracklet_debt_tags(ta, 'source') if ta else []
        ttags = tracklet_debt_tags(tb, 'target') if tb else []
        q = dict(r)
        q['split_name'] = split_name
        q['source_debt_tags'] = '|'.join(stags)
        q['target_debt_tags'] = '|'.join(ttags)
        q['source_debt_count'] = len(stags)
        q['target_debt_count'] = len(ttags)
        q['edge_debt_score'] = len(stags) + len(ttags)
        q['edge_type'] = edge_type(q, stags, ttags)
        rows.append(q)

    # Rank by model score and appearance score in source/target competition groups.
    for score_col in ['aflink_score', 'appearance_max']:
        add_group_rank_fields(rows, score_col, 'out', 'track_a')
        add_group_rank_fields(rows, score_col, 'in', 'track_b')
    for r in rows:
        risk_and_score(r)
    rows = sorted(rows, key=lambda r: af(r.get('debt_adjusted_edge_score')), reverse=True)
    for i, r in enumerate(rows, start=1):
        r['debt_adjusted_rank_global'] = i
    write_csv(output_csv, rows)
    group_rows = build_group_status(rows)
    write_csv(group_csv, group_rows)
    by_seq = defaultdict(list)
    for r in rows:
        by_seq[r.get('seq')].append(r)
    seq_summary = {}
    for seq, rs in sorted(by_seq.items()):
        seq_summary[seq] = {
            'edges': len(rs),
            'old_thr015_pool': sum(ai(r.get('old_thr015_pool')) for r in rs),
            'old_thr020_pool': sum(ai(r.get('old_thr020_pool')) for r in rs),
            'old_thr030_pool': sum(ai(r.get('old_thr030_pool')) for r in rs),
            'debt_gated_pool': sum(ai(r.get('debt_gated_pool')) for r in rs),
            'rank_margin_gated_pool': sum(ai(r.get('rank_margin_gated_pool')) for r in rs),
            'solver_ready_v1': sum(ai(r.get('solver_ready_v1')) for r in rs),
            'solver_ready_strict_v1': sum(ai(r.get('solver_ready_strict_v1')) for r in rs),
            'edge_type_counts': dict(Counter(r.get('edge_type') for r in rs)),
        }
        if 'same_gt' in rs[0]:
            for flag in ['old_thr015_pool', 'solver_ready_v1', 'solver_ready_strict_v1']:
                sel = [r for r in rs if ai(r.get(flag))]
                tp = sum(str(r.get('same_gt')) == '1' for r in sel)
                seq_summary[seq][f'{flag}_label_precision'] = tp / len(sel) if sel else 0.0
                seq_summary[seq][f'{flag}_label_tp'] = tp
    summary = {
        'split_name': split_name,
        'pair_path': str(pair_path),
        'tracklet_path': str(tracklet_path),
        'edges': len(rows),
        'tracklets': len(tracklets),
        'missing_tracklet_pairs': missing_tracklet,
        'old_thr015_pool': sum(ai(r.get('old_thr015_pool')) for r in rows),
        'old_thr020_pool': sum(ai(r.get('old_thr020_pool')) for r in rows),
        'old_thr030_pool': sum(ai(r.get('old_thr030_pool')) for r in rows),
        'debt_gated_pool': sum(ai(r.get('debt_gated_pool')) for r in rows),
        'rank_margin_gated_pool': sum(ai(r.get('rank_margin_gated_pool')) for r in rows),
        'solver_ready_v1': sum(ai(r.get('solver_ready_v1')) for r in rows),
        'solver_ready_strict_v1': sum(ai(r.get('solver_ready_strict_v1')) for r in rows),
        'edge_type_counts': dict(Counter(r.get('edge_type') for r in rows)),
        'source_debt_tag_counts': dict(Counter(tag for r in rows for tag in r.get('source_debt_tags','').split('|') if tag)),
        'target_debt_tag_counts': dict(Counter(tag for r in rows for tag in r.get('target_debt_tags','').split('|') if tag)),
        'by_seq': seq_summary,
    }
    if rows and 'same_gt' in rows[0]:
        positives = sum(str(r.get('same_gt')) == '1' for r in rows)
        summary['label_positive_edges'] = positives
        summary['label_positive_rate'] = positives / len(rows) if rows else 0.0
        for flag in ['old_thr015_pool', 'old_thr020_pool', 'old_thr030_pool', 'debt_gated_pool', 'rank_margin_gated_pool', 'solver_ready_v1', 'solver_ready_strict_v1']:
            sel = [r for r in rows if ai(r.get(flag))]
            tp = sum(str(r.get('same_gt')) == '1' for r in sel)
            summary[f'{flag}_label_tp'] = tp
            summary[f'{flag}_label_precision'] = tp / len(sel) if sel else 0.0
            summary[f'{flag}_label_recall'] = tp / positives if positives else 0.0
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-pairs', required=True)
    ap.add_argument('--train-tracklets', required=True)
    ap.add_argument('--test-pairs', required=True)
    ap.add_argument('--test-tracklets', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_summary = build_manifest(
        Path(args.train_pairs), Path(args.train_tracklets),
        out / 'debt_edge_candidates_train_oof.csv', out / 'debt_edge_group_status_train.csv', 'train_oof')
    test_summary = build_manifest(
        Path(args.test_pairs), Path(args.test_tracklets),
        out / 'debt_edge_candidates_test.csv', out / 'debt_edge_group_status_test.csv', 'test')
    summary = {'train_oof': train_summary, 'test': test_summary, 'decision': 'A41_01_MANIFEST_BUILT_READY_FOR_SOLVER_V1'}
    (out / 'debt_candidate_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    md = [
        '# A41_01 Debt-Aware Candidate Manifest', '',
        '## Summary', '', '```json', json.dumps(summary, indent=2, sort_keys=True)[:9000], '```', '',
        '## Decision', '', '```text', 'A41_01 = MANIFEST_BUILT_READY_FOR_SOLVER_V1', 'Next = A41_02_global_solver_v1', '```', '',
        'The manifest does not use test GT. Train OOF labels are retained only for diagnostic precision/recall estimates.'
    ]
    (out / 'decision.md').write_text('\n'.join(md) + '\n')
    print(json.dumps({
        'out': str(out),
        'train_edges': train_summary['edges'],
        'test_edges': test_summary['edges'],
        'train_solver_ready_v1': train_summary['solver_ready_v1'],
        'test_solver_ready_v1': test_summary['solver_ready_v1'],
        'train_solver_ready_precision': train_summary.get('solver_ready_v1_label_precision'),
        'decision': summary['decision'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
