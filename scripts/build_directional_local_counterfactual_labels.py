from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from eval_assa_swap_merge_fusion import read_track_rows


SEQUENCES = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']
EVENT_KEYS = ['seq', 'canonical_rank', 'u', 'v', 'boundary_frame', 'transaction_type']
HORIZONS: list[tuple[str, int | None]] = [
    ('h30', 30),
    ('h60', 60),
    ('h120', 120),
    ('h300', 300),
    ('full', None),
]
LOCAL_METRIC_SUFFIXES = [
    'pairwise_net_norm',
    'idtp_delta_norm',
    'row_net',
    'donor_minus_receiver_support_frac',
]
GLOBAL_TARGETS = ['delta_AssA', 'delta_HOTA', 'delta_IDF1']
PASS_THRESHOLDS = {
    'pooled_spearman_delta_AssA': 0.50,
    'positive_hota_auc': 0.75,
    'positive_sequence_spearman_count': 3,
    'window_topone_utility_sum_strictly_positive': True,
    'window_topone_worst_sequence_nonnegative': True,
    'window_topone_positive_windows_min': 8,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rounded(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=[np.number]).columns:
        result[column] = result[column].round(12)
    return result


def load_baseline_records(path: Path) -> list[dict[str, Any]]:
    rows, _, _ = read_track_rows(path)
    records: list[dict[str, Any]] = []
    for source_index, parts in enumerate(rows):
        frame = int(float(parts[0]))
        track_id = int(float(parts[1]))
        records.append({
            'frame': frame,
            'raw_tid': track_id,
            'label': track_id,
            'parts': list(parts),
            'source_index': source_index,
        })
    records.sort(key=lambda record: (record['frame'], record['label']))
    return records


def record_geometry_key(parts: list[str]) -> tuple[str, ...]:
    """Identify one MOT row without using its identity label."""
    if len(parts) < 6:
        raise RuntimeError(f'invalid MOT row with {len(parts)} columns')
    return tuple([parts[0], *parts[2:]])


def build_geometry_index(records: list[dict[str, Any]]) -> dict[tuple[str, ...], int]:
    result: dict[tuple[str, ...], int] = {}
    for index, record in enumerate(records):
        key = record_geometry_key(record['parts'])
        if key in result:
            raise RuntimeError(f'duplicate baseline geometry key at rows {result[key]} and {index}')
        result[key] = index
    return result


def recover_changed_rows_from_counterfactual(
    edited_path: Path,
    records: list[dict[str, Any]],
    geometry_index: dict[tuple[str, ...], int],
    expected_sha256: str,
) -> tuple[list[int], int, int, dict[str, Any]]:
    actual_sha256 = sha256(edited_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f'counterfactual SHA mismatch for {edited_path}: '
            f'expected={expected_sha256}, actual={actual_sha256}'
        )

    seen = bytearray(len(records))
    changed_indices: list[int] = []
    identity_pairs: Counter[tuple[int, int]] = Counter()
    row_count = 0
    with edited_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip('\n').split(',')
            if len(parts) < 6:
                continue
            key = record_geometry_key(parts)
            baseline_index = geometry_index.get(key)
            if baseline_index is None:
                raise RuntimeError(
                    f'edited geometry absent from baseline: {edited_path}:{line_number}'
                )
            if seen[baseline_index]:
                raise RuntimeError(
                    f'duplicate edited geometry key: {edited_path}:{line_number}'
                )
            seen[baseline_index] = 1
            row_count += 1
            baseline_label = int(records[baseline_index]['label'])
            edited_label = int(float(parts[1]))
            if baseline_label != edited_label:
                changed_indices.append(baseline_index)
                identity_pairs[(baseline_label, edited_label)] += 1

    if row_count != len(records) or sum(seen) != len(records):
        raise RuntimeError(
            f'counterfactual geometry coverage mismatch for {edited_path}: '
            f'edited={row_count}, baseline={len(records)}, matched={sum(seen)}'
        )
    donor_anchors = {edited_label for _, edited_label in identity_pairs}
    if len(donor_anchors) != 1:
        raise RuntimeError(
            f'expected one edited donor anchor for {edited_path}, found {identity_pairs}'
        )
    donor_anchor = next(iter(donor_anchors))
    pair_rows = sum(identity_pairs.values())
    if pair_rows != len(changed_indices):
        raise RuntimeError(f'changed-row accounting mismatch for {edited_path}')
    receiver_anchor_counts = Counter({
        receiver_anchor: count
        for (receiver_anchor, edited_anchor), count in identity_pairs.items()
        if edited_anchor == donor_anchor
    })
    receiver_anchor = receiver_anchor_counts.most_common(1)[0][0]
    audit = {
        'edited_path': str(edited_path),
        'edited_sha256': actual_sha256,
        'geometry_rows': row_count,
        'changed_rows': len(changed_indices),
        'receiver_anchor': receiver_anchor,
        'receiver_anchor_count': len(receiver_anchor_counts),
        'receiver_anchor_counts': json.dumps(
            {str(key): value for key, value in sorted(receiver_anchor_counts.items())},
            sort_keys=True,
        ),
        'donor_anchor': donor_anchor,
    }
    return changed_indices, donor_anchor, receiver_anchor_counts, audit


def load_gt(path: Path) -> pd.DataFrame:
    columns = ['frame', 'gt_id', 'x', 'y', 'w', 'h', 'mark', 'class_id', 'visibility']
    gt = pd.read_csv(path, header=None, names=columns)
    gt = gt[(gt.mark == 1) & (gt.class_id == 1)].copy()
    gt[['frame', 'gt_id']] = gt[['frame', 'gt_id']].astype(int)
    return gt.reset_index(drop=True)


def boxes_from_records(records: list[dict[str, Any]], indices: list[int]) -> np.ndarray:
    if not indices:
        return np.empty((0, 4), dtype=float)
    return np.asarray([
        [
            float(records[index]['parts'][2]),
            float(records[index]['parts'][3]),
            float(records[index]['parts'][4]),
            float(records[index]['parts'][5]),
        ]
        for index in indices
    ], dtype=float)


def xywh_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=float)
    left_xy2 = left[:, :2] + np.maximum(left[:, 2:], 0.0)
    right_xy2 = right[:, :2] + np.maximum(right[:, 2:], 0.0)
    inter_xy1 = np.maximum(left[:, None, :2], right[None, :, :2])
    inter_xy2 = np.minimum(left_xy2[:, None, :], right_xy2[None, :, :])
    inter_wh = np.maximum(inter_xy2 - inter_xy1, 0.0)
    intersection = inter_wh[..., 0] * inter_wh[..., 1]
    left_area = np.maximum(left[:, 2], 0.0) * np.maximum(left[:, 3], 0.0)
    right_area = np.maximum(right[:, 2], 0.0) * np.maximum(right[:, 3], 0.0)
    union = left_area[:, None] + right_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_records_to_gt(
    records: list[dict[str, Any]],
    gt: pd.DataFrame,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    gt_ids = np.full(len(records), -1, dtype=np.int64)
    gt_ious = np.zeros(len(records), dtype=float)
    record_by_frame: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        record_by_frame[int(record['frame'])].append(index)
    gt_by_frame = {int(frame): group for frame, group in gt.groupby('frame', sort=False)}

    matched = 0
    evaluated_frames = 0
    for frame, indices in record_by_frame.items():
        gt_frame = gt_by_frame.get(frame)
        if gt_frame is None or not len(gt_frame):
            continue
        evaluated_frames += 1
        tracker_boxes = boxes_from_records(records, indices)
        truth_boxes = gt_frame[['x', 'y', 'w', 'h']].to_numpy(float)
        similarities = xywh_iou(tracker_boxes, truth_boxes)
        row_indices, column_indices = linear_sum_assignment(1.0 - similarities)
        truth_ids = gt_frame.gt_id.to_numpy(np.int64)
        for local_row, local_column in zip(row_indices, column_indices):
            iou = float(similarities[local_row, local_column])
            if iou < iou_threshold:
                continue
            record_index = indices[int(local_row)]
            gt_ids[record_index] = int(truth_ids[int(local_column)])
            gt_ious[record_index] = iou
            matched += 1
    summary = {
        'tracker_rows': len(records),
        'matched_tracker_rows': matched,
        'evaluated_frames': evaluated_frames,
    }
    return gt_ids, gt_ious, summary


def matched_counter(indices: list[int], matched_gt_ids: np.ndarray) -> Counter[int]:
    return Counter(int(matched_gt_ids[index]) for index in indices if matched_gt_ids[index] >= 0)


def dominant_identity(counts: Counter[int]) -> tuple[int, float]:
    if not counts:
        return -1, math.nan
    identity, support = max(counts.items(), key=lambda item: (item[1], -item[0]))
    return int(identity), float(support / sum(counts.values()))


def entropy(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return math.nan
    probabilities = np.asarray(list(counts.values()), dtype=float) / total
    return float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())


def dot_counts(left: Counter[int], right: Counter[int]) -> int:
    return int(sum(value * right.get(identity, 0) for identity, value in left.items()))


def assignment_idtp(groups: list[Counter[int]]) -> int:
    identities = sorted({identity for group in groups for identity in group})
    if not identities:
        return 0
    matrix = np.asarray([
        [group.get(identity, 0) for identity in identities]
        for group in groups
    ], dtype=float)
    rows, columns = linear_sum_assignment(-matrix)
    return int(matrix[rows, columns].sum())


def horizon_indices(
    records: list[dict[str, Any]],
    candidate_indices: list[int],
    start_frame: int,
    horizon: int | None,
    history: bool,
) -> list[int]:
    if history:
        lower = -10**12 if horizon is None else start_frame - horizon
        return [
            index for index in candidate_indices
            if lower <= int(records[index]['frame']) < start_frame
        ]
    upper = 10**12 if horizon is None else start_frame + horizon - 1
    return [
        index for index in candidate_indices
        if start_frame <= int(records[index]['frame']) <= upper
    ]


def compute_horizon_metrics(
    records: list[dict[str, Any]],
    matched_gt_ids: np.ndarray,
    gt_ious: np.ndarray,
    donor_history_all: list[int],
    receiver_history_by_anchor: dict[int, list[int]],
    future_by_anchor: dict[int, list[int]],
    start_frame: int,
    horizon: int | None,
) -> tuple[dict[str, float | int], dict[str, int | float]]:
    donor_indices = horizon_indices(
        records, donor_history_all, start_frame, horizon, history=True
    )
    receiver_indices_by_anchor = {
        anchor: horizon_indices(
            records, indices, start_frame, horizon, history=True
        )
        for anchor, indices in receiver_history_by_anchor.items()
    }
    future_indices_by_anchor = {
        anchor: horizon_indices(
            records, indices, start_frame, horizon, history=False
        )
        for anchor, indices in future_by_anchor.items()
    }
    receiver_indices = [
        index
        for indices in receiver_indices_by_anchor.values()
        for index in indices
    ]
    future_indices = [
        index
        for indices in future_indices_by_anchor.values()
        for index in indices
    ]

    donor_counts = matched_counter(donor_indices, matched_gt_ids)
    receiver_counts_by_anchor = {
        anchor: matched_counter(indices, matched_gt_ids)
        for anchor, indices in receiver_indices_by_anchor.items()
    }
    future_counts_by_anchor = {
        anchor: matched_counter(indices, matched_gt_ids)
        for anchor, indices in future_indices_by_anchor.items()
    }
    receiver_counts = sum(receiver_counts_by_anchor.values(), Counter())
    future_counts = sum(future_counts_by_anchor.values(), Counter())
    donor_identity, donor_purity = dominant_identity(donor_counts)
    receiver_identity, receiver_purity = dominant_identity(receiver_counts)
    future_identity, future_purity = dominant_identity(future_counts)
    receiver_identity_by_anchor = {
        anchor: dominant_identity(counts)[0]
        for anchor, counts in receiver_counts_by_anchor.items()
    }

    future_matched = sum(future_counts.values())
    donor_support = 0
    receiver_support = 0
    benefit_rows = 0
    harm_rows = 0
    receiver_valid_identities: list[int] = []
    for anchor, future_anchor_counts in future_counts_by_anchor.items():
        receiver_anchor_identity = receiver_identity_by_anchor.get(anchor, -1)
        if receiver_anchor_identity >= 0:
            receiver_valid_identities.append(receiver_anchor_identity)
        donor_anchor_support = (
            future_anchor_counts.get(donor_identity, 0) if donor_identity >= 0 else 0
        )
        receiver_anchor_support = (
            future_anchor_counts.get(receiver_anchor_identity, 0)
            if receiver_anchor_identity >= 0 else 0
        )
        donor_support += donor_anchor_support
        receiver_support += receiver_anchor_support
        if (
            donor_identity >= 0
            and receiver_anchor_identity >= 0
            and donor_identity != receiver_anchor_identity
        ):
            benefit_rows += donor_anchor_support
            harm_rows += receiver_anchor_support
    row_net = (benefit_rows - harm_rows) / max(future_matched, 1)

    donor_total_pairs = 0
    receiver_total_pairs = 0
    donor_same_pairs = 0
    receiver_same_pairs = 0
    donor_different_pairs = 0
    receiver_different_pairs = 0
    baseline_groups = [donor_counts]
    edited_groups = [donor_counts + future_counts]
    for anchor, future_anchor_counts in future_counts_by_anchor.items():
        receiver_anchor_counts = receiver_counts_by_anchor.get(anchor, Counter())
        future_anchor_matched = sum(future_anchor_counts.values())
        donor_anchor_total_pairs = sum(donor_counts.values()) * future_anchor_matched
        receiver_anchor_total_pairs = (
            sum(receiver_anchor_counts.values()) * future_anchor_matched
        )
        donor_anchor_same_pairs = dot_counts(donor_counts, future_anchor_counts)
        receiver_anchor_same_pairs = dot_counts(
            receiver_anchor_counts, future_anchor_counts
        )
        donor_total_pairs += donor_anchor_total_pairs
        receiver_total_pairs += receiver_anchor_total_pairs
        donor_same_pairs += donor_anchor_same_pairs
        receiver_same_pairs += receiver_anchor_same_pairs
        donor_different_pairs += donor_anchor_total_pairs - donor_anchor_same_pairs
        receiver_different_pairs += (
            receiver_anchor_total_pairs - receiver_anchor_same_pairs
        )
        baseline_groups.append(receiver_anchor_counts + future_anchor_counts)
        edited_groups.append(receiver_anchor_counts)
    pairwise_net = (
        donor_same_pairs
        - donor_different_pairs
        + receiver_different_pairs
        - receiver_same_pairs
    )
    affected_pairs = donor_total_pairs + receiver_total_pairs
    pairwise_net_norm = pairwise_net / max(affected_pairs, 1)

    baseline_idtp = assignment_idtp(baseline_groups)
    edited_idtp = assignment_idtp(edited_groups)
    total_matched = (
        sum(donor_counts.values())
        + sum(receiver_counts.values())
        + future_matched
    )
    idtp_delta = edited_idtp - baseline_idtp
    idtp_delta_norm = idtp_delta / max(total_matched, 1)

    matched_future_indices = [index for index in future_indices if matched_gt_ids[index] >= 0]
    mean_future_iou = (
        float(np.mean(gt_ious[matched_future_indices])) if matched_future_indices else math.nan
    )
    future_frames = [int(records[index]['frame']) for index in future_indices]
    metrics: dict[str, float | int] = {
        'history_donor_rows': len(donor_indices),
        'history_receiver_rows': len(receiver_indices),
        'receiver_anchor_count': len(receiver_indices_by_anchor),
        'future_rows': len(future_indices),
        'history_donor_matched': sum(donor_counts.values()),
        'history_receiver_matched': sum(receiver_counts.values()),
        'future_matched': future_matched,
        'future_match_rate': future_matched / max(len(future_indices), 1),
        'future_mean_match_iou': mean_future_iou,
        'donor_dominant_gt': donor_identity,
        'receiver_dominant_gt': receiver_identity,
        'future_dominant_gt': future_identity,
        'donor_gt_purity': donor_purity,
        'receiver_gt_purity': receiver_purity,
        'future_gt_purity': future_purity,
        'future_gt_entropy': entropy(future_counts),
        'future_unique_gt': len(future_counts),
        'history_dominant_same_gt': int(
            donor_identity >= 0
            and bool(receiver_valid_identities)
            and all(identity == donor_identity for identity in receiver_valid_identities)
        ),
        'donor_future_support_rows': donor_support,
        'receiver_future_support_rows': receiver_support,
        'benefit_rows': benefit_rows,
        'harm_rows': harm_rows,
        'row_net': row_net,
        'donor_minus_receiver_support_frac': (
            donor_support - receiver_support
        ) / max(future_matched, 1),
        'donor_same_pairs': donor_same_pairs,
        'donor_different_pairs': donor_different_pairs,
        'receiver_same_pairs': receiver_same_pairs,
        'receiver_different_pairs': receiver_different_pairs,
        'pairwise_net_pairs': pairwise_net,
        'pairwise_affected_pairs': affected_pairs,
        'pairwise_net_norm': pairwise_net_norm,
        'baseline_local_idtp': baseline_idtp,
        'edited_local_idtp': edited_idtp,
        'idtp_delta': idtp_delta,
        'idtp_delta_norm': idtp_delta_norm,
        'future_span_frames': (
            max(future_frames) - min(future_frames) + 1 if future_frames else 0
        ),
        'future_density': (
            len(future_indices)
            / max(max(future_frames) - min(future_frames) + 1, 1)
            if future_frames else 0.0
        ),
    }
    identities = {
        'donor_identity': donor_identity,
        'receiver_identity': receiver_identity,
        'future_identity': future_identity,
        'receiver_identity_by_anchor': receiver_identity_by_anchor,
    }
    return metrics, identities


def safe_correlation(left: pd.Series, right: pd.Series, method: str) -> tuple[float, float, int]:
    frame = pd.DataFrame({'left': left, 'right': right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame.left.nunique() < 2 or frame.right.nunique() < 2:
        return math.nan, math.nan, len(frame)
    if method == 'pearson':
        statistic, pvalue = pearsonr(frame.left, frame.right)
    elif method == 'spearman':
        statistic, pvalue = spearmanr(frame.left, frame.right)
    else:
        raise ValueError(method)
    return float(statistic), float(pvalue), len(frame)


def metric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column for column in frame.columns
        if any(column.endswith(suffix) for suffix in LOCAL_METRIC_SUFFIXES)
    ]


def correlation_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    positive_target = (frame.delta_HOTA > 0).astype(int)
    for metric in metric_columns(frame):
        for target in GLOBAL_TARGETS:
            pearson, pearson_p, pearson_n = safe_correlation(frame[metric], frame[target], 'pearson')
            spearman, spearman_p, spearman_n = safe_correlation(frame[metric], frame[target], 'spearman')
            rows.append({
                'metric': metric,
                'target': target,
                'pearson': pearson,
                'pearson_p': pearson_p,
                'pearson_n': pearson_n,
                'spearman': spearman,
                'spearman_p': spearman_p,
                'spearman_n': spearman_n,
            })
        valid = frame[[metric]].replace([np.inf, -np.inf], np.nan).dropna().index
        auc = math.nan
        if len(valid) >= 3 and positive_target.loc[valid].nunique() == 2:
            auc = float(roc_auc_score(positive_target.loc[valid], frame.loc[valid, metric]))
        rows.append({
            'metric': metric,
            'target': 'positive_delta_HOTA',
            'roc_auc': auc,
            'auc_n': len(valid),
        })
        for sequence in SEQUENCES:
            subset = frame[frame.seq == sequence]
            statistic, pvalue, count = safe_correlation(
                subset[metric], subset.delta_AssA, 'spearman'
            )
            rows.append({
                'metric': metric,
                'target': 'delta_AssA',
                'sequence': sequence,
                'sequence_spearman': statistic,
                'sequence_spearman_p': pvalue,
                'sequence_n': count,
            })
    return pd.DataFrame(rows)


def rank_windows(frame: pd.DataFrame) -> pd.Series:
    return ((frame.canonical_rank.astype(int) - 21) // 20) * 20 + 21


def window_ranking_diagnostics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[pd.Series] = []
    summary_rows: list[dict[str, Any]] = []
    work = frame.copy()
    work['rank_start'] = rank_windows(work)
    work['rank_end'] = work.rank_start + 19
    for metric in metric_columns(frame):
        selections = []
        for _, group in work.groupby(['seq', 'rank_start', 'rank_end'], sort=True):
            eligible = group.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])
            if not len(eligible):
                continue
            chosen = eligible.sort_values(
                [metric, 'canonical_rank', 'transaction_type'],
                ascending=[False, True, True],
            ).iloc[0].copy()
            chosen['ranking_metric'] = metric
            selections.append(chosen)
            detail_rows.append(chosen)
        selected = pd.DataFrame(selections)
        sequence_utility = {
            sequence: float(selected.loc[selected.seq == sequence, 'delta_HOTA'].sum())
            for sequence in SEQUENCES
        }
        summary_rows.append({
            'metric': metric,
            'selected_windows': len(selected),
            'positive_windows': int((selected.delta_HOTA > 0).sum()) if len(selected) else 0,
            'negative_windows': int((selected.delta_HOTA < 0).sum()) if len(selected) else 0,
            'zero_windows': int((selected.delta_HOTA == 0).sum()) if len(selected) else 0,
            'utility_sum': float(selected.delta_HOTA.sum()) if len(selected) else 0.0,
            'worst_sequence_utility': min(sequence_utility.values()),
            'sequence_utility': json.dumps(sequence_utility, sort_keys=True),
        })
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def qualification_summary(
    event_labels: pd.DataFrame,
    correlations: pd.DataFrame,
    window_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for metric in metric_columns(event_labels):
        pooled = correlations[
            (correlations.metric == metric)
            & (correlations.target == 'delta_AssA')
            & correlations.sequence.isna()
        ]
        pooled_spearman = float(pooled.spearman.iloc[0]) if len(pooled) else math.nan
        auc_rows = correlations[
            (correlations.metric == metric)
            & (correlations.target == 'positive_delta_HOTA')
        ]
        auc = float(auc_rows.roc_auc.iloc[0]) if len(auc_rows) else math.nan
        sequence_rows = correlations[
            (correlations.metric == metric)
            & (correlations.target == 'delta_AssA')
            & correlations.sequence.notna()
        ]
        positive_sequences = int((sequence_rows.sequence_spearman > 0).sum())
        window = window_summary[window_summary.metric == metric].iloc[0]
        passes = bool(
            pooled_spearman >= PASS_THRESHOLDS['pooled_spearman_delta_AssA']
            and auc >= PASS_THRESHOLDS['positive_hota_auc']
            and positive_sequences >= PASS_THRESHOLDS['positive_sequence_spearman_count']
            and float(window.utility_sum) > 0
            and float(window.worst_sequence_utility) >= 0
            and int(window.positive_windows) >= PASS_THRESHOLDS['window_topone_positive_windows_min']
        )
        rows.append({
            'metric': metric,
            'pooled_spearman_delta_AssA': pooled_spearman,
            'positive_hota_auc': auc,
            'positive_sequence_spearman_count': positive_sequences,
            'window_selected': int(window.selected_windows),
            'window_positive': int(window.positive_windows),
            'window_utility_sum': float(window.utility_sum),
            'window_worst_sequence_utility': float(window.worst_sequence_utility),
            'qualified_auxiliary_target': int(passes),
        })
    return pd.DataFrame(rows).sort_values(
        ['qualified_auxiliary_target', 'pooled_spearman_delta_AssA', 'positive_hota_auc'],
        ascending=[False, False, False],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--utility', required=True)
    parser.add_argument('--baseline-root', required=True)
    parser.add_argument('--edited-root', required=True)
    parser.add_argument('--gt-root', default='datasets/MOT20/train')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    args = parser.parse_args()

    utility_all = pd.read_csv(args.utility)
    required_columns = EVENT_KEYS + [
        'name', 'accepted', 'changed_rows', 'effective_start_frame',
        'delta_HOTA', 'delta_AssA', 'delta_IDF1', 'track_sha256',
    ]
    missing = [column for column in required_columns if column not in utility_all.columns]
    if missing:
        raise RuntimeError(f'utility file missing columns: {missing}')
    utility = utility_all[utility_all.accepted == 1].copy().reset_index(drop=True)
    if len(utility) != 225:
        raise RuntimeError(f'expected 225 accepted train events, found {len(utility)}')

    event_rows: list[dict[str, Any]] = []
    changed_row_labels: list[dict[str, Any]] = []
    sequence_matching = []
    replay_mismatches = []
    replay_audits = []

    for sequence in SEQUENCES:
        baseline_path = Path(args.baseline_root) / f'{sequence}.txt'
        gt_path = Path(args.gt_root) / sequence / 'gt' / 'gt.txt'
        records = load_baseline_records(baseline_path)
        geometry_index = build_geometry_index(records)
        matched_gt_ids, gt_ious, match_summary = match_records_to_gt(
            records, load_gt(gt_path), args.iou_threshold
        )
        sequence_matching.append({'seq': sequence, **match_summary})

        label_indices: dict[int, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            label_indices[int(record['label'])].append(index)

        for utility_row in utility[utility.seq == sequence].to_dict('records'):
            edited_path = (
                Path(args.edited_root)
                / sequence
                / str(utility_row['name'])
                / 'track_results'
                / f'{sequence}.txt'
            )
            target_indices, donor_anchor, receiver_anchor_counts, replay_audit = (
                recover_changed_rows_from_counterfactual(
                    edited_path,
                    records,
                    geometry_index,
                    str(utility_row['track_sha256']),
                )
            )
            replay_audit['name'] = utility_row['name']
            replay_audit['seq'] = sequence
            replay_audits.append(replay_audit)
            if len(target_indices) != int(utility_row['changed_rows']):
                replay_mismatches.append({
                    'name': utility_row['name'],
                    'expected_changed_rows': int(utility_row['changed_rows']),
                    'target_indices': len(target_indices),
                })
            start_frame = int(utility_row['effective_start_frame'])
            target_frames = [int(records[index]['frame']) for index in target_indices]
            if min(target_frames) != start_frame:
                raise RuntimeError(
                    f'effective-start mismatch for {utility_row["name"]}: '
                    f'expected={start_frame}, recovered={min(target_frames)}'
                )
            primary_receiver_anchor = int(replay_audit['receiver_anchor'])
            recovered_receiver_counts = Counter(
                int(records[index]['label']) for index in target_indices
            )
            if recovered_receiver_counts != receiver_anchor_counts:
                raise RuntimeError(
                    f'receiver-anchor accounting mismatch for {utility_row["name"]}'
                )
            if 'donor_anchor' in utility_row and not pd.isna(utility_row['donor_anchor']):
                if int(utility_row['donor_anchor']) != donor_anchor:
                    raise RuntimeError(f'donor-anchor mismatch for {utility_row["name"]}')
            if 'receiver_anchor' in utility_row and not pd.isna(utility_row['receiver_anchor']):
                if int(utility_row['receiver_anchor']) not in receiver_anchor_counts:
                    raise RuntimeError(f'receiver-anchor mismatch for {utility_row["name"]}')
            donor_history_all = [
                index for index in label_indices[donor_anchor]
                if int(records[index]['frame']) < start_frame
            ]
            receiver_history_by_anchor = {
                anchor: [
                    index for index in label_indices[anchor]
                    if int(records[index]['frame']) < start_frame
                ]
                for anchor in receiver_anchor_counts
            }
            future_by_anchor: dict[int, list[int]] = defaultdict(list)
            for index in target_indices:
                future_by_anchor[int(records[index]['label'])].append(index)
            output: dict[str, Any] = {
                column: utility_row[column] for column in EVENT_KEYS + GLOBAL_TARGETS
            }
            output.update({
                'name': utility_row['name'],
                'effective_start_frame': start_frame,
                'changed_rows': len(target_indices),
                'donor_anchor': donor_anchor,
                'receiver_anchor': primary_receiver_anchor,
                'receiver_anchor_count': len(receiver_anchor_counts),
                'receiver_anchor_counts': json.dumps(
                    {
                        str(key): value
                        for key, value in sorted(receiver_anchor_counts.items())
                    },
                    sort_keys=True,
                ),
                'target_rows_with_receiver_anchor': int(sum(
                    int(records[index]['label']) == primary_receiver_anchor
                    for index in target_indices
                )),
            })
            full_identities: dict[str, Any] | None = None
            for horizon_name, horizon in HORIZONS:
                metrics, identities = compute_horizon_metrics(
                    records,
                    matched_gt_ids,
                    gt_ious,
                    donor_history_all,
                    receiver_history_by_anchor,
                    dict(future_by_anchor),
                    start_frame,
                    horizon,
                )
                output.update({f'{horizon_name}_{key}': value for key, value in metrics.items()})
                if horizon_name == 'full':
                    full_identities = identities
            event_rows.append(output)

            assert full_identities is not None
            donor_gt = int(full_identities['donor_identity'])
            aggregate_receiver_gt = int(full_identities['receiver_identity'])
            receiver_identity_by_anchor = {
                int(anchor): int(identity)
                for anchor, identity in full_identities[
                    'receiver_identity_by_anchor'
                ].items()
            }
            for index in target_indices:
                gt_id = int(matched_gt_ids[index])
                baseline_anchor = int(records[index]['label'])
                receiver_gt = receiver_identity_by_anchor.get(baseline_anchor, -1)
                if gt_id < 0:
                    row_class = 'unmatched'
                elif donor_gt >= 0 and donor_gt == receiver_gt and gt_id == donor_gt:
                    row_class = 'shared_history_identity'
                elif donor_gt >= 0 and gt_id == donor_gt:
                    row_class = 'benefit'
                elif receiver_gt >= 0 and gt_id == receiver_gt:
                    row_class = 'harm'
                else:
                    row_class = 'other_gt'
                changed_row_labels.append({
                    **{column: utility_row[column] for column in EVENT_KEYS},
                    'name': utility_row['name'],
                    'frame': int(records[index]['frame']),
                    'frames_after_handoff': int(records[index]['frame']) - start_frame,
                    'baseline_label': baseline_anchor,
                    'edited_label': donor_anchor,
                    'matched_gt_id': gt_id,
                    'matched_gt_iou': float(gt_ious[index]),
                    'donor_dominant_gt_full': donor_gt,
                    'receiver_dominant_gt_full': receiver_gt,
                    'aggregate_receiver_dominant_gt_full': aggregate_receiver_gt,
                    'row_class': row_class,
                })

    event_labels = pd.DataFrame(event_rows)
    row_labels = pd.DataFrame(changed_row_labels)
    correlations = correlation_diagnostics(event_labels)
    window_detail, window_summary = window_ranking_diagnostics(event_labels)
    qualifications = qualification_summary(event_labels, correlations, window_summary)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    rounded(event_labels).to_csv(out_dir / 'event_local_labels.csv', index=False)
    rounded(row_labels).to_csv(out_dir / 'changed_row_labels.csv', index=False)
    rounded(correlations).to_csv(out_dir / 'correlations.csv', index=False)
    rounded(window_detail).to_csv(out_dir / 'window_topone_detail.csv', index=False)
    rounded(window_summary).to_csv(out_dir / 'window_topone_summary.csv', index=False)
    rounded(qualifications).to_csv(out_dir / 'auxiliary_target_qualification.csv', index=False)
    rounded(pd.DataFrame(sequence_matching)).to_csv(
        out_dir / 'sequence_gt_matching_summary.csv', index=False
    )
    rounded(pd.DataFrame(replay_audits)).to_csv(
        out_dir / 'counterfactual_replay_audit.csv', index=False
    )

    qualified = qualifications[qualifications.qualified_auxiliary_target == 1]
    report = {
        'protocol': {
            'scope': 'Train ranks21-100 only. No locked feature, utility, or TrackEval artifact is read.',
            'baseline': 'identity_transaction_fusion_eval_v2/aggressive15_merge_then_swap',
            'counterfactual_source': 'Previously audited accepted train-event track files; every file is SHA-verified and geometry-matched to the formal baseline.',
            'gt_matching': 'Per-frame Hungarian assignment at IoU >= threshold; MOT mark=1,class=1 only.',
            'horizons': {name: horizon for name, horizon in HORIZONS},
            'local_pairwise_delta': 'Net change in correct-minus-incorrect cross-cluster GT association pairs when future receiver rows move from receiver anchor to donor anchor.',
            'local_idtp_delta': 'Change in maximum local tracker-to-GT assignment support under baseline versus edited two-cluster partitions.',
            'qualification_thresholds_preregistered': PASS_THRESHOLDS,
        },
        'dataset': {
            'utility_rows_total': len(utility_all),
            'accepted_train_events': len(event_labels),
            'changed_rows_labeled': len(row_labels),
            'sequences': SEQUENCES,
            'replay_changed_row_mismatches': len(replay_mismatches),
            'counterfactual_sha_verified': len(replay_audits),
        },
        'gt_matching': sequence_matching,
        'qualified_auxiliary_targets': qualified.metric.tolist(),
        'qualified_auxiliary_target_count': len(qualified),
        'locked_artifacts_read': False,
        'next_step_allowed': bool(len(qualified)),
    }
    (out_dir / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    manifest = {
        path.name: sha256(path)
        for path in sorted(out_dir.iterdir())
        if path.is_file() and path.name != 'manifest.json'
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
