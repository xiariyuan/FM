#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

SUMMARY_FIELDS = [
    "source_runs",
    "anchor_label_mode",
    "anchor_supervision_mode",
    "anchor_future_window",
    "anchor_future_min_nonnegative_rows",
    "anchor_future_negative_raw_gain_thresh",
    "anchor_rows",
    "anchor_base_positive_rows",
    "anchor_positive_rows",
    "anchor_relabeled_negative_rows",
    "anchor_softened_positive_rows",
    "anchor_avg_train_target",
    "anchor_avg_sample_weight",
    "anchor_negative_rows",
    "extension_rows",
    "extension_positive_rows",
    "extension_negative_rows",
    "deduped_rows",
    "conflicting_rows",
    "status",
    "error",
]

PER_SEQUENCE_FIELDS = [
    "dataset",
    "seq_name",
    "rows",
    "positive_rows",
    "negative_rows",
]

FEATURE_NAMES = [
    "acceptance_gate_score",
    "acceptance_gate_effective_score",
    "acceptance_gate_temporal_bonus",
    "best_box_iou",
    "age_gap",
    "track_age",
    "owner_age",
    "track_hits",
    "owner_hits",
    "edge_score",
    "owner_edge_score",
    "edge_advantage_vs_owner",
    "force_rewrite_keep_plan_score",
    "force_rewrite_rewrite_plan_score",
    "force_rewrite_score",
    "force_rewrite_neighborhood_gain",
    "force_rewrite_raw_neighborhood_gain",
    "force_rewrite_edge_deficit",
    "force_rewrite_recovery_bonus",
    "force_rewrite_reroute_ready_penalty",
    "force_rewrite_recovery_priority",
    "challenger_best_alt_det_box_iou",
    "challenger_best_alt_det_score",
    "owner_best_alt_det_box_iou",
    "owner_best_alt_det_score",
    "local_contention_ranker_score",
    "local_contention_ranker_margin_to_second",
    "takeover_risk_scale",
    "force_rewrite_temporal_bonus",
    "force_rewrite_temporal_frame_gap",
    "force_rewrite_temporal_warmup_streak",
    "force_rewrite_temporal_anchor_edge_deficit",
    "derived_candidate_history_exists",
    "derived_candidate_history_frame_gap",
    "derived_candidate_history_warmup_streak",
    "derived_candidate_history_last_force_rewrite_score",
    "derived_candidate_history_last_neighborhood_gain",
    "derived_candidate_history_last_raw_neighborhood_gain",
    "derived_candidate_history_last_selected_forced",
    "det_rank",
    "det_top_score",
    "track_claim_matches_det_gt",
    "owner_claim_matches_det_gt",
    "owner_alt_base_viable",
    "owner_alt_reroute_ready",
    "acceptance_gate_accepted",
    "takeover_risk_accepted",
    "force_rewrite_shared_alt_conflict",
    "selected_forced",
    "force_rewrite_accepted",
    "force_rewrite_temporal_active",
    "force_rewrite_temporal_anchor_ready",
    "derived_prior_anchor_exists",
    "derived_prior_anchor_frame_gap",
    "derived_prior_anchor_edge_deficit",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build offline anchor and extension datasets from anchored recovery pre-association exports."
    )
    parser.add_argument(
        "--run-roots",
        nargs="*",
        default=[
            str(
                REPO_ROOT
                / "outputs"
                / "deep_ocsort_preassoc_force_recovery_anchor_seq0090_debug"
                / "runs"
                / "anchor_gap3_safe015"
            ),
            str(
                REPO_ROOT
                / "outputs"
                / "deep_ocsort_preassoc_force_recovery_anchor_seq0090_debug"
                / "runs"
                / "anchor_gap2_safe015"
            ),
            str(
                REPO_ROOT
                / "outputs"
                / "deep_ocsort_preassoc_force_recovery_anchor_seq0090_debug"
                / "runs"
                / "anchor_gap3_safe020"
            ),
            str(
                REPO_ROOT
                / "outputs"
                / "deep_ocsort_preassoc_force_recovery_anchor_confirm_dance3_debug"
                / "runs"
                / "anchor_gap2_safe015_dance3"
            ),
        ],
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--anchor-min-raw-neighborhood-gain", type=float, default=0.05)
    parser.add_argument("--anchor-max-edge-deficit", type=float, default=0.35)
    parser.add_argument(
        "--anchor-future-window",
        type=int,
        default=4,
        help="Future frame window used to keep only anchors that still show support in the same challenger-owner pair.",
    )
    parser.add_argument(
        "--anchor-future-min-nonnegative-rows",
        type=int,
        default=1,
        help="When future rows exist for the same challenger-owner pair, require at least this many rows with nonnegative raw neighborhood gain.",
    )
    parser.add_argument(
        "--anchor-future-negative-raw-gain-thresh",
        type=float,
        default=-0.05,
        help="Threshold used for reporting strongly negative future raw neighborhood gain rows in derived diagnostics.",
    )
    parser.add_argument(
        "--anchor-supervision-mode",
        choices=["hard_short_horizon", "soft_future_support"],
        default="hard_short_horizon",
        help="How future support is injected into anchor supervision.",
    )
    parser.add_argument(
        "--anchor-label-mode",
        choices=[
            "base_label_v1",
            "gt_reclaim_dense_v1",
            "gt_reclaim_xor_v1",
            "gt_reclaim_softneg_v1",
        ],
        default="base_label_v1",
        help=(
            "Anchor label definition. "
            "base_label_v1 keeps the original rule-imitation target. "
            "gt_reclaim_dense_v1 uses ground-truth reclaim correctness for all anchor candidates. "
            "gt_reclaim_xor_v1 keeps only clear reclaim supervision where challenger and owner disagree. "
            "gt_reclaim_softneg_v1 keeps clear reclaim positives, clear owner-correct negatives, and downweights ambiguous negatives."
        ),
    )
    parser.add_argument(
        "--anchor-ambiguous-negative-weight",
        type=float,
        default=0.05,
        help="Sample weight used by gt_reclaim_softneg_v1 for ambiguous rows where challenger and owner do not disagree cleanly.",
    )
    parser.add_argument(
        "--anchor-soft-min-target",
        type=float,
        default=0.25,
        help="Minimum soft target assigned to base-positive anchors that have future rows but no future support.",
    )
    parser.add_argument(
        "--anchor-soft-trailing-target",
        type=float,
        default=0.85,
        help="Soft target assigned to base-positive anchors with no future rows in the observation window.",
    )
    parser.add_argument(
        "--anchor-soft-max-extra-weight",
        type=float,
        default=1.0,
        help="Maximum additional sample weight assigned to base-positive anchors with future evidence.",
    )
    parser.add_argument(
        "--anchor-soft-negative-penalty-weight",
        type=float,
        default=0.5,
        help="Additional sample weight for base-positive anchors whose future window is strongly negative.",
    )
    parser.add_argument("--candidate-history-max-frame-gap", type=int, default=2)
    parser.add_argument("--candidate-history-min-score", type=float, default=0.85)
    parser.add_argument("--candidate-history-min-box-iou", type=float, default=0.80)
    parser.add_argument("--candidate-history-min-challenger-alt-box-iou", type=float, default=0.12)
    parser.add_argument("--candidate-history-min-neighborhood-gain", type=float, default=-0.08)
    parser.add_argument("--extension-max-frame-gap", type=int, default=3)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_single_row_csv(path: Path, fieldnames: Iterable[str], row: Dict[str, object]) -> None:
    write_rows(path, fieldnames, [row])


def append_registry(args: argparse.Namespace, summary_csv: Path, status: str, notes: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(args.registry_csv),
        "--kind",
        "analysis",
        "--status",
        status,
        "--script",
        "scripts/build_recovery_anchor_extension_dataset.py",
        "--dataset",
        "DanceTrack",
        "--split",
        "anchor_extension_jsonl",
        "--tracker-family",
        "deep_ocsort_preassoc_force_recovery_anchor",
        "--variant",
        Path(args.out_dir).name,
        "--tag",
        Path(args.out_dir).name,
        "--run-root",
        str(Path(args.out_dir).resolve()),
        "--summary-csv",
        str(summary_csv.resolve()),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def to_float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def to_int(value: object, default: int = 0) -> int:
    try:
        if value in ("", None):
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def build_feature_vector(row: Dict[str, object]) -> List[float]:
    vector: List[float] = []
    for name in FEATURE_NAMES:
        value = row.get(name, 0.0)
        if name in {
            "owner_alt_base_viable",
            "owner_alt_reroute_ready",
            "acceptance_gate_accepted",
            "takeover_risk_accepted",
            "force_rewrite_shared_alt_conflict",
            "selected_forced",
            "force_rewrite_accepted",
            "force_rewrite_temporal_active",
            "force_rewrite_temporal_anchor_ready",
            "track_claim_matches_det_gt",
            "owner_claim_matches_det_gt",
            "derived_candidate_history_exists",
            "derived_candidate_history_last_selected_forced",
        }:
            vector.append(float(to_int(value, 0)))
        else:
            vector.append(float(to_float(value, 0.0)))
    return vector


def dedupe_key(row: Dict[str, object], dataset: str) -> Tuple[str, str, int, int, int, int]:
    return (
        str(dataset),
        str(row.get("seq_name", "")),
        int(to_int(row.get("frame_id", -1), -1)),
        int(to_int(row.get("track_internal_id", -1), -1)),
        int(to_int(row.get("raw_owner_internal_id", -1), -1)),
        int(to_int(row.get("det_idx", -1), -1)),
    )


def future_pair_key(source_run: str, row: Dict[str, object]) -> Tuple[str, str, int, int, int]:
    return (
        str(source_run),
        str(row.get("seq_name", "")),
        int(to_int(row.get("track_internal_id", -1), -1)),
        int(to_int(row.get("raw_owner_internal_id", -1), -1)),
        int(to_int(row.get("det_gt_id", -1), -1)),
    )


def future_row_key(source_run: str, row: Dict[str, object]) -> Tuple[str, str, int, int, int, int, int]:
    return (
        str(source_run),
        str(row.get("seq_name", "")),
        int(to_int(row.get("frame_id", -1), -1)),
        int(to_int(row.get("track_internal_id", -1), -1)),
        int(to_int(row.get("raw_owner_internal_id", -1), -1)),
        int(to_int(row.get("det_idx", -1), -1)),
        int(to_int(row.get("det_gt_id", -1), -1)),
    )


def candidate_history_pair_key(source_run: str, row: Dict[str, object]) -> Tuple[str, str, int, int]:
    return (
        str(source_run),
        str(row.get("seq_name", "")),
        int(to_int(row.get("track_internal_id", -1), -1)),
        int(to_int(row.get("raw_owner_internal_id", -1), -1)),
    )


def anchor_candidate(row: Dict[str, object]) -> bool:
    return int(to_int(row.get("force_rewrite_temporal_anchor_ready", 0), 0)) == 0 and float(
        to_float(row.get("acceptance_gate_score", -1.0), -1.0)
    ) >= 0.0


def anchor_base_label(row: Dict[str, object], args: argparse.Namespace) -> int:
    accepted = int(to_int(row.get("force_rewrite_accepted", 0), 0)) == 1
    gate_accepted = int(to_int(row.get("acceptance_gate_accepted", 0), 0)) == 1
    raw_gain = float(to_float(row.get("force_rewrite_raw_neighborhood_gain", -1e9), -1e9))
    edge_deficit = float(to_float(row.get("force_rewrite_edge_deficit", 1e9), 1e9))
    return int(
        accepted
        and gate_accepted
        and raw_gain >= float(args.anchor_min_raw_neighborhood_gain)
        and edge_deficit <= float(args.anchor_max_edge_deficit)
    )


def anchor_gt_reclaim_label(row: Dict[str, object]) -> int | None:
    track_claim_matches_det_gt = int(to_int(row.get("track_claim_matches_det_gt", 0), 0)) == 1
    owner_claim_matches_det_gt = int(to_int(row.get("owner_claim_matches_det_gt", 0), 0)) == 1
    if track_claim_matches_det_gt and not owner_claim_matches_det_gt:
        return 1
    if owner_claim_matches_det_gt and not track_claim_matches_det_gt:
        return 0
    return None


def build_anchor_future_stats(
    rows: List[Dict[str, object]],
    *,
    source_run: str,
    args: argparse.Namespace,
) -> Dict[Tuple[str, str, int, int, int, int, int], Dict[str, object]]:
    rows_by_pair: Dict[Tuple[str, str, int, int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_pair[future_pair_key(source_run, row)].append(row)

    stats_by_row: Dict[Tuple[str, str, int, int, int, int, int], Dict[str, object]] = {}
    future_window = max(int(args.anchor_future_window), 0)
    negative_raw_gain_thresh = float(args.anchor_future_negative_raw_gain_thresh)
    min_nonnegative_rows = max(int(args.anchor_future_min_nonnegative_rows), 0)

    for pair_rows in rows_by_pair.values():
        pair_rows.sort(key=lambda row: int(to_int(row.get("frame_id", -1), -1)))
        for row in pair_rows:
            frame_id = int(to_int(row.get("frame_id", -1), -1))
            future_rows = [
                future_row
                for future_row in pair_rows
                if frame_id < int(to_int(future_row.get("frame_id", -1), -1)) <= frame_id + future_window
            ]
            future_nonnegative_rows = sum(
                1
                for future_row in future_rows
                if float(to_float(future_row.get("force_rewrite_raw_neighborhood_gain", -1e9), -1e9)) >= 0.0
            )
            future_negative_rows = sum(
                1
                for future_row in future_rows
                if float(
                    to_float(future_row.get("force_rewrite_raw_neighborhood_gain", -1e9), -1e9)
                )
                < negative_raw_gain_thresh
            )
            future_owner_claim_rows = sum(
                1
                for future_row in future_rows
                if int(to_int(future_row.get("owner_claim_matches_det_gt", 0), 0)) == 1
            )
            future_row_count = int(len(future_rows))
            future_ratio_denom = float(max(future_row_count, 1))
            future_support = int(
                future_row_count == 0 or future_nonnegative_rows >= min_nonnegative_rows
            )
            stats_by_row[future_row_key(source_run, row)] = {
                "derived_future_rows": int(future_row_count),
                "derived_future_nonnegative_raw_gain_rows": int(future_nonnegative_rows),
                "derived_future_negative_raw_gain_rows": int(future_negative_rows),
                "derived_future_owner_claim_rows": int(future_owner_claim_rows),
                "derived_future_nonnegative_raw_gain_ratio": float(
                    float(future_nonnegative_rows) / future_ratio_denom
                ),
                "derived_future_negative_raw_gain_ratio": float(
                    float(future_negative_rows) / future_ratio_denom
                ),
                "derived_future_owner_claim_ratio": float(
                    float(future_owner_claim_rows) / future_ratio_denom
                ),
                "derived_future_support": int(future_support),
            }
    return stats_by_row


def build_candidate_history_stats(
    rows: List[Dict[str, object]],
    *,
    source_run: str,
    args: argparse.Namespace,
) -> Dict[Tuple[str, str, int, int, int, int, int], Dict[str, object]]:
    rows_by_pair: Dict[Tuple[str, str, int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_pair[candidate_history_pair_key(source_run, row)].append(row)

    stats_by_row: Dict[Tuple[str, str, int, int, int, int, int], Dict[str, object]] = {}
    max_frame_gap = max(int(args.candidate_history_max_frame_gap), 1)
    min_score = float(args.candidate_history_min_score)
    min_box_iou = float(args.candidate_history_min_box_iou)
    min_challenger_alt_box_iou = float(args.candidate_history_min_challenger_alt_box_iou)
    min_neighborhood_gain = float(args.candidate_history_min_neighborhood_gain)

    for pair_rows in rows_by_pair.values():
        pair_rows.sort(
            key=lambda row: (
                int(to_int(row.get("frame_id", -1), -1)),
                int(to_int(row.get("det_idx", -1), -1)),
            )
        )
        state: Dict[str, object] | None = None
        for row in pair_rows:
            frame_id = int(to_int(row.get("frame_id", -1), -1))
            context = {
                "derived_candidate_history_exists": 0,
                "derived_candidate_history_frame_gap": 0,
                "derived_candidate_history_warmup_streak": 0,
                "derived_candidate_history_last_force_rewrite_score": 0.0,
                "derived_candidate_history_last_neighborhood_gain": 0.0,
                "derived_candidate_history_last_raw_neighborhood_gain": 0.0,
                "derived_candidate_history_last_selected_forced": 0,
            }
            if state is not None:
                frame_gap = int(frame_id - int(to_int(state.get("last_frame_id", -1), -1)))
                if 0 < frame_gap <= max_frame_gap:
                    context = {
                        "derived_candidate_history_exists": 1,
                        "derived_candidate_history_frame_gap": int(frame_gap),
                        "derived_candidate_history_warmup_streak": int(
                            to_int(state.get("warmup_streak", 0), 0)
                        ),
                        "derived_candidate_history_last_force_rewrite_score": float(
                            to_float(state.get("last_force_rewrite_score", 0.0), 0.0)
                        ),
                        "derived_candidate_history_last_neighborhood_gain": float(
                            to_float(state.get("last_neighborhood_gain", 0.0), 0.0)
                        ),
                        "derived_candidate_history_last_raw_neighborhood_gain": float(
                            to_float(state.get("last_raw_neighborhood_gain", 0.0), 0.0)
                        ),
                        "derived_candidate_history_last_selected_forced": int(
                            to_int(state.get("last_selected_forced", 0), 0)
                        ),
                    }
            stats_by_row[future_row_key(source_run, row)] = context

            owner_alt_reroute_ready = int(to_int(row.get("owner_alt_reroute_ready", 0), 0)) == 1
            force_rewrite_score = float(to_float(row.get("force_rewrite_score", -1.0), -1.0))
            best_box_iou = float(to_float(row.get("best_box_iou", 0.0), 0.0))
            challenger_alt_box_iou = float(
                to_float(row.get("challenger_best_alt_det_box_iou", 0.0), 0.0)
            )
            neighborhood_gain = float(to_float(row.get("force_rewrite_neighborhood_gain", -1e9), -1e9))
            raw_neighborhood_gain = float(
                to_float(row.get("force_rewrite_raw_neighborhood_gain", 0.0), 0.0)
            )
            selected_forced = int(to_int(row.get("selected_forced", 0), 0))
            eligible = (
                not owner_alt_reroute_ready
                and force_rewrite_score + 1e-6 >= min_score
                and best_box_iou + 1e-6 >= min_box_iou
                and challenger_alt_box_iou + 1e-6 >= min_challenger_alt_box_iou
                and neighborhood_gain + 1e-6 >= min_neighborhood_gain
            )
            if not eligible:
                continue

            previous_warmup = 0
            if state is not None:
                frame_gap = int(frame_id - int(to_int(state.get("last_frame_id", -1), -1)))
                if 0 < frame_gap <= max_frame_gap:
                    previous_warmup = int(to_int(state.get("warmup_streak", 0), 0))
            state = {
                "last_frame_id": int(frame_id),
                "warmup_streak": int(previous_warmup + 1 if previous_warmup > 0 else 1),
                "last_force_rewrite_score": float(force_rewrite_score),
                "last_neighborhood_gain": float(neighborhood_gain),
                "last_raw_neighborhood_gain": float(raw_neighborhood_gain),
                "last_selected_forced": int(selected_forced),
            }
    return stats_by_row


def anchor_label(
    row: Dict[str, object],
    args: argparse.Namespace,
    future_stats: Dict[str, object] | None = None,
) -> int | None:
    if str(args.anchor_label_mode) == "gt_reclaim_dense_v1":
        gt_label = anchor_gt_reclaim_label(row)
        return 0 if gt_label is None else int(gt_label)
    if str(args.anchor_label_mode) == "gt_reclaim_xor_v1":
        return anchor_gt_reclaim_label(row)
    if str(args.anchor_label_mode) == "gt_reclaim_softneg_v1":
        gt_label = anchor_gt_reclaim_label(row)
        return 0 if gt_label is None else int(gt_label)
    base_label = anchor_base_label(row, args)
    if str(args.anchor_supervision_mode) == "soft_future_support":
        return int(base_label)
    support = int((future_stats or {}).get("derived_future_support", 1))
    return int(base_label == 1 and support == 1)


def anchor_train_target_and_weight(
    row: Dict[str, object],
    args: argparse.Namespace,
    future_stats: Dict[str, object] | None = None,
) -> Tuple[float, float]:
    if str(args.anchor_label_mode) == "gt_reclaim_softneg_v1":
        gt_label = anchor_gt_reclaim_label(row)
        if gt_label is None:
            return 0.0, float(max(args.anchor_ambiguous_negative_weight, 0.0))
        return float(gt_label), 1.0

    if str(args.anchor_label_mode) != "base_label_v1":
        label = anchor_label(row, args, future_stats)
        if label is None:
            return 0.0, 0.0
        return float(label), 1.0

    base_label = anchor_base_label(row, args)
    if str(args.anchor_supervision_mode) != "soft_future_support":
        label = float(anchor_label(row, args, future_stats))
        return label, 1.0

    if base_label != 1:
        return 0.0, 1.0

    stats = future_stats or {}
    future_rows = int(stats.get("derived_future_rows", 0))
    future_nonnegative_rows = int(stats.get("derived_future_nonnegative_raw_gain_rows", 0))
    future_negative_rows = int(stats.get("derived_future_negative_raw_gain_rows", 0))

    if future_rows <= 0:
        return float(args.anchor_soft_trailing_target), 1.0

    support_ratio = float(future_nonnegative_rows) / float(max(future_rows, 1))
    min_target = float(args.anchor_soft_min_target)
    target = float(min_target + (1.0 - min_target) * support_ratio)
    extra_weight = float(args.anchor_soft_max_extra_weight) * min(float(future_rows), 4.0) / 4.0
    weight = 1.0 + extra_weight
    if future_nonnegative_rows == 0 and future_negative_rows > 0:
        weight += float(args.anchor_soft_negative_penalty_weight)
    return target, weight


def extension_candidate(row: Dict[str, object]) -> bool:
    return False


def extension_label(row: Dict[str, object]) -> int:
    return int(int(to_int(row.get("force_rewrite_accepted", 0), 0)) == 1)


def make_dataset_row(
    *,
    dataset: str,
    source_run: str,
    row: Dict[str, object],
    label: int,
    extras: Dict[str, object] | None = None,
) -> Dict[str, object]:
    item = dict(row)
    item.update(
        {
            "dataset": dataset,
            "source_run": source_run,
            "seq_name": str(row.get("seq_name", "")),
            "frame_id": int(to_int(row.get("frame_id", -1), -1)),
            "output_frame_id": int(to_int(row.get("output_frame_id", -1), -1)),
            "track_internal_id": int(to_int(row.get("track_internal_id", -1), -1)),
            "track_output_id": int(to_int(row.get("track_output_id", -1), -1)),
            "raw_owner_internal_id": int(to_int(row.get("raw_owner_internal_id", -1), -1)),
            "raw_owner_output_id": int(to_int(row.get("raw_owner_output_id", -1), -1)),
            "det_idx": int(to_int(row.get("det_idx", -1), -1)),
            "det_gt_id": int(to_int(row.get("det_gt_id", -1), -1)),
            "track_gt_id": int(to_int(row.get("track_gt_id", -1), -1)),
            "label": int(label),
            "feature_names": list(FEATURE_NAMES),
            "derived_candidate_history_exists": 0,
            "derived_candidate_history_frame_gap": 0,
            "derived_candidate_history_warmup_streak": 0,
            "derived_candidate_history_last_force_rewrite_score": 0.0,
            "derived_candidate_history_last_neighborhood_gain": 0.0,
            "derived_candidate_history_last_raw_neighborhood_gain": 0.0,
            "derived_candidate_history_last_selected_forced": 0,
            "derived_prior_anchor_exists": 0,
            "derived_prior_anchor_frame_gap": 0,
            "derived_prior_anchor_edge_deficit": 0.0,
            "anchor_base_label": 0,
            "derived_future_rows": 0,
            "derived_future_nonnegative_raw_gain_rows": 0,
            "derived_future_negative_raw_gain_rows": 0,
            "derived_future_owner_claim_rows": 0,
            "derived_future_nonnegative_raw_gain_ratio": 0.0,
            "derived_future_negative_raw_gain_ratio": 0.0,
            "derived_future_owner_claim_ratio": 0.0,
            "derived_future_support": 1,
            "train_target": float(label),
            "sample_weight": 1.0,
        }
    )
    if extras:
        item.update(extras)
    item["features"] = build_feature_vector(item)
    return item


def load_jsonl_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "summary.csv"
    per_sequence_csv = out_dir / "per_sequence_summary.csv"
    anchor_jsonl = out_dir / "anchor_dataset.jsonl"
    extension_jsonl = out_dir / "extension_dataset.jsonl"

    summary_row: Dict[str, object] = {
        "source_runs": "|".join(str(Path(item).expanduser().resolve()) for item in args.run_roots),
        "anchor_label_mode": str(args.anchor_label_mode),
        "anchor_supervision_mode": str(args.anchor_supervision_mode),
        "anchor_future_window": int(args.anchor_future_window),
        "anchor_future_min_nonnegative_rows": int(args.anchor_future_min_nonnegative_rows),
        "anchor_future_negative_raw_gain_thresh": float(args.anchor_future_negative_raw_gain_thresh),
        "anchor_rows": 0,
        "anchor_base_positive_rows": 0,
        "anchor_positive_rows": 0,
        "anchor_relabeled_negative_rows": 0,
        "anchor_softened_positive_rows": 0,
        "anchor_avg_train_target": 0.0,
        "anchor_avg_sample_weight": 0.0,
        "anchor_negative_rows": 0,
        "extension_rows": 0,
        "extension_positive_rows": 0,
        "extension_negative_rows": 0,
        "deduped_rows": 0,
        "conflicting_rows": 0,
        "status": "running",
        "error": "",
    }
    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
    write_rows(per_sequence_csv, PER_SEQUENCE_FIELDS, [])
    append_registry(args, summary_csv, "running", "building recovery anchor and extension datasets")

    try:
        anchor_rows_by_key: Dict[Tuple[str, str, int, int, int, int], Dict[str, object]] = {}
        extension_rows_by_key: Dict[Tuple[str, str, int, int, int, int], Dict[str, object]] = {}
        base_rows_by_key: Dict[Tuple[str, str, int, int, int, int], Dict[str, object]] = {}
        conflicting_rows = 0

        for run_root_str in args.run_roots:
            run_root = Path(run_root_str).expanduser().resolve()
            candidate_jsonl = run_root / "preassoc_candidates.jsonl"
            if not candidate_jsonl.is_file():
                raise FileNotFoundError(f"Missing candidate export: {candidate_jsonl}")

            run_rows = load_jsonl_rows(candidate_jsonl)
            future_stats_by_row = build_anchor_future_stats(run_rows, source_run=run_root.name, args=args)
            candidate_history_stats_by_row = build_candidate_history_stats(
                run_rows,
                source_run=run_root.name,
                args=args,
            )

            for row in run_rows:
                row["source_run"] = run_root.name
                base_key = dedupe_key(row, "base")
                previous_base = base_rows_by_key.get(base_key)
                if previous_base is None:
                    base_rows_by_key[base_key] = dict(row)

                if anchor_candidate(row):
                    future_stats = future_stats_by_row.get(future_row_key(run_root.name, row), {})
                    candidate_history_stats = candidate_history_stats_by_row.get(
                        future_row_key(run_root.name, row),
                        {},
                    )
                    label = anchor_label(row, args, future_stats)
                    if label is None:
                        continue
                    base_label = anchor_base_label(row, args)
                    train_target, sample_weight = anchor_train_target_and_weight(row, args, future_stats)
                    item = make_dataset_row(
                        dataset="anchor",
                        source_run=run_root.name,
                        row=row,
                        label=int(label),
                        extras={
                            "anchor_base_label": int(base_label),
                            "derived_candidate_history_exists": int(
                                candidate_history_stats.get("derived_candidate_history_exists", 0)
                            ),
                            "derived_candidate_history_frame_gap": int(
                                candidate_history_stats.get("derived_candidate_history_frame_gap", 0)
                            ),
                            "derived_candidate_history_warmup_streak": int(
                                candidate_history_stats.get("derived_candidate_history_warmup_streak", 0)
                            ),
                            "derived_candidate_history_last_force_rewrite_score": float(
                                candidate_history_stats.get(
                                    "derived_candidate_history_last_force_rewrite_score",
                                    0.0,
                                )
                            ),
                            "derived_candidate_history_last_neighborhood_gain": float(
                                candidate_history_stats.get(
                                    "derived_candidate_history_last_neighborhood_gain",
                                    0.0,
                                )
                            ),
                            "derived_candidate_history_last_raw_neighborhood_gain": float(
                                candidate_history_stats.get(
                                    "derived_candidate_history_last_raw_neighborhood_gain",
                                    0.0,
                                )
                            ),
                            "derived_candidate_history_last_selected_forced": int(
                                candidate_history_stats.get(
                                    "derived_candidate_history_last_selected_forced",
                                    0,
                                )
                            ),
                            "derived_future_rows": int(future_stats.get("derived_future_rows", 0)),
                            "derived_future_nonnegative_raw_gain_rows": int(
                                future_stats.get("derived_future_nonnegative_raw_gain_rows", 0)
                            ),
                            "derived_future_negative_raw_gain_rows": int(
                                future_stats.get("derived_future_negative_raw_gain_rows", 0)
                            ),
                            "derived_future_owner_claim_rows": int(
                                future_stats.get("derived_future_owner_claim_rows", 0)
                            ),
                            "derived_future_nonnegative_raw_gain_ratio": float(
                                future_stats.get("derived_future_nonnegative_raw_gain_ratio", 0.0)
                            ),
                            "derived_future_negative_raw_gain_ratio": float(
                                future_stats.get("derived_future_negative_raw_gain_ratio", 0.0)
                            ),
                            "derived_future_owner_claim_ratio": float(
                                future_stats.get("derived_future_owner_claim_ratio", 0.0)
                            ),
                            "derived_future_support": int(future_stats.get("derived_future_support", 1)),
                            "train_target": float(train_target),
                            "sample_weight": float(sample_weight),
                        },
                    )
                    key = dedupe_key(row, "anchor")
                    previous = anchor_rows_by_key.get(key)
                    if previous is not None and int(previous["label"]) != int(item["label"]):
                        conflicting_rows += 1
                        if int(item["label"]) > int(previous["label"]):
                            anchor_rows_by_key[key] = item
                    elif previous is None:
                        anchor_rows_by_key[key] = item

        anchor_rows = sorted(
            anchor_rows_by_key.values(),
            key=lambda row: (
                str(row.get("seq_name", "")),
                int(row.get("frame_id", -1)),
                int(row.get("track_internal_id", -1)),
                int(row.get("det_idx", -1)),
            ),
        )

        pair_to_positive_anchors: Dict[Tuple[str, int, int], List[Dict[str, object]]] = defaultdict(list)
        for row in anchor_rows:
            if int(row.get("label", 0)) != 1:
                continue
            pair_key = (
                str(row.get("seq_name", "")),
                int(row.get("track_internal_id", -1)),
                int(row.get("raw_owner_internal_id", -1)),
            )
            pair_to_positive_anchors[pair_key].append(row)
        for rows in pair_to_positive_anchors.values():
            rows.sort(key=lambda row: int(row.get("frame_id", -1)))

        base_rows = sorted(
            base_rows_by_key.values(),
            key=lambda row: (
                str(row.get("seq_name", "")),
                int(to_int(row.get("frame_id", -1), -1)),
                int(to_int(row.get("track_internal_id", -1), -1)),
                int(to_int(row.get("raw_owner_internal_id", -1), -1)),
                int(to_int(row.get("det_idx", -1), -1)),
            ),
        )
        for row in base_rows:
            pair_key = (
                str(row.get("seq_name", "")),
                int(to_int(row.get("track_internal_id", -1), -1)),
                int(to_int(row.get("raw_owner_internal_id", -1), -1)),
            )
            frame_id = int(to_int(row.get("frame_id", -1), -1))
            prior_anchor = None
            for anchor_row in pair_to_positive_anchors.get(pair_key, []):
                anchor_frame = int(anchor_row.get("frame_id", -1))
                frame_gap = frame_id - anchor_frame
                if 0 < frame_gap <= int(args.extension_max_frame_gap):
                    prior_anchor = anchor_row
            if prior_anchor is None:
                continue
            item = make_dataset_row(
                dataset="extension",
                source_run=str(row.get("source_run", "")) or "deduped_anchor_exports",
                row=row,
                label=extension_label(row),
                extras={
                    "derived_prior_anchor_exists": 1,
                    "derived_prior_anchor_frame_gap": int(frame_id - int(prior_anchor.get("frame_id", -1))),
                    "derived_prior_anchor_edge_deficit": float(
                        to_float(prior_anchor.get("force_rewrite_edge_deficit", 0.0), 0.0)
                    ),
                },
            )
            key = dedupe_key(row, "extension")
            previous = extension_rows_by_key.get(key)
            if previous is not None and int(previous["label"]) != int(item["label"]):
                conflicting_rows += 1
                if int(item["label"]) > int(previous["label"]):
                    extension_rows_by_key[key] = item
            elif previous is None:
                extension_rows_by_key[key] = item

        extension_rows = sorted(
            extension_rows_by_key.values(),
            key=lambda row: (
                str(row.get("seq_name", "")),
                int(row.get("frame_id", -1)),
                int(row.get("track_internal_id", -1)),
                int(row.get("det_idx", -1)),
            ),
        )

        write_jsonl(anchor_jsonl, anchor_rows)
        write_jsonl(extension_jsonl, extension_rows)

        per_sequence_counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
            lambda: {"rows": 0, "positive_rows": 0, "negative_rows": 0}
        )
        for dataset_name, rows in (("anchor", anchor_rows), ("extension", extension_rows)):
            for row in rows:
                key = (dataset_name, str(row.get("seq_name", "")))
                per_sequence_counts[key]["rows"] += 1
                if int(row.get("label", 0)) == 1:
                    per_sequence_counts[key]["positive_rows"] += 1
                else:
                    per_sequence_counts[key]["negative_rows"] += 1

        per_sequence_rows = [
            {
                "dataset": dataset,
                "seq_name": seq_name,
                "rows": counts["rows"],
                "positive_rows": counts["positive_rows"],
                "negative_rows": counts["negative_rows"],
            }
            for (dataset, seq_name), counts in sorted(per_sequence_counts.items())
        ]
        write_rows(per_sequence_csv, PER_SEQUENCE_FIELDS, per_sequence_rows)

        anchor_positive_rows = sum(int(row["label"]) for row in anchor_rows)
        anchor_base_positive_rows = sum(int(row.get("anchor_base_label", row["label"])) for row in anchor_rows)
        anchor_relabeled_negative_rows = sum(
            1
            for row in anchor_rows
            if int(row.get("anchor_base_label", row["label"])) == 1 and int(row.get("label", 0)) == 0
        )
        anchor_softened_positive_rows = sum(
            1
            for row in anchor_rows
            if int(row.get("anchor_base_label", 0)) == 1
            and float(to_float(row.get("train_target", row.get("label", 0)), 0.0)) < 0.999
        )
        anchor_avg_train_target = (
            sum(float(to_float(row.get("train_target", row.get("label", 0)), 0.0)) for row in anchor_rows)
            / float(max(len(anchor_rows), 1))
        )
        anchor_avg_sample_weight = (
            sum(float(to_float(row.get("sample_weight", 1.0), 1.0)) for row in anchor_rows)
            / float(max(len(anchor_rows), 1))
        )

        summary_row.update(
            {
                "anchor_rows": len(anchor_rows),
                "anchor_base_positive_rows": anchor_base_positive_rows,
                "anchor_positive_rows": anchor_positive_rows,
                "anchor_relabeled_negative_rows": anchor_relabeled_negative_rows,
                "anchor_softened_positive_rows": anchor_softened_positive_rows,
                "anchor_avg_train_target": anchor_avg_train_target,
                "anchor_avg_sample_weight": anchor_avg_sample_weight,
                "anchor_negative_rows": len(anchor_rows) - anchor_positive_rows,
                "extension_rows": len(extension_rows),
                "extension_positive_rows": sum(int(row["label"]) for row in extension_rows),
                "extension_negative_rows": len(extension_rows) - sum(int(row["label"]) for row in extension_rows),
                "deduped_rows": len(anchor_rows_by_key) + len(extension_rows_by_key),
                "conflicting_rows": conflicting_rows,
                "status": "success",
                "error": "",
            }
        )
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, "success", "built recovery anchor and extension datasets")
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = str(exc)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, "failed", f"failed to build recovery anchor and extension datasets: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
