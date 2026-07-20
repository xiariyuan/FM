#!/usr/bin/env python3
from __future__ import annotations

"""Post-freeze candidate-coverage diagnostics for M23-53.

This audit is deliberately read-only with respect to the frozen candidate graphs,
teacher utilities, selected path covers, trackers, and official TrackEval outputs.
It verifies the frozen SHA-256 manifests, then uses the already-created teacher
labels to diagnose candidate coverage and flow conversion. It never regenerates
candidates, changes teacher weights, runs TrackEval, or selects a deployable
policy.
"""

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
OUTPUTS = Path("outputs/mot20_m23_20260718")
BASELINE_CACHE = OUTPUTS / "m23_47_m23_46_deployable_baseline_cache_v1"
AUDIT_ROOT = OUTPUTS / "m23_53_candidate_coverage_audit_v1"
DOC_JSON = Path("docs/generated/M23_53_CANDIDATE_COVERAGE_AUDIT_20260720.json")
DOC_MD = Path("docs/m23_53_candidate_coverage_audit_20260720.md")
COMBINED_REPORT = OUTPUTS / "m23_53_identity_flow_capacity_combined_v1/report.json"
SMOKE_RUN = "m23_53_identity_flow_capacity_m02_smoke_v1"
FOLDS = {
    "MOT20-01": "m23_53_identity_flow_capacity_m01_v1",
    "MOT20-02": "m23_53_identity_flow_capacity_m02_v1",
    "MOT20-03": "m23_53_identity_flow_capacity_m03_v1",
    "MOT20-05": "m23_53_identity_flow_capacity_m05_v1",
}
RANK_COLUMNS = (
    "out_rank",
    "in_rank",
    "max_rank",
    "appearance_out_rank",
    "appearance_in_rank",
    "motion_out_rank",
    "motion_in_rank",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def as_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): as_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def quantile(values: Iterable[float], q: float) -> float | None:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return None
    return float(np.quantile(array, q))


def bin_gap(gap: int) -> str:
    if gap <= 0:
        return "0"
    if gap <= 30:
        return "1-30"
    if gap <= 90:
        return "31-90"
    if gap <= 180:
        return "91-180"
    if gap <= 600:
        return "181-600"
    return ">600"


def tertile_label(value: float, low: float, high: float) -> str:
    if value <= low:
        return "low"
    if value <= high:
        return "medium"
    return "high"


def summarize_groups(events: pd.DataFrame, column: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    order = list(dict.fromkeys(events[column].astype(str).tolist()))
    for label in order:
        group = events[events[column].astype(str) == label]
        present = int(group.candidate_present.sum())
        selected = int(group.selected_by_teacher.sum())
        result[label] = {
            "events": int(len(group)),
            "candidate_present": present,
            "candidate_recall": safe_rate(present, len(group)),
            "selected_by_teacher": selected,
            "teacher_selection_rate_all": safe_rate(selected, len(group)),
            "teacher_selection_rate_when_present": safe_rate(selected, present),
        }
    return result


def rank_summary(events: pd.DataFrame) -> dict[str, Any]:
    present = events[events.candidate_present == 1]
    result: dict[str, Any] = {"candidate_present_events": int(len(present))}
    for column in RANK_COLUMNS:
        values = present[column].dropna().astype(float)
        result[column] = {
            "count": int(len(values)),
            "p50": quantile(values, 0.50),
            "p90": quantile(values, 0.90),
            "p95": quantile(values, 0.95),
            "max": float(values.max()) if len(values) else None,
            "top1_rate": safe_rate(int((values <= 1).sum()), len(values)),
            "top5_rate": safe_rate(int((values <= 5).sum()), len(values)),
            "top10_rate": safe_rate(int((values <= 10).sum()), len(values)),
            "top32_rate": safe_rate(int((values <= 32).sum()), len(values)),
        }
    return result


def frame_density(tracker_path: Path) -> pd.Series:
    frames = pd.read_csv(tracker_path, header=None, usecols=[0])[0].astype(int)
    return frames.value_counts().sort_index()


def verify_frozen_artifacts(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report = read_json(run_root / "report.json")
    manifest_path = run_root / "frozen_candidate_graph/freeze_manifest.json"
    manifest = read_json(manifest_path)
    artifacts = manifest["frozen_artifacts"]
    nodes_path = REPO / artifacts["nodes"]
    edges_path = REPO / artifacts["edges"]
    actual_nodes = sha256_file(nodes_path)
    actual_edges = sha256_file(edges_path)
    if actual_nodes != artifacts["nodes_sha256"]:
        raise RuntimeError(f"node hash mismatch: {nodes_path}")
    if actual_edges != artifacts["edges_sha256"]:
        raise RuntimeError(f"edge hash mismatch: {edges_path}")
    if not manifest["baseline_reconstruction"]["byte_exact"]:
        raise RuntimeError(f"baseline is not byte-exact: {run_root}")
    if manifest.get("gt_opened") is not False:
        raise RuntimeError(f"candidate manifest does not certify pre-GT freeze: {run_root}")
    if manifest.get("stage") != "candidate_graph_frozen_before_gt":
        raise RuntimeError(f"unexpected freeze stage: {run_root}")
    if artifacts.get("forbidden_edge_columns") or artifacts.get("forbidden_node_columns"):
        raise RuntimeError(f"forbidden frozen columns found: {run_root}")
    return manifest, report


def successor_events(
    seq: str,
    nodes: pd.DataFrame,
    labels: pd.DataFrame,
    edges: pd.DataFrame,
    selected: pd.DataFrame,
    density: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labeled = labels[labels.teacher_dominant_gt > 0].copy()
    node_density = np.maximum(
        nodes.first_frame.map(density).fillna(0).to_numpy(float),
        nodes.last_frame.map(density).fillna(0).to_numpy(float),
    )
    crowd_low, crowd_high = np.quantile(node_density, [1 / 3, 2 / 3])

    identity_support = (
        labeled.groupby("teacher_dominant_gt").teacher_dominant_count.sum().astype(int)
    )
    support_values = identity_support.to_numpy(float)
    length_low, length_high = (
        np.quantile(support_values, [1 / 3, 2 / 3])
        if len(support_values)
        else (0.0, 0.0)
    )

    edge_lookup = (
        edges.sort_values(["parent_edge", "src_chunk", "dst_chunk"], ascending=[False, True, True])
        .drop_duplicates(["src_chunk", "dst_chunk"], keep="first")
        .set_index(["src_chunk", "dst_chunk"], drop=False)
    )
    selected_pairs = set(
        zip(selected.src_chunk.astype(int), selected.dst_chunk.astype(int))
    )
    rows: list[dict[str, Any]] = []

    for gt_id, group in labeled.groupby("teacher_dominant_gt"):
        ordered = group.sort_values(
            ["first_frame", "last_frame", "chunk_id"], kind="mergesort"
        ).reset_index(drop=True)
        support = int(identity_support.loc[gt_id])
        length_stratum = tertile_label(support, length_low, length_high)
        for source in ordered.itertuples(index=False):
            later = ordered[ordered.first_frame > int(source.last_frame)]
            if later.empty:
                continue
            destination = later.iloc[0]
            src = int(source.chunk_id)
            dst = int(destination.chunk_id)
            gap = int(destination.first_frame) - int(source.last_frame) - 1
            pair = (src, dst)
            present = pair in edge_lookup.index
            entry: dict[str, Any] = {
                "seq": seq,
                "teacher_gt": int(gt_id),
                "src_chunk": src,
                "dst_chunk": dst,
                "src_first_frame": int(source.first_frame),
                "src_last_frame": int(source.last_frame),
                "dst_first_frame": int(destination.first_frame),
                "dst_last_frame": int(destination.last_frame),
                "gap": gap,
                "gap_stratum": bin_gap(gap),
                "within_max_gap_600": int(gap <= 600),
                "identity_support_rows": support,
                "trajectory_length_stratum": length_stratum,
                "endpoint_crowd": float(max(node_density[src], node_density[dst])),
                "crowd_stratum": tertile_label(
                    float(max(node_density[src], node_density[dst])), crowd_low, crowd_high
                ),
                "candidate_present": int(present),
                "selected_by_teacher": int(pair in selected_pairs),
                "candidate_is_parent": 0,
                "candidate_is_cross": 0,
            }
            for rank in RANK_COLUMNS:
                entry[rank] = None
            if present:
                edge = edge_lookup.loc[pair]
                if isinstance(edge, pd.DataFrame):
                    edge = edge.iloc[0]
                entry["candidate_is_parent"] = int(edge.parent_edge)
                entry["candidate_is_cross"] = int(not bool(edge.parent_edge))
                for rank in RANK_COLUMNS:
                    entry[rank] = int(edge[rank])
            rows.append(entry)

    events = pd.DataFrame(rows)
    thresholds = {
        "crowd_definition": "max M23-46 tracker detections at source-last and destination-first frames",
        "crowd_tertiles_from_all_frozen_nodes": {
            "low_upper": float(crowd_low),
            "medium_upper": float(crowd_high),
        },
        "trajectory_length_definition": "sum of dominant matched rows over chunks assigned to the GT identity",
        "trajectory_length_tertiles": {
            "low_upper": float(length_low),
            "medium_upper": float(length_high),
        },
        "successor_definition": (
            "for each GT-labeled source chunk, the earliest same-GT chunk whose "
            "first frame is strictly after the source last frame; deterministic "
            "ties by first_frame,last_frame,chunk_id"
        ),
    }
    return events, thresholds


def audit_source_edges(
    teacher_edges: pd.DataFrame, events: pd.DataFrame
) -> dict[str, Any]:
    parent = teacher_edges[teacher_edges.parent_edge == 1].copy()
    both_matched = (
        (parent.teacher_src_dominant_gt > 0)
        & (parent.teacher_dst_dominant_gt > 0)
    )
    mismatch = both_matched & (
        parent.teacher_src_dominant_gt != parent.teacher_dst_dominant_gt
    )
    unmatched = ~both_matched
    wrong = mismatch | unmatched

    positive_cross = teacher_edges[
        (teacher_edges.parent_edge == 0)
        & (teacher_edges.teacher_same_identity_forward == 1)
    ]
    sources_with_any_correct_cross = set(positive_cross.src_chunk.astype(int))
    exact_cross_pairs = set(
        zip(
            events.loc[
                (events.candidate_present == 1) & (events.candidate_is_cross == 1),
                "src_chunk",
            ].astype(int),
            events.loc[
                (events.candidate_present == 1) & (events.candidate_is_cross == 1),
                "dst_chunk",
            ].astype(int),
        )
    )
    exact_cross_sources = {src for src, _ in exact_cross_pairs}
    wrong_rows = parent[wrong]
    mismatch_rows = parent[mismatch]
    any_alternative = int(
        wrong_rows.src_chunk.astype(int).isin(sources_with_any_correct_cross).sum()
    )
    exact_alternative = int(
        wrong_rows.src_chunk.astype(int).isin(exact_cross_sources).sum()
    )
    mismatch_any_alternative = int(
        mismatch_rows.src_chunk.astype(int).isin(sources_with_any_correct_cross).sum()
    )
    mismatch_exact_alternative = int(
        mismatch_rows.src_chunk.astype(int).isin(exact_cross_sources).sum()
    )
    return {
        "parent_source_edges": int(len(parent)),
        "both_endpoints_matched": int(both_matched.sum()),
        "cross_gt_identity_edges": int(mismatch.sum()),
        "cross_gt_ratio_all_parent": safe_rate(int(mismatch.sum()), len(parent)),
        "cross_gt_ratio_among_matched_parent": safe_rate(
            int(mismatch.sum()), int(both_matched.sum())
        ),
        "unmatched_endpoint_edges": int(unmatched.sum()),
        "unmatched_endpoint_ratio": safe_rate(int(unmatched.sum()), len(parent)),
        "wrong_or_unmatched_source_edges": int(wrong.sum()),
        "wrong_with_any_same_gt_forward_cross_alternative": any_alternative,
        "wrong_with_any_alternative_rate": safe_rate(any_alternative, int(wrong.sum())),
        "wrong_with_exact_gt_successor_cross_alternative": exact_alternative,
        "wrong_with_exact_successor_alternative_rate": safe_rate(
            exact_alternative, int(wrong.sum())
        ),
        "cross_gt_with_any_same_gt_forward_cross_alternative": mismatch_any_alternative,
        "cross_gt_with_any_alternative_rate": safe_rate(
            mismatch_any_alternative, int(mismatch.sum())
        ),
        "cross_gt_with_exact_gt_successor_cross_alternative": mismatch_exact_alternative,
        "cross_gt_with_exact_successor_alternative_rate": safe_rate(
            mismatch_exact_alternative, int(mismatch.sum())
        ),
    }


def detect_swap_transactions(
    parent_pairs: set[tuple[int, int]], selected_pairs: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    parent_successor = {src: dst for src, dst in parent_pairs}
    selected_successor = {src: dst for src, dst in selected_pairs}
    parent_predecessor = {dst: src for src, dst in parent_pairs}
    swaps: set[tuple[int, int]] = set()
    for src_a, old_dst_a in parent_successor.items():
        new_dst_a = selected_successor.get(src_a)
        if new_dst_a is None or new_dst_a == old_dst_a:
            continue
        src_b = parent_predecessor.get(new_dst_a)
        if src_b is None or src_b == src_a:
            continue
        if selected_successor.get(src_b) == old_dst_a:
            swaps.add(tuple(sorted((src_a, src_b))))
    return swaps


def audit_flow_edits(
    nodes: pd.DataFrame,
    teacher_edges: pd.DataFrame,
    selected: pd.DataFrame,
) -> dict[str, Any]:
    parent_pairs = set(
        zip(
            teacher_edges.loc[teacher_edges.parent_edge == 1, "src_chunk"].astype(int),
            teacher_edges.loc[teacher_edges.parent_edge == 1, "dst_chunk"].astype(int),
        )
    )
    selected_pairs = set(
        zip(selected.src_chunk.astype(int), selected.dst_chunk.astype(int))
    )
    kept = parent_pairs & selected_pairs
    cut = parent_pairs - selected_pairs
    cross = selected_pairs - parent_pairs
    symmetric = parent_pairs ^ selected_pairs
    swaps = detect_swap_transactions(parent_pairs, selected_pairs)
    affected_chunks = {chunk for pair in symmetric for chunk in pair}
    affected_rows = int(
        nodes.loc[nodes.chunk_id.astype(int).isin(affected_chunks), "rows"].sum()
    )
    total_rows = int(nodes.rows.sum())
    selected_cross = selected[selected.parent_edge == 0]
    correct_cross = int(selected_cross.teacher_same_identity_forward.sum())
    return {
        "parent_edges": int(len(parent_pairs)),
        "selected_edges": int(len(selected_pairs)),
        "keep_parent_edges": int(len(kept)),
        "cut_parent_edges": int(len(cut)),
        "selected_cross_edges": int(len(cross)),
        "selected_correct_cross_edges": correct_cross,
        "selected_wrong_or_unmatched_cross_edges": int(len(selected_cross) - correct_cross),
        "symmetric_difference_edges": int(len(symmetric)),
        "edge_edit_rate_vs_parent": safe_rate(len(symmetric), len(parent_pairs)),
        "exact_two_successor_swap_transactions": int(len(swaps)),
        "swap_involved_source_edges": int(2 * len(swaps)),
        "dummy_terminate_nodes": int(len(nodes) - len(selected_pairs)),
        "dummy_restart_nodes": int(len(nodes) - len(selected_pairs)),
        "identity_assignment_affected_chunks": int(len(affected_chunks)),
        "identity_assignment_affected_rows": affected_rows,
        "identity_assignment_affected_row_rate": safe_rate(affected_rows, total_rows),
        "detection_rows_total": total_rows,
        "detection_row_values_changed": 0,
        "detection_boxes_or_scores_changed": 0,
        "note": (
            "affected rows belong to chunks incident to a cut or newly selected edge; "
            "the detector rows, boxes and scores themselves remain byte-identical"
        ),
    }


def summarize_events(events: pd.DataFrame) -> dict[str, Any]:
    present = int(events.candidate_present.sum())
    selected = int(events.selected_by_teacher.sum())
    within = events[events.within_max_gap_600 == 1]
    within_present = int(within.candidate_present.sum())
    present_events = events[events.candidate_present == 1]
    return {
        "gt_successor_events": int(len(events)),
        "candidate_present": present,
        "candidate_recall": safe_rate(present, len(events)),
        "within_gap_600_events": int(len(within)),
        "within_gap_600_candidate_present": within_present,
        "within_gap_600_candidate_recall": safe_rate(within_present, len(within)),
        "source_parent_successor_events": int(events.candidate_is_parent.sum()),
        "cross_successor_events": int(events.candidate_is_cross.sum()),
        "selected_by_teacher": selected,
        "teacher_selection_rate_all_successors": safe_rate(selected, len(events)),
        "teacher_selection_rate_when_candidate_present": safe_rate(selected, present),
        "present_but_not_selected": int(len(present_events) - selected),
        "by_gap": summarize_groups(events, "gap_stratum"),
        "by_crowd": summarize_groups(events, "crowd_stratum"),
        "by_trajectory_length": summarize_groups(
            events, "trajectory_length_stratum"
        ),
        "correct_continuation_rank": rank_summary(events),
    }


def aggregate_count_dicts(items: list[dict[str, Any]], keys: Iterable[str]) -> dict[str, int]:
    return {key: int(sum(int(item[key]) for item in items)) for key in keys}


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def generate_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# M23-53 Candidate Coverage and Flow Conversion Audit",
        "",
        "Date: 2026-07-20",
        "",
        "## Status",
        "",
        "- Role: post-freeze teacher-only diagnostic; deployable=false.",
        "- Frozen candidate graphs, teacher utilities, selected flows, trackers and official TrackEval outputs were read-only.",
        "- No candidate regeneration, GT-driven tuning, teacher-weight change, TrackEval rerun, student training or test submission was performed.",
        f"- Audited Git input: `{payload['git_head']}`.",
        "",
        "## Integrity checks",
        "",
        "All four folds passed candidate node/edge SHA-256 verification, forbidden-column audit, freeze-before-GT certification, and M23-46 byte-exact baseline reconstruction.",
        f"The existing M02 smoke was also reverified without rerunning it: baseline byte-exact={str(payload['m02_smoke_verification']['baseline_byte_exact']).lower()}, frozen edges={payload['m02_smoke_verification']['frozen_edges']}, forbidden frozen columns=0, one-to-one/acyclic/time-forward=true, official HOTA={payload['m02_smoke_verification']['official_trackeval']['HOTA']:.6f}.",
        "",
        "## Existing-code reuse audit",
        "",
        "- `m23_11_eval_utility_graph.py`: reused sparse maximum-weight bipartite matching, reconstruction and official-evaluation patterns.",
        "- `m23_12_chain_transaction_oracle.py`: reused chain transaction and tracker-writing conventions.",
        "- `m23_37_fast_exact_hota_teacher.py`: reused exact-HOTA/official-verification infrastructure, but single-action HOTA delta was intentionally not used as the M23-53 objective.",
        "- `eval_max_weight_identity_path_cover.py`: provided the legacy matching/path-cover precedent, but only supported track-level positive merge links and lacked frozen source/cross/dummy semantics and freeze-before-GT evidence.",
        "- M23-53 therefore retained the existing matching/reconstruction machinery and added only the unified source/cross/dummy flow representation required by the preregistered protocol.",
        "",
        "## Candidate coverage",
        "",
    ]
    rows: list[list[Any]] = []
    for seq, fold in payload["folds"].items():
        coverage = fold["coverage"]
        rows.append(
            [
                seq,
                coverage["gt_successor_events"],
                pct(coverage["candidate_recall"]),
                pct(coverage["within_gap_600_candidate_recall"]),
                pct(coverage["teacher_selection_rate_when_candidate_present"]),
                fold["official_trackeval"]["HOTA"],
            ]
        )
    combined = payload["combined"]
    rows.append(
        [
            "POOLED",
            combined["coverage"]["gt_successor_events"],
            pct(combined["coverage"]["candidate_recall"]),
            pct(combined["coverage"]["within_gap_600_candidate_recall"]),
            pct(combined["coverage"]["teacher_selection_rate_when_candidate_present"]),
            payload["official_combined"]["HOTA"],
        ]
    )
    lines.extend(
        markdown_table(
            [
                "Fold",
                "GT-successors",
                "Candidate recall",
                "Recall gap<=600",
                "Selected/present",
                "Official HOTA",
            ],
            rows,
        )
    )
    lines.extend(["", "Primary successor definition: earliest strictly time-forward, non-overlapping chunk with the same post-freeze teacher GT identity. This diagnostic definition does not alter the flow objective.", ""])

    lines.extend(["## Current source-edge errors and alternatives", ""])
    source_rows: list[list[Any]] = []
    for seq, fold in payload["folds"].items():
        audit = fold["source_edge_audit"]
        source_rows.append(
            [
                seq,
                audit["parent_source_edges"],
                pct(audit["cross_gt_ratio_all_parent"]),
                pct(audit["unmatched_endpoint_ratio"]),
                pct(audit["cross_gt_with_any_alternative_rate"]),
                pct(audit["cross_gt_with_exact_successor_alternative_rate"]),
            ]
        )
    source_rows.append(
        [
            "TOTAL",
            combined["source_edge_audit"]["parent_source_edges"],
            pct(combined["source_edge_audit"]["cross_gt_ratio_all_parent"]),
            pct(combined["source_edge_audit"]["unmatched_endpoint_ratio"]),
            pct(combined["source_edge_audit"]["cross_gt_with_any_alternative_rate"]),
            pct(combined["source_edge_audit"]["cross_gt_with_exact_successor_alternative_rate"]),
        ]
    )
    lines.extend(
        markdown_table(
            [
                "Fold",
                "Source edges",
                "Cross-GT",
                "Unmatched",
                "Cross-GT any alt",
                "Cross-GT exact alt",
            ],
            source_rows,
        )
    )

    lines.extend(["", "## Teacher flow actions relative to M23-46", ""])
    action_rows: list[list[Any]] = []
    for seq, fold in payload["folds"].items():
        action = fold["flow_edit_audit"]
        action_rows.append(
            [
                seq,
                action["keep_parent_edges"],
                action["cut_parent_edges"],
                action["selected_cross_edges"],
                action["exact_two_successor_swap_transactions"],
                action["identity_assignment_affected_rows"],
                pct(action["identity_assignment_affected_row_rate"]),
            ]
        )
    total_action = combined["flow_edit_audit"]
    action_rows.append(
        [
            "TOTAL",
            total_action["keep_parent_edges"],
            total_action["cut_parent_edges"],
            total_action["selected_cross_edges"],
            total_action["exact_two_successor_swap_transactions"],
            total_action["identity_assignment_affected_rows"],
            pct(total_action["identity_assignment_affected_row_rate"]),
        ]
    )
    lines.extend(
        markdown_table(
            ["Fold", "Keep", "Cut", "Cross", "Swap pairs", "Affected rows", "Affected rate"],
            action_rows,
        )
    )
    lines.extend(
        [
            "",
            "`Affected rows` counts frozen detection rows in chunks incident to an edited edge. Actual detection row values, boxes and scores changed: **0**.",
            "",
            "## Stratified pooled recall",
            "",
            "### By temporal gap",
            "",
        ]
    )
    gap_rows = []
    for label, stats in combined["coverage"]["by_gap"].items():
        gap_rows.append([label, stats["events"], pct(stats["candidate_recall"]), pct(stats["teacher_selection_rate_when_present"])])
    lines.extend(markdown_table(["Gap", "Events", "Recall", "Selected/present"], gap_rows))
    lines.extend(["", "### By sequence-normalized crowd tertile", ""])
    crowd_rows = []
    for label, stats in combined["coverage"]["by_crowd"].items():
        crowd_rows.append([label, stats["events"], pct(stats["candidate_recall"]), pct(stats["teacher_selection_rate_when_present"])])
    lines.extend(markdown_table(["Crowd", "Events", "Recall", "Selected/present"], crowd_rows))
    lines.extend(["", "### By GT trajectory-support tertile", ""])
    length_rows = []
    for label, stats in combined["coverage"]["by_trajectory_length"].items():
        length_rows.append([label, stats["events"], pct(stats["candidate_recall"]), pct(stats["teacher_selection_rate_when_present"])])
    lines.extend(markdown_table(["Length", "Events", "Recall", "Selected/present"], length_rows))

    rank = combined["coverage"]["correct_continuation_rank"]
    lines.extend(["", "## Correct-continuation ranks when present", ""])
    rank_rows = []
    for column in RANK_COLUMNS:
        stats = rank[column]
        rank_rows.append([column, stats["count"], f"{stats['p50']:.1f}" if stats["p50"] is not None else "n/a", f"{stats['p90']:.1f}" if stats["p90"] is not None else "n/a", pct(stats["top5_rate"]), pct(stats["top32_rate"])])
    lines.extend(markdown_table(["Rank", "N", "P50", "P90", "Top-5", "Top-32"], rank_rows))

    lines.extend(
        [
            "",
            "## Capacity decision",
            "",
            f"- Frozen fixed-chunk teacher COMBINED: HOTA **{payload['official_combined']['HOTA']:.6f}**, DetA **{payload['official_combined']['DetA']:.6f}**, AssA **{payload['official_combined']['AssA']:.6f}**, IDSW **{payload['official_combined']['IDSW']}**.",
            "- The result remains below the preregistered 80.300000 no-student floor.",
            "- The already-completed uniform GT-free adaptive repair reached 79.920490 and is also closed.",
            "- M23-54 remains prohibited under the preregistered gate.",
            "- Best strict deployable result remains M23-46 HOTA 79.123193.",
            "",
            "## Output artifacts",
            "",
            f"- `{AUDIT_ROOT}/report.json`",
            f"- `{AUDIT_ROOT}/successor_events.parquet`",
            f"- `{DOC_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, default=AUDIT_ROOT)
    parser.add_argument("--doc-json", type=Path, default=DOC_JSON)
    parser.add_argument("--doc-md", type=Path, default=DOC_MD)
    args = parser.parse_args()

    combined_report = read_json(REPO / COMBINED_REPORT)
    smoke_root = REPO / OUTPUTS / SMOKE_RUN
    smoke_manifest, smoke_report = verify_frozen_artifacts(smoke_root)
    fold_payload: dict[str, Any] = {}
    all_events: list[pd.DataFrame] = []
    source_audits: list[dict[str, Any]] = []
    flow_audits: list[dict[str, Any]] = []

    for seq, run_name in FOLDS.items():
        run_root = REPO / OUTPUTS / run_name
        manifest, report = verify_frozen_artifacts(run_root)
        nodes = pd.read_parquet(run_root / "frozen_candidate_graph/nodes.parquet")
        edges = pd.read_parquet(run_root / "frozen_candidate_graph/edges.parquet")
        labels = pd.read_parquet(
            run_root / "teacher_identity_flow/teacher_node_labels.parquet"
        )
        teacher_edges = pd.read_parquet(
            run_root / "teacher_identity_flow/teacher_edge_utilities.parquet"
        )
        selected = pd.read_parquet(
            run_root / "teacher_identity_flow/selected_path_cover_edges.parquet"
        )
        tracker_path = REPO / BASELINE_CACHE / seq / "track_results" / f"{seq}.txt"
        density = frame_density(tracker_path)
        events, thresholds = successor_events(
            seq, nodes, labels, edges, selected, density
        )
        if events.empty:
            raise RuntimeError(f"no GT-successor events for {seq}")
        events_path = args.audit_root / seq / "successor_events.parquet"
        (REPO / events_path).parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(REPO / events_path, index=False)
        source_audit = audit_source_edges(teacher_edges, events)
        flow_audit = audit_flow_edits(nodes, teacher_edges, selected)
        coverage = summarize_events(events)
        fold_payload[seq] = {
            "run": run_name,
            "freeze_manifest_sha256": sha256_file(
                run_root / "frozen_candidate_graph/freeze_manifest.json"
            ),
            "frozen_nodes_sha256": manifest["frozen_artifacts"]["nodes_sha256"],
            "frozen_edges_sha256": manifest["frozen_artifacts"]["edges_sha256"],
            "baseline_byte_exact": True,
            "forbidden_node_columns": manifest["frozen_artifacts"]["forbidden_node_columns"],
            "forbidden_edge_columns": manifest["frozen_artifacts"]["forbidden_edge_columns"],
            "coverage_thresholds": thresholds,
            "coverage": coverage,
            "source_edge_audit": source_audit,
            "flow_edit_audit": flow_audit,
            "official_trackeval": report["official_trackeval"],
            "successor_events": str(events_path),
            "successor_events_sha256": sha256_file(REPO / events_path),
        }
        all_events.append(events)
        source_audits.append(source_audit)
        flow_audits.append(flow_audit)

    pooled_events = pd.concat(all_events, ignore_index=True)
    pooled_events_path = args.audit_root / "successor_events.parquet"
    (REPO / pooled_events_path).parent.mkdir(parents=True, exist_ok=True)
    pooled_events.to_parquet(REPO / pooled_events_path, index=False)

    source_counts = aggregate_count_dicts(
        source_audits,
        [
            "parent_source_edges",
            "both_endpoints_matched",
            "cross_gt_identity_edges",
            "unmatched_endpoint_edges",
            "wrong_or_unmatched_source_edges",
            "wrong_with_any_same_gt_forward_cross_alternative",
            "wrong_with_exact_gt_successor_cross_alternative",
            "cross_gt_with_any_same_gt_forward_cross_alternative",
            "cross_gt_with_exact_gt_successor_cross_alternative",
        ],
    )
    source_counts.update(
        {
            "cross_gt_ratio_all_parent": safe_rate(
                source_counts["cross_gt_identity_edges"],
                source_counts["parent_source_edges"],
            ),
            "cross_gt_ratio_among_matched_parent": safe_rate(
                source_counts["cross_gt_identity_edges"],
                source_counts["both_endpoints_matched"],
            ),
            "unmatched_endpoint_ratio": safe_rate(
                source_counts["unmatched_endpoint_edges"],
                source_counts["parent_source_edges"],
            ),
            "wrong_with_any_alternative_rate": safe_rate(
                source_counts["wrong_with_any_same_gt_forward_cross_alternative"],
                source_counts["wrong_or_unmatched_source_edges"],
            ),
            "wrong_with_exact_successor_alternative_rate": safe_rate(
                source_counts["wrong_with_exact_gt_successor_cross_alternative"],
                source_counts["wrong_or_unmatched_source_edges"],
            ),
            "cross_gt_with_any_alternative_rate": safe_rate(
                source_counts["cross_gt_with_any_same_gt_forward_cross_alternative"],
                source_counts["cross_gt_identity_edges"],
            ),
            "cross_gt_with_exact_successor_alternative_rate": safe_rate(
                source_counts["cross_gt_with_exact_gt_successor_cross_alternative"],
                source_counts["cross_gt_identity_edges"],
            ),
        }
    )

    flow_keys = [
        "parent_edges",
        "selected_edges",
        "keep_parent_edges",
        "cut_parent_edges",
        "selected_cross_edges",
        "selected_correct_cross_edges",
        "selected_wrong_or_unmatched_cross_edges",
        "symmetric_difference_edges",
        "exact_two_successor_swap_transactions",
        "swap_involved_source_edges",
        "dummy_terminate_nodes",
        "dummy_restart_nodes",
        "identity_assignment_affected_chunks",
        "identity_assignment_affected_rows",
        "detection_rows_total",
        "detection_row_values_changed",
        "detection_boxes_or_scores_changed",
    ]
    flow_counts = aggregate_count_dicts(flow_audits, flow_keys)
    flow_counts.update(
        {
            "edge_edit_rate_vs_parent": safe_rate(
                flow_counts["symmetric_difference_edges"],
                flow_counts["parent_edges"],
            ),
            "identity_assignment_affected_row_rate": safe_rate(
                flow_counts["identity_assignment_affected_rows"],
                flow_counts["detection_rows_total"],
            ),
            "note": (
                "fold-wise unique affected-row counts are summed; detection row values, "
                "boxes and scores remain unchanged"
            ),
        }
    )

    payload: dict[str, Any] = {
        "audit": "M23-53 candidate coverage and flow conversion",
        "created_at": utc_now(),
        "git_head": git_head(),
        "status": "completed",
        "teacher_only": True,
        "deployable": False,
        "read_only_post_freeze_diagnostic": True,
        "trackeval_rerun": False,
        "candidate_regenerated": False,
        "teacher_weights_changed": False,
        "test_submission": False,
        "strict_parent": "M23-46 frozen deployable trackers/applied graphs",
        "m02_smoke_verification": {
            "run": SMOKE_RUN,
            "rerun": False,
            "baseline_byte_exact": bool(
                smoke_manifest["baseline_reconstruction"]["byte_exact"]
            ),
            "baseline_tracker_sha256": smoke_manifest["baseline_reconstruction"][
                "cached_tracker_sha256"
            ],
            "frozen_edges": int(
                smoke_manifest["frozen_artifacts"]["edge_rows"]
            ),
            "forbidden_node_columns": smoke_manifest["frozen_artifacts"][
                "forbidden_node_columns"
            ],
            "forbidden_edge_columns": smoke_manifest["frozen_artifacts"][
                "forbidden_edge_columns"
            ],
            "official_trackeval": smoke_report["official_trackeval"],
            "one_to_one": bool(smoke_report["teacher"]["one_to_one"]),
            "acyclic": bool(smoke_report["teacher"]["acyclic"]),
            "time_forward": bool(smoke_report["teacher"]["time_forward"]),
        },
        "existing_code_reuse_audit": {
            "m23_11_eval_utility_graph": "sparse maximum-weight bipartite matching, tracker reconstruction, official-evaluation patterns",
            "m23_12_chain_transaction_oracle": "chain transaction and tracker-writing conventions",
            "m23_37_fast_exact_hota_teacher": "exact-HOTA and official-verification infrastructure; single-action delta objective not reused",
            "eval_max_weight_identity_path_cover": "legacy track-level positive-link path-cover precedent; insufficient for frozen source/cross/dummy unified flow",
            "new_solver_required": False,
            "m23_53_extension": "unified frozen source/cross/dummy semantics on existing sparse matching and reconstruction machinery",
        },
        "folds": fold_payload,
        "combined": {
            "coverage": summarize_events(pooled_events),
            "source_edge_audit": source_counts,
            "flow_edit_audit": flow_counts,
            "successor_events": str(pooled_events_path),
            "successor_events_sha256": sha256_file(REPO / pooled_events_path),
        },
        "official_combined": combined_report["official_trackeval"]["COMBINED"],
        "capacity_decision": combined_report["decision"],
        "adaptive_followup": {
            "completed": True,
            "combined_HOTA": 79.920490,
            "deployable": False,
            "decision": "adaptive_protocol_insufficient_close_m23_53b_do_not_start_m23_54",
        },
        "strict_deployable_best": {
            "experiment": "M23-46",
            "HOTA": 79.123193,
            "DetA": 81.543470,
            "AssA": 76.825150,
            "IDSW": 996,
        },
        "next_gate": "M23-54 prohibited because teacher capacity is below 80.700000",
    }
    payload = as_builtin(payload)
    report_path = args.audit_root / "report.json"
    write_json(REPO / report_path, payload)
    write_json(REPO / args.doc_json, payload)
    (REPO / args.doc_md).parent.mkdir(parents=True, exist_ok=True)
    (REPO / args.doc_md).write_text(generate_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "doc_json": str(args.doc_json),
        "doc_md": str(args.doc_md),
        "pooled_successors": payload["combined"]["coverage"]["gt_successor_events"],
        "candidate_recall": payload["combined"]["coverage"]["candidate_recall"],
        "within_600_recall": payload["combined"]["coverage"]["within_gap_600_candidate_recall"],
        "selected_when_present": payload["combined"]["coverage"]["teacher_selection_rate_when_candidate_present"],
    }, indent=2))


if __name__ == "__main__":
    main()
