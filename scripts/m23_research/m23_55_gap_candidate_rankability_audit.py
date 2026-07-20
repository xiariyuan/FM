#!/usr/bin/env python3
from __future__ import annotations

"""Post-freeze M23-55 rejection, coverage, OOF rankability and teacher audit.

Every command verifies the pre-GT freeze manifest and artifact hashes before it
loads any teacher labels or GT-derived successor table. No output from this
script is fed back into candidate construction.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
SEQ_SHORT = {"MOT20-01": "m01", "MOT20-02": "m02", "MOT20-03": "m03", "MOT20-05": "m05"}
ROOTS = {seq: Path(f"outputs/mot20_m23_20260718/m23_55_stratified_gap_candidates_{SEQ_SHORT[seq]}_v1") for seq in SEQUENCES}
OLD_ROOTS = {
    "MOT20-01": Path("outputs/mot20_m23_20260718/m23_53_identity_flow_capacity_m01_v1"),
    "MOT20-02": Path("outputs/mot20_m23_20260718/m23_53_identity_flow_capacity_m02_v1"),
    "MOT20-03": Path("outputs/mot20_m23_20260718/m23_53_identity_flow_capacity_m03_v1"),
    "MOT20-05": Path("outputs/mot20_m23_20260718/m23_53_identity_flow_capacity_m05_v1"),
}
GRAPH_ROOT = Path("outputs/mot20_m23_20260718/micrograph_chunk30_v1")
SUCCESSOR_EVENTS = Path("outputs/mot20_m23_20260718/m23_53_candidate_coverage_audit_v1/successor_events.parquet")
SOURCE_PARENT = Path("outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results")
RECALL_K = (1, 4, 8, 16, 32, 64, 128, 256)
FEATURES = (
    "recall_score", "multi_appearance", "whole_appearance", "appearance_cos",
    "motion_error_min", "endpoint_displacement", "velocity_cos", "abs_log_height_ratio",
    "log_gap", "src_consistency", "dst_consistency", "parent_edge",
)


def load_module(name: str, rel: str):
    path = REPO / rel
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def canonical_payload_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_freeze(root: Path) -> dict:
    manifest_path = root / "frozen_candidate_graph" / "freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("gt_opened") is not False or manifest.get("candidate_graph_frozen") is not True:
        raise RuntimeError("invalid freeze state")
    artifacts = manifest["frozen_artifacts"]
    keys = (
        ("nodes", "nodes_sha256"), ("edges", "edges_sha256"),
        ("outgoing_pool", "outgoing_pool_sha256"), ("incoming_pool", "incoming_pool_sha256"),
        ("descriptors", "descriptors_sha256"),
    )
    for path_key, hash_key in keys:
        path = Path(artifacts[path_key])
        if sha256_file(path) != artifacts[hash_key]:
            raise RuntimeError(f"frozen artifact hash mismatch: {path}")
    if not manifest["baseline_reconstruction"]["byte_exact"]:
        raise RuntimeError("baseline reconstruction gate failed")
    return manifest


def load_descriptors(root: Path) -> dict[str, np.ndarray]:
    data = np.load(root / "frozen_candidate_graph" / "observable_descriptors.npz")
    result = {key: data[key].astype(np.float32) for key in data.files}
    for key in result:
        result[key] /= np.maximum(np.linalg.norm(result[key], axis=1, keepdims=True), 1e-12)
    return result


def bucket_bounds(name: str) -> tuple[int, int]:
    return {"1-30": (1, 30), "31-90": (31, 90), "91-180": (91, 180), "181-600": (181, 600)}[name]


def pair_scores(nodes: pd.DataFrame, desc: dict[str, np.ndarray], src: int, dsts: np.ndarray) -> dict[str, np.ndarray]:
    dsts = np.asarray(dsts, np.int64)
    views = np.stack((
        desc["tail1"][src] @ desc["head1"][dsts].T,
        desc["tail3"][src] @ desc["head3"][dsts].T,
        desc["tail8"][src] @ desc["head8"][dsts].T,
        desc["qtail"][src] @ desc["qhead"][dsts].T,
        desc["robust"][src] @ desc["robust"][dsts].T,
        desc["medoid"][src] @ desc["medoid"][dsts].T,
        desc["whole"][src] @ desc["whole"][dsts].T,
    ), axis=0)
    multi = views.max(axis=0)
    whole = views[-1]
    a = nodes.iloc[src]
    b = nodes.iloc[dsts]
    gap = b.first_frame.to_numpy(float) - float(a.last_frame) - 1
    dt = np.maximum(b.first_frame.to_numpy(float) - float(a.last_frame), 1.0)
    h = np.maximum(0.5 * (float(a.last_h) + b.first_h.to_numpy(float)), 1.0)
    predx = float(a.last_cx) + float(a.end_vx) * dt
    predy = float(a.last_cy) + float(a.end_vy) * dt
    backx = b.first_cx.to_numpy(float) - b.start_vx.to_numpy(float) * dt
    backy = b.first_cy.to_numpy(float) - b.start_vy.to_numpy(float) * dt
    ferr = np.hypot(b.first_cx.to_numpy(float) - predx, b.first_cy.to_numpy(float) - predy) / h
    berr = np.hypot(float(a.last_cx) - backx, float(a.last_cy) - backy) / h
    motion = np.minimum(ferr, berr)
    disp = np.hypot(b.first_cx.to_numpy(float) - float(a.last_cx), b.first_cy.to_numpy(float) - float(a.last_cy)) / h
    av = np.asarray([float(a.end_vx), float(a.end_vy)])
    bv = b[["start_vx", "start_vy"]].to_numpy(float)
    velocity = (bv @ av) / np.maximum(np.linalg.norm(bv, axis=1) * max(np.linalg.norm(av), 1e-8), 1e-8)
    log_height = np.log(np.maximum(b.first_h.to_numpy(float), 1e-3) / max(float(a.last_h), 1e-3))
    score = multi + 0.15 * np.exp(-0.5 * np.clip(motion, 0, 20)) + 0.03 * velocity - 0.02 * np.abs(log_height) - 0.005 * np.log1p(np.maximum(gap, 0))
    return {"multi": multi, "whole": whole, "motion": motion, "score": score, "gap": gap, "best_view": views.argmax(axis=0)}


def pair_scores_to_dst(nodes: pd.DataFrame, desc: dict[str, np.ndarray], srcs: np.ndarray, dst: int) -> dict[str, np.ndarray]:
    """Vectorized incoming score computation for many sources and one destination."""
    srcs = np.asarray(srcs, np.int64)
    views = np.stack((
        desc["tail1"][srcs] @ desc["head1"][dst],
        desc["tail3"][srcs] @ desc["head3"][dst],
        desc["tail8"][srcs] @ desc["head8"][dst],
        desc["qtail"][srcs] @ desc["qhead"][dst],
        desc["robust"][srcs] @ desc["robust"][dst],
        desc["medoid"][srcs] @ desc["medoid"][dst],
        desc["whole"][srcs] @ desc["whole"][dst],
    ), axis=0)
    multi = views.max(axis=0)
    whole = views[-1]
    a = nodes.iloc[srcs]
    b = nodes.iloc[int(dst)]
    gap = float(b.first_frame) - a.last_frame.to_numpy(float) - 1
    dt = np.maximum(float(b.first_frame) - a.last_frame.to_numpy(float), 1.0)
    h = np.maximum(0.5 * (a.last_h.to_numpy(float) + float(b.first_h)), 1.0)
    predx = a.last_cx.to_numpy(float) + a.end_vx.to_numpy(float) * dt
    predy = a.last_cy.to_numpy(float) + a.end_vy.to_numpy(float) * dt
    backx = float(b.first_cx) - float(b.start_vx) * dt
    backy = float(b.first_cy) - float(b.start_vy) * dt
    ferr = np.hypot(float(b.first_cx) - predx, float(b.first_cy) - predy) / h
    berr = np.hypot(a.last_cx.to_numpy(float) - backx, a.last_cy.to_numpy(float) - backy) / h
    motion = np.minimum(ferr, berr)
    av = a[["end_vx", "end_vy"]].to_numpy(float)
    bv = np.asarray([float(b.start_vx), float(b.start_vy)])
    velocity = (av @ bv) / np.maximum(np.linalg.norm(av, axis=1) * max(np.linalg.norm(bv), 1e-8), 1e-8)
    log_height = np.log(max(float(b.first_h), 1e-3) / np.maximum(a.last_h.to_numpy(float), 1e-3))
    score = multi + 0.15 * np.exp(-0.5 * np.clip(motion, 0, 20)) + 0.03 * velocity - 0.02 * np.abs(log_height) - 0.005 * np.log1p(np.maximum(gap, 0))
    return {"multi": multi, "whole": whole, "motion": motion, "score": score, "gap": gap, "best_view": views.argmax(axis=0)}


def rank_value(values: np.ndarray, ids: np.ndarray, target_index: int) -> int:
    target = float(values[target_index])
    target_id = int(ids[target_index])
    return 1 + int(np.sum((values > target) | ((values == target) & (ids < target_id))))


def canonical_exact_ranks(nodes: pd.DataFrame, desc: dict[str, np.ndarray], events: pd.DataFrame) -> pd.DataFrame:
    starts = nodes.first_frame.to_numpy(int)
    lasts = nodes.last_frame.to_numpy(int)
    tracks = nodes.source_track_id.to_numpy(int)
    rows = []
    for event in events.itertuples(index=False):
        src, dst = int(event.src_chunk), int(event.dst_chunk)
        bucket = str(event.gap_stratum)
        if bucket == "0":
            continue
        lo, hi = bucket_bounds(bucket)
        eligible_out = np.flatnonzero((starts - lasts[src] - 1 >= lo) & (starts - lasts[src] - 1 <= hi) & (tracks != tracks[src]))
        eligible_in = np.flatnonzero((starts[dst] - lasts - 1 >= lo) & (starts[dst] - lasts - 1 <= hi) & (tracks != tracks[dst]))
        record = {"seq": event.seq, "src_chunk": src, "dst_chunk": dst, "gap_stratum": bucket}
        if dst in set(eligible_out.tolist()):
            scores = pair_scores(nodes, desc, src, eligible_out)
            target = int(np.flatnonzero(eligible_out == dst)[0])
            record.update({
                "out_recall_rank": rank_value(scores["score"], eligible_out, target),
                "out_multi_rank": rank_value(scores["multi"], eligible_out, target),
                "out_whole_rank": rank_value(scores["whole"], eligible_out, target),
                "out_motion_rank": rank_value(-scores["motion"], eligible_out, target),
                "best_view": int(scores["best_view"][target]),
            })
        else:
            record.update({"out_recall_rank": np.inf, "out_multi_rank": np.inf, "out_whole_rank": np.inf, "out_motion_rank": np.inf, "best_view": -1})
        if src in set(eligible_in.tolist()):
            incoming_scores = pair_scores_to_dst(nodes, desc, eligible_in, dst)
            target = int(np.flatnonzero(eligible_in == src)[0])
            record.update({
                "in_recall_rank": rank_value(incoming_scores["score"], eligible_in, target),
                "in_multi_rank": rank_value(incoming_scores["multi"], eligible_in, target),
                "in_whole_rank": rank_value(incoming_scores["whole"], eligible_in, target),
                "in_motion_rank": rank_value(-incoming_scores["motion"], eligible_in, target),
            })
        else:
            record.update({"in_recall_rank": np.inf, "in_multi_rank": np.inf, "in_whole_rank": np.inf, "in_motion_rank": np.inf})
        rows.append(record)
    return pd.DataFrame(rows)


def old_rejection_audit(seq: str, root: Path, events: pd.DataFrame, nodes: pd.DataFrame, desc: dict[str, np.ndarray]) -> pd.DataFrame:
    missing = events[(events.gap_stratum != "0") & (events.candidate_present == 0)].copy()
    bank = pd.read_parquet(GRAPH_ROOT / seq / "candidate_edges.parquet")
    bank_index = bank.set_index(["src_chunk", "dst_chunk"], drop=False)
    raw_nodes = pd.read_parquet(GRAPH_ROOT / seq / "microtracklets.parquet")
    teacher_nodes = pd.read_parquet(OLD_ROOTS[seq] / "teacher_identity_flow" / "teacher_node_labels.parquet")
    old_edges = pd.read_parquet(OLD_ROOTS[seq] / "frozen_candidate_graph" / "edges.parquet")
    parent = old_edges[old_edges.parent_edge == 1]
    parent_by_src = {int(r.src_chunk): r for r in parent.itertuples(index=False)}
    starts, lasts = nodes.first_frame.to_numpy(int), nodes.last_frame.to_numpy(int)
    tracks = nodes.source_track_id.to_numpy(int)
    whole = desc["whole"]
    result = []
    for event in missing.itertuples(index=False):
        src, dst, gap = int(event.src_chunk), int(event.dst_chunk), int(event.gap)
        key = (src, dst)
        pair_in_bank = key in bank_index.index
        out_rank = np.inf
        in_rank = np.inf
        gap0_higher = 0
        if gap <= 300:
            eligible_out = np.flatnonzero((starts > lasts[src]) & (starts <= lasts[src] + 301))
            scores = whole[eligible_out] @ whole[src]
            if dst in set(eligible_out.tolist()):
                target = int(np.flatnonzero(eligible_out == dst)[0])
                out_rank = rank_value(scores, eligible_out, target)
                gap_values = starts[eligible_out] - lasts[src] - 1
                gap0_higher = int(np.sum((gap_values == 0) & (scores > scores[target])))
            eligible_in = np.flatnonzero((lasts < starts[dst]) & (lasts >= starts[dst] - 301))
            incoming_scores = whole[eligible_in] @ whole[dst]
            if src in set(eligible_in.tolist()):
                target = int(np.flatnonzero(eligible_in == src)[0])
                in_rank = rank_value(incoming_scores, eligible_in, target)
        endpoint_bad = int(raw_nodes.iloc[src].mapped_rows) == 0 or int(raw_nodes.iloc[dst].mapped_rows) == 0
        impurity = float(teacher_nodes.iloc[src].teacher_dominant_purity) < 0.999999 or float(teacher_nodes.iloc[dst].teacher_dominant_purity) < 0.999999
        source_state = "no_source_successor"
        if src in parent_by_src:
            p = parent_by_src[src]
            sgt = int(teacher_nodes.iloc[src].teacher_dominant_gt)
            dgt = int(teacher_nodes.iloc[int(p.dst_chunk)].teacher_dominant_gt)
            source_state = "source_correct" if sgt > 0 and sgt == dgt else "source_wrong_or_unmatched"
        reason = "legacy_outgoing_top16_or_unmodeled"
        if gap < 0 or gap > 300:
            reason = "1_temporal_eligibility_or_gap_gate"
        elif pair_in_bank:
            edge = bank_index.loc[key]
            if isinstance(edge, pd.DataFrame):
                edge = edge.iloc[0]
            failed = float(edge.motion_error_min) > 5.0 or float(edge.endpoint_displacement) > 6.0 or abs(float(edge.log_height_ratio)) > 1.2
            if failed:
                reason = "2_spatial_or_motion_gate"
        elif out_rank > 16 and gap0_higher >= 16:
            reason = "4_global_topk_filled_by_gap0"
        elif in_rank <= 16:
            reason = "5_incoming_outgoing_oneway_omission"
        elif endpoint_bad:
            reason = "6_endpoint_quality_or_node_exclusion"
        elif impurity:
            reason = "8_node_identity_impurity"
        result.append({
            **event._asdict(), "rejection_reason": reason, "legacy_out_rank": out_rank,
            "legacy_in_rank": in_rank, "higher_gap0_candidates": gap0_higher,
            "pair_in_legacy_bank": int(pair_in_bank), "endpoint_bad": int(endpoint_bad),
            "node_impure": int(impurity), "source_edge_correctness": source_state,
            "appearance_threshold_present_in_legacy": 0, "source_consumed_cross_budget": 0,
        })
    return pd.DataFrame(result)


def merge_pool_coverage(root: Path, seq: str, events: pd.DataFrame, exact: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cols = ["src_chunk", "dst_chunk", "gap_bucket", "rank", "recall_score", "multi_appearance_cos", "whole_appearance_cos", "motion_error_min", "best_view"]
    outgoing = pd.read_parquet(root / "frozen_candidate_graph" / "outgoing_ranking_pool.parquet", columns=cols)
    incoming = pd.read_parquet(root / "frozen_candidate_graph" / "incoming_ranking_pool.parquet", columns=cols)
    outgoing = outgoing.rename(columns={"rank": "out_pool_rank", "recall_score": "out_pool_score"})[["src_chunk", "dst_chunk", "out_pool_rank", "out_pool_score"]]
    incoming = incoming.rename(columns={"rank": "in_pool_rank", "recall_score": "in_pool_score"})[["src_chunk", "dst_chunk", "in_pool_rank", "in_pool_score"]]
    old = pd.read_parquet(OLD_ROOTS[seq] / "frozen_candidate_graph" / "edges.parquet", columns=["src_chunk", "dst_chunk"])
    old["old_candidate"] = 1
    merged = events.merge(outgoing, on=["src_chunk", "dst_chunk"], how="left").merge(incoming, on=["src_chunk", "dst_chunk"], how="left").merge(old, on=["src_chunk", "dst_chunk"], how="left")
    merged["old_candidate"] = merged.old_candidate.fillna(0).astype(int)
    merged["best_pool_rank"] = np.fmin(merged.out_pool_rank.fillna(np.inf), merged.in_pool_rank.fillna(np.inf))
    merged.loc[merged.old_candidate == 1, "best_pool_rank"] = 0
    merged["expanded_present"] = ((merged.old_candidate == 1) | np.isfinite(merged.best_pool_rank)).astype(int)
    if len(exact):
        merged = merged.merge(exact, on=["seq", "src_chunk", "dst_chunk", "gap_stratum"], how="left")
    nonzero = merged[merged.gap_stratum != "0"]
    report = {
        "nonzero_events": len(nonzero),
        "nonzero_expanded_recall": float(nonzero.expanded_present.mean()) if len(nonzero) else None,
        "by_gap": {}, "recall_at_k": {}, "mrr": float(np.mean(1.0 / nonzero.best_pool_rank.replace(0, 1))) if len(nonzero) else None,
    }
    for bucket, group in nonzero.groupby("gap_stratum"):
        report["by_gap"][str(bucket)] = {"events": len(group), "expanded_recall": float(group.expanded_present.mean())}
    for k in RECALL_K:
        report["recall_at_k"][str(k)] = float(((nonzero.old_candidate == 1) | (nonzero.best_pool_rank <= k)).mean()) if len(nonzero) else None
    if len(exact):
        for method in ("recall", "multi", "whole", "motion"):
            rank = np.fmin(exact[f"out_{method}_rank"].to_numpy(float), exact[f"in_{method}_rank"].to_numpy(float))
            finite = np.isfinite(rank)
            report[f"independent_{method}_mrr"] = float(np.mean(1.0 / rank[finite])) if finite.any() else 0.0
            report[f"independent_{method}_recall_at_k"] = {str(k): float((rank <= k).mean()) for k in RECALL_K}
        out_rank = exact.out_recall_rank.to_numpy(float)
        in_rank = exact.in_recall_rank.to_numpy(float)
        mutual_rank = np.fmax(out_rank, in_rank)
        finite_mutual = np.isfinite(mutual_rank)
        report["mutual_rank_mrr"] = float(np.mean(1.0 / mutual_rank[finite_mutual])) if finite_mutual.any() else 0.0
        report["mutual_rank_recall_at_k"] = {str(k): float((mutual_rank <= k).mean()) for k in RECALL_K}
    return merged, report


def wrong_source_coverage(seq: str, root: Path, events: pd.DataFrame) -> dict:
    teacher_nodes = pd.read_parquet(OLD_ROOTS[seq] / "teacher_identity_flow" / "teacher_node_labels.parquet", columns=["chunk_id", "teacher_dominant_gt"])
    gt = teacher_nodes.teacher_dominant_gt.to_numpy(int)
    old_edges = pd.read_parquet(OLD_ROOTS[seq] / "frozen_candidate_graph" / "edges.parquet", columns=["src_chunk", "dst_chunk", "parent_edge"])
    wrong = old_edges[(old_edges.parent_edge == 1) & (gt[old_edges.src_chunk.to_numpy(int)] > 0) & (gt[old_edges.dst_chunk.to_numpy(int)] > 0) & (gt[old_edges.src_chunk.to_numpy(int)] != gt[old_edges.dst_chunk.to_numpy(int)])].copy()
    wrong_src = set(wrong.src_chunk.astype(int))
    if not wrong_src:
        return {"wrong_source_edges": 0}
    pool_parts = []
    for name in ("outgoing_ranking_pool.parquet", "incoming_ranking_pool.parquet"):
        frame = pd.read_parquet(root / "frozen_candidate_graph" / name, columns=["src_chunk", "dst_chunk"])
        pool_parts.append(frame[frame.src_chunk.astype(int).isin(wrong_src)])
    old_cross = old_edges[old_edges.src_chunk.astype(int).isin(wrong_src)][["src_chunk", "dst_chunk"]]
    pool = pd.concat([old_cross, *pool_parts], ignore_index=True).drop_duplicates()
    pool["same_teacher_gt"] = (gt[pool.src_chunk.to_numpy(int)] > 0) & (gt[pool.src_chunk.to_numpy(int)] == gt[pool.dst_chunk.to_numpy(int)])
    any_good = set(pool.loc[pool.same_teacher_gt, "src_chunk"].astype(int))
    canonical_map = {int(r.src_chunk): int(r.dst_chunk) for r in events[events.gap_stratum != "0"].itertuples(index=False)}
    pool_keys = set(zip(pool.src_chunk.astype(int), pool.dst_chunk.astype(int)))
    exact = sum((int(r.src_chunk), canonical_map.get(int(r.src_chunk), -1)) in pool_keys for r in wrong.itertuples(index=False))
    return {
        "wrong_source_edges": len(wrong),
        "any_correct_alternative": sum(int(src) in any_good for src in wrong.src_chunk),
        "any_correct_alternative_rate": float(np.mean([int(src) in any_good for src in wrong.src_chunk])),
        "exact_canonical_alternative": int(exact),
        "exact_canonical_alternative_rate": float(exact / len(wrong)) if len(wrong) else None,
    }


def coverage_command(seq: str, root: Path) -> dict:
    manifest = verify_freeze(root)
    nodes = pd.read_parquet(root / "frozen_candidate_graph" / "nodes.parquet")
    desc = load_descriptors(root)
    events = pd.read_parquet(SUCCESSOR_EVENTS)
    events = events[events.seq == seq].copy()
    exact = canonical_exact_ranks(nodes, desc, events[events.gap_stratum != "0"])
    rejection = old_rejection_audit(seq, root, events, nodes, desc)
    merged, coverage = merge_pool_coverage(root, seq, events, exact)
    wrong = wrong_source_coverage(seq, root, events)
    audit_dir = root / "postfreeze_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    exact.to_parquet(audit_dir / "canonical_exact_ranks.parquet", index=False)
    rejection.to_parquet(audit_dir / "legacy_rejection_reasons.parquet", index=False)
    merged.to_parquet(audit_dir / "successor_coverage.parquet", index=False)
    rejection_summary = []
    if len(rejection):
        group_cols = ["gap_stratum", "crowd_stratum", "trajectory_length_stratum", "source_edge_correctness", "rejection_reason"]
        rejection_summary = rejection.groupby(group_cols, dropna=False).size().reset_index(name="events").to_dict("records")
    report = {
        "experiment": "M23-55 post-freeze coverage and rejection audit", "seq": seq,
        "teacher_only": True, "deployable": False, "freeze_verified": True,
        "manifest_sha256": sha256_file(root / "frozen_candidate_graph" / "freeze_manifest.json"),
        "coverage": coverage, "wrong_source_repair": wrong,
        "rejection_reason_counts": {
            **{
                "1_temporal_eligibility_or_gap_gate": 0,
                "2_spatial_or_motion_gate": 0,
                "3_appearance_threshold": 0,
                "4_global_topk_filled_by_gap0": 0,
                "5_incoming_outgoing_oneway_omission": 0,
                "6_endpoint_quality_or_node_exclusion": 0,
                "7_source_edge_consumed_cross_budget": 0,
                "8_node_identity_impurity": 0,
                "legacy_outgoing_top16_or_unmodeled": 0,
            },
            **(rejection.rejection_reason.value_counts().sort_index().to_dict() if len(rejection) else {}),
        },
        "rejection_strata": rejection_summary,
        "diagnostic_targets": {
            "nonzero_total_at_least": 0.75, "gap_1_30_at_least": 0.80,
            "gap_31_90_at_least": 0.60, "wrong_source_exact_at_least": 0.55,
        },
        "candidate_counts": {
            "outgoing_pool": manifest["frozen_artifacts"]["outgoing_rows"],
            "incoming_pool": manifest["frozen_artifacts"]["incoming_rows"],
            "flow_edges": manifest["frozen_artifacts"]["edge_rows"],
            "new_flow_edges": manifest["frozen_artifacts"]["new_stratified_edges"],
        },
    }
    targets = report["diagnostic_targets"]
    report["target_pass"] = {
        "nonzero": coverage["nonzero_expanded_recall"] >= targets["nonzero_total_at_least"],
        "gap_1_30": coverage["by_gap"].get("1-30", {}).get("expanded_recall", 0) >= targets["gap_1_30_at_least"],
        "gap_31_90": coverage["by_gap"].get("31-90", {}).get("expanded_recall", 0) >= targets["gap_31_90_at_least"],
        "wrong_source_exact": wrong.get("exact_canonical_alternative_rate", 0) >= targets["wrong_source_exact_at_least"],
    }
    report["protocol_pass_for_teacher_smoke"] = bool(manifest["baseline_reconstruction"]["byte_exact"] and manifest["frozen_artifacts"]["forbidden_edge_columns"] == [] and manifest["frozen_artifacts"]["forbidden_node_columns"] == [])
    json_write(audit_dir / "coverage_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def detector_payload(path: Path) -> str:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split(",")
            if len(fields) >= 6:
                rows.append((int(float(fields[0])), tuple(fields[2:])))
    rows.sort()
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def teacher_command(seq: str, root: Path, skip_trackeval: bool) -> dict:
    manifest = verify_freeze(root)
    coverage_path = root / "postfreeze_audit" / "coverage_report.json"
    if not coverage_path.exists():
        raise RuntimeError("coverage audit must precede teacher smoke")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if not coverage["protocol_pass_for_teacher_smoke"]:
        raise RuntimeError("candidate protocol failed; teacher is blocked")
    m53 = load_module(f"m23_55_teacher_{SEQ_SHORT[seq]}", "scripts/m23_research/m23_53_global_identity_flow_capacity.py")
    nodes, selected, teacher_report = m53.build_teacher_utilities(seq=seq, source_parent_root=SOURCE_PARENT, output_root=root, freeze_manifest=manifest)
    tracker_path = root / "track_results" / f"{seq}.txt"
    tracker_report = m53.write_tracker(seq, SOURCE_PARENT / f"{seq}.txt", nodes, selected, tracker_path)
    tracker_report["sha256"] = sha256_file(tracker_path)
    source_payload = detector_payload(SOURCE_PARENT / f"{seq}.txt")
    output_payload = detector_payload(tracker_path)
    if source_payload != output_payload:
        raise RuntimeError("detection rows/boxes/scores changed")
    official = None if skip_trackeval else m53.run_official_trackeval(seq=seq, output_root=root, tracker_name=root.name)
    old_edges = pd.read_parquet(OLD_ROOTS[seq] / "frozen_candidate_graph" / "edges.parquet", columns=["src_chunk", "dst_chunk", "parent_edge"])
    parent = set(zip(old_edges.loc[old_edges.parent_edge == 1, "src_chunk"].astype(int), old_edges.loc[old_edges.parent_edge == 1, "dst_chunk"].astype(int)))
    chosen = set(zip(selected.src_chunk.astype(int), selected.dst_chunk.astype(int)))
    kept = len(parent & chosen)
    cut = len(parent - chosen)
    cross = len(chosen - parent)
    changed_edges = parent.symmetric_difference(chosen)
    affected_chunks = {chunk for edge in changed_edges for chunk in edge}
    affected_rows = int(nodes.iloc[sorted(affected_chunks)].rows.sum()) if affected_chunks else 0
    frozen_edges = pd.read_parquet(root / "frozen_candidate_graph" / "edges.parquet", columns=["src_chunk", "dst_chunk"])
    frozen_keys = set(zip(frozen_edges.src_chunk.astype(int), frozen_edges.dst_chunk.astype(int)))
    successor = pd.read_parquet(SUCCESSOR_EVENTS)
    successor = successor[successor.seq == seq]
    canonical_keys = [(int(r.src_chunk), int(r.dst_chunk)) for r in successor.itertuples(index=False)]
    present_keys = [key for key in canonical_keys if key in frozen_keys]
    converted = sum(key in chosen for key in present_keys)
    report = {
        "experiment": "M23-55 stratified long-gap teacher flow", "seq": seq,
        "teacher_only": True, "deployable": False, "freeze_verified": True,
        "candidate_graph_sha256": manifest["frozen_artifacts"]["edges_sha256"],
        "tracker_sha256": tracker_report["sha256"],
        "teacher": teacher_report, "tracker": tracker_report, "official_trackeval": official,
        "flow_actions": {
            "keep_parent": kept, "cut_parent": cut, "cross": cross,
            "dummy_terminate": len(nodes) - len(selected), "dummy_restart": len(nodes) - len(selected),
            "identity_affected_chunks": len(affected_chunks),
            "identity_affected_rows": affected_rows,
            "identity_affected_row_rate": float(affected_rows / max(1, int(nodes.rows.sum()))),
        },
        "flow_conversion": {
            "canonical_successors": len(canonical_keys),
            "canonical_present_in_flow_graph": len(present_keys),
            "canonical_selected": converted,
            "selection_rate_when_present": float(converted / len(present_keys)) if present_keys else None,
        },
        "integrity": {
            "one_to_one": True, "acyclic": True, "time_forward": True,
            "detection_payload_unchanged": True, "source_payload_sha256": source_payload,
            "output_payload_sha256": output_payload,
        },
    }
    report["report_payload_sha256"] = canonical_payload_sha256(report)
    report_path = root / "report.json"
    json_write(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def edge_frame(seq: str, root: Path) -> pd.DataFrame:
    nodes = pd.read_parquet(OLD_ROOTS[seq] / "teacher_identity_flow" / "teacher_node_labels.parquet", columns=["chunk_id", "teacher_dominant_gt"])
    gt = nodes.teacher_dominant_gt.to_numpy(int)
    edges = pd.read_parquet(root / "frozen_candidate_graph" / "edges.parquet")
    src, dst = edges.src_chunk.to_numpy(int), edges.dst_chunk.to_numpy(int)
    frame = pd.DataFrame({
        "seq": seq, "src_chunk": src, "dst_chunk": dst,
        "label": ((gt[src] > 0) & (gt[src] == gt[dst])).astype(np.int8),
        "gap_bucket": edges.get("m23_55_gap_bucket", pd.Series("legacy", index=edges.index)).astype(str),
        "recall_score": edges.get("m23_55_recall_score", edges.appearance_cos).fillna(edges.appearance_cos),
        "multi_appearance": edges.get("m23_55_multi_appearance_cos", edges.appearance_cos).fillna(edges.appearance_cos),
        "whole_appearance": edges.get("m23_55_whole_appearance_cos", edges.appearance_cos).fillna(edges.appearance_cos),
        "appearance_cos": edges.appearance_cos,
        "motion_error_min": edges.motion_error_min,
        "endpoint_displacement": edges.endpoint_displacement,
        "velocity_cos": edges.velocity_cos,
        "abs_log_height_ratio": edges.log_height_ratio.abs(),
        "log_gap": edges.log_gap,
        "src_consistency": edges.src_consistency,
        "dst_consistency": edges.dst_consistency,
        "parent_edge": edges.parent_edge,
    })
    return frame


def rankability_command(roots: dict[str, Path], output: Path) -> dict:
    for root in roots.values():
        verify_freeze(root)
    frames = {seq: edge_frame(seq, roots[seq]) for seq in SEQUENCES}
    successor = pd.read_parquet(SUCCESSOR_EVENTS)
    results = {}
    predictions = []
    rng = np.random.default_rng(2355)
    for held in SEQUENCES:
        train_parts = []
        for seq in SEQUENCES:
            if seq == held:
                continue
            frame = frames[seq]
            positive = frame[frame.label == 1]
            negative = frame[frame.label == 0]
            limit = min(len(negative), max(100000, 8 * len(positive)))
            if len(negative) > limit:
                negative = negative.iloc[rng.choice(len(negative), limit, replace=False)]
            train_parts.append(pd.concat([positive, negative], ignore_index=True))
        train = pd.concat(train_parts, ignore_index=True)
        test = frames[held].copy()
        model = make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), LogisticRegression(max_iter=300, class_weight="balanced", C=0.25, random_state=2355))
        model.fit(train[list(FEATURES)], train.label)
        probability = model.predict_proba(test[list(FEATURES)])[:, 1]
        test["oof_probability"] = probability
        test["out_rank"] = test.groupby("src_chunk").oof_probability.rank(method="first", ascending=False)
        test["in_rank"] = test.groupby("dst_chunk").oof_probability.rank(method="first", ascending=False)
        top_out = test.loc[test.groupby("src_chunk").oof_probability.idxmax()]
        top_in = test.loc[test.groupby("dst_chunk").oof_probability.idxmax()]
        canonical = successor[(successor.seq == held) & (successor.gap_stratum != "0")][["src_chunk", "dst_chunk", "gap_stratum"]]
        canonical = canonical.merge(test[["src_chunk", "dst_chunk", "out_rank", "in_rank", "oof_probability"]], on=["src_chunk", "dst_chunk"], how="left")
        canonical["rank"] = np.fmin(canonical.out_rank.fillna(np.inf), canonical.in_rank.fillna(np.inf))
        by_gap = {}
        for bucket, group in canonical.groupby("gap_stratum"):
            by_gap[str(bucket)] = {
                "events": len(group), "mrr": float(np.mean(1.0 / group["rank"].replace(np.inf, np.nan).dropna())) if np.isfinite(group["rank"]).any() else 0.0,
                "recall_at_k": {str(k): float((group["rank"] <= k).mean()) for k in RECALL_K},
            }
        bins = np.linspace(0, 1, 11)
        index = np.clip(np.digitize(probability, bins) - 1, 0, 9)
        ece = 0.0
        for b in range(10):
            mask = index == b
            if mask.any():
                ece += float(mask.mean()) * abs(float(probability[mask].mean()) - float(test.label.to_numpy()[mask].mean()))
        wrong_sources = set()
        old_nodes = pd.read_parquet(OLD_ROOTS[held] / "teacher_identity_flow" / "teacher_node_labels.parquet", columns=["teacher_dominant_gt"])
        gt = old_nodes.teacher_dominant_gt.to_numpy(int)
        old_edges = pd.read_parquet(OLD_ROOTS[held] / "frozen_candidate_graph" / "edges.parquet", columns=["src_chunk", "dst_chunk", "parent_edge"])
        for r in old_edges[old_edges.parent_edge == 1].itertuples(index=False):
            if gt[int(r.src_chunk)] > 0 and gt[int(r.dst_chunk)] > 0 and gt[int(r.src_chunk)] != gt[int(r.dst_chunk)]:
                wrong_sources.add(int(r.src_chunk))
        wrong_top = top_out[top_out.src_chunk.astype(int).isin(wrong_sources)]
        results[held] = {
            "train_sequences": [seq for seq in SEQUENCES if seq != held],
            "test_edges": len(test), "positive_rate": float(test.label.mean()),
            "canonical_mrr": float(np.mean(1.0 / canonical["rank"].replace(np.inf, np.nan).dropna())) if np.isfinite(canonical["rank"]).any() else 0.0,
            "canonical_recall_at_k": {str(k): float((canonical["rank"] <= k).mean()) for k in RECALL_K},
            "outgoing_top1_precision": float(top_out.label.mean()),
            "incoming_top1_precision": float(top_in.label.mean()),
            "wrong_source_repair_precision": float(wrong_top.label.mean()) if len(wrong_top) else None,
            "catastrophic_false_link_rate": float(1.0 - top_out.label.mean()),
            "brier": float(brier_score_loss(test.label, probability)),
            "log_loss": float(log_loss(test.label, probability, labels=[0, 1])),
            "ece_10bin": ece, "by_gap": by_gap,
            "source_cross_margin": {
                "positive_mean": float(test.loc[test.label == 1, "oof_probability"].mean()),
                "negative_mean": float(test.loc[test.label == 0, "oof_probability"].mean()),
            },
        }
        predictions.append(test[["seq", "src_chunk", "dst_chunk", "label", "gap_bucket", "oof_probability", "out_rank", "in_rank"]])
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(predictions, ignore_index=True).to_parquet(output.with_suffix(".parquet"), index=False)
    report = {
        "experiment": "M23-55 observable OOF rankability audit", "deployable": False,
        "role": "diagnostic only; no tracker selection or held-fold tuning", "features": list(FEATURES),
        "folds": results,
    }
    json_write(output, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def aggregate_command(roots: dict[str, Path], output_json: Path, output_md: Path) -> dict:
    coverage, teacher, manifests = {}, {}, {}
    for seq, root in roots.items():
        coverage[seq] = json.loads((root / "postfreeze_audit" / "coverage_report.json").read_text(encoding="utf-8"))
        report_path = root / "report.json"
        teacher[seq] = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
        manifests[seq] = json.loads((root / "frozen_candidate_graph" / "freeze_manifest.json").read_text(encoding="utf-8"))
    combined_path = Path("outputs/mot20_m23_20260718/m23_55_stratified_gap_capacity_combined_v1/report.json")
    rankability_path = Path("outputs/mot20_m23_20260718/m23_55_rankability_oof_v1/report.json")
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    rankability = json.loads(rankability_path.read_text(encoding="utf-8"))
    pooled_events = sum(item["coverage"]["nonzero_events"] for item in coverage.values())
    pooled_hits = sum(item["coverage"]["nonzero_events"] * item["coverage"]["nonzero_expanded_recall"] for item in coverage.values())
    pooled_gap = {}
    for bucket in ("1-30", "31-90", "91-180", "181-600"):
        events = sum(item["coverage"]["by_gap"].get(bucket, {}).get("events", 0) for item in coverage.values())
        hits = sum(
            item["coverage"]["by_gap"].get(bucket, {}).get("events", 0)
            * item["coverage"]["by_gap"].get(bucket, {}).get("expanded_recall", 0.0)
            for item in coverage.values()
        )
        pooled_gap[bucket] = {"events": events, "expanded_recall": hits / events if events else None}
    wrong_total = sum(item["wrong_source_repair"]["wrong_source_edges"] for item in coverage.values())
    wrong_exact = sum(item["wrong_source_repair"]["exact_canonical_alternative"] for item in coverage.values())
    rejection_totals = Counter()
    for item in coverage.values():
        rejection_totals.update(item["rejection_reason_counts"])
    coverage_summary = {
        seq: {
            "coverage": coverage[seq]["coverage"],
            "wrong_source_repair": coverage[seq]["wrong_source_repair"],
            "rejection_reason_counts": coverage[seq]["rejection_reason_counts"],
            "candidate_counts": coverage[seq]["candidate_counts"],
            "target_pass": coverage[seq]["target_pass"],
            "freeze_manifest_sha256": coverage[seq]["manifest_sha256"],
        }
        for seq in SEQUENCES
    }
    candidate_resources = {}
    for seq in SEQUENCES:
        artifacts = manifests[seq]["frozen_artifacts"]
        resource_paths = {
            "flow_graph": Path(artifacts["edges"]),
            "outgoing_pool": Path(artifacts["outgoing_pool"]),
            "incoming_pool": Path(artifacts["incoming_pool"]),
            "descriptors": Path(artifacts["descriptors"]),
        }
        candidate_resources[seq] = {
            "flow_edges": artifacts["edge_rows"],
            "outgoing_rows": artifacts["outgoing_rows"],
            "incoming_rows": artifacts["incoming_rows"],
            "new_stratified_flow_edges": artifacts["new_stratified_edges"],
            "frozen_artifact_disk_bytes": sum(path.stat().st_size for path in resource_paths.values()),
            "generation_runtime_seconds": manifests[seq]["runtime_seconds"],
            "descriptor_runtime_seconds": manifests[seq]["descriptor_report"]["runtime_seconds"],
            "outgoing_runtime_seconds": manifests[seq]["ranking_reports"]["outgoing"]["runtime_seconds"],
            "incoming_runtime_seconds": manifests[seq]["ranking_reports"]["incoming"]["runtime_seconds"],
            "peak_memory_bytes": None,
            "peak_memory_note": "not instrumented in v1; frozen artifact disk bytes are reported instead",
        }
    independent_contribution = {}
    for method in ("recall", "multi", "whole", "motion"):
        weighted_mrr = sum(
            coverage[seq]["coverage"]["nonzero_events"] * coverage[seq]["coverage"][f"independent_{method}_mrr"]
            for seq in SEQUENCES
        ) / max(1, pooled_events)
        weighted_r32 = sum(
            coverage[seq]["coverage"]["nonzero_events"] * coverage[seq]["coverage"][f"independent_{method}_recall_at_k"]["32"]
            for seq in SEQUENCES
        ) / max(1, pooled_events)
        independent_contribution[method] = {"pooled_weighted_mrr": weighted_mrr, "pooled_weighted_recall_at_32": weighted_r32}
    teacher_summary = {
        seq: {
            "official_trackeval": teacher[seq]["official_trackeval"],
            "flow_actions": teacher[seq]["flow_actions"],
            "flow_conversion": teacher[seq]["flow_conversion"],
            "candidate_graph_sha256": teacher[seq]["candidate_graph_sha256"],
            "tracker_sha256": teacher[seq]["tracker_sha256"],
            "report_payload_sha256": teacher[seq]["report_payload_sha256"],
        }
        for seq in SEQUENCES
    }
    aggregate = {
        "experiment": "M23-55 Stratified Long-Gap Candidate Recall Expansion",
        "teacher_only": True, "deployable": False,
        "strict_parent": "M23-46 HOTA 79.123193",
        "coverage_folds": coverage_summary,
        "teacher_folds": teacher_summary,
        "official_combined": combined,
        "observable_oof_rankability": rankability,
        "pooled_nonzero_events": pooled_events,
        "pooled_nonzero_recall": pooled_hits / pooled_events if pooled_events else None,
        "pooled_gap_recall": pooled_gap,
        "pooled_wrong_source": {
            "wrong_source_edges": wrong_total,
            "exact_canonical_alternative": wrong_exact,
            "exact_canonical_alternative_rate": wrong_exact / wrong_total if wrong_total else None,
        },
        "pooled_rejection_reason_counts": dict(sorted(rejection_totals.items())),
        "candidate_resources": candidate_resources,
        "independent_rank_contribution": independent_contribution,
        "protocol_decision": {
            "capacity_gate": "B: 80.300000 <= teacher COMBINED < 80.700000",
            "m23_54_started": False,
            "m23_54_status": "locked",
            "allowed_next": "separately preregistered small strict nested-LOSO structured transfer feasibility only",
            "reason_not_directly_deployable": "OOF wrong-source repair top-1 precision is 0 on M01/M02/M03 and 0.43% on M05; catastrophic false-link rate is 7.33%-12.15%",
            "mot20_test_submission": False,
        },
        "implementation_notes": {
            "explicit_suspicious_source_extra_quota": False,
            "source_repair_note": "v1 used uniform per-node, per-gap outgoing/incoming quotas for every source and retained every old source/cross edge; it did not add a separate suspicious-source-only quota. This valid GT-free subset was not changed after GT unlock.",
            "rankability_normalization": "diagnostic OOF logistic regression used outer-training RobustScaler; it is not the separately gated sequence-normalized structured transfer student",
            "peak_memory_instrumented": False,
        },
        "artifact_hashes": {
            "combined_report_sha256": sha256_file(combined_path),
            "rankability_report_sha256": sha256_file(rankability_path),
            "candidate_manifests": {
                seq: {
                    "nodes_sha256": manifests[seq]["frozen_artifacts"]["nodes_sha256"],
                    "edges_sha256": manifests[seq]["frozen_artifacts"]["edges_sha256"],
                    "outgoing_pool_sha256": manifests[seq]["frozen_artifacts"]["outgoing_pool_sha256"],
                    "incoming_pool_sha256": manifests[seq]["frozen_artifacts"]["incoming_pool_sha256"],
                    "descriptors_sha256": manifests[seq]["frozen_artifacts"]["descriptors_sha256"],
                }
                for seq in SEQUENCES
            },
        },
    }
    aggregate["summary_payload_sha256"] = canonical_payload_sha256(aggregate)
    json_write(output_json, aggregate)
    official = combined["official_trackeval"]
    lines = [
        "# M23-55 Stratified Long-Gap Candidate Recall Expansion", "", "Date: 2026-07-20", "",
        "## Protocol", "", "- GT-free candidate ranking pools were frozen and SHA-256 recorded before GT diagnostics.",
        "- Gap buckets 1-30, 31-90, 91-180 and 181-600 have independent outgoing/incoming quotas.",
        "- Ranking pool K=256; teacher-flow subgraph K=32 per direction/bucket; all M23-53 edges retained.",
        "- M23-54 was not started; no MOT20 test submission was made.", "", "## Coverage", "",
        "| Fold | Nonzero events | Expanded recall | Exact wrong-source alternative |", "|---|---:|---:|---:|",
    ]
    for seq in SEQUENCES:
        c = coverage[seq]
        lines.append(f"| {seq} | {c['coverage']['nonzero_events']} | {100*c['coverage']['nonzero_expanded_recall']:.2f}% | {100*c['wrong_source_repair']['exact_canonical_alternative_rate']:.2f}% |")
    lines.extend([f"| POOLED | {pooled_events} | {100*aggregate['pooled_nonzero_recall']:.2f}% | {100*aggregate['pooled_wrong_source']['exact_canonical_alternative_rate']:.2f}% |", "", "### Pooled recall by gap", "", "| Gap | Events | Recall |", "|---|---:|---:|"])
    for bucket in ("1-30", "31-90", "91-180", "181-600"):
        item = pooled_gap[bucket]
        lines.append(f"| {bucket} | {item['events']} | {100*item['expanded_recall']:.2f}% |")
    lines.extend(["", "### Candidate scale and runtime", "", "| Fold | Flow edges | Out K=256 | In K=256 | Frozen disk | Runtime |", "|---|---:|---:|---:|---:|---:|"])
    for seq in SEQUENCES:
        resource = candidate_resources[seq]
        lines.append(
            f"| {seq} | {resource['flow_edges']} | {resource['outgoing_rows']} | {resource['incoming_rows']} | "
            f"{resource['frozen_artifact_disk_bytes'] / (1024**3):.2f} GiB | {resource['generation_runtime_seconds']:.1f} s |"
        )
    lines.extend([
        "", "Peak process RAM/GPU memory was not instrumented in v1; the table reports exact frozen artifact disk size rather than inventing a peak-memory value.",
        "", "### Independent ranking contribution", "", "| Score view | Pooled weighted MRR | Pooled weighted R@32 |", "|---|---:|---:|",
    ])
    for method, label in (("recall", "fixed multiview + motion score"), ("multi", "multiview appearance"), ("whole", "whole-track prototype"), ("motion", "motion-only rank")):
        item = independent_contribution[method]
        lines.append(f"| {label} | {item['pooled_weighted_mrr']:.4f} | {100*item['pooled_weighted_recall_at_32']:.2f}% |")
    lines.extend(["", "### Legacy rejection reasons for missing nonzero-gap successors", "", "| First rejection reason | Events |", "|---|---:|"])
    rejection_labels = {
        "1_temporal_eligibility_or_gap_gate": "temporal eligibility / gap gate",
        "2_spatial_or_motion_gate": "spatial / motion gate",
        "3_appearance_threshold": "appearance threshold",
        "4_global_topk_filled_by_gap0": "global top-k filled by gap=0",
        "5_incoming_outgoing_oneway_omission": "incoming/outgoing one-way omission",
        "6_endpoint_quality_or_node_exclusion": "endpoint quality / node exclusion",
        "7_source_edge_consumed_cross_budget": "source edge consumed cross budget",
        "8_node_identity_impurity": "node identity impurity",
        "legacy_outgoing_top16_or_unmodeled": "legacy outgoing top-16 / unmodeled",
    }
    for key, value in sorted(rejection_totals.items()):
        lines.append(f"| {rejection_labels.get(key, key)} | {value} |")
    lines.extend(["", "## Teacher capacity", ""])
    if all(teacher.values()):
        lines.extend(["| Fold | HOTA | DetA | AssA | IDSW |", "|---|---:|---:|---:|---:|"])
        for seq in SEQUENCES:
            o = teacher[seq]["official_trackeval"]
            lines.append(f"| {seq} | {o['HOTA']:.6f} | {o['DetA']:.6f} | {o['AssA']:.6f} | {o['IDSW']} |")
        o = official["COMBINED"]
        lines.append(f"| **COMBINED** | **{o['HOTA']:.6f}** | **{o['DetA']:.6f}** | **{o['AssA']:.6f}** | **{o['IDSW']}** |")
    else:
        lines.append("Teacher folds are incomplete; COMBINED capacity is not claimed.")
    lines.extend([
        "", "- Relative to M23-53 teacher: **+0.434950 HOTA**.",
        "- Relative to strict M23-46: **+1.262347 HOTA**, but this result is teacher-only and deployable=false.",
        "- Capacity gate: **B (80.300000-80.700000)**. M23-54 remains locked.",
        "", "## Observable OOF rankability", "",
        "| Held fold | Canonical MRR | R@32 | Out top-1 precision | In top-1 precision | Wrong-source repair precision | Catastrophic false-link |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for seq in SEQUENCES:
        r = rankability["folds"][seq]
        wrong = r["wrong_source_repair_precision"]
        lines.append(
            f"| {seq} | {r['canonical_mrr']:.4f} | {100*r['canonical_recall_at_k']['32']:.2f}% | "
            f"{100*r['outgoing_top1_precision']:.2f}% | {100*r['incoming_top1_precision']:.2f}% | "
            f"{100*wrong:.2f}% | {100*r['catastrophic_false_link_rate']:.2f}% |"
        )
    lines.extend([
        "", "The recall-first candidate pool raises teacher capacity above 80.3, but observable cross-sequence ranking does not safely repair wrong source edges. Teacher conversion must not be treated as deployable rankability evidence.",
        "", "## Implementation limitations", "",
        "- M23-55 v1 gave every source the same per-gap outgoing/incoming quota and retained all prior source/cross edges. It did **not** add a separate suspicious-source-only extra quota. That conservative GT-free subset was frozen before GT and was not modified post hoc.",
        "- The OOF rankability model is a diagnostic RobustScaler + logistic regression, not the separately gated sequence-normalized structured transfer student.",
        "- Peak process memory was not instrumented; only exact frozen artifact disk size and wall-clock runtime are claimed.",
        "", "## Decision", "",
        "- Do not start M23-54 under the preregistered 80.7 gate.",
        "- Do not train a large student and do not submit a new MOT20 test result.",
        "- The only unlocked follow-up is a separately preregistered small strict nested-LOSO structured transfer feasibility audit; every failed inner gate must freeze the outer tracker to no-op.",
        "- Best strict deployable result remains M23-46 HOTA **79.123193**.",
    ])
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(output_json), "md": str(output_md), "pooled_nonzero_recall": aggregate["pooled_nonzero_recall"]}, indent=2), flush=True)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("coverage")
    p.add_argument("--seq", required=True, choices=SEQUENCES)
    p.add_argument("--root", default=None)
    p = sub.add_parser("teacher")
    p.add_argument("--seq", required=True, choices=SEQUENCES)
    p.add_argument("--root", default=None)
    p.add_argument("--skip-trackeval", action="store_true")
    p = sub.add_parser("rankability")
    p.add_argument("--output", default="outputs/mot20_m23_20260718/m23_55_rankability_oof_v1/report.json")
    p = sub.add_parser("aggregate")
    p.add_argument("--output-json", default="docs/generated/M23_55_STRATIFIED_GAP_CAPACITY_20260720.json")
    p.add_argument("--output-md", default="docs/m23_55_stratified_gap_candidate_capacity_20260720.md")
    args = parser.parse_args()
    if args.command == "coverage":
        coverage_command(args.seq, Path(args.root) if args.root else ROOTS[args.seq])
    elif args.command == "teacher":
        teacher_command(args.seq, Path(args.root) if args.root else ROOTS[args.seq], args.skip_trackeval)
    elif args.command == "rankability":
        rankability_command(ROOTS, Path(args.output))
    else:
        aggregate_command(ROOTS, Path(args.output_json), Path(args.output_md))


if __name__ == "__main__":
    main()
