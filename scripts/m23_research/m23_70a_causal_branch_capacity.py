#!/usr/bin/env python3
from __future__ import annotations

"""M23-70A: genuinely bounded-delay K=3 causal branch capacity audit.

This is a teacher-only MOT20-train capacity experiment. GT may define which
split/action is correct only after the causal candidate graph is frozen. No
model is trained and no MOT20 test data is accessed.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
ROOT = Path("outputs/mot20_m23_20260718/m23_70a_causal_branch_capacity_v1")
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
PREREG_SHA256 = "0e4f875ad0445088d90e9a6c9b5fe72f7656b27eb84db1cda0841cecd1b810be"
M46_REPORT = Path("outputs/mot20_m23_20260718/m23_46_pure_hgb_topk_combined_oof_v1/report.json")
M57_ROOT = Path("outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2")
SOURCE_PARENT = Path("outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results")
BASELINE_CACHE = Path("outputs/mot20_m23_20260718/m23_47_m23_46_deployable_baseline_cache_v1")
DELAY = 8
K = 3
ALTERNATIVES = 2
GO_HOTA = 80.80

# Frozen causal candidate score. These coefficients are preregistered and are
# never tuned per sequence.
W_APPEARANCE = 1.0
W_MOTION = 0.35
W_GAP = 0.03
W_HEIGHT = 0.08


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def event(root: Path, name: str, **payload: Any) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "protocol_events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": now(), "event": name, **payload}, sort_keys=True) + "\n")


def load_module(name: str, relative: str):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def verify_prereg(root: Path) -> dict[str, Any]:
    path = root / "preregistered_protocol.json"
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha(path)
    if actual != PREREG_SHA256:
        raise RuntimeError(f"preregistration hash mismatch: {actual} != {PREREG_SHA256}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["fixed"]["delay_frames"] != DELAY or payload["fixed"]["branch_count"] != K:
        raise RuntimeError("frozen protocol constants do not match implementation")
    return payload


def freeze_inputs(root: Path) -> dict[str, Any]:
    verify_prereg(root)
    output = root / "input_manifest.json"
    if output.exists():
        raise FileExistsError(output)
    required = [
        M46_REPORT,
        M57_ROOT / "capacity_combined/report.json",
        Path("scripts/m23_research/m23_57_intra_node_change_point_capacity.py"),
        Path("scripts/m23_research/m23_53b_build_adaptive_micrograph.py"),
        Path("scripts/m23_research/m23_53_global_identity_flow_capacity.py"),
        Path("scripts/m23_research/m23_11_add_micrograph_utility.py"),
        Path("scripts/m23_research/m23_45_domain_robust_source_cut_student.py"),
    ]
    for seq in SEQUENCES:
        required += [
            M57_ROOT / f"boundary_universe/{seq}/freeze_manifest.json",
            M57_ROOT / f"boundary_universe/{seq}/boundary_features.parquet",
            SOURCE_PARENT / f"{seq}.txt",
            BASELINE_CACHE / seq / "track_results" / f"{seq}.txt",
        ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    m46 = json.loads(M46_REPORT.read_text(encoding="utf-8"))
    m57 = json.loads((M57_ROOT / "capacity_combined/report.json").read_text(encoding="utf-8"))
    payload = {
        "experiment_id": "M23-70A",
        "created_at": now(),
        "git_head": git_head(),
        "m23_46": m46["metrics"],
        "m23_57_teacher_reference": m57["official_trackeval"]["COMBINED"],
        "artifacts": {str(p): {"sha256": sha(p), "bytes": p.stat().st_size} for p in required},
        "training_runs": 0,
        "mot20_test_reads": 0,
        "trackeval_runs": 0,
    }
    write_json(output, payload)
    event(root, "inputs_frozen", manifest_sha256=sha(output))
    return payload


def freeze_implementation(root: Path) -> dict[str, Any]:
    verify_prereg(root)
    input_manifest = root / "input_manifest.json"
    if not input_manifest.exists():
        raise RuntimeError("freeze inputs first")
    output = root / "implementation_manifest.json"
    if output.exists():
        raise FileExistsError(output)
    script = Path("scripts/m23_research/m23_70a_causal_branch_capacity.py")
    test = Path("scripts/m23_research/test_m23_70a_causal_branch_capacity.py")
    payload = {
        "experiment_id": "M23-70A",
        "created_at": now(),
        "git_head": git_head(),
        "preregistration_sha256": PREREG_SHA256,
        "input_manifest_sha256": sha(input_manifest),
        "script_sha256": sha(script),
        "test_sha256": sha(test),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "execution_device": "cpu",
            "cuda_probe_executed": False,
            "gpu": None,
            "cpu_count": os.cpu_count(),
        },
        "score": {
            "appearance": W_APPEARANCE,
            "motion_penalty": W_MOTION,
            "gap_penalty": W_GAP,
            "height_penalty": W_HEIGHT,
        },
        "training_runs": 0,
    }
    write_json(output, payload)
    event(root, "implementation_frozen", manifest_sha256=sha(output))
    return payload


def ownership_runs(values: np.ndarray, matched_positions: np.ndarray) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for gt, pos in zip(values[matched_positions], matched_positions):
        gt, pos = int(gt), int(pos)
        if gt <= 0:
            continue
        if not runs or int(runs[-1]["identity"]) != gt:
            runs.append({"identity": gt, "positions": [pos]})
        else:
            runs[-1]["positions"].append(pos)
    return runs


def causal_teacher_cuts(
    fixed_nodes: pd.DataFrame,
    chunk_rows: dict[int, list[int]],
    source_rows: list[dict[str, Any]],
    row_gt: np.ndarray,
    mandatory: dict[int, list[int]],
) -> tuple[dict[int, list[int]], pd.DataFrame, dict[str, Any]]:
    cuts = {int(k): sorted(set(map(int, v))) for k, v in mandatory.items()}
    records: list[dict[str, Any]] = []
    for node in fixed_nodes.itertuples(index=False):
        fixed_id = int(node.chunk_id)
        indices = np.asarray(chunk_rows[fixed_id], np.int64)
        own = row_gt[indices]
        matched = np.flatnonzero(own > 0)
        runs = ownership_runs(own, matched)
        for transition_index, (left, right) in enumerate(zip(runs, runs[1:])):
            position = int(left["positions"][-1]) + 1
            decision_frame = int(source_rows[int(indices[position])]["frame"])
            evidence_frame = None
            if len(right["positions"]) >= 2:
                evidence_frame = int(source_rows[int(indices[int(right["positions"][1])])]["frame"])
            evidence_delay = None if evidence_frame is None else evidence_frame - decision_frame
            eligible = bool(
                len(left["positions"]) >= 2
                and len(right["positions"]) >= 2
                and evidence_delay is not None
                and 0 <= evidence_delay <= DELAY
            )
            if eligible:
                cuts.setdefault(fixed_id, []).append(position)
            records.append({
                "fixed_chunk_id": fixed_id,
                "transition_index": transition_index,
                "position": position,
                "left_identity": int(left["identity"]),
                "right_identity": int(right["identity"]),
                "left_support": len(left["positions"]),
                "right_support": len(right["positions"]),
                "decision_frame": decision_frame,
                "evidence_frame": evidence_frame,
                "evidence_delay": evidence_delay,
                "causal_eligible": int(eligible),
            })
    cuts = {k: sorted(set(v)) for k, v in cuts.items()}
    transitions = pd.DataFrame(records)
    eligible = int(transitions.causal_eligible.sum()) if len(transitions) else 0
    return cuts, transitions, {
        "teacher_transitions": len(transitions),
        "causal_eligible_transitions": eligible,
        "causal_eligible_rate": eligible / len(transitions) if len(transitions) else None,
        "delay_frames": DELAY,
        "mandatory_chunks": len(mandatory),
        "chunks_with_cut": len(cuts),
    }


def adaptive_row_map(
    fixed_nodes: pd.DataFrame,
    chunk_rows: dict[int, list[int]],
    cuts: dict[int, list[int]],
    nodes: pd.DataFrame,
) -> dict[int, list[int]]:
    by_track: dict[int, list[Any]] = defaultdict(list)
    for node in fixed_nodes.itertuples(index=False):
        by_track[int(node.source_track_id)].append(node)
    result: dict[int, list[int]] = {}
    chunk_id = 0
    for _, track_nodes in sorted(by_track.items()):
        track_nodes.sort(key=lambda n: int(n.source_ordinal))
        for fixed in track_nodes:
            fixed_id = int(fixed.chunk_id)
            rows = chunk_rows[fixed_id]
            boundaries = [0, *sorted(cuts.get(fixed_id, [])), len(rows)]
            for start, end in zip(boundaries, boundaries[1:]):
                part = list(map(int, rows[start:end]))
                if not part:
                    raise RuntimeError("empty adaptive segment")
                result[chunk_id] = part
                observed = nodes.iloc[chunk_id]
                if int(observed.first_line) != part[0] or int(observed.last_line) != part[-1]:
                    raise RuntimeError("adaptive row-map mismatch")
                chunk_id += 1
    if chunk_id != len(nodes):
        raise RuntimeError("adaptive row-map size mismatch")
    return result


def normalized_mean(vectors: np.ndarray) -> np.ndarray | None:
    if len(vectors) == 0:
        return None
    value = vectors.mean(axis=0)
    norm = float(np.linalg.norm(value))
    return None if norm <= 1e-12 else (value / norm).astype(np.float32)


def causal_candidate_edges(
    nodes: pd.DataFrame,
    rows_by_node: dict[int, list[int]],
    source_rows: list[dict[str, Any]],
    row_embeddings: np.ndarray,
    mapped: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    # B0 parent edges: consecutive segments carrying the same frozen M23-46 ID.
    parent_records: list[dict[str, Any]] = []
    parent_pairs: set[tuple[int, int]] = set()
    for _, group in nodes.groupby("parent_tracker_id", sort=True):
        ordered = group.sort_values(["first_frame", "last_frame", "chunk_id"], kind="mergesort")
        ids = ordered.chunk_id.astype(int).tolist()
        for src, dst in zip(ids, ids[1:]):
            if int(nodes.iloc[dst].first_frame) <= int(nodes.iloc[src].last_frame):
                raise RuntimeError("overlapping nodes share a parent ID")
            pair = (src, dst)
            parent_pairs.add(pair)
            parent_records.append({
                "src_chunk": src,
                "dst_chunk": dst,
                "gap": int(nodes.iloc[dst].first_frame - nodes.iloc[src].last_frame - 1),
                "parent_edge": 1,
                "edge_role": "m23_46_parent",
                "causal_score": 0.0,
                "appearance_cos": np.nan,
                "motion_error": np.nan,
                "log_height_ratio": np.nan,
                "visible_dst_rows": 0,
                "branch_rank": 0,
            })

    first_frame_index: dict[int, list[int]] = defaultdict(list)
    for row in nodes.itertuples(index=False):
        first_frame_index[int(row.first_frame)].append(int(row.chunk_id))

    src_proto: dict[int, np.ndarray | None] = {}
    for src in range(len(nodes)):
        last = int(nodes.iloc[src].last_frame)
        indices = [
            i for i in rows_by_node[src]
            if int(source_rows[i]["frame"]) >= last - (DELAY - 1) and bool(mapped[i])
        ]
        src_proto[src] = normalized_mean(row_embeddings[np.asarray(indices, np.int64)]) if indices else None

    candidates: list[dict[str, Any]] = []
    for src in range(len(nodes)):
        source = nodes.iloc[src]
        source_last = int(source.last_frame)
        cutoff = source_last + DELAY
        possible: list[int] = []
        for frame in range(source_last + 1, cutoff + 2):
            possible.extend(first_frame_index.get(frame, []))
        for dst in possible:
            if src == dst or (src, dst) in parent_pairs:
                continue
            destination = nodes.iloc[dst]
            gap = int(destination.first_frame - source_last - 1)
            if gap < 0 or gap > DELAY:
                continue
            visible = [i for i in rows_by_node[dst] if int(source_rows[i]["frame"]) <= cutoff]
            visible_mapped = [i for i in visible if bool(mapped[i])]
            dst_proto = normalized_mean(row_embeddings[np.asarray(visible_mapped, np.int64)]) if visible_mapped else None
            appearance = -1.0 if src_proto[src] is None or dst_proto is None else float(src_proto[src] @ dst_proto)
            dt = max(float(destination.first_frame - source_last), 1.0)
            predicted_x = float(source.last_cx + source.end_vx * dt)
            predicted_y = float(source.last_cy + source.end_vy * dt)
            scale = max(float(0.5 * (source.last_h + destination.first_h)), 1.0)
            motion_error = math.hypot(float(destination.first_cx) - predicted_x, float(destination.first_cy) - predicted_y) / scale
            height_ratio = abs(math.log(max(float(destination.first_h), 1e-3) / max(float(source.last_h), 1e-3)))
            score = W_APPEARANCE * appearance - W_MOTION * motion_error - W_GAP * gap - W_HEIGHT * height_ratio
            candidates.append({
                "src_chunk": src,
                "dst_chunk": dst,
                "gap": gap,
                "parent_edge": 0,
                "edge_role": "m23_70a_causal_alternative",
                "causal_score": score,
                "appearance_cos": appearance,
                "motion_error": motion_error,
                "log_height_ratio": height_ratio,
                "visible_dst_rows": len(visible),
            })

    cross = pd.DataFrame(candidates)
    if len(cross):
        cross.sort_values(
            ["src_chunk", "causal_score", "motion_error", "gap", "dst_chunk"],
            ascending=[True, False, True, True, True],
            kind="mergesort",
            inplace=True,
        )
        cross["branch_rank"] = cross.groupby("src_chunk", sort=False).cumcount() + 1
        all_cross = cross.copy()
        cross = cross[cross.branch_rank <= ALTERNATIVES].copy()
    else:
        all_cross = cross.copy()
        cross["branch_rank"] = pd.Series(dtype=np.int64)
    parent = pd.DataFrame(parent_records)
    edges = pd.concat([parent, cross], ignore_index=True, sort=False)
    edges.sort_values(["src_chunk", "parent_edge", "branch_rank", "dst_chunk"], ascending=[True, False, True, True], kind="mergesort", inplace=True)
    edges.drop_duplicates(["src_chunk", "dst_chunk"], keep="first", inplace=True)
    edges.reset_index(drop=True, inplace=True)
    branch_table = cross.copy()
    counts = cross.groupby("src_chunk").size() if len(cross) else pd.Series(dtype=int)
    if len(counts) and int(counts.max()) > ALTERNATIVES:
        raise RuntimeError("K=3 alternative budget exceeded")
    report = {
        "parent_edges": len(parent),
        "causal_cross_candidates_before_k": len(all_cross),
        "retained_cross_edges": len(cross),
        "retained_edges": len(edges),
        "sources_with_alternative": int(len(counts)),
        "sources_with_two_alternatives": int((counts == ALTERNATIVES).sum()) if len(counts) else 0,
        "max_alternatives_per_source": int(counts.max()) if len(counts) else 0,
        "delay_frames": DELAY,
        "complete_future_destination_descriptor_used": False,
    }
    return edges, branch_table, report


def run_sequence(seq: str, root: Path, skip_trackeval: bool = False) -> dict[str, Any]:
    verify_prereg(root)
    if not (root / "input_manifest.json").exists() or not (root / "implementation_manifest.json").exists():
        raise RuntimeError("input and implementation freezes are required")
    output = root / "capacity" / seq
    report_path = output / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    started = time.time()
    m57 = load_module(f"m70_m57_{seq[-2:]}", "scripts/m23_research/m23_57_intra_node_change_point_capacity.py")
    m53 = load_module(f"m70_m53_{seq[-2:]}", "scripts/m23_research/m23_53_global_identity_flow_capacity.py")
    (
        _m10, _m53_from_57, m53b, source_path, baseline_path, source_rows,
        fixed_nodes, fixed_chunk_rows, parent_ids, crowd, mapped, _match_iou, row_embeddings,
    ) = m57.prepare_observable_rows(seq)
    boundaries = pd.read_parquet(M57_ROOT / f"boundary_universe/{seq}/boundary_features.parquet")
    mandatory = (
        boundaries.loc[boundaries.parent_id_transition > 0]
        .groupby("fixed_chunk_id").position.apply(lambda x: sorted(set(map(int, x)))).to_dict()
    )
    event(root, "split_teacher_opened_after_freezes", seq=seq, teacher_only=True)
    labeler = load_module(f"m70_label_{seq[-2:]}", "scripts/m23_research/m23_11_add_micrograph_utility.py")
    labeler.PARENT = SOURCE_PARENT
    m23 = labeler.load_m23()
    label_rows = labeler.read_tracker(source_path)
    row_gt = labeler.matched_gt_per_row(label_rows, seq, m23)
    cuts, transitions, split_report = causal_teacher_cuts(
        fixed_nodes, fixed_chunk_rows, source_rows, row_gt, mandatory
    )
    nodes, _full_prototypes_unused = m53b.build_adaptive_nodes(
        source_rows=source_rows,
        fixed_nodes=fixed_nodes,
        chunk_rows=fixed_chunk_rows,
        selected_boundaries=cuts,
        row_embeddings=row_embeddings,
        mapped=mapped,
        parent_ids=parent_ids,
        crowd_density=crowd,
    )
    rows_by_node = adaptive_row_map(fixed_nodes, fixed_chunk_rows, cuts, nodes)
    edges, branches, candidate_report = causal_candidate_edges(
        nodes, rows_by_node, source_rows, row_embeddings, mapped
    )

    audit = output / "postfreeze_audit"
    audit.mkdir(parents=True, exist_ok=True)
    transition_path = audit / "causal_transition_eligibility.parquet"
    transitions.to_parquet(transition_path, index=False)
    frozen = output / "frozen_candidate_graph"
    frozen.mkdir(parents=True, exist_ok=True)
    nodes_path, edges_path = frozen / "nodes.parquet", frozen / "edges.parquet"
    branch_path = frozen / "causal_branch_table.parquet"
    nodes.to_parquet(nodes_path, index=False)
    edges.to_parquet(edges_path, index=False)
    branches.to_parquet(branch_path, index=False)

    empty = pd.DataFrame(columns=["src_chunk", "dst_chunk"])
    reconstructed = output / "baseline_reconstruction/track_results" / f"{seq}.txt"
    baseline_report = m53.write_tracker(seq, source_path, nodes, empty, reconstructed, preserve_parent_ids=True)
    baseline_exact = reconstructed.read_bytes() == baseline_path.read_bytes()
    if not baseline_exact:
        raise RuntimeError(f"{seq}: parent-ID reconstruction is not byte-exact M23-46")

    graph_manifest = {
        "experiment_id": "M23-70A",
        "seq": seq,
        "teacher_only": True,
        "deployable": False,
        "candidate_graph_frozen_before_path_cover_teacher": True,
        "split_state_gt_derived_capacity_only": True,
        "input_manifest_sha256": sha(root / "input_manifest.json"),
        "implementation_manifest_sha256": sha(root / "implementation_manifest.json"),
        "split_report": split_report,
        "candidate_report": candidate_report,
        "baseline_reconstruction": {**baseline_report, "byte_exact": baseline_exact},
        "frozen_artifacts": {
            "nodes": str(nodes_path), "nodes_sha256": sha(nodes_path), "node_rows": len(nodes),
            "edges": str(edges_path), "edges_sha256": sha(edges_path), "edge_rows": len(edges),
            "branch_table": str(branch_path), "branch_table_sha256": sha(branch_path),
        },
        "protocol": {
            "branch_count": K,
            "alternatives_per_source": ALTERNATIVES,
            "delay_frames": DELAY,
            "complete_future_destination_descriptor_used": False,
            "teacher_objective": "M23-53 maximum-weight path cover over frozen causal candidates",
        },
    }
    manifest_path = frozen / "freeze_manifest.json"
    write_json(manifest_path, graph_manifest)
    event(root, "causal_candidate_graph_frozen", seq=seq, manifest_sha256=sha(manifest_path))

    nodes, selected, teacher_report = m53.build_teacher_utilities(
        seq=seq, source_parent_root=SOURCE_PARENT, output_root=output, freeze_manifest=graph_manifest
    )
    tracker = output / "track_results" / f"{seq}.txt"
    tracker_report = m53.write_tracker(seq, source_path, nodes, selected, tracker)
    payload_unchanged = m57.detector_payload(source_path) == m57.detector_payload(tracker)
    if not payload_unchanged:
        raise RuntimeError(f"{seq}: detector payload changed")
    official = None if skip_trackeval else m53.run_official_trackeval(
        seq=seq, output_root=output, tracker_name=f"m23_70a_{seq[-2:]}"
    )
    report = {
        "experiment_id": "M23-70A",
        "seq": seq,
        "status": "completed",
        "teacher_only": True,
        "deployable": False,
        "split_report": split_report,
        "candidate_report": candidate_report,
        "teacher": teacher_report,
        "integrity": {
            "baseline_m23_46_byte_exact": baseline_exact,
            "detection_payload_unchanged": payload_unchanged,
            "one_to_one": bool(teacher_report["one_to_one"]),
            "time_forward": bool(teacher_report["time_forward"]),
            "acyclic": bool(teacher_report["acyclic"]),
            "complete_future_destination_descriptor_used": False,
        },
        "official_trackeval": official,
        "tracker": {**tracker_report, "sha256": sha(tracker)},
        "artifacts": {
            "transition_eligibility_sha256": sha(transition_path),
            "candidate_manifest_sha256": sha(manifest_path),
            "tracker_sha256": sha(tracker),
        },
        "runtime_seconds": time.time() - started,
        "peak_rss_mb": peak_rss_mb(),
        # This audit is CPU-only. Avoid initializing CUDA solely for telemetry.
        "peak_gpu_memory_mb": 0.0,
    }
    write_json(report_path, report)
    event(root, "sequence_completed", seq=seq, HOTA=None if official is None else official["HOTA"])
    return report


def combine(root: Path) -> dict[str, Any]:
    verify_prereg(root)
    combined_root = root / "capacity_combined"
    report_path = combined_root / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    baseline = json.loads(M46_REPORT.read_text(encoding="utf-8"))
    m57 = json.loads((M57_ROOT / "capacity_combined/report.json").read_text(encoding="utf-8"))["official_trackeval"]
    track_results = combined_root / "track_results"
    track_results.mkdir(parents=True, exist_ok=True)
    fold_reports: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for seq in SEQUENCES:
        report = root / "capacity" / seq / "report.json"
        if not report.exists():
            raise FileNotFoundError(report)
        fold_reports[seq] = json.loads(report.read_text(encoding="utf-8"))
        src = root / "capacity" / seq / "track_results" / f"{seq}.txt"
        dst = track_results / f"{seq}.txt"
        shutil.copy2(src, dst)
        hashes[f"{seq}.txt"] = sha(dst)
    evaluator = load_module("m70_combined_eval", "scripts/m23_research/m23_45_domain_robust_source_cut_student.py")
    metrics = evaluator.evaluate_detailed(track_results, combined_root / "official_eval", "m23_70a_causal_branch_capacity", SEQUENCES)
    non_degradation = {
        seq: metrics[seq]["HOTA"] >= baseline["folds"][seq]["HOTA"] for seq in SEQUENCES
    }
    integrity = {
        seq: all([
            fold_reports[seq]["integrity"]["baseline_m23_46_byte_exact"],
            fold_reports[seq]["integrity"]["detection_payload_unchanged"],
            fold_reports[seq]["integrity"]["one_to_one"],
            fold_reports[seq]["integrity"]["time_forward"],
            fold_reports[seq]["integrity"]["acyclic"],
            not fold_reports[seq]["integrity"]["complete_future_destination_descriptor_used"],
        ]) for seq in SEQUENCES
    }
    combined = metrics["COMBINED"]
    passed = bool(combined["HOTA"] >= GO_HOTA and all(non_degradation.values()) and all(integrity.values()))
    decision = "PASS_CAUSAL_BRANCH_CAPACITY_AUTHORIZE_M23_70B" if passed else "FAIL_CAUSAL_BRANCH_CAPACITY_CLOSE_OR_REDESIGN_BEFORE_TRAINING"
    delta46 = {
        scope: {
            key: metrics[scope][key] - (baseline["metrics"][key] if scope == "COMBINED" else baseline["folds"][scope][key])
            for key in ("HOTA", "DetA", "AssA", "IDSW")
        } for scope in (*SEQUENCES, "COMBINED")
    }
    payload = {
        "experiment_id": "M23-70A",
        "status": "completed",
        "teacher_only": True,
        "deployable": False,
        "official_trackeval": metrics,
        "baseline_m23_46": baseline,
        "m23_57_unconstrained_teacher_reference": m57,
        "delta_vs_m23_46": delta46,
        "capacity_gate": {
            "threshold_HOTA": GO_HOTA,
            "combined_hota_pass": combined["HOTA"] >= GO_HOTA,
            "per_sequence_non_degradation": non_degradation,
            "integrity": integrity,
            "pass": passed,
            "margin": combined["HOTA"] - GO_HOTA,
        },
        "decision": decision,
        "m23_70b_authorized": passed,
        "training_runs": 0,
        "mot20_test_reads": 0,
        "test_submission": False,
        "trackeval_runs": 5,
        "tracker_sha256": hashes,
    }
    write_json(report_path, payload)
    rows = [{
        "experiment_id": "M23-70A", "scope": scope,
        **{k: metrics[scope][k] for k in ("HOTA", "DetA", "AssA", "IDSW")},
        "delta_hota_vs_m23_46": delta46[scope]["HOTA"],
        "teacher_only": 1, "deployable": 0, "gate_pass": int(passed), "decision": decision,
    } for scope in (*SEQUENCES, "COMBINED")]
    pd.DataFrame(rows).to_csv(root / "summary.csv", index=False)
    final = {
        "experiment_id": "M23-70A", "status": "completed", "decision": decision,
        "hota": combined["HOTA"], "teacher_only": True, "deployable": False,
        "m23_70b_authorized": passed, "training_runs": 0, "trackeval_runs": 5,
        "mot20_test_reads": 0, "test_submission": False, "capacity_gate": payload["capacity_gate"],
    }
    write_json(root / "final_summary.json", final)
    lines = [
        "# M23-70A Genuine Causal Branch Capacity — Result", "", f"Decision: **{decision}**", "",
        "This is a teacher-only capacity audit, not a deployable result.", "",
        "| Scope | HOTA | DetA | AssA | IDSW | ΔHOTA vs M23-46 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scope in (*SEQUENCES, "COMBINED"):
        lines.append(f"| {scope} | {metrics[scope]['HOTA']:.6f} | {metrics[scope]['DetA']:.6f} | {metrics[scope]['AssA']:.6f} | {int(metrics[scope]['IDSW'])} | {delta46[scope]['HOTA']:+.6f} |")
    lines += ["", f"- K: `{K}`.", f"- Delay: `{DELAY}` frames.", f"- Gate: `{GO_HOTA:.2f}` HOTA.", f"- Combined HOTA: `{combined['HOTA']:.6f}`.", f"- M23-70B authorized: `{str(passed).lower()}`.", "- MOT20 test submission: `false`."]
    result_doc = Path("docs/m23_70a_causal_branch_capacity_result_20260725.md")
    result_doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    event(root, "combined_completed", HOTA=combined["HOTA"], decision=decision, result_doc_sha256=sha(result_doc))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze-inputs")
    sub.add_parser("freeze-implementation")
    fold = sub.add_parser("run-sequence")
    fold.add_argument("--seq", required=True, choices=SEQUENCES)
    fold.add_argument("--skip-trackeval", action="store_true")
    sub.add_parser("combine")
    args = parser.parse_args()
    if args.command == "freeze-inputs":
        result = freeze_inputs(args.root)
    elif args.command == "freeze-implementation":
        result = freeze_implementation(args.root)
    elif args.command == "run-sequence":
        result = run_sequence(args.seq, args.root, args.skip_trackeval)
    elif args.command == "combine":
        result = combine(args.root)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
