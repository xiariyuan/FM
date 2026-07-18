from __future__ import annotations

# Research artifact for the MOT20 M23 sequence-LOSO audit.

import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


SEQS = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
PARENT = Path(
    "outputs/assocriskbench_p15_20260714/"
    "fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results"
)
DATA = Path("outputs/mot20_m23_20260718/micrograph_chunk30_v1")
PRED = Path("outputs/mot20_m23_20260718/micrograph_chunk30_role_loso_v1")
OUT = Path("outputs/mot20_m23_20260718/micrograph_chunk30_role_policy_explore_v1")
SYN_BASE = 10_000_000
STRIDE = 1_000_000
CHUNK_SPAN = 30
GAP_BREAK = 1


def read_parent(path: Path) -> list[dict]:
    rows = []
    for line_index, line in enumerate(path.open()):
        fields = line.rstrip("\n").split(",")
        if len(fields) < 6:
            continue
        rows.append(
            {
                "line": line_index,
                "frame": int(float(fields[0])),
                "track_id": int(float(fields[1])),
                "fields": fields,
            }
        )
    return rows


def line_chunks(rows: list[dict], meta: pd.DataFrame) -> dict[int, int]:
    by_track = defaultdict(list)
    for row_index, row in enumerate(rows):
        by_track[row["track_id"]].append(row_index)
    mapping = {}
    chunk_id = 0
    for track_id, indices in sorted(by_track.items()):
        indices.sort(key=lambda index: (rows[index]["frame"], index))
        start = 0
        ordinal = 0
        for end in range(1, len(indices) + 1):
            boundary = (
                end == len(indices)
                or rows[indices[end]]["frame"] - rows[indices[end - 1]]["frame"] > GAP_BREAK
                or rows[indices[end]]["frame"] - rows[indices[start]]["frame"] >= CHUNK_SPAN
            )
            if not boundary:
                continue
            record = meta.iloc[chunk_id]
            if (
                int(record.source_track_id) != track_id
                or int(record.source_ordinal) != ordinal
                or int(record.first_frame) != rows[indices[start]]["frame"]
                or int(record.last_frame) != rows[indices[end - 1]]["frame"]
            ):
                raise RuntimeError(f"chunk reconstruction mismatch at {chunk_id}")
            for index in indices[start:end]:
                mapping[index] = chunk_id
            chunk_id += 1
            ordinal += 1
            start = end
    if chunk_id != len(meta) or len(mapping) != len(rows):
        raise RuntimeError(f"chunk count mismatch: {chunk_id}, {len(meta)}, {len(mapping)}, {len(rows)}")
    return mapping


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def role_candidates(frame: pd.DataFrame, probability: str, threshold: float, role: str) -> pd.DataFrame:
    probability_values = np.clip(frame[probability].to_numpy(float), 1e-6, 1 - 1e-6)
    logit = np.log(probability_values / (1 - probability_values))
    threshold_logit = math.log(threshold / (1 - threshold))
    keep = probability_values >= threshold
    selected = frame.loc[keep].copy()
    selected["edge_role"] = role
    selected["role_probability"] = probability_values[keep]
    selected["role_threshold"] = threshold
    selected["benefit"] = logit[keep] - threshold_logit
    selected["utility"] = sigmoid(selected.benefit.to_numpy(float))
    return selected


def maximum_weight_path_edges(candidates: pd.DataFrame, chunk_count: int) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    src = candidates.src_chunk.to_numpy(int)
    dst = candidates.dst_chunk.to_numpy(int)
    if np.any(src < 0) or np.any(src >= chunk_count) or np.any(dst < 0) or np.any(dst >= chunk_count):
        raise RuntimeError("chunk id outside graph")
    utility = np.clip(candidates.utility.to_numpy(float), 0, 1)
    cost = 1.000001 - utility
    rows = np.concatenate([src, np.arange(chunk_count, dtype=int)])
    cols = np.concatenate([dst, chunk_count + np.arange(chunk_count, dtype=int)])
    data = np.concatenate([cost, np.ones(chunk_count, float)])
    matrix = coo_matrix((data, (rows, cols)), shape=(chunk_count, 2 * chunk_count)).tocsr()
    row_index, col_index = min_weight_full_bipartite_matching(matrix)
    real = col_index < chunk_count
    pairs = set(zip(row_index[real].tolist(), col_index[real].tolist()))
    return candidates[
        [(int(src_id), int(dst_id)) in pairs for src_id, dst_id in zip(candidates.src_chunk, candidates.dst_chunk)]
    ].copy()


def chains(selected: pd.DataFrame, chunk_count: int) -> dict[int, int]:
    successor = {int(row.src_chunk): int(row.dst_chunk) for row in selected.itertuples()}
    predecessor = {dst: src for src, dst in successor.items()}
    if len(successor) != len(predecessor):
        raise RuntimeError("selected graph is not one-to-one")
    roots = [chunk_id for chunk_id in range(chunk_count) if chunk_id not in predecessor]
    assignment = {}
    for root in roots:
        current = root
        seen = set()
        while current not in seen:
            seen.add(current)
            assignment[current] = root
            if current not in successor:
                break
            current = successor[current]
        else:
            raise RuntimeError("cycle in selected graph")
    if len(assignment) != chunk_count:
        raise RuntimeError(f"unassigned chunks: {chunk_count - len(assignment)}")
    return assignment


def write_tracker(seq: str, meta: pd.DataFrame, selected: pd.DataFrame, path: Path) -> dict:
    rows = read_parent(PARENT / f"{seq}.txt")
    mapping = line_chunks(rows, meta)
    assignment = chains(selected, len(meta))
    base = SYN_BASE + SEQS.index(seq) * STRIDE
    output = []
    seen = set()
    for row_index, row in enumerate(rows):
        chunk_id = mapping[row_index]
        new_id = base + assignment[chunk_id]
        if new_id >= (1 << 24):
            raise RuntimeError("synthetic ID exceeds exact float32 integer range")
        fields = list(row["fields"])
        fields[1] = str(new_id)
        key = (row["frame"], new_id)
        if key in seen:
            raise RuntimeError(f"{seq}: duplicate frame/id {key}")
        seen.add(key)
        output.append(fields)
    output.sort(key=lambda fields: (int(float(fields[0])), int(float(fields[1])), float(fields[2]), float(fields[3])))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for fields in output:
            handle.write(",".join(fields) + "\n")
    cross = selected[selected.edge_role == "cross"] if len(selected) else selected
    source = selected[selected.edge_role == "source"] if len(selected) else selected
    return {
        "seq": seq,
        "rows": len(rows),
        "chunks": len(meta),
        "chains": len(set(assignment.values())),
        "selected_edges": len(selected),
        "selected_source_edges": len(source),
        "selected_cross_edges": len(cross),
        "diagnostic_source_same_gt": int(source.same_gt.sum()) if len(source) else 0,
        "diagnostic_source_precision": float(source.same_gt.mean()) if len(source) else None,
        "diagnostic_cross_same_gt": int(cross.same_gt.sum()) if len(cross) else 0,
        "diagnostic_cross_precision": float(cross.same_gt.mean()) if len(cross) else None,
    }


def evaluate(root: Path, name: str) -> dict:
    command = [
        sys.executable,
        "scripts/eval_motstyle_trackeval.py",
        "--benchmark-name",
        "MOT20",
        "--split-to-eval",
        "train",
        "--gt-root",
        "datasets/MOT20/train",
        "--results-dir",
        str(root / "track_results"),
        "--tracker-name",
        name,
        "--work-dir",
        str(root / "eval_work"),
        "--seqs",
        *SEQS,
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (root / "eval.log").write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(completed.stdout[-4000:])
    rows = list(csv.DictReader((root / "eval_work" / "eval" / name / "pedestrian_detailed.csv").open()))
    result = {}
    for seq in SEQS + ["COMBINED"]:
        row = next(item for item in rows if item["seq"] == seq)
        result[seq] = {
            "HOTA": 100 * float(row["HOTA___AUC"]),
            "DetA": 100 * float(row["DetA___AUC"]),
            "AssA": 100 * float(row["AssA___AUC"]),
            "IDSW": int(float(row["IDSW"])),
        }
    return result


def select_policy(
    name: str,
    boundary: pd.DataFrame,
    cross: pd.DataFrame,
    chunk_count: int,
) -> tuple[pd.DataFrame, dict]:
    all_source = role_candidates(boundary, "pred_keep_prob", 1e-5, "source")
    all_source["utility"] = 0.999999
    all_source["benefit"] = 20.0
    if name == "parent_reconstruction":
        return all_source, {"source_keep_threshold": 0.0, "cross_link_threshold": None}
    if name == "cross95_preserve":
        cross_candidates = role_candidates(cross, "pred_link_prob", 0.95, "cross")
        used_src = set(all_source.src_chunk.astype(int))
        used_dst = set(all_source.dst_chunk.astype(int))
        cross_candidates = cross_candidates[
            ~cross_candidates.src_chunk.astype(int).isin(used_src)
            & ~cross_candidates.dst_chunk.astype(int).isin(used_dst)
        ]
        cross_selected = maximum_weight_path_edges(cross_candidates, chunk_count)
        return pd.concat([all_source, cross_selected], ignore_index=True, sort=False), {
            "source_keep_threshold": 0.0,
            "cross_link_threshold": 0.95,
        }
    source_candidates = role_candidates(boundary, "pred_keep_prob", 0.30, "source")
    if name == "split30_only":
        return source_candidates, {"source_keep_threshold": 0.30, "cross_link_threshold": None}
    if name == "joint30_95":
        cross_candidates = role_candidates(cross, "pred_link_prob", 0.95, "cross")
        candidates = pd.concat([source_candidates, cross_candidates], ignore_index=True, sort=False)
        return maximum_weight_path_edges(candidates, chunk_count), {
            "source_keep_threshold": 0.30,
            "cross_link_threshold": 0.95,
        }
    raise ValueError(name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {}
    for seq in SEQS:
        cache[seq] = (
            pd.read_parquet(DATA / seq / "microtracklets.parquet"),
            pd.read_parquet(PRED / f"{seq}_boundary_predictions.parquet"),
            pd.read_parquet(PRED / f"{seq}_cross_predictions.parquet"),
        )
    reports = []
    for name in ["parent_reconstruction", "cross95_preserve", "split30_only", "joint30_95"]:
        root = OUT / name
        sequence_reports = []
        policy_parameters = None
        for seq in SEQS:
            meta, boundary, cross = cache[seq]
            selected, parameters = select_policy(name, boundary, cross, len(meta))
            if policy_parameters is None:
                policy_parameters = parameters
            elif policy_parameters != parameters:
                raise RuntimeError("policy parameters changed across sequences")
            selected_dir = root / "selected_edges"
            selected_dir.mkdir(parents=True, exist_ok=True)
            selected.to_parquet(selected_dir / f"{seq}.parquet", index=False)
            sequence_reports.append(write_tracker(seq, meta, selected, root / "track_results" / f"{seq}.txt"))
        report = {
            "name": name,
            "status": "exploratory mechanism ablation; not nested policy selection",
            "deployment_allowed": False,
            "gt_used_in_selection_or_inference": False,
            "held_sequence_excluded_from_role_model_fit": True,
            "policy_parameters": policy_parameters,
            "by_seq": sequence_reports,
            "eval": evaluate(root, name),
        }
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)
        print(json.dumps({"name": name, "by_seq": sequence_reports, "combined": report["eval"]["COMBINED"]}, indent=2), flush=True)
    (OUT / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
