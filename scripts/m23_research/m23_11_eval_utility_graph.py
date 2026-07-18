from __future__ import annotations

# Research artifact for the MOT20 M23 sequence-LOSO audit.

import csv
import json
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
PRED = Path("outputs/mot20_m23_20260718/micrograph_chunk30_utility_loso_v1")
OUT = Path("outputs/mot20_m23_20260718/micrograph_chunk30_utility_graph_explore_v1")
NAME = "raw_expected_utility"
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
        raise RuntimeError(
            f"chunk count mismatch: {chunk_id}, {len(meta)}, {len(mapping)}, {len(rows)}"
        )
    return mapping


def fixed_candidates(source: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    source = source.copy()
    source["edge_role"] = "source"
    source["utility"] = np.maximum(
        source.pred_keep_expected_utility.to_numpy(float), 1e-6
    )
    cross = cross[cross.pred_risk_adjusted_utility > 0].copy()
    cross["edge_role"] = "cross"
    cross["utility"] = cross.pred_risk_adjusted_utility.to_numpy(float)
    candidates = pd.concat([source, cross], ignore_index=True, sort=False)
    candidates.sort_values("utility", ascending=False, inplace=True)
    candidates.drop_duplicates(["src_chunk", "dst_chunk"], keep="first", inplace=True)
    return candidates


def maximum_weight_path_edges(candidates: pd.DataFrame, chunk_count: int) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    src = candidates.src_chunk.to_numpy(int)
    dst = candidates.dst_chunk.to_numpy(int)
    utility = candidates.utility.to_numpy(float)
    if (
        np.any(src < 0)
        or np.any(src >= chunk_count)
        or np.any(dst < 0)
        or np.any(dst >= chunk_count)
        or np.any(~np.isfinite(utility))
        or np.any(utility <= 0)
    ):
        raise RuntimeError("invalid candidate graph")
    offset = float(utility.max()) + 1.0
    rows = np.concatenate([src, np.arange(chunk_count, dtype=int)])
    cols = np.concatenate([dst, chunk_count + np.arange(chunk_count, dtype=int)])
    data = np.concatenate([offset - utility, np.full(chunk_count, offset, float)])
    matrix = coo_matrix(
        (data, (rows, cols)), shape=(chunk_count, 2 * chunk_count)
    ).tocsr()
    row_index, col_index = min_weight_full_bipartite_matching(matrix)
    real = col_index < chunk_count
    pairs = set(zip(row_index[real].tolist(), col_index[real].tolist()))
    keep = [
        (int(src_id), int(dst_id)) in pairs
        for src_id, dst_id in zip(candidates.src_chunk, candidates.dst_chunk)
    ]
    return candidates[keep].copy()


def chains(selected: pd.DataFrame, chunk_count: int) -> dict[int, int]:
    successor = {int(row.src_chunk): int(row.dst_chunk) for row in selected.itertuples()}
    predecessor = {dst: src for src, dst in successor.items()}
    if len(successor) != len(selected) or len(predecessor) != len(selected):
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
    output.sort(
        key=lambda fields: (
            int(float(fields[0])),
            int(float(fields[1])),
            float(fields[2]),
            float(fields[3]),
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for fields in output:
            handle.write(",".join(fields) + "\n")
    source = selected[selected.edge_role == "source"]
    cross = selected[selected.edge_role == "cross"]
    source_total = int((meta.source_ordinal.to_numpy(int) > 0).sum())
    return {
        "seq": seq,
        "rows": len(rows),
        "chunks": len(meta),
        "chains": len(set(assignment.values())),
        "selected_edges": len(selected),
        "selected_source_edges": len(source),
        "source_edges_replaced": source_total - len(source),
        "selected_cross_edges": len(cross),
        "predicted_utility_sum": float(selected.utility.sum()),
        "diagnostic_actual_assa_proxy_sum": float(
            selected.assa_edge_delta_proxy.fillna(0).sum()
        ),
        "diagnostic_cross_actual_assa_proxy_sum": float(
            cross.assa_edge_delta_proxy.fillna(0).sum()
        ),
        "diagnostic_cross_positive": int(cross.assa_edge_positive.sum()),
        "diagnostic_cross_precision": float(cross.assa_edge_positive.mean())
        if len(cross)
        else None,
    }


def evaluate(root: Path) -> dict:
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
        NAME,
        "--work-dir",
        str(root / "eval_work"),
        "--seqs",
        *SEQS,
    ]
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    (root / "eval.log").write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(completed.stdout[-4000:])
    detailed = root / "eval_work" / "eval" / NAME / "pedestrian_detailed.csv"
    rows = list(csv.DictReader(detailed.open()))
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


def main() -> None:
    root = OUT / NAME
    sequence_reports = []
    for seq in SEQS:
        meta = pd.read_parquet(DATA / seq / "microtracklets.parquet")
        source = pd.read_parquet(PRED / f"{seq}_source_utility_predictions.parquet")
        cross = pd.read_parquet(PRED / f"{seq}_cross_utility_predictions.parquet")
        candidates = fixed_candidates(source, cross)
        selected = maximum_weight_path_edges(candidates, len(meta))
        selected_dir = root / "selected_edges"
        selected_dir.mkdir(parents=True, exist_ok=True)
        selected.to_parquet(selected_dir / f"{seq}.parquet", index=False)
        sequence_reports.append(
            write_tracker(seq, meta, selected, root / "track_results" / f"{seq}.txt")
        )
    report = {
        "name": NAME,
        "status": "exploratory fixed mechanism test; not a nested selected policy",
        "deployment_allowed": False,
        "gt_used_in_selection_or_inference": False,
        "held_sequence_excluded_from_utility_model_fit": True,
        "rule": {
            "source": "all source-adjacent edges; weight=max(pred_keep_expected_utility, 1e-6)",
            "cross": "all pred_risk_adjusted_utility > 0; weight=pred_risk_adjusted_utility",
            "optimizer": "maximum predicted-utility one-to-one temporal path cover",
            "held_out_threshold_tuning": "none",
        },
        "by_seq": sequence_reports,
        "eval": evaluate(root),
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
