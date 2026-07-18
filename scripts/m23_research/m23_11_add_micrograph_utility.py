from __future__ import annotations

# Research artifact for the MOT20 M23 sequence-LOSO audit.

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SEQS = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
PARENT = Path(
    "outputs/assocriskbench_p15_20260714/"
    "fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results"
)
ROOT = Path("outputs/mot20_m23_20260718/micrograph_chunk30_v1")
OUT = Path("outputs/mot20_m23_20260718/micrograph_chunk30_assa_utility_v1")
CHUNK_SPAN = 30
GAP_BREAK = 1


def load_m23():
    spec = importlib.util.spec_from_file_location(
        "m23", "scripts/audit_m23_mot20_expanded_evidence_oracle.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["m23"] = module
    spec.loader.exec_module(module)
    return module


def read_tracker(path: Path) -> list[dict]:
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
            }
        )
    return rows


def line_chunks(rows: list[dict], meta: pd.DataFrame) -> np.ndarray:
    by_track = defaultdict(list)
    for row_index, row in enumerate(rows):
        by_track[row["track_id"]].append(row_index)
    mapping = np.full(len(rows), -1, np.int32)
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
            mapping[np.asarray(indices[start:end], dtype=np.int64)] = chunk_id
            chunk_id += 1
            ordinal += 1
            start = end
    if chunk_id != len(meta) or np.any(mapping < 0):
        raise RuntimeError(f"chunk reconstruction incomplete: {chunk_id}, {len(meta)}")
    return mapping


def matched_gt_per_row(rows: list[dict], seq: str, m23) -> np.ndarray:
    labels = np.zeros(len(rows), np.int32)
    baseline = m23.load_baseline(PARENT / f"{seq}.txt")
    gt = m23.load_gt(Path("datasets/MOT20/train") / seq / "gt" / "gt.txt")
    for frame in sorted(baseline):
        kept, valid, _ = m23.valid_and_distractor_filtered(baseline[frame], gt.get(frame, []))
        for candidate_index, gt_index, _ in m23.match_candidates(kept, valid):
            line_index = int(kept[candidate_index].uid & ((1 << 32) - 1))
            labels[line_index] = int(valid[gt_index].gt_id)
    return labels


def gt_counts(seq: str) -> Counter:
    counts = Counter()
    path = Path("datasets/MOT20/train") / seq / "gt" / "gt.txt"
    for line in path.open():
        fields = line.strip().split(",")
        if len(fields) < 8:
            continue
        mark = int(float(fields[6]))
        category = int(float(fields[7]))
        if mark > 0 and category == 1:
            counts[int(float(fields[1]))] += 1
    return counts


def association_contribution(gt_total: Counter, tracker_rows: int, matches: Counter) -> float:
    return float(
        sum(
            (matched * matched) / max(1, gt_total[gt_id] + tracker_rows - matched)
            for gt_id, matched in matches.items()
        )
    )


def quantiles(values: pd.Series) -> dict[str, float]:
    if values.empty:
        return {}
    levels = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
    return {str(level): float(value) for level, value in values.quantile(levels).items()}


def label_sequence(seq: str, m23) -> dict:
    rows = read_tracker(PARENT / f"{seq}.txt")
    meta = pd.read_parquet(ROOT / seq / "microtracklets.parquet")
    edges = pd.read_parquet(ROOT / seq / "candidate_edges.parquet")
    row_chunk = line_chunks(rows, meta)
    row_gt = matched_gt_per_row(rows, seq, m23)
    totals = gt_counts(seq)

    chunk_matches = [Counter() for _ in range(len(meta))]
    for chunk_id, gt_id in zip(row_chunk, row_gt):
        if gt_id > 0:
            chunk_matches[int(chunk_id)][int(gt_id)] += 1
    chunk_rows = meta.rows.to_numpy(int)
    before = np.asarray(
        [
            association_contribution(totals, int(chunk_rows[index]), chunk_matches[index])
            for index in range(len(meta))
        ],
        dtype=np.float64,
    )

    delta = np.empty(len(edges), np.float64)
    after = np.empty(len(edges), np.float64)
    before_pair = np.empty(len(edges), np.float64)
    shared_gt = np.zeros(len(edges), np.int8)
    combined_dominant_gt = np.full(len(edges), -1, np.int32)
    src_matched = np.empty(len(edges), np.int32)
    dst_matched = np.empty(len(edges), np.int32)
    for edge_index, edge in enumerate(edges.itertuples(index=False)):
        src = int(edge.src_chunk)
        dst = int(edge.dst_chunk)
        src_counts = chunk_matches[src]
        dst_counts = chunk_matches[dst]
        merged_counts = src_counts + dst_counts
        pair_before = before[src] + before[dst]
        pair_after = association_contribution(
            totals,
            int(chunk_rows[src] + chunk_rows[dst]),
            merged_counts,
        )
        before_pair[edge_index] = pair_before
        after[edge_index] = pair_after
        delta[edge_index] = pair_after - pair_before
        shared_gt[edge_index] = int(bool(set(src_counts) & set(dst_counts)))
        if merged_counts:
            combined_dominant_gt[edge_index] = int(merged_counts.most_common(1)[0][0])
        src_matched[edge_index] = int(sum(src_counts.values()))
        dst_matched[edge_index] = int(sum(dst_counts.values()))

    edges["assa_edge_before_proxy"] = before_pair
    edges["assa_edge_after_proxy"] = after
    edges["assa_edge_delta_proxy"] = delta
    edges["assa_edge_positive"] = (delta > 0).astype(np.int8)
    edges["assa_edge_negative"] = (delta < 0).astype(np.int8)
    edges["assa_edge_delta_per_row"] = delta / np.maximum(
        1, edges.src_rows.to_numpy(float) + edges.dst_rows.to_numpy(float)
    )
    edges["shares_any_gt"] = shared_gt
    edges["combined_dominant_gt"] = combined_dominant_gt
    edges["diagnostic_src_matched_rows"] = src_matched
    edges["diagnostic_dst_matched_rows"] = dst_matched
    output = OUT / seq
    output.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(output / "candidate_edges_utility.parquet", index=False)

    roles = {
        "all": np.ones(len(edges), dtype=bool),
        "source_adjacent": edges.source_adjacent.to_numpy(int) == 1,
        "cross": edges.same_source.to_numpy(int) == 0,
    }
    role_reports = {}
    for role, mask in roles.items():
        part = edges.loc[mask]
        role_reports[role] = {
            "rows": len(part),
            "positive": int(part.assa_edge_positive.sum()),
            "negative": int(part.assa_edge_negative.sum()),
            "zero": int((part.assa_edge_delta_proxy == 0).sum()),
            "positive_rate": float(part.assa_edge_positive.mean()) if len(part) else 0.0,
            "delta_sum": float(part.assa_edge_delta_proxy.sum()),
            "positive_sum": float(part.loc[part.assa_edge_delta_proxy > 0, "assa_edge_delta_proxy"].sum()),
            "negative_sum": float(part.loc[part.assa_edge_delta_proxy < 0, "assa_edge_delta_proxy"].sum()),
            "positive_quantiles": quantiles(part.loc[part.assa_edge_delta_proxy > 0, "assa_edge_delta_proxy"]),
            "negative_quantiles": quantiles(part.loc[part.assa_edge_delta_proxy < 0, "assa_edge_delta_proxy"]),
        }
    report = {
        "seq": seq,
        "tracker_rows": len(rows),
        "chunks": len(meta),
        "edges": len(edges),
        "matched_rows": int((row_gt > 0).sum()),
        "roles": role_reports,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m23 = load_m23()
    reports = [label_sequence(seq, m23) for seq in SEQS]
    report = {
        "oracle_labels": True,
        "deployment_allowed": False,
        "gt_use": "training/evaluation edge-utility labels only",
        "formula": "sum_g m_g^2 / (gt_count_g + tracker_rows - m_g); delta after merging two chunks minus separate contributions",
        "features": "unchanged GT-free micrograph observables; diagnostic GT fields must be excluded by explicit feature allowlist",
        "sequences": reports,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
