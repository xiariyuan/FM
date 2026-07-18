from __future__ import annotations

# Research artifact for the MOT20 M23 sequence-LOSO audit.

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


SEQS = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
PARENT = Path(
    "outputs/assocriskbench_p15_20260714/"
    "fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results"
)
DATA = Path("outputs/mot20_m23_20260718/micrograph_chunk30_v1")
UTILITY = Path("outputs/mot20_m23_20260718/micrograph_chunk30_assa_utility_v1")
OUT = Path("outputs/mot20_m23_20260718/micrograph_chain_transaction_oracle_v1")
NAME = "greedy_disjoint_positive_chain_transaction_oracle"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def add_counts(left: Counter, right: Counter) -> Counter:
    output = left.copy()
    output.update(right)
    return output


def build_track_parts(meta: pd.DataFrame, chunk_matches: list[Counter]):
    by_track = defaultdict(list)
    for chunk in meta.itertuples():
        by_track[int(chunk.source_track_id)].append(
            (int(chunk.source_ordinal), int(chunk.chunk_id), int(chunk.rows))
        )
    parts = {}
    for track_id, chunks in by_track.items():
        chunks.sort()
        ids = [chunk_id for _, chunk_id, _ in chunks]
        if [ordinal for ordinal, _, _ in chunks] != list(range(len(chunks))):
            raise RuntimeError(f"non-contiguous ordinals for track {track_id}")
        prefix = []
        running_counts = Counter()
        running_rows = 0
        for _, chunk_id, rows in chunks:
            running_counts = add_counts(running_counts, chunk_matches[chunk_id])
            running_rows += rows
            prefix.append((running_counts, running_rows))
        suffix = [None] * len(chunks)
        running_counts = Counter()
        running_rows = 0
        for index in range(len(chunks) - 1, -1, -1):
            _, chunk_id, rows = chunks[index]
            running_counts = add_counts(running_counts, chunk_matches[chunk_id])
            running_rows += rows
            suffix[index] = (running_counts, running_rows)
        parts[track_id] = {
            "ids": ids,
            "prefix": prefix,
            "suffix": suffix,
            "all": prefix[-1],
        }
    return parts


def contribution(labeler, totals: Counter, part: tuple[Counter, int] | None) -> float:
    if part is None:
        return 0.0
    counts, rows = part
    return labeler.association_contribution(totals, rows, counts)


def empty_part() -> tuple[Counter, int]:
    return Counter(), 0


def merge_parts(
    left: tuple[Counter, int], right: tuple[Counter, int]
) -> tuple[Counter, int]:
    return add_counts(left[0], right[0]), left[1] + right[1]


def label_transactions(seq: str, labeler):
    rows = labeler.read_tracker(PARENT / f"{seq}.txt")
    meta = pd.read_parquet(DATA / seq / "microtracklets.parquet")
    edges = pd.read_parquet(UTILITY / seq / "candidate_edges_utility.parquet")
    row_chunk = labeler.line_chunks(rows, meta)
    row_gt = labeler.matched_gt_per_row(rows, seq, labeler.load_m23())
    totals = labeler.gt_counts(seq)
    chunk_matches = [Counter() for _ in range(len(meta))]
    for chunk_id, gt_id in zip(row_chunk, row_gt):
        if gt_id > 0:
            chunk_matches[int(chunk_id)][int(gt_id)] += 1
    parts = build_track_parts(meta, chunk_matches)
    track_for_chunk = meta.source_track_id.to_numpy(int)
    ordinal_for_chunk = meta.source_ordinal.to_numpy(int)

    cross = edges[edges.same_source.to_numpy(int) == 0].copy()
    deltas = []
    src_tracks = []
    dst_tracks = []
    removed_out = []
    removed_in = []
    for edge in cross.itertuples(index=False):
        src = int(edge.src_chunk)
        dst = int(edge.dst_chunk)
        src_track = int(track_for_chunk[src])
        dst_track = int(track_for_chunk[dst])
        src_ordinal = int(ordinal_for_chunk[src])
        dst_ordinal = int(ordinal_for_chunk[dst])
        src_part = parts[src_track]
        dst_part = parts[dst_track]
        src_prefix = src_part["prefix"][src_ordinal]
        src_suffix = (
            src_part["suffix"][src_ordinal + 1]
            if src_ordinal + 1 < len(src_part["ids"])
            else empty_part()
        )
        dst_prefix = (
            dst_part["prefix"][dst_ordinal - 1]
            if dst_ordinal > 0
            else empty_part()
        )
        dst_suffix = dst_part["suffix"][dst_ordinal]
        before = contribution(labeler, totals, src_part["all"]) + contribution(
            labeler, totals, dst_part["all"]
        )
        after = (
            contribution(labeler, totals, merge_parts(src_prefix, dst_suffix))
            + contribution(labeler, totals, src_suffix)
            + contribution(labeler, totals, dst_prefix)
        )
        deltas.append(after - before)
        src_tracks.append(src_track)
        dst_tracks.append(dst_track)
        removed_out.append(int(src_ordinal + 1 < len(src_part["ids"])))
        removed_in.append(int(dst_ordinal > 0))
    cross["transaction_src_track_id"] = src_tracks
    cross["transaction_dst_track_id"] = dst_tracks
    cross["chain_transaction_delta_proxy"] = deltas
    cross["chain_transaction_positive"] = (
        cross.chain_transaction_delta_proxy.to_numpy(float) > 0
    ).astype("int8")
    cross["transaction_removes_source_out"] = removed_out
    cross["transaction_removes_source_in"] = removed_in
    return meta, edges, cross


def greedy_disjoint(cross: pd.DataFrame) -> pd.DataFrame:
    positive = cross[cross.chain_transaction_delta_proxy > 0].copy()
    positive.sort_values(
        ["chain_transaction_delta_proxy", "src_chunk", "dst_chunk"],
        ascending=[False, True, True],
        inplace=True,
    )
    used_tracks = set()
    selected_indices = []
    for index, edge in positive.iterrows():
        src_track = int(edge.transaction_src_track_id)
        dst_track = int(edge.transaction_dst_track_id)
        if src_track in used_tracks or dst_track in used_tracks:
            continue
        used_tracks.add(src_track)
        used_tracks.add(dst_track)
        selected_indices.append(index)
    return positive.loc[selected_indices].copy()


def apply_transactions(edges: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    source = edges[edges.source_adjacent.to_numpy(int) == 1].copy()
    remove_src = set(selected.src_chunk.astype(int))
    remove_dst = set(selected.dst_chunk.astype(int))
    source = source[
        ~source.src_chunk.astype(int).isin(remove_src)
        & ~source.dst_chunk.astype(int).isin(remove_dst)
    ].copy()
    source["edge_role"] = "source"
    source["utility"] = 1.0
    selected = selected.copy()
    selected["edge_role"] = "cross"
    selected["utility"] = selected.chain_transaction_delta_proxy.to_numpy(float)
    output = pd.concat([source, selected], ignore_index=True, sort=False)
    if output.src_chunk.duplicated().any() or output.dst_chunk.duplicated().any():
        raise RuntimeError("transaction application is not one-to-one")
    return output


def main() -> None:
    labeler = load_module(
        "m23_11_utility_labeler",
        Path("scripts/m23_research/m23_11_add_micrograph_utility.py"),
    )
    evaluator = load_module(
        "m23_11_utility_evaluator",
        Path("scripts/m23_research/m23_11_eval_utility_graph.py"),
    )
    evaluator.NAME = NAME
    root = OUT / NAME
    sequence_reports = []
    for seq in SEQS:
        meta, edges, cross = label_transactions(seq, labeler)
        seq_dir = OUT / "labels" / seq
        seq_dir.mkdir(parents=True, exist_ok=True)
        cross.to_parquet(seq_dir / "cross_chain_transaction_utility.parquet", index=False)
        selected = greedy_disjoint(cross)
        selected_dir = root / "selected_edges"
        selected_dir.mkdir(parents=True, exist_ok=True)
        selected.to_parquet(selected_dir / f"{seq}_transactions.parquet", index=False)
        applied = apply_transactions(edges, selected)
        applied.to_parquet(selected_dir / f"{seq}.parquet", index=False)
        tracker_report = evaluator.write_tracker(
            seq, meta, applied, root / "track_results" / f"{seq}.txt"
        )
        sequence_reports.append(
            {
                "seq": seq,
                "cross_candidates": len(cross),
                "positive_transactions": int(cross.chain_transaction_positive.sum()),
                "selected_disjoint_transactions": len(selected),
                "selected_proxy_delta_sum": float(
                    selected.chain_transaction_delta_proxy.sum()
                ),
                **tracker_report,
            }
        )
        print(json.dumps(sequence_reports[-1]), flush=True)
    report = {
        "name": NAME,
        "status": "GT-oracle action-space lower bound; nondeployable",
        "deployment_allowed": False,
        "gt_use": "chain-transaction utility labels and greedy action selection",
        "action": (
            "split source after src chunk; split destination before dst chunk; "
            "merge source prefix to destination suffix"
        ),
        "constraint": (
            "each original parent track participates in at most one transaction; "
            "disjoint transaction utilities are additive"
        ),
        "optimizer": (
            "deterministic descending-utility greedy matching; this is a lower "
            "bound, not the maximum-weight oracle"
        ),
        "by_seq": sequence_reports,
        "eval": evaluator.evaluate(root),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
