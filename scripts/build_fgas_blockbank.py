#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.fgas.fgas.data.block_types import ConflictBlock, EdgeSample
from projects.fgas.fgas.data.blockbank_io import write_blockbank_jsonl
from projects.fgas.fgas.features.edge_features import (
    COL_CONTEXT_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
    ROW_CONTEXT_FEATURE_NAMES,
    build_col_context_from_rows,
    build_row_context_from_rows,
    feature_vector_from_row,
)


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
SUMMARY_FIELDS = [
    "source_pairbank",
    "stage_name",
    "rows",
    "blocks",
    "ambiguous_blocks",
    "multi_row_blocks",
    "multi_col_blocks",
    "avg_rows_per_block",
    "avg_cols_per_block",
    "max_rows_seen",
    "max_cols_seen",
    "max_rows_kept",
    "max_cols_kept",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an FGAS block-bank from an existing pair-bank.")
    parser.add_argument("--pairbank-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stage-name", default="primary")
    parser.add_argument("--max-rows", type=int, default=3)
    parser.add_argument("--max-cols", type=int, default=3)
    parser.add_argument("--min-edges", type=int, default=2)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_single_row_csv(path: Path, fieldnames: Sequence[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_registry(args: argparse.Namespace, summary_csv: Path, status: str, notes: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/append_experiment_record.py"),
        "--csv",
        str(args.registry_csv),
        "--kind",
        "analysis",
        "--status",
        status,
        "--script",
        "scripts/build_fgas_blockbank.py",
        "--dataset",
        "MOT17",
        "--split",
        "pairbank_to_blockbank",
        "--tracker-family",
        "botsort_fgas",
        "--variant",
        Path(args.out_dir).name,
        "--tag",
        Path(args.out_dir).name,
        "--run-root",
        str(Path(args.out_dir)),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, check=False)


def load_pairbank_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def det_key_from_row(row: Dict[str, object]) -> Tuple[object, ...]:
    det_box = [round(float(v), 2) for v in row.get("det_box", [])]
    return (
        row.get("seq_name", ""),
        int(row.get("frame_id", -1)),
        tuple(det_box),
        round(float(row.get("det_score", 0.0)), 4),
        int(row.get("det_gt_id", -1)),
    )


def build_connected_components(frame_rows: Sequence[Dict[str, object]]) -> List[Tuple[Set[int], Set[Tuple[object, ...]]]]:
    row_to_dets: Dict[int, Set[Tuple[object, ...]]] = defaultdict(set)
    det_to_rows: Dict[Tuple[object, ...], Set[int]] = defaultdict(set)
    all_rows: Set[int] = set()
    all_dets: Set[Tuple[object, ...]] = set()
    for row in frame_rows:
        track_id = int(row.get("track_gt_id", -1))
        det_key = det_key_from_row(row)
        row_to_dets[track_id].add(det_key)
        det_to_rows[det_key].add(track_id)
        all_rows.add(track_id)
        all_dets.add(det_key)

    visited_rows: Set[int] = set()
    visited_dets: Set[Tuple[object, ...]] = set()
    components: List[Tuple[Set[int], Set[Tuple[object, ...]]]] = []

    for start_row in sorted(all_rows):
        if start_row in visited_rows:
            continue
        comp_rows: Set[int] = set()
        comp_dets: Set[Tuple[object, ...]] = set()
        queue: deque[Tuple[str, object]] = deque([("row", start_row)])
        while queue:
            kind, node = queue.popleft()
            if kind == "row":
                row_id = int(node)
                if row_id in visited_rows:
                    continue
                visited_rows.add(row_id)
                comp_rows.add(row_id)
                for det_key in row_to_dets.get(row_id, set()):
                    if det_key not in visited_dets:
                        queue.append(("det", det_key))
            else:
                det_key = node
                if det_key in visited_dets:
                    continue
                visited_dets.add(det_key)
                comp_dets.add(det_key)
                for row_id in det_to_rows.get(det_key, set()):
                    if row_id not in visited_rows:
                        queue.append(("row", row_id))
        if comp_rows and comp_dets:
            components.append((comp_rows, comp_dets))
    return components


def truncate_nodes(nodes: Sequence[object], scored_rows: Sequence[Tuple[object, float]], limit: int) -> List[object]:
    ordered = [node for node, _ in sorted(scored_rows, key=lambda item: item[1], reverse=True)]
    return list(ordered[: max(1, int(limit))])


def build_blocks(rows: Sequence[Dict[str, object]], args: argparse.Namespace) -> Tuple[List[ConflictBlock], Dict[str, object]]:
    per_frame: Dict[Tuple[str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        per_frame[(str(row.get("seq_name", "")), int(row.get("frame_id", -1)))].append(row)

    blocks: List[ConflictBlock] = []
    ambiguous_blocks = 0
    multi_row_blocks = 0
    multi_col_blocks = 0
    total_rows = 0
    total_cols = 0
    max_rows_seen = 0
    max_cols_seen = 0

    for (seq_name, frame_id), frame_rows in sorted(per_frame.items()):
        components = build_connected_components(frame_rows)
        row_lookup: Dict[int, List[Dict[str, object]]] = defaultdict(list)
        det_lookup: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
        for row in frame_rows:
            row_lookup[int(row.get("track_gt_id", -1))].append(row)
            det_lookup[det_key_from_row(row)].append(row)

        for comp_idx, (row_nodes, det_nodes) in enumerate(components):
            max_rows_seen = max(max_rows_seen, len(row_nodes))
            max_cols_seen = max(max_cols_seen, len(det_nodes))
            if len(row_nodes) == 0 or len(det_nodes) == 0:
                continue
            row_scores = []
            for row_id in row_nodes:
                score = max(float(r.get("base_similarity", 0.0)) for r in row_lookup[row_id])
                row_scores.append((row_id, score))
            det_scores = []
            for det_key in det_nodes:
                score = max(float(r.get("det_score", 0.0)) for r in det_lookup[det_key])
                det_scores.append((det_key, score))
            kept_rows = truncate_nodes(sorted(row_nodes), row_scores, int(args.max_rows))
            kept_dets = truncate_nodes(list(det_nodes), det_scores, int(args.max_cols))
            row_index = {int(row_id): idx for idx, row_id in enumerate(kept_rows)}
            det_index = {det_key: idx for idx, det_key in enumerate(kept_dets)}

            edge_rows: List[EdgeSample] = []
            row_targets = [-1 for _ in kept_rows]
            ambiguous_flag = False
            kept_row_candidates: Dict[int, List[Dict[str, object]]] = {}
            for row_id in kept_rows:
                candidates = [r for r in row_lookup[int(row_id)] if det_key_from_row(r) in det_index]
                kept_row_candidates[int(row_id)] = list(
                    sorted(candidates, key=lambda item: float(item.get("base_similarity", 0.0)), reverse=True)
                )
                for candidate in sorted(candidates, key=lambda item: float(item.get("base_similarity", 0.0)), reverse=True):
                    det_key = det_key_from_row(candidate)
                    sample = EdgeSample(
                        row_index=row_index[int(row_id)],
                        col_index=det_index[det_key],
                        feature_names=list(EDGE_FEATURE_NAMES),
                        features=feature_vector_from_row(candidate),
                        label=int(candidate.get("label", 0)),
                        valid=True,
                    )
                    edge_rows.append(sample)
                    if int(candidate.get("label", 0)) == 1:
                        row_targets[row_index[int(row_id)]] = det_index[det_key]
                    ambiguous_flag = ambiguous_flag or bool(int(candidate.get("ambiguous_flag", 0)))
            if len(edge_rows) < int(args.min_edges):
                continue

            row_features = [
                build_row_context_from_rows(
                    kept_row_candidates.get(int(row_id), []),
                    candidate_limit=float(max(len(kept_dets), 1)),
                )
                for row_id in kept_rows
            ]
            col_features = []
            for det_key in kept_dets:
                col_rows = [row for row_id in kept_rows for row in kept_row_candidates.get(int(row_id), []) if det_key_from_row(row) == det_key]
                col_features.append(
                    build_col_context_from_rows(
                        col_rows,
                        candidate_limit=float(max(len(kept_rows), 1)),
                    )
                )

            block = ConflictBlock(
                block_key=f"{seq_name}:{frame_id}:{args.stage_name}:component:{comp_idx}",
                stage_name=str(args.stage_name),
                row_track_ids=[int(v) for v in kept_rows],
                col_det_ids=list(range(len(kept_dets))),
                edges=edge_rows,
                row_feature_names=list(ROW_CONTEXT_FEATURE_NAMES),
                row_features=row_features,
                col_feature_names=list(COL_CONTEXT_FEATURE_NAMES),
                col_features=col_features,
                row_match_targets=row_targets,
                ambiguous=bool(ambiguous_flag),
                metadata={
                    "seq_name": seq_name,
                    "frame_id": int(frame_id),
                    "original_row_count": int(len(row_nodes)),
                    "original_col_count": int(len(det_nodes)),
                },
            )
            blocks.append(block)
            total_rows += len(kept_rows)
            total_cols += len(kept_dets)
            if block.ambiguous:
                ambiguous_blocks += 1
            if len(kept_rows) > 1:
                multi_row_blocks += 1
            if len(kept_dets) > 1:
                multi_col_blocks += 1

    summary = {
        "rows": int(len(rows)),
        "blocks": int(len(blocks)),
        "ambiguous_blocks": int(ambiguous_blocks),
        "multi_row_blocks": int(multi_row_blocks),
        "multi_col_blocks": int(multi_col_blocks),
        "avg_rows_per_block": float(total_rows / len(blocks)) if blocks else 0.0,
        "avg_cols_per_block": float(total_cols / len(blocks)) if blocks else 0.0,
        "max_rows_seen": int(max_rows_seen),
        "max_cols_seen": int(max_cols_seen),
        "max_rows_kept": int(args.max_rows),
        "max_cols_kept": int(args.max_cols),
    }
    return blocks, summary


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    blockbank_jsonl = out_dir / "blockbank.jsonl"
    summary_row: Dict[str, object] = {
        "source_pairbank": str(Path(args.pairbank_jsonl)),
        "stage_name": str(args.stage_name),
        "rows": 0,
        "blocks": 0,
        "ambiguous_blocks": 0,
        "multi_row_blocks": 0,
        "multi_col_blocks": 0,
        "avg_rows_per_block": 0.0,
        "avg_cols_per_block": 0.0,
        "max_rows_seen": 0,
        "max_cols_seen": 0,
        "max_rows_kept": int(args.max_rows),
        "max_cols_kept": int(args.max_cols),
        "status": "running",
        "error": "",
    }
    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
    append_registry(args, summary_csv, "running", f"stage={args.stage_name} source={args.pairbank_jsonl}")
    try:
        rows = load_pairbank_rows(Path(args.pairbank_jsonl))
        blocks, stats = build_blocks(rows, args)
        write_blockbank_jsonl(blockbank_jsonl, blocks)
        summary_row.update(stats)
        summary_row["status"] = "success"
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, "success", f"stage={args.stage_name} source={args.pairbank_jsonl}")
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = repr(exc)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, "failed", f"stage={args.stage_name} source={args.pairbank_jsonl}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
