#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.fgas.fgas.data.block_types import collate_conflict_blocks
from projects.fgas.fgas.data.blockbank_io import load_blockbank_jsonl
from projects.fgas.fgas.features.acceptance_features import ACCEPTANCE_FEATURE_NAMES, build_acceptance_feature_vector
from projects.fgas.fgas.runtime.block_primitive_runtime import decode_block_primitive_output
from projects.fgas.fgas.runtime.block_refiner import FGASBlockRefiner, FGASConfig


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
SUMMARY_FIELDS = [
    "blockbank_jsonl",
    "checkpoint",
    "feature_mode",
    "assignment_mode",
    "blend_weight",
    "row_nomatch_weight",
    "soft_mode",
    "blocks_total",
    "takeover_blocks",
    "changed_blocks",
    "rows_total",
    "changed_rows",
    "positive_rows",
    "negative_rows",
    "harmful_rows",
    "neutral_rows",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an FGAS acceptance dataset with local Hungarian labels aligned to the runtime primitive soft-refinement path."
    )
    parser.add_argument("--blockbank-jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fgas-assignment-mode", default="blend", choices=["blend", "replace"])
    parser.add_argument("--fgas-blend-weight", type=float, default=0.5)
    parser.add_argument("--fgas-row-nomatch-weight", type=float, default=0.0)
    parser.add_argument("--fgas-block-primitive-conf-thresh", type=float, default=0.0)
    parser.add_argument("--fgas-soft-row-base-margin-thresh", type=float, default=1.0)
    parser.add_argument("--fgas-soft-changed-row-flip-gap-thresh", type=float, default=0.0)
    parser.add_argument("--fgas-soft-changed-row-refined-margin-thresh", type=float, default=0.0)
    parser.add_argument("--fgas-soft-only-changed-blocks", action="store_true")
    parser.add_argument("--fgas-soft-only-changed-rows", action="store_true")
    parser.add_argument("--fgas-soft-only-changed-frontier", action="store_true")
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
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(args.registry_csv),
        "--kind",
        "analysis",
        "--status",
        status,
        "--script",
        "scripts/build_fgas_acceptance_runtime_dataset.py",
        "--dataset",
        "MOT17",
        "--split",
        "blockbank_acceptance_runtime",
        "--tracker-family",
        "deep_ocsort_fgas",
        "--variant",
        Path(args.out_dir).name,
        "--tag",
        Path(args.out_dir).name,
        "--run-root",
        str(Path(args.out_dir)),
        "--summary-csv",
        str(summary_csv),
        "--checkpoint",
        str(Path(args.checkpoint)),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int)
    try:
        import lap

        _, x, _ = lap.lapjv(cost_matrix, extend_cost=True)
        return np.asarray([[row_idx, col_idx] for row_idx, col_idx in enumerate(x) if col_idx >= 0], dtype=int)
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost_matrix)
        return np.asarray(list(zip(rows, cols)), dtype=int)


def assignment_from_similarity(similarity: np.ndarray, valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    row_count, col_count = similarity.shape
    row_assignment = np.full((row_count,), fill_value=-1, dtype=int)
    col_assignment = np.full((col_count,), fill_value=-1, dtype=int)
    if row_count == 0 or col_count == 0:
        return row_assignment, col_assignment

    large_cost = 1e6
    cost_matrix = np.full((row_count, col_count), fill_value=large_cost, dtype=np.float32)
    cost_matrix[valid_mask] = 1.0 - np.asarray(similarity[valid_mask], dtype=np.float32)
    matched = linear_assignment(cost_matrix)
    for row_idx, col_idx in matched.tolist():
        row_idx = int(row_idx)
        col_idx = int(col_idx)
        if row_idx < 0 or row_idx >= row_count or col_idx < 0 or col_idx >= col_count:
            continue
        if not bool(valid_mask[row_idx, col_idx]):
            continue
        row_assignment[row_idx] = col_idx
        col_assignment[col_idx] = row_idx
    return row_assignment, col_assignment


def resolve_soft_mode(args: argparse.Namespace) -> str:
    if bool(args.fgas_soft_only_changed_frontier):
        return "changed_frontier"
    if bool(args.fgas_soft_only_changed_rows):
        return "changed_rows"
    if bool(args.fgas_soft_only_changed_blocks):
        return "changed_blocks"
    return "all_touched"


def build_refined_similarity(
    *,
    base_similarity: np.ndarray,
    primitive_probs: np.ndarray,
    valid_mask: np.ndarray,
    row_nomatch_probs: np.ndarray,
    row_changed_mask: np.ndarray,
    frontier_col_mask: np.ndarray,
    component_changed: bool,
    args: argparse.Namespace,
) -> np.ndarray:
    refined_similarity = np.asarray(base_similarity, dtype=np.float32).copy()
    apply_soft_on_component = bool(component_changed or not bool(args.fgas_soft_only_changed_blocks))
    for local_r in range(base_similarity.shape[0]):
        for local_c in np.where(valid_mask[local_r])[0].tolist():
            base_score = float(base_similarity[local_r, local_c])
            pred_score = float(primitive_probs[local_r, local_c])
            if str(args.fgas_assignment_mode) == "replace":
                score = pred_score
            else:
                score = (1.0 - float(args.fgas_blend_weight)) * base_score + float(args.fgas_blend_weight) * pred_score
            if float(args.fgas_row_nomatch_weight) > 0.0:
                score *= max(0.0, 1.0 - float(args.fgas_row_nomatch_weight) * float(row_nomatch_probs[local_r]))

            row_soft_allowed = bool(apply_soft_on_component)
            if bool(args.fgas_soft_only_changed_frontier):
                row_soft_allowed = bool(row_changed_mask[local_r] or frontier_col_mask[local_c])
            elif bool(args.fgas_soft_only_changed_rows):
                row_soft_allowed = bool(row_changed_mask[local_r])
            elif bool(args.fgas_soft_only_changed_blocks):
                row_soft_allowed = bool(component_changed)

            if row_soft_allowed:
                refined_similarity[local_r, local_c] = float(np.clip(score, 0.0, 1.0))
    return refined_similarity


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    dataset_jsonl = out_dir / "acceptance_dataset.jsonl"
    summary_row: Dict[str, object] = {
        "blockbank_jsonl": str(Path(args.blockbank_jsonl)),
        "checkpoint": str(Path(args.checkpoint)),
        "feature_mode": "",
        "assignment_mode": str(args.fgas_assignment_mode),
        "blend_weight": float(args.fgas_blend_weight),
        "row_nomatch_weight": float(args.fgas_row_nomatch_weight),
        "soft_mode": resolve_soft_mode(args),
        "blocks_total": 0,
        "takeover_blocks": 0,
        "changed_blocks": 0,
        "rows_total": 0,
        "changed_rows": 0,
        "positive_rows": 0,
        "negative_rows": 0,
        "harmful_rows": 0,
        "neutral_rows": 0,
        "status": "running",
        "error": "",
    }
    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
    append_registry(args, summary_csv, "running", "building FGAS runtime-aligned acceptance dataset")

    try:
        refiner = FGASBlockRefiner(
            FGASConfig(
                enabled=True,
                block_primitive_checkpoint=str(Path(args.checkpoint)),
                block_primitive_conf_thresh=float(args.fgas_block_primitive_conf_thresh),
                soft_apply_only_changed_blocks=bool(args.fgas_soft_only_changed_blocks),
                soft_apply_only_changed_rows=bool(args.fgas_soft_only_changed_rows),
                soft_apply_only_changed_frontier=bool(args.fgas_soft_only_changed_frontier),
                soft_row_base_margin_thresh=float(args.fgas_soft_row_base_margin_thresh),
                soft_changed_row_flip_gap_thresh=float(args.fgas_soft_changed_row_flip_gap_thresh),
                soft_changed_row_refined_margin_thresh=float(args.fgas_soft_changed_row_refined_margin_thresh),
                blend_weight=float(args.fgas_blend_weight),
                assignment_mode=str(args.fgas_assignment_mode),
                row_nomatch_weight=float(args.fgas_row_nomatch_weight),
                device=str(args.device),
            )
        )
        if refiner.block_primitive is None:
            raise ValueError("Expected a block primitive checkpoint.")
        device = refiner.device
        runtime_feature_names = list(refiner.block_primitive_feature_names or refiner.feature_names)
        blocks = load_blockbank_jsonl(Path(args.blockbank_jsonl))
        summary_row["blocks_total"] = int(len(blocks))
        summary_row["feature_mode"] = "full" if any(name in {"s_low", "s_mid", "s_high"} for name in runtime_feature_names) else "nofreq"
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)

        rows_total = 0
        changed_rows = 0
        positive_rows = 0
        negative_rows = 0
        harmful_rows = 0
        neutral_rows = 0
        changed_blocks = 0
        takeover_blocks = 0

        with dataset_jsonl.open("w", encoding="utf-8") as handle:
            for block in blocks:
                if not block.edges:
                    continue
                batch = collate_conflict_blocks([block])
                full_edge_names = list(block.edges[0].feature_names)
                edge_features_full = batch["edge_features"][0].cpu().numpy()
                valid_edge_mask = batch["edge_mask"][0].cpu().numpy().astype(bool)
                row_features_full = batch["row_features"][0].cpu().numpy()
                col_features_full = batch["col_features"][0].cpu().numpy()
                row_targets = batch["row_targets"][0].cpu().numpy()
                rows_total += int(batch["row_mask"][0].sum().item())

                feature_indices = torch.tensor(
                    [full_edge_names.index(name) for name in runtime_feature_names],
                    dtype=torch.long,
                    device=device,
                )
                edge_features = batch["edge_features"].to(device).index_select(dim=-1, index=feature_indices)
                with torch.no_grad():
                    output = refiner.block_primitive(
                        edge_features=edge_features,
                        edge_mask=batch["edge_mask"].to(device),
                        stage_ids=torch.tensor([0], dtype=torch.long, device=device),
                        row_context=batch["row_features"].to(device),
                        col_context=batch["col_features"].to(device),
                    )
                primitive_probs = torch.sigmoid(output.edge_logits)[0].cpu().numpy()
                row_nomatch_probs = torch.sigmoid(output.row_no_match_logits)[0].cpu().numpy()
                primitive_decision = decode_block_primitive_output(
                    output,
                    edge_mask=batch["edge_mask"].to(device),
                    confidence_threshold=float(args.fgas_block_primitive_conf_thresh),
                )
                primitive_takeover = bool(primitive_decision.takeover.cpu().numpy()[0])
                row_no_match = primitive_decision.row_no_match.cpu().numpy()[0]
                col_newborn = primitive_decision.col_newborn.cpu().numpy()[0]

                base_idx = full_edge_names.index("base_similarity")
                base_similarity = edge_features_full[:, :, base_idx]
                base_best, base_row_margin = refiner._row_top1_and_margin(base_similarity, valid_edge_mask)
                refined_best, refined_row_margin, row_changed_mask, frontier_col_mask, component_changed = refiner._compute_changed_row_state(
                    valid_edge_mask=valid_edge_mask,
                    base_best=base_best,
                    base_row_margin=base_row_margin,
                    probs=primitive_probs,
                )
                if np.any(row_no_match):
                    row_changed_mask = row_changed_mask | ((base_best >= 0) & row_no_match)
                    frontier_col_mask = refiner._build_frontier_col_mask(
                        base_best=base_best,
                        refined_best=refined_best,
                        row_changed_mask=row_changed_mask,
                        col_count=int(valid_edge_mask.shape[1]),
                    )
                component_changed = bool(np.any(row_changed_mask) or np.any(col_newborn))
                if component_changed:
                    changed_blocks += 1
                if primitive_takeover:
                    takeover_blocks += 1
                if not primitive_takeover:
                    continue

                refined_similarity = build_refined_similarity(
                    base_similarity=base_similarity,
                    primitive_probs=primitive_probs,
                    valid_mask=valid_edge_mask,
                    row_nomatch_probs=row_nomatch_probs,
                    row_changed_mask=row_changed_mask,
                    frontier_col_mask=frontier_col_mask,
                    component_changed=component_changed,
                    args=args,
                )
                base_assignment, _ = assignment_from_similarity(base_similarity, valid_edge_mask)
                refined_assignment, _ = assignment_from_similarity(refined_similarity, valid_edge_mask)

                for row_idx in np.where(row_changed_mask)[0].tolist():
                    changed_rows += 1
                    target_col = int(row_targets[row_idx])
                    base_col = int(base_assignment[row_idx])
                    refined_col = int(refined_assignment[row_idx])
                    base_correct = int(target_col >= 0 and base_col == target_col)
                    refined_correct = int(target_col >= 0 and refined_col == target_col)

                    label = int(refined_correct == 1 and base_correct == 0)
                    if label == 1:
                        positive_rows += 1
                        flip_type = "beneficial"
                    else:
                        negative_rows += 1
                        if base_correct == 1 and refined_correct == 0:
                            harmful_rows += 1
                            flip_type = "harmful"
                        else:
                            neutral_rows += 1
                            flip_type = "neutral"

                    features = build_acceptance_feature_vector(
                        edge_feature_names=full_edge_names,
                        row_feature_names=block.row_feature_names,
                        col_feature_names=block.col_feature_names,
                        edge_features=edge_features_full,
                        row_features=row_features_full,
                        col_features=col_features_full,
                        valid_mask=valid_edge_mask,
                        probs=primitive_probs,
                        row_nomatch_probs=row_nomatch_probs,
                        base_best=base_best,
                        refined_best=refined_best,
                        base_row_margin=base_row_margin,
                        refined_row_margin=refined_row_margin,
                        row_idx=int(row_idx),
                    )
                    record = {
                        "block_key": str(block.block_key),
                        "seq_name": str(block.metadata.get("seq_name", "")),
                        "frame_id": int(block.metadata.get("frame_id", -1)),
                        "row_index": int(row_idx),
                        "track_gt_id": int(block.row_track_ids[row_idx]),
                        "target_col": int(target_col),
                        "base_best": int(base_best[row_idx]),
                        "refined_best": int(refined_best[row_idx]),
                        "base_assignment": int(base_col),
                        "refined_assignment": int(refined_col),
                        "base_correct": int(base_correct),
                        "refined_correct": int(refined_correct),
                        "row_no_match": int(bool(row_no_match[row_idx])),
                        "primitive_takeover": int(primitive_takeover),
                        "flip_type": flip_type,
                        "label": int(label),
                        "feature_names": list(ACCEPTANCE_FEATURE_NAMES),
                        "features": [float(v) for v in features],
                    }
                    handle.write(json.dumps(record))
                    handle.write("\n")

        summary_row.update(
            {
                "takeover_blocks": int(takeover_blocks),
                "changed_blocks": int(changed_blocks),
                "rows_total": int(rows_total),
                "changed_rows": int(changed_rows),
                "positive_rows": int(positive_rows),
                "negative_rows": int(negative_rows),
                "harmful_rows": int(harmful_rows),
                "neutral_rows": int(neutral_rows),
                "status": "success",
            }
        )
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, "success", "built FGAS runtime-aligned acceptance dataset")
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = repr(exc)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, "failed", "building FGAS runtime-aligned acceptance dataset")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
