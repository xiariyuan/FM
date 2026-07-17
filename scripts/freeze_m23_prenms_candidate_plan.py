"""Freeze the M23-1 pre-NMS candidate plan without reading ground truth.

This stage is intentionally separated from oracle evaluation. It reads only:
- frozen MOT20 baseline tracker outputs,
- frozen post-NMS Phase-0 detector geometry,
- verified pre-NMS suppression dumps.

It writes immutable candidate index arrays and manifests. There is no GT path
argument and no TrackEval call. The downstream oracle audit must consume these
files verbatim rather than recomputing or changing the candidate policy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_dump(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_under(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def save_array(path: Path, array: np.ndarray) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(array), allow_pickle=False)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "dtype": str(np.asarray(array).dtype),
        "shape": [int(value) for value in np.asarray(array).shape],
        "size_bytes": path.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument(
        "--baseline-dir",
        default="outputs/trusttrack_review_20260710/all4_baseline_oracle/baseline/eval_work/trackers/all4_baseline/data",
    )
    parser.add_argument("--phase0-root", default="outputs/alink_train_inputs/phase0_root")
    parser.add_argument("--prenms-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    baseline_dir = resolve_under(repo, args.baseline_dir)
    phase0_root = resolve_under(repo, args.phase0_root)
    prenms_root = resolve_under(repo, args.prenms_root)
    output_dir = resolve_under(repo, args.out_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    m23 = load_module(
        repo / "scripts" / "audit_m23_mot20_expanded_evidence_oracle.py",
        "fmtrack_m23_0_for_freeze",
    )
    audit = load_module(
        repo / "scripts" / "audit_m23_prenms_suppressed_evidence_oracle.py",
        "fmtrack_m23_1_for_freeze",
    )

    combined_dump_report_path = prenms_root / "report.json"
    combined_dump_manifest_path = prenms_root / "manifest.json"
    combined_dump_report = json.loads(combined_dump_report_path.read_text(encoding="utf-8"))
    if not bool(combined_dump_report.get("all_reference_equivalence_passed")):
        raise RuntimeError("pre-NMS suppression dump has not passed frozen Phase-0 equivalence")
    if tuple(combined_dump_report.get("sequences", [])) != SEQUENCES:
        raise RuntimeError(
            f"unexpected pre-NMS sequence order: {combined_dump_report.get('sequences')}"
        )

    inventory_rows = []
    sequence_manifests: Dict[str, dict] = {}
    for sequence in SEQUENCES:
        print(f"[M23-1 freeze] {sequence}", flush=True)
        baseline_path = baseline_dir / f"{sequence}.txt"
        phase0_path = phase0_root / sequence / "dump_yolox_reid.npz"
        baseline = m23.load_baseline(baseline_path)
        phase0, phase0_metadata = m23.load_phase0(phase0_path)
        baseline_rows = sum(len(rows) for rows in baseline.values())
        postnms_novel_rows = 0
        for frame in sorted(set(baseline) | set(phase0)):
            novel, _ = m23.novel_phase0(phase0.get(frame, []), baseline.get(frame, []))
            postnms_novel_rows += len(novel)

        store = audit.load_prenms_store(prenms_root, sequence)
        plan = audit.build_budget_plan(
            store,
            baseline_rows=baseline_rows,
            postnms_novel_rows=postnms_novel_rows,
        )
        full_indices = np.flatnonzero(plan.full_mask).astype("<i8")
        budget_indices = np.flatnonzero(plan.budget_mask).astype("<i8")
        sequence_dir = output_dir / sequence
        files = {
            "full_indices": save_array(sequence_dir / "full_indices.npy", full_indices),
            "full_priority": save_array(
                sequence_dir / "full_priority.npy",
                plan.priority[full_indices].astype("<f4", copy=False),
            ),
            "full_family_rank": save_array(
                sequence_dir / "full_family_rank.npy",
                plan.family_rank[full_indices].astype("<i2", copy=False),
            ),
            "full_local_rank": save_array(
                sequence_dir / "full_local_rank.npy",
                plan.local_rank[full_indices].astype("<i2", copy=False),
            ),
            "budget_indices": save_array(sequence_dir / "budget_indices.npy", budget_indices),
            "budget_priority": save_array(
                sequence_dir / "budget_priority.npy",
                plan.priority[budget_indices].astype("<f4", copy=False),
            ),
            "budget_family_rank": save_array(
                sequence_dir / "budget_family_rank.npy",
                plan.family_rank[budget_indices].astype("<i2", copy=False),
            ),
            "budget_local_rank": save_array(
                sequence_dir / "budget_local_rank.npy",
                plan.local_rank[budget_indices].astype("<i2", copy=False),
            ),
        }
        inventory = {"sequence": sequence, **plan.stats}
        inventory_rows.append(inventory)
        sequence_report = {
            "schema": "fmtrack.m23.prenms_candidate_plan.v1",
            "sequence": sequence,
            "protocol": {
                "ground_truth_read": False,
                "trackeval_calls": 0,
                "selector_trained": False,
                "threshold_sweep": False,
                "pool_ratio_limit": float(audit.POOL_RATIO_LIMIT),
                "exact_duplicate_iou": float(audit.EXACT_DUPLICATE_IOU),
                "family_dedupe_iou": float(audit.FAMILY_DEDUPE_IOU),
                "priority": "score * clip((0.99 - suppressor_iou) / (0.99 - nms_threshold), 0, 1)",
                "selection_order": "frame-local rank, descending priority, frame, stable global index",
            },
            "inventory": inventory,
            "sources": {
                "baseline": {
                    "path": str(baseline_path),
                    "sha256": sha256_file(baseline_path),
                    "rows": int(baseline_rows),
                },
                "phase0": phase0_metadata,
                "prenms_manifest": {
                    "path": str(prenms_root / sequence / "manifest.json"),
                    "sha256": sha256_file(prenms_root / sequence / "manifest.json"),
                },
                "prenms_data": {
                    "path": str(prenms_root / sequence / "suppressed_candidates.f32"),
                    "sha256": store.manifest["file_hashes"]["suppressed_candidates.f32"],
                },
            },
            "files": files,
            "locked_state": {
                "locked_label_reads": 0,
                "locked_trackeval_calls": 0,
                "remaining_locked_rows_untouched": 156,
                "p15_policy": "no_op",
            },
        }
        canonical_json_dump(sequence_report, sequence_dir / "report.json")
        sequence_manifest = {
            "schema": "fmtrack.m23.prenms_candidate_plan.manifest.v1",
            "sequence": sequence,
            "report_sha256": sha256_file(sequence_dir / "report.json"),
            "file_hashes": {
                path.name: sha256_file(path)
                for path in sorted(sequence_dir.iterdir())
                if path.is_file()
            },
        }
        canonical_json_dump(sequence_manifest, sequence_dir / "manifest.json")
        sequence_manifests[sequence] = {
            "path": f"{sequence}/manifest.json",
            "sha256": sha256_file(sequence_dir / "manifest.json"),
        }

    combined = audit.aggregate_inventory(inventory_rows)
    combined["sequence"] = "COMBINED"
    inventory_all = inventory_rows + [combined]
    inventory_fields = ["sequence"] + sorted(
        {key for row in inventory_all for key in row if key != "sequence"}
    )
    write_csv(output_dir / "candidate_inventory.csv", inventory_all, inventory_fields)

    report = {
        "schema": "fmtrack.m23.prenms_candidate_plan.combined.v1",
        "protocol": {
            "ground_truth_read": False,
            "trackeval_calls": 0,
            "selector_trained": False,
            "threshold_sweep": False,
            "candidate_plan_frozen_before_oracle": True,
        },
        "inventory": inventory_all,
        "decision": {
            "candidate_budget_passed": bool(combined["budget_passed"]),
            "candidate_pool_ratio": float(combined["budget_pool_ratio"]),
            "candidate_plan_frozen": True,
            "deployment_allowed": False,
            "locked_manifest_created": False,
        },
        "sources": {
            "prenms_combined_report": {
                "path": str(combined_dump_report_path),
                "sha256": sha256_file(combined_dump_report_path),
            },
            "prenms_combined_manifest": {
                "path": str(combined_dump_manifest_path),
                "sha256": sha256_file(combined_dump_manifest_path),
            },
            "candidate_plan_script": {
                "path": "scripts/freeze_m23_prenms_candidate_plan.py",
                "sha256": sha256_file(repo / "scripts" / "freeze_m23_prenms_candidate_plan.py"),
            },
            "policy_script": {
                "path": "scripts/audit_m23_prenms_suppressed_evidence_oracle.py",
                "sha256": sha256_file(repo / "scripts" / "audit_m23_prenms_suppressed_evidence_oracle.py"),
            },
        },
        "sequence_manifests": sequence_manifests,
        "locked_state": {
            "locked_label_reads": 0,
            "locked_trackeval_calls": 0,
            "remaining_locked_rows_untouched": 156,
            "p15_policy": "no_op",
        },
    }
    canonical_json_dump(report, output_dir / "report.json")
    manifest = {
        "schema": "fmtrack.m23.prenms_candidate_plan.combined_manifest.v1",
        "report_sha256": sha256_file(output_dir / "report.json"),
        "candidate_inventory_sha256": sha256_file(output_dir / "candidate_inventory.csv"),
        "sequence_manifests": sequence_manifests,
    }
    canonical_json_dump(manifest, output_dir / "manifest.json")
    print(json.dumps(report["decision"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
