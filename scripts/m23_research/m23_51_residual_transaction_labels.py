#!/usr/bin/env python3
from __future__ import annotations

"""M23-51 exact residual transaction labels around frozen M23-39 trackers.

This is a thin protocol wrapper over the already validated M23-38 exact-HOTA
label engine.  It substitutes the byte-verified, deployable M23-39 tracker,
selected transactions and applied graph as the frozen baseline, while retaining
the original GT-free transaction prediction bank for residual candidate
construction.  GT is used only after the residual shortlist is frozen.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
DEFAULT_CACHE = Path(
    "outputs/mot20_m23_20260718/m23_45_m23_39_deployable_baseline_cache_v1"
)
FORBIDDEN = (
    "same_gt", "modal_gt", "purity", "label_confidence", "actual_assa",
)


def load_module(name: str, relative: str):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def symlink_exact(source: Path, target: Path) -> None:
    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.is_symlink() and target.resolve() == source:
            return
        raise RuntimeError(f"refusing to overwrite prepared baseline path: {target}")
    os.symlink(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True, choices=SEQUENCES)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--graph-root", default=None)
    parser.add_argument("--source-parent", default=None)
    parser.add_argument("--max-replacements", type=int, default=256)
    parser.add_argument(
        "--replacement-selection", choices=("policy", "diverse"), default="diverse"
    )
    parser.add_argument("--skip-drops", action="store_true")
    args = parser.parse_args()

    m39 = load_module(
        "m23_51_m39", "scripts/m23_research/m23_39_exact_advantage_student_nested_loso.py"
    )
    m38 = load_module(
        "m23_51_m38", "scripts/m23_research/m23_38_exact_hota_advantage_labels.py"
    )

    seq = args.seq
    output_root = Path(args.output_root)
    cache = Path(args.baseline_cache)
    seq_cache = cache / seq
    output_root.mkdir(parents=True, exist_ok=True)
    prepared = output_root / "prepared_m23_39_baseline"

    tracker = seq_cache / "track_results" / f"{seq}.txt"
    selected = seq_cache / "frozen_selected_transactions.parquet"
    applied = seq_cache / "frozen_applied_edges.parquet"
    prediction = (
        m39.BASELINE_ROOTS[seq] / "predictions" / f"{seq}_predictions.parquet"
    )
    for path in (tracker, selected, applied, prediction):
        if not path.exists():
            raise FileNotFoundError(path)

    symlink_exact(tracker, prepared / "track_results" / f"{seq}.txt")
    symlink_exact(prediction, prepared / "predictions" / f"{seq}_predictions.parquet")
    symlink_exact(selected, prepared / f"{seq}_selected_transactions.parquet")
    symlink_exact(applied, prepared / f"{seq}_applied_edges.parquet")

    original_argv = sys.argv[:]
    delegated = [
        str(REPO / "scripts/m23_research/m23_38_exact_hota_advantage_labels.py"),
        "--seq", seq,
        "--baseline-root", str(prepared),
        "--output-root", str(output_root),
        "--max-replacements", str(args.max_replacements),
        "--replacement-selection", args.replacement_selection,
    ]
    if args.graph_root:
        delegated.extend(["--graph-root", args.graph_root])
    if args.source_parent:
        delegated.extend(["--source-parent", args.source_parent])
    if args.skip_drops:
        delegated.append("--skip-drops")
    sys.argv = delegated
    try:
        m38.main()
    finally:
        sys.argv = original_argv

    label_path = output_root / "exact_action_labels.parquet"
    labels = pd.read_parquet(label_path)
    forbidden_columns = [
        column for column in labels.columns
        if any(token in column.lower() for token in FORBIDDEN)
    ]
    if forbidden_columns:
        raise RuntimeError(f"GT-derived residual label feature columns: {forbidden_columns}")

    report_path = output_root / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update({
        "experiment": "M23-51 exact residual transaction labels around frozen M23-39",
        "teacher_only": True,
        "deployable": False,
        "baseline_family": "strict deployable M23-39 transaction tracker",
        "baseline_cache": str(cache),
        "candidate_construction": "GT-free residual transaction candidates from frozen M23-39 selected set",
        "held_protocol": "labels may be used only when the sequence belongs to outer-training or an already exposed diagnostic fold",
        "label_regeneration_of_m23_41": False,
        "forbidden_candidate_columns": forbidden_columns,
    })
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    protocol_path = output_root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol.update({
        "experiment": "M23-51 exact residual transaction labels around frozen M23-39",
        "teacher_only": True,
        "deployable": False,
        "baseline_family": "strict deployable M23-39 transaction tracker",
        "candidate_construction": "GT-free residual transaction candidates from frozen M23-39 selected set",
    })
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "stage": "m23_51_completed",
        "seq": seq,
        "rows": int(len(labels)),
        "positive_actions": int((labels.delta_HOTA > 0.0).sum()),
        "best_delta_HOTA": float(labels.delta_HOTA.max()),
        "forbidden_candidate_columns": forbidden_columns,
    }), flush=True)


if __name__ == "__main__":
    main()
