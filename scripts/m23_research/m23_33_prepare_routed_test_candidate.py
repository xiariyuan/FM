#!/usr/bin/env python3
from __future__ import annotations

"""Prepare a GT-free MOT20 test candidate from the M23-32 all-train model.

The absolute train probability threshold is intentionally not reused because a
GT-free audit found a train/test calibration shift. Instead, each test sequence
is routed to the nearest train sequence using graph and parent-tracker summary
statistics only. The selected-action fraction of that train sequence defines a
sequence-relative probability quantile. No MOT20 test GT path is opened.
"""

import argparse
import hashlib
import importlib.util
import json
import pickle
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
TRAIN_SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
TEST_SEQUENCES = ("MOT20-04", "MOT20-06", "MOT20-07", "MOT20-08")
DEFAULT_MODEL_ROOT = (
    REPO
    / "outputs"
    / "mot20_m23_20260718"
    / "m23_32_alltrain_structured_fit_v1"
)
DEFAULT_GRAPH_ROOT = (
    REPO
    / "outputs"
    / "mot20_m23_20260718"
    / "m23_26_test_deploy_oof_ensemble_v1"
)
DEFAULT_TRAIN_PARENT = (
    REPO
    / "outputs"
    / "assocriskbench_p15_20260714"
    / "fused_a42_adaptive_risk_eval"
    / "l1_r2_q75_adaptive_priority"
    / "track_results"
)
DEFAULT_TEST_PARENT = (
    REPO
    / "outputs"
    / "assocriskbench_p15_20260713"
    / "identity_debt_a43_test_submission"
    / "track_results"
)
DEFAULT_OUTPUT = (
    REPO
    / "outputs"
    / "mot20_m23_20260718"
    / "m23_33_routed_alltrain_test_candidate_v1"
)
TARGET = "chain_transaction_delta_proxy"
ROUTING_FEATURES = (
    "rows",
    "frames",
    "tracks",
    "chunks",
    "cross_edges",
    "rows_per_frame",
    "chunks_per_track",
    "cross_per_chunk",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_module(name: str, relative_path: str):
    path = REPO / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parent_stats(path: Path) -> Tuple[int, int, int]:
    rows = 0
    max_frame = 0
    track_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.split(",")
            rows += 1
            max_frame = max(max_frame, int(float(fields[0])))
            track_ids.add(int(float(fields[1])))
    return rows, max_frame, len(track_ids)


def sequence_descriptor(
    seq: str, graph_root: Path, parent_root: Path
) -> Dict[str, float]:
    meta = pd.read_parquet(graph_root / seq / "microtracklets.parquet")
    edges = pd.read_parquet(graph_root / seq / "candidate_edges.parquet")
    rows, frames, tracks = parent_stats(parent_root / f"{seq}.txt")
    cross_edges = int((edges.same_source.to_numpy(int) == 0).sum())
    return {
        "sequence": seq,
        "rows": float(rows),
        "frames": float(frames),
        "tracks": float(tracks),
        "chunks": float(len(meta)),
        "cross_edges": float(cross_edges),
        "rows_per_frame": float(rows) / max(frames, 1),
        "chunks_per_track": float(len(meta)) / max(tracks, 1),
        "cross_per_chunk": float(cross_edges) / max(len(meta), 1),
    }


def route_sequences(
    train_descriptors: Dict[str, Dict[str, float]],
    test_descriptors: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, str], Dict[str, Dict[str, float]]]:
    train_matrix = np.asarray(
        [
            [np.log1p(train_descriptors[seq][feature]) for feature in ROUTING_FEATURES]
            for seq in TRAIN_SEQUENCES
        ],
        dtype=float,
    )
    mean = train_matrix.mean(axis=0)
    std = train_matrix.std(axis=0)
    std[std == 0.0] = 1.0
    normalized_train = (train_matrix - mean) / std
    routes = {}
    distances = {}
    for test_seq in TEST_SEQUENCES:
        vector = np.asarray(
            [
                np.log1p(test_descriptors[test_seq][feature])
                for feature in ROUTING_FEATURES
            ],
            dtype=float,
        )
        normalized = (vector - mean) / std
        current = {
            train_seq: float(np.linalg.norm(normalized - normalized_train[index]))
            for index, train_seq in enumerate(TRAIN_SEQUENCES)
        }
        routes[test_seq] = min(current, key=current.get)
        distances[test_seq] = current
    return routes, distances


def validate_submission(package_root: Path, zip_path: Path, output_root: Path) -> None:
    commands = [
        (
            [
                sys.executable,
                str(REPO / "scripts" / "check_mot20_submission.py"),
                "--results-dir",
                str(package_root),
                "--profile",
                "mot20_test_4",
            ],
            output_root / "precheck_results_dir.log",
        ),
        (
            [
                sys.executable,
                str(REPO / "scripts" / "check_mot20_submission.py"),
                "--zip-path",
                str(zip_path),
                "--profile",
                "mot20_test_4",
            ],
            output_root / "precheck_zip.log",
        ),
    ]
    for command, log_path in commands:
        completed = subprocess.run(
            command,
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(completed.stdout[-5000:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--graph-root", default=str(DEFAULT_GRAPH_ROOT))
    parser.add_argument("--train-parent", default=str(DEFAULT_TRAIN_PARENT))
    parser.add_argument("--test-parent", default=str(DEFAULT_TEST_PARENT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    model_root = Path(args.model_root).resolve()
    graph_root = Path(args.graph_root).resolve()
    train_parent = Path(args.train_parent).resolve()
    test_parent = Path(args.test_parent).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_graph = graph_root / "train_oof_micrograph"
    test_graph = graph_root / "test_micrograph"
    with (model_root / "structured_model.pkl").open("rb") as handle:
        artifact = pickle.load(handle)
    model = artifact["model"]
    features: List[str] = list(artifact["features"])

    m26 = load_module("m23_33_m26", "scripts/m23_research/m23_26_prepare_test_submission.py")
    m28 = load_module("m23_33_m28", "scripts/m23_research/m23_28_structured_oracle_imitation_loso.py")
    base = load_module("m23_33_base", "scripts/m23_research/m23_12_train_chain_transaction_loso.py")
    chain = load_module("m23_33_chain", "scripts/m23_research/m23_12_chain_transaction_oracle.py")
    evaluator = load_module("m23_33_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py")

    train_descriptors = {
        seq: sequence_descriptor(seq, train_graph, train_parent)
        for seq in TRAIN_SEQUENCES
    }
    test_descriptors = {
        seq: sequence_descriptor(seq, test_graph, test_parent)
        for seq in TEST_SEQUENCES
    }
    routes, route_distances = route_sequences(train_descriptors, test_descriptors)

    train_fractions = {}
    for seq in TRAIN_SEQUENCES:
        predictions = pd.read_parquet(model_root / "predictions" / f"{seq}.parquet")
        selected = pd.read_parquet(
            model_root / "selected_transactions" / f"{seq}.parquet"
        )
        train_fractions[seq] = float(len(selected)) / max(len(predictions), 1)

    prediction_root = output_root / "predictions"
    selected_root = output_root / "selected_transactions"
    applied_root = output_root / "applied_edges"
    track_root = output_root / "track_results"
    package_root = output_root / "package_root"
    for path in (
        prediction_root,
        selected_root,
        applied_root,
        track_root,
        package_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    base.META = test_graph
    evaluator.DATA = test_graph
    evaluator.PARENT = test_parent
    evaluator.SEQS = list(TEST_SEQUENCES)
    sequence_reports = []
    for seq in TEST_SEQUENCES:
        meta = pd.read_parquet(test_graph / seq / "microtracklets.parquet")
        edges = pd.read_parquet(test_graph / seq / "candidate_edges.parquet")
        structural = m26.structural_transactions(meta, edges)
        frame = m26.add_conflict_graph_features(base.add_chain_features(seq, structural))
        frame["oracle_selection_probability"] = model.predict_proba(
            frame[features]
        )[:, 1]
        frame.to_parquet(prediction_root / f"{seq}.parquet", index=False)

        routed_train = routes[seq]
        quantile = 1.0 - train_fractions[routed_train]
        selected = m28.maximum_weight_matching(
            frame, "oracle_selection_probability", quantile
        )
        selected.to_parquet(selected_root / f"{seq}.parquet", index=False)
        applied = chain.apply_transactions(
            edges,
            selected.assign(**{TARGET: selected.policy_score.to_numpy(float)}),
        )
        for column, default in (
            ("assa_edge_delta_proxy", 0.0),
            ("assa_edge_positive", 0),
            ("assa_edge_negative", 0),
        ):
            if column not in applied:
                applied[column] = default
        applied.to_parquet(applied_root / f"{seq}.parquet", index=False)
        output_txt = track_root / f"{seq}.txt"
        tracker_report = evaluator.write_tracker(seq, meta, applied, output_txt)
        shutil.copy2(output_txt, package_root / f"{seq}.txt")
        report = {
            "sequence": seq,
            "routed_train_sequence": routed_train,
            "routing_distances": route_distances[seq],
            "train_selected_fraction": train_fractions[routed_train],
            "score_quantile": quantile,
            "candidates": int(len(frame)),
            "selected_actions": int(len(selected)),
            "selected_score_min": (
                float(selected.oracle_selection_probability.min())
                if len(selected)
                else None
            ),
            "selected_score_median": (
                float(selected.oracle_selection_probability.median())
                if len(selected)
                else None
            ),
            "selected_score_max": (
                float(selected.oracle_selection_probability.max())
                if len(selected)
                else None
            ),
            "parent_sha256": sha256(test_parent / f"{seq}.txt"),
            "output_sha256": sha256(output_txt),
            "output_bytes": output_txt.stat().st_size,
            **tracker_report,
        }
        sequence_reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)

    zip_path = output_root / "MOT20_M23_33_routed_alltrain_candidate.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for seq in TEST_SEQUENCES:
            archive.write(package_root / f"{seq}.txt", arcname=f"{seq}.txt")
    validate_submission(package_root, zip_path, output_root)

    training_report = json.loads((model_root / "report.json").read_text(encoding="utf-8"))
    manifest = {
        "status": "completed",
        "created_at": now_iso(),
        "experiment": "M23-33 routed M23-32 all-train test candidate",
        "test_gt_read": False,
        "test_inference_gt_free": True,
        "training_fit_HOTA": training_report["metrics"]["COMBINED"]["HOTA"],
        "training_fit_role": training_report["score_role"],
        "strict_oof_reference_HOTA": 79.53797,
        "absolute_threshold_rejected": artifact["probability_threshold"],
        "calibration": (
            "nearest train-sequence descriptor route plus routed train selected-action fraction"
        ),
        "routing_features": ROUTING_FEATURES,
        "train_descriptors": train_descriptors,
        "test_descriptors": test_descriptors,
        "routes": routes,
        "route_distances": route_distances,
        "train_selected_fractions": train_fractions,
        "sequences": sequence_reports,
        "package_root": str(package_root.relative_to(REPO)),
        "zip_path": str(zip_path.relative_to(REPO)),
        "zip_sha256": sha256(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "files": [
            {
                "name": f"{seq}.txt",
                "sha256": sha256(package_root / f"{seq}.txt"),
                "bytes": (package_root / f"{seq}.txt").stat().st_size,
            }
            for seq in TEST_SEQUENCES
        ],
    }
    (output_root / "submission_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
