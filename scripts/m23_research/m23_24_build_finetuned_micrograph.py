#!/usr/bin/env python3
from __future__ import annotations

"""Rebuild the M23 microtracklet graph from LOSO-fine-tuned embeddings."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
BASE_SCRIPT = REPO / "scripts" / "m23_research" / "m23_10_build_micrograph.py"
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
DEFAULT_PARENT = "outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results"


def load_base():
    spec = importlib.util.spec_from_file_location("m23_10_graph", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-seq", required=True, choices=SEQUENCES)
    parser.add_argument("--phase-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--parent", default=DEFAULT_PARENT)
    parser.add_argument("--seq", action="append", choices=SEQUENCES)
    args = parser.parse_args()

    base = load_base()
    base.PHASE = Path(args.phase_root)
    base.OUT = Path(args.out_root)
    base.PARENT = Path(args.parent)
    base.OUT.mkdir(parents=True, exist_ok=True)
    sequences = args.seq or list(SEQUENCES)
    label_tools = base.load_m23()
    rng = np.random.default_rng(base.SEED)
    projection = rng.normal(size=(2048, base.DIM)).astype(np.float32) / np.sqrt(base.DIM)
    records = []
    for seq in sequences:
        phase_file = base.PHASE / seq / "dump_yolox_reid.npz"
        protocol_file = base.PHASE / seq / "m23_24_reembed_protocol.json"
        if not phase_file.is_file() or not protocol_file.is_file():
            raise FileNotFoundError(f"phase={phase_file} protocol={protocol_file}")
        protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
        if not protocol.get("complete", False):
            raise RuntimeError(f"refusing incomplete re-embedding for {seq}: {protocol_file}")
        if protocol.get("held_sequence") != args.held_seq:
            raise RuntimeError(f"fold mismatch for {seq}: {protocol.get('held_sequence')} != {args.held_seq}")
        tracker_rows = base.read_tracker(base.PARENT / f"{seq}.txt")
        phase, match_iou, embedding, position = base.map_phase(tracker_rows, seq, projection)
        # The outer held graph is built exactly as test inference would be: no GT
        # file is opened.  Training-sequence labels are retained only so later
        # utility-label generation can train the association model.
        if seq == args.held_seq:
            gt = np.zeros(len(tracker_rows), dtype=np.int32)
            gt_usage = "none (outer held inference graph)"
        else:
            gt = base.gt_labels(tracker_rows, seq, label_tools)
            gt_usage = "training labels/diagnostics only"
        metadata, prototypes = base.build_chunks(tracker_rows, phase, match_iou, embedding, position, gt)
        edges = base.build_edges(metadata, prototypes)
        sequence_dir = base.OUT / seq
        sequence_dir.mkdir(parents=True, exist_ok=True)
        metadata.to_parquet(sequence_dir / "microtracklets.parquet", index=False)
        np.save(sequence_dir / "prototypes.f16.npy", prototypes.astype(np.float16))
        edges.to_parquet(sequence_dir / "candidate_edges.parquet", index=False)
        clean = edges[
            (edges.src_purity >= 0.7)
            & (edges.dst_purity >= 0.7)
            & (edges.src_modal_gt > 0)
            & (edges.dst_modal_gt > 0)
        ]
        record = {
            "held_sequence": args.held_seq,
            "seq": seq,
            "tracker_rows": len(tracker_rows),
            "mapped_rows": int((phase >= 0).sum()),
            "chunks": len(metadata),
            "edges": len(edges),
            "clean_edges": len(clean),
            "clean_positive": int(clean.same_gt.sum()),
            "cross_edges": int((edges.same_source == 0).sum()),
            "cross_positive": int(edges.loc[edges.same_source == 0, "same_gt"].sum()),
            "gt_usage": gt_usage,
            "candidate_generation_gt_use": "none",
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    report = {
        "experiment": "M23-24 LOSO-fine-tuned appearance micrograph",
        "held_sequence": args.held_seq,
        "phase_root": str(base.PHASE),
        "parent": str(base.PARENT),
        "projection_seed": base.SEED,
        "projected_dim": base.DIM,
        "sequences": records,
    }
    (base.OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
