from __future__ import annotations

# Research artifact for the MOT20 M23 sequence-LOSO audit.

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEQS = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
DATA = Path("outputs/mot20_m23_20260718/micrograph_chunk30_v1")
PRED = Path("outputs/mot20_m23_20260718/micrograph_chunk30_utility_loso_v1")
OUT = Path("outputs/mot20_m23_20260718/micrograph_chunk30_endpoint_utility_explore_v1")
NAME = "preserve_source_positive_utility"


def load_base():
    path = Path(__file__).with_name("m23_11_eval_utility_graph.py")
    spec = importlib.util.spec_from_file_location("m23_11_eval_utility_graph", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.NAME = NAME
    return module


def select_edges(source: pd.DataFrame, cross: pd.DataFrame, chunk_count: int, base):
    source = source.copy()
    source["edge_role"] = "source"
    source["utility"] = np.maximum(
        source.pred_keep_expected_utility.to_numpy(float), 1e-6
    )
    used_src = set(source.src_chunk.astype(int))
    used_dst = set(source.dst_chunk.astype(int))
    cross = cross[
        (cross.pred_risk_adjusted_utility > 0)
        & ~cross.src_chunk.astype(int).isin(used_src)
        & ~cross.dst_chunk.astype(int).isin(used_dst)
    ].copy()
    cross["edge_role"] = "cross"
    cross["utility"] = cross.pred_risk_adjusted_utility.to_numpy(float)
    cross.sort_values("utility", ascending=False, inplace=True)
    cross.drop_duplicates(["src_chunk", "dst_chunk"], keep="first", inplace=True)
    cross = base.maximum_weight_path_edges(cross, chunk_count)
    return pd.concat([source, cross], ignore_index=True, sort=False)


def main() -> None:
    base = load_base()
    root = OUT / NAME
    sequence_reports = []
    for seq in SEQS:
        meta = pd.read_parquet(DATA / seq / "microtracklets.parquet")
        source = pd.read_parquet(PRED / f"{seq}_source_utility_predictions.parquet")
        cross = pd.read_parquet(PRED / f"{seq}_cross_utility_predictions.parquet")
        selected = select_edges(source, cross, len(meta), base)
        selected_dir = root / "selected_edges"
        selected_dir.mkdir(parents=True, exist_ok=True)
        selected.to_parquet(selected_dir / f"{seq}.parquet", index=False)
        sequence_reports.append(
            base.write_tracker(
                seq, meta, selected, root / "track_results" / f"{seq}.txt"
            )
        )
    report = {
        "name": NAME,
        "status": "exploratory fixed safety control; not a nested selected policy",
        "deployment_allowed": False,
        "gt_used_in_selection_or_inference": False,
        "held_sequence_excluded_from_utility_model_fit": True,
        "rule": {
            "source": "preserve every source-adjacent edge exactly",
            "cross": (
                "among free parent-chain endpoints, use every positive "
                "pred_risk_adjusted_utility candidate"
            ),
            "optimizer": "maximum predicted-utility one-to-one temporal matching",
            "held_out_threshold_tuning": "none",
        },
        "by_seq": sequence_reports,
        "eval": base.evaluate(root),
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
