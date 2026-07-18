from __future__ import annotations

# Research artifact for the MOT20 M23 conflict-graph feature audit.

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def group_rank(
    frame: pd.DataFrame,
    value: str,
    group: str,
    ascending: bool,
) -> pd.Series:
    return frame.groupby(group, sort=False)[value].rank(
        method="average", pct=True, ascending=ascending
    )


def add_conflict_graph_features(
    seq: str,
    frame: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    del seq
    output = frame.copy()
    src = "transaction_src_track_id"
    dst = "transaction_dst_track_id"

    src_out = output.groupby(src, sort=False)[src].transform("size").astype(float)
    dst_in = output.groupby(dst, sort=False)[dst].transform("size").astype(float)
    source_role_counts = output[src].value_counts()
    destination_role_counts = output[dst].value_counts()
    total_counts = source_role_counts.add(destination_role_counts, fill_value=0)
    src_total = output[src].map(total_counts).to_numpy(float)
    dst_total = output[dst].map(total_counts).to_numpy(float)

    added = []

    def assign(name: str, values) -> None:
        output[name] = np.asarray(values, dtype=np.float32)
        added.append(name)

    assign("transaction_src_out_degree", src_out)
    assign("transaction_dst_in_degree", dst_in)
    assign("transaction_src_total_degree", src_total)
    assign("transaction_dst_total_degree", dst_total)
    assign(
        "transaction_src_role_fraction",
        src_out.to_numpy(float) / np.maximum(src_total, 1.0),
    )
    assign(
        "transaction_dst_role_fraction",
        dst_in.to_numpy(float) / np.maximum(dst_total, 1.0),
    )
    assign(
        "transaction_pair_multiplicity",
        output.groupby([src, dst], sort=False)[src].transform("size"),
    )
    assign(
        "transaction_src_chunk_degree",
        output.groupby("src_chunk", sort=False)["src_chunk"].transform("size"),
    )
    assign(
        "transaction_dst_chunk_degree",
        output.groupby("dst_chunk", sort=False)["dst_chunk"].transform("size"),
    )

    rank_specs = [
        ("segment_appearance_cos", False, "segment_app"),
        ("track_appearance_cos", False, "track_app"),
        ("merged_coherence_gain", False, "coherence_gain"),
        ("motion_error_min", True, "motion_error"),
    ]
    for value, ascending, prefix in rank_specs:
        src_rank = group_rank(output, value, src, ascending)
        dst_rank = group_rank(output, value, dst, ascending)
        assign(f"transaction_src_{prefix}_rank", src_rank)
        assign(f"transaction_dst_{prefix}_rank", dst_rank)
        assign(
            f"transaction_mutual_{prefix}_rank_max",
            np.maximum(src_rank.to_numpy(float), dst_rank.to_numpy(float)),
        )

    return output, [*features, *added]


def main() -> None:
    base = load_module(
        "m23_17_sequence_normalized_conflict_base",
        Path("scripts/m23_research/m23_17_sequence_normalized_utility.py"),
    )
    base.OUT = Path(
        "outputs/mot20_m23_20260718/"
        "m23_20_conflict_graph_normalized_utility_v1"
    )
    base.NAME = "conflict_graph_normalized_utility_ood_policy_v1"
    base.SCORE = "pred_conflict_graph_normalized_utility"
    base.FEATURE_TRANSFORM_DESCRIPTION = (
        "M23-17 within-sequence percentile features plus GT-free transaction-track "
        "conflict degrees and mutual within-track appearance/motion ranks"
    )
    base.STATUS_DESCRIPTION = (
        "nested conflict-graph normalized utility audit on reused development "
        "sequences; fixed-parent provenance remains exploratory"
    )
    base.augment_model_features = add_conflict_graph_features
    base.main()


if __name__ == "__main__":
    main()
