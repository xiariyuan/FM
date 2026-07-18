from __future__ import annotations

# Research artifact for the MOT20 M23 truncate-before-conflict audit.

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def truncate_then_disjoint(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Truncate the ranking first; remove conflicts without refilling the budget."""
    candidates = frame.copy()
    candidates["policy_score"] = candidates[
        "pred_expected_transaction_utility"
    ].to_numpy(float)
    candidates.sort_values(
        ["policy_score", "src_chunk", "dst_chunk"],
        ascending=[False, True, True],
        inplace=True,
    )
    candidates = candidates.head(top_k)
    used_tracks: set[int] = set()
    selected_indices = []
    for index, edge in candidates.iterrows():
        src_track = int(edge.transaction_src_track_id)
        dst_track = int(edge.transaction_dst_track_id)
        if src_track in used_tracks or dst_track in used_tracks:
            continue
        used_tracks.add(src_track)
        used_tracks.add(dst_track)
        selected_indices.append(index)
    return candidates.loc[selected_indices].copy()


def main() -> None:
    base = load_module(
        "m23_15_rank_budget_base",
        Path("scripts/m23_research/m23_15_nested_rank_budget_policy.py"),
    )
    base.OUT = Path(
        "outputs/mot20_m23_20260718/"
        "m23_16_truncate_then_disjoint_policy_v1"
    )
    base.NAME = "nested_truncate_then_disjoint_ood_policy_v1"
    base.topk_disjoint = truncate_then_disjoint

    original_calibrate = base.calibrate_rank_budget

    def calibrate(inner_predictions):
        chosen, report = original_calibrate(inner_predictions)
        report["feasibility_operator"] = (
            "sort all candidates by GT-free expected utility; truncate to the "
            "first K rows; greedily remove track conflicts inside that fixed "
            "prefix; never refill from ranks below K"
        )
        report["difference_from_m23_15"] = (
            "M23-15 refilled each rejected conflict from lower ranks until K "
            "executable transactions were selected"
        )
        return chosen, report

    base.calibrate_rank_budget = calibrate
    base.main()


if __name__ == "__main__":
    main()
