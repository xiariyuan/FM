from __future__ import annotations

# Research artifact for the MOT20 M23 gate-free pair-state policy audit.

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


SEQS = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
TOP_K_GRID = [10, 25, 50, 100, 250, 500, 1000, 2500]
SOURCE = Path("outputs/mot20_m23_20260718/m23_21_pair_state_metric_utility_v1")
OUT = Path("outputs/mot20_m23_20260718/m23_22_pair_state_budget_policy_v1")
NAME = "pair_state_nested_budget_policy_v1"
SCORE = "pred_pair_state_normalized_utility"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_module(
        "m23_17_pair_state_budget_base",
        Path("scripts/m23_research/m23_17_sequence_normalized_utility.py"),
    )
    oracle = load_module(
        "m23_12_pair_state_budget_oracle",
        Path("scripts/m23_research/m23_12_chain_transaction_oracle.py"),
    )
    evaluator = load_module(
        "m23_11_pair_state_budget_evaluator",
        Path("scripts/m23_research/m23_11_eval_utility_graph.py"),
    )
    base.SCORE = SCORE
    base.TOP_K_GRID = TOP_K_GRID
    evaluator.NAME = NAME

    source_protocol = json.loads((SOURCE / "frozen_protocol.json").read_text())
    frozen_manifests = []
    frozen_policy_rows = []
    fold_reports = []
    selected_for_evaluation = {}
    for outer_held in SEQS:
        training_sequences = [seq for seq in SEQS if seq != outer_held]
        inner_predictions = {
            inner_held: pd.read_parquet(
                SOURCE
                / "inner_predictions"
                / f"outer_{outer_held}"
                / f"{inner_held}_predictions.parquet"
            )
            for inner_held in training_sequences
        }
        calibrated_top_k, calibration = base.calibrate_top_k(inner_predictions)
        outer = pd.read_parquet(
            SOURCE
            / "outer_predictions"
            / f"{outer_held}_predictions_gt_free.parquet"
        )
        selected = (
            base.empty_selection(outer)
            if calibrated_top_k is None
            else base.truncate_then_disjoint(outer, calibrated_top_k)
        )
        selected_for_evaluation[outer_held] = selected
        frozen_manifests.append(
            base.inference_manifest(selected, outer_held, calibrated_top_k)
        )
        policy_row = {
            "seq": outer_held,
            "calibrated_top_k": calibrated_top_k,
            "applied_top_k": calibrated_top_k,
            "outer_selected": len(selected),
            "outer_predicted_score_sum": float(selected.policy_score.sum()),
            "ood_gate": "none",
        }
        frozen_policy_rows.append(policy_row)
        fold_reports.append(
            {
                "outer_held_seq": outer_held,
                "outer_model_fit_sequences": training_sequences,
                "outer_gt_used_in_model_or_policy_selection": False,
                "calibration": calibration,
                "ood_gate": {
                    "enabled": False,
                    "reason": (
                        "the pair-state representation is sequence-rank normalized; "
                        "the frozen M23-14 scalar-model probability gate is not reused"
                    ),
                },
                "applied_top_k": calibrated_top_k,
                "outer_selected": len(selected),
                "outer_predicted_score_sum": float(selected.policy_score.sum()),
            }
        )
        print(json.dumps(policy_row), flush=True)

    manifest = pd.concat(frozen_manifests, ignore_index=True, sort=False)
    manifest_path = OUT / "frozen_outer_selection.csv"
    manifest.to_csv(manifest_path, index=False)
    selection_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    policy_frame = pd.DataFrame(frozen_policy_rows)
    policy_path = OUT / "frozen_outer_policy.csv"
    policy_frame.to_csv(policy_path, index=False)
    policy_sha256 = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    frozen_protocol = {
        "status": "outer policy and transaction selection frozen before TrackEval",
        "selection_sha256": selection_sha256,
        "policy_sha256": policy_sha256,
        "outer_gt_used_in_model_or_policy_selection": False,
        "training_gt_use": (
            "M23-21 sequence-normalized utility and sign labels on each fold's fit "
            "sequences; inner labels from outer-training sequences calibrate budget"
        ),
        "inference_gt_use": "none",
        "source_pair_state_protocol_sha256": hashlib.sha256(
            (SOURCE / "frozen_protocol.json").read_bytes()
        ).hexdigest(),
        "source_pair_state_selection_sha256": source_protocol["selection_sha256"],
        "score": SCORE,
        "top_k_grid": TOP_K_GRID,
        "ood_gate": "none",
        "folds": fold_reports,
    }
    (OUT / "frozen_protocol.json").write_text(
        json.dumps(frozen_protocol, indent=2) + "\n"
    )

    root = OUT / NAME
    tracker_reports = []
    outer_diagnostics = []
    for held in SEQS:
        selected = selected_for_evaluation[held]
        # Held-sequence labels are consumed only after all frozen artifacts exist.
        labels = pd.read_parquet(
            base.LABEL_ROOT / held / "cross_chain_transaction_utility.parquet",
            columns=[
                "src_chunk",
                "dst_chunk",
                base.TARGET,
            ],
        )
        selected_with_labels = selected.merge(
            labels, on=["src_chunk", "dst_chunk"], how="left", validate="one_to_one"
        )
        outer_diagnostics.append(
            {"seq": held, **base.selection_audit(selected_with_labels)}
        )
        edges = pd.read_parquet(
            base.UTILITY / held / "candidate_edges_utility.parquet"
        )
        applied = oracle.apply_transactions(edges, selected_with_labels)
        meta = pd.read_parquet(base.META / held / "microtracklets.parquet")
        tracker_reports.append(
            evaluator.write_tracker(
                held,
                meta,
                applied,
                root / "track_results" / f"{held}.txt",
            )
        )

    report = {
        "name": NAME,
        "status": (
            "strict nested gate-free pair-state budget audit on reused development "
            "sequences; fixed-parent provenance remains exploratory"
        ),
        "deployment_allowed": False,
        "parent": (
            "fixed GT-free exploratory fused A42 parent at 78.763497 HOTA; "
            "formal deployable anchor remains 77.699"
        ),
        "selection_sha256": selection_sha256,
        "policy_sha256": policy_sha256,
        "protocol": frozen_protocol,
        "outer_postfreeze_diagnostics": outer_diagnostics,
        "tracker_reports": tracker_reports,
        "eval": evaluator.evaluate(root),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
