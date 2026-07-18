from __future__ import annotations

# Research artifact for the MOT20 M23 nested sequence-level OOD audit.

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SEQS = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
LAMBDAS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]
INNER_ROOT = Path(
    "outputs/mot20_m23_20260718/m23_13_nested_chain_risk_policy_v1/inner_predictions"
)
OUTER_PRED = Path(
    "outputs/mot20_m23_20260718/"
    "micrograph_chain_expected_utility_loso_v2_segment_app"
)
META = Path("outputs/mot20_m23_20260718/micrograph_chunk30_v1")
UTILITY = Path("outputs/mot20_m23_20260718/micrograph_chunk30_assa_utility_v1")
OUT = Path("outputs/mot20_m23_20260718/m23_14_sequence_ood_gate_v1")
NAME = "sequence_ood_median_risk_policy_v1"
PROBABILITY_COLUMN = "pred_transaction_positive_prob_unweighted"
OOD_QUANTILE = 0.90
EPS = 1e-9


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def risk_score(frame: pd.DataFrame, loss_multiplier: float) -> np.ndarray:
    probability = frame.pred_transaction_positive_prob_unweighted.to_numpy(float)
    gain = frame.pred_transaction_positive_gain.to_numpy(float)
    loss = frame.pred_transaction_negative_loss.to_numpy(float)
    return probability * gain - loss_multiplier * (1.0 - probability) * loss


def greedy_disjoint(frame: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    candidates = frame.copy()
    candidates["policy_score"] = score
    candidates = candidates[candidates.policy_score > 0].copy()
    candidates.sort_values(
        ["policy_score", "src_chunk", "dst_chunk"],
        ascending=[False, True, True],
        inplace=True,
    )
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


def selection_audit(selected: pd.DataFrame) -> dict:
    actual = selected.chain_transaction_delta_proxy.to_numpy(float)
    return {
        "selected": len(selected),
        "true_positive": int((actual > 0).sum()),
        "true_precision": float((actual > 0).mean()) if len(actual) else None,
        "true_delta_sum": float(actual.sum()),
        "predicted_score_sum": float(selected.policy_score.sum()),
    }


def load_inner_predictions(outer_held: str) -> dict[str, pd.DataFrame]:
    predictions = {}
    root = INNER_ROOT / f"outer_{outer_held}"
    for inner_held in SEQS:
        if inner_held == outer_held:
            continue
        predictions[inner_held] = pd.read_parquet(
            root / f"{inner_held}_predictions.parquet"
        )
    return predictions


def calibrate_median_lambda(
    inner_predictions: dict[str, pd.DataFrame],
) -> tuple[float | None, dict]:
    candidates = []
    for loss_multiplier in LAMBDAS:
        by_seq = {}
        deltas = []
        for seq, prediction in inner_predictions.items():
            selected = greedy_disjoint(
                prediction, risk_score(prediction, loss_multiplier)
            )
            audit = selection_audit(selected)
            by_seq[seq] = audit
            deltas.append(audit["true_delta_sum"])
        candidates.append(
            {
                "loss_multiplier": loss_multiplier,
                "median_true_delta_sum": float(np.median(deltas)),
                "total_true_delta_sum": float(np.sum(deltas)),
                "total_selected": int(sum(item["selected"] for item in by_seq.values())),
                "by_seq": by_seq,
            }
        )

    positive = [item for item in candidates if item["median_true_delta_sum"] > EPS]
    chosen = None
    if positive:
        best = max(
            positive,
            key=lambda item: (
                item["median_true_delta_sum"],
                -item["loss_multiplier"],
            ),
        )
        chosen = float(best["loss_multiplier"])
    return chosen, {
        "rule": (
            "choose the pre-registered loss multiplier with the largest positive "
            "median true transaction-utility sum over the three inner held "
            "sequences; ties choose the smaller multiplier; otherwise no-op"
        ),
        "lambda_grid": LAMBDAS,
        "chosen_loss_multiplier_before_ood_gate": chosen,
        "candidates": candidates,
    }


def probability_q90(frame: pd.DataFrame) -> float:
    return float(frame[PROBABILITY_COLUMN].quantile(OOD_QUANTILE))


def empty_selection(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.iloc[0:0].copy()
    selected["policy_score"] = np.asarray([], dtype=float)
    return selected


def inference_manifest(
    frame: pd.DataFrame,
    held: str,
    applied_loss_multiplier: float | None,
) -> pd.DataFrame:
    columns = [
        "src_chunk",
        "dst_chunk",
        "transaction_src_track_id",
        "transaction_dst_track_id",
        "pred_transaction_positive_prob_unweighted",
        "pred_transaction_positive_gain",
        "pred_transaction_negative_loss",
        "policy_score",
    ]
    output = frame[columns].copy()
    output.insert(0, "seq", held)
    output.insert(1, "applied_loss_multiplier", applied_loss_multiplier)
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    oracle = load_module(
        "m23_12_oracle",
        Path("scripts/m23_research/m23_12_chain_transaction_oracle.py"),
    )
    evaluator = load_module(
        "m23_11_evaluator",
        Path("scripts/m23_research/m23_11_eval_utility_graph.py"),
    )
    evaluator.NAME = NAME

    fold_reports = []
    frozen_manifests = []
    frozen_policy_rows = []
    selected_for_evaluation = {}
    for outer_held in SEQS:
        inner_predictions = load_inner_predictions(outer_held)
        calibrated_lambda, calibration = calibrate_median_lambda(inner_predictions)
        outer = pd.read_parquet(
            OUTER_PRED
            / f"{outer_held}_chain_expected_utility_predictions.parquet"
        )

        inner_q90 = {
            seq: probability_q90(frame)
            for seq, frame in inner_predictions.items()
        }
        outer_q90 = probability_q90(outer)
        ood_upper_threshold = float(max(inner_q90.values()))
        upper_ood = outer_q90 > ood_upper_threshold
        applied_lambda = None if upper_ood else calibrated_lambda
        if applied_lambda is None:
            selected = empty_selection(outer)
        else:
            selected = greedy_disjoint(outer, risk_score(outer, applied_lambda))

        selected_for_evaluation[outer_held] = selected
        frozen_manifests.append(
            inference_manifest(selected, outer_held, applied_lambda)
        )
        policy_row = {
            "seq": outer_held,
            "calibrated_loss_multiplier": calibrated_lambda,
            "outer_probability_q90": outer_q90,
            "inner_probability_q90_max": ood_upper_threshold,
            "upper_ood": upper_ood,
            "applied_loss_multiplier": applied_lambda,
            "outer_selected": len(selected),
            "outer_predicted_score_sum": float(selected.policy_score.sum()),
        }
        frozen_policy_rows.append(policy_row)
        fold_reports.append(
            {
                "outer_held_seq": outer_held,
                "outer_model_fit_sequences": [seq for seq in SEQS if seq != outer_held],
                "outer_gt_used_in_model_or_policy_selection": False,
                "calibration": calibration,
                "ood_gate": {
                    "rule": (
                        "force no-op when the outer GT-free positive-probability "
                        "q90 is greater than the maximum q90 of the three inner "
                        "held-sequence predictions; lower-tail extrapolation is "
                        "not gated because the modeled failure is over-selection"
                    ),
                    "probability_column": PROBABILITY_COLUMN,
                    "quantile": OOD_QUANTILE,
                    "inner_q90": inner_q90,
                    "upper_threshold": ood_upper_threshold,
                    "outer_q90": outer_q90,
                    "upper_ood": upper_ood,
                },
                "applied_loss_multiplier": applied_lambda,
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
        "inner_gt_use": (
            "training-sequence chain-transaction utility labels only for median "
            "loss-multiplier calibration"
        ),
        "feature_gt_use": "none; q90 gate and transaction score use GT-free predictions",
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
        # GT-derived columns below are consumed only after all frozen artifacts
        # are written. They are diagnostics and never enter outer selection.
        outer_diagnostics.append({"seq": held, **selection_audit(selected)})
        edges = pd.read_parquet(
            UTILITY / held / "candidate_edges_utility.parquet"
        )
        applied = oracle.apply_transactions(edges, selected)
        meta = pd.read_parquet(META / held / "microtracklets.parquet")
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
            "nested sequence-level GT-free OOD audit on reused development "
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
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
