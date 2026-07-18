from __future__ import annotations

# Research artifact for the MOT20 M23 nested sequence-LOSO audit.

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SEQS = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
LAMBDAS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]
LABEL_ROOT = Path(
    "outputs/mot20_m23_20260718/micrograph_chain_transaction_oracle_v1/labels"
)
OUTER_PRED = Path(
    "outputs/mot20_m23_20260718/"
    "micrograph_chain_expected_utility_loso_v2_segment_app"
)
META = Path("outputs/mot20_m23_20260718/micrograph_chunk30_v1")
UTILITY = Path("outputs/mot20_m23_20260718/micrograph_chunk30_assa_utility_v1")
OUT = Path("outputs/mot20_m23_20260718/m23_13_nested_chain_risk_policy_v1")
NAME = "nested_chain_risk_policy_v1"
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


def fit_inner_predictions(
    expected,
    all_data: pd.DataFrame,
    outer_held: str,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    training_sequences = [seq for seq in SEQS if seq != outer_held]
    predictions = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for inner_index, inner_held in enumerate(training_sequences):
        fit_sequences = [
            seq for seq in training_sequences if seq != inner_held
        ]
        train = all_data[all_data.seq.isin(fit_sequences)].copy()
        test = all_data[all_data.seq == inner_held].copy()
        seed_index = SEQS.index(outer_held) * 10 + inner_index
        models = expected.fit_models(train, seed_index)
        prediction = expected.predict(test, models)
        compact_columns = [
            "seq",
            "src_chunk",
            "dst_chunk",
            "transaction_src_track_id",
            "transaction_dst_track_id",
            "chain_transaction_delta_proxy",
            "chain_transaction_positive",
            "pred_transaction_positive_prob_unweighted",
            "pred_transaction_positive_gain",
            "pred_transaction_negative_loss",
            "pred_expected_transaction_utility",
        ]
        prediction[compact_columns].to_parquet(
            output_dir / f"{inner_held}_predictions.parquet", index=False
        )
        predictions[inner_held] = prediction
    return predictions


def calibrate_lambda(inner_predictions: dict[str, pd.DataFrame]) -> tuple[float | None, dict]:
    candidates = []
    chosen = None
    for loss_multiplier in LAMBDAS:
        by_seq = {}
        total_selected = 0
        safe = True
        for seq, prediction in inner_predictions.items():
            selected = greedy_disjoint(
                prediction, risk_score(prediction, loss_multiplier)
            )
            audit = selection_audit(selected)
            by_seq[seq] = audit
            total_selected += audit["selected"]
            safe = safe and audit["true_delta_sum"] >= -EPS
        record = {
            "loss_multiplier": loss_multiplier,
            "safe_on_every_inner_sequence": safe,
            "total_selected": total_selected,
            "by_seq": by_seq,
        }
        candidates.append(record)
        if chosen is None and safe and total_selected > 0:
            chosen = loss_multiplier
    return chosen, {
        "rule": (
            "choose the smallest pre-registered loss multiplier with nonnegative "
            "true transaction-utility sum on every inner held sequence and at "
            "least one selected action; otherwise choose fold-level no-op"
        ),
        "lambda_grid": LAMBDAS,
        "chosen_loss_multiplier": chosen,
        "candidates": candidates,
    }


def inference_manifest(frame: pd.DataFrame, held: str, loss_multiplier: float | None) -> pd.DataFrame:
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
    output.insert(1, "chosen_loss_multiplier", loss_multiplier)
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    expected = load_module(
        "m23_12_expected",
        Path("scripts/m23_research/m23_12_train_chain_expected_utility_loso.py"),
    )
    oracle = load_module(
        "m23_12_oracle",
        Path("scripts/m23_research/m23_12_chain_transaction_oracle.py"),
    )
    evaluator = load_module(
        "m23_11_evaluator",
        Path("scripts/m23_research/m23_11_eval_utility_graph.py"),
    )
    evaluator.NAME = NAME

    frames = []
    for seq in SEQS:
        raw = pd.read_parquet(
            LABEL_ROOT / seq / "cross_chain_transaction_utility.parquet"
        )
        frames.append(expected.base.add_chain_features(seq, raw))
    all_data = pd.concat(frames, ignore_index=True, sort=False)

    fold_reports = []
    frozen_manifests = []
    selected_for_evaluation = {}
    for outer_held in SEQS:
        inner_predictions = fit_inner_predictions(
            expected,
            all_data,
            outer_held,
            OUT / "inner_predictions" / f"outer_{outer_held}",
        )
        chosen_lambda, calibration = calibrate_lambda(inner_predictions)
        outer = pd.read_parquet(
            OUTER_PRED
            / f"{outer_held}_chain_expected_utility_predictions.parquet"
        )
        if chosen_lambda is None:
            selected = outer.iloc[0:0].copy()
            selected["policy_score"] = np.asarray([], dtype=float)
        else:
            selected = greedy_disjoint(
                outer, risk_score(outer, chosen_lambda)
            )
        selected_for_evaluation[outer_held] = selected
        frozen_manifests.append(
            inference_manifest(selected, outer_held, chosen_lambda)
        )
        fold_reports.append(
            {
                "outer_held_seq": outer_held,
                "outer_model_fit_sequences": [
                    seq for seq in SEQS if seq != outer_held
                ],
                "outer_gt_used_in_model_or_policy_selection": False,
                "calibration": calibration,
                "outer_selected": len(selected),
                "outer_predicted_score_sum": float(selected.policy_score.sum()),
            }
        )
        print(
            json.dumps(
                {
                    "outer_held_seq": outer_held,
                    "chosen_loss_multiplier": chosen_lambda,
                    "outer_selected": len(selected),
                }
            ),
            flush=True,
        )

    manifest = pd.concat(frozen_manifests, ignore_index=True, sort=False)
    manifest_path = OUT / "frozen_outer_selection.csv"
    manifest.to_csv(manifest_path, index=False)
    selection_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    frozen_protocol = {
        "status": "outer selection frozen before TrackEval",
        "selection_sha256": selection_sha256,
        "outer_gt_used_in_model_or_policy_selection": False,
        "inner_gt_use": (
            "training-sequence chain-transaction utility labels only for "
            "loss-multiplier calibration"
        ),
        "feature_gt_use": "none; inherited explicit GT-free allowlist",
        "loss_multiplier_grid": LAMBDAS,
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
        # GT-derived columns below are consumed only after the frozen manifest
        # is written. They are diagnostics and never enter selection.
        outer_diagnostics.append(
            {"seq": held, **selection_audit(selected)}
        )
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
            "nested sequence-LOSO policy audit on reused development "
            "sequences; fixed-parent provenance remains exploratory"
        ),
        "deployment_allowed": False,
        "parent": (
            "fixed GT-free exploratory fused A42 parent at 78.763497 HOTA; "
            "formal deployable anchor remains 77.699"
        ),
        "selection_sha256": selection_sha256,
        "protocol": frozen_protocol,
        "outer_postfreeze_diagnostics": outer_diagnostics,
        "tracker_reports": tracker_reports,
        "eval": evaluator.evaluate(root),
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
