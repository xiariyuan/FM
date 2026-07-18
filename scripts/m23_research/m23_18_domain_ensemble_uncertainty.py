from __future__ import annotations

# Research artifact for the MOT20 M23 leave-domain-out uncertainty audit.

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SEQS = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
TOP_K_GRID = [10, 25, 50, 100, 250]
SCORE_MODES = ["full_positive", "median_positive", "lcb_0p5_positive", "minimum_positive"]
KEYS = [
    "src_chunk",
    "dst_chunk",
    "transaction_src_track_id",
    "transaction_dst_track_id",
]
LABEL_ROOT = Path(
    "outputs/mot20_m23_20260718/micrograph_chain_transaction_oracle_v1/labels"
)
M17 = Path("outputs/mot20_m23_20260718/m23_17_sequence_normalized_utility_v1")
META = Path("outputs/mot20_m23_20260718/micrograph_chunk30_v1")
UTILITY = Path("outputs/mot20_m23_20260718/micrograph_chunk30_assa_utility_v1")
OUT = Path("outputs/mot20_m23_20260718/m23_18_domain_ensemble_uncertainty_v1")
NAME = "domain_ensemble_uncertainty_ood_policy_v1"
TARGET = "chain_transaction_delta_proxy"
M17_SCORE = "pred_sequence_normalized_utility"
EPS = 1e-9


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_score(path: Path, reference: pd.DataFrame) -> np.ndarray:
    prediction = pd.read_parquet(path)
    left = reference[KEYS].reset_index(drop=True)
    right = prediction[KEYS].reset_index(drop=True)
    if not left.equals(right):
        raise RuntimeError(f"prediction keys do not align: {path}")
    return prediction[M17_SCORE].to_numpy(float)


def add_ensemble_scores(
    frame: pd.DataFrame,
    full_score: np.ndarray,
    jackknife_scores: list[np.ndarray],
) -> pd.DataFrame:
    output = frame.copy()
    components = np.column_stack([full_score, *jackknife_scores])
    median = np.median(components, axis=1)
    component_range = components.max(axis=1) - components.min(axis=1)
    output["ensemble_full"] = full_score
    output["ensemble_median"] = median
    output["ensemble_range"] = component_range
    output["ensemble_lcb_0p5"] = median - 0.5 * component_range
    output["ensemble_minimum"] = components.min(axis=1)
    return output


def score_column(mode: str) -> str:
    return {
        "full_positive": "ensemble_full",
        "median_positive": "ensemble_median",
        "lcb_0p5_positive": "ensemble_lcb_0p5",
        "minimum_positive": "ensemble_minimum",
    }[mode]


def select_transactions(frame: pd.DataFrame, mode: str, top_k: int) -> pd.DataFrame:
    column = score_column(mode)
    candidates = frame[frame[column] > 0].copy()
    candidates["policy_score"] = candidates[column].to_numpy(float)
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


def selection_audit(selected: pd.DataFrame) -> dict:
    actual = selected[TARGET].to_numpy(float)
    return {
        "selected": len(selected),
        "true_positive": int((actual > 0).sum()),
        "true_precision": float((actual > 0).mean()) if len(actual) else None,
        "true_delta_sum": float(actual.sum()),
        "predicted_score_sum": float(selected.policy_score.sum()),
    }


def calibrate_policy(
    inner_predictions: dict[str, pd.DataFrame],
) -> tuple[dict | None, dict]:
    candidates = []
    mode_preference = {mode: len(SCORE_MODES) - index for index, mode in enumerate(SCORE_MODES)}
    for mode in SCORE_MODES:
        for top_k in TOP_K_GRID:
            by_seq = {}
            deltas = []
            for seq, prediction in inner_predictions.items():
                selected = select_transactions(prediction, mode, top_k)
                audit = selection_audit(selected)
                by_seq[seq] = audit
                deltas.append(audit["true_delta_sum"])
            candidates.append(
                {
                    "score_mode": mode,
                    "top_k": top_k,
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
                item["total_true_delta_sum"],
                mode_preference[item["score_mode"]],
                -item["top_k"],
            ),
        )
        chosen = {"score_mode": best["score_mode"], "top_k": int(best["top_k"])}
    return chosen, {
        "rule": (
            "choose the positive-score ensemble mode and truncate-then-disjoint K "
            "with the largest positive median true utility across the three inner "
            "held sequences; total utility then the fixed mode order break ties"
        ),
        "score_modes": SCORE_MODES,
        "top_k_grid": TOP_K_GRID,
        "chosen_before_sequence_ood_gate": chosen,
        "candidates": candidates,
    }


def empty_selection(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.iloc[0:0].copy()
    selected["policy_score"] = np.asarray([], dtype=float)
    return selected


def inference_manifest(
    frame: pd.DataFrame,
    held: str,
    score_mode: str | None,
    top_k: int | None,
) -> pd.DataFrame:
    columns = [
        *KEYS,
        "ensemble_full",
        "ensemble_median",
        "ensemble_range",
        "ensemble_lcb_0p5",
        "ensemble_minimum",
        "policy_score",
    ]
    output = frame[columns].copy()
    output.insert(0, "seq", held)
    output.insert(1, "applied_score_mode", score_mode)
    output.insert(2, "applied_top_k", top_k)
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m17 = load_module(
        "m23_17_base_ensemble",
        Path("scripts/m23_research/m23_17_sequence_normalized_utility.py"),
    )
    expected = load_module(
        "m23_12_expected_ensemble",
        Path("scripts/m23_research/m23_12_train_chain_expected_utility_loso.py"),
    )
    oracle = load_module(
        "m23_12_oracle_ensemble",
        Path("scripts/m23_research/m23_12_chain_transaction_oracle.py"),
    )
    evaluator = load_module(
        "m23_11_evaluator_ensemble",
        Path("scripts/m23_research/m23_11_eval_utility_graph.py"),
    )
    evaluator.NAME = NAME

    frames = []
    for seq in SEQS:
        raw = pd.read_parquet(LABEL_ROOT / seq / "cross_chain_transaction_utility.parquet")
        enriched = expected.base.add_chain_features(seq, raw)
        enriched, normalized_features = m17.rank_normalize(enriched, expected.base.FEATURES)
        frames.append(enriched)
    all_data = pd.concat(frames, ignore_index=True, sort=False)
    by_seq = {seq: all_data[all_data.seq == seq].copy() for seq in SEQS}

    singleton_predictions = {}
    singleton_reports = {}
    singleton_root = OUT / "singleton_predictions"
    singleton_root.mkdir(parents=True, exist_ok=True)
    for fit_seq in SEQS:
        model, metadata = m17.fit_model(by_seq[fit_seq], normalized_features, 18000 + SEQS.index(fit_seq))
        singleton_reports[fit_seq] = {
            "fit_sequence": fit_seq,
            "train_rows": len(by_seq[fit_seq]),
            **metadata,
        }
        for target_seq in SEQS:
            if target_seq == fit_seq:
                continue
            score = model.predict(by_seq[target_seq][normalized_features])
            singleton_predictions[(fit_seq, target_seq)] = score
            compact = by_seq[target_seq][KEYS].copy()
            compact[M17_SCORE] = score
            compact.to_parquet(
                singleton_root / f"fit_{fit_seq}_target_{target_seq}.parquet",
                index=False,
            )
        print(json.dumps({"singleton_fit": fit_seq}), flush=True)

    m17_policy = pd.read_csv(M17 / "frozen_outer_policy.csv").set_index("seq")
    fold_reports = []
    frozen_manifests = []
    frozen_policy_rows = []
    selected_for_evaluation = {}
    for outer_held in SEQS:
        training_sequences = [seq for seq in SEQS if seq != outer_held]
        inner_predictions = {}
        for inner_held in training_sequences:
            fit_sequences = [seq for seq in training_sequences if seq != inner_held]
            reference = by_seq[inner_held]
            full_score = load_score(
                M17
                / "inner_predictions"
                / f"outer_{outer_held}"
                / f"{inner_held}_predictions.parquet",
                reference,
            )
            jackknife_scores = [
                singleton_predictions[(fit_seq, inner_held)] for fit_seq in fit_sequences
            ]
            inner_predictions[inner_held] = add_ensemble_scores(
                reference, full_score, jackknife_scores
            )

        chosen, calibration = calibrate_policy(inner_predictions)
        reference = by_seq[outer_held]
        full_score = load_score(
            M17 / "outer_predictions" / f"{outer_held}_predictions_gt_free.parquet",
            reference,
        )
        pair_scores = []
        for excluded_training_seq in training_sequences:
            pair_scores.append(
                load_score(
                    M17
                    / "inner_predictions"
                    / f"outer_{excluded_training_seq}"
                    / f"{outer_held}_predictions.parquet",
                    reference,
                )
            )
        outer = add_ensemble_scores(reference, full_score, pair_scores)

        upper_ood = bool(m17_policy.loc[outer_held, "upper_ood"])
        applied = None if upper_ood else chosen
        if applied is None:
            selected = empty_selection(outer)
            applied_mode = None
            applied_top_k = None
        else:
            applied_mode = applied["score_mode"]
            applied_top_k = int(applied["top_k"])
            selected = select_transactions(outer, applied_mode, applied_top_k)
        selected_for_evaluation[outer_held] = selected
        frozen_manifests.append(
            inference_manifest(selected, outer_held, applied_mode, applied_top_k)
        )
        policy_row = {
            "seq": outer_held,
            "calibrated_score_mode": None if chosen is None else chosen["score_mode"],
            "calibrated_top_k": None if chosen is None else chosen["top_k"],
            "upper_ood": upper_ood,
            "applied_score_mode": applied_mode,
            "applied_top_k": applied_top_k,
            "outer_selected": len(selected),
            "outer_predicted_score_sum": float(selected.policy_score.sum()),
        }
        frozen_policy_rows.append(policy_row)
        fold_reports.append(
            {
                "outer_held_seq": outer_held,
                "outer_model_fit_sequences": training_sequences,
                "outer_gt_used_in_model_or_policy_selection": False,
                "calibration": calibration,
                "sequence_ood_gate": {
                    "source": "frozen M23-17/M23-14 q90 gate",
                    "upper_ood": upper_ood,
                },
                "applied_policy": applied,
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
        "status": "outer ensemble policy and selection frozen before TrackEval",
        "selection_sha256": selection_sha256,
        "policy_sha256": policy_sha256,
        "outer_gt_used_in_model_or_policy_selection": False,
        "training_gt_use": "fit-sequence transaction utility targets only",
        "inference_gt_use": "none",
        "ensemble": (
            "outer full three-domain model plus three leave-one-training-domain-out "
            "pair models; inner full pair model plus its two singleton models"
        ),
        "singleton_models": singleton_reports,
        "folds": fold_reports,
    }
    (OUT / "frozen_protocol.json").write_text(json.dumps(frozen_protocol, indent=2) + "\n")

    root = OUT / NAME
    tracker_reports = []
    outer_diagnostics = []
    for held in SEQS:
        selected = selected_for_evaluation[held]
        # Outer labels are consumed only after both frozen manifests are written.
        outer_diagnostics.append({"seq": held, **selection_audit(selected)})
        edges = pd.read_parquet(UTILITY / held / "candidate_edges_utility.parquet")
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
            "nested leave-domain-out ensemble audit on reused development "
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
