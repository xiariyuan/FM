#!/usr/bin/env python3
from __future__ import annotations

"""M23-33 strict nested-LOSO pairwise transaction ranker.

For one outer-held MOT20-train sequence, a RankNet scorer is trained only on
structured-oracle transaction labels from the other sequences.  All scorer
inputs are GT-free and transformed to within-sequence percentiles.  Score
fusion and selection quantile are chosen by exact TrackEval on inner-held
outer-training sequences.  The outer-held sequence is evaluated exactly once
only after that policy is frozen.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
SOURCE_ROOT = (
    REPO
    / "outputs"
    / "mot20_m23_20260718"
    / "m23_26_test_deploy_oof_ensemble_v1"
)
DEFAULT_PARENT = (
    REPO
    / "outputs"
    / "assocriskbench_p15_20260714"
    / "fused_a42_adaptive_risk_eval"
    / "l1_r2_q75_adaptive_priority"
    / "track_results"
)
TARGET = "chain_transaction_delta_proxy"
ORACLE_TARGET = "structured_oracle_selected"


def load_module(name: str, relative_path: str):
    path = REPO / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_float_grid(raw: str) -> List[float]:
    values = sorted({float(value.strip()) for value in raw.split(",") if value.strip()})
    if not values:
        raise ValueError("empty float grid")
    return values


def robust_rank(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    finite = np.isfinite(series.to_numpy(float))
    output = np.full(len(series), 0.5, dtype=np.float32)
    if finite.any():
        ranked = series[finite].rank(method="average", pct=True).to_numpy(np.float32)
        output[finite] = ranked
    return output


def unique_preserve(values: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


@dataclass
class SequenceData:
    seq: str
    labelled: pd.DataFrame
    inference: pd.DataFrame
    matrix: np.ndarray
    utility_rank: np.ndarray
    oracle_count: int


class RankNet(nn.Module):
    def __init__(self, dimension: int, hidden: Sequence[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        current = dimension
        for width in hidden:
            layers.extend(
                [
                    nn.Linear(current, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current = width
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def prepare_sequence(
    seq: str,
    prediction_root: Path,
    feature_names: Sequence[str],
    m28,
) -> SequenceData:
    frame = pd.read_parquet(prediction_root / f"oof_{seq}.parquet")
    frame, _ = m28.add_stack_features(frame)
    frame[ORACLE_TARGET] = m28.derive_structured_oracle_label(frame)
    probability = frame.pred_positive_probability.clip(1e-6, 1.0 - 1e-6).to_numpy(float)
    utility_score = (
        probability * frame.pred_normalized_gain.to_numpy(float)
        - (1.0 - probability) * frame.pred_normalized_loss.to_numpy(float)
    )
    utility_rank = robust_rank(utility_score)
    matrix_columns = []
    missing = []
    for feature in feature_names:
        if feature not in frame:
            missing.append(feature)
            matrix_columns.append(np.full(len(frame), 0.5, dtype=np.float32))
            continue
        values = frame[feature].to_numpy(float)
        matrix_columns.append(robust_rank(values))
    matrix = np.column_stack(matrix_columns).astype(np.float32, copy=False)
    matrix = 2.0 * matrix - 1.0
    inference = frame.drop(columns=[TARGET, ORACLE_TARGET], errors="ignore").copy()
    inference["m23_33_utility_rank"] = utility_rank
    return SequenceData(
        seq=seq,
        labelled=frame,
        inference=inference,
        matrix=matrix,
        utility_rank=utility_rank,
        oracle_count=int(frame[ORACLE_TARGET].sum()),
    )


def hard_negative_pairs(
    data: Mapping[str, SequenceData],
    training_sequences: Sequence[str],
    seed: int,
    conflict_per_positive: int,
    global_per_positive: int,
    random_per_positive: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    rng = np.random.default_rng(seed)
    positive_rows: List[np.ndarray] = []
    negative_rows: List[np.ndarray] = []
    pair_weights: List[np.ndarray] = []
    reports = []
    for seq in training_sequences:
        item = data[seq]
        frame = item.labelled
        labels = frame[ORACLE_TARGET].to_numpy(np.int8)
        positives = np.flatnonzero(labels == 1)
        negatives = np.flatnonzero(labels == 0)
        if not len(positives) or not len(negatives):
            raise RuntimeError(f"insufficient RankNet labels for {seq}")
        utility = item.utility_rank
        src_tracks = frame.transaction_src_track_id.to_numpy(np.int64)
        dst_tracks = frame.transaction_dst_track_id.to_numpy(np.int64)
        negative_order = negatives[np.argsort(-utility[negatives], kind="mergesort")]
        track_to_negatives: Dict[int, List[int]] = {}
        for index in negative_order:
            for track in {
                int(src_tracks[int(index)]),
                int(dst_tracks[int(index)]),
            }:
                bucket = track_to_negatives.setdefault(track, [])
                if len(bucket) < max(64, 2 * conflict_per_positive):
                    bucket.append(int(index))
        positive_utility = frame.loc[positives, TARGET].to_numpy(float)
        scale = max(float(np.median(positive_utility[positive_utility > 0])), 1e-3)
        sequence_positive = []
        sequence_negative = []
        sequence_weight = []
        global_top = negative_order[: max(256, global_per_positive * 8)]
        for positive_index in positives:
            conflict_candidates: List[int] = []
            for track in {
                int(src_tracks[int(positive_index)]),
                int(dst_tracks[int(positive_index)]),
            }:
                conflict_candidates.extend(track_to_negatives.get(track, []))
            conflict_candidates = list(dict.fromkeys(conflict_candidates))
            conflict_candidates.sort(key=lambda index: (-float(utility[index]), index))
            chosen: List[int] = conflict_candidates[:conflict_per_positive]
            for index in global_top:
                if len(chosen) >= conflict_per_positive + global_per_positive:
                    break
                integer = int(index)
                if integer not in chosen:
                    chosen.append(integer)
            if random_per_positive:
                sampled = rng.choice(
                    negatives,
                    size=min(random_per_positive * 3, len(negatives)),
                    replace=False,
                )
                for index in sampled:
                    integer = int(index)
                    if integer not in chosen:
                        chosen.append(integer)
                    if len(chosen) >= (
                        conflict_per_positive + global_per_positive + random_per_positive
                    ):
                        break
            if not chosen:
                continue
            sequence_positive.extend([int(positive_index)] * len(chosen))
            sequence_negative.extend(chosen)
            normalized_value = math.log1p(
                max(float(frame.iloc[int(positive_index)][TARGET]), 0.0) / scale
            )
            sequence_weight.extend([1.0 + normalized_value] * len(chosen))
        positive_rows.append(item.matrix[np.asarray(sequence_positive, dtype=np.int64)])
        negative_rows.append(item.matrix[np.asarray(sequence_negative, dtype=np.int64)])
        pair_weights.append(np.asarray(sequence_weight, dtype=np.float32))
        reports.append(
            {
                "seq": seq,
                "oracle_positives": int(len(positives)),
                "negative_candidates": int(len(negatives)),
                "training_pairs": int(len(sequence_positive)),
            }
        )
    positive_matrix = np.concatenate(positive_rows, axis=0)
    negative_matrix = np.concatenate(negative_rows, axis=0)
    weights = np.concatenate(pair_weights, axis=0)
    weights /= max(float(weights.mean()), 1e-6)
    return positive_matrix, negative_matrix, weights, {
        "seed": seed,
        "training_sequences": list(training_sequences),
        "pairs": int(len(weights)),
        "by_sequence": reports,
    }


def train_one_model(
    positive: np.ndarray,
    negative: np.ndarray,
    weights: np.ndarray,
    dimension: int,
    hidden: Sequence[int],
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> RankNet:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = RankNet(dimension, hidden, dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    positive_tensor = torch.from_numpy(positive)
    negative_tensor = torch.from_numpy(negative)
    weight_tensor = torch.from_numpy(weights)
    generator = torch.Generator().manual_seed(seed + 1000)
    for epoch in range(epochs):
        permutation = torch.randperm(len(weights), generator=generator)
        model.train()
        epoch_loss = 0.0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            pos = positive_tensor[indices].to(device, non_blocking=True)
            neg = negative_tensor[indices].to(device, non_blocking=True)
            current_weight = weight_tensor[indices].to(device, non_blocking=True)
            positive_score = model(pos)
            negative_score = model(neg)
            pair_loss = torch.nn.functional.softplus(
                -(positive_score - negative_score)
            )
            anchor_loss = 0.15 * (
                torch.nn.functional.softplus(-positive_score)
                + torch.nn.functional.softplus(negative_score)
            )
            loss = ((pair_loss + anchor_loss) * current_weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(indices)
        if epoch in {0, epochs - 1} or (epoch + 1) % 10 == 0:
            print(
                json.dumps(
                    {
                        "stage": "ranknet_train",
                        "seed": seed,
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "loss": epoch_loss / len(weights),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return model


def predict_model(
    model: RankNet,
    matrix: np.ndarray,
    device: torch.device,
    batch_size: int = 16384,
) -> np.ndarray:
    output = np.empty(len(matrix), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(matrix), batch_size):
            end = min(start + batch_size, len(matrix))
            values = torch.from_numpy(matrix[start:end]).to(device)
            output[start:end] = model(values).detach().cpu().numpy().astype(np.float32)
    return output


def fit_and_predict(
    data: Mapping[str, SequenceData],
    training_sequences: Sequence[str],
    prediction_sequences: Sequence[str],
    seeds: Sequence[int],
    args,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    predictions = {
        seq: np.zeros(len(data[seq].matrix), dtype=np.float64)
        for seq in prediction_sequences
    }
    model_reports = []
    for seed in seeds:
        positive, negative, weights, pair_report = hard_negative_pairs(
            data,
            training_sequences,
            seed,
            args.conflict_negatives,
            args.global_negatives,
            args.random_negatives,
        )
        model = train_one_model(
            positive,
            negative,
            weights,
            positive.shape[1],
            args.hidden,
            args.dropout,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.weight_decay,
            seed,
            device,
        )
        for seq in prediction_sequences:
            predictions[seq] += predict_model(model, data[seq].matrix, device)
        model_reports.append(pair_report)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for seq in predictions:
        predictions[seq] = robust_rank(predictions[seq] / len(seeds))
    return predictions, {
        "training_sequences": list(training_sequences),
        "prediction_sequences": list(prediction_sequences),
        "seeds": list(seeds),
        "models": model_reports,
    }


def candidate_score(
    mode: str,
    ranknet_rank: np.ndarray,
    utility_rank: np.ndarray,
) -> np.ndarray:
    if mode == "ranknet":
        return ranknet_rank
    if mode == "blend75":
        return 0.75 * ranknet_rank + 0.25 * utility_rank
    if mode == "blend50":
        return 0.50 * ranknet_rank + 0.50 * utility_rank
    if mode == "product":
        return np.sqrt(np.maximum(ranknet_rank * utility_rank, 0.0))
    raise ValueError(f"unknown score mode: {mode}")


def candidate_id(mode: str, quantile: Optional[float]) -> str:
    if quantile is None:
        return "noop"
    return f"{mode}_q{quantile:.5f}".replace(".", "p")


def build_candidate_tracks(
    sequences: Sequence[str],
    data: Mapping[str, SequenceData],
    rank_predictions: Mapping[str, np.ndarray],
    mode: Optional[str],
    quantile: Optional[float],
    graph_root: Path,
    parent: Path,
    output_root: Path,
    m28,
    chain,
    evaluator,
) -> Tuple[Path, Dict[str, object]]:
    identifier = candidate_id(mode or "noop", quantile)
    candidate_root = output_root / "candidates" / identifier
    if candidate_root.exists():
        shutil.rmtree(candidate_root)
    track_root = candidate_root / "track_results"
    selected_root = candidate_root / "selected_transactions"
    track_root.mkdir(parents=True, exist_ok=True)
    selected_root.mkdir(parents=True, exist_ok=True)
    reports = {}
    for seq in sequences:
        inference = data[seq].inference.copy()
        if quantile is None:
            selected = inference.iloc[:0].copy()
            selected["policy_score"] = np.asarray([], dtype=float)
        else:
            score = candidate_score(
                mode or "ranknet", rank_predictions[seq], data[seq].utility_rank
            )
            inference["m23_33_score"] = score
            selected = m28.maximum_weight_matching(
                inference, "m23_33_score", float(quantile)
            )
        selected.to_parquet(selected_root / f"{seq}.parquet", index=False)
        meta = pd.read_parquet(graph_root / seq / "microtracklets.parquet")
        edges = pd.read_parquet(graph_root / seq / "candidate_edges.parquet")
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
        tracker_report = evaluator.write_tracker(
            seq, meta, applied, track_root / f"{seq}.txt"
        )
        reports[seq] = {
            "selected_actions": int(len(selected)),
            "selected_score_sum": float(selected.policy_score.sum()) if len(selected) else 0.0,
            **tracker_report,
        }
    return candidate_root, reports


def write_metrics(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-seq", required=True, choices=SEQUENCES)
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--parent", default=str(DEFAULT_PARENT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--hidden", default="96,48")
    parser.add_argument("--seeds", default="3301,3302,3303")
    parser.add_argument("--conflict-negatives", type=int, default=20)
    parser.add_argument("--global-negatives", type=int, default=8)
    parser.add_argument("--random-negatives", type=int, default=4)
    parser.add_argument("--score-modes", default="ranknet,blend75,blend50,product")
    parser.add_argument("--quantile-grid", default="0.9985,0.999,0.99925,0.9995")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.hidden = [int(value) for value in args.hidden.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    modes = [value.strip() for value in args.score_modes.split(",") if value.strip()]
    quantiles = parse_float_grid(args.quantile_grid)
    held = args.held_seq
    training_sequences = [seq for seq in SEQUENCES if seq != held]
    source_root = Path(args.source_root).resolve()
    parent = Path(args.parent).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )

    m26 = load_module("m23_33_m26", "scripts/m23_research/m23_26_prepare_test_submission.py")
    m27 = load_module("m23_33_m27", "scripts/m23_research/m23_27_oof_hota_policy_search.py")
    m28 = load_module("m23_33_m28", "scripts/m23_research/m23_28_structured_oracle_imitation_loso.py")
    base = load_module("m23_33_base", "scripts/m23_research/m23_12_train_chain_transaction_loso.py")
    chain = load_module("m23_33_chain", "scripts/m23_research/m23_12_chain_transaction_oracle.py")
    evaluator = load_module("m23_33_eval", "scripts/m23_research/m23_11_eval_utility_graph.py")

    graph_root = source_root / "train_oof_micrograph"
    prediction_root = source_root / "predictions"
    probe_frame = pd.read_parquet(prediction_root / f"oof_{SEQUENCES[0]}.parquet")
    probe_frame, stack_features = m28.add_stack_features(probe_frame)
    feature_names = unique_preserve(
        list(base.FEATURES) + list(m26.GRAPH_FEATURES) + list(stack_features)
    )
    data = {
        seq: prepare_sequence(seq, prediction_root, feature_names, m28)
        for seq in SEQUENCES
    }
    feature_report = {
        "feature_count": len(feature_names),
        "features": feature_names,
        "transform": "within-sequence empirical percentile mapped to [-1,1]",
        "oracle_counts": {seq: data[seq].oracle_count for seq in SEQUENCES},
    }
    (output_root / "feature_protocol.json").write_text(
        json.dumps(feature_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    inner_rank_predictions: Dict[str, np.ndarray] = {}
    inner_model_reports = []
    for inner_held in training_sequences:
        inner_training = [seq for seq in training_sequences if seq != inner_held]
        prediction, report = fit_and_predict(
            data,
            inner_training,
            [inner_held],
            seeds,
            args,
            device,
        )
        inner_rank_predictions[inner_held] = prediction[inner_held]
        report["inner_held"] = inner_held
        inner_model_reports.append(report)
        print(
            json.dumps(
                {
                    "stage": "inner_prediction_complete",
                    "inner_held": inner_held,
                    "training_sequences": inner_training,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    evaluator.DATA = graph_root
    evaluator.PARENT = parent
    evaluator.SEQS = list(training_sequences)
    inner_rows: List[Dict[str, object]] = []
    candidate_specs: List[Tuple[Optional[str], Optional[float]]] = [(None, None)]
    candidate_specs.extend((mode, quantile) for mode in modes for quantile in quantiles)
    for mode, quantile in candidate_specs:
        identifier = candidate_id(mode or "noop", quantile)
        candidate_root, sequence_reports = build_candidate_tracks(
            training_sequences,
            data,
            inner_rank_predictions,
            mode,
            quantile,
            graph_root,
            parent,
            output_root / "inner_selection",
            m28,
            chain,
            evaluator,
        )
        metrics = m27.evaluate_combined(
            candidate_root / "track_results",
            candidate_root,
            f"m23_33_inner_{identifier}",
            tuple(training_sequences),
        )
        row = {
            "candidate_id": identifier,
            "score_mode": "noop" if mode is None else mode,
            "score_quantile": "" if quantile is None else quantile,
            "selected_actions": int(
                sum(report["selected_actions"] for report in sequence_reports.values())
            ),
            **metrics,
        }
        inner_rows.append(row)
        (candidate_root / "report.json").write_text(
            json.dumps(
                {
                    "candidate": row,
                    "sequences": sequence_reports,
                    "protocol": "inner-held predictions; exact inner-training-sequence TrackEval model selection",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"stage": "inner_trackeval", **row}, sort_keys=True), flush=True)
    write_metrics(output_root / "inner_metrics.csv", inner_rows)
    best = max(
        inner_rows,
        key=lambda row: (
            float(row["HOTA"]),
            float(row["AssA"]),
            -int(row["selected_actions"]),
            str(row["candidate_id"]),
        ),
    )
    selected_mode = None if best["score_mode"] == "noop" else str(best["score_mode"])
    selected_quantile = None if selected_mode is None else float(best["score_quantile"])
    selection = {
        "status": "frozen_before_outer_held_prediction_and_trackeval",
        "outer_held_sequence": held,
        "inner_sequences": training_sequences,
        "outer_held_gt_used_in_training_or_model_selection": False,
        "selection_rule": "maximize inner exact COMBINED HOTA; tie AssA; tie fewer actions; tie candidate id",
        "selected": best,
        "candidates": inner_rows,
        "inner_model_reports": inner_model_reports,
        "feature_protocol": feature_report,
    }
    selection_path = output_root / "frozen_inner_selection.json"
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"stage": "policy_frozen", "selected": best}, sort_keys=True), flush=True)

    outer_predictions, outer_model_report = fit_and_predict(
        data,
        training_sequences,
        [held],
        seeds,
        args,
        device,
    )
    held_meta = pd.read_parquet(graph_root / held / "microtracklets.parquet")
    held_edges = pd.read_parquet(graph_root / held / "candidate_edges.parquet")
    held_gt_diagnostics = {
        "modal_gt_sum": int(held_meta.modal_gt.sum()),
        "same_gt_sum": int(held_edges.same_gt.sum()),
        "label_confidence_max": float(held_edges.label_confidence.max()),
    }
    if held_gt_diagnostics != {
        "modal_gt_sum": 0,
        "same_gt_sum": 0,
        "label_confidence_max": 0.0,
    }:
        raise RuntimeError(f"outer-held graph contains GT diagnostics: {held_gt_diagnostics}")
    evaluator.DATA = graph_root
    evaluator.PARENT = parent
    evaluator.SEQS = [held]
    outer_candidate_root, outer_sequence_reports = build_candidate_tracks(
        [held],
        data,
        outer_predictions,
        selected_mode,
        selected_quantile,
        graph_root,
        parent,
        output_root / "outer_final",
        m28,
        chain,
        evaluator,
    )
    outer_metrics = m27.evaluate_combined(
        outer_candidate_root / "track_results",
        outer_candidate_root,
        "m23_33_outer_final",
        (held,),
    )
    final_report = {
        "status": "completed",
        "experiment": "M23-33 strict nested-LOSO pairwise RankNet transaction scorer",
        "outer_held_sequence": held,
        "outer_training_sequences": training_sequences,
        "outer_held_gt_use": "final TrackEval only",
        "outer_held_gt_read_before_frozen_tracker": False,
        "candidate_feature_gt_use": "none",
        "training_gt_use": "outer-training structured-oracle transaction labels only",
        "frozen_selection_path": str(selection_path.relative_to(REPO)),
        "frozen_selection_sha256": sha256_file(selection_path),
        "selected_policy": best,
        "outer_model_report": outer_model_report,
        "held_gt_diagnostics": held_gt_diagnostics,
        "outer_sequence_report": outer_sequence_reports[held],
        "metrics": outer_metrics,
    }
    (output_root / "report.json").write_text(
        json.dumps(final_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
