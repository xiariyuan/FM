#!/usr/bin/env python3
from __future__ import annotations

"""Search a deployable transaction policy with out-of-fold TrackEval calibration.

This stage reuses the strict outer-held graphs and OOF transaction predictions
created by M23-26.  Every sequence prediction is produced by a model that did
not train on that sequence.  MOT20-train GT is used only by TrackEval to choose
one global deployment policy for the hidden MOT20 test set.

The resulting OOF HOTA is therefore a *model-selection score*, not an unbiased
strict validation estimate.  Test inference remains fully GT-free.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
SOURCE_ROOT = (
    REPO
    / "outputs"
    / "mot20_m23_20260718"
    / "m23_26_test_deploy_oof_ensemble_v1"
)
DEFAULT_OUTPUT_ROOT = (
    REPO
    / "outputs"
    / "mot20_m23_20260718"
    / "m23_27_oof_hota_policy_search_v1"
)
TRAIN_PARENT = (
    REPO
    / "outputs"
    / "assocriskbench_p15_20260714"
    / "fused_a42_adaptive_risk_eval"
    / "l1_r2_q75_adaptive_priority"
    / "track_results"
)
TARGET = "chain_transaction_delta_proxy"
POLICY_FIELDS = ("loss_multiplier", "min_probability", "score_quantile")
METRIC_FIELDS = (
    "candidate_id",
    "status",
    "HOTA",
    "DetA",
    "AssA",
    "IDSW",
    "delta_vs_parent",
    "delta_vs_m23_25_strict",
    "loss_multiplier",
    "min_probability",
    "score_quantile",
    "selected_actions",
    "selected_MOT20_01",
    "selected_MOT20_02",
    "selected_MOT20_03",
    "selected_MOT20_05",
    "proxy_sum",
    "proxy_worst_sequence",
    "source_reason",
    "message",
)
PARENT_HOTA = 78.763
M23_25_STRICT_HOTA = 79.02501


def load_module(name: str, relative_path: str):
    path = REPO / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def policy_tuple(report: Dict[str, object]) -> Optional[Tuple[float, float, float]]:
    if report.get("loss_multiplier") is None:
        return None
    return tuple(float(report[field]) for field in POLICY_FIELDS)  # type: ignore[return-value]


def policy_id(policy: Optional[Tuple[float, float, float]]) -> str:
    if policy is None:
        return "noop"
    loss, probability, quantile = policy
    raw = f"L{loss:g}_P{probability:.4f}_Q{quantile:.5f}"
    return raw.replace(".", "p")


def selection_signature(selected_by_sequence: Dict[str, pd.DataFrame]) -> str:
    payload = []
    for seq in SEQUENCES:
        frame = selected_by_sequence[seq]
        rows = sorted(
            (
                int(row.transaction_src_track_id),
                int(row.transaction_dst_track_id),
                int(row.src_chunk),
                int(row.dst_chunk),
            )
            for _, row in frame.iterrows()
        )
        payload.append((seq, rows))
    return sha256_text(json.dumps(payload, separators=(",", ":")))


def select_transactions(
    m26,
    predictions: Dict[str, pd.DataFrame],
    policy: Optional[Tuple[float, float, float]],
) -> Dict[str, pd.DataFrame]:
    selected: Dict[str, pd.DataFrame] = {}
    for seq, frame in predictions.items():
        if policy is None:
            current = frame.iloc[:0].copy()
            current["policy_score"] = np.asarray([], dtype=float)
        else:
            current = m26.maximum_weight_transaction_matching(frame, *policy)
        selected[seq] = current
    return selected


def candidate_reason_map(
    reports: Sequence[Dict[str, object]],
    max_candidates: int,
) -> Dict[Optional[Tuple[float, float, float]], List[str]]:
    """Choose a diverse fast-pass subset while supporting a complete search."""

    reasons: Dict[Optional[Tuple[float, float, float]], List[str]] = {None: ["no-op baseline"]}

    def add(report: Dict[str, object], reason: str) -> None:
        policy = policy_tuple(report)
        reasons.setdefault(policy, []).append(reason)

    non_noop = [report for report in reports if policy_tuple(report) is not None]
    if max_candidates <= 0 or max_candidates >= len(non_noop) + 1:
        for report in non_noop:
            add(report, "full grid")
        return reasons

    # Frozen policies already used by M23-25/M23-26.
    frozen = {
        (16.0, 0.9, 0.999): "M23-26 frozen",
        (4.0, 0.9, 0.9975): "M23-25 M01",
        (1.0, 0.5, 0.999): "M23-25 M02",
        (4.0, 0.5, 0.9995): "M23-25 M05",
    }
    by_policy = {policy_tuple(report): report for report in non_noop}
    for policy, reason in frozen.items():
        if policy in by_policy:
            add(by_policy[policy], reason)

    ranked_views: List[Tuple[str, List[Dict[str, object]]]] = []
    ranked_views.append(
        (
            "proxy total",
            sorted(non_noop, key=lambda r: float(r["sum_normalized_utility"]), reverse=True),
        )
    )
    ranked_views.append(
        (
            "proxy worst",
            sorted(non_noop, key=lambda r: float(r["worst_normalized_utility"]), reverse=True),
        )
    )
    ranked_views.append(
        (
            "positive sequence coverage",
            sorted(
                non_noop,
                key=lambda r: (
                    int(r["positive_sequences"]),
                    float(r["sum_normalized_utility"]),
                ),
                reverse=True,
            ),
        )
    )
    for seq in SEQUENCES:
        ranked_views.append(
            (
                f"{seq} proxy",
                sorted(
                    non_noop,
                    key=lambda r: next(
                        float(item["actual_proxy_sum"])
                        for item in r["by_sequence"]  # type: ignore[index]
                        if item["seq"] == seq
                    ),
                    reverse=True,
                ),
            )
        )

    cursor = 0
    while len(reasons) < max_candidates:
        added = False
        for name, ranked in ranked_views:
            if cursor < len(ranked):
                before = len(reasons)
                add(ranked[cursor], f"top {cursor + 1} {name}")
                added |= len(reasons) > before
                if len(reasons) >= max_candidates:
                    break
        if not added and cursor >= max(len(ranked) for _, ranked in ranked_views):
            break
        cursor += 1

    # Add action-count quantiles if room remains.
    if len(reasons) < max_candidates:
        ordered = sorted(non_noop, key=lambda r: int(r["actions"]))
        for fraction in np.linspace(0.0, 1.0, num=min(max_candidates, 11)):
            report = ordered[int(round(fraction * (len(ordered) - 1)))]
            add(report, f"action-count quantile {fraction:.2f}")
            if len(reasons) >= max_candidates:
                break
    return reasons


def parse_combined_metrics(detailed_path: Path) -> Dict[str, float]:
    with detailed_path.open(encoding="utf-8") as handle:
        row = next(item for item in csv.DictReader(handle) if item["seq"] == "COMBINED")
    return {
        "HOTA": 100.0 * float(row["HOTA___AUC"]),
        "DetA": 100.0 * float(row["DetA___AUC"]),
        "AssA": 100.0 * float(row["AssA___AUC"]),
        "IDSW": int(float(row["IDSW"])),
    }


def evaluate_combined(track_results: Path, candidate_root: Path, tracker_name: str) -> Dict[str, float]:
    work_dir = candidate_root / "eval_work"
    command = [
        sys.executable,
        "scripts/eval_motstyle_trackeval.py",
        "--benchmark-name",
        "MOT20",
        "--split-to-eval",
        "train",
        "--gt-root",
        "datasets/MOT20/train",
        "--results-dir",
        str(track_results),
        "--tracker-name",
        tracker_name,
        "--work-dir",
        str(work_dir),
        "--seqs",
        *SEQUENCES,
    ]
    completed = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (candidate_root / "eval.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stdout[-5000:])
    detailed = work_dir / "eval" / tracker_name / "pedestrian_detailed.csv"
    return parse_combined_metrics(detailed)


def load_existing_rows(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {row["candidate_id"]: row for row in csv.DictReader(handle)}


def write_rows(path: Path, rows: Dict[str, Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for candidate_id in sorted(rows):
            writer.writerow({field: rows[candidate_id].get(field, "") for field in METRIC_FIELDS})


def build_candidate_tracks(
    m26,
    chain,
    evaluator,
    graph_root: Path,
    predictions: Dict[str, pd.DataFrame],
    selected_by_sequence: Dict[str, pd.DataFrame],
    candidate_root: Path,
) -> Dict[str, Dict[str, object]]:
    track_root = candidate_root / "track_results"
    selected_root = candidate_root / "selected_transactions"
    track_root.mkdir(parents=True, exist_ok=True)
    selected_root.mkdir(parents=True, exist_ok=True)
    reports: Dict[str, Dict[str, object]] = {}
    for seq in SEQUENCES:
        meta = pd.read_parquet(graph_root / seq / "microtracklets.parquet")
        edges = pd.read_parquet(graph_root / seq / "candidate_edges.parquet")
        selected = selected_by_sequence[seq].copy()
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
        tracker_report = evaluator.write_tracker(
            seq,
            meta,
            applied,
            track_root / f"{seq}.txt",
        )
        reports[seq] = {
            "selected_actions": int(len(selected)),
            "selected_proxy_sum": float(selected[TARGET].sum()) if len(selected) else 0.0,
            "selected_score_sum": float(selected.policy_score.sum()) if len(selected) else 0.0,
            **tracker_report,
        }
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=24,
        help="0 evaluates the full policy grid; positive values run a diverse fast pass",
    )
    parser.add_argument(
        "--candidate-ids",
        default="",
        help=(
            "optional comma-separated policy IDs from the selected candidate set; "
            "use this to run disjoint parallel shards"
        ),
    )
    parser.add_argument("--keep-candidate-tracks", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "metrics.csv"
    report_path = output_root / "report.json"

    m26 = load_module("m23_27_m26", "scripts/m23_research/m23_26_prepare_test_submission.py")
    chain = load_module("m23_27_chain", "scripts/m23_research/m23_12_chain_transaction_oracle.py")
    evaluator = load_module("m23_27_evaluator", "scripts/m23_research/m23_11_eval_utility_graph.py")

    graph_root = source_root / "train_oof_micrograph"
    prediction_root = source_root / "predictions"
    calibration_path = source_root / "deployment_calibration.json"
    for path in (graph_root, prediction_root, calibration_path, TRAIN_PARENT):
        if not path.exists():
            raise FileNotFoundError(path)

    predictions = {
        seq: pd.read_parquet(prediction_root / f"oof_{seq}.parquet")
        for seq in SEQUENCES
    }
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    reports: List[Dict[str, object]] = calibration["candidates"]
    reasons = candidate_reason_map(reports, args.max_candidates)

    evaluator.DATA = graph_root
    evaluator.PARENT = TRAIN_PARENT
    evaluator.SEQS = list(SEQUENCES)

    existing = {} if args.overwrite else load_existing_rows(metrics_path)
    metric_rows: Dict[str, Dict[str, object]] = dict(existing)
    seen_signatures: Dict[str, str] = {}
    candidate_manifest = []

    ordered_policies = sorted(
        reasons,
        key=lambda policy: (
            policy is not None,
            policy if policy is not None else (0.0, 0.0, 0.0),
        ),
    )
    requested_ids = {
        value.strip() for value in args.candidate_ids.split(",") if value.strip()
    }
    if requested_ids:
        available = {policy_id(policy): policy for policy in ordered_policies}
        unknown = sorted(requested_ids.difference(available))
        if unknown:
            raise ValueError(
                f"unknown candidate IDs for max-candidates={args.max_candidates}: {unknown}"
            )
        ordered_policies = [
            policy for policy in ordered_policies if policy_id(policy) in requested_ids
        ]
    for policy in ordered_policies:
        selected = select_transactions(m26, predictions, policy)
        signature = selection_signature(selected)
        candidate_id = policy_id(policy)
        if signature in seen_signatures:
            candidate_manifest.append(
                {
                    "candidate_id": candidate_id,
                    "policy": policy,
                    "status": "duplicate-selection",
                    "duplicate_of": seen_signatures[signature],
                    "reasons": reasons[policy],
                }
            )
            continue
        seen_signatures[signature] = candidate_id
        counts = {seq: int(len(selected[seq])) for seq in SEQUENCES}
        proxy_sums = {
            seq: float(selected[seq][TARGET].sum()) if len(selected[seq]) else 0.0
            for seq in SEQUENCES
        }
        candidate_manifest.append(
            {
                "candidate_id": candidate_id,
                "policy": policy,
                "selection_signature": signature,
                "counts": counts,
                "proxy_sums": proxy_sums,
                "reasons": reasons[policy],
            }
        )
        if candidate_id in existing and existing[candidate_id].get("status") == "success":
            print(f"skip completed {candidate_id}", flush=True)
            continue

        candidate_root = output_root / "candidates" / candidate_id
        if candidate_root.exists() and args.overwrite:
            shutil.rmtree(candidate_root)
        candidate_root.mkdir(parents=True, exist_ok=True)
        row: Dict[str, object] = {
            "candidate_id": candidate_id,
            "status": "running",
            "loss_multiplier": "" if policy is None else policy[0],
            "min_probability": "" if policy is None else policy[1],
            "score_quantile": "" if policy is None else policy[2],
            "selected_actions": sum(counts.values()),
            "selected_MOT20_01": counts["MOT20-01"],
            "selected_MOT20_02": counts["MOT20-02"],
            "selected_MOT20_03": counts["MOT20-03"],
            "selected_MOT20_05": counts["MOT20-05"],
            "proxy_sum": sum(proxy_sums.values()),
            "proxy_worst_sequence": min(proxy_sums.values()),
            "source_reason": "; ".join(sorted(set(reasons[policy]))),
            "message": "",
        }
        metric_rows[candidate_id] = row
        write_rows(metrics_path, metric_rows)
        print(
            f"evaluate {candidate_id}: counts={counts} proxy={proxy_sums}",
            flush=True,
        )
        try:
            sequence_reports = build_candidate_tracks(
                m26,
                chain,
                evaluator,
                graph_root,
                predictions,
                selected,
                candidate_root,
            )
            tracker_name = f"m23_27_{candidate_id}"
            metrics = evaluate_combined(
                candidate_root / "track_results",
                candidate_root,
                tracker_name,
            )
            row.update(
                {
                    "status": "success",
                    **metrics,
                    "delta_vs_parent": metrics["HOTA"] - PARENT_HOTA,
                    "delta_vs_m23_25_strict": metrics["HOTA"] - M23_25_STRICT_HOTA,
                    "message": "OOF TrackEval model-selection score",
                }
            )
            (candidate_root / "report.json").write_text(
                json.dumps(
                    {
                        "candidate_id": candidate_id,
                        "policy": policy,
                        "selection_signature": signature,
                        "sequence_reports": sequence_reports,
                        "metrics": metrics,
                        "protocol": "OOF predictions; train GT used only by TrackEval model selection",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"completed {candidate_id}: {metrics}", flush=True)
            if not args.keep_candidate_tracks:
                shutil.rmtree(candidate_root / "track_results")
                shutil.rmtree(candidate_root / "eval_work")
        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"failed {candidate_id}: {exc}", flush=True)
        metric_rows[candidate_id] = row
        write_rows(metrics_path, metric_rows)

    successful = [
        row for row in metric_rows.values() if row.get("status") == "success" and row.get("HOTA") not in ("", None)
    ]
    best = max(successful, key=lambda row: float(row["HOTA"])) if successful else None
    final_report = {
        "status": "completed" if successful else "failed",
        "protocol": (
            "strict OOF predictions with MOT20-train TrackEval policy calibration; "
            "score is for final test model selection, not unbiased validation"
        ),
        "test_inference_gt_free": True,
        "source_root": str(source_root.relative_to(REPO)),
        "output_root": str(output_root.relative_to(REPO)),
        "requested_max_candidates": args.max_candidates,
        "requested_candidate_ids": sorted(requested_ids),
        "unique_candidates_considered": len(seen_signatures),
        "completed_candidates": len(successful),
        "best": best,
        "candidate_manifest": candidate_manifest,
    }
    report_path.write_text(
        json.dumps(final_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
