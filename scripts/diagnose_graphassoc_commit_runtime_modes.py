#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.graph_assoc_commit_runtime import GraphAssocCommitScorer


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

META_FIELDS = [
    "run_root",
    "checkpoint",
    "device",
    "candidate_files",
    "variants",
    "status",
    "error",
]

SUMMARY_FIELDS = [
    "variant",
    "decision_mode",
    "threshold",
    "neutral_risk_weight",
    "positive_threshold",
    "neutral_threshold",
    "candidates",
    "accept_count",
    "accept_rate",
    "mean_decision_score",
    "mean_score_delta",
    "mean_positive_probability",
    "mean_neutral_probability",
    "mean_reject_probability",
    "mean_policy_score",
    "mean_selection_score",
    "mean_legacy_policy_score",
    "mean_gain_pred",
    "diff_accept_vs_policy_t000",
    "diff_accept_vs_recorded",
    "diff_mode_vs_recorded",
    "policy_score_mode",
    "recorded_decision_mode",
]

PER_SEQ_FIELDS = [
    "variant",
    "seq",
    "decision_mode",
    "threshold",
    "candidates",
    "accept_count",
    "accept_rate",
    "mean_decision_score",
    "mean_score_delta",
    "diff_accept_vs_policy_t000",
    "diff_accept_vs_recorded",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Diagnose graph-association commit runtime decision modes on dumped candidate rows.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate-jsonls", nargs="+", required=True)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / f"{ts}_graphassoc_commit_runtime_diag"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_single_row_csv(path: Path, fieldnames: Sequence[str], row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_rows_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_registry(args: argparse.Namespace, summary_csv: Path, status: str, notes: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(args.registry_csv),
        "--kind",
        "analysis",
        "--status",
        status,
        "--script",
        "scripts/diagnose_graphassoc_commit_runtime_modes.py",
        "--dataset",
        "MOT20",
        "--split",
        "runtime_diag",
        "--tracker-family",
        "BoT-SORT",
        "--variant",
        Path(args.out_dir).name,
        "--tag",
        Path(args.out_dir).name,
        "--run-root",
        str(Path(args.out_dir).resolve()),
        "--summary-csv",
        str(summary_csv.resolve()),
        "--checkpoint",
        str(Path(args.checkpoint).resolve()),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _seq_name_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_candidates"):
        return stem[: -len("_candidates")]
    return stem


def _load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def _build_variants() -> List[Dict[str, Any]]:
    return [
        {
            "variant": "policy_t000",
            "decision_mode": "policy_score",
            "threshold": 0.0,
            "neutral_risk_weight": "",
            "positive_threshold": "",
            "neutral_threshold": "",
            "scorer_kwargs": {"decision_mode": "policy_score", "threshold": 0.0},
        },
        {
            "variant": "policy_t003",
            "decision_mode": "policy_score",
            "threshold": 0.03,
            "neutral_risk_weight": "",
            "positive_threshold": "",
            "neutral_threshold": "",
            "scorer_kwargs": {"decision_mode": "policy_score", "threshold": 0.03},
        },
        {
            "variant": "posminus_t003",
            "decision_mode": "positive_minus_neutral",
            "threshold": 0.03,
            "neutral_risk_weight": 1.0,
            "positive_threshold": "",
            "neutral_threshold": "",
            "scorer_kwargs": {
                "decision_mode": "positive_minus_neutral",
                "threshold": 0.03,
                "neutral_risk_weight": 1.0,
            },
        },
        {
            "variant": "weighted_t003",
            "decision_mode": "positive_minus_weighted_neutral",
            "threshold": 0.03,
            "neutral_risk_weight": 1.0,
            "positive_threshold": "",
            "neutral_threshold": "",
            "scorer_kwargs": {
                "decision_mode": "positive_minus_weighted_neutral",
                "threshold": 0.03,
                "neutral_risk_weight": 1.0,
            },
        },
        {
            "variant": "dualthr_p040_n040",
            "decision_mode": "dual_threshold",
            "threshold": 0.0,
            "neutral_risk_weight": "",
            "positive_threshold": 0.40,
            "neutral_threshold": 0.40,
            "scorer_kwargs": {
                "decision_mode": "dual_threshold",
                "positive_threshold": 0.40,
                "neutral_threshold": 0.40,
            },
        },
    ]


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_csv = out_dir / "meta_summary.csv"
    summary_csv = out_dir / "summary.csv"
    per_seq_csv = out_dir / "per_sequence_summary.csv"

    candidate_paths = [Path(path).expanduser().resolve() for path in args.candidate_jsonls]
    variants = _build_variants()
    meta_row: Dict[str, object] = {
        "run_root": str(out_dir),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "device": str(args.device),
        "candidate_files": json.dumps([str(path) for path in candidate_paths], ensure_ascii=False),
        "variants": json.dumps(
            [{key: value for key, value in variant.items() if key != "scorer_kwargs"} for variant in variants],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "status": "running",
        "error": "",
    }
    write_single_row_csv(meta_csv, META_FIELDS, meta_row)
    write_rows_csv(summary_csv, SUMMARY_FIELDS, [])
    write_rows_csv(per_seq_csv, PER_SEQ_FIELDS, [])
    append_registry(args, summary_csv, "running", "diagnosing graph-association commit runtime mode divergence on dumped candidate rows")

    try:
        scorer_by_variant = {
            variant["variant"]: GraphAssocCommitScorer(
                str(Path(args.checkpoint).expanduser().resolve()),
                device=str(args.device),
                **variant["scorer_kwargs"],
            )
            for variant in variants
        }
        seq_rows = {
            _seq_name_from_path(path): _load_jsonl_rows(path)
            for path in candidate_paths
        }

        recorded_mode_counts: Dict[str, int] = defaultdict(int)
        baseline_accepts: Dict[tuple[str, int], bool] = {}
        recorded_accepts: Dict[tuple[str, int], bool] = {}
        result_cache: Dict[tuple[str, str, int], Dict[str, Any]] = {}

        for seq_name, rows in seq_rows.items():
            for row_idx, row in enumerate(rows):
                recorded_mode = str(row.get("learned_commit_decision_mode", "") or "")
                if recorded_mode:
                    recorded_mode_counts[recorded_mode] += 1
                recorded_accepts[(seq_name, row_idx)] = bool(row.get("learned_commit_score_margin_pass", False))

        summary_rows: List[Dict[str, object]] = []
        per_seq_rows: List[Dict[str, object]] = []

        for variant in variants:
            variant_name = str(variant["variant"])
            scorer = scorer_by_variant[variant_name]
            variant_scores: List[float] = []
            variant_deltas: List[float] = []
            variant_accepts = 0
            variant_pos: List[float] = []
            variant_neu: List[float] = []
            variant_rej: List[float] = []
            variant_policy: List[float] = []
            variant_selection: List[float] = []
            variant_legacy: List[float] = []
            variant_gain: List[float] = []
            diff_accept_vs_recorded = 0
            diff_mode_vs_recorded = 0
            policy_score_mode = ""

            for seq_name, rows in seq_rows.items():
                seq_scores: List[float] = []
                seq_deltas: List[float] = []
                seq_accepts = 0
                diff_accept_vs_policy = 0
                diff_accept_vs_recorded_seq = 0
                for row_idx, row in enumerate(rows):
                    score = scorer.score_candidate_row(row)
                    result_cache[(variant_name, seq_name, row_idx)] = {
                        "accept": bool(score.accept),
                        "decision_score": float(score.decision_score),
                        "score_delta": float(score.score_delta),
                    }
                    if variant_name == "policy_t000":
                        baseline_accepts[(seq_name, row_idx)] = bool(score.accept)
                    policy_score_mode = str(score.policy_score_mode or policy_score_mode)
                    seq_scores.append(float(score.decision_score))
                    seq_deltas.append(float(score.score_delta))
                    variant_scores.append(float(score.decision_score))
                    variant_deltas.append(float(score.score_delta))
                    variant_pos.append(float(score.positive_probability))
                    variant_neu.append(float(score.neutral_probability))
                    variant_rej.append(float(score.reject_probability))
                    variant_policy.append(float(score.policy_score))
                    variant_selection.append(float(score.selection_score))
                    variant_legacy.append(float(score.legacy_policy_score))
                    variant_gain.append(float(score.gain_pred))
                    if bool(score.accept):
                        seq_accepts += 1
                        variant_accepts += 1
                    recorded_accept = recorded_accepts[(seq_name, row_idx)]
                    if bool(score.accept) != recorded_accept:
                        diff_accept_vs_recorded += 1
                        diff_accept_vs_recorded_seq += 1
                    recorded_mode = str(row.get("learned_commit_decision_mode", "") or "")
                    if recorded_mode and str(score.decision_mode) != recorded_mode:
                        diff_mode_vs_recorded += 1
                per_seq_rows.append(
                    {
                        "variant": variant_name,
                        "seq": seq_name,
                        "decision_mode": str(variant["decision_mode"]),
                        "threshold": variant["threshold"],
                        "candidates": len(rows),
                        "accept_count": seq_accepts,
                        "accept_rate": float(seq_accepts) / float(len(rows)) if rows else 0.0,
                        "mean_decision_score": _mean(seq_scores),
                        "mean_score_delta": _mean(seq_deltas),
                        "diff_accept_vs_policy_t000": diff_accept_vs_policy,
                        "diff_accept_vs_recorded": diff_accept_vs_recorded_seq,
                    }
                )

            diff_accept_vs_policy_total = 0
            if variant_name != "policy_t000":
                for seq_name, rows in seq_rows.items():
                    for row_idx, _row in enumerate(rows):
                        current_accept = bool(result_cache[(variant_name, seq_name, row_idx)]["accept"])
                        baseline_accept = bool(baseline_accepts[(seq_name, row_idx)])
                        if current_accept != baseline_accept:
                            diff_accept_vs_policy_total += 1
                for seq_row in per_seq_rows:
                    if str(seq_row["variant"]) != variant_name:
                        continue
                    seq_name = str(seq_row["seq"])
                    diff_count = 0
                    for row_idx, _row in enumerate(seq_rows[seq_name]):
                        current_accept = bool(result_cache[(variant_name, seq_name, row_idx)]["accept"])
                        baseline_accept = bool(baseline_accepts[(seq_name, row_idx)])
                        if current_accept != baseline_accept:
                            diff_count += 1
                    seq_row["diff_accept_vs_policy_t000"] = diff_count

            summary_rows.append(
                {
                    "variant": variant_name,
                    "decision_mode": str(variant["decision_mode"]),
                    "threshold": variant["threshold"],
                    "neutral_risk_weight": variant["neutral_risk_weight"],
                    "positive_threshold": variant["positive_threshold"],
                    "neutral_threshold": variant["neutral_threshold"],
                    "candidates": sum(len(rows) for rows in seq_rows.values()),
                    "accept_count": variant_accepts,
                    "accept_rate": (
                        float(variant_accepts) / float(sum(len(rows) for rows in seq_rows.values()))
                        if seq_rows
                        else 0.0
                    ),
                    "mean_decision_score": _mean(variant_scores),
                    "mean_score_delta": _mean(variant_deltas),
                    "mean_positive_probability": _mean(variant_pos),
                    "mean_neutral_probability": _mean(variant_neu),
                    "mean_reject_probability": _mean(variant_rej),
                    "mean_policy_score": _mean(variant_policy),
                    "mean_selection_score": _mean(variant_selection),
                    "mean_legacy_policy_score": _mean(variant_legacy),
                    "mean_gain_pred": _mean(variant_gain),
                    "diff_accept_vs_policy_t000": diff_accept_vs_policy_total,
                    "diff_accept_vs_recorded": diff_accept_vs_recorded,
                    "diff_mode_vs_recorded": diff_mode_vs_recorded,
                    "policy_score_mode": policy_score_mode,
                    "recorded_decision_mode": max(recorded_mode_counts, key=recorded_mode_counts.get) if recorded_mode_counts else "",
                }
            )

        write_rows_csv(summary_csv, SUMMARY_FIELDS, summary_rows)
        write_rows_csv(per_seq_csv, PER_SEQ_FIELDS, per_seq_rows)

        meta_row["status"] = "success"
        write_single_row_csv(meta_csv, META_FIELDS, meta_row)
        append_registry(
            args,
            summary_csv,
            "success",
            "runtime decision modes diverged after fixing commit scorer decision surface, threshold wiring, and checkpoint policy_score_mode restore",
        )
        return 0
    except Exception as exc:
        meta_row["status"] = "failed"
        meta_row["error"] = str(exc)
        write_single_row_csv(meta_csv, META_FIELDS, meta_row)
        append_registry(args, summary_csv, "failed", f"runtime diagnosis failed: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
