#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")


VAL_RE = re.compile(
    r"\[INFO\] \[Val\] HOTA=(?P<HOTA>-?\d+(?:\.\d+)?) "
    r"DetA=(?P<DetA>-?\d+(?:\.\d+)?) "
    r"AssA=(?P<AssA>-?\d+(?:\.\d+)?) "
    r"IDF1=(?P<IDF1>-?\d+(?:\.\d+)?) "
    r"MOTA=(?P<MOTA>-?\d+(?:\.\d+)?) "
    r"IDSW=(?P<IDSW>-?\d+(?:\.\d+)?) "
    r"Frag=(?P<Frag>-?\d+(?:\.\d+)?)"
)


@dataclass
class ValMetric:
    epoch: int
    hota: float
    deta: float
    assa: float
    idf1: float
    mota: float
    idsw: float
    frag: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit legacy frequency/laplace module families.")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Default: outputs/legacy_module_forensic_audit_<timestamp>",
    )
    return parser.parse_args()


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_csv(path: Path, rows: list[dict], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_val_metrics_from_log(path: Path) -> list[ValMetric]:
    metrics: list[ValMetric] = []
    current_epoch: int | None = None
    for line in read_text(path).splitlines():
        if "Validation finished for epoch " in line:
            try:
                current_epoch = int(line.split("Validation finished for epoch ", 1)[1].split(".", 1)[0])
            except Exception:
                current_epoch = None
            continue
        match = VAL_RE.search(line)
        if match is None or current_epoch is None:
            continue
        metrics.append(
            ValMetric(
                epoch=current_epoch,
                hota=float(match.group("HOTA")),
                deta=float(match.group("DetA")),
                assa=float(match.group("AssA")),
                idf1=float(match.group("IDF1")),
                mota=float(match.group("MOTA")),
                idsw=float(match.group("IDSW")),
                frag=float(match.group("Frag")),
            )
        )
    return metrics


def parse_summary_txt(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        values = f.readline().strip().split()
    return {k: float(v) for k, v in zip(header, values)}


def count_nan_lines(path: Path) -> int:
    return sum(1 for line in read_text(path).splitlines() if " nan " in f" {line.lower()} " or "=(nan)" in line.lower())


def pair_bucket_label(value: float) -> str:
    buckets = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for left, right in buckets:
        if left <= value < right:
            return f"{left:.1f}-{min(right, 1.0):.1f}"
    return "unknown"


def summarize_pair_log(path: Path) -> tuple[dict[str, float], list[dict[str, str]]]:
    valid_rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["track_gt_id"]) <= 0:
                continue
            valid_rows.append(row)

    total = len(valid_rows)
    chosen_rows = [row for row in valid_rows if int(row["chosen"]) == 1]
    chosen_true = [row for row in chosen_rows if int(row["is_true_match"]) == 1]

    bucket_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in valid_rows:
        bucket_rows[pair_bucket_label(float(row["pair_rel"]))].append(row)

    bucket_summary: list[dict[str, str]] = []
    for bucket in sorted(bucket_rows.keys()):
        rows = bucket_rows[bucket]
        chosen = [row for row in rows if int(row["chosen"]) == 1]
        chosen_true_rate = (
            sum(int(row["is_true_match"]) for row in chosen) / len(chosen) if chosen else 0.0
        )
        bucket_summary.append(
            {
                "sequence": path.stem.replace("_pairs", ""),
                "bucket": bucket,
                "pairs": str(len(rows)),
                "chosen_pairs": str(len(chosen)),
                "true_rate": f"{sum(int(row['is_true_match']) for row in rows) / len(rows):.6f}",
                "chosen_true_rate": f"{chosen_true_rate:.6f}",
                "avg_pair_rel": f"{sum(float(row['pair_rel']) for row in rows) / len(rows):.6f}",
                "avg_motion_sim": f"{sum(float(row['motion_sim']) for row in rows) / len(rows):.6f}",
                "avg_appearance_sim": f"{sum(float(row['appearance_sim']) for row in rows) / len(rows):.6f}",
            }
        )

    high_bucket_ctr = 0.0
    for row in bucket_summary:
        if row["bucket"] == "0.9-1.0":
            high_bucket_ctr = float(row["chosen_true_rate"])
            break

    overall = {
        "sequence": path.stem.replace("_pairs", ""),
        "valid_pairs": total,
        "chosen_rate": len(chosen_rows) / total if total else 0.0,
        "true_rate": sum(int(row["is_true_match"]) for row in valid_rows) / total if total else 0.0,
        "chosen_true_rate": len(chosen_true) / len(chosen_rows) if chosen_rows else 0.0,
        "avg_pair_rel": sum(float(row["pair_rel"]) for row in valid_rows) / total if total else 0.0,
        "high_rel_chosen_true_rate": high_bucket_ctr,
    }
    return overall, bucket_summary


def append_registry(row: dict[str, str]) -> None:
    registry_path = REPO_ROOT / "outputs" / "experiment_registry.csv"
    with registry_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
    merged = {name: "" for name in fieldnames}
    merged.update(row)
    with registry_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(merged)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "outputs" / f"legacy_module_forensic_audit_{now_ts()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    freq_main_log = REPO_ROOT / "outputs/bytetrack_fa_mot_mot17_v13_tf_only_val0213/train/log.txt"
    freq_fixnan_v4_log = REPO_ROOT / "outputs/bytetrack_fa_mot_mot17_v13_tf_only_val0213_smoke_fixnan_v4/train/log.txt"
    freq_sfi_mot20_log = REPO_ROOT / "outputs/bytetrack_fa_mot_mot20_v14_assoc_only_val05_sfi_20260305_091550/train/log.txt"
    freq_smoke_dirs = sorted((REPO_ROOT / "outputs").glob("bytetrack_fa_mot_mot17_v13_tf_only_val0213_smoke_fixnan*"))

    laplace_base_summary = REPO_ROOT / "outputs/v15_laplace_proxy0213/base/tracker/MOT17-train/pedestrian_summary.txt"
    laplace_full_summary = REPO_ROOT / "outputs/v15_laplace_proxy0213/laplace/tracker/MOT17-train/pedestrian_summary.txt"
    laplace_v15_train_log = REPO_ROOT / "outputs/bytetrack_fa_mot_mot17_v15_laplace_proxy0213_20260307_001459/train/log.txt"
    laplace_v16_log = REPO_ROOT / "outputs/bytetrack_fa_mot_mot17_v16_laplace_gate_proxy0213_20260307_113345/train/log.txt"
    pair_02 = REPO_ROOT / "outputs/laplace_pair_logs/MOT17_full/MOT17-02-FRCNN_pairs.csv"
    pair_13 = REPO_ROOT / "outputs/laplace_pair_logs/MOT17_full/MOT17-13-FRCNN_pairs.csv"

    family_rows: list[dict[str, str]] = []

    freq_main_text = read_text(freq_main_log)
    freq_main_nan_lines = count_nan_lines(freq_main_log)
    freq_main_tracker_root_missing = "tracker root not found" in freq_main_text.lower()
    freq_main_val_metrics = parse_val_metrics_from_log(freq_main_log)
    family_rows.append(
        {
            "family": "frequency",
            "run_name": "v13_tf_plus_lowfreq_mamba_main",
            "status": "unstable_nan",
            "dataset_scope": "MOT17 proxy0213-trainline",
            "evidence_path": str(freq_main_log),
            "key_metrics": f"nan_lines={freq_main_nan_lines}; val_epochs={len(freq_main_val_metrics)}; tracker_root_missing={int(freq_main_tracker_root_missing)}",
            "failure_mode": "optimization instability and broken validation bookkeeping",
            "diagnosis": "Epoch 1 develops NaNs across id/match/frequency losses. Epoch 0 validation finishes but tracker root is missing, so the run never produces a usable paired validation record.",
        }
    )

    freq_fixnan_metrics = parse_val_metrics_from_log(freq_fixnan_v4_log)
    best_freq_fixnan = max(freq_fixnan_metrics, key=lambda item: item.hota)
    family_rows.append(
        {
            "family": "frequency",
            "run_name": "v13_tf_plus_lowfreq_mamba_fixnan_v4",
            "status": "stable_but_collapsed",
            "dataset_scope": "MOT17 proxy0213",
            "evidence_path": str(freq_fixnan_v4_log),
            "key_metrics": (
                f"best_epoch={best_freq_fixnan.epoch}; HOTA={best_freq_fixnan.hota:.2f}; "
                f"AssA={best_freq_fixnan.assa:.2f}; IDF1={best_freq_fixnan.idf1:.2f}; "
                f"MOTA={best_freq_fixnan.mota:.2f}; IDSW={best_freq_fixnan.idsw:.0f}"
            ),
            "failure_mode": "identity semantics collapse after numerical stabilization",
            "diagnosis": "Repeated fixnan attempts stop the NaNs, but the stabilized run collapses to extremely low association quality and massive switch count, so the family is not rescued by simple numerical fixes.",
        }
    )

    freq_sfi_metrics = parse_val_metrics_from_log(freq_sfi_mot20_log)
    sfi_metric = freq_sfi_metrics[-1]
    family_rows.append(
        {
            "family": "frequency",
            "run_name": "v14_tf_sfi_mot20_val05",
            "status": "stable_but_off_carrier",
            "dataset_scope": "MOT20 val05",
            "evidence_path": str(freq_sfi_mot20_log),
            "key_metrics": (
                f"epoch={sfi_metric.epoch}; HOTA={sfi_metric.hota:.2f}; AssA={sfi_metric.assa:.2f}; "
                f"IDF1={sfi_metric.idf1:.2f}; IDSW={sfi_metric.idsw:.0f}"
            ),
            "failure_mode": "evidence gap on the canonical carrier",
            "diagnosis": "The conservative SFI-lite branch is numerically stable on MOT20 val05, but it was never carried through a clean paired MOT17 canonical-baseline audit, so it cannot answer the current paper question.",
        }
    )

    base_metrics = parse_summary_txt(laplace_base_summary)
    laplace_metrics = parse_summary_txt(laplace_full_summary)
    laplace_deltas = {k: laplace_metrics[k] - base_metrics[k] for k in ["HOTA", "DetA", "AssA", "MOTA", "IDSW", "IDF1"]}
    family_rows.append(
        {
            "family": "laplace",
            "run_name": "v15_laplace_proxy0213_heuristic",
            "status": "effective_proxy_positive",
            "dataset_scope": "MOT17 proxy0213",
            "evidence_path": str(laplace_full_summary),
            "key_metrics": (
                f"delta_HOTA={laplace_deltas['HOTA']:.3f}; delta_AssA={laplace_deltas['AssA']:.3f}; "
                f"delta_IDF1={laplace_deltas['IDF1']:.3f}; delta_MOTA={laplace_deltas['MOTA']:.3f}; "
                f"delta_IDSW={laplace_deltas['IDSW']:.0f}"
            ),
            "failure_mode": "not globally invalid; useful but slice-dependent",
            "diagnosis": "The fixed Laplace reliability branch improves proxy0213 on every main metric and reduces ID switches. This family had a valid positive regime before the current official-host work.",
        }
    )

    v16_metrics = parse_val_metrics_from_log(laplace_v16_log)
    best_v16 = max(v16_metrics, key=lambda item: item.hota)
    last_v16 = v16_metrics[-1]
    family_rows.append(
        {
            "family": "laplace",
            "run_name": "v16_laplace_trainable_gate",
            "status": "trained_but_regressed",
            "dataset_scope": "MOT17 proxy0213",
            "evidence_path": str(laplace_v16_log),
            "key_metrics": (
                f"best_epoch={best_v16.epoch}; best_HOTA={best_v16.hota:.2f}; best_AssA={best_v16.assa:.2f}; "
                f"last_epoch={last_v16.epoch}; last_HOTA={last_v16.hota:.2f}; last_AssA={last_v16.assa:.2f}; "
                f"last_IDSW={last_v16.idsw:.0f}"
            ),
            "failure_mode": "trainable supervision degrades the reliable heuristic branch",
            "diagnosis": "The trainable Laplace gate does not explode numerically, but validation degrades after epoch 0 and never establishes a positive regime. The learned objective appears to hurt reliability calibration instead of helping it.",
        }
    )

    pair_overall_rows: list[dict[str, str]] = []
    pair_bucket_rows: list[dict[str, str]] = []
    for pair_path in [pair_02, pair_13]:
        overall, buckets = summarize_pair_log(pair_path)
        pair_overall_rows.append(
            {
                "sequence": overall["sequence"],
                "valid_pairs": str(overall["valid_pairs"]),
                "chosen_rate": f"{overall['chosen_rate']:.6f}",
                "true_rate": f"{overall['true_rate']:.6f}",
                "chosen_true_rate": f"{overall['chosen_true_rate']:.6f}",
                "avg_pair_rel": f"{overall['avg_pair_rel']:.6f}",
                "high_rel_chosen_true_rate": f"{overall['high_rel_chosen_true_rate']:.6f}",
            }
        )
        pair_bucket_rows.extend(buckets)

    write_csv(
        out_dir / "family_runs.csv",
        family_rows,
        ["family", "run_name", "status", "dataset_scope", "evidence_path", "key_metrics", "failure_mode", "diagnosis"],
    )
    write_csv(
        out_dir / "laplace_pair_overview.csv",
        pair_overall_rows,
        ["sequence", "valid_pairs", "chosen_rate", "true_rate", "chosen_true_rate", "avg_pair_rel", "high_rel_chosen_true_rate"],
    )
    write_csv(
        out_dir / "laplace_pair_buckets.csv",
        pair_bucket_rows,
        ["sequence", "bucket", "pairs", "chosen_pairs", "true_rate", "chosen_true_rate", "avg_pair_rel", "avg_motion_sim", "avg_appearance_sim"],
    )

    summary_row = {
        "audit_name": out_dir.name,
        "status": "success",
        "frequency_smoke_fixnan_attempts": str(len(freq_smoke_dirs)),
        "frequency_main_nan_lines": str(freq_main_nan_lines),
        "frequency_main_tracker_root_missing": str(int(freq_main_tracker_root_missing)),
        "frequency_fixnan_best_HOTA": f"{best_freq_fixnan.hota:.3f}",
        "frequency_fixnan_best_AssA": f"{best_freq_fixnan.assa:.3f}",
        "frequency_fixnan_best_IDF1": f"{best_freq_fixnan.idf1:.3f}",
        "frequency_fixnan_best_MOTA": f"{best_freq_fixnan.mota:.3f}",
        "frequency_fixnan_best_IDSW": f"{best_freq_fixnan.idsw:.0f}",
        "frequency_sfi_mot20_HOTA": f"{sfi_metric.hota:.3f}",
        "frequency_sfi_mot20_AssA": f"{sfi_metric.assa:.3f}",
        "frequency_sfi_mot20_IDF1": f"{sfi_metric.idf1:.3f}",
        "laplace_v15_delta_HOTA": f"{laplace_deltas['HOTA']:.3f}",
        "laplace_v15_delta_AssA": f"{laplace_deltas['AssA']:.3f}",
        "laplace_v15_delta_IDF1": f"{laplace_deltas['IDF1']:.3f}",
        "laplace_v15_delta_MOTA": f"{laplace_deltas['MOTA']:.3f}",
        "laplace_v15_delta_IDSW": f"{laplace_deltas['IDSW']:.0f}",
        "laplace_v16_best_epoch": str(best_v16.epoch),
        "laplace_v16_best_HOTA": f"{best_v16.hota:.3f}",
        "laplace_v16_best_AssA": f"{best_v16.assa:.3f}",
        "laplace_v16_best_IDF1": f"{best_v16.idf1:.3f}",
        "laplace_v16_last_epoch": str(last_v16.epoch),
        "laplace_v16_last_HOTA": f"{last_v16.hota:.3f}",
        "laplace_v16_last_AssA": f"{last_v16.assa:.3f}",
        "laplace_pair02_highrel_ctr": next(row["high_rel_chosen_true_rate"] for row in pair_overall_rows if row["sequence"] == "MOT17-02-FRCNN"),
        "laplace_pair13_highrel_ctr": next(row["high_rel_chosen_true_rate"] for row in pair_overall_rows if row["sequence"] == "MOT17-13-FRCNN"),
        "overall_verdict": "frequency_family_failed_on_stability_then_identity_collapse; laplace_family_had_real_proxy_value_but_learned_gate_regressed_and_hard_slice_calibration_broke",
    }
    write_csv(out_dir / "summary.csv", [summary_row], summary_row.keys())
    write_csv(out_dir / "result.csv", [summary_row], summary_row.keys())

    report = f"""# Legacy Module Forensic Audit

Date: {datetime.now().isoformat(timespec="seconds")}

## Scope

This audit revisits two older idea families that were implemented before the current diagnosis-driven official-ByteTrack work:

- `frequency family`
- `laplace family`

The goal is not to rerun them. The goal is to answer a narrower question:

- were they truly "not validated",
- or were they run already but never investigated deeply enough to understand why they failed or where they actually worked?

## Frequency Family

### Main findings

1. The original `v13_tf_plus_lowfreq_mamba` line was not merely "non-positive". It was optimization-unstable.
   - Main evidence: `{freq_main_log}`
   - `nan_lines={freq_main_nan_lines}`
   - epoch 0 validation finishes, but the log reports `tracker root not found`, so the run never produces a clean usable validation artifact.
2. There were at least `{len(freq_smoke_dirs)}` follow-up `smoke_fixnan` rescue attempts.
3. The strongest rescue attempt that actually validates, `v13_tf_only_val0213_smoke_fixnan_v4`, is numerically stable but behaviorally collapsed.
   - best HOTA = `{best_freq_fixnan.hota:.2f}`
   - best AssA = `{best_freq_fixnan.assa:.2f}`
   - best IDF1 = `{best_freq_fixnan.idf1:.2f}`
   - best MOTA = `{best_freq_fixnan.mota:.2f}`
   - best IDSW = `{best_freq_fixnan.idsw:.0f}`
4. There is one conservative `SFI-lite` branch on MOT20 val05 that is stable:
   - HOTA = `{sfi_metric.hota:.2f}`
   - AssA = `{sfi_metric.assa:.2f}`
   - IDF1 = `{sfi_metric.idf1:.2f}`
   - IDSW = `{sfi_metric.idsw:.0f}`
   - but this is off the canonical MOT17 carrier and was never audited in a strict paired setting.

### Diagnosis

The frequency family failed in two stages:

- first on optimization stability,
- then on identity semantics after numerical stabilization.

So the missing work was not "we forgot to validate it". The missing work was that we never explicitly closed the loop and wrote down that the family moved from `NaN instability` to `stable but identity-collapsed`, which is a much stronger and more useful conclusion.

## Laplace Family

### Main findings

1. The fixed `v15` Laplace branch had a real positive regime on `proxy0213`.
   - base vs laplace evidence:
     - base: `{laplace_base_summary}`
     - laplace: `{laplace_full_summary}`
   - delta HOTA = `{laplace_deltas['HOTA']:.3f}`
   - delta AssA = `{laplace_deltas['AssA']:.3f}`
   - delta IDF1 = `{laplace_deltas['IDF1']:.3f}`
   - delta MOTA = `{laplace_deltas['MOTA']:.3f}`
   - delta IDSW = `{laplace_deltas['IDSW']:.0f}`
2. Pair-log evidence shows the Laplace trust signal is useful but slice-dependent.
   - `MOT17-02-FRCNN`: high-reliability bucket chosen-true-rate = `{next(row["high_rel_chosen_true_rate"] for row in pair_overall_rows if row["sequence"] == "MOT17-02-FRCNN")}`
   - `MOT17-13-FRCNN`: high-reliability bucket chosen-true-rate = `{next(row["high_rel_chosen_true_rate"] for row in pair_overall_rows if row["sequence"] == "MOT17-13-FRCNN")}`
3. The trainable `v16` Laplace gate did not explode, but it regressed steadily.
   - best epoch = `{best_v16.epoch}` with HOTA `{best_v16.hota:.2f}`, AssA `{best_v16.assa:.2f}`, IDF1 `{best_v16.idf1:.2f}`
   - latest recorded epoch = `{last_v16.epoch}` with HOTA `{last_v16.hota:.2f}`, AssA `{last_v16.assa:.2f}`, IDF1 `{last_v16.idf1:.2f}`, IDSW `{last_v16.idsw:.0f}`

### Diagnosis

The Laplace family should not be labeled "invalid".

The stronger conclusion is:

- the heuristic/fixed branch worked on a real proxy slice,
- but the learned/trainable gate objective degraded that branch,
- and the hard slice calibration breaks badly on `MOT17-13-FRCNN`.

So the missing work was not proving that Laplace never worked. The missing work was proving exactly where it worked, and why the learned version made it worse.

## Final Verdict

1. `frequency family`
   - already ran
   - not merely unvalidated
   - failed first by instability, then by semantic collapse
2. `laplace family`
   - already ran
   - had a genuine positive regime on proxy0213
   - but was never carried into a clean canonical-carrier diagnosis package
   - and its learned gate version regressed

## What This Changes

The current project should not treat both older families as the same kind of "dead end".

- `frequency`: evidence now says the family needs a stability-and-semantics redesign before it deserves new budget.
- `laplace`: evidence now says the core intuition had value, but the learned calibration/supervision was the weak point.

This matters for the next redesign decision: if we reuse anything from legacy lines, Laplace-style `reliability over temporal history` is the more credible seed than the old heavy frequency stack.
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    append_registry(
        {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "kind": "analysis",
            "status": "success",
            "script": "scripts/run_legacy_module_forensic_audit.py",
            "dataset": "MOT17+MOT20",
            "split": "historical_forensic",
            "tracker_family": "legacy_module_audit",
            "variant": out_dir.name,
            "tag": "forensic_audit",
            "run_root": str(out_dir),
            "summary_csv": str(out_dir / "summary.csv"),
            "log_path": str(out_dir / "report.md"),
            "notes": "forensic audit of frequency and laplace legacy module families",
            "name": out_dir.name,
        }
    )

    print(out_dir)


if __name__ == "__main__":
    main()
