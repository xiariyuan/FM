"""Unified reproducibility and decision audit for M23-3-1 replacement oracle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
VARIANTS = (
    "baseline_raw",
    "m23_selected_replacement_oracle",
    "dense_replacement_oracle",
)
COMPACT_FILES = (
    "candidate_inventory.csv",
    "utility_bins.csv",
    "variant_metrics.csv",
    "per_sequence_metrics.csv",
    "report.json",
    "manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def tracker_skeleton(path: Path) -> List[Tuple[int, int, Tuple[str, ...]]]:
    result = []
    previous_frame = -1
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(",")
        if len(fields) < 7:
            raise RuntimeError(f"invalid tracker line in {path}: {line!r}")
        frame = int(fields[0])
        identity = int(fields[1])
        if frame < previous_frame:
            raise RuntimeError(f"non-monotonic tracker frames in {path}")
        previous_frame = frame
        key = (frame, identity)
        if key in seen:
            raise RuntimeError(f"duplicate frame/identity in {path}: {key}")
        seen.add(key)
        result.append((frame, identity, tuple(fields[7:])))
    return result


def add_check(checks: List[dict], name: str, passed: bool, details: object = None) -> None:
    item = {"name": name, "passed": bool(passed), "details": details}
    checks.append(item)
    if not passed:
        raise RuntimeError(f"unified audit failed: {name}: {details}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--formal-dir", required=True)
    parser.add_argument("--repro-dir", required=True)
    parser.add_argument("--precision-diagnostic-dir", required=True)
    parser.add_argument("--preregister-v1", required=True)
    parser.add_argument("--preregister-v2", required=True)
    parser.add_argument("--action-script", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    formal = resolve(repo, args.formal_dir)
    repro = resolve(repo, args.repro_dir)
    diagnostic = resolve(repo, args.precision_diagnostic_dir)
    prereg_v1_path = resolve(repo, args.preregister_v1)
    prereg_v2_path = resolve(repo, args.preregister_v2)
    action_script_path = resolve(repo, args.action_script)
    output_path = resolve(repo, args.out)

    prereg_v1 = json.loads(prereg_v1_path.read_text(encoding="utf-8"))
    prereg_v2 = json.loads(prereg_v2_path.read_text(encoding="utf-8"))
    report = json.loads((formal / "report.json").read_text(encoding="utf-8"))
    repro_report = json.loads((repro / "report.json").read_text(encoding="utf-8"))
    diagnostic_report = json.loads((diagnostic / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((formal / "manifest.json").read_text(encoding="utf-8"))
    checks: List[dict] = []

    add_check(checks, "preregister_v1_schema", prereg_v1["schema"].endswith("preregister.v1"), prereg_v1["schema"])
    add_check(checks, "preregister_v2_schema", prereg_v2["schema"].endswith("preregister.v2"), prereg_v2["schema"])
    add_check(
        checks,
        "v2_binds_action_script",
        prereg_v2["script_sha256"] == sha256_file(action_script_path),
        {"expected": prereg_v2["script_sha256"], "actual": sha256_file(action_script_path)},
    )
    add_check(
        checks,
        "v2_binds_v1_preregister",
        prereg_v2["correction"]["previous_preregister_sha256"] == sha256_file(prereg_v1_path),
        prereg_v2["correction"],
    )
    add_check(checks, "v2_metric_or_gate_changes_false", prereg_v2["correction"]["metric_or_gate_changes"] is False)
    add_check(checks, "v2_candidate_or_tracker_changes_false", prereg_v2["correction"]["candidate_or_tracker_changes"] is False)
    add_check(
        checks,
        "v2_only_combined_metric_source_changed",
        "detailed COMBINED" in prereg_v2["correction"]["only_change"],
        prereg_v2["correction"]["only_change"],
    )

    compact_hashes: Dict[str, dict] = {}
    for name in COMPACT_FILES:
        formal_hash = sha256_file(formal / name)
        repro_hash = sha256_file(repro / name)
        compact_hashes[name] = {"formal": formal_hash, "repro": repro_hash}
        add_check(checks, f"compact_byte_identity:{name}", formal_hash == repro_hash, compact_hashes[name])

    for name in ("candidate_inventory.csv", "utility_bins.csv", "variant_metrics.csv", "per_sequence_metrics.csv"):
        add_check(
            checks,
            f"precision_correction_preserved_artifact:{name}",
            sha256_file(formal / name) == sha256_file(diagnostic / name),
            {"formal": sha256_file(formal / name), "diagnostic": sha256_file(diagnostic / name)},
        )

    tracker_hashes: Dict[str, dict] = {}
    eval_hashes: Dict[str, dict] = {}
    for variant in VARIANTS:
        tracker_hashes[variant] = {}
        for sequence in SEQUENCES:
            relative = Path("eval_work/trackers") / variant / "data" / f"{sequence}.txt"
            formal_hash = sha256_file(formal / relative)
            repro_hash = sha256_file(repro / relative)
            diagnostic_hash = sha256_file(diagnostic / relative)
            tracker_hashes[variant][sequence] = formal_hash
            add_check(
                checks,
                f"tracker_byte_identity:{variant}:{sequence}",
                formal_hash == repro_hash,
                {"formal": formal_hash, "repro": repro_hash},
            )
            add_check(
                checks,
                f"precision_correction_preserved_tracker:{variant}:{sequence}",
                formal_hash == diagnostic_hash,
                {"formal": formal_hash, "diagnostic": diagnostic_hash},
            )
        eval_hashes[variant] = {}
        for name in ("pedestrian_summary.txt", "pedestrian_detailed.csv"):
            relative = Path("eval_work/eval") / variant / name
            formal_hash = sha256_file(formal / relative)
            repro_hash = sha256_file(repro / relative)
            diagnostic_hash = sha256_file(diagnostic / relative)
            eval_hashes[variant][name] = formal_hash
            add_check(
                checks,
                f"trackeval_byte_identity:{variant}:{name}",
                formal_hash == repro_hash,
                {"formal": formal_hash, "repro": repro_hash},
            )
            add_check(
                checks,
                f"precision_correction_preserved_trackeval:{variant}:{name}",
                formal_hash == diagnostic_hash,
                {"formal": formal_hash, "diagnostic": diagnostic_hash},
            )

    add_check(checks, "formal_repro_report_identity", report == repro_report)
    add_check(
        checks,
        "manifest_report_hash",
        manifest["report_sha256"] == sha256_file(formal / "report.json"),
        manifest["report_sha256"],
    )
    add_check(
        checks,
        "formal_binds_v2_preregister",
        report["sources"]["preregister"]["sha256"] == sha256_file(prereg_v2_path),
        report["sources"]["preregister"],
    )
    add_check(checks, "diagnostic_detected_only_combined_precision", diagnostic_report["decision"]["baseline_sequences_reproduced"] is True)
    add_check(checks, "diagnostic_combined_false_due_precision", diagnostic_report["decision"]["baseline_combined_reproduced"] is False)

    baseline_sources = report["sources"]["baseline"]
    for sequence in SEQUENCES:
        baseline_tracker = formal / "eval_work" / "trackers" / "baseline_raw" / "data" / f"{sequence}.txt"
        add_check(
            checks,
            f"baseline_tracker_exact_source:{sequence}",
            sha256_file(baseline_tracker) == baseline_sources[sequence],
            {"tracker": sha256_file(baseline_tracker), "source": baseline_sources[sequence]},
        )
        baseline_skeleton = tracker_skeleton(baseline_tracker)
        for variant in ("m23_selected_replacement_oracle", "dense_replacement_oracle"):
            variant_tracker = formal / "eval_work" / "trackers" / variant / "data" / f"{sequence}.txt"
            add_check(
                checks,
                f"identity_frame_tail_skeleton_preserved:{variant}:{sequence}",
                tracker_skeleton(variant_tracker) == baseline_skeleton,
                {"baseline_rows": len(baseline_skeleton), "variant_rows": len(tracker_skeleton(variant_tracker))},
            )

    decision = report["decision"]
    add_check(checks, "baseline_combined_reproduced", decision["baseline_combined_reproduced"] is True)
    add_check(checks, "baseline_sequences_reproduced", decision["baseline_sequences_reproduced"] is True)
    add_check(
        checks,
        "baseline_metric_deltas_zero",
        all(abs(float(value)) <= 1e-12 for value in decision["baseline_metric_deltas"].values()),
        decision["baseline_metric_deltas"],
    )
    add_check(checks, "selected_all_sequences_nonnegative", decision["selected_all_sequence_hota_nonnegative"] is True)
    add_check(checks, "dense_all_sequences_nonnegative", decision["dense_all_sequence_hota_nonnegative"] is True)
    add_check(checks, "selected_combined_gate_failed", decision["selected_combined_hota_passed"] is False)
    add_check(checks, "dense_combined_gate_failed", decision["dense_combined_hota_passed"] is False)
    add_check(checks, "replacement_graph_ceiling_failed", decision["replacement_graph_ceiling_passed"] is False)
    add_check(checks, "deployment_false", decision["deployment_allowed"] is False)
    add_check(checks, "locked_manifest_false", decision["locked_manifest_created"] is False)
    add_check(checks, "p15_no_op", report["locked_state"]["p15_policy"] == "no_op")
    add_check(checks, "locked_label_reads_zero", int(report["locked_state"]["locked_label_reads"]) == 0)
    add_check(checks, "locked_trackeval_zero", int(report["locked_state"]["locked_trackeval_calls"]) == 0)
    add_check(checks, "locked_rows_untouched", int(report["locked_state"]["remaining_locked_rows_untouched"]) == 156)

    inventory = {row["sequence"]: row for row in read_csv(formal / "candidate_inventory.csv")}
    combined_inventory = inventory["COMBINED"]
    add_check(checks, "candidate_rows_match_preregister", int(combined_inventory["candidate_rows"]) == 459899)
    add_check(checks, "eligible_candidates_match_preregister", int(combined_inventory["eligible_candidates"]) == 414387)
    add_check(checks, "dense_positive_groups_match_preregister", int(combined_inventory["dense_positive_groups"]) == 174497)
    add_check(checks, "selected_positive_groups_match_preregister", int(combined_inventory["selected_positive_groups"]) == 171212)
    add_check(
        checks,
        "replacement_plan_hashes_reproduced",
        report["replacement_plan_hashes"] == repro_report["replacement_plan_hashes"],
        report["replacement_plan_hashes"],
    )

    bins = read_csv(formal / "utility_bins.csv")
    dense_group_bins = {
        row["utility_bin"]: int(row["count"])
        for row in bins
        if row["sequence"] == "COMBINED" and row["scope"] == "dense_group_best"
    }
    positive_dense_groups = sum(
        count for label, count in dense_group_bins.items() if label != "(-inf,0]"
    )
    small_positive_dense_groups = sum(
        dense_group_bins[label]
        for label in ("(0,0.01]", "(0.01,0.02]", "(0.02,0.05]")
    )
    add_check(checks, "positive_dense_group_bin_total", positive_dense_groups == 174497, dense_group_bins)

    exact_metrics = report["exact_combined_metrics"]
    baseline_hota = float(exact_metrics["baseline_raw"]["HOTA"])
    selected_hota = float(exact_metrics["m23_selected_replacement_oracle"]["HOTA"])
    dense_hota = float(exact_metrics["dense_replacement_oracle"]["HOTA"])
    interpretation = {
        "selected_hota_gain": selected_hota - baseline_hota,
        "dense_hota_gain": dense_hota - baseline_hota,
        "dense_increment_over_selected": dense_hota - selected_hota,
        "dense_positive_groups": positive_dense_groups,
        "dense_positive_groups_with_utility_at_most_0_05": small_positive_dense_groups,
        "small_positive_fraction": small_positive_dense_groups / positive_dense_groups,
        "dense_fp_reduction": int(exact_metrics["baseline_raw"]["CLR_FP"]) - int(exact_metrics["dense_replacement_oracle"]["CLR_FP"]),
        "dense_fn_reduction": int(exact_metrics["baseline_raw"]["CLR_FN"]) - int(exact_metrics["dense_replacement_oracle"]["CLR_FN"]),
        "dense_idsw_change": int(exact_metrics["dense_replacement_oracle"]["IDSW"]) - int(exact_metrics["baseline_raw"]["IDSW"]),
        "raw_row_count_preserved_but_trackeval_dets_can_change": "MOTChallenge preprocessing can retain/remove different rows after boxes move",
    }
    research_decision = {
        "direct_segment_replacement_graph_closed": True,
        "reason": "even a GT oracle with 174,497 positive group replacements reaches only 78.00133 HOTA, below the preregistered 78.5 ceiling",
        "do_not_train_replacement_scorer": True,
        "appearance_role": "auxiliary only; unchanged",
        "next_stage": "M23-4-0 fixed 30-frame inter-segment gap-bridge addition oracle from raw baseline, inheriting an existing track ID only when absent in the frame; combine with dense replacement, no spawn ID and no identity relabeling",
        "deployment_allowed": False,
        "locked_manifest_created": False,
    }

    audit = {
        "schema": "fmtrack.m23_3_1.unified_audit.v1",
        "checks": checks,
        "all_checks_passed": all(item["passed"] for item in checks),
        "counts": {
            "checks": len(checks),
            "compact_files_reproduced": len(COMPACT_FILES),
            "tracker_files_reproduced": len(VARIANTS) * len(SEQUENCES),
            "trackeval_files_reproduced": len(VARIANTS) * 2,
            "precision_diagnostic_trackers_verified": len(VARIANTS) * len(SEQUENCES),
            "precision_diagnostic_trackeval_files_verified": len(VARIANTS) * 2,
        },
        "hashes": {
            "action_script": sha256_file(action_script_path),
            "preregister_v1": sha256_file(prereg_v1_path),
            "preregister_v2": sha256_file(prereg_v2_path),
            "formal_report": sha256_file(formal / "report.json"),
            "formal_manifest": sha256_file(formal / "manifest.json"),
            "compact": compact_hashes,
            "trackers": tracker_hashes,
            "trackeval": eval_hashes,
        },
        "formal_decision": decision,
        "interpretation": interpretation,
        "research_decision": research_decision,
        "locked_state": report["locked_state"],
    }
    canonical_json_dump(audit, output_path)
    print(json.dumps({
        "all_checks_passed": audit["all_checks_passed"],
        "checks": len(checks),
        "output_sha256": sha256_file(output_path),
        "interpretation": interpretation,
        "research_decision": research_decision,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
