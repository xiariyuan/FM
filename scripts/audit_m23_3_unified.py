"""Unified reproducibility and decision audit for M23-3-0."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
VARIANTS = (
    "postnms_context",
    "candidate_default_safe",
    "candidate_suppressor_safe",
    "candidate_suppressor_episode_spawn",
    "oracle_linked_spawn",
)
COMPACT_FILES = (
    "tracklet_segments.csv",
    "action_summary.csv",
    "action_reason_summary.csv",
    "spawn_episodes.csv",
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


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def check(checks: List[dict], name: str, passed: bool, details: object = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "details": details})
    if not passed:
        raise RuntimeError(f"unified audit failed: {name}: {details}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--formal-dir", required=True)
    parser.add_argument("--repro-dir", required=True)
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
    prereg_v1_path = resolve(repo, args.preregister_v1)
    prereg_v2_path = resolve(repo, args.preregister_v2)
    action_script_path = resolve(repo, args.action_script)
    output_path = resolve(repo, args.out)

    checks: List[dict] = []
    prereg_v1 = json.loads(prereg_v1_path.read_text(encoding="utf-8"))
    prereg_v2 = json.loads(prereg_v2_path.read_text(encoding="utf-8"))
    report = json.loads((formal / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((formal / "manifest.json").read_text(encoding="utf-8"))
    repro_report = json.loads((repro / "report.json").read_text(encoding="utf-8"))

    check(checks, "preregister_v1_schema", prereg_v1["schema"].endswith("preregister.v1"), prereg_v1["schema"])
    check(checks, "preregister_v2_schema", prereg_v2["schema"].endswith("preregister.v2"), prereg_v2["schema"])
    check(
        checks,
        "v2_binds_action_script",
        prereg_v2["script_sha256"] == sha256_file(action_script_path),
        {"expected": prereg_v2["script_sha256"], "actual": sha256_file(action_script_path)},
    )
    check(
        checks,
        "v2_preserves_v1_hash",
        prereg_v2["correction"]["failed_preregister_sha256"] == sha256_file(prereg_v1_path),
        prereg_v2["correction"],
    )
    check(checks, "v2_metric_or_gate_changes_false", prereg_v2["correction"]["metric_or_gate_changes"] is False)
    check(checks, "v2_action_or_episode_changes_false", prereg_v2["correction"]["action_or_episode_changes"] is False)
    check(
        checks,
        "synthetic_namespace_float32_safe",
        int(prereg_v2["protocol"]["synthetic_spawn_id_base"]) < (1 << 24),
        prereg_v2["protocol"],
    )

    compact_hashes: Dict[str, dict] = {}
    for name in COMPACT_FILES:
        formal_hash = sha256_file(formal / name)
        repro_hash = sha256_file(repro / name)
        compact_hashes[name] = {"formal": formal_hash, "repro": repro_hash}
        check(checks, f"compact_byte_identity:{name}", formal_hash == repro_hash, compact_hashes[name])

    tracker_hashes: Dict[str, dict] = {}
    eval_hashes: Dict[str, dict] = {}
    for variant in VARIANTS:
        tracker_hashes[variant] = {}
        for sequence in SEQUENCES:
            relative = Path("eval_work/trackers") / variant / "data" / f"{sequence}.txt"
            formal_hash = sha256_file(formal / relative)
            repro_hash = sha256_file(repro / relative)
            tracker_hashes[variant][sequence] = formal_hash
            check(
                checks,
                f"tracker_byte_identity:{variant}:{sequence}",
                formal_hash == repro_hash,
                {"formal": formal_hash, "repro": repro_hash},
            )
        eval_hashes[variant] = {}
        for name in ("pedestrian_summary.txt", "pedestrian_detailed.csv"):
            relative = Path("eval_work/eval") / variant / name
            formal_hash = sha256_file(formal / relative)
            repro_hash = sha256_file(repro / relative)
            eval_hashes[variant][name] = formal_hash
            check(
                checks,
                f"trackeval_byte_identity:{variant}:{name}",
                formal_hash == repro_hash,
                {"formal": formal_hash, "repro": repro_hash},
            )

    check(checks, "formal_repro_report_object_identity", report == repro_report)
    check(
        checks,
        "manifest_report_hash",
        manifest["report_sha256"] == sha256_file(formal / "report.json"),
        manifest["report_sha256"],
    )
    check(checks, "oracle_linked_spawn_reproduced", report["decision"]["oracle_linked_spawn_reproduced"] is True)
    check(
        checks,
        "oracle_linked_zero_metric_delta",
        all(abs(float(value)) <= 1e-12 for value in report["decision"]["oracle_linked_spawn_metric_deltas"].values()),
        report["decision"]["oracle_linked_spawn_metric_deltas"],
    )
    check(checks, "candidate_suppressor_combined_gate", report["decision"]["candidate_suppressor_combined_hota_passed"] is True)
    check(checks, "candidate_suppressor_worst_sequence_gate", report["decision"]["candidate_suppressor_worst_sequence_hota_passed"] is True)
    check(checks, "candidate_suppressor_all_sequences_nonnegative", report["decision"]["candidate_suppressor_all_sequence_increment_nonnegative"] is True)
    check(checks, "episode_spawn_combined_gate_failed", report["decision"]["episode_spawn_combined_hota_passed"] is False)
    check(checks, "episode_spawn_worst_sequence_gate_failed", report["decision"]["episode_spawn_worst_sequence_hota_passed"] is False)
    check(checks, "deployment_false", report["decision"]["deployment_allowed"] is False)
    check(checks, "locked_manifest_false", report["decision"]["locked_manifest_created"] is False)
    check(checks, "p15_no_op", report["locked_state"]["p15_policy"] == "no_op")
    check(checks, "locked_rows_untouched", int(report["locked_state"]["remaining_locked_rows_untouched"]) == 156)
    check(checks, "locked_label_reads_zero", int(report["locked_state"]["locked_label_reads"]) == 0)
    check(checks, "locked_trackeval_zero", int(report["locked_state"]["locked_trackeval_calls"]) == 0)

    variant_rows = {row["variant"]: row for row in read_csv(formal / "variant_metrics.csv")}
    action_rows = read_csv(formal / "action_summary.csv")
    combined_actions = {
        row["action"]: row for row in action_rows if row["sequence"] == "COMBINED"
    }
    reason_counter = Counter()
    for row in read_csv(formal / "action_reason_summary.csv"):
        reason_counter[(row["action"], row["candidate_reason"], row["suppressor_reason"])] += int(row["events"])

    candidate_hota = float(variant_rows["candidate_default_safe"]["HOTA"])
    candidate_suppressor_hota = float(variant_rows["candidate_suppressor_safe"]["HOTA"])
    episode_hota = float(variant_rows["candidate_suppressor_episode_spawn"]["HOTA"])
    oracle_hota = float(variant_rows["oracle_linked_spawn"]["HOTA"])
    context_hota = float(variant_rows["postnms_context"]["HOTA"])
    candidate_replacement_events = int(combined_actions["candidate_default"]["replacement_events"])
    candidate_events = int(combined_actions["candidate_default"]["events"])

    interpretation = {
        "candidate_default_safe_hota_gain_vs_postnms": candidate_hota - context_hota,
        "suppressor_override_incremental_hota": candidate_suppressor_hota - candidate_hota,
        "episode_spawn_hota_delta_vs_candidate_suppressor_safe": episode_hota - candidate_suppressor_hota,
        "oracle_linkage_hota_gain_vs_episode_spawn": oracle_hota - episode_hota,
        "candidate_default_replacement_fraction": candidate_replacement_events / candidate_events,
        "candidate_default_events": candidate_events,
        "suppressor_override_events": int(combined_actions["suppressor_override"]["events"]),
        "spawn_events": int(combined_actions["spawn"]["events"]),
        "spawn_conflicting_other_gt_events": reason_counter[("spawn", "candidate_support_other_gt", "suppressor_support_other_gt")],
        "spawn_both_segments_missing_events": reason_counter[("spawn", "candidate_segment_missing", "suppressor_segment_missing")],
    }
    research_decision = {
        "candidate_default_action_space_retained": True,
        "suppressor_override_as_separate_learned_branch": False,
        "suppressor_decision_status": "post-hoc engineering interpretation: only 123 oracle events and +0.001 rounded HOTA beyond candidate-default",
        "episode_spawn_branch_retained": False,
        "spawn_decision_status": "preregistered TrackEval gates failed; new-track episodes caused severe association fragmentation",
        "appearance_role": "auxiliary edge evidence only, unchanged from M23-2",
        "next_stage": "M23-3-1 sequence-LOSO segment-conditioned observation replacement graph with baseline no-op, one observation per segment/frame, candidate inheritance only, spawn disabled, suppressor diagnostic only",
        "deployment_allowed": False,
        "locked_manifest_created": False,
    }

    audit = {
        "schema": "fmtrack.m23_3.unified_audit.v1",
        "checks": checks,
        "all_checks_passed": all(item["passed"] for item in checks),
        "counts": {
            "checks": len(checks),
            "compact_files_reproduced": len(COMPACT_FILES),
            "tracker_files_reproduced": len(VARIANTS) * len(SEQUENCES),
            "trackeval_files_reproduced": len(VARIANTS) * 2,
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
        "formal_decision": report["decision"],
        "interpretation": interpretation,
        "research_decision": research_decision,
        "locked_state": report["locked_state"],
    }
    canonical_json_dump(audit, output_path)
    print(json.dumps({
        "all_checks_passed": audit["all_checks_passed"],
        "checks": len(checks),
        "output_sha256": sha256_file(output_path),
        "research_decision": research_decision,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
