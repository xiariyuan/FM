"""Unified reproducibility and decision audit for M23-5-0 mechanism oracle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
VARIANTS = (
    "dense_replacement_context", "safe_gap_context", "all_additions_reference",
    "distinct_person_dual_retention_oracle", "mechanism_decomposed_oracle",
    "safe_gap_plus_mechanism_oracle",
)
COMPACT_FILES = (
    "conflict_summary.csv", "mechanism_actions.csv", "per_sequence_metrics.csv",
    "variant_metrics.csv", "report.json", "manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_key(line: str) -> tuple[int, int]:
    fields = line.split(",", 2)
    return int(fields[0]), int(fields[1])


def tracker_index(path: Path) -> dict[tuple[int, int], str]:
    result: dict[tuple[int, int], str] = {}
    with path.open("r", encoding="utf-8") as handle:
        previous_frame = -1
        for raw in handle:
            line = raw.rstrip("\n")
            if not line:
                continue
            key = parse_key(line)
            if key[0] < previous_frame:
                raise RuntimeError(f"non-monotonic tracker {path}")
            previous_frame = key[0]
            if key in result:
                raise RuntimeError(f"duplicate key {key} in {path}")
            result[key] = line
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-dir", default="outputs/mot20_m23_20260717/mechanism_decomposed_nms_conflict_v1")
    parser.add_argument("--repro-dir", default="outputs/mot20_m23_20260717/mechanism_decomposed_nms_conflict_v1_repro")
    parser.add_argument("--preregister", default="outputs/mot20_m23_20260717/mechanism_decomposed_nms_conflict_preregister_v1.json")
    parser.add_argument("--script", default="scripts/audit_m23_5_mechanism_decomposed_nms_conflict_oracle.py")
    parser.add_argument("--gap-dir", default="outputs/mot20_m23_20260717/existing_id_gap_bridge_oracle_v1")
    parser.add_argument("--replacement-dir", default="outputs/mot20_m23_20260717/deployable_replacement_oracle_v1")
    parser.add_argument("--out", default="deliverables/mot20_m23_5_audit_20260717.json")
    args = parser.parse_args()

    repo = Path.cwd()
    formal, repro = repo / args.formal_dir, repo / args.repro_dir
    prereg = load_json(repo / args.preregister)
    report = load_json(formal / "report.json")
    manifest = load_json(formal / "manifest.json")
    gap_dir, replacement_dir = repo / args.gap_dir, repo / args.replacement_dir
    out = repo / args.out
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    script_sha = sha256_file(repo / args.script)
    check("script_sha_matches_preregister", script_sha == prereg["script_sha256"], {
        "actual": script_sha, "expected": prereg["script_sha256"]
    })
    check("report_sha_matches_manifest", sha256_file(formal / "report.json") == manifest["report_sha256"])
    check("report_plan_matches_preregister", report["plan_hashes"] == prereg["expected"]["plan_hashes"])
    check("report_sources_match_preregister", report["source_hashes"] == prereg["sources"])
    check("report_counts_match_preregister", report["counts"] == prereg["expected"]["summary_rows"][-1])

    compact_identical = 0
    for name in COMPACT_FILES:
        formal_sha, repro_sha = sha256_file(formal / name), sha256_file(repro / name)
        same = formal_sha == repro_sha
        compact_identical += int(same)
        check(f"compact_byte_identical::{name}", same, {"formal": formal_sha, "repro": repro_sha})

    tracker_identical = 0
    for variant in VARIANTS:
        for sequence in SEQUENCES:
            formal_path = formal / "eval_work/trackers" / variant / "data" / f"{sequence}.txt"
            repro_path = repro / "eval_work/trackers" / variant / "data" / f"{sequence}.txt"
            same = sha256_file(formal_path) == sha256_file(repro_path)
            tracker_identical += int(same)
            check(f"tracker_byte_identical::{variant}::{sequence}", same)

    trackeval_identical = 0
    for variant in VARIANTS:
        for name in ("pedestrian_summary.txt", "pedestrian_detailed.csv"):
            formal_path = formal / "eval_work/eval" / variant / name
            repro_path = repro / "eval_work/eval" / variant / name
            same = sha256_file(formal_path) == sha256_file(repro_path)
            trackeval_identical += int(same)
            check(f"trackeval_byte_identical::{variant}::{name}", same)

    actions = read_csv(formal / "mechanism_actions.csv")
    check("action_count", len(actions) == 2994, len(actions))
    mechanism_counts = Counter(row["mechanism"] for row in actions)
    box_counts = Counter(row["box_choice"] for row in actions)
    check("mechanism_counts", mechanism_counts == Counter({
        "other_target": 2603, "same_target": 384, "unmatched": 7,
    }), dict(mechanism_counts))
    check("box_choice_counts", box_counts == Counter({"candidate": 2782, "source": 212}), dict(box_counts))
    check("unique_source_actions", len({(row["sequence"], int(row["source_line_index"])) for row in actions}) == len(actions))
    check("unique_target_frame_ids", len({(row["sequence"], int(row["frame"]), int(row["inherited_track_id"])) for row in actions}) == len(actions))
    check("candidate_target_iou_gate", all(float(row["candidate_target_iou"]) >= 0.5 for row in actions))
    check("candidate_host_iou_gate", all(float(row["candidate_host_iou"]) >= 0.5 for row in actions))
    check("source_box_only_same_target", all(row["mechanism"] == "same_target" for row in actions if row["box_choice"] == "source"))

    actions_by_sequence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in actions:
        actions_by_sequence[row["sequence"]].append(row)

    for sequence in SEQUENCES:
        dense_source = replacement_dir / "eval_work/trackers/dense_replacement_oracle/data" / f"{sequence}.txt"
        safe_source = gap_dir / "eval_work/trackers/safe_existing_id_gap_bridge_oracle/data" / f"{sequence}.txt"
        all_source = gap_dir / "eval_work/trackers/all_existing_id_gap_bridge_oracle/data" / f"{sequence}.txt"
        reference_paths = {
            "dense_replacement_context": dense_source,
            "safe_gap_context": safe_source,
            "all_additions_reference": all_source,
        }
        for variant, source in reference_paths.items():
            formal_path = formal / "eval_work/trackers" / variant / "data" / f"{sequence}.txt"
            check(f"reference_tracker_exact::{variant}::{sequence}", sha256_file(formal_path) == sha256_file(source))

        dense = tracker_index(dense_source)
        safe = tracker_index(safe_source)
        dual = tracker_index(formal / "eval_work/trackers/distinct_person_dual_retention_oracle/data" / f"{sequence}.txt")
        mechanism = tracker_index(formal / "eval_work/trackers/mechanism_decomposed_oracle/data" / f"{sequence}.txt")
        combined = tracker_index(formal / "eval_work/trackers/safe_gap_plus_mechanism_oracle/data" / f"{sequence}.txt")
        sequence_actions = actions_by_sequence[sequence]
        other_count = sum(row["mechanism"] == "other_target" for row in sequence_actions)
        replacement_count = len(sequence_actions) - other_count
        check(f"dual_row_delta::{sequence}", len(dual) - len(dense) == other_count)
        check(f"mechanism_row_delta::{sequence}", len(mechanism) - len(dense) == other_count)
        check(f"combined_row_delta::{sequence}", len(combined) - len(safe) == other_count)
        # Dual retention must preserve every dense row exactly.
        check(f"dual_preserves_dense::{sequence}", all(dual.get(key) == line for key, line in dense.items()))
        # Mechanism changes exactly the same/unmatched source keys and preserves all others.
        changed_keys = {
            (int(row["frame"]), int(row["source_track_id"]))
            for row in sequence_actions if row["mechanism"] != "other_target"
        }
        preserved = all(mechanism.get(key) == line for key, line in dense.items() if key not in changed_keys)
        removed = all(key not in mechanism for key in changed_keys)
        check(f"mechanism_preserves_unedited::{sequence}", preserved)
        check(f"mechanism_removes_edited_source_keys::{sequence}", removed, {
            "expected": replacement_count, "observed": sum(key not in mechanism for key in changed_keys)
        })
        # Safe combined preserves unedited safe rows; edited source keys disappear.
        combined_preserved = all(combined.get(key) == line for key, line in safe.items() if key not in changed_keys)
        combined_removed = all(key not in combined for key in changed_keys)
        check(f"combined_preserves_unedited::{sequence}", combined_preserved)
        check(f"combined_removes_edited_source_keys::{sequence}", combined_removed)

    decision = report["decision"]
    check("references_reproduced", all(decision[key] is True for key in (
        "dense_context_reproduced", "safe_context_reproduced", "all_additions_reference_reproduced"
    )))
    check("mechanism_gate_failed", decision["mechanism_gate_passed"] is False)
    check("combined_gate_failed", decision["safe_plus_mechanism_gate_passed"] is False)
    check("ceiling_failed", decision["mechanism_action_space_ceiling_passed"] is False)
    check("deployment_disallowed", decision["deployment_allowed"] is False)
    check("locked_manifest_absent", decision["locked_manifest_created"] is False)
    check("all_sequence_deltas_positive", all(value > 0.0 for value in decision["mechanism_sequence_hota_deltas"].values()) and all(value > 0.0 for value in decision["safe_plus_mechanism_sequence_hota_deltas"].values()))
    check("idsw_nonincreasing", decision["mechanism_idsw_change_vs_dense"] <= 0 and decision["safe_plus_mechanism_idsw_change_vs_safe"] <= 0)
    protocol = report["protocol"]
    check("no_locked_access", protocol["locked_label_reads"] == 0 and protocol["locked_trackeval_calls"] == 0)
    check("no_models_or_sweeps", protocol["models_trained"] == 0 and protocol["threshold_sweeps"] == 0 and protocol["parameter_sweeps"] == 0)

    metrics = {(row["variant"], row["sequence"]): row for row in read_csv(formal / "per_sequence_metrics.csv")}
    def m(variant: str, field: str) -> float:
        return float(metrics[(variant, "COMBINED")][field])
    interpretation = {
        "dual_retention_hota_gain_over_dense": m("distinct_person_dual_retention_oracle", "HOTA") - m("dense_replacement_context", "HOTA"),
        "mechanism_increment_over_dual_retention": m("mechanism_decomposed_oracle", "HOTA") - m("distinct_person_dual_retention_oracle", "HOTA"),
        "mechanism_hota_gain_over_dense": m("mechanism_decomposed_oracle", "HOTA") - m("dense_replacement_context", "HOTA"),
        "safe_plus_mechanism_hota_gain_over_safe": m("safe_gap_plus_mechanism_oracle", "HOTA") - m("safe_gap_context", "HOTA"),
        "all_additions_advantage_over_safe_plus_mechanism": m("all_additions_reference", "HOTA") - m("safe_gap_plus_mechanism_oracle", "HOTA"),
        "all_additions_idsw_advantage": int(float(metrics[("safe_gap_plus_mechanism_oracle", "COMBINED")]["IDSW"])) - int(float(metrics[("all_additions_reference", "COMBINED")]["IDSW"])),
        "formal_repro_compact_identical": compact_identical,
        "formal_repro_tracker_identical": tracker_identical,
        "formal_repro_trackeval_identical": trackeval_identical,
    }
    research_decision = {
        "local_nms_conflict_action_space_closed": True,
        "do_not_train_local_mechanism_selector": True,
        "reason": "mechanism 78.102845 < 78.20 and safe+mechanism 78.137070 < 78.25 despite positive sequence deltas and lower IDSW",
        "next_stage": "M23-6-0 longer-horizon cross-tracklet identity graph oracle using segment/episode actions rather than frame-local edits",
        "deployment_allowed": False,
        "locked_manifest_created": False,
        "p15_policy": "no_op; 156 locked rows untouched",
    }

    payload = {
        "schema": "fmtrack.m23_5.unified_audit.v1",
        "all_checks_passed": all(item["passed"] for item in checks),
        "checks": len(checks), "check_results": checks,
        "interpretation": interpretation, "research_decision": research_decision,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_checks_passed": payload["all_checks_passed"], "checks": len(checks),
        "interpretation": interpretation, "research_decision": research_decision,
        "output_sha256": sha256_file(out),
    }, indent=2, sort_keys=True))
    if not payload["all_checks_passed"]:
        failed = [item for item in checks if not item["passed"]]
        raise SystemExit(f"unified audit failed: {failed}")


if __name__ == "__main__":
    main()
