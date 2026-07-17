"""Unified reproducibility and decision audit for M23-6-0 tracklet chains."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
VARIANTS = (
    "all_additions_context", "pure_segment_chain_oracle", "modal_segment_chain_oracle",
    "modal_core_chain_oracle", "full_matched_identity_ceiling",
)
COMPACT_FILES = (
    "chain_summary.csv", "chain_segments.csv", "per_sequence_metrics.csv",
    "variant_metrics.csv", "report.json", "manifest.json",
)
SYNTHETIC_BASE = 10_000_000
MAX_EXACT_FLOAT32_INTEGER = 1 << 24


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


def parse_fields(line: str) -> list[str]:
    fields = line.rstrip("\n").split(",")
    if len(fields) < 7:
        raise RuntimeError(f"invalid tracker line: {line!r}")
    return fields


def audit_variant(source_path: Path, variant_path: Path) -> dict[str, int]:
    changed = rows = synthetic_changed = 0
    seen: set[tuple[int, int]] = set()
    previous_frame = -1
    with source_path.open("r", encoding="utf-8") as source, variant_path.open("r", encoding="utf-8") as variant:
        for source_line, variant_line in zip(source, variant):
            source_fields, variant_fields = parse_fields(source_line), parse_fields(variant_line)
            rows += 1
            if source_fields[0] != variant_fields[0] or source_fields[2:] != variant_fields[2:]:
                raise RuntimeError(f"non-identity field changed in {variant_path} at row {rows}")
            frame, identity = int(variant_fields[0]), int(variant_fields[1])
            if frame < previous_frame:
                raise RuntimeError(f"non-monotonic frames in {variant_path}")
            previous_frame = frame
            key = (frame, identity)
            if key in seen:
                raise RuntimeError(f"duplicate frame/ID {key} in {variant_path}")
            seen.add(key)
            if source_fields[1] != variant_fields[1]:
                changed += 1
                synthetic_changed += int(SYNTHETIC_BASE <= identity < MAX_EXACT_FLOAT32_INTEGER)
        if source.readline() or variant.readline():
            raise RuntimeError(f"row count mismatch: {source_path} vs {variant_path}")
    return {"rows": rows, "changed_ids": changed, "synthetic_changed_ids": synthetic_changed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-dir", default="outputs/mot20_m23_20260717/long_horizon_tracklet_chain_v1")
    parser.add_argument("--repro-dir", default="outputs/mot20_m23_20260717/long_horizon_tracklet_chain_v1_repro")
    parser.add_argument("--preregister", default="outputs/mot20_m23_20260717/long_horizon_tracklet_chain_preregister_v1.json")
    parser.add_argument("--script", default="scripts/audit_m23_6_long_horizon_tracklet_chain_oracle.py")
    parser.add_argument("--parent-dir", default="outputs/mot20_m23_20260717/existing_id_gap_bridge_oracle_v1")
    parser.add_argument("--out", default="deliverables/mot20_m23_6_audit_20260718.json")
    args = parser.parse_args()

    repo = Path.cwd()
    formal, repro = repo / args.formal_dir, repo / args.repro_dir
    prereg = load_json(repo / args.preregister)
    report = load_json(formal / "report.json")
    manifest = load_json(formal / "manifest.json")
    parent = repo / args.parent_dir
    out = repo / args.out
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    script_sha = sha256_file(repo / args.script)
    check("script_sha_matches_preregister", script_sha == prereg["script_sha256"], {"actual": script_sha, "expected": prereg["script_sha256"]})
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

    summary = report["counts"]
    expected_changed = {
        "all_additions_context": 0,
        "pure_segment_chain_oracle": int(summary["pure_chain_rows"]),
        "modal_segment_chain_oracle": int(summary["modal_chain_rows"]),
        "modal_core_chain_oracle": int(summary["modal_core_row_actions"]),
        "full_matched_identity_ceiling": int(summary["full_identity_row_actions"]),
    }
    changed_totals = Counter()
    for sequence in SEQUENCES:
        source = formal / "eval_work/trackers/all_additions_context/data" / f"{sequence}.txt"
        parent_source = parent / "eval_work/trackers/all_existing_id_gap_bridge_oracle/data" / f"{sequence}.txt"
        check(f"source_tracker_exact::{sequence}", sha256_file(source) == sha256_file(parent_source))
        for variant in VARIANTS:
            path = formal / "eval_work/trackers" / variant / "data" / f"{sequence}.txt"
            result = audit_variant(source, path)
            changed_totals[variant] += result["changed_ids"]
            check(f"identity_only_semantics::{variant}::{sequence}", result["changed_ids"] == result["synthetic_changed_ids"], result)
            check(f"row_count_preserved::{variant}::{sequence}", result["rows"] > 0, result["rows"])
    for variant, expected in expected_changed.items():
        check(f"changed_row_count::{variant}", changed_totals[variant] == expected, {"actual": changed_totals[variant], "expected": expected})

    segment_rows = read_csv(formal / "chain_segments.csv")
    check("segment_row_count", len(segment_rows) == int(summary["segments_total"]), len(segment_rows))
    pure_selected = [row for row in segment_rows if row["selected_pure_chain"] == "True"]
    modal_selected = [row for row in segment_rows if row["selected_modal_chain"] == "True"]
    check("pure_selected_count", len(pure_selected) == int(summary["pure_chain_segments"]), len(pure_selected))
    check("modal_selected_count", len(modal_selected) == int(summary["modal_chain_segments"]), len(modal_selected))
    check("pure_selection_is_pure", all(float(row["modal_purity"]) == 1.0 and int(row["modal_gt_id"]) > 0 for row in pure_selected))
    check("modal_selection_positive", all(int(row["modal_gt_id"]) > 0 for row in modal_selected))
    check("synthetic_segment_ids_exact_range", all(SYNTHETIC_BASE <= int(row["synthetic_chain_id"]) < MAX_EXACT_FLOAT32_INTEGER for row in segment_rows if int(row["modal_gt_id"]) > 0))
    # Selected intervals must be non-overlapping within each sequence/GT.
    grouped: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for row in modal_selected:
        grouped.setdefault((row["sequence"], int(row["modal_gt_id"])), []).append((int(row["first_frame"]), int(row["last_frame"])))
    nonoverlap = True
    for intervals in grouped.values():
        intervals.sort()
        nonoverlap &= all(a[1] < b[0] for a, b in zip(intervals, intervals[1:]))
    check("modal_selected_nonoverlap", nonoverlap)

    decision = report["decision"]
    check("source_reproduced", decision["source_context_reproduced"] is True)
    check("modal_segment_gate_failed", decision["modal_segment_gate_passed"] is False)
    check("modal_core_gate_failed", decision["modal_core_gate_passed"] is False)
    check("full_identity_gate_passed", decision["full_identity_gate_passed"] is True)
    check("overall_ceiling_failed", decision["long_horizon_chain_ceiling_passed"] is False)
    check("deployment_disallowed", decision["deployment_allowed"] is False)
    check("locked_manifest_absent", decision["locked_manifest_created"] is False)
    check("all_hota_deltas_positive", all(value > 0.0 for key in (
        "modal_segment_sequence_hota_deltas", "modal_core_sequence_hota_deltas", "full_identity_sequence_hota_deltas"
    ) for value in decision[key].values()))
    check("modal_segment_hota_threshold_pass_only", decision["modal_segment_hota"] >= float(report["gates"]["modal_segment_combined_hota_min"]) and decision["modal_segment_idsw"] > int(report["gates"]["modal_segment_idsw_max"]))
    check("modal_core_hota_threshold_pass_only", decision["modal_core_hota"] >= float(report["gates"]["modal_core_combined_hota_min"]) and decision["modal_core_idsw"] > int(report["gates"]["modal_core_idsw_max"]))
    protocol = report["protocol"]
    check("no_row_or_box_changes", protocol["row_additions"] == 0 and protocol["row_deletions"] == 0 and protocol["box_changes"] == 0)
    check("no_locked_access", protocol["locked_label_reads"] == 0 and protocol["locked_trackeval_calls"] == 0)
    check("no_models_or_sweeps", protocol["models_trained"] == 0 and protocol["threshold_sweeps"] == 0 and protocol["parameter_sweeps"] == 0)

    metrics = {(row["variant"], row["sequence"]): row for row in read_csv(formal / "per_sequence_metrics.csv")}
    def metric(variant: str, field: str) -> float:
        return float(metrics[(variant, "COMBINED")][field])
    interpretation = {
        "pure_hota_gain": metric("pure_segment_chain_oracle", "HOTA") - metric("all_additions_context", "HOTA"),
        "modal_segment_hota_gain": metric("modal_segment_chain_oracle", "HOTA") - metric("all_additions_context", "HOTA"),
        "modal_core_hota_gain": metric("modal_core_chain_oracle", "HOTA") - metric("all_additions_context", "HOTA"),
        "full_identity_hota_gain": metric("full_matched_identity_ceiling", "HOTA") - metric("all_additions_context", "HOTA"),
        "modal_segment_idsw_change": int(metric("modal_segment_chain_oracle", "IDSW") - metric("all_additions_context", "IDSW")),
        "modal_core_idsw_change": int(metric("modal_core_chain_oracle", "IDSW") - metric("all_additions_context", "IDSW")),
        "full_identity_idsw_change": int(metric("full_matched_identity_ceiling", "IDSW") - metric("all_additions_context", "IDSW")),
        "modal_core_to_full_hota_gap": metric("full_matched_identity_ceiling", "HOTA") - metric("modal_core_chain_oracle", "HOTA"),
        "formal_repro_compact_identical": compact_identical,
        "formal_repro_tracker_identical": tracker_identical,
        "formal_repro_trackeval_identical": trackeval_identical,
    }
    research_decision = {
        "current_segment_chain_representation_closed": True,
        "do_not_train_current_graph": True,
        "reason": "modal segment reaches 80.99495 but IDSW 949; modal core reaches 82.298195 but IDSW 2623, violating preregistered identity-safety gates",
        "identity_target_feasible": True,
        "evidence": "full matched identity ceiling reaches 83.08602 HOTA with IDSW 4",
        "next_stage": "M23-7-0 purified support-span chain oracle: other-GT hard boundaries, unmatched-hole filling, span-level WIS relinking",
        "deployment_allowed": False,
        "locked_manifest_created": False,
        "p15_policy": "no_op; 156 locked rows untouched",
    }

    payload = {
        "schema": "fmtrack.m23_6.unified_audit.v1",
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
