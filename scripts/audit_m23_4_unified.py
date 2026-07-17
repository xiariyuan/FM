"""Unified reproducibility and decision audit for M23-4-0 gap-bridge oracle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
VARIANTS = (
    "baseline_raw",
    "dense_replacement_context",
    "all_existing_id_gap_bridge_oracle",
    "safe_existing_id_gap_bridge_oracle",
)
COMPACT_FILES = (
    "gap_pairs.csv", "gap_summary.csv", "bridge_events.csv",
    "per_sequence_metrics.csv", "variant_metrics.csv", "report.json", "manifest.json",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_key(line: str) -> tuple[int, int]:
    fields = line.split(",", 2)
    return int(fields[0]), int(fields[1])


def load_unique_tracker(path: Path) -> dict[tuple[int, int], str]:
    rows: dict[tuple[int, int], str] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            key = parse_key(line)
            if key in rows:
                raise RuntimeError(f"duplicate frame-ID in {path}: {key}")
            rows[key] = line
    return rows


def audit_extension(path: Path, source: dict[tuple[int, int], str]) -> dict[str, int]:
    seen_source: set[tuple[int, int]] = set()
    seen_all: set[tuple[int, int]] = set()
    additions = altered = rows = 0
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            rows += 1
            key = parse_key(line)
            if key in seen_all:
                raise RuntimeError(f"duplicate frame-ID in {path}: {key}")
            seen_all.add(key)
            if key in source:
                seen_source.add(key)
                altered += int(source[key] != line)
            else:
                additions += 1
    return {
        "rows": rows,
        "additions": additions,
        "altered_source_rows": altered,
        "missing_source_rows": len(source) - len(seen_source),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal-dir", default="outputs/mot20_m23_20260717/existing_id_gap_bridge_oracle_v1")
    ap.add_argument("--repro-dir", default="outputs/mot20_m23_20260717/existing_id_gap_bridge_oracle_v1_repro")
    ap.add_argument("--preregister", default="outputs/mot20_m23_20260717/existing_id_gap_bridge_oracle_preregister_v1.json")
    ap.add_argument("--script", default="scripts/audit_m23_4_existing_id_gap_bridge_oracle.py")
    ap.add_argument("--replacement-dir", default="outputs/mot20_m23_20260717/deployable_replacement_oracle_v1")
    ap.add_argument("--out", default="deliverables/mot20_m23_4_audit_20260717.json")
    args = ap.parse_args()

    repo = Path.cwd()
    formal, repro = repo / args.formal_dir, repo / args.repro_dir
    prereg = load_json(repo / args.preregister)
    report = load_json(formal / "report.json")
    manifest = load_json(formal / "manifest.json")
    replacement = repo / args.replacement_dir
    out = repo / args.out
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    script_sha = sha256_file(repo / args.script)
    check("script_sha_matches_preregister", script_sha == prereg["script_sha256"], {"actual": script_sha, "expected": prereg["script_sha256"]})
    check("formal_report_sha_matches_manifest", sha256_file(formal / "report.json") == manifest["report_sha256"])

    compact_identical = 0
    for name in COMPACT_FILES:
        fsha, rsha = sha256_file(formal / name), sha256_file(repro / name)
        same = fsha == rsha
        compact_identical += int(same)
        check(f"compact_byte_identical::{name}", same, {"formal": fsha, "repro": rsha})

    tracker_identical = 0
    for variant in VARIANTS:
        for sequence in SEQUENCES:
            fp = formal / "eval_work/trackers" / variant / "data" / f"{sequence}.txt"
            rp = repro / "eval_work/trackers" / variant / "data" / f"{sequence}.txt"
            fsha, rsha = sha256_file(fp), sha256_file(rp)
            same = fsha == rsha
            tracker_identical += int(same)
            check(f"tracker_byte_identical::{variant}::{sequence}", same, {"formal": fsha, "repro": rsha})

    trackeval_identical = 0
    for variant in VARIANTS:
        for name in ("pedestrian_summary.txt", "pedestrian_detailed.csv"):
            fp = formal / "eval_work/eval" / variant / name
            rp = repro / "eval_work/eval" / variant / name
            fsha, rsha = sha256_file(fp), sha256_file(rp)
            same = fsha == rsha
            trackeval_identical += int(same)
            check(f"trackeval_byte_identical::{variant}::{name}", same, {"formal": fsha, "repro": rsha})

    events = read_csv(formal / "bridge_events.csv")
    safe_events = [r for r in events if r["context_exclusive"] == "True"]
    per_seq_all = {s: sum(r["sequence"] == s for r in events) for s in SEQUENCES}
    per_seq_safe = {s: sum(r["sequence"] == s for r in safe_events) for s in SEQUENCES}

    for sequence in SEQUENCES:
        baseline = formal / "eval_work/trackers/baseline_raw/data" / f"{sequence}.txt"
        dense = formal / "eval_work/trackers/dense_replacement_context/data" / f"{sequence}.txt"
        source_baseline = repo / "outputs/trusttrack_review_20260710/all4_baseline_oracle/baseline/eval_work/trackers/all4_baseline/data" / f"{sequence}.txt"
        source_dense = replacement / "eval_work/trackers/dense_replacement_oracle/data" / f"{sequence}.txt"
        check(f"baseline_source_exact::{sequence}", sha256_file(baseline) == sha256_file(source_baseline))
        check(f"dense_source_exact::{sequence}", sha256_file(dense) == sha256_file(source_dense))
        source = load_unique_tracker(dense)
        for variant, expected in (
            ("all_existing_id_gap_bridge_oracle", per_seq_all[sequence]),
            ("safe_existing_id_gap_bridge_oracle", per_seq_safe[sequence]),
        ):
            result = audit_extension(formal / "eval_work/trackers" / variant / "data" / f"{sequence}.txt", source)
            check(f"extension_exact::{variant}::{sequence}", result["additions"] == expected and result["altered_source_rows"] == 0 and result["missing_source_rows"] == 0, {**result, "expected_additions": expected})

    check("all_plan_event_count", len(events) == 3825, len(events))
    check("safe_plan_event_count", len(safe_events) == 829, len(safe_events))
    check("safe_events_context_iou_lt_0_5", all(float(r["maximum_context_iou"]) < 0.5 for r in safe_events))
    check("all_event_frame_inherited_unique", len({(r["sequence"], r["frame"], r["inherited_track_id"]) for r in events}) == len(events))
    check("all_gap_lengths_fixed_range", all(1 <= int(r["gap_frames"]) <= 30 for r in events))
    check("all_target_iou_at_least_0_5", all(float(r["target_iou"]) >= 0.5 for r in events))

    d = report["decision"]
    check("baseline_reproduced", d["baseline_reproduced"] is True)
    check("dense_context_reproduced", d["dense_context_reproduced"] is True)
    check("safe_gate_failed", d["safe_additions_combined_hota_passed"] is False)
    check("safe_all_sequences_nonnegative", d["safe_additions_all_sequence_hota_nonnegative"] is True)
    check("ceiling_closed", d["internal_gap_bridge_ceiling_passed"] is False)
    check("deployment_disallowed", d["deployment_allowed"] is False)
    check("locked_manifest_absent", d["locked_manifest_created"] is False)
    proto = report["protocol"]
    check("no_locked_access", proto["locked_label_reads"] == 0 and proto["locked_trackeval_calls"] == 0)
    check("no_training_or_sweeps", proto["models_trained"] == 0 and proto["threshold_sweeps"] == 0 and proto["parameter_sweeps"] == 0 and proto["gap_sweeps"] == 0)

    metrics = {(r["variant"], r["sequence"]): r for r in read_csv(formal / "per_sequence_metrics.csv")}
    dense_m = metrics[("dense_replacement_context", "COMBINED")]
    safe_m = metrics[("safe_existing_id_gap_bridge_oracle", "COMBINED")]
    all_m = metrics[("all_existing_id_gap_bridge_oracle", "COMBINED")]
    overlapping = sum(float(r["maximum_context_iou"]) >= 0.5 for r in events)
    interpretation = {
        "safe_hota_gain_over_dense": float(safe_m["HOTA"]) - float(dense_m["HOTA"]),
        "all_hota_gain_over_dense": float(all_m["HOTA"]) - float(dense_m["HOTA"]),
        "safe_fn_reduction": int(dense_m["CLR_FN"]) - int(safe_m["CLR_FN"]),
        "safe_fp_change": int(safe_m["CLR_FP"]) - int(dense_m["CLR_FP"]),
        "safe_idsw_change": int(safe_m["IDSW"]) - int(dense_m["IDSW"]),
        "all_fn_reduction": int(dense_m["CLR_FN"]) - int(all_m["CLR_FN"]),
        "all_fp_change": int(all_m["CLR_FP"]) - int(dense_m["CLR_FP"]),
        "all_idsw_change": int(all_m["IDSW"]) - int(dense_m["IDSW"]),
        "overlapping_conflict_events": overlapping,
        "context_exclusive_events": len(safe_events),
        "conflict_fraction": overlapping / len(events),
        "formal_repro_compact_identical": compact_identical,
        "formal_repro_tracker_identical": tracker_identical,
        "formal_repro_trackeval_identical": trackeval_identical,
    }
    research_decision = {
        "same_id_internal_gap_fill_closed": True,
        "do_not_train_same_id_gap_selector": True,
        "reason": "829 context-exclusive GT oracle additions reach only 78.03543 HOTA, below the preregistered 78.10 gate",
        "next_stage": "M23-5-0 cross-ID tracklet relinking/reassignment action-space oracle on the 2,996 overlapping conflict events",
        "deployment_allowed": False,
        "locked_manifest_created": False,
        "p15_policy": "no_op; 156 locked rows untouched",
    }
    payload = {
        "schema": "fmtrack.m23_4.unified_audit.v1",
        "all_checks_passed": all(x["passed"] for x in checks),
        "checks": len(checks),
        "check_results": checks,
        "interpretation": interpretation,
        "research_decision": research_decision,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"all_checks_passed": payload["all_checks_passed"], "checks": len(checks), "interpretation": interpretation, "research_decision": research_decision}, indent=2, sort_keys=True))
    if not payload["all_checks_passed"]:
        raise SystemExit("unified audit failed")


if __name__ == "__main__":
    main()
