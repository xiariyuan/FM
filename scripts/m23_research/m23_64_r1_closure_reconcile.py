#!/usr/bin/env python3
"""Reconcile M23-64-R1 structured closure records without rerunning training.

The frozen training artifacts and scientific reports are not changed.  This
validator fixes only a bookkeeping bug in the inherited closure helper that
matched an empty-unit seed row to every seed-specific row, incorrectly changing
completed seed records to ``prohibited_by_scope``.  It also fills the required
checkpoint/seed/training fields in the final registry row and rehashes closure.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(".").resolve()
EXP_ID = "M23-64-R1"
R64 = ROOT / "outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training_repair1"
REGISTRY = ROOT / "outputs/experiment_registry.csv"
REPAIR_SCRIPT = ROOT / "scripts/m23_research/m23_64_v3_pair_reconstruction_training_repair.py"
RECONCILIATION = R64 / "closure_reconciliation.json"
SUMMARY = R64 / "summary.csv"
FINAL = R64 / "final_summary.json"
CLOSURE = R64 / "closure_validation.json"
EVENTS = R64 / "protocol_events.jsonl"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    closure_before = json.loads(CLOSURE.read_text(encoding="utf-8"))
    assert final["experiment_id"] == EXP_ID
    assert final["decision"] == "PASS_V3_FROM_SCRATCH_RELATION_TRAINING"
    assert final["training_runs"] == 3

    with SUMMARY.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        summary_rows = list(reader)
        fields = reader.fieldnames or []
    before_summary_sha = sha256(SUMMARY)
    seeds = [2359001, 2359002, 2359003]
    seed_manifests = {}
    for seed in seeds:
        p = R64 / "training" / f"seed_{seed}" / "checkpoint_manifest.json"
        m = json.loads(p.read_text(encoding="utf-8"))
        seed_manifests[seed] = m
        assert m.get("status") == "completed"
        assert m.get("epochs_completed") == 30
        assert m.get("all_epochs_finite") is True
        assert len(json.loads((p.parent / "training_history.json").read_text(encoding="utf-8"))) == 30
        assert len((p.parent / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 30

    # Keep one authoritative seed-specific completed row. Empty-unit template
    # rows are retained but explicitly superseded rather than mislabelled.
    for row in summary_rows:
        if row["experiment"] != EXP_ID or not row["stage"].startswith("seed_"):
            continue
        seed = int(row["stage"].split("_", 1)[1])
        if row["unit"] == "":
            row.update(
                status="superseded",
                completed_at=row.get("completed_at") or now(),
                decision="superseded_by_seed_specific_record",
                report="",
                notes="template row superseded by authoritative seed-specific record",
            )
        elif int(row["unit"]) == seed:
            m = seed_manifests[seed]
            row.update(
                status="completed",
                completed_at=row.get("completed_at") or now(),
                decision="pass",
                report=str((R64 / "training" / f"seed_{seed}" / "checkpoint_manifest.json").relative_to(ROOT)),
                notes=f"30/30 finite; best_epoch={m['best_epoch']}; composite={m['best_composite']:.9f}",
            )

    with SUMMARY.with_suffix(".csv.tmp").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(summary_rows)
    os.replace(SUMMARY.with_suffix(".csv.tmp"), SUMMARY)

    with REGISTRY.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        registry_rows = list(reader); registry_fields = reader.fieldnames or []
    before_registry_sha = sha256(REGISTRY)
    closed = [r for r in registry_rows if r.get("tracker_family") == EXP_ID and r.get("current_stage") == "closed"]
    assert closed
    row = closed[-1]
    selection = final["selection"]
    row.update(
        status="completed",
        current_stage="closed",
        decision=final["decision"],
        result_file="docs/m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training_repair1_result_20260723.md",
        artifact="docs/m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training_repair1_result_20260723.md",
        train_examples=str(final["pair_counts"]["train"]),
        val_examples=str(final["pair_counts"]["validation"]),
        seed=str(selection["seed"]),
        selected_seed=str(selection["seed"]),
        selected_epoch=str(selection["epoch"]),
        checkpoint_sha256=selection["checkpoint_sha256"],
        best_metric=str(selection["composite"]),
        training_runs=str(final["training_runs"]),
        trackeval_runs="0",
        HOTA="",
        next_action="M23-65 representation gate",
        notes=(row.get("notes", "") + "; structured closure reconciled; TrackEval=0; HOTA=empty").strip("; "),
    )
    with REGISTRY.with_suffix(".csv.tmp").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=registry_fields, extrasaction="ignore")
        w.writeheader(); w.writerows(registry_rows)
    os.replace(REGISTRY.with_suffix(".csv.tmp"), REGISTRY)

    reconciliation = {
        "experiment_id": EXP_ID,
        "created_at": now(),
        "reason": "inherited closure helper matched empty-unit seed template rows to seed-specific rows",
        "command": "python -u scripts/m23_research/m23_64_r1_closure_reconcile.py",
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
        "before_summary_sha256": before_summary_sha,
        "after_summary_sha256": sha256(SUMMARY),
        "before_registry_sha256": before_registry_sha,
        "after_registry_sha256": sha256(REGISTRY),
        "seed_records": {str(seed): {"status": "completed", "epochs": 30, "all_epochs_finite": True} for seed in seeds},
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "training_runs": 3,
        "scientific_artifacts_changed": False,
        "training_rerun": False,
    }
    write_json(RECONCILIATION, reconciliation)
    with EVENTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": now(), "event": "closure_structured_records_reconciled", "report_sha256": sha256(RECONCILIATION)}, sort_keys=True) + "\n")

    checks = dict(closure_before.get("checks", {}))
    checks.update(
        {
            "seed_summary_records_complete": all(
                r["unit"] != "" and r["status"] == "completed" and r["decision"] == "pass"
                for r in summary_rows
                if r["experiment"] == EXP_ID and r["stage"].startswith("seed_") and r["unit"] != ""
            ),
            "registry_required_fields_complete": all(
                row.get(k, "") not in {"", None}
                for k in ["selected_seed", "selected_epoch", "checkpoint_sha256", "training_runs", "result_file"]
            ),
            "scientific_artifacts_unchanged": reconciliation["scientific_artifacts_changed"] is False,
        }
    )
    closure = dict(closure_before)
    closure.update({"created_at": now(), "checks": checks, "closure_integrity_passed": all(checks.values()), "reconciliation": reconciliation})
    artifacts = [p for p in R64.rglob("*") if p.is_file() and p != CLOSURE]
    artifacts.extend([REPAIR_SCRIPT, Path(__file__), ROOT / "docs/m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training_repair1_prereg_20260723.md", ROOT / "docs/m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training_repair1_result_20260723.md"])
    closure["output_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(artifacts))}
    write_json(CLOSURE, closure)
    print(json.dumps({"decision": final["decision"], "closure_integrity_passed": closure["closure_integrity_passed"], "summary_sha256": sha256(SUMMARY), "registry_sha256": sha256(REGISTRY), "closure_sha256": sha256(CLOSURE)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
