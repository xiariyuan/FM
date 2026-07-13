#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def ai(v, d=0):
    try:
        return int(float(v))
    except Exception:
        return d


def af(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def safe_label(row):
    wrong = ai(row.get("diag_wrong_rows"))
    skipped = ai(row.get("skipped_collision_rows"))
    applied = ai(row.get("applied_rows"))
    idsw = ai(row.get("IDSW", 999999), 999999)
    hota = af(row.get("HOTA"))
    idf1 = af(row.get("IDF1"))
    return int(wrong <= 1 and skipped == 0 and applied >= 10 and idsw <= 443 and hota >= 68.430 and idf1 >= 74.413)


def reject_reason(row):
    if ai(row.get("diag_wrong_rows")) > 1:
        return "wrong_rows"
    if ai(row.get("skipped_collision_rows")) > 0:
        return "collision_skips"
    if ai(row.get("applied_rows")) < 10:
        return "too_few_rows"
    if row.get("IDSW") and ai(row.get("IDSW"), 999999) > 443:
        return "idsw_worse"
    if row.get("HOTA") and af(row.get("HOTA")) < 68.430:
        return "hota_drop"
    return "safe"


def enrich(row):
    run_dir = Path(row.get("source_run_dir", ""))
    rw = {}
    if (run_dir / "rewrite_summary.json").exists():
        rw = json.loads((run_dir / "rewrite_summary.json").read_text(encoding="utf-8"))
    selected = read_csv(run_dir / "selected_fragments.csv")
    gaps = read_csv(run_dir / "gap_rows.csv")
    if rw:
        row["selected_fragments"] = "|".join(rw.get("selected_fragments", [])) or row.get("selected_fragments", "")
        for k_src, k_dst in [
            ("selected_fragment_count", "selected_fragment_count"),
            ("gap_row_count", "gap_row_count"),
            ("planned_rows", "planned_rows"),
            ("applied_rows", "applied_rows"),
            ("skipped_collision_rows", "skipped_collision_rows"),
            ("gt_diag_selected_rows_same_anchor", "diag_same_rows"),
            ("gt_diag_selected_rows_wrong_or_unknown", "diag_wrong_rows"),
            ("correct_fragment_count_diag", "correct_fragment_count_diag"),
            ("wrong_fragment_count_diag", "wrong_fragment_count_diag"),
            ("correct_gap_rows_diag", "correct_gap_rows_diag"),
            ("wrong_gap_rows_diag", "wrong_gap_rows_diag"),
        ]:
            if k_src in rw:
                row[k_dst] = rw[k_src]
    if selected:
        row["selected_fragment_count"] = len(selected)
        row["high_reid_fragment_count"] = sum(1 for r in selected if r.get("selected_stage") == "high_reid")
        row["bridge_fragment_count"] = sum(1 for r in selected if r.get("selected_stage") == "bridge_fragment")
        bridge = [r for r in selected if r.get("selected_stage") == "bridge_fragment"]
        if bridge:
            vals = [af(r.get("bridge_score")) for r in bridge]
            sims = [af(r.get("sim_to_anchor")) for r in bridge]
            row["max_bridge_score_selected"] = max(vals)
            row["min_bridge_sim_selected"] = min(sims)
    if gaps:
        row["gap_row_count"] = len(gaps)
        row["max_gap_dist"] = max(af(r.get("dist")) for r in gaps)
        row["min_gap_height_ratio"] = min(af(r.get("height_ratio")) for r in gaps)
        row["max_gap_height_ratio"] = max(af(r.get("height_ratio")) for r in gaps)
    row["label_safe_to_rewrite"] = safe_label(row)
    row["reject_reason"] = reject_reason(row)
    return row


def main():
    ap = argparse.ArgumentParser(description="Finalize/enrich A39_04a matcher dataset path examples from rewrite summaries.")
    ap.add_argument("--dataset-dir", default="outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_04_learned_path_gate_dataset")
    args = ap.parse_args()
    d = Path(args.dataset_dir)
    path_rows = [enrich(dict(r)) for r in read_csv(d / "path_transaction_examples.csv")]
    # Write enriched all examples.
    fields = []
    seen = set()
    for r in path_rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    write_csv(d / "path_transaction_examples_enriched.csv", fields, path_rows)
    # Deduplicate by anchor_id, keeping the richest run: prefer non-hard-negative explicit positive/unsafe recovered, then max applied rows.
    priority = {
        "A39_03e1_ultrasafe_positive": 4,
        "A39_03e0_positive": 3,
        "A39_03h_recovered_unsafe": 2,
        "A39_03f0_hard_negative": 1,
    }
    best = {}
    for r in path_rows:
        aid = r.get("anchor_id")
        key = (priority.get(r.get("source_family"), 0), ai(r.get("applied_rows")), -ai(r.get("diag_wrong_rows")))
        if aid not in best or key > best[aid][0]:
            best[aid] = (key, r)
    dedup = [x[1] for x in best.values()]
    write_csv(d / "path_transaction_examples_dedup.csv", fields, sorted(dedup, key=lambda r: r.get("anchor_id", "")))
    summary = {
        "path_examples_enriched": len(path_rows),
        "path_examples_dedup": len(dedup),
        "safe_enriched": sum(ai(r.get("label_safe_to_rewrite")) for r in path_rows),
        "unsafe_enriched": sum(1 - ai(r.get("label_safe_to_rewrite")) for r in path_rows),
        "safe_dedup": sum(ai(r.get("label_safe_to_rewrite")) for r in dedup),
        "unsafe_dedup": sum(1 - ai(r.get("label_safe_to_rewrite")) for r in dedup),
    }
    (d / "finalized_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
