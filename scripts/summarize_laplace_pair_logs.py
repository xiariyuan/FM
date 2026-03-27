#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


REL_BUCKETS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
GAP_BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 999999)]
HISTORY_BUCKETS = [(0, 1), (2, 2), (3, 4), (5, 8), (9, 999999)]
AMB_BUCKETS = [0.0, 0.05, 0.10, 0.20, 0.40, 1.01]


def rel_bucket_label(value: float) -> str:
    for left, right in zip(REL_BUCKETS[:-1], REL_BUCKETS[1:]):
        if left <= value < right:
            return f"{left:.1f}-{min(right, 1.0):.1f}"
    return "unknown"


def gap_bucket_label(value: int) -> str:
    for left, right in GAP_BUCKETS:
        if left <= value <= right:
            return f"{left}" if left == right else f"{left}-{right if right < 999999 else '+'}"
    return "unknown"


def history_bucket_label(value: int) -> str:
    for left, right in HISTORY_BUCKETS:
        if left <= value <= right:
            return f"{left}" if left == right else f"{left}-{right if right < 999999 else '+'}"
    return "unknown"


def amb_bucket_label(value: float) -> str:
    for left, right in zip(AMB_BUCKETS[:-1], AMB_BUCKETS[1:]):
        if left <= value < right:
            return f"{left:.2f}-{min(right, 1.0):.2f}"
    return "unknown"


def mean(values):
    vals = [v for v in values if v == v]
    return sum(vals) / len(vals) if vals else 0.0


def _to_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def choose_trust_key(rows: list[dict[str, object]]) -> str:
    for row in rows:
        value = row.get("learned_r", float("nan"))
        if isinstance(value, float) and value == value:
            return "learned_r"
    return "pair_rel"


def pair_ambiguity(row: dict[str, object]) -> float:
    vals = [
        float(row.get("amb_spa", float("nan"))),
        float(row.get("amb_lap", float("nan"))),
        float(row.get("amb_mot", float("nan"))),
    ]
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan")
    return min(vals)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize LTRA pairwise analysis logs.")
    parser.add_argument("csv_path", help="Path to *_pairs.csv log")
    parser.add_argument("--out-dir", required=True, help="Directory to store summary csv files")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["gap"] = int(row["gap"])
            row["history_len"] = int(row.get("history_len", 0))
            row["chosen"] = int(row["chosen"])
            row["is_true_match"] = int(row["is_true_match"])
            row["track_gt_id"] = int(row["track_gt_id"])
            row["det_gt_id"] = int(row["det_gt_id"])
            for key in [
                "pair_rel",
                "learned_alpha",
                "learned_r",
                "appearance_sim",
                "fused_sim",
                "motion_sim",
                "spatial_sim",
                "laplace_sim",
                "agreement",
                "stability",
                "coherence",
                "det_score",
                "prod_sim",
                "amb_spa",
                "amb_lap",
                "amb_mot",
            ]:
                row[key] = _to_float(row, key)
            rows.append(row)

    valid_rows = [row for row in rows if row["track_gt_id"] > 0]
    chosen_rows = [row for row in valid_rows if row["chosen"] == 1]
    trust_key = choose_trust_key(valid_rows)
    trust_source = "learned_r" if trust_key == "learned_r" else "pair_rel"

    rel_groups = defaultdict(list)
    for row in valid_rows:
        rel_groups[rel_bucket_label(row[trust_key])].append(row)

    rel_summary = []
    for bucket in sorted(rel_groups.keys()):
        group = rel_groups[bucket]
        chosen_group = [row for row in group if row["chosen"] == 1]
        rel_summary.append(
            {
                "bucket": bucket,
                "trust_source": trust_source,
                "pairs": len(group),
                "true_rate": f"{mean([row['is_true_match'] for row in group]):.6f}",
                "chosen_rate": f"{mean([row['chosen'] for row in group]):.6f}",
                "chosen_true_rate": f"{mean([row['is_true_match'] for row in chosen_group]):.6f}" if chosen_group else "",
                "avg_trust": f"{mean([row[trust_key] for row in group]):.6f}",
                "avg_motion_sim": f"{mean([row['motion_sim'] for row in group]):.6f}",
                "avg_appearance_sim": f"{mean([row['appearance_sim'] for row in group]):.6f}",
            }
        )

    gap_groups = defaultdict(list)
    for row in chosen_rows:
        gap_groups[gap_bucket_label(row["gap"])].append(row)

    gap_summary = []
    for bucket in sorted(gap_groups.keys(), key=lambda x: (999999 if x == "unknown" else int(x.split("-")[0].replace("+", "")))):
        group = gap_groups[bucket]
        gap_summary.append(
            {
                "bucket": bucket,
                "trust_source": trust_source,
                "matches": len(group),
                "correct_rate": f"{mean([row['is_true_match'] for row in group]):.6f}",
                "avg_trust": f"{mean([row[trust_key] for row in group]):.6f}",
                "avg_pair_rel": f"{mean([row['pair_rel'] for row in group]):.6f}",
                "avg_learned_r": f"{mean([row['learned_r'] for row in group]):.6f}",
                "avg_motion_sim": f"{mean([row['motion_sim'] for row in group]):.6f}",
                "avg_appearance_sim": f"{mean([row['appearance_sim'] for row in group]):.6f}",
            }
        )

    history_groups = defaultdict(list)
    for row in chosen_rows:
        history_groups[history_bucket_label(row["history_len"])].append(row)

    history_summary = []
    for bucket in sorted(history_groups.keys(), key=lambda x: (999999 if x == "unknown" else int(x.split("-")[0].replace("+", "")))):
        group = history_groups[bucket]
        history_summary.append(
            {
                "bucket": bucket,
                "trust_source": trust_source,
                "matches": len(group),
                "correct_rate": f"{mean([row['is_true_match'] for row in group]):.6f}",
                "avg_trust": f"{mean([row[trust_key] for row in group]):.6f}",
                "avg_pair_rel": f"{mean([row['pair_rel'] for row in group]):.6f}",
                "avg_learned_r": f"{mean([row['learned_r'] for row in group]):.6f}",
                "avg_amb_spa": f"{mean([row['amb_spa'] for row in group]):.6f}",
                "avg_amb_lap": f"{mean([row['amb_lap'] for row in group]):.6f}",
                "avg_amb_mot": f"{mean([row['amb_mot'] for row in group]):.6f}",
            }
        )

    ambiguity_groups = defaultdict(list)
    for row in chosen_rows:
        amb_value = pair_ambiguity(row)
        if amb_value == amb_value:
            ambiguity_groups[amb_bucket_label(amb_value)].append(row)

    ambiguity_summary = []
    for bucket in sorted(ambiguity_groups.keys()):
        group = ambiguity_groups[bucket]
        ambiguity_summary.append(
            {
                "bucket": bucket,
                "trust_source": trust_source,
                "matches": len(group),
                "correct_rate": f"{mean([row['is_true_match'] for row in group]):.6f}",
                "avg_trust": f"{mean([row[trust_key] for row in group]):.6f}",
                "avg_pair_rel": f"{mean([row['pair_rel'] for row in group]):.6f}",
                "avg_learned_r": f"{mean([row['learned_r'] for row in group]):.6f}",
                "avg_pair_ambiguity": f"{mean([pair_ambiguity(row) for row in group]):.6f}",
                "avg_motion_sim": f"{mean([row['motion_sim'] for row in group]):.6f}",
                "avg_appearance_sim": f"{mean([row['appearance_sim'] for row in group]):.6f}",
            }
        )

    outputs = [
        ("reliability_calibration.csv", rel_summary),
        ("gap_bucket_summary.csv", gap_summary),
        ("history_bucket_summary.csv", history_summary),
        ("ambiguity_bucket_summary.csv", ambiguity_summary),
    ]
    if trust_key == "learned_r":
        outputs.extend(
            [
                ("learned_r_calibration.csv", rel_summary),
                ("learned_r_gap_bucket_summary.csv", gap_summary),
            ]
        )

    for name, summary in outputs:
        out_path = out_dir / name
        if not summary:
            out_path.write_text("")
            continue
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
