#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def score(row):
    return float(row.get("aflink_score", 0.0) or 0.0)


def annotate(rows):
    by_a = defaultdict(list)
    by_b = defaultdict(list)
    for i, row in enumerate(rows):
        seq = row["seq"]
        a = int(float(row["track_a"]))
        b = int(float(row["track_b"]))
        by_a[(seq, a)].append(i)
        by_b[(seq, b)].append(i)

    rank_a = {}
    rank_b = {}
    margin_a = {}
    margin_b = {}
    for group, idxs in by_a.items():
        ordered = sorted(idxs, key=lambda i: -score(rows[i]))
        top_score = score(rows[ordered[0]]) if ordered else 0.0
        second_score = score(rows[ordered[1]]) if len(ordered) > 1 else 0.0
        top_margin = top_score - second_score
        for rank, i in enumerate(ordered, start=1):
            rank_a[i] = rank
            margin_a[i] = top_margin if rank == 1 else score(rows[i]) - score(rows[ordered[rank - 2]])
    for group, idxs in by_b.items():
        ordered = sorted(idxs, key=lambda i: -score(rows[i]))
        top_score = score(rows[ordered[0]]) if ordered else 0.0
        second_score = score(rows[ordered[1]]) if len(ordered) > 1 else 0.0
        top_margin = top_score - second_score
        for rank, i in enumerate(ordered, start=1):
            rank_b[i] = rank
            margin_b[i] = top_margin if rank == 1 else score(rows[i]) - score(rows[ordered[rank - 2]])

    out = []
    for i, row in enumerate(rows):
        rr = dict(row)
        ra = rank_a.get(i, 999999)
        rb = rank_b.get(i, 999999)
        ma = margin_a.get(i, 0.0)
        mb = margin_b.get(i, 0.0)
        rr["rank_as_successor"] = ra
        rr["rank_as_predecessor"] = rb
        rr["score_margin_for_a"] = ma
        rr["score_margin_for_b"] = mb
        rr["min_score_margin"] = min(ma, mb)
        rr["is_mutual_top1"] = int(ra == 1 and rb == 1)
        rr["is_mutual_top2"] = int(ra <= 2 and rb <= 2)
        out.append(rr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--max-rank", type=int, default=0, help="0 means no rank filter; 1 mutual-top1; 2 mutual-top2")
    ap.add_argument("--min-margin", type=float, default=-999.0)
    ap.add_argument("--summary-json", default="")
    args = ap.parse_args()

    with open(args.input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = annotate(rows)

    selected = []
    by_seq = defaultdict(lambda: {"input": 0, "selected": 0})
    for row in rows:
        seq = row["seq"]
        by_seq[seq]["input"] += 1
        s = score(row)
        if s < args.min_score:
            continue
        if args.max_rank > 0:
            if int(row["rank_as_successor"]) > args.max_rank or int(row["rank_as_predecessor"]) > args.max_rank:
                continue
        if float(row["min_score_margin"]) < args.min_margin:
            continue
        selected.append(row)
        by_seq[seq]["selected"] += 1

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    summary = {
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "min_score": args.min_score,
        "max_rank": args.max_rank,
        "min_margin": args.min_margin,
        "by_seq": dict(by_seq),
    }
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
