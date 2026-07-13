#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def ff(row, key):
    return float(row.get(key, 0.0) or 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-min", type=float, default=0.05)
    ap.add_argument("--aux-min", type=float, default=0.2)
    ap.add_argument("--mode", choices=["base", "aux", "mul"], default="mul")
    ap.add_argument("--summary", default="")
    args = ap.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            base = ff(row, "aflink_score")
            aux = ff(row, "validator_score")
            if base < args.base_min or aux < args.aux_min:
                continue
            rr = dict(row)
            rr["base_score_saved"] = base
            rr["aux_score_saved"] = aux
            rr["mul_score_saved"] = base * aux
            if args.mode == "aux":
                rr["aflink_score"] = aux
            elif args.mode == "mul":
                rr["aflink_score"] = base * aux
            else:
                rr["aflink_score"] = base
            rows.append(rr)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    else:
        out.write_text("", encoding="utf-8")

    by = {}
    tp = 0
    for row in rows:
        seq = row["seq"]
        y = int(float(row.get("same_gt", 0) or 0))
        tp += y
        by.setdefault(seq, {"selected": 0, "tp": 0})
        by[seq]["selected"] += 1
        by[seq]["tp"] += y
    for seq in by:
        by[seq]["precision"] = by[seq]["tp"] / by[seq]["selected"] if by[seq]["selected"] else 0.0
    summary = {
        "selected": len(rows),
        "tp": tp,
        "precision": tp / len(rows) if rows else 0.0,
        "base_min": args.base_min,
        "aux_min": args.aux_min,
        "mode": args.mode,
        "by_seq": by,
    }
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
