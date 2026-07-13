#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    report = {"files": {}, "total_duplicate_keys": 0, "total_duplicate_rows": 0, "ok": True}
    for txt in sorted(Path(args.results_dir).glob("MOT20-*.txt")):
        counts = defaultdict(int)
        rows = 0
        with txt.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                frame = int(float(parts[0]))
                track_id = int(float(parts[1]))
                counts[(frame, track_id)] += 1
                rows += 1
        duplicate_keys = {k: v for k, v in counts.items() if v > 1}
        duplicate_rows = sum(v - 1 for v in duplicate_keys.values())
        report["files"][txt.name] = {
            "rows": rows,
            "duplicate_keys": len(duplicate_keys),
            "duplicate_rows": duplicate_rows,
            "examples": [
                {"frame": int(k[0]), "track_id": int(k[1]), "count": int(v)}
                for k, v in list(duplicate_keys.items())[:20]
            ],
        }
        report["total_duplicate_keys"] += len(duplicate_keys)
        report["total_duplicate_rows"] += duplicate_rows
    report["ok"] = report["total_duplicate_keys"] == 0
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
