#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDNAMES = [
    "rows_csv",
    "group_jsonl",
    "host_variant",
    "source_tag",
    "split_tag",
    "dataset_tag",
    "feature_version",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a manifest CSV for local-conflict set-predictor dataset sources.")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--rows-csv", required=True)
    parser.add_argument("--group-jsonl", required=True)
    parser.add_argument("--host-variant", default="base_reid_da")
    parser.add_argument("--source-tag", default="base_runtime")
    parser.add_argument("--split-tag", default="auto")
    parser.add_argument("--dataset-tag", default="local_conflict_set_predictor_large_base")
    parser.add_argument("--feature-version", default="v2_hostnorm_geom")
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def _load_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def main() -> int:
    args = parse_args()
    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "rows_csv": str(Path(args.rows_csv).resolve()),
        "group_jsonl": str(Path(args.group_jsonl).resolve()),
        "host_variant": str(args.host_variant),
        "source_tag": str(args.source_tag),
        "split_tag": str(args.split_tag),
        "dataset_tag": str(args.dataset_tag),
        "feature_version": str(args.feature_version),
    }

    rows = _load_existing_rows(out_csv) if bool(args.append) else []
    dedupe = {
        (
            str(existing.get("rows_csv", "")),
            str(existing.get("group_jsonl", "")),
            str(existing.get("source_tag", "")),
        )
        for existing in rows
    }
    key = (row["rows_csv"], row["group_jsonl"], row["source_tag"])
    if key not in dedupe:
        rows.append(row)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for existing in rows:
            writer.writerow({name: existing.get(name, "") for name in FIELDNAMES})

    summary = {
        "out_csv": str(out_csv),
        "rows": len(rows),
        "host_variants": sorted(
            {
                str(existing.get("host_variant", ""))
                for existing in rows
                if str(existing.get("host_variant", "")).strip()
            }
        ),
        "source_tags": sorted(
            {
                str(existing.get("source_tag", ""))
                for existing in rows
                if str(existing.get("source_tag", "")).strip()
            }
        ),
        "dataset_tags": sorted(
            {
                str(existing.get("dataset_tag", ""))
                for existing in rows
                if str(existing.get("dataset_tag", "")).strip()
            }
        ),
        "feature_versions": sorted(
            {
                str(existing.get("feature_version", ""))
                for existing in rows
                if str(existing.get("feature_version", "")).strip()
            }
        ),
        "status": "ok",
    }
    summary_path = out_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
