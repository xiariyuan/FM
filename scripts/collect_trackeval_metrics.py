#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


def find_summary_file(run_dir: Path) -> Path | None:
    if run_dir.is_file():
        return run_dir
    candidates = sorted(run_dir.glob("tracker/*/pedestrian_summary.txt"))
    if candidates:
        return candidates[0]
    candidates = sorted(run_dir.glob("**/pedestrian_summary.txt"))
    if candidates:
        return candidates[0]
    return None


def parse_summary(path: Path) -> dict[str, str]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Invalid summary file: {path}")
    keys = lines[0].split()
    values = lines[1].split()
    if len(values) < len(keys):
        raise ValueError(f"Summary row shorter than header: {path}")
    metrics = dict(zip(keys, values))
    metrics["summary_file"] = str(path)
    metrics["dataset_split"] = path.parent.name
    metrics["run_dir"] = str(path.parent.parent.parent)
    metrics["name"] = path.parent.parent.parent.name
    return metrics


def select_fields(rows: list[dict[str, str]], requested: Iterable[str]) -> list[str]:
    default_fields = [
        "name",
        "HOTA",
        "DetA",
        "AssA",
        "IDF1",
        "MOTA",
        "IDSW",
        "Frag",
        "dataset_split",
        "run_dir",
    ]
    if requested:
        return list(requested)
    fields: list[str] = []
    for field in default_fields:
        if any(field in row for row in rows):
            fields.append(field)
    return fields


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect TrackEval summary metrics into a compact table.")
    parser.add_argument("paths", nargs="+", help="Run directories or pedestrian_summary.txt files")
    parser.add_argument("--csv", dest="csv_path", help="Optional CSV output path")
    parser.add_argument("--fields", nargs="*", default=None, help="Fields to print/export")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for item in args.paths:
        summary = find_summary_file(Path(item))
        if summary is None:
            print(f"[skip] summary not found: {item}")
            continue
        rows.append(parse_summary(summary))

    if not rows:
        raise SystemExit("No valid TrackEval summaries found.")

    fields = select_fields(rows, args.fields or [])
    widths = {field: max(len(field), *(len(row.get(field, "")) for row in rows)) for field in fields}
    header = "  ".join(field.ljust(widths[field]) for field in fields)
    print(header)
    print("  ".join("-" * widths[field] for field in fields))
    for row in rows:
        print("  ".join(row.get(field, "").ljust(widths[field]) for field in fields))

    if args.csv_path:
        csv_path = Path(args.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
        print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
