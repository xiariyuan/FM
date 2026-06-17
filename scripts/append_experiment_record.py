#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
import fcntl


CORE_FIELDS = [
    "timestamp",
    "kind",
    "status",
    "script",
    "dataset",
    "split",
    "tracker_family",
    "variant",
    "tag",
    "run_root",
    "summary_csv",
    "checkpoint",
    "calibrator_npz",
    "log_path",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append experiment results to a central CSV registry.")
    parser.add_argument("--csv", required=True, help="Registry CSV path")
    parser.add_argument("--kind", required=True, choices=["train", "eval", "analysis", "queue", "other"])
    parser.add_argument("--status", default="success")
    parser.add_argument("--script", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--tracker-family", dest="tracker_family", default="")
    parser.add_argument("--variant", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--run-root", dest="run_root", default="")
    parser.add_argument("--summary-csv", dest="summary_csv", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--calibrator-npz", dest="calibrator_npz", default="")
    parser.add_argument("--log-path", dest="log_path", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--extra", nargs="*", default=[], help="Extra key=value fields to store")
    return parser.parse_args()


def parse_extra(extra_items: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in extra_items:
        if "=" not in item:
            raise ValueError(f"Invalid --extra item (expected key=value): {item}")
        key, value = item.split("=", 1)
        values[key] = value
    return values


def base_record(args: argparse.Namespace) -> dict[str, str]:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "kind": args.kind,
        "status": args.status,
        "script": args.script,
        "dataset": args.dataset,
        "split": args.split,
        "tracker_family": args.tracker_family,
        "variant": args.variant,
        "tag": args.tag,
        "run_root": args.run_root,
        "summary_csv": args.summary_csv,
        "checkpoint": args.checkpoint,
        "calibrator_npz": args.calibrator_npz,
        "log_path": args.log_path,
        "notes": args.notes,
    }
    record.update(parse_extra(args.extra))
    return record


def summary_rows(summary_csv: str) -> list[dict[str, str]]:
    if not summary_csv:
        return []
    path = Path(summary_csv)
    if not path.is_file():
        return []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def merge_rows(base: dict[str, str], summary: list[dict[str, str]]) -> list[dict[str, str]]:
    if not summary:
        return [base]
    rows: list[dict[str, str]] = []
    for row in summary:
        merged = dict(base)
        merged.update(row)
        rows.append(merged)
    return rows


def ordered_fields(existing_rows: list[dict[str, str]], new_rows: list[dict[str, str]]) -> list[str]:
    fields = list(CORE_FIELDS)
    seen = set(fields)
    for rows in (existing_rows, new_rows):
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
    return fields


def normalize_pathish(value: str) -> str:
    if not value:
        return ""
    try:
        return str(Path(value).expanduser().resolve(strict=False))
    except Exception:
        return value


def dedupe_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("kind", ""),
        row.get("script", ""),
        row.get("dataset", ""),
        row.get("split", ""),
        row.get("tracker_family", ""),
        row.get("variant", ""),
        normalize_pathish(row.get("run_root", "")),
        normalize_pathish(row.get("summary_csv", "")),
        normalize_pathish(row.get("checkpoint", "")),
        row.get("phase", ""),
        row.get("name", ""),
        normalize_pathish(row.get("run_dir", "")),
    )


def merge_unique_rows(existing_rows: list[dict[str, str]], new_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int, int]:
    row_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    order: list[tuple[str, ...]] = []
    for row in existing_rows:
        key = dedupe_key(row)
        if key in row_by_key:
            continue
        row_by_key[key] = row
        order.append(key)

    appended = 0
    updated = 0
    for row in new_rows:
        key = dedupe_key(row)
        if key in row_by_key:
            row_by_key[key] = row
            updated += 1
            continue
        row_by_key[key] = row
        order.append(key)
        appended += 1

    unique_rows = [row_by_key[key] for key in order]
    return unique_rows, appended, updated


def read_existing(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not csv_path.is_file():
        return [], []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        return rows, reader.fieldnames or []


def write_registry(csv_path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(tmp_path, csv_path)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    base = base_record(args)
    new_rows = merge_rows(base, summary_rows(args.summary_csv))

    with lock_path.open("w") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        existing_rows, _ = read_existing(csv_path)
        all_rows, appended, updated = merge_unique_rows(existing_rows, new_rows)
        fieldnames = ordered_fields(existing_rows, new_rows)
        write_registry(csv_path, all_rows, fieldnames)
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)

    skipped = len(new_rows) - appended - updated
    print(f"[registry] appended {appended} row(s), updated {updated} row(s), skipped {skipped} duplicate row(s) in {csv_path}")


if __name__ == "__main__":
    main()
