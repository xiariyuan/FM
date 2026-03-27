#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
import fcntl


CORE_FIELDS = [
    "plan_key",
    "created_at",
    "updated_at",
    "status",
    "kind",
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
    parser = argparse.ArgumentParser(description="Upsert experiment plan/status rows in a central CSV.")
    parser.add_argument("--csv", required=True, help="Plan CSV path")
    parser.add_argument("--status", required=True, choices=["queued", "running", "completed", "failed", "cancelled"])
    parser.add_argument("--kind", default="other", choices=["train", "eval", "analysis", "other"])
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
    parser.add_argument("--key", default="", help="Stable unique plan key. Defaults to a normalized path-based key.")
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


def normalize_pathish(value: str) -> str:
    if not value:
        return ""
    try:
        return str(Path(value).expanduser().resolve(strict=False))
    except Exception:
        return value


def derive_key(args: argparse.Namespace) -> str:
    if args.key:
        return args.key
    for value in (args.run_root, args.checkpoint, args.summary_csv):
        normalized = normalize_pathish(value)
        if normalized:
            return f"path:{normalized}"
    pieces = [
        args.kind,
        args.script,
        args.dataset,
        args.split,
        args.tracker_family,
        args.variant,
        args.tag,
    ]
    return "meta:" + "|".join(pieces)


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


def read_existing(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not csv_path.is_file():
        return [], []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        return rows, reader.fieldnames or []


def write_csv(csv_path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(tmp_path, csv_path)


def build_record(args: argparse.Namespace) -> dict[str, str]:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    record = {
        "plan_key": derive_key(args),
        "created_at": now,
        "updated_at": now,
        "status": args.status,
        "kind": args.kind,
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


def merge_record(existing: dict[str, str], new_record: dict[str, str]) -> dict[str, str]:
    merged = dict(existing)
    merged["updated_at"] = new_record["updated_at"]
    merged["status"] = new_record["status"]
    for key, value in new_record.items():
        if key in {"plan_key", "created_at", "updated_at", "status"}:
            continue
        if value != "":
            merged[key] = value
    return merged


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    new_record = build_record(args)

    with lock_path.open("w") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        existing_rows, _ = read_existing(csv_path)
        updated_rows: list[dict[str, str]] = []
        found = False
        for row in existing_rows:
            if row.get("plan_key", "") == new_record["plan_key"]:
                updated_rows.append(merge_record(row, new_record))
                found = True
            else:
                updated_rows.append(row)
        if not found:
            updated_rows.append(new_record)
        fieldnames = ordered_fields(existing_rows, [new_record])
        write_csv(csv_path, updated_rows, fieldnames)
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)

    action = "updated" if found else "inserted"
    print(f"[plan] {action} {new_record['plan_key']} in {csv_path}")


if __name__ == "__main__":
    main()
