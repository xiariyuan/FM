#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_value(raw: str):
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "null":
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def set_nested(doc, dotted_key: str, value):
    cursor = doc
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def main():
    ap = argparse.ArgumentParser(description="Merge dotted key-values into a run manifest JSON file.")
    ap.add_argument("--manifest", required=True, help="Path to run_manifest.json")
    ap.add_argument(
        "--set",
        dest="updates",
        action="append",
        default=[],
        help="Dotted update in the form key=value. Value accepts JSON literals.",
    )
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
    else:
        doc = {}

    for item in args.updates:
        if "=" not in item:
            raise SystemExit(f"Invalid --set value (expected key=value): {item}")
        key, raw_value = item.split("=", 1)
        set_nested(doc, key, parse_value(raw_value))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
