#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

SUMMARY_FIELDS = [
    "source_blockbank",
    "output_blockbank",
    "ambiguous_only",
    "repeat_spec",
    "source_blocks",
    "output_blocks",
    "extra_copies",
    "source_ambiguous_blocks",
    "output_ambiguous_blocks",
    "status",
    "error",
]

PER_SEQ_FIELDS = [
    "seq_name",
    "source_blocks",
    "source_ambiguous_blocks",
    "output_blocks",
    "output_ambiguous_blocks",
    "repeat_factor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reweighted FGAS blockbank by repeating selected sequence blocks.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--seq-repeat",
        action="append",
        default=[],
        help="repeat factor spec in the form SEQ_NAME=FACTOR. Factor is total copies including the original.",
    )
    parser.add_argument("--ambiguous-only", action="store_true", help="only repeat blocks marked ambiguous")
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_registry(args: argparse.Namespace, summary_csv: Path, status: str, notes: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(args.registry_csv),
        "--kind",
        "analysis",
        "--status",
        status,
        "--script",
        "scripts/reweight_fgas_blockbank.py",
        "--dataset",
        "MOT17",
        "--split",
        "blockbank_reweight",
        "--tracker-family",
        "deep_ocsort_fgas",
        "--variant",
        Path(args.out_dir).name,
        "--tag",
        Path(args.out_dir).name,
        "--run-root",
        str(Path(args.out_dir)),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def parse_repeat_specs(values: List[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for value in values:
        if "=" not in str(value):
            raise ValueError(f"Invalid --seq-repeat value: {value}")
        seq_name, factor_text = str(value).split("=", 1)
        seq_name = seq_name.strip()
        factor = int(factor_text)
        if not seq_name:
            raise ValueError(f"Missing sequence name in --seq-repeat value: {value}")
        if factor < 1:
            raise ValueError(f"Repeat factor must be >= 1: {value}")
        mapping[seq_name] = factor
    return mapping


def infer_seq_name(payload: Dict[str, object]) -> str:
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("seq_name"):
        return str(metadata["seq_name"])
    block_key = str(payload.get("block_key", ""))
    return block_key.split(":", 1)[0] if ":" in block_key else ""


def is_ambiguous(payload: Dict[str, object]) -> bool:
    return bool(int(payload.get("ambiguous", 0)))


def main() -> int:
    args = parse_args()
    repeat_map = parse_repeat_specs(list(args.seq_repeat))
    if not repeat_map:
        raise ValueError("At least one --seq-repeat spec is required.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    output_jsonl = out_dir / "blockbank.jsonl"
    per_seq_csv = out_dir / "per_sequence_counts.csv"
    summary_row: Dict[str, object] = {
        "source_blockbank": str(Path(args.input_jsonl)),
        "output_blockbank": str(output_jsonl),
        "ambiguous_only": int(bool(args.ambiguous_only)),
        "repeat_spec": "|".join(f"{key}={value}" for key, value in sorted(repeat_map.items())),
        "source_blocks": 0,
        "output_blocks": 0,
        "extra_copies": 0,
        "source_ambiguous_blocks": 0,
        "output_ambiguous_blocks": 0,
        "status": "running",
        "error": "",
    }
    write_rows(summary_csv, SUMMARY_FIELDS, [summary_row])
    append_registry(args, summary_csv, "running", f"reweight source={args.input_jsonl} spec={summary_row['repeat_spec']}")

    try:
        source_rows: List[Dict[str, object]] = []
        with Path(args.input_jsonl).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    source_rows.append(json.loads(line))

        source_per_seq = Counter()
        source_amb_per_seq = Counter()
        output_per_seq = Counter()
        output_amb_per_seq = Counter()
        extra_copies = 0

        with output_jsonl.open("w", encoding="utf-8") as handle:
            for payload in source_rows:
                seq_name = infer_seq_name(payload)
                ambiguous = is_ambiguous(payload)
                source_per_seq[seq_name] += 1
                if ambiguous:
                    source_amb_per_seq[seq_name] += 1

                repeat_factor = int(repeat_map.get(seq_name, 1))
                if args.ambiguous_only and not ambiguous:
                    repeat_factor = 1

                for _copy_idx in range(repeat_factor):
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    output_per_seq[seq_name] += 1
                    if ambiguous:
                        output_amb_per_seq[seq_name] += 1
                extra_copies += max(repeat_factor - 1, 0)

        per_seq_rows: List[Dict[str, object]] = []
        for seq_name in sorted(set(source_per_seq) | set(output_per_seq) | set(repeat_map)):
            per_seq_rows.append(
                {
                    "seq_name": seq_name,
                    "source_blocks": int(source_per_seq.get(seq_name, 0)),
                    "source_ambiguous_blocks": int(source_amb_per_seq.get(seq_name, 0)),
                    "output_blocks": int(output_per_seq.get(seq_name, 0)),
                    "output_ambiguous_blocks": int(output_amb_per_seq.get(seq_name, 0)),
                    "repeat_factor": int(repeat_map.get(seq_name, 1)),
                }
            )
        write_rows(per_seq_csv, PER_SEQ_FIELDS, per_seq_rows)

        summary_row.update(
            {
                "source_blocks": int(len(source_rows)),
                "output_blocks": int(sum(output_per_seq.values())),
                "extra_copies": int(extra_copies),
                "source_ambiguous_blocks": int(sum(source_amb_per_seq.values())),
                "output_ambiguous_blocks": int(sum(output_amb_per_seq.values())),
                "status": "success",
                "error": "",
            }
        )
        write_rows(summary_csv, SUMMARY_FIELDS, [summary_row])
        append_registry(args, summary_csv, "success", f"reweight complete source={args.input_jsonl} spec={summary_row['repeat_spec']}")
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = str(exc)
        write_rows(summary_csv, SUMMARY_FIELDS, [summary_row])
        append_registry(args, summary_csv, "failed", f"reweight failed source={args.input_jsonl}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
