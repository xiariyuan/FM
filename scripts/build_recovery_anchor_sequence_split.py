#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

SUMMARY_FIELDS = [
    "anchor_jsonl",
    "split_mode",
    "val_seq_names",
    "target_val_fraction",
    "source_rows",
    "source_sequences",
    "train_rows",
    "train_pos",
    "train_neg",
    "val_rows",
    "val_pos",
    "val_neg",
    "status",
    "error",
]

PER_SEQUENCE_FIELDS = [
    "seq_name",
    "rows",
    "positive_rows",
    "negative_rows",
    "split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sequence-level train/val JSONL splits for recovery-anchor candidate learning."
    )
    parser.add_argument("--anchor-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-seq-name", dest="val_seq_names", nargs="*", default=[])
    parser.add_argument("--target-val-fraction", type=float, default=0.25)
    parser.add_argument("--min-val-positive", type=int, default=1)
    parser.add_argument("--min-train-positive", type=int, default=1)
    parser.add_argument("--max-auto-val-seqs", type=int, default=3)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_single_row_csv(path: Path, fieldnames: Iterable[str], row: Dict[str, object]) -> None:
    write_rows(path, fieldnames, [row])


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
        "scripts/build_recovery_anchor_sequence_split.py",
        "--dataset",
        "DanceTrack",
        "--split",
        "recovery_anchor_seqsplit",
        "--tracker-family",
        "deep_ocsort_preassoc_force_recovery_anchor",
        "--variant",
        Path(args.out_dir).name,
        "--tag",
        Path(args.out_dir).name,
        "--run-root",
        str(Path(args.out_dir).resolve()),
        "--summary-csv",
        str(summary_csv.resolve()),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def load_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def choose_val_sequences(
    stats_by_seq: Dict[str, Dict[str, int]],
    *,
    target_val_fraction: float,
    min_val_positive: int,
    min_train_positive: int,
    max_auto_val_seqs: int,
) -> List[str]:
    seq_names = sorted(stats_by_seq)
    if len(seq_names) < 2:
        raise ValueError("Need at least two sequences to build a sequence-level split.")

    total_rows = sum(int(stats["rows"]) for stats in stats_by_seq.values())
    total_pos = sum(int(stats["positive_rows"]) for stats in stats_by_seq.values())
    target_val_pos = max(int(min_val_positive), int(round(float(total_pos) * float(target_val_fraction))))

    candidates: List[tuple[tuple[float, ...], List[str]]] = []
    max_group = max(1, min(int(max_auto_val_seqs), len(seq_names) - 1))
    for group_size in range(1, max_group + 1):
        for combo in itertools.combinations(seq_names, group_size):
            val_rows = sum(int(stats_by_seq[name]["rows"]) for name in combo)
            val_pos = sum(int(stats_by_seq[name]["positive_rows"]) for name in combo)
            train_rows = int(total_rows) - int(val_rows)
            train_pos = int(total_pos) - int(val_pos)
            if train_rows <= 0 or val_rows <= 0:
                continue
            if val_pos < int(min_val_positive) or train_pos < int(min_train_positive):
                continue
            row_fraction_error = abs((float(val_rows) / float(max(total_rows, 1))) - float(target_val_fraction))
            pos_balance = float(abs(int(val_pos) - int(target_val_pos)))
            score = (
                pos_balance,
                row_fraction_error,
                float(group_size),
                float(-min(int(val_pos), int(train_pos))),
            )
            candidates.append((score, list(combo)))

    if not candidates:
        raise ValueError(
            "Could not find an automatic sequence split that leaves positives in both train and val."
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1]


def main() -> int:
    args = parse_args()
    anchor_jsonl = Path(args.anchor_jsonl).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "summary.csv"
    per_sequence_csv = out_dir / "per_sequence_summary.csv"
    train_jsonl = out_dir / "train.jsonl"
    val_jsonl = out_dir / "val.jsonl"

    summary_row: Dict[str, object] = {
        "anchor_jsonl": str(anchor_jsonl),
        "split_mode": "manual" if args.val_seq_names else "auto",
        "val_seq_names": "|".join(str(name) for name in list(args.val_seq_names or [])),
        "target_val_fraction": float(args.target_val_fraction),
        "source_rows": 0,
        "source_sequences": 0,
        "train_rows": 0,
        "train_pos": 0,
        "train_neg": 0,
        "val_rows": 0,
        "val_pos": 0,
        "val_neg": 0,
        "status": "running",
        "error": "",
    }
    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
    write_rows(per_sequence_csv, PER_SEQUENCE_FIELDS, [])
    append_registry(args, summary_csv, "running", "building recovery-anchor sequence split")

    try:
        rows = load_rows(anchor_jsonl)
        if not rows:
            raise ValueError("Anchor dataset is empty.")

        rows_by_seq: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in rows:
            seq_name = str(row.get("seq_name", "")).strip()
            if not seq_name:
                raise ValueError("Encountered anchor row without seq_name.")
            rows_by_seq[seq_name].append(row)

        stats_by_seq: Dict[str, Dict[str, int]] = {}
        for seq_name, seq_rows in rows_by_seq.items():
            pos_rows = sum(int(row.get("label", 0)) for row in seq_rows)
            stats_by_seq[seq_name] = {
                "rows": int(len(seq_rows)),
                "positive_rows": int(pos_rows),
                "negative_rows": int(len(seq_rows) - pos_rows),
            }

        summary_row["source_rows"] = int(len(rows))
        summary_row["source_sequences"] = int(len(rows_by_seq))

        if args.val_seq_names:
            val_seq_names = sorted({str(name) for name in list(args.val_seq_names or [])})
            unknown = [name for name in val_seq_names if name not in rows_by_seq]
            if unknown:
                raise ValueError(f"Unknown validation sequences: {unknown}")
        else:
            val_seq_names = choose_val_sequences(
                stats_by_seq,
                target_val_fraction=float(args.target_val_fraction),
                min_val_positive=int(args.min_val_positive),
                min_train_positive=int(args.min_train_positive),
                max_auto_val_seqs=int(args.max_auto_val_seqs),
            )
        val_seq_name_set = set(val_seq_names)
        summary_row["val_seq_names"] = "|".join(val_seq_names)

        train_rows: List[Dict[str, object]] = []
        val_rows: List[Dict[str, object]] = []
        per_sequence_rows: List[Dict[str, object]] = []
        for seq_name in sorted(rows_by_seq):
            split = "val" if seq_name in val_seq_name_set else "train"
            seq_rows = rows_by_seq[seq_name]
            if split == "val":
                val_rows.extend(seq_rows)
            else:
                train_rows.extend(seq_rows)
            per_sequence_rows.append(
                {
                    "seq_name": seq_name,
                    "rows": int(stats_by_seq[seq_name]["rows"]),
                    "positive_rows": int(stats_by_seq[seq_name]["positive_rows"]),
                    "negative_rows": int(stats_by_seq[seq_name]["negative_rows"]),
                    "split": split,
                }
            )

        train_pos = sum(int(row.get("label", 0)) for row in train_rows)
        val_pos = sum(int(row.get("label", 0)) for row in val_rows)
        if not train_rows or not val_rows:
            raise ValueError("Train or val split is empty.")
        if train_pos < int(args.min_train_positive):
            raise ValueError("Train split does not contain enough positive rows.")
        if val_pos < int(args.min_val_positive):
            raise ValueError("Val split does not contain enough positive rows.")

        write_jsonl(train_jsonl, train_rows)
        write_jsonl(val_jsonl, val_rows)
        write_rows(per_sequence_csv, PER_SEQUENCE_FIELDS, per_sequence_rows)

        summary_row.update(
            {
                "train_rows": int(len(train_rows)),
                "train_pos": int(train_pos),
                "train_neg": int(len(train_rows) - train_pos),
                "val_rows": int(len(val_rows)),
                "val_pos": int(val_pos),
                "val_neg": int(len(val_rows) - val_pos),
                "status": "success",
                "error": "",
            }
        )
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(
            args,
            summary_csv,
            "success",
            f"built recovery-anchor split: train={summary_row['train_rows']} val={summary_row['val_rows']}",
        )
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = str(exc)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, "failed", f"failed to build recovery-anchor split: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
