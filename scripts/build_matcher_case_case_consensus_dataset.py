#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
SUMMARY_FIELDS = [
    "rows_jsonl",
    "label_source",
    "source_rows",
    "unique_cases",
    "stable_cases",
    "conflicting_cases",
    "tie_cases",
    "stable_binary_cases",
    "pos_cases",
    "neg_cases",
    "min_support",
    "status",
    "error",
]
CASE_SUMMARY_FIELDS = [
    "base_run_name",
    "seq_name",
    "frame_id",
    "track_gt_id",
    "raw_det_index",
    "fgas_det_index",
    "base_best_det_index",
    "base_best_det_raw_owner_track_gt_id",
    "case_rows",
    "unique_sources",
    "beneficial_rows",
    "harmful_rows",
    "tie_rows",
    "case_consensus_status",
    "stable_binary_case",
    "stable_label",
    "representative_source_analysis_name",
    "representative_alt_run_name",
    "source_analysis_names",
    "alt_run_names",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate divergence-aligned matcher-case rows into stable case-level labels."
    )
    parser.add_argument("--rows-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--min-support",
        type=int,
        default=1,
        help="minimum number of source rows required before a stable binary case is exported",
    )
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_single_row_csv(path: Path, fieldnames: Iterable[str], row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_rows_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
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
        "scripts/build_matcher_case_case_consensus_dataset.py",
        "--dataset",
        "MOT17",
        "--split",
        "matcher_case_case_consensus_dataset",
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
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _case_key(row: Mapping[str, object]) -> Tuple[object, ...]:
    return (
        str(row.get("base_run_name", "")),
        str(row.get("seq_name", "")),
        int(row.get("frame_id", -1)),
        int(row.get("track_gt_id", -1)),
        int(row.get("raw_det_index", -1)),
        int(row.get("fgas_det_index", -1)),
        int(row.get("base_best_det_index", -1)),
        int(row.get("base_best_det_raw_owner_track_gt_id", -1)),
    )


def _case_status(counts: Counter[str]) -> str:
    has_beneficial = counts.get("beneficial", 0) > 0
    has_harmful = counts.get("harmful", 0) > 0
    has_tie = counts.get("tie", 0) > 0
    if has_beneficial and has_harmful:
        return "conflict"
    if has_beneficial and not has_harmful and not has_tie:
        return "beneficial"
    if has_harmful and not has_beneficial and not has_tie:
        return "harmful"
    return "tie"


def _label_value(status: str) -> int:
    if status == "beneficial":
        return 1
    if status == "harmful":
        return 0
    return -1


def _pick_representative(rows: Sequence[Mapping[str, object]], desired_status: str) -> Mapping[str, object]:
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            str(row.get("source_analysis_name", "")),
            str(row.get("alt_run_name", "")),
        ),
    )
    for row in ordered_rows:
        if str(row.get("utility_label_status", "tie")) == desired_status:
            return row
    return ordered_rows[0]


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    case_summary_csv = out_dir / "case_summary.csv"
    rows_jsonl = out_dir / "rows.jsonl"

    summary_row: Dict[str, object] = {
        "rows_jsonl": str(Path(args.rows_jsonl)),
        "label_source": "",
        "source_rows": 0,
        "unique_cases": 0,
        "stable_cases": 0,
        "conflicting_cases": 0,
        "tie_cases": 0,
        "stable_binary_cases": 0,
        "pos_cases": 0,
        "neg_cases": 0,
        "min_support": int(args.min_support),
        "status": "running",
        "error": "",
    }
    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
    append_registry(args, summary_csv, "running", "building stable matcher-case case-consensus dataset")

    try:
        grouped_rows: dict[Tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
        with Path(args.rows_jsonl).open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                grouped_rows[_case_key(row)].append(row)
                summary_row["source_rows"] = int(summary_row["source_rows"]) + 1
                if not summary_row["label_source"]:
                    summary_row["label_source"] = str(row.get("utility_label_source", ""))

        summary_row["unique_cases"] = int(len(grouped_rows))

        case_summary_rows: list[dict[str, object]] = []
        with rows_jsonl.open("w", encoding="utf-8") as out_handle:
            for key in sorted(grouped_rows):
                rows = grouped_rows[key]
                label_counts: Counter[str] = Counter(
                    str(row.get("utility_label_status", "tie")) for row in rows
                )
                case_status = _case_status(label_counts)
                stable_binary_case = int(
                    case_status in {"beneficial", "harmful"} and len(rows) >= int(args.min_support)
                )
                representative = _pick_representative(rows, desired_status=case_status)

                if case_status == "conflict":
                    summary_row["conflicting_cases"] = int(summary_row["conflicting_cases"]) + 1
                else:
                    summary_row["stable_cases"] = int(summary_row["stable_cases"]) + 1
                    if case_status == "tie":
                        summary_row["tie_cases"] = int(summary_row["tie_cases"]) + 1

                unique_sources = sorted(
                    {
                        str(row.get("source_analysis_name", ""))
                        for row in rows
                        if str(row.get("source_analysis_name", ""))
                    }
                )
                alt_run_names = sorted(
                    {
                        str(row.get("alt_run_name", ""))
                        for row in rows
                        if str(row.get("alt_run_name", ""))
                    }
                )
                stable_label = int(_label_value(case_status))

                case_summary_rows.append(
                    {
                        "base_run_name": str(representative.get("base_run_name", "")),
                        "seq_name": str(representative.get("seq_name", "")),
                        "frame_id": int(representative.get("frame_id", -1)),
                        "track_gt_id": int(representative.get("track_gt_id", -1)),
                        "raw_det_index": int(representative.get("raw_det_index", -1)),
                        "fgas_det_index": int(representative.get("fgas_det_index", -1)),
                        "base_best_det_index": int(representative.get("base_best_det_index", -1)),
                        "base_best_det_raw_owner_track_gt_id": int(
                            representative.get("base_best_det_raw_owner_track_gt_id", -1)
                        ),
                        "case_rows": int(len(rows)),
                        "unique_sources": int(len(unique_sources)),
                        "beneficial_rows": int(label_counts.get("beneficial", 0)),
                        "harmful_rows": int(label_counts.get("harmful", 0)),
                        "tie_rows": int(label_counts.get("tie", 0)),
                        "case_consensus_status": str(case_status),
                        "stable_binary_case": int(stable_binary_case),
                        "stable_label": int(stable_label),
                        "representative_source_analysis_name": str(
                            representative.get("source_analysis_name", "")
                        ),
                        "representative_alt_run_name": str(representative.get("alt_run_name", "")),
                        "source_analysis_names": "|".join(unique_sources),
                        "alt_run_names": "|".join(alt_run_names),
                    }
                )

                if not stable_binary_case:
                    continue

                export_row = dict(representative)
                export_row["case_consensus_status"] = str(case_status)
                export_row["case_consensus_label"] = int(stable_label)
                export_row["case_support"] = int(len(rows))
                export_row["case_unique_sources"] = int(len(unique_sources))
                export_row["case_source_analysis_names"] = list(unique_sources)
                export_row["case_alt_run_names"] = list(alt_run_names)
                export_row["case_beneficial_rows"] = int(label_counts.get("beneficial", 0))
                export_row["case_harmful_rows"] = int(label_counts.get("harmful", 0))
                export_row["case_tie_rows"] = int(label_counts.get("tie", 0))
                out_handle.write(json.dumps(export_row))
                out_handle.write("\n")

                summary_row["stable_binary_cases"] = int(summary_row["stable_binary_cases"]) + 1
                if stable_label == 1:
                    summary_row["pos_cases"] = int(summary_row["pos_cases"]) + 1
                elif stable_label == 0:
                    summary_row["neg_cases"] = int(summary_row["neg_cases"]) + 1

        write_rows_csv(case_summary_csv, CASE_SUMMARY_FIELDS, case_summary_rows)
        summary_row["status"] = "success"
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(
            args,
            summary_csv,
            "success",
            f"stable matcher-case case-consensus dataset built: stable_binary_cases={summary_row['stable_binary_cases']}",
        )
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = str(exc)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, "failed", f"stable matcher-case case-consensus dataset failed: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
