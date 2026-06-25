#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.spot_common.io_utils import (
    append_registry,
    ensure_dir,
    read_json,
    upsert_plan,
    write_json,
    write_manifest,
    write_markdown,
    write_rows,
    write_single_row_csv,
)
from scripts.spot_common.metrics import median_or_none, safe_ratio, summarize_latencies


SUMMARY_FIELDS = [
    "status",
    "error",
    "dataset",
    "split",
    "seq_name",
    "alignment_rows",
    "matched_rows",
    "match_rate",
    "mean_gt_iou",
    "detection_error_rows",
    "detection_error_rate",
    "association_error_rows",
    "association_error_rate",
    "switch_events",
    "protectable_events",
    "protectable_rate",
    "idsw_reduction_percent",
    "median_recovery_latency",
    "median_evidence_latency",
    "median_contamination_gap",
    "oracle0c_fixable_percent",
    "oracle0c_analysis_scope",
]

DERIVED_FIELDS = [
    "is_detection_error",
    "is_association_error",
    "switch_event",
    "switch_from_gt",
    "switch_to_gt",
    "protectable_event",
    "recover_within_window",
    "recovery_latency",
    "evidence_latency",
    "contamination_gap",
    "latency_source",
]

EVENT_FIELDS = [
    "track_id",
    "frame",
    "from_gt",
    "to_gt",
    "protectable_event",
    "recover_within_window",
    "recovery_latency",
    "evidence_latency",
    "contamination_gap",
    "latency_source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the SPOT P0 attribution bundle.")
    parser.add_argument("--alignment-csv", default="", help="GT alignment rows CSV")
    parser.add_argument("--alignment-json", default="", help="Optional GT alignment JSON with a 'rows' array")
    parser.add_argument("--oracle-0a-json", required=True, help="Oracle 0A switch events JSON")
    parser.add_argument("--oracle-0b-json", default="", help="Optional Oracle 0B delay events JSON")
    parser.add_argument("--oracle-0c-json", default="", help="Optional Oracle 0C metrics JSON")
    parser.add_argument("--out-dir", default="outputs/p0_dump", help="Output directory")
    parser.add_argument("--det-iou-error-thresh", type=float, default=0.3, help="IoU threshold for a clear detection error")
    parser.add_argument("--recovery-window", type=int, default=30, help="Window used to count protectable Oracle 0A events")
    return parser.parse_args()


def _parse_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _load_alignment_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    if args.alignment_json:
        payload = read_json(args.alignment_json)
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"Alignment JSON {args.alignment_json} does not contain a 'rows' list")
        if not rows:
            raise ValueError(f"Alignment JSON {args.alignment_json} contains no rows")
        fieldnames = list(rows[0].keys())
        return rows, fieldnames

    if not args.alignment_csv:
        raise ValueError("Provide either --alignment-csv or --alignment-json")

    csv_path = Path(args.alignment_csv).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Alignment CSV {csv_path} is missing headers")
        rows = list(reader)
        if not rows:
            raise ValueError(f"Alignment CSV {csv_path} contains no rows")
        return rows, list(reader.fieldnames)


def _event_key(track_id: int, frame: int, from_gt: int | None = None, to_gt: int | None = None) -> tuple[int, int, int | None, int | None]:
    return (int(track_id), int(frame), None if from_gt is None else int(from_gt), None if to_gt is None else int(to_gt))


def _load_events(path: str, *, source: str) -> dict[tuple[int, int, int | None, int | None], dict[str, Any]]:
    if not path:
        return {}
    payload = read_json(path)
    rows = payload.get("events")
    if not isinstance(rows, list):
        raise ValueError(f"{source} JSON {path} does not contain an 'events' list")
    out: dict[tuple[int, int, int | None, int | None], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        track_id = _parse_int(row.get("track_id", -1), -1)
        frame = _parse_int(row.get("frame", -1), -1)
        from_gt = row.get("from_gt")
        to_gt = row.get("to_gt")
        key = _event_key(track_id, frame, _parse_int(from_gt, -1) if from_gt not in (None, "") else None, _parse_int(to_gt, -1) if to_gt not in (None, "") else None)
        out[key] = row
        out[_event_key(track_id, frame, None, None)] = row
    return out


def _maybe_load_oracle0c(path: str) -> dict[str, Any]:
    if not path:
        return {}
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Oracle 0C metrics at {path} must be a JSON object")
    return payload


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    reports_dir = ensure_dir(out_dir / "reports")
    figures_dir = ensure_dir(out_dir / "figures")
    summary_csv = out_dir / "summary.csv"

    rows, input_fieldnames = _load_alignment_rows(args)
    oracle0a_events = _load_events(args.oracle_0a_json, source="Oracle 0A")
    oracle0b_events = _load_events(args.oracle_0b_json, source="Oracle 0B") if args.oracle_0b_json else {}
    oracle0c = _maybe_load_oracle0c(args.oracle_0c_json)

    dataset = str(rows[0].get("dataset", "unknown")) if rows else "unknown"
    split = str(rows[0].get("split", "unknown")) if rows else "unknown"
    seq_name = str(rows[0].get("seq_name", "unknown_seq")) if rows else "unknown_seq"

    script_path = str(Path(__file__).resolve().relative_to(REPO_ROOT))
    variant = out_dir.name
    tag = variant

    summary_row = {
        "status": "running",
        "error": "",
        "dataset": dataset,
        "split": split,
        "seq_name": seq_name,
        "alignment_rows": 0,
        "matched_rows": 0,
        "match_rate": 0.0,
        "mean_gt_iou": 0.0,
        "detection_error_rows": 0,
        "detection_error_rate": 0.0,
        "association_error_rows": 0,
        "association_error_rate": 0.0,
        "switch_events": 0,
        "protectable_events": 0,
        "protectable_rate": 0.0,
        "idsw_reduction_percent": 0.0,
        "median_recovery_latency": "",
        "median_evidence_latency": "",
        "median_contamination_gap": "",
        "oracle0c_fixable_percent": "",
        "oracle0c_analysis_scope": "",
    }
    write_single_row_csv(summary_csv, summary_row, SUMMARY_FIELDS)

    append_registry(
        kind="analysis",
        status="running",
        script=script_path,
        dataset=dataset,
        split=split,
        tracker_family="spot_p0_dump",
        variant=variant,
        tag=tag,
        run_root=out_dir,
        summary_csv=summary_csv,
        notes=f"P0 dump running for {seq_name}",
    )
    upsert_plan(
        status="running",
        kind="analysis",
        script=script_path,
        dataset=dataset,
        split=split,
        tracker_family="spot_p0_dump",
        variant=variant,
        tag=tag,
        run_root=out_dir,
        summary_csv=summary_csv,
        notes=f"P0 dump running for {seq_name}",
        key=f"spot_p0_dump:{out_dir}",
    )

    p0_rows_csv = reports_dir / "p0_rows.csv"
    p0_events_csv = reports_dir / "p0_events.csv"
    p0_events_json = reports_dir / "p0_events.json"
    snapshot_json = reports_dir / "p0_snapshot.json"
    p0_alignment_summary_json = reports_dir / "p0_alignment_summary.json"

    try:
        fieldnames = list(input_fieldnames)
        for extra in DERIVED_FIELDS:
            if extra not in fieldnames:
                fieldnames.append(extra)

        total_rows = 0
        matched_rows = 0
        gt_iou_sum = 0.0
        detection_error_rows = 0
        association_error_rows = 0
        event_rows: list[dict[str, Any]] = []
        recovery_latencies: list[int] = []
        evidence_latencies: list[int] = []
        contamination_gaps: list[int] = []
        protectable_events = 0
        switch_events = 0

        with p0_rows_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                total_rows += 1
                track_id = _parse_int(row.get("track_id", -1), -1)
                frame = _parse_int(row.get("frame", -1), -1)
                gt_iou = _parse_float(row.get("gt_iou", 0.0), 0.0)
                is_match = _parse_int(row.get("is_match", 0), 0)
                if is_match:
                    matched_rows += 1
                    gt_iou_sum += gt_iou
                is_detection_error = int(gt_iou < float(args.det_iou_error_thresh))
                is_association_error = int((not is_detection_error) and is_match == 0)
                if is_detection_error:
                    detection_error_rows += 1
                if is_association_error:
                    association_error_rows += 1

                event = oracle0a_events.get(_event_key(track_id, frame, None, None))
                if event is None:
                    event = oracle0a_events.get(
                        _event_key(
                            track_id,
                            frame,
                            _parse_int(row.get("gt_id", -1), -1) if row.get("gt_id", "") not in ("", None) else None,
                            None,
                        )
                    )
                derived: dict[str, Any] = {
                    "is_detection_error": is_detection_error,
                    "is_association_error": is_association_error,
                    "switch_event": 1 if event is not None else 0,
                    "switch_from_gt": "",
                    "switch_to_gt": "",
                    "protectable_event": 0,
                    "recover_within_window": 0,
                    "recovery_latency": "",
                    "evidence_latency": "",
                    "contamination_gap": "",
                    "latency_source": "",
                }

                if event is not None:
                    switch_events += 1
                    from_gt = event.get("from_gt")
                    to_gt = event.get("to_gt")
                    recovery_latency = event.get("recovery_latency")
                    protectable = int(event.get("protectable", 0))
                    recover_within_window = int(event.get("recover_within_window", 0))
                    if recovery_latency is not None:
                        recovery_latencies.append(_parse_int(recovery_latency, 0))
                    derived.update(
                        {
                            "switch_from_gt": from_gt if from_gt is not None else "",
                            "switch_to_gt": to_gt if to_gt is not None else "",
                            "protectable_event": protectable,
                            "recover_within_window": recover_within_window,
                            "recovery_latency": recovery_latency if recovery_latency is not None else "",
                        }
                    )
                    if protectable:
                        protectable_events += 1

                    delay_event = oracle0b_events.get(_event_key(track_id, frame, _parse_int(from_gt, -1) if from_gt not in (None, "") else None, _parse_int(to_gt, -1) if to_gt not in (None, "") else None))
                    evidence_latency = None
                    latency_source = ""
                    if delay_event is not None:
                        evidence_latency = delay_event.get("evidence_latency")
                        latency_source = "delay_json"
                    elif recovery_latency is not None:
                        evidence_latency = recovery_latency
                        latency_source = "recovery_proxy"
                    if evidence_latency is not None:
                        evidence_latencies.append(_parse_int(evidence_latency, 0))
                    if recovery_latency is not None and evidence_latency is not None:
                        contamination_gap = _parse_int(recovery_latency, 0) - _parse_int(evidence_latency, 0)
                        contamination_gaps.append(contamination_gap)
                    else:
                        contamination_gap = ""
                    derived.update(
                        {
                            "evidence_latency": evidence_latency if evidence_latency is not None else "",
                            "contamination_gap": contamination_gap,
                            "latency_source": latency_source,
                        }
                    )

                    event_rows.append(
                        {
                            "track_id": track_id,
                            "frame": frame,
                            "from_gt": from_gt if from_gt is not None else "",
                            "to_gt": to_gt if to_gt is not None else "",
                            "protectable_event": protectable,
                            "recover_within_window": recover_within_window,
                            "recovery_latency": recovery_latency if recovery_latency is not None else "",
                            "evidence_latency": evidence_latency if evidence_latency is not None else "",
                            "contamination_gap": contamination_gap,
                            "latency_source": latency_source,
                        }
                    )

                merged = dict(row)
                merged.update(derived)
                writer.writerow(merged)

        write_rows(p0_events_csv, EVENT_FIELDS, event_rows)
        write_json({"events": event_rows}, p0_events_json)

        oracle0c_scope = str(oracle0c.get("analysis_scope", ""))
        oracle0c_fixable_percent = oracle0c.get("fixable_percent", "")
        summary_row.update(
            {
                "status": "completed",
                "alignment_rows": total_rows,
                "matched_rows": matched_rows,
                "match_rate": round(safe_ratio(matched_rows, total_rows), 6),
                "mean_gt_iou": round((gt_iou_sum / matched_rows) if matched_rows else 0.0, 6),
                "detection_error_rows": detection_error_rows,
                "detection_error_rate": round(safe_ratio(detection_error_rows, total_rows), 6),
                "association_error_rows": association_error_rows,
                "association_error_rate": round(safe_ratio(association_error_rows, total_rows), 6),
                "switch_events": switch_events,
                "protectable_events": protectable_events,
                "protectable_rate": round(safe_ratio(protectable_events, switch_events), 6),
                "idsw_reduction_percent": round(100.0 * safe_ratio(protectable_events, switch_events), 6),
                "median_recovery_latency": median_or_none(recovery_latencies),
                "median_evidence_latency": median_or_none(evidence_latencies),
                "median_contamination_gap": median_or_none(contamination_gaps),
                "oracle0c_fixable_percent": oracle0c_fixable_percent,
                "oracle0c_analysis_scope": oracle0c_scope,
            }
        )
        write_single_row_csv(summary_csv, summary_row, SUMMARY_FIELDS)

        snapshot = {
            "dataset": dataset,
            "split": split,
            "seq_name": seq_name,
            "alignment": {
                "alignment_rows": total_rows,
                "matched_rows": matched_rows,
                "match_rate": summary_row["match_rate"],
                "mean_gt_iou": summary_row["mean_gt_iou"],
                "detection_error_rows": detection_error_rows,
                "detection_error_rate": summary_row["detection_error_rate"],
                "association_error_rows": association_error_rows,
                "association_error_rate": summary_row["association_error_rate"],
            },
            "oracle0a": {
                "switch_events": switch_events,
                "protectable_events": protectable_events,
                "protectable_rate": summary_row["protectable_rate"],
                "idsw_reduction_percent": summary_row["idsw_reduction_percent"],
                "median_recovery_latency": summary_row["median_recovery_latency"],
                "recovery_latency_summary": summarize_latencies(recovery_latencies),
            },
            "oracle0b_proxy": {
                "evidence_latency_summary": summarize_latencies(evidence_latencies),
                "median_evidence_latency": summary_row["median_evidence_latency"],
                "median_contamination_gap": summary_row["median_contamination_gap"],
                "available": bool(args.oracle_0b_json),
            },
            "oracle0c": {
                "analysis_scope": oracle0c_scope,
                "fixable_percent": oracle0c_fixable_percent,
                "median_positive_rank": oracle0c.get("median_positive_rank", ""),
                "groups_with_gt": oracle0c.get("groups_with_gt", ""),
                "wrong_selected_groups": oracle0c.get("wrong_selected_groups", ""),
                "fixable_groups": oracle0c.get("fixable_groups", ""),
            },
            "sources": {
                "alignment_csv": str(Path(args.alignment_csv).expanduser().resolve()) if args.alignment_csv else "",
                "alignment_json": str(Path(args.alignment_json).expanduser().resolve()) if args.alignment_json else "",
                "oracle_0a_json": str(Path(args.oracle_0a_json).expanduser().resolve()),
                "oracle_0b_json": str(Path(args.oracle_0b_json).expanduser().resolve()) if args.oracle_0b_json else "",
                "oracle_0c_json": str(Path(args.oracle_0c_json).expanduser().resolve()) if args.oracle_0c_json else "",
            },
            "recommendation": {
                "state_protection": "go" if float(summary_row["idsw_reduction_percent"]) >= 5.0 else ("ablation_only" if float(summary_row["idsw_reduction_percent"]) >= 3.0 else "kill"),
                "delayed_commitment": "pending_real_0b" if not args.oracle_0b_json else ("candidate_extension" if (summary_row["median_evidence_latency"] not in (None, "") and 2 <= float(summary_row["median_evidence_latency"]) <= 5) else "skip"),
                "pcc": "provisional_support" if oracle0c_scope == "partial" else ("support" if float(oracle0c_fixable_percent or 0.0) >= 10.0 else "ablation_only"),
            },
        }
        write_json(snapshot, snapshot_json)
        write_json(snapshot, p0_alignment_summary_json)

        report_md = [
            f"# P0 Attribution Bundle: {seq_name}",
            "",
            "## Context",
            "This bundle combines the frozen-detector alignment, Oracle 0A state-protection evidence, and the current 0C rerank proxy into a single reproducible P0 snapshot.",
            "",
            "## Key results",
            f"- alignment_rows: {summary_row['alignment_rows']}",
            f"- matched_rows: {summary_row['matched_rows']}",
            f"- match_rate: {summary_row['match_rate']}",
            f"- mean_gt_iou: {summary_row['mean_gt_iou']}",
            f"- detection_error_rows: {summary_row['detection_error_rows']}",
            f"- association_error_rows: {summary_row['association_error_rows']}",
            f"- switch_events: {summary_row['switch_events']}",
            f"- protectable_events: {summary_row['protectable_events']}",
            f"- protectable_rate: {summary_row['protectable_rate']}",
            f"- idsw_reduction_percent: {summary_row['idsw_reduction_percent']}",
            f"- median_recovery_latency: {summary_row['median_recovery_latency']}",
            f"- median_evidence_latency: {summary_row['median_evidence_latency']}",
            f"- median_contamination_gap: {summary_row['median_contamination_gap']}",
            f"- oracle0c_analysis_scope: {summary_row['oracle0c_analysis_scope']}",
            f"- oracle0c_fixable_percent: {summary_row['oracle0c_fixable_percent']}",
            "",
            "## Recommendation",
            f"- state_protection: {snapshot['recommendation']['state_protection']}",
            f"- delayed_commitment: {snapshot['recommendation']['delayed_commitment']}",
            f"- pcc: {snapshot['recommendation']['pcc']}",
            "",
            "## Notes",
            "- Evidence latency is currently a proxy derived from the available oracle outputs unless a dedicated Oracle 0B run is supplied.",
            "- The current 0C result is still marked partial and should be rerun for the final paper-grade conclusion.",
        ]
        write_markdown("\n".join(report_md), reports_dir / "P0_dump_report.md")

        metrics = {
            "alignment_rows": total_rows,
            "matched_rows": matched_rows,
            "match_rate": summary_row["match_rate"],
            "mean_gt_iou": summary_row["mean_gt_iou"],
            "detection_error_rows": detection_error_rows,
            "association_error_rows": association_error_rows,
            "switch_events": switch_events,
            "protectable_events": protectable_events,
            "protectable_rate": summary_row["protectable_rate"],
            "idsw_reduction_percent": summary_row["idsw_reduction_percent"],
            "median_recovery_latency": summary_row["median_recovery_latency"],
            "median_evidence_latency": summary_row["median_evidence_latency"],
            "median_contamination_gap": summary_row["median_contamination_gap"],
            "oracle0c_fixable_percent": summary_row["oracle0c_fixable_percent"],
            "oracle0c_analysis_scope": summary_row["oracle0c_analysis_scope"],
        }
        write_manifest(
            out_dir,
            phase="spot_p0_dump",
            script=script_path,
            args=vars(args),
            status="ok",
            metrics=metrics,
            artifacts={
                "summary_csv": str(summary_csv),
                "snapshot_json": str(snapshot_json),
                "alignment_rows_csv": str(p0_rows_csv),
                "events_csv": str(p0_events_csv),
                "events_json": str(p0_events_json),
            },
            notes=f"P0 attribution bundle for {seq_name}",
        )
        append_registry(
            kind="analysis",
            status="success",
            script=script_path,
            dataset=dataset,
            split=split,
            tracker_family="spot_p0_dump",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"P0 attribution bundle complete for {seq_name}",
        )
        upsert_plan(
            status="completed",
            kind="analysis",
            script=script_path,
            dataset=dataset,
            split=split,
            tracker_family="spot_p0_dump",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"P0 attribution bundle complete for {seq_name}",
            key=f"spot_p0_dump:{out_dir}",
        )
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = str(exc)
        write_single_row_csv(summary_csv, summary_row, SUMMARY_FIELDS)
        append_registry(
            kind="analysis",
            status="failed",
            script=script_path,
            dataset=dataset,
            split=split,
            tracker_family="spot_p0_dump",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"P0 attribution bundle failed: {exc}",
        )
        upsert_plan(
            status="failed",
            kind="analysis",
            script=script_path,
            dataset=dataset,
            split=split,
            tracker_family="spot_p0_dump",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"P0 attribution bundle failed: {exc}",
            key=f"spot_p0_dump:{out_dir}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
