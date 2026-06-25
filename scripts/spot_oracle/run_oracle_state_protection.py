#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

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
    write_single_row_csv,
)
from scripts.spot_common.metrics import median_or_none, safe_ratio


SUMMARY_FIELDS = [
    "status",
    "error",
    "dataset",
    "split",
    "seq_name",
    "switch_events",
    "protectable_events",
    "protectable_rate",
    "idsw_reduction_percent",
    "median_recovery_latency",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle 0A proxy for SPOT state protection.")
    parser.add_argument("--alignment-json", required=True)
    parser.add_argument("--out-dir", default="outputs/oracle_gate/0A_state_protection")
    parser.add_argument("--recovery-window", type=int, default=10)
    return parser.parse_args()


def _find_switch_events(rows: list[dict]) -> tuple[list[dict], list[int]]:
    by_track: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if int(row.get("gt_id", -1)) > 0:
            by_track[int(row["track_id"])].append(row)
    events: list[dict] = []
    latencies: list[int] = []
    for track_id, track_rows in by_track.items():
        ordered = sorted(track_rows, key=lambda item: int(item["frame"]))
        for prev, cur in zip(ordered[:-1], ordered[1:]):
            prev_gt = int(prev["gt_id"])
            cur_gt = int(cur["gt_id"])
            if prev_gt <= 0 or cur_gt <= 0 or prev_gt == cur_gt:
                continue
            latency = None
            for future in ordered:
                future_frame = int(future["frame"])
                if future_frame <= int(cur["frame"]):
                    continue
                if int(future["gt_id"]) == prev_gt:
                    latency = future_frame - int(cur["frame"])
                    break
            event = {
                "track_id": track_id,
                "frame": int(cur["frame"]),
                "from_gt": prev_gt,
                "to_gt": cur_gt,
                "protectable": int(latency is not None),
                "recovery_latency": latency,
            }
            events.append(event)
            if latency is not None:
                latencies.append(int(latency))
    return events, latencies


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    summary_csv = out_dir / "summary.csv"
    script_path = str(Path(__file__).resolve().relative_to(REPO_ROOT))
    variant = out_dir.name
    tag = variant

    payload = read_json(args.alignment_json)
    dataset = str(payload.get("dataset", "unknown"))
    split = str(payload.get("split", "unknown"))
    seq_name = str(payload.get("seq_name", "unknown_seq"))
    summary_row = {
        "status": "running",
        "error": "",
        "dataset": dataset,
        "split": split,
        "seq_name": seq_name,
        "switch_events": 0,
        "protectable_events": 0,
        "protectable_rate": 0.0,
        "idsw_reduction_percent": 0.0,
        "median_recovery_latency": "",
    }
    write_single_row_csv(summary_csv, summary_row, SUMMARY_FIELDS)
    append_registry(
        kind="analysis",
        status="running",
        script=script_path,
        dataset=dataset,
        split=split,
        tracker_family="spot_oracle_0A",
        variant=variant,
        tag=tag,
        run_root=out_dir,
        summary_csv=summary_csv,
        notes=f"state protection oracle running for {seq_name}",
    )
    upsert_plan(
        status="running",
        kind="analysis",
        script=script_path,
        dataset=dataset,
        split=split,
        tracker_family="spot_oracle_0A",
        variant=variant,
        tag=tag,
        run_root=out_dir,
        summary_csv=summary_csv,
        notes=f"state protection oracle running for {seq_name}",
        key=f"spot_oracle_0A:{out_dir}",
    )

    try:
        events, latencies = _find_switch_events(list(payload.get("rows", [])))
        filtered_events = []
        for event in events:
            latency = event.get("recovery_latency")
            event["recover_within_window"] = int(latency is not None and int(latency) <= int(args.recovery_window))
            filtered_events.append(event)
        protectable = sum(int(event["recover_within_window"]) for event in filtered_events)
        summary_row.update(
            {
                "status": "completed",
                "switch_events": len(filtered_events),
                "protectable_events": protectable,
                "protectable_rate": round(safe_ratio(protectable, len(filtered_events)), 6),
                "idsw_reduction_percent": round(100.0 * safe_ratio(protectable, len(filtered_events)), 6),
                "median_recovery_latency": median_or_none(latencies),
            }
        )
        write_single_row_csv(summary_csv, summary_row, SUMMARY_FIELDS)
        metrics = {
            "switch_events": len(filtered_events),
            "protectable_events": protectable,
            "protectable_rate": summary_row["protectable_rate"],
            "idsw_reduction_percent": summary_row["idsw_reduction_percent"],
            "median_recovery_latency": summary_row["median_recovery_latency"],
            "recovery_window": int(args.recovery_window),
        }
        write_json(metrics, out_dir / "oracle_state_protection_metrics.json")
        write_json({"events": filtered_events}, out_dir / "switch_events.json")
        write_markdown(
            "\n".join(
                [
                    "# Oracle 0A State Protection",
                    "",
                    f"- seq_name: {seq_name}",
                    f"- switch_events: {len(filtered_events)}",
                    f"- protectable_events: {protectable}",
                    f"- protectable_rate: {summary_row['protectable_rate']}",
                    f"- idsw_reduction_percent: {summary_row['idsw_reduction_percent']}",
                    f"- median_recovery_latency: {summary_row['median_recovery_latency']}",
                ]
            ),
            out_dir / "oracle_state_protection_report.md",
        )
        write_manifest(
            out_dir,
            phase="oracle_0A_state_protection",
            script=script_path,
            args=vars(args),
            status="ok",
            metrics=metrics,
            artifacts={
                "summary_csv": str(summary_csv),
                "metrics_json": str(out_dir / "oracle_state_protection_metrics.json"),
                "events_json": str(out_dir / "switch_events.json"),
            },
            notes=f"state protection oracle for {seq_name}",
        )
        append_registry(
            kind="analysis",
            status="success",
            script=script_path,
            dataset=dataset,
            split=split,
            tracker_family="spot_oracle_0A",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"state protection oracle complete for {seq_name}",
        )
        upsert_plan(
            status="completed",
            kind="analysis",
            script=script_path,
            dataset=dataset,
            split=split,
            tracker_family="spot_oracle_0A",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"state protection oracle complete for {seq_name}",
            key=f"spot_oracle_0A:{out_dir}",
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
            tracker_family="spot_oracle_0A",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"state protection oracle failed: {exc}",
        )
        upsert_plan(
            status="failed",
            kind="analysis",
            script=script_path,
            dataset=dataset,
            split=split,
            tracker_family="spot_oracle_0A",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"state protection oracle failed: {exc}",
            key=f"spot_oracle_0A:{out_dir}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
