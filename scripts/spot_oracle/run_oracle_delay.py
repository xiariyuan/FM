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
    "recoverable_at_2",
    "recoverable_at_5",
    "recoverable_at_10",
    "median_evidence_latency",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle 0B proxy for delayed commitment.")
    parser.add_argument("--alignment-json", required=True)
    parser.add_argument("--out-dir", default="outputs/oracle_gate/0B_delayed_commitment")
    parser.add_argument("--delays", default="2,5,10")
    parser.add_argument("--trusted-only", action="store_true", help="mark output as untrusted if off")
    return parser.parse_args()


def _switch_events(rows: list[dict]) -> list[dict]:
    by_track: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if int(row.get("gt_id", -1)) > 0:
            by_track[int(row["track_id"])].append(row)
    events: list[dict] = []
    for track_id, track_rows in by_track.items():
        ordered = sorted(track_rows, key=lambda item: int(item["frame"]))
        for prev, cur in zip(ordered[:-1], ordered[1:]):
            prev_gt = int(prev["gt_id"])
            cur_gt = int(cur["gt_id"])
            if prev_gt <= 0 or cur_gt <= 0 or prev_gt == cur_gt:
                continue
            latency = None
            for future in ordered:
                frame = int(future["frame"])
                if frame <= int(cur["frame"]):
                    continue
                if int(future["gt_id"]) == prev_gt:
                    latency = frame - int(cur["frame"])
                    break
            events.append(
                {
                    "track_id": track_id,
                    "frame": int(cur["frame"]),
                    "from_gt": prev_gt,
                    "to_gt": cur_gt,
                    "evidence_latency": latency,
                }
            )
    return events


def main() -> int:
    args = parse_args()
    delays = [int(chunk) for chunk in str(args.delays).split(",") if chunk.strip()]
    out_dir = ensure_dir(args.out_dir)
    summary_csv = out_dir / "summary.csv"
    script_path = str(Path(__file__).resolve().relative_to(REPO_ROOT))
    payload = read_json(args.alignment_json)
    dataset = str(payload.get("dataset", "unknown"))
    split = str(payload.get("split", "unknown"))
    seq_name = str(payload.get("seq_name", "unknown_seq"))
    variant = out_dir.name
    tag = variant
    summary_row = {
        "status": "running",
        "error": "",
        "dataset": dataset,
        "split": split,
        "seq_name": seq_name,
        "switch_events": 0,
        "recoverable_at_2": 0,
        "recoverable_at_5": 0,
        "recoverable_at_10": 0,
        "median_evidence_latency": "",
    }
    write_single_row_csv(summary_csv, summary_row, SUMMARY_FIELDS)
    append_registry(
        kind="analysis",
        status="running",
        script=script_path,
        dataset=dataset,
        split=split,
        tracker_family="spot_oracle_0B",
        variant=variant,
        tag=tag,
        run_root=out_dir,
        summary_csv=summary_csv,
        notes=f"delayed commitment oracle running for {seq_name}",
    )
    upsert_plan(
        status="running",
        kind="analysis",
        script=script_path,
        dataset=dataset,
        split=split,
        tracker_family="spot_oracle_0B",
        variant=variant,
        tag=tag,
        run_root=out_dir,
        summary_csv=summary_csv,
        notes=f"delayed commitment oracle running for {seq_name}",
        key=f"spot_oracle_0B:{out_dir}",
    )

    try:
        events = _switch_events(list(payload.get("rows", [])))
        latencies = [int(event["evidence_latency"]) for event in events if event.get("evidence_latency") is not None]
        metrics = {
            "num_switch_events": len(events),
            "recoverable": {},
            "median_evidence_latency": median_or_none(latencies),
            "recoverable_rate": {},
        }
        for delay in delays:
            count = sum(
                1
                for event in events
                if event.get("evidence_latency") is not None and int(event["evidence_latency"]) <= delay
            )
            metrics["recoverable"][str(delay)] = count
            metrics["recoverable_rate"][str(delay)] = round(safe_ratio(count, len(events)), 6)
        summary_row.update(
            {
                "status": "completed",
                "switch_events": len(events),
                "recoverable_at_2": metrics["recoverable"].get("2", 0),
                "recoverable_at_5": metrics["recoverable"].get("5", 0),
                "recoverable_at_10": metrics["recoverable"].get("10", 0),
                "median_evidence_latency": metrics["median_evidence_latency"],
            }
        )
        write_single_row_csv(summary_csv, summary_row, SUMMARY_FIELDS)
        write_json(metrics, out_dir / "oracle_delay_metrics.json")
        write_json({"events": events}, out_dir / "delay_events.json")
        lines = [
            "# Oracle 0B Delayed Commitment",
            "",
            f"- seq_name: {seq_name}",
            f"- num_switch_events: {len(events)}",
            f"- median_evidence_latency: {metrics['median_evidence_latency']}",
        ]
        for delay in delays:
            lines.append(f"- recoverable_at_{delay}: {metrics['recoverable'][str(delay)]}")
            lines.append(f"- recoverable_rate_at_{delay}: {metrics['recoverable_rate'][str(delay)]}")
        write_markdown("\n".join(lines), out_dir / "oracle_delay_report.md")
        write_manifest(
            out_dir,
            phase="oracle_0B_delay",
            script=script_path,
            args=vars(args),
            status="ok",
            metrics=metrics,
            artifacts={
                "summary_csv": str(summary_csv),
                "metrics_json": str(out_dir / "oracle_delay_metrics.json"),
                "events_json": str(out_dir / "delay_events.json"),
            },
            notes=f"delayed commitment oracle for {seq_name}",
        )
        append_registry(
            kind="analysis",
            status="success",
            script=script_path,
            dataset=dataset,
            split=split,
            tracker_family="spot_oracle_0B",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"delayed commitment oracle complete for {seq_name}",
        )
        upsert_plan(
            status="completed",
            kind="analysis",
            script=script_path,
            dataset=dataset,
            split=split,
            tracker_family="spot_oracle_0B",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"delayed commitment oracle complete for {seq_name}",
            key=f"spot_oracle_0B:{out_dir}",
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
            tracker_family="spot_oracle_0B",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"delayed commitment oracle failed: {exc}",
        )
        upsert_plan(
            status="failed",
            kind="analysis",
            script=script_path,
            dataset=dataset,
            split=split,
            tracker_family="spot_oracle_0B",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"delayed commitment oracle failed: {exc}",
            key=f"spot_oracle_0B:{out_dir}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
