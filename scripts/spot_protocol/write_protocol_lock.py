#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.spot_common.io_utils import (
    append_registry,
    ensure_dir,
    now_iso,
    upsert_plan,
    write_manifest,
    write_markdown,
    write_single_row_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write SPOT protocol lock artifacts.")
    parser.add_argument("--out-dir", default="outputs/protocol_lock")
    parser.add_argument("--dataset", default="MOT20")
    parser.add_argument("--split", default="val")
    parser.add_argument("--title", default="SPOT-Track Phase 0 Protocol Lock")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    summary_csv = out_dir / "summary.csv"
    variant = out_dir.name
    tag = variant
    script_path = str(Path(__file__).resolve().relative_to(REPO_ROOT))

    summary_row = {
        "status": "completed",
        "phase": "protocol_lock",
        "title": args.title,
        "dataset": args.dataset,
        "split": args.split,
        "forbid_tracker_core_edits": 1,
        "next_gate": "oracle_0A_state_protection",
        "generated_at": now_iso(),
        "notes": args.notes,
    }
    write_single_row_csv(summary_csv, summary_row)

    md = f"""# {args.title}

## Objective
- Lock the first SPOT execution round to protocol scaffolding, GT alignment, and oracle analysis only.

## Allowed Work
- Add new files under `scripts/spot_common/`, `scripts/spot_protocol/`, `scripts/spot_p0/`, `scripts/spot_oracle/`
- Generate structured outputs under a dedicated `outputs/` run root
- Run smoke analysis on synthetic or tiny local fixtures

## Forbidden Work
- Do not modify `external/BoT-SORT-main/tracker/*.py`
- Do not modify `external/BoT-SORT-main/tools/track.py`
- Do not train ADG
- Do not implement delayed commitment runtime behavior

## Required Go/No-Go Order
1. Protocol lock
2. GT alignment
3. Oracle 0A and Oracle 0C
4. Oracle 0D and Oracle 0B
5. Oracle 0E joint decision

## Current Decision
- Runtime tracker patches are blocked until oracle evidence is written and reviewed.

## Notes
- {args.notes or "none"}
"""
    write_markdown(md, out_dir / "protocol_lock.md")
    write_manifest(
        out_dir,
        phase="protocol_lock",
        script=script_path,
        args=vars(args),
        status="ok",
        metrics={"forbid_tracker_core_edits": 1, "next_gate": "oracle_0A_state_protection"},
        artifacts={
            "summary_csv": str(summary_csv),
            "protocol_lock_md": str(out_dir / "protocol_lock.md"),
        },
        notes=args.notes,
    )
    append_registry(
        kind="analysis",
        status="success",
        script=script_path,
        dataset=args.dataset,
        split=args.split,
        tracker_family="spot_protocol",
        variant=variant,
        tag=tag,
        run_root=out_dir,
        summary_csv=summary_csv,
        notes="SPOT protocol lock written",
    )
    upsert_plan(
        status="completed",
        kind="analysis",
        script=script_path,
        dataset=args.dataset,
        split=args.split,
        tracker_family="spot_protocol",
        variant=variant,
        tag=tag,
        run_root=out_dir,
        summary_csv=summary_csv,
        notes="SPOT protocol lock complete",
        key=f"spot_protocol:{out_dir}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
