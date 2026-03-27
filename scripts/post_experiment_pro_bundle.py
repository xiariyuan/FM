#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build and record a Pro review bundle for a finished experiment run.")
    ap.add_argument("--run-root", required=True, help="Experiment output directory to attach as latest_run evidence.")
    ap.add_argument("--tag", default="pro_review_bundle")
    ap.add_argument("--label", default="latest_run")
    ap.add_argument("--status", default="")
    ap.add_argument("--extra-evidence", nargs="*", default=[])
    return ap.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    args = parse_args()
    repo_root = _repo_root()
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "build_pro_review_bundle.py"),
        "--tag",
        args.tag,
        "--latest-run-root",
        str(run_root),
        "--latest-run-label",
        args.label,
    ]
    if args.extra_evidence:
        cmd.extend(["--extra-evidence", *args.extra_evidence])

    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode

    stdout = proc.stdout.strip()
    summary = json.loads(stdout) if stdout else {}
    bundle_record = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "status": args.status,
        "tag": args.tag,
        "label": args.label,
        "bundle_dir": summary.get("bundle_dir", ""),
        "tar_gz": summary.get("tar_gz", ""),
        "zip": summary.get("zip", ""),
        "missing_count": summary.get("missing_count", 0),
        "missing_entries": summary.get("missing_entries", []),
    }

    bundle_json = run_root / "pro_review_bundle.json"
    bundle_json.write_text(json.dumps(bundle_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(bundle_record, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
