#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
PLAN_CSV = REPO_ROOT / "outputs" / "experiment_plan.csv"


def ensure_dir(path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def write_json(data: Any, path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def write_markdown(text: str, path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text.rstrip() + "\n", encoding="utf-8")
    return target


def write_rows(path: str | Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = list(fieldnames)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ordered})
    return target


def write_single_row_csv(path: str | Path, row: dict[str, Any], fieldnames: Iterable[str] | None = None) -> Path:
    ordered = list(fieldnames) if fieldnames is not None else list(row.keys())
    return write_rows(path, ordered, [row])


def write_manifest(
    out_dir: str | Path,
    *,
    phase: str,
    script: str,
    args: dict[str, Any],
    status: str,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, str] | None = None,
    notes: str = "",
) -> Path:
    payload = {
        "phase": phase,
        "script": script,
        "args": args,
        "status": status,
        "metrics": metrics or {},
        "artifacts": artifacts or {},
        "notes": notes,
        "generated_at": now_iso(),
    }
    return write_json(payload, ensure_dir(out_dir) / "run_manifest.json")


def _run_subprocess(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def append_registry(
    *,
    kind: str,
    status: str,
    script: str,
    dataset: str,
    split: str,
    tracker_family: str,
    variant: str,
    tag: str,
    run_root: str | Path,
    summary_csv: str | Path,
    notes: str,
    extra: dict[str, Any] | None = None,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(REGISTRY_CSV),
        "--kind",
        kind,
        "--status",
        status,
        "--script",
        script,
        "--dataset",
        dataset,
        "--split",
        split,
        "--tracker-family",
        tracker_family,
        "--variant",
        variant,
        "--tag",
        tag,
        "--run-root",
        str(Path(run_root).expanduser().resolve()),
        "--summary-csv",
        str(Path(summary_csv).expanduser().resolve()),
        "--notes",
        notes,
    ]
    if extra:
        cmd.append("--extra")
        for key, value in extra.items():
            cmd.append(f"{key}={value}")
    _run_subprocess(cmd)


def upsert_plan(
    *,
    status: str,
    kind: str,
    script: str,
    dataset: str,
    split: str,
    tracker_family: str,
    variant: str,
    tag: str,
    run_root: str | Path,
    summary_csv: str | Path,
    notes: str,
    key: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "upsert_experiment_plan.py"),
        "--csv",
        str(PLAN_CSV),
        "--status",
        status,
        "--kind",
        kind,
        "--script",
        script,
        "--dataset",
        dataset,
        "--split",
        split,
        "--tracker-family",
        tracker_family,
        "--variant",
        variant,
        "--tag",
        tag,
        "--run-root",
        str(Path(run_root).expanduser().resolve()),
        "--summary-csv",
        str(Path(summary_csv).expanduser().resolve()),
        "--notes",
        notes,
    ]
    if key:
        cmd.extend(["--key", key])
    if extra:
        cmd.append("--extra")
        for item_key, value in extra.items():
            cmd.append(f"{item_key}={value}")
    _run_subprocess(cmd)
