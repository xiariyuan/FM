#!/usr/bin/env python3
"""Run or summarize a three-condition SPOT parity audit.

The audit has three intended conditions:
  00_baseline       no SPOT flags
  01_spot_observe   --spot-enable only; must be track-output identical
  02_spot_freeze    --spot-enable --spot-freeze-app; may change outputs

This script intentionally does not hard-code a dataset or tracker command. You can
use it in two modes:

1) Reuse existing result directories:
   python scripts/run_spot_parity_audit.py \
     --out-root outputs/spot_runtime/e1e1121_parity_mot20_05 \
     --baseline-results-dir outputs/.../00_baseline/track_results \
     --observe-results-dir outputs/.../01_spot_observe/track_results \
     --freeze-results-dir outputs/.../02_spot_freeze_app/track_results

2) Run commands and then compare their outputs. Command strings may use
   {run_dir} and {results_dir} placeholders, which are replaced with the
   condition working directory and default track_results directory:
   python scripts/run_spot_parity_audit.py --out-root outputs/audit \
     --baseline-cmd 'python ... --output {run_dir}' \
     --observe-cmd  'python ... --output {run_dir} --spot-enable --spot-debug-dir {run_dir}/spot_debug' \
     --freeze-cmd   'python ... --output {run_dir} --spot-enable --spot-freeze-app --spot-debug-dir {run_dir}/spot_debug'
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from spot_compare_tracking_outputs import compare_dirs  # noqa: E402


def git_value(args: list[str], default: str = "") -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return default


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or summarize SPOT baseline/observe/freeze parity audit.")
    parser.add_argument("--out-root", required=True, type=Path, help="Audit output directory")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root")
    parser.add_argument("--pattern", default="*.txt", help="Tracking result glob pattern")
    parser.add_argument("--numeric", action="store_true", help="Use numeric MOT comparison instead of byte-exact comparison")
    parser.add_argument("--tol", type=float, default=1e-6, help="Numeric tolerance when --numeric is used")
    parser.add_argument("--allow-observe-diff", action="store_true", help="Do not return failure when baseline vs observe differs")
    parser.add_argument("--skip-freeze", action="store_true", help="Do not require or compare the freeze condition")

    parser.add_argument("--baseline-results-dir", type=Path, default=None)
    parser.add_argument("--observe-results-dir", type=Path, default=None)
    parser.add_argument("--freeze-results-dir", type=Path, default=None)

    parser.add_argument("--baseline-cmd", default=None, help="Shell command for baseline condition")
    parser.add_argument("--observe-cmd", default=None, help="Shell command for observe-only condition")
    parser.add_argument("--freeze-cmd", default=None, help="Shell command for freeze condition")
    return parser.parse_args(list(argv) if argv is not None else None)


def condition_paths(out_root: Path, condition: str) -> tuple[Path, Path]:
    run_dir = out_root / condition
    return run_dir, run_dir / "track_results"


def render_command(command: str, run_dir: Path, results_dir: Path) -> str:
    return command.format(run_dir=str(run_dir), results_dir=str(results_dir))


def run_condition(command: Optional[str], condition: str, out_root: Path, repo_root: Path) -> Optional[Path]:
    if not command:
        return None
    run_dir, results_dir = condition_paths(out_root, condition)
    run_dir.mkdir(parents=True, exist_ok=True)
    rendered = render_command(command, run_dir, results_dir)
    log_path = run_dir / "command.log"
    env = dict(os.environ)
    env.update(
        {
            "SPOT_AUDIT_CONDITION": condition,
            "SPOT_AUDIT_RUN_DIR": str(run_dir),
            "SPOT_AUDIT_RESULTS_DIR": str(results_dir),
        }
    )
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {rendered}\n\n")
        proc = subprocess.run(
            rendered,
            cwd=repo_root,
            shell=True,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"{condition} command failed with exit code {proc.returncode}; see {log_path}")
    return results_dir


def resolve_results_dir(explicit_dir: Optional[Path], command: Optional[str], condition: str, out_root: Path, repo_root: Path) -> Path:
    ran_dir = run_condition(command, condition, out_root, repo_root)
    if explicit_dir is not None:
        return explicit_dir
    if ran_dir is not None:
        return ran_dir
    raise ValueError(f"Missing --{condition.split('_', 1)[1] if '_' in condition else condition}-results-dir or command for {condition}")


def summarize_spot_csv(run_dir: Path) -> Dict[str, Any]:
    candidates = sorted(run_dir.rglob("*_summary.csv")) + sorted(run_dir.rglob("spot_summary.csv"))
    if not candidates:
        return {"summary_found": False}
    # Keep this parser deliberately simple: SPOT summaries are one-row CSV files.
    import csv

    path = candidates[0]
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    data: Dict[str, Any] = {"summary_found": True, "summary_csv": str(path), "rows": len(rows)}
    if rows:
        for key in [
            "enabled",
            "freeze_app",
            "frames",
            "primary_matches",
            "ambiguous_matches",
            "forced_freeze_updates",
            "already_freeze_updates",
            "ambiguity_rate",
            "freeze_rate",
            "pairs_csv",
        ]:
            if key in rows[0]:
                data[key] = rows[0][key]
    return data


def write_markdown_report(path: Path, manifest: Dict[str, Any], observe_report: Dict[str, Any], freeze_report: Optional[Dict[str, Any]]) -> None:
    lines = [
        "# SPOT Parity Audit Report",
        "",
        f"- Generated: `{manifest['generated_at']}`",
        f"- Git SHA: `{manifest['git_sha']}`",
        f"- Git short SHA: `{manifest['git_short_sha']}`",
        f"- Compare mode: `{observe_report['mode']}`",
        f"- Pattern: `{observe_report['pattern']}`",
        "",
        "## Baseline vs observe-only",
        "",
        f"- Parity OK: **{observe_report['parity_ok']}**",
        f"- Baseline files: `{observe_report['baseline_file_count']}`",
        f"- Observe files: `{observe_report['candidate_file_count']}`",
        f"- Missing in observe: `{len(observe_report['missing_in_candidate'])}`",
        f"- Extra in observe: `{len(observe_report['extra_in_candidate'])}`",
        f"- Changed files: `{len(observe_report['changed'])}`",
        "",
    ]
    if observe_report["changed"]:
        lines.extend(["### Changed files", ""])
        for item in observe_report["changed"][:50]:
            lines.append(f"- `{item.get('file', '')}`: `{item.get('reason', 'sha256 differs')}`")
        if len(observe_report["changed"]) > 50:
            lines.append(f"- ... {len(observe_report['changed']) - 50} more")
        lines.append("")

    observe_summary = manifest["conditions"].get("01_spot_observe", {}).get("spot_summary", {})
    lines.extend(["## Observe-only SPOT summary", ""])
    if observe_summary.get("summary_found"):
        for key, value in observe_summary.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- No SPOT summary CSV found. This is acceptable only if the run did not enable `--spot-enable` or did not complete summary writing.")
    lines.append("")

    if freeze_report is not None:
        lines.extend(
            [
                "## Baseline vs freeze-app",
                "",
                f"- Outputs changed: **{not freeze_report['parity_ok']}**",
                f"- Freeze files: `{freeze_report['candidate_file_count']}`",
                f"- Missing in freeze: `{len(freeze_report['missing_in_candidate'])}`",
                f"- Extra in freeze: `{len(freeze_report['extra_in_candidate'])}`",
                f"- Changed files: `{len(freeze_report['changed'])}`",
                "",
            ]
        )
        freeze_summary = manifest["conditions"].get("02_spot_freeze_app", {}).get("spot_summary", {})
        lines.extend(["## Freeze-app SPOT summary", ""])
        if freeze_summary.get("summary_found"):
            for key, value in freeze_summary.items():
                lines.append(f"- `{key}`: `{value}`")
        else:
            lines.append("- No SPOT summary CSV found.")
        lines.append("")

    lines.extend(
        [
            "## Gate interpretation",
            "",
            "- If baseline vs observe-only is not exactly identical, stop and debug observation-layer side effects before interpreting freeze results.",
            "- If observe-only parity passes, freeze-app results may be evaluated, but `runtime_patch_allowed` remains 0 until real paired metrics are positive.",
            "- This report checks tracker-output parity; HOTA/IDSW/AssA metric evaluation must be run separately or layered on top of this manifest.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    repo_root = args.repo_root.resolve()

    baseline_dir = resolve_results_dir(args.baseline_results_dir, args.baseline_cmd, "00_baseline", out_root, repo_root)
    observe_dir = resolve_results_dir(args.observe_results_dir, args.observe_cmd, "01_spot_observe", out_root, repo_root)
    freeze_dir: Optional[Path] = None
    if not args.skip_freeze or args.freeze_results_dir or args.freeze_cmd:
        freeze_dir = resolve_results_dir(args.freeze_results_dir, args.freeze_cmd, "02_spot_freeze_app", out_root, repo_root)

    observe_report = compare_dirs(baseline_dir, observe_dir, args.pattern, args.numeric, args.tol)
    freeze_report = compare_dirs(baseline_dir, freeze_dir, args.pattern, args.numeric, args.tol) if freeze_dir is not None else None

    manifest: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "git_sha": git_value(["rev-parse", "HEAD"]),
        "git_short_sha": git_value(["rev-parse", "--short", "HEAD"]),
        "git_status_short": git_value(["status", "--short"], default="unavailable"),
        "out_root": str(out_root),
        "pattern": args.pattern,
        "numeric": bool(args.numeric),
        "tol": float(args.tol),
        "conditions": {
            "00_baseline": {"results_dir": str(baseline_dir)},
            "01_spot_observe": {"results_dir": str(observe_dir), "spot_summary": summarize_spot_csv(out_root / "01_spot_observe")},
        },
        "observe_parity_ok": bool(observe_report["parity_ok"]),
    }
    if freeze_dir is not None:
        manifest["conditions"]["02_spot_freeze_app"] = {
            "results_dir": str(freeze_dir),
            "spot_summary": summarize_spot_csv(out_root / "02_spot_freeze_app"),
        }
        manifest["freeze_outputs_changed"] = bool(not freeze_report["parity_ok"] if freeze_report is not None else False)

    (out_root / "observe_parity_report.json").write_text(json.dumps(observe_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if freeze_report is not None:
        (out_root / "freeze_diff_report.json").write_text(json.dumps(freeze_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown_report(out_root / "parity_report.md", manifest, observe_report, freeze_report)

    print(json.dumps({"out_root": str(out_root), "observe_parity_ok": observe_report["parity_ok"], "freeze_compared": freeze_report is not None}, indent=2, sort_keys=True))
    if not observe_report["parity_ok"] and not args.allow_observe_diff:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
