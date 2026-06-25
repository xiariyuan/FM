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
    read_json,
    upsert_plan,
    write_json,
    write_manifest,
    write_markdown,
    write_single_row_csv,
)


SUMMARY_FIELDS = [
    "status",
    "error",
    "state_gain",
    "rerank_gain_proxy",
    "median_evidence_latency",
    "decision_confidence",
    "runtime_patch_allowed",
    "block_reason",
    "final_route",
    "pcc_role",
    "p5_role",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle 0E joint SPOT routing decision.")
    parser.add_argument("--state-json", required=True)
    parser.add_argument("--delay-json", required=True)
    parser.add_argument("--rerank-json", required=True)
    parser.add_argument("--out-dir", default="outputs/oracle_gate/0E_joint_oracle")
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--split", default="unknown")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    summary_csv = out_dir / "summary.csv"
    script_path = str(Path(__file__).resolve().relative_to(REPO_ROOT))
    variant = out_dir.name
    tag = variant
    summary_row = {
        "status": "running",
        "error": "",
        "state_gain": 0.0,
        "rerank_gain_proxy": 0.0,
        "median_evidence_latency": "",
        "decision_confidence": "",
        "runtime_patch_allowed": 0,
        "block_reason": "",
        "final_route": "",
        "pcc_role": "",
        "p5_role": "",
    }
    write_single_row_csv(summary_csv, summary_row, SUMMARY_FIELDS)
    append_registry(
        kind="analysis",
        status="running",
        script=script_path,
        dataset=args.dataset,
        split=args.split,
        tracker_family="spot_oracle_0E",
        variant=variant,
        tag=tag,
        run_root=out_dir,
        summary_csv=summary_csv,
        notes="joint oracle running",
    )
    upsert_plan(
        status="running",
        kind="analysis",
        script=script_path,
        dataset=args.dataset,
        split=args.split,
        tracker_family="spot_oracle_0E",
        variant=variant,
        tag=tag,
        run_root=out_dir,
        summary_csv=summary_csv,
        notes="joint oracle running",
        key=f"spot_oracle_0E:{out_dir}",
    )

    try:
        state = read_json(args.state_json)
        delay = read_json(args.delay_json)
        rerank = read_json(args.rerank_json)
        state_gain = float(state.get("idsw_reduction_percent", 0.0))
        rerank_gain = float(rerank.get("fixable_percent", 0.0))
        median_latency = delay.get("median_evidence_latency")

        block_reasons = []
        if str(state.get("status", "completed")) not in {"completed", "success", "ok"}:
            block_reasons.append("0A state oracle is not completed")
        if str(rerank.get("status", "completed")) not in {"completed", "success", "ok"}:
            block_reasons.append("0C rerank oracle is not completed")
        if str(rerank.get("analysis_scope", "full")) != "full":
            block_reasons.append("0C rerank oracle is not full-file")
        if int(rerank.get("trusted", 1)) != 1:
            block_reasons.append("0C rerank oracle is not trusted")

        if block_reasons:
            final_route = "NOT_CLOSED"
            decision_confidence = "not_closed"
            runtime_patch_allowed = 0
        elif state_gain >= 5.0:
            final_route = "SPOT_MAINLINE"
            decision_confidence = "closed"
            runtime_patch_allowed = 1
        elif state_gain >= 3.0:
            final_route = "SPOT_ABLATION_ONLY"
            decision_confidence = "closed"
            runtime_patch_allowed = 1
        else:
            final_route = "B_PLAN_P0_CAUSAL_DIAGNOSTIC"
            decision_confidence = "closed"
            runtime_patch_allowed = 0
        block_reason = "; ".join(block_reasons)

        if rerank_gain >= 10.0:
            pcc_role = "strong_support"
        elif rerank_gain >= 5.0:
            pcc_role = "support"
        else:
            pcc_role = "skip"

        p5_role = "skip"
        if median_latency is not None and 2 <= float(median_latency) <= 5 and state_gain >= 5.0:
            p5_role = "candidate_extension"

        summary_row.update(
            {
                "status": "completed",
                "state_gain": state_gain,
                "rerank_gain_proxy": rerank_gain,
                "median_evidence_latency": median_latency,
                "decision_confidence": decision_confidence,
                "runtime_patch_allowed": runtime_patch_allowed,
                "block_reason": block_reason,
                "final_route": final_route,
                "pcc_role": pcc_role,
                "p5_role": p5_role,
            }
        )
        write_single_row_csv(summary_csv, summary_row, SUMMARY_FIELDS)
        decision = {
            "state_gain": state_gain,
            "rerank_gain_proxy": rerank_gain,
            "median_evidence_latency": median_latency,
            "decision_confidence": decision_confidence,
            "runtime_patch_allowed": runtime_patch_allowed,
            "block_reason": block_reason,
            "final_route": final_route,
            "pcc_role": pcc_role,
            "p5_role": p5_role,
        }
        write_json(decision, out_dir / "joint_oracle_decision.json")
        write_markdown(
            "\n".join(
                [
                    "# Oracle 0E Joint Decision",
                    "",
                    f"- state_gain: {state_gain}",
                    f"- rerank_gain_proxy: {rerank_gain}",
                    f"- median_evidence_latency: {median_latency}",
                    f"- decision_confidence: {decision_confidence}",
                    f"- runtime_patch_allowed: {runtime_patch_allowed}",
                    f"- block_reason: {block_reason}",
                    f"- final_route: {final_route}",
                    f"- pcc_role: {pcc_role}",
                    f"- p5_role: {p5_role}",
                ]
            ),
            out_dir / "joint_oracle_decision.md",
        )
        write_manifest(
            out_dir,
            phase="oracle_0E_joint",
            script=script_path,
            args=vars(args),
            status="ok",
            metrics=decision,
            artifacts={
                "summary_csv": str(summary_csv),
                "decision_json": str(out_dir / "joint_oracle_decision.json"),
                "decision_md": str(out_dir / "joint_oracle_decision.md"),
            },
            notes="joint oracle decision complete",
        )
        append_registry(
            kind="analysis",
            status="success",
            script=script_path,
            dataset=args.dataset,
            split=args.split,
            tracker_family="spot_oracle_0E",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes="joint oracle complete",
        )
        upsert_plan(
            status="completed",
            kind="analysis",
            script=script_path,
            dataset=args.dataset,
            split=args.split,
            tracker_family="spot_oracle_0E",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes="joint oracle complete",
            key=f"spot_oracle_0E:{out_dir}",
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
            dataset=args.dataset,
            split=args.split,
            tracker_family="spot_oracle_0E",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"joint oracle failed: {exc}",
        )
        upsert_plan(
            status="failed",
            kind="analysis",
            script=script_path,
            dataset=args.dataset,
            split=args.split,
            tracker_family="spot_oracle_0E",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"joint oracle failed: {exc}",
            key=f"spot_oracle_0E:{out_dir}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
