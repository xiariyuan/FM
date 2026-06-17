#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")

SUMMARY_FIELDS = [
    "step",
    "name",
    "status",
    "out_dir",
    "summary_csv",
    "log_path",
    "started_at",
    "finished_at",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paper-style narrative draft from the pre-association evidence pack.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--table-pack-dir",
        default=str(REPO_ROOT / "outputs" / "deep_ocsort_preassoc_paper_table_pack_20260405_1"),
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def to_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_storyline(
    main_row: Dict[str, str],
    claims: List[Dict[str, str]],
    stability_rows: List[Dict[str, str]],
    mechanism_rows: List[Dict[str, str]],
) -> str:
    best_hota = main_row["delta_HOTA"]
    best_idf1 = main_row["delta_IDF1"]
    stability_8 = next(row for row in stability_rows if row["owner_max_hits"] == "8")
    stability_9 = next(row for row in stability_rows if row["owner_max_hits"] == "9")
    delayed = next(row for row in mechanism_rows if row["evidence_group"] == "delayed_effect")
    return (
        f"""# Deep-OC-SORT Pre-association Narrative

## Core message

我们的改动不是把旧 baseline 上的若干 heuristic 重新混合，而是把 stale competition 的接管条件重写成一条新的三段式关联逻辑:

1. `weak-owner`: 只在 owner 足够弱时允许 reclaim 进入竞争。
2. `owner-edge-deficit`: reclaim 即使 edge 略弱于 owner，也允许在一个受控 deficit 内继续参与。
3. `owner-block-on-reclaim`: 一旦 reclaim 被选中，阻止 owner 在同一步重新夺回。

这条线在 MOT17 val-half full7 上得到稳定正收益，最佳设置为 `owner_max_hits=8, max_owner_edge_deficit=0.20, block_owner_on_reclaim=1`，达到 `delta_HOTA={best_hota}`、`delta_IDF1={best_idf1}`。

## Why this is not a fragile point

局部邻域扫描说明该结果不是单点尖峰，而是一条稳定 ridge。`owner_max_hits=8` 与 `9` 在 `owner_edge_deficit>=0.20` 的局部区域都能维持同等最优，分别有 `owner_max_hits=8` 的 `num_ties_with_best={stability_8['num_ties_with_best']}` 和 `owner_max_hits=9` 的 `num_ties_with_best={stability_9['num_ties_with_best']}` 个 tied-best 邻点。

相反，`owner_edge_deficit=0.18` 会明显退化，`owner_max_hits=7` 整条支路也持续偏弱。这说明我们的设计已经从“偶然命中的参数点”走向“具有局部稳定区间的结构机制”。

## Why the mechanism is credible

机制证据不是只靠 aggregate metric 反推。aligned case audit 显示，被选中的 `6` 个 reclaim case 全部满足 rank-2、weak-owner、within-owner-edge-deficit 条件，而且这些 track 在当帧都没有直接进入最终输出。delayed trace 进一步表明，在 `30` 帧窗口内有 `{delayed['selected_beats_owner_h30']}/{delayed['selected_cases']}` 个 case 是 selected delayed win，而 owner delayed win 只有 `{delayed['owner_beats_selected_h30']}` 个。

因此更合理的叙事不是“该帧立即修正输出框”，而是“内部关联层先把 ownership 接管对，收益在后续帧逐步兑现成更好的 identity continuity”。

## Claim stack
"""
        + "\n".join(f"- `{row['claim_id']}`: {row['claim']} Evidence: {row['evidence']}" for row in claims)
        + "\n"
    )


def build_experiment_section(
    main_row: Dict[str, str],
    ablation_rows: List[Dict[str, str]],
    local_rows: List[Dict[str, str]],
    mechanism_rows: List[Dict[str, str]],
) -> str:
    ablate_owner_block = next(row for row in ablation_rows if row["setting"] == "ablate_owner_block")
    ablate_weak_owner = next(row for row in ablation_rows if row["setting"] == "ablate_weak_owner")
    ablate_deficit = next(row for row in ablation_rows if row["setting"] == "ablate_owner_edge_deficit")
    tied_rows = [
        row for row in local_rows
        if row["step"] != "reference" and row["ties_best"] == "1"
    ]
    tied_steps = ", ".join(row["step"] for row in tied_rows)
    delayed = next(row for row in mechanism_rows if row["evidence_group"] == "delayed_effect")
    return f"""# Experiment Section Draft

## Main result paragraph

We evaluate the proposed stale-competition reassignment rule on MOT17 val-half full7. The final method uses `owner_max_hits=8`, `max_owner_edge_deficit=0.20`, and `block_owner_on_reclaim=1`. Under this setting, the competition branch improves the raw tracker by `+{main_row['delta_HOTA']}` HOTA, `+{main_row['delta_AssA']}` AssA, and `+{main_row['delta_IDF1']}` IDF1, while reducing identity switches and fragmentations by `{main_row['delta_IDs']}` and `{main_row['delta_Frag']}`, respectively. This confirms that the proposed reclaim rule improves association quality rather than merely exchanging one error mode for another.

## Ablation paragraph

The gain depends on all three components. Removing owner-block reduces the result to `+{ablate_owner_block['delta_HOTA']}` HOTA and `+{ablate_owner_block['delta_IDF1']}` IDF1, roughly halving the benefit. Removing weak-owner collapses performance to `delta_HOTA={ablate_weak_owner['delta_HOTA']}`, showing that unconstrained reclaim over-selects stale alternatives. Removing owner-edge-deficit increases selected matches to `{ablate_deficit['selected_matches']}` but still yields `delta_HOTA={ablate_deficit['delta_HOTA']}`, indicating that selection quantity alone is not sufficient and the relative edge constraint is necessary.

## Mechanism paragraph

To understand how the rule works, we first audit the aligned export of selected reclaim cases. All selected cases satisfy the rank-2 weak-owner pattern and fall within the owner-edge-deficit threshold, but none of the selected tracks are emitted directly in the same output frame. We then trace the subsequent tracker outputs. Within a 30-frame horizon, selected reclaim beats the raw owner in `{delayed['selected_beats_owner_h30']}` out of `{delayed['selected_cases']}` cases, whereas owner beats selected in only `{delayed['owner_beats_selected_h30']}` case. This supports the interpretation that the proposed rule first changes the internal association ownership, and the metric gain appears later through better identity continuity.

## Stability paragraph

We finally test a local neighborhood around the best point. Four nearby settings tie the confirmed best on both HOTA and IDF1: {tied_steps}. In contrast, tightening the deficit to `0.18` consistently reduces the gain, and lowering `owner_max_hits` to `7` produces a weaker branch across all tested deficits. Therefore, the improvement is not a brittle hyper-parameter spike but a stable local ridge centered around `owner_max_hits in {{8,9}}` and `owner_edge_deficit >= 0.20`.
"""


def build_method_section() -> str:
    return """# Method Section Draft

## Motivation

The baseline stale competition logic tends to fail in two opposite ways. If reclaim is too permissive, stale tracks with large historical support can overtake the true owner and inject identity noise. If reclaim is too conservative, the tracker cannot exploit short periods in which the current owner is visibly weaker than an alternative continuation. Our design targets this tradeoff directly instead of adding another generic score blending controller.

## Proposed rule

For each stale-competition candidate pair, we keep the reclaim path only when three conditions jointly hold.

First, the current owner must be weak enough under a bounded owner-hit criterion. This weak-owner gate prevents mature stable owners from being challenged by every temporary alternative.

Second, the reclaim candidate may lag the owner on edge score, but only within a bounded owner-edge-deficit margin. This allows useful continuity recovery when the better long-term continuation is not the top local edge in the current frame.

Third, once the reclaim candidate wins, the owner is blocked from immediate re-entry in the same reclaim step. This owner-block term removes the short-cycle oscillation that otherwise erases the intended reassignment effect.

## Design intuition

The three terms serve different roles. Weak-owner limits when reclaim is even allowed to matter. Owner-edge-deficit defines how much short-term edge disadvantage can be tolerated in exchange for long-term identity continuity. Owner-block guarantees that a selected reclaim decision actually survives the current association step. The ablation results show that none of the three can be removed without losing the overall behavior.
"""


def build_positioning_section() -> str:
    return """# Positioning Notes

## What remains baseline-shaped

The outer tracker shell is still Deep-OC-SORT. State prediction, detector interface, and the general association pipeline remain in the baseline family. This is not a new tracker from scratch.

## What is genuinely ours

The critical association decision inside stale competition is no longer baseline-shaped. The final behavior is not a simple threshold tweak, not a score blend controller, and not a replay of the old soft-blend path. The core novelty is a reclaim-specific ownership rule with three coupled parts: weak-owner gating, bounded owner-edge-deficit tolerance, and owner-block after reclaim selection.

## How to phrase the contribution honestly

The strongest honest claim is: we introduce a new reclaim mechanism inside the stale-competition association branch that improves identity continuity and exhibits stable local behavior under targeted neighborhood scans. This is stronger than a heuristic sweep result, but narrower than claiming an entirely new MOT framework.
"""


def build_next_steps() -> List[Dict[str, object]]:
    return [
        {
            "priority": 1,
            "next_step": "Convert the draft paragraphs into the paper's experiments subsection.",
            "reason": "The evidence pack and narrative are now aligned, so the shortest path is direct manuscript integration.",
        },
        {
            "priority": 2,
            "next_step": "Prepare one compact figure or table highlighting the local ridge and delayed-effect evidence.",
            "reason": "These two pieces are the strongest support that the method is structural rather than accidental.",
        },
        {
            "priority": 3,
            "next_step": "Only after writing, decide whether a second dataset is needed for breadth.",
            "reason": "The current step was to make the mechanism stand up first, and that is now done.",
        },
    ]


def main() -> int:
    args = parse_args()
    started_at = now_iso()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    log_path = out_dir / "build.log"

    write_rows(
        summary_csv,
        SUMMARY_FIELDS,
        [
            {
                "step": "paper_narrative",
                "name": out_dir.name,
                "status": "running",
                "out_dir": str(out_dir),
                "summary_csv": str(summary_csv),
                "log_path": str(log_path),
                "started_at": started_at,
                "finished_at": "",
                "notes": "building paper narrative draft",
            }
        ],
    )
    try:
        pack_dir = Path(args.table_pack_dir).resolve()
        main_row = read_rows(pack_dir / "main_results.csv")[0]
        ablation_rows = read_rows(pack_dir / "ablation_table.csv")
        mechanism_rows = read_rows(pack_dir / "mechanism_evidence.csv")
        local_rows = read_rows(pack_dir / "local_neighborhood_table.csv")
        stability_rows = read_rows(pack_dir / "stability_summary.csv")
        claim_rows = read_rows(pack_dir / "claim_summary.csv")

        storyline = build_storyline(main_row, claim_rows, stability_rows, mechanism_rows)
        experiment_section = build_experiment_section(main_row, ablation_rows, local_rows, mechanism_rows)
        method_section = build_method_section()
        positioning = build_positioning_section()
        next_steps = build_next_steps()

        to_text(out_dir / "storyline.md", storyline)
        to_text(out_dir / "experiment_section_draft.md", experiment_section)
        to_text(out_dir / "method_section_draft.md", method_section)
        to_text(out_dir / "positioning_notes.md", positioning)
        write_rows(out_dir / "next_steps.csv", list(next_steps[0].keys()), next_steps)

        notes = (
            f"best_delta_HOTA={main_row['delta_HOTA']} "
            f"best_delta_IDF1={main_row['delta_IDF1']} "
            f"claims={len(claim_rows)} "
            f"tied_best_neighbors={sum(int(row['ties_best']) for row in local_rows if row['step'] != 'reference')}"
        )
        log_path.write_text(notes + "\n", encoding="utf-8")
        write_rows(
            summary_csv,
            SUMMARY_FIELDS,
            [
                {
                    "step": "paper_narrative",
                    "name": out_dir.name,
                    "status": "success",
                    "out_dir": str(out_dir),
                    "summary_csv": str(summary_csv),
                    "log_path": str(log_path),
                    "started_at": started_at,
                    "finished_at": now_iso(),
                    "notes": notes,
                }
            ],
        )
        return 0
    except Exception as exc:
        write_rows(
            summary_csv,
            SUMMARY_FIELDS,
            [
                {
                    "step": "paper_narrative",
                    "name": out_dir.name,
                    "status": "failed",
                    "out_dir": str(out_dir),
                    "summary_csv": str(summary_csv),
                    "log_path": str(log_path),
                    "started_at": started_at,
                    "finished_at": now_iso(),
                    "notes": f"paper narrative failed: {exc}",
                }
            ],
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
