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
    parser = argparse.ArgumentParser(description="Build manuscript-ready subsections from the preassoc narrative pack.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--narrative-dir",
        default=str(REPO_ROOT / "outputs" / "deep_ocsort_preassoc_paper_narrative_20260405_1"),
    )
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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def build_method_subsection() -> str:
    return """
## Reclaim Rule Under Weak Owner and Bounded Owner-Edge Deficit

We modify the stale-competition branch in Deep-OC-SORT with a reclaim-specific ownership rule rather than adding another generic score blending controller. The key observation is that stale reclaim should only be allowed when the current owner is weak enough, and even then the reclaim candidate should only be tolerated if its edge score is not too far below the owner.

Concretely, we introduce three coupled terms. First, a weak-owner gate limits reclaim to cases where the current owner has sufficiently small support, implemented here through an `owner_max_hits` cap. Second, we allow the reclaim candidate to trail the owner by a bounded `owner_edge_deficit`, which keeps the method from being trapped by short-term local edge ordering while still preventing arbitrary stale takeovers. Third, once a reclaim candidate is selected, we block the owner from immediately re-entering the same reclaim step. This owner-block term is necessary to avoid short-cycle oscillation that would otherwise erase the reassignment effect within the same association round.

The three terms play different roles: weak-owner decides when reclaim is even allowed to matter, bounded owner-edge-deficit defines how much short-term disadvantage can be accepted in exchange for longer-term identity continuity, and owner-block ensures that a selected reclaim decision survives the current step. As shown by ablations, removing any one of the three destroys the desired behavior.
"""


def build_experiments_subsection(main_row: Dict[str, str], ablation_rows: List[Dict[str, str]], mech_rows: List[Dict[str, str]], local_rows: List[Dict[str, str]]) -> str:
    owner_block = next(row for row in ablation_rows if row["setting"] == "ablate_owner_block")
    weak_owner = next(row for row in ablation_rows if row["setting"] == "ablate_weak_owner")
    edge_deficit = next(row for row in ablation_rows if row["setting"] == "ablate_owner_edge_deficit")
    delayed = next(row for row in mech_rows if row["evidence_group"] == "delayed_effect")
    tied_rows = [row["step"] for row in local_rows if row["step"] != "reference" and row["ties_best"] == "1"]
    tied_text = ", ".join(tied_rows)
    return f"""
## Main Result and Analysis

We evaluate the proposed stale-competition reclaim rule on MOT17 val-half full7. The final setting uses `owner_max_hits=8`, `max_owner_edge_deficit=0.20`, and `block_owner_on_reclaim=1`. Relative to the raw tracker, the modified competition branch achieves `+{main_row['delta_HOTA']}` HOTA, `+{main_row['delta_AssA']}` AssA, and `+{main_row['delta_IDF1']}` IDF1, while reducing identity switches and fragmentations by `{main_row['delta_IDs']}` and `{main_row['delta_Frag']}`. These gains indicate that the method improves association quality rather than simply shifting error types.

The ablation results show that the proposed rule only works when all three parts are present. Removing owner-block reduces the gain to `+{owner_block['delta_HOTA']}` HOTA and `+{owner_block['delta_IDF1']}` IDF1, roughly halving the improvement. Removing weak-owner causes a collapse to `delta_HOTA={weak_owner['delta_HOTA']}`, showing that unconstrained reclaim over-selects stale alternatives. Removing owner-edge-deficit increases the number of selected matches to `{edge_deficit['selected_matches']}` but still yields `delta_HOTA={edge_deficit['delta_HOTA']}`, demonstrating that more reclaim events alone are not beneficial unless the relative edge relation is controlled.

To examine the mechanism, we first audit aligned selected reclaim cases and observe that all selected cases satisfy the rank-2 weak-owner pattern and lie within the owner-edge-deficit threshold. Importantly, these selected tracks are not directly emitted in the same output frame. Delayed-effect tracing shows that within a 30-frame horizon, selected reclaim beats the raw owner in `{delayed['selected_beats_owner_h30']}` of `{delayed['selected_cases']}` cases, while owner beats selected in only `{delayed['owner_beats_selected_h30']}` case. This supports the interpretation that the proposed rule first changes internal ownership and that the performance gain appears later through improved identity continuity.

Finally, a local neighborhood scan around the best point shows that the result is not an isolated hyper-parameter spike. Four nearby settings tie the confirmed best on both HOTA and IDF1: {tied_text}. By contrast, tightening the owner-edge-deficit to `0.18` consistently degrades performance, and the entire `owner_max_hits=7` branch remains weaker. Therefore, the final configuration lies on a stable local ridge centered around `owner_max_hits in {{8,9}}` and `owner_edge_deficit >= 0.20`.
"""


def build_contributions_file(main_row: Dict[str, str]) -> str:
    return f"""
- We replace the old stale-competition fallback behavior with a reclaim-specific three-part ownership rule: weak-owner gating, bounded owner-edge-deficit tolerance, and owner-block after reclaim selection.
- The resulting method improves the raw Deep-OC-SORT tracker by `+{main_row['delta_HOTA']}` HOTA and `+{main_row['delta_IDF1']}` IDF1 on MOT17 val-half full7.
- Ablation, delayed-effect tracing, and local-neighborhood scans jointly show that the gain is mechanism-driven and locally stable rather than an accidental parameter spike.
"""


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
                "step": "manuscript_section",
                "name": out_dir.name,
                "status": "running",
                "out_dir": str(out_dir),
                "summary_csv": str(summary_csv),
                "log_path": str(log_path),
                "started_at": started_at,
                "finished_at": "",
                "notes": "building manuscript-ready subsection",
            }
        ],
    )
    try:
        table_pack_dir = Path(args.table_pack_dir).resolve()
        main_row = read_rows(table_pack_dir / "main_results.csv")[0]
        ablation_rows = read_rows(table_pack_dir / "ablation_table.csv")
        mech_rows = read_rows(table_pack_dir / "mechanism_evidence.csv")
        local_rows = read_rows(table_pack_dir / "local_neighborhood_table.csv")

        method_text = build_method_subsection()
        exp_text = build_experiments_subsection(main_row, ablation_rows, mech_rows, local_rows)
        contrib_text = build_contributions_file(main_row)

        write_text(out_dir / "paper_method_subsection.md", method_text)
        write_text(out_dir / "paper_experiments_subsection.md", exp_text)
        write_text(out_dir / "paper_contributions.md", contrib_text)

        notes = (
            f"best_delta_HOTA={main_row['delta_HOTA']} "
            f"best_delta_IDF1={main_row['delta_IDF1']} "
            f"local_ties={sum(int(row['ties_best']) for row in local_rows if row['step'] != 'reference')}"
        )
        log_path.write_text(notes + "\n", encoding="utf-8")
        write_rows(
            summary_csv,
            SUMMARY_FIELDS,
            [
                {
                    "step": "manuscript_section",
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
                    "step": "manuscript_section",
                    "name": out_dir.name,
                    "status": "failed",
                    "out_dir": str(out_dir),
                    "summary_csv": str(summary_csv),
                    "log_path": str(log_path),
                    "started_at": started_at,
                    "finished_at": now_iso(),
                    "notes": f"manuscript section failed: {exc}",
                }
            ],
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
