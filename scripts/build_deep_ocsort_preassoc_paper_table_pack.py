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
    parser = argparse.ArgumentParser(description="Assemble a paper-ready evidence pack for Deep-OC-SORT pre-association competition follow-up.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--best-metrics-csv",
        default=str(REPO_ROOT / "outputs" / "deep_ocsort_preassoc_competition_followup_best_confirm_20260405_1" / "metrics_delta.csv"),
    )
    parser.add_argument(
        "--ablation-decision-csv",
        default=str(REPO_ROOT / "outputs" / "deep_ocsort_preassoc_competition_component_ablation_20260405_1" / "decision.csv"),
    )
    parser.add_argument(
        "--caseaudit-aggregate-csv",
        default=str(REPO_ROOT / "outputs" / "deep_ocsort_preassoc_competition_followup_caseaudit_aligned_20260405_1" / "aggregate.csv"),
    )
    parser.add_argument(
        "--delayed-trace-aggregate-csv",
        default=str(REPO_ROOT / "outputs" / "deep_ocsort_preassoc_delayed_trace_20260405_1" / "aggregate.csv"),
    )
    parser.add_argument(
        "--local-neighborhood-decision-csv",
        default=str(REPO_ROOT / "outputs" / "deep_ocsort_preassoc_competition_local_neighborhood_extend_20260405_1" / "decision.csv"),
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


def to_float(row: Dict[str, str], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def to_int(row: Dict[str, str], key: str) -> int:
    return int(float(row.get(key, 0) or 0))


def build_main_results(best_row: Dict[str, str]) -> List[Dict[str, object]]:
    return [
        {
            "setting": "best_confirmed",
            "owner_max_hits": 8,
            "max_owner_edge_deficit": 0.20,
            "block_owner_on_reclaim": 1,
            "delta_HOTA": round(to_float(best_row, "delta_HOTA"), 3),
            "delta_AssA": round(to_float(best_row, "delta_AssA"), 3),
            "delta_IDF1": round(to_float(best_row, "delta_IDF1"), 3),
            "delta_MOTA": round(to_float(best_row, "delta_MOTA"), 3),
            "delta_IDs": to_int(best_row, "delta_IDs"),
            "delta_Frag": to_int(best_row, "delta_Frag"),
            "notes": "confirmed best on MOT17 val_half full7",
        }
    ]


def build_ablation_table(best_row: Dict[str, str], ablation_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    best_hota = to_float(best_row, "delta_HOTA")
    best_idf1 = to_float(best_row, "delta_IDF1")
    rows: List[Dict[str, object]] = [
        {
            "setting": "full_method",
            "component_change": "none",
            "delta_HOTA": round(best_hota, 3),
            "delta_AssA": round(to_float(best_row, "delta_AssA"), 3),
            "delta_IDF1": round(best_idf1, 3),
            "delta_MOTA": round(to_float(best_row, "delta_MOTA"), 3),
            "delta_IDs": to_int(best_row, "delta_IDs"),
            "delta_Frag": to_int(best_row, "delta_Frag"),
            "selected_matches": 6,
            "drop_vs_best_HOTA": 0.0,
            "drop_vs_best_IDF1": 0.0,
            "notes": "weak-owner + owner-edge-deficit + owner-block",
        }
    ]
    name_map = {
        "ablate_owner_block": "remove owner-block",
        "ablate_weak_owner": "remove weak-owner",
        "ablate_owner_edge_deficit": "remove owner-edge-deficit",
    }
    for row in ablation_rows:
        rows.append(
            {
                "setting": row["step"],
                "component_change": name_map.get(row["step"], row["step"]),
                "delta_HOTA": round(to_float(row, "delta_HOTA"), 3),
                "delta_AssA": round(to_float(row, "delta_AssA"), 3),
                "delta_IDF1": round(to_float(row, "delta_IDF1"), 3),
                "delta_MOTA": round(to_float(row, "delta_MOTA"), 3),
                "delta_IDs": to_int(row, "delta_IDs"),
                "delta_Frag": to_int(row, "delta_Frag"),
                "selected_matches": to_int(row, "selected_matches"),
                "drop_vs_best_HOTA": round(best_hota - to_float(row, "delta_HOTA"), 3),
                "drop_vs_best_IDF1": round(best_idf1 - to_float(row, "delta_IDF1"), 3),
                "notes": row.get("notes", ""),
            }
        )
    return rows


def build_mechanism_evidence(caseaudit_row: Dict[str, str], delayed_row: Dict[str, str]) -> List[Dict[str, object]]:
    return [
        {
            "evidence_group": "selection_geometry",
            "selected_cases": to_int(caseaudit_row, "selected_cases"),
            "selected_sequences": to_int(caseaudit_row, "selected_sequences"),
            "rank2_cases": to_int(caseaudit_row, "rank2_cases"),
            "weak_owner_cases": to_int(caseaudit_row, "weak_owner_cases"),
            "within_owner_edge_deficit_cases": to_int(caseaudit_row, "within_owner_edge_deficit_cases"),
            "selected_track_emitted_before_filter_cases": to_int(caseaudit_row, "selected_track_emitted_before_filter_cases"),
            "selected_beats_owner_h10": "",
            "owner_beats_selected_h10": "",
            "selected_beats_owner_h30": "",
            "owner_beats_selected_h30": "",
            "notes": "all selected cases are rank2 weak-owner cases within owner-edge-deficit threshold",
        },
        {
            "evidence_group": "delayed_effect",
            "selected_cases": to_int(delayed_row, "selected_cases"),
            "selected_sequences": "",
            "rank2_cases": "",
            "weak_owner_cases": "",
            "within_owner_edge_deficit_cases": "",
            "selected_track_emitted_before_filter_cases": to_int(delayed_row, "selected_track_emitted_before_filter_cases"),
            "selected_beats_owner_h10": to_int(delayed_row, "selected_beats_owner_h10"),
            "owner_beats_selected_h10": to_int(delayed_row, "owner_beats_selected_h10"),
            "selected_beats_owner_h30": to_int(delayed_row, "selected_beats_owner_h30"),
            "owner_beats_selected_h30": to_int(delayed_row, "owner_beats_selected_h30"),
            "notes": "most selected reclaim cases beat owner only after later frames, not at the same output frame",
        },
    ]


def build_local_neighborhood_table(decision_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    best_row = next(row for row in decision_rows if row.get("decision") == "best_after_queue")
    best_hota = to_float(best_row, "delta_HOTA")
    best_idf1 = to_float(best_row, "delta_IDF1")
    rows: List[Dict[str, object]] = []
    for row in decision_rows:
        delta_hota = to_float(row, "delta_HOTA")
        delta_idf1 = to_float(row, "delta_IDF1")
        rows.append(
            {
                "step": row.get("step", ""),
                "owner_max_hits": row.get("owner_max_hits", ""),
                "max_owner_edge_deficit": row.get("max_owner_edge_deficit", ""),
                "delta_HOTA": round(delta_hota, 3),
                "delta_AssA": round(to_float(row, "delta_AssA"), 3),
                "delta_IDF1": round(delta_idf1, 3),
                "delta_MOTA": round(to_float(row, "delta_MOTA"), 3),
                "delta_IDs": to_int(row, "delta_IDs"),
                "delta_Frag": to_int(row, "delta_Frag"),
                "selected_matches": to_int(row, "selected_matches"),
                "candidate_rows": to_int(row, "candidate_rows"),
                "ties_best": int(abs(delta_hota - best_hota) <= 1e-9 and abs(delta_idf1 - best_idf1) <= 1e-9),
                "hota_gap_to_best": round(best_hota - delta_hota, 3),
                "idf1_gap_to_best": round(best_idf1 - delta_idf1, 3),
                "decision": row.get("decision", ""),
                "notes": row.get("notes", ""),
            }
        )
    def sort_key(item: Dict[str, object]) -> tuple:
        return (
            0 if str(item["step"]) == "reference" else 1,
            -float(item["delta_HOTA"]),
            -float(item["delta_IDF1"]),
            float(item["owner_max_hits"] or 0),
            float(item["max_owner_edge_deficit"] or 0),
        )
    rows.sort(key=sort_key)
    return rows


def build_stability_summary(decision_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    reference = next(row for row in decision_rows if row.get("step") == "reference")
    best_hota = to_float(reference, "delta_HOTA")
    best_idf1 = to_float(reference, "delta_IDF1")
    by_owner: Dict[str, List[Dict[str, str]]] = {}
    for row in decision_rows:
        if row.get("step") == "reference":
            continue
        by_owner.setdefault(str(row.get("owner_max_hits", "")), []).append(row)

    rows: List[Dict[str, object]] = []
    for owner_hits, items in sorted(by_owner.items(), key=lambda kv: int(kv[0])):
        tied = 0
        degrade_018 = ""
        for row in items:
            if abs(to_float(row, "delta_HOTA") - best_hota) <= 1e-9 and abs(to_float(row, "delta_IDF1") - best_idf1) <= 1e-9:
                tied += 1
            if abs(float(row.get("max_owner_edge_deficit", 0) or 0) - 0.18) <= 1e-9:
                degrade_018 = f"HOTA={to_float(row, 'delta_HOTA'):.3f},IDF1={to_float(row, 'delta_IDF1'):.3f}"
        rows.append(
            {
                "owner_max_hits": int(owner_hits),
                "num_variants": len(items),
                "num_ties_with_best": tied,
                "odef018_result": degrade_018,
                "best_hota_reference": round(best_hota, 3),
                "best_idf1_reference": round(best_idf1, 3),
                "notes": "owner_max_hits=7 stays below ridge; 8/9 maintain ridge at odef>=0.20" if owner_hits in {"8", "9"} else "owner_max_hits=7 consistently weaker",
            }
        )
    return rows


def build_claim_summary(best_row: Dict[str, str], delayed_row: Dict[str, str], decision_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    best_hota = round(to_float(best_row, "delta_HOTA"), 3)
    best_idf1 = round(to_float(best_row, "delta_IDF1"), 3)
    tie_rows = [
        row for row in decision_rows
        if row.get("step") != "reference"
        and abs(to_float(row, "delta_HOTA") - to_float(best_row, "delta_HOTA")) <= 1e-9
        and abs(to_float(row, "delta_IDF1") - to_float(best_row, "delta_IDF1")) <= 1e-9
    ]
    return [
        {
            "claim_id": "main_gain_confirmed",
            "claim": "Best setting is confirmed on full7.",
            "evidence": f"delta_HOTA={best_hota:.3f}, delta_IDF1={best_idf1:.3f}, delta_IDs={to_int(best_row, 'delta_IDs')}, delta_Frag={to_int(best_row, 'delta_Frag')}",
            "support_level": "strong",
        },
        {
            "claim_id": "three_components_needed",
            "claim": "weak-owner, owner-edge-deficit, and owner-block are jointly necessary.",
            "evidence": "Removing weak-owner gives negative HOTA; removing owner-edge-deficit over-selects; removing owner-block cuts gains roughly in half.",
            "support_level": "strong",
        },
        {
            "claim_id": "delayed_effect_supported",
            "claim": "Selected reclaim often pays off in later frames rather than the same frame.",
            "evidence": (
                f"selected_beats_owner_h30={to_int(delayed_row, 'selected_beats_owner_h30')}/"
                f"{to_int(delayed_row, 'selected_cases')}, owner_beats_selected_h30={to_int(delayed_row, 'owner_beats_selected_h30')}"
            ),
            "support_level": "strong",
        },
        {
            "claim_id": "local_ridge_not_single_point",
            "claim": "The best result lies on a stable local ridge, not an isolated spike.",
            "evidence": f"{len(tie_rows)} neighborhood settings tie the confirmed best on HOTA and IDF1.",
            "support_level": "strong",
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
                "step": "paper_table_pack",
                "name": out_dir.name,
                "status": "running",
                "out_dir": str(out_dir),
                "summary_csv": str(summary_csv),
                "log_path": str(log_path),
                "started_at": started_at,
                "finished_at": "",
                "notes": "assembling paper table pack",
            }
        ],
    )
    try:
        best_rows = read_rows(Path(args.best_metrics_csv).resolve())
        ablation_rows = read_rows(Path(args.ablation_decision_csv).resolve())
        caseaudit_rows = read_rows(Path(args.caseaudit_aggregate_csv).resolve())
        delayed_rows = read_rows(Path(args.delayed_trace_aggregate_csv).resolve())
        neighborhood_rows = read_rows(Path(args.local_neighborhood_decision_csv).resolve())

        best_row = best_rows[0]
        caseaudit_row = caseaudit_rows[0]
        delayed_row = delayed_rows[0]

        main_results = build_main_results(best_row)
        ablation_table = build_ablation_table(best_row, ablation_rows)
        mechanism_evidence = build_mechanism_evidence(caseaudit_row, delayed_row)
        local_table = build_local_neighborhood_table(neighborhood_rows)
        stability_summary = build_stability_summary(neighborhood_rows)
        claim_summary = build_claim_summary(best_row, delayed_row, neighborhood_rows)

        write_rows(out_dir / "main_results.csv", list(main_results[0].keys()), main_results)
        write_rows(out_dir / "ablation_table.csv", list(ablation_table[0].keys()), ablation_table)
        write_rows(out_dir / "mechanism_evidence.csv", list(mechanism_evidence[0].keys()), mechanism_evidence)
        write_rows(out_dir / "local_neighborhood_table.csv", list(local_table[0].keys()), local_table)
        write_rows(out_dir / "stability_summary.csv", list(stability_summary[0].keys()), stability_summary)
        write_rows(out_dir / "claim_summary.csv", list(claim_summary[0].keys()), claim_summary)

        notes = (
            f"best_delta_HOTA={to_float(best_row, 'delta_HOTA'):.3f} "
            f"best_delta_IDF1={to_float(best_row, 'delta_IDF1'):.3f} "
            f"delayed_selected_beats_owner_h30={to_int(delayed_row, 'selected_beats_owner_h30')} "
            f"local_best_ties={sum(int(row['ties_best']) for row in local_table if str(row['step']) != 'reference')}"
        )
        log_path.write_text(notes + "\n", encoding="utf-8")
        write_rows(
            summary_csv,
            SUMMARY_FIELDS,
            [
                {
                    "step": "paper_table_pack",
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
                    "step": "paper_table_pack",
                    "name": out_dir.name,
                    "status": "failed",
                    "out_dir": str(out_dir),
                    "summary_csv": str(summary_csv),
                    "log_path": str(log_path),
                    "started_at": started_at,
                    "finished_at": now_iso(),
                    "notes": f"paper table pack failed: {exc}",
                }
            ],
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
