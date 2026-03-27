#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ACTION_TO_ID = {
    "keep": 0,
    "rerank": 1,
    "null": 2,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build group-level competition-association cases from runtime replay group jsonl."
    )
    ap.add_argument("--group-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--horizon", type=int, default=16)
    return ap.parse_args()


def _load_groups(path: Path) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            groups.append(json.loads(line))
    if not groups:
        raise ValueError(f"No groups found in {path}")
    return groups


def _classify_action(group: dict[str, Any]) -> str:
    det_gt_id = int(group.get("det_gt_id", -1))
    det_ignore = int(group.get("det_ignore", 0))
    if int(group.get("group_is_background", 0)) > 0 or det_gt_id <= 0 or det_ignore > 0:
        return "null"
    if int(group.get("group_is_recoverable", 0)) > 0 and int(group.get("rank_top1_correct", 0)) == 0:
        return "rerank"
    return "keep"


def _classify_continuity(action_target: str, next_gap: int, horizon: int) -> str:
    if action_target == "null":
        return "none"
    if next_gap < 0:
        return "terminal"
    if next_gap == 1:
        return "immediate"
    if 2 <= next_gap <= int(horizon):
        return "bridge"
    return "long_gap"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def main() -> int:
    args = parse_args()
    group_jsonl = Path(args.group_jsonl).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = _load_groups(group_jsonl)
    groups_by_seq_gt: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        det_gt_id = _safe_int(group.get("det_gt_id", -1), -1)
        if det_gt_id > 0 and _safe_int(group.get("det_ignore", 0), 0) == 0:
            groups_by_seq_gt[(str(group.get("seq", "")), det_gt_id)].append(group)

    future_index: dict[str, dict[str, int]] = {}
    for timeline in groups_by_seq_gt.values():
        timeline.sort(key=lambda g: (_safe_int(g.get("frame", 0)), _safe_int(g.get("det_index", 0))))
        for idx, group in enumerate(timeline):
            frame = _safe_int(group.get("frame", 0))
            next_gap = -1
            future_visible_count = 0
            future_ambiguous_count = 0
            future_recoverable_count = 0
            for nxt in timeline[idx + 1 :]:
                gap = _safe_int(nxt.get("frame", 0)) - frame
                if gap <= 0:
                    continue
                if next_gap < 0:
                    next_gap = gap
                if gap <= int(args.horizon):
                    future_visible_count += 1
                    future_ambiguous_count += _safe_int(nxt.get("group_is_ambiguous", 0))
                    future_recoverable_count += _safe_int(nxt.get("group_is_recoverable", 0))
                else:
                    break
            future_index[str(group.get("group_id", ""))] = {
                "next_same_gt_gap": int(next_gap),
                "future_visible_count": int(future_visible_count),
                "future_ambiguous_count": int(future_ambiguous_count),
                "future_recoverable_count": int(future_recoverable_count),
            }

    case_rows: list[dict[str, Any]] = []
    seq_counter: dict[str, Counter[str]] = defaultdict(Counter)
    summary_counter: Counter[str] = Counter()

    for group in groups:
        group_id = str(group.get("group_id", ""))
        seq = str(group.get("seq", ""))
        candidates = list(group.get("candidates", []))
        positive_rank = _safe_int(group.get("positive_rank", -1), -1)
        action_target = _classify_action(group)
        future_stats = future_index.get(
            group_id,
            {
                "next_same_gt_gap": -1,
                "future_visible_count": 0,
                "future_ambiguous_count": 0,
                "future_recoverable_count": 0,
            },
        )
        continuity_target = _classify_continuity(
            action_target=action_target,
            next_gap=int(future_stats["next_same_gt_gap"]),
            horizon=int(args.horizon),
        )

        valid_candidates = [cand for cand in candidates if _safe_int(cand.get("valid_train_row", 1), 1) > 0]
        if valid_candidates:
            track_gaps = [_safe_float(cand.get("track_gap", 0.0), 0.0) for cand in valid_candidates]
            hist_lens = [_safe_float(cand.get("track_hist_len", 0.0), 0.0) for cand in valid_candidates]
        else:
            track_gaps = [0.0]
            hist_lens = [0.0]

        row = {
            "group_id": group_id,
            "seq": seq,
            "frame": _safe_int(group.get("frame", 0)),
            "det_index": _safe_int(group.get("det_index", 0)),
            "det_gt_id": _safe_int(group.get("det_gt_id", -1), -1),
            "det_ignore": _safe_int(group.get("det_ignore", 0)),
            "group_size": _safe_int(group.get("group_size", len(candidates))),
            "candidate_count_total": _safe_int(group.get("candidate_count_total", len(candidates))),
            "group_has_positive": _safe_int(group.get("group_has_positive", 0)),
            "group_is_background": _safe_int(group.get("group_is_background", 0)),
            "group_is_ambiguous": _safe_int(group.get("group_is_ambiguous", 0)),
            "group_is_recoverable": _safe_int(group.get("group_is_recoverable", 0)),
            "positive_in_topk": _safe_int(group.get("positive_in_topk", 0)),
            "positive_rank": positive_rank,
            "rank_top1_correct": _safe_int(group.get("rank_top1_correct", 0)),
            "rank_margin": _safe_float(group.get("rank_margin", 0.0)),
            "rank_entropy": _safe_float(group.get("rank_entropy", 0.0)),
            "base_margin": _safe_float(group.get("base_margin", 0.0)),
            "refined_margin": _safe_float(group.get("refined_margin", 0.0)),
            "det_score": _safe_float(next(iter(candidates), {}).get("det_score", 0.0), 0.0),
            "track_gap_min": min(track_gaps),
            "track_gap_mean": sum(track_gaps) / float(max(len(track_gaps), 1)),
            "track_hist_len_mean": sum(hist_lens) / float(max(len(hist_lens), 1)),
            "future_visible_count": int(future_stats["future_visible_count"]),
            "future_ambiguous_count": int(future_stats["future_ambiguous_count"]),
            "future_recoverable_count": int(future_stats["future_recoverable_count"]),
            "next_same_gt_gap": int(future_stats["next_same_gt_gap"]),
            "action_target": action_target,
            "action_target_id": ACTION_TO_ID[action_target],
            "continuity_target": continuity_target,
            "continuity_bridge": int(continuity_target == "bridge"),
            "target_candidate_rank": positive_rank if action_target == "rerank" else -1,
            "hard_case": int(
                _safe_int(group.get("group_is_ambiguous", 0)) > 0
                or _safe_int(group.get("group_is_recoverable", 0)) > 0
                or continuity_target == "bridge"
            ),
        }
        case_rows.append(row)

        seq_counter[seq].update(
            {
                "groups": 1,
                "positive_groups": int(row["group_has_positive"] > 0),
                "background_groups": int(row["group_is_background"] > 0),
                "ambiguous_groups": int(row["group_is_ambiguous"] > 0),
                "recoverable_groups": int(row["group_is_recoverable"] > 0),
                "action_keep": int(action_target == "keep"),
                "action_rerank": int(action_target == "rerank"),
                "action_null": int(action_target == "null"),
                "continuity_bridge": int(continuity_target == "bridge"),
                "hard_case": int(row["hard_case"] > 0),
            }
        )
        summary_counter.update(
            {
                "groups": 1,
                "positive_groups": int(row["group_has_positive"] > 0),
                "background_groups": int(row["group_is_background"] > 0),
                "ambiguous_groups": int(row["group_is_ambiguous"] > 0),
                "recoverable_groups": int(row["group_is_recoverable"] > 0),
                "action_keep": int(action_target == "keep"),
                "action_rerank": int(action_target == "rerank"),
                "action_null": int(action_target == "null"),
                "continuity_bridge": int(continuity_target == "bridge"),
                "hard_case": int(row["hard_case"] > 0),
            }
        )

    cases_csv = out_dir / "competition_cases.csv"
    with cases_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)

    seq_summary_csv = out_dir / "sequence_summary.csv"
    seq_rows: list[dict[str, Any]] = []
    for seq, counter in sorted(seq_counter.items()):
        groups_n = max(int(counter["groups"]), 1)
        positive_n = max(int(counter["positive_groups"]), 1)
        seq_rows.append(
            {
                "seq": seq,
                "groups": int(counter["groups"]),
                "positive_groups": int(counter["positive_groups"]),
                "background_groups": int(counter["background_groups"]),
                "ambiguous_groups": int(counter["ambiguous_groups"]),
                "recoverable_groups": int(counter["recoverable_groups"]),
                "action_keep": int(counter["action_keep"]),
                "action_rerank": int(counter["action_rerank"]),
                "action_null": int(counter["action_null"]),
                "continuity_bridge": int(counter["continuity_bridge"]),
                "hard_case": int(counter["hard_case"]),
                "ambiguous_rate": float(counter["ambiguous_groups"]) / float(groups_n),
                "recoverable_rate_among_positive": float(counter["recoverable_groups"]) / float(positive_n),
                "rerank_rate_among_positive": float(counter["action_rerank"]) / float(positive_n),
                "bridge_rate_among_positive": float(counter["continuity_bridge"]) / float(positive_n),
            }
        )
    with seq_summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(seq_rows[0].keys()))
        writer.writeheader()
        writer.writerows(seq_rows)

    total_groups = max(int(summary_counter["groups"]), 1)
    positive_groups = max(int(summary_counter["positive_groups"]), 1)
    summary = {
        "group_jsonl": str(group_jsonl),
        "horizon": int(args.horizon),
        "groups": int(summary_counter["groups"]),
        "positive_groups": int(summary_counter["positive_groups"]),
        "background_groups": int(summary_counter["background_groups"]),
        "ambiguous_groups": int(summary_counter["ambiguous_groups"]),
        "recoverable_groups": int(summary_counter["recoverable_groups"]),
        "action_keep": int(summary_counter["action_keep"]),
        "action_rerank": int(summary_counter["action_rerank"]),
        "action_null": int(summary_counter["action_null"]),
        "continuity_bridge": int(summary_counter["continuity_bridge"]),
        "hard_case_groups": int(summary_counter["hard_case"]),
        "ambiguous_rate": float(summary_counter["ambiguous_groups"]) / float(total_groups),
        "recoverable_rate_among_positive": float(summary_counter["recoverable_groups"]) / float(positive_groups),
        "rerank_rate_among_positive": float(summary_counter["action_rerank"]) / float(positive_groups),
        "bridge_rate_among_positive": float(summary_counter["continuity_bridge"]) / float(positive_groups),
        "hard_case_rate": float(summary_counter["hard_case"]) / float(total_groups),
    }
    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
