from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from tracker import matching


@dataclass
class LocalGraphReassocConfig:
    enabled: bool = False
    dump_candidate_rows: bool = False
    allow_col_only_blocks: bool = True
    require_row_involved_strict_reclaim: bool = False
    protect_young_active_rows: bool = False
    top_k: int = 3
    max_rows: int = 4
    max_cols: int = 4
    row_margin: float = 0.03
    col_margin: float = 0.03
    min_reclaim_time_since_update: int = 1
    max_reclaim_time_since_update: int = 8
    min_reclaim_tracklet_len: int = 20
    recent_owner_max_time_since_update: int = 1
    recent_owner_max_tracklet_len: int = 8
    young_active_max_time_since_update: int = 1
    young_active_max_tracklet_len: int = 20
    young_active_min_reclaim_gap: int = 2
    young_active_max_cost_delta: float = -1.0
    protect_stale_lost_owner_rows: bool = False
    stale_lost_owner_min_time_since_update: int = 9
    stale_lost_owner_min_tracklet_len: int = 100
    stale_lost_owner_active_max_time_since_update: int = 1
    stale_lost_owner_min_introduced_edge_utility: float = 0.0
    min_box_iou: float = 0.6
    reclaim_bonus: float = 0.08
    recent_owner_penalty: float = 0.05
    iou_bonus: float = 0.04
    score_bonus: float = 0.02
    min_assignment_gain: float = 0.01
    max_cost_delta: float = 0.05
    row_involved_min_assignment_gain: float = 0.01
    col_only_min_assignment_gain: float = 0.01
    col_only_max_cost_delta: float = 0.05
    force_match_cost: float = 0.0
    require_same_match_count: bool = True
    candidate_rerank_top_k: int = 6
    learned_commit_rerank_candidates: bool = False
    learned_commit_scorer: Any = None
    learned_commit_replace_rules: bool = False
    learned_commit_gate_only: bool = False
    learned_commit_score_margin: float = 0.0
    learned_commit_safety_min_gain: float | None = None
    learned_commit_safety_max_cost_delta: float | None = None
    learned_commit_safety_require_reclaim_improve: bool = False
    learned_commit_safety_require_same_match_count: bool = False


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _sorted_local_pairs(pair_map: Dict[int, int]) -> List[List[int]]:
    return [[int(det_idx), int(track_idx)] for det_idx, track_idx in sorted(pair_map.items())]


class LocalGraphReassocRefiner:
    def __init__(self, config: LocalGraphReassocConfig) -> None:
        self.config = config
        self.stats: Dict[str, int | float | bool] = {
            "enabled": bool(config.enabled),
            "dump_candidate_rows": bool(config.dump_candidate_rows),
            "allow_col_only_blocks": bool(config.allow_col_only_blocks),
            "require_row_involved_strict_reclaim": bool(config.require_row_involved_strict_reclaim),
            "protect_young_active_rows": bool(config.protect_young_active_rows),
            "learned_commit_enabled": bool(config.learned_commit_scorer is not None),
            "learned_commit_replace_rules": bool(config.learned_commit_replace_rules),
            "learned_commit_gate_only": bool(config.learned_commit_gate_only),
            "learned_commit_score_margin": float(config.learned_commit_score_margin),
            "learned_commit_safety_min_gain": float(config.learned_commit_safety_min_gain)
            if config.learned_commit_safety_min_gain is not None
            else "",
            "learned_commit_safety_max_cost_delta": float(config.learned_commit_safety_max_cost_delta)
            if config.learned_commit_safety_max_cost_delta is not None
            else "",
            "learned_commit_safety_require_reclaim_improve": bool(config.learned_commit_safety_require_reclaim_improve),
            "learned_commit_safety_require_same_match_count": bool(config.learned_commit_safety_require_same_match_count),
            "learned_commit_decision_mode": str(getattr(config.learned_commit_scorer, "decision_mode", "") or ""),
            "learned_commit_threshold": float(getattr(config.learned_commit_scorer, "threshold", 0.0))
            if config.learned_commit_scorer is not None
            else "",
            "learned_commit_positive_threshold": float(getattr(config.learned_commit_scorer, "positive_threshold", 0.0))
            if config.learned_commit_scorer is not None
            else "",
            "learned_commit_neutral_threshold": float(getattr(config.learned_commit_scorer, "neutral_threshold", 0.0))
            if config.learned_commit_scorer is not None
            else "",
            "learned_commit_neutral_risk_weight": float(getattr(config.learned_commit_scorer, "neutral_risk_weight", 0.0))
            if config.learned_commit_scorer is not None
            else "",
            "top_k": int(config.top_k),
            "max_rows": int(config.max_rows),
            "max_cols": int(config.max_cols),
            "row_margin": float(config.row_margin),
            "col_margin": float(config.col_margin),
            "min_reclaim_time_since_update": int(config.min_reclaim_time_since_update),
            "max_reclaim_time_since_update": int(config.max_reclaim_time_since_update),
            "min_reclaim_tracklet_len": int(config.min_reclaim_tracklet_len),
            "recent_owner_max_time_since_update": int(config.recent_owner_max_time_since_update),
            "recent_owner_max_tracklet_len": int(config.recent_owner_max_tracklet_len),
            "young_active_max_time_since_update": int(config.young_active_max_time_since_update),
            "young_active_max_tracklet_len": int(config.young_active_max_tracklet_len),
            "young_active_min_reclaim_gap": int(config.young_active_min_reclaim_gap),
            "young_active_max_cost_delta": float(config.young_active_max_cost_delta),
            "protect_stale_lost_owner_rows": bool(config.protect_stale_lost_owner_rows),
            "stale_lost_owner_min_time_since_update": int(config.stale_lost_owner_min_time_since_update),
            "stale_lost_owner_min_tracklet_len": int(config.stale_lost_owner_min_tracklet_len),
            "stale_lost_owner_active_max_time_since_update": int(config.stale_lost_owner_active_max_time_since_update),
            "stale_lost_owner_min_introduced_edge_utility": float(config.stale_lost_owner_min_introduced_edge_utility),
            "min_box_iou": float(config.min_box_iou),
            "reclaim_bonus": float(config.reclaim_bonus),
            "recent_owner_penalty": float(config.recent_owner_penalty),
            "iou_bonus": float(config.iou_bonus),
            "score_bonus": float(config.score_bonus),
            "min_assignment_gain": float(config.min_assignment_gain),
            "max_cost_delta": float(config.max_cost_delta),
            "row_involved_min_assignment_gain": float(config.row_involved_min_assignment_gain),
            "col_only_min_assignment_gain": float(config.col_only_min_assignment_gain),
            "col_only_max_cost_delta": float(config.col_only_max_cost_delta),
            "force_match_cost": float(config.force_match_cost),
            "require_same_match_count": bool(config.require_same_match_count),
            "candidate_rerank_top_k": int(config.candidate_rerank_top_k),
            "learned_commit_rerank_candidates": bool(config.learned_commit_rerank_candidates),
            "frames": 0,
            "trigger_blocks": 0,
            "changed_blocks": 0,
            "trigger_rows": 0,
            "trigger_cols": 0,
            "ambiguous_rows": 0,
            "ambiguous_cols": 0,
            "enumerated_assignments": 0,
            "forced_matches": 0,
            "forced_rows": 0,
            "suppressed_rows": 0,
            "event_count": 0,
            "candidate_count": 0,
            "candidate_accepted_count": 0,
            "candidate_rejected_count": 0,
            "candidate_rerank_scored_count": 0,
            "candidate_rerank_selected_count": 0,
            "candidate_rerank_selected_rank_sum": 0,
            "skip_too_large": 0,
            "skip_no_reclaim_rows": 0,
            "skip_same_assignment": 0,
            "skip_pair_count_change": 0,
            "skip_low_gain": 0,
            "skip_high_cost_delta": 0,
            "skip_no_reclaim_improvement": 0,
            "skip_match_count_drop_bad_tradeoff": 0,
            "skip_col_only_gate": 0,
            "skip_col_only_block": 0,
            "skip_row_strict_reclaim": 0,
            "skip_young_active_protection": 0,
            "skip_young_active_high_cost": 0,
            "skip_stale_lost_owner_protection": 0,
            "skip_learned_commit_margin": 0,
            "skip_learned_commit_gate": 0,
            "skip_learned_commit_safety_min_gain": 0,
            "skip_learned_commit_safety_max_cost_delta": 0,
            "skip_learned_commit_safety_reclaim_improve": 0,
            "skip_learned_commit_safety_same_match_count": 0,
            "learned_commit_scored_candidates": 0,
            "learned_commit_margin_accept_count": 0,
            "learned_commit_margin_reject_count": 0,
            "learned_commit_gate_applied_count": 0,
            "learned_commit_error_count": 0,
            "candidate_rerank_scored_count": 0,
            "candidate_rerank_selected_count": 0,
            "candidate_rerank_selected_rank_sum": 0,
        }
        self.event_rows: List[Dict[str, object]] = []
        self.candidate_rows: List[Dict[str, object]] = []

    def is_active(self) -> bool:
        return bool(self.config.enabled)

    def _frame_debug(self) -> Dict[str, object]:
        return {
            "trigger_blocks": 0,
            "changed_blocks": 0,
            "trigger_rows": 0,
            "trigger_cols": 0,
            "ambiguous_rows": 0,
            "ambiguous_cols": 0,
            "enumerated_assignments": 0,
            "forced_matches": 0,
            "forced_rows": 0,
            "suppressed_rows": 0,
            "event_rows": [],
            "candidate_rows": [],
            "candidate_count": 0,
            "candidate_accepted_count": 0,
            "candidate_rejected_count": 0,
            "candidate_rerank_scored_count": 0,
            "candidate_rerank_selected_count": 0,
            "candidate_rerank_selected_rank_sum": 0,
            "skip_too_large": 0,
            "skip_no_reclaim_rows": 0,
            "skip_same_assignment": 0,
            "skip_pair_count_change": 0,
            "skip_low_gain": 0,
            "skip_high_cost_delta": 0,
            "skip_no_reclaim_improvement": 0,
            "skip_match_count_drop_bad_tradeoff": 0,
            "skip_col_only_gate": 0,
            "skip_col_only_block": 0,
            "skip_row_strict_reclaim": 0,
            "skip_young_active_protection": 0,
            "skip_young_active_high_cost": 0,
            "skip_stale_lost_owner_protection": 0,
            "skip_learned_commit_margin": 0,
            "skip_learned_commit_gate": 0,
            "skip_learned_commit_safety_min_gain": 0,
            "skip_learned_commit_safety_max_cost_delta": 0,
            "skip_learned_commit_safety_reclaim_improve": 0,
            "skip_learned_commit_safety_same_match_count": 0,
            "learned_commit_scored_candidates": 0,
            "learned_commit_margin_accept_count": 0,
            "learned_commit_margin_reject_count": 0,
            "learned_commit_gate_applied_count": 0,
            "learned_commit_error_count": 0,
        }

    def _track_gap(self, track: object, frame_id: int) -> int:
        last_frame = _safe_int(getattr(track, "frame_id", frame_id), frame_id)
        return max(0, int(frame_id) - int(last_frame))

    def _tracklet_len(self, track: object) -> int:
        return max(0, _safe_int(getattr(track, "tracklet_len", 0), 0))

    def _track_state_name(self, track: object) -> str:
        state_value = _safe_int(getattr(track, "state", -1), -1)
        return {
            0: "New",
            1: "Tracked",
            2: "Lost",
            3: "LongLost",
            4: "Removed",
        }.get(state_value, f"Unknown({state_value})")

    def _box_to_list(self, box: object) -> List[float]:
        if box is None:
            return []
        try:
            arr = np.asarray(box, dtype=np.float32).reshape(-1)
        except Exception:
            return []
        return [float(v) for v in arr.tolist()]

    def _row_topk_cols(self, dists: np.ndarray, match_thresh: float) -> List[List[int]]:
        top_cols: List[List[int]] = []
        for row_idx in range(dists.shape[0]):
            valid_cols = [
                int(col_idx)
                for col_idx in np.argsort(dists[row_idx])
                if np.isfinite(dists[row_idx, col_idx]) and float(dists[row_idx, col_idx]) <= float(match_thresh)
            ]
            top_cols.append(valid_cols[: max(1, int(self.config.top_k))])
        return top_cols

    def _col_topk_rows(self, dists: np.ndarray, match_thresh: float) -> List[List[int]]:
        top_rows: List[List[int]] = []
        for col_idx in range(dists.shape[1]):
            valid_rows = [
                int(row_idx)
                for row_idx in np.argsort(dists[:, col_idx])
                if np.isfinite(dists[row_idx, col_idx]) and float(dists[row_idx, col_idx]) <= float(match_thresh)
            ]
            top_rows.append(valid_rows[: max(1, int(self.config.top_k))])
        return top_rows

    def _ambiguous_rows(self, dists: np.ndarray, match_thresh: float) -> set[int]:
        rows: set[int] = set()
        for row_idx in range(dists.shape[0]):
            valid_costs = [
                float(dists[row_idx, col_idx])
                for col_idx in np.argsort(dists[row_idx])
                if np.isfinite(dists[row_idx, col_idx]) and float(dists[row_idx, col_idx]) <= float(match_thresh)
            ]
            if len(valid_costs) < 2:
                continue
            if float(valid_costs[1]) - float(valid_costs[0]) <= float(self.config.row_margin):
                rows.add(int(row_idx))
        return rows

    def _ambiguous_cols(self, dists: np.ndarray, match_thresh: float) -> set[int]:
        cols: set[int] = set()
        for col_idx in range(dists.shape[1]):
            valid_costs = [
                float(dists[row_idx, col_idx])
                for row_idx in np.argsort(dists[:, col_idx])
                if np.isfinite(dists[row_idx, col_idx]) and float(dists[row_idx, col_idx]) <= float(match_thresh)
            ]
            if len(valid_costs) < 2:
                continue
            if float(valid_costs[1]) - float(valid_costs[0]) <= float(self.config.col_margin):
                cols.add(int(col_idx))
        return cols

    def _build_trigger_components(self, dists: np.ndarray, match_thresh: float) -> List[Dict[str, object]]:
        row_top_cols = self._row_topk_cols(dists, match_thresh)
        col_top_rows = self._col_topk_rows(dists, match_thresh)
        ambiguous_rows = self._ambiguous_rows(dists, match_thresh)
        ambiguous_cols = self._ambiguous_cols(dists, match_thresh)

        row_adj: Dict[int, set[int]] = {idx: set(cols) for idx, cols in enumerate(row_top_cols)}
        col_adj: Dict[int, set[int]] = {idx: set(rows) for idx, rows in enumerate(col_top_rows)}

        visited_rows: set[int] = set()
        visited_cols: set[int] = set()
        components: List[Dict[str, object]] = []

        def bfs(start_rows: set[int], start_cols: set[int]) -> tuple[set[int], set[int]]:
            comp_rows = set(start_rows)
            comp_cols = set(start_cols)
            queue_rows = list(start_rows)
            queue_cols = list(start_cols)
            while queue_rows or queue_cols:
                while queue_rows:
                    row_idx = queue_rows.pop()
                    if row_idx in visited_rows:
                        continue
                    visited_rows.add(int(row_idx))
                    for col_idx in row_adj.get(int(row_idx), set()):
                        if col_idx not in comp_cols:
                            comp_cols.add(int(col_idx))
                            queue_cols.append(int(col_idx))
                while queue_cols:
                    col_idx = queue_cols.pop()
                    if col_idx in visited_cols:
                        continue
                    visited_cols.add(int(col_idx))
                    for row_idx in col_adj.get(int(col_idx), set()):
                        if row_idx not in comp_rows:
                            comp_rows.add(int(row_idx))
                            queue_rows.append(int(row_idx))
            return comp_rows, comp_cols

        for row_idx in sorted(ambiguous_rows):
            if row_idx in visited_rows:
                continue
            rows, cols = bfs({int(row_idx)}, set(row_adj.get(int(row_idx), set())))
            components.append(
                {
                    "rows": sorted(rows),
                    "cols": sorted(cols),
                    "ambiguous_rows": sorted(rows.intersection(ambiguous_rows)),
                    "ambiguous_cols": sorted(cols.intersection(ambiguous_cols)),
                }
            )
        for col_idx in sorted(ambiguous_cols):
            if col_idx in visited_cols:
                continue
            rows, cols = bfs(set(col_adj.get(int(col_idx), set())), {int(col_idx)})
            components.append(
                {
                    "rows": sorted(rows),
                    "cols": sorted(cols),
                    "ambiguous_rows": sorted(rows.intersection(ambiguous_rows)),
                    "ambiguous_cols": sorted(cols.intersection(ambiguous_cols)),
                }
            )
        deduped: List[Dict[str, object]] = []
        seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        for comp in components:
            key = (tuple(comp["rows"]), tuple(comp["cols"]))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(comp)
        return deduped

    def _row_reclaim_strength(self, track: object, frame_id: int) -> float:
        gap = self._track_gap(track, frame_id)
        length = self._tracklet_len(track)
        if gap < int(self.config.min_reclaim_time_since_update) or gap > int(self.config.max_reclaim_time_since_update):
            return 0.0
        if length < int(self.config.min_reclaim_tracklet_len):
            return 0.0
        gap_span = max(1, int(self.config.max_reclaim_time_since_update) - int(self.config.min_reclaim_time_since_update) + 1)
        gap_norm = float(gap - int(self.config.min_reclaim_time_since_update) + 1) / float(gap_span)
        len_norm = min(1.0, float(length - int(self.config.min_reclaim_tracklet_len) + 1) / float(max(1, int(self.config.min_reclaim_tracklet_len))))
        return max(0.0, 0.5 * gap_norm + 0.5 * len_norm)

    def _row_recent_penalty(self, track: object, frame_id: int) -> float:
        gap = self._track_gap(track, frame_id)
        length = self._tracklet_len(track)
        if gap > int(self.config.recent_owner_max_time_since_update):
            return 0.0
        if length > int(self.config.recent_owner_max_tracklet_len):
            return 0.0
        len_limit = max(1, int(self.config.recent_owner_max_tracklet_len))
        return 1.0 - min(1.0, float(length) / float(len_limit))

    def _edge_utility(
        self,
        *,
        row_idx: int,
        col_idx: int,
        track_pool: Sequence[object],
        detections: Sequence[object],
        dists: np.ndarray,
        raw_ious_dists: np.ndarray,
        frame_id: int,
    ) -> Dict[str, float]:
        track = track_pool[row_idx]
        det = detections[col_idx]
        cost = float(dists[row_idx, col_idx])
        box_iou = 1.0 - float(raw_ious_dists[row_idx, col_idx])
        det_score = _safe_float(getattr(det, "score", 0.0), 0.0)
        reclaim_strength = self._row_reclaim_strength(track, frame_id)
        recent_penalty = self._row_recent_penalty(track, frame_id)
        iou_lift = max(0.0, box_iou - float(self.config.min_box_iou))
        utility = -float(cost)
        utility += float(self.config.reclaim_bonus) * float(reclaim_strength)
        utility -= float(self.config.recent_owner_penalty) * float(recent_penalty)
        utility += float(self.config.iou_bonus) * float(iou_lift)
        utility += float(self.config.score_bonus) * float(det_score)
        return {
            "utility": float(utility),
            "cost": float(cost),
            "box_iou": float(box_iou),
            "det_score": float(det_score),
            "reclaim_strength": float(reclaim_strength),
            "recent_penalty": float(recent_penalty),
        }

    def _record_candidate_row(
        self,
        debug: Dict[str, object],
        candidate_row: Dict[str, object],
    ) -> None:
        debug["candidate_count"] = int(debug.get("candidate_count", 0)) + 1
        if bool(candidate_row.get("accepted", False)):
            debug["candidate_accepted_count"] = int(debug.get("candidate_accepted_count", 0)) + 1
        else:
            debug["candidate_rejected_count"] = int(debug.get("candidate_rejected_count", 0)) + 1
        if not bool(self.config.dump_candidate_rows):
            return
        debug_rows = debug.setdefault("candidate_rows", [])
        if isinstance(debug_rows, list):
            debug_rows.append(candidate_row)
        self.candidate_rows.append(candidate_row)

    def _track_debug_row(self, track: object, row_idx: int, frame_id: int) -> Dict[str, object]:
        return {
            "row_idx": int(row_idx),
            "track_id": _safe_int(getattr(track, "track_id", row_idx), row_idx),
            "gap": self._track_gap(track, frame_id),
            "tracklet_len": self._tracklet_len(track),
            "frame_id": _safe_int(getattr(track, "frame_id", frame_id), frame_id),
            "start_frame": _safe_int(getattr(track, "start_frame", 0), 0),
            "is_activated": bool(getattr(track, "is_activated", False)),
            "state": self._track_state_name(track),
            "score": _safe_float(getattr(track, "score", 0.0), 0.0),
            "tlwh": self._box_to_list(getattr(track, "tlwh", None)),
        }

    def _det_debug_row(self, det: object, col_idx: int) -> Dict[str, object]:
        return {
            "col_idx": int(col_idx),
            "score": _safe_float(getattr(det, "score", 0.0), 0.0),
            "tlwh": self._box_to_list(getattr(det, "tlwh", None)),
        }

    def _enumerate_best_assignment(
        self,
        *,
        rows: List[int],
        cols: List[int],
        candidate_cols: Dict[int, List[int]],
        track_pool: Sequence[object],
        detections: Sequence[object],
        dists: np.ndarray,
        raw_ious_dists: np.ndarray,
        frame_id: int,
        max_candidates: int | None = None,
    ) -> Dict[str, object]:
        candidates = self._enumerate_assignment_candidates(
            rows=rows,
            cols=cols,
            candidate_cols=candidate_cols,
            track_pool=track_pool,
            detections=detections,
            dists=dists,
            raw_ious_dists=raw_ious_dists,
            frame_id=frame_id,
            max_candidates=int(max_candidates) if max_candidates is not None else 1,
        )
        if not candidates:
            return {
                "pairs": [],
                "utility": float("-inf"),
                "raw_cost": float("inf"),
                "reclaim_matches": -1,
                "recent_matches": 10**9,
            }
        return dict(candidates[0])

    def _candidate_assignment_local_map(
        self,
        *,
        rows: List[int],
        cols: List[int],
        candidate_pairs: List[tuple[int, int]],
    ) -> Dict[int, int]:
        row_to_local = {int(row_idx): local_idx for local_idx, row_idx in enumerate(rows)}
        col_to_local = {int(col_idx): local_idx for local_idx, col_idx in enumerate(cols)}
        local_map: Dict[int, int] = {}
        for row_idx, col_idx in candidate_pairs:
            if row_idx not in row_to_local or col_idx not in col_to_local:
                continue
            local_map[int(col_to_local[int(col_idx)])] = int(row_to_local[int(row_idx)])
        return local_map

    def _enumerate_assignment_candidates(
        self,
        *,
        rows: List[int],
        cols: List[int],
        candidate_cols: Dict[int, List[int]],
        track_pool: Sequence[object],
        detections: Sequence[object],
        dists: np.ndarray,
        raw_ious_dists: np.ndarray,
        frame_id: int,
        max_candidates: int,
    ) -> List[Dict[str, object]]:
        edge_meta: Dict[tuple[int, int], Dict[str, float]] = {}
        for row_idx in rows:
            for col_idx in candidate_cols.get(int(row_idx), []):
                edge_meta[(int(row_idx), int(col_idx))] = self._edge_utility(
                    row_idx=int(row_idx),
                    col_idx=int(col_idx),
                    track_pool=track_pool,
                    detections=detections,
                    dists=dists,
                    raw_ious_dists=raw_ious_dists,
                    frame_id=frame_id,
                )

        best: Dict[str, object] = {
            "pairs": [],
            "utility": float("-inf"),
            "raw_cost": float("inf"),
            "reclaim_matches": -1,
            "recent_matches": 10**9,
        }
        candidates: List[Dict[str, object]] = []
        seen_pairs: set[tuple[tuple[int, int], ...]] = set()

        def dfs(row_pos: int, used_cols: set[int], pairs: List[tuple[int, int]], utility: float, raw_cost: float, reclaim_matches: int, recent_matches: int) -> None:
            self._enum_count += 1
            if row_pos >= len(rows):
                pair_key = tuple(sorted((int(row_idx), int(col_idx)) for row_idx, col_idx in pairs))
                if pair_key in seen_pairs:
                    return
                seen_pairs.add(pair_key)
                candidates.append(
                    {
                        "pairs": list(pairs),
                        "utility": float(utility),
                        "raw_cost": float(raw_cost),
                        "reclaim_matches": int(reclaim_matches),
                        "recent_matches": int(recent_matches),
                    }
                )
                return

            row_idx = int(rows[row_pos])
            dfs(row_pos + 1, used_cols, pairs, utility, raw_cost, reclaim_matches, recent_matches)
            for col_idx in candidate_cols.get(row_idx, []):
                col_idx = int(col_idx)
                if col_idx in used_cols:
                    continue
                meta = edge_meta[(row_idx, col_idx)]
                next_reclaim = reclaim_matches + int(meta["reclaim_strength"] > 0.0)
                next_recent = recent_matches + int(meta["recent_penalty"] > 0.0)
                pairs.append((row_idx, col_idx))
                used_cols.add(col_idx)
                dfs(
                    row_pos + 1,
                    used_cols,
                    pairs,
                    float(utility) + float(meta["utility"]),
                    float(raw_cost) + float(meta["cost"]),
                    next_reclaim,
                    next_recent,
                )
                used_cols.remove(col_idx)
                pairs.pop()

        dfs(0, set(), [], 0.0, 0.0, 0, 0)
        candidates.sort(
            key=lambda item: (
                int(len(item["pairs"])),
                int(item["reclaim_matches"]),
                -int(item["recent_matches"]),
                float(item["utility"]),
                -float(item["raw_cost"]),
            ),
            reverse=True,
        )
        if int(max_candidates) > 0:
            candidates = candidates[: int(max_candidates)]
        return candidates

    def refine_primary_cost(
        self,
        *,
        track_pool: Sequence[object],
        detections: Sequence[object],
        dists: np.ndarray,
        raw_ious_dists: np.ndarray,
        frame_id: int,
        match_thresh: float,
    ) -> tuple[np.ndarray, Dict[str, object]]:
        debug = self._frame_debug()
        if not self.is_active():
            return dists, debug

        self.stats["frames"] = int(self.stats["frames"]) + 1
        if dists.size == 0 or len(track_pool) == 0 or len(detections) == 0:
            return dists, debug

        modified = np.asarray(dists, dtype=np.float32).copy()
        components = self._build_trigger_components(modified, match_thresh)
        debug["trigger_blocks"] = int(len(components))

        for block_idx, comp in enumerate(components):
            rows = [int(v) for v in comp.get("rows", [])]
            cols = [int(v) for v in comp.get("cols", [])]
            ambiguous_rows = [int(v) for v in comp.get("ambiguous_rows", [])]
            ambiguous_cols = [int(v) for v in comp.get("ambiguous_cols", [])]
            debug["trigger_rows"] = int(debug["trigger_rows"]) + int(len(rows))
            debug["trigger_cols"] = int(debug["trigger_cols"]) + int(len(cols))
            debug["ambiguous_rows"] = int(debug["ambiguous_rows"]) + int(len(ambiguous_rows))
            debug["ambiguous_cols"] = int(debug["ambiguous_cols"]) + int(len(ambiguous_cols))

            if len(rows) < 2 or len(cols) < 2:
                debug["skip_too_large"] = int(debug["skip_too_large"]) + 1
                continue
            if len(rows) > int(self.config.max_rows) or len(cols) > int(self.config.max_cols):
                debug["skip_too_large"] = int(debug["skip_too_large"]) + 1
                continue
            if (not bool(self.config.allow_col_only_blocks)) and ambiguous_cols and (not ambiguous_rows):
                debug["skip_col_only_block"] = int(debug["skip_col_only_block"]) + 1
                continue

            candidate_cols: Dict[int, List[int]] = {}
            reclaimable_rows: set[int] = set()
            recent_owner_rows: set[int] = set()
            protected_young_active_rows: set[int] = set()
            for row_idx in rows:
                for col_idx in cols:
                    if not np.isfinite(modified[row_idx, col_idx]) or float(modified[row_idx, col_idx]) > float(match_thresh):
                        continue
                    box_iou = 1.0 - float(raw_ious_dists[row_idx, col_idx])
                    if box_iou < float(self.config.min_box_iou):
                        continue
                    candidate_cols.setdefault(int(row_idx), []).append(int(col_idx))
                candidate_cols[int(row_idx)] = sorted(candidate_cols.get(int(row_idx), []), key=lambda col_idx: float(modified[row_idx, col_idx]))
                track = track_pool[row_idx]
                if self._row_reclaim_strength(track, frame_id) > 0.0:
                    reclaimable_rows.add(int(row_idx))
                if self._row_recent_penalty(track, frame_id) > 0.0:
                    recent_owner_rows.add(int(row_idx))
                if bool(self.config.protect_young_active_rows):
                    gap = self._track_gap(track, frame_id)
                    length = self._tracklet_len(track)
                    if (
                        gap <= int(self.config.young_active_max_time_since_update)
                        and length <= int(self.config.young_active_max_tracklet_len)
                    ):
                        protected_young_active_rows.add(int(row_idx))

            if not reclaimable_rows:
                debug["skip_no_reclaim_rows"] = int(debug["skip_no_reclaim_rows"]) + 1
                continue
            protected_young_active_rows.difference_update(reclaimable_rows)

            submatrix = modified[np.ix_(rows, cols)]
            baseline_pairs_local, _, _ = matching.linear_assignment(submatrix, thresh=float(match_thresh))
            baseline_pairs = [(int(rows[r]), int(cols[c])) for r, c in baseline_pairs_local.tolist()] if len(baseline_pairs_local) > 0 else []
            baseline_rows = {int(row_idx) for row_idx, _ in baseline_pairs}
            baseline_reclaim = sum(1 for row_idx, _ in baseline_pairs if row_idx in reclaimable_rows)
            baseline_recent = sum(1 for row_idx, _ in baseline_pairs if row_idx in recent_owner_rows)
            baseline_cost = float(sum(float(modified[row_idx, col_idx]) for row_idx, col_idx in baseline_pairs))
            baseline_utility = 0.0
            for row_idx, col_idx in baseline_pairs:
                baseline_utility += float(
                    self._edge_utility(
                        row_idx=row_idx,
                        col_idx=col_idx,
                        track_pool=track_pool,
                        detections=detections,
                        dists=modified,
                        raw_ious_dists=raw_ious_dists,
                        frame_id=frame_id,
                    )["utility"]
                )

            candidate_limit = max(1, int(self.config.candidate_rerank_top_k))
            self._enum_count = 0
            candidate_assignments = self._enumerate_assignment_candidates(
                rows=rows,
                cols=cols,
                candidate_cols=candidate_cols,
                track_pool=track_pool,
                detections=detections,
                dists=modified,
                raw_ious_dists=raw_ious_dists,
                frame_id=frame_id,
                max_candidates=candidate_limit,
            )
            debug["enumerated_assignments"] = int(debug["enumerated_assignments"]) + int(self._enum_count)
            if not candidate_assignments:
                candidate_assignments = [
                    {
                        "pairs": [],
                        "utility": float("-inf"),
                        "raw_cost": float("inf"),
                        "reclaim_matches": -1,
                        "recent_matches": 10**9,
                    }
                ]

            scorer = self.config.learned_commit_scorer
            scored_candidates: List[Dict[str, object]] = []
            for rank, candidate in enumerate(candidate_assignments, 1):
                candidate_pairs = [(int(row_idx), int(col_idx)) for row_idx, col_idx in candidate["pairs"]]
                candidate_rows = {int(row_idx) for row_idx, _ in candidate_pairs}
                candidate_reclaim = int(candidate["reclaim_matches"])
                candidate_recent = int(candidate["recent_matches"])
                candidate_utility = float(candidate["utility"])
                candidate_cost = float(candidate["raw_cost"])
                candidate_match_count_delta = int(len(candidate_pairs)) - int(len(baseline_pairs))
                candidate_drop_rows = sorted(int(row_idx) for row_idx in baseline_rows.difference(candidate_rows))
                candidate_drop_recent_owner = any(int(row_idx) in recent_owner_rows for row_idx in candidate_drop_rows)
                candidate_drop_reclaimable = any(int(row_idx) in reclaimable_rows for row_idx in candidate_drop_rows)
                candidate_drop_safe = bool(candidate_match_count_delta >= 0) or (
                    not bool(candidate_drop_recent_owner) and bool(candidate_drop_reclaimable)
                )
                variant_result = None
                if scorer is not None and bool(self.config.learned_commit_rerank_candidates):
                    candidate_local = self._candidate_assignment_local_map(
                        rows=rows,
                        cols=cols,
                        candidate_pairs=candidate_pairs,
                    )
                    try:
                        variant_result = scorer.score_candidate_row_variant(
                            {
                                "frame_id": int(frame_id),
                                "block_id": int(block_idx),
                                "rows": list(rows),
                                "cols": list(cols),
                                "ambiguous_rows": list(ambiguous_rows),
                                "ambiguous_cols": list(ambiguous_cols),
                                "baseline_pairs": [list(v) for v in baseline_pairs],
                                "chosen_pairs": [list(v) for v in candidate_pairs],
                                "baseline_reclaim_matches": int(baseline_reclaim),
                                "chosen_reclaim_matches": int(candidate_reclaim),
                                "baseline_recent_matches": int(baseline_recent),
                                "chosen_recent_matches": int(candidate_recent),
                                "baseline_cost": float(baseline_cost),
                                "chosen_cost": float(candidate_cost),
                                "cost_delta": float(candidate_cost - baseline_cost),
                                "baseline_utility": float(baseline_utility),
                                "chosen_utility": float(candidate_utility),
                                "utility_gain": float(candidate_utility - baseline_utility),
                                "row_states": row_states,
                                "col_states": col_states,
                                "edge_meta": edge_meta_rows,
                                "baseline_pair_meta": baseline_pair_meta,
                                "chosen_pair_meta": _pair_debug_rows(candidate_pairs),
                                "introduced_rows": list(int(v) for v in introduced_rows),
                                "suppressed_rows": list(int(v) for v in suppressed_rows),
                                "recent_owner_rows": list(int(v) for v in recent_owner_rows),
                                "protected_young_active_rows": list(int(v) for v in protected_young_active_rows),
                                "candidate_local_map": candidate_local,
                            },
                            candidate_local,
                            assignment_key="candidate_local_map",
                        )
                    except Exception as exc:
                        variant_result = {"error": str(exc)}
                variant_summary = {
                    "baseline_score": _safe_float(variant_result.get("baseline_score", 0.0), 0.0)
                    if isinstance(variant_result, dict)
                    else 0.0,
                    "variant_score": _safe_float(variant_result.get("variant_score", 0.0), 0.0)
                    if isinstance(variant_result, dict)
                    else 0.0,
                    "assignment_delta": _safe_float(variant_result.get("assignment_delta", 0.0), 0.0)
                    if isinstance(variant_result, dict)
                    else 0.0,
                    "decision_score": _safe_float(variant_result.get("decision_score", 0.0), 0.0)
                    if isinstance(variant_result, dict)
                    else 0.0,
                    "decision_margin": _safe_float(variant_result.get("decision_margin", 0.0), 0.0)
                    if isinstance(variant_result, dict)
                    else 0.0,
                    "accept_threshold": _safe_float(variant_result.get("accept_threshold", 0.0), 0.0)
                    if isinstance(variant_result, dict)
                    else 0.0,
                    "score_delta": _safe_float(variant_result.get("score_delta", 0.0), 0.0)
                    if isinstance(variant_result, dict)
                    else 0.0,
                    "chosen_better": bool(variant_result["chosen_better"]) if isinstance(variant_result, dict) and "chosen_better" in variant_result else False,
                    "pred_score": _safe_float(variant_result.get("pred_score", 0.0), 0.0)
                    if isinstance(variant_result, dict)
                    else 0.0,
                }
                scored_candidates.append(
                    {
                        "rank": int(rank),
                        "pairs": list(candidate_pairs),
                        "utility": float(candidate_utility),
                        "raw_cost": float(candidate_cost),
                        "reclaim_matches": int(candidate_reclaim),
                        "recent_matches": int(candidate_recent),
                        "match_count_delta": int(candidate_match_count_delta),
                        "drop_safe": bool(candidate_drop_safe),
                        "drop_recent_owner": bool(candidate_drop_recent_owner),
                        "drop_reclaimable": bool(candidate_drop_reclaimable),
                        "dropped_rows": list(candidate_drop_rows),
                        "variant_result": variant_summary,
                    }
                )

            if scorer is not None and bool(self.config.learned_commit_rerank_candidates):
                def _candidate_sort_key(item: Dict[str, object]) -> tuple[float, float, float, float, float, int, int]:
                    match_count_delta = int(item.get("match_count_delta", 0))
                    drop_safe = bool(item.get("drop_safe", False))
                    if match_count_delta >= 0:
                        drop_priority = 2.0
                    elif drop_safe:
                        drop_priority = 1.0
                    else:
                        drop_priority = 0.0
                    variant_result = item.get("variant_result", None)
                    if isinstance(variant_result, dict):
                        decision_score = _safe_float(variant_result.get("decision_score", float("-inf")), float("-inf"))
                        decision_delta = _safe_float(variant_result.get("assignment_delta", float("-inf")), float("-inf"))
                        decision_margin = _safe_float(variant_result.get("decision_margin", float("-inf")), float("-inf"))
                    else:
                        decision_score = float("-inf")
                        decision_delta = float("-inf")
                        decision_margin = float("-inf")
                    return (
                        float(drop_priority),
                        float(decision_score),
                        float(decision_margin),
                        float(decision_delta),
                        float(item.get("utility", float("-inf"))),
                        int(item.get("reclaim_matches", -1)),
                        int(item.get("recent_matches", 10**9)),
                    )

                scored_candidates.sort(key=_candidate_sort_key, reverse=True)
                if scored_candidates:
                    debug["candidate_rerank_scored_count"] = int(debug.get("candidate_rerank_scored_count", 0)) + int(len(scored_candidates))
                    debug["candidate_rerank_selected_count"] = int(debug.get("candidate_rerank_selected_count", 0)) + 1
                    debug["candidate_rerank_selected_rank_sum"] = int(debug.get("candidate_rerank_selected_rank_sum", 0)) + int(scored_candidates[0]["rank"])

            best = scored_candidates[0]
            chosen_pairs = [(int(row_idx), int(col_idx)) for row_idx, col_idx in best["pairs"]]
            chosen_reclaim = int(best["reclaim_matches"])
            chosen_recent = int(best["recent_matches"])
            chosen_utility = float(best["utility"])
            chosen_cost = float(best["raw_cost"])
            gain = float(chosen_utility) - float(baseline_utility)
            cost_delta = float(chosen_cost) - float(baseline_cost)
            match_count_delta = int(len(chosen_pairs)) - int(len(baseline_pairs))
            row_involved_block = bool(ambiguous_rows)
            col_only_block = bool(ambiguous_cols) and not bool(ambiguous_rows)
            min_gain = float(self.config.min_assignment_gain)
            max_cost_delta = float(self.config.max_cost_delta)
            if row_involved_block:
                min_gain = max(min_gain, float(self.config.row_involved_min_assignment_gain))
            if col_only_block:
                min_gain = max(min_gain, float(self.config.col_only_min_assignment_gain))
                max_cost_delta = min(max_cost_delta, float(self.config.col_only_max_cost_delta))

            if sorted(chosen_pairs) == sorted(baseline_pairs):
                debug["skip_same_assignment"] = int(debug["skip_same_assignment"]) + 1
                continue

            baseline_rows = {int(row_idx) for row_idx, _ in baseline_pairs}
            chosen_rows = {int(row_idx) for row_idx, _ in chosen_pairs}
            chosen_cols = {int(col_idx) for _, col_idx in chosen_pairs}
            introduced_rows = [int(row_idx) for row_idx in chosen_rows if row_idx not in baseline_rows]
            suppressed_rows = [int(row_idx) for row_idx in rows if row_idx not in chosen_rows]
            suppressed_protected_rows = [int(row_idx) for row_idx in suppressed_rows if row_idx in protected_young_active_rows]
            has_gap_advantaged_reclaim = any(
                self._track_gap(track_pool[row_idx], frame_id) >= int(self.config.young_active_min_reclaim_gap)
                for row_idx in introduced_rows
            )
            suppressed_stale_lost_rows: List[int] = []
            active_introduced_rows: List[int] = []
            introduced_utilities: List[float] = []
            if bool(self.config.protect_stale_lost_owner_rows) and suppressed_rows and introduced_rows:
                suppressed_stale_lost_rows = [
                    int(row_idx)
                    for row_idx in suppressed_rows
                    if row_idx not in reclaimable_rows
                    and self._track_gap(track_pool[row_idx], frame_id) >= int(self.config.stale_lost_owner_min_time_since_update)
                    and self._tracklet_len(track_pool[row_idx]) >= int(self.config.stale_lost_owner_min_tracklet_len)
                ]
                active_introduced_rows = [
                    int(row_idx)
                    for row_idx in introduced_rows
                    if self._track_gap(track_pool[row_idx], frame_id) <= int(self.config.stale_lost_owner_active_max_time_since_update)
                ]
                if suppressed_stale_lost_rows and active_introduced_rows:
                    for row_idx, col_idx in chosen_pairs:
                        if int(row_idx) not in active_introduced_rows:
                            continue
                        introduced_utilities.append(
                            float(
                                self._edge_utility(
                                    row_idx=int(row_idx),
                                    col_idx=int(col_idx),
                                    track_pool=track_pool,
                                    detections=detections,
                                    dists=modified,
                                    raw_ious_dists=raw_ious_dists,
                                    frame_id=frame_id,
                                )["utility"]
                            )
                        )
            row_states: List[Dict[str, object]] = []
            for row_idx in rows:
                track = track_pool[row_idx]
                row_state = self._track_debug_row(track, row_idx, frame_id)
                row_state["reclaim_strength"] = float(self._row_reclaim_strength(track, frame_id))
                row_state["recent_penalty"] = float(self._row_recent_penalty(track, frame_id))
                row_state["is_reclaimable"] = bool(row_idx in reclaimable_rows)
                row_state["is_recent_owner"] = bool(row_idx in recent_owner_rows)
                row_state["is_protected_young_active"] = bool(row_idx in protected_young_active_rows)
                row_state["is_baseline_matched"] = bool(any(pair_row == row_idx for pair_row, _ in baseline_pairs))
                row_state["is_chosen_matched"] = bool(any(pair_row == row_idx for pair_row, _ in chosen_pairs))
                row_state["is_suppressed"] = bool(row_idx in suppressed_rows)
                row_states.append(row_state)
            col_states = [self._det_debug_row(detections[col_idx], col_idx) for col_idx in cols]
            edge_meta_rows: List[Dict[str, object]] = []
            for row_idx in rows:
                track_id = _safe_int(getattr(track_pool[row_idx], "track_id", row_idx), row_idx)
                for col_idx in candidate_cols.get(int(row_idx), []):
                    meta = self._edge_utility(
                        row_idx=int(row_idx),
                        col_idx=int(col_idx),
                        track_pool=track_pool,
                        detections=detections,
                        dists=modified,
                        raw_ious_dists=raw_ious_dists,
                        frame_id=frame_id,
                    )
                    edge_meta_rows.append(
                        {
                            "row_idx": int(row_idx),
                            "track_id": int(track_id),
                            "col_idx": int(col_idx),
                            **{key: float(value) for key, value in meta.items()},
                        }
                    )

            def _pair_debug_rows(pairs: List[tuple[int, int]]) -> List[Dict[str, object]]:
                pair_rows: List[Dict[str, object]] = []
                for row_idx, col_idx in pairs:
                    meta = self._edge_utility(
                        row_idx=int(row_idx),
                        col_idx=int(col_idx),
                        track_pool=track_pool,
                        detections=detections,
                        dists=modified,
                        raw_ious_dists=raw_ious_dists,
                        frame_id=frame_id,
                    )
                    pair_rows.append(
                        {
                            "row_idx": int(row_idx),
                            "track_id": _safe_int(getattr(track_pool[row_idx], "track_id", row_idx), row_idx),
                            "col_idx": int(col_idx),
                            **{key: float(value) for key, value in meta.items()},
                        }
                    )
                return pair_rows

            baseline_pair_meta = _pair_debug_rows(baseline_pairs)
            chosen_pair_meta = _pair_debug_rows(chosen_pairs)
            candidate_row_base = {
                "frame_id": int(frame_id),
                "block_id": int(block_idx),
                "rows": list(rows),
                "cols": list(cols),
                "ambiguous_rows": list(ambiguous_rows),
                "ambiguous_cols": list(ambiguous_cols),
                "row_involved_block": bool(row_involved_block),
                "col_only_block": bool(col_only_block),
                "baseline_pairs": [list(v) for v in baseline_pairs],
                "chosen_pairs": [list(v) for v in chosen_pairs],
                "baseline_reclaim_matches": int(baseline_reclaim),
                "chosen_reclaim_matches": int(chosen_reclaim),
                "baseline_recent_matches": int(baseline_recent),
                "chosen_recent_matches": int(chosen_recent),
                "baseline_cost": float(baseline_cost),
                "chosen_cost": float(chosen_cost),
                "cost_delta": float(cost_delta),
                "baseline_utility": float(baseline_utility),
                "chosen_utility": float(chosen_utility),
                "utility_gain": float(gain),
                "required_min_gain": float(min_gain),
                "required_max_cost_delta": float(max_cost_delta),
                "enumerated_assignments": int(self._enum_count),
                "candidate_rerank_top_k": int(self.config.candidate_rerank_top_k),
                "candidate_rerank_enabled": bool(self.config.learned_commit_rerank_candidates),
                "candidate_rerank_candidates": [
                    {
                        "rank": int(item["rank"]),
                        "utility": float(item["utility"]),
                        "raw_cost": float(item["raw_cost"]),
                        "reclaim_matches": int(item["reclaim_matches"]),
                        "recent_matches": int(item["recent_matches"]),
                        "match_count_delta": int(item.get("match_count_delta", 0)),
                        "drop_safe": bool(item.get("drop_safe", False)),
                        "drop_recent_owner": bool(item.get("drop_recent_owner", False)),
                        "drop_reclaimable": bool(item.get("drop_reclaimable", False)),
                        "dropped_rows": list(item.get("dropped_rows", [])),
                        "variant_assignment_delta": _safe_float(
                            dict(item.get("variant_result", {})).get("assignment_delta", 0.0),
                            0.0,
                        )
                        if isinstance(item.get("variant_result", None), dict)
                        else 0.0,
                        "variant_decision_score": _safe_float(
                            dict(item.get("variant_result", {})).get("decision_score", 0.0),
                            0.0,
                        )
                        if isinstance(item.get("variant_result", None), dict)
                        else 0.0,
                        "variant_decision_margin": _safe_float(
                            dict(item.get("variant_result", {})).get("decision_margin", 0.0),
                            0.0,
                        )
                        if isinstance(item.get("variant_result", None), dict)
                        else 0.0,
                        "variant_chosen_better": bool(
                            dict(item.get("variant_result", {})).get("chosen_better", False)
                        )
                        if isinstance(item.get("variant_result", None), dict)
                        else False,
                    }
                    for item in scored_candidates
                ]
                if bool(self.config.learned_commit_rerank_candidates)
                else [],
                "reclaimable_rows": sorted(reclaimable_rows),
                "recent_owner_rows": sorted(recent_owner_rows),
                "protected_young_active_rows": sorted(protected_young_active_rows),
                "introduced_rows": list(int(v) for v in introduced_rows),
                "suppressed_rows": list(suppressed_rows),
                "suppressed_protected_rows": list(int(v) for v in suppressed_protected_rows),
                "suppressed_stale_lost_rows": list(int(v) for v in suppressed_stale_lost_rows),
                "active_introduced_rows": list(int(v) for v in active_introduced_rows),
                "has_gap_advantaged_reclaim": bool(has_gap_advantaged_reclaim),
                "max_introduced_utility": float(max(introduced_utilities)) if introduced_utilities else "",
                "row_states": row_states,
                "col_states": col_states,
                "edge_meta": edge_meta_rows,
                "baseline_pair_meta": baseline_pair_meta,
                "chosen_pair_meta": chosen_pair_meta,
            }

            scorer = self.config.learned_commit_scorer
            learned_margin = float(self.config.learned_commit_score_margin)
            candidate_row_base.update(
                {
                    "decision_source": "rules",
                    "learned_commit_enabled": bool(scorer is not None),
                    "learned_commit_replace_rules": bool(self.config.learned_commit_replace_rules),
                    "learned_commit_gate_only": bool(self.config.learned_commit_gate_only),
                    "learned_commit_score_margin": float(learned_margin),
                    "learned_commit_applied_to_decision": False,
                    "learned_commit_available": False,
                    "learned_commit_baseline_score": "",
                    "learned_commit_chosen_score": "",
                    "learned_commit_score_delta": "",
                    "learned_commit_pred_score": "",
                    "learned_commit_decision_score": "",
                    "learned_commit_gain_pred": "",
                    "learned_commit_policy_score": "",
                    "learned_commit_action_margin": "",
                    "learned_commit_positive_score": "",
                    "learned_commit_neutral_score": "",
                    "learned_commit_model_type": "",
                    "learned_commit_decision_mode": "",
                    "learned_commit_threshold": "",
                    "learned_commit_positive_threshold": "",
                    "learned_commit_neutral_threshold": "",
                    "learned_commit_neutral_risk_weight": "",
                    "learned_commit_chosen_better": "",
                    "learned_commit_score_margin_pass": "",
                    "learned_commit_pred_pairs_local": [],
                    "learned_commit_baseline_pairs_local": [],
                    "learned_commit_chosen_pairs_local": [],
                    "learned_commit_error": "",
                    "rules_passed_before_learned_gate": False,
                }
            )
            learned_margin_accept = False
            learned_score_available = False
            if scorer is not None and not bool(self.config.learned_commit_rerank_candidates):
                try:
                    learned = scorer.score_candidate_row(candidate_row_base)
                    learned_score_available = True
                    learned_margin_accept = bool(learned.score_delta >= learned_margin)
                    debug["learned_commit_scored_candidates"] = int(debug["learned_commit_scored_candidates"]) + 1
                    if learned_margin_accept:
                        debug["learned_commit_margin_accept_count"] = int(debug["learned_commit_margin_accept_count"]) + 1
                    else:
                        debug["learned_commit_margin_reject_count"] = int(debug["learned_commit_margin_reject_count"]) + 1
                    candidate_row_base.update(
                        {
                            "learned_commit_available": True,
                            "learned_commit_baseline_score": _safe_float(learned.baseline_score, 0.0),
                            "learned_commit_chosen_score": _safe_float(learned.chosen_score, 0.0),
                            "learned_commit_score_delta": _safe_float(learned.score_delta, 0.0),
                            "learned_commit_pred_score": _safe_float(learned.pred_score, 0.0),
                            "learned_commit_decision_score": _safe_float(getattr(learned, "decision_score", learned.pred_score), 0.0),
                            "learned_commit_gain_pred": _safe_float(getattr(learned, "gain_pred", 0.0), 0.0),
                            "learned_commit_policy_score": _safe_float(getattr(learned, "policy_score", 0.0), 0.0),
                            "learned_commit_action_margin": _safe_float(getattr(learned, "action_margin", 0.0), 0.0),
                            "learned_commit_positive_score": _safe_float(
                                getattr(learned, "positive_probability", getattr(learned, "probability", 0.0)),
                                0.0,
                            ),
                            "learned_commit_neutral_score": _safe_float(getattr(learned, "neutral_probability", 0.0), 0.0),
                            "learned_commit_model_type": str(getattr(learned, "model_type", "single_head")),
                            "learned_commit_decision_mode": str(getattr(learned, "decision_mode", "")),
                            "learned_commit_threshold": _safe_float(getattr(learned, "threshold", 0.0), 0.0),
                            "learned_commit_positive_threshold": _safe_float(getattr(learned, "positive_threshold", 0.0), 0.0),
                            "learned_commit_neutral_threshold": _safe_float(getattr(learned, "neutral_threshold", 0.0), 0.0),
                            "learned_commit_neutral_risk_weight": _safe_float(getattr(learned, "neutral_risk_weight", 0.0), 0.0),
                            "learned_commit_chosen_better": bool(learned.chosen_better),
                            "learned_commit_score_margin_pass": bool(learned_margin_accept),
                            "learned_commit_pred_pairs_local": _sorted_local_pairs(learned.pred_pairs_local),
                            "learned_commit_baseline_pairs_local": _sorted_local_pairs(learned.baseline_pairs_local),
                            "learned_commit_chosen_pairs_local": _sorted_local_pairs(learned.chosen_pairs_local),
                        }
                    )
                except Exception as exc:
                    debug["learned_commit_error_count"] = int(debug["learned_commit_error_count"]) + 1
                    candidate_row_base["learned_commit_error"] = str(exc)

            def _record_rejected_candidate(reason: str, *, decision_source: str = "rules") -> None:
                self._record_candidate_row(
                    debug,
                    {
                        **candidate_row_base,
                        "accepted": False,
                        "decision": "rejected",
                        "decision_source": str(decision_source),
                        "skip_reason": str(reason),
                    },
                )

            def _learned_commit_safety_reject_reason() -> str | None:
                if bool(self.config.learned_commit_safety_require_same_match_count) and len(chosen_pairs) != len(baseline_pairs):
                    return "learned_commit_safety_same_match_count"
                if (
                    self.config.learned_commit_safety_min_gain is not None
                    and gain < float(self.config.learned_commit_safety_min_gain)
                ):
                    return "learned_commit_safety_min_gain"
                if (
                    self.config.learned_commit_safety_max_cost_delta is not None
                    and cost_delta > float(self.config.learned_commit_safety_max_cost_delta)
                ):
                    return "learned_commit_safety_max_cost_delta"
                if bool(self.config.learned_commit_safety_require_reclaim_improve):
                    if chosen_reclaim < baseline_reclaim or (chosen_reclaim == baseline_reclaim and chosen_recent >= baseline_recent):
                        return "learned_commit_safety_reclaim_improve"
                return None

            learned_replace_active = bool(self.config.learned_commit_replace_rules) and learned_score_available
            learned_gate_active = (
                bool(self.config.learned_commit_gate_only)
                and not bool(self.config.learned_commit_replace_rules)
                and learned_score_available
            )
            if learned_replace_active:
                candidate_row_base["learned_commit_applied_to_decision"] = True
                if not learned_margin_accept:
                    debug["skip_learned_commit_margin"] = int(debug["skip_learned_commit_margin"]) + 1
                    _record_rejected_candidate("learned_commit_margin", decision_source="learned_commit")
                    continue
                safety_reason = _learned_commit_safety_reject_reason()
                if safety_reason is not None:
                    debug[f"skip_{safety_reason}"] = int(debug[f"skip_{safety_reason}"]) + 1
                    _record_rejected_candidate(safety_reason, decision_source="learned_commit")
                    continue

            if not learned_replace_active:
                if bool(self.config.require_same_match_count) and len(chosen_pairs) != len(baseline_pairs):
                    debug["skip_pair_count_change"] = int(debug["skip_pair_count_change"]) + 1
                    if match_count_delta < 0:
                        dropped_rows = sorted(int(row_idx) for row_idx in baseline_rows.difference(chosen_rows))
                        dropped_recent_owner = any(int(row_idx) in recent_owner_rows for row_idx in dropped_rows)
                        dropped_reclaimable = any(int(row_idx) in reclaimable_rows for row_idx in dropped_rows)
                        if dropped_recent_owner or not dropped_reclaimable:
                            debug["skip_match_count_drop_bad_tradeoff"] = int(debug["skip_match_count_drop_bad_tradeoff"]) + 1
                            _record_rejected_candidate("match_count_drop_bad_tradeoff")
                            continue
                if gain < float(min_gain):
                    debug["skip_low_gain"] = int(debug["skip_low_gain"]) + 1
                    if col_only_block and float(min_gain) > float(self.config.min_assignment_gain):
                        debug["skip_col_only_gate"] = int(debug["skip_col_only_gate"]) + 1
                    _record_rejected_candidate("low_gain")
                    continue
                if cost_delta > float(max_cost_delta):
                    debug["skip_high_cost_delta"] = int(debug["skip_high_cost_delta"]) + 1
                    if col_only_block and float(max_cost_delta) < float(self.config.max_cost_delta):
                        debug["skip_col_only_gate"] = int(debug["skip_col_only_gate"]) + 1
                    _record_rejected_candidate("high_cost_delta")
                    continue
                if row_involved_block and bool(self.config.require_row_involved_strict_reclaim) and chosen_reclaim <= baseline_reclaim:
                    debug["skip_row_strict_reclaim"] = int(debug["skip_row_strict_reclaim"]) + 1
                    _record_rejected_candidate("row_strict_reclaim")
                    continue
                if chosen_reclaim < baseline_reclaim or (chosen_reclaim == baseline_reclaim and chosen_recent >= baseline_recent):
                    debug["skip_no_reclaim_improvement"] = int(debug["skip_no_reclaim_improvement"]) + 1
                    _record_rejected_candidate("no_reclaim_improvement")
                    continue
                if bool(self.config.protect_young_active_rows) and suppressed_protected_rows:
                    if not has_gap_advantaged_reclaim:
                        debug["skip_young_active_protection"] = int(debug["skip_young_active_protection"]) + 1
                        _record_rejected_candidate("young_active_protection")
                        continue
                    if (
                        float(self.config.young_active_max_cost_delta) >= 0.0
                        and float(cost_delta) > float(self.config.young_active_max_cost_delta)
                    ):
                        debug["skip_young_active_high_cost"] = int(debug["skip_young_active_high_cost"]) + 1
                        _record_rejected_candidate("young_active_high_cost")
                        continue
                if suppressed_stale_lost_rows and active_introduced_rows:
                    if introduced_utilities and max(introduced_utilities) < float(self.config.stale_lost_owner_min_introduced_edge_utility):
                        debug["skip_stale_lost_owner_protection"] = int(debug["skip_stale_lost_owner_protection"]) + 1
                        _record_rejected_candidate("stale_lost_owner_protection")
                        continue

            candidate_row_base["rules_passed_before_learned_gate"] = True
            if learned_gate_active:
                candidate_row_base["learned_commit_applied_to_decision"] = True
                debug["learned_commit_gate_applied_count"] = int(debug["learned_commit_gate_applied_count"]) + 1
                if not learned_margin_accept:
                    debug["skip_learned_commit_gate"] = int(debug["skip_learned_commit_gate"]) + 1
                    _record_rejected_candidate("learned_commit_gate", decision_source="learned_commit_gate")
                    continue
                safety_reason = _learned_commit_safety_reject_reason()
                if safety_reason is not None:
                    debug[f"skip_{safety_reason}"] = int(debug[f"skip_{safety_reason}"]) + 1
                    _record_rejected_candidate(safety_reason, decision_source="learned_commit_gate")
                    continue

            for row_idx in chosen_rows:
                modified[row_idx, :] = 1.0
            for col_idx in chosen_cols:
                modified[:, col_idx] = 1.0
            for row_idx in suppressed_rows:
                modified[row_idx, cols] = 1.0
            for row_idx, col_idx in chosen_pairs:
                modified[row_idx, col_idx] = min(float(self.config.force_match_cost), float(dists[row_idx, col_idx]))

            debug["changed_blocks"] = int(debug["changed_blocks"]) + 1
            debug["forced_matches"] = int(debug["forced_matches"]) + int(len(chosen_pairs))
            debug["forced_rows"] = int(debug["forced_rows"]) + int(len(chosen_rows))
            debug["suppressed_rows"] = int(debug["suppressed_rows"]) + int(len(suppressed_rows))
            event_row = {
                **candidate_row_base,
                "accepted": True,
                "decision": "accepted",
                "decision_source": "learned_commit" if learned_replace_active else ("rules+learned_gate" if learned_gate_active else "rules"),
                "learned_commit_applied_to_decision": bool(learned_replace_active or learned_gate_active),
                "skip_reason": "",
            }
            self._record_candidate_row(debug, event_row)
            debug["event_rows"].append(event_row)
            self.event_rows.append(event_row)

        self.stats["trigger_blocks"] = int(self.stats["trigger_blocks"]) + int(debug["trigger_blocks"])
        self.stats["changed_blocks"] = int(self.stats["changed_blocks"]) + int(debug["changed_blocks"])
        self.stats["trigger_rows"] = int(self.stats["trigger_rows"]) + int(debug["trigger_rows"])
        self.stats["trigger_cols"] = int(self.stats["trigger_cols"]) + int(debug["trigger_cols"])
        self.stats["ambiguous_rows"] = int(self.stats["ambiguous_rows"]) + int(debug["ambiguous_rows"])
        self.stats["ambiguous_cols"] = int(self.stats["ambiguous_cols"]) + int(debug["ambiguous_cols"])
        self.stats["enumerated_assignments"] = int(self.stats["enumerated_assignments"]) + int(debug["enumerated_assignments"])
        self.stats["forced_matches"] = int(self.stats["forced_matches"]) + int(debug["forced_matches"])
        self.stats["forced_rows"] = int(self.stats["forced_rows"]) + int(debug["forced_rows"])
        self.stats["suppressed_rows"] = int(self.stats["suppressed_rows"]) + int(debug["suppressed_rows"])
        self.stats["event_count"] = int(self.stats["event_count"]) + int(len(debug["event_rows"]))
        self.stats["candidate_count"] = int(self.stats["candidate_count"]) + int(debug.get("candidate_count", 0))
        self.stats["candidate_accepted_count"] = int(self.stats["candidate_accepted_count"]) + int(debug.get("candidate_accepted_count", 0))
        self.stats["candidate_rejected_count"] = int(self.stats["candidate_rejected_count"]) + int(debug.get("candidate_rejected_count", 0))
        self.stats["candidate_rerank_scored_count"] = int(self.stats["candidate_rerank_scored_count"]) + int(debug.get("candidate_rerank_scored_count", 0))
        self.stats["candidate_rerank_selected_count"] = int(self.stats["candidate_rerank_selected_count"]) + int(debug.get("candidate_rerank_selected_count", 0))
        self.stats["candidate_rerank_selected_rank_sum"] = int(self.stats["candidate_rerank_selected_rank_sum"]) + int(debug.get("candidate_rerank_selected_rank_sum", 0))
        for key in [
            "skip_too_large",
            "skip_no_reclaim_rows",
            "skip_same_assignment",
            "skip_pair_count_change",
            "skip_low_gain",
            "skip_high_cost_delta",
            "skip_no_reclaim_improvement",
            "skip_match_count_drop_bad_tradeoff",
            "skip_col_only_gate",
            "skip_col_only_block",
            "skip_row_strict_reclaim",
            "skip_young_active_protection",
            "skip_young_active_high_cost",
            "skip_stale_lost_owner_protection",
            "skip_learned_commit_margin",
            "skip_learned_commit_gate",
            "skip_learned_commit_safety_min_gain",
            "skip_learned_commit_safety_max_cost_delta",
            "skip_learned_commit_safety_reclaim_improve",
            "skip_learned_commit_safety_same_match_count",
            "learned_commit_scored_candidates",
            "learned_commit_margin_accept_count",
            "learned_commit_margin_reject_count",
            "learned_commit_gate_applied_count",
            "learned_commit_error_count",
        ]:
            self.stats[key] = int(self.stats[key]) + int(debug[key])
        return modified, debug

    def get_summary(self) -> Dict[str, int | float | bool]:
        summary = dict(self.stats)
        trigger_blocks = int(summary.get("trigger_blocks", 0))
        changed_blocks = int(summary.get("changed_blocks", 0))
        forced_matches = int(summary.get("forced_matches", 0))
        candidate_count = int(summary.get("candidate_count", 0))
        candidate_accepted_count = int(summary.get("candidate_accepted_count", 0))
        summary["changed_block_rate"] = float(changed_blocks) / float(trigger_blocks) if trigger_blocks > 0 else 0.0
        summary["candidate_accept_rate"] = (
            float(candidate_accepted_count) / float(candidate_count)
            if candidate_count > 0
            else 0.0
        )
        learned_commit_scored_candidates = int(summary.get("learned_commit_scored_candidates", 0))
        learned_commit_margin_accept_count = int(summary.get("learned_commit_margin_accept_count", 0))
        summary["learned_commit_margin_accept_rate"] = (
            float(learned_commit_margin_accept_count) / float(learned_commit_scored_candidates)
            if learned_commit_scored_candidates > 0
            else 0.0
        )
        summary["avg_forced_matches_per_changed_block"] = (
            float(forced_matches) / float(changed_blocks)
            if changed_blocks > 0
            else 0.0
        )
        summary["avg_assignments_per_block"] = (
            float(summary.get("enumerated_assignments", 0)) / float(trigger_blocks)
            if trigger_blocks > 0
            else 0.0
        )
        return summary

    def get_event_rows(self) -> List[Dict[str, object]]:
        return list(self.event_rows)

    def drain_event_rows(self) -> List[Dict[str, object]]:
        rows = list(self.event_rows)
        self.event_rows.clear()
        return rows

    def get_candidate_rows(self) -> List[Dict[str, object]]:
        return list(self.candidate_rows)

    def drain_candidate_rows(self) -> List[Dict[str, object]]:
        rows = list(self.candidate_rows)
        self.candidate_rows.clear()
        return rows
