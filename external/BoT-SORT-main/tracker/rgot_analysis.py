from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class RgotAnalysisConfig:
    enabled: bool = False
    analysis_only: bool = False
    top_k: int = 3
    row_margin: float = 0.03
    col_margin: float = 0.03
    max_rows: int = 4
    max_cols: int = 4
    owner_recent_max_time_since_update: int = 1
    owner_recent_max_tracklet_len: int = 8
    reclaim_min_time_since_update: int = 1
    reclaim_max_time_since_update: int = 8
    reclaim_min_tracklet_len: int = 20
    owneralt_min_box_iou: float = 0.75
    owneralt_max_owner_edge_deficit: float = 0.10


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


class RgotAnalysisCollector:
    def __init__(self, config: RgotAnalysisConfig) -> None:
        self.config = config
        self.stats: Dict[str, int | float | bool] = {
            "enabled": bool(config.enabled),
            "analysis_only": bool(config.analysis_only),
            "top_k": int(config.top_k),
            "row_margin": float(config.row_margin),
            "col_margin": float(config.col_margin),
            "max_rows": int(config.max_rows),
            "max_cols": int(config.max_cols),
            "frames": 0,
            "candidate_blocks": 0,
            "skipped_too_large_blocks": 0,
            "skip_no_focus_column": 0,
            "trigger_blocks": 0,
            "row_involved_blocks": 0,
            "col_only_blocks": 0,
            "ambiguous_rows": 0,
            "ambiguous_cols": 0,
            "owner_weak_cases": 0,
            "challenger_reclaim_cases": 0,
            "alt_available_cases": 0,
            "buffer_like_cases": 0,
            "reentry_like_cases": 0,
            "owneralt_overlap_events": 0,
            "graph_assoc_overlap_events": 0,
            "owneralt_like_proxy_cases": 0,
            "joint_cost_viable_cases": 0,
            "not_explained_by_buffer_or_reentry_cases": 0,
            "missing_appearance_cases": 0,
            "event_count": 0,
        }
        self.event_rows: List[Dict[str, object]] = []
        self._owner_edge_deficits: List[float] = []
        self._joint_cost_deltas: List[float] = []

    def is_active(self) -> bool:
        return bool(self.config.enabled)

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

    def _safe_cosine_similarity(self, vec_a: object, vec_b: object) -> float:
        if vec_a is None or vec_b is None:
            return 0.0
        try:
            a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
            b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
        except Exception:
            return 0.0
        if a.size == 0 or b.size == 0:
            return 0.0
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-8:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _row_topk_cols(self, dists: np.ndarray, match_thresh: float) -> List[List[int]]:
        top_cols: List[List[int]] = []
        limit = max(1, int(self.config.top_k))
        for row_idx in range(dists.shape[0]):
            valid_cols = [
                int(col_idx)
                for col_idx in np.argsort(dists[row_idx])
                if np.isfinite(dists[row_idx, col_idx]) and float(dists[row_idx, col_idx]) <= float(match_thresh)
            ]
            top_cols.append(valid_cols[:limit])
        return top_cols

    def _col_topk_rows(self, dists: np.ndarray, match_thresh: float) -> List[List[int]]:
        top_rows: List[List[int]] = []
        limit = max(1, int(self.config.top_k))
        for col_idx in range(dists.shape[1]):
            valid_rows = [
                int(row_idx)
                for row_idx in np.argsort(dists[:, col_idx])
                if np.isfinite(dists[row_idx, col_idx]) and float(dists[row_idx, col_idx]) <= float(match_thresh)
            ]
            top_rows.append(valid_rows[:limit])
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

    def _choose_focus_column(
        self,
        *,
        rows: Sequence[int],
        cols: Sequence[int],
        dists: np.ndarray,
        match_thresh: float,
    ) -> tuple[int | None, List[int], float]:
        best_col: int | None = None
        best_rows: List[int] = []
        best_gap = float("inf")
        for col_idx in cols:
            valid_rows = [
                int(row_idx)
                for row_idx in sorted(rows, key=lambda row_id: float(dists[int(row_id), int(col_idx)]))
                if np.isfinite(dists[int(row_idx), int(col_idx)]) and float(dists[int(row_idx), int(col_idx)]) <= float(match_thresh)
            ]
            if len(valid_rows) < 2:
                continue
            gap = float(dists[valid_rows[1], int(col_idx)]) - float(dists[valid_rows[0], int(col_idx)])
            if gap < best_gap:
                best_gap = gap
                best_col = int(col_idx)
                best_rows = valid_rows
        return best_col, best_rows, best_gap

    def _best_owner_alt(
        self,
        *,
        owner_row: int,
        focus_col: int,
        dists: np.ndarray,
        raw_ious_dists: np.ndarray,
        detections: Sequence[object],
        match_thresh: float,
    ) -> Dict[str, object]:
        best_alt: Dict[str, object] = {
            "exists": False,
            "alt_col": -1,
            "alt_cost": float("inf"),
            "alt_box_iou": 0.0,
            "alt_score": 0.0,
        }
        for alt_col in np.argsort(dists[int(owner_row)]):
            alt_col = int(alt_col)
            if alt_col == int(focus_col):
                continue
            alt_cost = float(dists[int(owner_row), alt_col])
            if not np.isfinite(alt_cost) or alt_cost > float(match_thresh):
                continue
            best_alt = {
                "exists": True,
                "alt_col": int(alt_col),
                "alt_cost": float(alt_cost),
                "alt_box_iou": float(1.0 - float(raw_ious_dists[int(owner_row), alt_col])),
                "alt_score": _safe_float(getattr(detections[int(alt_col)], "score", 0.0), 0.0),
            }
            break
        return best_alt

    def _matrix_value(self, debug: Dict[str, object] | None, key: str, row_idx: int, col_idx: int, default: float = 0.0) -> float:
        if debug is None:
            return float(default)
        values = debug.get(key)
        if values is None:
            return float(default)
        try:
            arr = np.asarray(values, dtype=np.float32)
            if arr.ndim != 2:
                return float(default)
            return float(arr[int(row_idx), int(col_idx)])
        except Exception:
            return float(default)

    def _vector_value(self, debug: Dict[str, object] | None, key: str, col_idx: int, default: float = 0.0) -> float:
        if debug is None:
            return float(default)
        values = debug.get(key)
        if values is None:
            return float(default)
        try:
            arr = np.asarray(values, dtype=np.float32).reshape(-1)
            return float(arr[int(col_idx)])
        except Exception:
            return float(default)

    def inspect_primary_cost(
        self,
        *,
        track_pool: Sequence[object],
        detections: Sequence[object],
        baseline_dists: np.ndarray,
        raw_ious_dists: np.ndarray,
        frame_id: int,
        match_thresh: float,
        max_time_lost: int,
        laplace_debug: Dict[str, object] | None = None,
        owneralt_event_rows: Sequence[Dict[str, object]] | None = None,
        graph_assoc_event_rows: Sequence[Dict[str, object]] | None = None,
    ) -> Dict[str, int | float]:
        debug = {
            "candidate_blocks": 0,
            "skipped_too_large_blocks": 0,
            "skip_no_focus_column": 0,
            "trigger_blocks": 0,
            "row_involved_blocks": 0,
            "col_only_blocks": 0,
            "ambiguous_rows": 0,
            "ambiguous_cols": 0,
            "event_rows": [],
        }
        if not self.is_active():
            return debug
        if baseline_dists.size == 0 or len(track_pool) == 0 or len(detections) == 0:
            return debug

        self.stats["frames"] = int(self.stats["frames"]) + 1
        components = self._build_trigger_components(baseline_dists, match_thresh)
        debug["candidate_blocks"] = int(len(components))
        self.stats["candidate_blocks"] = int(self.stats["candidate_blocks"]) + int(len(components))
        owneralt_event_rows = owneralt_event_rows or []
        graph_assoc_event_rows = graph_assoc_event_rows or []

        for block_idx, comp in enumerate(components):
            rows = [int(v) for v in comp.get("rows", [])]
            cols = [int(v) for v in comp.get("cols", [])]
            ambiguous_rows = [int(v) for v in comp.get("ambiguous_rows", [])]
            ambiguous_cols = [int(v) for v in comp.get("ambiguous_cols", [])]
            if not rows or not cols:
                continue
            if len(rows) > int(self.config.max_rows) or len(cols) > int(self.config.max_cols):
                debug["skipped_too_large_blocks"] = int(debug["skipped_too_large_blocks"]) + 1
                self.stats["skipped_too_large_blocks"] = int(self.stats["skipped_too_large_blocks"]) + 1
                continue

            focus_col, focus_rows, focus_gap = self._choose_focus_column(
                rows=rows,
                cols=cols,
                dists=baseline_dists,
                match_thresh=match_thresh,
            )
            if focus_col is None or len(focus_rows) < 2:
                debug["skip_no_focus_column"] = int(debug["skip_no_focus_column"]) + 1
                self.stats["skip_no_focus_column"] = int(self.stats["skip_no_focus_column"]) + 1
                continue

            owner_row = int(focus_rows[0])
            challenger_row = int(focus_rows[1])
            owner_track = track_pool[owner_row]
            challenger_track = track_pool[challenger_row]
            det = detections[int(focus_col)]

            owner_gap = self._track_gap(owner_track, frame_id)
            challenger_gap = self._track_gap(challenger_track, frame_id)
            owner_len = self._tracklet_len(owner_track)
            challenger_len = self._tracklet_len(challenger_track)
            owner_state = self._track_state_name(owner_track)
            challenger_state = self._track_state_name(challenger_track)
            owner_cost = float(baseline_dists[owner_row, int(focus_col)])
            challenger_cost = float(baseline_dists[challenger_row, int(focus_col)])
            owner_box_iou = float(1.0 - float(raw_ious_dists[owner_row, int(focus_col)]))
            challenger_box_iou = float(1.0 - float(raw_ious_dists[challenger_row, int(focus_col)]))
            owner_edge_deficit = float(challenger_cost - owner_cost)

            best_alt = self._best_owner_alt(
                owner_row=owner_row,
                focus_col=int(focus_col),
                dists=baseline_dists,
                raw_ious_dists=raw_ious_dists,
                detections=detections,
                match_thresh=match_thresh,
            )
            alt_exists = bool(best_alt["exists"])
            joint_cost_delta = float(best_alt["alt_cost"] + challenger_cost - owner_cost) if alt_exists else float("inf")

            owner_recent_weak = bool(
                owner_gap <= int(self.config.owner_recent_max_time_since_update)
                and owner_len <= int(self.config.owner_recent_max_tracklet_len)
            )
            challenger_reclaimable = bool(
                challenger_gap >= int(self.config.reclaim_min_time_since_update)
                and challenger_gap <= int(self.config.reclaim_max_time_since_update)
                and challenger_len >= int(self.config.reclaim_min_tracklet_len)
            )
            buffer_like_proxy = bool(challenger_gap > 0 and challenger_gap <= int(max_time_lost))
            reentry_like_proxy = bool(challenger_state in {"Lost", "LongLost"} and challenger_gap > 0)
            owneralt_like_proxy = bool(
                challenger_reclaimable
                and owner_recent_weak
                and alt_exists
                and challenger_box_iou >= float(self.config.owneralt_min_box_iou)
                and owner_edge_deficit <= float(self.config.owneralt_max_owner_edge_deficit)
            )

            owneralt_overlap_event = any(
                int(row.get("frame_id", -1)) == int(frame_id)
                and int(row.get("det_col", -1)) == int(focus_col)
                and int(row.get("owner_row", -2)) == int(owner_row)
                and int(row.get("challenger_row", -3)) == int(challenger_row)
                for row in owneralt_event_rows
            )
            graph_assoc_overlap_event = any(
                int(row.get("frame_id", -1)) == int(frame_id)
                and int(focus_col) in [int(v) for v in row.get("cols", [])]
                and int(owner_row) in [int(v) for v in row.get("rows", [])]
                and int(challenger_row) in [int(v) for v in row.get("rows", [])]
                for row in graph_assoc_event_rows
            )

            app_available = bool(
                getattr(owner_track, "smooth_feat", None) is not None
                and getattr(det, "curr_feat", None) is not None
            )
            appearance_cos = self._safe_cosine_similarity(getattr(owner_track, "smooth_feat", None), getattr(det, "curr_feat", None))

            row_involved_block = bool(ambiguous_rows)
            col_only_block = bool(ambiguous_cols) and not row_involved_block
            det_score = _safe_float(getattr(det, "score", 0.0), 0.0)
            owner_final_sim = self._matrix_value(laplace_debug, "final_sim", owner_row, int(focus_col), default=0.0)
            challenger_final_sim = self._matrix_value(laplace_debug, "final_sim", challenger_row, int(focus_col), default=0.0)
            if owner_final_sim == 0.0 and laplace_debug is not None and "final_sim" not in laplace_debug:
                owner_final_sim = self._matrix_value(laplace_debug, "anchor_sim", owner_row, int(focus_col), default=0.0)
                challenger_final_sim = self._matrix_value(laplace_debug, "anchor_sim", challenger_row, int(focus_col), default=0.0)
            event_row = {
                "frame_id": int(frame_id),
                "block_id": int(block_idx),
                "rows": list(rows),
                "cols": list(cols),
                "num_rows": int(len(rows)),
                "num_cols": int(len(cols)),
                "ambiguous_rows": list(ambiguous_rows),
                "ambiguous_cols": list(ambiguous_cols),
                "row_involved_block": bool(row_involved_block),
                "col_only_block": bool(col_only_block),
                "focus_det_col": int(focus_col),
                "focus_det_score": float(det_score),
                "focus_margin": float(focus_gap),
                "owner_row": int(owner_row),
                "owner_track_id": _safe_int(getattr(owner_track, "track_id", owner_row), owner_row),
                "owner_state": str(owner_state),
                "owner_gap": int(owner_gap),
                "owner_tracklet_len": int(owner_len),
                "owner_cost": float(owner_cost),
                "owner_box_iou": float(owner_box_iou),
                "owner_final_sim": float(owner_final_sim),
                "owner_recent_weak": bool(owner_recent_weak),
                "challenger_row": int(challenger_row),
                "challenger_track_id": _safe_int(getattr(challenger_track, "track_id", challenger_row), challenger_row),
                "challenger_state": str(challenger_state),
                "challenger_gap": int(challenger_gap),
                "challenger_tracklet_len": int(challenger_len),
                "challenger_cost": float(challenger_cost),
                "challenger_box_iou": float(challenger_box_iou),
                "challenger_final_sim": float(challenger_final_sim),
                "challenger_reclaimable": bool(challenger_reclaimable),
                "owner_edge_deficit": float(owner_edge_deficit),
                "alt_exists": bool(alt_exists),
                "alt_col": int(best_alt["alt_col"]),
                "alt_cost": float(best_alt["alt_cost"]) if alt_exists else "",
                "alt_box_iou": float(best_alt["alt_box_iou"]) if alt_exists else "",
                "alt_score": float(best_alt["alt_score"]) if alt_exists else "",
                "joint_cost_delta_proxy": float(joint_cost_delta) if alt_exists else "",
                "joint_cost_viable_proxy": bool(alt_exists and joint_cost_delta <= float(self.config.owneralt_max_owner_edge_deficit)),
                "buffer_like_proxy": bool(buffer_like_proxy),
                "reentry_like_proxy": bool(reentry_like_proxy),
                "not_explained_by_buffer_or_reentry": bool(not buffer_like_proxy and not reentry_like_proxy),
                "owneralt_like_proxy": bool(owneralt_like_proxy),
                "owneralt_overlap_event": bool(owneralt_overlap_event),
                "graph_assoc_overlap_event": bool(graph_assoc_overlap_event),
                "appearance_available": bool(app_available),
                "appearance_cosine": float(appearance_cos),
                "haca_margin": float(self._vector_value(laplace_debug, "haca_comp_margin", int(focus_col), default=0.0)),
                "haca_bg_prob": float(self._vector_value(laplace_debug, "haca_background", int(focus_col), default=0.0)),
                "haca_owner_active": float(self._matrix_value(laplace_debug, "haca_comp_active", owner_row, int(focus_col), default=0.0)),
                "haca_challenger_active": float(self._matrix_value(laplace_debug, "haca_comp_active", challenger_row, int(focus_col), default=0.0)),
            }

            debug["trigger_blocks"] = int(debug["trigger_blocks"]) + 1
            debug["row_involved_blocks"] = int(debug["row_involved_blocks"]) + int(row_involved_block)
            debug["col_only_blocks"] = int(debug["col_only_blocks"]) + int(col_only_block)
            debug["ambiguous_rows"] = int(debug["ambiguous_rows"]) + int(len(ambiguous_rows))
            debug["ambiguous_cols"] = int(debug["ambiguous_cols"]) + int(len(ambiguous_cols))
            debug["event_rows"].append(event_row)
            self.event_rows.append(event_row)
            self._owner_edge_deficits.append(float(owner_edge_deficit))
            if alt_exists:
                self._joint_cost_deltas.append(float(joint_cost_delta))

            self.stats["trigger_blocks"] = int(self.stats["trigger_blocks"]) + 1
            self.stats["row_involved_blocks"] = int(self.stats["row_involved_blocks"]) + int(row_involved_block)
            self.stats["col_only_blocks"] = int(self.stats["col_only_blocks"]) + int(col_only_block)
            self.stats["ambiguous_rows"] = int(self.stats["ambiguous_rows"]) + int(len(ambiguous_rows))
            self.stats["ambiguous_cols"] = int(self.stats["ambiguous_cols"]) + int(len(ambiguous_cols))
            self.stats["owner_weak_cases"] = int(self.stats["owner_weak_cases"]) + int(owner_recent_weak)
            self.stats["challenger_reclaim_cases"] = int(self.stats["challenger_reclaim_cases"]) + int(challenger_reclaimable)
            self.stats["alt_available_cases"] = int(self.stats["alt_available_cases"]) + int(alt_exists)
            self.stats["buffer_like_cases"] = int(self.stats["buffer_like_cases"]) + int(buffer_like_proxy)
            self.stats["reentry_like_cases"] = int(self.stats["reentry_like_cases"]) + int(reentry_like_proxy)
            self.stats["owneralt_overlap_events"] = int(self.stats["owneralt_overlap_events"]) + int(owneralt_overlap_event)
            self.stats["graph_assoc_overlap_events"] = int(self.stats["graph_assoc_overlap_events"]) + int(graph_assoc_overlap_event)
            self.stats["owneralt_like_proxy_cases"] = int(self.stats["owneralt_like_proxy_cases"]) + int(owneralt_like_proxy)
            self.stats["joint_cost_viable_cases"] = int(self.stats["joint_cost_viable_cases"]) + int(
                alt_exists and joint_cost_delta <= float(self.config.owneralt_max_owner_edge_deficit)
            )
            self.stats["not_explained_by_buffer_or_reentry_cases"] = int(self.stats["not_explained_by_buffer_or_reentry_cases"]) + int(
                not buffer_like_proxy and not reentry_like_proxy
            )
            self.stats["missing_appearance_cases"] = int(self.stats["missing_appearance_cases"]) + int(not app_available)
            self.stats["event_count"] = int(self.stats["event_count"]) + 1

        return debug

    def get_summary(self) -> Dict[str, int | float | bool]:
        summary = dict(self.stats)
        trigger_blocks = int(summary.get("trigger_blocks", 0))
        candidate_blocks = int(summary.get("candidate_blocks", 0))
        event_count = int(summary.get("event_count", 0))
        summary["trigger_block_rate"] = (
            float(trigger_blocks) / float(candidate_blocks) if candidate_blocks > 0 else 0.0
        )
        summary["owner_weak_rate"] = (
            float(summary.get("owner_weak_cases", 0)) / float(event_count) if event_count > 0 else 0.0
        )
        summary["challenger_reclaim_rate"] = (
            float(summary.get("challenger_reclaim_cases", 0)) / float(event_count) if event_count > 0 else 0.0
        )
        summary["alt_available_rate"] = (
            float(summary.get("alt_available_cases", 0)) / float(event_count) if event_count > 0 else 0.0
        )
        summary["buffer_like_rate"] = (
            float(summary.get("buffer_like_cases", 0)) / float(event_count) if event_count > 0 else 0.0
        )
        summary["reentry_like_rate"] = (
            float(summary.get("reentry_like_cases", 0)) / float(event_count) if event_count > 0 else 0.0
        )
        summary["owneralt_overlap_rate"] = (
            float(summary.get("owneralt_overlap_events", 0)) / float(event_count) if event_count > 0 else 0.0
        )
        summary["graph_assoc_overlap_rate"] = (
            float(summary.get("graph_assoc_overlap_events", 0)) / float(event_count) if event_count > 0 else 0.0
        )
        summary["not_explained_by_buffer_or_reentry_rate"] = (
            float(summary.get("not_explained_by_buffer_or_reentry_cases", 0)) / float(event_count) if event_count > 0 else 0.0
        )
        summary["mean_owner_edge_deficit"] = (
            float(sum(self._owner_edge_deficits) / len(self._owner_edge_deficits)) if self._owner_edge_deficits else 0.0
        )
        summary["median_owner_edge_deficit"] = float(median(self._owner_edge_deficits)) if self._owner_edge_deficits else 0.0
        finite_joint = [value for value in self._joint_cost_deltas if np.isfinite(value)]
        summary["mean_joint_cost_delta_proxy"] = (
            float(sum(finite_joint) / len(finite_joint)) if finite_joint else 0.0
        )
        summary["median_joint_cost_delta_proxy"] = float(median(finite_joint)) if finite_joint else 0.0
        return summary

    def get_event_rows(self) -> List[Dict[str, object]]:
        return list(self.event_rows)

    def drain_event_rows(self) -> List[Dict[str, object]]:
        rows = list(self.event_rows)
        self.event_rows.clear()
        return rows
