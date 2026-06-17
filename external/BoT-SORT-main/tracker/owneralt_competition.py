from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class OwnerAltCompetitionConfig:
    enabled: bool = False
    min_time_since_update: int = 2
    max_time_since_update: int = 8
    min_tracklet_len: int = 20
    min_box_iou: float = 0.75
    gap1_min_box_iou: float = -1.0
    owner_max_tracklet_len: int = 8
    owner_alt_det_min_score: float = 0.0
    owner_alt_det_min_box_iou: float = 0.0
    gap1_owner_alt_det_min_box_iou: float = -1.0
    max_owner_edge_deficit: float = 0.10
    gap1_max_owner_edge_deficit: float = -1.0
    evidence_mode: str = "legacy"
    max_joint_penalty: float = -1.0
    gap1_max_joint_penalty: float = -1.0
    owner_alt_bonus: float = 0.10
    block_owner_on_reclaim: bool = True


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


class OwnerAltCompetitionRefiner:
    def __init__(self, config: OwnerAltCompetitionConfig) -> None:
        mode = str(config.evidence_mode).strip().lower()
        if mode not in {"legacy", "joint"}:
            raise ValueError(f"Unsupported owneralt evidence_mode: {config.evidence_mode}")
        config.evidence_mode = mode
        self.config = config
        self.stats: Dict[str, int | float | bool] = {
            "enabled": bool(config.enabled),
            "min_time_since_update": int(config.min_time_since_update),
            "max_time_since_update": int(config.max_time_since_update),
            "min_tracklet_len": int(config.min_tracklet_len),
            "min_box_iou": float(config.min_box_iou),
            "gap1_min_box_iou": float(config.gap1_min_box_iou),
            "owner_max_tracklet_len": int(config.owner_max_tracklet_len),
            "owner_alt_det_min_score": float(config.owner_alt_det_min_score),
            "owner_alt_det_min_box_iou": float(config.owner_alt_det_min_box_iou),
            "gap1_owner_alt_det_min_box_iou": float(config.gap1_owner_alt_det_min_box_iou),
            "max_owner_edge_deficit": float(config.max_owner_edge_deficit),
            "gap1_max_owner_edge_deficit": float(config.gap1_max_owner_edge_deficit),
            "evidence_mode": str(config.evidence_mode),
            "max_joint_penalty": float(config.max_joint_penalty),
            "gap1_max_joint_penalty": float(config.gap1_max_joint_penalty),
            "owner_alt_bonus": float(config.owner_alt_bonus),
            "block_owner_on_reclaim": bool(config.block_owner_on_reclaim),
            "frames": 0,
            "candidate_detections": 0,
            "candidate_pairs": 0,
            "rewrites": 0,
            "owner_rows_released": 0,
            "alt_edges_reweighted": 0,
            "blocked_owner_reclaims": 0,
            "event_count": 0,
            "skip_single_valid_row": 0,
            "skip_owner_too_long": 0,
            "challenger_reject_used_conflict": 0,
            "challenger_reject_gap_too_small": 0,
            "challenger_reject_gap_too_large": 0,
            "challenger_reject_tracklet_too_short": 0,
            "challenger_reject_box_iou_too_low": 0,
            "challenger_reject_gap1_box_iou_too_low": 0,
            "challenger_reject_edge_deficit_too_large": 0,
            "challenger_reject_gap1_edge_deficit_too_large": 0,
            "challenger_reject_joint_penalty_too_large": 0,
            "challenger_reject_gap1_joint_penalty_too_large": 0,
            "challenger_reject_no_owner_alt": 0,
            "alt_reject_same_detection": 0,
            "alt_reject_used_slot": 0,
            "alt_reject_cost_too_large": 0,
            "alt_reject_score_too_low": 0,
            "alt_reject_box_iou_too_low": 0,
            "alt_reject_gap1_box_iou_too_low": 0,
            "alt_reject_owner_not_top1": 0,
        }
        self.event_rows: List[Dict[str, int | float | str]] = []

    def is_active(self) -> bool:
        return bool(self.config.enabled)

    def _frame_debug(self) -> Dict[str, object]:
        return {
            "candidate_detections": 0,
            "candidate_pairs": 0,
            "rewrites": 0,
            "owner_rows_released": 0,
            "alt_edges_reweighted": 0,
            "blocked_owner_reclaims": 0,
            "skip_single_valid_row": 0,
            "skip_owner_too_long": 0,
            "challenger_reject_used_conflict": 0,
            "challenger_reject_gap_too_small": 0,
            "challenger_reject_gap_too_large": 0,
            "challenger_reject_tracklet_too_short": 0,
            "challenger_reject_box_iou_too_low": 0,
            "challenger_reject_gap1_box_iou_too_low": 0,
            "challenger_reject_edge_deficit_too_large": 0,
            "challenger_reject_gap1_edge_deficit_too_large": 0,
            "challenger_reject_joint_penalty_too_large": 0,
            "challenger_reject_gap1_joint_penalty_too_large": 0,
            "challenger_reject_no_owner_alt": 0,
            "alt_reject_same_detection": 0,
            "alt_reject_used_slot": 0,
            "alt_reject_cost_too_large": 0,
            "alt_reject_score_too_low": 0,
            "alt_reject_box_iou_too_low": 0,
            "alt_reject_gap1_box_iou_too_low": 0,
            "alt_reject_owner_not_top1": 0,
            "event_rows": [],
        }

    def _valid_rows(self, column_costs: np.ndarray, match_thresh: float) -> List[int]:
        valid_rows: List[int] = []
        for row_idx in np.argsort(column_costs):
            value = float(column_costs[row_idx])
            if not np.isfinite(value):
                continue
            if value > float(match_thresh):
                continue
            valid_rows.append(int(row_idx))
        return valid_rows

    def _track_gap(self, track: object, frame_id: int) -> int:
        last_frame = _safe_int(getattr(track, "frame_id", frame_id), frame_id)
        return max(0, int(frame_id) - int(last_frame))

    def _tracklet_len(self, track: object) -> int:
        return max(0, _safe_int(getattr(track, "tracklet_len", 0), 0))

    def _challenger_ok(
        self,
        track: object,
        challenger_cost: float,
        owner_cost: float,
        raw_ious_dists: np.ndarray,
        challenger_row: int,
        det_col: int,
        frame_id: int,
    ) -> tuple[bool, Dict[str, float | int], str]:
        gap = self._track_gap(track, frame_id)
        length = self._tracklet_len(track)
        box_iou = 1.0 - float(raw_ious_dists[challenger_row, det_col])
        edge_deficit = float(challenger_cost) - float(owner_cost)

        if gap < int(self.config.min_time_since_update):
            return False, {}, "gap_too_small"
        if gap > int(self.config.max_time_since_update):
            return False, {}, "gap_too_large"
        if length < int(self.config.min_tracklet_len):
            return False, {}, "tracklet_too_short"
        if box_iou < float(self.config.min_box_iou):
            return False, {}, "box_iou_too_low"
        if gap == 1 and float(self.config.gap1_min_box_iou) >= 0.0 and box_iou < float(self.config.gap1_min_box_iou):
            return False, {}, "gap1_box_iou_too_low"
        if edge_deficit > float(self.config.max_owner_edge_deficit):
            return False, {}, "edge_deficit_too_large"
        if (
            gap == 1
            and float(self.config.gap1_max_owner_edge_deficit) >= 0.0
            and edge_deficit > float(self.config.gap1_max_owner_edge_deficit)
        ):
            return False, {}, "gap1_edge_deficit_too_large"
        return True, {
            "challenger_gap": int(gap),
            "challenger_tracklet_len": int(length),
            "challenger_box_iou": float(box_iou),
            "owner_edge_deficit": float(edge_deficit),
        }, "ok"

    def _select_owner_alt(
        self,
        owner_row: int,
        det_col: int,
        modified: np.ndarray,
        raw_ious_dists: np.ndarray,
        detections: Sequence[object],
        match_thresh: float,
        challenger_gap: int,
        used_det_cols: set[int],
        used_alt_cols: set[int],
    ) -> tuple[Dict[str, float | int] | None, Dict[str, int]]:
        owner_costs = modified[owner_row]
        best_choice: Dict[str, float | int] | None = None
        best_key: tuple[float, float, float, int] | None = None
        reject_counts = {
            "same_detection": 0,
            "used_slot": 0,
            "cost_too_large": 0,
            "score_too_low": 0,
            "box_iou_too_low": 0,
            "gap1_box_iou_too_low": 0,
            "owner_not_top1": 0,
        }
        for alt_col in np.argsort(owner_costs):
            alt_col = int(alt_col)
            if alt_col == int(det_col):
                reject_counts["same_detection"] += 1
                continue
            if alt_col in used_det_cols or alt_col in used_alt_cols:
                reject_counts["used_slot"] += 1
                continue
            alt_cost = float(owner_costs[alt_col])
            if not np.isfinite(alt_cost) or alt_cost > float(match_thresh):
                reject_counts["cost_too_large"] += 1
                continue
            alt_det = detections[alt_col]
            alt_score = _safe_float(getattr(alt_det, "score", 0.0), 0.0)
            if alt_score < float(self.config.owner_alt_det_min_score):
                reject_counts["score_too_low"] += 1
                continue
            alt_box_iou = 1.0 - float(raw_ious_dists[owner_row, alt_col])
            if alt_box_iou < float(self.config.owner_alt_det_min_box_iou):
                reject_counts["box_iou_too_low"] += 1
                continue
            if (
                int(challenger_gap) == 1
                and float(self.config.gap1_owner_alt_det_min_box_iou) >= 0.0
                and alt_box_iou < float(self.config.gap1_owner_alt_det_min_box_iou)
            ):
                reject_counts["gap1_box_iou_too_low"] += 1
                continue
            alt_valid_rows = self._valid_rows(modified[:, alt_col], match_thresh)
            if not alt_valid_rows or int(alt_valid_rows[0]) != int(owner_row):
                reject_counts["owner_not_top1"] += 1
                continue
            effective_cost = max(0.0, alt_cost - float(self.config.owner_alt_bonus))
            key = (float(effective_cost), -float(alt_box_iou), -float(alt_score), int(alt_col))
            if best_key is None or key < best_key:
                best_key = key
                best_choice = {
                    "alt_col": int(alt_col),
                    "alt_cost": float(alt_cost),
                    "effective_alt_cost": float(effective_cost),
                    "alt_box_iou": float(alt_box_iou),
                    "alt_score": float(alt_score),
                }
        return best_choice, reject_counts

    def _joint_penalty_ok(
        self,
        challenger_meta: Dict[str, float | int],
        alt_choice: Dict[str, float | int],
    ) -> tuple[bool, float, str]:
        joint_penalty = float(challenger_meta["owner_edge_deficit"]) + float(alt_choice["effective_alt_cost"])
        if float(self.config.max_joint_penalty) >= 0.0 and joint_penalty > float(self.config.max_joint_penalty):
            return False, float(joint_penalty), "joint_penalty_too_large"
        if (
            int(challenger_meta["challenger_gap"]) == 1
            and float(self.config.gap1_max_joint_penalty) >= 0.0
            and joint_penalty > float(self.config.gap1_max_joint_penalty)
        ):
            return False, float(joint_penalty), "gap1_joint_penalty_too_large"
        return True, float(joint_penalty), "ok"

    def _candidate_key(
        self,
        challenger_meta: Dict[str, float | int],
        alt_choice: Dict[str, float | int],
        joint_penalty: float,
    ) -> tuple[float, ...]:
        if str(self.config.evidence_mode) == "joint":
            return (
                float(joint_penalty),
                float(challenger_meta["owner_edge_deficit"]),
                float(alt_choice["effective_alt_cost"]),
                -float(challenger_meta["challenger_box_iou"]),
                -float(alt_choice["alt_box_iou"]),
                -float(challenger_meta["challenger_gap"]),
                float(int(alt_choice["alt_col"])),
            )
        return (
            float(challenger_meta["owner_edge_deficit"]),
            float(alt_choice["effective_alt_cost"]),
            -float(challenger_meta["challenger_box_iou"]),
            -float(challenger_meta["challenger_gap"]),
            float(int(alt_choice["alt_col"])),
        )

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
        used_rows: set[int] = set()
        used_det_cols: set[int] = set()
        used_alt_cols: set[int] = set()

        for det_col in range(modified.shape[1]):
            valid_rows = self._valid_rows(modified[:, det_col], match_thresh)
            if len(valid_rows) < 2:
                debug["skip_single_valid_row"] = int(debug["skip_single_valid_row"]) + 1
                continue
            owner_row = int(valid_rows[0])
            owner_track = track_pool[owner_row]
            owner_len = self._tracklet_len(owner_track)
            if int(self.config.owner_max_tracklet_len) > 0 and owner_len > int(self.config.owner_max_tracklet_len):
                debug["skip_owner_too_long"] = int(debug["skip_owner_too_long"]) + 1
                continue

            debug["candidate_detections"] = int(debug["candidate_detections"]) + 1
            owner_cost = float(modified[owner_row, det_col])
            best_choice: Dict[str, float | int] | None = None
            best_key: tuple[float, float, float, float, int] | None = None

            for challenger_row in valid_rows[1:]:
                challenger_row = int(challenger_row)
                debug["candidate_pairs"] = int(debug["candidate_pairs"]) + 1
                if owner_row in used_rows or challenger_row in used_rows or det_col in used_det_cols:
                    debug["challenger_reject_used_conflict"] = int(debug["challenger_reject_used_conflict"]) + 1
                    continue
                challenger_track = track_pool[challenger_row]
                challenger_cost = float(modified[challenger_row, det_col])
                ok, challenger_meta, challenger_reason = self._challenger_ok(
                    challenger_track,
                    challenger_cost,
                    owner_cost,
                    raw_ious_dists,
                    challenger_row,
                    det_col,
                    frame_id,
                )
                if not ok:
                    debug[f"challenger_reject_{challenger_reason}"] = int(debug[f"challenger_reject_{challenger_reason}"]) + 1
                    continue
                alt_choice, alt_reject_counts = self._select_owner_alt(
                    owner_row,
                    det_col,
                    modified,
                    raw_ious_dists,
                    detections,
                    match_thresh,
                    int(challenger_meta["challenger_gap"]),
                    used_det_cols,
                    used_alt_cols,
                )
                for reject_name, reject_count in alt_reject_counts.items():
                    debug[f"alt_reject_{reject_name}"] = int(debug[f"alt_reject_{reject_name}"]) + int(reject_count)
                if alt_choice is None:
                    debug["challenger_reject_no_owner_alt"] = int(debug["challenger_reject_no_owner_alt"]) + 1
                    continue
                joint_ok, joint_penalty, joint_reason = self._joint_penalty_ok(challenger_meta, alt_choice)
                if not joint_ok:
                    debug[f"challenger_reject_{joint_reason}"] = int(debug[f"challenger_reject_{joint_reason}"]) + 1
                    continue
                key = self._candidate_key(challenger_meta, alt_choice, joint_penalty)
                if best_key is None or key < best_key:
                    best_key = key
                    best_choice = {
                        "owner_row": int(owner_row),
                        "challenger_row": int(challenger_row),
                        "det_col": int(det_col),
                        "owner_cost": float(owner_cost),
                        "challenger_cost": float(challenger_cost),
                        "owner_track_id": _safe_int(getattr(owner_track, "track_id", -1), -1),
                        "challenger_track_id": _safe_int(getattr(challenger_track, "track_id", -1), -1),
                        "owner_tracklet_len": int(owner_len),
                        "joint_penalty": float(joint_penalty),
                        **challenger_meta,
                        **alt_choice,
                    }

            if best_choice is None:
                continue

            owner_row = int(best_choice["owner_row"])
            challenger_row = int(best_choice["challenger_row"])
            det_col = int(best_choice["det_col"])
            alt_col = int(best_choice["alt_col"])
            modified[owner_row, alt_col] = float(best_choice["effective_alt_cost"])
            debug["alt_edges_reweighted"] = int(debug["alt_edges_reweighted"]) + 1
            if bool(self.config.block_owner_on_reclaim):
                modified[owner_row, det_col] = 1.0
                debug["blocked_owner_reclaims"] = int(debug["blocked_owner_reclaims"]) + 1
            else:
                modified[owner_row, det_col] = min(
                    1.0,
                    max(float(modified[owner_row, det_col]), float(best_choice["challenger_cost"]) + 1e-3),
                )
            used_rows.add(owner_row)
            used_rows.add(challenger_row)
            used_det_cols.add(det_col)
            used_alt_cols.add(alt_col)
            debug["rewrites"] = int(debug["rewrites"]) + 1
            debug["owner_rows_released"] = int(debug["owner_rows_released"]) + 1
            event_row = {
                "frame_id": int(frame_id),
                "owner_track_id": int(best_choice["owner_track_id"]),
                "challenger_track_id": int(best_choice["challenger_track_id"]),
                "owner_row": int(owner_row),
                "challenger_row": int(challenger_row),
                "det_col": int(det_col),
                "alt_col": int(alt_col),
                "owner_cost": float(best_choice["owner_cost"]),
                "challenger_cost": float(best_choice["challenger_cost"]),
                "owner_edge_deficit": float(best_choice["owner_edge_deficit"]),
                "joint_penalty": float(best_choice["joint_penalty"]),
                "challenger_gap": int(best_choice["challenger_gap"]),
                "challenger_tracklet_len": int(best_choice["challenger_tracklet_len"]),
                "owner_tracklet_len": int(best_choice["owner_tracklet_len"]),
                "challenger_box_iou": float(best_choice["challenger_box_iou"]),
                "alt_cost": float(best_choice["alt_cost"]),
                "effective_alt_cost": float(best_choice["effective_alt_cost"]),
                "alt_box_iou": float(best_choice["alt_box_iou"]),
                "alt_score": float(best_choice["alt_score"]),
            }
            debug["event_rows"].append(event_row)
            self.event_rows.append(event_row)

        self.stats["candidate_detections"] = int(self.stats["candidate_detections"]) + int(debug["candidate_detections"])
        self.stats["candidate_pairs"] = int(self.stats["candidate_pairs"]) + int(debug["candidate_pairs"])
        self.stats["rewrites"] = int(self.stats["rewrites"]) + int(debug["rewrites"])
        self.stats["owner_rows_released"] = int(self.stats["owner_rows_released"]) + int(debug["owner_rows_released"])
        self.stats["alt_edges_reweighted"] = int(self.stats["alt_edges_reweighted"]) + int(debug["alt_edges_reweighted"])
        self.stats["blocked_owner_reclaims"] = int(self.stats["blocked_owner_reclaims"]) + int(debug["blocked_owner_reclaims"])
        self.stats["event_count"] = int(self.stats["event_count"]) + int(len(debug["event_rows"]))
        for key in [
            "skip_single_valid_row",
            "skip_owner_too_long",
            "challenger_reject_used_conflict",
            "challenger_reject_gap_too_small",
            "challenger_reject_gap_too_large",
            "challenger_reject_tracklet_too_short",
            "challenger_reject_box_iou_too_low",
            "challenger_reject_gap1_box_iou_too_low",
            "challenger_reject_edge_deficit_too_large",
            "challenger_reject_gap1_edge_deficit_too_large",
            "challenger_reject_joint_penalty_too_large",
            "challenger_reject_gap1_joint_penalty_too_large",
            "challenger_reject_no_owner_alt",
            "alt_reject_same_detection",
            "alt_reject_used_slot",
            "alt_reject_cost_too_large",
            "alt_reject_score_too_low",
            "alt_reject_box_iou_too_low",
            "alt_reject_gap1_box_iou_too_low",
            "alt_reject_owner_not_top1",
        ]:
            self.stats[key] = int(self.stats[key]) + int(debug[key])
        return modified, debug

    def get_summary(self) -> Dict[str, int | float | bool]:
        summary = dict(self.stats)
        candidate_detections = int(summary.get("candidate_detections", 0))
        rewrites = int(summary.get("rewrites", 0))
        summary["rewrite_rate"] = float(rewrites) / float(candidate_detections) if candidate_detections > 0 else 0.0
        summary["avg_candidate_pairs_per_detection"] = (
            float(summary.get("candidate_pairs", 0)) / float(candidate_detections)
            if candidate_detections > 0
            else 0.0
        )
        summary["avg_alt_edges_per_rewrite"] = float(summary.get("alt_edges_reweighted", 0)) / float(rewrites) if rewrites > 0 else 0.0
        return summary

    def get_event_rows(self) -> List[Dict[str, int | float | str]]:
        return list(self.event_rows)

    def drain_event_rows(self) -> List[Dict[str, int | float | str]]:
        rows = list(self.event_rows)
        self.event_rows.clear()
        return rows
