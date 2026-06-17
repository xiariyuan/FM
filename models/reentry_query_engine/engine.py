"""ReentryQueryEngine — facade composing the four-layer architecture.

    Detection → QueryPlanner → SpatialRouter → VectorSearcher → CommitController
                                ↕ ArchiveIndex

The engine replaces the monolithic _recover_removed_tracks() in bot_sort.py
with a clean pipeline.  Each layer is independently testable and swappable.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from models.reentry_query_engine.archive_index import ArchiveIndex, IdentityEntry
from models.reentry_query_engine.commit_controller import CommitController, CommitDecision
from models.reentry_query_engine.metrics import ReentryMetrics
from models.reentry_query_engine.query_planner import QueryMode, QueryPlan, QueryPlanner
from models.reentry_query_engine.spatial_router import SpatialRouter
from models.reentry_query_engine.vector_searcher import VectorMatch, VectorSearcher


class ReentryQueryEngine:
    """Database-style query engine for identity re-entry recovery.

    Parameters
    ----------
    max_gap : int
        Maximum frame gap before an identity entry expires.
    hilbert_order : int
        Order of the Hilbert curve for spatial sharding (2^order cells per axis).
    brute_force_threshold : int
        Archive size below which brute-force scan is used.
    confirm_streak : int
        Consecutive frames of match needed to confirm re-entry.
    confirm_gap : int
        Max interruption before pending proposal expires.
    confirm_min_similarity : float
        Minimum score to confirm.
    app_weight, iou_weight, score_weight, gap_weight : float
        Scoring weights (same semantics as existing bot_sort params).
    min_similarity : float
        Minimum composite score to consider a match.
    min_det_score : float
        Minimum detection score to even attempt a query.
    spatial_radius : int
        Base Hilbert neighbourhood radius.
    """

    def __init__(
        self,
        max_gap: int = 120,
        max_size: int = 256,
        hilbert_order: int = 8,
        brute_force_threshold: int = 50,
        confirm_streak: int = 2,
        confirm_gap: int = 2,
        confirm_min_similarity: float = 0.65,
        app_weight: float = 0.55,
        iou_weight: float = 0.25,
        score_weight: float = 0.10,
        gap_weight: float = 0.10,
        min_similarity: float = 0.60,
        min_det_score: float = 0.10,
        spatial_radius: int = 2,
        max_spatial_radius: int = 4,
        base_top_k: int = 8,
        short_gap_threshold: int = 0,
        num_prototypes: int = 3,
        recent_score_margin: float = 0.0,
        recent_min_exit_frame_advantage: int = 0,
    ):
        self.max_gap = max_gap
        self.max_size = max(1, int(max_size))
        self.short_gap_threshold = short_gap_threshold
        self.num_prototypes = max(1, int(num_prototypes))
        self.recent_score_margin = max(0.0, float(recent_score_margin))
        self.recent_min_exit_frame_advantage = max(0, int(recent_min_exit_frame_advantage))
        self.recent_rerank_considered = 0
        self.recent_rerank_swaps = 0
        self.match_dump_enabled = False
        self._match_log: list[dict[str, object]] = []

        self.archive = ArchiveIndex(
            hilbert_order=hilbert_order,
            max_gap=max_gap,
        )
        self.spatial_router = SpatialRouter(
            base_radius=spatial_radius,
            max_radius=max_spatial_radius,
            min_candidates=3,
        )
        self.vector_searcher = VectorSearcher(
            app_weight=app_weight,
            iou_weight=iou_weight,
            score_weight=score_weight,
            gap_weight=gap_weight,
            min_similarity=min_similarity,
            min_det_score=min_det_score,
        )
        self.query_planner = QueryPlanner(
            brute_force_threshold=brute_force_threshold,
            min_det_score=min_det_score,
            base_spatial_radius=spatial_radius,
            base_top_k=base_top_k,
        )
        self.commit_controller = CommitController(
            confirm_streak=confirm_streak,
            confirm_gap=confirm_gap,
            confirm_min_similarity=confirm_min_similarity,
        )
        self.metrics = ReentryMetrics()

    # -- archive management -------------------------------------------------

    def archive_track(self, track) -> None:
        """Convert a removed STrack to an IdentityEntry and insert into archive.

        Parameters
        ----------
        track : STrack
            A removed track object from BoT-SORT with standard attributes.
        """
        from tracker.basetrack import BaseTrack

        # Extract centre from tlwh or tlbr
        tlwh = getattr(track, "_tlwh", None)
        if tlwh is None:
            cx, cy = 0.0, 0.0
        else:
            cx = float(tlwh[0]) + float(tlwh[2]) / 2.0
            cy = float(tlwh[1]) + float(tlwh[3]) / 2.0

        # Velocity from Kalman state
        mean = getattr(track, "mean", None)
        velocity = (0.0, 0.0)
        if mean is not None and len(mean) >= 8:
            velocity = (float(mean[6]), float(mean[7]))

        entry = IdentityEntry(
            track_id=int(getattr(track, "track_id", -1)),
            exit_x=cx,
            exit_y=cy,
            hilbert_key=0,
            exit_frame=int(getattr(track, "frame_id", 0)),
            smooth_feat=getattr(track, "smooth_feat", None),
            velocity=velocity,
            mean_state=mean.copy() if mean is not None else None,
            covariance=getattr(track, "covariance", None),
            exit_score=float(getattr(track, "score", 0.0)),
            tracklet_len=int(getattr(track, "tracklet_len", 0)),
            prototype_set=self._build_prototype_set(track),
        )
        self.archive.insert(entry)

    def remove_from_archive(self, track_id: int) -> bool:
        return self.archive.delete(track_id)

    def expire_entries(self, current_frame: int) -> int:
        return self.archive.expire(current_frame)

    def _build_prototype_set(self, track, max_prototypes: int | None = None) -> list[np.ndarray]:
        """Build a set of appearance prototypes from a track's feature history.

        Returns up to *max_prototypes* (default: self.num_prototypes) feature
        vectors that are diverse representations of this identity:
        1. The EMA smooth_feat (long-term stable)
        2. Recent raw features from the deque (captures recent appearance)
        3. Optionally: cluster centroids from the feature history
        """
        k = max_prototypes or self.num_prototypes
        prototypes: list[np.ndarray] = []

        # Always include smooth_feat as the primary prototype
        smooth = getattr(track, "smooth_feat", None)
        if smooth is not None:
            prototypes.append(np.asarray(smooth, dtype=np.float32).copy())

        # Add recent raw features from the deque
        features = getattr(track, "features", None)
        if features is not None and len(features) > 0:
            raw_list = list(features)
            # Take evenly spaced samples from the history
            n = len(raw_list)
            if n <= k - len(prototypes):
                # Fewer features than slots — take all
                for f in raw_list:
                    feat = np.asarray(f, dtype=np.float32)
                    if not self._is_duplicate(feat, prototypes):
                        prototypes.append(feat)
            else:
                # Sample evenly: most recent, middle, oldest
                indices = []
                remaining = k - len(prototypes)
                if remaining >= 1:
                    indices.append(n - 1)  # most recent
                if remaining >= 2:
                    indices.append(0)  # oldest
                if remaining >= 3:
                    indices.append(n // 2)  # middle
                if remaining >= 4:
                    indices.append(n // 4)
                if remaining >= 5:
                    indices.append(3 * n // 4)
                for idx in indices[:remaining]:
                    feat = np.asarray(raw_list[idx], dtype=np.float32)
                    if not self._is_duplicate(feat, prototypes):
                        prototypes.append(feat)

        return prototypes[:k]

    @staticmethod
    def _is_duplicate(feat: np.ndarray, existing: list[np.ndarray], threshold: float = 0.98) -> bool:
        """Check if feat is near-duplicate of any existing prototype."""
        if not existing:
            return False
        for e in existing:
            cos = np.dot(feat, e) / (max(np.linalg.norm(feat), 1e-8) * max(np.linalg.norm(e), 1e-8))
            if cos > threshold:
                return True
        return False

    # -- main query interface -----------------------------------------------

    def query(
        self,
        detections: list,
        detection_features: Sequence[Optional[np.ndarray]],
        current_frame: int,
        det_ambiguities: Optional[Sequence[float]] = None,
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Run re-entry queries for unmatched detections.

        Parameters
        ----------
        detections : list[STrack]
            Unmatched detection objects (each has .score, .curr_feat, ._tlwh).
        detection_features : np.ndarray
            Re-ID features for each detection, shape (N, D).
        current_frame : int
            Current frame number.

        Returns
        -------
        recovered_indices : list[int]
            Indices into *detections* that were matched and recovered.
        match_pairs : list[(track_id, det_index)]
            The (archive_track_id, detection_index) pairs that were committed.
        """
        if len(detections) == 0 or self.archive.size == 0:
            return [], []

        t0 = time.perf_counter()
        self.metrics.begin_frame(current_frame, len(detections))

        recovered_indices = []
        match_pairs = []

        for det_idx, det in enumerate(detections):
            det_score = float(getattr(det, "score", 0.0))
            det_feat = detection_features[det_idx] if det_idx < len(detection_features) else None
            if det_feat is not None:
                det_feat = np.asarray(det_feat, dtype=np.float32)
            det_ambiguity = float(det_ambiguities[det_idx]) if det_ambiguities is not None and det_idx < len(det_ambiguities) else 0.0
            det_tlbr = self._get_tlbr(det)

            # Plan
            plan = self.query_planner.plan(
                archive_size=self.archive.size,
                det_score=det_score,
                det_ambiguity=det_ambiguity,
            )

            if plan.mode == QueryMode.SKIP:
                self.metrics.record_query(0, 0, skipped=True)
                continue

            # Route
            det_cx = det_cy = 0.0
            tlwh = getattr(det, "_tlwh", None)
            if tlwh is not None:
                det_cx = float(tlwh[0]) + float(tlwh[2]) / 2.0
                det_cy = float(tlwh[1]) + float(tlwh[3]) / 2.0

            if plan.mode == QueryMode.BRUTE_FORCE:
                candidates = self.archive.scan_all()
            else:
                candidates = self.spatial_router.route(
                    self.archive, det_cx, det_cy,
                    override_radius=plan.spatial_radius,
                )

            if not candidates:
                self.metrics.record_query(0, 0)
                continue

            # Search
            if det_feat is None:
                self.metrics.record_query(len(candidates), 0)
                continue

            t_search = time.perf_counter()
            matches = self.vector_searcher.search(
                candidates=candidates,
                det_feat=det_feat,
                det_score=det_score,
                det_tlbr=det_tlbr,
                current_frame=current_frame,
                max_gap=self.max_gap,
                top_k=plan.top_k,
            )
            search_ms = (time.perf_counter() - t_search) * 1000.0

            self.metrics.record_query(len(candidates), len(matches), query_time_ms=search_ms)

            if not matches:
                continue

            # Commit: select the final candidate after optional recentness-aware rerank.
            best = self._select_best_match(matches)
            track_id = best.entry.track_id
            gap = best.entry.gap(current_frame)
            self.metrics.record_recovery_attempt(gap)

            if self.match_dump_enabled:
                self._match_log.append({
                    "frame": int(current_frame),
                    "det_index": int(det_idx),
                    "det_score": float(det_score),
                    "selected_track_id": int(track_id),
                    "selected_composite_score": float(best.composite_score),
                    "selected_app_sim": float(best.app_sim),
                    "selected_exit_frame": int(best.entry.exit_frame),
                    "top_k_tracks": [int(m.entry.track_id) for m in matches],
                    "top_k_scores": [float(m.composite_score) for m in matches],
                    "top_k_app_sims": [float(m.app_sim) for m in matches],
                    "top_k_exit_frames": [int(m.entry.exit_frame) for m in matches],
                    "num_candidates": len(matches),
                })

            # Short-gap guard: skip commit for short gaps where the host
            # tracker's own re-association is likely correct.
            if self.short_gap_threshold > 0 and gap <= self.short_gap_threshold:
                self.metrics.record_query(len(candidates), len(matches))
                continue

            decision = self.commit_controller.propose(
                track_id=track_id,
                det_index=det_idx,
                score=best.composite_score,
                frame_id=current_frame,
                origin="engine",
            )

            if decision == CommitDecision.CONFIRMED:
                recovered_indices.append(det_idx)
                match_pairs.append((track_id, det_idx))
                self.metrics.record_commit(gap)
                # Remove from archive
                self.archive.delete(track_id)

        # Cleanup stale proposals
        active_ids = {e.track_id for e in self.archive.all_entries()}
        self.commit_controller.cleanup(current_frame, active_ids)

        # Expire old entries
        self.archive.expire(current_frame)
        # Hard cap to keep the archive bounded even when tracks are constantly re-removed.
        if self.archive.size > self.max_size:
            self._prune_archive_to_size(self.max_size, current_frame)

        total_ms = (time.perf_counter() - t0) * 1000.0
        self.metrics.end_frame()

        return recovered_indices, match_pairs

    def get_stats(self) -> Dict[str, Any]:
        """Return combined statistics from all layers."""
        return {
            "archive_size": self.archive.size,
            "archive_shards": self.archive.shard_count,
            "commit": self.commit_controller.stats.as_dict(),
            "metrics": self.metrics.summary(),
            "recent_rerank": {
                "enabled": bool(self.recent_score_margin > 0.0 and self.recent_min_exit_frame_advantage > 0),
                "score_margin": float(self.recent_score_margin),
                "min_exit_frame_advantage": int(self.recent_min_exit_frame_advantage),
                "considered": int(self.recent_rerank_considered),
                "swaps": int(self.recent_rerank_swaps),
            },
        }

    # -- match dump ----------------------------------------------------------

    def enable_match_dump(self) -> None:
        self.match_dump_enabled = True
        self._match_log = []

    def drain_match_log(self) -> list[dict[str, object]]:
        rows = list(self._match_log)
        self._match_log.clear()
        return rows

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _get_tlbr(det) -> Optional[np.ndarray]:
        tlwh = getattr(det, "_tlwh", None)
        if tlwh is None:
            return None
        tlwh = np.asarray(tlwh, dtype=np.float32)
        tlbr = np.array([
            tlwh[0],
            tlwh[1],
            tlwh[0] + tlwh[2],
            tlwh[1] + tlwh[3],
        ], dtype=np.float32)
        return tlbr

    def _prune_archive_to_size(self, max_size: int, current_frame: int) -> int:
        """Remove the oldest archive entries until the archive fits the size budget."""
        if max_size <= 0 or self.archive.size <= max_size:
            return 0
        entries = sorted(
            self.archive.all_entries(),
            key=lambda e: (int(e.exit_frame), int(e.track_id)),
        )
        to_remove = max(0, self.archive.size - max_size)
        removed = 0
        for entry in entries[:to_remove]:
            if self.archive.delete(int(entry.track_id)):
                removed += 1
        return removed

    def _select_best_match(self, matches: Sequence[VectorMatch]) -> VectorMatch:
        """Select the final candidate after a minimal recentness-aware tie-break.

        Default behavior is unchanged unless both of these are enabled:
        - recent_score_margin > 0
        - recent_min_exit_frame_advantage > 0

        If the top-ranked candidate only wins by a small score margin and a more
        recent candidate exists with a sufficiently newer exit frame, prefer the
        more recent candidate. This guards against reviving an older lineage
        branch when appearance scores are nearly tied.
        """
        best = matches[0]
        if (
            len(matches) <= 1
            or self.recent_score_margin <= 0.0
            or self.recent_min_exit_frame_advantage <= 0
        ):
            return best

        self.recent_rerank_considered += 1
        head_score = float(best.composite_score)
        eligible = [
            match
            for match in matches
            if head_score - float(match.composite_score) <= self.recent_score_margin
        ]
        if len(eligible) <= 1:
            return best

        recent_best = max(
            eligible,
            key=lambda match: (int(match.entry.exit_frame), float(match.composite_score)),
        )
        exit_frame_advantage = int(recent_best.entry.exit_frame) - int(best.entry.exit_frame)
        if (
            recent_best.entry.track_id != best.entry.track_id
            and exit_frame_advantage >= self.recent_min_exit_frame_advantage
        ):
            self.recent_rerank_swaps += 1
            return recent_best
        return best
