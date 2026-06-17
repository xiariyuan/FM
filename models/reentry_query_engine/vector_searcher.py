"""VectorSearcher — appearance-based kNN search within candidate set.

Phase 1: flat cosine scan over a small candidate list (produced by
SpatialRouter or brute-force).  The interface is designed so that a future
version can swap in HNSW / DiskANN without changing callers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from models.reentry_query_engine.archive_index import IdentityEntry


@dataclass
class VectorMatch:
    """A single candidate after vector scoring."""

    entry: IdentityEntry
    app_sim: float  # cosine similarity on appearance
    iou_sim: float  # spatial IoU proxy
    gap_factor: float  # temporal decay
    det_score: float  # detection confidence
    composite_score: float  # weighted combination


class VectorSearcher:
    """Score archive entries against a detection's appearance vector.

    Parameters
    ----------
    app_weight : float
        Weight for appearance cosine similarity.
    iou_weight : float
        Weight for spatial IoU proxy.
    score_weight : float
        Weight for detection confidence.
    gap_weight : float
        Weight for temporal gap decay.
    min_similarity : float
        Threshold below which pairs are discarded.
    min_det_score : float
        Minimum detection score to consider.
    """

    def __init__(
        self,
        app_weight: float = 0.55,
        iou_weight: float = 0.25,
        score_weight: float = 0.10,
        gap_weight: float = 0.10,
        min_similarity: float = 0.60,
        min_det_score: float = 0.10,
    ):
        self.app_weight = app_weight
        self.iou_weight = iou_weight
        self.score_weight = score_weight
        self.gap_weight = gap_weight
        self.min_similarity = min_similarity
        self.min_det_score = min_det_score

    def score_pair(
        self,
        entry: IdentityEntry,
        det_feat: np.ndarray,
        det_score: float,
        det_tlbr: Optional[np.ndarray],
        current_frame: int,
        max_gap: int,
    ) -> Optional[VectorMatch]:
        """Score a single archive entry against one detection.

        Returns None if the pair fails any gating condition.
        """
        if det_score < self.min_det_score:
            return None
        if entry.smooth_feat is None:
            return None
        if det_feat is None:
            return None

        gap = entry.gap(current_frame)
        if gap > max_gap:
            return None

        # Appearance cosine similarity — use multi-prototype max if available
        app_sim = self._best_prototype_similarity(entry, det_feat)

        # IoU proxy: if we have the archive entry's predicted bbox, compute IoU.
        # For now use a simple distance-based proxy from velocity extrapolation.
        iou_sim = self._iou_proxy(entry, det_tlbr, current_frame)

        # Temporal decay
        gap_factor = math.exp(-float(gap) / float(max(max_gap, 1)))

        # Detection confidence
        score_val = float(det_score)

        composite = (
            self.app_weight * max(0.0, app_sim)
            + self.iou_weight * max(0.0, iou_sim)
            + self.score_weight * score_val
            + self.gap_weight * gap_factor
        )

        if composite < self.min_similarity:
            return None

        return VectorMatch(
            entry=entry,
            app_sim=app_sim,
            iou_sim=iou_sim,
            gap_factor=gap_factor,
            det_score=score_val,
            composite_score=float(np.clip(composite, 0.0, 1.0)),
        )

    def search(
        self,
        candidates: List[IdentityEntry],
        det_feat: np.ndarray,
        det_score: float,
        det_tlbr: Optional[np.ndarray],
        current_frame: int,
        max_gap: int,
        top_k: int = 0,
    ) -> List[VectorMatch]:
        """Score all candidates and return sorted matches above threshold.

        Parameters
        ----------
        top_k : int
            If > 0, return only the top-k matches.  Otherwise return all above threshold.
        """
        matches: List[VectorMatch] = []
        for entry in candidates:
            m = self.score_pair(entry, det_feat, det_score, det_tlbr, current_frame, max_gap)
            if m is not None:
                matches.append(m)

        matches.sort(key=lambda m: m.composite_score, reverse=True)
        if top_k > 0:
            matches = matches[:top_k]
        return matches

    def search_multi(
        self,
        candidates: List[IdentityEntry],
        detections_feat: np.ndarray,
        detections_score: np.ndarray,
        detections_tlbr: Optional[np.ndarray],
        current_frame: int,
        max_gap: int,
        top_k_per_det: int = 0,
    ) -> List[List[VectorMatch]]:
        """Score multiple detections against the same candidate set.

        Returns a list of match lists, one per detection.
        """
        results = []
        for i in range(len(detections_feat)):
            tlbr = detections_tlbr[i] if detections_tlbr is not None else None
            matches = self.search(
                candidates,
                detections_feat[i],
                float(detections_score[i]),
                tlbr,
                current_frame,
                max_gap,
                top_k=top_k_per_det,
            )
            results.append(matches)
        return results

    # -- internals ----------------------------------------------------------

    def _best_prototype_similarity(self, entry: "IdentityEntry", det_feat: np.ndarray) -> float:
        """Compute best cosine similarity over the entry's prototype set.

        If prototype_set is available, returns max cosine over all prototypes.
        Otherwise falls back to smooth_feat cosine.
        """
        proto_set = getattr(entry, "prototype_set", None)
        if proto_set and len(proto_set) > 0:
            best = 0.0
            for proto in proto_set:
                sim = self._cosine(proto, det_feat)
                if sim > best:
                    best = sim
            return best
        # Fallback to smooth_feat
        smooth = getattr(entry, "smooth_feat", None)
        if smooth is not None:
            return self._cosine(smooth, det_feat)
        return 0.0

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _iou_proxy(
        entry: IdentityEntry,
        det_tlbr: Optional[np.ndarray],
        current_frame: int,
    ) -> float:
        """Approximate IoU from velocity extrapolation.

        If mean_state is available (Kalman [cx, cy, a, h, vx, vy, va, vh]),
        predict current position and compute a rough overlap proxy.
        Otherwise fall back to 0.
        """
        if entry.mean_state is None or det_tlbr is None:
            return 0.0
        try:
            gap = entry.gap(current_frame)
            # predicted centre
            pred_cx = entry.mean_state[0] + gap * entry.velocity[0]
            pred_cy = entry.mean_state[1] + gap * entry.velocity[1]
            # det centre
            det_cx = (det_tlbr[0] + det_tlbr[2]) / 2.0
            det_cy = (det_tlbr[1] + det_tlbr[3]) / 2.0
            # distance-based proxy: sigmoid of normalised distance
            dist = math.sqrt((pred_cx - det_cx) ** 2 + (pred_cy - det_cy) ** 2)
            # normalise by a typical bbox size (~80 px)
            norm_dist = dist / 80.0
            return max(0.0, 1.0 - norm_dist)
        except Exception:
            return 0.0
