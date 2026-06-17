"""QueryPlanner — cost-aware query strategy selection.

Decides *how* to query the identity archive based on archive size, detection
quality, and gap.  Returns a QueryPlan that the engine uses to drive
SpatialRouter + VectorSearcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class QueryMode(Enum):
    BRUTE_FORCE = "brute-force"  # scan all entries (small archive)
    SPATIAL_FIRST = "spatial-first"  # Hilbert route → vector re-rank
    VECTOR_FIRST = "vector-first"  # vector scan on full archive (fallback)
    SKIP = "skip"  # do not query (archive empty or det too weak)


@dataclass
class QueryPlan:
    mode: QueryMode
    spatial_radius: int  # for SPATIAL_FIRST
    top_k: int  # max candidates to return from vector search
    max_pair_budget: int  # max (entry, det) pairs to evaluate
    reason: str  # human-readable explanation


class QueryPlanner:
    """Cost-based planner for identity archive queries.

    Parameters
    ----------
    brute_force_threshold : int
        If archive size <= this, always use brute-force.
    min_det_score : float
        Detections below this score are skipped entirely.
    base_spatial_radius : int
        Default radius for spatial-first queries.
    base_top_k : int
        Default top-k for vector search results.
    max_pair_budget : int
        Hard cap on the number of (entry, det) pairs evaluated per query.
    """

    def __init__(
        self,
        brute_force_threshold: int = 50,
        min_det_score: float = 0.10,
        base_spatial_radius: int = 2,
        base_top_k: int = 8,
        max_pair_budget: int = 512,
    ):
        self.brute_force_threshold = brute_force_threshold
        self.min_det_score = min_det_score
        self.base_spatial_radius = base_spatial_radius
        self.base_top_k = base_top_k
        self.max_pair_budget = max_pair_budget

    def plan(
        self,
        archive_size: int,
        det_score: float,
        det_ambiguity: float = 0.0,
        expected_gap: int | None = None,
    ) -> QueryPlan:
        """Select a query strategy for a single detection.

        Parameters
        ----------
        archive_size : int
            Current number of entries in the identity archive.
        det_score : float
            Detection confidence.
        det_ambiguity : float
            0–1 indicating how ambiguous the primary association was.
            Higher means "more likely to need re-entry recovery."
        expected_gap : int | None
            If known, the expected gap for the best-matching archive entry.
        """
        # Skip if archive is empty
        if archive_size == 0:
            return QueryPlan(
                mode=QueryMode.SKIP,
                spatial_radius=0,
                top_k=0,
                max_pair_budget=0,
                reason="empty archive",
            )

        # Skip weak detections
        if det_score < self.min_det_score:
            return QueryPlan(
                mode=QueryMode.SKIP,
                spatial_radius=0,
                top_k=0,
                max_pair_budget=0,
                reason=f"det_score {det_score:.3f} < min {self.min_det_score}",
            )

        # Small archive → brute force
        if archive_size <= self.brute_force_threshold:
            return QueryPlan(
                mode=QueryMode.BRUTE_FORCE,
                spatial_radius=0,
                top_k=min(archive_size, self.base_top_k),
                max_pair_budget=archive_size,
                reason=f"archive_size {archive_size} <= threshold {self.brute_force_threshold}",
            )

        # Large archive → spatial-first
        # Adjust radius based on gap expectation
        radius = self.base_spatial_radius
        if expected_gap is not None:
            if expected_gap > 60:
                radius = min(self.base_spatial_radius + 2, 6)
            elif expected_gap > 30:
                radius = min(self.base_spatial_radius + 1, 5)

        # Adjust top_k based on ambiguity
        top_k = self.base_top_k
        if det_ambiguity > 0.5:
            top_k = min(top_k + 4, 16)

        return QueryPlan(
            mode=QueryMode.SPATIAL_FIRST,
            spatial_radius=radius,
            top_k=top_k,
            max_pair_budget=self.max_pair_budget,
            reason=f"spatial-first: radius={radius}, top_k={top_k}, archive={archive_size}",
        )

    def plan_batch(
        self,
        archive_size: int,
        det_scores: np.ndarray,
        det_ambiguities: np.ndarray | None = None,
    ) -> list[QueryPlan]:
        """Plan queries for a batch of detections."""
        plans = []
        for i in range(len(det_scores)):
            amb = float(det_ambiguities[i]) if det_ambiguities is not None else 0.0
            plans.append(self.plan(archive_size, float(det_scores[i]), amb))
        return plans
