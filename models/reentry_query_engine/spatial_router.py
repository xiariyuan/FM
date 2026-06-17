"""SpatialRouter — coarse spatial routing via Hilbert curve.

Maps detections to Hilbert shards and selects a neighbourhood of archive
shards to probe.  The router is stateless; it delegates storage to ArchiveIndex.
"""

from __future__ import annotations

from typing import List, Tuple

from models.reentry_query_engine.archive_index import (
    ArchiveIndex,
    IdentityEntry,
    _hilbert_bucket,
)


class SpatialRouter:
    """Route a detection to candidate archive shards.

    Parameters
    ----------
    base_radius : int
        Starting neighbourhood radius for spatial probe.
    max_radius : int
        If the initial probe returns too few candidates, expand up to this.
    min_candidates : int
        Keep expanding radius until at least this many candidates are found
        (or max_radius is hit).
    """

    def __init__(
        self,
        base_radius: int = 1,
        max_radius: int = 4,
        min_candidates: int = 3,
    ):
        self.base_radius = base_radius
        self.max_radius = max_radius
        self.min_candidates = min_candidates

    def route(
        self,
        archive: ArchiveIndex,
        x: float,
        y: float,
        override_radius: int | None = None,
    ) -> List[IdentityEntry]:
        """Return spatially routed candidates for a detection at (x, y).

        Tries *base_radius* first, expands if fewer than *min_candidates*.
        """
        if archive.size == 0:
            return []

        radius = override_radius if override_radius is not None else self.base_radius
        candidates = archive.spatial_candidates(x, y, radius=radius)

        while len(candidates) < self.min_candidates and radius < self.max_radius:
            radius += 1
            candidates = archive.spatial_candidates(x, y, radius=radius)

        return candidates

    def route_radius_for_gap(self, gap: int) -> int:
        """Suggest a larger radius for older tracks that may have drifted far."""
        if gap <= 10:
            return self.base_radius
        if gap <= 30:
            return min(self.base_radius + 1, self.max_radius)
        if gap <= 60:
            return min(self.base_radius + 2, self.max_radius)
        return self.max_radius

    def route_adaptive(
        self,
        archive: ArchiveIndex,
        x: float,
        y: float,
        expected_gap: int | None = None,
    ) -> List[IdentityEntry]:
        """Route with radius adapted to expected gap."""
        if expected_gap is not None:
            radius = self.route_radius_for_gap(expected_gap)
        else:
            radius = self.base_radius
        return self.route(archive, x, y, override_radius=radius)
