"""Stage 3 retrieval late recovery for RGSA.

For unmatched detections after Stage 2, query the archive (removed_stracks)
using ReID + spatial proximity. Lightweight: no independent Hungarian.

Uses the existing ReentryQueryEngine infrastructure.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from models.rgsa_contract import Stage3Output


class Stage3RetrievalHead:
    """Rule-based + optional learned scoring for archive retrieval.

    Phase 1 (default): uses hand-crafted composite score
      score = app_weight * appearance_sim
            + iou_weight * iou_sim
            + score_weight * det_score
            + gap_weight * gap_decay

    Phase 2 (future): replace with learned scorer
    """

    def __init__(
        self,
        recovery_threshold: float = 0.60,
        app_weight: float = 0.55,
        iou_weight: float = 0.25,
        det_score_weight: float = 0.10,
        gap_weight: float = 0.10,
        max_gap: int = 30,
        min_det_score: float = 0.10,
    ):
        self.recovery_threshold = recovery_threshold
        self.app_weight = app_weight
        self.iou_weight = iou_weight
        self.det_score_weight = det_score_weight
        self.gap_weight = gap_weight
        self.max_gap = max_gap
        self.min_det_score = min_det_score

    def recover(
        self,
        unmatched_det_ids: List[int],
        detections: list,
        archive_tracks: list,
        frame_id: int,
        cosine_similarity_fn=None,
        iou_fn=None,
    ) -> Stage3Output:
        """Attempt recovery of unmatched detections from archive.

        Args:
            unmatched_det_ids: indices into detections array
            detections: list of detection objects with .tlwh, .score, .curr_feat
            archive_tracks: list of removed STrack objects with .smooth_feat, .end_frame
            frame_id: current frame id
            cosine_similarity_fn: callable(track_feat, det_feat) -> float
            iou_fn: callable(track_tlwh, det_tlwh) -> float

        Returns:
            Stage3Output
        """
        output = Stage3Output()
        if not unmatched_det_ids or not archive_tracks:
            output.remaining_unmatched_det_ids = list(unmatched_det_ids)
            return output

        if cosine_similarity_fn is None:
            cosine_similarity_fn = self._default_cosine_similarity
        if iou_fn is None:
            iou_fn = self._default_iou

        for det_id in unmatched_det_ids:
            det = detections[det_id]
            det_score = float(getattr(det, "score", 0.0))
            if det_score < self.min_det_score:
                output.remaining_unmatched_det_ids.append(det_id)
                continue

            best_score = -1.0
            best_track_id = -1

            for track in archive_tracks:
                gap = int(frame_id) - int(getattr(track, "end_frame", frame_id))
                if gap < 1 or gap > self.max_gap:
                    continue

                app_sim = cosine_similarity_fn(
                    getattr(track, "smooth_feat", None),
                    getattr(det, "curr_feat", None),
                )
                track_tlwh = getattr(track, "tlwh", None)
                det_tlwh = getattr(det, "tlwh", None)
                iou_sim = 0.0
                if track_tlwh is not None and det_tlwh is not None:
                    iou_sim = max(0.0, iou_fn(track_tlwh, det_tlwh))

                gap_factor = np.exp(-float(gap) / float(max(self.max_gap, 1)))

                score = (
                    self.app_weight * max(0.0, app_sim)
                    + self.iou_weight * max(0.0, iou_sim)
                    + self.det_score_weight * det_score
                    + self.gap_weight * gap_factor
                )

                if score > best_score:
                    best_score = score
                    best_track_id = int(getattr(track, "track_id", -1))

            if best_score >= self.recovery_threshold and best_track_id >= 0:
                output.recovered_matches[det_id] = best_track_id
                output.recovery_scores[det_id] = best_score
            else:
                output.remaining_unmatched_det_ids.append(det_id)

        return output

    @staticmethod
    def _default_cosine_similarity(feat_a, feat_b) -> float:
        if feat_a is None or feat_b is None:
            return 0.0
        a = np.asarray(feat_a, dtype=np.float32).ravel()
        b = np.asarray(feat_b, dtype=np.float32).ravel()
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm < 1e-8:
            return 0.0
        return float(np.dot(a, b) / norm)

    @staticmethod
    def _default_iou(tlwh_a, tlwh_b) -> float:
        a = np.asarray(tlwh_a, dtype=np.float32)
        b = np.asarray(tlwh_b, dtype=np.float32)
        xa, ya = a[0], a[1]
        xb, yb = b[0], b[1]
        wa, ha = a[2], a[3]
        wb, hb = b[2], b[3]
        x1 = max(xa, xb)
        y1 = max(ya, yb)
        x2 = min(xa + wa, xb + wb)
        y2 = min(ya + ha, yb + hb)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = wa * ha + wb * hb - inter
        if union < 1e-8:
            return 0.0
        return float(inter / union)
