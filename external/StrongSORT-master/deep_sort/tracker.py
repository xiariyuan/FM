# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .haca_assoc import HACAV1Checkpoint, haca_fuse_cost
from .laplace_assoc import laplace_fuse_cost
from .laplace_calibrator import LaplaceAlphaRCalibrator
from .track import Track
from opts import opt

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3, laplace_analysis=None):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.laplace_calibrator = None
        self.laplace_analysis = laplace_analysis
        self.laplace_calibrator_path = getattr(opt, "laplace_calibrator", "") or ""
        if getattr(opt, "LAPLACE", False) and self.laplace_calibrator_path:
            self.laplace_calibrator = LaplaceAlphaRCalibrator.from_npz(self.laplace_calibrator_path)
        self.laplace_assoc_mode = str(getattr(opt, "laplace_assoc_mode", "auto") or "auto").lower()
        self.laplace_haca_checkpoint_path = getattr(opt, "laplace_haca_checkpoint", "") or ""
        self.laplace_haca_no_set_encoder = getattr(opt, "laplace_haca_no_set_encoder", False)
        self.laplace_haca_no_background = getattr(opt, "laplace_haca_no_background", False)
        delta_scale_override = getattr(opt, "laplace_haca_delta_scale", float("nan"))
        self.laplace_haca_delta_scale = float(delta_scale_override) if np.isfinite(delta_scale_override) else None
        self.laplace_haca_checkpoint = None
        if getattr(opt, "LAPLACE", False) and self.laplace_haca_checkpoint_path:
            self.laplace_haca_checkpoint = HACAV1Checkpoint.from_npz(self.laplace_haca_checkpoint_path)
        if self.laplace_assoc_mode not in {"auto", "heuristic", "current_learned", "haca_v1", "haca_v2", "haca_v3"}:
            raise ValueError(f"Unsupported laplace_assoc_mode: {self.laplace_assoc_mode}")
        if self.laplace_assoc_mode == "current_learned" and self.laplace_calibrator is None:
            raise ValueError("laplace_assoc_mode=current_learned requires --laplace-calibrator")
        if self.laplace_assoc_mode in {"haca_v1", "haca_v2", "haca_v3"} and self.laplace_haca_checkpoint is None:
            raise ValueError(f"laplace_assoc_mode={self.laplace_assoc_mode} requires --laplace-haca-checkpoint")

        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict()

    def camera_update(self, video, frame):
        for track in self.tracks:
            track.camera_update(video, frame)

    def update(self, detections, frame_id=None):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections, frame_id=frame_id)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            if not opt.EMA:
                track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections, frame_id=None):
        laplace_debug_states = []

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            debug = None
            use_haca_primary = bool(
                getattr(opt, "LAPLACE", False)
                and (
                    self.laplace_assoc_mode in {"haca_v1", "haca_v2", "haca_v3"}
                    or (self.laplace_assoc_mode == "auto" and self.laplace_haca_checkpoint is not None)
                )
            )
            if self.laplace_assoc_mode == "heuristic":
                active_calibrator = None
            elif self.laplace_assoc_mode == "current_learned":
                active_calibrator = self.laplace_calibrator
            elif self.laplace_assoc_mode in {"haca_v1", "haca_v2", "haca_v3"}:
                active_calibrator = None
            else:
                active_calibrator = self.laplace_calibrator
            if getattr(opt, "LAPLACE", False):
                return_debug = self.laplace_analysis is not None and frame_id is not None
                if use_haca_primary:
                    laplace_out = haca_fuse_cost(
                        tracks,
                        dets,
                        track_indices,
                        detection_indices,
                        spatial_cost=cost_matrix,
                        gating_threshold=kalman_filter.chi2inv95[4],
                        checkpoint=self.laplace_haca_checkpoint,
                        decay_scales=getattr(opt, "laplace_decay_scales", [1.0, 2.0, 4.0]),
                        min_history=getattr(opt, "laplace_min_history", 3),
                        use_set_encoder=not getattr(opt, "laplace_haca_no_set_encoder", False),
                        use_background=not getattr(opt, "laplace_haca_no_background", False),
                        delta_scale=self.laplace_haca_delta_scale,
                        return_debug=return_debug,
                    )
                if self.laplace_analysis is not None and frame_id is not None:
                    if use_haca_primary:
                        cost_matrix, debug = laplace_out
                    else:
                        cost_matrix, debug = laplace_fuse_cost(
                            tracks,
                            dets,
                            track_indices,
                            detection_indices,
                            spatial_cost=cost_matrix,
                            gating_threshold=kalman_filter.chi2inv95[4],
                            decay_scales=getattr(opt, "laplace_decay_scales", [1.0, 2.0, 4.0]),
                            appearance_alpha=getattr(opt, "laplace_weight", 0.35),
                            min_history=getattr(opt, "laplace_min_history", 3),
                            calibrator=active_calibrator,
                            use_pole_bank=not getattr(opt, "laplace_disable_pole_bank", False),
                            return_debug=True,
                        )
                else:
                    if use_haca_primary:
                        debug = None
                        cost_matrix = laplace_out
                    else:
                        debug = None
                        cost_matrix = laplace_fuse_cost(
                            tracks,
                            dets,
                            track_indices,
                            detection_indices,
                            spatial_cost=cost_matrix,
                            gating_threshold=kalman_filter.chi2inv95[4],
                            decay_scales=getattr(opt, "laplace_decay_scales", [1.0, 2.0, 4.0]),
                            appearance_alpha=getattr(opt, "laplace_weight", 0.35),
                            min_history=getattr(opt, "laplace_min_history", 3),
                            calibrator=active_calibrator,
                            use_pole_bank=not getattr(opt, "laplace_disable_pole_bank", False),
                        )
            cost_matrix = linear_assignment.gate_cost_matrix(
                cost_matrix, tracks, dets, track_indices,
                detection_indices)
            if debug is not None:
                laplace_debug_states.append(
                    {
                        "track_indices": list(track_indices),
                        "detection_indices": list(detection_indices),
                        "debug": debug,
                        "valid_mask": (cost_matrix < linear_assignment.INFTY_COST).copy(),
                    }
                )

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)
        if self.laplace_analysis is not None and frame_id is not None and laplace_debug_states:
            self._log_laplace_assoc(frame_id, detections, matches_a, laplace_debug_states)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _log_laplace_assoc(self, frame_id, detections, matches, debug_states):
        matched_pairs = set(matches)
        for state in debug_states:
            track_indices = state["track_indices"]
            detection_indices = state["detection_indices"]
            if not track_indices or not detection_indices:
                continue
            row_of = {track_idx: row for row, track_idx in enumerate(track_indices)}
            col_of = {det_idx: col for col, det_idx in enumerate(detection_indices)}
            chosen_pairs = set()
            for track_idx, det_idx in matched_pairs:
                row = row_of.get(track_idx)
                col = col_of.get(det_idx)
                if row is not None and col is not None:
                    chosen_pairs.add((row, col))
            track_subset = [self.tracks[idx] for idx in track_indices]
            det_subset = [detections[idx] for idx in detection_indices]
            self.laplace_analysis.log_first_assoc(
                frame_id=frame_id,
                tracks=track_subset,
                detections=det_subset,
                debug=state["debug"],
                chosen_pairs=chosen_pairs,
                valid_mask=state["valid_mask"],
                assoc_stage="primary",
            )

    def _initiate_track(self, detection):
        self.tracks.append(Track(
            detection.to_xyah(), self._next_id, self.n_init, self.max_age,
            detection.feature, detection.confidence))
        self._next_id += 1
