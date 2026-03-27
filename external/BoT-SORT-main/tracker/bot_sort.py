import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from tracker import matching
from tracker.gmc import GMC
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter
from tracker.laplace_analysis import LaplaceAnalysisWriter
from tracker.haca_assoc import HACAV1Checkpoint, haca_fuse_distance
from tracker.laplace_assoc import laplace_fuse_distance
from tracker.laplace_calibrator import LaplaceAlphaRCalibrator

from fast_reid.fast_reid_interfece import FastReIDInterface


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9
        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)

    def freq_gate(self, gate_min=0.2, gate_max=1.0):
        """
        Frequency-guided stability gate for appearance association.
        Computes high/low frequency energy ratio over feature history.
        Higher instability => lower gate (down-weight appearance).
        """
        if len(self.features) < 4:
            return gate_max
        feats = np.stack(self.features, axis=0)  # (T, D)
        feats = feats - feats.mean(axis=0, keepdims=True)
        spectrum = np.fft.rfft(feats, axis=0)
        power = np.abs(spectrum) ** 2
        if power.shape[0] <= 2:
            return gate_max
        low = power[:2].mean()
        high = power[2:].mean()
        ratio = high / (low + high + 1e-6)
        gate = 1.0 - ratio
        gate = np.clip(gate, 0.0, 1.0)
        return gate_min + (gate_max - gate_min) * gate

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BoTSORT(object):
    def __init__(self, args, frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        if args.with_reid:
            self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)

        self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])
        self.freq_gate = getattr(args, "freq_gate", False)
        self.freq_gate_min = getattr(args, "freq_gate_min", 0.2)
        self.freq_gate_max = getattr(args, "freq_gate_max", 1.0)
        self.laplace_assoc = getattr(args, "laplace_assoc", False)
        self.laplace_primary_only = getattr(args, "laplace_primary_only", False)
        self.laplace_weight = getattr(args, "laplace_weight", 0.35)
        self.laplace_decay_scales = getattr(args, "laplace_decay_scales", [1.0, 2.0, 4.0])
        self.laplace_min_history = getattr(args, "laplace_min_history", 3)
        self.laplace_proto_mode = getattr(args, "laplace_proto_mode", "multi")
        self.laplace_no_reliability = getattr(args, "laplace_no_reliability", False)
        self.laplace_no_det_score = getattr(args, "laplace_no_det_score", False)
        self.laplace_disable_pole_bank = getattr(args, "laplace_disable_pole_bank", False)
        self.laplace_reliability_scale = getattr(args, "laplace_reliability_scale", 1.0)
        self.laplace_agreement_mode = getattr(args, "laplace_agreement_mode", "absdiff")
        self.laplace_use_history_len = getattr(args, "laplace_use_history_len", False)
        self.laplace_history_len_gamma = getattr(args, "laplace_history_len_gamma", 1.0)
        self.laplace_assoc_mode = str(getattr(args, "laplace_assoc_mode", "auto") or "auto").lower()
        self.laplace_haca_checkpoint_path = getattr(args, "laplace_haca_checkpoint", "") or ""
        self.laplace_haca_no_set_encoder = getattr(args, "laplace_haca_no_set_encoder", False)
        self.laplace_haca_no_background = getattr(args, "laplace_haca_no_background", False)
        delta_scale_override = getattr(args, "laplace_haca_delta_scale", float("nan"))
        self.laplace_haca_delta_scale = float(delta_scale_override) if np.isfinite(delta_scale_override) else None
        self.laplace_analysis = None
        self.laplace_analysis_dir = getattr(args, "laplace_analysis_dir", "")
        if self.laplace_analysis_dir:
            self.laplace_analysis = LaplaceAnalysisWriter(
                seq_name=args.name,
                img_dir=args.path,
                out_dir=self.laplace_analysis_dir,
            )

        self.laplace_calibrator = None
        self.laplace_calibrator_path = getattr(args, "laplace_calibrator", "") or ""
        if self.laplace_calibrator_path:
            self.laplace_calibrator = LaplaceAlphaRCalibrator.from_npz(self.laplace_calibrator_path)
        self.laplace_haca_checkpoint = None
        if self.laplace_haca_checkpoint_path:
            self.laplace_haca_checkpoint = HACAV1Checkpoint.from_npz(self.laplace_haca_checkpoint_path)

        if self.laplace_assoc_mode not in {"auto", "heuristic", "current_learned", "haca_v1", "haca_v2", "haca_v3"}:
            raise ValueError(f"Unsupported laplace_assoc_mode: {self.laplace_assoc_mode}")
        if self.laplace_assoc_mode == "current_learned" and self.laplace_calibrator is None:
            raise ValueError("laplace_assoc_mode=current_learned requires --laplace-calibrator")
        if self.laplace_assoc_mode in {"haca_v1", "haca_v2", "haca_v3"} and self.laplace_haca_checkpoint is None:
            raise ValueError(f"laplace_assoc_mode={self.laplace_assoc_mode} requires --laplace-haca-checkpoint")
        if self.laplace_assoc_mode in {"haca_v1", "haca_v2", "haca_v3"} and self.laplace_no_reliability:
            raise ValueError("HACA association does not support --laplace-no-reliability")

    def __del__(self):
        if getattr(self, "laplace_analysis", None) is not None:
            self.laplace_analysis.close()

    def update(self, output_results, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, -1]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                classes = output_results[:, -1]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]

        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        '''Extract embeddings '''
        if self.args.with_reid:
            features_keep = self.encoder.inference(img, dets)

        if len(dets) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                              (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        if self.laplace_analysis is not None:
            for det in detections:
                self.laplace_analysis.assign_detection_gt(det, self.frame_id)

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        raw_ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = (raw_ious_dists > self.proximity_thresh)
        ious_dists = raw_ious_dists.copy()
        use_haca_primary = bool(
                self.laplace_assoc
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

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
            if self.laplace_assoc:
                det_scores = np.array([det.score for det in detections], dtype=np.float32) if len(detections) > 0 else None
                track_gaps = np.array(
                    [max(0, int(self.frame_id) - int(getattr(t, "frame_id", self.frame_id))) for t in strack_pool],
                    dtype=np.float32,
                )
                return_debug = self.laplace_analysis is not None
                if use_haca_primary:
                    laplace_out = haca_fuse_distance(
                        strack_pool,
                        detections,
                        spatial_cost=emb_dists,
                        motion_cost=raw_ious_dists,
                        checkpoint=self.laplace_haca_checkpoint,
                        det_scores=det_scores,
                        decay_scales=self.laplace_decay_scales,
                        min_history=self.laplace_min_history,
                        proto_mode=self.laplace_proto_mode,
                        use_det_score=not self.laplace_no_det_score,
                        track_gaps=track_gaps,
                        valid_mask=np.logical_not(ious_dists_mask),
                        use_set_encoder=not self.laplace_haca_no_set_encoder,
                        use_background=not self.laplace_haca_no_background,
                        delta_scale=self.laplace_haca_delta_scale,
                        return_debug=return_debug,
                    )
                else:
                    laplace_out = laplace_fuse_distance(
                        strack_pool,
                        detections,
                        spatial_cost=emb_dists,
                        motion_cost=raw_ious_dists,
                        det_scores=det_scores,
                        decay_scales=self.laplace_decay_scales,
                        appearance_alpha=self.laplace_weight,
                        min_history=self.laplace_min_history,
                        proto_mode=self.laplace_proto_mode,
                        use_reliability=not self.laplace_no_reliability,
                        use_det_score=not self.laplace_no_det_score,
                        reliability_scale=self.laplace_reliability_scale,
                        agreement_mode=self.laplace_agreement_mode,
                        use_history_len=self.laplace_use_history_len,
                        history_len_gamma=self.laplace_history_len_gamma,
                        track_gaps=track_gaps,
                        valid_mask=np.logical_not(ious_dists_mask),
                        calibrator=active_calibrator,
                        use_pole_bank=not self.laplace_disable_pole_bank,
                        return_debug=return_debug,
                    )
                if return_debug:
                    laplace_dists, laplace_debug = laplace_out
                else:
                    laplace_dists = laplace_out
                    laplace_debug = None
                if self.laplace_no_reliability:
                    emb_dists = laplace_dists
                    emb_dists[emb_dists > self.appearance_thresh] = 1.0
                    emb_dists[ious_dists_mask] = 1.0
                    dists = np.minimum(ious_dists, emb_dists)
                else:
                    dists = laplace_dists
            else:
                if self.freq_gate:
                    gates = np.array(
                        [t.freq_gate(self.freq_gate_min, self.freq_gate_max) for t in strack_pool],
                        dtype=np.float32,
                    )
                    if gates.size > 0:
                        emb_dists = emb_dists / gates.reshape(-1, 1)
                raw_emb_dists = emb_dists.copy()
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                dists = np.minimum(ious_dists, emb_dists)
            dists[ious_dists_mask] = 1.0

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        if self.laplace_analysis is not None and self.laplace_assoc and len(strack_pool) > 0 and len(detections) > 0:
            valid_mask = np.logical_not(ious_dists_mask)
            chosen_pairs = set((int(r), int(c)) for r, c in matches)
            self.laplace_analysis.log_first_assoc(
                frame_id=self.frame_id,
                tracks=strack_pool,
                detections=detections,
                debug=laplace_debug,
                chosen_pairs=chosen_pairs,
                valid_mask=valid_mask,
                assoc_stage="primary",
            )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                track.analysis_gt_id = getattr(det, "analysis_gt_id", -1)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                track.analysis_gt_id = getattr(det, "analysis_gt_id", -1)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        raw_ious_dists = matching.iou_distance(unconfirmed, detections)
        ious_dists_mask = (raw_ious_dists > self.proximity_thresh)
        ious_dists = raw_ious_dists.copy()
        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            if self.laplace_assoc and not self.laplace_primary_only and not use_haca_primary:
                det_scores = np.array([det.score for det in detections], dtype=np.float32) if len(detections) > 0 else None
                track_gaps = np.array(
                    [max(0, int(self.frame_id) - int(getattr(t, "frame_id", self.frame_id))) for t in unconfirmed],
                    dtype=np.float32,
                )
                laplace_dists = laplace_fuse_distance(
                    unconfirmed,
                    detections,
                    spatial_cost=emb_dists,
                    motion_cost=raw_ious_dists,
                    det_scores=det_scores,
                    decay_scales=self.laplace_decay_scales,
                    appearance_alpha=self.laplace_weight,
                    min_history=self.laplace_min_history,
                    proto_mode=self.laplace_proto_mode,
                    use_reliability=not self.laplace_no_reliability,
                    use_det_score=not self.laplace_no_det_score,
                    reliability_scale=self.laplace_reliability_scale,
                    agreement_mode=self.laplace_agreement_mode,
                    use_history_len=self.laplace_use_history_len,
                    history_len_gamma=self.laplace_history_len_gamma,
                    track_gaps=track_gaps,
                    valid_mask=np.logical_not(ious_dists_mask),
                    calibrator=active_calibrator,
                    use_pole_bank=not self.laplace_disable_pole_bank,
                )
                if self.laplace_no_reliability:
                    emb_dists = laplace_dists
                    emb_dists[emb_dists > self.appearance_thresh] = 1.0
                    emb_dists[ious_dists_mask] = 1.0
                    dists = np.minimum(ious_dists, emb_dists)
                else:
                    dists = laplace_dists
                    dists[ious_dists_mask] = 1.0
            else:
                raw_emb_dists = emb_dists.copy()
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            track.analysis_gt_id = getattr(track, "analysis_gt_id", -1)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]


        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
