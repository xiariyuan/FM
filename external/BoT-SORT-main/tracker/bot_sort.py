import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import math
import os.path as osp
import sys

from tracker import matching
from tracker.gmc import GMC
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter
from tracker.laplace_analysis import LaplaceAnalysisWriter
from tracker.haca_assoc import HACAV1Checkpoint, haca_fuse_distance
from tracker.laplace_assoc import laplace_fuse_distance
from tracker.laplace_calibrator import LaplaceAlphaRCalibrator
from tracker.local_graph_reassoc import LocalGraphReassocConfig, LocalGraphReassocRefiner
from tracker.owneralt_competition import OwnerAltCompetitionConfig, OwnerAltCompetitionRefiner

from fast_reid.fast_reid_interfece import FastReIDInterface

_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from projects.fcaa.fcaa.runtime.fcaa_refiner import FCAAConfig, FCAARefiner
except Exception:
    FCAAConfig = None
    FCAARefiner = None

try:
    from projects.fgas.fgas.runtime.block_refiner import FGASConfig, FGASBlockRefiner
except Exception:
    FGASConfig = None
    FGASBlockRefiner = None

try:
    from models.graph_assoc_commit_runtime import GraphAssocCommitScorer
except Exception:
    GraphAssocCommitScorer = None

try:
    from models.graph_assoc_gate_runtime import GraphAssocGateScorer, checkpoint_looks_like_graph_assoc_gate
except Exception:
    GraphAssocGateScorer = None
    checkpoint_looks_like_graph_assoc_gate = None

try:
    from models.graph_assoc_competition_runtime import (
        GraphAssocCompetitionScorer,
        checkpoint_looks_like_graph_assoc_competition,
    )
except Exception:
    GraphAssocCompetitionScorer = None
    checkpoint_looks_like_graph_assoc_competition = None

try:
    from models.reentry_query_engine import ReentryQueryEngine
except Exception:
    ReentryQueryEngine = None

try:
    from models.rgsa_stage1_deferral import Stage1DeferralHead
    from models.rgsa_stage2_recovery import Stage2RecoveryHead
    from models.rgsa_stage2_verifier import HeuristicVerifier
    from models.ccrc_calibrators import PlattScaling
except Exception:
    Stage1DeferralHead = None
    Stage2RecoveryHead = None


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
        self.fcaa_low = None
        self.fcaa_mid = None
        self.fcaa_high = None
        self.fcaa_band_desc = None
        if feat is not None:
            self.update_features(feat)

    def release_memory(self):
        self.features.clear()
        self.smooth_feat = None
        self.curr_feat = None
        self.fcaa_low = None
        self.fcaa_mid = None
        self.fcaa_high = None
        self.fcaa_band_desc = None
        self.mean = None
        self.covariance = None
        self.kalman_filter = None

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

    def update_features(self, feat, mode="normal", alpha_override=None, append_history=True):
        # Normalize with epsilon guard; avoid in-place mutation of input array
        norm = max(float(np.linalg.norm(feat)), 1e-12)
        feat_normed = (feat / norm).copy()
        self.curr_feat = feat_normed

        if mode == "freeze":
            # freeze: do NOT update smooth_feat, do NOT append to history
            # curr_feat is already updated above
            pass
        elif mode == "soft":
            # soft: conservative EMA update
            alpha = alpha_override if alpha_override is not None else self.alpha
            if self.smooth_feat is None:
                self.smooth_feat = feat_normed.copy()
            else:
                self.smooth_feat = alpha * self.smooth_feat + (1 - alpha) * feat_normed
            smooth_norm = max(float(np.linalg.norm(self.smooth_feat)), 1e-12)
            self.smooth_feat = self.smooth_feat / smooth_norm
            if append_history:
                self.features.append(feat_normed.copy())
        else:
            # normal: original behavior
            if self.smooth_feat is None:
                self.smooth_feat = feat_normed.copy()
            else:
                self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat_normed
            smooth_norm = max(float(np.linalg.norm(self.smooth_feat)), 1e-12)
            self.smooth_feat = self.smooth_feat / smooth_norm
            self.features.append(feat_normed.copy())

    def update_fcaa_bands(self, desc, momentum=0.9):
        if desc is None:
            return
        self.fcaa_band_desc = desc
        for band_name in ("low", "mid", "high"):
            new_value = np.asarray(getattr(desc, band_name), dtype=np.float32)
            current = getattr(self, f"fcaa_{band_name}", None)
            if current is None or float(momentum) <= 0.0:
                setattr(self, f"fcaa_{band_name}", new_value)
                continue
            mixed = float(momentum) * np.asarray(current, dtype=np.float32) + (1.0 - float(momentum)) * new_value
            norm = float(np.linalg.norm(mixed))
            if norm > 1e-8:
                mixed = mixed / norm
            setattr(self, f"fcaa_{band_name}", mixed.astype(np.float32))

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
        if getattr(self, "fcaa_band_desc", None) is not None:
            self.update_fcaa_bands(self.fcaa_band_desc, momentum=0.0)

        # TOS-Track attributes: initialized on first activation
        self.tos_is_shadow = False       # True when track is in occlusion shadow state
        self.tos_shadow_start = -1      # frame_id when shadow state began
        self.tos_shadow_duration = 0    # frames spent in shadow
        self.tos_reconnect_det_id = -1  # detection id used for reconnection (analysis)
        self.tos_reconnect_app_sim = 0.0  # appearance similarity at reconnection (analysis)
        self.tos_occlusion_score = 0.0  # HACA-derived occlusion score (0-1)
        self.tos_last_seen = frame_id   # last frame_id this track saw a detection
        self.tos_hold_countdown = 0     # countdown frames before creating newborn ID
        self.tos_reconnect_cooldown = 0  # frames since last reconnection (prevents rapid cycles)

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            mode = getattr(new_track, "tcgau_update_mode", "normal")
            alpha_override = getattr(new_track, "tcgau_alpha_override", None)
            append_history = getattr(new_track, "tcgau_append_history", True)
            self.update_features(new_track.curr_feat, mode=mode, alpha_override=alpha_override, append_history=append_history)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        if getattr(new_track, "fcaa_band_desc", None) is not None:
            self.update_fcaa_bands(new_track.fcaa_band_desc)
        # TOS: reset shadow state on reconnection
        self.tos_is_shadow = False
        self.tos_shadow_start = -1
        self.tos_shadow_duration = 0
        self.tos_last_seen = frame_id
        self.tos_reconnect_cooldown = 0

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
            mode = getattr(new_track, "tcgau_update_mode", "normal")
            alpha_override = getattr(new_track, "tcgau_alpha_override", None)
            append_history = getattr(new_track, "tcgau_append_history", True)
            self.update_features(new_track.curr_feat, mode=mode, alpha_override=alpha_override, append_history=append_history)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if getattr(new_track, "fcaa_band_desc", None) is not None:
            self.update_fcaa_bands(new_track.fcaa_band_desc)

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
        # RGSA oracle dump: collect HACA debug data per frame
        self.rgsa_dump_dir = getattr(args, "rgsa_dump_dir", "") or ""
        self._rgsa_frame_rows = [] if self.rgsa_dump_dir else None
        self._rgsa_pairbank_path = None
        self._rgsa_pairbank_fieldnames = None
        if self.rgsa_dump_dir:
            import os as _os
            _os.makedirs(self.rgsa_dump_dir, exist_ok=True)
            self._rgsa_pairbank_path = _os.path.join(self.rgsa_dump_dir, "pairbank.csv")
            if _os.path.isfile(self._rgsa_pairbank_path):
                _os.remove(self._rgsa_pairbank_path)
        self.rgsa_enable = bool(getattr(args, "rgsa_enable", False))
        self.rgsa_stage1_checkpoint_path = str(getattr(args, "rgsa_stage1_checkpoint", "") or "")
        self.rgsa_stage2_checkpoint_path = str(getattr(args, "rgsa_stage2_checkpoint", "") or "")
        self.rgsa_device = str(getattr(args, "rgsa_device", "") or "").strip()
        if not self.rgsa_device:
            self.rgsa_device = "cuda:0" if str(getattr(args, "device", "gpu")).lower() == "gpu" else "cpu"
        if (
            self.rgsa_device.startswith("cuda")
            and (not hasattr(cv2, "cuda") or not hasattr(cv2.cuda, "getCudaEnabledDeviceCount") or not cv2.cuda.getCudaEnabledDeviceCount())
        ):
            self.rgsa_device = "cpu"
        self.rgsa_topk = max(1, int(getattr(args, "rgsa_topk", 5)))
        self.rgsa_stage1_lambda_defer = float(getattr(args, "rgsa_stage1_lambda_defer", 0.3))
        self.rgsa_stage1_lambda_reject = float(getattr(args, "rgsa_stage1_lambda_reject", 0.8))
        self.rgsa_stage2_rewrite_gain = float(getattr(args, "rgsa_stage2_rewrite_gain", 0.35))
        # Dual-param HACA Stage2: use different HACA competitive params for deferred subset
        self.rgsa_stage2_haca_mode = str(getattr(args, "rgsa_stage2_haca_mode", "learned") or "learned")
        # aggressive HACA params for deferred re-scoring (used when mode=dual_haca)
        self.rgsa_stage2_comp_delta_scale = float(getattr(args, "rgsa_stage2_comp_delta_scale", 2.0))
        self.rgsa_stage2_comp_margin_temperature = float(getattr(args, "rgsa_stage2_comp_margin_temperature", 0.05))
        self.rgsa_stage2_rewrite_threshold = float(getattr(args, "rgsa_stage2_rewrite_threshold", 0.05))
        self.rgsa_stage2_cost_discount = float(getattr(args, "rgsa_stage2_cost_discount", 0.2))
        # Verifier mode
        self.rgsa_verifier_mode = str(getattr(args, "rgsa_verifier_mode", "none") or "none")
        self.rgsa_verifier = None
        # CCRC Platt calibrator
        self.ccrc_platt_checkpoint_path = str(getattr(args, "ccrc_platt_checkpoint", "") or "")
        self.ccrc_platt_a = 1.0
        self.ccrc_platt_b = 0.0
        self.ccrc_tau_commit = float(getattr(args, "ccrc_tau_commit", 0.0))
        self.ccrc_enable = bool(getattr(args, "ccrc_enable", False))
        self.ccrc_stats = {"frames": 0, "commits": 0, "abstains": 0, "abstain_correct": 0, "abstain_wrong": 0}
        # TCGAU: Track-Coherence-Gated Appearance Update
        self.tcgau_enable = bool(getattr(args, "tcgau_enable", False))
        self.tcgau_freeze_thresh = float(getattr(args, "tcgau_freeze_thresh", 0.30))
        self.tcgau_soft_thresh = float(getattr(args, "tcgau_soft_thresh", 0.70))
        self.tcgau_soft_alpha = float(getattr(args, "tcgau_soft_alpha", 0.97))
        self.tcgau_margin_thresh = float(getattr(args, "tcgau_margin_thresh", 0.05))
        self.tcgau_history_norm_denom = float(getattr(args, "tcgau_history_norm_denom", 10.0))
        self.tcgau_log_pairs = bool(getattr(args, "tcgau_log_pairs", False))
        self.tcgau_pair_log = []
        self.tcgau_stats = {
            "frames": 0,
            "primary_matches": 0,
            "normal_updates": 0,
            "soft_updates": 0,
            "freeze_updates": 0,
            "avg_q_update": 0.0,
            "avg_app_sim": 0.0,
            "avg_pair_rel": 0.0,
        }
        self.tcgau_debug_available = False
        self.rgsa_stage1_head = None
        self.rgsa_stage2_head = None
        self.rgsa_stats = {
            "enabled": bool(self.rgsa_enable),
            "frames": 0,
            "active_frames": 0,
            "inactive_frames": 0,
            "stage1_accept": 0,
            "stage1_defer": 0,
            "stage1_reject": 0,
            "stage1_penalized_edges": 0,
            "stage1_reject_penalized_edges": 0,
            "stage1_newborn_blocked": 0,
            "stage1_hard_blocked_cols": 0,
            "stage2_rewrite": 0,
            "stage2_defer": 0,
            "stage2_reject": 0,
            "stage2_rewrite_edges": 0,
            "stage2_newborn_blocked": 0,
            "stage2_hard_blocked_cols": 0,
            "newborn_blocked": 0,
            "hard_blocked_cols": 0,
        }

        self.laplace_calibrator = None
        self.laplace_calibrator_path = getattr(args, "laplace_calibrator", "") or ""
        if self.laplace_calibrator_path:
            self.laplace_calibrator = LaplaceAlphaRCalibrator.from_npz(self.laplace_calibrator_path)
        self.laplace_haca_checkpoint = None
        if self.laplace_haca_checkpoint_path:
            self.laplace_haca_checkpoint = HACAV1Checkpoint.from_npz(self.laplace_haca_checkpoint_path)

        self.fcaa_refiner = None
        self.fgas_refiner = None
        self.graph_assoc_refiner = None
        self.graph_assoc_commit_scorer = None
        self.owneralt_refiner = None
        self.fcaa_last_debug = {
            "triggered_tracks": 0,
            "refined_pairs": 0,
            "trigger_groups": 0,
            "changed_groups": 0,
            "trigger_rows": [],
        }
        self.fgas_last_debug = {
            "trigger_blocks": 0,
            "changed_blocks": 0,
            "rows_touched": 0,
            "edges_touched": 0,
            "row_nomatch_rows": 0,
            "controller_blocks": 0,
            "controller_forced_candidates": 0,
            "controller_forced_matches": 0,
            "controller_blocked_rows": 0,
            "controller_blocked_cols": 0,
            "controller_conflicts_dropped": 0,
            "controller_applied_forced_matches": 0,
            "controller_applied_blocked_rows": 0,
            "controller_applied_blocked_cols": 0,
        }
        self.owneralt_last_debug = {
            "candidate_detections": 0,
            "candidate_pairs": 0,
            "rewrites": 0,
            "owner_rows_released": 0,
            "alt_edges_reweighted": 0,
            "blocked_owner_reclaims": 0,
            "event_rows": [],
        }
        self.graph_assoc_last_debug = {
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
        }
        self.fcaa_enable = bool(getattr(args, "fcaa_enable", False))
        self.fgas_enable = bool(getattr(args, "fgas_enable", False))
        self.graph_assoc_enable = bool(getattr(args, "graph_assoc_enable", False))
        self.owneralt_enable = bool(getattr(args, "owneralt_competition_enable", False))
        self.graph_assoc_commit_checkpoint_path = str(getattr(args, "graph_assoc_commit_checkpoint", "") or "")
        if self.graph_assoc_commit_checkpoint_path and not osp.isabs(self.graph_assoc_commit_checkpoint_path):
            self.graph_assoc_commit_checkpoint_path = osp.join(_REPO_ROOT, self.graph_assoc_commit_checkpoint_path)
        self.graph_assoc_commit_replace_rules = bool(getattr(args, "graph_assoc_commit_replace_rules", False))
        self.graph_assoc_commit_gate_only = bool(getattr(args, "graph_assoc_commit_gate_only", False))
        self.graph_assoc_commit_score_margin = float(getattr(args, "graph_assoc_commit_score_margin", 0.0))
        self.graph_assoc_commit_device = str(getattr(args, "graph_assoc_commit_device", "") or "").strip()
        self.graph_assoc_commit_decision_mode = str(getattr(args, "graph_assoc_commit_decision_mode", "") or "").strip()
        decision_threshold = float(getattr(args, "graph_assoc_commit_threshold", float("nan")))
        self.graph_assoc_commit_threshold = None if math.isnan(decision_threshold) else float(decision_threshold)
        neutral_risk_weight = float(getattr(args, "graph_assoc_commit_neutral_risk_weight", float("nan")))
        self.graph_assoc_commit_neutral_risk_weight = None if math.isnan(neutral_risk_weight) else float(neutral_risk_weight)
        positive_threshold = float(getattr(args, "graph_assoc_commit_positive_threshold", float("nan")))
        self.graph_assoc_commit_positive_threshold = None if math.isnan(positive_threshold) else float(positive_threshold)
        neutral_threshold = float(getattr(args, "graph_assoc_commit_neutral_threshold", float("nan")))
        self.graph_assoc_commit_neutral_threshold = None if math.isnan(neutral_threshold) else float(neutral_threshold)
        safety_min_gain = float(getattr(args, "graph_assoc_commit_safety_min_gain", float("nan")))
        self.graph_assoc_commit_safety_min_gain = None if math.isnan(safety_min_gain) else float(safety_min_gain)
        safety_max_cost_delta = float(getattr(args, "graph_assoc_commit_safety_max_cost_delta", float("nan")))
        self.graph_assoc_commit_safety_max_cost_delta = None if math.isnan(safety_max_cost_delta) else float(safety_max_cost_delta)
        self.graph_assoc_commit_safety_require_reclaim_improve = bool(
            getattr(args, "graph_assoc_commit_safety_require_reclaim_improve", False)
        )
        self.graph_assoc_commit_safety_require_same_match_count = bool(
            getattr(args, "graph_assoc_commit_safety_require_same_match_count", False)
        )
        if not self.graph_assoc_commit_device:
            self.graph_assoc_commit_device = "cuda:0" if str(getattr(args, "device", "gpu")).lower() == "gpu" else "cpu"
        self.track_feat_history = 50
        if self.owneralt_enable and not (self.freq_gate or self.laplace_assoc or self.fcaa_enable or self.fgas_enable):
            # OwnerAlt standalone only uses the smoothed feature for appearance matching.
            self.track_feat_history = 1
        # Long-term recovery memory: keep removed tracks around for a bounded gap window so they can be
        # re-activated if the same identity re-enters after a long occlusion.
        self.reentry_memory_enable = bool(getattr(args, "reentry_memory_enable", False))
        self.reentry_memory_max_gap = max(
            1,
            int(getattr(args, "reentry_memory_max_gap", max(1, int(self.max_time_lost) * 2))),
        )
        self.reentry_memory_max_size = max(1, int(getattr(args, "reentry_memory_max_size", 256)))
        self.reentry_memory_min_similarity = float(getattr(args, "reentry_memory_min_similarity", 0.60))
        self.reentry_memory_confirm_streak = max(1, int(getattr(args, "reentry_memory_confirm_streak", 2)))
        self.reentry_memory_confirm_gap = max(1, int(getattr(args, "reentry_memory_confirm_gap", self.reentry_memory_confirm_streak)))
        self.reentry_memory_confirm_min_similarity = float(
            getattr(
                args,
                "reentry_memory_confirm_min_similarity",
                min(0.95, max(self.reentry_memory_min_similarity, self.reentry_memory_min_similarity + 0.05)),
            )
        )
        self.reentry_memory_min_det_score = float(getattr(args, "reentry_memory_min_det_score", 0.10))
        self.reentry_memory_app_weight = float(getattr(args, "reentry_memory_app_weight", 0.55))
        self.reentry_memory_iou_weight = float(getattr(args, "reentry_memory_iou_weight", 0.25))
        self.reentry_memory_score_weight = float(getattr(args, "reentry_memory_score_weight", 0.10))
        self.reentry_memory_gap_weight = float(getattr(args, "reentry_memory_gap_weight", 0.10))
        self.reentry_memory_use_low_score = bool(getattr(args, "reentry_memory_use_low_score", False))
        self.reentry_memory_compete_primary = bool(getattr(args, "reentry_memory_compete_primary", False))
        self.reentry_memory_stats = {
            "enabled": bool(self.reentry_memory_enable),
            "compete_primary": bool(self.reentry_memory_compete_primary),
            "frames": 0,
            "archive_size": 0,
            "candidate_tracks": 0,
            "candidate_detections": 0,
            "candidate_pairs": 0,
            "matches": 0,
            "pending_proposals": 0,
            "pending_updates": 0,
            "pending_confirmations": 0,
            "pending_resets": 0,
            "reactivated_tracks": 0,
            "competitive_reactivated_tracks": 0,
            "competitive_archive_candidates": 0,
            "competitive_primary_matches": 0,
            "pruned_tracks": 0,
        }
        self.reentry_memory_pending = {}
        self.removed_archive_retention = max(1, int(max(self.max_time_lost, self.reentry_memory_max_gap)))

        # TOS-Track (Track Occlusion Shadow): analysis-only + behavioral system
        self.tos_enable = bool(getattr(args, "tos_enable", False))
        self.tos_analysis_only = bool(getattr(args, "tos_analysis_only", False))
        self.tos_analysis_dir = str(getattr(args, "tos_analysis_dir", ""))
        self.tos_hold_buffer = max(0, int(getattr(args, "tos_hold_buffer", 30)))
        self.tos_newborn_delay = max(0, int(getattr(args, "tos_newborn_delay", 5)))
        self.tos_memory_frames = max(1, int(getattr(args, "tos_memory_frames", 150)))
        self.tos_reconnect_gap_max = max(1, int(getattr(args, "tos_reconnect_gap_max", 60)))
        self.tos_reconnect_min_similarity = float(getattr(args, "tos_reconnect_min_similarity", 0.70))
        self.tos_occlusion_thresh = float(getattr(args, "tos_occlusion_thresh", 0.5))
        self.tos_freeze_on_occlusion = bool(getattr(args, "tos_freeze_on_occlusion", False))
        self.tos_disable_reentry = bool(getattr(args, "tos_disable_reentry", False))
        self.tos_stats = {
            "enabled": bool(self.tos_enable),
            "analysis_only": bool(self.tos_analysis_only),
            "frames": 0,
            "active_tracks": 0,
            "shadow_tracks": 0,
            "occluded_tracks": 0,
            "reconnected_tracks": 0,
            "held_newborns": 0,
            "reconnect_attempts": 0,
            "reconnect_successes": 0,
        }
        # Per-frame analysis rows (drained after each sequence)
        self._tos_analysis_rows = []
        self._tos_seq_writer = None

        # If tos_freeze_on_occlusion is set, auto-enable tos
        if self.tos_freeze_on_occlusion:
            self.tos_enable = True

        # Disable reentry memory when TOS is active (for isolation in v0)
        if self.tos_enable and self.tos_disable_reentry:
            self.reentry_memory_enable = False
            self.reentry_engine_enable = False

        # Reentry Query Engine: database-style identity query processing
        self.reentry_engine_enable = bool(getattr(args, "reentry_engine_enable", False))
        self.reentry_engine: "ReentryQueryEngine | None" = None
        if self.reentry_engine_enable and ReentryQueryEngine is not None:
            self.reentry_engine = ReentryQueryEngine(
                max_gap=self.reentry_memory_max_gap,
                max_size=self.reentry_memory_max_size,
                hilbert_order=int(getattr(args, "reentry_engine_hilbert_order", 8)),
                brute_force_threshold=int(getattr(args, "reentry_engine_bf_threshold", 50)),
                confirm_streak=self.reentry_memory_confirm_streak,
                confirm_gap=self.reentry_memory_confirm_gap,
                confirm_min_similarity=self.reentry_memory_confirm_min_similarity,
                app_weight=self.reentry_memory_app_weight,
                iou_weight=self.reentry_memory_iou_weight,
                score_weight=self.reentry_memory_score_weight,
                gap_weight=self.reentry_memory_gap_weight,
                min_similarity=self.reentry_memory_min_similarity,
                min_det_score=self.reentry_memory_min_det_score,
                spatial_radius=int(getattr(args, "reentry_engine_spatial_radius", 2)),
                max_spatial_radius=int(getattr(args, "reentry_engine_max_spatial_radius", 4)),
                short_gap_threshold=int(getattr(args, "reentry_engine_short_gap_threshold", 0)),
                num_prototypes=int(getattr(args, "reentry_engine_num_prototypes", 1)),
                recent_score_margin=float(getattr(args, "reentry_engine_recent_score_margin", 0.0)),
                recent_min_exit_frame_advantage=int(getattr(args, "reentry_engine_recent_min_exit_frame_advantage", 0)),
            )
            if bool(getattr(args, "reentry_engine_dump_matches", False)):
                self.reentry_engine.enable_match_dump()
        self.bc_lost_track_promote = bool(getattr(args, "bc_lost_track_promote", False))
        self.bc_max_gap = int(getattr(args, "bc_lost_track_max_gap", 30))
        self.bc_appearance_thresh = float(getattr(args, "bc_appearance_thresh", 0.45))
        self.bc_cost_margin = float(getattr(args, "bc_cost_margin", 0.15))
        self.bc_promotion_weight = float(getattr(args, "bc_promotion_weight", 0.3))
        self.bc_stats = {
            "enabled": self.bc_lost_track_promote,
            "frames": 0,
            "promotions": 0,
            "lost_tracks_considered": 0,
        }
        self._bc_trace_log: list[dict] = []
        self.fcaa_stats = {
            "enabled": bool(self.fcaa_enable),
            "trigger_mode": str(getattr(args, "fcaa_trigger_mode", "row_margin") or "row_margin"),
            "trigger_margin": float(getattr(args, "fcaa_trigger_margin", 0.05)),
            "lambda_weight": float(getattr(args, "fcaa_lambda", 0.3)),
            "top_k": int(getattr(args, "fcaa_topk", 3)),
            "scorer_mode": "",
            "frames": 0,
            "frames_with_trigger_groups": 0,
            "trigger_groups": 0,
            "changed_groups": 0,
            "unchanged_groups": 0,
            "triggered_tracks": 0,
            "refined_pairs": 0,
        }
        self.fgas_stats = {
            "enabled": bool(self.fgas_enable),
            "trigger_blocks": 0,
            "changed_blocks": 0,
            "rows_touched": 0,
            "edges_touched": 0,
            "row_nomatch_rows": 0,
            "frames": 0,
            "frames_with_trigger_blocks": 0,
            "blend_weight": float(getattr(args, "fgas_blend_weight", 0.5)),
            "assignment_mode": str(getattr(args, "fgas_assignment_mode", "blend")),
            "row_nomatch_weight": float(getattr(args, "fgas_row_nomatch_weight", 0.0)),
            "top_k": int(getattr(args, "fgas_topk", 5)),
            "max_rows": int(getattr(args, "fgas_max_rows", 3)),
            "max_cols": int(getattr(args, "fgas_max_cols", 3)),
            "controller_enable": bool(getattr(args, "fgas_controller_enable", False)),
            "controller_edge_thresh": float(getattr(args, "fgas_controller_edge_thresh", 0.6)),
            "controller_row_defer_thresh": float(getattr(args, "fgas_controller_row_defer_thresh", 0.6)),
            "controller_col_newborn_thresh": float(getattr(args, "fgas_controller_col_newborn_thresh", 0.6)),
            "controller_margin_thresh": float(getattr(args, "fgas_controller_margin_thresh", 0.05)),
            "controller_ambiguity_margin": float(getattr(args, "fgas_controller_ambiguity_margin", 0.05)),
            "controller_blocks": 0,
            "controller_forced_candidates": 0,
            "controller_forced_matches": 0,
            "controller_blocked_rows": 0,
            "controller_blocked_cols": 0,
            "controller_conflicts_dropped": 0,
            "controller_applied_forced_matches": 0,
            "controller_applied_blocked_rows": 0,
            "controller_applied_blocked_cols": 0,
            "frames_with_controller_actions": 0,
        }
        if self.fcaa_enable and self.fgas_enable:
            raise ValueError("FCAA and FGAS cannot be enabled at the same time.")
        if self.graph_assoc_enable and self.owneralt_enable:
            raise ValueError("Graph association and OwnerAlt cannot be enabled at the same time.")
        if self.graph_assoc_enable and (self.freq_gate or self.laplace_assoc or self.fcaa_enable or self.fgas_enable):
            raise ValueError("Graph association must run as a standalone primary-association rewrite and cannot be mixed with freq-gate, Laplace, FCAA, or FGAS.")
        if self.owneralt_enable and (self.freq_gate or self.laplace_assoc or self.fcaa_enable or self.fgas_enable):
            raise ValueError("OwnerAlt competition must run as a standalone primary-association rewrite and cannot be mixed with freq-gate, Laplace, FCAA, or FGAS.")
        if self.graph_assoc_commit_replace_rules and self.graph_assoc_commit_gate_only:
            raise ValueError("graph_assoc_commit_replace_rules and graph_assoc_commit_gate_only cannot be enabled at the same time")
        if self.graph_assoc_commit_replace_rules and not self.graph_assoc_commit_checkpoint_path:
            raise ValueError("graph_assoc_commit_replace_rules requires --graph-assoc-commit-checkpoint")
        if self.graph_assoc_commit_gate_only and not self.graph_assoc_commit_checkpoint_path:
            raise ValueError("graph_assoc_commit_gate_only requires --graph-assoc-commit-checkpoint")
        if self.graph_assoc_enable and self.graph_assoc_commit_checkpoint_path:
            use_competition_scorer = (
                checkpoint_looks_like_graph_assoc_competition is not None
                and bool(checkpoint_looks_like_graph_assoc_competition(self.graph_assoc_commit_checkpoint_path))
            )
            use_gate_scorer = (
                checkpoint_looks_like_graph_assoc_gate is not None
                and bool(checkpoint_looks_like_graph_assoc_gate(self.graph_assoc_commit_checkpoint_path))
            )
            if use_competition_scorer:
                if GraphAssocCompetitionScorer is None:
                    raise ImportError("Failed to import GraphAssocCompetitionScorer for graph-assoc competition runtime.")
                self.graph_assoc_commit_scorer = GraphAssocCompetitionScorer(
                    self.graph_assoc_commit_checkpoint_path,
                    device=self.graph_assoc_commit_device,
                    decision_mode=self.graph_assoc_commit_decision_mode,
                    threshold=self.graph_assoc_commit_threshold,
                    neutral_risk_weight=self.graph_assoc_commit_neutral_risk_weight,
                    positive_threshold=self.graph_assoc_commit_positive_threshold,
                    neutral_threshold=self.graph_assoc_commit_neutral_threshold,
                )
            elif use_gate_scorer:
                if GraphAssocGateScorer is None:
                    raise ImportError("Failed to import GraphAssocGateScorer for graph-assoc learned gate runtime.")
                self.graph_assoc_commit_scorer = GraphAssocGateScorer(
                    self.graph_assoc_commit_checkpoint_path,
                    device=self.graph_assoc_commit_device,
                    decision_mode=self.graph_assoc_commit_decision_mode,
                    threshold=self.graph_assoc_commit_threshold,
                    neutral_risk_weight=self.graph_assoc_commit_neutral_risk_weight,
                    positive_threshold=self.graph_assoc_commit_positive_threshold,
                    neutral_threshold=self.graph_assoc_commit_neutral_threshold,
                )
            else:
                if GraphAssocCommitScorer is None:
                    raise ImportError("Failed to import GraphAssocCommitScorer for graph-assoc learned commit runtime.")
                self.graph_assoc_commit_scorer = GraphAssocCommitScorer(
                    self.graph_assoc_commit_checkpoint_path,
                    device=self.graph_assoc_commit_device,
                    decision_mode=self.graph_assoc_commit_decision_mode,
                    threshold=self.graph_assoc_commit_threshold,
                    neutral_risk_weight=self.graph_assoc_commit_neutral_risk_weight,
                    positive_threshold=self.graph_assoc_commit_positive_threshold,
                    neutral_threshold=self.graph_assoc_commit_neutral_threshold,
                )
        self.fcaa_trigger_rows = []
        if self.fcaa_enable:
            if not args.with_reid:
                raise ValueError("FCAA requires --with-reid because it rescales appearance association.")
            if FCAAConfig is None or FCAARefiner is None:
                raise ImportError("FCAA modules are unavailable but --fcaa-enable was requested.")
            self.fcaa_refiner = FCAARefiner(
                FCAAConfig(
                    enabled=True,
                    scorer_checkpoint=getattr(args, "fcaa_scorer_checkpoint", "") or "",
                    trigger_mode=str(getattr(args, "fcaa_trigger_mode", "row_margin") or "row_margin"),
                    trigger_margin=float(getattr(args, "fcaa_trigger_margin", 0.05)),
                    lambda_weight=float(getattr(args, "fcaa_lambda", 0.3)),
                    top_k=int(getattr(args, "fcaa_topk", 3)),
                    appearance_thresh=float(self.appearance_thresh),
                    crop_height=int(getattr(args, "fcaa_crop_height", 128)),
                    crop_width=int(getattr(args, "fcaa_crop_width", 64)),
                    device=str(getattr(args, "device", "cpu")),
                )
            )
            self.fcaa_stats["scorer_mode"] = str(getattr(self.fcaa_refiner, "mode", ""))
        if self.fgas_enable:
            if not args.with_reid:
                raise ValueError("FGAS requires --with-reid.")
            if FGASConfig is None or FGASBlockRefiner is None:
                raise ImportError("FGAS modules are unavailable but --fgas-enable was requested.")
            self.fgas_refiner = FGASBlockRefiner(
                FGASConfig(
                    enabled=True,
                    resolver_checkpoint=getattr(args, "fgas_resolver_checkpoint", "") or "",
                    top_k=int(getattr(args, "fgas_topk", 5)),
                    proximity_thresh=float(self.proximity_thresh),
                    appearance_thresh=float(self.appearance_thresh),
                    max_rows=int(getattr(args, "fgas_max_rows", 3)),
                    max_cols=int(getattr(args, "fgas_max_cols", 3)),
                    crop_height=int(getattr(args, "fgas_crop_height", 128)),
                    crop_width=int(getattr(args, "fgas_crop_width", 64)),
                    blend_weight=float(getattr(args, "fgas_blend_weight", 0.5)),
                    assignment_mode=str(getattr(args, "fgas_assignment_mode", "blend")),
                    row_nomatch_weight=float(getattr(args, "fgas_row_nomatch_weight", 0.0)),
                    controller_enable=bool(getattr(args, "fgas_controller_enable", False)),
                    controller_edge_thresh=float(getattr(args, "fgas_controller_edge_thresh", 0.6)),
                    controller_row_defer_thresh=float(getattr(args, "fgas_controller_row_defer_thresh", 0.6)),
                    controller_col_newborn_thresh=float(getattr(args, "fgas_controller_col_newborn_thresh", 0.6)),
                    controller_margin_thresh=float(getattr(args, "fgas_controller_margin_thresh", 0.05)),
                    controller_ambiguity_margin=float(getattr(args, "fgas_controller_ambiguity_margin", 0.05)),
                    device=str(getattr(args, "device", "cpu")),
                )
            )
        if self.owneralt_enable:
            self.owneralt_refiner = OwnerAltCompetitionRefiner(
                OwnerAltCompetitionConfig(
                    enabled=True,
                    min_time_since_update=int(getattr(args, "owneralt_competition_min_time_since_update", 2)),
                    max_time_since_update=int(getattr(args, "owneralt_competition_max_time_since_update", 8)),
                    min_tracklet_len=int(getattr(args, "owneralt_competition_min_tracklet_len", 20)),
                    min_box_iou=float(getattr(args, "owneralt_competition_min_box_iou", 0.75)),
                    gap1_min_box_iou=float(getattr(args, "owneralt_competition_gap1_min_box_iou", -1.0)),
                    owner_max_tracklet_len=int(getattr(args, "owneralt_competition_owner_max_tracklet_len", 8)),
                    owner_alt_det_min_score=float(getattr(args, "owneralt_competition_owner_alt_det_min_score", 0.0)),
                    owner_alt_det_min_box_iou=float(getattr(args, "owneralt_competition_owner_alt_det_min_box_iou", 0.0)),
                    gap1_owner_alt_det_min_box_iou=float(
                        getattr(args, "owneralt_competition_gap1_owner_alt_det_min_box_iou", -1.0)
                    ),
                    max_owner_edge_deficit=float(getattr(args, "owneralt_competition_max_owner_edge_deficit", 0.10)),
                    gap1_max_owner_edge_deficit=float(
                        getattr(args, "owneralt_competition_gap1_max_owner_edge_deficit", -1.0)
                    ),
                    evidence_mode=str(getattr(args, "owneralt_competition_evidence_mode", "legacy")),
                    max_joint_penalty=float(getattr(args, "owneralt_competition_max_joint_penalty", -1.0)),
                    gap1_max_joint_penalty=float(
                        getattr(args, "owneralt_competition_gap1_max_joint_penalty", -1.0)
                    ),
                    owner_alt_bonus=float(getattr(args, "owneralt_competition_owner_alt_bonus", 0.10)),
                    block_owner_on_reclaim=bool(getattr(args, "owneralt_competition_block_owner_on_reclaim", False)),
                )
            )
        if self.graph_assoc_enable:
            self.graph_assoc_refiner = LocalGraphReassocRefiner(
                LocalGraphReassocConfig(
                    enabled=True,
                    dump_candidate_rows=bool(getattr(args, "graph_assoc_dump_candidate_rows", False)),
                    allow_col_only_blocks=bool(getattr(args, "graph_assoc_allow_col_only_blocks", True)),
                    require_row_involved_strict_reclaim=bool(getattr(args, "graph_assoc_require_row_involved_strict_reclaim", False)),
                    protect_young_active_rows=bool(getattr(args, "graph_assoc_protect_young_active_rows", False)),
                    top_k=int(getattr(args, "graph_assoc_top_k", 3)),
                    max_rows=int(getattr(args, "graph_assoc_max_rows", 4)),
                    max_cols=int(getattr(args, "graph_assoc_max_cols", 4)),
                    row_margin=float(getattr(args, "graph_assoc_row_margin", 0.03)),
                    col_margin=float(getattr(args, "graph_assoc_col_margin", 0.03)),
                    min_reclaim_time_since_update=int(getattr(args, "graph_assoc_min_reclaim_time_since_update", 1)),
                    max_reclaim_time_since_update=int(getattr(args, "graph_assoc_max_reclaim_time_since_update", 8)),
                    min_reclaim_tracklet_len=int(getattr(args, "graph_assoc_min_reclaim_tracklet_len", 20)),
                    recent_owner_max_time_since_update=int(getattr(args, "graph_assoc_recent_owner_max_time_since_update", 1)),
                    recent_owner_max_tracklet_len=int(getattr(args, "graph_assoc_recent_owner_max_tracklet_len", 8)),
                    young_active_max_time_since_update=int(getattr(args, "graph_assoc_young_active_max_time_since_update", 1)),
                    young_active_max_tracklet_len=int(getattr(args, "graph_assoc_young_active_max_tracklet_len", 20)),
                    young_active_min_reclaim_gap=int(getattr(args, "graph_assoc_young_active_min_reclaim_gap", 2)),
                    young_active_max_cost_delta=float(getattr(args, "graph_assoc_young_active_max_cost_delta", -1.0)),
                    protect_stale_lost_owner_rows=bool(getattr(args, "graph_assoc_protect_stale_lost_owner_rows", False)),
                    stale_lost_owner_min_time_since_update=int(getattr(args, "graph_assoc_stale_lost_owner_min_time_since_update", 9)),
                    stale_lost_owner_min_tracklet_len=int(getattr(args, "graph_assoc_stale_lost_owner_min_tracklet_len", 100)),
                    stale_lost_owner_active_max_time_since_update=int(getattr(args, "graph_assoc_stale_lost_owner_active_max_time_since_update", 1)),
                    stale_lost_owner_min_introduced_edge_utility=float(getattr(args, "graph_assoc_stale_lost_owner_min_introduced_edge_utility", 0.0)),
                    min_box_iou=float(getattr(args, "graph_assoc_min_box_iou", 0.6)),
                    reclaim_bonus=float(getattr(args, "graph_assoc_reclaim_bonus", 0.08)),
                    recent_owner_penalty=float(getattr(args, "graph_assoc_recent_owner_penalty", 0.05)),
                    iou_bonus=float(getattr(args, "graph_assoc_iou_bonus", 0.04)),
                    score_bonus=float(getattr(args, "graph_assoc_score_bonus", 0.02)),
                    min_assignment_gain=float(getattr(args, "graph_assoc_min_assignment_gain", 0.01)),
                    max_cost_delta=float(getattr(args, "graph_assoc_max_cost_delta", 0.05)),
                    row_involved_min_assignment_gain=float(getattr(args, "graph_assoc_row_involved_min_assignment_gain", 0.01)),
                    col_only_min_assignment_gain=float(getattr(args, "graph_assoc_col_only_min_assignment_gain", 0.01)),
                    col_only_max_cost_delta=float(getattr(args, "graph_assoc_col_only_max_cost_delta", 0.05)),
                    force_match_cost=float(getattr(args, "graph_assoc_force_match_cost", 0.0)),
                    require_same_match_count=not bool(getattr(args, "graph_assoc_allow_match_count_drop", False)),
                    candidate_rerank_top_k=int(getattr(args, "graph_assoc_candidate_rerank_top_k", 6)),
                    learned_commit_rerank_candidates=bool(getattr(args, "graph_assoc_learned_commit_rerank_candidates", False)),
                    learned_commit_scorer=self.graph_assoc_commit_scorer,
                    learned_commit_replace_rules=self.graph_assoc_commit_replace_rules,
                    learned_commit_gate_only=self.graph_assoc_commit_gate_only,
                    learned_commit_score_margin=self.graph_assoc_commit_score_margin,
                    learned_commit_safety_min_gain=self.graph_assoc_commit_safety_min_gain,
                    learned_commit_safety_max_cost_delta=self.graph_assoc_commit_safety_max_cost_delta,
                    learned_commit_safety_require_reclaim_improve=self.graph_assoc_commit_safety_require_reclaim_improve,
                    learned_commit_safety_require_same_match_count=self.graph_assoc_commit_safety_require_same_match_count,
                )
            )

        if self.laplace_assoc_mode not in {"auto", "heuristic", "current_learned", "haca_v1", "haca_v2", "haca_v3"}:
            raise ValueError(f"Unsupported laplace_assoc_mode: {self.laplace_assoc_mode}")
        if self.laplace_assoc_mode == "current_learned" and self.laplace_calibrator is None:
            raise ValueError("laplace_assoc_mode=current_learned requires --laplace-calibrator")
        if self.laplace_assoc_mode in {"haca_v1", "haca_v2", "haca_v3"} and self.laplace_haca_checkpoint is None:
            raise ValueError(f"laplace_assoc_mode={self.laplace_assoc_mode} requires --laplace-haca-checkpoint")
        if self.laplace_assoc_mode in {"haca_v1", "haca_v2", "haca_v3"} and self.laplace_no_reliability:
            raise ValueError("HACA association does not support --laplace-no-reliability")
        if self.tcgau_enable:
            if not bool(getattr(args, "with_reid", False)):
                raise ValueError("TCGAU requires --with-reid because it gates appearance memory updates.")
            if not self.laplace_assoc:
                raise ValueError("TCGAU requires --laplace-assoc on the HACA carrier.")
            if self.laplace_assoc_mode not in {"auto", "haca_v1", "haca_v2", "haca_v3"}:
                raise ValueError(
                    f"TCGAU requires a HACA-capable laplace_assoc_mode; got {self.laplace_assoc_mode}."
                )
            if self.laplace_assoc_mode == "auto" and self.laplace_haca_checkpoint is None:
                raise ValueError("TCGAU with laplace_assoc_mode=auto requires --laplace-haca-checkpoint.")
        if self.rgsa_enable:
            if not bool(getattr(args, "with_reid", False)):
                raise ValueError("RGSA runtime requires --with-reid because it consumes HACA appearance-time features.")
            if not self.laplace_assoc:
                raise ValueError("RGSA runtime requires --laplace-assoc on the HACA carrier.")
            if Stage1DeferralHead is None or Stage2RecoveryHead is None:
                raise ImportError("RGSA runtime heads are unavailable but --rgsa-enable was requested.")
            if not self.rgsa_stage1_checkpoint_path:
                raise ValueError("RGSA runtime requires --rgsa-stage1-checkpoint")
            if not osp.isabs(self.rgsa_stage1_checkpoint_path):
                self.rgsa_stage1_checkpoint_path = osp.join(_REPO_ROOT, self.rgsa_stage1_checkpoint_path)
            self.rgsa_stage1_head = Stage1DeferralHead.from_checkpoint(
                self.rgsa_stage1_checkpoint_path,
                device=self.rgsa_device,
            )
            self.rgsa_stage1_head.eval()
            if self.rgsa_stage2_checkpoint_path:
                if not osp.isabs(self.rgsa_stage2_checkpoint_path):
                    self.rgsa_stage2_checkpoint_path = osp.join(_REPO_ROOT, self.rgsa_stage2_checkpoint_path)
                self.rgsa_stage2_head = Stage2RecoveryHead.from_checkpoint(
                    self.rgsa_stage2_checkpoint_path,
                    device=self.rgsa_device,
                )
                self.rgsa_stage2_head.eval()
            # Initialize verifier if requested
            if self.rgsa_verifier_mode == "heuristic":
                self.rgsa_verifier = HeuristicVerifier()
            elif self.rgsa_verifier_mode == "heuristic_tight":
                self.rgsa_verifier = HeuristicVerifier(
                    rule1_s_final_min=0.9, rule1_margin_min=0.15, rule1_entropy_max=0.3,
                    rule2_activation_min=0.6, rule2_margin_min=0.1, rule2_bg_prob_max=0.2,
                    rule3_beta_product_min=0.4, rule3_margin_min=0.1,
                )
            # Load CCRC Platt calibrator
            if self.ccrc_enable and self.ccrc_platt_checkpoint_path:
                import torch as _torch
                ckpt_path = self.ccrc_platt_checkpoint_path
                if not osp.isabs(ckpt_path):
                    ckpt_path = osp.join(_REPO_ROOT, ckpt_path)
                ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
                sd = ckpt.get("state_dict", {})
                self.ccrc_platt_a = float(sd["a"].item()) if "a" in sd else float(ckpt.get("metadata", {}).get("a", 1.0))
                self.ccrc_platt_b = float(sd["b"].item()) if "b" in sd else float(ckpt.get("metadata", {}).get("b", 0.0))

    def __del__(self):
        if getattr(self, "laplace_analysis", None) is not None:
            self.laplace_analysis.close()
        self._flush_rgsa_frame_rows()

    def _flush_rgsa_frame_rows(self):
        if not getattr(self, "_rgsa_frame_rows", None):
            return
        out_path = getattr(self, "_rgsa_pairbank_path", None)
        if not out_path:
            return
        import csv as _csv
        if self._rgsa_pairbank_fieldnames is None:
            self._rgsa_pairbank_fieldnames = list(self._rgsa_frame_rows[0].keys())
        write_header = not osp.isfile(out_path)
        with open(out_path, "a", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=self._rgsa_pairbank_fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(self._rgsa_frame_rows)
        self._rgsa_frame_rows.clear()

    def _promote_lost_tracks(self, dists, strack_pool, detections, emb_dists):
        """Branch-consistency v2: reduce cost for recently-lost tracks with strong appearance match."""
        if not self.bc_lost_track_promote or dists.size == 0:
            return dists
        if emb_dists is None:
            return dists  # no appearance features available
        self.bc_stats["frames"] += 1
        num_tracks = len(strack_pool)
        num_dets = len(detections)
        for row in range(num_tracks):
            track = strack_pool[row]
            if track.state != TrackState.Lost:
                continue
            time_since = self.frame_id - int(getattr(track, "end_frame", self.frame_id))
            if time_since < 1 or time_since > self.bc_max_gap:
                continue
            self.bc_stats["lost_tracks_considered"] += 1
            for col in range(num_dets):
                if emb_dists[row, col] >= self.bc_appearance_thresh:
                    continue
                best_other = 1.0
                for r in range(num_tracks):
                    if r != row and dists[r, col] < best_other:
                        best_other = dists[r, col]
                cost_delta = dists[row, col] - best_other
                if cost_delta > self.bc_cost_margin:
                    continue
                dists[row, col] *= (1.0 - self.bc_promotion_weight)
                self.bc_stats["promotions"] += 1
                self._bc_trace_log.append({
                    "frame": int(self.frame_id),
                    "lost_track_id": int(getattr(track, "track_id", -1)),
                    "det_index": int(col),
                    "time_since_update": int(time_since),
                    "emb_dist": round(float(emb_dists[row, col]), 6),
                    "original_cost": round(float(dists[row, col] / (1.0 - self.bc_promotion_weight)), 6),
                    "promoted_cost": round(float(dists[row, col]), 6),
                    "best_other_cost": round(float(best_other), 6),
                    "cost_delta": round(float(cost_delta), 6),
                })
        return dists

    def drain_bc_trace_log(self):
        rows = list(self._bc_trace_log)
        self._bc_trace_log.clear()
        return rows

    def _apply_fgas_controller(self, dists, thresh, controller_actions):
        empty_matches = np.empty((0, 2), dtype=int)
        base_debug = {
            "controller_applied_forced_matches": 0,
            "controller_applied_blocked_rows": 0,
            "controller_applied_blocked_cols": 0,
        }
        if dists.size == 0:
            return empty_matches, tuple(range(dists.shape[0])), tuple(range(dists.shape[1])), base_debug
        if controller_actions is None:
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=thresh)
            return matches, u_track, u_detection, base_debug

        raw_forced = list(getattr(controller_actions, "forced_matches", []) or [])
        blocked_rows = {int(v) for v in (getattr(controller_actions, "blocked_rows", []) or [])}
        blocked_cols = {int(v) for v in (getattr(controller_actions, "blocked_cols", []) or [])}
        forced_matches = []
        used_rows = set()
        used_cols = set()
        for row_idx, col_idx in raw_forced:
            row_idx = int(row_idx)
            col_idx = int(col_idx)
            if row_idx < 0 or row_idx >= dists.shape[0] or col_idx < 0 or col_idx >= dists.shape[1]:
                continue
            if row_idx in blocked_rows or col_idx in blocked_cols:
                continue
            if row_idx in used_rows or col_idx in used_cols:
                continue
            if not np.isfinite(dists[row_idx, col_idx]) or float(dists[row_idx, col_idx]) > float(thresh):
                continue
            forced_matches.append((row_idx, col_idx))
            used_rows.add(row_idx)
            used_cols.add(col_idx)

        blocked_rows.difference_update(used_rows)
        blocked_cols.difference_update(used_cols)
        remain_rows = [idx for idx in range(dists.shape[0]) if idx not in used_rows and idx not in blocked_rows]
        remain_cols = [idx for idx in range(dists.shape[1]) if idx not in used_cols and idx not in blocked_cols]

        hungarian_matches = []
        unmatched_rows = set(blocked_rows)
        unmatched_cols = set(blocked_cols)
        if remain_rows and remain_cols:
            sub_dists = dists[np.ix_(remain_rows, remain_cols)]
            sub_matches, sub_u_rows, sub_u_cols = matching.linear_assignment(sub_dists, thresh=thresh)
            hungarian_matches = [(int(remain_rows[r]), int(remain_cols[c])) for r, c in sub_matches.tolist()]
            unmatched_rows.update(int(remain_rows[idx]) for idx in sub_u_rows)
            unmatched_cols.update(int(remain_cols[idx]) for idx in sub_u_cols)
        else:
            unmatched_rows.update(int(idx) for idx in remain_rows)
            unmatched_cols.update(int(idx) for idx in remain_cols)

        matches = forced_matches + hungarian_matches
        matches_array = np.asarray(matches, dtype=int) if matches else empty_matches
        debug = {
            "controller_applied_forced_matches": int(len(forced_matches)),
            "controller_applied_blocked_rows": int(len(blocked_rows)),
            "controller_applied_blocked_cols": int(len(blocked_cols)),
        }
        return matches_array, np.asarray(sorted(unmatched_rows), dtype=int), np.asarray(sorted(unmatched_cols), dtype=int), debug

    def update(self, output_results, img):
        self.frame_id += 1
        if self.reentry_memory_enable:
            self.reentry_memory_stats["frames"] += 1
        if self.tcgau_enable:
            self.tcgau_stats["frames"] = int(self.tcgau_stats.get("frames", 0)) + 1
        self.fcaa_stats["frames"] = int(self.fcaa_stats["frames"]) + 1
        self.fgas_stats["frames"] = int(self.fgas_stats["frames"]) + 1
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
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f, feat_history=self.track_feat_history) for
                              (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, feat_history=self.track_feat_history) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        if self.fcaa_refiner is not None and self.fcaa_refiner.is_active() and len(detections) > 0:
            det_descs = self.fcaa_refiner.extract_detection_descriptors(img, detections)
            for det, desc in zip(detections, det_descs):
                det.fcaa_band_desc = desc
                det.update_fcaa_bands(desc, momentum=0.0)
        if self.fgas_refiner is not None and self.fgas_refiner.is_active() and self.fgas_refiner.uses_frequency() and len(detections) > 0:
            det_descs = self.fgas_refiner.extract_detection_descriptors(img, detections)
            for det, desc in zip(detections, det_descs):
                det.fcaa_band_desc = desc
                det.update_fcaa_bands(desc, momentum=0.0)

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

        bc_raw_emb_dists = None
        if self.args.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
            bc_raw_emb_dists = emb_dists.copy()  # save before gating for bc promotion
            fgas_controller_actions = None
            if self.laplace_assoc:
                det_scores = np.array([det.score for det in detections], dtype=np.float32) if len(detections) > 0 else None
                track_gaps = np.array(
                    [max(0, int(self.frame_id) - int(getattr(t, "frame_id", self.frame_id))) for t in strack_pool],
                    dtype=np.float32,
                )
                return_debug = self.laplace_analysis is not None or self.rgsa_dump_dir or self.rgsa_enable or self.tcgau_enable or self.tos_enable
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
                # RGSA oracle dump: collect HACA debug per frame
                if self._rgsa_frame_rows is not None and laplace_debug is not None and use_haca_primary:
                    anchor_sim = laplace_debug.get("anchor_sim")
                    if anchor_sim is not None:
                        n_tracks, n_dets = anchor_sim.shape
                        anchor_z = laplace_debug.get("haca_anchor_z", np.zeros_like(anchor_sim))
                        anchor_margin = laplace_debug.get("haca_anchor_margin", np.zeros_like(anchor_sim))
                        anchor_rank = laplace_debug.get("haca_anchor_rank", np.zeros_like(anchor_sim))
                        hist_last = laplace_debug.get("haca_hist_last", np.zeros_like(anchor_sim))
                        hist_max = laplace_debug.get("haca_hist_max", np.zeros_like(anchor_sim))
                        hist_std = laplace_debug.get("haca_hist_std", np.zeros_like(anchor_sim))
                        stability = laplace_debug.get("stability")
                        if stability is None:
                            stability = np.exp(-np.maximum(hist_std, 0.0)).astype(np.float32)
                        else:
                            stability = np.asarray(stability, dtype=np.float32).reshape(-1, 1).repeat(n_dets, axis=1)
                        coherence = laplace_debug.get("coherence")
                        if coherence is None:
                            coherence = np.divide(
                                hist_last,
                                np.clip(hist_max, 1e-6, None),
                                out=hist_last.copy().astype(np.float32),
                                where=np.abs(hist_max) > 1e-6,
                            ).astype(np.float32)
                        else:
                            coherence = np.asarray(coherence, dtype=np.float32).reshape(-1, 1).repeat(n_dets, axis=1)
                        for det_idx in range(n_dets):
                            det = detections[det_idx]
                            det_score = float(getattr(det, "score", 0.0))
                            det_tlwh = getattr(det, "tlwh", None)
                            det_anchor = anchor_sim[:, det_idx]
                            det_rank_order = np.argsort(-det_anchor)
                            det_rank_lookup = {int(track_row): rank for rank, track_row in enumerate(det_rank_order.tolist())}
                            for t_idx in range(n_tracks):
                                track = strack_pool[t_idx]
                                track_id = int(getattr(track, "track_id", t_idx))
                                track_gap = max(0, int(self.frame_id) - int(getattr(track, "frame_id", self.frame_id)))
                                track_age = int(self.frame_id) - int(getattr(track, "start_frame", self.frame_id)) if hasattr(track, "start_frame") else 0
                                hist = getattr(track, "features", None)
                                hist_len = len(hist) if hist is not None else 0
                                hist_norm = min(1.0, float(hist_len) / max(float(self.laplace_min_history), 1.0))
                                gap_log1p = float(np.log1p(max(float(track_gap), 0.0)))
                                topk_rank = int(det_rank_lookup.get(int(t_idx), 0))
                                self._rgsa_frame_rows.append({
                                    "seq_name": str(getattr(self.args, "name", "")),
                                    "frame_id": int(self.frame_id),
                                    "det_id": det_idx,
                                    "track_id": track_id,
                                    "topk_rank": topk_rank,
                                    "anchor_sim": float(anchor_sim[t_idx, det_idx]),
                                    "spatial_sim": float(laplace_debug.get("spatial_sim", np.zeros_like(anchor_sim))[t_idx, det_idx]) if "spatial_sim" in laplace_debug else 0.0,
                                    "motion_sim": float(laplace_debug.get("motion_sim", np.zeros_like(anchor_sim))[t_idx, det_idx]) if "motion_sim" in laplace_debug else 0.0,
                                    "temp_sim": float(laplace_debug.get("haca_temp_sim", np.zeros_like(anchor_sim))[t_idx, det_idx]),
                                    "hist_last_sim": float(hist_last[t_idx, det_idx]),
                                    "hist_max_sim": float(hist_max[t_idx, det_idx]),
                                    "hist_std_sim": float(hist_std[t_idx, det_idx]),
                                    "gap_log1p": gap_log1p,
                                    "hist_norm": hist_norm,
                                    "stability": float(stability[t_idx, det_idx]),
                                    "coherence": float(coherence[t_idx, det_idx]),
                                    "anchor_z": float(anchor_z[t_idx, det_idx]),
                                    "anchor_margin": float(anchor_margin[t_idx, det_idx]),
                                    "anchor_rank": float(anchor_rank[t_idx, det_idx]),
                                    "s_final": float(laplace_debug.get("final_sim", anchor_sim)[t_idx, det_idx]),
                                    "activation": float(laplace_debug.get("haca_comp_active", np.zeros_like(anchor_sim))[t_idx, det_idx]) if "haca_comp_active" in laplace_debug else 0.0,
                                    "margin": float(laplace_debug.get("haca_comp_margin", np.zeros(n_dets))[det_idx]) if "haca_comp_margin" in laplace_debug else 0.0,
                                    "entropy": float(laplace_debug.get("haca_comp_entropy", np.zeros(n_dets))[det_idx]) if "haca_comp_entropy" in laplace_debug else 0.0,
                                    "bg_prob": float(laplace_debug.get("haca_background", np.zeros(n_dets))[det_idx]) if "haca_background" in laplace_debug else 0.0,
                                    "beta_pred": float(laplace_debug.get("haca_beta_pred", np.zeros_like(anchor_sim))[t_idx, det_idx]),
                                    "beta_hist": float(laplace_debug.get("haca_beta_hist", np.zeros_like(anchor_sim))[t_idx, det_idx]),
                                    "beta_ood": float(laplace_debug.get("haca_beta_ood", np.zeros_like(anchor_sim))[t_idx, det_idx]),
                                    "ood_score": float(laplace_debug.get("haca_ood_score", np.zeros_like(anchor_sim))[t_idx, det_idx]),
                                    "track_gap": track_gap,
                                    "track_age": track_age,
                                    "history_len": hist_len,
                                    "det_score": det_score,
                                    "det_tlwh": ",".join(f"{v:.1f}" for v in det_tlwh) if det_tlwh is not None else "",
                                    "track_tlwh": ",".join(f"{v:.1f}" for v in getattr(track, "tlwh", [0,0,0,0])),
                                })
                        self._flush_rgsa_frame_rows()
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
                if self.fgas_refiner is not None and self.fgas_refiner.is_active():
                    emb_dists, self.fgas_last_debug, fgas_controller_actions = self.fgas_refiner.refine_primary_cost(
                        track_pool=strack_pool,
                        detections=detections,
                        emb_dists=emb_dists,
                        raw_ious_dists=raw_ious_dists,
                        image=img,
                    )
                    frame_trigger_blocks = int(self.fgas_last_debug.get("trigger_blocks", 0))
                    if frame_trigger_blocks > 0:
                        self.fgas_stats["frames_with_trigger_blocks"] = int(self.fgas_stats["frames_with_trigger_blocks"]) + 1
                    self.fgas_stats["trigger_blocks"] = int(self.fgas_stats["trigger_blocks"]) + frame_trigger_blocks
                    self.fgas_stats["changed_blocks"] = int(self.fgas_stats["changed_blocks"]) + int(self.fgas_last_debug.get("changed_blocks", 0))
                    self.fgas_stats["rows_touched"] = int(self.fgas_stats["rows_touched"]) + int(self.fgas_last_debug.get("rows_touched", 0))
                    self.fgas_stats["edges_touched"] = int(self.fgas_stats["edges_touched"]) + int(self.fgas_last_debug.get("edges_touched", 0))
                    self.fgas_stats["row_nomatch_rows"] = int(self.fgas_stats["row_nomatch_rows"]) + int(self.fgas_last_debug.get("row_nomatch_rows", 0))
                    self.fgas_stats["controller_blocks"] = int(self.fgas_stats["controller_blocks"]) + int(self.fgas_last_debug.get("controller_blocks", 0))
                    self.fgas_stats["controller_forced_candidates"] = int(self.fgas_stats["controller_forced_candidates"]) + int(self.fgas_last_debug.get("controller_forced_candidates", 0))
                    self.fgas_stats["controller_forced_matches"] = int(self.fgas_stats["controller_forced_matches"]) + int(self.fgas_last_debug.get("controller_forced_matches", 0))
                    self.fgas_stats["controller_blocked_rows"] = int(self.fgas_stats["controller_blocked_rows"]) + int(self.fgas_last_debug.get("controller_blocked_rows", 0))
                    self.fgas_stats["controller_blocked_cols"] = int(self.fgas_stats["controller_blocked_cols"]) + int(self.fgas_last_debug.get("controller_blocked_cols", 0))
                    self.fgas_stats["controller_conflicts_dropped"] = int(self.fgas_stats["controller_conflicts_dropped"]) + int(self.fgas_last_debug.get("controller_conflicts_dropped", 0))
                elif self.fcaa_refiner is not None and self.fcaa_refiner.is_active():
                    emb_dists, self.fcaa_last_debug = self.fcaa_refiner.refine_embedding_cost(
                        track_pool=strack_pool,
                        detections=detections,
                        emb_dists=emb_dists,
                        raw_ious_dists=ious_dists,
                        ious_dists_mask=ious_dists_mask,
                        image=img,
                    )
                    frame_trigger_groups = int(self.fcaa_last_debug.get("trigger_groups", 0))
                    frame_changed_groups = int(self.fcaa_last_debug.get("changed_groups", 0))
                    if frame_trigger_groups > 0:
                        self.fcaa_stats["frames_with_trigger_groups"] = int(self.fcaa_stats["frames_with_trigger_groups"]) + 1
                    self.fcaa_stats["trigger_groups"] = int(self.fcaa_stats["trigger_groups"]) + frame_trigger_groups
                    self.fcaa_stats["changed_groups"] = int(self.fcaa_stats["changed_groups"]) + frame_changed_groups
                    self.fcaa_stats["unchanged_groups"] = int(self.fcaa_stats["unchanged_groups"]) + max(0, frame_trigger_groups - frame_changed_groups)
                    self.fcaa_stats["triggered_tracks"] = int(self.fcaa_stats["triggered_tracks"]) + int(self.fcaa_last_debug.get("triggered_tracks", 0))
                    self.fcaa_stats["refined_pairs"] = int(self.fcaa_stats["refined_pairs"]) + int(self.fcaa_last_debug.get("refined_pairs", 0))
                    for row in self.fcaa_last_debug.get("trigger_rows", []):
                        payload = dict(row)
                        payload["frame_id"] = int(self.frame_id)
                        self.fcaa_trigger_rows.append(payload)
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

        rgsa_newborn_blocked_det_ids = set()
        rgsa_hard_block_det_ids = set()

        if self.bc_lost_track_promote and len(strack_pool) > 0 and len(detections) > 0:
            dists = self._promote_lost_tracks(dists, strack_pool, detections, bc_raw_emb_dists if self.args.with_reid else None)

        if self.owneralt_refiner is not None and self.owneralt_refiner.is_active():
            dists, self.owneralt_last_debug = self.owneralt_refiner.refine_primary_cost(
                track_pool=strack_pool,
                detections=detections,
                dists=dists,
                raw_ious_dists=raw_ious_dists,
                frame_id=self.frame_id,
                match_thresh=float(self.args.match_thresh),
            )
        if self.graph_assoc_refiner is not None and self.graph_assoc_refiner.is_active():
            dists, self.graph_assoc_last_debug = self.graph_assoc_refiner.refine_primary_cost(
                track_pool=strack_pool,
                detections=detections,
                dists=dists,
                raw_ious_dists=raw_ious_dists,
                frame_id=self.frame_id,
                match_thresh=float(self.args.match_thresh),
            )
        dists, rgsa_newborn_blocked_det_ids, rgsa_hard_block_det_ids = self._apply_rgsa_primary_runtime(
            dists=dists,
            strack_pool=strack_pool,
            detections=detections,
            ious_dists_mask=ious_dists_mask,
            laplace_debug=laplace_debug if self.args.with_reid else None,
            use_haca_primary=use_haca_primary,
        )

        competitive_archive_start = len(strack_pool)
        competitive_archive_count = 0
        competitive_archive_map = {}
        if self.reentry_memory_enable and self.reentry_memory_compete_primary and len(detections) > 0 and len(self.removed_stracks) > 0:
            archive_tracks = []
            for track in self.removed_stracks:
                gap = int(self.frame_id) - int(getattr(track, "end_frame", self.frame_id))
                if gap < 1 or gap > self.reentry_memory_max_gap:
                    continue
                if getattr(track, "mean", None) is None or getattr(track, "covariance", None) is None:
                    continue
                archive_tracks.append(track)
            if archive_tracks:
                archive_cost = np.full((len(archive_tracks), len(detections)), 1.0, dtype=np.float32)
                for row, track in enumerate(archive_tracks):
                    gap = max(1, int(self.frame_id) - int(getattr(track, "end_frame", self.frame_id)))
                    gap_factor = math.exp(-float(gap) / float(max(self.reentry_memory_max_gap, 1)))
                    for col, det in enumerate(detections):
                        det_score = float(getattr(det, "score", 0.0))
                        if det_score < self.reentry_memory_min_det_score:
                            continue
                        app_sim = self._safe_cosine_similarity(getattr(track, "smooth_feat", None), getattr(det, "curr_feat", None))
                        try:
                            iou_sim = 1.0 - float(matching.iou_distance([track], [det])[0, 0])
                        except Exception:
                            iou_sim = 0.0
                        score = (
                            self.reentry_memory_app_weight * max(0.0, app_sim)
                            + self.reentry_memory_iou_weight * max(0.0, iou_sim)
                            + self.reentry_memory_score_weight * det_score
                            + self.reentry_memory_gap_weight * gap_factor
                        )
                        if score < self.reentry_memory_min_similarity:
                            continue
                        archive_cost[row, col] = float(np.clip(1.0 - score, 0.0, 1.0))
                if np.isfinite(archive_cost).any():
                    competitive_archive_count = int(len(archive_tracks))
                    competitive_archive_start = len(strack_pool)
                    strack_pool = list(strack_pool) + archive_tracks
                    competitive_archive_map = {
                        competitive_archive_start + int(idx): archive_tracks[int(idx)]
                        for idx in range(len(archive_tracks))
                    }
                    dists = np.concatenate([np.asarray(dists, dtype=np.float32), archive_cost], axis=0)
                    self.reentry_memory_stats["competitive_archive_candidates"] += int(competitive_archive_count)
        if rgsa_hard_block_det_ids:
            for det_id in rgsa_hard_block_det_ids:
                if 0 <= int(det_id) < dists.shape[1]:
                    dists[:, int(det_id)] = 1.0

        if (
            self.fgas_refiner is not None
            and self.fgas_refiner.is_active()
            and bool(getattr(self.fgas_refiner.config, "controller_enable", False))
        ):
            matches, u_track, u_detection, controller_debug = self._apply_fgas_controller(
                dists,
                thresh=self.args.match_thresh,
                controller_actions=fgas_controller_actions,
            )
            self.fgas_last_debug.update(controller_debug)
            self.fgas_stats["controller_applied_forced_matches"] = int(self.fgas_stats["controller_applied_forced_matches"]) + int(controller_debug.get("controller_applied_forced_matches", 0))
            self.fgas_stats["controller_applied_blocked_rows"] = int(self.fgas_stats["controller_applied_blocked_rows"]) + int(controller_debug.get("controller_applied_blocked_rows", 0))
            self.fgas_stats["controller_applied_blocked_cols"] = int(self.fgas_stats["controller_applied_blocked_cols"]) + int(controller_debug.get("controller_applied_blocked_cols", 0))
            if any(int(controller_debug.get(key, 0)) > 0 for key in controller_debug):
                self.fgas_stats["frames_with_controller_actions"] = int(self.fgas_stats["frames_with_controller_actions"]) + 1
        else:
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        # CCRC: Platt-calibrated commit/abstain on primary matches
        if self.ccrc_enable and self.ccrc_tau_commit > 0 and len(matches) > 0 and laplace_debug is not None:
            anchor_sim = laplace_debug.get("anchor_sim")
            if anchor_sim is not None:
                import math as _math
                ccrc_matches = []
                ccrc_abstained = []
                for itracked, idet in matches:
                    if itracked < anchor_sim.shape[0] and idet < anchor_sim.shape[1]:
                        s_final = float(anchor_sim[itracked, idet])
                    else:
                        s_final = 0.0
                    # Platt calibration: sigmoid(a * logit(s_final) + b)
                    s_clamped = max(min(s_final, 1 - 1e-6), 1e-6)
                    logit_s = _math.log(s_clamped / (1 - s_clamped))
                    p_correct = 1.0 / (1.0 + _math.exp(-(self.ccrc_platt_a * logit_s + self.ccrc_platt_b)))
                    if p_correct >= self.ccrc_tau_commit:
                        ccrc_matches.append((itracked, idet))
                        self.ccrc_stats["commits"] += 1
                    else:
                        ccrc_abstained.append((itracked, idet))
                        self.ccrc_stats["abstains"] += 1
                        # Return detection to unmatched pool for newborn/reentry
                        u_detection = list(u_detection) + [idet]
                        u_track = list(u_track) + [itracked]
                matches = ccrc_matches
                self.ccrc_stats["frames"] += 1
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

        competitive_recovered_tracks = []
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if competitive_archive_count > 0 and int(itracked) >= int(competitive_archive_start):
                archive_track = competitive_archive_map.get(int(itracked), track)
                pair_score = self._score_archive_detection_pair(archive_track, det)
                if pair_score is None:
                    self.reentry_memory_stats["competitive_primary_matches"] += 1
                    continue
                if self._update_pending_reentry(archive_track, det, pair_score, origin="competitive"):
                    self._commit_confirmed_reentry(archive_track, det, origin="competitive")
                    competitive_recovered_tracks.append(archive_track)
                    refind_stracks.append(archive_track)
                self.reentry_memory_stats["competitive_primary_matches"] += 1
                continue

            # TCGAU: compute update policy before update/re_activate
            tcgau_policy = self._compute_tcgau_policy(track, det, itracked, idet, laplace_debug if self.laplace_assoc else None)
            det.tcgau_update_mode = tcgau_policy["mode"]
            det.tcgau_alpha_override = tcgau_policy["alpha_override"]
            det.tcgau_append_history = tcgau_policy["append_history"]

            # TOS-Track: occlusion-triggered freeze
            if self.tos_enable and self.tos_freeze_on_occlusion:
                tos_occlusion = self._compute_tos_occlusion_score(track, det, itracked, idet, laplace_debug if self.laplace_assoc else None)
                det.tos_occlusion_score = tos_occlusion
                # Record on track for analysis
                track.tos_occlusion_score = tos_occlusion
                track.tos_last_seen = int(self.frame_id)
                if tos_occlusion >= self.tos_occlusion_thresh and det.tcgau_update_mode != "freeze":
                    # Override: freeze appearance during occlusion
                    det.tcgau_update_mode = "freeze"
                    det.tcgau_append_history = False
                    det.tcgau_alpha_override = None
                    self.tos_stats["occluded_tracks"] += 1

            if self.tos_enable and self.tos_analysis_only:
                # Record per-track per-frame analysis row
                track_len = int(getattr(track, "tracklet_len", 0))
                hist_len = int(len(getattr(track, "features", []) or []))
                gap = max(0, int(self.frame_id) - int(getattr(track, "frame_id", self.frame_id)))
                occ = float(getattr(track, "tos_occlusion_score", 0.0))
                shadow = int(getattr(track, "tos_is_shadow", False))
                shadow_dur = int(getattr(track, "tos_shadow_duration", 0))
                det_score = float(getattr(det, "score", 0.0))
                app_sim = float(tcgau_policy["app_sim"])
                pair_rel = float(tcgau_policy["pair_rel"])
                q_up = float(tcgau_policy["q_update"])
                mode = str(det.tcgau_update_mode)
                self._tos_analysis_rows.append({
                    "frame": int(self.frame_id),
                    "track_id": int(getattr(track, "track_id", -1)),
                    "det_id": int(idet),
                    "track_state": str(getattr(track, "state", "Unknown")),
                    "track_len": track_len,
                    "hist_len": hist_len,
                    "gap": gap,
                    "det_score": det_score,
                    "app_sim": round(app_sim, 4),
                    "pair_rel": round(pair_rel, 4),
                    "q_update": round(q_up, 4),
                    "update_mode": mode,
                    "tos_occlusion": round(occ, 4),
                    "tos_shadow": shadow,
                    "tos_shadow_dur": shadow_dur,
                    "haca_active": float(self._rgsa_matrix_value(laplace_debug, "haca_comp_active", itracked, idet, default=0.0)),
                    "haca_margin": float(self._rgsa_vector_value(laplace_debug, "haca_comp_margin", idet, default=0.0)),
                    "haca_entropy": float(self._rgsa_vector_value(laplace_debug, "haca_comp_entropy", idet, default=0.0)),
                    "haca_bg": float(self._rgsa_vector_value(laplace_debug, "haca_background", idet, default=0.0)),
                })

            if self.tcgau_enable:
                self.tcgau_stats["primary_matches"] += 1
                self.tcgau_stats[f"{tcgau_policy['mode']}_updates"] += 1
                self.tcgau_stats["avg_q_update"] += tcgau_policy["q_update"]
                self.tcgau_stats["avg_app_sim"] += tcgau_policy["app_sim"]
                self.tcgau_stats["avg_pair_rel"] += tcgau_policy["pair_rel"]
                if self.tcgau_log_pairs:
                    self.tcgau_pair_log.append({
                        "seq": str(getattr(self.args, "name", "")),
                        "frame": int(self.frame_id),
                        "track_id": int(getattr(track, "track_id", -1)),
                        "det_index": int(idet),
                        "mode": tcgau_policy["mode"],
                        "q_update": tcgau_policy["q_update"],
                        "app_sim": tcgau_policy["app_sim"],
                        "pair_rel": tcgau_policy["pair_rel"],
                        "stability": tcgau_policy["stability"],
                        "coherence": tcgau_policy["coherence"],
                        "margin": tcgau_policy["margin"],
                        "hist_norm": tcgau_policy["hist_norm"],
                        "track_gap": max(0, int(self.frame_id) - int(getattr(track, "frame_id", self.frame_id))),
                        "det_score": float(getattr(det, "score", 0.0)),
                    })

            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                track.analysis_gt_id = getattr(det, "analysis_gt_id", -1)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                track.analysis_gt_id = getattr(det, "analysis_gt_id", -1)
                refind_stracks.append(track)

        if competitive_recovered_tracks:
            self.removed_stracks = sub_stracks(self.removed_stracks, competitive_recovered_tracks)
            # Also remove from query engine archive
            if self.reentry_engine is not None:
                for ct in competitive_recovered_tracks:
                    self.reentry_engine.remove_from_archive(int(getattr(ct, "track_id", -1)))
            self.reentry_memory_stats["competitive_reactivated_tracks"] += int(len(competitive_recovered_tracks))
            self.reentry_memory_stats["reactivated_tracks"] += int(len(competitive_recovered_tracks))
            self.reentry_memory_stats["matches"] += int(len(competitive_recovered_tracks))
            self.reentry_memory_stats["archive_size"] = int(len(self.removed_stracks))

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
            # Keep host low-score association aligned with the baseline path.
            # ReID features for low-score detections are only materialized on demand
            # if low-score re-entry recovery is explicitly enabled later.
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
            if competitive_archive_count > 0 and int(it) >= int(competitive_archive_start):
                continue
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

        # Re-entry recovery: try to resurrect archived removed tracks before creating newborn IDs.
        if self.reentry_engine is not None and self.reentry_engine_enable:
            # Database-style re-entry query engine
            engine_recovered = []
            if len(detections) > 0 and len(u_detection) > 0:
                u_det_list = [detections[i] for i in u_detection]
                if self.args.with_reid and len(u_det_list) > 0:
                    u_det_feats = []
                    for det in u_det_list:
                        feat = getattr(det, "curr_feat", None)
                        u_det_feats.append(feat)
                    u_det_ambiguities = [
                        float(max(0.0, 1.0 - float(getattr(det, "score", 0.0))))
                        for det in u_det_list
                    ]
                    remaining, pairs = self.reentry_engine.query(
                        u_det_list,
                        u_det_feats,
                        self.frame_id,
                        det_ambiguities=u_det_ambiguities,
                    )
                    # Build recovered tracks
                    recovered_track_ids = {tid for tid, _ in pairs}
                    recovered_tracks = []
                    for track in self.removed_stracks:
                        if int(getattr(track, "track_id", -1)) in recovered_track_ids:
                            # Find the matching det
                            for tid, det_idx in pairs:
                                if tid == int(getattr(track, "track_id", -1)):
                                    det = u_det_list[det_idx]
                                    track.re_activate(det, self.frame_id, new_id=False)
                                    track.analysis_gt_id = getattr(det, "analysis_gt_id", -1)
                                    refind_stracks.append(track)
                                    recovered_tracks.append(track)
                                    break
                    if recovered_tracks:
                        self.removed_stracks = sub_stracks(self.removed_stracks, recovered_tracks)
                    # Filter out recovered detections from u_detection
                    recovered_det_set = set()
                    for _, det_idx in pairs:
                        # Map back to original u_detection index
                        if det_idx < len(u_detection):
                            recovered_det_set.add(int(u_detection[det_idx]))
                    u_detection = np.asarray([i for i in u_detection if int(i) not in recovered_det_set], dtype=int)
        elif self.reentry_memory_enable:
            if len(detections_second) > 0 and len(u_detection_second) > 0 and self.reentry_memory_use_low_score:
                if self.args.with_reid:
                    low_score_indices = [int(idx) for idx in u_detection_second]
                    if low_score_indices:
                        recovery_dets = [detections_second[idx] for idx in low_score_indices]
                        recovery_boxes = np.asarray([det.tlbr for det in recovery_dets], dtype=np.float32)
                        recovery_feats = self.encoder.inference(img, recovery_boxes)
                        for det, feat in zip(recovery_dets, recovery_feats):
                            det.update_features(feat)
                u_detection_second, recovered_low = self._recover_removed_tracks(detections_second, list(u_detection_second))
                if recovered_low:
                    refind_stracks.extend(recovered_low)
            if len(detections) > 0 and len(u_detection) > 0:
                u_detection, recovered_high = self._recover_removed_tracks(detections, list(u_detection))
                if recovered_high:
                    refind_stracks.extend(recovered_high)
            self._cleanup_pending_reentry()

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            if int(inew) in rgsa_newborn_blocked_det_ids:
                continue
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
        self.lost_stracks = sub_stracks(self.lost_stracks, removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        # Archive newly removed tracks into the query engine
        if self.reentry_engine is not None and removed_stracks:
            for track in removed_stracks:
                if int(getattr(track, "track_id", -1)) >= 0:
                    self.reentry_engine.archive_track(track)
        if self.removed_stracks:
            min_end_frame = int(self.frame_id) - int(self.removed_archive_retention)
            pruned_removed = []
            pruned_count = 0
            for track in self.removed_stracks:
                if int(getattr(track, "end_frame", -1)) >= min_end_frame:
                    pruned_removed.append(track)
                else:
                    track.release_memory()
                    pruned_count += 1
            self.removed_stracks = pruned_removed
            if self.reentry_memory_enable:
                self.reentry_memory_stats["archive_size"] = int(len(self.removed_stracks))
                self.reentry_memory_stats["pruned_tracks"] += int(pruned_count)
                self._cleanup_pending_reentry()
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]


        return output_stracks

    def get_fcaa_summary(self):
        summary = dict(self.fcaa_stats)
        trigger_groups = int(summary.get("trigger_groups", 0))
        summary["changed_group_rate"] = float(summary.get("changed_groups", 0)) / float(trigger_groups) if trigger_groups > 0 else 0.0
        summary["avg_refined_pairs_per_trigger_group"] = float(summary.get("refined_pairs", 0)) / float(trigger_groups) if trigger_groups > 0 else 0.0
        summary["trigger_row_count"] = int(len(self.fcaa_trigger_rows))
        return summary

    def get_fcaa_trigger_rows(self):
        return list(self.fcaa_trigger_rows)

    def get_fgas_summary(self):
        summary = dict(self.fgas_stats)
        trigger_blocks = int(summary.get("trigger_blocks", 0))
        controller_blocks = int(summary.get("controller_blocks", 0))
        summary["changed_block_rate"] = float(summary.get("changed_blocks", 0)) / float(trigger_blocks) if trigger_blocks > 0 else 0.0
        summary["avg_edges_per_block"] = float(summary.get("edges_touched", 0)) / float(trigger_blocks) if trigger_blocks > 0 else 0.0
        summary["avg_forced_matches_per_controller_block"] = float(summary.get("controller_applied_forced_matches", 0)) / float(controller_blocks) if controller_blocks > 0 else 0.0
        summary["avg_blocked_rows_per_controller_block"] = float(summary.get("controller_applied_blocked_rows", 0)) / float(controller_blocks) if controller_blocks > 0 else 0.0
        summary["avg_blocked_cols_per_controller_block"] = float(summary.get("controller_applied_blocked_cols", 0)) / float(controller_blocks) if controller_blocks > 0 else 0.0
        return summary

    def get_owneralt_summary(self):
        if self.owneralt_refiner is None:
            return {
                "enabled": False,
                "frames": 0,
                "candidate_detections": 0,
                "candidate_pairs": 0,
                "rewrites": 0,
                "owner_rows_released": 0,
                "alt_edges_reweighted": 0,
                "blocked_owner_reclaims": 0,
                "event_count": 0,
                "rewrite_rate": 0.0,
                "avg_candidate_pairs_per_detection": 0.0,
                "avg_alt_edges_per_rewrite": 0.0,
            }
        return self.owneralt_refiner.get_summary()

    def get_owneralt_event_rows(self):
        if self.owneralt_refiner is None:
            return []
        return self.owneralt_refiner.get_event_rows()

    def get_reentry_summary(self):
        summary = dict(self.reentry_memory_stats)
        summary["enabled"] = bool(self.reentry_memory_enable)
        summary["archive_size"] = int(len(self.removed_stracks))
        summary["archive_retention"] = int(self.removed_archive_retention)
        summary["max_gap"] = int(self.reentry_memory_max_gap)
        summary["min_similarity"] = float(self.reentry_memory_min_similarity)
        summary["confirm_streak"] = int(self.reentry_memory_confirm_streak)
        summary["confirm_gap"] = int(self.reentry_memory_confirm_gap)
        summary["confirm_min_similarity"] = float(self.reentry_memory_confirm_min_similarity)
        summary["min_det_score"] = float(self.reentry_memory_min_det_score)
        if int(summary.get("candidate_detections", 0)) > 0:
            summary["reactivation_rate"] = float(summary.get("reactivated_tracks", 0)) / float(
                summary.get("candidate_detections", 1)
            )
        else:
            summary["reactivation_rate"] = 0.0
        # Merge query engine stats
        if self.reentry_engine is not None:
            summary["engine"] = self.reentry_engine.get_stats()
        return summary

    def get_tos_summary(self):
        summary = dict(self.tos_stats)
        summary["enabled"] = bool(self.tos_enable)
        summary["analysis_only"] = bool(self.tos_analysis_only)
        summary["hold_buffer"] = int(self.tos_hold_buffer)
        summary["newborn_delay"] = int(self.tos_newborn_delay)
        summary["memory_frames"] = int(self.tos_memory_frames)
        summary["reconnect_gap_max"] = int(self.tos_reconnect_gap_max)
        summary["reconnect_min_similarity"] = float(self.tos_reconnect_min_similarity)
        summary["occlusion_thresh"] = float(self.tos_occlusion_thresh)
        summary["freeze_on_occlusion"] = bool(self.tos_freeze_on_occlusion)
        summary["disable_reentry"] = bool(self.tos_disable_reentry)
        return summary

    def get_rgsa_summary(self):
        summary = dict(self.rgsa_stats)
        summary["ccrc_enable"] = self.ccrc_enable
        summary["ccrc_tau_commit"] = self.ccrc_tau_commit
        summary["ccrc_platt_a"] = self.ccrc_platt_a
        summary["ccrc_platt_b"] = self.ccrc_platt_b
        summary["ccrc_frames"] = self.ccrc_stats.get("frames", 0)
        summary["ccrc_commits"] = self.ccrc_stats.get("commits", 0)
        summary["ccrc_abstains"] = self.ccrc_stats.get("abstains", 0)
        # TCGAU stats
        tcgau = self.tcgau_stats
        summary["tcgau_enable"] = self.tcgau_enable
        summary["tcgau_freeze_thresh"] = self.tcgau_freeze_thresh
        summary["tcgau_soft_thresh"] = self.tcgau_soft_thresh
        summary["tcgau_soft_alpha"] = self.tcgau_soft_alpha
        summary["tcgau_frames"] = tcgau.get("frames", 0)
        summary["tcgau_primary_matches"] = tcgau.get("primary_matches", 0)
        summary["tcgau_normal_updates"] = tcgau.get("normal_updates", 0)
        summary["tcgau_soft_updates"] = tcgau.get("soft_updates", 0)
        summary["tcgau_freeze_updates"] = tcgau.get("freeze_updates", 0)
        pm = max(tcgau.get("primary_matches", 0), 1)
        summary["tcgau_avg_q_update"] = round(tcgau.get("avg_q_update", 0) / pm, 6)
        summary["tcgau_avg_app_sim"] = round(tcgau.get("avg_app_sim", 0) / pm, 6)
        summary["tcgau_avg_pair_rel"] = round(tcgau.get("avg_pair_rel", 0) / pm, 6)
        frames = int(summary.get("frames", 0))
        active_frames = int(summary.get("active_frames", 0))
        summary["active_frame_rate"] = float(active_frames) / float(frames) if frames > 0 else 0.0
        summary["stage1_total"] = int(summary.get("stage1_accept", 0)) + int(summary.get("stage1_defer", 0)) + int(summary.get("stage1_reject", 0))
        summary["stage2_total"] = int(summary.get("stage2_rewrite", 0)) + int(summary.get("stage2_defer", 0)) + int(summary.get("stage2_reject", 0))
        summary["stage1_checkpoint"] = str(self.rgsa_stage1_checkpoint_path or "")
        summary["stage2_checkpoint"] = str(self.rgsa_stage2_checkpoint_path or "")
        summary["rgsa_topk"] = int(self.rgsa_topk)
        summary["stage1_lambda_defer"] = float(self.rgsa_stage1_lambda_defer)
        summary["stage1_lambda_reject"] = float(self.rgsa_stage1_lambda_reject)
        summary["stage2_rewrite_gain"] = float(self.rgsa_stage2_rewrite_gain)
        return summary

    @staticmethod
    def _rgsa_safe_value(value, default=0.0):
        try:
            out = float(value)
        except Exception:
            return float(default)
        if not np.isfinite(out):
            return float(default)
        return out

    def _rgsa_matrix_value(self, laplace_debug, key, row_idx, det_idx, default=0.0):
        if laplace_debug is None:
            return float(default)
        values = laplace_debug.get(key)
        if values is None:
            return float(default)
        values = np.asarray(values)
        if values.ndim != 2:
            return float(default)
        if row_idx < 0 or det_idx < 0 or row_idx >= values.shape[0] or det_idx >= values.shape[1]:
            return float(default)
        return self._rgsa_safe_value(values[row_idx, det_idx], default=default)

    def _rgsa_vector_value(self, laplace_debug, key, det_idx, default=0.0):
        if laplace_debug is None:
            return float(default)
        values = laplace_debug.get(key)
        if values is None:
            return float(default)
        values = np.asarray(values)
        if values.ndim == 0:
            return self._rgsa_safe_value(values, default=default)
        if det_idx < 0 or det_idx >= values.shape[0]:
            return float(default)
        return self._rgsa_safe_value(values[det_idx], default=default)

    def _rgsa_select_candidate_rows(self, score_column, valid_column_mask, topk):
        score_column = np.asarray(score_column, dtype=np.float32).reshape(-1)
        score_column = np.nan_to_num(score_column, nan=-1.0, posinf=1.0, neginf=-1.0)
        if score_column.size == 0:
            return []
        candidate_rows = np.arange(score_column.shape[0], dtype=int)
        if valid_column_mask is not None:
            valid_column_mask = np.asarray(valid_column_mask, dtype=bool).reshape(-1)
            valid_rows = candidate_rows[valid_column_mask[: candidate_rows.shape[0]]]
            if valid_rows.size > 0:
                candidate_rows = valid_rows
        order = np.argsort(-score_column[candidate_rows])
        return [int(candidate_rows[idx]) for idx in order[: max(1, int(topk))].tolist()]

    def _rgsa_build_stage1_feature(self, track, det, row_idx, det_idx, laplace_debug):
        hist = getattr(track, "features", None)
        history_len = len(hist) if hist is not None else 0
        track_gap = max(0, int(self.frame_id) - int(getattr(track, "frame_id", self.frame_id)))
        track_age = max(0, int(self.frame_id) - int(getattr(track, "start_frame", self.frame_id)))
        return np.asarray(
            [
                self._rgsa_matrix_value(laplace_debug, "haca_comp_active", row_idx, det_idx, default=0.0),
                self._rgsa_vector_value(laplace_debug, "haca_comp_margin", det_idx, default=0.0),
                self._rgsa_vector_value(laplace_debug, "haca_comp_entropy", det_idx, default=0.0),
                self._rgsa_vector_value(laplace_debug, "haca_background", det_idx, default=0.0),
                self._rgsa_matrix_value(laplace_debug, "haca_beta_hist", row_idx, det_idx, default=0.0),
                self._rgsa_matrix_value(laplace_debug, "haca_beta_ood", row_idx, det_idx, default=0.0),
                self._rgsa_matrix_value(laplace_debug, "haca_ood_score", row_idx, det_idx, default=0.0),
                float(track_gap),
                float(track_age),
                float(history_len),
                self._rgsa_safe_value(getattr(det, "score", 0.0), default=0.0),
            ],
            dtype=np.float32,
        )

    def _rgsa_build_stage2_pair_feature(self, track, det, row_idx, det_idx, candidate_rows, rank, laplace_debug):
        final_values = np.asarray(
            [self._rgsa_matrix_value(laplace_debug, "final_sim", cand_row, det_idx, default=0.0) for cand_row in candidate_rows],
            dtype=np.float32,
        )
        final_mean = float(final_values.mean()) if final_values.size > 0 else 0.0
        final_std = float(final_values.std()) if final_values.size > 0 else 0.0
        top1_final = float(final_values[0]) if final_values.size > 0 else 0.0
        top2_final = float(final_values[1]) if final_values.size > 1 else top1_final
        track_gap = max(0, int(self.frame_id) - int(getattr(track, "frame_id", self.frame_id)))
        hist = getattr(track, "features", None)
        history_len = len(hist) if hist is not None else 0
        hist_last = self._rgsa_matrix_value(laplace_debug, "haca_hist_last", row_idx, det_idx, default=0.0)
        hist_max = self._rgsa_matrix_value(laplace_debug, "haca_hist_max", row_idx, det_idx, default=0.0)
        hist_std = self._rgsa_matrix_value(laplace_debug, "haca_hist_std", row_idx, det_idx, default=0.0)
        gap_log1p = math.log1p(max(float(track_gap), 0.0))
        hist_norm = min(1.0, float(history_len) / max(float(self.laplace_min_history), 1.0))
        stability = math.exp(-max(hist_std, 0.0))
        coherence = hist_last / hist_max if hist_max > 1e-6 else hist_last
        final_sim = self._rgsa_matrix_value(laplace_debug, "final_sim", row_idx, det_idx, default=0.0)
        return np.asarray(
            [
                final_sim,
                self._rgsa_matrix_value(laplace_debug, "spatial_sim", row_idx, det_idx, default=0.0),
                self._rgsa_matrix_value(laplace_debug, "motion_sim", row_idx, det_idx, default=0.0),
                self._rgsa_matrix_value(laplace_debug, "haca_temp_sim", row_idx, det_idx, default=0.0),
                hist_last,
                hist_max,
                hist_std,
                gap_log1p,
                min(1.0, max(0.0, hist_norm)),
                min(1.0, max(0.0, stability)),
                min(1.0, max(0.0, coherence)),
                self._rgsa_matrix_value(
                    laplace_debug,
                    "haca_anchor_z",
                    row_idx,
                    det_idx,
                    default=(final_sim - final_mean) / max(final_std, 1e-6),
                ),
                self._rgsa_matrix_value(
                    laplace_debug,
                    "haca_anchor_margin",
                    row_idx,
                    det_idx,
                    default=top1_final - top2_final,
                ),
                self._rgsa_matrix_value(laplace_debug, "haca_anchor_rank", row_idx, det_idx, default=float(rank)),
                self._rgsa_safe_value(getattr(det, "score", 0.0), default=0.0),
            ],
            dtype=np.float32,
        )

    def _apply_rgsa_primary_runtime(self, dists, strack_pool, detections, ious_dists_mask, laplace_debug, use_haca_primary):
        blocked_newborn_det_ids = set()
        hard_block_det_ids = set()
        self.rgsa_stats["frames"] = int(self.rgsa_stats.get("frames", 0)) + 1
        if not self.rgsa_enable:
            self.rgsa_stats["inactive_frames"] = int(self.rgsa_stats.get("inactive_frames", 0)) + 1
            return np.asarray(dists, dtype=np.float32), blocked_newborn_det_ids, hard_block_det_ids
        if (
            self.rgsa_stage1_head is None
            or laplace_debug is None
            or not use_haca_primary
            or len(strack_pool) == 0
            or len(detections) == 0
        ):
            self.rgsa_stats["inactive_frames"] = int(self.rgsa_stats.get("inactive_frames", 0)) + 1
            return np.asarray(dists, dtype=np.float32), blocked_newborn_det_ids, hard_block_det_ids
        final_sim = laplace_debug.get("final_sim")
        if final_sim is None:
            final_sim = laplace_debug.get("anchor_sim")
        if final_sim is None:
            self.rgsa_stats["inactive_frames"] = int(self.rgsa_stats.get("inactive_frames", 0)) + 1
            return np.asarray(dists, dtype=np.float32), blocked_newborn_det_ids, hard_block_det_ids
        final_sim = np.asarray(final_sim, dtype=np.float32)
        if final_sim.ndim != 2 or final_sim.shape[0] != len(strack_pool) or final_sim.shape[1] != len(detections):
            self.rgsa_stats["inactive_frames"] = int(self.rgsa_stats.get("inactive_frames", 0)) + 1
            return np.asarray(dists, dtype=np.float32), blocked_newborn_det_ids, hard_block_det_ids

        self.rgsa_stats["active_frames"] = int(self.rgsa_stats.get("active_frames", 0)) + 1
        dists = np.asarray(dists, dtype=np.float32).copy()
        valid_mask = None
        if ious_dists_mask is not None:
            valid_mask = np.logical_not(np.asarray(ious_dists_mask, dtype=bool))
            if valid_mask.shape != final_sim.shape:
                valid_mask = None

        stage1_features = []
        stage1_det_ids = []
        stage1_host_rows = []
        candidate_rows_per_det = {}
        candidate_features_per_det = {}
        candidate_row_ids_per_det = {}
        for det_idx, det in enumerate(detections):
            candidate_rows = self._rgsa_select_candidate_rows(
                final_sim[:, det_idx],
                valid_mask[:, det_idx] if valid_mask is not None else None,
                topk=min(self.rgsa_topk, len(strack_pool)),
            )
            if not candidate_rows:
                continue
            host_row = int(candidate_rows[0])
            stage1_features.append(self._rgsa_build_stage1_feature(strack_pool[host_row], det, host_row, det_idx, laplace_debug))
            stage1_det_ids.append(int(det_idx))
            stage1_host_rows.append(host_row)
            candidate_rows_per_det[int(det_idx)] = list(candidate_rows)
            candidate_row_ids_per_det[int(det_idx)] = [int(row_idx) for row_idx in candidate_rows]
            candidate_features_per_det[int(det_idx)] = np.asarray(
                [
                    self._rgsa_build_stage2_pair_feature(
                        strack_pool[int(row_idx)],
                        det,
                        int(row_idx),
                        int(det_idx),
                        candidate_rows,
                        rank,
                        laplace_debug,
                    )
                    for rank, row_idx in enumerate(candidate_rows)
                ],
                dtype=np.float32,
            )
        if not stage1_features:
            self.rgsa_stats["inactive_frames"] = int(self.rgsa_stats.get("inactive_frames", 0)) + 1
            self.rgsa_stats["active_frames"] = max(0, int(self.rgsa_stats.get("active_frames", 0)) - 1)
            return dists, blocked_newborn_det_ids, hard_block_det_ids

        stage1_output = self.rgsa_stage1_head.apply_soft_deferral(
            np.asarray(stage1_features, dtype=np.float32),
            det_ids=stage1_det_ids,
            track_ids=stage1_host_rows,
            device=self.rgsa_device,
            lambda_defer=self.rgsa_stage1_lambda_defer,
            lambda_reject=self.rgsa_stage1_lambda_reject,
        )
        # Inject HACA runtime signals into deferred_host_signals for verifier
        if self.rgsa_verifier is not None:
            final_sim = laplace_debug.get("final_sim") if laplace_debug else None
            if final_sim is None and laplace_debug is not None:
                final_sim = laplace_debug.get("anchor_sim")
            for det_id in stage1_output.deferred_det_ids:
                det_id = int(det_id)
                candidate_rows = candidate_rows_per_det.get(det_id, [])
                if not candidate_rows:
                    continue
                host_row = int(candidate_rows[0])
                det_idx = det_id
                if final_sim is not None and host_row < final_sim.shape[0] and det_idx < final_sim.shape[1]:
                    col = final_sim[:, det_idx]
                    sorted_col = np.sort(col)[::-1]
                    margin = float(sorted_col[0] - sorted_col[1]) if len(sorted_col) > 1 else float(sorted_col[0])
                else:
                    margin = 0.0
                stage1_output.deferred_host_signals.setdefault(det_id, {}).update({
                    "s_final": float(final_sim[host_row, det_idx]) if final_sim is not None and host_row < final_sim.shape[0] and det_idx < final_sim.shape[1] else 0.0,
                    "margin": margin,
                    "entropy": float(self._rgsa_vector_value(laplace_debug, "haca_comp_entropy", det_idx, default=0.0)),
                    "activation": float(self._rgsa_matrix_value(laplace_debug, "haca_comp_active", host_row, det_idx, default=0.0)),
                    "bg_prob": float(self._rgsa_vector_value(laplace_debug, "haca_background", det_idx, default=0.0)),
                    "beta_hist": float(self._rgsa_matrix_value(laplace_debug, "haca_beta_hist", host_row, det_idx, default=0.0)),
                    "beta_ood": float(self._rgsa_matrix_value(laplace_debug, "haca_beta_ood", host_row, det_idx, default=0.0)),
                    "track_gap": float(max(0, int(self.frame_id) - int(getattr(strack_pool[host_row], "frame_id", self.frame_id)))),
                    "track_age": float(max(0, int(self.frame_id) - int(getattr(strack_pool[host_row], "start_frame", self.frame_id)))),
                    "history_len": float(len(getattr(strack_pool[host_row], "features", None) or [])),
                    "det_score": float(getattr(detections[det_idx], "score", 0.0)),
                })
        for det_id in stage1_output.deferred_det_ids:
            det_id = int(det_id)
            candidate_rows = candidate_rows_per_det.get(det_id, [])
            if not candidate_rows:
                continue
            host_row = int(candidate_rows[0])
            bias = float(stage1_output.deferred_cost_bias.get(det_id, self.rgsa_stage1_lambda_defer))
            dists[host_row, det_id] = float(np.clip(dists[host_row, det_id] + bias, 0.0, 1.0))
            self.rgsa_stats["stage1_penalized_edges"] = int(self.rgsa_stats.get("stage1_penalized_edges", 0)) + 1
        for det_id in stage1_output.rejected_det_ids:
            det_id = int(det_id)
            if det_id < 0 or det_id >= dists.shape[1]:
                continue
            candidate_rows = candidate_rows_per_det.get(det_id, [])
            bias = float(stage1_output.deferred_cost_bias.get(det_id, self.rgsa_stage1_lambda_reject))
            if candidate_rows:
                host_row = int(candidate_rows[0])
                dists[host_row, det_id] = float(np.clip(dists[host_row, det_id] + bias, 0.0, 1.0))
                self.rgsa_stats["stage1_reject_penalized_edges"] = int(self.rgsa_stats.get("stage1_reject_penalized_edges", 0)) + 1
        self.rgsa_stats["stage1_accept"] = int(self.rgsa_stats.get("stage1_accept", 0)) + int(len(stage1_output.accepted_det_ids))
        self.rgsa_stats["stage1_defer"] = int(self.rgsa_stats.get("stage1_defer", 0)) + int(len(stage1_output.deferred_det_ids))
        self.rgsa_stats["stage1_reject"] = int(self.rgsa_stats.get("stage1_reject", 0)) + int(len(stage1_output.rejected_det_ids))

        if stage1_output.deferred_det_ids and self.rgsa_verifier is not None:
            # VERIFIER path: confirm or veto host's top-1 candidate
            deferred_ids = [int(d) for d in stage1_output.deferred_det_ids]
            signals_per_det = {}
            track_ids_per_det = {}
            for det_id in deferred_ids:
                host_signals = stage1_output.deferred_host_signals.get(det_id, {})
                signals_per_det[det_id] = host_signals
                candidate_rows = candidate_rows_per_det.get(det_id, [])
                if candidate_rows:
                    track_ids_per_det[det_id] = int(candidate_rows[0])

            verifier_output = self.rgsa_verifier.verify_batch(
                deferred_det_ids=deferred_ids,
                signals_per_det=signals_per_det,
                track_ids_per_det=track_ids_per_det,
            )

            # Confirmed: accept host's top-1, no cost modification needed
            self.rgsa_stats["stage2_confirm"] = int(self.rgsa_stats.get("stage2_confirm", 0)) + int(len(verifier_output.confirmed_matches))

            # Vetoed: penalize host edge, allow Hungarian to find alternative
            for det_id in verifier_output.vetoed_det_ids:
                det_id = int(det_id)
                candidate_rows = candidate_rows_per_det.get(det_id, [])
                if candidate_rows:
                    host_row = int(candidate_rows[0])
                    if 0 <= host_row < dists.shape[0] and 0 <= det_id < dists.shape[1]:
                        dists[host_row, det_id] = float(np.clip(dists[host_row, det_id] + self.rgsa_stage1_lambda_defer, 0.0, 1.0))
                self.rgsa_stats["stage2_veto"] = int(self.rgsa_stats.get("stage2_veto", 0)) + 1

            self.rgsa_stats["verifier_active_frames"] = int(self.rgsa_stats.get("verifier_active_frames", 0)) + 1

        elif stage1_output.deferred_det_ids and self.rgsa_stage2_haca_mode == "dual_haca":
            # Dual-param HACA Stage2: for deferred detections, apply cost discount to
            # ALL local candidate edges to give Hungarian more freedom to re-match.
            # This is equivalent to "lowering the bar" for deferred detections.
            deferred_det_ids = [int(d) for d in stage1_output.deferred_det_ids]
            for det_id in deferred_det_ids:
                candidate_rows = candidate_rows_per_det.get(det_id, [])
                if not candidate_rows:
                    continue
                host_row = int(candidate_rows[0])
                host_cost = float(dists[host_row, det_id])

                # Apply cost discount to ALL candidate edges (not just host)
                for row_idx in candidate_rows:
                    row_idx = int(row_idx)
                    if 0 <= row_idx < dists.shape[0] and 0 <= det_id < dists.shape[1]:
                        old_cost = float(dists[row_idx, det_id])
                        new_cost = float(np.clip(old_cost - self.rgsa_stage2_cost_discount, 0.0, 1.0))
                        dists[row_idx, det_id] = new_cost

                # If any non-host candidate got a lower cost than host, count as rewrite
                best_other_cost = float("inf")
                best_other_row = host_row
                for row_idx in candidate_rows:
                    row_idx = int(row_idx)
                    if row_idx == host_row:
                        continue
                    c = float(dists[row_idx, det_id])
                    if c < best_other_cost:
                        best_other_cost = c
                        best_other_row = row_idx

                if best_other_cost < host_cost - self.rgsa_stage2_rewrite_threshold:
                    self.rgsa_stats["stage2_rewrite"] = int(self.rgsa_stats.get("stage2_rewrite", 0)) + 1
                else:
                    self.rgsa_stats["stage2_defer"] = int(self.rgsa_stats.get("stage2_defer", 0)) + 1

            self.rgsa_stats["stage2_haca_dual_active_frames"] = int(self.rgsa_stats.get("stage2_haca_dual_active_frames", 0)) + 1

        elif self.rgsa_stage2_head is not None and stage1_output.deferred_det_ids:
            # Original learned Stage2 path
            stage2_output = self.rgsa_stage2_head.apply_recovery(
                deferred_det_ids=[int(det_id) for det_id in stage1_output.deferred_det_ids],
                candidate_features_per_det=candidate_features_per_det,
                candidate_track_ids_per_det=candidate_row_ids_per_det,
                device=self.rgsa_device,
            )
            for det_id, row_idx in stage2_output.rewritten_matches.items():
                det_id = int(det_id)
                row_idx = int(row_idx)
                if row_idx < 0 or row_idx >= dists.shape[0] or det_id < 0 or det_id >= dists.shape[1]:
                    continue
                dists[row_idx, det_id] = float(np.clip(dists[row_idx, det_id] - self.rgsa_stage2_rewrite_gain, 0.0, 1.0))
                self.rgsa_stats["stage2_rewrite_edges"] = int(self.rgsa_stats.get("stage2_rewrite_edges", 0)) + 1
            for det_id in stage2_output.still_deferred_det_ids:
                det_id = int(det_id)
                # Soft-only: raise cost on local candidate rows, do NOT hard-mask to 1.0
                for row_idx in candidate_rows_per_det.get(det_id, []):
                    if 0 <= int(row_idx) < dists.shape[0] and 0 <= det_id < dists.shape[1]:
                        dists[int(row_idx), det_id] = float(np.clip(dists[int(row_idx), det_id] + self.rgsa_stage1_lambda_defer, 0.0, 1.0))
                self.rgsa_stats["stage2_stilldefer_penalized_edges"] = int(self.rgsa_stats.get("stage2_stilldefer_penalized_edges", 0)) + int(len(candidate_rows_per_det.get(det_id, [])))
            for det_id in stage2_output.rejected_det_ids:
                det_id = int(det_id)
                if det_id < 0 or det_id >= dists.shape[1]:
                    continue
                # Soft-only: raise cost on local candidate rows, do NOT block newborn
                for row_idx in candidate_rows_per_det.get(det_id, []):
                    if 0 <= int(row_idx) < dists.shape[0]:
                        dists[int(row_idx), det_id] = float(np.clip(dists[int(row_idx), det_id] + self.rgsa_stage1_lambda_reject, 0.0, 1.0))
                # Do NOT add to blocked_newborn_det_ids or hard_block_det_ids
            self.rgsa_stats["stage2_rewrite"] = int(self.rgsa_stats.get("stage2_rewrite", 0)) + int(len(stage2_output.rewritten_matches))
            self.rgsa_stats["stage2_defer"] = int(self.rgsa_stats.get("stage2_defer", 0)) + int(len(stage2_output.still_deferred_det_ids))
            self.rgsa_stats["stage2_reject"] = int(self.rgsa_stats.get("stage2_reject", 0)) + int(len(stage2_output.rejected_det_ids))
            self.rgsa_stats["stage2_reject_penalized_edges"] = int(self.rgsa_stats.get("stage2_reject_penalized_edges", 0)) + int(len(stage2_output.rejected_det_ids))

        self.rgsa_stats["stage1_newborn_blocked"] = int(self.rgsa_stats.get("stage1_newborn_blocked", 0))
        self.rgsa_stats["stage1_hard_blocked_cols"] = int(self.rgsa_stats.get("stage1_hard_blocked_cols", 0))
        self.rgsa_stats["newborn_blocked"] = int(self.rgsa_stats.get("newborn_blocked", 0)) + int(len(blocked_newborn_det_ids))
        self.rgsa_stats["hard_blocked_cols"] = int(self.rgsa_stats.get("hard_blocked_cols", 0)) + int(len(hard_block_det_ids))
        return dists, blocked_newborn_det_ids, hard_block_det_ids

    def _compute_tos_occlusion_score(self, track, det, itracked, idet, laplace_debug):
        """Compute TOS occlusion score for a matched pair.

        Returns float in [0, 1]: higher = more likely occlusion.
        Uses final_sim (appearance association strength) as primary signal,
        since comp_active is only available with haca_v3 competition head.
        """
        if laplace_debug is None:
            return 0.0

        # Track gap: only consider occlusion if track has been missing
        track_gap = max(0, int(self.frame_id) - int(getattr(track, "frame_id", self.frame_id)))
        if track_gap == 0:
            return 0.0

        # Primary: final_sim (association quality, available in all HACA versions)
        final_sim = laplace_debug.get("final_sim") or laplace_debug.get("anchor_sim")
        if final_sim is None:
            return 0.0
        sim = float(final_sim[itracked, idet]) if itracked < final_sim.shape[0] and idet < final_sim.shape[1] else 0.0

        # Detection score ambiguity
        det_score = float(getattr(det, "score", 0.0))

        # Gap factor
        gap_factor = min(1.0, float(track_gap) / max(float(self.tos_hold_buffer), 1.0))

        # Low sim + high gap = occlusion
        # sim_range: clamp [0,1] — occlusion_haca = 1 - sim
        occlusion_sim = 1.0 - max(0.0, min(1.0, sim))
        score = (
            0.5 * occlusion_sim
            + 0.3 * gap_factor
            + 0.2 * max(0.0, 1.0 - det_score)
        )
        return float(np.clip(score, 0.0, 1.0))

    def _compute_tcgau_policy(self, track, det, itracked, idet, laplace_debug):
        """Compute TCGAU update policy for a matched pair.

        Returns dict with mode, q_update, signals, and STrack parameters.
        """
        policy = {
            "mode": "normal",
            "q_update": 1.0,
            "app_sim": 0.0,
            "pair_rel": 0.0,
            "stability": 0.0,
            "coherence": 0.0,
            "margin": 0.0,
            "hist_norm": 0.0,
            "alpha_override": None,
            "append_history": True,
        }

        if not self.tcgau_enable or laplace_debug is None:
            return policy

        appearance_sim = laplace_debug.get("appearance_sim")
        if appearance_sim is not None and itracked < appearance_sim.shape[0] and idet < appearance_sim.shape[1]:
            policy["app_sim"] = float(appearance_sim[itracked, idet])
        else:
            policy["app_sim"] = self._safe_cosine_similarity(
                getattr(track, "smooth_feat", None), getattr(det, "curr_feat", None)
            )

        pair_rel = laplace_debug.get("pair_rel")
        if pair_rel is not None and itracked < pair_rel.shape[0] and idet < pair_rel.shape[1]:
            policy["pair_rel"] = float(pair_rel[itracked, idet])
        else:
            policy["pair_rel"] = float(np.clip(policy["app_sim"], 0.0, 1.0))

        self.tcgau_debug_available = bool(appearance_sim is not None and pair_rel is not None)

        stability = laplace_debug.get("stability")
        if stability is not None:
            st = np.asarray(stability).reshape(-1)
            if itracked < len(st):
                policy["stability"] = float(st[itracked])

        coherence = laplace_debug.get("coherence")
        if coherence is not None:
            co = np.asarray(coherence).reshape(-1)
            if itracked < len(co):
                policy["coherence"] = float(co[itracked])

        margin = laplace_debug.get("haca_comp_margin")
        if margin is not None:
            mg = np.asarray(margin).reshape(-1)
            if idet < len(mg):
                policy["margin"] = float(mg[idet])

        hist = getattr(track, "features", None)
        policy["hist_norm"] = min(1.0, float(len(hist)) / max(self.tcgau_history_norm_denom, 1.0)) if hist else 0.0

        # q_update formula
        margin_gate = 1.0 if policy["margin"] > self.tcgau_margin_thresh else 0.0
        q = (
            max(0.0, policy["app_sim"])
            * max(0.0, policy["pair_rel"])
            * max(0.0, policy["stability"])
            * max(0.0, policy["coherence"])
            * max(0.0, policy["hist_norm"])
            * margin_gate
        )
        policy["q_update"] = float(np.clip(q, 0.0, 1.0))

        # Three-tier decision
        if policy["q_update"] <= self.tcgau_freeze_thresh:
            policy["mode"] = "freeze"
            policy["append_history"] = False
            policy["alpha_override"] = None
        elif policy["q_update"] <= self.tcgau_soft_thresh:
            policy["mode"] = "soft"
            policy["alpha_override"] = self.tcgau_soft_alpha
            policy["append_history"] = True
        else:
            policy["mode"] = "normal"
            policy["alpha_override"] = None
            policy["append_history"] = True

        return policy

    @staticmethod
    def _safe_cosine_similarity(vec_a, vec_b):
        if vec_a is None or vec_b is None:
            return 0.0
        a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
        b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
        if a.size == 0 or b.size == 0:
            return 0.0
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-8:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _recover_removed_tracks(self, detections, candidate_indices):
        """
        Try to reactivate archived removed tracks on top of the current unmatched detections.

        This keeps the core tracker unchanged and only adds a bounded re-entry pass before newborn IDs
        are created.
        """
        if not self.reentry_memory_enable:
            return list(candidate_indices), []
        if len(candidate_indices) == 0 or len(self.removed_stracks) == 0:
            return list(candidate_indices), []

        det_indices = [int(idx) for idx in candidate_indices if int(idx) >= 0 and int(idx) < len(detections)]
        if not det_indices:
            return list(candidate_indices), []

        archive_tracks = []
        for track in self.removed_stracks:
            gap = int(self.frame_id) - int(getattr(track, "end_frame", self.frame_id))
            if gap < 1 or gap > self.reentry_memory_max_gap:
                continue
            archive_tracks.append(track)

        if not archive_tracks:
            return list(candidate_indices), []

        candidate_dets = [detections[idx] for idx in det_indices]
        score_matrix = np.full((len(archive_tracks), len(candidate_dets)), -1.0, dtype=np.float32)
        for row, track in enumerate(archive_tracks):
            gap = max(1, int(self.frame_id) - int(getattr(track, "end_frame", self.frame_id)))
            gap_factor = math.exp(-float(gap) / float(max(self.reentry_memory_max_gap, 1)))
            for col, det in enumerate(candidate_dets):
                det_score = float(getattr(det, "score", 0.0))
                if det_score < self.reentry_memory_min_det_score:
                    continue
                app_sim = self._safe_cosine_similarity(getattr(track, "smooth_feat", None), getattr(det, "curr_feat", None))
                try:
                    iou_sim = 1.0 - float(matching.iou_distance([track], [det])[0, 0])
                except Exception:
                    iou_sim = 0.0
                score = (
                    self.reentry_memory_app_weight * max(0.0, app_sim)
                    + self.reentry_memory_iou_weight * max(0.0, iou_sim)
                    + self.reentry_memory_score_weight * det_score
                    + self.reentry_memory_gap_weight * gap_factor
                )
                score_matrix[row, col] = float(np.clip(score, 0.0, 1.0))

        valid_mask = score_matrix >= self.reentry_memory_min_similarity
        self.reentry_memory_stats["candidate_tracks"] += int(len(archive_tracks))
        self.reentry_memory_stats["candidate_detections"] += int(len(candidate_dets))
        self.reentry_memory_stats["candidate_pairs"] += int(np.count_nonzero(np.isfinite(score_matrix) & (score_matrix >= 0.0)))
        if not bool(valid_mask.any()):
            self.reentry_memory_stats["archive_size"] = int(len(self.removed_stracks))
            return list(candidate_indices), []

        cost_matrix = 1.0 - np.clip(score_matrix, 0.0, 1.0)
        cost_matrix[~valid_mask] = 1.0
        matches, _, _ = matching.linear_assignment(cost_matrix, thresh=float(1.0 - self.reentry_memory_min_similarity))

        recovered_tracks = []
        recovered_det_indices = set()
        withheld_det_indices = set()
        for trk_idx, det_idx in matches:
            trk_idx = int(trk_idx)
            det_idx = int(det_idx)
            if trk_idx < 0 or det_idx < 0:
                continue
            if trk_idx >= len(archive_tracks) or det_idx >= len(candidate_dets):
                continue
            if not bool(valid_mask[trk_idx, det_idx]):
                continue
            track = archive_tracks[trk_idx]
            det = candidate_dets[det_idx]
            score = self._score_archive_detection_pair(track, det)
            if score is None:
                continue
            withheld_det_indices.add(det_indices[det_idx])
            if self._update_pending_reentry(track, det, score, origin="unmatched"):
                self._commit_confirmed_reentry(track, det, origin="unmatched")
                recovered_tracks.append(track)
                recovered_det_indices.add(det_indices[det_idx])

        if recovered_tracks:
            self.removed_stracks = sub_stracks(self.removed_stracks, recovered_tracks)
        self.reentry_memory_stats["matches"] += int(len(matches))
        self.reentry_memory_stats["reactivated_tracks"] += int(len(recovered_tracks))
        self.reentry_memory_stats["archive_size"] = int(len(self.removed_stracks))
        remaining_indices = [idx for idx in det_indices if idx not in recovered_det_indices and idx not in withheld_det_indices]
        return remaining_indices, recovered_tracks

    def _competitive_recover_removed_tracks(
        self,
        tracks,
        detections,
        dists,
        raw_ious_dists,
        valid_track_indices,
        valid_det_indices,
        match_thresh,
    ):
        """
        Let archived removed tracks compete with current active tracks for the same detections.

        This is stricter than unmatched-only recovery: if an archived track is sufficiently better
        than the current owner, it can reclaim the detection before newborn creation.
        """
        if not self.reentry_memory_enable:
            return np.asarray([], dtype=int), np.asarray([], dtype=int), []
        if len(valid_track_indices) == 0 or len(valid_det_indices) == 0 or len(self.removed_stracks) == 0:
            return np.asarray(valid_track_indices, dtype=int), np.asarray(valid_det_indices, dtype=int), []
        active_tracks = [tracks[int(idx)] for idx in valid_track_indices]
        archive_tracks = []
        for track in self.removed_stracks:
            gap = int(self.frame_id) - int(getattr(track, "end_frame", self.frame_id))
            if gap < 1 or gap > self.reentry_memory_max_gap:
                continue
            archive_tracks.append(track)

        if not archive_tracks:
            return np.asarray(valid_track_indices, dtype=int), np.asarray(valid_det_indices, dtype=int), []

        candidate_dets = [detections[int(idx)] for idx in valid_det_indices]
        active_cost = np.asarray(dists, dtype=np.float32)
        archive_cost = np.full((len(archive_tracks), len(candidate_dets)), 1.0, dtype=np.float32)
        for row, track in enumerate(archive_tracks):
            gap = max(1, int(self.frame_id) - int(getattr(track, "end_frame", self.frame_id)))
            gap_factor = math.exp(-float(gap) / float(max(self.reentry_memory_max_gap, 1)))
            for col, det in enumerate(candidate_dets):
                det_score = float(getattr(det, "score", 0.0))
                if det_score < self.reentry_memory_min_det_score:
                    continue
                app_sim = self._safe_cosine_similarity(getattr(track, "smooth_feat", None), getattr(det, "curr_feat", None))
                try:
                    iou_sim = 1.0 - float(matching.iou_distance([track], [det])[0, 0])
                except Exception:
                    iou_sim = 0.0
                score = (
                    self.reentry_memory_app_weight * max(0.0, app_sim)
                    + self.reentry_memory_iou_weight * max(0.0, iou_sim)
                    + self.reentry_memory_score_weight * det_score
                    + self.reentry_memory_gap_weight * gap_factor
                )
                if score < self.reentry_memory_min_similarity:
                    continue
                archive_cost[row, col] = float(np.clip(1.0 - score, 0.0, 1.0))

        if not np.isfinite(archive_cost).any():
            return np.asarray(valid_track_indices, dtype=int), np.asarray(valid_det_indices, dtype=int), []

        active_matches, _, _ = matching.linear_assignment(active_cost, thresh=float(match_thresh))
        archive_matches, _, _ = matching.linear_assignment(archive_cost, thresh=float(match_thresh))

        active_assignment: Dict[int, tuple[int, float]] = {}
        for row_idx, col_idx in active_matches.tolist():
            active_assignment[int(col_idx)] = (int(row_idx), float(active_cost[int(row_idx), int(col_idx)]))

        archive_assignment: Dict[int, tuple[int, float]] = {}
        for row_idx, col_idx in archive_matches.tolist():
            archive_assignment[int(col_idx)] = (int(row_idx), float(archive_cost[int(row_idx), int(col_idx)]))

        reclaimed_det_indices: set[int] = set()
        recovered_tracks: List[object] = []
        for local_det_idx, (archive_row, archive_cost_value) in archive_assignment.items():
            active_choice = active_assignment.get(int(local_det_idx))
            if active_choice is None:
                continue
            active_row, active_cost_value = active_choice
            if not np.isfinite(active_cost_value):
                continue
            # Only reclaim when the archived track is clearly better, not merely tied.
            if archive_cost_value + 1e-4 >= active_cost_value:
                continue
            if active_cost_value - archive_cost_value < 0.02:
                continue
            archive_track = archive_tracks[int(archive_row)]
            det = candidate_dets[int(local_det_idx)]
            archive_track.re_activate(det, self.frame_id, new_id=False)
            archive_track.analysis_gt_id = getattr(det, "analysis_gt_id", -1)
            recovered_tracks.append(archive_track)
            reclaimed_det_indices.add(int(valid_det_indices[int(local_det_idx)]))

        if recovered_tracks:
            self.removed_stracks = sub_stracks(self.removed_stracks, recovered_tracks)
            self.reentry_memory_stats["competitive_reactivated_tracks"] += int(len(recovered_tracks))
            self.reentry_memory_stats["archive_size"] = int(len(self.removed_stracks))

        remaining_track_indices = np.asarray(
            [int(idx) for idx in valid_track_indices],
            dtype=int,
        )
        remaining_det_indices = np.asarray(
            [int(idx) for idx in valid_det_indices if int(idx) not in reclaimed_det_indices],
            dtype=int,
        )
        self.reentry_memory_stats["competitive_archive_candidates"] += int(len(archive_tracks))
        self.reentry_memory_stats["competitive_primary_matches"] += int(len(active_matches))
        return remaining_track_indices, remaining_det_indices, recovered_tracks

    def _score_archive_detection_pair(self, track, det):
        det_score = float(getattr(det, "score", 0.0))
        if det_score < self.reentry_memory_min_det_score:
            return None
        gap = max(1, int(self.frame_id) - int(getattr(track, "end_frame", self.frame_id)))
        gap_factor = math.exp(-float(gap) / float(max(self.reentry_memory_max_gap, 1)))
        app_sim = self._safe_cosine_similarity(getattr(track, "smooth_feat", None), getattr(det, "curr_feat", None))
        try:
            iou_sim = 1.0 - float(matching.iou_distance([track], [det])[0, 0])
        except Exception:
            iou_sim = 0.0
        score = (
            self.reentry_memory_app_weight * max(0.0, app_sim)
            + self.reentry_memory_iou_weight * max(0.0, iou_sim)
            + self.reentry_memory_score_weight * det_score
            + self.reentry_memory_gap_weight * gap_factor
        )
        if score < self.reentry_memory_min_similarity:
            return None
        return float(np.clip(score, 0.0, 1.0))

    def _update_pending_reentry(self, track, det, score, origin="competitive"):
        key = int(getattr(track, "track_id", -1))
        if key < 0:
            return False
        pending = self.reentry_memory_pending.get(key)
        if pending is None:
            self.reentry_memory_pending[key] = {
                "track": track,
                "det": det,
                "score": float(score),
                "streak": 1,
                "last_frame": int(self.frame_id),
                "origin": str(origin),
            }
            self.reentry_memory_stats["pending_proposals"] += 1
            return False

        if int(pending.get("last_frame", -1)) != int(self.frame_id) - 1:
            pending.update(
                {
                    "track": track,
                    "det": det,
                    "score": float(score),
                    "streak": 1,
                    "last_frame": int(self.frame_id),
                    "origin": str(origin),
                }
            )
            self.reentry_memory_stats["pending_resets"] += 1
            return False

        pending["track"] = track
        pending["det"] = det
        pending["score"] = float(score)
        pending["streak"] = int(pending.get("streak", 0)) + 1
        pending["last_frame"] = int(self.frame_id)
        pending["origin"] = str(origin)
        self.reentry_memory_stats["pending_updates"] += 1
        if int(pending["streak"]) >= int(self.reentry_memory_confirm_streak) and float(score) >= float(self.reentry_memory_confirm_min_similarity):
            self.reentry_memory_pending.pop(key, None)
            self.reentry_memory_stats["pending_confirmations"] += 1
            return True
        return False

    def _commit_confirmed_reentry(self, track, det, origin="competitive"):
        track.re_activate(det, self.frame_id, new_id=False)
        track.analysis_gt_id = getattr(det, "analysis_gt_id", -1)
        self.removed_stracks = sub_stracks(self.removed_stracks, [track])
        if str(origin) == "competitive":
            self.reentry_memory_stats["competitive_reactivated_tracks"] += 1
        self.reentry_memory_stats["reactivated_tracks"] += 1
        self.reentry_memory_stats["matches"] += 1
        self.reentry_memory_stats["archive_size"] = int(len(self.removed_stracks))

    def _cleanup_pending_reentry(self):
        if not self.reentry_memory_pending:
            return
        active_ids = {
            int(getattr(track, "track_id", -1))
            for track in self.removed_stracks
            if int(getattr(track, "track_id", -1)) >= 0
        }
        stale_keys = []
        for key, pending in self.reentry_memory_pending.items():
            track = pending.get("track")
            track_id = int(getattr(track, "track_id", -1)) if track is not None else -1
            if track_id < 0 or track_id not in active_ids:
                stale_keys.append(key)
                continue
            last_frame = int(pending.get("last_frame", -1))
            if last_frame >= 0 and int(self.frame_id) - last_frame > int(self.reentry_memory_confirm_gap):
                stale_keys.append(key)
        for key in stale_keys:
            self.reentry_memory_pending.pop(key, None)
            self.reentry_memory_stats["pending_resets"] += 1

    def drain_owneralt_event_rows(self):
        if self.owneralt_refiner is None:
            return []
        return self.owneralt_refiner.drain_event_rows()

    def get_graph_assoc_summary(self):
        if self.graph_assoc_refiner is None:
            return {
                "enabled": False,
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
                "changed_block_rate": 0.0,
                "candidate_accept_rate": 0.0,
                "avg_forced_matches_per_changed_block": 0.0,
                "avg_assignments_per_block": 0.0,
            }
        return self.graph_assoc_refiner.get_summary()

    def get_graph_assoc_event_rows(self):
        if self.graph_assoc_refiner is None:
            return []
        return self.graph_assoc_refiner.get_event_rows()

    def drain_graph_assoc_event_rows(self):
        if self.graph_assoc_refiner is None:
            return []
        return self.graph_assoc_refiner.drain_event_rows()

    def get_graph_assoc_candidate_rows(self):
        if self.graph_assoc_refiner is None:
            return []
        return self.graph_assoc_refiner.get_candidate_rows()

    def drain_graph_assoc_candidate_rows(self):
        if self.graph_assoc_refiner is None:
            return []
        return self.graph_assoc_refiner.drain_candidate_rows()


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
