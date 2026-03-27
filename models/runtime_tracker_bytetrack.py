# Copyright (c) 2024. All Rights Reserved.
"""
使用 ByteTrack 特征的运行时跟踪器

这个跟踪器:
1. 使用 ByteTrack (YOLOX) 进行目标检测和特征提取
2. 使用训练好的频域关联模块进行 ID 关联
3. 不依赖 DINO 检测器
"""

import torch
import einops
import os
import csv
import json
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Any
import warnings
import hashlib
from collections import defaultdict

from structures.instances import Instances
from structures.ordered_set import OrderedSet
from utils.misc import distributed_device
try:
    from vnext_laplace_assoc import LaplaceAssociationAdapter
    _LAPLACE_ASSOC_AVAILABLE = True
except Exception:
    LaplaceAssociationAdapter = None
    _LAPLACE_ASSOC_AVAILABLE = False

try:
    from models.mtcr_assoc import MTCRAssociationAdapter
    _MTCR_ASSOC_AVAILABLE = True
except Exception:
    MTCRAssociationAdapter = None
    _MTCR_ASSOC_AVAILABLE = False

try:
    from models.runtime_replay_assoc import RuntimeReplayAssociationAdapter
    _RUNTIME_REPLAY_ASSOC_AVAILABLE = True
except Exception:
    RuntimeReplayAssociationAdapter = None
    _RUNTIME_REPLAY_ASSOC_AVAILABLE = False

try:
    from models.competition_assoc import (
        ACTION_LABELS as _COMP_ACTION_LABELS,
        CANDIDATE_FEATURES as _COMP_CANDIDATE_FEATURES,
        OBSERVED_GROUP_FEATURES as _COMP_GROUP_FEATURES,
        CompetitionAssociationController,
    )
    _COMPETITION_ASSOC_AVAILABLE = True
except Exception:
    _COMP_ACTION_LABELS = ("keep", "rerank", "null")
    _COMP_CANDIDATE_FEATURES = ()
    _COMP_GROUP_FEATURES = ()
    CompetitionAssociationController = None
    _COMPETITION_ASSOC_AVAILABLE = False

try:
    from models.local_conflict_graph_common import (
        build_topk_bipartite_components,
        filter_local_conflict_clusters_by_size,
        solve_assignment_with_private_defer,
        solve_assignment_with_private_null,
    )
    _LOCAL_CONFLICT_GRAPH_AVAILABLE = True
except Exception:
    build_topk_bipartite_components = None
    filter_local_conflict_clusters_by_size = None
    solve_assignment_with_private_defer = None
    solve_assignment_with_private_null = None
    _LOCAL_CONFLICT_GRAPH_AVAILABLE = False

try:
    from models.local_conflict_commit import LocalConflictCommitRefiner
    _LOCAL_CONFLICT_COMMIT_AVAILABLE = True
except Exception:
    LocalConflictCommitRefiner = None
    _LOCAL_CONFLICT_COMMIT_AVAILABLE = False

try:
    from models.local_conflict_set_predictor import (
        FEATURE_VERSION as _LOCAL_CONFLICT_SET_PREDICTOR_FEATURE_VERSION,
        HostConditionedLocalConflictSetPredictor,
        encode_host_variant as _encode_local_conflict_host_variant,
        normalize_host_vocab as _normalize_local_conflict_host_vocab,
        pair_geometry_features as _local_conflict_pair_geometry_features,
        softmax_probs_1d as _local_conflict_softmax_probs_1d,
        zscore_1d as _local_conflict_zscore_1d,
    )
    _LOCAL_CONFLICT_SET_PREDICTOR_AVAILABLE = True
except Exception:
    _LOCAL_CONFLICT_SET_PREDICTOR_FEATURE_VERSION = "v2_hostnorm_geom"
    HostConditionedLocalConflictSetPredictor = None
    _encode_local_conflict_host_variant = None
    _normalize_local_conflict_host_vocab = None
    _local_conflict_pair_geometry_features = None
    _local_conflict_softmax_probs_1d = None
    _local_conflict_zscore_1d = None
    _LOCAL_CONFLICT_SET_PREDICTOR_AVAILABLE = False

try:
    from yolox.motdt_tracker.kalman_filter import KalmanFilter as ByteTrackKalman
    _KALMAN_AVAILABLE = True
except Exception:
    # Fallback: allow importing ByteTrack/YOLOX without requiring an editable install.
    # This mirrors the logic in ByteTrackFeatureExtractor and is important because this
    # module is imported before we construct the feature extractor (which may patch sys.path).
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bytetrack_root = os.path.join(repo_root, "third_party", "ByteTrack")
    if os.path.isdir(bytetrack_root) and bytetrack_root not in sys.path:
        sys.path.insert(0, bytetrack_root)
    try:
        from yolox.motdt_tracker.kalman_filter import KalmanFilter as ByteTrackKalman
        _KALMAN_AVAILABLE = True
    except Exception:
        ByteTrackKalman = None
        _KALMAN_AVAILABLE = False


_linear_sum_assignment = None


def _get_linear_sum_assignment():
    global _linear_sum_assignment
    if _linear_sum_assignment is None:
        try:
            from scipy.optimize import linear_sum_assignment
            _linear_sum_assignment = linear_sum_assignment
        except Exception:
            _linear_sum_assignment = None
    return _linear_sum_assignment


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2 - inter + 1e-6
    return inter / union


def _score_entropy_from_probs(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() == 0:
        return torch.zeros((0,), device=scores.device, dtype=scores.dtype)
    clipped = scores.clamp(min=1e-4, max=1.0 - 1e-4)
    logits = torch.logit(clipped, eps=1e-4)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    prob = torch.softmax(logits, dim=-1)
    return -(prob * torch.log(prob.clamp(min=1e-8))).sum(dim=-1)


def _cxcywh_to_xyah(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(dim=-1)
    a = w / (h + 1e-6)
    return torch.stack([cx, cy, a, h], dim=-1)


def _xyah_to_cxcywh(xyah: torch.Tensor) -> torch.Tensor:
    x, y, a, h = xyah.unbind(dim=-1)
    w = a * h
    return torch.stack([x, y, w, h], dim=-1)
from utils.box_ops import box_cxcywh_to_xywh, box_xywh_to_cxcywh
from models.misc import get_model
from models.bytetrack_feature_extractor import (
    ByteTrackFeatureConfig,
    ByteTrackFeatureExtractor,
)

# Optional advanced strategies
try:
    from models.motip.advanced_strategies import FrequencyAwareConfidenceCalibration, TestTimeAugmentation
    _ADV_STRATEGIES_AVAILABLE = True
except Exception:
    FrequencyAwareConfidenceCalibration = None
    TestTimeAugmentation = None
    _ADV_STRATEGIES_AVAILABLE = False

# Optional memory bank (Top-Conference strategy)
try:
    from models.motip.topconf_losses import build_memory_bank
    _MEMORY_AVAILABLE = True
except Exception:
    build_memory_bank = None
    _MEMORY_AVAILABLE = False


class RuntimeTrackerByteTrack:
    """
    使用 ByteTrack 特征的运行时跟踪器

    整个流程：
    1. ByteTrack 检测目标并提取特征
    2. 使用频域轨迹建模进行时序建模
    3. 使用 ID 解码器进行身份关联
    """

    def __init__(
        self,
        trajectory_modeling: torch.nn.Module,
        id_decoder: torch.nn.Module,
        feature_extractor: ByteTrackFeatureExtractor,
        sequence_hw: Tuple[int, int],
        # 推理设置
        miss_tolerance: int = 30,
        det_thresh: float = 0.3,
        det_max_per_frame: int = 0,
        newborn_thresh: float = 0.5,
        id_thresh: float = 0.1,
        area_thresh: int = 0,
        num_id_vocabulary: int = 500,
        feature_dim: int = 256,
        track_window: int = 30,
        matching_method: str = "hungarian",
        assoc_iou_gate: float = 0.0,
        assoc_id_weight: float = 1.0,
        assoc_iou_weight: float = 0.0,
        assoc_logit_temp: float = 1.0,
        assoc_use_det_score: bool = False,
        assoc_mode: str = "logit",
        assoc_feat_weight: float = 1.0,
        assoc_feat_agg: str = "last",
        assoc_feat_k: int = 5,
        assoc_feat_tau: float = 1.0,
        assoc_feat_source: str = "yolox",
        assoc_feat_score_mode: str = "raw",
        assoc_freq_gate: bool = False,
        assoc_freq_gate_min: float = 0.2,
        assoc_freq_gate_max: float = 1.0,
        assoc_use_laplace: bool = False,
        assoc_laplace_weight: float = 0.35,
        assoc_laplace_decay_scales: Optional[List[float]] = None,
        assoc_laplace_hidden_dim: int = 16,
        assoc_use_mtcr: bool = False,
        assoc_mtcr_checkpoint: str = "",
        assoc_mtcr_hist_hidden: int = 16,
        assoc_mtcr_comp_hidden: int = 64,
        assoc_mtcr_topk: int = 3,
        assoc_mtcr_margin_threshold: float = 0.10,
        assoc_mtcr_margin_temperature: float = 0.03,
        assoc_mtcr_delta_scale: float = 1.0,
        assoc_mtcr_min_history: int = 3,
        assoc_mtcr_decay_scales: Optional[List[float]] = None,
        assoc_use_runtime_replay: bool = False,
        assoc_runtime_replay_checkpoint: str = "",
        assoc_runtime_replay_hard_margin_gate: bool = False,
        assoc_runtime_replay_margin_threshold: Optional[float] = None,
        assoc_use_competition: bool = False,
        assoc_competition_checkpoint: str = "",
        assoc_competition_topk: int = 3,
        assoc_competition_delta_scale: float = 0.05,
        assoc_competition_mode: str = "rerank_only",
        assoc_competition_hard_action: bool = True,
        assoc_competition_margin_threshold: Optional[float] = None,
        assoc_use_competition_oracle: bool = False,
        assoc_competition_oracle_csv: str = "",
        assoc_use_local_conflict_graph: bool = False,
        assoc_local_conflict_graph_mode: str = "disabled",
        assoc_use_local_conflict_graph_oracle: bool = False,
        assoc_local_conflict_graph_oracle_jsonl: str = "",
        assoc_local_conflict_graph_checkpoint: str = "",
        assoc_local_conflict_graph_topk: int = 8,
        assoc_local_conflict_graph_min_detections: int = 2,
        assoc_local_conflict_graph_min_committed_matches: int = 1,
        assoc_local_conflict_graph_max_detections: int = 8,
        assoc_local_conflict_graph_max_tracks: int = 32,
        assoc_local_conflict_graph_cluster_gate_thresh: float = 0.5,
        assoc_local_conflict_graph_cluster_gate_temp: float = 1.0,
        assoc_local_conflict_graph_cluster_gate_bias: float = 0.0,
        assoc_local_conflict_graph_host_variant: str = "",
        assoc_bbox_dist_weight: float = 0.0,
        assoc_bbox_dist_tau: float = 1.0,
        assoc_bbox_dist_use_cal_factor: bool = False,
        assoc_two_stage: bool = False,
        assoc_stage2_iou_gate: Optional[float] = None,
        assoc_stage2_id_thresh: Optional[float] = None,
        assoc_stage2_bbox_weight: Optional[float] = None,
        reid_encoder: Optional[Any] = None,
        use_kalman: bool = True,
        dtype: torch.dtype = torch.float32,
        id_label_strategy: str = "random",
        sequence_name: Optional[str] = None,
        # Advanced inference options
        use_confidence_calibration: bool = False,
        calibration_strength: float = 0.5,
        min_confidence: float = 0.1,
        use_tta: bool = False,
        tta_scales: Optional[List[float]] = None,
        tta_flip: bool = True,
        tta_fusion: str = "average",
        use_memory_bank: bool = False,
        memory_lambda: float = 0.9,
        memory_update_threshold: float = 0.5,
        det_source: str = "model",
        external_detections: Optional[Dict[int, List[Tuple[float, float, float, float, float]]]] = None,
        assoc_runtime_dump_path: str = "",
        assoc_runtime_dump_topk: int = 8,
        assoc_runtime_dump_min_score: float = 0.0,
        assoc_runtime_dump_save_tensors: bool = False,
        assoc_runtime_dump_npz_every_n_groups: int = 2048,
    ):
        self.trajectory_modeling = trajectory_modeling
        self.id_decoder = id_decoder
        self.feature_extractor = feature_extractor

        self.trajectory_modeling.eval()
        self.id_decoder.eval()
        self.feature_extractor.eval()

        self.dtype = dtype
        self.sequence_hw = sequence_hw
        self.miss_tolerance = miss_tolerance
        self.det_thresh = det_thresh
        self.det_max_per_frame = int(det_max_per_frame) if det_max_per_frame is not None else 0
        self.newborn_thresh = newborn_thresh
        self.id_thresh = id_thresh
        self.area_thresh = area_thresh
        self.num_id_vocabulary = num_id_vocabulary
        self.feature_dim = feature_dim
        self.track_window = max(int(track_window), 1)
        det_source = str(det_source).lower()
        if det_source in ("txt", "public"):
            det_source = "external"
        if det_source not in ("model", "external"):
            raise ValueError(f"Unsupported det_source: {det_source}")
        if det_source == "external" and external_detections is None:
            raise RuntimeError("det_source=external but external_detections is None")
        self.det_source = det_source
        self.external_detections = external_detections if external_detections is not None else {}
        self.last_dets: List[Tuple[float, float, float, float, float]] = []
        self.assoc_runtime_dump_path = str(assoc_runtime_dump_path or "")
        # `0` means dump the full gated candidate set for each detection, which is useful for
        # replay recoverability analysis. Positive values keep only the top-k rows by host score.
        self.assoc_runtime_dump_topk = max(int(assoc_runtime_dump_topk), 0)
        self.assoc_runtime_dump_min_score = float(assoc_runtime_dump_min_score)
        self.assoc_runtime_dump_save_tensors = bool(assoc_runtime_dump_save_tensors)
        self.assoc_runtime_dump_npz_every_n_groups = max(int(assoc_runtime_dump_npz_every_n_groups), 1)
        self._assoc_dump_file = None
        self._assoc_dump_writer = None
        self._assoc_tensor_buffer = self._new_assoc_tensor_buffer()
        self._assoc_tensor_shard_index = 0
        self.matching_method = str(matching_method).lower()
        self.assoc_iou_gate = float(assoc_iou_gate)
        self.assoc_id_weight = float(assoc_id_weight)
        self.assoc_iou_weight = float(assoc_iou_weight)
        self.assoc_logit_temp = float(assoc_logit_temp)
        self.assoc_use_det_score = bool(assoc_use_det_score)
        self.assoc_mode = str(assoc_mode).lower()
        self.assoc_feat_weight = float(assoc_feat_weight)
        self.assoc_feat_agg = str(assoc_feat_agg).lower()
        self.assoc_feat_k = max(int(assoc_feat_k), 1)
        self.assoc_feat_tau = float(assoc_feat_tau)
        self.assoc_feat_source = str(assoc_feat_source).lower()
        if self.assoc_feat_source not in ("yolox", "reid"):
            warnings.warn(
                f"[RuntimeTrackerByteTrack] Unsupported ASSOC_FEAT_SOURCE={assoc_feat_source}; fallback to 'yolox'."
            )
            self.assoc_feat_source = "yolox"
        self.assoc_feat_score_mode = str(assoc_feat_score_mode).lower()
        if self.assoc_feat_score_mode not in ("raw", "softmax"):
            warnings.warn(
                f"[RuntimeTrackerByteTrack] Unsupported ASSOC_FEAT_SCORE_MODE={assoc_feat_score_mode}; fallback to 'raw'."
            )
            self.assoc_feat_score_mode = "raw"
        self.assoc_freq_gate = bool(assoc_freq_gate)
        self.assoc_freq_gate_min = float(assoc_freq_gate_min)
        self.assoc_freq_gate_max = float(assoc_freq_gate_max)
        self.assoc_use_laplace = bool(assoc_use_laplace) and _LAPLACE_ASSOC_AVAILABLE
        self.assoc_laplace_weight = float(assoc_laplace_weight)
        self.assoc_laplace_decay_scales = list(assoc_laplace_decay_scales or [1.0, 2.0, 4.0])
        self.assoc_laplace_hidden_dim = int(assoc_laplace_hidden_dim)
        self.laplace_assoc = None
        if self.assoc_use_laplace:
            try:
                self.laplace_assoc = LaplaceAssociationAdapter(
                    decay_scales=self.assoc_laplace_decay_scales,
                    hidden_dim=self.assoc_laplace_hidden_dim,
                    blend=self.assoc_laplace_weight,
                )
            except Exception as exc:
                warnings.warn(f"[RuntimeTrackerByteTrack] Failed to init Laplace association; disabled. Error: {exc}")
                self.assoc_use_laplace = False
                self.laplace_assoc = None

        self.assoc_use_mtcr = bool(assoc_use_mtcr) and _MTCR_ASSOC_AVAILABLE
        self.assoc_mtcr_checkpoint = str(assoc_mtcr_checkpoint or "")
        self.assoc_mtcr_hist_hidden = int(assoc_mtcr_hist_hidden)
        self.assoc_mtcr_comp_hidden = int(assoc_mtcr_comp_hidden)
        self.assoc_mtcr_topk = int(max(assoc_mtcr_topk, 1))
        self.assoc_mtcr_margin_threshold = float(assoc_mtcr_margin_threshold)
        self.assoc_mtcr_margin_temperature = float(max(assoc_mtcr_margin_temperature, 1e-4))
        self.assoc_mtcr_delta_scale = float(assoc_mtcr_delta_scale)
        self.assoc_mtcr_min_history = int(max(assoc_mtcr_min_history, 1))
        self.assoc_mtcr_decay_scales = list(assoc_mtcr_decay_scales or [1.0, 2.0, 4.0])
        self.mtcr_assoc = None
        if self.assoc_use_mtcr:
            try:
                if self.assoc_mtcr_checkpoint:
                    self.mtcr_assoc = MTCRAssociationAdapter.from_npz(self.assoc_mtcr_checkpoint)
                else:
                    self.mtcr_assoc = MTCRAssociationAdapter(
                        hist_hidden=self.assoc_mtcr_hist_hidden,
                        comp_hidden=self.assoc_mtcr_comp_hidden,
                        topk=self.assoc_mtcr_topk,
                        margin_threshold=self.assoc_mtcr_margin_threshold,
                        margin_temperature=self.assoc_mtcr_margin_temperature,
                        delta_scale=self.assoc_mtcr_delta_scale,
                        min_history=self.assoc_mtcr_min_history,
                        decay_scales=self.assoc_mtcr_decay_scales,
                    )
                    warnings.warn(
                        "[RuntimeTrackerByteTrack] ASSOC_USE_MTCR=True but ASSOC_MTCR_CHECKPOINT is empty; "
                        "using safe no-op initialization."
                    )
                self.mtcr_assoc = self.mtcr_assoc.to(distributed_device())
                self.mtcr_assoc.eval()
            except Exception as exc:
                warnings.warn(f"[RuntimeTrackerByteTrack] Failed to init MTCR association; disabled. Error: {exc}")
                self.assoc_use_mtcr = False
                self.mtcr_assoc = None

        self.assoc_use_runtime_replay = bool(assoc_use_runtime_replay) and _RUNTIME_REPLAY_ASSOC_AVAILABLE
        self.assoc_runtime_replay_checkpoint = str(assoc_runtime_replay_checkpoint or "")
        self.assoc_runtime_replay_hard_margin_gate = bool(assoc_runtime_replay_hard_margin_gate)
        self.assoc_runtime_replay_margin_threshold = (
            float(assoc_runtime_replay_margin_threshold)
            if assoc_runtime_replay_margin_threshold is not None
            else None
        )
        self.runtime_replay_assoc = None
        if self.assoc_use_runtime_replay:
            try:
                if self.assoc_runtime_replay_checkpoint:
                    self.runtime_replay_assoc = RuntimeReplayAssociationAdapter.from_checkpoint(
                        self.assoc_runtime_replay_checkpoint
                    )
                else:
                    self.runtime_replay_assoc = RuntimeReplayAssociationAdapter()
                    warnings.warn(
                        "[RuntimeTrackerByteTrack] ASSOC_USE_RUNTIME_REPLAY=True but "
                        "ASSOC_RUNTIME_REPLAY_CHECKPOINT is empty; using safe no-op initialization."
                    )
                self.runtime_replay_assoc = self.runtime_replay_assoc.to(distributed_device())
                self.runtime_replay_assoc.eval()
            except Exception as exc:
                warnings.warn(
                    f"[RuntimeTrackerByteTrack] Failed to init runtime replay association; disabled. Error: {exc}"
                )
                self.assoc_use_runtime_replay = False
                self.runtime_replay_assoc = None

        self.assoc_use_competition = bool(assoc_use_competition) and _COMPETITION_ASSOC_AVAILABLE
        self.assoc_competition_checkpoint = str(assoc_competition_checkpoint or "")
        self.assoc_competition_topk = int(max(assoc_competition_topk, 1))
        self.assoc_competition_delta_scale = float(max(assoc_competition_delta_scale, 0.0))
        self.assoc_competition_mode = str(assoc_competition_mode or "rerank_only").lower()
        self.assoc_competition_hard_action = bool(assoc_competition_hard_action)
        self.assoc_competition_margin_threshold = (
            float(assoc_competition_margin_threshold)
            if assoc_competition_margin_threshold is not None
            else None
        )
        self.assoc_use_competition_oracle = bool(assoc_use_competition_oracle)
        self.assoc_competition_oracle_csv = str(assoc_competition_oracle_csv or "")
        self.competition_oracle = {}
        if self.assoc_use_competition_oracle:
            try:
                if not self.assoc_competition_oracle_csv:
                    raise RuntimeError("ASSOC_COMPETITION_ORACLE_CSV is empty")
                self.competition_oracle = self._load_competition_oracle(self.assoc_competition_oracle_csv)
            except Exception as exc:
                warnings.warn(
                    f"[RuntimeTrackerByteTrack] Failed to load competition oracle; disabled. Error: {exc}"
                )
                self.assoc_use_competition_oracle = False
                self.competition_oracle = {}
        self.competition_assoc = None
        if self.assoc_use_competition and not self.assoc_use_competition_oracle:
            try:
                if not self.assoc_competition_checkpoint:
                    raise RuntimeError("ASSOC_COMPETITION_CHECKPOINT is empty")
                self.competition_assoc = CompetitionAssociationController.from_checkpoint(
                    self.assoc_competition_checkpoint,
                    map_location=distributed_device(),
                )
                self.competition_assoc = self.competition_assoc.to(distributed_device())
                self.competition_assoc.eval()
            except Exception as exc:
                warnings.warn(
                    f"[RuntimeTrackerByteTrack] Failed to init competition association; disabled. Error: {exc}"
                )
                self.assoc_use_competition = False
                self.competition_assoc = None

        self.assoc_use_local_conflict_graph = bool(assoc_use_local_conflict_graph) and _LOCAL_CONFLICT_GRAPH_AVAILABLE
        self.assoc_local_conflict_graph_mode = str(assoc_local_conflict_graph_mode or "disabled").lower()
        if self.assoc_local_conflict_graph_mode not in (
            "disabled",
            "oracle_full",
            "oracle_commit_matches",
            "learned_commit",
        ):
            warnings.warn(
                f"[RuntimeTrackerByteTrack] Unsupported ASSOC_LOCAL_CONFLICT_GRAPH_MODE={assoc_local_conflict_graph_mode}; "
                "fallback to disabled."
            )
            self.assoc_local_conflict_graph_mode = "disabled"
        self.assoc_use_local_conflict_graph_oracle = bool(assoc_use_local_conflict_graph_oracle)
        self.assoc_local_conflict_graph_oracle_jsonl = str(assoc_local_conflict_graph_oracle_jsonl or "")
        self.assoc_local_conflict_graph_checkpoint = str(assoc_local_conflict_graph_checkpoint or "")
        self.assoc_local_conflict_graph_topk = max(int(assoc_local_conflict_graph_topk), 1)
        self.assoc_local_conflict_graph_min_detections = max(int(assoc_local_conflict_graph_min_detections), 2)
        self.assoc_local_conflict_graph_min_committed_matches = max(
            int(assoc_local_conflict_graph_min_committed_matches),
            1,
        )
        self.assoc_local_conflict_graph_max_detections = max(int(assoc_local_conflict_graph_max_detections), 0)
        self.assoc_local_conflict_graph_max_tracks = max(int(assoc_local_conflict_graph_max_tracks), 0)
        self.assoc_local_conflict_graph_cluster_gate_thresh = float(assoc_local_conflict_graph_cluster_gate_thresh)
        self.assoc_local_conflict_graph_cluster_gate_temp = float(assoc_local_conflict_graph_cluster_gate_temp)
        self.assoc_local_conflict_graph_cluster_gate_bias = float(assoc_local_conflict_graph_cluster_gate_bias)
        self.assoc_local_conflict_graph_host_variant = str(assoc_local_conflict_graph_host_variant or "").strip()
        self.local_conflict_graph_oracle = {}
        self.local_conflict_commit_model = None
        self.local_conflict_model_family = "mlp_commit_v1"
        self.local_conflict_feature_version = "v1_raw"
        self.local_conflict_host_vocab = ["unknown"]
        self._local_conflict_graph_last_stats: Dict[str, Any] = {}
        need_local_graph_oracle = bool(
            self.assoc_use_local_conflict_graph_oracle
            or (
                self.assoc_use_local_conflict_graph
                and self.assoc_local_conflict_graph_mode in ("oracle_full", "oracle_commit_matches")
            )
        )
        if self.assoc_use_local_conflict_graph and not _LOCAL_CONFLICT_GRAPH_AVAILABLE:
            warnings.warn(
                "[RuntimeTrackerByteTrack] ASSOC_USE_LOCAL_CONFLICT_GRAPH=True but local_conflict_graph_common "
                "is unavailable; disabled."
            )
            self.assoc_use_local_conflict_graph = False
            self.assoc_local_conflict_graph_mode = "disabled"
        if need_local_graph_oracle:
            try:
                if not self.assoc_local_conflict_graph_oracle_jsonl:
                    raise RuntimeError("ASSOC_LOCAL_CONFLICT_GRAPH_ORACLE_JSONL is empty")
                self.local_conflict_graph_oracle = self._load_local_conflict_graph_oracle(
                    self.assoc_local_conflict_graph_oracle_jsonl
                )
            except Exception as exc:
                warnings.warn(
                    f"[RuntimeTrackerByteTrack] Failed to load local conflict graph oracle; disabled. Error: {exc}"
                )
                self.assoc_use_local_conflict_graph = False
                self.assoc_local_conflict_graph_mode = "disabled"
                self.assoc_use_local_conflict_graph_oracle = False
                self.local_conflict_graph_oracle = {}
        if self.assoc_use_local_conflict_graph and self.assoc_local_conflict_graph_mode == "learned_commit":
            try:
                if not self.assoc_local_conflict_graph_checkpoint:
                    raise RuntimeError("ASSOC_LOCAL_CONFLICT_GRAPH_CHECKPOINT is empty")
                checkpoint_meta = torch.load(
                    self.assoc_local_conflict_graph_checkpoint,
                    map_location="cpu",
                )
                model_family = str(checkpoint_meta.get("model_family", "mlp_commit_v1")).strip() or "mlp_commit_v1"
                feature_version = str(
                    checkpoint_meta.get(
                        "feature_version",
                        _LOCAL_CONFLICT_SET_PREDICTOR_FEATURE_VERSION
                        if model_family == "set_predictor_v2"
                        else "v1_raw",
                    )
                ).strip()
                if model_family == "set_predictor_v2":
                    if not _LOCAL_CONFLICT_SET_PREDICTOR_AVAILABLE:
                        raise RuntimeError("models.local_conflict_set_predictor is unavailable")
                    self.local_conflict_commit_model = HostConditionedLocalConflictSetPredictor.from_checkpoint(
                        self.assoc_local_conflict_graph_checkpoint,
                        map_location=distributed_device(),
                    )
                    if _normalize_local_conflict_host_vocab is not None:
                        self.local_conflict_host_vocab = _normalize_local_conflict_host_vocab(
                            checkpoint_meta.get("host_vocab", ["unknown"])
                        )
                else:
                    if not _LOCAL_CONFLICT_COMMIT_AVAILABLE:
                        raise RuntimeError("models.local_conflict_commit is unavailable")
                    self.local_conflict_commit_model = LocalConflictCommitRefiner.from_checkpoint(
                        self.assoc_local_conflict_graph_checkpoint,
                        map_location=distributed_device(),
                    )
                    self.local_conflict_host_vocab = ["unknown"]
                self.local_conflict_model_family = model_family
                self.local_conflict_feature_version = feature_version or "v1_raw"
                self.local_conflict_commit_model = self.local_conflict_commit_model.to(distributed_device())
                self.local_conflict_commit_model.eval()
            except Exception as exc:
                warnings.warn(
                    f"[RuntimeTrackerByteTrack] Failed to init learned local conflict commit model; disabled. Error: {exc}"
                )
                self.assoc_use_local_conflict_graph = False
                self.assoc_local_conflict_graph_mode = "disabled"
                self.local_conflict_commit_model = None
                self.local_conflict_model_family = "mlp_commit_v1"
                self.local_conflict_feature_version = "v1_raw"
                self.local_conflict_host_vocab = ["unknown"]

        self.assoc_bbox_dist_weight = float(assoc_bbox_dist_weight)
        self.assoc_bbox_dist_tau = float(assoc_bbox_dist_tau)
        self.assoc_bbox_dist_use_cal_factor = bool(assoc_bbox_dist_use_cal_factor)
        if not (self.assoc_bbox_dist_tau > 0.0):
            warnings.warn(
                f"[RuntimeTrackerByteTrack] Invalid ASSOC_BBOX_DIST_TAU={self.assoc_bbox_dist_tau}; fallback to 1.0."
            )
            self.assoc_bbox_dist_tau = 1.0

        # Optional two-stage matching (opt-in): stage2 uses bbox-distance to recover missed matches.
        self.assoc_two_stage = bool(assoc_two_stage)
        self.assoc_stage2_iou_gate = float(assoc_stage2_iou_gate) if assoc_stage2_iou_gate is not None else None
        self.assoc_stage2_id_thresh = float(assoc_stage2_id_thresh) if assoc_stage2_id_thresh is not None else None
        self.assoc_stage2_bbox_weight = (
            float(assoc_stage2_bbox_weight) if assoc_stage2_bbox_weight is not None else None
        )
        self._warned_stage2_defaults = False
        self.reid_encoder = reid_encoder
        self.assoc_feature_dim = int(getattr(reid_encoder, "out_dim", feature_dim)) if reid_encoder is not None else int(feature_dim)
        self._hungarian_available = None
        self.use_kalman = bool(use_kalman) and _KALMAN_AVAILABLE
        self.kf = ByteTrackKalman() if self.use_kalman else None
        if use_kalman and not _KALMAN_AVAILABLE:
            warnings.warn("[RuntimeTrackerByteTrack] KalmanFilter not available, disabled.")

        # Advanced strategies
        self.use_confidence_calibration = bool(use_confidence_calibration) and _ADV_STRATEGIES_AVAILABLE
        if self.use_confidence_calibration:
            self.calibrator = FrequencyAwareConfidenceCalibration(
                num_bands=int(getattr(id_decoder, "num_bands", 4)),
                calibration_strength=calibration_strength,
                min_confidence=min_confidence,
            ).to(distributed_device())
        else:
            self.calibrator = None

        self.use_tta = bool(use_tta) and _ADV_STRATEGIES_AVAILABLE
        if self.use_tta:
            scales = tta_scales if tta_scales is not None else [0.8, 1.0, 1.2]
            self.tta = TestTimeAugmentation(scales=scales, flip=tta_flip, fusion_method=tta_fusion)
        else:
            self.tta = None

        self.use_memory_bank = bool(use_memory_bank) and _MEMORY_AVAILABLE
        if self.use_memory_bank:
            self.memory_bank = build_memory_bank({
                "FEATURE_DIM": feature_dim,
                "MEMORY_LAMBDA": memory_lambda,
                "MEMORY_UPDATE_THRESHOLD": memory_update_threshold,
            }).to(distributed_device())
        else:
            self.memory_bank = None
            if use_memory_bank and not _MEMORY_AVAILABLE:
                warnings.warn("[RuntimeTrackerByteTrack] MemoryBank not available, disabled.")

        # 跟踪状态
        self.frame_id = 0
        self.active_ids = OrderedSet()
        self.trajectory_infos = {}  # id -> trajectory info (history)
        self.max_id = 0
        self.id_label_strategy = str(id_label_strategy).lower()
        self.sequence_name = sequence_name
        self._seq_perm = None
        self.track_vocab_map = {}
        self.available_vocab = list(range(int(self.num_id_vocabulary)))
        self.available_vocab_set = set(self.available_vocab)
        if self.id_label_strategy != "random":
            seq_key = self._get_sequence_key()
            if seq_key is not None:
                self._seq_perm = self._build_sequence_perm(seq_key)

        # 结果
        self.current_track_results = None

        # 归一化因子
        h, w = sequence_hw
        self.bbox_unnorm = torch.tensor([w, h, w, h], dtype=dtype, device=distributed_device())
        self._local_conflict_graph_cumulative_stats = self._new_local_conflict_graph_cumulative_stats()

    def _new_local_conflict_graph_cumulative_stats(self) -> Dict[str, Any]:
        if self.assoc_use_local_conflict_graph:
            graph_mode = str(self.assoc_local_conflict_graph_mode or "disabled")
        elif self.assoc_use_local_conflict_graph_oracle:
            graph_mode = "legacy_partial_oracle"
        else:
            graph_mode = "disabled"
        return {
            "graph_mode": graph_mode,
            "frames_seen": 0,
            "frames_with_eligible_clusters": 0,
            "frames_with_replaced_clusters": 0,
            "eligible_clusters": 0,
            "replaced_clusters": 0,
            "resolved_dets": 0,
            "matched_dets": 0,
            "null_dets": 0,
            "deferred_dets": 0,
            "blocked_tracks": 0,
            "gate_pass_clusters": 0,
            "gate_filtered_clusters": 0,
            "trigger_filtered_clusters": 0,
            "skipped_large_clusters": 0,
        }

    def _accumulate_local_conflict_graph_stats(self, stats: Optional[Dict[str, Any]]) -> None:
        if not stats:
            return
        accum = self._local_conflict_graph_cumulative_stats
        accum["frames_seen"] += 1
        if int(stats.get("eligible_clusters", 0) or 0) > 0:
            accum["frames_with_eligible_clusters"] += 1
        if int(stats.get("replaced_clusters", 0) or 0) > 0:
            accum["frames_with_replaced_clusters"] += 1
        for key in (
            "eligible_clusters",
            "replaced_clusters",
            "resolved_dets",
            "matched_dets",
            "null_dets",
            "deferred_dets",
            "blocked_tracks",
            "gate_pass_clusters",
            "gate_filtered_clusters",
            "trigger_filtered_clusters",
            "skipped_large_clusters",
        ):
            accum[key] += int(stats.get(key, 0) or 0)

    def get_local_conflict_graph_diagnostics(self) -> Dict[str, Any]:
        return dict(self._local_conflict_graph_cumulative_stats)

    def _new_assoc_tensor_buffer(self) -> dict[str, list]:
        return {
            "group_ids": [],
            "group_offsets": [0],
            "det_feat": [],
            "det_box": [],
            "det_score": [],
            "cand_track_rank": [],
            "cand_track_id": [],
            "cand_hist_feat": [],
            "cand_hist_mask": [],
            "cand_hist_time": [],
            "cand_track_box": [],
        }

    def _assoc_tensor_dir(self) -> str:
        dump_root = str(self.assoc_runtime_dump_path or "")
        seq_name = str(self.sequence_name or "unknown_seq")
        if dump_root.lower().endswith(".csv"):
            parent = os.path.dirname(dump_root)
            stem = os.path.splitext(os.path.basename(dump_root))[0]
            return os.path.join(parent, f"{stem}_tensor_shards", seq_name)
        return os.path.join(dump_root, "tensor_shards", seq_name)

    def _restore_assoc_tensor_shard_index(self) -> None:
        if not self.assoc_runtime_dump_save_tensors:
            self._assoc_tensor_shard_index = 0
            return
        out_dir = self._assoc_tensor_dir()
        next_index = 0
        try:
            if os.path.isdir(out_dir):
                for name in os.listdir(out_dir):
                    if not (name.startswith("runtime_tensor_shard_") and name.endswith(".npz")):
                        continue
                    stem = os.path.splitext(name)[0]
                    suffix = stem.rsplit("_", 1)[-1]
                    if suffix.isdigit():
                        next_index = max(next_index, int(suffix) + 1)
        except Exception:
            # Resume support is best-effort. On any filesystem parsing issue, fall back to
            # the current in-memory index to avoid crashing inference.
            return
        self._assoc_tensor_shard_index = int(next_index)

    def _flush_assoc_tensor_shard(self, force: bool = False) -> None:
        if not self.assoc_runtime_dump_save_tensors:
            return
        if len(self._assoc_tensor_buffer["group_ids"]) == 0:
            return
        if not force and len(self._assoc_tensor_buffer["group_ids"]) < self.assoc_runtime_dump_npz_every_n_groups:
            return
        try:
            out_dir = self._assoc_tensor_dir()
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"runtime_tensor_shard_{self._assoc_tensor_shard_index:05d}.npz")
            payload = {
                "group_ids": np.asarray(self._assoc_tensor_buffer["group_ids"], dtype=object),
                "group_offsets": np.asarray(self._assoc_tensor_buffer["group_offsets"], dtype=np.int64),
                "det_feat": np.stack(self._assoc_tensor_buffer["det_feat"], axis=0).astype(np.float16),
                "det_box": np.stack(self._assoc_tensor_buffer["det_box"], axis=0).astype(np.float16),
                "det_score": np.asarray(self._assoc_tensor_buffer["det_score"], dtype=np.float32),
                "cand_track_rank": np.asarray(self._assoc_tensor_buffer["cand_track_rank"], dtype=np.int16),
                "cand_track_id": np.asarray(self._assoc_tensor_buffer["cand_track_id"], dtype=np.int32),
                "cand_hist_feat": np.stack(self._assoc_tensor_buffer["cand_hist_feat"], axis=0).astype(np.float16),
                "cand_hist_mask": np.stack(self._assoc_tensor_buffer["cand_hist_mask"], axis=0).astype(np.uint8),
                "cand_hist_time": np.stack(self._assoc_tensor_buffer["cand_hist_time"], axis=0).astype(np.int32),
                "cand_track_box": np.stack(self._assoc_tensor_buffer["cand_track_box"], axis=0).astype(np.float16),
            }
            np.savez_compressed(out_path, **payload)
            self._assoc_tensor_shard_index += 1
            self._assoc_tensor_buffer = self._new_assoc_tensor_buffer()
        except Exception as exc:
            if not getattr(self, "_warned_assoc_tensor_flush_error", False):
                warnings.warn(
                    f"[RuntimeTrackerByteTrack] Failed to flush runtime tensor shard; "
                    f"subsequent replay training data may be incomplete. Error: {exc}"
                )
                self._warned_assoc_tensor_flush_error = True

    def _ensure_assoc_dump_writer(self) -> None:
        if not self.assoc_runtime_dump_path or self._assoc_dump_writer is not None:
            return
        self._restore_assoc_tensor_shard_index()
        dump_root = self.assoc_runtime_dump_path
        if dump_root.lower().endswith(".csv"):
            dump_path = dump_root
        else:
            seq_name = str(self.sequence_name or "unknown_seq")
            dump_path = os.path.join(dump_root, f"{seq_name}.csv")
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        self._assoc_dump_file = open(dump_path, "a", encoding="utf-8", newline="")
        self._assoc_dump_writer = csv.writer(self._assoc_dump_file)
        if os.path.getsize(dump_path) == 0:
            self._assoc_dump_writer.writerow(
                [
                    "seq",
                    "frame",
                    "assoc_mode",
                    "group_id",
                    "candidate_count_total",
                    "det_index",
                    "track_rank",
                    "track_id",
                    "is_selected",
                    "det_score",
                    "base_score",
                    "refined_score",
                    "motion_score",
                    "track_gap",
                    "track_hist_len",
                    "det_cx",
                    "det_cy",
                    "det_w",
                    "det_h",
                    "track_cx",
                    "track_cy",
                    "track_w",
                    "track_h",
                ]
            )
            self._assoc_dump_file.flush()

    def _maybe_dump_feature_assoc_candidates(
        self,
        base_scores: torch.Tensor,
        refined_scores: torch.Tensor,
        motion_scores: Optional[torch.Tensor],
        scores: torch.Tensor,
        active_ids_list: List[int],
        det_features: Optional[torch.Tensor],
        track_history_features: Optional[torch.Tensor],
        track_history_masks: torch.Tensor,
        track_history_times: Optional[torch.Tensor],
        det_boxes_cxcywh: Optional[torch.Tensor],
        track_boxes_cxcywh: Optional[torch.Tensor],
        id_labels: torch.Tensor,
    ) -> None:
        if not self.assoc_runtime_dump_path:
            return
        if base_scores.numel() == 0 or refined_scores.numel() == 0:
            return
        try:
            self._ensure_assoc_dump_writer()
            if self._assoc_dump_writer is None:
                return

            num_dets, num_tracks = refined_scores.shape
            hist_counts = (~track_history_masks).sum(dim=1).to(dtype=torch.long)
            track_gaps = torch.full((num_tracks,), -1, device=refined_scores.device, dtype=torch.long)
            if track_history_times is not None and track_history_times.numel() > 0:
                valid = ~track_history_masks
                last_idx = (
                    valid.to(dtype=torch.long)
                    * (torch.arange(track_history_times.shape[1], device=track_history_times.device).view(1, -1) + 1)
                ).max(dim=1).values - 1
                last_idx = last_idx.clamp(min=0)
                last_times = torch.gather(track_history_times, dim=1, index=last_idx.view(-1, 1)).squeeze(1)
                track_gaps = (track_history_times.new_full((track_history_times.shape[0],), int(self.frame_id)) - last_times).clamp(min=0)

            for det_idx in range(num_dets):
                row = refined_scores[det_idx]
                k = int(row.numel()) if self.assoc_runtime_dump_topk <= 0 else min(self.assoc_runtime_dump_topk, int(row.numel()))
                top_idx = torch.topk(row, k=k).indices.tolist()
                group_id = f"{self.sequence_name or ''}:{int(self.frame_id)}:{int(det_idx)}"
                candidate_count_total = int(row.numel())
                saved_count = 0
                group_track_ranks = []
                group_track_ids = []
                group_hist_feat = []
                group_hist_mask = []
                group_hist_time = []
                group_track_boxes = []
                for rank, track_idx in enumerate(top_idx, start=1):
                    refined_val = float(refined_scores[det_idx, track_idx].item())
                    if refined_val < self.assoc_runtime_dump_min_score:
                        continue
                    track_id = int(active_ids_list[track_idx])
                    selected = 1 if int(id_labels[det_idx].item()) == track_id else 0
                    det_box = det_boxes_cxcywh[det_idx] if det_boxes_cxcywh is not None else None
                    trk_box = track_boxes_cxcywh[track_idx] if track_boxes_cxcywh is not None else None
                    self._assoc_dump_writer.writerow(
                        [
                            str(self.sequence_name or ""),
                            int(self.frame_id),
                            "feature",
                            group_id,
                            candidate_count_total,
                            int(det_idx),
                            int(rank),
                            track_id,
                            selected,
                            float(scores[det_idx].item()) if scores is not None else 0.0,
                            float(base_scores[det_idx, track_idx].item()),
                            refined_val,
                            float(motion_scores[det_idx, track_idx].item()) if motion_scores is not None else 0.0,
                            int(track_gaps[track_idx].item()) if track_gaps.numel() > track_idx else -1,
                            int(hist_counts[track_idx].item()) if hist_counts.numel() > track_idx else 0,
                            float(det_box[0].item()) if det_box is not None else 0.0,
                            float(det_box[1].item()) if det_box is not None else 0.0,
                            float(det_box[2].item()) if det_box is not None else 0.0,
                            float(det_box[3].item()) if det_box is not None else 0.0,
                            float(trk_box[0].item()) if trk_box is not None else 0.0,
                            float(trk_box[1].item()) if trk_box is not None else 0.0,
                            float(trk_box[2].item()) if trk_box is not None else 0.0,
                            float(trk_box[3].item()) if trk_box is not None else 0.0,
                        ]
                    )
                    saved_count += 1
                    if self.assoc_runtime_dump_save_tensors and det_features is not None and track_history_features is not None:
                        group_track_ranks.append(int(rank))
                        group_track_ids.append(track_id)
                        group_hist_feat.append(track_history_features[track_idx].detach().to(dtype=torch.float16).cpu().numpy())
                        group_hist_mask.append(track_history_masks[track_idx].detach().to(dtype=torch.uint8).cpu().numpy())
                        if track_history_times is not None and track_history_times.numel() > 0:
                            group_hist_time.append(track_history_times[track_idx].detach().to(dtype=torch.int32).cpu().numpy())
                        else:
                            group_hist_time.append(np.zeros((track_history_masks.shape[1],), dtype=np.int32))
                        if trk_box is not None:
                            group_track_boxes.append(trk_box.detach().to(dtype=torch.float16).cpu().numpy())
                        else:
                            group_track_boxes.append(np.zeros((4,), dtype=np.float16))
                if self.assoc_runtime_dump_save_tensors and saved_count > 0 and det_features is not None and track_history_features is not None:
                    det_box = det_boxes_cxcywh[det_idx] if det_boxes_cxcywh is not None else None
                    self._assoc_tensor_buffer["group_ids"].append(group_id)
                    self._assoc_tensor_buffer["det_feat"].append(det_features[det_idx].detach().to(dtype=torch.float16).cpu().numpy())
                    if det_box is not None:
                        self._assoc_tensor_buffer["det_box"].append(det_box.detach().to(dtype=torch.float16).cpu().numpy())
                    else:
                        self._assoc_tensor_buffer["det_box"].append(np.zeros((4,), dtype=np.float16))
                    self._assoc_tensor_buffer["det_score"].append(float(scores[det_idx].item()) if scores is not None else 0.0)
                    self._assoc_tensor_buffer["cand_track_rank"].extend(group_track_ranks)
                    self._assoc_tensor_buffer["cand_track_id"].extend(group_track_ids)
                    self._assoc_tensor_buffer["cand_hist_feat"].extend(group_hist_feat)
                    self._assoc_tensor_buffer["cand_hist_mask"].extend(group_hist_mask)
                    self._assoc_tensor_buffer["cand_hist_time"].extend(group_hist_time)
                    self._assoc_tensor_buffer["cand_track_box"].extend(group_track_boxes)
                    self._assoc_tensor_buffer["group_offsets"].append(
                        int(self._assoc_tensor_buffer["group_offsets"][-1]) + int(saved_count)
                    )
                    self._flush_assoc_tensor_shard(force=False)
            self._assoc_dump_file.flush()
        except Exception as exc:
            if not getattr(self, "_warned_assoc_dump_error", False):
                warnings.warn(
                    f"[RuntimeTrackerByteTrack] Failed to dump runtime association candidates; "
                    f"runtime replay data may be incomplete. Error: {exc}"
                )
                self._warned_assoc_dump_error = True

    def reset(self):
        """重置跟踪器状态"""
        self.frame_id = 0
        self.active_ids = OrderedSet()
        self.trajectory_infos = {}
        self.max_id = 0
        self._seq_perm = None
        self.current_track_results = None
        self.track_vocab_map = {}
        self.available_vocab = list(range(int(self.num_id_vocabulary)))
        self.available_vocab_set = set(self.available_vocab)
        if self.id_label_strategy != "random":
            seq_key = self._get_sequence_key()
            if seq_key is not None:
                self._seq_perm = self._build_sequence_perm(seq_key)
        if self._assoc_dump_file is not None:
            try:
                self._assoc_dump_file.close()
            except Exception:
                pass
        self._assoc_dump_file = None
        self._assoc_dump_writer = None
        self._flush_assoc_tensor_shard(force=True)
        self._assoc_tensor_buffer = self._new_assoc_tensor_buffer()
        self._assoc_tensor_shard_index = 0
        self._local_conflict_graph_last_stats = {}
        self._local_conflict_graph_cumulative_stats = self._new_local_conflict_graph_cumulative_stats()

    def _get_sequence_key(self) -> Optional[str]:
        if not self.sequence_name:
            return None
        return str(self.sequence_name)

    def _build_sequence_perm(self, seq_key: str) -> torch.Tensor:
        seed = int(hashlib.md5(seq_key.encode("utf-8")).hexdigest(), 16) % (2**31 - 1)
        g = torch.Generator()
        g.manual_seed(seed)
        vocab = max(int(self.num_id_vocabulary), 1)
        return torch.randperm(vocab, generator=g)

    def _evict_one_track_for_vocab(self, avoid_track_id: Optional[int] = None) -> Optional[int]:
        """
        Evict one least-recently-seen track to free a vocab id.

        This is only used when `num_id_vocabulary` is exhausted. Reusing vocab ids without eviction
        will create collisions and can collapse association logits.
        """
        candidates = [tid for tid in list(self.active_ids) if tid != avoid_track_id]
        if not candidates:
            return None

        def _key(tid: int) -> Tuple[int, int, int]:
            info = self.trajectory_infos.get(tid, {})
            last_seen = int(info.get("last_seen", -1))
            miss_count = int(info.get("miss_count", 0))
            # Evict the oldest track; break ties by higher miss_count (more likely inactive).
            return (last_seen, -miss_count, int(tid))

        evict_id = min(candidates, key=_key)
        self.active_ids.discard(evict_id)
        self.trajectory_infos.pop(evict_id, None)
        self._release_vocab_id(evict_id)

        if not getattr(self, "_warned_vocab_eviction", False):
            warnings.warn(
                f"[RuntimeTrackerByteTrack] Vocab pool exhausted (num_id_vocabulary={self.num_id_vocabulary}); "
                "evicting least-recent tracks to keep vocab IDs unique."
            )
            self._warned_vocab_eviction = True
        return int(evict_id)

    def _get_or_assign_vocab_id(self, track_id: int) -> int:
        # Non-random strategies should still guarantee uniqueness within the active set.
        # Otherwise, different tracks may map to the same vocab index (e.g., when track_id grows > vocab_size),
        # which can collapse association logits and make ID matching unreliable.
        if self.id_label_strategy != "random":
            if track_id in self.track_vocab_map:
                return int(self.track_vocab_map[track_id])

            vocab_size = int(self.num_id_vocabulary) if int(self.num_id_vocabulary) > 0 else 1
            perm = self._seq_perm
            start_idx = int(track_id) % vocab_size

            # Prefer assigning an unused vocab id deterministically (probe perm order starting at start_idx).
            while not self.available_vocab_set:
                # Keep vocab ids unique by evicting least-recent tracks when the pool is exhausted.
                evicted = self._evict_one_track_for_vocab(avoid_track_id=track_id)
                if evicted is None:
                    break

            if self.available_vocab_set:
                if perm is not None and int(perm.numel()) >= vocab_size:
                    for off in range(vocab_size):
                        idx = (start_idx + off) % vocab_size
                        cand = int(perm[idx].item())
                        if cand in self.available_vocab_set:
                            # remove from pool
                            self.available_vocab_set.discard(cand)
                            try:
                                self.available_vocab.remove(cand)
                            except ValueError:
                                pass
                            self.track_vocab_map[track_id] = cand
                            return cand

                # Fallback (perm missing/invalid): pick a free vocab id from the pool deterministically.
                cand = int(min(self.available_vocab_set))
                self.available_vocab_set.discard(cand)
                try:
                    self.available_vocab.remove(cand)
                except ValueError:
                    pass
                self.track_vocab_map[track_id] = cand
                return cand

            # Last resort: no available vocab id and no eviction candidate; fall back deterministically.
            # This can theoretically collide, but should only happen in degenerate states.
            if not getattr(self, "_warned_vocab_collision_fallback", False):
                warnings.warn(
                    "[RuntimeTrackerByteTrack] Failed to allocate a free vocab id; falling back to a potentially "
                    "colliding mapping. This indicates an inconsistent tracker state."
                )
                self._warned_vocab_collision_fallback = True
            vocab_id = (
                int(perm[start_idx].item())
                if perm is not None and int(getattr(perm, "numel", lambda: 0)()) >= vocab_size
                else int(start_idx)
            )
            self.track_vocab_map[track_id] = vocab_id
            return vocab_id
        if track_id in self.track_vocab_map:
            return self.track_vocab_map[track_id]
        vocab_size = int(self.num_id_vocabulary) if int(self.num_id_vocabulary) > 0 else 1
        while not self.available_vocab:
            evicted = self._evict_one_track_for_vocab(avoid_track_id=track_id)
            if evicted is None:
                break
        if not self.available_vocab:
            return int(track_id % vocab_size)
        idx = int(track_id % len(self.available_vocab))
        vocab_id = self.available_vocab.pop(idx)
        self.available_vocab_set.discard(vocab_id)
        self.track_vocab_map[track_id] = int(vocab_id)
        return int(vocab_id)

    def _release_vocab_id(self, track_id: int):
        vocab_id = self.track_vocab_map.pop(track_id, None)
        if self.id_label_strategy != "random":
            # Return to pool so future tracks can reuse it deterministically.
            if vocab_id is None:
                return
            if int(vocab_id) not in self.available_vocab_set:
                self.available_vocab.append(int(vocab_id))
                self.available_vocab_set.add(int(vocab_id))
            return
        if vocab_id is None:
            return
        if vocab_id not in self.available_vocab_set:
            self.available_vocab.append(int(vocab_id))
            self.available_vocab_set.add(int(vocab_id))

    @torch.no_grad()
    def update(self, image_path: str) -> Dict:
        """
        处理一帧图像

        参数：
            image_path: 图像路径

        返回：
            结果字典，包含 score, category, bbox, id
        """
        self.frame_id += 1
        device = distributed_device()

        # 1. 获取检测框并提取特征（model / external）
        if self.det_source == "external":
            frame_dets = self.external_detections.get(int(self.frame_id), [])
            detections = []
            for det in frame_dets:
                if len(det) < 5:
                    continue
                x, y, w, h, conf = det[:5]
                detections.append((float(x), float(y), float(w), float(h), float(conf)))
            if len(detections) > 0 and not (
                self.assoc_feat_source == "reid" and self.assoc_mode == "feature" and self.reid_encoder is not None
            ):
                boxes_xywh = [(d[0], d[1], d[2], d[3]) for d in detections]
                try:
                    features = self.feature_extractor.extract_features_from_boxes(
                        image_path=image_path,
                        boxes_xywh=boxes_xywh,
                    )
                except Exception as exc:
                    warnings.warn(
                        f"[RuntimeTrackerByteTrack] External det feature extraction failed at "
                        f"frame {self.frame_id}: {exc}"
                    )
                    features = torch.zeros((len(detections), self.feature_dim), device=device, dtype=self.dtype)
            else:
                # Fast path: association relies on external ReID only, so we can skip YOLOX RoI
                # features. Still keep a placeholder tensor aligned with detections for downstream
                # indexing (e.g., features[filtered_indices]).
                features = torch.zeros((len(detections), self.feature_dim), device=device, dtype=self.dtype)
        else:
            if self.use_tta and hasattr(self.feature_extractor, "detect_with_features_tta"):
                detections, features = self.feature_extractor.detect_with_features_tta(
                    image_path=image_path,
                    tta=self.tta,
                )
            else:
                # Fast path: if association relies on external ReID features only, skip YOLOX RoI features.
                if (
                    self.assoc_feat_source == "reid"
                    and self.assoc_mode == "feature"
                    and self.reid_encoder is not None
                    and hasattr(self.feature_extractor, "detect")
                ):
                    detections = self.feature_extractor.detect(image_path)
                    features = torch.zeros((len(detections), self.feature_dim), device=device, dtype=self.dtype)
                else:
                    detections, features = self.feature_extractor.detect_with_features(image_path)

        self.last_dets = list(detections)
        if not torch.is_tensor(features):
            features = torch.as_tensor(features, device=device)
        features = features.to(device=device, dtype=self.dtype)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.shape[0] != len(detections):
            # Keep code robust: feature extraction can be skipped/fail for external det or return
            # unexpected shapes. We prefer a safe placeholder over crashing in evaluation/sweeps.
            if features.shape[0] == 0:
                features = torch.zeros((len(detections), self.feature_dim), device=device, dtype=self.dtype)
            else:
                warnings.warn(
                    f"[RuntimeTrackerByteTrack] Feature count mismatch at frame {self.frame_id}: "
                    f"{features.shape[0]} features for {len(detections)} detections. "
                    f"Padding/truncating with zeros."
                )
                if features.shape[0] < len(detections):
                    pad = torch.zeros(
                        (len(detections) - features.shape[0], self.feature_dim),
                        device=device,
                        dtype=self.dtype,
                    )
                    features = torch.cat([features, pad], dim=0)
                else:
                    features = features[: len(detections)]

        if len(detections) == 0:
            # 无检测，更新丢失容忍度
            self._update_miss_counts()
            self.current_track_results = {
                "score": torch.tensor([], device=device),
                "category": torch.tensor([], dtype=torch.long, device=device),
                "bbox": torch.zeros((0, 4), device=device),
                "id": torch.tensor([], dtype=torch.long, device=device),
            }
            return self.current_track_results

        # 2. 过滤低置信度检测
        filtered_dets = []
        filtered_indices = []
        for i, (x, y, w, h, conf) in enumerate(detections):
            area = w * h
            if conf >= self.det_thresh and area >= self.area_thresh:
                filtered_dets.append((x, y, w, h, conf))
                filtered_indices.append(i)

        # Optional: limit detections per frame by confidence (inference only)
        if self.det_max_per_frame and len(filtered_dets) > self.det_max_per_frame:
            confs = [d[4] for d in filtered_dets]
            topk_idx = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)[: self.det_max_per_frame]
            filtered_dets = [filtered_dets[i] for i in topk_idx]
            filtered_indices = [filtered_indices[i] for i in topk_idx]

        if len(filtered_dets) == 0:
            self._update_miss_counts()
            self.current_track_results = {
                "score": torch.tensor([], device=device),
                "category": torch.tensor([], dtype=torch.long, device=device),
                "bbox": torch.zeros((0, 4), device=device),
                "id": torch.tensor([], dtype=torch.long, device=device),
            }
            return self.current_track_results

        # 获取过滤后的特征
        model_features = features[filtered_indices]  # (N, feature_dim)

        # 3. 转换检测框格式
        boxes_xywh = torch.tensor(
            [[d[0], d[1], d[2], d[3]] for d in filtered_dets],
            dtype=self.dtype,
            device=device
        )
        scores = torch.tensor([d[4] for d in filtered_dets], dtype=self.dtype, device=device)

        # 归一化框为 cxcywh
        boxes_cxcywh = box_xywh_to_cxcywh(boxes_xywh) / self.bbox_unnorm

        num_dets = len(filtered_dets)

        # Optional: external ReID embeddings for association (inference only).
        assoc_features = model_features
        if self.reid_encoder is not None and self.assoc_feat_source == "reid":
            try:
                reid_feats = self.reid_encoder.encode(image_path=image_path, boxes_xywh=boxes_xywh)
                if torch.is_tensor(reid_feats) and reid_feats.shape[0] == num_dets:
                    assoc_features = reid_feats.to(device=device, dtype=self.dtype)
                else:
                    warnings.warn(
                        f"[RuntimeTrackerByteTrack] ReID encoder returned invalid shape at frame {self.frame_id}: "
                        f"{getattr(reid_feats, 'shape', None)}; fallback to YOLOX features."
                    )
                    assoc_features = model_features
            except Exception as exc:
                warnings.warn(
                    f"[RuntimeTrackerByteTrack] ReID feature extraction failed at frame {self.frame_id}: {exc}"
                )
                assoc_features = model_features

        # 4. 预测轨迹状态（Kalman）
        if self.use_kalman:
            for track_id in list(self.active_ids):
                info = self.trajectory_infos.get(track_id)
                if info is None:
                    continue
                mean = info.get("mean")
                cov = info.get("cov")
                if mean is not None and cov is not None:
                    mean, cov = self.kf.predict(mean, cov)
                    info["mean"] = mean
                    info["cov"] = cov
                    pred = _xyah_to_cxcywh(torch.tensor(mean[:4], device=device, dtype=self.dtype))
                    info["pred_box"] = pred

        # 5. 准备轨迹特征
        if len(self.active_ids) == 0:
            # 第一帧，所有检测都是新目标
            id_labels = self._assign_new_ids(num_dets, scores)
        else:
            # 使用频域关联进行匹配
            id_labels, scores = self._get_id_pred_labels(model_features, assoc_features, boxes_cxcywh, scores)

        # 6. 更新轨迹信息
        self._update_trajectory_infos(boxes_cxcywh, model_features, assoc_features, id_labels, scores)

        # 7. 构造输出
        self.current_track_results = {
            "score": scores,
            "category": torch.zeros(num_dets, dtype=torch.long, device=device),
            "bbox": boxes_xywh,
            "id": id_labels,
        }

        return self.current_track_results

    def _assign_new_ids(self, num_dets: int, scores: torch.Tensor) -> torch.Tensor:
        """为新检测分配新 ID"""
        device = distributed_device()
        id_labels = torch.zeros(num_dets, dtype=torch.long, device=device)

        for i in range(num_dets):
            if scores[i] >= self.newborn_thresh:
                self.max_id += 1
                new_id = self.max_id
                id_labels[i] = new_id
                self.active_ids.add(new_id)
                self._get_or_assign_vocab_id(new_id)
            else:
                id_labels[i] = -1  # 低置信度，不分配 ID

        return id_labels

    def _build_track_feature_matrix(self, active_ids_list: List[int]) -> torch.Tensor:
        device = distributed_device()
        feats = []
        for track_id in active_ids_list:
            info = self.trajectory_infos.get(track_id)
            track_feats = None
            if info is not None:
                track_feats = info.get("assoc_features", None)
                if not track_feats:
                    track_feats = info.get("features", None)
            if not track_feats:
                feats.append(torch.zeros((self.assoc_feature_dim,), device=device, dtype=self.dtype))
                continue
            if self.assoc_feat_agg == "mean":
                k = min(len(track_feats), self.assoc_feat_k)
                feat = torch.stack(track_feats[-k:], dim=0).mean(dim=0)
            else:
                feat = track_feats[-1]
            feats.append(feat.to(device=device, dtype=self.dtype))
        if len(feats) == 0:
            return torch.zeros((0, self.assoc_feature_dim), device=device, dtype=self.dtype)
        return torch.stack(feats, dim=0)

    def _refine_assoc_scores_with_laplace(
        self,
        base_scores: torch.Tensor,
        det_features: torch.Tensor,
        track_history_features: torch.Tensor,
        track_history_masks: torch.Tensor,
        det_scores: torch.Tensor,
        motion_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.assoc_use_laplace or self.laplace_assoc is None:
            return base_scores
        if base_scores.numel() == 0:
            return base_scores
        try:
            self.laplace_assoc = self.laplace_assoc.to(device=base_scores.device)
            result = self.laplace_assoc(
                spatial_scores=base_scores,
                det_features=det_features,
                track_history_features=track_history_features,
                track_history_masks=track_history_masks,
                motion_scores=motion_scores,
                det_scores=det_scores,
            )
            fused = result.get("fused_scores", None)
            if torch.is_tensor(fused) and fused.shape == base_scores.shape:
                return fused.to(device=base_scores.device, dtype=base_scores.dtype)
        except Exception as exc:
            warnings.warn(f"[RuntimeTrackerByteTrack] Laplace refinement failed; fallback to base scores. Error: {exc}")
        return base_scores

    def _refine_assoc_scores_with_mtcr(
        self,
        base_scores: torch.Tensor,
        det_features: torch.Tensor,
        track_history_features: torch.Tensor,
        track_history_masks: torch.Tensor,
        track_history_times: Optional[torch.Tensor],
        det_scores: torch.Tensor,
        motion_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.assoc_use_mtcr or self.mtcr_assoc is None:
            return base_scores
        if base_scores.numel() == 0:
            return base_scores
        try:
            self.mtcr_assoc = self.mtcr_assoc.to(device=base_scores.device)
            track_gaps = None
            if track_history_times is not None and track_history_times.numel() > 0:
                valid = ~track_history_masks
                last_idx = (valid.to(dtype=torch.long) * (torch.arange(track_history_times.shape[1], device=track_history_times.device).view(1, -1) + 1)).max(dim=1).values - 1
                last_idx = last_idx.clamp(min=0)
                last_times = torch.gather(track_history_times, dim=1, index=last_idx.view(-1, 1)).squeeze(1)
                track_gaps = (track_history_times.new_full((track_history_times.shape[0],), int(self.frame_id)) - last_times).clamp(min=0)
            result = self.mtcr_assoc(
                anchor_scores=base_scores,
                det_features=det_features,
                track_history_features=track_history_features,
                track_history_masks=track_history_masks,
                motion_scores=motion_scores,
                det_scores=det_scores,
                track_gaps=track_gaps,
            )
            fused = result.get("final_scores", None)
            if torch.is_tensor(fused) and fused.shape == base_scores.shape:
                return fused.to(device=base_scores.device, dtype=base_scores.dtype)
        except Exception as exc:
            warnings.warn(f"[RuntimeTrackerByteTrack] MTCR refinement failed; fallback to base scores. Error: {exc}")
        return base_scores

    def _build_runtime_replay_track_gaps(
        self,
        track_history_masks: torch.Tensor,
        track_history_times: Optional[torch.Tensor],
    ) -> torch.Tensor:
        num_tracks = int(track_history_masks.shape[0]) if track_history_masks.ndim >= 2 else 0
        if num_tracks == 0:
            return torch.zeros((0,), device=track_history_masks.device, dtype=torch.long)
        gaps = torch.zeros((num_tracks,), device=track_history_masks.device, dtype=torch.long)
        if track_history_times is None or track_history_times.numel() == 0:
            return gaps
        valid = ~track_history_masks
        hist_len = valid.sum(dim=1)
        last_idx = (
            valid.to(dtype=torch.long)
            * (torch.arange(track_history_times.shape[1], device=track_history_times.device).view(1, -1) + 1)
        ).max(dim=1).values - 1
        last_idx = last_idx.clamp(min=0)
        last_times = torch.gather(track_history_times, dim=1, index=last_idx.view(-1, 1)).squeeze(1)
        gaps = (track_history_times.new_full((num_tracks,), int(self.frame_id)) - last_times).clamp(min=0)
        gaps = torch.where(hist_len > 0, gaps, torch.zeros_like(gaps))
        return gaps

    def _build_runtime_replay_scalar_features(
        self,
        anchor_scores: torch.Tensor,
        motion_scores: torch.Tensor,
        det_score: torch.Tensor,
        track_gaps: torch.Tensor,
        hist_len: torch.Tensor,
        det_box: torch.Tensor,
        track_boxes: torch.Tensor,
    ) -> torch.Tensor:
        if anchor_scores.ndim == 2:
            batch, cand = anchor_scores.shape
            if cand == 0:
                return torch.zeros((batch, 0, 18), device=anchor_scores.device, dtype=anchor_scores.dtype)

            if cand <= 1:
                margin = torch.zeros((batch,), device=anchor_scores.device, dtype=anchor_scores.dtype)
                rank_frac = torch.ones_like(anchor_scores)
            else:
                top2 = torch.topk(anchor_scores, k=min(2, cand), dim=1, sorted=True).values
                margin = top2[:, 0] - top2[:, 1]
                rank_vals = torch.arange(cand, device=anchor_scores.device, dtype=anchor_scores.dtype).view(1, -1)
                rank_frac = 1.0 - (rank_vals / float(max(cand - 1, 1)))
                rank_frac = rank_frac.expand(batch, -1)
            entropy = _score_entropy_from_probs(anchor_scores)

            det_cx = det_box[:, 0].unsqueeze(1)
            det_cy = det_box[:, 1].unsqueeze(1)
            det_w = det_box[:, 2].clamp(min=1e-6).unsqueeze(1)
            det_h = det_box[:, 3].clamp(min=1e-6).unsqueeze(1)
            trk_cx = track_boxes[:, :, 0]
            trk_cy = track_boxes[:, :, 1]
            trk_w = track_boxes[:, :, 2].clamp(min=1e-6)
            trk_h = track_boxes[:, :, 3].clamp(min=1e-6)

            dx_norm = (det_cx - trk_cx) / det_w
            dy_norm = (det_cy - trk_cy) / det_h
            log_w_ratio = torch.log(det_w / trk_w)
            log_h_ratio = torch.log(det_h / trk_h)
            log_area_ratio = torch.log((det_w * det_h) / (trk_w * trk_h).clamp(min=1e-6))

            det_xyxy = _cxcywh_to_xyxy(det_box).unsqueeze(1)
            trk_xyxy = _cxcywh_to_xyxy(track_boxes)
            lt = torch.maximum(det_xyxy[..., :2], trk_xyxy[..., :2])
            rb = torch.minimum(det_xyxy[..., 2:], trk_xyxy[..., 2:])
            wh = (rb - lt).clamp(min=0)
            inter = wh[..., 0] * wh[..., 1]
            det_area = ((det_xyxy[..., 2] - det_xyxy[..., 0]).clamp(min=0) * (det_xyxy[..., 3] - det_xyxy[..., 1]).clamp(min=0))
            trk_area = ((trk_xyxy[..., 2] - trk_xyxy[..., 0]).clamp(min=0) * (trk_xyxy[..., 3] - trk_xyxy[..., 1]).clamp(min=0))
            det_iou = inter / (det_area + trk_area - inter + 1e-6)

            margin_col = margin.unsqueeze(1).expand(-1, cand)
            entropy_col = entropy.unsqueeze(1).expand(-1, cand)
            det_score_col = det_score.to(dtype=anchor_scores.dtype).view(batch, 1).expand(-1, cand)
            scalar = torch.stack(
                [
                    anchor_scores,
                    anchor_scores,
                    anchor_scores,
                    motion_scores,
                    det_score_col,
                    torch.log1p(track_gaps.to(dtype=anchor_scores.dtype).clamp(min=0)),
                    torch.log1p(hist_len.to(dtype=anchor_scores.dtype).clamp(min=0)),
                    margin_col,
                    margin_col,
                    margin_col,
                    entropy_col,
                    rank_frac,
                    dx_norm,
                    dy_norm,
                    log_w_ratio,
                    log_h_ratio,
                    log_area_ratio,
                    det_iou.to(dtype=anchor_scores.dtype),
                ],
                dim=-1,
            )
            return scalar

        cand = int(anchor_scores.numel())
        if cand == 0:
            return torch.zeros((0, 18), device=anchor_scores.device, dtype=anchor_scores.dtype)

        if cand <= 1:
            margin = anchor_scores.new_tensor(0.0)
        else:
            top2 = torch.topk(anchor_scores, k=min(2, cand), sorted=True).values
            margin = top2[0] - top2[1]
        entropy = _score_entropy_from_probs(anchor_scores.view(1, -1)).view(-1)[0]

        det_cx, det_cy, det_w, det_h = det_box.unbind(dim=-1)
        trk_cx = track_boxes[:, 0]
        trk_cy = track_boxes[:, 1]
        trk_w = track_boxes[:, 2].clamp(min=1e-6)
        trk_h = track_boxes[:, 3].clamp(min=1e-6)
        det_w = det_w.clamp(min=1e-6)
        det_h = det_h.clamp(min=1e-6)

        dx_norm = (det_cx - trk_cx) / det_w
        dy_norm = (det_cy - trk_cy) / det_h
        log_w_ratio = torch.log(det_w / trk_w)
        log_h_ratio = torch.log(det_h / trk_h)
        log_area_ratio = torch.log((det_w * det_h) / (trk_w * trk_h).clamp(min=1e-6))

        det_iou = _box_iou(_cxcywh_to_xyxy(det_box.view(1, 4)), _cxcywh_to_xyxy(track_boxes)).view(-1)
        if cand <= 1:
            rank_frac = torch.ones_like(anchor_scores)
        else:
            rank_frac = 1.0 - (
                torch.arange(cand, device=anchor_scores.device, dtype=anchor_scores.dtype)
                / float(max(cand - 1, 1))
            )

        margin_col = anchor_scores.new_full((cand,), float(margin.item()))
        entropy_col = anchor_scores.new_full((cand,), float(entropy.item()))
        det_score_col = det_score.expand(cand)
        scalar = torch.stack(
            [
                anchor_scores,
                anchor_scores,
                anchor_scores,
                motion_scores,
                det_score_col,
                torch.log1p(track_gaps.to(dtype=anchor_scores.dtype).clamp(min=0)),
                torch.log1p(hist_len.to(dtype=anchor_scores.dtype).clamp(min=0)),
                margin_col,
                margin_col,
                margin_col,
                entropy_col,
                rank_frac,
                dx_norm,
                dy_norm,
                log_w_ratio,
                log_h_ratio,
                log_area_ratio,
                det_iou.to(dtype=anchor_scores.dtype),
            ],
            dim=-1,
        )
        return scalar

    def _load_competition_oracle(self, csv_path: str) -> Dict[str, Dict[str, Any]]:
        path = os.path.expanduser(csv_path)
        records: Dict[str, Dict[str, Any]] = {}
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                group_id = str(row.get("group_id", "")).strip()
                if not group_id:
                    continue
                try:
                    action_target_id = int(float(row.get("action_target_id", 0)))
                except Exception:
                    action_target_id = 0
                try:
                    target_candidate_rank = int(float(row.get("target_candidate_rank", -1)))
                except Exception:
                    target_candidate_rank = -1
                records[group_id] = {
                    "action_target_id": action_target_id,
                    "target_candidate_rank": target_candidate_rank,
                    "raw": dict(row),
                }
        return records

    def _load_local_conflict_graph_oracle(self, jsonl_path: str) -> Dict[str, Dict[str, Any]]:
        path = os.path.expanduser(jsonl_path)
        records: Dict[str, Dict[str, Any]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                group_id = str(row.get("group_id", "")).strip()
                if not group_id:
                    continue
                positive_track_ids = []
                candidate_track_ids = []
                for cand in row.get("candidates", []):
                    try:
                        valid = int(cand.get("valid_train_row", 1))
                    except Exception:
                        valid = 1
                    try:
                        track_id = int(cand.get("track_id", -1))
                    except Exception:
                        track_id = -1
                    if valid <= 0 or track_id < 0:
                        continue
                    candidate_track_ids.append(track_id)
                    try:
                        label = int(cand.get("label", 0))
                    except Exception:
                        label = 0
                    if label == 1:
                        positive_track_ids.append(track_id)
                records[group_id] = {
                    "seq": str(row.get("seq", "")),
                    "frame": int(row.get("frame", 0)),
                    "det_index": int(row.get("det_index", 0)),
                    "candidate_track_ids": sorted(set(candidate_track_ids)),
                    "positive_track_ids": sorted(set(positive_track_ids)),
                    "group_has_positive": bool(int(row.get("group_has_positive", 1 if positive_track_ids else 0))),
                }
        return records

    def _build_local_conflict_graph_oracle_plan(
        self,
        score_mat: torch.Tensor,
        active_ids_list: List[int],
        thresh: float,
        graph_mode: str,
        finalize_null_dets: bool,
        block_all_cluster_tracks: bool,
    ) -> Dict[str, Any]:
        plan = {
            "assignments": [],
            "resolved_dets": set(),
            "blocked_tracks": set(),
            "stats": {
                "graph_mode": str(graph_mode),
                "eligible_clusters": 0,
                "replaced_clusters": 0,
                "resolved_dets": 0,
                "matched_dets": 0,
                "null_dets": 0,
                "deferred_dets": 0,
                "blocked_tracks": 0,
                "trigger_filtered_clusters": 0,
                "skipped_large_clusters": 0,
            },
        }
        if (
            not self.assoc_use_local_conflict_graph
            or self.assoc_local_conflict_graph_mode != str(graph_mode)
            or not self.local_conflict_graph_oracle
            or build_topk_bipartite_components is None
            or solve_assignment_with_private_null is None
        ):
            return plan
        if score_mat.ndim != 2 or score_mat.numel() == 0:
            return plan

        num_dets, num_tracks = score_mat.shape
        if num_dets == 0 or num_tracks == 0:
            return plan

        det_rows_with_oracle: List[int] = []
        for det_row in range(num_dets):
            group_id = f"{self.sequence_name or ''}:{int(self.frame_id)}:{int(det_row)}"
            if group_id in self.local_conflict_graph_oracle:
                det_rows_with_oracle.append(int(det_row))
        if not det_rows_with_oracle:
            return plan

        clusters = build_topk_bipartite_components(
            score_mat=score_mat,
            topk=int(self.assoc_local_conflict_graph_topk),
            min_edge_score=0.0,
            det_rows=det_rows_with_oracle,
        )

        for cluster in clusters:
            det_rows = [int(x) for x in cluster.get("det_rows", [])]
            track_cols = [int(x) for x in cluster.get("track_cols", [])]
            if len(det_rows) < int(self.assoc_local_conflict_graph_min_detections):
                continue
            if not det_rows or not track_cols:
                continue

            oracle_rows: List[Dict[str, Any]] = []
            missing_oracle = False
            for det_row in det_rows:
                group_id = f"{self.sequence_name or ''}:{int(self.frame_id)}:{int(det_row)}"
                oracle = self.local_conflict_graph_oracle.get(group_id)
                if oracle is None:
                    missing_oracle = True
                    break
                oracle_rows.append(oracle)
            if missing_oracle:
                continue

            plan["stats"]["eligible_clusters"] += 1
            score_sub = score_mat[det_rows][:, track_cols]
            feasible_mask = torch.zeros_like(score_sub, dtype=torch.bool)
            null_scores = torch.zeros((len(det_rows),), device=score_sub.device, dtype=score_sub.dtype)

            for local_det_idx, oracle in enumerate(oracle_rows):
                positive_track_ids = set(int(tid) for tid in oracle.get("positive_track_ids", []))
                if not positive_track_ids:
                    continue
                for local_track_idx, col in enumerate(track_cols):
                    track_id = int(active_ids_list[col])
                    if track_id not in positive_track_ids:
                        continue
                    if float(score_sub[local_det_idx, local_track_idx].item()) < float(thresh):
                        continue
                    feasible_mask[local_det_idx, local_track_idx] = True

            local_assignments = solve_assignment_with_private_null(
                score_sub=score_sub,
                feasible_mask=feasible_mask,
                null_scores=null_scores,
                use_hungarian=(self.matching_method == "hungarian"),
            )

            matched_track_cols = set()
            matched_det_rows = []
            deferred_det_rows = []
            for assignment in local_assignments:
                det_local_idx = int(assignment.get("det_local_idx", -1))
                if det_local_idx < 0 or det_local_idx >= len(det_rows):
                    continue
                det_row = int(det_rows[det_local_idx])
                track_local_idx = assignment.get("track_local_idx", None)
                if track_local_idx is None:
                    if finalize_null_dets:
                        plan["resolved_dets"].add(det_row)
                        plan["stats"]["null_dets"] += 1
                    else:
                        deferred_det_rows.append(det_row)
                    continue
                track_local_idx = int(track_local_idx)
                if track_local_idx < 0 or track_local_idx >= len(track_cols):
                    if finalize_null_dets:
                        plan["resolved_dets"].add(det_row)
                        plan["stats"]["null_dets"] += 1
                    else:
                        deferred_det_rows.append(det_row)
                    continue
                track_col = int(track_cols[track_local_idx])
                matched_track_cols.add(track_col)
                matched_det_rows.append((det_row, track_col))

            if (not finalize_null_dets) and (
                len(matched_track_cols) < int(self.assoc_local_conflict_graph_min_committed_matches)
            ):
                plan["stats"]["trigger_filtered_clusters"] += 1
                plan["stats"]["deferred_dets"] += int(len(det_rows))
                continue

            plan["stats"]["replaced_clusters"] += 1
            for det_row, track_col in matched_det_rows:
                plan["resolved_dets"].add(det_row)
                plan["assignments"].append((det_row, track_col))
                plan["stats"]["matched_dets"] += 1
            if not finalize_null_dets:
                plan["stats"]["deferred_dets"] += int(len(deferred_det_rows))

            if block_all_cluster_tracks:
                for track_col in track_cols:
                    plan["blocked_tracks"].add(int(track_col))
            else:
                for track_col in matched_track_cols:
                    plan["blocked_tracks"].add(int(track_col))

        plan["assignments"].sort(key=lambda x: (int(x[0]), int(x[1])))
        plan["stats"]["resolved_dets"] = int(len(plan["resolved_dets"]))
        plan["stats"]["blocked_tracks"] = int(len(plan["blocked_tracks"]))
        return plan

    def _get_local_conflict_graph_oracle_full_plan(
        self,
        score_mat: torch.Tensor,
        active_ids_list: List[int],
        thresh: float,
    ) -> Dict[str, Any]:
        return self._build_local_conflict_graph_oracle_plan(
            score_mat=score_mat,
            active_ids_list=active_ids_list,
            thresh=thresh,
            graph_mode="oracle_full",
            finalize_null_dets=True,
            block_all_cluster_tracks=True,
        )

    def _get_local_conflict_graph_oracle_commit_matches_plan(
        self,
        score_mat: torch.Tensor,
        active_ids_list: List[int],
        thresh: float,
    ) -> Dict[str, Any]:
        return self._build_local_conflict_graph_oracle_plan(
            score_mat=score_mat,
            active_ids_list=active_ids_list,
            thresh=thresh,
            graph_mode="oracle_commit_matches",
            finalize_null_dets=False,
            block_all_cluster_tracks=False,
        )

    def _get_local_conflict_graph_oracle_assignments(
        self,
        base_scores: torch.Tensor,
        score_mat: torch.Tensor,
        active_ids_list: List[int],
        thresh: float,
    ) -> Dict[str, Any]:
        plan = {
            "assignments": [],
            "stats": {
                "graph_mode": "legacy_partial_oracle",
                "eligible_clusters": 0,
                "replaced_clusters": 0,
                "resolved_dets": 0,
                "matched_dets": 0,
                "null_dets": 0,
                "deferred_dets": 0,
                "blocked_tracks": 0,
                "trigger_filtered_clusters": 0,
                "skipped_large_clusters": 0,
            },
        }
        if not self.assoc_use_local_conflict_graph_oracle or not self.local_conflict_graph_oracle:
            return plan
        if base_scores.numel() == 0 or base_scores.ndim != 2:
            return plan

        num_dets, num_tracks = base_scores.shape
        if num_dets == 0 or num_tracks == 0:
            return plan

        topk = int(max(min(self.assoc_local_conflict_graph_topk, num_tracks), 1))
        top_idx = torch.topk(base_scores, k=topk, dim=1, sorted=True).indices

        det_to_tracks: Dict[int, List[int]] = {}
        det_to_positive_tracks: Dict[int, set[int]] = {}
        track_to_dets: Dict[int, set[int]] = defaultdict(set)
        eligible_det_rows: List[int] = []

        for det_row in range(num_dets):
            group_id = f"{self.sequence_name or ''}:{int(self.frame_id)}:{int(det_row)}"
            oracle = self.local_conflict_graph_oracle.get(group_id)
            if not oracle:
                continue
            row_track_cols: List[int] = []
            row_positive_cols: set[int] = set()
            positive_track_ids = set(int(tid) for tid in oracle.get("positive_track_ids", []))
            for local_rank in range(topk):
                col = int(top_idx[det_row, local_rank].item())
                track_id = int(active_ids_list[col])
                row_track_cols.append(col)
                if track_id in positive_track_ids:
                    row_positive_cols.add(col)
                track_to_dets[col].add(det_row)
            if not row_track_cols:
                continue
            det_to_tracks[det_row] = sorted(set(row_track_cols))
            det_to_positive_tracks[det_row] = row_positive_cols
            eligible_det_rows.append(det_row)

        if not eligible_det_rows:
            return plan

        visited_dets: set[int] = set()
        visited_tracks: set[int] = set()
        lsa = _get_linear_sum_assignment()

        for start_det in eligible_det_rows:
            if start_det in visited_dets:
                continue
            det_stack = [start_det]
            cluster_dets: set[int] = set()
            cluster_tracks: set[int] = set()
            while det_stack:
                det_row = det_stack.pop()
                if det_row in visited_dets:
                    continue
                visited_dets.add(det_row)
                cluster_dets.add(det_row)
                for col in det_to_tracks.get(det_row, []):
                    if col not in cluster_tracks:
                        cluster_tracks.add(col)
                    if col in visited_tracks:
                        continue
                    visited_tracks.add(col)
                    for nxt_det in track_to_dets.get(col, set()):
                        if nxt_det not in visited_dets:
                            det_stack.append(nxt_det)

            if len(cluster_dets) < self.assoc_local_conflict_graph_min_detections:
                continue

            det_rows = sorted(cluster_dets)
            track_cols = sorted(cluster_tracks)
            if not det_rows or not track_cols:
                continue
            plan["stats"]["eligible_clusters"] += 1

            score_sub = score_mat[det_rows][:, track_cols]
            feasible = torch.zeros_like(score_sub, dtype=torch.bool)
            for local_det_idx, det_row in enumerate(det_rows):
                positive_cols = det_to_positive_tracks.get(det_row, set())
                if not positive_cols:
                    continue
                for local_track_idx, col in enumerate(track_cols):
                    if col not in positive_cols:
                        continue
                    if float(score_sub[local_det_idx, local_track_idx].item()) < float(thresh):
                        continue
                    feasible[local_det_idx, local_track_idx] = True

            if not bool(feasible.any()):
                continue
            plan["stats"]["replaced_clusters"] += 1

            num_local_dets = len(det_rows)
            num_local_tracks = len(track_cols)
            assign_scores = torch.zeros(
                (num_local_dets, num_local_tracks + num_local_dets),
                device=score_sub.device,
                dtype=score_sub.dtype,
            )
            assign_scores[:, :num_local_tracks] = torch.where(
                feasible,
                score_sub,
                torch.full_like(score_sub, -1e6),
            )
            if lsa is None:
                used_local_tracks: set[int] = set()
                for local_det_idx, det_row in enumerate(det_rows):
                    feasible_cols = feasible[local_det_idx].nonzero(as_tuple=True)[0]
                    if feasible_cols.numel() == 0:
                        continue
                    best_local_track = None
                    best_score = None
                    for local_track_idx in feasible_cols.tolist():
                        if local_track_idx in used_local_tracks:
                            continue
                        cand_score = float(score_sub[local_det_idx, local_track_idx].item())
                        if best_score is None or cand_score > best_score:
                            best_score = cand_score
                            best_local_track = int(local_track_idx)
                    if best_local_track is None:
                        continue
                    used_local_tracks.add(best_local_track)
                    plan["assignments"].append((int(det_row), int(track_cols[best_local_track])))
                    plan["stats"]["matched_dets"] += 1
                continue

            cost = (-assign_scores).detach().cpu().numpy()
            rows, cols = lsa(cost)
            for local_det_idx, local_assign_idx in zip(rows, cols):
                if int(local_assign_idx) >= num_local_tracks:
                    continue
                if not bool(feasible[local_det_idx, local_assign_idx].item()):
                    continue
                plan["assignments"].append(
                    (
                        int(det_rows[int(local_det_idx)]),
                        int(track_cols[int(local_assign_idx)]),
                    )
                )
                plan["stats"]["matched_dets"] += 1

        plan["assignments"].sort(key=lambda x: (x[0], x[1]))
        plan["stats"]["resolved_dets"] = int(plan["stats"]["matched_dets"])
        return plan

    def _build_local_conflict_runtime_features(
        self,
        *,
        base_scores: torch.Tensor,
        score_mat: torch.Tensor,
        det_scores: torch.Tensor,
        det_rows: List[int],
        track_cols: List[int],
        track_history_masks: Optional[torch.Tensor],
        track_history_times: Optional[torch.Tensor],
        motion_scores: Optional[torch.Tensor],
        det_boxes_cxcywh: Optional[torch.Tensor] = None,
        track_boxes_cxcywh: Optional[torch.Tensor] = None,
        host_variant: str = "",
    ) -> Dict[str, Any]:
        if self.local_conflict_model_family == "set_predictor_v2" or str(
            getattr(self, "local_conflict_feature_version", "")
        ).lower().startswith("v2"):
            return self._build_local_conflict_runtime_features_v2(
                base_scores=base_scores,
                score_mat=score_mat,
                det_scores=det_scores,
                det_rows=det_rows,
                track_cols=track_cols,
                track_history_masks=track_history_masks,
                track_history_times=track_history_times,
                motion_scores=motion_scores,
                det_boxes_cxcywh=det_boxes_cxcywh,
                track_boxes_cxcywh=track_boxes_cxcywh,
                host_variant=host_variant,
            )
        return self._build_local_conflict_runtime_features_v1(
            base_scores=base_scores,
            score_mat=score_mat,
            det_scores=det_scores,
            det_rows=det_rows,
            track_cols=track_cols,
            track_history_masks=track_history_masks,
            track_history_times=track_history_times,
            motion_scores=motion_scores,
        )

    def _build_local_conflict_runtime_features_v1(
        self,
        *,
        base_scores: torch.Tensor,
        score_mat: torch.Tensor,
        det_scores: torch.Tensor,
        det_rows: List[int],
        track_cols: List[int],
        track_history_masks: Optional[torch.Tensor],
        track_history_times: Optional[torch.Tensor],
        motion_scores: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        device = score_mat.device
        dtype = score_mat.dtype
        det_rows = [int(x) for x in det_rows]
        track_cols = [int(x) for x in track_cols]
        num_local_dets = len(det_rows)
        num_local_tracks = len(track_cols)
        topk = int(max(min(self.assoc_local_conflict_graph_topk, score_mat.shape[1]), 1))
        track_col_to_local = {int(track_col): idx for idx, track_col in enumerate(track_cols)}

        score_rows = score_mat.index_select(0, torch.as_tensor(det_rows, device=device))
        top_vals, top_idx = torch.topk(score_rows, k=topk, dim=1, sorted=True)
        if track_history_masks is not None:
            hist_len_all = (~track_history_masks).sum(dim=1).to(device=device, dtype=dtype)
            track_gaps_all = self._build_runtime_replay_track_gaps(track_history_masks, track_history_times).to(
                device=device,
                dtype=dtype,
            )
        else:
            hist_len_all = torch.zeros((score_mat.shape[1],), device=device, dtype=dtype)
            track_gaps_all = torch.zeros((score_mat.shape[1],), device=device, dtype=dtype)

        det_features: List[List[float]] = []
        edge_features: List[List[float]] = []
        edge_det_index: List[int] = []
        edge_track_index: List[int] = []
        dense_edge_mask = torch.zeros((num_local_dets, num_local_tracks), device=device, dtype=torch.bool)
        dense_refined_scores = torch.zeros((num_local_dets, num_local_tracks), device=device, dtype=dtype)

        for local_det_idx, det_row in enumerate(det_rows):
            row_vals = top_vals[local_det_idx]
            row_prob = torch.softmax(row_vals - row_vals.max(), dim=0)
            row_entropy = float(-(row_prob * torch.log(row_prob.clamp(min=1e-8))).sum().item())
            row_margin = 0.0
            if row_vals.numel() > 1:
                row_margin = float((row_vals[0] - row_vals[1]).item())
            det_features.append(
                [
                    float(det_scores[det_row].item()) if det_scores is not None else 0.0,
                    0.0,
                    row_margin,
                    row_entropy,
                ]
            )
            for local_rank in range(int(top_vals.shape[1])):
                track_col = int(top_idx[local_det_idx, local_rank].item())
                if track_col not in track_col_to_local:
                    continue
                score_val = float(top_vals[local_det_idx, local_rank].item())
                if score_val <= 0.0:
                    continue
                local_track_idx = int(track_col_to_local[track_col])
                edge_det_index.append(local_det_idx)
                edge_track_index.append(local_track_idx)
                dense_edge_mask[local_det_idx, local_track_idx] = True
                dense_refined_scores[local_det_idx, local_track_idx] = score_mat[det_row, track_col]
                edge_features.append(
                    [
                        float(base_scores[det_row, track_col].item()),
                        float(score_mat[det_row, track_col].item()),
                        float(motion_scores[det_row, track_col].item()) if motion_scores is not None else 0.0,
                        float(track_gaps_all[track_col].item()),
                        float(hist_len_all[track_col].item()),
                        float(local_rank + 1) / float(max(topk, 1)),
                    ]
                )

        if edge_features:
            edge_features_tensor = torch.tensor(edge_features, device=device, dtype=dtype)
            edge_det_index_tensor = torch.tensor(edge_det_index, device=device, dtype=torch.long)
            edge_track_index_tensor = torch.tensor(edge_track_index, device=device, dtype=torch.long)
        else:
            edge_features_tensor = torch.zeros((0, 6), device=device, dtype=dtype)
            edge_det_index_tensor = torch.zeros((0,), device=device, dtype=torch.long)
            edge_track_index_tensor = torch.zeros((0,), device=device, dtype=torch.long)

        row_degree = torch.zeros((num_local_dets,), device=device, dtype=dtype)
        col_degree = torch.zeros((num_local_tracks,), device=device, dtype=dtype)
        for det_local_idx, track_local_idx in zip(edge_det_index, edge_track_index):
            row_degree[int(det_local_idx)] += 1.0
            col_degree[int(track_local_idx)] += 1.0
        for det_local_idx in range(num_local_dets):
            det_features[det_local_idx][1] = float(row_degree[det_local_idx].item())

        track_features: List[List[float]] = []
        for track_col in track_cols:
            local_track_idx = int(track_col_to_local[track_col])
            track_features.append(
                [
                    float(track_gaps_all[track_col].item()),
                    float(hist_len_all[track_col].item()),
                    float(col_degree[local_track_idx].item()),
                ]
            )

        cluster_features = torch.tensor(
            [
                float(num_local_dets),
                float(num_local_tracks),
                float(len(edge_features)),
                float(row_degree.mean().item()) if row_degree.numel() > 0 else 0.0,
                float(row_degree.max().item()) if row_degree.numel() > 0 else 0.0,
                float(col_degree.mean().item()) if col_degree.numel() > 0 else 0.0,
                float(col_degree.max().item()) if col_degree.numel() > 0 else 0.0,
            ],
            device=device,
            dtype=dtype,
        )

        return {
            "det_features": torch.tensor(det_features, device=device, dtype=dtype),
            "track_features": torch.tensor(track_features, device=device, dtype=dtype),
            "edge_features": edge_features_tensor,
            "edge_det_index": edge_det_index_tensor,
            "edge_track_index": edge_track_index_tensor,
            "cluster_features": cluster_features,
            "dense_edge_mask": dense_edge_mask,
            "dense_refined_scores": dense_refined_scores,
        }

    def _build_local_conflict_runtime_features_v2(
        self,
        *,
        base_scores: torch.Tensor,
        score_mat: torch.Tensor,
        det_scores: torch.Tensor,
        det_rows: List[int],
        track_cols: List[int],
        track_history_masks: Optional[torch.Tensor],
        track_history_times: Optional[torch.Tensor],
        motion_scores: Optional[torch.Tensor],
        det_boxes_cxcywh: Optional[torch.Tensor],
        track_boxes_cxcywh: Optional[torch.Tensor],
        host_variant: str,
    ) -> Dict[str, Any]:
        device = score_mat.device
        dtype = score_mat.dtype
        det_rows = [int(x) for x in det_rows]
        track_cols = [int(x) for x in track_cols]
        num_local_dets = len(det_rows)
        num_local_tracks = len(track_cols)
        topk = int(max(min(self.assoc_local_conflict_graph_topk, score_mat.shape[1]), 1))
        det_row_tensor = torch.as_tensor(det_rows, device=device, dtype=torch.long)
        track_col_tensor = torch.as_tensor(track_cols, device=device, dtype=torch.long)
        track_col_to_local = {int(track_col): idx for idx, track_col in enumerate(track_cols)}

        score_rows = score_mat.index_select(0, det_row_tensor)
        top_vals, top_idx = torch.topk(score_rows, k=topk, dim=1, sorted=True)
        top_base = torch.gather(base_scores.index_select(0, det_row_tensor), dim=1, index=top_idx)
        if motion_scores is not None:
            top_motion = torch.gather(motion_scores.index_select(0, det_row_tensor).to(dtype=dtype), dim=1, index=top_idx)
        else:
            top_motion = torch.zeros_like(top_vals)

        if track_history_masks is not None:
            hist_len_all = (~track_history_masks).sum(dim=1).to(device=device, dtype=dtype)
            track_gaps_all = self._build_runtime_replay_track_gaps(track_history_masks, track_history_times).to(
                device=device,
                dtype=dtype,
            )
        else:
            hist_len_all = torch.zeros((score_mat.shape[1],), device=device, dtype=dtype)
            track_gaps_all = torch.zeros((score_mat.shape[1],), device=device, dtype=dtype)

        if det_boxes_cxcywh is not None and det_boxes_cxcywh.shape[0] > 0:
            det_boxes_local = det_boxes_cxcywh.index_select(0, det_row_tensor).to(device=device, dtype=torch.float32)
        else:
            det_boxes_local = torch.zeros((num_local_dets, 4), device=device, dtype=torch.float32)
        if track_boxes_cxcywh is not None and track_boxes_cxcywh.shape[0] > 0 and num_local_tracks > 0:
            track_boxes_local = track_boxes_cxcywh.index_select(0, track_col_tensor).to(device=device, dtype=torch.float32)
        else:
            track_boxes_local = torch.zeros((num_local_tracks, 4), device=device, dtype=torch.float32)

        det_features: List[List[float]] = []
        edge_records: List[Dict[str, Any]] = []
        dense_edge_mask = torch.zeros((num_local_dets, num_local_tracks), device=device, dtype=torch.bool)
        dense_refined_scores = torch.zeros((num_local_dets, num_local_tracks), device=device, dtype=dtype)
        track_gap_acc: Dict[int, List[float]] = defaultdict(list)
        track_hist_acc: Dict[int, List[float]] = defaultdict(list)
        track_box_acc: Dict[int, List[torch.Tensor]] = defaultdict(list)
        col_refined_scores: Dict[int, List[float]] = defaultdict(list)
        row_entropy_values: List[float] = []
        row_margin_values: List[float] = []

        for local_det_idx, det_row in enumerate(det_rows):
            row_refined = top_vals[local_det_idx].to(dtype=torch.float32)
            row_base = top_base[local_det_idx].to(dtype=torch.float32)
            row_motion = top_motion[local_det_idx].to(dtype=torch.float32)
            row_refined_probs = (
                _local_conflict_softmax_probs_1d(row_refined)
                if _local_conflict_softmax_probs_1d is not None
                else torch.softmax(row_refined - row_refined.max(), dim=0)
            )
            row_entropy = float((-(row_refined_probs * torch.log(row_refined_probs.clamp(min=1e-8))).sum()).item())
            row_margin = 0.0
            if row_refined.numel() > 1:
                row_margin = float((row_refined[0] - row_refined[1]).item())
            row_entropy_values.append(row_entropy)
            row_margin_values.append(row_margin)
            det_box = det_boxes_local[local_det_idx]
            det_cx = float(det_box[0].item())
            det_cy = float(det_box[1].item())
            det_w = max(float(det_box[2].item()), 1e-6)
            det_h = max(float(det_box[3].item()), 1e-6)
            det_features.append(
                [
                    float(det_scores[det_row].item()) if det_scores is not None else 0.0,
                    0.0,
                    row_margin,
                    row_entropy,
                    det_cx,
                    det_cy,
                    float(np.log(det_w)),
                    float(np.log(det_h)),
                    float(det_w / max(det_h, 1e-6)),
                ]
            )

            row_base_z = (
                _local_conflict_zscore_1d(row_base)
                if _local_conflict_zscore_1d is not None
                else (row_base - row_base.mean()) / (row_base.std(unbiased=False) + 1e-6)
            )
            row_refined_z = (
                _local_conflict_zscore_1d(row_refined)
                if _local_conflict_zscore_1d is not None
                else (row_refined - row_refined.mean()) / (row_refined.std(unbiased=False) + 1e-6)
            )
            row_motion_z = (
                _local_conflict_zscore_1d(row_motion)
                if _local_conflict_zscore_1d is not None
                else (row_motion - row_motion.mean()) / (row_motion.std(unbiased=False) + 1e-6)
            )
            row_top1 = float(row_refined.max().item()) if row_refined.numel() > 0 else 0.0

            for local_rank in range(int(top_vals.shape[1])):
                track_col = int(top_idx[local_det_idx, local_rank].item())
                if track_col not in track_col_to_local:
                    continue
                score_val = float(top_vals[local_det_idx, local_rank].item())
                if score_val <= 0.0:
                    continue
                local_track_idx = int(track_col_to_local[track_col])
                dense_edge_mask[local_det_idx, local_track_idx] = True
                dense_refined_scores[local_det_idx, local_track_idx] = score_mat[det_row, track_col]
                track_gap_acc[local_track_idx].append(float(track_gaps_all[track_col].item()))
                track_hist_acc[local_track_idx].append(float(hist_len_all[track_col].item()))
                track_box = track_boxes_local[local_track_idx] if num_local_tracks > 0 else det_box.new_zeros((4,))
                track_box_acc[local_track_idx].append(track_box.detach().clone())
                if _local_conflict_pair_geometry_features is not None:
                    geom = _local_conflict_pair_geometry_features(
                        det_box.view(1, 4),
                        track_box.view(1, 4),
                    )
                    iou_score = float(geom["iou"].view(-1)[0].item())
                    bbox_dist_score = float(geom["bbox_dist_score"].view(-1)[0].item())
                    delta_cx_norm = float(geom["delta_cx_norm"].view(-1)[0].item())
                    delta_cy_norm = float(geom["delta_cy_norm"].view(-1)[0].item())
                    delta_log_w = float(geom["delta_log_w"].view(-1)[0].item())
                    delta_log_h = float(geom["delta_log_h"].view(-1)[0].item())
                else:
                    iou_score = 0.0
                    bbox_dist_score = 0.0
                    delta_cx_norm = 0.0
                    delta_cy_norm = 0.0
                    delta_log_w = 0.0
                    delta_log_h = 0.0
                edge_records.append(
                    {
                        "det_local_idx": int(local_det_idx),
                        "track_local_idx": int(local_track_idx),
                        "base_score_raw": float(top_base[local_det_idx, local_rank].item()),
                        "refined_score_raw": score_val,
                        "motion_score_raw": float(top_motion[local_det_idx, local_rank].item()),
                        "base_score_row_z": float(row_base_z[local_rank].item()),
                        "refined_score_row_z": float(row_refined_z[local_rank].item()),
                        "motion_score_row_z": float(row_motion_z[local_rank].item()),
                        "refined_score_row_softmax": float(row_refined_probs[local_rank].item()),
                        "refined_gap_to_row_top1": float(row_top1 - score_val),
                        "rank_frac": float(local_rank + 1) / float(max(int(top_vals.shape[1]), 1)),
                        "iou": iou_score,
                        "bbox_dist_score": bbox_dist_score,
                        "delta_cx_norm": delta_cx_norm,
                        "delta_cy_norm": delta_cy_norm,
                        "delta_log_w": delta_log_w,
                        "delta_log_h": delta_log_h,
                    }
                )
                col_refined_scores[local_track_idx].append(score_val)

        edge_det_index = [int(record["det_local_idx"]) for record in edge_records]
        edge_track_index = [int(record["track_local_idx"]) for record in edge_records]
        row_degree = torch.zeros((num_local_dets,), device=device, dtype=dtype)
        col_degree = torch.zeros((num_local_tracks,), device=device, dtype=dtype)
        for det_local_idx, track_local_idx in zip(edge_det_index, edge_track_index):
            row_degree[int(det_local_idx)] += 1.0
            col_degree[int(track_local_idx)] += 1.0
        for det_local_idx in range(num_local_dets):
            det_features[det_local_idx][1] = float(row_degree[det_local_idx].item())

        track_features: List[List[float]] = []
        for local_track_idx in range(num_local_tracks):
            gap_values = track_gap_acc.get(local_track_idx, [])
            hist_values = track_hist_acc.get(local_track_idx, [])
            track_boxes = track_box_acc.get(local_track_idx, [])
            gap_mean = float(sum(gap_values) / len(gap_values)) if gap_values else 0.0
            hist_mean = float(sum(hist_values) / len(hist_values)) if hist_values else 0.0
            if track_boxes:
                stacked = torch.stack(track_boxes, dim=0).mean(dim=0)
                track_cx = float(stacked[0].item())
                track_cy = float(stacked[1].item())
                track_w = max(float(stacked[2].item()), 1e-6)
                track_h = max(float(stacked[3].item()), 1e-6)
            else:
                track_cx = track_cy = 0.0
                track_w = track_h = 1e-6
            track_features.append(
                [
                    float(np.log1p(max(gap_mean, 0.0))),
                    float(np.log1p(max(hist_mean, 0.0))),
                    float(col_degree[local_track_idx].item()) if local_track_idx < col_degree.numel() else 0.0,
                    track_cx,
                    track_cy,
                    float(np.log(track_w)),
                    float(np.log(track_h)),
                    float(track_w / max(track_h, 1e-6)),
                ]
            )

        col_z_by_track: Dict[int, List[float]] = {}
        for local_track_idx, scores_for_track in col_refined_scores.items():
            score_tensor = torch.tensor(scores_for_track, device=device, dtype=torch.float32)
            col_z = (
                _local_conflict_zscore_1d(score_tensor)
                if _local_conflict_zscore_1d is not None
                else (score_tensor - score_tensor.mean()) / (score_tensor.std(unbiased=False) + 1e-6)
            )
            col_z_by_track[local_track_idx] = [float(x) for x in col_z.tolist()]

        col_offsets: Dict[int, int] = defaultdict(int)
        edge_features: List[List[float]] = []
        for record in edge_records:
            local_track_idx = int(record["track_local_idx"])
            local_offset = int(col_offsets[local_track_idx])
            col_offsets[local_track_idx] += 1
            refined_score_col_z = 0.0
            if local_track_idx in col_z_by_track and local_offset < len(col_z_by_track[local_track_idx]):
                refined_score_col_z = float(col_z_by_track[local_track_idx][local_offset])
            edge_features.append(
                [
                    float(record["base_score_raw"]),
                    float(record["refined_score_raw"]),
                    float(record["motion_score_raw"]),
                    float(record["base_score_row_z"]),
                    float(record["refined_score_row_z"]),
                    float(record["motion_score_row_z"]),
                    float(record["refined_score_row_softmax"]),
                    float(record["refined_gap_to_row_top1"]),
                    float(record["rank_frac"]),
                    refined_score_col_z,
                    float(record["refined_score_raw"] - record["base_score_raw"]),
                    float(record["motion_score_raw"] - record["refined_score_raw"]),
                    float(record["iou"]),
                    float(record["bbox_dist_score"]),
                    float(record["delta_cx_norm"]),
                    float(record["delta_cy_norm"]),
                    float(record["delta_log_w"]),
                    float(record["delta_log_h"]),
                ]
            )

        cluster_features = torch.tensor(
            [
                float(num_local_dets),
                float(num_local_tracks),
                float(len(edge_features)),
                float(row_degree.mean().item()) if row_degree.numel() > 0 else 0.0,
                float(row_degree.max().item()) if row_degree.numel() > 0 else 0.0,
                float(col_degree.mean().item()) if col_degree.numel() > 0 else 0.0,
                float(col_degree.max().item()) if col_degree.numel() > 0 else 0.0,
                float(np.mean(row_entropy_values)) if row_entropy_values else 0.0,
                float(np.max(row_entropy_values)) if row_entropy_values else 0.0,
                float(np.mean(row_margin_values)) if row_margin_values else 0.0,
                float(np.max(row_margin_values)) if row_margin_values else 0.0,
            ],
            device=device,
            dtype=dtype,
        )

        host_variant_id = 0
        if _encode_local_conflict_host_variant is not None:
            host_variant_id = int(
                _encode_local_conflict_host_variant(
                    str(host_variant or self.assoc_local_conflict_graph_host_variant or ""),
                    getattr(self, "local_conflict_host_vocab", ["unknown"]),
                )
            )

        if edge_features:
            edge_features_tensor = torch.tensor(edge_features, device=device, dtype=dtype)
            edge_det_index_tensor = torch.tensor(edge_det_index, device=device, dtype=torch.long)
            edge_track_index_tensor = torch.tensor(edge_track_index, device=device, dtype=torch.long)
        else:
            edge_features_tensor = torch.zeros((0, 18), device=device, dtype=dtype)
            edge_det_index_tensor = torch.zeros((0,), device=device, dtype=torch.long)
            edge_track_index_tensor = torch.zeros((0,), device=device, dtype=torch.long)

        return {
            "det_features": torch.tensor(det_features, device=device, dtype=dtype),
            "track_features": torch.tensor(track_features, device=device, dtype=dtype),
            "edge_features": edge_features_tensor,
            "edge_det_index": edge_det_index_tensor,
            "edge_track_index": edge_track_index_tensor,
            "cluster_features": cluster_features,
            "dense_edge_mask": dense_edge_mask,
            "dense_refined_scores": dense_refined_scores,
            "host_variant_id": host_variant_id,
        }

    def _get_local_conflict_graph_learned_commit_plan(
        self,
        *,
        base_scores: torch.Tensor,
        score_mat: torch.Tensor,
        det_scores: torch.Tensor,
        active_ids_list: List[int],
        track_history_masks: Optional[torch.Tensor],
        track_history_times: Optional[torch.Tensor],
        motion_scores: Optional[torch.Tensor],
        det_boxes_cxcywh: Optional[torch.Tensor],
        track_boxes_cxcywh: Optional[torch.Tensor],
        thresh: float,
    ) -> Dict[str, Any]:
        plan = {
            "assignments": [],
            "resolved_dets": set(),
            "blocked_tracks": set(),
            "stats": {
                "graph_mode": "learned_commit",
                "eligible_clusters": 0,
                "replaced_clusters": 0,
                "resolved_dets": 0,
                "matched_dets": 0,
                "null_dets": 0,
                "deferred_dets": 0,
                "blocked_tracks": 0,
                "gate_pass_clusters": 0,
                "gate_filtered_clusters": 0,
                "trigger_filtered_clusters": 0,
                "skipped_large_clusters": 0,
            },
        }
        if (
            not self.assoc_use_local_conflict_graph
            or self.assoc_local_conflict_graph_mode != "learned_commit"
            or self.local_conflict_commit_model is None
            or build_topk_bipartite_components is None
            or filter_local_conflict_clusters_by_size is None
            or solve_assignment_with_private_defer is None
            or not _LOCAL_CONFLICT_COMMIT_AVAILABLE
        ):
            return plan
        if score_mat.ndim != 2 or score_mat.numel() == 0:
            return plan
        if score_mat.shape[0] == 0 or score_mat.shape[1] == 0:
            return plan

        components = build_topk_bipartite_components(
            score_mat=score_mat,
            topk=int(self.assoc_local_conflict_graph_topk),
            min_edge_score=0.0,
        )
        eligible_components, skipped_large = filter_local_conflict_clusters_by_size(
            components,
            min_detections=int(self.assoc_local_conflict_graph_min_detections),
            max_detections=int(self.assoc_local_conflict_graph_max_detections),
            max_tracks=int(self.assoc_local_conflict_graph_max_tracks),
        )
        plan["stats"]["skipped_large_clusters"] = int(skipped_large)
        if not eligible_components:
            return plan

        self.local_conflict_commit_model = self.local_conflict_commit_model.to(device=score_mat.device)
        for component in eligible_components:
            det_rows = [int(x) for x in component.get("det_rows", [])]
            track_cols = [int(x) for x in component.get("track_cols", [])]
            if not det_rows or not track_cols:
                continue
            plan["stats"]["eligible_clusters"] += 1
            feature_pack = self._build_local_conflict_runtime_features(
                base_scores=base_scores,
                score_mat=score_mat,
                det_scores=det_scores,
                det_rows=det_rows,
                track_cols=track_cols,
                track_history_masks=track_history_masks,
                track_history_times=track_history_times,
                motion_scores=motion_scores,
                det_boxes_cxcywh=det_boxes_cxcywh,
                track_boxes_cxcywh=track_boxes_cxcywh,
                host_variant=str(self.assoc_local_conflict_graph_host_variant or ""),
            )
            if feature_pack["edge_features"].shape[0] == 0:
                plan["stats"]["trigger_filtered_clusters"] += 1
                plan["stats"]["deferred_dets"] += int(len(det_rows))
                continue

            with torch.inference_mode():
                model_kwargs = {
                    "det_features": feature_pack["det_features"],
                    "track_features": feature_pack["track_features"],
                    "edge_features": feature_pack["edge_features"],
                    "edge_det_index": feature_pack["edge_det_index"],
                    "edge_track_index": feature_pack["edge_track_index"],
                    "cluster_features": feature_pack["cluster_features"],
                }
                if self.local_conflict_model_family == "set_predictor_v2":
                    model_kwargs["host_variant_id"] = int(feature_pack.get("host_variant_id", 0))
                outputs = self.local_conflict_commit_model(**model_kwargs)
            if self.local_conflict_model_family == "set_predictor_v2":
                gate_logit = outputs["cluster_commit_logit"].view(())
                gate_logit = (
                    gate_logit / max(float(self.assoc_local_conflict_graph_cluster_gate_temp), 1e-6)
                    + float(self.assoc_local_conflict_graph_cluster_gate_bias)
                )
                cluster_gate_prob = float(torch.sigmoid(gate_logit).item())
                if cluster_gate_prob < float(self.assoc_local_conflict_graph_cluster_gate_thresh):
                    plan["stats"]["gate_filtered_clusters"] += 1
                    plan["stats"]["deferred_dets"] += int(len(det_rows))
                    continue
                plan["stats"]["gate_pass_clusters"] += 1
                dense_logits = HostConditionedLocalConflictSetPredictor.build_dense_assignment_logits(
                    num_detections=len(det_rows),
                    num_tracks=len(track_cols),
                    edge_logits=outputs["edge_logits"],
                    edge_det_index=feature_pack["edge_det_index"],
                    edge_track_index=feature_pack["edge_track_index"],
                    defer_logits=outputs["defer_logits"],
                )
            else:
                dense_logits = LocalConflictCommitRefiner.build_dense_assignment_logits(
                    num_detections=len(det_rows),
                    num_tracks=len(track_cols),
                    edge_logits=outputs["edge_logits"],
                    edge_det_index=feature_pack["edge_det_index"],
                    edge_track_index=feature_pack["edge_track_index"],
                    defer_logits=outputs["defer_logits"],
                )
            feasible_mask = feature_pack["dense_edge_mask"] & (
                feature_pack["dense_refined_scores"] >= float(thresh)
            )
            assignments = solve_assignment_with_private_defer(
                score_sub=dense_logits[:, : len(track_cols)],
                feasible_mask=feasible_mask,
                defer_scores=dense_logits[:, len(track_cols)],
                use_hungarian=(self.matching_method == "hungarian"),
            )

            matched_pairs: List[Tuple[int, int]] = []
            matched_track_cols: set[int] = set()
            for assignment in assignments:
                det_local_idx = int(assignment.get("det_local_idx", -1))
                if det_local_idx < 0 or det_local_idx >= len(det_rows):
                    continue
                track_local_idx = assignment.get("track_local_idx", None)
                if track_local_idx is None:
                    continue
                track_local_idx = int(track_local_idx)
                if track_local_idx < 0 or track_local_idx >= len(track_cols):
                    continue
                if not bool(feasible_mask[det_local_idx, track_local_idx].item()):
                    continue
                det_row = int(det_rows[det_local_idx])
                track_col = int(track_cols[track_local_idx])
                matched_pairs.append((det_row, track_col))
                matched_track_cols.add(track_col)

            if len(matched_pairs) < int(self.assoc_local_conflict_graph_min_committed_matches):
                plan["stats"]["trigger_filtered_clusters"] += 1
                plan["stats"]["deferred_dets"] += int(len(det_rows))
                continue

            plan["stats"]["replaced_clusters"] += 1
            plan["stats"]["deferred_dets"] += int(len(det_rows) - len(matched_pairs))
            for det_row, track_col in matched_pairs:
                plan["resolved_dets"].add(det_row)
                plan["assignments"].append((det_row, track_col))
                plan["stats"]["matched_dets"] += 1
            for track_col in matched_track_cols:
                plan["blocked_tracks"].add(int(track_col))

        plan["assignments"].sort(key=lambda x: (int(x[0]), int(x[1])))
        plan["stats"]["resolved_dets"] = int(len(plan["resolved_dets"]))
        plan["stats"]["blocked_tracks"] = int(len(plan["blocked_tracks"]))
        return plan

    def _refine_assoc_scores_with_competition(
        self,
        base_scores: torch.Tensor,
        det_scores: torch.Tensor,
        track_history_masks: Optional[torch.Tensor],
        track_history_times: Optional[torch.Tensor],
        motion_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.assoc_use_competition:
            return base_scores
        if not self.assoc_use_competition_oracle and self.competition_assoc is None:
            return base_scores
        if base_scores.numel() == 0 or base_scores.ndim != 2:
            return base_scores

        num_dets, num_tracks = base_scores.shape
        if num_dets == 0 or num_tracks == 0:
            return base_scores

        topk = int(max(min(self.assoc_competition_topk, num_tracks), 1))
        if topk <= 0:
            return base_scores

        if num_tracks > 1:
            top2 = torch.topk(base_scores, k=min(2, num_tracks), dim=1, sorted=True).values
            rank_margin_all = top2[:, 0] - top2[:, 1]
        else:
            rank_margin_all = torch.zeros((num_dets,), device=base_scores.device, dtype=base_scores.dtype)

        if self.assoc_competition_margin_threshold is not None and num_tracks > 1:
            eligible_mask = rank_margin_all < float(self.assoc_competition_margin_threshold)
            eligible_idx = eligible_mask.nonzero(as_tuple=True)[0]
            if eligible_idx.numel() == 0:
                return base_scores
        else:
            eligible_idx = torch.arange(num_dets, device=base_scores.device, dtype=torch.long)

        base_scores_sel = base_scores.index_select(0, eligible_idx)
        det_scores_sel = det_scores.index_select(0, eligible_idx).to(dtype=base_scores.dtype)
        top_idx = torch.topk(base_scores_sel, k=topk, dim=1, sorted=True).indices
        top_base = torch.gather(base_scores_sel, dim=1, index=top_idx)
        top_motion = (
            torch.gather(motion_scores.index_select(0, eligible_idx).to(dtype=base_scores.dtype), dim=1, index=top_idx)
            if motion_scores is not None
            else torch.zeros_like(top_base)
        )

        if track_history_masks is not None:
            hist_len_all = (~track_history_masks).sum(dim=1).to(device=base_scores.device, dtype=base_scores.dtype)
            track_gaps_all = self._build_runtime_replay_track_gaps(track_history_masks, track_history_times).to(
                device=base_scores.device,
                dtype=base_scores.dtype,
            )
        else:
            hist_len_all = torch.zeros((num_tracks,), device=base_scores.device, dtype=base_scores.dtype)
            track_gaps_all = torch.zeros((num_tracks,), device=base_scores.device, dtype=base_scores.dtype)

        top_track_gaps = track_gaps_all[top_idx]
        top_hist_len = hist_len_all[top_idx]
        rank_entropy = _score_entropy_from_probs(top_base)
        group_features = torch.stack(
            [
                torch.full((eligible_idx.shape[0],), float(topk), device=base_scores.device, dtype=base_scores.dtype),
                torch.full((eligible_idx.shape[0],), float(num_tracks), device=base_scores.device, dtype=base_scores.dtype),
                rank_margin_all.index_select(0, eligible_idx).to(dtype=base_scores.dtype),
                rank_entropy.to(dtype=base_scores.dtype),
                top_track_gaps.min(dim=1).values.to(dtype=base_scores.dtype),
                top_track_gaps.mean(dim=1).to(dtype=base_scores.dtype),
                top_hist_len.mean(dim=1).to(dtype=base_scores.dtype),
                det_scores_sel,
            ],
            dim=-1,
        )

        rank_frac = (
            (torch.arange(topk, device=base_scores.device, dtype=base_scores.dtype).view(1, -1) + 1.0)
            / float(max(topk, 1))
        ).expand(eligible_idx.shape[0], -1)
        candidate_features = torch.stack(
            [
                top_base,
                top_base,
                top_motion,
                top_track_gaps,
                top_hist_len,
                rank_frac,
            ],
            dim=-1,
        )
        valid_mask = torch.ones((eligible_idx.shape[0], topk), device=base_scores.device, dtype=torch.bool)

        if self.assoc_use_competition_oracle:
            refined_scores = base_scores.clone()
            target = refined_scores.index_select(0, eligible_idx)
            local_max = top_base.max(dim=1).values
            for local_row, det_row in enumerate(eligible_idx.tolist()):
                group_id = f"{self.sequence_name or ''}:{int(self.frame_id)}:{int(det_row)}"
                oracle = self.competition_oracle.get(group_id)
                if not oracle:
                    continue
                if int(oracle.get("action_target_id", 0)) != _COMP_ACTION_LABELS.index("rerank"):
                    continue
                target_rank = int(oracle.get("target_candidate_rank", -1))
                if target_rank < 1 or target_rank > topk:
                    continue
                winner_col = top_idx[local_row, target_rank - 1]
                target[local_row, winner_col] = local_max[local_row] + 1e-4
            refined_scores.index_copy_(0, eligible_idx, target)
            return refined_scores

        try:
            self.competition_assoc = self.competition_assoc.to(device=base_scores.device)
            with torch.inference_mode():
                outputs = self.competition_assoc(
                    group_features=group_features,
                    candidate_features=candidate_features,
                    valid_mask=valid_mask,
                )
        except Exception as exc:
            warnings.warn(
                f"[RuntimeTrackerByteTrack] Competition refinement failed; fallback to base scores. Error: {exc}"
            )
            return base_scores

        if self.assoc_competition_mode == "noop":
            return base_scores

        action_prob = outputs["action_prob"]
        action_pred = outputs["action_logits"].argmax(dim=-1)
        if self.assoc_competition_hard_action:
            rerank_weight = (action_pred == _COMP_ACTION_LABELS.index("rerank")).to(dtype=base_scores.dtype)
        else:
            rerank_weight = action_prob[:, _COMP_ACTION_LABELS.index("rerank")].to(dtype=base_scores.dtype)

        if not bool((rerank_weight > 0).any()):
            return base_scores

        if self.assoc_competition_mode == "rerank_minimal":
            candidate_pred = outputs["candidate_logits"].argmax(dim=-1)
            refined_scores = base_scores.clone()
            target = refined_scores.index_select(0, eligible_idx)
            local_max = top_base.max(dim=1).values
            winner_rows = (rerank_weight > 0).nonzero(as_tuple=True)[0]
            if winner_rows.numel() > 0:
                winner_cols = top_idx[winner_rows, candidate_pred[winner_rows]]
                target[winner_rows, winner_cols] = local_max[winner_rows] + 1e-4
                refined_scores.index_copy_(0, eligible_idx, target)
            return refined_scores

        candidate_prob = outputs["candidate_prob"].to(dtype=base_scores.dtype)
        valid_float = valid_mask.to(dtype=base_scores.dtype)
        mean_prob = (candidate_prob * valid_float).sum(dim=1, keepdim=True) / valid_float.sum(dim=1, keepdim=True).clamp(min=1.0)
        delta = (candidate_prob - mean_prob) * float(self.assoc_competition_delta_scale)
        delta = delta * rerank_weight.unsqueeze(1)
        refined_top = top_base + delta

        refined_scores = base_scores.clone()
        target = refined_scores.index_select(0, eligible_idx)
        target.scatter_(1, top_idx, refined_top.to(dtype=refined_scores.dtype))
        refined_scores.index_copy_(0, eligible_idx, target)
        return refined_scores

    def _refine_assoc_scores_with_runtime_replay(
        self,
        base_scores: torch.Tensor,
        det_features: torch.Tensor,
        track_history_features: torch.Tensor,
        track_history_masks: torch.Tensor,
        track_history_times: Optional[torch.Tensor],
        det_scores: torch.Tensor,
        motion_scores: Optional[torch.Tensor] = None,
        det_boxes_cxcywh: Optional[torch.Tensor] = None,
        track_boxes_cxcywh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.assoc_use_runtime_replay or self.runtime_replay_assoc is None:
            return base_scores
        if base_scores.numel() == 0:
            return base_scores
        if det_features is None or track_history_features is None or track_history_masks is None:
            return base_scores
        feat_dim = int(getattr(self.runtime_replay_assoc, "feat_dim", det_features.shape[-1]))
        if det_features.shape[-1] != feat_dim or track_history_features.shape[-1] != feat_dim:
            if not getattr(self, "_warned_runtime_replay_feat_dim", False):
                warnings.warn(
                    "[RuntimeTrackerByteTrack] Runtime replay feature-dim mismatch; "
                    f"expected {feat_dim}, got det={det_features.shape[-1]} hist={track_history_features.shape[-1]}. "
                    "Falling back to base scores."
                )
                self._warned_runtime_replay_feat_dim = True
            return base_scores
        if track_boxes_cxcywh is None or det_boxes_cxcywh is None:
            return base_scores

        try:
            self.runtime_replay_assoc = self.runtime_replay_assoc.to(device=base_scores.device)
            refined_scores = base_scores.clone()
            num_dets, num_tracks = base_scores.shape
            if num_tracks == 0:
                return refined_scores

            hist_len_all = (~track_history_masks).sum(dim=1)
            track_gaps_all = self._build_runtime_replay_track_gaps(track_history_masks, track_history_times)
            topk = int(max(min(getattr(self.runtime_replay_assoc, "topk", num_tracks), num_tracks), 1))
            if self.assoc_runtime_replay_hard_margin_gate and num_tracks > 1:
                gate_threshold = self.assoc_runtime_replay_margin_threshold
                if gate_threshold is None:
                    gate_threshold = float(getattr(self.runtime_replay_assoc, "margin_threshold", 0.10))
                top2 = torch.topk(base_scores, k=2, dim=1, sorted=True).values
                base_margin = top2[:, 0] - top2[:, 1]
                eligible_mask = base_margin < float(gate_threshold)
                eligible_idx = eligible_mask.nonzero(as_tuple=True)[0]
                if eligible_idx.numel() == 0:
                    return refined_scores
            else:
                eligible_idx = torch.arange(num_dets, device=base_scores.device, dtype=torch.long)

            base_scores_sel = base_scores.index_select(0, eligible_idx)
            det_features_sel = det_features.index_select(0, eligible_idx)
            det_scores_sel = det_scores.index_select(0, eligible_idx)
            det_boxes_sel = det_boxes_cxcywh.index_select(0, eligible_idx)
            motion_scores_sel = (
                motion_scores.index_select(0, eligible_idx).to(dtype=base_scores.dtype)
                if motion_scores is not None
                else None
            )

            top_idx = torch.topk(base_scores_sel, k=min(topk, num_tracks), dim=1, sorted=True).indices
            if top_idx.numel() == 0:
                return refined_scores

            anchor = torch.gather(base_scores_sel, dim=1, index=top_idx).clamp(min=1e-4, max=1.0 - 1e-4)
            motion = (
                torch.gather(motion_scores_sel, dim=1, index=top_idx)
                if motion_scores_sel is not None
                else torch.zeros_like(anchor)
            )
            top_track_gaps = track_gaps_all[top_idx]
            top_hist_len = hist_len_all[top_idx]
            top_track_boxes = track_boxes_cxcywh[top_idx].to(dtype=base_scores.dtype)
            scalar = self._build_runtime_replay_scalar_features(
                anchor_scores=anchor,
                motion_scores=motion,
                det_score=det_scores_sel.to(dtype=base_scores.dtype),
                track_gaps=top_track_gaps,
                hist_len=top_hist_len,
                det_box=det_boxes_sel.to(dtype=base_scores.dtype),
                track_boxes=top_track_boxes,
            )
            hist_features = track_history_features[top_idx]
            hist_masks = track_history_masks[top_idx]
            hist_times = (
                track_history_times[top_idx]
                if track_history_times is not None and track_history_times.numel() > 0
                else None
            )
            det_times = torch.full((eligible_idx.shape[0],), int(self.frame_id), device=base_scores.device, dtype=torch.long)

            with torch.inference_mode():
                result = self.runtime_replay_assoc(
                    anchor_scores=anchor,
                    scalar_features=scalar,
                    det_features=det_features_sel,
                    hist_features=hist_features,
                    hist_masks=hist_masks,
                    hist_times=hist_times,
                    det_times=det_times,
                    det_scores=det_scores_sel,
                    valid_mask=torch.ones(anchor.shape, device=base_scores.device, dtype=torch.bool),
                )
            candidate_scores = result.get("candidate_scores", None)
            if torch.is_tensor(candidate_scores) and candidate_scores.shape == anchor.shape:
                target = refined_scores.index_select(0, eligible_idx)
                target.scatter_(
                    1,
                    top_idx,
                    candidate_scores.to(device=base_scores.device, dtype=base_scores.dtype),
                )
                refined_scores.index_copy_(0, eligible_idx, target)
            return refined_scores
        except Exception as exc:
            warnings.warn(
                f"[RuntimeTrackerByteTrack] Runtime replay refinement failed; fallback to base scores. Error: {exc}"
            )
            return base_scores

    def _compute_feature_scores(
        self,
        det_features: torch.Tensor,
        track_features: torch.Tensor,
    ) -> torch.Tensor:
        if det_features.numel() == 0 or track_features.numel() == 0:
            return torch.zeros(
                (det_features.shape[0], track_features.shape[0]),
                device=det_features.device,
                dtype=det_features.dtype,
            )
        det_norm = F.normalize(det_features, dim=-1)
        track_norm = F.normalize(track_features, dim=-1)
        tau = max(self.assoc_feat_tau, 1e-6)
        sim = (det_norm @ track_norm.t()) / tau
        if self.assoc_feat_score_mode == "softmax":
            sim = sim - sim.max(dim=1, keepdim=True).values
            return torch.softmax(sim, dim=1)
        sim = (sim + 1.0) * 0.5
        return sim.clamp(min=0.0, max=1.0)

    def _compute_bbox_distance_scores(
        self,
        det_boxes_cxcywh: torch.Tensor,
        track_boxes_cxcywh: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute a bbox-based similarity score in [0, 1] between detections and tracks.

        This is a StableTrack-inspired geometry cue (bbox distance instead of Mahalanobis).
        Inputs are expected to be normalized cxcywh boxes.
        """
        if det_boxes_cxcywh is None or track_boxes_cxcywh is None:
            return torch.zeros((0, 0), device=distributed_device(), dtype=self.dtype)
        if det_boxes_cxcywh.numel() == 0 or track_boxes_cxcywh.numel() == 0:
            return torch.zeros(
                (det_boxes_cxcywh.shape[0], track_boxes_cxcywh.shape[0]),
                device=det_boxes_cxcywh.device,
                dtype=det_boxes_cxcywh.dtype,
            )

        # Do all computations in float32 for numerical stability (then cast back).
        det = det_boxes_cxcywh.to(dtype=torch.float32)
        trk = track_boxes_cxcywh.to(dtype=torch.float32)
        eps = 1e-6

        # (N, 1) vs (1, M)
        det_cx = det[:, 0].unsqueeze(1)
        det_cy = det[:, 1].unsqueeze(1)
        det_w = det[:, 2].unsqueeze(1).clamp(min=eps)
        det_h = det[:, 3].unsqueeze(1).clamp(min=eps)

        trk_cx = trk[:, 0].unsqueeze(0)
        trk_cy = trk[:, 1].unsqueeze(0)
        trk_w = trk[:, 2].unsqueeze(0).clamp(min=eps)
        trk_h = trk[:, 3].unsqueeze(0).clamp(min=eps)

        # Center distance normalized by track size + log-scale size difference.
        dx = (det_cx - trk_cx) / trk_w
        dy = (det_cy - trk_cy) / trk_h
        dw = torch.log(det_w / trk_w)
        dh = torch.log(det_h / trk_h)

        dist = torch.sqrt(dx * dx + dy * dy + dw * dw + dh * dh + eps)  # (N, M)
        tau = max(float(self.assoc_bbox_dist_tau), eps)
        sim = torch.exp(-dist / tau)
        return sim.to(device=det_boxes_cxcywh.device, dtype=det_boxes_cxcywh.dtype).clamp(min=0.0, max=1.0)

    def _match_with_scores(
        self,
        base_scores: torch.Tensor,
        scores: torch.Tensor,
        active_ids_list: List[int],
        track_history_masks: Optional[torch.Tensor] = None,
        track_history_times: Optional[torch.Tensor] = None,
        motion_scores: Optional[torch.Tensor] = None,
        det_boxes_cxcywh: Optional[torch.Tensor] = None,
        track_boxes_cxcywh: Optional[torch.Tensor] = None,
        cal_factor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = distributed_device()
        num_dets = base_scores.shape[0]
        num_tracks = base_scores.shape[1] if base_scores.dim() == 2 else 0

        id_labels = torch.full((num_dets,), -1, dtype=torch.long, device=device)

        # Optional: use detection confidence as a multiplier (keeps matching conservative).
        if self.assoc_use_det_score and scores is not None and base_scores.numel() > 0:
            base_scores = base_scores * scores.view(-1, 1).clamp(min=0.0, max=1.0)

        # Optional geometry cues: IoU and bbox-distance.
        iou = None
        want_iou = (
            (self.assoc_iou_gate > 0.0)
            or (self.assoc_iou_weight > 0.0)
            or (self.assoc_two_stage and (self.assoc_stage2_iou_gate is not None) and (self.assoc_stage2_iou_gate > 0.0))
        )
        if want_iou and det_boxes_cxcywh is not None and track_boxes_cxcywh is not None and num_tracks > 0:
            det_xyxy = _cxcywh_to_xyxy(det_boxes_cxcywh)
            track_xyxy = _cxcywh_to_xyxy(track_boxes_cxcywh)
            iou = _box_iou(det_xyxy, track_xyxy)

        bbox_scores = None
        want_bbox_scores = (
            (self.assoc_bbox_dist_weight > 0.0)
            or (self.assoc_two_stage and det_boxes_cxcywh is not None and track_boxes_cxcywh is not None and num_tracks > 0)
        )
        if want_bbox_scores and det_boxes_cxcywh is not None and track_boxes_cxcywh is not None and num_tracks > 0:
            bbox_scores = self._compute_bbox_distance_scores(det_boxes_cxcywh, track_boxes_cxcywh)

        # Stage-1 matching scores: base_scores (+ optional bbox distance), then IoU gating/weighting.
        score_mat = base_scores
        if bbox_scores is not None and self.assoc_bbox_dist_weight > 0.0:
            bbox_w = float(self.assoc_bbox_dist_weight)
            # Our twist (not a direct copy): optionally gate bbox-distance by the model's own
            # frequency-consistency-based reliability factor (cal_factor in [0,1]).
            # - low cal_factor => model uncertain => trust geometry more
            # - high cal_factor => model confident  => trust learned association more
            if self.assoc_bbox_dist_use_cal_factor and cal_factor is not None:
                try:
                    cf = cal_factor.detach()
                    if torch.is_tensor(cf):
                        cf = cf.to(device=score_mat.device, dtype=torch.float32).reshape(-1)
                        if cf.numel() == num_dets:
                            bbox_w_det = (1.0 - cf.clamp(0.0, 1.0)).view(-1, 1).to(score_mat.dtype)
                            score_mat = score_mat + bbox_w * bbox_w_det * bbox_scores.to(score_mat.dtype)
                        else:
                            score_mat = score_mat + bbox_w * bbox_scores.to(score_mat.dtype)
                    else:
                        score_mat = score_mat + bbox_w * bbox_scores.to(score_mat.dtype)
                except Exception:
                    score_mat = score_mat + bbox_w * bbox_scores.to(score_mat.dtype)
            else:
                score_mat = score_mat + bbox_w * bbox_scores.to(score_mat.dtype)

        if self.assoc_iou_gate > 0.0 and iou is not None:
            gate = iou >= self.assoc_iou_gate
            score_mat = score_mat * gate.to(score_mat.dtype)

        if self.assoc_iou_weight > 0.0 and iou is not None:
            # Keep backward compatibility: in logit-mode score_mat is already probs,
            # but in feature/hybrid it is a mixture. We still treat it as a generic similarity.
            score_mat = score_mat + float(self.assoc_iou_weight) * iou.to(score_mat.dtype)

        # Hungarian matching (if available)
        use_hungarian = self.matching_method == "hungarian"
        lsa = _get_linear_sum_assignment() if use_hungarian else None
        if use_hungarian and lsa is None:
            if self._hungarian_available is not False:
                print("[RuntimeTrackerByteTrack] scipy not available, fallback to greedy matching.")
                self._hungarian_available = False
            use_hungarian = False
        assigned_dets = set()
        assigned_tracks = set()
        resolved_dets = set()
        blocked_tracks = set()
        self._local_conflict_graph_last_stats = {}

        if self.assoc_use_local_conflict_graph and self.assoc_local_conflict_graph_mode in (
            "oracle_full",
            "oracle_commit_matches",
            "learned_commit",
        ):
            if self.assoc_local_conflict_graph_mode == "oracle_full":
                local_graph_plan = self._get_local_conflict_graph_oracle_full_plan(
                    score_mat=score_mat,
                    active_ids_list=active_ids_list,
                    thresh=float(self.id_thresh),
                )
            elif self.assoc_local_conflict_graph_mode == "oracle_commit_matches":
                local_graph_plan = self._get_local_conflict_graph_oracle_commit_matches_plan(
                    score_mat=score_mat,
                    active_ids_list=active_ids_list,
                    thresh=float(self.id_thresh),
                )
            else:
                local_graph_plan = self._get_local_conflict_graph_learned_commit_plan(
                    base_scores=base_scores,
                    score_mat=score_mat,
                    det_scores=scores,
                    active_ids_list=active_ids_list,
                    track_history_masks=track_history_masks,
                    track_history_times=track_history_times,
                    motion_scores=motion_scores,
                    det_boxes_cxcywh=det_boxes_cxcywh,
                    track_boxes_cxcywh=track_boxes_cxcywh,
                    thresh=float(self.id_thresh),
                )
            resolved_dets.update(int(x) for x in local_graph_plan.get("resolved_dets", set()))
            blocked_tracks.update(int(x) for x in local_graph_plan.get("blocked_tracks", set()))
            for det_i, trk_j in local_graph_plan.get("assignments", []):
                det_i = int(det_i)
                trk_j = int(trk_j)
                if det_i in assigned_dets or trk_j in assigned_tracks:
                    continue
                id_labels[det_i] = int(active_ids_list[trk_j])
                assigned_dets.add(det_i)
                assigned_tracks.add(trk_j)
            self._local_conflict_graph_last_stats = dict(local_graph_plan.get("stats", {}))
        elif self.assoc_use_local_conflict_graph_oracle:
            local_graph_plan = self._get_local_conflict_graph_oracle_assignments(
                base_scores=base_scores,
                score_mat=score_mat,
                active_ids_list=active_ids_list,
                thresh=float(self.id_thresh),
            )
            for det_i, trk_j in local_graph_plan.get("assignments", []):
                if det_i in assigned_dets or trk_j in assigned_tracks:
                    continue
                id_labels[det_i] = int(active_ids_list[trk_j])
                assigned_dets.add(det_i)
                assigned_tracks.add(trk_j)
            self._local_conflict_graph_last_stats = dict(local_graph_plan.get("stats", {}))

        if self._local_conflict_graph_last_stats:
            self._accumulate_local_conflict_graph_stats(self._local_conflict_graph_last_stats)

        def _assign_from_score_mat(
            score_sub: torch.Tensor,
            det_indices: List[int],
            track_indices: List[int],
            thresh: float,
        ) -> None:
            if score_sub.numel() == 0:
                return
            if lsa is not None and use_hungarian:
                cost = (-score_sub).detach().cpu().numpy()
                rows, cols = lsa(cost)
                for r, c in zip(rows, cols):
                    if score_sub[r, c] < thresh:
                        continue
                    det_i = int(det_indices[int(r)])
                    trk_j = int(track_indices[int(c)])
                    if det_i in assigned_dets or trk_j in assigned_tracks:
                        continue
                    id_labels[det_i] = int(active_ids_list[trk_j])
                    assigned_dets.add(det_i)
                    assigned_tracks.add(trk_j)
                return

            # Greedy matching fallback
            used_tracks = set()
            for r, det_i in enumerate(det_indices):
                det_i = int(det_i)
                if det_i in assigned_dets:
                    continue
                row = score_sub[r]
                if row.numel() == 0 or float(row.max().item()) < thresh:
                    continue
                best_local = int(row.argmax().item())
                if best_local in used_tracks:
                    continue
                trk_j = int(track_indices[best_local])
                if trk_j in assigned_tracks:
                    continue
                id_labels[det_i] = int(active_ids_list[trk_j])
                assigned_dets.add(det_i)
                assigned_tracks.add(trk_j)
                used_tracks.add(best_local)

        # Stage-1 assignment on all dets/tracks.
        det_all = [i for i in range(num_dets) if i not in resolved_dets]
        trk_all = [j for j in range(num_tracks) if j not in blocked_tracks]
        if det_all and trk_all:
            _assign_from_score_mat(score_mat[det_all][:, trk_all], det_all, trk_all, float(self.id_thresh))

        # Optional stage-2: recover matches using bbox-distance only (less strict than stage-1).
        if self.assoc_two_stage and bbox_scores is not None and num_tracks > 0:
            rem_dets = [i for i in range(num_dets) if i not in assigned_dets and i not in resolved_dets]
            rem_trks = [j for j in range(num_tracks) if j not in assigned_tracks and j not in blocked_tracks]
            if rem_dets and rem_trks:
                stage2_thresh = self.assoc_stage2_id_thresh
                if stage2_thresh is None:
                    # A conservative default: bbox similarity is in [0,1], so too-low thresholds can over-match.
                    stage2_thresh = 0.30
                    if not self._warned_stage2_defaults:
                        warnings.warn(
                            "[RuntimeTrackerByteTrack] ASSOC_TWO_STAGE=True but ASSOC_STAGE2_ID_THRESH is not set; "
                            "defaulting to 0.30."
                        )
                        self._warned_stage2_defaults = True
                stage2_iou_gate = float(self.assoc_stage2_iou_gate) if self.assoc_stage2_iou_gate is not None else 0.0
                stage2_bbox_w = float(self.assoc_stage2_bbox_weight) if self.assoc_stage2_bbox_weight is not None else 1.0

                score2 = bbox_scores[rem_dets][:, rem_trks].to(score_mat.dtype) * stage2_bbox_w
                if self.assoc_bbox_dist_use_cal_factor and cal_factor is not None:
                    try:
                        cf = cal_factor.detach()
                        if torch.is_tensor(cf):
                            cf = cf.to(device=score_mat.device, dtype=torch.float32).reshape(-1)
                            if cf.numel() == num_dets:
                                gate_w = (1.0 - cf.clamp(0.0, 1.0))[rem_dets].view(-1, 1).to(score2.dtype)
                                score2 = score2 * gate_w
                    except Exception:
                        pass
                if self.assoc_use_det_score and scores is not None:
                    score2 = score2 * scores[rem_dets].view(-1, 1).clamp(min=0.0, max=1.0)
                if stage2_iou_gate > 0.0 and iou is not None:
                    gate2 = iou[rem_dets][:, rem_trks] >= stage2_iou_gate
                    score2 = score2 * gate2.to(score2.dtype)

                _assign_from_score_mat(score2, rem_dets, rem_trks, float(stage2_thresh))

        # 未匹配检测分配新 ID 或标记未知
        for i in range(num_dets):
            if i in assigned_dets:
                continue
            # Full cluster replacement may explicitly resolve a detection to null.
            # Those rows must not fall through to newborn ID creation.
            if i in resolved_dets:
                continue
            if scores[i] >= self.newborn_thresh:
                self.max_id += 1
                new_id = self.max_id
                id_labels[i] = new_id
                self.active_ids.add(new_id)
                self._get_or_assign_vocab_id(new_id)

        return id_labels

    def _get_id_pred_labels(
        self,
        model_features: torch.Tensor,
        assoc_features: torch.Tensor,
        boxes_cxcywh: torch.Tensor,
        scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用频域关联模块进行 ID 预测

        参数：
            model_features: (N, feature_dim) - 当前帧检测特征（用于频域关联模型）
            assoc_features: (N, D) - 用于纯特征匹配/混合匹配的外观特征（可选 ReID）
            boxes_cxcywh: (N, 4) - 归一化的 cxcywh 框
            scores: (N,) - 检测置信度

        返回：
            id_labels: (N,) - 预测的 ID
        """
        device = distributed_device()
        num_dets = model_features.shape[0]

        # 构建 seq_info
        # 需要将历史轨迹和当前检测组合
        active_ids_list = list(self.active_ids)
        num_tracks = len(active_ids_list)
        max_tracks = int(self.num_id_vocabulary)
        if max_tracks > 0 and num_tracks > max_tracks:
            # 保留最近出现的轨迹，避免超过词表导致维度不匹配
            warnings.warn(
                f"[RuntimeTrackerByteTrack] Active tracks ({num_tracks}) exceed vocab ({max_tracks}); "
                f"truncating to most-recent {max_tracks}."
            )
            def _last_seen(tid: int) -> int:
                info = self.trajectory_infos.get(tid)
                if info is None:
                    return -1
                return int(info.get("last_seen", -1))

            active_ids_list = sorted(active_ids_list, key=_last_seen, reverse=True)
            keep_ids = set(active_ids_list[:max_tracks])
            remove_ids = [tid for tid in list(self.active_ids) if tid not in keep_ids]
            for tid in remove_ids:
                if tid in self.active_ids:
                    self.active_ids.remove(tid)
                self.trajectory_infos.pop(tid, None)
                self._release_vocab_id(tid)
            active_ids_list = active_ids_list[:max_tracks]
            num_tracks = len(active_ids_list)
        if max_tracks > 0 and num_tracks > max_tracks:
            active_ids_list = active_ids_list[:max_tracks]
            num_tracks = len(active_ids_list)

        if num_tracks == 0:
            return self._assign_new_ids(num_dets, scores)

        # 构建轨迹特征序列（滑动窗口）
        T = self.track_window
        trajectory_features = torch.zeros((num_tracks, T, self.feature_dim), device=device, dtype=self.dtype)
        trajectory_assoc_features = torch.zeros((num_tracks, T, self.assoc_feature_dim), device=device, dtype=self.dtype)
        trajectory_boxes = torch.zeros((num_tracks, T, 4), device=device, dtype=self.dtype)
        trajectory_times = torch.zeros((num_tracks, T), device=device, dtype=torch.long)
        trajectory_masks = torch.ones((num_tracks, T), device=device, dtype=torch.bool)

        for idx, track_id in enumerate(active_ids_list):
            info = self.trajectory_infos.get(track_id)
            if info is None:
                continue
            feats = info.get("features", [])
            boxes = info.get("boxes", [])
            times = info.get("times", [])
            if len(feats) == 0:
                continue
            feats = feats[-T:]
            assoc_hist = info.get("assoc_features", [])
            if not assoc_hist:
                assoc_hist = feats
            assoc_hist = assoc_hist[-T:]
            boxes = boxes[-T:]
            times = times[-T:]
            L = len(feats)
            start = T - L
            trajectory_features[idx, start:] = torch.stack(feats, dim=0)
            if len(assoc_hist) == L:
                trajectory_assoc_features[idx, start:] = torch.stack(assoc_hist, dim=0).to(device=device, dtype=self.dtype)
            else:
                # Keep runtime feature space aligned with the scores being refined. Fallback to model features if the
                # assoc-feature history is unavailable or length-mismatched.
                trajectory_assoc_features[idx, start:] = torch.stack(feats, dim=0).to(device=device, dtype=self.dtype)
            trajectory_boxes[idx, start:] = torch.stack(boxes, dim=0)
            trajectory_times[idx, start:] = torch.tensor(times, device=device, dtype=torch.long)
            trajectory_masks[idx, start:] = False

        # 轨迹最近一次的框用于 IoU gating / feature matching
        track_last_boxes = torch.zeros((num_tracks, 4), device=device, dtype=self.dtype)
        for i, track_id in enumerate(active_ids_list):
            info = self.trajectory_infos.get(track_id, {})
            pred_box = info.get("pred_box")
            if pred_box is not None:
                track_last_boxes[i] = pred_box.to(device=device, dtype=self.dtype)
                continue
            valid = ~trajectory_masks[i]
            if valid.any():
                last_idx = valid.nonzero(as_tuple=True)[0][-1].item()
                track_last_boxes[i] = trajectory_boxes[i, last_idx]

        # 纯特征相似度关联（不走 vocab logits）
        if self.assoc_mode == "feature":
            track_feat_matrix = self._build_track_feature_matrix(active_ids_list)
            feat_scores = self._compute_feature_scores(assoc_features, track_feat_matrix)
            motion_scores = self._compute_bbox_distance_scores(boxes_cxcywh, track_last_boxes)
            pre_refine_scores = feat_scores.clone()
            refine_det_features = assoc_features if assoc_features is not None else model_features
            refine_track_features = trajectory_assoc_features
            feat_scores = self._refine_assoc_scores_with_laplace(
                base_scores=feat_scores,
                det_features=refine_det_features,
                track_history_features=refine_track_features,
                track_history_masks=trajectory_masks,
                det_scores=scores,
                motion_scores=motion_scores,
            )
            feat_scores = self._refine_assoc_scores_with_mtcr(
                base_scores=feat_scores,
                det_features=refine_det_features,
                track_history_features=refine_track_features,
                track_history_masks=trajectory_masks,
                track_history_times=trajectory_times,
                det_scores=scores,
                motion_scores=motion_scores,
            )
            feat_scores = self._refine_assoc_scores_with_runtime_replay(
                base_scores=feat_scores,
                det_features=refine_det_features,
                track_history_features=refine_track_features,
                track_history_masks=trajectory_masks,
                track_history_times=trajectory_times,
                det_scores=scores,
                motion_scores=motion_scores,
                det_boxes_cxcywh=boxes_cxcywh,
                track_boxes_cxcywh=track_last_boxes,
            )
            feat_scores = self._refine_assoc_scores_with_competition(
                base_scores=feat_scores,
                det_scores=scores,
                track_history_masks=trajectory_masks,
                track_history_times=trajectory_times,
                motion_scores=motion_scores,
            )
            id_labels = self._match_with_scores(
                feat_scores,
                scores,
                active_ids_list,
                track_history_masks=trajectory_masks,
                track_history_times=trajectory_times,
                motion_scores=motion_scores,
                det_boxes_cxcywh=boxes_cxcywh,
                track_boxes_cxcywh=track_last_boxes,
            )
            self._maybe_dump_feature_assoc_candidates(
                base_scores=pre_refine_scores,
                refined_scores=feat_scores,
                motion_scores=motion_scores,
                scores=scores,
                active_ids_list=active_ids_list,
                det_features=refine_det_features,
                track_history_features=refine_track_features,
                track_history_masks=trajectory_masks,
                track_history_times=trajectory_times,
                det_boxes_cxcywh=boxes_cxcywh,
                track_boxes_cxcywh=track_last_boxes,
                id_labels=id_labels,
            )
            return id_labels, scores

        # 构建 seq_info
        # 格式: (B=1, G=1, T, N, C)
        # 轨迹局部索引与真实ID映射（推理时只用局部索引做关联）
        # NOTE: decoder logits include a newborn class (vocab_size = num_id_vocabulary + 1), but track vocab ids
        # should be within [0, num_id_vocabulary) to avoid mapping a track to the newborn index.
        track_vocab_cap = int(getattr(self.id_decoder, "num_id_vocabulary", self.num_id_vocabulary))
        track_vocab_cap = max(track_vocab_cap, 1)
        track_vocab_indices = [self._get_or_assign_vocab_id(tid) % track_vocab_cap for tid in active_ids_list]
        trajectory_id_labels = torch.tensor(track_vocab_indices, device=device, dtype=torch.long).view(1, 1, 1, num_tracks)
        trajectory_id_labels = trajectory_id_labels.repeat(1, 1, T, 1)
        trajectory_id_map = torch.tensor(active_ids_list, device=device, dtype=torch.long).view(1, 1, 1, num_tracks)
        trajectory_id_map = trajectory_id_map.repeat(1, 1, T, 1)

        seq_info = {
            "trajectory_features": trajectory_features.permute(1, 0, 2).unsqueeze(0).unsqueeze(0),  # (1,1,T,N,C)
            "trajectory_boxes": trajectory_boxes.permute(1, 0, 2).unsqueeze(0).unsqueeze(0),
            "trajectory_masks": trajectory_masks.permute(1, 0).unsqueeze(0).unsqueeze(0),
            "trajectory_id_labels": trajectory_id_labels,
            "trajectory_id_map": trajectory_id_map,
            "trajectory_times": trajectory_times.permute(1, 0).unsqueeze(0).unsqueeze(0),
            "unknown_features": model_features.unsqueeze(0).unsqueeze(0).unsqueeze(0),  # (1, 1, 1, num_dets, C)
            "unknown_boxes": boxes_cxcywh.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            "unknown_masks": torch.zeros((1, 1, 1, num_dets), dtype=torch.bool, device=device),
            "unknown_id_labels": -torch.ones((1, 1, 1, num_dets), dtype=torch.long, device=device),
            "unknown_times": torch.full((1, 1, 1, num_dets), self.frame_id, dtype=torch.long, device=device),
        }

        # 前向传播
        seq_info = self.trajectory_modeling(seq_info)
        id_decoder_output = self.id_decoder(seq_info, use_decoder_checkpoint=False)

        extra_info = None
        if isinstance(id_decoder_output, tuple) and len(id_decoder_output) >= 3:
            id_logits = id_decoder_output[0]  # (layers, B, G, T, N, vocab) or (B,G,T,N,vocab)
            if len(id_decoder_output) >= 4:
                extra_info = id_decoder_output[3]
        else:
            id_logits = id_decoder_output

        # Frequency-guided gating signal (per detection)
        freq_gate = None
        if extra_info is not None:
            freq_conf = None
            try:
                fusion_info = extra_info.get("fusion_info", None)
                if isinstance(fusion_info, dict):
                    freq_conf = fusion_info.get("freq_confidence", None)
                if freq_conf is None:
                    freq_info = extra_info.get("freq_branch_info", None)
                    if isinstance(freq_info, dict):
                        freq_conf = freq_info.get("freq_confidence", None)
                if torch.is_tensor(freq_conf):
                    fc = freq_conf
                    while fc.dim() > 1:
                        fc = fc.squeeze(0)
                    if fc.numel() == num_dets:
                        freq_gate = fc.detach()
            except Exception:
                freq_gate = None

        # 使用最后一层的 logits
        if id_logits.dim() == 6:
            id_logits = id_logits[-1]  # (B, G, T, N, vocab)

        id_logits = id_logits.squeeze(0).squeeze(0).squeeze(0)  # (num_dets, vocab)

        # Optional: confidence calibration using band consistency
        cal_factor = None
        calibration_applied = False
        if extra_info is not None:
            cal_factor = extra_info.get("calibration_factor", None)
            calibration_applied = bool(extra_info.get("calibration_applied", False))

        if cal_factor is None and self.calibrator is not None and extra_info is not None:
            try:
                freq_info = extra_info.get("freq_branch_info", None)
                band_logits = None
                if isinstance(freq_info, dict):
                    band_logits = freq_info.get("band_logits", None)
                if isinstance(band_logits, list) and len(band_logits) > 0:
                    band_logits_squeezed = []
                    for bl in band_logits:
                        # (B,G,T,N,V) -> (B,T,N,V)
                        if bl.dim() == 5:
                            bl = bl.squeeze(1)  # remove G
                        band_logits_squeezed.append(bl)
                    fused_logits = id_logits.unsqueeze(0).unsqueeze(0)  # (1,1,N,V)
                    raw_scores = scores.view(1, 1, -1)
                    calibrated = self.calibrator(
                        raw_scores=raw_scores,
                        band_logits=band_logits_squeezed,
                        fused_logits=fused_logits,
                    )
                    scores = calibrated.view(-1)
                    cal_factor = self.calibrator.compute_calibration_factor(
                        band_logits=band_logits_squeezed,
                        fused_logits=fused_logits,
                    )
            except Exception:
                cal_factor = None

        if calibration_applied:
            cal_factor = None

        # 预测 ID（logit / hybrid）
        feat_scores = None
        if self.assoc_mode == "hybrid":
            track_feat_matrix = self._build_track_feature_matrix(active_ids_list)
            feat_scores = self._compute_feature_scores(assoc_features, track_feat_matrix)

        refine_det_features = model_features
        refine_track_features = trajectory_features
        if self.assoc_feat_source == "reid" and assoc_features is not None:
            refine_det_features = assoc_features
            refine_track_features = trajectory_assoc_features

        id_labels = self._decode_id_labels(
            id_logits=id_logits,
            scores=scores,
            active_ids_list=active_ids_list,
            det_boxes_cxcywh=boxes_cxcywh,
            track_boxes_cxcywh=track_last_boxes,
            cal_factor=cal_factor,
            feat_scores=feat_scores,
            freq_gate=freq_gate,
            det_features=refine_det_features,
            track_history_features=refine_track_features,
            track_history_masks=trajectory_masks,
            track_history_times=trajectory_times,
        )

        return id_labels, scores

    def _decode_id_labels(
        self,
        id_logits: torch.Tensor,
        scores: torch.Tensor,
        active_ids_list: List[int],
        det_boxes_cxcywh: Optional[torch.Tensor] = None,
        track_boxes_cxcywh: Optional[torch.Tensor] = None,
        cal_factor: Optional[torch.Tensor] = None,
        feat_scores: Optional[torch.Tensor] = None,
        freq_gate: Optional[torch.Tensor] = None,
        det_features: Optional[torch.Tensor] = None,
        track_history_features: Optional[torch.Tensor] = None,
        track_history_masks: Optional[torch.Tensor] = None,
        track_history_times: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        从 ID logits 解码 ID 标签

        参数：
            id_logits: (N, vocab) - ID 预测 logits
            scores: (N,) - 检测置信度
            active_ids_list: 当前活跃的 ID 列表

        返回：
            id_labels: (N,) - 预测的 ID
        """
        device = distributed_device()
        num_dets = id_logits.shape[0]
        num_tracks = len(active_ids_list)

        id_labels = torch.full((num_dets,), -1, dtype=torch.long, device=device)

        # 使用 softmax 获取概率（可选温度）
        logit_temp = max(self.assoc_logit_temp, 1e-6)
        id_probs = torch.softmax(id_logits / logit_temp, dim=-1)
        # Apply calibration factor to association probabilities (if available)
        if cal_factor is not None:
            try:
                factor = cal_factor.view(-1, 1).to(id_probs.device, dtype=id_probs.dtype)
                uniform = torch.full_like(id_probs, 1.0 / id_probs.shape[-1])
                id_probs = factor * id_probs + (1.0 - factor) * uniform
            except Exception:
                pass

        # Use vocab indices assigned per track (unique within active set)
        vocab_size = int(id_probs.shape[-1])
        track_vocab_cap = int(getattr(self.id_decoder, "num_id_vocabulary", self.num_id_vocabulary))
        track_vocab_cap = max(min(track_vocab_cap, vocab_size), 1)
        track_vocab_indices = [self._get_or_assign_vocab_id(tid) % track_vocab_cap for tid in active_ids_list]

        # Extract per-track probabilities
        if num_tracks > 0:
            probs = id_probs[:, track_vocab_indices]  # (num_dets, num_tracks)
        else:
            probs = id_probs[:, :0]

        if self.assoc_mode == "hybrid" and feat_scores is not None:
            if self.assoc_freq_gate and freq_gate is not None:
                try:
                    gate = freq_gate.to(device=feat_scores.device, dtype=feat_scores.dtype).view(-1, 1)
                    gate = gate.clamp(0.0, 1.0)
                    gate = self.assoc_freq_gate_min + (self.assoc_freq_gate_max - self.assoc_freq_gate_min) * gate
                    feat_scores = feat_scores * gate
                except Exception:
                    pass
            base_scores = self.assoc_id_weight * probs + self.assoc_feat_weight * feat_scores
        else:
            base_scores = probs

        pre_refine_scores = base_scores.clone()
        motion_scores = None
        if (
            det_features is not None
            and track_history_features is not None
            and track_history_masks is not None
            and det_boxes_cxcywh is not None
            and track_boxes_cxcywh is not None
        ):
            motion_scores = self._compute_bbox_distance_scores(det_boxes_cxcywh, track_boxes_cxcywh)
            base_scores = self._refine_assoc_scores_with_laplace(
                base_scores=base_scores,
                det_features=det_features,
                track_history_features=track_history_features,
                track_history_masks=track_history_masks,
                det_scores=scores,
                motion_scores=motion_scores,
            )
            base_scores = self._refine_assoc_scores_with_mtcr(
                base_scores=base_scores,
                det_features=det_features,
                track_history_features=track_history_features,
                track_history_masks=track_history_masks,
                track_history_times=track_history_times,
                det_scores=scores,
                motion_scores=motion_scores,
            )
            base_scores = self._refine_assoc_scores_with_runtime_replay(
                base_scores=base_scores,
                det_features=det_features,
                track_history_features=track_history_features,
                track_history_masks=track_history_masks,
                track_history_times=track_history_times,
                det_scores=scores,
                motion_scores=motion_scores,
                det_boxes_cxcywh=det_boxes_cxcywh,
                track_boxes_cxcywh=track_boxes_cxcywh,
            )
            base_scores = self._refine_assoc_scores_with_competition(
                base_scores=base_scores,
                det_scores=scores,
                track_history_masks=track_history_masks,
                track_history_times=track_history_times,
                motion_scores=motion_scores,
            )

        id_labels = self._match_with_scores(
            base_scores,
            scores,
            active_ids_list,
            track_history_masks=track_history_masks,
            track_history_times=track_history_times,
            motion_scores=motion_scores,
            det_boxes_cxcywh=det_boxes_cxcywh,
            track_boxes_cxcywh=track_boxes_cxcywh,
            cal_factor=cal_factor,
        )

        # Runtime candidate dumps are needed for honest conflict-set analysis on the actual host.
        # The old implementation only dumped in ASSOC_MODE=feature, which made hybrid hosts invisible
        # to the replay/competition pipeline even though their matching path was valid.
        if (
            self.assoc_runtime_dump_path
            and det_features is not None
            and track_history_features is not None
            and track_history_masks is not None
            and det_boxes_cxcywh is not None
            and track_boxes_cxcywh is not None
        ):
            self._maybe_dump_feature_assoc_candidates(
                base_scores=pre_refine_scores,
                refined_scores=base_scores,
                motion_scores=motion_scores,
                scores=scores,
                active_ids_list=active_ids_list,
                det_features=det_features,
                track_history_features=track_history_features,
                track_history_masks=track_history_masks,
                track_history_times=track_history_times,
                det_boxes_cxcywh=det_boxes_cxcywh,
                track_boxes_cxcywh=track_boxes_cxcywh,
                id_labels=id_labels,
            )

        return id_labels

    def _update_trajectory_infos(
        self,
        boxes_cxcywh: torch.Tensor,
        model_features: torch.Tensor,
        assoc_features: torch.Tensor,
        id_labels: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ):
        """更新轨迹信息"""
        device = distributed_device()

        # 更新匹配到的轨迹
        matched_ids = set()
        for i, id_label in enumerate(id_labels):
            track_id = id_label.item()
            if track_id > 0:
                info = self.trajectory_infos.get(track_id)
                if info is None:
                    info = {
                        "features": [],
                        "assoc_features": [],
                        "boxes": [],
                        "times": [],
                        "mean": None,
                        "cov": None,
                        "pred_box": None,
                        "last_seen": self.frame_id,
                        "miss_count": 0,
                    }
                    self.trajectory_infos[track_id] = info
                feat_to_store = model_features[i].detach()
                if self.memory_bank is not None and scores is not None:
                    long_mem = info.get("long_memory", feat_to_store)
                    last_feat = info.get("last_feature", feat_to_store)
                    sc = scores[i].view(1)
                    updated_long, query_feat = self.memory_bank.update(
                        current_features=feat_to_store.view(1, -1),
                        long_memory=long_mem.view(1, -1),
                        last_features=last_feat.view(1, -1),
                        scores=sc,
                    )
                    info["long_memory"] = updated_long.view(-1).detach()
                    info["last_feature"] = feat_to_store
                    feat_to_store = query_feat.view(-1).detach()
                info["features"].append(feat_to_store)
                if assoc_features is not None and torch.is_tensor(assoc_features):
                    info.setdefault("assoc_features", []).append(assoc_features[i].detach())
                info["boxes"].append(boxes_cxcywh[i].detach())
                info["times"].append(self.frame_id)
                if len(info["features"]) > self.track_window:
                    info["features"] = info["features"][-self.track_window:]
                    if "assoc_features" in info and isinstance(info.get("assoc_features"), list):
                        info["assoc_features"] = info["assoc_features"][-self.track_window:]
                    info["boxes"] = info["boxes"][-self.track_window:]
                    info["times"] = info["times"][-self.track_window:]
                info["last_seen"] = self.frame_id
                info["miss_count"] = 0
                if self.use_kalman:
                    meas = _cxcywh_to_xyah(boxes_cxcywh[i].unsqueeze(0)).squeeze(0).cpu().numpy()
                    if info["mean"] is None or info["cov"] is None:
                        mean, cov = self.kf.initiate(meas)
                    else:
                        mean, cov = self.kf.update(info["mean"], info["cov"], meas)
                    info["mean"] = mean
                    info["cov"] = cov
                    info["pred_box"] = boxes_cxcywh[i].detach()
                matched_ids.add(track_id)

        # 更新未匹配的轨迹的 miss_count
        ids_to_remove = []
        for track_id in list(self.active_ids):
            if track_id not in matched_ids:
                if track_id in self.trajectory_infos:
                    self.trajectory_infos[track_id]["miss_count"] += 1
                    if self.trajectory_infos[track_id]["miss_count"] > self.miss_tolerance:
                        ids_to_remove.append(track_id)

        # 移除超时的轨迹
        for track_id in ids_to_remove:
            self.active_ids.discard(track_id)
            if track_id in self.trajectory_infos:
                del self.trajectory_infos[track_id]
            self._release_vocab_id(track_id)

    def _update_miss_counts(self):
        """当没有检测时，更新所有轨迹的 miss_count"""
        ids_to_remove = []
        for track_id in list(self.active_ids):
            if track_id in self.trajectory_infos:
                self.trajectory_infos[track_id]["miss_count"] += 1
                if self.trajectory_infos[track_id]["miss_count"] > self.miss_tolerance:
                    ids_to_remove.append(track_id)

        for track_id in ids_to_remove:
            self.active_ids.discard(track_id)
            if track_id in self.trajectory_infos:
                del self.trajectory_infos[track_id]
            self._release_vocab_id(track_id)
