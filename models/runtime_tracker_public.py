# Copyright (c) Ruopeng Gao. All Rights Reserved.
# Modified to support public detections for MOT17/MOT20

import torch
import einops
import os

# SciPy 延迟导入：仅在使用 hungarian 协议时才需要
# 这样可以避免在某些缺少 SciPy 的云镜像上直接 import 失败
_linear_sum_assignment = None

def _get_linear_sum_assignment():
    """延迟导入 scipy.optimize.linear_sum_assignment"""
    global _linear_sum_assignment
    if _linear_sum_assignment is None:
        try:
            from scipy.optimize import linear_sum_assignment
            _linear_sum_assignment = linear_sum_assignment
        except ImportError:
            raise ImportError(
                "scipy is required for 'hungarian' assignment protocol. "
                "Please install it with: pip install scipy, "
                "or use 'object-max' / 'id-max' protocol instead."
            )
    return _linear_sum_assignment

from structures.instances import Instances
from structures.ordered_set import OrderedSet
from utils.misc import distributed_device
from utils.box_ops import box_cxcywh_to_xywh, box_xywh_to_cxcywh
from models.misc import get_model


def load_public_detections(det_path: str) -> dict:
    """
    Load public detections from MOT format det.txt file.
    Format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
    
    Returns: dict[frame_id] -> list of (x, y, w, h, conf)
    """
    detections = {}
    if not os.path.exists(det_path):
        raise FileNotFoundError(f"Detection file not found: {det_path}")
    
    with open(det_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            frame_id = int(parts[0])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf = float(parts[6])
            
            if frame_id not in detections:
                detections[frame_id] = []
            detections[frame_id].append((x, y, w, h, conf))
    
    return detections


def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    boxes1: (N, 4) in xywh format
    boxes2: (M, 4) in xywh format
    Returns: (N, M) IoU matrix
    """
    # Convert to xyxy
    boxes1_xyxy = boxes1.clone()
    boxes1_xyxy[:, 2:] = boxes1_xyxy[:, :2] + boxes1_xyxy[:, 2:]
    boxes2_xyxy = boxes2.clone()
    boxes2_xyxy[:, 2:] = boxes2_xyxy[:, :2] + boxes2_xyxy[:, 2:]
    
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    
    lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
    rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-6)
    
    return iou


class RuntimeTrackerPublic:
    """
    Runtime tracker that uses external detections (public or ByteTrack).
    It matches DINO detections with external detections and uses matched
    features for ID association (unless using external ReID embeddings).
    """
    def __init__(
            self,
            model,
            # Sequence infos:
            sequence_hw: tuple,
            # Detections:
            public_detections: dict | None = None,  # frame_id -> list of (x, y, w, h, conf)
            detector=None,                          # ByteTrack detector
            det_source: str = "public",
            # Inference settings:
            use_sigmoid: bool = False,
            assignment_protocol: str = "object-priority",  # object-priority 正确返回检测数量的标签
            miss_tolerance: int = 30,
            det_thresh: float = 0.5,  # For public detection confidence
            newborn_thresh: float = 0.5,
            id_thresh: float = 0.1,
            area_thresh: int = 0,
            iou_thresh: float = 0.5,  # IoU threshold for matching DINO with public
            # P1-b: public det + external ReID embeddings (crop -> encoder)
            public_reid_encoder=None,
            use_public_reid: bool = False,
            only_detr: bool = False,
            dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.model.eval()

        self.dtype = dtype
        self.public_detections = public_detections
        self.detector = detector
        self.det_source = str(det_source).lower()
        self.last_dets = []
        if self.det_source not in ["public", "bytetrack"]:
            raise ValueError(f"Unsupported det_source: {self.det_source}")
        if self.det_source == "public" and self.public_detections is None:
            raise RuntimeError("det_source=public but public_detections is None")
        if self.det_source == "bytetrack" and self.detector is None:
            raise RuntimeError("det_source=bytetrack but detector is None")
        self.iou_thresh = iou_thresh
        self.public_reid_encoder = public_reid_encoder
        self.use_public_reid = bool(use_public_reid)
        self.frame_id = 0

        # For FP16:
        if self.dtype != torch.float32:
            if self.dtype == torch.float16:
                self.model.half()
            else:
                raise NotImplementedError(f"Unsupported dtype {self.dtype}.")

        self.use_sigmoid = use_sigmoid
        self.assignment_protocol = assignment_protocol.lower()
        self.miss_tolerance = miss_tolerance
        self.det_thresh = det_thresh
        self.newborn_thresh = newborn_thresh
        self.id_thresh = id_thresh
        self.area_thresh = area_thresh
        self.only_detr = only_detr
        self.num_id_vocabulary = get_model(model).num_id_vocabulary

        # Check for the legality of settings:
        assert self.assignment_protocol in ["hungarian", "id-max", "object-max", "object-priority", "id-priority"], \
            f"Assignment protocol {self.assignment_protocol} is not supported."

        self.sequence_hw = sequence_hw
        self.bbox_unnorm = torch.tensor(
            [sequence_hw[1], sequence_hw[0], sequence_hw[1], sequence_hw[0]],
            dtype=dtype,
            device=distributed_device(),
        )

        # Trajectory fields:
        self.next_id = 0
        self.id_label_to_id = {}
        self.id_queue = OrderedSet()
        # Init id_queue:
        for i in range(self.num_id_vocabulary):
            self.id_queue.add(i)
        # All fields are in shape (T, N, ...)
        # Get feature_dim from model config instead of hardcoding
        feature_dim = get_model(model).feature_dim if hasattr(get_model(model), 'feature_dim') else 256
        self.feature_dim = feature_dim  # Store for later use
        self.trajectory_features = torch.zeros(
            (0, 0, feature_dim), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_boxes = torch.zeros(
            (0, 0, 4), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_id_labels = torch.zeros(
            (0, 0), dtype=torch.int64, device=distributed_device(),
        )
        self.trajectory_times = torch.zeros(
            (0, 0), dtype=torch.int64, device=distributed_device(),
        )
        self.trajectory_masks = torch.zeros(
            (0, 0), dtype=torch.bool, device=distributed_device(),
        )

        self.current_track_results = {}
        return

    @torch.no_grad()
    def update(self, image, image_path: str | None = None):
        """
        Update with detections from selected source.
        优化：先检查 detections，若为空则跳过 DINO forward 以节省计算
        1. Get detections for current frame (先检查，避免无效 DINO 计算)
        2. (Option A) Get DINO detections and features (只在需要时运行)
           (Option B, P1-b) Extract ReID embeddings from raw image crops
        3. (Option A) Match DINO detections with public detections using IoU
        4. Use matched features with public detection boxes
        5. Perform ID association
        """
        self.frame_id += 1

        # Step 1: get detections from selected source
        if self.det_source == "bytetrack":
            if self.detector is None:
                raise RuntimeError("det_source=bytetrack but detector is None")
            if image_path is None:
                raise ValueError("ByteTrack detector requires image_path")
            public_dets = self.detector.detect(image_path)
        else:
            if self.public_detections is None:
                raise RuntimeError("det_source=public but public_detections is None")
            public_dets = self.public_detections.get(self.frame_id, [])

        self.last_dets = list(public_dets)

        # Handle empty public detections (不运行 DINO，节省计算)
        if len(public_dets) == 0:
            self.current_track_results = {
                "score": torch.tensor([], dtype=self.dtype, device=distributed_device()),
                "category": torch.tensor([], dtype=torch.int64, device=distributed_device()),
                "bbox": torch.zeros((0, 4), dtype=self.dtype, device=distributed_device()),
                "id": torch.tensor([], dtype=torch.int64, device=distributed_device()),
            }
            # 推进轨迹时间轴，即使没有检测也要更新时间步
            empty_boxes = torch.zeros((0, 4), dtype=self.dtype, device=distributed_device())
            empty_embeds = torch.zeros((0, self.feature_dim), dtype=self.dtype, device=distributed_device())
            empty_ids = torch.zeros((0,), dtype=torch.int64, device=distributed_device())
            self._update_trajectory_infos(boxes=empty_boxes, output_embeds=empty_embeds, id_labels=empty_ids)
            self._filter_out_inactive_tracks()
            return

        # Filter by confidence and area
        filtered_public_dets = []
        for det in public_dets:
            x, y, w, h, conf = det
            area = w * h
            if conf >= self.det_thresh and area >= self.area_thresh:
                filtered_public_dets.append(det)

        if len(filtered_public_dets) == 0:
            self.current_track_results = {
                "score": torch.tensor([], dtype=self.dtype, device=distributed_device()),
                "category": torch.tensor([], dtype=torch.int64, device=distributed_device()),
                "bbox": torch.zeros((0, 4), dtype=self.dtype, device=distributed_device()),
                "id": torch.tensor([], dtype=torch.int64, device=distributed_device()),
            }
            # 推进轨迹时间轴，即使过滤后为空也要更新时间步
            empty_boxes = torch.zeros((0, 4), dtype=self.dtype, device=distributed_device())
            empty_embeds = torch.zeros((0, self.feature_dim), dtype=self.dtype, device=distributed_device())
            empty_ids = torch.zeros((0,), dtype=torch.int64, device=distributed_device())
            self._update_trajectory_infos(boxes=empty_boxes, output_embeds=empty_embeds, id_labels=empty_ids)
            self._filter_out_inactive_tracks()
            return

        # Convert public dets to tensors
        public_boxes_xywh = torch.tensor(
            [[d[0], d[1], d[2], d[3]] for d in filtered_public_dets],
            dtype=self.dtype, device=distributed_device()
        )
        public_scores = torch.tensor(
            [d[4] for d in filtered_public_dets],
            dtype=self.dtype, device=distributed_device()
        )

        # Step 2/3: get per-box embeddings
        if self.use_public_reid:
            if self.public_reid_encoder is None:
                raise RuntimeError("use_public_reid=True but public_reid_encoder is None")
            if image_path is None:
                raise ValueError("P1-b requires image_path (raw frame) for cropping")

            boxes = public_boxes_xywh
            boxes_norm = box_xywh_to_cxcywh(boxes / self.bbox_unnorm)
            output_embeds = self.public_reid_encoder.encode(image_path=image_path, boxes_xywh=boxes)
            scores = public_scores
            categories = torch.zeros((boxes.shape[0],), dtype=torch.int64, device=distributed_device())
        else:
            # Step 2: Now run DINO forward (只在有有效 public dets 时才运行，节省计算)
            detr_out = self.model(frames=image, part="detr")
            _, dino_categories, dino_boxes, dino_embeds = self._get_dino_detections(detr_out)

            # Step 3: Match DINO detections with public detections
            # IMPORTANT: keep *all* public detections to evaluate association fairly.
            # We assign a DINO embedding to every public box using best-IoU match (argmax)
            # 注意：移除了Hungarian匹配以避免O(n³)复杂度导致的超时问题
            if dino_boxes.shape[0] > 0:
                # Convert DINO boxes from normalized cxcywh to xywh (pixel coordinates)
                dino_boxes_xywh = box_cxcywh_to_xywh(dino_boxes) * self.bbox_unnorm

                # Compute IoU
                iou_matrix = box_iou(public_boxes_xywh, dino_boxes_xywh)  # (P, D)

                # Default assignment: best IoU per public box (fast, O(P*D))
                max_iou_values, assign_dino_idx = torch.max(iou_matrix, dim=1)  # (P,), (P,)

                # ✅ Keep ALL public detections, don't drop any
                boxes = public_boxes_xywh
                boxes_norm = box_xywh_to_cxcywh(boxes / self.bbox_unnorm)
                output_embeds = dino_embeds[assign_dino_idx]
                scores = public_scores
                categories = dino_categories[assign_dino_idx]

                # B3优化: 当 max IoU < iou_thresh 时，使用零 embedding（更稳定）
                # 这样可以让 iou_thresh 参数真正发挥作用，避免使用不匹配的 DINO 特征
                low_iou_mask = max_iou_values < self.iou_thresh
                if low_iou_mask.any():
                    zero_embed = torch.zeros(self.feature_dim, dtype=self.dtype, device=distributed_device())
                    output_embeds[low_iou_mask] = zero_embed
            else:
                # No DINO detections - cannot extract features
                self.current_track_results = {
                    "score": torch.tensor([], dtype=self.dtype, device=distributed_device()),
                    "category": torch.tensor([], dtype=torch.int64, device=distributed_device()),
                    "bbox": torch.zeros((0, 4), dtype=self.dtype, device=distributed_device()),
                    "id": torch.tensor([], dtype=torch.int64, device=distributed_device()),
                }
                # 推进轨迹时间轴，即使DINO检测为空也要更新时间步
                empty_boxes = torch.zeros((0, 4), dtype=self.dtype, device=distributed_device())
                empty_embeds = torch.zeros((0, self.feature_dim), dtype=self.dtype, device=distributed_device())
                empty_ids = torch.zeros((0,), dtype=torch.int64, device=distributed_device())
                self._update_trajectory_infos(boxes=empty_boxes, output_embeds=empty_embeds, id_labels=empty_ids)
                self._filter_out_inactive_tracks()
                return
        
        # Step 4: ID association (same as original)
        if self.only_detr:
            id_pred_labels = self.num_id_vocabulary * torch.ones(boxes_norm.shape[0], dtype=torch.int64, device=boxes_norm.device)
        else:
            id_pred_labels = self._get_id_pred_labels(boxes=boxes_norm, output_embeds=output_embeds)
        
        # Filter out illegal newborn detections:
        keep_idxs = (id_pred_labels != self.num_id_vocabulary) | (scores > self.newborn_thresh)
        scores = scores[keep_idxs]
        categories = categories[keep_idxs]
        boxes = boxes[keep_idxs]  # xywh pixel coordinates
        boxes_norm = boxes_norm[keep_idxs]
        output_embeds = output_embeds[keep_idxs]
        id_pred_labels = id_pred_labels[keep_idxs]

        # Handle empty after filtering
        if boxes.shape[0] == 0:
            self.current_track_results = {
                "score": scores,
                "category": categories,
                "bbox": boxes,
                "id": torch.tensor([], dtype=torch.int64, device=boxes.device),
            }
            # 推进轨迹时间轴，即使newborn过滤后为空也要更新时间步
            empty_boxes = torch.zeros((0, 4), dtype=self.dtype, device=distributed_device())
            empty_embeds = torch.zeros((0, self.feature_dim), dtype=self.dtype, device=distributed_device())
            empty_ids = torch.zeros((0,), dtype=torch.int64, device=distributed_device())
            self._update_trajectory_infos(boxes=empty_boxes, output_embeds=empty_embeds, id_labels=empty_ids)
            self._filter_out_inactive_tracks()
            return

        # A hack implementation, before assign new id labels, update the id_queue
        n_activate_id_labels = 0
        n_newborn_targets = 0
        for _ in range(len(id_pred_labels)):
            if id_pred_labels[_].item() != self.num_id_vocabulary:
                n_activate_id_labels += 1
                self.id_queue.add(id_pred_labels[_].item())
            else:
                n_newborn_targets += 1

        # Make sure the length of newborn instances is less than the length of remaining IDs:
        n_remaining_ids = len(self.id_queue) - n_activate_id_labels
        if n_newborn_targets > n_remaining_ids:
            newborn_idxs = (id_pred_labels == self.num_id_vocabulary)
            newborn_positions = newborn_idxs.nonzero(as_tuple=False).view(-1)
            keep_idxs = torch.ones(len(id_pred_labels), dtype=torch.bool, device=id_pred_labels.device)
            if n_remaining_ids < len(newborn_positions):
                drop_positions = newborn_positions[n_remaining_ids:]
                keep_idxs[drop_positions] = False
            scores = scores[keep_idxs]
            categories = categories[keep_idxs]
            boxes = boxes[keep_idxs]
            boxes_norm = boxes_norm[keep_idxs]
            output_embeds = output_embeds[keep_idxs]
            id_pred_labels = id_pred_labels[keep_idxs]

        # Assign new id labels:
        id_labels = self._assign_newborn_id_labels(pred_id_labels=id_pred_labels)

        if len(torch.unique(id_labels)) != len(id_labels):
            print("[RuntimeTrackerPublic] Duplicate IDs detected; reassigning duplicates.")
            id_labels = id_labels.clone()
            seen = set()
            for i in range(len(id_labels)):
                val = id_labels[i].item()
                if val in seen:
                    id_labels[i] = self.num_id_vocabulary
                else:
                    seen.add(val)
            id_labels = self._assign_newborn_id_labels(pred_id_labels=id_labels)
            if len(torch.unique(id_labels)) != len(id_labels):
                raise RuntimeError(f"Duplicate ID labels remain after reassignment: {id_labels}")

        # Update the results (using public detection boxes in xywh format):
        self.current_track_results = {
            "score": scores,
            "category": categories,
            "bbox": boxes,  # Already in xywh pixel coordinates
            "id": torch.tensor(
                [self.id_label_to_id[_] for _ in id_labels.tolist()], dtype=torch.int64,
            ),
        }

        # Update id_queue:
        for _ in range(len(id_labels)):
            self.id_queue.add(id_labels[_].item())

        # Update trajectory infos:
        self._update_trajectory_infos(boxes=boxes_norm, output_embeds=output_embeds, id_labels=id_labels)

        # Filter out inactive tracks:
        self._filter_out_inactive_tracks()
        return

    def get_track_results(self):
        return self.current_track_results

    def _get_dino_detections(self, detr_out: dict):
        """Get all DINO detections (low threshold to maximize matching)"""
        logits = detr_out["pred_logits"][0]
        boxes = detr_out["pred_boxes"][0]
        output_embeds = detr_out["outputs"][0]
        scores = logits.sigmoid()
        scores, categories = torch.max(scores, dim=-1)
        
        # Use lower threshold to get more candidates for matching
        # 设为0.0以避免dino_boxes为空导致整帧输出空结果
        low_thresh = 0.0
        activate_indices = scores > low_thresh
        
        boxes = boxes[activate_indices]
        output_embeds = output_embeds[activate_indices]
        scores = scores[activate_indices]
        categories = categories[activate_indices]
        return scores, categories, boxes, output_embeds

    def _get_id_pred_labels(self, boxes: torch.Tensor, output_embeds: torch.Tensor):
        # Handle empty detections
        if boxes.shape[0] == 0:
            return torch.tensor([], dtype=torch.int64, device=boxes.device)
        if self.trajectory_features.shape[0] == 0 or self.trajectory_features.shape[1] == 0:
            return self.num_id_vocabulary * torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device)
        else:
            # 1. prepare current infos:
            current_features = output_embeds[None, ...]     # (T, N, ...)
            current_boxes = boxes[None, ...]                # (T, N, 4)
            current_masks = torch.zeros((1, output_embeds.shape[0]), dtype=torch.bool, device=distributed_device())
            current_times = self.trajectory_times.shape[0] * torch.ones(
                (1, output_embeds.shape[0]), dtype=torch.int64, device=distributed_device(),
            )
            # 2. prepare seq_info:
            seq_info = {
                "trajectory_features": self.trajectory_features[None, None, ...],
                "trajectory_boxes": self.trajectory_boxes[None, None, ...],
                "trajectory_id_labels": self.trajectory_id_labels[None, None, ...],
                "trajectory_times": self.trajectory_times[None, None, ...],
                "trajectory_masks": self.trajectory_masks[None, None, ...],
                "unknown_features": current_features[None, None, ...],
                "unknown_boxes": current_boxes[None, None, ...],
                "unknown_masks": current_masks[None, None, ...],
                "unknown_times": current_times[None, None, ...],
            }
            # 3. forward:
            seq_info = self.model(seq_info=seq_info, part="trajectory_modeling")
            id_decoder_output = self.model(seq_info=seq_info, part="id_decoder")
            if isinstance(id_decoder_output, tuple):
                id_logits = id_decoder_output[0]
            else:
                id_logits = id_decoder_output
            # 4. get scores:
            id_logits = id_logits[0, 0, 0]
            if not self.use_sigmoid:
                id_scores = id_logits.softmax(dim=-1)
            else:
                id_scores = id_logits.sigmoid()
            # 5. assign id labels:
            match self.assignment_protocol:
                case "hungarian":
                    id_pred_labels = self._hungarian_assignment(id_scores=id_scores)
                case "id-max":
                    id_pred_labels = self._id_max_assignment(id_scores=id_scores)
                case "object-max":
                    id_pred_labels = self._object_max_assignment(id_scores=id_scores)
                case "object-priority":
                    id_pred_labels = self._object_priority_assignment(id_scores=id_scores)
                case "id-priority":
                    id_pred_labels = self._id_priority_assignment(id_scores=id_scores)
                case _:
                    raise ValueError(f"Unknown assignment protocol: {self.assignment_protocol}")
            return id_pred_labels

    def _assign_newborn_id_labels(self, pred_id_labels: torch.Tensor):
        # 1. how many newborn instances?
        n_newborns = (pred_id_labels == self.num_id_vocabulary).sum().item()
        if n_newborns == 0:
            return pred_id_labels
        else:
            # 2. get available id labels from id_queue:
            newborn_id_labels = torch.tensor(
                list(self.id_queue)[:n_newborns], dtype=torch.int64, device=distributed_device(),
            )
            # 3. make sure these id labels are not in trajectory infos:
            trajectory_remove_idxs = torch.zeros(
                self.trajectory_id_labels.shape[1], dtype=torch.bool, device=distributed_device(),
            )
            for _ in range(len(newborn_id_labels)):
                if self.trajectory_id_labels.shape[0] > 0:
                    trajectory_remove_idxs |= (self.trajectory_id_labels[0] == newborn_id_labels[_])
                if newborn_id_labels[_].item() in self.id_label_to_id:
                    self.id_label_to_id.pop(newborn_id_labels[_].item())
            # remove from trajectory infos:
            self.trajectory_features = self.trajectory_features[:, ~trajectory_remove_idxs]
            self.trajectory_boxes = self.trajectory_boxes[:, ~trajectory_remove_idxs]
            self.trajectory_id_labels = self.trajectory_id_labels[:, ~trajectory_remove_idxs]
            self.trajectory_times = self.trajectory_times[:, ~trajectory_remove_idxs]
            self.trajectory_masks = self.trajectory_masks[:, ~trajectory_remove_idxs]
            # 4. assign id labels to newborn instances:
            pred_id_labels[pred_id_labels == self.num_id_vocabulary] = newborn_id_labels
            # 5. update id infos:
            for _ in range(len(newborn_id_labels)):
                self.id_label_to_id[newborn_id_labels[_].item()] = self.next_id
                self.next_id += 1

            return pred_id_labels

    def _update_trajectory_infos(self, boxes: torch.Tensor, output_embeds: torch.Tensor, id_labels: torch.Tensor):
        # 1. cut trajectory infos:
        self.trajectory_features = self.trajectory_features[-self.miss_tolerance + 2:, ...]
        self.trajectory_boxes = self.trajectory_boxes[-self.miss_tolerance + 2:, ...]
        self.trajectory_id_labels = self.trajectory_id_labels[-self.miss_tolerance + 2:, ...]
        self.trajectory_times = self.trajectory_times[-self.miss_tolerance + 2:, ...]
        self.trajectory_masks = self.trajectory_masks[-self.miss_tolerance + 2:, ...]
        # 2. add the new trajectory info:
        all_id_labels = set(id_labels.tolist())
        for _ in range(self.trajectory_id_labels.shape[1]):
            if self.trajectory_id_labels[0, _].item() not in all_id_labels:
                all_id_labels.add(self.trajectory_id_labels[0, _].item())
        all_id_labels = list(all_id_labels)
        n_ids = len(all_id_labels)
        T = self.trajectory_features.shape[0] + 1
        new_trajectory_features = torch.zeros(
            (T, n_ids, self.feature_dim), dtype=self.dtype, device=distributed_device(),
        )
        new_trajectory_boxes = torch.zeros(
            (T, n_ids, 4), dtype=self.dtype, device=distributed_device(),
        )
        new_trajectory_id_labels = torch.zeros(
            (T, n_ids), dtype=torch.int64, device=distributed_device(),
        )
        new_trajectory_times = torch.zeros(
            (T, n_ids), dtype=torch.int64, device=distributed_device(),
        )
        new_trajectory_masks = torch.ones(
            (T, n_ids), dtype=torch.bool, device=distributed_device(),
        )
        for i, _id in enumerate(all_id_labels):
            new_trajectory_id_labels[:, i] = _id
            if _id in id_labels.tolist():
                idx = id_labels.tolist().index(_id)
                new_trajectory_features[-1, i] = output_embeds[idx]
                new_trajectory_boxes[-1, i] = boxes[idx]
                new_trajectory_times[-1, i] = T - 1
                new_trajectory_masks[-1, i] = False
            if self.trajectory_id_labels.shape[1] > 0 and _id in self.trajectory_id_labels[0].tolist():
                old_idx = self.trajectory_id_labels[0].tolist().index(_id)
                new_trajectory_features[:-1, i] = self.trajectory_features[:, old_idx]
                new_trajectory_boxes[:-1, i] = self.trajectory_boxes[:, old_idx]
                new_trajectory_times[:-1, i] = self.trajectory_times[:, old_idx]
                new_trajectory_masks[:-1, i] = self.trajectory_masks[:, old_idx]
        self.trajectory_features = new_trajectory_features.contiguous().clone()
        self.trajectory_boxes = new_trajectory_boxes.contiguous().clone()
        self.trajectory_id_labels = new_trajectory_id_labels.contiguous().clone()
        self.trajectory_times = new_trajectory_times.contiguous().clone()
        self.trajectory_masks = new_trajectory_masks.contiguous().clone()
        return

    def _filter_out_inactive_tracks(self):
        is_active = torch.sum((~self.trajectory_masks).to(torch.int64), dim=0) > 0
        self.trajectory_features = self.trajectory_features[:, is_active]
        self.trajectory_boxes = self.trajectory_boxes[:, is_active]
        self.trajectory_id_labels = self.trajectory_id_labels[:, is_active]
        self.trajectory_times = self.trajectory_times[:, is_active]
        self.trajectory_masks = self.trajectory_masks[:, is_active]
        return

    def _hungarian_assignment(self, id_scores: torch.Tensor):
        # 修复：返回 len(id_scores) 个标签（每个检测一个）
        num_detections = len(id_scores)
        id_labels = [self.num_id_vocabulary] * num_detections  # 每个检测默认为 newborn

        if num_detections > 0 and self.trajectory_id_labels.shape[1] > 0:
            active_id_labels = self.trajectory_id_labels[0, :].tolist()
            cost_matrix = -id_scores[:, active_id_labels].cpu().numpy()
            # 使用延迟导入的 linear_sum_assignment
            linear_sum_assignment = _get_linear_sum_assignment()
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_indices, col_indices):
                if id_scores[r, active_id_labels[c]] > self.id_thresh:
                    id_labels[r] = active_id_labels[c]  # 分配到正确的检测索引

        id_pred_labels = torch.tensor(id_labels, dtype=torch.int64, device=distributed_device())
        return id_pred_labels

    def _id_max_assignment(self, id_scores: torch.Tensor):
        id_labels = []
        active_id_labels = self.trajectory_id_labels[0, :].tolist()
        for i in range(len(id_scores)):
            max_score, max_idx = torch.max(id_scores[i, active_id_labels], dim=0)
            if max_score > self.id_thresh:
                id_labels.append(active_id_labels[max_idx.item()])
            else:
                id_labels.append(self.num_id_vocabulary)
        id_pred_labels = torch.tensor(id_labels, dtype=torch.int64, device=distributed_device())
        return id_pred_labels

    def _object_max_assignment(self, id_scores: torch.Tensor):
        # 修复：返回 len(id_scores) 个标签（每个检测一个），而非 len(active_id_labels) 个
        active_id_labels = self.trajectory_id_labels[0, :].tolist()
        id_labels = [self.num_id_vocabulary] * len(id_scores)  # 每个检测默认为 newborn
        for id_label in active_id_labels:
            max_score, max_idx = torch.max(id_scores[:, id_label], dim=0)
            if max_score > self.id_thresh:
                id_labels[max_idx.item()] = id_label  # 将该 ID 分配给最匹配的检测
        id_pred_labels = torch.tensor(id_labels, dtype=torch.int64, device=distributed_device())
        return id_pred_labels

    def _object_priority_assignment(self, id_scores: torch.Tensor):
        active_id_labels = self.trajectory_id_labels[0, :].tolist()
        id_labels = [self.num_id_vocabulary] * len(id_scores)
        assigned_objects = set()
        for id_label in active_id_labels:
            max_score, max_idx = torch.max(id_scores[:, id_label], dim=0)
            if max_score > self.id_thresh and max_idx.item() not in assigned_objects:
                id_labels[max_idx.item()] = id_label
                assigned_objects.add(max_idx.item())
        id_pred_labels = torch.tensor(id_labels, dtype=torch.int64, device=distributed_device())
        return id_pred_labels

    def _id_priority_assignment(self, id_scores: torch.Tensor):
        active_id_labels = self.trajectory_id_labels[0, :].tolist()
        id_labels = [self.num_id_vocabulary] * len(id_scores)
        assigned_ids = set()
        for i in range(len(id_scores)):
            max_score, max_idx = torch.max(id_scores[i, active_id_labels], dim=0)
            if max_score > self.id_thresh and active_id_labels[max_idx.item()] not in assigned_ids:
                id_labels[i] = active_id_labels[max_idx.item()]
                assigned_ids.add(active_id_labels[max_idx.item()])
        id_pred_labels = torch.tensor(id_labels, dtype=torch.int64, device=distributed_device())
        return id_pred_labels
