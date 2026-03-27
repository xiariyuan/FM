# Copyright (c) 2024. All Rights Reserved.
"""
ByteTrack/YOLOX Feature Extractor for FM-Track

从冻结的 YOLOX 模型中提取检测框和对应的特征向量，
用于训练频域关联模块。

特征提取策略：
1. 使用 YOLOX backbone (PAFPN) 获取多尺度特征图
2. 使用 YOLOX head 的 stems 统一特征通道到 256
3. 根据检测框位置，使用 RoIAlign 提取对应特征
4. 投影到统一的特征维度

这样可以保证：
- 检测框和特征完全对应
- 特征维度与频域模块兼容
- 整个检测器保持冻结
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
import warnings


@dataclass
class ByteTrackFeatureConfig:
    exp_file: str
    ckpt: str
    fp16: bool = True
    test_size: Optional[Tuple[int, int]] = None  # (h, w)
    conf_thre: float = 0.01
    nms_thre: float = 0.7
    class_agnostic_nms: bool = True
    feature_dim: int = 256  # 输出特征维度
    roi_size: int = 7  # RoI pooling 输出大小
    canonical_scale: float = 224.0
    canonical_level: int = 4
    roi_multi_level: bool = False
    roi_level_fusion: str = "scale"  # "avg" or "scale"


class ByteTrackFeatureExtractor(nn.Module):
    """
    从 YOLOX 模型提取检测框和特征的模块。

    整个 YOLOX 模型保持冻结，只提供特征。
    可以单独训练一个特征投影层来适配下游任务。

    输出格式：
        detections: List[(x, y, w, h, conf)] - 像素坐标的检测框
        features: torch.Tensor (N, feature_dim) - 对应的特征向量
    """

    def __init__(self, cfg: ByteTrackFeatureConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self._warned_invalid_boxes = False
        self._warned_nonfinite_roi = False
        self._warned_nonfinite_features = False

        # 延迟导入
        try:
            from yolox.exp import get_exp
        except Exception:
            # Fallback: make local third_party/ByteTrack importable without requiring
            # an editable install (common after server restart / fresh environment).
            #
            # This keeps the project runnable even if users forgot to run:
            #   pip install -e third_party/ByteTrack
            import sys

            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            bytetrack_root = os.path.join(repo_root, "third_party", "ByteTrack")
            if os.path.isdir(bytetrack_root) and bytetrack_root not in sys.path:
                sys.path.insert(0, bytetrack_root)
            try:
                from yolox.exp import get_exp
            except Exception as exc:
                raise ImportError(
                    "Cannot import YOLOX/ByteTrack. Install it with: pip install -e third_party/ByteTrack"
                ) from exc

        exp_file = cfg.exp_file
        ckpt_file = cfg.ckpt

        if not os.path.exists(exp_file):
            raise FileNotFoundError(
                f"BYTETRACK_EXP_FILE not found: {exp_file}. "
                f"Make sure you cloned ByteTrack into 'third_party/ByteTrack'."
            )
        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(
                f"BYTETRACK_CKPT not found: {ckpt_file}."
            )

        self.exp = get_exp(exp_file, None)
        self.yolox_model = self.exp.get_model().to(device).eval()

        ckpt = torch.load(ckpt_file, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.yolox_model.load_state_dict(state_dict, strict=False)

        # 冻结 YOLOX 模型
        for param in self.yolox_model.parameters():
            param.requires_grad = False

        if cfg.fp16:
            self.yolox_model.half()

        self.test_size = cfg.test_size if cfg.test_size is not None else tuple(self.exp.test_size)
        self.num_classes = getattr(self.exp, "num_classes", 1)

        # FPN strides (P3, P4, P5)
        self.fpn_strides = getattr(self.exp, "strides", [8, 16, 32])
        if len(self.fpn_strides) != 3:
            self.fpn_strides = [8, 16, 32]
        self.canonical_scale = float(cfg.canonical_scale)
        self.canonical_level = int(cfg.canonical_level)  # P4
        self.roi_multi_level = bool(cfg.roi_multi_level)
        self.roi_level_fusion = str(cfg.roi_level_fusion).lower()

        # 获取 YOLOX 的特征通道数
        # YOLOX-X: width=1.25, 所以 stems 输出通道为 int(256 * 1.25) = 320
        # YOLOX-L: width=1.0, 所以 stems 输出通道为 256
        width = getattr(self.exp, "width", 1.0)
        self.stem_channels = int(256 * width)

        # 特征投影层：从 stem_channels 投影到 feature_dim
        # 这个投影层可以训练
        self.feature_proj = nn.Sequential(
            nn.Linear(self.stem_channels, cfg.feature_dim),
            nn.LayerNorm(cfg.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.feature_dim, cfg.feature_dim),
        ).to(device)

        self.roi_size = cfg.roi_size
        self.feature_dim = cfg.feature_dim

    @staticmethod
    def _valid_boxes_xyxy(boxes_xyxy: torch.Tensor, *, min_size: float = 1.0) -> torch.Tensor:
        """
        Validate boxes for RoIAlign.

        RoIAlign is not robust to NaN/Inf coords and can behave poorly on degenerate boxes.
        """
        if boxes_xyxy.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool, device=boxes_xyxy.device)
        finite = torch.isfinite(boxes_xyxy).all(dim=-1)
        w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        size_ok = (w > float(min_size)) & (h > float(min_size))
        return finite & size_ok

    def _preprocess_bgr(self, img_bgr: np.ndarray, input_size: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, float]:
        """预处理 BGR 图像（numpy HWC），返回模型输入与缩放比例"""
        from yolox.data.data_augment import preproc
        import inspect

        size = input_size if input_size is not None else self.test_size

        try:
            n_params = len(inspect.signature(preproc).parameters)
        except Exception:
            n_params = 2

        if n_params <= 2:
            img, ratio = preproc(img_bgr, size)
        else:
            rgb_means = getattr(self.exp, "rgb_means", None)
            std = getattr(self.exp, "std", None)
            try:
                img, ratio = preproc(img_bgr, size, rgb_means, std)
            except TypeError:
                img, ratio = preproc(img_bgr, size, rgb_means, std, (2, 0, 1))

        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        img = img.half() if self.cfg.fp16 else img.float()
        return img, ratio

    def _preprocess(self, img_bgr: np.ndarray) -> Tuple[torch.Tensor, float]:
        """预处理图像（BGR numpy）"""
        return self._preprocess_bgr(img_bgr, input_size=self.test_size)

    def _preprocess_tensor(self, img_tensor: torch.Tensor, input_size: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, float]:
        """
        预处理已经增强后的图像张量（RGB, CxHxW 或 HxWxC）。
        会先转换为 BGR numpy，再调用 YOLOX preproc。
        """
        if torch.is_tensor(img_tensor):
            img = img_tensor.detach().cpu()
            if img.dim() == 3 and img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0)  # HWC
            elif img.dim() == 3 and img.shape[2] in (1, 3):
                pass
            else:
                raise ValueError(f"Unsupported image tensor shape: {img_tensor.shape}")
            img_np = img.numpy()
        else:
            raise ValueError("img_tensor must be a torch.Tensor")

        if img_np.dtype != np.uint8:
            max_val = float(img_np.max()) if img_np.size > 0 else 0.0
            if max_val <= 1.5:
                img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img_np = img_np.clip(0, 255).astype(np.uint8)

        # RGB -> BGR for YOLOX preproc
        img_bgr = img_np[:, :, ::-1].copy()
        return self._preprocess_bgr(img_bgr, input_size=input_size)

    @staticmethod
    def _safe_postprocess(outputs, num_classes: int, conf_thre: float, nms_thre: float, class_agnostic: bool):
        """Call YOLOX postprocess with signature compatibility across versions."""
        from yolox.utils import postprocess
        try:
            return postprocess(
                outputs,
                num_classes=num_classes,
                conf_thre=conf_thre,
                nms_thre=nms_thre,
                class_agnostic=class_agnostic,
            )
        except TypeError:
            # Older YOLOX does not support class_agnostic arg
            return postprocess(
                outputs,
                num_classes=num_classes,
                conf_thre=conf_thre,
                nms_thre=nms_thre,
            )

    def _extract_fpn_features(self, img: torch.Tensor) -> List[torch.Tensor]:
        """
        提取 YOLOX backbone (PAFPN) 的多尺度特征。

        返回：
            fpn_features: List[Tensor] - 3个尺度的特征图
                - scale 0: stride 8, shape (B, C0, H/8, W/8)
                - scale 1: stride 16, shape (B, C1, H/16, W/16)
                - scale 2: stride 32, shape (B, C2, H/32, W/32)
        """
        with torch.no_grad():
            fpn_outs = self.yolox_model.backbone(img)
        return fpn_outs

    def _extract_stem_features(self, fpn_outs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        使用 YOLOX head 的 stems 将多尺度特征统一到相同通道数。

        stems 将各尺度特征投影到 256*width 通道。
        """
        stem_features = []
        with torch.no_grad():
            for k, x in enumerate(fpn_outs):
                x = self.yolox_model.head.stems[k](x)
                stem_features.append(x)
        return stem_features

    def _roi_align_features(
        self,
        stem_features: List[torch.Tensor],
        boxes_xyxy: torch.Tensor,
        img_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        使用 RoI Align 从多尺度特征图中提取检测框对应的特征。

        参数：
            stem_features: List[Tensor] - 多尺度特征图
            boxes_xyxy: Tensor (N, 4) - 检测框，格式为 (x1, y1, x2, y2)，像素坐标
            img_size: (H, W) - 输入图像大小

        返回：
            features: Tensor (N, stem_channels) - 每个框的特征向量
        """
        if boxes_xyxy.shape[0] == 0:
            return torch.zeros((0, self.stem_channels), device=self.device)

        img_h, img_w = img_size
        boxes_xyxy = boxes_xyxy.clone()
        boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp(0, img_w - 1)
        boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp(0, img_h - 1)

        # 根据 box 尺度选择 FPN 层级
        x1, y1, x2, y2 = boxes_xyxy.unbind(dim=1)
        box_w = (x2 - x1).clamp(min=1.0)
        box_h = (y2 - y1).clamp(min=1.0)
        box_scale = torch.sqrt(box_w * box_h)
        target_level = torch.log2(box_scale / self.canonical_scale + 1e-6) + float(self.canonical_level)
        level = (target_level - 3).clamp(0, len(self.fpn_strides) - 1).long()

        # 多尺度 RoIAlign
        features = torch.zeros(
            (boxes_xyxy.shape[0], self.stem_channels),
            device=self.device,
            dtype=stem_features[0].dtype,
        )

        if self.roi_multi_level:
            pooled_all = []
            for lvl, stride in enumerate(self.fpn_strides):
                feat_dtype = stem_features[lvl].dtype
                batch_idx = torch.zeros((boxes_xyxy.shape[0], 1), device=self.device, dtype=feat_dtype)
                rois = torch.cat([batch_idx, boxes_xyxy.to(dtype=feat_dtype)], dim=1)
                pooled = roi_align(
                    stem_features[lvl],
                    rois,
                    output_size=(self.roi_size, self.roi_size),
                    spatial_scale=1.0 / float(stride),
                    aligned=True,
                )
                pooled_all.append(pooled.mean(dim=[2, 3]))

            if len(pooled_all) > 0:
                pooled_stack = torch.stack(pooled_all, dim=1)  # (N, L, C)
                if self.roi_level_fusion == "avg":
                    features = pooled_stack.mean(dim=1)
                else:
                    # scale-aware fusion: weight by distance to target_level
                    level_nums = torch.arange(
                        3, 3 + len(self.fpn_strides),
                        device=boxes_xyxy.device,
                        dtype=target_level.dtype,
                    ).unsqueeze(0)
                    diff = (target_level.unsqueeze(1) - level_nums).abs()
                    weights = torch.softmax(-diff, dim=1).unsqueeze(-1)  # (N, L, 1)
                    features = (pooled_stack * weights).sum(dim=1)
        else:
            for lvl, stride in enumerate(self.fpn_strides):
                idx = (level == lvl).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                feat_dtype = stem_features[lvl].dtype
                batch_idx = torch.zeros((idx.shape[0], 1), device=self.device, dtype=feat_dtype)
                rois = torch.cat([batch_idx, boxes_xyxy[idx].to(dtype=feat_dtype)], dim=1)
                pooled = roi_align(
                    stem_features[lvl],
                    rois,
                    output_size=(self.roi_size, self.roi_size),
                    spatial_scale=1.0 / float(stride),
                    aligned=True,
                )
                features[idx] = pooled.mean(dim=[2, 3])

        return features

    @torch.no_grad()
    def detect_with_features(
        self,
        image_path: str
    ) -> Tuple[List[Tuple[float, float, float, float, float]], torch.Tensor]:
        """
        检测目标并提取对应特征。

        参数：
            image_path: 图像路径

        返回：
            detections: List[(x, y, w, h, conf)] - 像素坐标的检测框
            features: Tensor (N, feature_dim) - 对应的特征向量
        """
        import cv2
        from yolox.utils import postprocess

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        img_h, img_w = img_bgr.shape[:2]
        img, ratio = self._preprocess(img_bgr)

        # 1. 提取 FPN 特征
        fpn_outs = self._extract_fpn_features(img)

        # 2. 提取 stem 特征
        stem_features = self._extract_stem_features(fpn_outs)

        # 3. 运行检测头获取检测结果
        with torch.no_grad():
            outputs = self.yolox_model.head(fpn_outs)

        outputs = self._safe_postprocess(
            outputs,
            num_classes=self.num_classes,
            conf_thre=self.cfg.conf_thre,
            nms_thre=self.cfg.nms_thre,
            class_agnostic=self.cfg.class_agnostic_nms,
        )

        if outputs[0] is None:
            empty_features = torch.zeros((0, self.feature_dim), device=self.device)
            return [], empty_features

        dets = outputs[0]  # (N, 7): x1, y1, x2, y2, obj_conf, cls_conf, cls

        # 过滤非人类目标
        person_mask = dets[:, 6] == 0
        dets = dets[person_mask]

        if dets.shape[0] == 0:
            empty_features = torch.zeros((0, self.feature_dim), device=self.device)
            return [], empty_features

        # 4. 使用 RoI Align 提取特征（在原始特征图尺度上）
        # Filter invalid/degenerate boxes BEFORE RoIAlign to avoid NaN/Inf propagation.
        boxes_xyxy = dets[:, :4].clone()
        valid = self._valid_boxes_xyxy(boxes_xyxy, min_size=1.0)
        if not valid.all():
            if not self._warned_invalid_boxes:
                warnings.warn("[ByteTrackFeatureExtractor] Dropping invalid/degenerate YOLOX boxes before RoIAlign.")
                self._warned_invalid_boxes = True
            dets = dets[valid]
            boxes_xyxy = boxes_xyxy[valid]
        if dets.numel() == 0:
            empty_features = torch.zeros((0, self.feature_dim), device=self.device)
            return [], empty_features

        raw_features = self._roi_align_features(
            stem_features,
            boxes_xyxy,
            img_size=self.test_size,
        )

        # 5. 投影到目标特征维度
        if not torch.isfinite(raw_features).all():
            if not self._warned_nonfinite_roi:
                warnings.warn("[ByteTrackFeatureExtractor] Non-finite RoI features detected; applying nan_to_num().")
                self._warned_nonfinite_roi = True
            raw_features = torch.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0)
        raw_features = raw_features.float()  # 确保是 float32
        features = self.feature_proj(raw_features)  # (N, feature_dim)
        if not torch.isfinite(features).all():
            if not self._warned_nonfinite_features:
                warnings.warn("[ByteTrackFeatureExtractor] Non-finite projected features detected; applying nan_to_num().")
                self._warned_nonfinite_features = True
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # 6. 将检测框转换到原始图像坐标
        dets_np = dets.cpu().numpy()
        dets_np[:, :4] /= ratio
        dets_np[:, 0::2] = np.clip(dets_np[:, 0::2], 0, img_w - 1)
        dets_np[:, 1::2] = np.clip(dets_np[:, 1::2], 0, img_h - 1)

        results = []
        valid_indices = []
        for i, (x1, y1, x2, y2, obj_conf, cls_conf, cls_id) in enumerate(dets_np):
            conf = float(obj_conf * cls_conf)
            x = float(x1)
            y = float(y1)
            w = float(x2 - x1)
            h = float(y2 - y1)
            if w <= 1 or h <= 1:
                continue
            results.append((x, y, w, h, conf))
            valid_indices.append(i)

        # 只保留有效检测框对应的特征
        if len(valid_indices) > 0:
            features = features[valid_indices]
        else:
            features = torch.zeros((0, self.feature_dim), device=self.device)

        return results, features

    @torch.no_grad()
    def detect(
        self,
        image_path: str,
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Detect objects only (no RoI feature extraction).

        This is useful for inference modes that rely on external appearance features (e.g., ReID)
        and don't need YOLOX RoI features, saving compute vs `detect_with_features()`.
        """
        import cv2

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        img_h, img_w = img_bgr.shape[:2]
        img, ratio = self._preprocess(img_bgr)

        # 1) Extract FPN features (backbone+FPN)
        fpn_outs = self._extract_fpn_features(img)

        # 2) Run detection head
        outputs = self.yolox_model.head(fpn_outs)
        outputs = self._safe_postprocess(
            outputs,
            num_classes=self.num_classes,
            conf_thre=self.cfg.conf_thre,
            nms_thre=self.cfg.nms_thre,
            class_agnostic=self.cfg.class_agnostic_nms,
        )

        if outputs[0] is None:
            return []

        dets = outputs[0]  # (N, 7): x1, y1, x2, y2, obj_conf, cls_conf, cls

        # Person-only
        person_mask = dets[:, 6] == 0
        dets = dets[person_mask]
        if dets.numel() == 0:
            return []

        # Convert boxes to original image coordinates
        dets_np = dets.detach().cpu().numpy()
        dets_np[:, :4] /= ratio
        dets_np[:, 0::2] = np.clip(dets_np[:, 0::2], 0, img_w - 1)
        dets_np[:, 1::2] = np.clip(dets_np[:, 1::2], 0, img_h - 1)

        results: List[Tuple[float, float, float, float, float]] = []
        for x1, y1, x2, y2, obj_conf, cls_conf, _cls_id in dets_np:
            conf = float(obj_conf * cls_conf)
            x = float(x1)
            y = float(y1)
            w = float(x2 - x1)
            h = float(y2 - y1)
            if w <= 1 or h <= 1:
                continue
            results.append((x, y, w, h, conf))

        return results

    @torch.no_grad()
    def detect_with_features_tensor(
        self,
        image_tensor: torch.Tensor,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[Tuple[float, float, float, float, float]], torch.Tensor]:
        """
        使用已经增强后的图像张量进行检测与特征提取。

        参数：
            image_tensor: Tensor (C,H,W) 或 (H,W,C)，RGB
            input_size: 可选，YOLOX preproc 输入尺寸；默认使用 self.test_size

        返回：
            detections: List[(x, y, w, h, conf)] - 像素坐标的检测框（增强后图像坐标系）
            features: Tensor (N, feature_dim)
        """
        from yolox.utils import postprocess

        if image_tensor is None:
            raise ValueError("image_tensor is required")

        img_h = int(image_tensor.shape[-2]) if image_tensor.dim() >= 2 else None
        img_w = int(image_tensor.shape[-1]) if image_tensor.dim() >= 2 else None
        if img_h is None or img_w is None:
            raise ValueError("Invalid image_tensor shape for height/width.")

        img, ratio = self._preprocess_tensor(image_tensor, input_size=input_size or self.test_size)

        # 1. 提取 FPN 特征
        fpn_outs = self._extract_fpn_features(img)
        # 2. 提取 stem 特征
        stem_features = self._extract_stem_features(fpn_outs)
        # 3. 运行检测头获取检测结果
        outputs = self.yolox_model.head(fpn_outs)
        outputs = self._safe_postprocess(
            outputs,
            num_classes=self.num_classes,
            conf_thre=self.cfg.conf_thre,
            nms_thre=self.cfg.nms_thre,
            class_agnostic=self.cfg.class_agnostic_nms,
        )

        if outputs[0] is None:
            empty_features = torch.zeros((0, self.feature_dim), device=self.device)
            return [], empty_features

        dets = outputs[0]  # (N, 7): x1, y1, x2, y2, obj_conf, cls_conf, cls

        # 过滤非人类目标
        person_mask = dets[:, 6] == 0
        dets = dets[person_mask]

        if dets.shape[0] == 0:
            empty_features = torch.zeros((0, self.feature_dim), device=self.device)
            return [], empty_features

        # 4. RoI Align 提取特征（在预处理后的坐标系）
        # Filter invalid/degenerate boxes BEFORE RoIAlign to avoid NaN/Inf propagation.
        boxes_xyxy = dets[:, :4].clone()
        valid = self._valid_boxes_xyxy(boxes_xyxy, min_size=1.0)
        if not valid.all():
            if not self._warned_invalid_boxes:
                warnings.warn("[ByteTrackFeatureExtractor] Dropping invalid/degenerate YOLOX boxes before RoIAlign.")
                self._warned_invalid_boxes = True
            dets = dets[valid]
            boxes_xyxy = boxes_xyxy[valid]
        if dets.numel() == 0:
            empty_features = torch.zeros((0, self.feature_dim), device=self.device)
            return [], empty_features

        roi_size = input_size if input_size is not None else self.test_size
        raw_features = self._roi_align_features(
            stem_features,
            boxes_xyxy,
            img_size=roi_size,
        )

        # 5. 投影到目标特征维度
        if not torch.isfinite(raw_features).all():
            if not self._warned_nonfinite_roi:
                warnings.warn("[ByteTrackFeatureExtractor] Non-finite RoI features detected; applying nan_to_num().")
                self._warned_nonfinite_roi = True
            raw_features = torch.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0)
        raw_features = raw_features.float()
        features = self.feature_proj(raw_features)
        if not torch.isfinite(features).all():
            if not self._warned_nonfinite_features:
                warnings.warn("[ByteTrackFeatureExtractor] Non-finite projected features detected; applying nan_to_num().")
                self._warned_nonfinite_features = True
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # 6. 将检测框映射回增强后图像坐标
        dets_np = dets.cpu().numpy()
        dets_np[:, :4] /= ratio
        dets_np[:, 0::2] = np.clip(dets_np[:, 0::2], 0, img_w - 1)
        dets_np[:, 1::2] = np.clip(dets_np[:, 1::2], 0, img_h - 1)

        results = []
        valid_indices = []
        for i, (x1, y1, x2, y2, obj_conf, cls_conf, cls_id) in enumerate(dets_np):
            conf = float(obj_conf * cls_conf)
            x = float(x1)
            y = float(y1)
            w = float(x2 - x1)
            h = float(y2 - y1)
            if w <= 1 or h <= 1:
                continue
            results.append((x, y, w, h, conf))
            valid_indices.append(i)

        if len(valid_indices) > 0:
            features = features[valid_indices]
        else:
            features = torch.zeros((0, self.feature_dim), device=self.device)

        return results, features

    @torch.no_grad()
    def detect_with_features_tta(
        self,
        image_path: str,
        tta,
    ) -> Tuple[List[Tuple[float, float, float, float, float]], torch.Tensor]:
        """
        Test-Time Augmentation detection with feature extraction.

        Args:
            image_path: path to the image file
            tta: TestTimeAugmentation instance

        Returns:
            detections: List[(x, y, w, h, conf)] in original image coords
            features: Tensor (N, feature_dim) aligned with detections
        """
        import cv2
        from torchvision.ops import nms

        if tta is None:
            return self.detect_with_features(image_path)

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        img_h, img_w = img_bgr.shape[:2]
        # BGR -> RGB
        img_rgb = img_bgr[:, :, ::-1].copy()
        base = torch.from_numpy(img_rgb).permute(2, 0, 1).float()

        all_boxes_xyxy = []
        all_scores = []
        all_features = []

        scales = getattr(tta, "scales", [1.0])
        flip_enabled = bool(getattr(tta, "flip", False))
        flips = [False, True] if flip_enabled else [False]

        for scale in scales:
            for do_flip in flips:
                aug = tta.augment_image(base, scale=scale, flip=do_flip)
                dets, feats = self.detect_with_features_tensor(aug, input_size=None)
                if len(dets) == 0:
                    continue

                # Convert dets to xyxy in augmented coords
                boxes = torch.tensor(
                    [[d[0], d[1], d[0] + d[2], d[1] + d[3]] for d in dets],
                    dtype=torch.float32,
                    device=self.device,
                )
                scores = torch.tensor([d[4] for d in dets], dtype=torch.float32, device=self.device)

                # Reverse augmentation to original coords
                boxes = tta.reverse_boxes(boxes, scale=scale, flip=do_flip, orig_size=(img_h, img_w))
                boxes[:, 0::2] = boxes[:, 0::2].clamp(0, img_w - 1)
                boxes[:, 1::2] = boxes[:, 1::2].clamp(0, img_h - 1)

                all_boxes_xyxy.append(boxes)
                all_scores.append(scores)
                all_features.append(feats)

        if len(all_boxes_xyxy) == 0:
            empty_features = torch.zeros((0, self.feature_dim), device=self.device)
            return [], empty_features

        boxes_all = torch.cat(all_boxes_xyxy, dim=0)
        scores_all = torch.cat(all_scores, dim=0)
        feats_all = torch.cat(all_features, dim=0) if len(all_features) > 1 else all_features[0]

        keep = nms(boxes_all, scores_all, iou_threshold=0.5)
        boxes_all = boxes_all[keep]
        scores_all = scores_all[keep]
        feats_all = feats_all[keep]

        # Convert back to xywh list
        results = []
        for i in range(boxes_all.shape[0]):
            x1, y1, x2, y2 = boxes_all[i].tolist()
            results.append((x1, y1, x2 - x1, y2 - y1, float(scores_all[i].item())))

        # Re-extract features on original image for consistency
        try:
            boxes_xywh = [(r[0], r[1], r[2], r[3]) for r in results]
            feats_all = self.extract_features_from_boxes(image_path, boxes_xywh)
        except Exception:
            # fallback to TTA features if re-extraction fails
            pass

        return results, feats_all

    def extract_features_from_boxes(
        self,
        image_path: str,
        boxes_xywh: List[Tuple[float, float, float, float]],
    ) -> torch.Tensor:
        """
        给定检测框，提取对应的特征。

        用于训练时，使用 GT 框提取特征。

        参数：
            image_path: 图像路径
            boxes_xywh: List[(x, y, w, h)] - 像素坐标的检测框

        返回：
            features: Tensor (N, feature_dim) - 对应的特征向量
        """
        import cv2

        if len(boxes_xywh) == 0:
            return torch.zeros((0, self.feature_dim), device=self.device)

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        img_h, img_w = img_bgr.shape[:2]
        img, ratio = self._preprocess(img_bgr)

        # 提取特征图
        fpn_outs = self._extract_fpn_features(img)
        stem_features = self._extract_stem_features(fpn_outs)

        # 将 xywh 转换为 xyxy 并缩放到特征图坐标系
        boxes_xyxy = []
        for x, y, w, h in boxes_xywh:
            x1 = x * ratio
            y1 = y * ratio
            x2 = (x + w) * ratio
            y2 = (y + h) * ratio
            boxes_xyxy.append([x1, y1, x2, y2])

        boxes_xyxy = torch.tensor(boxes_xyxy, dtype=torch.float32, device=self.device)
        valid = self._valid_boxes_xyxy(boxes_xyxy, min_size=1.0)
        if not valid.all():
            if not self._warned_invalid_boxes:
                warnings.warn("[ByteTrackFeatureExtractor] Dropping invalid boxes in extract_features_from_boxes().")
                self._warned_invalid_boxes = True
            # keep output shape aligned with input ordering
            out = torch.zeros((boxes_xyxy.shape[0], self.feature_dim), device=self.device)
            if valid.any():
                raw_features = self._roi_align_features(
                    stem_features,
                    boxes_xyxy[valid],
                    img_size=self.test_size,
                )
                if not torch.isfinite(raw_features).all():
                    if not self._warned_nonfinite_roi:
                        warnings.warn("[ByteTrackFeatureExtractor] Non-finite RoI features detected; applying nan_to_num().")
                        self._warned_nonfinite_roi = True
                    raw_features = torch.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0)
                feats = self.feature_proj(raw_features.float())
                if not torch.isfinite(feats).all():
                    if not self._warned_nonfinite_features:
                        warnings.warn("[ByteTrackFeatureExtractor] Non-finite projected features detected; applying nan_to_num().")
                        self._warned_nonfinite_features = True
                    feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
                out[valid] = feats
            return out

        # RoI Align 提取特征
        raw_features = self._roi_align_features(
            stem_features,
            boxes_xyxy,
            img_size=self.test_size,
        )

        # 投影
        if not torch.isfinite(raw_features).all():
            if not self._warned_nonfinite_roi:
                warnings.warn("[ByteTrackFeatureExtractor] Non-finite RoI features detected; applying nan_to_num().")
                self._warned_nonfinite_roi = True
            raw_features = torch.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0)
        raw_features = raw_features.float()
        features = self.feature_proj(raw_features)
        if not torch.isfinite(features).all():
            if not self._warned_nonfinite_features:
                warnings.warn("[ByteTrackFeatureExtractor] Non-finite projected features detected; applying nan_to_num().")
                self._warned_nonfinite_features = True
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def extract_features_from_boxes_tensor(
        self,
        image_tensor: torch.Tensor,
        boxes_xywh: List[Tuple[float, float, float, float]],
        input_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        给定增强后的图像张量与检测框（像素坐标，增强后坐标系），提取对应特征。
        """
        if len(boxes_xywh) == 0:
            return torch.zeros((0, self.feature_dim), device=self.device)

        if image_tensor is None:
            raise ValueError("image_tensor is required")

        img_h = int(image_tensor.shape[-2]) if image_tensor.dim() >= 2 else None
        img_w = int(image_tensor.shape[-1]) if image_tensor.dim() >= 2 else None
        if img_h is None or img_w is None:
            raise ValueError("Invalid image_tensor shape for height/width.")

        img, ratio = self._preprocess_tensor(image_tensor, input_size=input_size or self.test_size)

        # 提取特征图
        fpn_outs = self._extract_fpn_features(img)
        stem_features = self._extract_stem_features(fpn_outs)

        # 将 xywh 转换为 xyxy 并缩放到预处理坐标系
        boxes_xyxy = []
        for x, y, w, h in boxes_xywh:
            x1 = x * ratio
            y1 = y * ratio
            x2 = (x + w) * ratio
            y2 = (y + h) * ratio
            boxes_xyxy.append([x1, y1, x2, y2])

        boxes_xyxy = torch.tensor(boxes_xyxy, dtype=torch.float32, device=self.device)
        roi_size = input_size if input_size is not None else self.test_size
        valid = self._valid_boxes_xyxy(boxes_xyxy, min_size=1.0)
        if not valid.all():
            if not self._warned_invalid_boxes:
                warnings.warn("[ByteTrackFeatureExtractor] Dropping invalid boxes in extract_features_from_boxes_tensor().")
                self._warned_invalid_boxes = True
            out = torch.zeros((boxes_xyxy.shape[0], self.feature_dim), device=self.device)
            if valid.any():
                raw_features = self._roi_align_features(
                    stem_features,
                    boxes_xyxy[valid],
                    img_size=roi_size,
                )
                if not torch.isfinite(raw_features).all():
                    if not self._warned_nonfinite_roi:
                        warnings.warn("[ByteTrackFeatureExtractor] Non-finite RoI features detected; applying nan_to_num().")
                        self._warned_nonfinite_roi = True
                    raw_features = torch.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0)
                feats = self.feature_proj(raw_features.float())
                if not torch.isfinite(feats).all():
                    if not self._warned_nonfinite_features:
                        warnings.warn("[ByteTrackFeatureExtractor] Non-finite projected features detected; applying nan_to_num().")
                        self._warned_nonfinite_features = True
                    feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
                out[valid] = feats
            return out

        raw_features = self._roi_align_features(
            stem_features,
            boxes_xyxy,
            img_size=roi_size,
        )

        if not torch.isfinite(raw_features).all():
            if not self._warned_nonfinite_roi:
                warnings.warn("[ByteTrackFeatureExtractor] Non-finite RoI features detected; applying nan_to_num().")
                self._warned_nonfinite_roi = True
            raw_features = torch.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0)
        raw_features = raw_features.float()
        features = self.feature_proj(raw_features)
        if not torch.isfinite(features).all():
            if not self._warned_nonfinite_features:
                warnings.warn("[ByteTrackFeatureExtractor] Non-finite projected features detected; applying nan_to_num().")
                self._warned_nonfinite_features = True
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features


class ByteTrackFeatureProvider(nn.Module):
    """
    为训练提供 ByteTrack 特征的包装器。

    这个类用于在训练循环中：
    1. 从图像中提取 YOLOX 特征（冻结）
    2. 使用 GT 框从特征图中提取对应特征
    3. 返回格式与 DINO 输出兼容的字典

    替代原有的 DINO forward，使得训练流程可以使用 ByteTrack 特征。
    """

    def __init__(self, cfg: ByteTrackFeatureConfig, device: torch.device):
        super().__init__()
        self.extractor = ByteTrackFeatureExtractor(cfg, device)
        self.feature_dim = cfg.feature_dim
        self.device = device

    def forward(
        self,
        image_paths: List[str],
        gt_boxes_list: List[List[Tuple[float, float, float, float]]],
    ) -> Dict[str, torch.Tensor]:
        """
        提取一批图像的特征。

        参数：
            image_paths: List[str] - B*T 个图像路径
            gt_boxes_list: List[List[(x,y,w,h)]] - 每帧的 GT 框列表

        返回：
            dict 包含：
                "outputs": Tensor (B*T, max_N, feature_dim) - 特征
                "pred_boxes": Tensor (B*T, max_N, 4) - 归一化的 cxcywh 框
                "pred_logits": Tensor (B*T, max_N, 1) - 伪置信度（全1）
        """
        import cv2

        batch_size = len(image_paths)

        all_features = []
        all_boxes = []
        all_logits = []
        max_num_boxes = max(len(boxes) for boxes in gt_boxes_list) if gt_boxes_list else 0
        max_num_boxes = max(max_num_boxes, 1)  # 至少1个

        for i, (image_path, gt_boxes) in enumerate(zip(image_paths, gt_boxes_list)):
            # 读取图像获取尺寸
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")
            img_h, img_w = img.shape[:2]

            # 提取特征
            if len(gt_boxes) > 0:
                features = self.extractor.extract_features_from_boxes(image_path, gt_boxes)
            else:
                features = torch.zeros((0, self.feature_dim), device=self.device)

            # 转换框格式：xywh -> cxcywh (归一化)
            boxes_cxcywh = []
            for x, y, w, h in gt_boxes:
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                boxes_cxcywh.append([cx, cy, nw, nh])

            if len(boxes_cxcywh) > 0:
                boxes = torch.tensor(boxes_cxcywh, dtype=torch.float32, device=self.device)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32, device=self.device)

            # 伪置信度
            logits = torch.ones((features.shape[0], 1), dtype=torch.float32, device=self.device)

            # Padding 到 max_num_boxes
            num_boxes = features.shape[0]
            if num_boxes < max_num_boxes:
                pad_features = torch.zeros((max_num_boxes - num_boxes, self.feature_dim), device=self.device)
                features = torch.cat([features, pad_features], dim=0)

                pad_boxes = torch.zeros((max_num_boxes - num_boxes, 4), device=self.device)
                boxes = torch.cat([boxes, pad_boxes], dim=0)

                pad_logits = torch.zeros((max_num_boxes - num_boxes, 1), device=self.device)
                logits = torch.cat([logits, pad_logits], dim=0)

            all_features.append(features)
            all_boxes.append(boxes)
            all_logits.append(logits)

        # Stack
        outputs = torch.stack(all_features, dim=0)  # (B*T, max_N, feature_dim)
        pred_boxes = torch.stack(all_boxes, dim=0)  # (B*T, max_N, 4)
        pred_logits = torch.stack(all_logits, dim=0)  # (B*T, max_N, 1)

        return {
            "outputs": outputs,
            "pred_boxes": pred_boxes,
            "pred_logits": pred_logits,
        }

    def freeze(self):
        """冻结 YOLOX 模型，但保持投影层可训练"""
        for param in self.extractor.yolox_model.parameters():
            param.requires_grad = False
        # 投影层保持可训练
        for param in self.extractor.feature_proj.parameters():
            param.requires_grad = True

    def unfreeze_projection(self):
        """解冻投影层"""
        for param in self.extractor.feature_proj.parameters():
            param.requires_grad = True
