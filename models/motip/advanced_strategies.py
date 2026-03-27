# Copyright (c) 2024-2026. All Rights Reserved.
"""
Advanced Strategies for FM-Track

Core Innovation Extensions:
1. FrequencyAwareConfidenceCalibration - 频率感知置信度校准
2. FrequencyGuidedOcclusionRecovery - 频率引导遮挡恢复
3. AdaptiveBandSelection - 自适应频带数量选择

Inference Optimization:
4. TrackInterpolation - 轨迹插值后处理
5. TestTimeAugmentation - 测试时增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import warnings
from typing import Dict, List, Tuple, Optional
try:
    from scipy import interpolate as scipy_interp
    _SCIPY_AVAILABLE = True
except Exception:
    scipy_interp = None
    _SCIPY_AVAILABLE = False


# ============================================================================
# 1. Frequency-Aware Confidence Calibration (核心创新延伸)
# ============================================================================

class FrequencyAwareConfidenceCalibration(nn.Module):
    """
    频率感知置信度校准
    
    核心思想: 各频带预测的一致性程度反映了预测的可靠性
    - 高一致性 → 高置信度
    - 低一致性 → 降低置信度 (可能是困难样本或遮挡)
    
    论文贡献: 利用频率分解的多视角特性进行不确定性估计
    """
    
    def __init__(
        self,
        num_bands: int = 4,
        calibration_strength: float = 0.5,
        min_confidence: float = 0.1,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.calibration_strength = calibration_strength
        self.min_confidence = min_confidence
        
        # Learnable calibration parameters
        self.consistency_weight = nn.Parameter(torch.tensor(1.0))
        self.entropy_weight = nn.Parameter(torch.tensor(0.5))
        
    def compute_band_consistency(
        self,
        band_logits: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        计算各频带预测的一致性
        
        Args:
            band_logits: List of (B, T, N, V) logits from each band
            
        Returns:
            consistency: (B, T, N) consistency scores in [0, 1]
        """
        if len(band_logits) < 2:
            return torch.ones_like(band_logits[0][..., 0])
        
        # Convert to probabilities
        band_probs = [F.softmax(logits, dim=-1) for logits in band_logits]
        
        # Compute average probability
        avg_prob = torch.stack(band_probs, dim=0).mean(dim=0)
        
        # Compute KL divergence from each band to the average distribution.
        #
        # IMPORTANT: Avoid `F.kl_div(input_logp, target_prob)` with default `log_target=False` here.
        # When `target_prob` has zeros (common under float16 softmax underflow), PyTorch may compute
        # `target * log(target)` as `0 * (-inf)` which becomes NaN and poisons training/eval.
        eps = 1e-6
        avg_prob = avg_prob.clamp(min=eps, max=1.0)
        avg_log = avg_prob.log()
        kl_divs = []
        for prob in band_probs:
            prob_c = prob.clamp(min=eps, max=1.0)
            kl = (prob_c * (prob_c.log() - avg_log)).sum(dim=-1)
            kl_divs.append(kl)
        
        # Average KL divergence
        avg_kl = torch.stack(kl_divs, dim=0).mean(dim=0)
        
        # Convert to consistency score (lower KL = higher consistency)
        consistency = torch.exp(-avg_kl)
        
        return consistency
    
    def compute_prediction_entropy(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算预测熵 (不确定性指标)
        
        Args:
            logits: (B, T, N, V) prediction logits
            
        Returns:
            normalized_entropy: (B, T, N) in [0, 1]
        """
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * probs.log().clamp(min=-100)).sum(dim=-1)
        
        # Normalize by max entropy
        max_entropy = float(np.log(logits.shape[-1]))
        max_entropy = max(max_entropy, 1e-8)
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    def forward(
        self,
        raw_scores: torch.Tensor,
        band_logits: List[torch.Tensor],
        fused_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        校准置信度分数
        
        Args:
            raw_scores: (B, T, N) 原始置信度分数
            band_logits: 各频带的预测logits
            fused_logits: 融合后的logits
            
        Returns:
            calibrated_scores: (B, T, N) 校准后的分数
        """
        # Compute calibration factor
        calibration_factor = self.compute_calibration_factor(band_logits, fused_logits)
        
        # Apply calibration
        calibrated_scores = raw_scores * (
            1.0 - self.calibration_strength + 
            self.calibration_strength * calibration_factor
        )
        
        # Ensure minimum confidence
        calibrated_scores = calibrated_scores.clamp(min=self.min_confidence)
        
        return calibrated_scores

    def compute_calibration_factor(
        self,
        band_logits: List[torch.Tensor],
        fused_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reliability factor based on band consistency and prediction entropy.

        Returns:
            calibration_factor: (B, T, N) in [0, 1]
        """
        consistency = self.compute_band_consistency(band_logits)
        entropy = self.compute_prediction_entropy(fused_logits)
        confidence_from_entropy = 1.0 - entropy

        calibration_factor = (
            torch.sigmoid(self.consistency_weight) * consistency +
            torch.sigmoid(self.entropy_weight) * confidence_from_entropy
        ) / 2.0
        return calibration_factor


# ============================================================================
# 2. Frequency-Guided Occlusion Recovery (核心创新延伸)
# ============================================================================

class FrequencyGuidedOcclusionRecovery(nn.Module):
    """
    频率引导的遮挡恢复
    
    核心思想: 遮挡主要破坏高频信息，低频结构相对保持
    利用低频特征进行特征恢复和身份关联
    
    论文贡献: 展示频率分解在遮挡处理中的独特优势
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_bands: int = 4,
        recovery_ratio: float = 0.3,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_bands = num_bands
        self.recovery_ratio = recovery_ratio
        
        # Occlusion detection network
        self.occlusion_detector = nn.Sequential(
            nn.Linear(feature_dim * num_bands, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Feature recovery network (uses low-freq to recover high-freq)
        self.recovery_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        
        # Band importance predictor
        self.band_importance = nn.Sequential(
            nn.Linear(feature_dim, num_bands),
            nn.Softmax(dim=-1),
        )
        
    def detect_occlusion(
        self,
        band_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        检测遮挡程度
        
        Args:
            band_features: List of (B, T, N, D) features per band
            
        Returns:
            occlusion_score: (B, T, N) in [0, 1], higher = more occluded
        """
        # Concatenate all band features
        concat_features = torch.cat(band_features, dim=-1)
        occlusion_score = self.occlusion_detector(concat_features).squeeze(-1)
        return occlusion_score
    
    def recover_features(
        self,
        band_features: List[torch.Tensor],
        occlusion_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        基于低频特征恢复被遮挡的高频特征
        
        Args:
            band_features: List of (B, T, N, D) features per band
            occlusion_scores: (B, T, N) occlusion level
            
        Returns:
            recovered_features: (B, T, N, D) recovered full features
        """
        # Low-frequency features (first band)
        low_freq = band_features[0]
        
        # High-frequency features (last band)
        high_freq = band_features[-1]
        
        # Predict recovered high-freq from low-freq
        recovered_high = self.recovery_net(low_freq)
        
        # Blend based on occlusion score
        blend_weight = occlusion_scores.unsqueeze(-1) * self.recovery_ratio
        blended_high = (1 - blend_weight) * high_freq + blend_weight * recovered_high
        
        # Recompute fused features
        band_importance = self.band_importance(low_freq)  # (B, T, N, num_bands)
        
        # Weighted sum of all bands (with recovered high-freq)
        all_bands = band_features[:-1] + [blended_high]
        stacked = torch.stack(all_bands, dim=-1)  # (B, T, N, D, K)
        recovered = (stacked * band_importance.unsqueeze(-2)).sum(dim=-1)
        
        return recovered
    
    def forward(
        self,
        band_features: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with occlusion detection and recovery
        """
        occlusion_scores = self.detect_occlusion(band_features)
        recovered_features = self.recover_features(band_features, occlusion_scores)
        
        info = {
            'occlusion_scores': occlusion_scores,
            'recovery_applied': (occlusion_scores > 0.5).float().mean(),
        }
        
        return recovered_features, info


# ============================================================================
# 3. Adaptive Band Selection (核心创新延伸)
# ============================================================================

class AdaptiveBandSelector(nn.Module):
    """
    自适应频带数量选择
    
    核心思想: 不同场景需要不同的频率分辨率
    - 简单场景: 少量频带足够
    - 复杂场景 (拥挤/遮挡): 需要更多频带
    
    论文贡献: 证明频带数量与场景复杂度的关系
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        max_bands: int = 8,
        min_bands: int = 2,
        soft_band_temp: float = 1.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_bands = max_bands
        self.min_bands = min_bands
        self.soft_band_temp = soft_band_temp
        self._warned_nonfinite = False
        
        # Scene complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Band count predictor
        self.band_predictor = nn.Sequential(
            nn.Linear(feature_dim + 1, 32),
            nn.GELU(),
            nn.Linear(32, max_bands - min_bands + 1),
            nn.Softmax(dim=-1),
        )
        
    def estimate_complexity(
        self,
        features: torch.Tensor,
        num_objects: int,
    ) -> torch.Tensor:
        """
        估计场景复杂度
        
        Args:
            features: (B, T, N, D) scene features
            num_objects: number of objects in scene
            
        Returns:
            complexity: (B, T) complexity score in [0, 1]
        """
        # Global feature (mean over objects)
        global_feat = features.mean(dim=2)  # (B, T, D)
        
        # Estimate complexity
        complexity = self.complexity_estimator(global_feat).squeeze(-1)
        
        # Also consider object count
        density_factor = min(1.0, num_objects / 50.0)
        complexity = 0.7 * complexity + 0.3 * density_factor
        
        return complexity
    
    def forward(
        self,
        features: torch.Tensor,
        num_objects: int,
        total_bands: Optional[int] = None,
    ) -> Tuple[int, Dict]:
        """
        Forward pass to determine optimal band count
        """
        if total_bands is None:
            total_bands = self.max_bands
        total_bands = int(total_bands)

        # IMPORTANT:
        # If features are already NaN/Inf here, the *upstream* model has become numerically unstable.
        # Silently `nan_to_num`-ing will poison training (band_features/freq_decomposition already happened earlier).
        if not torch.isfinite(features).all():
            if self.training:
                raise FloatingPointError(
                    "[AdaptiveBandSelector] Non-finite features detected (NaN/Inf) in training. "
                    "This indicates an upstream numerical issue (e.g., MemoryBank / feature extractor / "
                    "trajectory modeling). Stop instead of continuing with corrupted gradients."
                )
            if not self._warned_nonfinite:
                warnings.warn(
                    "[AdaptiveBandSelector] Non-finite features detected in eval; replacing NaN/Inf with 0 "
                    "to keep inference running."
                )
                self._warned_nonfinite = True
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        complexity = self.estimate_complexity(features, num_objects)
        # Global scene descriptor
        global_feat = features.mean(dim=2)  # (B, T, D)
        global_feat_mean = global_feat.mean(dim=1)  # (B, D)
        complexity_mean = complexity.mean(dim=1, keepdim=True)  # (B, 1)

        # Predict band-count distribution (learnable)
        pred_in = torch.cat([global_feat_mean, complexity_mean], dim=-1)
        band_probs = self.band_predictor(pred_in)  # (B, K)
        if not torch.isfinite(band_probs).all():
            if self.training:
                raise FloatingPointError(
                    "[AdaptiveBandSelector] Non-finite band_probs detected (NaN/Inf) in training."
                )
            if not self._warned_nonfinite:
                warnings.warn(
                    "[AdaptiveBandSelector] Non-finite band_probs detected in eval; fallback to uniform distribution."
                )
                self._warned_nonfinite = True
            band_probs = torch.full_like(band_probs, 1.0 / max(int(band_probs.shape[-1]), 1))

        band_counts = torch.arange(
            self.min_bands,
            self.max_bands + 1,
            device=band_probs.device,
            dtype=band_probs.dtype,
        )
        expected_bands = (band_probs * band_counts.unsqueeze(0)).sum(dim=-1)
        expected_mean = expected_bands.mean()

        # Final discrete selection
        expected_mean_val = float(expected_mean.item())
        if not math.isfinite(expected_mean_val):
            if self.training:
                raise FloatingPointError(
                    "[AdaptiveBandSelector] expected_mean is NaN/Inf in training."
                )
            if not self._warned_nonfinite:
                warnings.warn(
                    "[AdaptiveBandSelector] expected_mean is NaN/Inf in eval; fallback to keep all bands enabled."
                )
                self._warned_nonfinite = True
            expected_mean_val = float(total_bands)
        num_bands = int(round(expected_mean_val))
        num_bands = max(self.min_bands, min(self.max_bands, num_bands))
        num_bands = max(1, min(total_bands, num_bands))

        # Soft band weights for differentiable gating
        expected_mean_safe = band_probs.new_tensor(expected_mean_val)
        band_indices = torch.arange(
            1,
            total_bands + 1,
            device=band_probs.device,
            dtype=band_probs.dtype,
        )
        band_weights = torch.sigmoid((expected_mean_safe - (band_indices - 0.5)) / max(1e-6, self.soft_band_temp))
        band_mask = band_weights > 0.5
        min_keep = min(int(self.min_bands), int(total_bands))
        if int(band_mask.sum().item()) < min_keep:
            band_mask[:min_keep] = True
        
        info = {
            'scene_complexity': float(torch.nan_to_num(complexity.mean(), nan=0.0).item()),
            'selected_bands': num_bands,
            'expected_bands': float(expected_mean_val),
            'band_probs': band_probs,
            'band_weights': band_weights,
            'band_mask': band_mask,
        }
        
        return num_bands, info


# ============================================================================
# 4. Track Interpolation (推理优化)
# ============================================================================

class TrackInterpolation:
    """
    轨迹插值后处理
    
    功能: 补全短暂丢失的轨迹片段
    
    方法:
    - 线性插值: 快速，适合短gap
    - 样条插值: 平滑，适合长gap
    """
    
    def __init__(
        self,
        max_gap: int = 10,
        min_track_length: int = 3,
        interpolation_method: str = 'linear',
    ):
        self.max_gap = max_gap
        self.min_track_length = min_track_length
        self.interpolation_method = interpolation_method
        
    def find_gaps(
        self,
        track_frames: List[int],
    ) -> List[Tuple[int, int]]:
        """
        找到轨迹中的间隙
        
        Args:
            track_frames: 轨迹出现的帧索引列表
            
        Returns:
            gaps: List of (start_frame, end_frame) gaps
        """
        if len(track_frames) < 2:
            return []
        
        gaps = []
        sorted_frames = sorted(track_frames)
        
        for i in range(len(sorted_frames) - 1):
            gap_size = sorted_frames[i + 1] - sorted_frames[i] - 1
            if 0 < gap_size <= self.max_gap:
                gaps.append((sorted_frames[i], sorted_frames[i + 1]))
        
        return gaps
    
    def interpolate_boxes(
        self,
        start_box: np.ndarray,
        end_box: np.ndarray,
        start_frame: int,
        end_frame: int,
    ) -> Dict[int, np.ndarray]:
        """
        插值两个边界框之间的位置
        
        Args:
            start_box: (4,) [x, y, w, h]
            end_box: (4,) [x, y, w, h]
            start_frame: 起始帧
            end_frame: 结束帧
            
        Returns:
            interpolated: Dict[frame_id, box]
        """
        num_frames = end_frame - start_frame - 1
        if num_frames <= 0:
            return {}
        
        interpolated = {}
        
        if self.interpolation_method == 'linear':
            for i in range(1, num_frames + 1):
                alpha = i / (num_frames + 1)
                box = (1 - alpha) * start_box + alpha * end_box
                interpolated[start_frame + i] = box
                
        elif self.interpolation_method == 'spline':
            if not _SCIPY_AVAILABLE:
                # Fallback to linear if scipy is unavailable.
                for i in range(1, num_frames + 1):
                    alpha = i / (num_frames + 1)
                    box = (1 - alpha) * start_box + alpha * end_box
                    interpolated[start_frame + i] = box
                return interpolated
            # Use cubic spline for smoother interpolation
            frames = [start_frame, end_frame]
            boxes = np.stack([start_box, end_box])
            # CubicSpline with only 2 control points degenerates to linear interpolation.
            # Use the linear path directly to avoid unnecessary SciPy overhead and potential numerical quirks.
            if len(frames) < 3:
                for i in range(1, num_frames + 1):
                    alpha = i / (num_frames + 1)
                    box = (1 - alpha) * start_box + alpha * end_box
                    interpolated[start_frame + i] = box
                return interpolated
            
            for dim in range(4):
                cs = scipy_interp.CubicSpline(frames, boxes[:, dim])
                for frame in range(start_frame + 1, end_frame):
                    if frame not in interpolated:
                        interpolated[frame] = np.zeros(4)
                    interpolated[frame][dim] = cs(frame)
        
        return interpolated
    
    def process_tracks(
        self,
        tracks: Dict[int, Dict[int, np.ndarray]],
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        处理所有轨迹，填补间隙
        
        Args:
            tracks: Dict[track_id, Dict[frame_id, box]]
            
        Returns:
            processed_tracks: 填补后的轨迹
        """
        processed = {}
        
        for track_id, track_data in tracks.items():
            frames = sorted(track_data.keys())
            
            if len(frames) < self.min_track_length:
                processed[track_id] = track_data
                continue
            
            # Find and fill gaps
            gaps = self.find_gaps(frames)
            new_track = dict(track_data)
            
            for start_frame, end_frame in gaps:
                start_box = track_data[start_frame]
                end_box = track_data[end_frame]
                
                interpolated = self.interpolate_boxes(
                    start_box, end_box, start_frame, end_frame
                )
                new_track.update(interpolated)
            
            processed[track_id] = new_track
        
        return processed


# ============================================================================
# 5. Test-Time Augmentation (推理优化)
# ============================================================================

class TestTimeAugmentation:
    """
    测试时增强
    
    功能: 多尺度/多翻转推理，提升鲁棒性
    
    方法:
    - 多尺度: [0.8, 1.0, 1.2]
    - 水平翻转: True/False
    - 融合策略: 平均 / 加权平均
    """
    
    def __init__(
        self,
        scales: List[float] = [0.8, 1.0, 1.2],
        flip: bool = True,
        fusion_method: str = 'average',
    ):
        self.scales = scales
        self.flip = flip
        self.fusion_method = fusion_method
        
    def augment_image(
        self,
        image: torch.Tensor,
        scale: float,
        flip: bool,
    ) -> torch.Tensor:
        """
        应用增强到图像
        
        Args:
            image: (C, H, W) or (B, C, H, W)
            scale: 缩放比例
            flip: 是否水平翻转
            
        Returns:
            augmented: 增强后的图像
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        B, C, H, W = image.shape
        
        # Scale
        if scale != 1.0:
            new_H = int(H * scale)
            new_W = int(W * scale)
            image = F.interpolate(
                image, size=(new_H, new_W), 
                mode='bilinear', align_corners=False
            )
        
        # Flip
        if flip:
            image = torch.flip(image, dims=[-1])
        
        return image
    
    def reverse_boxes(
        self,
        boxes: torch.Tensor,
        scale: float,
        flip: bool,
        orig_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        将增强后的检测框转换回原始坐标
        
        Args:
            boxes: (N, 4) [x1, y1, x2, y2] or [cx, cy, w, h]
            scale: 使用的缩放比例
            flip: 是否翻转
            orig_size: (H, W) 原始尺寸
            
        Returns:
            reversed_boxes: 原始坐标系下的框
        """
        H, W = orig_size
        boxes = boxes.clone()
        
        # Reverse flip
        if flip:
            if boxes.shape[-1] == 4:
                # Assuming [x1, y1, x2, y2] format
                boxes[:, [0, 2]] = W * scale - boxes[:, [2, 0]]
        
        # Reverse scale
        if scale != 1.0:
            boxes = boxes / scale
        
        return boxes
    
    def fuse_predictions(
        self,
        all_boxes: List[torch.Tensor],
        all_scores: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        融合多次增强的预测结果
        
        Args:
            all_boxes: List of (N_i, 4) boxes from each augmentation
            all_scores: List of (N_i,) scores
            
        Returns:
            fused_boxes: (M, 4) fused boxes
            fused_scores: (M,) fused scores
        """
        if self.fusion_method == 'average':
            # Simple concatenation and NMS
            boxes = torch.cat(all_boxes, dim=0)
            scores = torch.cat(all_scores, dim=0)
            
            # Apply NMS
            from torchvision.ops import nms
            keep = nms(boxes, scores, iou_threshold=0.5)
            
            return boxes[keep], scores[keep]
        
        elif self.fusion_method == 'weighted':
            # Weight by scale (larger scale = higher weight for small objects)
            # This is a simplified version
            boxes = torch.cat(all_boxes, dim=0)
            scores = torch.cat(all_scores, dim=0)
            
            from torchvision.ops import nms
            keep = nms(boxes, scores, iou_threshold=0.5)
            
            return boxes[keep], scores[keep]
        
        return all_boxes[0], all_scores[0]


# ============================================================================
# Utility Functions
# ============================================================================

def build_confidence_calibration(config: dict) -> FrequencyAwareConfidenceCalibration:
    return FrequencyAwareConfidenceCalibration(
        num_bands=config.get('NUM_FREQ_BANDS', config.get('NUM_BANDS', 4)),
        calibration_strength=config.get('CALIBRATION_STRENGTH', 0.5),
        min_confidence=config.get('MIN_CONFIDENCE', 0.1),
    )


def build_occlusion_recovery(config: dict) -> FrequencyGuidedOcclusionRecovery:
    return FrequencyGuidedOcclusionRecovery(
        feature_dim=config.get('FEATURE_DIM', 256),
        num_bands=config.get('NUM_FREQ_BANDS', config.get('NUM_BANDS', 4)),
        recovery_ratio=config.get('OCCLUSION_RECOVERY_RATIO', 0.3),
    )


def build_track_interpolation(config: dict) -> TrackInterpolation:
    return TrackInterpolation(
        max_gap=config.get('INTERPOLATION_MAX_GAP', 10),
        min_track_length=config.get('INTERPOLATION_MIN_LENGTH', 3),
        interpolation_method=config.get('INTERPOLATION_METHOD', 'linear'),
    )


def build_tta(config: dict) -> TestTimeAugmentation:
    return TestTimeAugmentation(
        scales=config.get('TTA_SCALES', [0.8, 1.0, 1.2]),
        flip=config.get('TTA_FLIP', True),
        fusion_method=config.get('TTA_FUSION', 'average'),
    )
