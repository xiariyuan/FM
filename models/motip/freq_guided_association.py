# Copyright (c) 2024. All Rights Reserved.
"""
Frequency-Guided Association (FGA) Module

核心创新点:
1. 遮挡感知的频率权重调整
   - 遮挡时：增加低频权重（更稳定）
   - 清晰时：增加高频权重（更精确）
2. 多频带ID解码
   - 每个频带独立预测ID
   - 通过投票或加权融合得到最终ID
3. 频率一致性约束
   - 不同频带的ID预测应该一致
   - 不一致时降低置信度

理论依据:
- 遮挡会主要破坏高频信息，但低频结构相对保持
- 运动模糊同样影响高频多于低频
- 通过动态调整频率权重，可以在困难场景下保持鲁棒性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from einops import rearrange


class OcclusionEstimator(nn.Module):
    """
    遮挡程度估计器
    
    基于特征分析估计目标的遮挡程度，用于后续的频率权重调整
    
    估计方法:
    1. 时序一致性：特征变化剧烈可能表示遮挡
    2. 空间完整性：基于检测置信度
    3. 频率分布异常：遮挡会改变频率分布
    """
    
    def __init__(
        self,
        dim: int,
        num_bands: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.num_bands = num_bands
        
        # 时序变化编码器
        self.temporal_diff_encoder = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 4),
        )
        
        # 频率分布编码器
        self.freq_dist_encoder = nn.Sequential(
            nn.Linear(num_bands, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim // 4),
        )
        
        # 遮挡预测头
        self.occlusion_head = nn.Sequential(
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        features: torch.Tensor,  # (B, T, N, C)
        band_energies: Optional[torch.Tensor] = None,  # (B, T, N, num_bands)
    ) -> torch.Tensor:
        """
        Args:
            features: 时序特征
            band_energies: 各频带的能量

        Returns:
            occlusion_scores: (B, T, N) 遮挡程度 [0, 1]
        """
        B, T, N, C = features.shape

        # 边界检查：空目标保护
        if N == 0:
            return torch.zeros((B, T, 0), device=features.device, dtype=features.dtype)

        # 计算时序差分
        if T > 1:
            temporal_diff = features[:, 1:] - features[:, :-1]
            temporal_diff = F.pad(temporal_diff, (0, 0, 0, 0, 1, 0))  # 补齐第一帧
        else:
            temporal_diff = torch.zeros_like(features)
        
        temporal_encoding = self.temporal_diff_encoder(temporal_diff)  # (B, T, N, C//4)
        
        # 频率分布编码
        if band_energies is not None:
            freq_encoding = self.freq_dist_encoder(band_energies)  # (B, T, N, C//4)
        else:
            freq_encoding = torch.zeros(
                B, T, N, self.dim // 4, device=features.device, dtype=features.dtype
            )
        
        # 拼接并预测遮挡
        combined = torch.cat([temporal_encoding, freq_encoding], dim=-1)  # (B, T, N, C//2)
        occlusion_scores = self.occlusion_head(combined).squeeze(-1)  # (B, T, N)
        
        return occlusion_scores


class FrequencyWeightPredictor(nn.Module):
    """
    频率权重预测器
    
    根据遮挡程度和内容特征，预测各频带的权重
    """
    
    def __init__(
        self,
        dim: int,
        num_bands: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.num_bands = num_bands
        
        # 内容编码器
        self.content_encoder = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 4),
        )
        
        # 遮挡条件编码
        self.occlusion_encoder = nn.Sequential(
            nn.Linear(1, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim // 4),
        )
        
        # 权重预测
        self.weight_predictor = nn.Sequential(
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, num_bands),
        )
        
        # 先验权重（无条件时的默认权重）
        self.prior_weights = nn.Parameter(torch.ones(num_bands) / num_bands)
        
        # 权重调制强度
        self.modulation_strength = nn.Parameter(torch.tensor(1.0))
        
    def forward(
        self,
        features: torch.Tensor,  # (B, T, N, C)
        occlusion_scores: torch.Tensor,  # (B, T, N)
    ) -> torch.Tensor:
        """
        Returns:
            weights: (B, T, N, num_bands) 频率权重，和为1
        """
        B, T, N, C = features.shape

        # 边界检查：空目标保护
        if N == 0:
            return torch.zeros((B, T, 0, self.num_bands), device=features.device, dtype=features.dtype)

        # 内容编码
        content = self.content_encoder(features)  # (B, T, N, C//4)
        
        # 遮挡条件编码
        occlusion = self.occlusion_encoder(occlusion_scores.unsqueeze(-1))  # (B, T, N, C//4)
        
        # 拼接并预测权重偏移
        combined = torch.cat([content, occlusion], dim=-1)  # (B, T, N, C//2)
        weight_logits = self.weight_predictor(combined)  # (B, T, N, num_bands)
        
        # 应用先验并归一化
        modulation = torch.sigmoid(self.modulation_strength)
        weights = F.softmax(
            self.prior_weights + modulation * weight_logits, 
            dim=-1
        )
        
        return weights


class MultiBandIDDecoder(nn.Module):
    """
    多频带ID解码器
    
    每个频带独立预测ID，然后通过加权融合得到最终预测
    这提供了:
    1. 更鲁棒的预测（多视角投票）
    2. 频率一致性可以作为置信度指标
    """
    
    def __init__(
        self,
        feature_dim: int,
        id_dim: int,
        num_bands: int = 4,
        num_id_vocabulary: int = 50,
    ):
        super().__init__()
        # NOTE: keep `feature_dim` in the signature for backward compatibility with older configs,
        # but do not store it since this decoder only operates on `id_dim` embeddings.
        self.id_dim = id_dim
        self.num_bands = num_bands
        self.num_id_vocabulary = num_id_vocabulary

        # 各频带的ID预测头
        self.band_embed_to_word = nn.ModuleList([
            nn.Linear(id_dim, num_id_vocabulary + 1, bias=False)
            for _ in range(num_bands)
        ])
        
        # 频带融合网络
        self.band_fusion = nn.Sequential(
            nn.Linear(num_bands * (num_id_vocabulary + 1), (num_id_vocabulary + 1) * 2),
            nn.ReLU(),
            nn.Linear((num_id_vocabulary + 1) * 2, num_id_vocabulary + 1),
        )
        
        # 一致性评估网络
        self.consistency_evaluator = nn.Sequential(
            nn.Linear(num_bands * (num_id_vocabulary + 1), 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Learnable fusion between (1) simple weighted logits and (2) nonlinear fused logits.
        # Using logits -> softmax keeps weights positive and summing to 1.
        self.fusion_logits = nn.Parameter(torch.zeros(2))
        
    def forward(
        self,
        band_id_embeds: List[torch.Tensor],  # 各频带的ID嵌入
        freq_weights: torch.Tensor,  # (B, T, N, num_bands) 频率权重
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            band_id_embeds: 各频带的ID嵌入，每个 (B, T, N, id_dim)
            freq_weights: 频率权重

        Returns:
            final_logits: (B, T, N, num_vocabulary+1) 融合后的ID logits
            info: 包含各频带预测和一致性分数
        """
        B, T, N, _ = band_id_embeds[0].shape
        vocab_size = self.num_id_vocabulary + 1
        device = band_id_embeds[0].device
        dtype = band_id_embeds[0].dtype

        # 边界检查：空目标保护
        if N == 0:
            empty_logits = torch.zeros((B, T, 0, vocab_size), device=device, dtype=dtype)
            info = {
                'band_logits': [empty_logits for _ in range(self.num_bands)],
                'consistency': torch.zeros((B, T, 0), device=device, dtype=dtype),
                'kl_consistency': torch.zeros((B, T, 0), device=device, dtype=dtype),
                'freq_weights': freq_weights,
            }
            return empty_logits, info

        # 各频带独立预测
        band_logits = []
        for i, embed in enumerate(band_id_embeds):
            logits = self.band_embed_to_word[i](embed)  # (B, T, N, vocab+1)
            band_logits.append(logits)
        
        # 堆叠各频带logits
        stacked_logits = torch.stack(band_logits, dim=-1)  # (B, T, N, vocab+1, num_bands)
        
        # 加权融合
        # 方法1: 简单加权平均
        weights = freq_weights.unsqueeze(-2)  # (B, T, N, 1, num_bands)
        weighted_logits = (stacked_logits * weights).sum(dim=-1)  # (B, T, N, vocab+1)
        
        # 方法2: 通过网络融合（学习非线性组合）
        concat_logits = rearrange(stacked_logits, 'b t n v k -> b t n (v k)')
        fused_logits = self.band_fusion(concat_logits)  # (B, T, N, vocab+1)
        
        # 结合两种方法
        fusion_w = torch.softmax(self.fusion_logits, dim=0)  # (2,)
        final_logits = fusion_w[0] * weighted_logits + fusion_w[1] * fused_logits
        
        # 计算一致性分数（各频带预测的一致程度）
        consistency = self.consistency_evaluator(concat_logits).squeeze(-1)  # (B, T, N)
        
        # 也可以用KL散度度量一致性
        band_probs = [F.softmax(l, dim=-1) for l in band_logits]
        avg_prob = sum(band_probs) / len(band_probs)
        avg_prob = torch.clamp(avg_prob, min=1e-6, max=1.0)
        kl_divs = []
        for prob in band_probs:
            # Clamp prob to prevent -inf/NaN in log
            prob_clamped = torch.clamp(prob, min=1e-6, max=1.0)
            # We want KL(prob || avg_prob) (each band should be close to the mean distribution).
            # Compute explicitly to avoid any ambiguity about kl_div conventions.
            kl = (prob_clamped * (prob_clamped.log() - avg_prob.log())).sum(dim=-1)
            kl_divs.append(kl)
        kl_consistency = 1.0 / (1.0 + sum(kl_divs) / len(kl_divs))  # 转换为一致性分数
        
        info = {
            'band_logits': band_logits,
            'consistency': consistency,
            'kl_consistency': kl_consistency,
            'freq_weights': freq_weights,
            'fusion_weights': fusion_w.detach(),
        }
        
        return final_logits, info


class FrequencyGuidedAssociation(nn.Module):
    """
    完整的频率引导关联模块
    
    整合:
    1. 遮挡估计
    2. 频率权重预测
    3. 多频带ID解码
    4. 频率一致性约束
    """
    
    def __init__(
        self,
        feature_dim: int,
        id_dim: int,
        num_bands: int = 4,
        num_id_vocabulary: int = 50,
        use_occlusion_aware: bool = True,
        consistency_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.id_dim = id_dim
        self.num_bands = num_bands
        self.num_id_vocabulary = num_id_vocabulary
        self.use_occlusion_aware = use_occlusion_aware
        
        # 遮挡估计器
        if use_occlusion_aware:
            self.occlusion_estimator = OcclusionEstimator(
                dim=feature_dim,
                num_bands=num_bands,
            )
        
        # 频率权重预测器
        self.freq_weight_predictor = FrequencyWeightPredictor(
            dim=feature_dim,
            num_bands=num_bands,
        )
        
        # 多频带ID解码器
        self.multiband_decoder = MultiBandIDDecoder(
            feature_dim=feature_dim,
            id_dim=id_dim,
            num_bands=num_bands,
            num_id_vocabulary=num_id_vocabulary,
        )
        
        # 各频带的特征投影（从频率分解输出到ID空间）
        self.band_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, id_dim),
            )
            for _ in range(num_bands)
        ])
        
        # 一致性损失权重
        # Keep this as a fixed hyper-parameter (do NOT learn it), otherwise the model can
        # trivially drive it to 0 and silently disable the loss.
        self.consistency_loss_weight = float(consistency_loss_weight)
        
    def compute_consistency_loss(self, info: Dict) -> torch.Tensor:
        """
        计算频率一致性损失
        
        鼓励不同频带的ID预测一致
        """
        band_logits = info['band_logits']
        
        # 计算所有频带对之间的KL散度
        total_kl = 0
        count = 0
        for i in range(len(band_logits)):
            for j in range(i + 1, len(band_logits)):
                prob_i = torch.clamp(F.softmax(band_logits[i], dim=-1), min=1e-6, max=1.0)
                prob_j = torch.clamp(F.softmax(band_logits[j], dim=-1), min=1e-6, max=1.0)
                
                # 对称KL散度
                # Match PyTorch's `reduction="batchmean"` scaling: sum over all non-batch dims / batch_size.
                batch = max(int(prob_i.shape[0]), 1)
                kl_ij = (prob_i * (prob_i.log() - prob_j.log())).sum(dim=-1).sum() / batch  # KL(prob_i || prob_j)
                kl_ji = (prob_j * (prob_j.log() - prob_i.log())).sum(dim=-1).sum() / batch  # KL(prob_j || prob_i)
                
                total_kl = total_kl + kl_ij + kl_ji
                count += 2
        
        if count > 0:
            consistency_loss = total_kl / count
        else:
            consistency_loss = torch.tensor(0.0, device=band_logits[0].device)
        
        return self.consistency_loss_weight * consistency_loss
    
    def forward(
        self,
        band_features: List[torch.Tensor],  # 各频带特征，每个 (B, G, T, N, C)
        trajectory_masks: Optional[torch.Tensor] = None,  # (B, G, T, N)
        band_energies: Optional[torch.Tensor] = None,  # (B, G, T, N, num_bands)
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            band_features: 频率分解后的各频带特征
            trajectory_masks: 轨迹mask
            band_energies: 各频带能量（用于遮挡估计）

        Returns:
            id_logits: (B, G, T, N, num_vocabulary+1)
            info: 中间信息字典
        """
        B, G, T, N, C = band_features[0].shape
        vocab_size = self.num_id_vocabulary + 1
        device = band_features[0].device
        dtype = band_features[0].dtype

        # 边界检查：空目标保护
        if N == 0:
            empty_logits = torch.zeros((B, G, T, 0, vocab_size), device=device, dtype=dtype)
            info = {
                'occlusion_scores': torch.zeros((B, G, T, 0), device=device, dtype=dtype),
                'freq_weights': torch.zeros((B, G, T, 0, self.num_bands), device=device, dtype=dtype),
                'band_id_embeds': [torch.zeros((B, G, T, 0, self.id_dim), device=device, dtype=dtype) for _ in range(self.num_bands)],
                'band_logits': [empty_logits for _ in range(self.num_bands)],
                'consistency': torch.zeros((B, G, T, 0), device=device, dtype=dtype),
                'kl_consistency': torch.zeros((B, G, T, 0), device=device, dtype=dtype),
            }
            return empty_logits, info

        # 展平B, G维度
        # NOTE: The effective batch dimension becomes (B*G) after flattening.
        band_features_flat = [f.reshape(B * G, T, N, C) for f in band_features]
        if trajectory_masks is not None:
            masks_flat = trajectory_masks.reshape(B * G, T, N)
        else:
            masks_flat = None
        if band_energies is not None:
            energies_flat = band_energies.reshape(B * G, T, N, -1)
        else:
            energies_flat = None

        # If energies are not provided, derive a simple normalized energy distribution from band features.
        # This activates the freq_dist_encoder path in OcclusionEstimator and avoids dead parameters.
        if energies_flat is None:
            try:
                # band_features_flat: list of (B*G, T, N, C) -> energies: (B*G, T, N, K)
                energies = torch.stack([f.pow(2).mean(dim=-1) for f in band_features_flat], dim=-1)
                energies = energies / (energies.sum(dim=-1, keepdim=True) + 1e-6)
                energies_flat = energies
            except Exception:
                energies_flat = None
        
        # 1. 估计遮挡程度
        # 使用第一个频带（通常是整体特征）来估计
        main_features = band_features_flat[0]
        
        if self.use_occlusion_aware:
            occlusion_scores = self.occlusion_estimator(
                main_features, energies_flat
            )  # (B*G, T, N)
        else:
            occlusion_scores = torch.zeros(
                B * G, T, N, device=main_features.device, dtype=main_features.dtype
            )
        
        # 2. 预测频率权重
        freq_weights = self.freq_weight_predictor(
            main_features, occlusion_scores
        )  # (B*G, T, N, num_bands)
        
        # 3. 各频带投影到ID空间
        band_id_embeds = []
        for i, feat in enumerate(band_features_flat):
            id_embed = self.band_projectors[i](feat)  # (B*G, T, N, id_dim)
            band_id_embeds.append(id_embed)
        
        # 4. 多频带ID解码
        id_logits, decoder_info = self.multiband_decoder(
            band_id_embeds, freq_weights
        )
        
        # 恢复形状
        id_logits = id_logits.reshape(B, G, T, N, -1)
        
        # 汇总信息
        info = {
            'occlusion_scores': occlusion_scores.reshape(B, G, T, N),
            'freq_weights': freq_weights.reshape(B, G, T, N, -1),
            'band_id_embeds': [e.reshape(B, G, T, N, -1) for e in band_id_embeds],
            **decoder_info,
        }
        
        # 计算一致性损失
        if self.training:
            info['consistency_loss'] = self.compute_consistency_loss(decoder_info)
        
        return id_logits, info
