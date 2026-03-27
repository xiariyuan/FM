# Copyright (c) 2024. All Rights Reserved.
"""
FA-MOT 可视化工具

用于生成论文所需的各种可视化图表：
1. 频率滤波器可视化
2. 频带响应可视化
3. 融合权重可视化
4. Attention热力图
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from typing import List, Optional, Dict
import cv2


class FrequencyVisualization:
    """频率相关的可视化"""
    
    def __init__(self, save_dir: str = "./visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置matplotlib风格
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
    
    def visualize_learned_filters(
        self,
        model,
        save_name: str = "learned_filters.pdf"
    ):
        """
        可视化学习到的频率滤波器
        
        对比：固定Laplacian vs 学习到的滤波器
        """
        # 获取滤波器参数
        try:
            lfd = model.trajectory_modeling.freq_decomposition
            filters = lfd.filter_module._construct_filters()  # (num_bands, 1, kernel_size)
        except Exception:
            print("Cannot access frequency filters from model")
            return
        
        num_bands = filters.shape[0]
        fig, axes = plt.subplots(2, num_bands, figsize=(4*num_bands, 8))
        
        # 第一行：学习到的滤波器（空间域）
        for i in range(num_bands):
            ax = axes[0, i]
            filter_1d = filters[i, 0].detach().cpu().numpy()
            ax.plot(filter_1d, 'b-', linewidth=2)
            ax.fill_between(range(len(filter_1d)), filter_1d, alpha=0.3)
            ax.set_title(f'Band {i} (Learned)')
            ax.set_xlabel('Spatial Position')
            ax.set_ylabel('Filter Response')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 第二行：频率响应（FFT）
        for i in range(num_bands):
            ax = axes[1, i]
            filter_1d = filters[i, 0].detach().cpu().numpy()
            
            # 计算FFT
            fft = np.fft.fft(filter_1d, n=256)
            freq = np.fft.fftfreq(256)
            magnitude = np.abs(fft)[:128]
            freq = freq[:128]
            
            ax.plot(freq, magnitude, 'r-', linewidth=2)
            ax.fill_between(freq, magnitude, alpha=0.3, color='r')
            ax.set_title(f'Band {i} Frequency Response')
            ax.set_xlabel('Normalized Frequency')
            ax.set_ylabel('Magnitude')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.save_dir / save_name}")
    
    def visualize_band_responses(
        self,
        features: torch.Tensor,
        band_features: List[torch.Tensor],
        frame_idx: int = 0,
        obj_idx: int = 0,
        save_name: str = "band_responses.pdf"
    ):
        """
        可视化不同频带对同一目标的响应
        
        Args:
            features: 原始特征 (B, G, T, N, C)
            band_features: 各频带特征列表
            frame_idx: 展示哪一帧
            obj_idx: 展示哪个目标
        """
        num_bands = len(band_features)
        
        fig, axes = plt.subplots(1, num_bands + 1, figsize=(4*(num_bands+1), 4))
        
        # 原始特征
        orig_feat = features[0, 0, frame_idx, obj_idx].detach().cpu().numpy()
        axes[0].bar(range(min(64, len(orig_feat))), orig_feat[:64])
        axes[0].set_title('Original Feature')
        axes[0].set_xlabel('Feature Dim')
        
        # 各频带特征
        for i, band_feat in enumerate(band_features):
            feat = band_feat[0, 0, frame_idx, obj_idx].detach().cpu().numpy()
            axes[i+1].bar(range(min(64, len(feat))), feat[:64])
            axes[i+1].set_title(f'Band {i} ({"Low" if i==0 else "High" if i==num_bands-1 else "Mid"})')
            axes[i+1].set_xlabel('Feature Dim')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.save_dir / save_name}")
    
    def visualize_fusion_weights(
        self,
        fusion_weights: torch.Tensor,
        timestamps: Optional[List[int]] = None,
        occlusion_labels: Optional[torch.Tensor] = None,
        save_name: str = "fusion_weights.pdf"
    ):
        """
        可视化融合权重随时间的变化
        
        Args:
            fusion_weights: (T, 2) 或 (B, T, 2) - [freq_weight, standard_weight]
            timestamps: 时间戳
            occlusion_labels: 遮挡标签，用于验证相关性
        """
        if fusion_weights.dim() == 3:
            fusion_weights = fusion_weights[0]  # 取第一个batch
        
        weights = fusion_weights.detach().cpu().numpy()
        T = len(weights)
        
        if timestamps is None:
            timestamps = list(range(T))
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        ax.plot(timestamps, weights[:, 0], 'b-', linewidth=2, label='Frequency Branch')
        ax.plot(timestamps, weights[:, 1], 'r-', linewidth=2, label='Standard Branch')
        ax.fill_between(timestamps, weights[:, 0], alpha=0.3, color='b')
        ax.fill_between(timestamps, weights[:, 1], alpha=0.3, color='r')
        
        # 如果有遮挡标签，标注遮挡区域
        if occlusion_labels is not None:
            occ = occlusion_labels.cpu().numpy()
            for i in range(T):
                if occ[i] > 0.5:
                    ax.axvspan(i-0.5, i+0.5, alpha=0.2, color='gray')
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Weight')
        ax.set_title('Fusion Weights over Time')
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.save_dir / save_name}")
    
    def visualize_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        band_idx: int = 0,
        save_name: str = "attention_heatmap.pdf"
    ):
        """
        可视化特定频带的attention热力图
        
        Args:
            attention_weights: (num_heads, T, T) 或 (B, num_heads, T, T)
        """
        if attention_weights.dim() == 4:
            attention_weights = attention_weights[0]  # 取第一个batch
        
        # 平均所有heads
        attn = attention_weights.mean(dim=0).detach().cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        sns.heatmap(attn, ax=ax, cmap='viridis', square=True,
                   xticklabels=5, yticklabels=5)
        ax.set_xlabel('Key (Trajectory Frame)')
        ax.set_ylabel('Query (Current Frame)')
        ax.set_title(f'Attention Weights (Band {band_idx})')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.save_dir / save_name}")
    
    def visualize_occlusion_vs_weights(
        self,
        occlusion_scores: torch.Tensor,
        freq_weights: torch.Tensor,
        save_name: str = "occlusion_vs_weights.pdf"
    ):
        """
        可视化遮挡程度与频率权重的相关性
        
        验证核心假设：遮挡增加 → 低频权重增加
        """
        occ = occlusion_scores.flatten().detach().cpu().numpy()
        weights = freq_weights.flatten().detach().cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 散点图
        axes[0].scatter(occ, weights, alpha=0.5, s=10)
        
        # 添加趋势线
        z = np.polyfit(occ, weights, 1)
        p = np.poly1d(z)
        x_line = np.linspace(occ.min(), occ.max(), 100)
        axes[0].plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend (slope={z[0]:.3f})')
        
        axes[0].set_xlabel('Occlusion Score')
        axes[0].set_ylabel('Low-Frequency Weight')
        axes[0].set_title('Occlusion vs Frequency Weight')
        axes[0].legend()
        
        # 分组统计
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_means = []
        bin_stds = []
        bin_centers = []
        
        for i in range(len(bins)-1):
            mask = (occ >= bins[i]) & (occ < bins[i+1])
            if mask.sum() > 0:
                bin_means.append(weights[mask].mean())
                bin_stds.append(weights[mask].std())
                bin_centers.append((bins[i] + bins[i+1]) / 2)
        
        axes[1].bar(bin_centers, bin_means, width=0.15, yerr=bin_stds, capsize=5)
        axes[1].set_xlabel('Occlusion Level')
        axes[1].set_ylabel('Mean Low-Frequency Weight')
        axes[1].set_title('Mean Weight by Occlusion Level')
        axes[1].set_xticks(bin_centers)
        axes[1].set_xticklabels(['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.save_dir / save_name}")


class TrackingVisualization:
    """跟踪结果可视化"""
    
    def __init__(self, save_dir: str = "./visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 颜色映射
        self.colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    def draw_tracking_results(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        ids: np.ndarray,
        scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        在图像上绘制跟踪结果
        
        Args:
            image: (H, W, 3) BGR图像
            boxes: (N, 4) [x, y, w, h]
            ids: (N,) 目标ID
            scores: (N,) 置信度分数
        """
        result = image.copy()
        
        for i, (box, id_) in enumerate(zip(boxes, ids)):
            x, y, w, h = map(int, box)
            color = self.colors[int(id_) % 20][:3]
            color = tuple(int(c * 255) for c in color)
            
            # 画框
            cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
            
            # 画标签背景
            label = f"ID:{id_}"
            if scores is not None:
                label += f" {scores[i]:.2f}"
            
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(result, (x, y-label_h-baseline), 
                         (x+label_w, y), color, -1)
            cv2.putText(result, label, (x, y-baseline),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        return result
    
    def compare_tracking_results(
        self,
        image: np.ndarray,
        boxes_baseline: np.ndarray,
        ids_baseline: np.ndarray,
        boxes_ours: np.ndarray,
        ids_ours: np.ndarray,
        save_name: str = "tracking_comparison.png"
    ):
        """
        对比两种方法的跟踪结果
        """
        result_baseline = self.draw_tracking_results(image, boxes_baseline, ids_baseline)
        result_ours = self.draw_tracking_results(image, boxes_ours, ids_ours)
        
        # 拼接
        result = np.hstack([result_baseline, result_ours])
        
        # 添加标题
        h, w = result.shape[:2]
        result_with_title = np.zeros((h+50, w, 3), dtype=np.uint8)
        result_with_title[50:] = result
        
        cv2.putText(result_with_title, "Baseline", (w//4-50, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(result_with_title, "FA-MOT (Ours)", (3*w//4-80, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        cv2.imwrite(str(self.save_dir / save_name), result_with_title)
        print(f"Saved: {self.save_dir / save_name}")


def create_ablation_table(results: Dict[str, Dict[str, float]], save_path: str):
    """
    创建消融实验表格
    
    Args:
        results: {
            "Full": {"HOTA": 65.0, "IDF1": 70.0, ...},
            "w/o LFD": {...},
            ...
        }
    """
    import pandas as pd
    
    df = pd.DataFrame(results).T
    
    # 高亮最好的结果
    styled = df.style.highlight_max(axis=0, color='lightgreen')
    
    # 保存为LaTeX
    latex = df.to_latex(float_format="%.1f", bold_rows=True)
    
    with open(save_path, 'w') as f:
        f.write(latex)
    
    print(f"Saved ablation table to {save_path}")
    return df


# 使用示例
if __name__ == "__main__":
    # 创建可视化对象
    freq_vis = FrequencyVisualization(save_dir="./vis_output")
    
    # 示例：可视化融合权重
    fake_weights = torch.rand(30, 2)
    fake_weights = fake_weights / fake_weights.sum(dim=-1, keepdim=True)
    freq_vis.visualize_fusion_weights(fake_weights, save_name="demo_fusion_weights.pdf")
    
    print("Visualization tools ready!")
