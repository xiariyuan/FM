# Copyright (c) 2024-2026. All Rights Reserved.
# Frequency Analysis and Visualization Tools for FM-Track

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
import os


class FrequencyVisualizer:
    def __init__(self, save_dir='visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'axes.labelsize': 14,
            'figure.figsize': (8, 6),
            'figure.dpi': 150,
            'savefig.dpi': 300,
        })
        self.colors = ['#0077BB', '#33BBEE', '#009988', '#EE7733', 
                       '#CC3311', '#EE3377', '#BBBBBB']
        
    def visualize_frequency_response(self, filter_weights, save_name='frequency_response'):
        if isinstance(filter_weights, torch.Tensor):
            filter_weights = filter_weights.detach().cpu().numpy()
        K, kernel_size = filter_weights.shape
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax1, ax2 = axes
        x = np.arange(kernel_size)
        for i in range(K):
            ax1.plot(x, filter_weights[i], color=self.colors[i % len(self.colors)], 
                    linewidth=2, label=f'Band {i}')
        ax1.set_xlabel('Kernel Position')
        ax1.set_ylabel('Weight')
        ax1.set_title('(a) Spatial Domain Filters')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        freqs = np.fft.fftfreq(kernel_size * 4)[:kernel_size * 2]
        for i in range(K):
            padded = np.zeros(kernel_size * 4)
            padded[:kernel_size] = filter_weights[i]
            fft_resp = np.abs(np.fft.fft(padded))[:kernel_size * 2]
            fft_resp = fft_resp / (fft_resp.max() + 1e-8)
            ax2.plot(freqs, fft_resp, color=self.colors[i % len(self.colors)],
                    linewidth=2, label=f'Band {i}')
        ax2.set_xlabel('Normalized Frequency')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('(b) Frequency Domain Response')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name + '.pdf')
        plt.savefig(save_path, format='pdf')
        plt.savefig(save_path.replace('.pdf', '.png'))
        plt.close()
        return save_path

    def visualize_orthogonality(self, filter_responses, save_name='orthogonality'):
        if isinstance(filter_responses, torch.Tensor):
            filter_responses = filter_responses.detach().cpu().numpy()
        K = filter_responses.shape[0]
        resp_norm = filter_responses / (np.linalg.norm(filter_responses, axis=1, keepdims=True) + 1e-8)
        corr = np.dot(resp_norm, resp_norm.T)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(['Band ' + str(i) for i in range(K)])
        ax.set_yticklabels(['Band ' + str(i) for i in range(K)])
        plt.colorbar(im, ax=ax, label='Correlation')
        for i in range(K):
            for j in range(K):
                val = corr[i][j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, '%.2f' % val, ha='center', va='center', color=color)
        plt.title('Frequency Band Correlation Matrix')
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name + '.pdf')
        plt.savefig(save_path, format='pdf')
        plt.close()
        return save_path

    def visualize_occlusion_weights(self, occ_levels, freq_weights, save_name='occlusion_weights'):
        if isinstance(freq_weights, torch.Tensor):
            freq_weights = freq_weights.detach().cpu().numpy()
        num_levels, K = freq_weights.shape
        fig, ax = plt.subplots(figsize=(8, 5))
        for i in range(K):
            label = 'Low-freq' if i == 0 else ('High-freq' if i == K-1 else 'Mid-freq')
            ax.plot(occ_levels, freq_weights[:, i], color=self.colors[i % len(self.colors)],
                   linewidth=2.5, marker='o', markersize=8, label=label)
        ax.set_xlabel('Occlusion Level')
        ax.set_ylabel('Frequency Weight')
        ax.set_title('Adaptive Frequency Weights under Occlusion')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name + '.pdf')
        plt.savefig(save_path, format='pdf')
        plt.close()
        return save_path


def test_visualization():
    vis = FrequencyVisualizer('test_vis')
    K, kernel_size = 4, 7
    filters = np.random.randn(K, kernel_size)
    vis.visualize_frequency_response(torch.tensor(filters))
    vis.visualize_orthogonality(torch.tensor(filters))
    print('Test completed. Check test_vis/')

if __name__ == '__main__':
    test_visualization()
