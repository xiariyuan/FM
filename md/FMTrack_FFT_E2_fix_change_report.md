# FM-Track LFD 后续修复（FFT padding + E2 分桶证据链）修改对照（文件/行号/修改前后）

本文件基于你上传的「最新修改版」代码（src）与我在其基础上继续修复后的版本（patched）做逐段对照。

## 本次修复目标
- **Fix-1**：频域正交/重叠度量使用 **zero-padding FFT**（避免 kernel_size 太短导致谱分辨率不稳）
- **Fix-2**：训练时诊断 dump 额外写出 `trajectory_boxes/trajectory_masks`，支撑 **speed/occlusion 分桶**
- **Fix-3**：新增 `tools/lfd_importance_binning.py`，一键生成 **importance 在速度/遮挡桶下的统计与曲线**

## 文件：`models/motip/learnable_freq_decomposition.py`
- 覆盖目标：Fix-1
- Hunk 数量：1

### 变更块 1
- 修改前（src）：L318–L333  （共 16 行）
- 修改后（patched）：L318–L334  （共 17 行）

**修改前**
```python
        """
        if filters is None or filters.numel() == 0:
            return torch.tensor(0.0, device=self.output_norm.weight.device, dtype=self.output_norm.weight.dtype)

        filt = filters
        if filt.dim() == 3:
            filt = filt.squeeze(1)
        elif filt.dim() > 3:
            filt = filt.reshape(filt.shape[0], -1)

        K = filt.shape[0]
        if K <= 1:
            return torch.tensor(0.0, device=filt.device, dtype=filt.dtype)

        fft = torch.fft.rfft(filt, dim=-1)
        power_raw = fft.real.pow(2) + fft.imag.pow(2)  # (K, F)
```

**修改后**
```python
        """
        if filters is None or filters.numel() == 0:
            return torch.tensor(0.0, device=self.output_norm.weight.device, dtype=self.output_norm.weight.dtype)

        filt = filters
        if filt.dim() == 3:
            filt = filt.squeeze(1)
        elif filt.dim() > 3:
            filt = filt.reshape(filt.shape[0], -1)

        K = filt.shape[0]
        if K <= 1:
            return torch.tensor(0.0, device=filt.device, dtype=filt.dtype)

        n_fft = max(256, int(filt.shape[-1]) * 8)
        fft = torch.fft.rfft(filt, n=n_fft, dim=-1)
        power_raw = fft.real.pow(2) + fft.imag.pow(2)  # (K, F)
```

> 说明：`n_fft=max(256,k*8)` 能显著提升频谱分辨率，让 dot/cos 或 JS 度量更稳定、更可解释。

## 文件：`train.py`
- 覆盖目标：Fix-2
- Hunk 数量：1

### 变更块 1
- 修改前（src）：L690–L713  （共 24 行）
- 修改后（patched）：L690–L725  （共 36 行）

**修改前**
```python
            )
            freq_losses = seq_info.get("freq_losses", {}) if not only_detr else {}
            # Optional: dump LFD diagnostics tensors for plots (FFT response / overlap heatmap / importance stats)
            if (not only_detr) and save_lfd_diagnostics and (outputs_dir is not None) and accelerator.is_main_process:
                try:
                    if int(states.get("global_step", 0)) % max(int(lfd_diag_interval), 1) == 0:
                        diag_dir = os.path.join(outputs_dir, "lfd_diag")
                        os.makedirs(diag_dir, exist_ok=True)
                        freq_info = (seq_info.get("freq_info", {}) or {}).get("decomposition_info", {})
                        band_features = (freq_info.get("band_features", {}) or {})
                        diag = {
                            "epoch": int(epoch),
                            "global_step": int(states.get("global_step", 0)),
                        }
                        for k in ["filters_detached", "center_freqs", "sigmas", "importance", "importance_logits", "importance_tau"]:
                            v = band_features.get(k, None)
                            if isinstance(v, torch.Tensor):
                                diag[k] = v.detach().to("cpu")
                        # Also record current losses (cpu scalar)
                        ol = freq_info.get("ortho_loss", None)
                        if isinstance(ol, torch.Tensor):
                            diag["ortho_loss"] = ol.detach().to("cpu")
                        torch.save(diag, os.path.join(diag_dir, f"e{epoch}_g{states.get('global_step',0)}.pt"))
                except Exception:
```

**修改后**
```python
            )
            freq_losses = seq_info.get("freq_losses", {}) if not only_detr else {}
            # Optional: dump LFD diagnostics tensors for plots (FFT response / overlap heatmap / importance stats)
            if (not only_detr) and save_lfd_diagnostics and (outputs_dir is not None) and accelerator.is_main_process:
                try:
                    if int(states.get("global_step", 0)) % max(int(lfd_diag_interval), 1) == 0:
                        diag_dir = os.path.join(outputs_dir, "lfd_diag")
                        os.makedirs(diag_dir, exist_ok=True)
                        freq_info = (seq_info.get("freq_info", {}) or {}).get("decomposition_info", {})
                        band_features = (freq_info.get("band_features", {}) or {})
                        diag = {
                            "epoch": int(epoch),
                            "global_step": int(states.get("global_step", 0)),
                        }
                        # E2 evidence: dump trajectory boxes/masks for speed/occlusion binning
                        tb = seq_info.get("trajectory_boxes", None)
                        tm = seq_info.get("trajectory_masks", None)
                        if isinstance(tb, torch.Tensor):
                            diag["trajectory_boxes"] = tb.detach().to("cpu")
                            try:
                                B_, G_, T_, N_, _ = tb.shape
                                diag["B"], diag["G"], diag["T"], diag["N"] = int(B_), int(G_), int(T_), int(N_)
                            except Exception:
                                pass
                        if isinstance(tm, torch.Tensor):
                            diag["trajectory_masks"] = tm.detach().to("cpu")

                        for k in ["filters_detached", "center_freqs", "sigmas", "importance", "importance_logits", "importance_tau"]:
                            v = band_features.get(k, None)
                            if isinstance(v, torch.Tensor):
                                diag[k] = v.detach().to("cpu")
                        # Also record current losses (cpu scalar)
                        ol = freq_info.get("ortho_loss", None)
                        if isinstance(ol, torch.Tensor):
                            diag["ortho_loss"] = ol.detach().to("cpu")
                        torch.save(diag, os.path.join(diag_dir, f"e{epoch}_g{states.get('global_step',0)}.pt"))
```

> 说明：这让诊断 pt 文件包含 per-target 的 box/mask，可直接做 **速度/遮挡分桶**，形成审稿人能接受的“证据链”。

## 新增文件：`tools/lfd_importance_binning.py`
- 覆盖目标：Fix-3
- 行数：339

### 变更块 1
- 修改前（src）：（不存在）
- 修改后（patched）：新增分桶分析脚本

**修改后（片段 1：脚本用途 & 入口参数）**
```python
"""LFD importance binning diagnostics (speed / occlusion buckets).

This script is part of the "evidence chain" for top-tier review:
- Show that band-importance is *not* a static global gate.
- Demonstrate correlation between band usage and MOT phenomena (speed / occlusion).

It expects a .pt dump produced by train.py when SAVE_LFD_DIAGNOSTICS=True.
To enable speed/occlusion binning, the dump should contain:
  - trajectory_boxes: (B, G, T, N, 4)
  - trajectory_masks: (B, G, T, N)  (bool, True=masked)
  - importance:        (B*G, T, N, K) or (B, G, T, N, K)

Usage:
  python tools/lfd_importance_binning.py \
      --diag_dir outputs/exp_xxx/lfd_diag \
      --out_dir outputs/exp_xxx/lfd_diag_bins \
      --pick latest

Notes:
- Boxes are assumed to be DETR-style normalized (cx, cy, w, h) by default.
- "Speed" is computed as mean center displacement per frame (normalized units).
"""

from __future__ import annotations
```

**修改后（片段 2：速度/遮挡计算的核心实现）**
```python
def _centers_from_boxes(boxes: np.ndarray, box_format: str = 'cxcywh') -> Tuple[np.ndarray, np.ndarray]:
    # boxes: (..., 4)
    box_format = box_format.lower()
    if box_format in ('cxcywh', 'cxywh', 'detr'):
        cx, cy = boxes[..., 0], boxes[..., 1]
        return cx, cy
    if box_format in ('xyxy',):
        x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5
    if box_format in ('xywh',):
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        return x + 0.5 * w, y + 0.5 * h
    raise ValueError(f'Unknown box_format: {box_format}')


def _compute_speed(boxes: np.ndarray, masks: np.ndarray, box_format: str) -> np.ndarray:
    """Mean per-target speed over time.

    Returns:
      speed: (B, G, N) in normalized units/frame.
    """
    # boxes: (B,G,T,N,4)
    # masks: (B,G,T,N) bool, True=masked
    valid = ~masks
    cx, cy = _centers_from_boxes(boxes, box_format=box_format)  # (B,G,T,N)

    # consecutive pairs validity
    valid_pair = valid[:, :, 1:, :] & valid[:, :, :-1, :]
    dx = cx[:, :, 1:, :] - cx[:, :, :-1, :]
    dy = cy[:, :, 1:, :] - cy[:, :, :-1, :]
    dist = np.sqrt(dx * dx + dy * dy)

    # masked pairs -> 0, and count them out
    dist = dist * valid_pair.astype(dist.dtype)
    denom = valid_pair.sum(axis=2)  # (B,G,N)
    denom = np.maximum(denom, 1)
    speed = dist.sum(axis=2) / denom
    return speed


def _compute_occ_ratio(masks: np.ndarray) -> np.ndarray:
    """Per-target occlusion ratio over time (masked fraction)."""
    return masks.astype(np.float32).mean(axis=2)  # (B,G,N)

```

**修改后（片段 3：输出文件与图）**
```python
    imp_all = np.concatenate(all_imp, axis=0)
    speed_all = np.concatenate(all_speed, axis=0)
    occ_all = np.concatenate(all_occ, axis=0)
    src_all = np.concatenate(all_valid, axis=0)

    K = imp_all.shape[-1]

    # ---------- speed bins ----------
    speed_bins = _parse_bins(args.speed_bins)
    if speed_bins is None:
        speed_bins = _quantile_bins(speed_all, n_bins=max(2, int(args.speed_nbins)))
    speed_bins = sorted(speed_bins)
    speed_bins[-1] = max(speed_bins[-1], float(speed_all.max()) + 1e-6)

    speed_bucket = np.digitize(speed_all, speed_bins[1:-1], right=False)  # 0..nb-1
    speed_nb = len(speed_bins) - 1
    speed_means, speed_counts = _bucket_means(imp_all, speed_bucket, speed_nb)

    # ---------- occlusion bins ----------
    occ_bins = _parse_bins(args.occ_bins) or [0.0, 0.1, 0.3, 0.6, 1.0]
    occ_bins = sorted(occ_bins)
    occ_bins[0] = min(occ_bins[0], 0.0)
    occ_bins[-1] = max(occ_bins[-1], 1.0)
    occ_bucket = np.digitize(occ_all, occ_bins[1:-1], right=False)
    occ_nb = len(occ_bins) - 1
    occ_means, occ_counts = _bucket_means(imp_all, occ_bucket, occ_nb)

    # ---------- entropy (optional sanity) ----------
    eps = 1e-12
    ent = -(imp_all * np.log(np.clip(imp_all, eps, 1.0))).sum(axis=-1)
    ent_speed = np.zeros((speed_nb,), dtype=np.float64)
    ent_occ = np.zeros((occ_nb,), dtype=np.float64)
    for b in range(speed_nb):
        idx = speed_bucket == b
        ent_speed[b] = float(ent[idx].mean()) if idx.any() else 0.0
    for b in range(occ_nb):
        idx = occ_bucket == b
        ent_occ[b] = float(ent[idx].mean()) if idx.any() else 0.0

    # ---------- save arrays ----------
    np.save(out_dir / 'speed_bins.npy', np.array(speed_bins, dtype=np.float64))
    np.save(out_dir / 'speed_means.npy', speed_means)
    np.save(out_dir / 'speed_counts.npy', speed_counts)

    np.save(out_dir / 'occ_bins.npy', np.array(occ_bins, dtype=np.float64))
    np.save(out_dir / 'occ_means.npy', occ_means)
    np.save(out_dir / 'occ_counts.npy', occ_counts)

    # ---------- write csv summary ----------
    def _fmt_edges(edges: List[float], i: int) -> str:
        return f'[{edges[i]:.6f}, {edges[i+1]:.6f})'

    lines = []
    header = ['bucket_type', 'bucket', 'count', 'entropy'] + [f'band_{i}' for i in range(K)]
    lines.append(','.join(header))

    for i in range(speed_nb):
        row = ['speed', _fmt_edges(speed_bins, i), str(int(speed_counts[i])), f'{ent_speed[i]:.6f}']
        row += [f'{speed_means[i, j]:.6f}' for j in range(K)]
        lines.append(','.join(row))

    for i in range(occ_nb):
        row = ['occlusion', _fmt_edges(occ_bins, i), str(int(occ_counts[i])), f'{ent_occ[i]:.6f}']
        row += [f'{occ_means[i, j]:.6f}' for j in range(K)]
        lines.append(','.join(row))

    (out_dir / 'importance_binning_summary.csv').write_text('\n'.join(lines), encoding='utf-8')

    # ---------- plots ----------
    import matplotlib.pyplot as plt

    # speed plot
    plt.figure()
    for k in range(K):
        plt.plot(np.arange(speed_nb), speed_means[:, k], marker='o', label=f'band_{k}')
    plt.xticks(np.arange(speed_nb), [f'{i}' for i in range(speed_nb)])
    plt.xlabel('Speed bucket index')
    plt.ylabel('Mean importance')
    plt.title('Band importance vs speed buckets')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'importance_vs_speed.png', dpi=200)
    plt.close()

    # occlusion plot
    plt.figure()
    for k in range(K):
        plt.plot(np.arange(occ_nb), occ_means[:, k], marker='o', label=f'band_{k}')
    plt.xticks(np.arange(occ_nb), [f'{i}' for i in range(occ_nb)])
    plt.xlabel('Occlusion bucket index')
    plt.ylabel('Mean importance')
    plt.title('Band importance vs occlusion buckets')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'importance_vs_occlusion.png', dpi=200)
    plt.close()

    # meta
    meta = [
        f'files: {len(picked)}',
        f'K(num_bands): {K}',
        f'speed_bins: {speed_bins}',
        f'occ_bins: {occ_bins}',
        f'total_targets: {imp_all.shape[0]}',
    ]
    (out_dir / 'meta.txt').write_text('\n'.join(meta), encoding='utf-8')

    print(f'[OK] Saved binning results to: {out_dir}')


if __name__ == '__main__':
    main()
```

### 使用方式（建议写进你的实验记录/论文附录）
1) 训练时打开诊断 dump：
```yaml
# config/*.yaml
SAVE_LFD_DIAGNOSTICS: true
LFD_DIAG_INTERVAL: 500
```

2) 训练结束后画图：
```bash
python tools/plot_lfd_diagnostics.py --diag_dir <outputs>/lfd_diag --out_dir <outputs>/lfd_diag_plots
python tools/lfd_importance_binning.py --diag_dir <outputs>/lfd_diag --out_dir <outputs>/lfd_diag_bins --pick all --max_files 20
```

输出（<out_dir>）：
- `importance_binning_summary.csv`：每个 bucket 的均值 importance/entropy
- `importance_vs_speed.png` / `importance_vs_occlusion.png`：曲线图
- `speed_bins.npy` / `occ_bins.npy`：桶边界（可复现）