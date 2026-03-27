# FMTrack 追加修复对照报告（基于你上传的 ff232...zip）

> 目标：保持你这版代码的 **A–D 修复不变**，并把 **E2（importance 的 speed/occlusion 分桶）**做成“不会因为 batch/group 维度而失效、且统计更可信”的版本。

## 结论
- **A–D（单调中心频率、DC suppression、importance logits+tau、JS/ dot loss）**：在你上传的 `ff232...zip` 中已经保持正确实现，本次不改动。
- **E2（分桶证据链）**：你上传版的 `tools/lfd_importance_binning.py` 做了简化，但会在常见形状下不稳定：
  - 当 `importance` 以 `(B*G,T,N,K)` 存储且 `G>1` 时，原脚本会错误假设 `G=1`，导致与 `trajectory_boxes (B,G,T,N,4)` 对不上，进而 **跳过文件或统计失真**。
  - 原脚本对 `importance` 直接 `mean(T)`，会把 **masked 帧**也计入平均，导致对遮挡样本的 importance 统计产生系统性偏差。

本次修复：
1) 用 `trajectory_boxes` 的形状可靠推断 `B,G,T,N`，并把 `importance` 正确 reshape 到 `(B,G,T,N,K)`；必要时用 dump 的 `B/G/T/N` 作为 fallback。
2) 统计 per-target importance 时，只对 **可见帧**做平均（visible-weighted mean）。

---

## 1) `tools/lfd_importance_binning.py`

### 修改位置 1.1：读取 diag tensor -> numpy（保证不会因 device/grad 引发异常）

**修改前（ff232...zip）**：L166–L168
```python
imp_np = imp.float().numpy()
boxes_np = boxes.float().numpy()
masks_np = masks.bool().numpy()
```

**修改后（robust fixed）**：L166–L168
```python
imp_np = imp.float().cpu().numpy()
boxes_np = boxes.float().cpu().numpy()
masks_np = masks.bool().cpu().numpy()
```

---

### 修改位置 1.2：importance 形状处理（支持 (B*G,T,N,K) + G>1）

**修改前（ff232...zip）**：L170–L186
```python
# importance shape: (B*G, T, N, K) or (B, G, T, N, K)
if imp_np.ndim == 4:
    # (B*G, T, N, K) -> (B,G,T,N,K) by inferring G=1
    BG, T, N, K = imp_np.shape
    imp_np = imp_np.reshape(BG, 1, T, N, K)
if imp_np.ndim != 5:
    continue

B, G, T, N, K = imp_np.shape
if boxes_np.shape[:4] != (B, G, T, N):
    # attempt to reshape if possible
    try:
        boxes_np = boxes_np.reshape(B, G, T, N, 4)
        masks_np = masks_np.reshape(B, G, T, N)
    except Exception:
        continue
```

**修改后（robust fixed）**：L170–L195
```python
# boxes: (B,G,T,N,4) ; masks: (B,G,T,N)
if boxes_np.ndim != 5 or masks_np.ndim != 4:
    continue
B, G, T, N, _ = boxes_np.shape

# importance can be (B*G, T, N, K) OR (B,G,T,N,K)
if imp_np.ndim == 4:
    BG, Ti, Ni, K = imp_np.shape
    if BG != B * G or Ti != T or Ni != N:
        # try meta fallback if present
        Bb = int(diag.get('B', B))
        Gg = int(diag.get('G', G))
        Tt = int(diag.get('T', T))
        Nn = int(diag.get('N', N))
        if BG == Bb * Gg and Ti == Tt and Ni == Nn:
            B, G, T, N = Bb, Gg, Tt, Nn
        else:
            continue
    imp_np = imp_np.reshape(B, G, T, N, K)
elif imp_np.ndim == 5:
    if imp_np.shape[:4] != (B, G, T, N):
        continue
    K = imp_np.shape[-1]
else:
    continue
```

**为什么要这样改**：
- 只要 `trajectory_boxes` 存在，我们就能可靠得到 `B,G,T,N`，因此 reshape 可以做到 **严格一致**。
- 只有在 `B/G/T/N` 的 meta 被保存且 boxes 的形状不可用时，才用 meta fallback。

---

### 修改位置 1.3：per-target importance 统计（只对可见帧平均）

**修改前（ff232...zip）**：L190–L200
```python
# flatten targets: (B,G,N,K)
imp_tgt = imp_np.mean(axis=2)  # mean over T
imp_flat = imp_tgt.reshape(-1, K)
speed_flat = speed.reshape(-1)
occ_flat = occ.reshape(-1)
valid_flat = (~masks_np).any(axis=2).reshape(-1)

all_imp.append(imp_flat)
all_speed.append(speed_flat)
all_occ.append(occ_flat)
all_valid.append(valid_flat)
```

**修改后（robust fixed）**：L199–L212
```python
# Per-target mean importance over *visible* frames (avoid bias from masked frames)
valid = (~masks_np).astype(np.float32)  # (B,G,T,N)
denom = np.maximum(valid.sum(axis=2, keepdims=True), 1.0)  # (B,G,1,N)
imp_mean = (imp_np * valid[..., None]).sum(axis=2) / denom[..., None]  # (B,G,N,K)

imp_flat = imp_mean.reshape(-1, K)
speed_flat = speed.reshape(-1)
occ_flat = occ.reshape(-1)
vis_cnt_flat = valid.sum(axis=2).reshape(-1)

all_imp.append(imp_flat)
all_speed.append(speed_flat)
all_occ.append(occ_flat)
all_valid.append(vis_cnt_flat)
```

**为什么要这样改**：
- `mean(T)` 会把 padding/缺失/遮挡帧也算进去，导致遮挡样本的 importance 被“稀释”或“偏置”。
- visible-weighted mean 是审稿更买账的统计（与你的“occlusion-aware”叙事一致）。

---

### 修改位置 1.4：过滤无效目标（按可见帧计数过滤）

**修改前（ff232...zip）**：L205–L214
```python
src_all = np.concatenate(all_valid, axis=0)

# filter invalid targets
keep = src_all.astype(bool)
imp_all = imp_all[keep]
speed_all = speed_all[keep]
occ_all = occ_all[keep]
```

**修改后（robust fixed）**：L217–L226
```python
vis_cnt_all = np.concatenate(all_valid, axis=0)

# filter invalid targets (no visible frames)
keep = vis_cnt_all > 0.0
imp_all = imp_all[keep]
speed_all = speed_all[keep]
occ_all = occ_all[keep]
```

---

## 产物
- 修复后的代码包（仅对 E2 分桶脚本做了“鲁棒 reshape + 可见帧平均”的增强）：
  - `FMTrack_ff232_E2_binning_robust_fixed.zip`

## 使用方式（不变）
```bash
python tools/lfd_importance_binning.py \
  --diag_dir outputs/<exp>/lfd_diag \
  --out_dir outputs/<exp>/lfd_diag_bins \
  --pick latest \
  --box_format cxcywh
```

> 提醒：确保训练时 `SAVE_LFD_DIAGNOSTICS=True`，并且 dump 里包含 `trajectory_boxes/trajectory_masks/importance`。
