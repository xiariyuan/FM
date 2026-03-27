# FM-Track LFD A–E 修改对照（文件/行号/修改前后）

本文件基于你上传的最新代码包（src）与我按 A–E 目标打完补丁后的版本（patched）做逐段对照。

## A–E 目标对应
- **A**：单调中心频率参数化（cumsum softplus + 稳定归一化到 (0, 0.5)）
- **B**：DC suppression（i>0 band 做零均值）
- **C**：importance 输出 logits + 温度 softmax（tau 在 train.py 控制）
- **D**：JS/KL(这里实现 JS) 谱距离（可选 metric），并修正 penalty 范围
- **E**：证据链落地：训练时 dump 诊断张量 + 提供画图脚本（FFT/overlap/importance）

## 文件：`models/motip/learnable_freq_decomposition.py`
- 覆盖目标：A/B/C/D + MultiScale wiring
- Hunk 数量：12

### 变更块 1
- 修改前（src）：L51–L58  （共 8 行）
- 修改后（patched）：L51–L60  （共 10 行）

**修改前**
```python
        # 频率调制参数（控制每个滤波器的中心频率和带宽）
        # center_freq: [0, 1] 表示从低频到高频
        # bandwidth: 控制频带宽度
        self.center_freqs = nn.Parameter(
            torch.linspace(0.1, 0.9, num_bands)
        )
        self.bandwidths = nn.Parameter(
            torch.ones(num_bands) * 0.2
```

**修改后**
```python
        # 频率调制参数（控制每个滤波器的中心频率和带宽）
        # center_freq: [0, 1] 表示从低频到高频
        # bandwidth: 控制频带宽度
        # 单调中心频率参数化：raw delta 参数（softplus 保证 > 0）
        # 通过 cumsum(softplus(delta)) 构造单调递增的中心频率，避免频带交换/坍塌
        self.freq_deltas = nn.Parameter(
            torch.ones(num_bands) * 0.5
        )
        self.bandwidths = nn.Parameter(
            torch.ones(num_bands) * 0.2
```

### 变更块 2
- 修改前（src）：L68–L83  （共 16 行）
- 修改后（patched）：L70–L87  （共 18 行）

**修改前**
```python
        self.out_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_bands)
        ])
        
        # 频带重要性（可学习的软注意力）
        self.band_importance = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, num_bands),
            nn.Softmax(dim=-1)
        )
        self.use_temporal_context = True
        if self.use_temporal_context:
            self.temporal_context = nn.Conv1d(
```

**修改后**
```python
        self.out_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_bands)
        ])

        # 频带重要性（可学习的软注意力）
        # 注意：这里只输出 logits，softmax( / tau ) 在 forward 中进行（便于 train.py 统一控制 tau，保证可复现）
        self.band_importance = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, num_bands),  # logits
        )
        # temperature for importance softmax (set in train.py per-epoch / per-step)
        self.register_buffer("importance_tau", torch.tensor(1.0))
        self.use_temporal_context = True
        if self.use_temporal_context:
            self.temporal_context = nn.Conv1d(
```

### 变更块 3
- 修改前（src）：L87–L93  （共 7 行）
- 修改后（patched）：L91–L97  （共 7 行）

**修改前**
```python
        self._init_filters()
        if self.use_fixed_filters:
            self.base_filters.requires_grad_(False)
            self.center_freqs.requires_grad_(False)
            self.bandwidths.requires_grad_(False)
    
    def _init_filters(self):
```

**修改后**
```python
        self._init_filters()
        if self.use_fixed_filters:
            self.base_filters.requires_grad_(False)
            self.freq_deltas.requires_grad_(False)
            self.bandwidths.requires_grad_(False)
    
    def _init_filters(self):
```

### 变更块 4
- 修改前（src）：L124–L138  （共 15 行）
- 修改后（patched）：L128–L150  （共 23 行）

**修改前**
```python
        # 离散时间轴（采样点单位），奇偶 k 都居中对齐
        n = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0  # (k,)

        filters = []
        for i in range(self.num_bands):
            # 基础滤波器
            base = self.base_filters[i]  # (1, k)

            # 中心频率：固定均匀基频 + 可学习小偏移
            base_freq = (i + 1) / (self.num_bands + 1) * 0.5
            freq_offset = torch.tanh(self.center_freqs[i]) * 0.05
            center_f = torch.clamp(base_freq + freq_offset, 0.02, 0.48)

            # sigma：高斯包络标准差（采样点单位）
            # v2: 避免 sigma 过大使 envelope 近似常数，导致频带高度重叠
```

**修改后**
```python
        # 离散时间轴（采样点单位），奇偶 k 都居中对齐
        n = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0  # (k,)

        # ========== 单调中心频率参数化（cycles/sample） ==========
        # deltas > 0 -> cumsum -> strictly increasing
        deltas = F.softplus(self.freq_deltas) + 1e-4  # (K,)
        raw_freqs = torch.cumsum(deltas, dim=0)       # (K,)
        raw_freqs = raw_freqs / (raw_freqs[-1] + 1e-6)
        min_freq, max_freq = 0.02, 0.48              # avoid DC & Nyquist boundary
        center_freqs = min_freq + raw_freqs * (max_freq - min_freq)  # (K,)
        # ==========================================================

        filters = []
        sigmas = []
        for i in range(self.num_bands):
            # 基础滤波器
            base = self.base_filters[i]  # (1, k)

            # 中心频率（已确保单调递增）：cycles per sample
            center_f = center_freqs[i]

            # sigma：高斯包络标准差（采样点单位）
            # v2: 避免 sigma 过大使 envelope 近似常数，导致频带高度重叠
```

### 变更块 5
- 修改前（src）：L140–L155  （共 16 行）
- 修改后（patched）：L152–L181  （共 30 行）

**修改前**
```python
            max_sigma = float(k) / 2.0  # e.g. k=7 -> 3.5
            sigma = min_sigma + (max_sigma - min_sigma) * torch.sigmoid(self.bandwidths[i])

            envelope = torch.exp(-0.5 * (n / (sigma + 1e-6)) ** 2)
            modulation = torch.cos(2 * math.pi * center_f * n)

            # 最终滤波器
            final_filter = base * modulation * envelope

            # 归一化（保持能量）
            final_filter = final_filter / (final_filter.norm() + 1e-6)
            filters.append(final_filter)

        return torch.stack(filters, dim=0)  # (num_bands, 1, k)
    
    def forward(
```

**修改后**
```python
            max_sigma = float(k) / 2.0  # e.g. k=7 -> 3.5
            sigma = min_sigma + (max_sigma - min_sigma) * torch.sigmoid(self.bandwidths[i])

            sigmas.append(sigma)

            envelope = torch.exp(-0.5 * (n / (sigma + 1e-6)) ** 2)
            modulation = torch.cos(2 * math.pi * center_f * n)

            # 最终滤波器
            final_filter = base * modulation * envelope

            # DC suppression：对带通/高通 band 做零均值，抑制低频/DC 偏置（建议只对 i>0）
            if i > 0:
                final_filter = final_filter - final_filter.mean(dim=-1, keepdim=True)

            # 归一化（保持能量）
            final_filter = final_filter / (final_filter.norm() + 1e-6)
            filters.append(final_filter)

        # cache for diagnostics/logging (no grad)
        try:
            self._cached_center_freqs = center_freqs.detach()
            self._cached_sigmas = torch.stack(sigmas, dim=0).detach() if len(sigmas) > 0 else None
        except Exception:
            self._cached_center_freqs = None
            self._cached_sigmas = None

        return torch.stack(filters, dim=0)  # (num_bands, 1, k)
    
    def forward(
```

### 变更块 6
- 修改前（src）：L200–L206  （共 7 行）
- 修改后（patched）：L226–L235  （共 10 行）

**修改前**
```python
        else:
            x_combined = x

        importance = self.band_importance(x_combined)  # (B, T, N, num_bands)

        # 加权融合
        output = torch.zeros_like(x)
```

**修改后**
```python
        else:
            x_combined = x

        importance_logits = self.band_importance(x_combined)  # (B, T, N, num_bands)

        tau = float(self.importance_tau.item()) if hasattr(self, 'importance_tau') else 1.0
        importance = F.softmax(importance_logits / max(tau, 1e-6), dim=-1)

        # 加权融合
        output = torch.zeros_like(x)
```

### 变更块 7
- 修改前（src）：L213–L218  （共 6 行）
- 修改后（patched）：L242–L254  （共 13 行）

**修改前**
```python
            f'band_{i}': band_outputs[i] for i in range(self.num_bands)
        }
        band_features['importance'] = importance
        band_features['filters'] = filters  # 不要 detach
        band_features['filters_detached'] = filters.detach()
        
```

**修改后**
```python
            f'band_{i}': band_outputs[i] for i in range(self.num_bands)
        }
        band_features['importance'] = importance
        band_features['importance_logits'] = importance_logits
        band_features['importance_tau'] = torch.tensor(tau, device=importance.device, dtype=importance.dtype)
        # (K,) cycles/sample and sigma (in samples) for logging/visualization
        if hasattr(self, '_cached_center_freqs') and self._cached_center_freqs is not None:
            band_features['center_freqs'] = self._cached_center_freqs.to(device=importance.device)
        if hasattr(self, '_cached_sigmas') and self._cached_sigmas is not None:
            band_features['sigmas'] = self._cached_sigmas.to(device=importance.device)
        band_features['filters'] = filters  # 不要 detach
        band_features['filters_detached'] = filters.detach()
        
```

### 变更块 8
- 修改前（src）：L239–L244  （共 6 行）
- 修改后（patched）：L275–L281  （共 7 行）

**修改前**
```python
        num_bands: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
        use_fixed_laplacian: bool = False,
    ):
        super().__init__()
```

**修改后**
```python
        num_bands: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
        freq_ortho_metric: str = "dot",
        use_fixed_laplacian: bool = False,
    ):
        super().__init__()
```

### 变更块 9
- 修改前（src）：L246–L251  （共 6 行）
- 修改后（patched）：L283–L289  （共 7 行）

**修改前**
```python
        self.num_bands = num_bands
        self.use_fixed_laplacian = use_fixed_laplacian
        
        # 输入归一化
        self.input_norm = nn.LayerNorm(dim)
        
```

**修改后**
```python
        self.num_bands = num_bands
        self.use_fixed_laplacian = use_fixed_laplacian
        
        self.freq_ortho_metric = freq_ortho_metric
        # 输入归一化
        self.input_norm = nn.LayerNorm(dim)
        
```

### 变更块 10
- 修改前（src）：L293–L308  （共 16 行）
- 修改后（patched）：L331–L377  （共 47 行）

**修改前**
```python
            return torch.tensor(0.0, device=filt.device, dtype=filt.dtype)

        fft = torch.fft.rfft(filt, dim=-1)
        power = fft.real.pow(2) + fft.imag.pow(2)
        power = power / (power.norm(dim=-1, keepdim=True) + 1e-6)

        sim = power @ power.transpose(0, 1)  # (K, K)
        off_diag_mask = ~torch.eye(K, dtype=torch.bool, device=sim.device)
        if not off_diag_mask.any():
            return torch.tensor(0.0, device=sim.device, dtype=sim.dtype)
        overlap_loss = sim[off_diag_mask].pow(2).mean()

        return overlap_loss

    def compute_feature_orthogonality(
        self,
```

**修改后**
```python
            return torch.tensor(0.0, device=filt.device, dtype=filt.dtype)

        fft = torch.fft.rfft(filt, dim=-1)
        power_raw = fft.real.pow(2) + fft.imag.pow(2)  # (K, F)

        metric = getattr(self, "freq_ortho_metric", "dot")
        metric = str(metric).lower()

        if metric in ("dot", "cos", "cosine"):
            # 与 v2 保持一致：用 L2 归一化后做相似度矩阵，惩罚非对角
            power = power_raw / (power_raw.norm(dim=-1, keepdim=True) + 1e-6)
            sim = power @ power.transpose(0, 1)  # (K, K)
            off_diag_mask = ~torch.eye(K, dtype=torch.bool, device=sim.device)
            if not off_diag_mask.any():
                return torch.tensor(0.0, device=sim.device, dtype=sim.dtype)
            overlap_loss = sim[off_diag_mask].pow(2).mean()
            return overlap_loss

        if metric in ("js", "jensen-shannon", "jsd"):
            # 频谱当作概率分布：sum-normalize
            p = power_raw / (power_raw.sum(dim=-1, keepdim=True) + 1e-8)
            # 计算所有 band 对的 JS divergence 平均
            # JS(p,q) = 0.5*KL(p||m) + 0.5*KL(q||m), m=0.5*(p+q)
            # JS ∈ [0, ln 2]
            eps = 1e-8
            js_vals = []
            for i in range(K):
                for j in range(i + 1, K):
                    pi = p[i].clamp(min=eps)
                    pj = p[j].clamp(min=eps)
                    m = 0.5 * (pi + pj)
                    kl_i = (pi * (pi.log() - m.log())).sum()
                    kl_j = (pj * (pj.log() - m.log())).sum()
                    js = 0.5 * (kl_i + kl_j)
                    js_vals.append(js)
            if len(js_vals) == 0:
                return torch.tensor(0.0, device=filt.device, dtype=filt.dtype)
            js_mean = torch.stack(js_vals).mean()
            js_max = math.log(2.0)
            # overlap penalty in [0,1], bigger means more overlap
            overlap_loss = 1.0 - (js_mean / js_max).clamp(0.0, 1.0)
            return overlap_loss

        raise ValueError(f"Unknown freq_ortho_metric: {metric}")

    def compute_feature_orthogonality(
        self,
```

### 变更块 11
- 修改前（src）：L453–L458  （共 6 行）
- 修改后（patched）：L522–L528  （共 7 行）

**修改前**
```python
        dim: int,
        num_bands: int = 4,
        num_scales: int = 3,
        base_kernel_size: int = 5,
    ):
        super().__init__()
```

**修改后**
```python
        dim: int,
        num_bands: int = 4,
        num_scales: int = 3,
        freq_ortho_metric: str = "dot",
        base_kernel_size: int = 5,
    ):
        super().__init__()
```

### 变更块 12
- 修改前（src）：L464–L469  （共 6 行）
- 修改后（patched）：L534–L540  （共 7 行）

**修改前**
```python
                dim=dim,
                num_bands=num_bands,
                kernel_size=base_kernel_size + i * 2,  # 递增的kernel size
            )
            for i in range(num_scales)
        ])
```

**修改后**
```python
                dim=dim,
                num_bands=num_bands,
                kernel_size=base_kernel_size + i * 2,  # 递增的kernel size
                freq_ortho_metric=freq_ortho_metric,
            )
            for i in range(num_scales)
        ])
```

## 文件：`models/motip/freq_aware_trajectory_modeling.py`
- 覆盖目标：D (metric wiring)
- Hunk 数量：4

### 变更块 1
- 修改前（src）：L77–L82  （共 6 行）
- 修改后（patched）：L77–L83  （共 7 行）

**修改前**
```python
        num_bands: int = 4,
        freq_kernel_size: int = 7,
        use_fixed_laplacian: bool = False,
        use_multiscale_freq: bool = False,
        num_freq_scales: int = 3,
        # 时序建模参数
```

**修改后**
```python
        num_bands: int = 4,
        freq_kernel_size: int = 7,
        use_fixed_laplacian: bool = False,
        freq_ortho_metric: str = "dot",
        use_multiscale_freq: bool = False,
        num_freq_scales: int = 3,
        # 时序建模参数
```

### 变更块 2
- 修改前（src）：L111–L116  （共 6 行）
- 修改后（patched）：L112–L118  （共 7 行）

**修改前**
```python
                num_bands=num_bands,
                num_scales=num_freq_scales,
                base_kernel_size=freq_kernel_size,
            )
        else:
            self.freq_decomposition = LearnableFrequencyDecomposition(
```

**修改后**
```python
                num_bands=num_bands,
                num_scales=num_freq_scales,
                base_kernel_size=freq_kernel_size,
                freq_ortho_metric=freq_ortho_metric,
            )
        else:
            self.freq_decomposition = LearnableFrequencyDecomposition(
```

### 变更块 3
- 修改前（src）：L118–L123  （共 6 行）
- 修改后（patched）：L120–L126  （共 7 行）

**修改前**
```python
                num_bands=num_bands,
                kernel_size=freq_kernel_size,
                dropout=dropout,
                use_fixed_laplacian=use_fixed_laplacian,
            )
        
```

**修改后**
```python
                num_bands=num_bands,
                kernel_size=freq_kernel_size,
                dropout=dropout,
                freq_ortho_metric=freq_ortho_metric,
                use_fixed_laplacian=use_fixed_laplacian,
            )
        
```

### 变更块 4
- 修改前（src）：L569–L572  （共 4 行）
- 修改后（patched）：L572–L576  （共 5 行）

**修改前**
```python
        max_seq_len=config.get("REL_PE_LENGTH", 30),
        use_mamba_for_lowfreq=config.get("USE_MAMBA_FOR_LOWFREQ", True),
        dropout=config.get("FREQ_DROPOUT", 0.1),
    )
```

**修改后**
```python
        max_seq_len=config.get("REL_PE_LENGTH", 30),
        use_mamba_for_lowfreq=config.get("USE_MAMBA_FOR_LOWFREQ", True),
        dropout=config.get("FREQ_DROPOUT", 0.1),
        freq_ortho_metric=config.get("FREQ_ORTHO_METRIC", "dot"),
    )
```

## 文件：`train.py`
- 覆盖目标：C (tau schedule) + E (diagnostics dumps)
- Hunk 数量：4

### 变更块 1
- 修改前（src）：L243–L248  （共 6 行）
- 修改后（patched）：L243–L270  （共 28 行）

**修改前**
```python
        epoch_start_timestamp = TPS.timestamp()
        # Prepare the sampler for the current epoch:
        train_sampler.prepare_for_epoch(epoch=epoch)
        # Train one epoch:
        train_metrics = train_one_epoch(
            accelerator=accelerator,
```

**修改后**
```python
        epoch_start_timestamp = TPS.timestamp()
        # Prepare the sampler for the current epoch:
        train_sampler.prepare_for_epoch(epoch=epoch)
        # --- LFD: importance temperature schedule (softmax(logits / tau)) ---
        # Default: linear anneal tau_start -> tau_end across epochs.
        if config.get("USE_IMPORTANCE_TAU_SCHEDULE", True):
            tau_start = float(config.get("IMPORTANCE_TAU_START", 2.0))
            tau_end = float(config.get("IMPORTANCE_TAU_END", 1.0))
            denom = max(config["EPOCHS"] - 1, 1)
            prog = float(epoch) / float(denom)
            importance_tau = tau_start + (tau_end - tau_start) * prog
        else:
            importance_tau = float(config.get("IMPORTANCE_TAU", 1.0))

        def _set_importance_tau(m):
            if hasattr(m, "importance_tau"):
                try:
                    m.importance_tau.fill_(float(importance_tau))
                except Exception:
                    pass

        model.apply(_set_importance_tau)
        if accelerator.is_main_process:
            logger.info(log=f"[LFD] importance_tau={importance_tau:.4f}")

        # Train one epoch:
        train_metrics = train_one_epoch(
            accelerator=accelerator,
```

### 变更块 2
- 修改前（src）：L267–L272  （共 6 行）
- 修改后（patched）：L289–L296  （共 8 行）

**修改前**
```python
            use_accelerate_clip_norm=config.get("USE_ACCELERATE_CLIP_NORM", True),
            freq_ortho_loss_weight=config.get("FREQ_ORTHO_LOSS_WEIGHT", 1.0),
            freq_consistency_loss_weight=config.get("FREQ_CONSISTENCY_LOSS_WEIGHT", 0.1),
            # For multi last checkpoints:
            outputs_dir=outputs_dir,
            is_last_epochs=(epoch == config["EPOCHS"] - 1),
```

**修改后**
```python
            use_accelerate_clip_norm=config.get("USE_ACCELERATE_CLIP_NORM", True),
            freq_ortho_loss_weight=config.get("FREQ_ORTHO_LOSS_WEIGHT", 1.0),
            freq_consistency_loss_weight=config.get("FREQ_CONSISTENCY_LOSS_WEIGHT", 0.1),
            save_lfd_diagnostics=config.get("SAVE_LFD_DIAGNOSTICS", False),
            lfd_diag_interval=int(config.get("LFD_DIAG_INTERVAL", 500)),
            # For multi last checkpoints:
            outputs_dir=outputs_dir,
            is_last_epochs=(epoch == config["EPOCHS"] - 1),
```

### 变更块 3
- 修改前（src）：L427–L432  （共 6 行）
- 修改后（patched）：L451–L459  （共 9 行）

**修改前**
```python
        logging_interval: int = 20,
        freq_ortho_loss_weight: float = 1.0,
        freq_consistency_loss_weight: float = 0.1,
        # For multi last checkpoints:
        outputs_dir: str = None,
        is_last_epochs: bool = False,
```

**修改后**
```python
        logging_interval: int = 20,
        freq_ortho_loss_weight: float = 1.0,
        freq_consistency_loss_weight: float = 0.1,
        # LFD diagnostics (optional evidence chain dumps)
        save_lfd_diagnostics: bool = False,
        lfd_diag_interval: int = 500,
        # For multi last checkpoints:
        outputs_dir: str = None,
        is_last_epochs: bool = False,
```

### 变更块 4
- 修改前（src）：L664–L669  （共 6 行）
- 修改后（patched）：L691–L721  （共 31 行）

**修改前**
```python
                detr_loss_dict[k] * detr_weight_dict[k] for k in detr_loss_dict.keys() if k in detr_weight_dict
            )
            freq_losses = seq_info.get("freq_losses", {}) if not only_detr else {}
            freq_ortho_loss = freq_losses.get("ortho_loss", 0.0)
            if not torch.is_tensor(freq_ortho_loss):
                freq_ortho_loss = torch.tensor(freq_ortho_loss, device=device, dtype=detr_loss.dtype)
```

**修改后**
```python
                detr_loss_dict[k] * detr_weight_dict[k] for k in detr_loss_dict.keys() if k in detr_weight_dict
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
                except Exception as e:
                    # avoid breaking training due to diagnostics
                    pass

            freq_ortho_loss = freq_losses.get("ortho_loss", 0.0)
            if not torch.is_tensor(freq_ortho_loss):
                freq_ortho_loss = torch.tensor(freq_ortho_loss, device=device, dtype=detr_loss.dtype)
```

## 新增文件：`tools/plot_lfd_diagnostics.py`（E：证据链可视化脚本）
- 说明：读取 `outputs_dir/lfd_diag/*.pt`（由 train.py 在 `SAVE_LFD_DIAGNOSTICS=True` 时写出），生成：
  - `fft_responses.png`（每个 band 的频响曲线）
  - `overlap_heatmap.png`（频谱 overlap 热力图）
  - `importance_mean.png` / `importance_hist.png`（importance 统计）

```python
"""Plot LFD diagnostics dumps.

This script reads *.pt files produced by train.py when SAVE_LFD_DIAGNOSTICS=True
and outputs:
  1) per-band frequency response curves (FFT power)
  2) overlap heatmap (cosine similarity on normalized power)
  3) importance statistics (mean + histogram)

Usage:
  python tools/plot_lfd_diagnostics.py \
      --diag_dir outputs/exp_xxx/lfd_diag \
      --out_dir outputs/exp_xxx/lfd_diag_plots \
      --pick latest

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch


def _list_pt_files(diag_dir: Path):
    return sorted([p for p in diag_dir.glob('*.pt') if p.is_file()])


def _pick_file(files, pick: str):
    if not files:
        raise FileNotFoundError('No .pt files found')
    if pick == 'latest':
        return files[-1]
    if pick.isdigit():
        idx = int(pick)
        idx = max(0, min(idx, len(files) - 1))
        return files[idx]
    # fallback: treat as filename
    p = Path(pick)
    if p.exists():
        return p
    raise ValueError(f'Unknown pick: {pick}')


def _fft_power(filters: np.ndarray, n_fft: int = 256):
    # filters: (K, k)
    fft = np.fft.rfft(filters, n=n_fft, axis=-1)
    power = (fft.real ** 2 + fft.imag ** 2)
    freqs = np.fft.rfftfreq(n_fft, d=1.0)  # cycles/sample in [0,0.5]
    return freqs, power


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag_dir', type=str, required=True)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--pick', type=str, default='latest', help='latest | integer index | path')
    ap.add_argument('--n_fft', type=int, default=256)
    args = ap.parse_args()

    diag_dir = Path(args.diag_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _list_pt_files(diag_dir)
    pt_path = _pick_file(files, args.pick)
    diag = torch.load(pt_path, map_location='cpu')

    # --------------- filters -> FFT curves + overlap heatmap ---------------
    filters = diag.get('filters_detached', None)
    if isinstance(filters, torch.Tensor):
        filt = filters.squeeze(1).float().numpy()  # (K,k)
        freqs, power = _fft_power(filt, n_fft=args.n_fft)  # (F,), (K,F)
        # normalize for comparison
        power_norm = power / (power.sum(axis=-1, keepdims=True) + 1e-12)

        # Save curves data
        np.save(out_dir / 'freqs.npy', freqs)
        np.save(out_dir / 'power_norm.npy', power_norm)

        # Plot curves
        import matplotlib.pyplot as plt
        plt.figure()
        for i in range(power_norm.shape[0]):
            plt.plot(freqs, power_norm[i], label=f'band_{i}')
        plt.xlabel('Frequency (cycles / sample)')
        plt.ylabel('Normalized power')
        plt.title(f'Band frequency responses\n{pt_path.name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'fft_responses.png', dpi=200)
        plt.close()

        # Overlap heatmap: cosine similarity on normalized power
        # (K,F) -> (K,K)
        p = power_norm
        p = p / (np.linalg.norm(p, axis=-1, keepdims=True) + 1e-12)
        sim = p @ p.T
        plt.figure()
        plt.imshow(sim, vmin=0.0, vmax=1.0)
        plt.colorbar()
        plt.xlabel('Band j')
        plt.ylabel('Band i')
        plt.title('Overlap heatmap (cosine similarity)')
        plt.tight_layout()
        plt.savefig(out_dir / 'overlap_heatmap.png', dpi=200)
        plt.close()

    # --------------- importance stats ---------------
    imp = diag.get('importance', None)
    if isinstance(imp, torch.Tensor):
        imp_np = imp.float().numpy()  # (...,K)
        K = imp_np.shape[-1]
        imp_flat = imp_np.reshape(-1, K)
        imp_mean = imp_flat.mean(axis=0)
        np.save(out_dir / 'importance_mean.npy', imp_mean)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.bar(np.arange(K), imp_mean)
        plt.xlabel('Band index')
        plt.ylabel('Mean importance')
        plt.title('Importance mean across (B,T,N)')
        plt.tight_layout()
        plt.savefig(out_dir / 'importance_mean.png', dpi=200)
        plt.close()

        # histogram per band
        plt.figure()
        for i in range(K):
            plt.hist(imp_flat[:, i], bins=50, alpha=0.4, label=f'band_{i}')
        plt.xlabel('Importance')
        plt.ylabel('Count')
        plt.title('Importance histograms')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'importance_hist.png', dpi=200)
        plt.close()

    # --------------- meta ---------------
    with open(out_dir / 'meta.txt', 'w', encoding='utf-8') as f:
        f.write(f'source_pt: {pt_path}\n')
        for k in ['epoch', 'global_step']:
            if k in diag:
                f.write(f'{k}: {diag[k]}\n')
        if 'importance_tau' in diag:
            tau = diag['importance_tau']
            if isinstance(tau, torch.Tensor):
                try:
                    f.write(f'importance_tau: {float(tau.item())}\n')
                except Exception:
                    pass

    print(f'[OK] Plots saved to: {out_dir}')


if __name__ == '__main__':
    main()
```
