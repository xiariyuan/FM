# Runtime Replay 主线重构与论文主线执行计划（按文件/按行号/按修改块）

版本：2026-03-17  
适用包：`gpt_pro_review_runtime_replay_20260317`  
目的：把当前项目从 **GT pseudo-track 代理训练** 彻底切到 **runtime candidate replay + ambiguity-aware top-K reranking** 主线，并把论文叙事、实验组织、代码修改一起锁死。

---

## 0. 先说死：从现在开始主线是什么

### 0.1 论文主线

论文主线不再是：

- “我们学了一个更好的 pairwise association score”
- “我们把 Laplace / frequency cue 融合得更好了”
- “association-only 可以把整个 tracker 救起来”

论文主线改成：

**在强 tracking-by-detection host 上，针对真实运行时 detection-centered candidate groups，训练一个仅在歧义场景触发的 bounded top-K reranker，用于修正 hard association，同时不破坏 easy cases。**

### 0.2 技术主线

技术主线不再是：

`build_gt_pseudotrack_groups.py -> train_mtcr_from_gt_tracks.py -> runtime MTCR`

技术主线改成：

`runtime tracker dump real candidate groups -> GT alignment labeling -> runtime replay shard building -> listwise / duel / safe rerank training -> frozen-host runtime integration -> on/off ablation`

### 0.3 为什么必须这样改

根据当前包内证据：

- `mtcr_gt_aligned_metrics.jsonl` 显示 GT pseudo-group 训练下，`base_top1` 已经接近 1，`top1_gain` 基本为 0，ambiguous gain 也为 0 或略负；这不是容量问题，而是训练对象问题。
- `sw_yolox_base_full7` 与 `sw_yolox_heuristic_full7` 说明强 host 明显更强，但 learned branch 还没有证明在强 host 上有额外价值。
- `runtime_assoc_dump` 相关代码已经存在，说明你已经具备最关键的基础设施：**真实运行时候选集导出能力**。

因此，**应该重构训练链，而不是继续打磨旧的 MTCR/GT pseudo-track 主线。**

---

## 1. 论文方向最终定稿（写作时就按这个方向）

### 1.1 论文应该怎么讲

#### 主标题方向

推荐主线标题风格：

- **Runtime-Replay-Trained Ambiguity-Aware Association Reranking for Strong MOT Hosts**
- 或中文内部表达：**面向强宿主 MOT 的真实运行时歧义关联重排序插件**

#### 摘要核心句

> 我们不重新设计整个 MOT 关联器，而是直接从真实 tracker 运行时导出 detection-centered candidate groups，训练一个仅在歧义场景触发的 bounded top-K reranker，并将其作为强宿主上的安全型 plug-in，用于提升困难关联场景中的 identity consistency，同时保持简单场景稳定性。

### 1.2 贡献点写成 3 条即可

1. **训练对象创新**：从 GT-clean pseudo-track 代理任务切换到真实 runtime candidate competition。  
2. **方法创新**：提出 ambiguity-triggered、bounded、top-K、zero-sum 风格的安全重排序模块。  
3. **实验创新**：在 frozen host 上做干净的 plugin on/off，对 easy / ambiguous / background 分层验证收益与风险。

### 1.3 不要再写成创新点的内容

不要再把下面内容写成创新点：

- pairwise MLP
- Laplace/frequency cue 本身
- “融合 motion 和 appearance”
- “更好的匹配分数”
- GT pseudo-track 组训练

这些最多只能当实现细节或失败路线对照。

---

## 2. 项目结构硬规定

### 2.1 主实验必须拆成两条线

#### A. public-comparable 主线

- 目标：论文 headline line
- 规则：provided/public comparable，不能混入 external-det ceiling 叙事
- 用途：回答 reviewer 的公平比较问题

#### B. external-det / system-recovery 次线

- 目标：开发沙盒 + ceiling line
- 用途：回答“在强 host 上，插件是否还有价值”
- 注意：它不能冒充主 benchmark 主表

### 2.2 learned module 的定位

从现在开始：

- **旧 MTCR / GT pseudo-group 路线降级为 legacy**
- **新的 runtime-replay reranker 只能是 plug-in，不是 main engine**

### 2.3 stop/go 判据

继续投资 learned association 的唯一条件：

- frozen host 上稳定提升 HOTA / AssA / IDF1；
- ambiguous group 上有稳定正增益；
- easy case 几乎不受损；
- 第二 host 或第二 benchmark 也能同号增益。

如果这些做不到，**停止 association learning，转向 host / ReID 主线**。

---

## 3. 代码总改法：不是补丁，是重构训练链

### 3.1 主保留

保留并作为新主线骨架：

- `src/core/runtime_tracker_bytetrack.py`
- `src/scripts/build_runtime_assoc_replay_labels.py`

### 3.2 主废弃

下面文件不再是主训练链：

- `src/scripts/build_gt_pseudotrack_groups.py`
- `src/scripts/train_mtcr_from_gt_tracks.py`
- `src/scripts/train_haca_v1_from_gt_tracks.py`

### 3.3 必须新增

必须新增这些文件：

- `src/scripts/build_runtime_assoc_group_shards.py`
- `src/scripts/train_mtcr_from_runtime_replay.py`
- `src/scripts/train_runtime_rerank_baseline.py`
- `src/scripts/eval_runtime_rerank_ablation.py`
- `src/models/runtime_replay_assoc.py`  （包里未包含 models 目录代码，但你必须新增这个模型文件）

---

# 4. 文件级详细修改清单

---

## 4.1 修改 `src/core/runtime_tracker_bytetrack.py`

这是最关键的 runtime 文件。你要做三件事：

1. 把当前的 `MTCR` 路径改成 **runtime replay rerank** 路径；
2. 把当前 CSV-only dump 改成 **group meta + tensor shard** 双输出；
3. 只在 **primary association** 处启用 reranker，不进入 secondary/newborn 链。

---

### 修改块 A：替换 MTCR import（第 30–35 行）

#### 修改前

```python
try:
    from models.mtcr_assoc import MTCRAssociationAdapter
    _MTCR_ASSOC_AVAILABLE = True
except Exception:
    MTCRAssociationAdapter = None
    _MTCR_ASSOC_AVAILABLE = False
```

#### 修改后

```python
try:
    from models.runtime_replay_assoc import RuntimeReplayAssociationAdapter
    _RUNTIME_RERANK_AVAILABLE = True
except Exception:
    RuntimeReplayAssociationAdapter = None
    _RUNTIME_RERANK_AVAILABLE = False

# backward-compat alias, 仅为了不让旧配置直接崩
MTCRAssociationAdapter = RuntimeReplayAssociationAdapter
_MTCR_ASSOC_AVAILABLE = _RUNTIME_RERANK_AVAILABLE
```

#### 原因

- 旧 `MTCRAssociationAdapter` 这个命名已经绑死在 GT pseudo-track 旧路线，不利于你后面写论文和做干净隔离。
- 新名字要体现：**这是 runtime replay learned rerank，不是旧的 GT pseudo-group MTCR。**

---

### 修改块 B：替换初始化参数（第 175–183 行 + 第 210–212 行）

#### 修改前

```python
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
...
assoc_runtime_dump_path: str = "",
assoc_runtime_dump_topk: int = 8,
assoc_runtime_dump_min_score: float = 0.0,
```

#### 修改后

```python
assoc_use_runtime_rerank: bool = False,
assoc_runtime_rerank_checkpoint: str = "",
assoc_runtime_rerank_hist_hidden: int = 16,
assoc_runtime_rerank_comp_hidden: int = 64,
assoc_runtime_rerank_topk: int = 3,
assoc_runtime_rerank_margin_threshold: float = 0.10,
assoc_runtime_rerank_margin_temperature: float = 0.03,
assoc_runtime_rerank_delta_scale: float = 0.75,
assoc_runtime_rerank_min_history: int = 3,
assoc_runtime_rerank_decay_scales: Optional[List[float]] = None,
assoc_runtime_rerank_enable_secondary: bool = False,
assoc_runtime_rerank_use_bg_head: bool = False,
...
assoc_runtime_dump_path: str = "",
assoc_runtime_dump_topk: int = 8,
assoc_runtime_dump_min_score: float = 0.0,
assoc_runtime_dump_save_tensors: bool = True,
assoc_runtime_dump_npz_every_n_groups: int = 2048,
assoc_runtime_dump_scope: str = "primary_only",
```

#### 原因

- 让配置语义和论文语义一致；
- 显式禁止 secondary stage 默认启用 rerank；
- `save_tensors` 和 `npz_every_n_groups` 是训练真正需要的，不然你只能得到 CSV 标量特征；
- `scope=primary_only` 是为了论文 clean attribution。

---

### 修改块 C：替换 MTCR 初始化（第 291–326 行）

#### 修改前

这一块是：

- 读取 `assoc_use_mtcr`
- 创建 `MTCRAssociationAdapter`
- 空 checkpoint 时 no-op init

#### 修改后

把整块替换为：

```python
self.assoc_use_runtime_rerank = bool(assoc_use_runtime_rerank) and _RUNTIME_RERANK_AVAILABLE
self.assoc_runtime_rerank_checkpoint = str(assoc_runtime_rerank_checkpoint or "")
self.assoc_runtime_rerank_hist_hidden = int(assoc_runtime_rerank_hist_hidden)
self.assoc_runtime_rerank_comp_hidden = int(assoc_runtime_rerank_comp_hidden)
self.assoc_runtime_rerank_topk = int(max(assoc_runtime_rerank_topk, 1))
self.assoc_runtime_rerank_margin_threshold = float(assoc_runtime_rerank_margin_threshold)
self.assoc_runtime_rerank_margin_temperature = float(max(assoc_runtime_rerank_margin_temperature, 1e-4))
self.assoc_runtime_rerank_delta_scale = float(assoc_runtime_rerank_delta_scale)
self.assoc_runtime_rerank_min_history = int(max(assoc_runtime_rerank_min_history, 1))
self.assoc_runtime_rerank_decay_scales = list(assoc_runtime_rerank_decay_scales or [1.0, 2.0, 4.0])
self.assoc_runtime_rerank_enable_secondary = bool(assoc_runtime_rerank_enable_secondary)
self.assoc_runtime_rerank_use_bg_head = bool(assoc_runtime_rerank_use_bg_head)
self.runtime_rerank_assoc = None

if self.assoc_use_runtime_rerank:
    try:
        if self.assoc_runtime_rerank_checkpoint:
            self.runtime_rerank_assoc = RuntimeReplayAssociationAdapter.from_npz(
                self.assoc_runtime_rerank_checkpoint
            )
        else:
            self.runtime_rerank_assoc = RuntimeReplayAssociationAdapter(
                hist_hidden=self.assoc_runtime_rerank_hist_hidden,
                comp_hidden=self.assoc_runtime_rerank_comp_hidden,
                topk=self.assoc_runtime_rerank_topk,
                margin_threshold=self.assoc_runtime_rerank_margin_threshold,
                margin_temperature=self.assoc_runtime_rerank_margin_temperature,
                delta_scale=self.assoc_runtime_rerank_delta_scale,
                min_history=self.assoc_runtime_rerank_min_history,
                decay_scales=self.assoc_runtime_rerank_decay_scales,
                use_bg_head=self.assoc_runtime_rerank_use_bg_head,
            )
        self.runtime_rerank_assoc = self.runtime_rerank_assoc.to(distributed_device())
        self.runtime_rerank_assoc.eval()
    except Exception as exc:
        warnings.warn(
            f"[RuntimeTrackerByteTrack] Failed to init runtime replay reranker; disabled. Error: {exc}"
        )
        self.assoc_use_runtime_rerank = False
        self.runtime_rerank_assoc = None
```

#### 原因

- 新旧路径彻底分开；
- `use_bg_head=False` 默认关闭，让 background/null 继续由 base host 负责；
- 推理时只接受 runtime replay 训练出的 checkpoint。

---

### 修改块 D：重写 `_ensure_assoc_dump_writer`（第 406–445 行）

#### 修改前

这一块只会创建一个 CSV，字段只有：

- `seq, frame, det_index, track_rank, ...`
- 没有 group-level uncertainty
- 没有 appearance/history tensor 索引
- 没有 shard flush 机制

#### 修改后

把整块替换成下面这个结构：

```python
def _ensure_assoc_dump_writer(self) -> None:
    if not self.assoc_runtime_dump_path:
        return
    if getattr(self, "_assoc_dump_initialized", False):
        return

    dump_root = self.assoc_runtime_dump_path
    os.makedirs(dump_root, exist_ok=True)
    seq_name = str(self.sequence_name or "unknown_seq")

    self._assoc_dump_meta_path = os.path.join(dump_root, f"{seq_name}.groups.csv")
    self._assoc_dump_meta_file = open(self._assoc_dump_meta_path, "a", encoding="utf-8", newline="")
    self._assoc_dump_writer = csv.writer(self._assoc_dump_meta_file)

    if os.path.getsize(self._assoc_dump_meta_path) == 0:
        self._assoc_dump_writer.writerow([
            "seq", "frame", "assoc_stage", "group_id", "det_index", "group_size",
            "base_margin", "base_entropy", "topk", "topk_positive_included",
            "track_rank", "track_id", "is_selected",
            "det_score", "base_score", "refined_score", "motion_score",
            "track_gap", "track_hist_len",
            "det_cx", "det_cy", "det_w", "det_h",
            "track_cx", "track_cy", "track_w", "track_h",
            "det_feat_offset", "track_feat_offset", "pair_feat_offset"
        ])

    self._assoc_dump_group_counter = 0
    self._assoc_dump_tensor_buffer = []
    self._assoc_dump_initialized = True
```

#### 原因

- 你的训练单位应该是 **group**，不是孤立 pair row；
- 后面训练必须读到 `group_id / group_size / margin / entropy / tensor offset`；
- 这是把 dump 从“可看 CSV”升级成“可训练数据接口”的第一步。

---

### 修改块 E：新增两个辅助函数（插入到 `_maybe_dump_feature_assoc_candidates` 之前）

#### 新增函数 1：group stats

```python
def _build_assoc_group_stats(self, row_scores: torch.Tensor) -> tuple[float, float]:
    if row_scores.numel() == 0:
        return 0.0, 0.0
    if row_scores.numel() == 1:
        p = torch.tensor([1.0], device=row_scores.device, dtype=row_scores.dtype)
        return float(row_scores[0].item()), float((-(p * torch.log(p))).sum().item())
    top2 = torch.topk(row_scores, k=min(2, row_scores.numel())).values
    margin = float((top2[0] - top2[1]).item())
    logits = torch.logit(row_scores.clamp(1e-4, 1.0 - 1e-4), eps=1e-4)
    p = torch.softmax(logits, dim=0)
    entropy = float((-(p * torch.log(p.clamp_min(1e-8)))).sum().item())
    return margin, entropy
```

#### 新增函数 2：flush tensor shard

```python
def _flush_assoc_tensor_shard(self) -> None:
    if not getattr(self, "_assoc_dump_tensor_buffer", None):
        return
    dump_root = self.assoc_runtime_dump_path
    seq_name = str(self.sequence_name or "unknown_seq")
    shard_idx = getattr(self, "_assoc_dump_shard_idx", 0)
    shard_path = os.path.join(dump_root, f"{seq_name}.shard_{shard_idx:05d}.npz")

    payload = {
        "records": np.asarray(self._assoc_dump_tensor_buffer, dtype=object),
    }
    np.savez_compressed(shard_path, **payload)
    self._assoc_dump_tensor_buffer = []
    self._assoc_dump_shard_idx = shard_idx + 1
```

#### 原因

- margin / entropy 是 ambiguity trigger 的核心输入；
- tensor shard flush 让你真正能把 det feat / track summary / pair token 存下来，而不是只存 CSV。

---

### 修改块 F：重写 `_maybe_dump_feature_assoc_candidates`（第 446–520 行）

#### 修改前

当前函数的问题：

1. 只写 CSV；
2. 只写 topK 分数，没有写 group-level uncertainty；
3. 没有 det feat / track feat / pair token；
4. 写的是 row，而不是真正的 group object；
5. 后续训练很难做 listwise candidate competition。

#### 修改后

把整个函数替换为下面这个结构（注意：这是执行级伪代码，不是最终逐字可运行版；你可以按这个结构直接实现）：

```python
def _maybe_dump_feature_assoc_candidates(
    self,
    base_scores: torch.Tensor,
    refined_scores: torch.Tensor,
    motion_scores: Optional[torch.Tensor],
    scores: torch.Tensor,
    active_ids_list: List[int],
    track_history_masks: torch.Tensor,
    track_history_times: Optional[torch.Tensor],
    det_boxes_cxcywh: Optional[torch.Tensor],
    track_boxes_cxcywh: Optional[torch.Tensor],
    id_labels: torch.Tensor,
    det_features: Optional[torch.Tensor] = None,
    track_history_features: Optional[torch.Tensor] = None,
    pair_tokens: Optional[torch.Tensor] = None,
    assoc_stage: str = "primary",
) -> None:
    if not self.assoc_runtime_dump_path:
        return
    if assoc_stage != "primary":
        return
    if base_scores.numel() == 0 or refined_scores.numel() == 0:
        return

    self._ensure_assoc_dump_writer()
    if self._assoc_dump_writer is None:
        return

    num_dets, num_tracks = refined_scores.shape
    hist_counts = (~track_history_masks).sum(dim=1).to(dtype=torch.long)
    track_gaps = torch.full((num_tracks,), -1, device=refined_scores.device, dtype=torch.long)
    if track_history_times is not None and track_history_times.numel() > 0:
        valid = ~track_history_masks
        last_idx = (valid.long() * (torch.arange(track_history_times.shape[1], device=track_history_times.device).view(1, -1) + 1)).max(dim=1).values - 1
        last_idx = last_idx.clamp(min=0)
        last_times = torch.gather(track_history_times, dim=1, index=last_idx.view(-1, 1)).squeeze(1)
        track_gaps = (track_history_times.new_full((track_history_times.shape[0],), int(self.frame_id)) - last_times).clamp(min=0)

    for det_idx in range(num_dets):
        row = refined_scores[det_idx]
        k = min(self.assoc_runtime_dump_topk, int(row.numel()))
        top_idx = torch.topk(row, k=k).indices.tolist()
        base_margin, base_entropy = self._build_assoc_group_stats(base_scores[det_idx, top_idx])
        group_id = f"{self.sequence_name}:{int(self.frame_id)}:{int(det_idx)}"

        for rank, track_idx in enumerate(top_idx, start=1):
            refined_val = float(refined_scores[det_idx, track_idx].item())
            if refined_val < self.assoc_runtime_dump_min_score:
                continue

            track_id = int(active_ids_list[track_idx])
            selected = 1 if int(id_labels[det_idx].item()) == track_id else 0
            det_box = det_boxes_cxcywh[det_idx] if det_boxes_cxcywh is not None else None
            trk_box = track_boxes_cxcywh[track_idx] if track_boxes_cxcywh is not None else None

            det_offset = -1
            trk_offset = -1
            pair_offset = -1
            if self.assoc_runtime_dump_save_tensors:
                det_feat_np = det_features[det_idx].detach().cpu().numpy() if det_features is not None else None
                hist_feat_np = track_history_features[track_idx].detach().cpu().numpy() if track_history_features is not None else None
                pair_feat_np = pair_tokens[det_idx, track_idx].detach().cpu().numpy() if pair_tokens is not None else None
                self._assoc_dump_tensor_buffer.append({
                    "group_id": group_id,
                    "rank": int(rank),
                    "det_feat": det_feat_np,
                    "hist_feat": hist_feat_np,
                    "pair_feat": pair_feat_np,
                })
                rec_idx = len(self._assoc_dump_tensor_buffer) - 1
                det_offset = rec_idx
                trk_offset = rec_idx
                pair_offset = rec_idx

            self._assoc_dump_writer.writerow([
                str(self.sequence_name or ""), int(self.frame_id), assoc_stage, group_id,
                int(det_idx), int(k),
                float(base_margin), float(base_entropy), int(k), 0,
                int(rank), track_id, selected,
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
                det_offset, trk_offset, pair_offset,
            ])

        self._assoc_dump_group_counter += 1
        if self.assoc_runtime_dump_save_tensors and self._assoc_dump_group_counter % self.assoc_runtime_dump_npz_every_n_groups == 0:
            self._flush_assoc_tensor_shard()

    self._assoc_dump_meta_file.flush()
```

#### 原因

这是整个项目最关键的改动之一：

- 它把 dump 的对象从“若干 row”变成“真实 runtime 候选 group”；
- 它为后面训练提供了真正的 tensor 输入；
- 它把 ambiguity trigger 需要的 `margin / entropy` 一起导出来；
- 它把训练对象正式从 GT pseudo groups 切成 runtime candidate groups。

---

### 修改块 G：重写 `_refine_assoc_scores_with_mtcr`（第 945–982 行）

#### 修改前

当前函数直接把 `base_scores / det_features / track_history_features / masks / motion / det_scores` 丢给旧 MTCR。

#### 修改后

把整个函数替换成新函数，并重命名：

```python
def _refine_assoc_scores_with_runtime_rerank(
    self,
    base_scores: torch.Tensor,
    det_features: torch.Tensor,
    track_history_features: torch.Tensor,
    track_history_masks: torch.Tensor,
    track_history_times: Optional[torch.Tensor],
    det_scores: torch.Tensor,
    motion_scores: Optional[torch.Tensor] = None,
    assoc_stage: str = "primary",
) -> torch.Tensor:
    if assoc_stage != "primary":
        return base_scores
    if not self.assoc_use_runtime_rerank or self.runtime_rerank_assoc is None:
        return base_scores
    if base_scores.numel() == 0:
        return base_scores

    try:
        self.runtime_rerank_assoc = self.runtime_rerank_assoc.to(device=base_scores.device)
        track_gaps = None
        if track_history_times is not None and track_history_times.numel() > 0:
            valid = ~track_history_masks
            last_idx = (valid.long() * (torch.arange(track_history_times.shape[1], device=track_history_times.device).view(1, -1) + 1)).max(dim=1).values - 1
            last_idx = last_idx.clamp(min=0)
            last_times = torch.gather(track_history_times, dim=1, index=last_idx.view(-1, 1)).squeeze(1)
            track_gaps = (track_history_times.new_full((track_history_times.shape[0],), int(self.frame_id)) - last_times).clamp(min=0)

        result = self.runtime_rerank_assoc(
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
        warnings.warn(
            f"[RuntimeTrackerByteTrack] runtime replay rerank failed; fallback to base scores. Error: {exc}"
        )
    return base_scores
```

#### 原因

- 让 runtime plugin 明确只属于 `primary association`；
- 函数名必须和论文主张一致；
- 把 secondary/newborn 排除掉，避免 reviewer 质疑“你是不是改了整个 tracker 的多个阶段”。

---

### 修改块 H：修改 primary association call site（第 1356–1381 行）

#### 修改前

```python
feat_scores = self._refine_assoc_scores_with_mtcr(...)
...
self._maybe_dump_feature_assoc_candidates(...)
```

#### 修改后

```python
feat_scores = self._refine_assoc_scores_with_runtime_rerank(
    base_scores=feat_scores,
    det_features=refine_det_features,
    track_history_features=refine_track_features,
    track_history_masks=trajectory_masks,
    track_history_times=trajectory_times,
    det_scores=scores,
    motion_scores=motion_scores,
    assoc_stage="primary",
)

id_labels = self._match_with_scores(
    feat_scores,
    scores,
    active_ids_list,
    det_boxes_cxcywh=boxes_cxcywh,
    track_boxes_cxcywh=track_last_boxes,
)

self._maybe_dump_feature_assoc_candidates(
    base_scores=pre_refine_scores,
    refined_scores=feat_scores,
    motion_scores=motion_scores,
    scores=scores,
    active_ids_list=active_ids_list,
    track_history_masks=trajectory_masks,
    track_history_times=trajectory_times,
    det_boxes_cxcywh=boxes_cxcywh,
    track_boxes_cxcywh=track_last_boxes,
    id_labels=id_labels,
    det_features=refine_det_features,
    track_history_features=refine_track_features,
    pair_tokens=None,
    assoc_stage="primary",
)
```

#### 原因

- dump 必须发生在你真正想学习的 runtime 接口上；
- 必须把 det feat / history feat 一起传进去；
- `pair_tokens=None` 可以第一版先空着，后面有现成 pair token 再补。

---

### 修改块 I：修改 secondary/newborn 路径（第 1602–1610 行）

#### 修改前

第二处也调用了 `_refine_assoc_scores_with_mtcr(...)`。

#### 修改后

直接替换为：

```python
# runtime replay rerank 默认不参与 secondary / newborn / recovery 阶段
# 为了 clean attribution 与稳定性，这里保持 base host 行为不变。
```

或者如果你需要保留代码结构：

```python
base_scores = self._refine_assoc_scores_with_runtime_rerank(
    base_scores=base_scores,
    det_features=det_features,
    track_history_features=track_history_features,
    track_history_masks=track_history_masks,
    track_history_times=track_history_times,
    det_scores=scores,
    motion_scores=motion_scores,
    assoc_stage="secondary",
)
```

并且在函数里因为 `assoc_stage != "primary"` 直接返回原分数。

#### 原因

- 你的论文故事必须非常干净：**只改主关联一步**。
- secondary/newborn 混进去会直接破坏归因。

---

### 修改块 J：reset 里 flush shard（第 537–543 行附近）

#### 修改前

只 close CSV file。

#### 修改后

在 close 之前增加：

```python
try:
    self._flush_assoc_tensor_shard()
except Exception:
    pass
```

#### 原因

不然最后一批未满 shard 的 tensor buffer 会丢。

---

## 4.2 修改 `src/scripts/build_runtime_assoc_replay_labels.py`

这是你新主线的第二个关键文件。当前它做的是：

- 读 dump CSV
- 用 GT 对齐 det 和 track
- 给每行打 label

这方向对，但它还不够；它现在只能生成 row-level CSV，不足以支撑真正的 group-level listwise 训练和 recoverability 诊断。

---

### 修改块 A：扩展 CLI（第 24–35 行）

#### 修改前

```python
ap.add_argument("--out-csv", required=True)
ap.add_argument("--summary-json", default="")
```

#### 修改后

```python
ap.add_argument("--out-csv", required=True)
ap.add_argument("--summary-json", default="")
ap.add_argument("--out-group-jsonl", default="")
ap.add_argument("--out-recoverability-json", default="")
ap.add_argument("--out-parquet", default="")
ap.add_argument("--topk", type=int, default=8)
```

#### 原因

你需要三类输出：

1. row-level CSV：方便查错；
2. group-level JSONL：方便训练与 group 诊断；
3. recoverability summary：决定这个方向值不值得继续。

---

### 修改块 B：扩展 fieldnames（第 104–136 行）

#### 修改前

当前字段只有：

- row-level label
- margin
n- group size
- 基本 box 信息

#### 修改后

把 fieldnames 改成：

```python
fieldnames = [
    "seq", "frame", "group_id", "det_index", "track_rank", "track_id",
    "is_selected", "det_score", "base_score", "refined_score", "motion_score",
    "track_gap", "track_hist_len",
    "det_gt_id", "track_gt_id",
    "det_ignore", "track_ignore",
    "label", "valid_train_row",
    "group_has_positive", "group_size",
    "base_margin", "refined_margin", "base_entropy",
    "base_top1_correct", "positive_in_topk", "positive_rank",
    "group_is_ambiguous", "group_is_background", "group_is_recoverable",
    "det_cx", "det_cy", "det_w", "det_h",
    "track_cx", "track_cy", "track_w", "track_h",
    "source_csv",
]
```

#### 原因

如果没有：

- `positive_in_topk`
- `positive_rank`
- `group_is_recoverable`
- `group_is_ambiguous`

你根本没法判断学习器有没有机会把错误翻回来。

---

### 修改块 C：group 逻辑重写（第 161–251 行）

#### 修改前

当前逻辑的问题：

- 只算了 `group_has_positive`；
- 没有判断 `base top1 是否正确`；
- 没有判断 `positive 是否在 topK`；
- 没有 recoverability 统计。

#### 修改后

把 `(frame, det_index)` 那个循环中的 group-level 处理重写成下面这个结构：

```python
base_vals = [float(r["base_score"]) for r in rows]
refined_vals = [float(r["refined_score"]) for r in rows]
base_margin = 0.0 if len(base_vals) <= 1 else float(sorted(base_vals, reverse=True)[0] - sorted(base_vals, reverse=True)[1])
refined_margin = 0.0 if len(refined_vals) <= 1 else float(sorted(refined_vals, reverse=True)[0] - sorted(refined_vals, reverse=True)[1])

base_arr = np.asarray(base_vals, dtype=np.float32)
if base_arr.size > 0:
    logits = np.log(np.clip(base_arr, 1e-4, 1.0 - 1e-4) / np.clip(1.0 - base_arr, 1e-4, 1.0))
    prob = np.exp(logits - logits.max())
    prob = prob / np.clip(prob.sum(), 1e-8, None)
    base_entropy = float(-(prob * np.log(np.clip(prob, 1e-8, None))).sum())
else:
    base_entropy = 0.0

# 逐行打 label
...

positive_rows = [idx for idx, r in enumerate(labeled_rows) if int(r["label"]) == 1 and int(r["valid_train_row"]) == 1]
group_has_positive = 1 if len(positive_rows) > 0 else 0
positive_rank = -1
positive_in_topk = 0
base_top1_correct = 0
group_is_background = 1 if group_has_positive == 0 else 0

if group_has_positive:
    best_idx = int(np.argmax(base_arr))
    pos_idx = int(positive_rows[0])
    positive_rank = int(pos_idx + 1)
    positive_in_topk = 1
    base_top1_correct = 1 if best_idx == pos_idx else 0

# ambiguity 定义：base top1 错 或 margin 小于阈值
margin_thresh = 0.10  # 第一版固定，后续可从训练集统计
group_is_ambiguous = 1 if (group_has_positive and (base_top1_correct == 0 or base_margin < margin_thresh)) else 0

# recoverable 定义：base top1 错，但 positive 仍在 topK 中
group_is_recoverable = 1 if (group_has_positive and base_top1_correct == 0 and positive_in_topk == 1) else 0
```

然后把这些 group-level 字段写回每一行。

#### 原因

这一步直接决定你接下来有没有必要继续做 learned rerank：

- 如果大部分错误根本不可恢复，那学习器没法救；
- 如果大量错误是“正确 continuation 在 topK 里，但排序错了”，那这个方向就很有价值。

---

### 修改块 D：新增 group-level JSONL 输出（在第 253 行之前）

新增逻辑：如果 `--out-group-jsonl` 非空，每个 group 写一条 JSON：

```python
{
  "group_id": group_id,
  "seq": seq_name,
  "frame": frame,
  "det_index": det_index,
  "group_size": len(rows),
  "det_gt_id": det_gt_id,
  "group_has_positive": group_has_positive,
  "base_margin": base_margin,
  "base_entropy": base_entropy,
  "base_top1_correct": base_top1_correct,
  "positive_rank": positive_rank,
  "positive_in_topk": positive_in_topk,
  "group_is_ambiguous": group_is_ambiguous,
  "group_is_background": group_is_background,
  "group_is_recoverable": group_is_recoverable,
}
```

#### 原因

训练脚本不应该再从 row CSV 里“猜 group”，而应该直接读 group object。

---

### 修改块 E：新增 recoverability summary（第 253–258 行之后）

#### 修改后新增

```python
recoverability = {
    "groups": ...,
    "positive_groups": ...,
    "background_groups": ...,
    "ambiguous_groups": ...,
    "recoverable_groups": ...,
    "recoverable_rate_among_positive": ...,
    "recoverable_rate_among_ambiguous": ...,
    "base_top1_acc_positive": ...,
    "base_top1_acc_ambiguous": ...,
    "positive_in_topk_rate": ...,
}
```

#### 原因

这是你的 go/no-go 总开关。没有这个 summary，就会继续盲目训模型。

---

## 4.3 修改 `src/scripts/build_gt_pseudotrack_groups.py`

这个文件不是删除，而是 **降级为 legacy**。

### 修改块 A：顶部加 deprecated 说明（第 1–10 行之后）

新增注释：

```python
"""
LEGACY ONLY.
This script builds GT-keyed pseudo-track training groups.
It is retained only for ablation / negative-result reproduction.
Do NOT use it as the main training data builder for the runtime replay paper line.
Mainline has moved to runtime candidate replay.
"""
```

### 修改块 B：在 `main()` 开始处加显式 warning

在 CLI 开始执行前新增：

```python
print(
    "[warning] build_gt_pseudotrack_groups.py is now legacy-only. "
    "Use build_runtime_assoc_replay_labels.py + build_runtime_assoc_group_shards.py for the mainline.",
    flush=True,
)
```

### 是否要继续改这个文件内部逻辑？

**主线不改。**

#### 原因

这条线已经被当前包内结果证伪：

- `mtcr_gt_aligned_metrics.jsonl` 里 base top1 极高；
- 模型 loss 下降但没有 top1 gain；
- 说明这类 GT-clean pseudo groups 不是学习真实 runtime competition 的正确对象。

所以这个文件只留作：

- ablation；
- negative result；
- “为什么我们必须转向 runtime replay” 的证据。

---

## 4.4 修改 `src/scripts/train_mtcr_from_gt_tracks.py`

这个文件也降级为 legacy，不再是主 trainer。

### 修改块 A：把 argparse 描述改成 legacy（第 21–58 行）

#### 修改前

```python
parser = argparse.ArgumentParser(description="Train MTCR from GT pseudo-track NPZ groups.")
```

#### 修改后

```python
parser = argparse.ArgumentParser(
    description="[LEGACY] Train MTCR from GT pseudo-track NPZ groups. "
                "Do not use for the runtime replay mainline."
)
```

### 修改块 B：在 `main()` 开头直接打印警告（第 456 行之后）

新增：

```python
print(
    "[warning] train_mtcr_from_gt_tracks.py is legacy-only. "
    "Mainline must use train_mtcr_from_runtime_replay.py.",
    flush=True,
)
```

### 这一文件是否继续大改？

**不要。**

#### 原因

- 它的数据输入假设是 GT pseudo-group NPZ；
- `_group_batch()`、`_fit_margin_threshold()`、`_group_loss()` 全都围绕旧对象构建；
- 越改越脏，不如直接新写一个 runtime trainer。

---

## 4.5 新增 `src/scripts/build_runtime_assoc_group_shards.py`

这是新主线必须新增的文件。

### 文件职责

把：

- `runtime dump groups.csv`
- `runtime dump shard_XXXXX.npz`
- `labeled row csv`
- `group jsonl`

整理成可训练的 **group-level shard NPZ / parquet**。

### 新文件完整结构建议

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    ...


def load_group_meta(...):
    ...


def load_group_labels(...):
    ...


def merge_meta_and_labels(...):
    ...


def build_group_record(group_df, tensor_lookup):
    # 输出一个 detection-centered candidate group
    return {
        "group_id": ...,
        "seq": ...,
        "frame": ...,
        "det_index": ...,
        "group_size": ...,
        "base_margin": ...,
        "base_entropy": ...,
        "group_is_ambiguous": ...,
        "group_is_background": ...,
        "group_is_recoverable": ...,
        "det_feat": ...,
        "hist_feat": ...,
        "hist_mask": ...,
        "anchor_scores": ...,
        "motion_scores": ...,
        "track_gaps": ...,
        "labels": ...,
    }


def main():
    ...
```

### 输出格式必须长这样

每个 group 是一个 object：

- `det_feat`: `[D]`
- `hist_feat`: `[K, T, D]`
- `hist_mask`: `[K, T]`
- `anchor_scores`: `[K]`
- `motion_scores`: `[K]`
- `track_gaps`: `[K]`
- `labels`: `[K]`
- `group_is_ambiguous`
- `group_is_background`
- `group_is_recoverable`

### 原因

这是你真正的训练对象。

不是 pair，不是 GT track group，而是：

**一个 detection 对多个 runtime candidates 的竞争组。**

---

## 4.6 新增 `src/models/runtime_replay_assoc.py`

这个文件虽然不在 review package 里，但你必须新增。旧的 `models.mtcr_assoc` 不应该继续承载主线语义。

### 模型目标

输入：一个 detection-centered candidate group  
输出：每个 candidate 的 reranked score

### 结构要求

- ambiguity-triggered
- top-K only
- zero-sum residual
- bounded residual
- background/null 默认不由它负责

### 推荐模型骨架

```python
class RuntimeReplayAssociationAdapter(nn.Module):
    def __init__(
        self,
        hist_hidden=16,
        comp_hidden=64,
        topk=3,
        margin_threshold=0.10,
        margin_temperature=0.03,
        delta_scale=0.75,
        min_history=3,
        decay_scales=(1.0, 2.0, 4.0),
        use_bg_head=False,
    ):
        ...

    def forward(
        self,
        anchor_scores,
        det_features,
        track_history_features,
        track_history_masks,
        motion_scores,
        det_scores,
        track_gaps,
    ):
        # 1) base logits z0
        # 2) margin / entropy / trust -> activation gate a
        # 3) 只对 top-K 做 rival interaction
        # 4) zero-sum residual rbar
        # 5) z = z0 + a * delta * tanh(rbar)
        # 6) final_scores = sigmoid(z)
        return {
            "final_scores": ...,
            "comp_margin": ...,
            "comp_entropy": ...,
            "comp_active": ...,
            "comp_residual": ...,
            "group_gate": ...,
            "bg_prob": ...,
        }
```

### 关键限制

#### 必须保留

- `rbar = r - mean(r)`
- `z = z0 + a * delta * tanh(rbar)`

#### 不要做

- 不要让这个头去替换整个 Hungarian / matching logic
- 不要默认学 background/null
- 不要全体候选 always-on 改分

### 原因

你现在最需要的是：

**一个安全型局部重排序头**，不是一个重写整个 tracker 的大模型。

---

## 4.7 新增 `src/scripts/train_runtime_rerank_baseline.py`

这个文件非常重要，优先级甚至高于深度模型 trainer。

### 作用

先训练一个简单 baseline：

- Logistic regression
- LightGBM / GBDT

只吃标量特征：

- `base_score`
- `motion_score`
- `track_gap`
- `track_hist_len`
- `base_margin`
- `base_entropy`
- box delta

### 为什么必须先做这个

因为你要先回答：

**这个方向到底是“训练对象终于对了”，还是“其实一个简单 rerank 就够了”？**

如果简单 baseline 都没有 gain：
- 说明候选集本身不可恢复；
- 或 host 不给学习器机会。

如果简单 baseline 就有 gain：
- 说明 runtime replay 方向是对的；
- 再上深度模型才有意义。

---

## 4.8 新增 `src/scripts/train_mtcr_from_runtime_replay.py`

这是新的主 trainer。

### 旧 trainer 为什么不能硬改

`train_mtcr_from_gt_tracks.py` 的问题不是几行逻辑，而是整个输入对象错了：

- `det_feat/hist_feat/ctx_feat/group_id/label` 都来自 GT pseudo-track builder；
- `_group_batch()` 默认 `anchor_scores` 直接取旧 `ctx_feat`；
- `_fit_margin_threshold()` 的 margin 也是基于旧代理分组统计；
- `_group_loss()` 的意义建立在旧对象之上。

所以：**直接新写。**

### 新 trainer 的输入单位

输入不再是 pair，而是：

**一个 detection-centered candidate group**

### 新 trainer 的结构

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from models.runtime_replay_assoc import RuntimeReplayAssociationAdapter


def parse_args():
    ...


def load_runtime_group_shards(...):
    ...


def group_to_tensors(group, device):
    return det_feat, hist_feat, hist_mask, anchor_scores, motion_scores, det_score, track_gaps, labels


def group_loss(outputs, labels, margin_threshold, temperature, duel_margin):
    # 1) listwise CE
    # 2) ambiguous group 上的 duel loss
    # 3) easy/background 上的 safe loss
    ...


def run_epoch(...):
    ...


def export_checkpoint(...):
    ...


def main():
    ...
```

### loss 必须这么写

#### 1. listwise CE

```python
L_list = CE(z_group, gt_index)
```

#### 2. duel margin（只在 ambiguous positive groups）

```python
L_duel = max(0, gamma - z_pos + max(z_neg))
```

#### 3. safe loss（只在 easy/background groups）

```python
L_safe = mean((z - z0)^2)
```

#### 4. 最终 loss

```python
L = L_list + lambda_duel * L_duel + lambda_safe * L_safe
```

### 新 trainer 必须记录的指标

每个 epoch 输出：

- `base_top1`
- `final_top1`
- `top1_gain`
- `amb_top1_gain`
- `easy_top1_gain`
- `recoverable_groups`
- `recovered_groups`
- `bg_suppression`
- `easy_shift_mean`
- `active_rate`

### 原因

新 trainer 的目标不是“loss 降了没”，而是：

**在 runtime candidate groups 上，是否真实修复了 base 排序错误。**

---

## 4.9 新增 `src/scripts/eval_runtime_rerank_ablation.py`

### 作用

给论文直接产出 on/off ablation：

- base
- heuristic
- runtime baseline rerank
- runtime learned rerank

### 输入

- frozen host profile
- rerank checkpoint
- split / benchmark / profile name

### 输出

- 总表 json
- per-sequence csv
- ambiguity subset csv
- easy/background subset csv

### 原因

论文最重要的不是训练日志，而是：

**固定 host 上，plugin on/off 是否成立。**

---

# 5. 训练与实验执行顺序（必须按顺序，不要跳）

## 阶段 1：先判定方向有没有机会

### 任务

1. 改 `runtime_tracker_bytetrack.py`，完成新的 dump
2. 改 `build_runtime_assoc_replay_labels.py`
3. 新建 `build_runtime_assoc_group_shards.py`
4. 先跑 `train_runtime_rerank_baseline.py`

### 通过标准

至少看到：

- recoverable groups 不少；
- baseline rerank 在 frozen host 上比 base top1 有正增益；
- ambiguous groups 上有改进，easy groups 基本不伤。

### 失败标准

如果连简单 baseline 都没 gain：

- 先不要训练深度模型；
- 先检查候选生成 / host / ReID；
- learned association 暂停。

---

## 阶段 2：再上 learned runtime rerank

### 任务

1. 新建 `runtime_replay_assoc.py`
2. 新建 `train_mtcr_from_runtime_replay.py`
3. 跑 frozen host on/off
4. 输出 ambiguity-stratified 指标

### 通过标准

至少满足一条：

- `+0.3 HOTA` 左右；
- 或 `+0.5 AssA / IDF1` 左右；
- 且 easy cases 不明显退化。

---

## 阶段 3：再做第二 host / 第二 benchmark

### 目的

回答 reviewer：

- 这不是某个 host 上的偶然 patch；
- 这不是只在一个 detector profile 上有用。

---

# 6. 论文实验矩阵（最小可发表版）

## Mandatory

1. Frozen Host A（public-comparable）
   - base
   - heuristic
   - runtime baseline rerank
   - runtime learned rerank

2. Frozen Host B（external-det strong host）
   - 同上四行

3. Training-object ablation
   - GT pseudo-group trained
   - runtime-replay trained

4. Recoverability diagnostics
   - positive-in-topK rate
   - recoverable-group rate
   - ambiguous-group fraction
   - recovered-IDSW-like subset

5. Safety ablation
   - ambiguity trigger on/off
   - bounded residual on/off
   - zero-sum residual on/off

## Strongly recommended

1. 第二 benchmark（MOT20 / DanceTrack）
2. 第二 host
3. simple baseline vs learned baseline
4. 多次 seed 重复

## Optional

1. external-det ceiling 扩展
2. offline post-linking appendix
3. 强 ReID appendix

---

# 7. reviewer attack 点与对策（写论文时一起准备）

## Attack 1：你没有真正超过 base host

### 反制证据

- frozen host 上 plugin on/off
- ambiguous groups 分层
- per-sequence 稳定增益

## Attack 2：新意太弱，只是 another score calibrator

### 反制证据

- 训练对象从 GT proxy 变成 runtime replay object
- 只在 ambiguity groups 激活
- bounded + zero-sum rerank
- 不碰 easy cases

## Attack 3：你的 strongest line 不 protocol-clean

### 反制证据

- public-comparable 主表
- external-det 只作为第二节

## Attack 4：深度模型没必要，GBDT 就够了

### 反制证据

- 跑 baseline rerank
- 再证明 learned 在第二 host / hard groups 上更强

## Attack 5：host/ReID 才是主贡献，不是 association

### 反制证据

- frozen detector
- frozen ReID
- frozen thresholds
- plugin only

---

# 8. 近几年最相关论文与我们应该借的思想

这一节是给你“论文方向想清楚”用的，不是让你照抄。

## 8.1 MAA / MAATrack（WACV 2022 Workshop）

### 借什么思想

- 把 ambiguous assignments 单独挑出来处理；
- 不要假设所有匹配都值得用同一逻辑解决。

### 我们和它的区别

- 它更偏 heuristic ambiguity handling；
- 我们做的是 **runtime-replay-trained learned rerank**。

## 8.2 Deep OC-SORT（2023）

### 借什么思想

- 在强 host 上做自适应 refinement；
- 不要 always-on 地全局加 appearance/learned correction。

### 我们和它的区别

- 它偏 heuristic adaptive integration；
- 我们强调 **真实运行时 group replay + top-K selective rerank**。

## 8.3 GeneralTrack（CVPR 2024）

### 借什么思想

- 从 point-wise relation 走向 instance-wise relation；
- 不要只盯单一 pair score。

### 我们和它的区别

- 它是更 general 的关系建模；
- 我们是 tracking-by-detection host 上的局部 plug-in。

## 8.4 MOTIP（CVPR 2025）

### 借什么思想

- 训练对象必须贴近真实推理对象；
- association 不一定非得是 surrogate pair classification。

### 我们和它的区别

- 它把 MOT 直接改写成 ID prediction；
- 我们不推翻 host，只做局部 runtime replay rerank。

## 8.5 TrackTrack（CVPR 2025）

### 借什么思想

- local candidate competition 值得单独建模；
- 不是所有错误都能被全局 Hungarian 自然修好。

### 我们和它的区别

- 它是 track-perspective association；
- 我们是 detection-centered ambiguity-triggered rerank。

## 8.6 LPC-MOT（CVPR 2021）

### 借什么思想

- 关联可以视作 proposal generation / proposal scoring / inference；
- 排序候选假设本身是合理研究对象。

### 我们和它的区别

- 它更像 proposal-classifier 范式；
- 我们更轻量，更适合插在强 host 上。

---

# 9. 最后一句硬结论

从现在开始，你不该再问：

**“怎么把旧 MTCR 再调好一点？”**

你应该执行的是：

**把主训练对象改成真实 runtime candidate groups，把 learned module 降级成安全型 ambiguity-aware rerank plug-in，并用 frozen-host on/off 实验来证明它的价值。**

这才是当前项目还有论文生命力的唯一主线。

---

# 10. 实施优先级（按今天开始的顺序）

## 本周必须完成

1. 改 `runtime_tracker_bytetrack.py` 的 dump 逻辑
2. 改 `build_runtime_assoc_replay_labels.py`
3. 新建 `build_runtime_assoc_group_shards.py`
4. 跑出第一版 recoverability summary
5. 跑 `train_runtime_rerank_baseline.py`

## baseline 如果有效，再做

6. 新建 `runtime_replay_assoc.py`
7. 新建 `train_mtcr_from_runtime_replay.py`
8. 跑 frozen host on/off

## 如果 baseline 都无效

9. 停 learned association
10. 转 host / ReID 主线
