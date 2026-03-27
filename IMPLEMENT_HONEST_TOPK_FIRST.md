# Honest-TopK 第一刀实施说明（按文件 / 按行号 / 按修改前后）

适用包：`gpt_pro_runtime_replay_followup_plus_20260321`

目标：不是继续大改结构，而是先做一版**信息价值最高**的诊断性改造：

1. 去掉训练时的 **positive 强塞回 top-K**；
2. 去掉 **hardfocus warm start**；
3. 去掉 **teacher distill**；
4. 保留现有 runtime-replay 模型主体，只改成更 **honest runtime** 的训练；
5. 用现有 `proxy epoch sweep` 继续选 online best epoch。

这版不是最终论文版，而是**判断这条 learned plugin 线是否还有真实 headroom** 的第一刀。

---

## 0. 这次先只改 4 个文件

### 必改

1. `scripts/train_runtime_replay_reranker.py`
2. `models/runtime_replay_assoc.py`
3. `scripts/run_runtime_replay_onlinealign_relaxed_longhaul.sh`
4. `PROJECT_STATUS.md`

### 新增

5. `scripts/run_runtime_replay_honest_topk_longhaul.sh`

> 先不要同时大改别的文件。你现在最需要的是一个**干净、可归因**的 A/B 诊断 run。

---

## 1. 修改 `scripts/train_runtime_replay_reranker.py`

---

### 修改块 1：给 trainer 增加 3 个新参数

**位置：第 78–137 行，`parse_args()` 内**

#### 修改前

当前这段末尾是：

```python
    ap.add_argument("--sample-ambiguous-weight", type=float, default=3.0)
    ap.add_argument("--sample-hard-positive-weight", type=float, default=8.0)
    ap.add_argument("--sample-easy-weight", type=float, default=0.35)
    ap.add_argument("--sample-background-weight", type=float, default=0.5)
    ap.add_argument("--sample-groups-per-shard", type=int, default=96)
    return ap.parse_args()
```

#### 修改后

替换为：

```python
    ap.add_argument("--sample-ambiguous-weight", type=float, default=3.0)
    ap.add_argument("--sample-hard-positive-weight", type=float, default=8.0)
    ap.add_argument("--sample-easy-weight", type=float, default=0.35)
    ap.add_argument("--sample-background-weight", type=float, default=0.5)
    ap.add_argument("--sample-groups-per-shard", type=int, default=96)

    # new: make training candidate selection faithful to runtime top-k
    ap.add_argument("--honest-topk", action="store_true",
                    help="Do not force-include the positive into top-k during training/validation.")
    # new: allow turning off teacher distillation cleanly
    ap.add_argument("--disable-distill", action="store_true",
                    help="Ignore teacher_score distillation even if teacher_score exists in shards.")
    # new: useful for logging / reproducibility
    ap.add_argument("--experiment-tag", default="",
                    help="Free-form tag written into metrics/checkpoints for bookkeeping.")
    return ap.parse_args()
```

#### 原因

你现在的 trainer 没有一个显式开关来控制“训练候选集是否忠实于 runtime top-K”。这正是当前最核心的问题。

---

### 修改块 2：把 `_select_candidates()` 改成 honest-topK / legacy-topK 双模式

**位置：第 326–353 行，函数 `_select_candidates`**

#### 修改前

```python
def _select_candidates(shard: PreparedShard, group_index: int, topk: int, valid_only: bool) -> np.ndarray:
    start = int(shard.group_offsets[group_index])
    end = int(shard.group_offsets[group_index + 1])
    idx = np.arange(start, end, dtype=np.int64)
    if idx.size == 0:
        return idx
    if valid_only:
        idx = idx[shard.valid_train_row[idx] > 0]
        if idx.size == 0:
            return idx
    if shard.rank_score_col == "base_score":
        rank_scores = shard.base_score[idx]
    else:
        rank_scores = shard.refined_score[idx]
    order = np.argsort(-rank_scores, kind="mergesort")
    idx = idx[order]
    if topk > 0 and idx.size > topk:
        keep = idx[:topk].tolist()
        label_mask = shard.label[idx] > 0
        pos_indices = idx[label_mask]
        if pos_indices.size > 0 and int(pos_indices[0]) not in keep:
            keep[-1] = int(pos_indices[0])
        if shard.rank_score_col == "base_score":
            keep = sorted(set(keep), key=lambda x: (-float(shard.base_score[x]), int(shard.track_rank[x])))
        else:
            keep = sorted(set(keep), key=lambda x: (-float(shard.refined_score[x]), int(shard.track_rank[x])))
        idx = np.asarray(keep, dtype=np.int64)
    return idx
```

#### 修改后

把函数签名和函数体一起替换成：

```python
def _select_candidates(
    shard: PreparedShard,
    group_index: int,
    topk: int,
    valid_only: bool,
    honest_topk: bool,
) -> np.ndarray:
    start = int(shard.group_offsets[group_index])
    end = int(shard.group_offsets[group_index + 1])
    idx = np.arange(start, end, dtype=np.int64)
    if idx.size == 0:
        return idx

    if valid_only:
        idx = idx[shard.valid_train_row[idx] > 0]
        if idx.size == 0:
            return idx

    if shard.rank_score_col == "base_score":
        rank_scores = shard.base_score[idx]
    else:
        rank_scores = shard.refined_score[idx]

    order = np.argsort(-rank_scores, kind="mergesort")
    idx = idx[order]

    if topk > 0 and idx.size > topk:
        if honest_topk:
            idx = idx[:topk]
        else:
            # backward-compatible legacy path
            keep = idx[:topk].tolist()
            label_mask = shard.label[idx] > 0
            pos_indices = idx[label_mask]
            if pos_indices.size > 0 and int(pos_indices[0]) not in keep:
                keep[-1] = int(pos_indices[0])
            if shard.rank_score_col == "base_score":
                keep = sorted(set(keep), key=lambda x: (-float(shard.base_score[x]), int(shard.track_rank[x])))
            else:
                keep = sorted(set(keep), key=lambda x: (-float(shard.refined_score[x]), int(shard.track_rank[x])))
            idx = np.asarray(keep, dtype=np.int64)

    return idx
```

#### 原因

这一刀就是整个第一版实验的核心：

- `honest_topk=True` 时，训练看到的就是 runtime 真正会看到的 top-K；
- `honest_topk=False` 时，保留旧行为，方便回归对照。

---

### 修改块 3：让 `_collate_batch()` 传入 `honest_topk`

**位置：第 356–363 行，函数 `_collate_batch()` 开头**

#### 修改前

```python
def _collate_batch(
    shard: PreparedShard,
    group_indices: list[int],
    topk: int,
    device: torch.device,
    valid_only: bool,
) -> dict[str, torch.Tensor]:
    selected_all = [_select_candidates(shard, group_idx, topk=topk, valid_only=valid_only) for group_idx in group_indices]
```

#### 修改后

```python
def _collate_batch(
    shard: PreparedShard,
    group_indices: list[int],
    topk: int,
    device: torch.device,
    valid_only: bool,
    honest_topk: bool,
) -> dict[str, torch.Tensor]:
    selected_all = [
        _select_candidates(
            shard,
            group_idx,
            topk=topk,
            valid_only=valid_only,
            honest_topk=honest_topk,
        )
        for group_idx in group_indices
    ]
```

#### 继续修改

搜索 `_collate_batch(` 的调用点，在 `_run_epoch(...)` 里把调用补成：

```python
batch = _collate_batch(
    shard=shard,
    group_indices=batch_group_indices,
    topk=int(args.topk),
    device=device,
    valid_only=bool(args.valid_only),
    honest_topk=bool(args.honest_topk),
)
```

> 你本地直接 `grep -n "_collate_batch(" -n scripts/train_runtime_replay_reranker.py` 找调用点即可。

---

### 修改块 4：把 `positive_in_topk` 重新按“当前选出的 honest candidates”计算

**位置：第 411–442 行，`_collate_batch()` 的 group 填充循环内**

#### 修改前

当前这里直接抄 shard 里的旧字段：

```python
        positive_in_topk[batch_idx] = bool(shard.positive_in_topk[group_idx] > 0)
```

#### 修改后

把这一行替换为：

```python
        positive_in_topk[batch_idx] = bool(np.any(shard.label[cand_idx] > 0))
```

#### 原因

shard 里保存的 `positive_in_topk` 是 dump 时的统计口径，不一定等于你现在 train-time `_select_candidates()` 真正拿到的 cand set。

第一版 honest-topK 诊断必须以**当前实际送进模型的候选集**为准。

---

### 修改块 5：distill loss 增加总开关

**位置：第 575–586 行，`_compute_losses()` 内 distill loss 段**

#### 修改前

```python
    teacher_score = batch["teacher_score"].clamp(min=1e-4, max=1.0 - 1e-4)
    teacher_logits = torch.logit(teacher_score, eps=1e-4)
    teacher_logits = teacher_logits.masked_fill(~supervised_mask, -1e9)
    student_logits = candidate_logits.masked_fill(~supervised_mask, -1e9)
    teacher_available = supervised_mask.any(dim=1) & (teacher_score.max(dim=1).values > 1e-4)
    distill_loss = torch.zeros_like(list_loss)
    if bool(torch.any(teacher_available)):
        rows = torch.nonzero(teacher_available, as_tuple=False).view(-1)
        t = max(float(args.distill_temperature), 1e-4)
        student_logp = torch.log_softmax(student_logits[rows] / t, dim=1)
        teacher_prob = torch.softmax(teacher_logits[rows] / t, dim=1)
        distill_loss[rows] = F.kl_div(student_logp, teacher_prob, reduction="none").sum(dim=1) * (t * t)
```

#### 修改后

```python
    distill_loss = torch.zeros_like(list_loss)
    if not bool(args.disable_distill) and float(args.loss_distill_weight) > 0.0:
        teacher_score = batch["teacher_score"].clamp(min=1e-4, max=1.0 - 1.0e-4)
        teacher_logits = torch.logit(teacher_score, eps=1e-4)
        teacher_logits = teacher_logits.masked_fill(~supervised_mask, -1e9)
        student_logits = candidate_logits.masked_fill(~supervised_mask, -1e9)
        teacher_available = supervised_mask.any(dim=1) & (teacher_score.max(dim=1).values > 1e-4)
        if bool(torch.any(teacher_available)):
            rows = torch.nonzero(teacher_available, as_tuple=False).view(-1)
            t = max(float(args.distill_temperature), 1e-4)
            student_logp = torch.log_softmax(student_logits[rows] / t, dim=1)
            teacher_prob = torch.softmax(teacher_logits[rows] / t, dim=1)
            distill_loss[rows] = F.kl_div(student_logp, teacher_prob, reduction="none").sum(dim=1) * (t * t)
```

#### 原因

第一版 honest-topK run 的目的，是看当前模型自己在更真实的训练任务下还有多少 online 增益。

所以 teacher distill 先关掉，避免继续把 hardfocus / previous replay 的 bias 带进来。

---

### 修改块 6：在 metrics jsonl 里写入 honest-topK / disable-distill 标签

**位置：第 1049–1056 行，`_append_jsonl(...)` 的 payload**

#### 修改前

```python
            _append_jsonl(
                metrics_path,
                {
                    "epoch": int(epoch),
                    "selection_score": float(selection),
                    "train": train_metrics,
                    "val": val_metrics,
                },
            )
```

#### 修改后

```python
            _append_jsonl(
                metrics_path,
                {
                    "epoch": int(epoch),
                    "selection_score": float(selection),
                    "experiment_tag": str(args.experiment_tag or ""),
                    "honest_topk": bool(args.honest_topk),
                    "disable_distill": bool(args.disable_distill),
                    "train": train_metrics,
                    "val": val_metrics,
                },
            )
```

#### 原因

你后面会同时有 legacy / honest / teacher-on / teacher-off 多条 run，没有这个字段，日志会非常难回看。

---

## 2. 修改 `models/runtime_replay_assoc.py`

这次先只做一个**非常小但必要**的改动：

> 不改变主体架构；只把 `candidate` 与 `null` 的 joint 计算从“先缩 candidate score，再取 log”改成“直接在 logits 空间做 joint softmax”。

这是为了减少当前 `null_prob` 直接缩放所有 candidate 的 calibration 扭曲。

---

### 修改块 1：替换 null / candidate joint logits 构造

**位置：第 503–512 行**

#### 修改前

```python
        residual = residual.masked_fill(~valid_mask, 0.0)
        residual_center = residual - _masked_mean(residual, valid_mask, dim=1, keepdim=True)
        final_logits = anchor_logits + group_activation.unsqueeze(-1) * beta * delta_cap.unsqueeze(-1) * torch.tanh(residual_center)
        final_logits = _masked_fill_logits(final_logits, valid_mask, fill=-20.0)
        null_prob = torch.sigmoid(null_logit)
        final_scores = (1.0 - null_prob.unsqueeze(-1)) * torch.sigmoid(final_logits)
        final_scores = final_scores * valid_mask.to(dtype=final_scores.dtype)
        candidate_logprob = torch.log(final_scores.clamp(min=1e-8))
        candidate_logprob = _masked_fill_logits(candidate_logprob, valid_mask, fill=-1e9)
        joint_logits = torch.cat([candidate_logprob, torch.log(null_prob.clamp(min=1e-8)).unsqueeze(-1)], dim=1)
```

#### 修改后

```python
        residual = residual.masked_fill(~valid_mask, 0.0)
        residual_center = residual - _masked_mean(residual, valid_mask, dim=1, keepdim=True)
        final_logits = anchor_logits + group_activation.unsqueeze(-1) * beta * delta_cap.unsqueeze(-1) * torch.tanh(residual_center)
        final_logits = _masked_fill_logits(final_logits, valid_mask, fill=-20.0)

        # honest joint decision in logit space: candidates compete with null directly
        joint_logits = torch.cat([final_logits, null_logit.unsqueeze(-1)], dim=1)
        joint_logprob = torch.log_softmax(joint_logits, dim=1)

        candidate_logprob = joint_logprob[:, :-1]
        null_logprob = joint_logprob[:, -1]
        final_scores = torch.exp(candidate_logprob) * valid_mask.to(dtype=final_logits.dtype)
        null_prob = torch.exp(null_logprob)
```

#### 然后继续改 return 字典

**位置：第 514–533 行返回字典中**

原来：

```python
            "null_logit": null_logit,
            "null_prob": null_prob,
```

改成：

```python
            "null_logit": null_logit,
            "null_prob": null_prob,
            "joint_logprob": joint_logprob,
```

#### 原因

这一刀不是终极 redesign，只是为了先把 current architecture 里一个很可能误导训练的点去掉：

- 旧写法里 `null_prob` 会整体压缩全部 candidate；
- 新写法里 `candidate` 和 `null` 是直接竞争关系。

这更符合 “选某个 candidate vs 选 null” 的训练目标。

---

## 3. 新增 `scripts/run_runtime_replay_honest_topk_longhaul.sh`

这是你接下来真正该跑的第一版主诊断脚本。

### 新文件内容

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

TAG="${1:-$(date +%Y%m%d_%H%M%S)}"

SHARD_ROOT="${SHARD_ROOT:-${REPO_ROOT}/outputs/runtime_replay_shards_sw_yolox_base_full7_20260318_full_learned_basefull7_fixreview}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/runtime_replay_honest_topk_longhaul_${TAG}}"
TRAIN_OUT="${OUT_ROOT}/train"
CKPT_PATH="${TRAIN_OUT}/runtime_replay_honest_topk.pt"
METRICS_PATH="${TRAIN_OUT}/runtime_replay_honest_topk.metrics.jsonl"
PROXY_SWEEP_OUT="${OUT_ROOT}/proxy_epoch_sweep"
LOG_PATH="${OUT_ROOT}/longhaul.log"
STATUS_PATH="${OUT_ROOT}/job_status.txt"
PID_PATH="${OUT_ROOT}/job_pid.txt"

mkdir -p "${TRAIN_OUT}" "${PROXY_SWEEP_OUT}"

echo "$$" > "${PID_PATH}"
echo "running" > "${STATUS_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1
trap 'status=$?; echo "[honest-topk] exit_code=${status} finished_at=$(date --iso-8601=seconds)"; echo "${status}" > "${STATUS_PATH}"' EXIT

echo "[honest-topk] tag=${TAG}"
echo "[honest-topk] shard_root=${SHARD_ROOT}"
echo "[honest-topk] out_root=${OUT_ROOT}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_runtime_replay_reranker.py" \
  --input-dir "${SHARD_ROOT}" \
  --out-ckpt "${CKPT_PATH}" \
  --metrics-path "${METRICS_PATH}" \
  --device cuda \
  --epochs 10 \
  --batch-groups 24 \
  --topk 5 \
  --valid-only \
  --fixed-val-sample \
  --honest-topk \
  --disable-distill \
  --loss-distill-weight 0.0 \
  --experiment-tag honest_topk_v1 \
  --train-seqs MOT17-04-FRCNN,MOT17-05-FRCNN,MOT17-09-FRCNN,MOT17-10-FRCNN,MOT17-11-FRCNN \
  --val-seqs MOT17-02-FRCNN,MOT17-13-FRCNN \
  --train-groups-per-epoch 4096 \
  --val-groups-per-epoch 2048 \
  --sample-groups-per-shard 128 \
  --sample-hard-positive-weight 8.0 \
  --sample-ambiguous-weight 3.5 \
  --sample-easy-weight 0.35 \
  --sample-background-weight 0.55 \
  --hard-positive-weight 4.0 \
  --ambiguous-weight 2.2 \
  --easy-weight 0.5 \
  --background-weight 0.75 \
  --loss-gate-weight 0.04 \
  --gate-positive-target 0.20 \
  --select-hard-weight 2.4 \
  --select-bg-weight 0.25 \
  --select-easy-weight 0.10

echo "[honest-topk] proxy sweep"
bash "${REPO_ROOT}/scripts/run_runtime_replay_proxy_epoch_sweep.sh" \
  sw_yolox "${TRAIN_OUT}" "${PROXY_SWEEP_OUT}" 1

echo "[honest-topk] done best ckpt under ${PROXY_SWEEP_OUT}"
```

### 原因

这版脚本故意做三件事：

- **不加载 `INIT_CKPT`**；
- **不开 distill**；
- 训练完后直接跑 **proxy epoch sweep + best epoch full7**。

也就是：

> 让模型在 honest runtime 训练条件下，自己证明自己是否还有真实 online gain。

---

## 4. 修改 `scripts/run_runtime_replay_onlinealign_relaxed_longhaul.sh`

这个文件不要删除，但要改成“legacy 对照脚本”，避免你后面自己混淆。

---

### 修改块 1：重命名日志 tag 提示

**位置：第 28–32 行附近**

#### 修改前

```bash
echo "[onlinealign_relaxed] tag=${TAG}"
echo "[onlinealign_relaxed] shard_root=${SHARD_ROOT}"
echo "[onlinealign_relaxed] init_ckpt=${INIT_CKPT}"
echo "[onlinealign_relaxed] out_root=${OUT_ROOT}"
echo "[onlinealign_relaxed] log_path=${LOG_PATH}"
```

#### 修改后

```bash
echo "[onlinealign_relaxed_legacy] tag=${TAG}"
echo "[onlinealign_relaxed_legacy] shard_root=${SHARD_ROOT}"
echo "[onlinealign_relaxed_legacy] init_ckpt=${INIT_CKPT}"
echo "[onlinealign_relaxed_legacy] out_root=${OUT_ROOT}"
echo "[onlinealign_relaxed_legacy] log_path=${LOG_PATH}"
```

### 修改块 2：训练命令显式写明 legacy 设置

**位置：第 34–62 行训练命令**

在原命令末尾追加这几项：

```bash
  --experiment-tag onlinealign_relaxed_legacy \
  --loss-distill-weight 0.25
```

> 不需要加 `--honest-topk`，也不要加 `--disable-distill`。

#### 原因

这样你后面比较 honest-topK 和 legacy，日志里一眼能看出来。

---

## 5. 修改 `PROJECT_STATUS.md`

在状态文件里新增一个小节，明确把下一刀实验定义下来，不然你后面很容易继续在旧路线上漂。

### 追加到文末

```md
## Next mandatory diagnostic run: honest-topK / no-distill / no-hardfocus-init

Purpose:
- test whether the current learned runtime-replay plugin still has real online headroom
  once training-time positive injection and teacher bias are removed.

Exact settings:
- trainer: `scripts/train_runtime_replay_reranker.py`
- candidate selection: `--honest-topk`
- init: none
- distill: off (`--disable-distill --loss-distill-weight 0.0`)
- model family: keep current `models/runtime_replay_assoc.py` backbone
- model selection: existing `scripts/run_runtime_replay_proxy_epoch_sweep.sh`

Primary stop/go criterion:
- if honest-topK best-epoch full7 still beats previous replay baseline by a stable margin,
  continue to redesign the plugin as a true runtime decision module;
- otherwise, demote association plugin work and pivot main effort to host / appearance / lifecycle.
```

---

## 6. 你改完之后，按什么顺序跑

### 第一步

先跑：

```bash
bash scripts/run_runtime_replay_honest_topk_longhaul.sh honest_topk_v1
```

### 第二步

看三个文件：

1. `outputs/runtime_replay_honest_topk_longhaul_*/train/runtime_replay_honest_topk.metrics.jsonl`
2. `outputs/runtime_replay_honest_topk_longhaul_*/proxy_epoch_sweep/epoch_proxy0213_scores.csv`
3. `outputs/runtime_replay_honest_topk_longhaul_*/proxy_epoch_sweep/best_epoch_full7/.../pedestrian_summary.txt`

### 第三步

只回答三个问题：

1. honest-topK 后，proxy best epoch 还在不在早期？
2. honest-topK best full7 还能不能稳定高于 previous replay baseline？
3. 如果 honest-topK 一开，full7 增益塌掉，是不是说明旧增益很大程度来自 training-time easier task？

---

## 7. 这一版不要做的事

这次先不要同时做下面这些：

- 不要再引入新大模型结构；
- 不要同时重写 host integration；
- 不要再从 hardfocus 初始化；
- 不要继续用 offline rerank 指标给自己讲 progress story；
- 不要同时把 background/null 大改到另一套复杂框架。

先把 **honest-topK 这一个因子** 单独测清楚。

---

## 8. 如果 honest-topK 结果仍然有真实 gain，第二刀再做什么

如果第一刀通过，再做第二刀：

1. 把 `models/runtime_replay_assoc.py` 拆成
   - intervene head
   - candidate head
   - null head
2. 加 short-horizon replay regret label
3. checkpoint selection 直接绑 proxy online HOTA

但这是第二阶段，不是现在。

---

## 9. 一句话执行摘要

你现在不是要“再调一个更好的 current line”，而是要先用一版**honest runtime training** 去验证：

> 这条 learned plugin 的 online 小增益，到底是真 headroom，还是训练任务比 runtime 更容易导致的假正信号。

这一版 MD 的全部修改，都是为这个问题服务。
