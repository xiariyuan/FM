# Round1 训练前最终检查清单

## ✅ 已完成的所有改进（GPT + Claude综合建议）

### 1. 硬断言系统（GPT强烈推荐）

**训练端断言** - `train.py`:
- ✅ 训练启动时自动检测VAL_SEQUENCES是否泄漏到训练集
- ✅ 如果检测到泄漏，立即抛出RuntimeError终止训练
- ✅ 成功时打印：`✅ Confirmed: VAL_SEQUENCES are excluded from TRAIN dataset.`

**评估端断言** - `submit_and_evaluate.py`:
- ✅ 每次评估时检测是否只在VAL_SEQUENCES上运行
- ✅ 如果检测到非VAL序列，立即抛出RuntimeError
- ✅ 成功时打印：`✅ Confirmed: Eval sequences are restricted to VAL_SEQUENCES=...`

**配置开关** - `configs/round1_minimal_change.yaml`:
```yaml
ASSERT_EVAL_ONLY_VAL: True  # 默认启用
```

### 2. Config调整（综合Claude + GPT建议）

```yaml
# 轨迹增强 - 极简对照
AUG_TRAJECTORY_OCCLUSION_PROB: 0.1    # GPT建议0.1~0.2，保留遮挡学习
AUG_TRAJECTORY_SWITCH_PROB: 0.0       # GPT建议0.0，关闭ID切换干扰

# BBox噪声 - 完全关闭
BBOX_POSITION_NOISE: 0.0
BBOX_SIZE_NOISE: 0.0
BBOX_DROP_PROB: 0.0
BBOX_AUG_PROB: 0.0

# 验证集
VAL_SEQUENCES: [MOT17-02, MOT17-10]   # 6个序列（3个检测器）

# 其他参数保持不变（最小改动原则）
LR: 1.0e-4
WEIGHT_DECAY: 0.001
DETR_DROPOUT: 0.1
```

### 3. 日志系统（GPT要求）

**训练集日志**:
```
[TRAIN] MOT17/train: 15 sequences loaded: ['MOT17-04-DPM', ...]
```

**评估集日志**:
```
[EVAL] Evaluating on 6 sequences: ['MOT17-02-DPM', 'MOT17-02-FRCNN', ...]
```

### 4. 确定性验证Transforms

- ✅ `data/transforms.py`: `build_transforms(is_validation=True)`
- ✅ 验证集：只做resize + normalize（确定性）
- ✅ 训练集：保留所有增强

### 5. 反证测试（防缓存）

- ✅ DET_THRESH=0.99测试：18837条→1264条（下降93%）
- ✅ 确认无缓存/复用问题

## 🎯 训练启动前必查3点

### ✅ 1. 配置文件就绪
```bash
cat configs/round1_minimal_change.yaml | grep -E "VAL_SEQUENCES|AUG_TRAJECTORY|BBOX_|ASSERT"
```
应该看到：
```
VAL_SEQUENCES: [MOT17-02, MOT17-10]
ASSERT_EVAL_ONLY_VAL: True
AUG_TRAJECTORY_OCCLUSION_PROB: 0.1
AUG_TRAJECTORY_SWITCH_PROB: 0.0
BBOX_POSITION_NOISE: 0.0
...
```

### ✅ 2. 代码修改就绪
```bash
# 训练端断言
grep -A 10 "Hard assert: VAL leakage" train.py | head -5

# 评估端断言
grep -A 10 "Optional hard assert: Eval only VAL" submit_and_evaluate.py | head -5
```

### ✅ 3. 验证测试通过
```bash
python test_val_split.py configs/round1_minimal_change.yaml
```
应该看到：
```
✅ PASS: No VAL sequences in training set (15 seqs)
✅ PASS: All validation sequences match VAL_SEQUENCES (6 seqs)
✅ ALL CHECKS PASSED!
```

## 🚀 启动训练（在80G显卡机器上）

```bash
cd /gemini/code/FMtrack-main/FM-Track
mkdir -p logs

# 后台训练
nohup python train.py --config-path configs/round1_minimal_change.yaml \
  > logs/round1_minimal_change.log 2>&1 &

# 保存PID
echo $! > logs/round1_pid.txt

# 立即查看前100行（检查断言）
head -100 logs/round1_minimal_change.log
```

## 🔍 启动后立即检查（必看！）

训练启动1-2分钟后，**必须**看到这3行：

```bash
head -100 logs/round1_minimal_change.log | grep "Confirmed"
```

**期望输出**:
```
✅ Confirmed: VAL_SEQUENCES are excluded from TRAIN dataset.
✅ Confirmed: Eval sequences are restricted to VAL_SEQUENCES=['MOT17-02', 'MOT17-10']
```

**如果看到 ❌ 或者没有 ✅ → 立即停止训练！**

```bash
kill $(cat logs/round1_pid.txt)
```

## 📊 训练监控

### 每个epoch后检查
```bash
# 查看最新验证结果
tail -100 logs/round1_minimal_change.log | grep -E "MOTA|IDF1|HOTA"
```

### 关键指标观察
- **MOTA on VAL**: 期望 >85% 且稳定
- **Train-Val Gap**: 应该比之前的43%（95.88% vs 52.79%）小得多
- **DetA vs AssA**: 如果DetA高但HOTA低 → 关联问题

## 📁 训练产出

- Checkpoints: `./outputs/round1_minimal_change/checkpoints/`
- 评估结果: `./outputs/round1_minimal_change/train/eval_during_train/`
- 日志: `logs/round1_minimal_change.log`

## 🎓 GPT关键建议总结

1. **最小改动原则**: 只改3件事（验证transforms + 轨迹增强 + BBox噪声）
2. **硬断言必加**: 防止"以为在验证，其实在看train"
3. **日志必看**: 训练开始必须确认 ✅ 出现
4. **隔离边界**: VAL必须held-out，不能碰训练数据

## ✅ 最终确认

- [x] 硬断言已添加（训练端 + 评估端）
- [x] Config符合最小改动原则
- [x] VAL_SEQUENCES正确配置
- [x] 确定性transforms已修复
- [x] 反证测试通过
- [x] 验证脚本测试通过
- [x] 文档已更新

**一切就绪！可以在80G显卡上开始Round1训练！**
