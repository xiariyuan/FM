# Round1 训练前最终检查清单（已实现GPT的5点补充建议）

## ✅ 所有改进已完成（Claude + GPT综合）

### 核心改进（之前完成）
1. ✅ 硬断言系统（训练端 + 评估端）
2. ✅ VAL_SEQUENCES分离（15训练 + 6验证）
3. ✅ 确定性验证transforms
4. ✅ 序列日志硬打印
5. ✅ Config按最小改动原则调整

### GPT补充建议（全部完成）✨

#### 1. ✅ 保存有效配置用于复现
**实现位置**: `train.py`

```python
# 保存最终生效的config
config_save_path = os.path.join(outputs_dir, "config_effective.yaml")
yaml.dump(config, f)

# 保存git commit
git_commit_path = os.path.join(outputs_dir, "git_commit.txt")
```

**效果**:
- `outputs/round1_minimal_change/config_effective.yaml` - 完整配置
- `outputs/round1_minimal_change/git_commit.txt` - 代码版本

#### 2. ✅ 双阈值评估（诊断 vs 默认）
**实现位置**: `train.py`

每次评估运行两套阈值：
- **诊断阈值**: DET=0.1, NEWBORN=0.0, ID=0.0（高召回，避免gating干扰）
- **默认阈值**: 配置中的实际阈值

输出目录分别为：
- `epoch_{N}_diagthr/` - 诊断阈值结果
- `epoch_{N}_default/` - 默认阈值结果

**配置开关**:
```yaml
EVAL_DIAGNOSTIC_THRESHOLDS: True  # 默认启用
```

#### 3. ✅ 诊断统计打印
**实现位置**: `submit_and_evaluate.py`

每次评估后打印：
```
[DIAG][VAL] det_before=xxxxx det_after=yyyyy keep=xx% unknown=xx% avg_det/frame=xx.x
```

**含义**:
- `det_before`: Score过滤后的检测数
- `det_after`: Newborn过滤后的保留数
- `keep`: 保留率（诊断gating是否过强）
- `unknown`: Unknown比例（诊断ID学习稳定性）
- `avg_det/frame`: 每帧平均检测数

#### 4. ✅ Best-on-Val checkpoint保存
**实现位置**: `train.py`

自动追踪验证集最佳MOTA：
```python
best_val_metric = {
    "epoch": -1,
    "MOTA": -inf,
    "IDF1": -inf,
    "HOTA": -inf,
}
```

当验证集MOTA提升时：
- 保存 `checkpoint_best.pth`
- 记录epoch和所有指标
- 日志打印：`[Best Model] Saved new best model at epoch X: MOTA=xx.xx%`

#### 5. ✅ 原始增强配置对照
**实现位置**: `configs/round1_baseline_contrast.yaml`

快速对照实验（只跑2 epoch）：
```yaml
# 原始高强度增强
AUG_TRAJECTORY_OCCLUSION_PROB: 0.5    # vs 0.1
AUG_TRAJECTORY_SWITCH_PROB: 0.5       # vs 0.0
BBOX_POSITION_NOISE: 0.02             # vs 0.0
BBOX_SIZE_NOISE: 0.03                 # vs 0.0
BBOX_AUG_PROB: 0.5                    # vs 0.0

EPOCHS: 2  # 快速对照
```

**目的**: 早期就能看到 unknown_ratio / det_after_filter 差异

## 🚀 启动训练（在80G显卡上）

### 主实验：Round1 Minimal Change
```bash
cd /gemini/code/FMtrack-main/FM-Track
mkdir -p logs

# 后台训练（40 epochs）
nohup python train.py --config-path configs/round1_minimal_change.yaml \
  > logs/round1_minimal_change.log 2>&1 &
echo $! > logs/round1_pid.txt
```

### 对照实验：Original Augmentation Baseline（可选）
```bash
# 快速对照（2 epochs）
nohup python train.py --config-path configs/round1_baseline_contrast.yaml \
  > logs/round1_baseline_contrast.log 2>&1 &
echo $! > logs/round1_contrast_pid.txt
```

## 🔍 启动后必看日志（GPT要求）

### 1. 训练启动时（前100行）
```bash
head -100 logs/round1_minimal_change.log | grep "Confirmed\|config_effective\|git"
```

**必须看到**:
```
Saved effective config to: ./outputs/round1_minimal_change/config_effective.yaml
Git commit: xxxxxxxx
✅ Confirmed: VAL_SEQUENCES are excluded from TRAIN dataset.
```

### 2. 每个epoch评估时
```bash
tail -200 logs/round1_minimal_change.log | grep "Eval\|DIAG\|Best"
```

**应该看到**:
```
[Eval] Running with Diagnostic (high-recall) thresholds: DET=0.1, NEWBORN=0.0, ID=0.0
[Eval epoch: X] [Diagnostic (high-recall)]  ...
✅ Confirmed: Eval sequences are restricted to VAL_SEQUENCES=['MOT17-02', 'MOT17-10']
[DIAG][VAL] det_before=xxxxx det_after=yyyyy keep=xx% unknown=xx% avg_det/frame=xx.x

[Eval] Running with Default thresholds: DET=0.3, NEWBORN=0.6, ID=0.2
[Eval epoch: X] [Default]  MOTA=xx.xx%, IDF1=xx.xx%, HOTA=xx.xx%
[Best Model] Saved new best model at epoch X: MOTA=xx.xx%, ...
```

## 📊 关键指标观察

### 诊断阈值 vs 默认阈值对比
- **如果 diagthr MOTA >> default MOTA**: Gating太强，砍掉了大量有效检测
- **如果两者接近**: Gating合理，问题在检测质量或关联

### 诊断统计观察
- **keep% 下降**: Newborn filter太强
- **unknown% 高**: ID学习不稳定，模型无法识别目标
- **avg_det/frame 低**: 检测召回本身就差

### Best model追踪
- 训练结束后使用 `checkpoint_best.pth` 而非 `checkpoint_39.pth`
- 防止选到过拟合的last epoch

## 📁 训练产出

```
outputs/round1_minimal_change/
├── config_effective.yaml          # 最终生效配置（复现用）
├── git_commit.txt                 # 代码版本
├── checkpoint_0.pth               # 每个epoch
├── checkpoint_1.pth
├── ...
├── checkpoint_best.pth            # ⭐ 验证集最佳模型
└── train/
    └── eval_during_train/
        ├── epoch_0_diagthr/       # 诊断阈值评估
        ├── epoch_0_default/       # 默认阈值评估
        ├── epoch_1_diagthr/
        ├── epoch_1_default/
        └── ...
```

## 🎓 GPT核心建议总结

1. **最小改动** + **双阈值** + **诊断统计** = 快速定位问题根源
2. **Hard assert** 确保实验边界干净
3. **Best-on-val** 避免选错模型
4. **Config dump** 确保可复现
5. **对照实验** 增强结论可信度

## ✅ 最终确认

- [x] 硬断言已添加（训练端 + 评估端）
- [x] Config符合最小改动原则
- [x] VAL_SEQUENCES正确配置
- [x] 确定性transforms已修复
- [x] **Config自动保存**（GPT建议1）
- [x] **双阈值评估**（GPT建议2）
- [x] **诊断统计打印**（GPT建议3）
- [x] **Best-on-val保存**（GPT建议4）
- [x] **对照配置创建**（GPT建议5）

**一切就绪！可以在80G显卡上开始训练！**
