# P1-b（Public det + 外部 ReID）权重落地补丁说明（Claude 同款格式）

> 目标：你已经把强 ReID 权重下载到 `weights/` 目录里，现在需要把 **配置文件**从默认的 torchvision ResNet（ImageNet）切换到你下载的 **OSNet / OSNet-AIN** 等强 ReID 权重。

---

## 变更概览

- **只改 1 个文件（配置）**：
  - `configs/r50_dino_fa_mot_v2_mot17_public_reid.yaml`

> 说明：你当前的 `models/public_reid.py` 已支持 `torchreid:<model_name>` 和 `PUBLIC_REID_WEIGHTS` 加载，因此这一步只需要改 config。

---

## 1) `configs/r50_dino_fa_mot_v2_mot17_public_reid.yaml`

### 修改位置
- **行号范围**：L5–L13

### 修改前（当前版本）
```yaml
# Option A (default): torchvision backbone (ImageNet pretrained)
PUBLIC_REID_BACKBONE: torchvision_resnet50
PUBLIC_REID_PRETRAINED: True
PUBLIC_REID_WEIGHTS:

# Option B (stronger ReID, requires `pip install torchreid`):
# PUBLIC_REID_BACKBONE: torchreid:osnet_ain_x1_0
# PUBLIC_REID_PRETRAINED: True
# PUBLIC_REID_WEIGHTS:
```

### 修改后（推荐：OSNet-AIN / OSNet + 本地权重）
> 你已经把权重放在 `weights/` 下，因此推荐：
> - `PUBLIC_REID_PRETRAINED: False`（避免 torchreid 额外下载/联网；完全使用你本地权重）
> - `PUBLIC_REID_WEIGHTS: weights/<你的权重文件名>.pth`

#### 方案 B1：OSNet-AIN（更推荐，跨域更稳）
```yaml
# Option B (stronger ReID): torchreid OSNet-AIN + local weights
PUBLIC_REID_BACKBONE: torchreid:osnet_ain_x1_0
PUBLIC_REID_PRETRAINED: False
PUBLIC_REID_WEIGHTS: weights/<osnet_ain_x1_0_msmt17_256x128_*.pth>
```

#### 方案 B2：OSNet x1.0（更经典、更通用）
```yaml
# Option B (stronger ReID): torchreid OSNet + local weights
PUBLIC_REID_BACKBONE: torchreid:osnet_x1_0
PUBLIC_REID_PRETRAINED: False
PUBLIC_REID_WEIGHTS: weights/<osnet_x1_0_msmt17_256x128_*.pth>
```

> 备注：
> - 你可以把 `PUBLIC_REID_WEIGHTS` 写成 **精确文件名**（最稳），例如：
>   - `weights/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_....pth`
> - 如果你不确定文件名，建议你先在命令行跑：`ls weights/`，复制粘贴精确名字到 yaml。

---

## 建议的 sanity check（不改代码，跑 10 秒就能验证“权重生效”）

1) **启动时不报错**
- 不能出现：`PUBLIC_REID_WEIGHTS not found`

2) **embedding 不是随机/不是全 0**
- 你可以临时在 `RuntimeTrackerPublic.update()`（或 `public_reid.encode()` 返回后）打印：
  - `output_embeds.norm(dim=-1).mean().item()`
- 如果你开启了 `PUBLIC_REID_L2_NORM: True`，这个值通常会接近 1（不是 0，也不会飘得很夸张）。

3) **输入尺寸与权重匹配**
- 你这份 config 里已经是：
  - `PUBLIC_REID_INPUT_H: 256`
  - `PUBLIC_REID_INPUT_W: 128`
- 请保持与权重文件名里的 `256x128` 一致（不要改）。

---

## 下一步（你开始跑主实验时的推荐设置）

- 主表（审稿最友好）：**MOT17/MOT20 Public det + OSNet-AIN**
- 私有上限（补充实验）：再接 ByteTrack/YOLOX detections（P2）

