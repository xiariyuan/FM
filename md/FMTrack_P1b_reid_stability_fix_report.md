# P1-b 后续修复补丁（ReID 稳定性）
> 对比：`fmtrack_code_configs_20260117_195600.zip`（修改前） → 当前修复版（修改后）

本次只动两处：
- `models/public_reid.py`：**默认使用 torchvision ImageNet 预训练权重**；对随机投影加入 **固定 seed 初始化**，避免每次运行 embedding 分布变化。
- `configs/r50_dino_fa_mot_v2_mot17_public_reid.yaml`：增加 `PUBLIC_REID_PROJ_SEED`，并允许 `PUBLIC_REID_WEIGHTS` 留空时走默认预训练。

---
## 文件：`models/public_reid.py`
### 修改块：L11-L16 → L11-L17
**修改前：**
```python
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models


class PublicDetReIDEncoder:
```
**修改后：**
```python
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights, ResNet18_Weights


class PublicDetReIDEncoder:
```
### 修改块：L95-L110 → L96-L119
**修改前：**
```python
) -> PublicDetReIDEncoder:
    backbone_name = str(config.get("PUBLIC_REID_BACKBONE", "torchvision_resnet50"))
    weights_path = config.get("PUBLIC_REID_WEIGHTS", None)
    input_h = int(config.get("PUBLIC_REID_INPUT_H", 256))
    input_w = int(config.get("PUBLIC_REID_INPUT_W", 128))
    batch_size = int(config.get("PUBLIC_REID_BATCH_SIZE", 64))
    l2_norm = bool(config.get("PUBLIC_REID_L2_NORM", True))

    if backbone_name == "torchvision_resnet50":
        backbone = models.resnet50(weights=None)
        feat_dim = 2048
    elif backbone_name == "torchvision_resnet18":
        backbone = models.resnet18(weights=None)
        feat_dim = 512
    else:
        raise ValueError(f"Unsupported PUBLIC_REID_BACKBONE: {backbone_name}")
```
**修改后：**
```python
) -> PublicDetReIDEncoder:
    backbone_name = str(config.get("PUBLIC_REID_BACKBONE", "torchvision_resnet50"))
    weights_path = config.get("PUBLIC_REID_WEIGHTS", None)
    # YAML 里常见的空值会解析成 None 或 ""，这里统一为 None
    if isinstance(weights_path, str) and weights_path.strip() == "":
        weights_path = None
    input_h = int(config.get("PUBLIC_REID_INPUT_H", 256))
    input_w = int(config.get("PUBLIC_REID_INPUT_W", 128))
    batch_size = int(config.get("PUBLIC_REID_BATCH_SIZE", 64))
    l2_norm = bool(config.get("PUBLIC_REID_L2_NORM", True))

    # ------------------------------------------------------------------
    # Backbone
    # 默认使用 torchvision 的 ImageNet 预训练权重（审稿友好、可复现），
    # 如果提供了 PUBLIC_REID_WEIGHTS 则加载用户权重覆盖。
    # ------------------------------------------------------------------
    if backbone_name == "torchvision_resnet50":
        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        feat_dim = 2048
    elif backbone_name == "torchvision_resnet18":
        backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        feat_dim = 512
    else:
        raise ValueError(f"Unsupported PUBLIC_REID_BACKBONE: {backbone_name}")
```
### 修改块：L120-L126 → L129-L144
**修改前：**
```python
        state = _strip_module_prefix(state)
        backbone.load_state_dict(state, strict=False)

    proj = nn.Linear(feat_dim, feature_dim)

    backbone = backbone.to(device=device, dtype=dtype)
    proj = proj.to(device=device, dtype=dtype)
```
**修改后：**
```python
        state = _strip_module_prefix(state)
        backbone.load_state_dict(state, strict=False)

    # ------------------------------------------------------------------
    # Projection
    # 说明：如果你没有端到端训练 adapter，这个投影需要“固定且可复现”。
    # 使用固定 seed 初始化，避免每次运行 embedding 分布不同。
    # ------------------------------------------------------------------
    proj_seed = int(config.get("PUBLIC_REID_PROJ_SEED", 12345))
    g = torch.Generator(device="cpu").manual_seed(proj_seed)
    proj = nn.Linear(feat_dim, feature_dim, bias=False)
    with torch.no_grad():
        proj.weight.copy_(torch.randn_like(proj.weight, generator=g) / (feat_dim ** 0.5))

    backbone = backbone.to(device=device, dtype=dtype)
    proj = proj.to(device=device, dtype=dtype)
```
## 文件：`configs/r50_dino_fa_mot_v2_mot17_public_reid.yaml`
### 修改块：L3-L8 → L3-L9
**修改前：**
```python
USE_PUBLIC_REID: True
PUBLIC_REID_BACKBONE: torchvision_resnet50
PUBLIC_REID_WEIGHTS:
PUBLIC_REID_INPUT_H: 256
PUBLIC_REID_INPUT_W: 128
PUBLIC_REID_BATCH_SIZE: 64
```
**修改后：**
```yaml
USE_PUBLIC_REID: True
PUBLIC_REID_BACKBONE: torchvision_resnet50
PUBLIC_REID_WEIGHTS:
PUBLIC_REID_PROJ_SEED: 12345
PUBLIC_REID_INPUT_H: 256
PUBLIC_REID_INPUT_W: 128
PUBLIC_REID_BATCH_SIZE: 64
```
