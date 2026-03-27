# FM-Track P1-b 强 ReID 权重支持补丁（v5）

> 目标：在不破坏现有 P1-b（public det + crop->embed）流程的前提下，让 `PUBLIC_REID_BACKBONE` 支持 **torchreid OSNet 系列（推荐）**，并把权重加载做成对审稿人更友好的“可复现 + 可迁移”实现。

---

## 0. 改动概览

- ✅ `models/public_reid.py`：
  - 新增 `torchreid:<name>` backbone（如 `torchreid:osnet_ain_x1_0`）
  - ckpt 加载更鲁棒（支持 `state_dict/model/net/network` 包装、自动去前缀）
  - 不再硬编码 `feat_dim=2048/512`，改为 dummy forward 自动推断（兼容 OSNet/ResNet 等）
  - `encode()` 支持 backbone 输出 tuple/list（常见于部分 ReID 模型）
- ✅ `configs/r50_dino_fa_mot_v2_mot17_public_reid.yaml`：
  - 增加 `PUBLIC_REID_PRETRAINED` 与 torchreid 使用示例（保持默认 torchvision baseline 不变）

---

## 1) `models/public_reid.py`

### 1.1 `encode()`：输出维度稳定 + 兼容 tuple 输出

**修改位置**：
- 修改前：约 L18–L86
- 修改后：约 L18–L95

**修改前**（节选）：
```py
class PublicDetReIDEncoder:
    ...
    @torch.no_grad()
    def encode(self, image_path: str, boxes_xywh: torch.Tensor) -> torch.Tensor:
        ...
        if num_boxes == 0:
            return torch.zeros((0, self.proj.out_features), device=self.device, dtype=self.dtype)

        out = torch.zeros((num_boxes, self.proj.out_features), device=self.device, dtype=self.dtype)

        def _flush():
            ...
            feats = self.backbone(batch)
            if feats.dim() > 2:
                feats = torch.flatten(feats, 1)
            emb = self.proj(feats)
            ...
```

**修改后**（节选）：
```py
class PublicDetReIDEncoder:
    def __init__(...):
        ...
        self.out_dim = int(getattr(self.proj, "out_features", 0))

    @torch.no_grad()
    def encode(self, image_path: str, boxes_xywh: torch.Tensor) -> torch.Tensor:
        """Encode all boxes in *one frame* into embeddings."""
        ...
        if num_boxes == 0:
            return torch.zeros((0, self.out_dim), device=self.device, dtype=self.dtype)

        out = torch.zeros((num_boxes, self.out_dim), device=self.device, dtype=self.dtype)

        def _flush():
            ...
            feats = self.backbone(batch)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            if feats.dim() > 2:
                feats = torch.flatten(feats, 1)
            emb = self.proj(feats)
            ...
```

**为什么要改**：
- torchreid/Transformer ReID 有时会返回 `(feat, aux)`；不处理会直接崩。
- 用 `self.out_dim` 统一输出维度，避免未来换投影层/非 Linear 时出现属性缺失。

---

### 1.2 权重加载：从“只去 module.”升级到“多前缀 + 多包装”

**修改位置**：
- 修改前：约 L89–L96（仅 `_strip_module_prefix`）
- 修改后：约 L98–L126（`_strip_known_prefixes` + `_extract_state_dict`）

**修改前**：
```py
def _strip_module_prefix(state_dict: dict) -> dict:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
```

**修改后**：
```py
def _strip_known_prefixes(state_dict: dict) -> dict:
    prefixes = ["module.", "model.", "backbone.", "encoder.", "reid."]
    ...

def _extract_state_dict(ckpt: Any) -> dict:
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "net", "network"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        ...
    raise ValueError("Unsupported checkpoint format for PUBLIC_REID_WEIGHTS")
```

**为什么要改**：
- 你后续很可能用到 FastReID/torchreid/自训练 ReID，ckpt key 前缀五花八门。
- 这版能让你“换权重不改代码”，对复现和写论文非常友好。

---

### 1.3 支持 torchreid OSNet：新增 `_build_backbone()`

**修改位置**：
- 修改前：无（原来只支持 torchvision）
- 修改后：约 L151–L195

**修改后**（新增）：
```py
def _build_backbone(backbone_name: str, pretrained: bool) -> nn.Module:
    if backbone_name == "torchvision_resnet50":
        ...

    if backbone_name.startswith("torchreid:"):
        try:
            import torchreid
        except ImportError as e:
            raise ImportError(
                "PUBLIC_REID_BACKBONE uses 'torchreid', but torchreid is not installed. "
                "Install it with: pip install torchreid"
            ) from e

        model_name = backbone_name.split(":", 1)[1]
        tr_models = torchreid.models
        if hasattr(tr_models, "build_model"):
            m = tr_models.build_model(name=model_name, num_classes=1, pretrained=pretrained)
        else:
            fn = getattr(tr_models, model_name, None)
            ...
        _remove_classification_head(m)
        return m
```

**为什么要改**：
- OSNet-AIN/IBN 是你要的“更强、更稳”的 ReID 选择之一；换成 torchreid 的 model zoo 之后，P1-b 就能直接吃到强 ReID embedding。

---

### 1.4 `build_public_reid_encoder()`：取消硬编码 feat_dim，新增 `PUBLIC_REID_PRETRAINED`

**修改位置**：
- 修改前：约 L100–L151（硬编码 feat_dim）
- 修改后：约 L198–L269（自动推断 feat_dim + torchreid 兼容）

**修改前**（节选）：
```py
if backbone_name == "torchvision_resnet50":
    backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    feat_dim = 2048
proj = nn.Linear(feat_dim, feature_dim, bias=False)
```

**修改后**（节选）：
```py
pretrained = bool(config.get("PUBLIC_REID_PRETRAINED", True))
backbone = _build_backbone(backbone_name=backbone_name, pretrained=pretrained)
feat_dim = _infer_feat_dim(backbone.cpu().float(), input_hw=(input_h, input_w))
proj = nn.Linear(feat_dim, feature_dim, bias=False)
```

**为什么要改**：
- OSNet/Transformer 等输出维度不是 2048/512，硬编码会直接挂。
- `PUBLIC_REID_PRETRAINED` 让实验更可控：你可以做消融（pretrained vs random）也更好写论文。

---

## 2) `configs/r50_dino_fa_mot_v2_mot17_public_reid.yaml`

**修改位置**：
- 修改前：L1–L12
- 修改后：L1–L33

**修改前**：
```yaml
USE_PUBLIC_REID: True
PUBLIC_REID_BACKBONE: torchvision_resnet50
PUBLIC_REID_WEIGHTS:
PUBLIC_REID_PROJ_SEED: 12345
```

**修改后**：
```yaml
USE_PUBLIC_REID: True

# Option A (default): torchvision backbone (ImageNet pretrained)
PUBLIC_REID_BACKBONE: torchvision_resnet50
PUBLIC_REID_PRETRAINED: True
PUBLIC_REID_WEIGHTS:

# Option B (stronger ReID, requires `pip install torchreid`):
# PUBLIC_REID_BACKBONE: torchreid:osnet_ain_x1_0
# PUBLIC_REID_PRETRAINED: True
# PUBLIC_REID_WEIGHTS:
```

---

## 3) 你接下来怎么用（推荐配置）

### 3.1 Baseline（先跑通、快迭代）
```yaml
PUBLIC_REID_BACKBONE: torchvision_resnet50
PUBLIC_REID_PRETRAINED: True
PUBLIC_REID_WEIGHTS:
```

### 3.2 Strong ReID（更稳，顶会更友好）
1) 安装：`pip install torchreid`
2) 配置：
```yaml
PUBLIC_REID_BACKBONE: torchreid:osnet_ain_x1_0
PUBLIC_REID_PRETRAINED: True
PUBLIC_REID_WEIGHTS:
```

如果你手里有 OSNet/FastReID 的本地 ckpt，把路径填到 `PUBLIC_REID_WEIGHTS` 即可。

