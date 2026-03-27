# FM-Track P1-b（Public Detections + 外部 ReID Embeddings）修改对照（文件/行号/修改前后）

本文件基于你上传的「当前代码」版本（src = `fmtrack_code_and_configs_only_20260117_192201.zip`）与我按 **P1-b** 方案补丁后的版本（patched）做逐段对照。

## 本次修改目标（P1-b）
- **目标**：在 `submit_public.py` 的 public detections 跑法中，不再依赖 DINO 的预测框/IoU 匹配来“借用” embedding，而是直接：
  - 用 **det/det.txt 的 public boxes**
  - 通过 **外部 ReID encoder（crop -> embedding）** 得到每个 public box 的 `output_embeds`
- **收益**：
  - 避免 “DINO 框差 -> IoU 匹配失败 -> 大量 zero embeddings” 的灾难链路
  - 让审稿人更认可：你的创新（频域关联）在 **同一检测输入条件** 下比较公平（MOT public protocol）

---

## 文件：`models/runtime_tracker_public.py`
- 覆盖目标：
  - 新增 P1-b 分支：`use_public_reid=True` 时使用外部 ReID embedding
  - `update()` 支持传入 `image_path`（用于 crop）
- Hunk 数量：3

### 变更块 1：`__init__` 增加 P1-b 参数并缓存
- 修改前（src）：L95–L120（共 26 行）
- 修改后（patched）：L95–L125（共 31 行）

**修改前**
```python
    def __init__(
            self,
            model,
            # Sequence infos:
            sequence_hw: tuple,
            # Public detections:
            public_detections: dict,  # frame_id -> list of (x, y, w, h, conf)
            # Inference settings:
            use_sigmoid: bool = False,
            assignment_protocol: str = "object-priority",  # object-priority 正确返回检测数量的标签
            miss_tolerance: int = 30,
            det_thresh: float = 0.5,  # For public detection confidence
            newborn_thresh: float = 0.5,
            id_thresh: float = 0.1,
            area_thresh: int = 0,
            iou_thresh: float = 0.5,  # IoU threshold for matching DINO with public
            only_detr: bool = False,
            dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.model.eval()

        self.dtype = dtype
        self.public_detections = public_detections
        self.iou_thresh = iou_thresh
        self.frame_id = 0
```

**修改后**
```python
    def __init__(
            self,
            model,
            # Sequence infos:
            sequence_hw: tuple,
            # Public detections:
            public_detections: dict,  # frame_id -> list of (x, y, w, h, conf)
            # Inference settings:
            use_sigmoid: bool = False,
            assignment_protocol: str = "object-priority",  # object-priority 正确返回检测数量的标签
            miss_tolerance: int = 30,
            det_thresh: float = 0.5,  # For public detection confidence
            newborn_thresh: float = 0.5,
            id_thresh: float = 0.1,
            area_thresh: int = 0,
            iou_thresh: float = 0.5,  # IoU threshold for matching DINO with public
            # P1-b: public det + external ReID embeddings (crop -> encoder)
            public_reid_encoder=None,
            use_public_reid: bool = False,
            only_detr: bool = False,
            dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.model.eval()

        self.dtype = dtype
        self.public_detections = public_detections
        self.iou_thresh = iou_thresh
        self.public_reid_encoder = public_reid_encoder
        self.use_public_reid = bool(use_public_reid)
        self.frame_id = 0
```

> 说明：`use_public_reid=True` 时，tracker 将完全跳过 DINO 的 IoU matching，改为直接为 public box 生成 embedding。

### 变更块 2：`update()` 增加 `image_path` 参数 + 文档说明
- 修改前（src）：L180–L190（共 11 行）
- 修改后（patched）：L185–L196（共 12 行）

**修改前**
```python
    @torch.no_grad()
    def update(self, image):
        """
        Update with public detections.
        优化：先检查 public detections，若为空则跳过 DINO forward 以节省计算
        1. Get public detections for current frame (先检查，避免无效 DINO 计算)
        2. Get DINO detections and features (只在需要时运行)
        3. Match DINO detections with public detections using IoU
        4. Use matched DINO features with public detection boxes
        5. Perform ID association
        """
```

**修改后**
```python
    @torch.no_grad()
    def update(self, image, image_path: str | None = None):
        """
        Update with public detections.
        优化：先检查 public detections，若为空则跳过 DINO forward 以节省计算
        1. Get public detections for current frame (先检查，避免无效 DINO 计算)
        2. (Option A) Get DINO detections and features (只在需要时运行)
           (Option B, P1-b) Extract ReID embeddings from raw image crops
        3. (Option A) Match DINO detections with public detections using IoU
        4. Use matched features with public detection boxes
        5. Perform ID association
        """
```

> 说明：P1-b 需要 `image_path` 打开原始帧做 crop（避免使用 DETR resize/normalize 后的 tensor 做裁剪带来坐标不一致）。

### 变更块 3：增加 P1-b 分支（public box -> ReID embed），并把 DINO forward 移入 else
- 修改前（src）：L235–L290（共 56 行）
- 修改后（patched）：L241–L301（共 61 行）

**修改前（核心片段）**
```python
        # Step 2: Now run DINO forward (只在有有效 public dets 时才运行，节省计算)
        detr_out = self.model(frames=image, part="detr")
        dino_scores, dino_categories, dino_boxes, dino_embeds = self._get_dino_detections(detr_out)

        # Convert to tensors
        public_boxes_xywh = torch.tensor(
            [[d[0], d[1], d[2], d[3]] for d in filtered_public_dets],
            dtype=self.dtype, device=distributed_device()
        )
        public_scores = torch.tensor(
            [d[4] for d in filtered_public_dets],
            dtype=self.dtype, device=distributed_device()
        )

        # Step 3: Match DINO detections with public detections
        ...
        output_embeds = dino_embeds[assign_dino_idx]
```

**修改后（核心片段）**
```python
        # Convert public dets to tensors
        public_boxes_xywh = torch.tensor(
            [[d[0], d[1], d[2], d[3]] for d in filtered_public_dets],
            dtype=self.dtype, device=distributed_device()
        )
        public_scores = torch.tensor(
            [d[4] for d in filtered_public_dets],
            dtype=self.dtype, device=distributed_device()
        )

        # Step 2/3: get per-box embeddings
        if self.use_public_reid:
            # P1-b: extract embeddings from raw image crops
            if self.public_reid_encoder is None:
                raise RuntimeError("use_public_reid=True but public_reid_encoder is None")
            if image_path is None:
                raise ValueError("P1-b requires image_path (raw frame) for cropping")

            boxes = public_boxes_xywh
            boxes_norm = box_xywh_to_cxcywh(boxes / self.bbox_unnorm)
            output_embeds = self.public_reid_encoder.encode(image_path=image_path, boxes_xywh=boxes)
            scores = public_scores
            categories = torch.zeros((boxes.shape[0],), dtype=torch.int64, device=distributed_device())
        else:
            # Option A: borrow DINO embeddings via IoU matching
            detr_out = self.model(frames=image, part="detr")
            _, dino_categories, dino_boxes, dino_embeds = self._get_dino_detections(detr_out)
            ...
```

> 说明：
> - P1-b 下不再依赖 DINO box quality，`output_embeds` 对每个 public box 都稳定存在。
> - `categories` 在 MOT17/MOT20 行人场景可以直接置 0（单类），避免把检测分类噪声带入关联。

---

## 文件：`submit_public.py`
- 覆盖目标：
  - 初始化 public ReID encoder，并传入 `RuntimeTrackerPublic`
  - 在 `update()` 时传入 `image_path`
- Hunk 数量：3

### 变更块 1：新增 ReID encoder import
- 修改前（src）：L22–L38（共 17 行）
- 修改后（patched）：L22–L39（共 18 行）

**修改前**
```python
from data.seq_dataset import SeqDataset
from models.runtime_tracker_public import RuntimeTrackerPublic, load_public_detections
```

**修改后**
```python
from data.seq_dataset import SeqDataset
from models.runtime_tracker_public import RuntimeTrackerPublic, load_public_detections
from models.public_reid import build_public_reid_encoder
```

### 变更块 2：dtype 提前 + 初始化 ReID encoder（可开关）
- 修改前（src）：L63–L104（共 42 行）
- 修改后（patched）：L64–L118（共 55 行）

**修改前（片段）**
```python
    model = model.to(device)
    model.eval()

    # Get dataset info
    dataset_name = config["INFERENCE_DATASET"]
    ...

    # Set dtype
    dtype_str = config.get("INFERENCE_DTYPE", "FP32")
    if dtype_str == "FP32":
        dtype = torch.float32
```

**修改后（片段）**
```python
    model = model.to(device)
    model.eval()

    # Set dtype (need before P1-b ReID encoder init)
    dtype_str = config.get("INFERENCE_DTYPE", "FP32")
    if dtype_str == "FP32":
        dtype = torch.float32
    elif dtype_str == "FP16":
        dtype = torch.float16
    else:
        raise ValueError(f"Unknown dtype '{dtype_str}'.")

    # ---------------------------------------------------------------------
    # P1-b: Optional public det + external ReID embeddings (crop -> encoder)
    # ---------------------------------------------------------------------
    use_public_reid = bool(config.get("USE_PUBLIC_REID", False))
    public_reid_encoder = None
    if use_public_reid:
        public_reid_encoder = build_public_reid_encoder(
            config=config,
            device=device,
            feature_dim=int(config.get("FEATURE_DIM", 256)),
            dtype=dtype,
        )
```

> 说明：`dtype` 必须在 encoder 初始化前确定（encoder 内部会把 backbone/proj 转到相同 dtype/device）。

### 变更块 3：把 encoder 传给 tracker + update 传入 `image_path`
- 修改前（src）：L180–L210（共 31 行）
- 修改后（patched）：L196–L227（共 32 行）

**修改前（片段）**
```python
        runtime_tracker = RuntimeTrackerPublic(
            ...
            iou_thresh=iou_thresh,
            only_detr=inference_only_detr,
            dtype=dtype,
        )

        ...
        runtime_tracker.update(image=image)
```

**修改后（片段）**
```python
        runtime_tracker = RuntimeTrackerPublic(
            ...
            iou_thresh=iou_thresh,
            public_reid_encoder=public_reid_encoder,
            use_public_reid=use_public_reid,
            only_detr=inference_only_detr,
            dtype=dtype,
        )

        ...
        runtime_tracker.update(image=image, image_path=image_path)
```

---

## 新增文件：`models/public_reid.py`
- 覆盖目标：为 public det boxes 提供可复现的 crop -> embedding 逻辑
- 行数：242

### 变更块 1：新增 ReID encoder 工厂 + `encode()` 接口
- 修改前（src）：（不存在）
- 修改后（patched）：新增

**修改后（片段 1：工厂函数入口）**
```python
def build_public_reid_encoder(
    config: dict,
    device: torch.device,
    feature_dim: int,
    dtype: torch.dtype = torch.float32,
) -> PublicDetReIDEncoder:
    ...
```

**修改后（片段 2：按 public boxes 做 crop + batch encode）**
```python
@torch.no_grad()
def encode(self, image_path: str, boxes_xywh: torch.Tensor) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    crops = []
    valid = []
    for b in boxes_xywh.detach().cpu().tolist():
        x, y, w, h = b
        ...
        crop = img.crop((x1, y1, x2, y2))
        crops.append(self.tf(crop))
        valid.append(True)

    batch = torch.stack(crops, dim=0).to(device=self.device, dtype=self.dtype)
    feats = self.backbone(batch)
    ...
    emb = self.proj(feats)
    emb = F.normalize(emb, p=2, dim=-1)
    return emb
```

> 说明：这里默认提供 `torchvision_resnet50/resnet18`（ImageNet 权重或你提供的 checkpoint）。
> 你后续可以把 `PUBLIC_REID_WEIGHTS` 指向更强的 ReID 权重（如 OSNet / CLIP-ReID 等）而不改 tracker 主逻辑。

---

## 新增文件：`configs/r50_dino_fa_mot_v2_mot17_public_reid.yaml`
- 覆盖目标：给 `submit_public.py` 一键开启 P1-b 的配置
- 行数：33

### 变更块 1：新增配置项（基于 super config 覆盖）
- 修改前（src）：（不存在）
- 修改后（patched）：新增

**修改后**
```yaml
SUPER_CONFIG_PATH: configs/r50_dino_fa_mot_v2_mot17.yaml

USE_PUBLIC_REID: True
PUBLIC_REID_BACKBONE: torchvision_resnet50
PUBLIC_REID_WEIGHTS:
PUBLIC_REID_INPUT_H: 256
PUBLIC_REID_INPUT_W: 128
PUBLIC_REID_BATCH_SIZE: 64
PUBLIC_REID_L2_NORM: True

EXP_NAME: fa_mot_v2_mot17_public_reid
```

---

## 使用方式（最短路径）
1) 直接跑 public protocol（MOT17）：
```bash
python submit_public.py --config-path configs/r50_dino_fa_mot_v2_mot17_public_reid.yaml
```
2) 如果你有更强的 ReID 权重：把 `PUBLIC_REID_WEIGHTS` 填成 checkpoint 路径即可（不需要再改代码）。

