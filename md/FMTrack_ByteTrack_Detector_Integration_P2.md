# FMTrack 接入 ByteTrack（YOLOX）目标检测器（P2 / Private Det）改动说明（给 Codex 实现）

> 目标：在**不训练 detector** 的前提下，把 **ByteTrack 提供的 YOLOX 预训练检测器**接入你现有推理/评测链路。
>
> - **Det 来源**：ByteTrack detector on-the-fly 产出 boxes/scores（而不是读取 MOT public `det.txt`）。
> - **特征来源**：沿用你现有 **P1-b**：`public_reid_encoder.encode(crop)` 给每个 box 提供 appearance embedding。
> - **训练/评测协议**：这是 **Private detections (P2)**。主表仍建议保留 public det（P1）。

---

## 0. 依赖与目录约定（Codex 先做这些）

### 0.1 拉取 ByteTrack（推荐作为 third_party）

在项目根目录新增：

- `third_party/ByteTrack/`（git clone 或 submodule 均可）

要求 ByteTrack 的 YOLOX 代码可 import：

- `from yolox.exp import get_exp`
- `from yolox.utils import postprocess`
- `from yolox.data.data_augment import preproc`

> 说明：ByteTrack 本质用 YOLOX 做检测器；你这里只拿它的 detector 推理即可。

### 0.2 安装依赖

确保环境能 import `yolox`（通常做法是）：

- `pip install -e third_party/ByteTrack`  （或按 ByteTrack README 安装）

### 0.3 预训练权重与 exp 文件

你需要在 config 里提供：

- `BYTETRACK_EXP_FILE`：ByteTrack repo 里的 exp python 文件（不同模型对应不同 exp）
- `BYTETRACK_CKPT`：你下载的 `.pth.tar` 权重路径

ByteTrack 官方 repo常用路径（示例）：

- MOT17 detector：`third_party/ByteTrack/exps/example/mot/yolox_x_mix_det.py`
- MOT20 detector：`third_party/ByteTrack/exps/example/mot/yolox_x_mix_mot20_ch.py`

（具体用哪个取决于你下载的 ckpt 对应的 exp，建议一一匹配。）

---

## 1. 新增配置项（config / yaml）

> 目标：让同一个 submit 脚本可以切换 det 来源：`public` / `bytetrack`。

在 yaml 里新增：

```yaml
# det source switch
DET_SOURCE: bytetrack   # [public, bytetrack]

# ByteTrack(YOLOX) detector
BYTETRACK_EXP_FILE: third_party/ByteTrack/exps/example/mot/yolox_x_mix_det.py
BYTETRACK_CKPT: weight/bytetrack_x_mot17.pth.tar
BYTETRACK_FP16: True
BYTETRACK_TEST_SIZE: [800, 1440]   # (h, w)；如果不填则用 exp.test_size
BYTETRACK_CONF_THRE: 0.01          # detector confidence threshold
BYTETRACK_NMS_THRE: 0.7
BYTETRACK_CLASS_AGNOSTIC_NMS: True

# 是否把 detector 输出缓存成 MOT det.txt（可选，但强烈推荐：方便复现和调参）
CACHE_PRIVATE_DET: True
CACHE_PRIVATE_DET_DIR: outputs/private_det_cache
```

> 注意：`DET_THRESH` 仍然用于 tracker 侧的过滤（在 `RuntimeTrackerPublic.update()` 内），不要和 `BYTETRACK_CONF_THRE` 混用。

---

## 2. 代码改动清单（逐文件 / 行号 / 改前改后）

> 下方行号基于你当前压缩包 `fmtrack_code_configs_20260117_213446.zip` 展开后的版本；Codex 实现时以同名文件为准。

---

### 2.1 【新增】`models/bytetrack_detector.py`

**文件：** `models/bytetrack_detector.py`（新文件）

**新增内容（完整文件）**：

```python
# models/bytetrack_detector.py

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch


@dataclass
class ByteTrackDetConfig:
    exp_file: str
    ckpt: str
    fp16: bool = True
    test_size: Optional[Tuple[int, int]] = None  # (h, w)
    conf_thre: float = 0.01
    nms_thre: float = 0.7
    class_agnostic_nms: bool = True


class ByteTrackDetector:
    """YOLOX detector wrapper used by ByteTrack.

    Output format matches RuntimeTrackerPublic expectation:
        List[(x, y, w, h, conf)]  in pixel xywh.
    """

    def __init__(self, cfg: ByteTrackDetConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        # Lazy import so the project still runs without ByteTrack installed.
        try:
            from yolox.exp import get_exp
        except Exception as e:
            raise ImportError(
                "Cannot import YOLOX/ByteTrack. Make sure third_party/ByteTrack is installed: "
                "pip install -e third_party/ByteTrack"
            ) from e

        assert os.path.exists(cfg.exp_file), f"BYTETRACK_EXP_FILE not found: {cfg.exp_file}"
        assert os.path.exists(cfg.ckpt), f"BYTETRACK_CKPT not found: {cfg.ckpt}"

        self.exp = get_exp(cfg.exp_file, None)
        self.model = self.exp.get_model().to(device).eval()

        ckpt = torch.load(cfg.ckpt, map_location="cpu")
        # ByteTrack/YOLOX checkpoints usually store weights under key 'model'
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=False)

        if cfg.fp16:
            self.model.half()

        # detector test size
        self.test_size = cfg.test_size if cfg.test_size is not None else tuple(self.exp.test_size)

        # postprocess uses exp.num_classes (ByteTrack MOT models are usually 1-class: person)
        self.num_classes = getattr(self.exp, "num_classes", 1)

    @torch.no_grad()
    def detect(self, image_path: str) -> List[Tuple[float, float, float, float, float]]:
        """Run detector on a single frame."""
        from yolox.data.data_augment import preproc
        from yolox.utils import postprocess

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        img_h, img_w = img_bgr.shape[:2]

        # YOLOX uses BGR->RGB inside preproc? (ByteTrack uses cv2 BGR; preproc handles normalization)
        img, ratio = preproc(img_bgr, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        img = img.half() if self.cfg.fp16 else img.float()

        outputs = self.model(img)
        outputs = postprocess(
            outputs,
            num_classes=self.num_classes,
            conf_thre=self.cfg.conf_thre,
            nms_thre=self.cfg.nms_thre,
            class_agnostic=self.cfg.class_agnostic_nms,
        )

        if outputs[0] is None:
            return []

        dets = outputs[0].cpu().numpy()  # (N, 7): x1,y1,x2,y2,obj_conf,cls_conf,cls

        # scale back
        dets[:, :4] /= ratio
        dets[:, 0::2] = np.clip(dets[:, 0::2], 0, img_w - 1)
        dets[:, 1::2] = np.clip(dets[:, 1::2], 0, img_h - 1)

        results = []
        for x1, y1, x2, y2, obj_conf, cls_conf, cls_id in dets:
            conf = float(obj_conf * cls_conf)
            # For MOT models, usually only person class exists; still keep a safety guard
            if int(cls_id) != 0:
                continue
            x = float(x1)
            y = float(y1)
            w = float(x2 - x1)
            h = float(y2 - y1)
            if w <= 1 or h <= 1:
                continue
            results.append((x, y, w, h, conf))

        return results
```

---

### 2.2 【修改】`models/runtime_tracker_public.py`：支持 det_source = bytetrack

#### 2.2.1 修改 `__init__` 签名与成员

**文件：** `models/runtime_tracker_public.py`

**修改位置：** 约 `L95-L125`

**修改前：**

```python
# L95-
class RuntimeTrackerPublic:
    def __init__(
            self,
            model,
            sequence_hw: tuple,
            public_detections: dict,  # frame_id -> list of (x, y, w, h, conf)
            ...
    ):
        ...
        self.public_detections = public_detections
        ...
```

**修改后：**

```python
class RuntimeTrackerPublic:
    def __init__(
            self,
            model,
            # Sequence infos:
            sequence_hw: tuple,
            # Detections:
            public_detections: dict | None = None,   # when DET_SOURCE=public
            detector=None,                           # when DET_SOURCE=bytetrack
            det_source: str = "public",
            ...
    ):
        ...
        self.public_detections = public_detections
        self.detector = detector
        self.det_source = det_source
        ...
```

> 约束：
> - `det_source == "public"` 时：必须传 `public_detections`（dict）
> - `det_source == "bytetrack"` 时：必须传 `detector`（ByteTrackDetector）

#### 2.2.2 修改 `update()` 中取 det 的逻辑

**文件：** `models/runtime_tracker_public.py`

**修改位置：** `L197-L201`

**修改前：**

```python
# L197-
self.frame_id += 1

# Step 1: Get public detections first
public_dets = self.public_detections.get(self.frame_id, [])
```

**修改后：**

```python
self.frame_id += 1

# Step 1: get detections from selected source
if self.det_source == "bytetrack":
    if self.detector is None:
        raise RuntimeError("det_source=bytetrack but detector is None")
    if image_path is None:
        raise ValueError("ByteTrack detector requires image_path")
    public_dets = self.detector.detect(image_path)
else:
    if self.public_detections is None:
        raise RuntimeError("det_source=public but public_detections is None")
    public_dets = self.public_detections.get(self.frame_id, [])
```

> 注意：后续逻辑（confidence/area filter、P1-b ReID crop、关联推理）全部不变。

---

### 2.3 【修改】`submit_public.py`：加入 `DET_SOURCE` 分支并构造 ByteTrackDetector

> 这里不改文件名也可以：当 `DET_SOURCE=bytetrack` 时，`submit_public.py` 实际上在跑 P2/private det。

#### 2.3.1 import ByteTrackDetector

**文件：** `submit_public.py`

**修改位置：** `L37-L39` import 区域

**修改前：**

```python
from models.runtime_tracker_public import RuntimeTrackerPublic, load_public_detections
from models.public_reid import build_public_reid_encoder
```

**修改后：**

```python
from models.runtime_tracker_public import RuntimeTrackerPublic, load_public_detections
from models.public_reid import build_public_reid_encoder
from models.bytetrack_detector import ByteTrackDetector, ByteTrackDetConfig
```

#### 2.3.2 在每个 sequence 构造 det provider

**文件：** `submit_public.py`

**修改位置：** `L183-L211`（加载 det + 初始化 RuntimeTrackerPublic）

**修改前：**

```python
# L183-
# Load public detections
if dataset_name == "MOT17":
    det_path = os.path.join(data_root, "MOT17", data_split, sequence_name, "det", "det.txt")
elif dataset_name == "MOT20":
    det_path = os.path.join(data_root, "MOT20", data_split, sequence_name, "det", "det.txt")
else:
    raise ValueError(f"Unknown dataset: {dataset_name}")

logger.info(f"Loading public detections from {det_path}")
public_detections = load_public_detections(det_path)

runtime_tracker = RuntimeTrackerPublic(
    model=model,
    sequence_hw=sequence_hw,
    public_detections=public_detections,
    ...
)
```

**修改后：**

```python
# decide det source
det_source = str(config.get("DET_SOURCE", "public")).lower()

public_detections = None
bytetrack_detector = None

if det_source == "bytetrack":
    # build ByteTrack detector once per process (OK) or once per sequence (simple)
    bt_cfg = ByteTrackDetConfig(
        exp_file=str(config["BYTETRACK_EXP_FILE"]),
        ckpt=str(config["BYTETRACK_CKPT"]),
        fp16=bool(config.get("BYTETRACK_FP16", True)),
        test_size=tuple(config.get("BYTETRACK_TEST_SIZE", None)) if config.get("BYTETRACK_TEST_SIZE", None) else None,
        conf_thre=float(config.get("BYTETRACK_CONF_THRE", 0.01)),
        nms_thre=float(config.get("BYTETRACK_NMS_THRE", 0.7)),
        class_agnostic_nms=bool(config.get("BYTETRACK_CLASS_AGNOSTIC_NMS", True)),
    )
    bytetrack_detector = ByteTrackDetector(cfg=bt_cfg, device=device)
    logger.info(f"Using ByteTrack detector: exp={bt_cfg.exp_file}, ckpt={bt_cfg.ckpt}")
else:
    # default: MOT public det.txt
    if dataset_name == "MOT17":
        det_path = os.path.join(data_root, "MOT17", data_split, sequence_name, "det", "det.txt")
    elif dataset_name == "MOT20":
        det_path = os.path.join(data_root, "MOT20", data_split, sequence_name, "det", "det.txt")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    logger.info(f"Loading public detections from {det_path}")
    public_detections = load_public_detections(det_path)

runtime_tracker = RuntimeTrackerPublic(
    model=model,
    sequence_hw=sequence_hw,
    public_detections=public_detections,
    detector=bytetrack_detector,
    det_source=det_source,
    use_sigmoid=use_sigmoid,
    assignment_protocol=assignment_protocol,
    miss_tolerance=miss_tolerance,
    det_thresh=det_thresh,
    newborn_thresh=newborn_thresh,
    id_thresh=id_thresh,
    area_thresh=area_thresh,
    iou_thresh=iou_thresh,
    public_reid_encoder=public_reid_encoder,
    use_public_reid=use_public_reid,
    only_detr=inference_only_detr,
    dtype=dtype,
)
```

> 重要：如果你 `DET_SOURCE=bytetrack`，强烈建议 `USE_PUBLIC_REID=True`，否则会回退到跑 DINO forward 来给 feature（这和你“完全不依赖 DINO detector”的目标冲突）。

---

### 2.4 【新增】一个 ByteTrack 私有 det 的 yaml（推荐复制现有 public_reid 配置）

**文件：** `configs/r50_dino_fa_mot_v2_mot17_bytetrack_det_public_reid.yaml`（新文件）

**内容（示例）**：

```yaml
SUPER_CONFIG_PATH: configs/r50_dino_fa_mot_v2_mot17_public_reid.yaml

# Use ByteTrack detector instead of MOT public det.txt
DET_SOURCE: bytetrack
BYTETRACK_EXP_FILE: third_party/ByteTrack/exps/example/mot/yolox_x_mix_det.py
BYTETRACK_CKPT: weight/bytetrack_x_mot17.pth.tar
BYTETRACK_FP16: True
BYTETRACK_CONF_THRE: 0.01
BYTETRACK_NMS_THRE: 0.7
BYTETRACK_CLASS_AGNOSTIC_NMS: True

# (optional) override test size; otherwise use exp.test_size
# BYTETRACK_TEST_SIZE: [800, 1440]

EXP_NAME: fa_mot_v2_mot17_bytetrack_det_public_reid
```

MOT20 对应新文件同理：
- `BYTETRACK_EXP_FILE: third_party/ByteTrack/exps/example/mot/yolox_x_mix_mot20_ch.py`
- `BYTETRACK_CKPT: weight/bytetrack_x_mot20.pth.tar`

---

## 3. 运行方式（给你快速验证接入是否成功）

### 3.1 用 ByteTrack detector（P2 / private det）跑 MOT17-val

```bash
python submit_public.py \
  --config-path configs/r50_dino_fa_mot_v2_mot17_bytetrack_det_public_reid.yaml \
  --inference-model /path/to/your_tracker_ckpt.pth \
  --inference-mode evaluate \
  --inference-dataset MOT17 \
  --inference-split val \
  --data-root /path/to/datasets \
  --outputs-dir ./outputs/eval_mot17_private_bytetrack
```

### 3.2 回退到 MOT public det（P1）跑同一套模型

把 config 换回：

- `configs/r50_dino_fa_mot_v2_mot17_public_reid.yaml`

即可。

---

## 4. 审稿友好建议（你现在就可以照着做）

- **主表（公平）**：P1 / `DET_SOURCE=public`（MOT 官方 det.txt）
- **上限（补充）**：P2 / `DET_SOURCE=bytetrack`（强 detector）
- 这两条只改 config，不改 tracker 本体。审稿人最喜欢这种“协议清晰、可复现”的对比。

---

## 5. 常见坑（Codex 实现时要直接规避）

1) **image_path 必须传到 RuntimeTrackerPublic.update()**
   - ByteTrack detector 与 crop-ReID 都需要 `image_path`。

2) ByteTrack checkpoint key
   - 多数是 `torch.load(ckpt)["model"]`；别用错 key。

3) YOLOX test_size / ratio
   - `preproc()` 会返回 `ratio`，务必用它把 xyxy 缩放回原图。

4) detector threshold vs tracker threshold
   - `BYTETRACK_CONF_THRE` 是 detector 出框门槛
   - `DET_THRESH` 是 tracker 侧再次过滤门槛（你已有）。

5) 只做人（person）
   - ByteTrack MOT detector 通常 1 类；仍建议保留 `cls_id==0` 防呆。

