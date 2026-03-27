# FMTrack：ByteTrack 接入后（训练前）最终检查 + 必要修复

> 你现在的代码整体已经能走通 **P2（private det: ByteTrack）+ P1-b（public ReID）** 的思路。
> 下面这份文档只做“训练/推理前必炸点”的最后修复，并给你一套**可直接开跑**的命令清单。

---

## 0. 你当前还会踩的 3 个硬坑（不修基本必报错）

1) **ByteTrack/YOLOX 的 `preproc()` 签名不统一**：不同 fork 有的需要 `(img, size, mean, std)`，有的只有 `(img, size)`。
   - 不兼容会直接 `TypeError`。

2) **MOT20 的 ByteTrack config 继承自 MOT17 config**，如果你不 override，`INFERENCE_DATASET` 会错用 MOT17。

3) `train.py` 里硬编码了 `TORCH_HOME=/gemini/...`：在你自己的服务器上这个路径一般不存在，会导致 torch cache / download 行为异常（甚至直接报错）。

---

## Fix-1：ByteTrackDetector 兼容不同 YOLOX `preproc()` 签名（强烈建议）

**文件**：`models/bytetrack_detector.py`

### 修改位置
- **修改后行号范围**：L44–L75、L94–L113

### 修改前
```python
        assert os.path.exists(cfg.exp_file), f"BYTETRACK_EXP_FILE not found: {cfg.exp_file}"
        assert os.path.exists(cfg.ckpt), f"BYTETRACK_CKPT not found: {cfg.ckpt}"

        self.exp = get_exp(cfg.exp_file, None)
        self.model = self.exp.get_model().to(device).eval()

        ckpt = torch.load(cfg.ckpt, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=False)
...
        img_h, img_w = img_bgr.shape[:2]
        img, ratio = preproc(img_bgr, self.test_size)
```

### 修改后
```python
        # Resolve file paths with clearer errors.
        # We keep this lightweight so users can either:
        #   (1) clone ByteTrack into third_party/ByteTrack, OR
        #   (2) point BYTETRACK_EXP_FILE to an absolute path.
        exp_file = cfg.exp_file
        ckpt_file = cfg.ckpt

        if not os.path.exists(exp_file):
            raise FileNotFoundError(
                f"BYTETRACK_EXP_FILE not found: {exp_file}. "
                f"Make sure you cloned ByteTrack into 'third_party/ByteTrack', "
                f"or set BYTETRACK_EXP_FILE to an absolute path."
            )
        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(
                f"BYTETRACK_CKPT not found: {ckpt_file}. "
                f"Put your weights under the configured path (e.g. ./weight/ or ./weights/) "
                f"and update BYTETRACK_CKPT accordingly."
            )

        self.exp = get_exp(exp_file, None)
        self.model = self.exp.get_model().to(device).eval()

        ckpt = torch.load(ckpt_file, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=False)
...
        # YOLOX forks differ on the preproc signature:
        #   - official: preproc(img, input_size, mean, std)
        #   - some forks: preproc(img, input_size)
        # To be robust across environments, we dynamically adapt.
        import inspect
        try:
            n_params = len(inspect.signature(preproc).parameters)
        except Exception:
            n_params = 2

        if n_params <= 2:
            img, ratio = preproc(img_bgr, self.test_size)
        else:
            rgb_means = getattr(self.exp, "rgb_means", None)
            std = getattr(self.exp, "std", None)
            try:
                img, ratio = preproc(img_bgr, self.test_size, rgb_means, std)
            except TypeError:
                # Some forks have an additional 'swap' argument.
                img, ratio = preproc(img_bgr, self.test_size, rgb_means, std, (2, 0, 1))
```

---

## Fix-2：MOT20 ByteTrack config 必须 override `INFERENCE_DATASET`

**文件**：`configs/r50_dino_fa_mot_v2_mot20_bytetrack_det_public_reid.yaml`

### 修改位置
- **修改后行号范围**：L3–L4

### 修改前
```yaml
SUPER_CONFIG_PATH: configs/r50_dino_fa_mot_v2_mot17_public_reid.yaml

# Use ByteTrack detector instead of MOT public det.txt
DET_SOURCE: bytetrack
...
```

### 修改后
```yaml
SUPER_CONFIG_PATH: configs/r50_dino_fa_mot_v2_mot17_public_reid.yaml

# Override inference dataset to MOT20 (the SUPER config defaults to MOT17).
INFERENCE_DATASET: MOT20

# Use ByteTrack detector instead of MOT public det.txt
DET_SOURCE: bytetrack
...
```

---

## Fix-3：`train.py` 里的 TORCH_HOME 硬编码改为可移植写法（建议）

**文件**：`train.py`

### 修改位置
- **修改后行号范围**：L32–L35

### 修改前
```python
os.environ["TORCH_HOME"] = "/gemini/code/FM-Track/FM-Track/.cache/torch"
```

### 修改后
```python
# Make TORCH_HOME portable across environments.
# If the user already set TORCH_HOME externally, respect it.
if "TORCH_HOME" not in os.environ:
    os.environ["TORCH_HOME"] = os.path.join(os.path.expanduser("~"), ".cache", "torch")
```

---

## 1) 你还需要“人工确认”的点（不改代码，但必须对齐）

### 1.1 ByteTrack 的代码位置
你的 config 里写的是：
- `BYTETRACK_EXP_FILE: third_party/ByteTrack/...`

**但你发的 code zip 里没有 `third_party/` 目录**。

你有两种稳的选择（选其一）：
- **方案 A（推荐，复现最稳）**：把 ByteTrack clone 到仓库：`third_party/ByteTrack/`，然后：
  - `pip install -e third_party/ByteTrack`
- **方案 B**：你已经在别处安装了 YOLOX/ByteTrack，那么就把 `BYTETRACK_EXP_FILE` 改成**绝对路径**。

### 1.2 你刚下载的权重路径
你说权重文件名是：
- `bytetrack_x_mot17.pth.tar`

请确认它真的在：
- `./weight/bytetrack_x_mot17.pth.tar`（你的 yaml 当前写的是这个）

> 如果你实际目录叫 `weights/`，就把 yaml 里的 `weight/` 改掉，或者建立一个软链接：
> `ln -s weights weight`

### 1.3 public ReID（OSNet-AIN）的依赖
如果你走 `PUBLIC_REID_BACKBONE: torchreid:osnet_ain_x1_0`，需要安装：
- `torchreid`
- `opencv-python`（ReID crop 也会用到）

---

## 2) 开始训练（稳定、审稿友好、且与你“冻结 detector”目标一致）

你现在的 config 已经支持：
- `DETR_NUM_TRAIN_FRAMES == 0` → **冻结 detr（检测部分）**，只训练你的关联/频域模块。

### 2.1 推荐训练命令（MOT17）
> 你只要把 `--data-root` 改成你机器上的数据路径。

```bash
python train.py \
  --config-path configs/r50_dino_fa_mot_v2_mot17.yaml \
  --data-root /ABS/PATH/TO/datasets \
  --exp-name fmtrack_mot17_frozen_detr \
  --detr-num-train-frames 0 \
  --batch-size 1
```

如果你是 40G/80G 显存、想稳一点加 batch：
- 先保持 `BATCH_SIZE=1` 不变
- 用 `--accumulate-steps 2/4` 做梯度累积（更稳定）

### 2.2 训练完立刻跑一次推理（MOT17, ByteTrack det + public ReID）
```bash
python submit_public.py \
  --config-path configs/r50_dino_fa_mot_v2_mot17_bytetrack_det_public_reid.yaml \
  --data-root /ABS/PATH/TO/datasets \
  --inference-model outputs/fmtrack_mot17_frozen_detr/checkpoints/xxx.pth \
  --inference-dataset MOT17 \
  --inference-split train \
  --inference-mode evaluate
```

> 第一次跑会慢（ByteTrack 每帧推一次）。
> 因为你开了 `CACHE_PRIVATE_DET: True`，它会把 private det 写到 `outputs/private_det_cache/.../det/det.txt`。

### 2.3 第二次推理（直接读 cache，速度接近 public det）
保持 config 不变，第二次跑会自动命中缓存：
```bash
python submit_public.py \
  --config-path configs/r50_dino_fa_mot_v2_mot17_bytetrack_det_public_reid.yaml \
  --data-root /ABS/PATH/TO/datasets \
  --inference-model outputs/fmtrack_mot17_frozen_detr/checkpoints/xxx.pth \
  --inference-dataset MOT17 \
  --inference-split train \
  --inference-mode evaluate
```

---

## 3) 我建议你“开跑前”的最小 sanity check（10 分钟就能做完）

1) **只跑 1 个序列 + 10 帧**：确认不会在 ByteTrack preproc / 路径上炸。
2) 检查输出目录：
   - `outputs/private_det_cache/.../det/det.txt` 是否生成
   - `outputs/.../tracker/.../*.txt` 是否生成
3) TrackEval 是否能跑起来（你现在默认 evaluate 会跑）。

---

# 交付物

- **修复说明文档**：本文件
- **已合并修复的代码包（preflightfixed_v2）**：用于你直接覆盖工程目录
