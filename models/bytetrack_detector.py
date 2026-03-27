# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ByteTrack (YOLOX) detector wrapper for private detections.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

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

    Output format:
        List[(x, y, w, h, conf)] in pixel xywh.
    """

    def __init__(self, cfg: ByteTrackDetConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        # Lazy import so project runs without ByteTrack installed.
        try:
            from yolox.exp import get_exp
        except Exception as exc:
            raise ImportError(
                "Cannot import YOLOX/ByteTrack. Install it with: pip install -e third_party/ByteTrack"
            ) from exc

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

        if cfg.fp16:
            self.model.half()

        self.test_size = cfg.test_size if cfg.test_size is not None else tuple(self.exp.test_size)
        self.num_classes = getattr(self.exp, "num_classes", 1)

    @torch.no_grad()
    def detect(self, image_path: str) -> List[Tuple[float, float, float, float, float]]:
        """Run detector on a single frame."""
        try:
            import cv2
        except Exception as exc:
            raise ImportError("opencv-python is required for ByteTrack detector inference.") from exc

        from yolox.data.data_augment import preproc
        from yolox.utils import postprocess

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        img_h, img_w = img_bgr.shape[:2]
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
        dets[:, :4] /= ratio
        dets[:, 0::2] = np.clip(dets[:, 0::2], 0, img_w - 1)
        dets[:, 1::2] = np.clip(dets[:, 1::2], 0, img_h - 1)

        results = []
        for x1, y1, x2, y2, obj_conf, cls_conf, cls_id in dets:
            conf = float(obj_conf * cls_conf)
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
