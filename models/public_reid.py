# Copyright (c) Ruopeng Gao. All Rights Reserved.
# Public detections + external ReID embeddings (crop -> encoder)

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from torchvision.ops import roi_align


class PublicDetReIDEncoder:
    def __init__(
        self,
        backbone: nn.Module,
        proj: nn.Module,
        tf: transforms.Compose,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int = 64,
        l2_norm: bool = True,
        input_hw: tuple[int, int] = (256, 128),
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        use_roi_align: bool = True,
        box_expand: float = 1.0,
    ):
        self.backbone = backbone
        self.proj = proj
        self.tf = tf
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.l2_norm = l2_norm
        self.out_dim = int(getattr(self.proj, "out_features", 0))
        self.input_h = int(input_hw[0])
        self.input_w = int(input_hw[1])
        self.use_roi_align = bool(use_roi_align)
        self.box_expand = float(box_expand) if box_expand is not None else 1.0
        # Normalization tensors (for ROIAlign path)
        self._mean = torch.tensor(mean, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(std, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        self.backbone.eval()
        self.proj.eval()

    @torch.no_grad()
    def encode(self, image_path: str, boxes_xywh: torch.Tensor) -> torch.Tensor:
        """Encode all boxes in one frame into embeddings."""
        img = Image.open(image_path).convert("RGB")
        img_w, img_h = img.size

        if not torch.is_tensor(boxes_xywh):
            boxes_xywh = torch.as_tensor(boxes_xywh)
        boxes_xywh = boxes_xywh.detach()
        num_boxes = int(boxes_xywh.shape[0])
        if num_boxes == 0:
            return torch.zeros((0, self.out_dim), device=self.device, dtype=self.dtype)

        out = torch.zeros((num_boxes, self.out_dim), device=self.device, dtype=self.dtype)

        # Fast path: ROIAlign crops on GPU (avoids Python/PIL per-box loop).
        if self.use_roi_align:
            # NOTE: Use torchvision's `to_tensor` for correctness (float32, CHW, [0,1]).
            # A previous "torch-only bytes" shortcut could leave values in [0,255],
            # which breaks ImageNet normalization and hurts embeddings.
            from torchvision.transforms.functional import to_tensor

            img_t = to_tensor(img).to(device=self.device, dtype=torch.float32).unsqueeze(0)  # (1,3,H,W)

            x = boxes_xywh[:, 0].to(dtype=torch.float32)
            y = boxes_xywh[:, 1].to(dtype=torch.float32)
            w = boxes_xywh[:, 2].to(dtype=torch.float32).clamp(min=1.0)
            h = boxes_xywh[:, 3].to(dtype=torch.float32).clamp(min=1.0)

            if self.box_expand != 1.0:
                s = float(max(self.box_expand, 1e-6))
                cx = x + 0.5 * w
                cy = y + 0.5 * h
                w = w * s
                h = h * s
                x = cx - 0.5 * w
                y = cy - 0.5 * h

            x1 = x.clamp(min=0.0, max=float(img_w - 1))
            y1 = y.clamp(min=0.0, max=float(img_h - 1))
            x2 = (x + w).clamp(min=0.0, max=float(img_w))
            y2 = (y + h).clamp(min=0.0, max=float(img_h))
            # Enforce minimal box size
            x2 = torch.maximum(x2, x1 + 1.0)
            y2 = torch.maximum(y2, y1 + 1.0)
            valid = (x2 > x1) & (y2 > y1)

            if valid.any():
                valid_idx = valid.nonzero(as_tuple=True)[0]
                rois = torch.zeros((int(valid_idx.numel()), 5), device=self.device, dtype=torch.float32)
                rois[:, 0] = 0  # batch idx
                rois[:, 1] = x1[valid_idx]
                rois[:, 2] = y1[valid_idx]
                rois[:, 3] = x2[valid_idx]
                rois[:, 4] = y2[valid_idx]

                crops = roi_align(
                    img_t,
                    rois,
                    output_size=(self.input_h, self.input_w),
                    spatial_scale=1.0,
                    aligned=True,
                ).to(dtype=self.dtype)
                crops = (crops - self._mean) / self._std

                # Forward in chunks to control memory
                emb_chunks = []
                for start in range(0, crops.shape[0], max(int(self.batch_size), 1)):
                    batch = crops[start : start + int(self.batch_size)]
                    feats = self.backbone(batch)
                    if isinstance(feats, (tuple, list)):
                        feats = feats[0]
                    if feats.dim() > 2:
                        feats = torch.flatten(feats, 1)
                    emb = self.proj(feats)
                    if self.l2_norm:
                        emb = F.normalize(emb, p=2, dim=-1)
                    emb_chunks.append(emb)
                emb_all = torch.cat(emb_chunks, dim=0) if emb_chunks else crops.new_zeros((0, self.out_dim))
                out[valid_idx] = emb_all
                return out

        # Slow fallback: PIL per-box crops (kept for robustness)
        boxes = boxes_xywh.detach().cpu().tolist()
        crops = []
        idxs = []

        def _flush():
            if not crops:
                return
            batch = torch.stack(crops, dim=0).to(device=self.device, dtype=self.dtype)
            feats = self.backbone(batch)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            if feats.dim() > 2:
                feats = torch.flatten(feats, 1)
            emb = self.proj(feats)
            if self.l2_norm:
                emb = F.normalize(emb, p=2, dim=-1)
            for i, e in zip(idxs, emb):
                out[i] = e
            crops.clear()
            idxs.clear()

        for i, b in enumerate(boxes):
            x, y, w, h = b
            w = max(float(w), 1.0)
            h = max(float(h), 1.0)
            if self.box_expand != 1.0:
                s = float(max(self.box_expand, 1e-6))
                cx = float(x) + 0.5 * w
                cy = float(y) + 0.5 * h
                w = w * s
                h = h * s
                x = cx - 0.5 * w
                y = cy - 0.5 * h

            x1 = max(0, min(int(round(x)), img_w - 1))
            y1 = max(0, min(int(round(y)), img_h - 1))
            x2 = max(x1 + 1, min(int(round(x + w)), img_w))
            y2 = max(y1 + 1, min(int(round(y + h)), img_h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img.crop((x1, y1, x2, y2))
            crops.append(self.tf(crop))
            idxs.append(i)
            if len(crops) >= self.batch_size:
                _flush()

        _flush()
        return out


def _strip_known_prefixes(state_dict: dict) -> dict:
    prefixes = ["module.", "model.", "backbone.", "encoder.", "reid."]
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def _extract_state_dict(ckpt: Any) -> dict:
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "net", "network"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError("Unsupported checkpoint format for PUBLIC_REID_WEIGHTS")


def _remove_classification_head(model: nn.Module) -> None:
    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    if hasattr(model, "classifier"):
        model.classifier = nn.Identity()
    if hasattr(model, "classif"):
        model.classif = nn.Identity()
    if hasattr(model, "head") and isinstance(model.head, nn.Module):
        model.head = nn.Identity()
    if hasattr(model, "logits") and isinstance(model.logits, nn.Module):
        model.logits = nn.Identity()


def _infer_feat_dim(backbone: nn.Module, input_hw: tuple[int, int]) -> int:
    backbone = backbone.eval()
    dummy = torch.zeros((1, 3, input_hw[0], input_hw[1]), dtype=torch.float32)
    with torch.no_grad():
        feats = backbone(dummy)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if feats.dim() > 2:
            feats = torch.flatten(feats, 1)
    return int(feats.shape[1])


def _build_backbone(backbone_name: str, pretrained: bool) -> nn.Module:
    if backbone_name == "torchvision_resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        _remove_classification_head(backbone)
        return backbone
    if backbone_name == "torchvision_resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        _remove_classification_head(backbone)
        return backbone
    if backbone_name.startswith("torchreid:"):
        try:
            import torchreid
        except ImportError as exc:
            raise ImportError(
                "PUBLIC_REID_BACKBONE uses 'torchreid', but torchreid is not installed. "
                "Install it with: pip install torchreid"
            ) from exc
        model_name = backbone_name.split(":", 1)[1]
        tr_models = torchreid.models
        if hasattr(tr_models, "build_model"):
            backbone = tr_models.build_model(name=model_name, num_classes=1, pretrained=pretrained)
        else:
            fn = getattr(tr_models, model_name, None)
            if fn is None:
                raise ValueError(f"Unsupported torchreid backbone: {model_name}")
            backbone = fn(num_classes=1, pretrained=pretrained)
        _remove_classification_head(backbone)
        return backbone
    raise ValueError(f"Unsupported PUBLIC_REID_BACKBONE: {backbone_name}")


def build_public_reid_encoder(
    config: dict,
    device: torch.device,
    feature_dim: int,
    dtype: torch.dtype = torch.float32,
) -> PublicDetReIDEncoder:
    backbone_name = str(config.get("PUBLIC_REID_BACKBONE", "torchvision_resnet50"))
    weights_path = config.get("PUBLIC_REID_WEIGHTS", None)
    if isinstance(weights_path, str) and weights_path.strip() == "":
        weights_path = None
    pretrained = bool(config.get("PUBLIC_REID_PRETRAINED", True))
    input_h = int(config.get("PUBLIC_REID_INPUT_H", 256))
    input_w = int(config.get("PUBLIC_REID_INPUT_W", 128))
    batch_size = int(config.get("PUBLIC_REID_BATCH_SIZE", 64))
    l2_norm = bool(config.get("PUBLIC_REID_L2_NORM", True))

    backbone = _build_backbone(backbone_name=backbone_name, pretrained=pretrained)

    if weights_path:
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"PUBLIC_REID_WEIGHTS not found: {weights_path}")
        ckpt = torch.load(weights_path, map_location="cpu")
        state = _extract_state_dict(ckpt)
        state = _strip_known_prefixes(state)
        backbone.load_state_dict(state, strict=False)

    feat_dim = _infer_feat_dim(backbone.cpu().float(), input_hw=(input_h, input_w))

    proj_seed = int(config.get("PUBLIC_REID_PROJ_SEED", 12345))
    g = torch.Generator(device="cpu").manual_seed(proj_seed)
    proj = nn.Linear(feat_dim, feature_dim, bias=False)
    with torch.no_grad():
        if int(feat_dim) == int(feature_dim):
            # Preserve the backbone embedding space if dimensions already match.
            # Random projection can distort pairwise distances and hurt ReID-based association.
            proj.weight.copy_(torch.eye(feat_dim, device="cpu", dtype=proj.weight.dtype))
        else:
            proj_weight = torch.randn(proj.weight.shape, generator=g, device="cpu", dtype=proj.weight.dtype)
            proj.weight.copy_(proj_weight / (feat_dim ** 0.5))

    backbone = backbone.to(device=device, dtype=dtype)
    proj = proj.to(device=device, dtype=dtype)
    backbone.eval()
    proj.eval()

    tf = transforms.Compose([
        transforms.Resize((input_h, input_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return PublicDetReIDEncoder(
        backbone=backbone,
        proj=proj,
        tf=tf,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        l2_norm=l2_norm,
        input_hw=(input_h, input_w),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        use_roi_align=bool(config.get("PUBLIC_REID_USE_ROI_ALIGN", True)),
        box_expand=float(config.get("PUBLIC_REID_BOX_EXPAND", 1.0)),
    )
