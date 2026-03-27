#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune an external ReID encoder on MOT GT identities (MOT17 and/or MOT20).

Why:
  MOT17 test (private detection) is currently bottlenecked by association on hard sequences.
  The default OSNet-AIN (MSMT17-pretrained) is generic and may not generalize to MOT-specific
  blur/occlusion/camera settings. Fine-tuning on MOT GT IDs is a high-ROI way to improve AssA/IDSW
  without changing the tracker architecture.

This script trains a torchreid backbone (default: osnet_ain_x1_0) with softmax + triplet loss.
It crops GT boxes on-the-fly from original frames to avoid massive pre-extraction on disk.
You can choose training datasets with --datasets (e.g. MOT20 only).

Example:
  python -u scripts/train_reid_mot.py \
    --data-root /gemini/code/datasets \
    --output-dir outputs/reid_ft_mot17mot20 \
    --datasets MOT17,MOT20 \
    --backbone osnet_ain_x1_0 \
    --init-weights weight/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth \
    --epochs 30 --batch-size 64 --num-instances 4
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torchreid
from torchreid.reid.data.sampler import RandomIdentitySampler
from torchreid.reid.losses import CrossEntropyLoss, TripletLoss


@dataclass(frozen=True)
class ReIDItem:
    img_path: str
    pid: int
    bbox_xywh: Tuple[float, float, float, float]


class MOTReIDDataset(Dataset):
    def __init__(
        self,
        items: List[ReIDItem],
        tf: transforms.Compose,
        box_expand: float = 1.0,
        box_jitter: float = 0.0,
        box_scale_jitter: float = 0.0,
    ):
        self.items = items
        self.tf = tf
        self.box_expand = float(box_expand) if box_expand is not None else 1.0
        self.box_jitter = float(box_jitter) if box_jitter is not None else 0.0
        self.box_scale_jitter = float(box_scale_jitter) if box_scale_jitter is not None else 0.0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        img = Image.open(it.img_path).convert("RGB")
        img_w, img_h = img.size

        x, y, w, h = it.bbox_xywh
        w = max(float(w), 1.0)
        h = max(float(h), 1.0)
        cx = float(x) + 0.5 * w
        cy = float(y) + 0.5 * h

        # Detector-shift simulation:
        # - scale jitter: imperfect box sizes (tight/loose)
        # - center jitter: imperfect localization
        if self.box_scale_jitter > 0.0:
            sj = float(self.box_scale_jitter)
            s = 1.0 + random.uniform(-sj, sj)
            w = max(w * s, 1.0)
            h = max(h * s, 1.0)
        if self.box_jitter > 0.0:
            jj = float(self.box_jitter)
            cx = cx + random.uniform(-jj, jj) * w
            cy = cy + random.uniform(-jj, jj) * h

        if self.box_expand != 1.0:
            s = float(max(self.box_expand, 1e-6))
            w = max(w * s, 1.0)
            h = max(h * s, 1.0)

        x = cx - 0.5 * w
        y = cy - 0.5 * h

        x1 = max(0, min(int(round(x)), img_w - 1))
        y1 = max(0, min(int(round(y)), img_h - 1))
        x2 = max(x1 + 1, min(int(round(x + w)), img_w))
        y2 = max(y1 + 1, min(int(round(y + h)), img_h))

        crop = img.crop((x1, y1, x2, y2))
        crop = self.tf(crop)
        return crop, int(it.pid)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_mot_gt(
    gt_path: Path,
    img_dir: Path,
    pid_offset_key: str,
    pid_map: Dict[Tuple[str, int], int],
    next_pid: int,
    min_vis: float,
    frame_stride: int,
) -> Tuple[List[ReIDItem], int]:
    """
    Parse MOT gt/gt.txt:
      frame, id, x, y, w, h, conf, class, vis
    """
    items: List[ReIDItem] = []
    if not gt_path.is_file():
        return items, next_pid

    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                frame = int(float(parts[0]))
                track_id = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
            except ValueError:
                continue

            if frame_stride > 1 and (frame % frame_stride != 0):
                continue
            if track_id <= 0:
                continue

            conf = None
            cls = None
            vis = None
            if len(parts) > 6:
                try:
                    conf = float(parts[6])
                except ValueError:
                    conf = None
            if len(parts) > 7:
                try:
                    cls = int(float(parts[7]))
                except ValueError:
                    cls = None
            if len(parts) > 8:
                try:
                    vis = float(parts[8])
                except ValueError:
                    vis = None

            # Standard MOT convention: conf==1 and class==1 means a valid pedestrian GT box.
            if conf is not None and conf != 1.0:
                continue
            if cls is not None and cls != 1:
                continue
            if vis is not None and vis < float(min_vis):
                continue

            img_path = img_dir / f"{frame:06d}.jpg"
            if not img_path.is_file():
                continue

            key = (pid_offset_key, track_id)
            if key not in pid_map:
                pid_map[key] = next_pid
                next_pid += 1
            pid = pid_map[key]

            items.append(ReIDItem(img_path=str(img_path), pid=int(pid), bbox_xywh=(x, y, w, h)))

    return items, next_pid


def build_mot_reid_items(
    data_root: str,
    min_vis: float,
    frame_stride: int,
    max_samples_per_id: int,
    datasets: List[str],
) -> Tuple[List[ReIDItem], int]:
    data_root_p = Path(data_root)

    mot17_seqs = [
        "MOT17-02-FRCNN",
        "MOT17-04-FRCNN",
        "MOT17-05-FRCNN",
        "MOT17-09-FRCNN",
        "MOT17-10-FRCNN",
        "MOT17-11-FRCNN",
        "MOT17-13-FRCNN",
    ]
    mot20_seqs = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]

    pid_map: Dict[Tuple[str, int], int] = {}
    next_pid = 0
    all_items: List[ReIDItem] = []

    ds_set = {str(d).strip().upper() for d in datasets if str(d).strip()}
    if len(ds_set) == 0:
        raise ValueError("datasets is empty. Expected at least one of: MOT17,MOT20")
    invalid = ds_set.difference({"MOT17", "MOT20"})
    if invalid:
        raise ValueError(f"Unsupported datasets={sorted(invalid)}. Allowed: MOT17,MOT20")

    if "MOT17" in ds_set:
        for seq in mot17_seqs:
            gt = data_root_p / "MOT17" / "train" / seq / "gt" / "gt.txt"
            img_dir = data_root_p / "MOT17" / "train" / seq / "img1"
            items, next_pid = _parse_mot_gt(
                gt_path=gt,
                img_dir=img_dir,
                pid_offset_key=f"MOT17:{seq}",
                pid_map=pid_map,
                next_pid=next_pid,
                min_vis=min_vis,
                frame_stride=frame_stride,
            )
            all_items.extend(items)

    if "MOT20" in ds_set:
        for seq in mot20_seqs:
            gt = data_root_p / "MOT20" / "train" / seq / "gt" / "gt.txt"
            img_dir = data_root_p / "MOT20" / "train" / seq / "img1"
            items, next_pid = _parse_mot_gt(
                gt_path=gt,
                img_dir=img_dir,
                pid_offset_key=f"MOT20:{seq}",
                pid_map=pid_map,
                next_pid=next_pid,
                min_vis=min_vis,
                frame_stride=frame_stride,
            )
            all_items.extend(items)

    # Optional subsampling per identity to keep training size manageable.
    if max_samples_per_id and max_samples_per_id > 0:
        by_pid: Dict[int, List[ReIDItem]] = defaultdict(list)
        for it in all_items:
            by_pid[int(it.pid)].append(it)
        sampled: List[ReIDItem] = []
        for pid, items in by_pid.items():
            if len(items) <= max_samples_per_id:
                sampled.extend(items)
            else:
                sampled.extend(random.sample(items, k=int(max_samples_per_id)))
        all_items = sampled

    num_classes = int(next_pid)
    return all_items, num_classes


def _split_train_val_items(
    items: List[ReIDItem],
    val_ratio: float,
    seed: int,
    min_train_per_id: int = 1,
    min_val_per_id: int = 1,
) -> Tuple[List[ReIDItem], List[ReIDItem]]:
    if val_ratio <= 0.0:
        return list(items), []

    by_pid: Dict[int, List[ReIDItem]] = defaultdict(list)
    for it in items:
        by_pid[int(it.pid)].append(it)

    rng = random.Random(int(seed) + 2026)
    train_items: List[ReIDItem] = []
    val_items: List[ReIDItem] = []

    for pid, pid_items in by_pid.items():
        cur = list(pid_items)
        rng.shuffle(cur)

        if len(cur) < (min_train_per_id + min_val_per_id):
            train_items.extend(cur)
            continue

        n_val = int(round(len(cur) * float(val_ratio)))
        n_val = max(n_val, int(min_val_per_id))
        n_val = min(n_val, len(cur) - int(min_train_per_id))
        n_val = max(n_val, 0)

        if n_val == 0:
            train_items.extend(cur)
            continue

        val_items.extend(cur[:n_val])
        train_items.extend(cur[n_val:])

    return train_items, val_items


def _remap_train_val_pids(
    train_items: List[ReIDItem],
    val_items: List[ReIDItem],
) -> Tuple[List[ReIDItem], List[ReIDItem], int]:
    train_pids = sorted({int(it.pid) for it in train_items})
    pid_map = {pid: i for i, pid in enumerate(train_pids)}

    remap_train = [
        ReIDItem(img_path=it.img_path, pid=int(pid_map[int(it.pid)]), bbox_xywh=it.bbox_xywh)
        for it in train_items
        if int(it.pid) in pid_map
    ]
    remap_val = [
        ReIDItem(img_path=it.img_path, pid=int(pid_map[int(it.pid)]), bbox_xywh=it.bbox_xywh)
        for it in val_items
        if int(it.pid) in pid_map
    ]
    return remap_train, remap_val, len(train_pids)


def _cap_items_per_pid(items: List[ReIDItem], max_items: int, seed: int) -> List[ReIDItem]:
    if max_items <= 0 or len(items) <= max_items:
        return items
    by_pid: Dict[int, List[ReIDItem]] = defaultdict(list)
    for it in items:
        by_pid[int(it.pid)].append(it)
    rng = random.Random(int(seed) + 99)
    pids = sorted(by_pid.keys())
    for pid in pids:
        rng.shuffle(by_pid[pid])

    capped: List[ReIDItem] = []
    ptr = {pid: 0 for pid in pids}
    while len(capped) < max_items:
        progressed = False
        for pid in pids:
            if len(capped) >= max_items:
                break
            i = ptr[pid]
            if i < len(by_pid[pid]):
                capped.append(by_pid[pid][i])
                ptr[pid] = i + 1
                progressed = True
        if not progressed:
            break
    return capped


@torch.no_grad()
def evaluate_reid_retrieval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    query_chunk_size: int = 256,
) -> Dict[str, float]:
    model.eval()
    all_feats: List[torch.Tensor] = []
    all_pids: List[torch.Tensor] = []

    for imgs, pids in loader:
        imgs = imgs.to(device=device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=bool(use_amp)):
            out = model(imgs)
        if isinstance(out, (tuple, list)):
            # torchreid triplet models return (logits, features)
            feats = out[1] if len(out) > 1 else out[0]
        else:
            feats = out
        feats = F.normalize(feats.float(), p=2, dim=-1)
        all_feats.append(feats.detach())
        all_pids.append(pids.to(device=device, non_blocking=True))

    if len(all_feats) == 0:
        return {"rank1": 0.0, "map": 0.0, "num_val": 0.0, "num_valid_queries": 0.0}

    feats = torch.cat(all_feats, dim=0)  # (N, D)
    pids = torch.cat(all_pids, dim=0).long()  # (N,)
    n = int(feats.shape[0])
    if n <= 1:
        return {"rank1": 0.0, "map": 0.0, "num_val": float(n), "num_valid_queries": 0.0}

    correct1 = 0.0
    ap_sum = 0.0
    valid_queries = 0
    gallery_feats_t = feats.t().contiguous()  # (D, N)
    arange_cache = torch.arange(n, device=device, dtype=torch.long)

    for start in range(0, n, int(query_chunk_size)):
        end = min(start + int(query_chunk_size), n)
        qf = feats[start:end]  # (B, D)
        qpid = pids[start:end]  # (B,)
        sim = qf @ gallery_feats_t  # (B, N)

        # Exclude self from gallery.
        row = torch.arange(end - start, device=device, dtype=torch.long)
        sim[row, arange_cache[start:end]] = -1e9

        order = torch.argsort(sim, dim=1, descending=True)  # (B, N)
        sorted_pids = pids[order]  # (B, N)
        matches = (sorted_pids == qpid.view(-1, 1))
        pos_counts = matches.sum(dim=1)
        valid_mask = pos_counts > 0
        if not valid_mask.any():
            continue

        valid_idx = valid_mask.nonzero(as_tuple=True)[0]
        valid_queries += int(valid_idx.numel())
        top1 = matches[valid_idx, 0].float().sum().item()
        correct1 += float(top1)

        # AP per valid query.
        for vi in valid_idx.tolist():
            m = matches[vi].float()  # (N,)
            denom = float(m.sum().item())
            if denom <= 0.0:
                continue
            cumsum = torch.cumsum(m, dim=0)
            precision = cumsum / (arange_cache.float() + 1.0)
            ap = (precision * m).sum() / denom
            ap_sum += float(ap.item())

    rank1 = (correct1 / max(valid_queries, 1)) * 100.0
    map_score = (ap_sum / max(valid_queries, 1)) * 100.0
    return {
        "rank1": float(rank1),
        "map": float(map_score),
        "num_val": float(n),
        "num_valid_queries": float(valid_queries),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--datasets",
        default="MOT17,MOT20",
        help="Comma-separated training datasets. Supported: MOT17,MOT20 (e.g. MOT20).",
    )
    ap.add_argument("--backbone", default="osnet_ain_x1_0")
    ap.add_argument("--init-weights", default=None, help="Optional pretrained weights to load (state_dict).")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-instances", type=int, default=4, help="K instances per identity in a batch (PK sampler).")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--min-vis", type=float, default=0.1)
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--max-samples-per-id", type=int, default=80)
    ap.add_argument("--box-expand", type=float, default=1.0)
    ap.add_argument("--box-jitter", type=float, default=0.0, help="Relative center jitter (e.g. 0.05).")
    ap.add_argument("--box-scale-jitter", type=float, default=0.0, help="Relative size jitter (e.g. 0.15).")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Per-ID validation split ratio.")
    ap.add_argument("--val-batch-size", type=int, default=128)
    ap.add_argument("--val-query-chunk", type=int, default=256)
    ap.add_argument("--val-max-items", type=int, default=6000, help="Cap validation items for faster per-epoch eval.")
    ap.add_argument("--early-stop-patience", type=int, default=6, help="Stop if no mAP improvement for N epochs.")
    ap.add_argument("--early-stop-min-delta", type=float, default=0.05, help="Minimum mAP improvement to reset patience.")
    ap.add_argument("--save-all-epochs", action="store_true", help="If set, save reid_epXXX.pth each epoch.")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    _seed_everything(int(args.seed))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = [s.strip().upper() for s in str(args.datasets).split(",") if s.strip()]
    print(f"Using datasets: {datasets}")

    items, _ = build_mot_reid_items(
        data_root=args.data_root,
        min_vis=float(args.min_vis),
        frame_stride=int(args.frame_stride),
        max_samples_per_id=int(args.max_samples_per_id),
        datasets=datasets,
    )
    if len(items) < 10:
        raise RuntimeError(f"ReID dataset too small: num_items={len(items)}")

    train_items_raw, val_items_raw = _split_train_val_items(
        items=items,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        min_train_per_id=1,
        min_val_per_id=1,
    )
    train_items, val_items, num_classes = _remap_train_val_pids(train_items_raw, val_items_raw)
    val_items = _cap_items_per_pid(val_items, max_items=int(args.val_max_items), seed=int(args.seed))

    if num_classes < 2 or len(train_items) < 10:
        raise RuntimeError(
            f"ReID train split too small after split/remap: num_classes={num_classes}, num_items={len(train_items)}"
        )
    if len(val_items) < 2:
        raise RuntimeError(
            "Validation split has fewer than 2 samples. Increase --val-ratio or dataset size."
        )
    print(
        f"Dataset split: train_items={len(train_items)} val_items={len(val_items)} "
        f"classes={num_classes} val_ratio={float(args.val_ratio):.3f}"
    )

    tf_train = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = MOTReIDDataset(
        items=train_items,
        tf=tf_train,
        box_expand=float(args.box_expand),
        box_jitter=float(args.box_jitter),
        box_scale_jitter=float(args.box_scale_jitter),
    )
    val_dataset = MOTReIDDataset(
        items=val_items,
        tf=tf_val,
        box_expand=float(args.box_expand),
        box_jitter=0.0,
        box_scale_jitter=0.0,
    )

    # torchreid's RandomIdentitySampler expects data_source as a list of tuples where pid is at index 1.
    # We keep a lightweight parallel list for the sampler to index, while the dataset holds full metadata.
    sampler_source = [(it.img_path, it.pid, 0, 0) for it in train_items]
    sampler = RandomIdentitySampler(
        data_source=sampler_source,
        batch_size=int(args.batch_size),
        num_instances=int(args.num_instances),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        sampler=sampler,
        num_workers=min(8, os.cpu_count() or 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.val_batch_size),
        shuffle=False,
        num_workers=min(8, os.cpu_count() or 4),
        pin_memory=True,
        drop_last=False,
    )

    model = torchreid.models.build_model(
        name=str(args.backbone),
        num_classes=int(num_classes),
        loss="triplet",
        pretrained=False,
        use_gpu=torch.cuda.is_available(),
    )

    if args.init_weights:
        init_path = str(args.init_weights)
        if not os.path.isfile(init_path):
            raise FileNotFoundError(f"--init-weights not found: {init_path}")
        ckpt = torch.load(init_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if not isinstance(ckpt, dict):
            raise ValueError("Unsupported init weights format (expected a state_dict dict)")
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"Loaded init weights: missing={len(missing)} unexpected={len(unexpected)}")

    model = model.to(device)

    ce_loss = CrossEntropyLoss(num_classes=int(num_classes), label_smooth=0.1).to(device)
    tri_loss = TripletLoss(margin=0.3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(args.epochs), 1))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))

    best_loss = float("inf")
    best_map = -1.0
    best_rank1 = -1.0
    no_improve_epochs = 0
    metrics_csv_path = out_dir / "metrics.csv"
    with metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_rank1",
            "val_map",
            "val_num",
            "val_valid_queries",
            "lr",
            "best_map",
            "best_rank1",
            "no_improve_epochs",
        ])

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        n = 0
        for imgs, pids in loader:
            imgs = imgs.to(device=device, non_blocking=True)
            pids = pids.to(device=device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                logits, feats = model(imgs)
            # torchreid TripletLoss internally builds an fp32 distance matrix and then
            # uses addmm_; forcing fp32 inputs avoids Half/Float dtype mismatch under AMP.
            logits = logits.float()
            feats = feats.float()
            loss_ce = ce_loss(logits, pids)
            loss_tri = tri_loss(feats, pids)
            loss = loss_ce + loss_tri
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item())
            n += 1
        scheduler.step()

        avg = running / max(n, 1)
        lr = float(optimizer.param_groups[0]["lr"])
        val_metrics = evaluate_reid_retrieval(
            model=model,
            loader=val_loader,
            device=device,
            use_amp=bool(args.amp),
            query_chunk_size=int(args.val_query_chunk),
        )
        cur_map = float(val_metrics["map"])
        cur_rank1 = float(val_metrics["rank1"])

        if bool(args.save_all_epochs):
            ckpt_path = out_dir / f"reid_ep{epoch:03d}.pth"
            torch.save(model.state_dict(), ckpt_path)

        improved = (cur_map - best_map) > float(args.early_stop_min_delta)
        if improved:
            best_map = cur_map
            best_rank1 = max(best_rank1, cur_rank1)
            no_improve_epochs = 0
            torch.save(model.state_dict(), out_dir / "reid_best.pth")
        else:
            no_improve_epochs += 1
            best_rank1 = max(best_rank1, cur_rank1)

        if avg < best_loss:
            best_loss = avg

        print(
            f"Epoch {epoch:03d}/{int(args.epochs)} | loss={avg:.4f} | "
            f"val_rank1={cur_rank1:.2f} | val_mAP={cur_map:.2f} | "
            f"lr={lr:.2e} | train_items={len(train_items)} | val_items={int(val_metrics['num_val'])} | "
            f"best_mAP={best_map:.2f} | wait={no_improve_epochs}"
        )

        with metrics_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{avg:.6f}",
                f"{cur_rank1:.4f}",
                f"{cur_map:.4f}",
                int(val_metrics["num_val"]),
                int(val_metrics["num_valid_queries"]),
                f"{lr:.8e}",
                f"{best_map:.4f}",
                f"{best_rank1:.4f}",
                no_improve_epochs,
            ])

        if no_improve_epochs >= int(args.early_stop_patience):
            print(
                f"Early stop at epoch {epoch}: no mAP improvement > {float(args.early_stop_min_delta):.4f} "
                f"for {int(args.early_stop_patience)} epochs."
            )
            break

    print(f"Done. Best train loss: {best_loss:.4f}")
    print(f"Best val mAP: {best_map:.2f}, Best val Rank-1: {best_rank1:.2f}")
    print(f"Best weights: {out_dir / 'reid_best.pth'}")
    print(f"Metrics CSV: {metrics_csv_path}")


if __name__ == "__main__":
    main()
