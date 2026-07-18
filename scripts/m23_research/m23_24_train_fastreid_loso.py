#!/usr/bin/env python3
from __future__ import annotations

"""Strict sequence-LOSO FastReID fine-tuning for the M23 association graph.

The held MOT20 sequence is never read while constructing the training/validation
sets or selecting a checkpoint.  The initial checkpoint is MOT17-only, so the
outer held MOT20 sequence is not present in the initialization data either.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms


REPO = Path(__file__).resolve().parents[2]
DEFAULT_BOT_ROOT = REPO / "external" / "BoT-SORT-main"
MOT20_SEQS = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")


@dataclass(frozen=True)
class Item:
    image: str
    pid: int
    bbox_xywh: Tuple[float, float, float, float]
    seq: str
    frame: int


class CropDataset(Dataset):
    def __init__(self, items: Sequence[Item], transform, box_expand: float) -> None:
        self.items = list(items)
        self.transform = transform
        self.box_expand = float(box_expand)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        image = Image.open(item.image).convert("RGB")
        image_w, image_h = image.size
        x, y, w, h = item.bbox_xywh
        w = max(float(w) * self.box_expand, 1.0)
        h = max(float(h) * self.box_expand, 1.0)
        cx = float(x) + 0.5 * float(item.bbox_xywh[2])
        cy = float(y) + 0.5 * float(item.bbox_xywh[3])
        x1 = max(0, min(int(round(cx - 0.5 * w)), image_w - 1))
        y1 = max(0, min(int(round(cy - 0.5 * h)), image_h - 1))
        x2 = max(x1 + 1, min(int(round(cx + 0.5 * w)), image_w))
        y2 = max(y1 + 1, min(int(round(cy + 0.5 * h)), image_h))
        crop = self.transform(image.crop((x1, y1, x2, y2)))
        # FastReID Baseline performs its own 0..255 mean/std normalization.
        return crop.mul(255.0), int(item.pid)


class RandomIdentityBatchSampler(Sampler[List[int]]):
    """Deterministic-per-epoch P x K sampler without a torchreid dependency."""

    def __init__(self, items: Sequence[Item], batch_size: int, num_instances: int, seed: int) -> None:
        if batch_size % num_instances != 0:
            raise ValueError("batch_size must be divisible by num_instances")
        self.batch_size = int(batch_size)
        self.num_instances = int(num_instances)
        self.pids_per_batch = self.batch_size // self.num_instances
        self.seed = int(seed)
        self.epoch = 0
        self.index_by_pid: Dict[int, List[int]] = defaultdict(list)
        for index, item in enumerate(items):
            self.index_by_pid[int(item.pid)].append(index)
        self.length = self._estimate_length()

    def _estimate_length(self) -> int:
        chunks = sum(max(1, math.ceil(len(rows) / self.num_instances)) for rows in self.index_by_pid.values())
        return max(1, chunks // self.pids_per_batch)

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        chunks: Dict[int, List[List[int]]] = {}
        for pid, source in self.index_by_pid.items():
            rows = list(source)
            rng.shuffle(rows)
            if len(rows) < self.num_instances:
                rows.extend(rng.choices(rows, k=self.num_instances - len(rows)))
            remainder = len(rows) % self.num_instances
            if remainder:
                rows.extend(rng.choices(rows, k=self.num_instances - remainder))
            chunks[pid] = [rows[i : i + self.num_instances] for i in range(0, len(rows), self.num_instances)]
            rng.shuffle(chunks[pid])
        available = [pid for pid, pid_chunks in chunks.items() if pid_chunks]
        while len(available) >= self.pids_per_batch:
            selected = rng.sample(available, self.pids_per_batch)
            batch: List[int] = []
            for pid in selected:
                batch.extend(chunks[pid].pop())
                if not chunks[pid]:
                    available.remove(pid)
            yield batch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sequence(
    data_root: Path,
    seq: str,
    pid_start: int,
    min_visibility: float,
    frame_stride: int,
) -> Tuple[List[Item], int]:
    seq_root = data_root / seq
    gt_path = seq_root / "gt" / "gt.txt"
    image_root = seq_root / "img1"
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)
    pid_map: Dict[int, int] = {}
    items: List[Item] = []
    next_pid = int(pid_start)
    with gt_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip().split(",")
            if len(fields) < 6:
                continue
            frame = int(float(fields[0]))
            source_pid = int(float(fields[1]))
            if source_pid <= 0 or (frame_stride > 1 and frame % frame_stride != 0):
                continue
            confidence = float(fields[6]) if len(fields) > 6 else 1.0
            category = int(float(fields[7])) if len(fields) > 7 else 1
            visibility = float(fields[8]) if len(fields) > 8 else 1.0
            if confidence != 1.0 or category != 1 or visibility < min_visibility:
                continue
            image_path = image_root / f"{frame:06d}.jpg"
            if not image_path.is_file():
                continue
            if source_pid not in pid_map:
                pid_map[source_pid] = next_pid
                next_pid += 1
            box = tuple(float(value) for value in fields[2:6])
            items.append(Item(str(image_path), pid_map[source_pid], box, seq, frame))
    return items, next_pid


def cap_per_identity(items: Sequence[Item], maximum: int, seed: int) -> List[Item]:
    if maximum <= 0:
        return list(items)
    grouped: Dict[int, List[Item]] = defaultdict(list)
    for item in items:
        grouped[int(item.pid)].append(item)
    output: List[Item] = []
    for pid in sorted(grouped):
        current = sorted(grouped[pid], key=lambda x: (x.frame, x.image))
        if len(current) > maximum:
            rng = random.Random(seed + 1009 * pid)
            current = sorted(rng.sample(current, maximum), key=lambda x: (x.frame, x.image))
        output.extend(current)
    return output


def split_by_identity(
    items: Sequence[Item], val_ratio: float, seed: int
) -> Tuple[List[Item], List[Item]]:
    grouped: Dict[int, List[Item]] = defaultdict(list)
    for item in items:
        grouped[int(item.pid)].append(item)
    train: List[Item] = []
    val: List[Item] = []
    for pid in sorted(grouped):
        current = sorted(grouped[pid], key=lambda x: (x.frame, x.image))
        rng = random.Random(seed + 7919 * pid)
        rng.shuffle(current)
        if len(current) < 3 or val_ratio <= 0:
            train.extend(current)
            continue
        count = max(1, min(len(current) - 2, int(round(len(current) * val_ratio))))
        val.extend(current[:count])
        train.extend(current[count:])
    return train, val


def remap_pids(train: Sequence[Item], val: Sequence[Item]) -> Tuple[List[Item], List[Item], int]:
    mapping = {pid: index for index, pid in enumerate(sorted({int(x.pid) for x in train}))}

    def apply(rows: Iterable[Item]) -> List[Item]:
        return [Item(x.image, mapping[int(x.pid)], x.bbox_xywh, x.seq, x.frame) for x in rows]

    return apply(train), apply(val), len(mapping)


def cap_validation(items: Sequence[Item], maximum: int, seed: int) -> List[Item]:
    if maximum <= 0 or len(items) <= maximum:
        return list(items)
    rng = random.Random(seed + 20260718)
    indices = sorted(rng.sample(range(len(items)), maximum))
    return [items[index] for index in indices]


@torch.no_grad()
def retrieval_metrics(model, loader: DataLoader, device: torch.device, amp: bool) -> Dict[str, float]:
    model.eval()
    features: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for images, pids in loader:
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            embedding = model(images)
        features.append(F.normalize(embedding.float(), dim=1).cpu())
        labels.append(pids.long().cpu())
    if not features:
        return {"rank1": 0.0, "map": 0.0, "queries": 0.0}
    feats = torch.cat(features)
    pids = torch.cat(labels)
    valid_queries = 0
    correct = 0
    ap_sum = 0.0
    for start in range(0, len(feats), 256):
        end = min(start + 256, len(feats))
        similarity = feats[start:end] @ feats.T
        rows = torch.arange(end - start)
        similarity[rows, torch.arange(start, end)] = -1e9
        order = similarity.argsort(dim=1, descending=True)
        matches = pids[order].eq(pids[start:end, None])
        for row in range(end - start):
            positives = matches[row]
            count = int(positives.sum())
            if count == 0:
                continue
            valid_queries += 1
            correct += int(positives[0])
            precision = positives.float().cumsum(0) / torch.arange(1, len(positives) + 1)
            ap_sum += float((precision * positives.float()).sum() / count)
    denominator = max(valid_queries, 1)
    return {
        "rank1": 100.0 * correct / denominator,
        "map": 100.0 * ap_sum / denominator,
        "queries": float(valid_queries),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-seq", required=True, choices=MOT20_SEQS)
    parser.add_argument("--data-root", default="datasets/MOT20/train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bot-sort-root", default=str(DEFAULT_BOT_ROOT))
    parser.add_argument("--config", default="fast_reid/configs/MOT20/sbs_S50.yml")
    parser.add_argument("--init-weights", default="pretrained/mot17_sbs_S50.pth")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--max-samples-per-id", type=int, default=60)
    parser.add_argument("--min-visibility", type=float, default=0.1)
    parser.add_argument("--box-expand", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--val-max-items", type=int, default=4000)
    parser.add_argument("--base-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2324)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--max-train-batches", type=int, default=0, help="Smoke-only batch cap; 0 is full epoch.")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("M23-24 FastReID training requires CUDA")
    device = torch.device("cuda")
    bot_root = Path(args.bot_sort_root).resolve()
    config_path = (bot_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    init_path = (bot_root / args.init_weights).resolve() if not Path(args.init_weights).is_absolute() else Path(args.init_weights)
    if not config_path.is_file() or not init_path.is_file():
        raise FileNotFoundError(f"config={config_path} init={init_path}")
    train_sequences = [seq for seq in MOT20_SEQS if seq != args.held_seq]
    all_items: List[Item] = []
    next_pid = 0
    counts: Dict[str, Dict[str, int]] = {}
    for seq in train_sequences:
        sequence_items, next_pid = parse_sequence(
            Path(args.data_root), seq, next_pid, args.min_visibility, args.frame_stride
        )
        counts[seq] = {"raw_items": len(sequence_items), "identities": len({x.pid for x in sequence_items})}
        all_items.extend(sequence_items)
    all_items = cap_per_identity(all_items, args.max_samples_per_id, args.seed)
    train_raw, val_raw = split_by_identity(all_items, args.val_ratio, args.seed)
    train_items, val_items, num_classes = remap_pids(train_raw, val_raw)
    val_items = cap_validation(val_items, args.val_max_items, args.seed)
    if num_classes < args.batch_size // args.num_instances or len(train_items) < args.batch_size:
        raise RuntimeError(f"insufficient training data: items={len(train_items)} classes={num_classes}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "experiment": "M23-24 strict sequence-LOSO FastReID fine-tuning",
        "held_sequence": args.held_seq,
        "training_sequences": train_sequences,
        "held_sequence_read_during_training": False,
        "checkpoint_selection": "inner per-identity validation split from training sequences only",
        "initialization": {"path": str(init_path), "sha256": sha256(init_path), "training_domain": "MOT17"},
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "counts_before_cap": counts,
        "train_items": len(train_items),
        "val_items": len(val_items),
        "num_classes": num_classes,
        "args": vars(args),
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    train_transform = transforms.Compose([
        transforms.Resize((384, 128), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 1.5))], p=0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.18), value="random"),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((384, 128), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    train_dataset = CropDataset(train_items, train_transform, args.box_expand)
    val_dataset = CropDataset(val_items, val_transform, args.box_expand)

    sampler = RandomIdentityBatchSampler(train_items, args.batch_size, args.num_instances, args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(args.batch_size, 128),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if str(bot_root) not in sys.path:
        sys.path.insert(0, str(bot_root))
    from fast_reid.fastreid.config import get_cfg
    from fast_reid.fastreid.modeling.meta_arch import build_model
    from fast_reid.fastreid.utils.checkpoint import Checkpointer
    from fast_reid.fastreid.utils.events import EventStorage

    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.HEADS.NUM_CLASSES = int(num_classes)
    cfg.freeze()
    model = build_model(cfg)
    Checkpointer(model).load(str(init_path))
    model.to(device)

    backbone_parameters = list(model.backbone.parameters())
    head_parameters = list(model.heads.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.base_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(["epoch", "train_loss", "val_rank1", "val_map", "queries", "backbone_lr", "head_lr", "batches"])

    best_map = -1.0
    wait = 0
    for epoch in range(1, args.epochs + 1):
        cosine = 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / max(args.epochs, 1)))
        optimizer.param_groups[0]["lr"] = 0.0 if epoch <= args.freeze_backbone_epochs else args.base_lr * cosine
        optimizer.param_groups[1]["lr"] = args.head_lr * cosine
        model.train()
        total_loss = 0.0
        batches = 0
        with EventStorage(start_iter=(epoch - 1) * max(len(train_loader), 1)) as storage:
            for images, pids in train_loader:
                images = images.to(device, non_blocking=True)
                pids = pids.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    losses = model({"images": images, "targets": pids})
                    loss = sum(losses.values())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach())
                batches += 1
                storage.step()
                if batches == 1 or batches % 100 == 0:
                    print(
                        f"[M23-24 train] held={args.held_seq} epoch={epoch}/{args.epochs} "
                        f"batch={batches}/{len(train_loader)} running_loss={total_loss / batches:.5f}",
                        flush=True,
                    )
                if args.max_train_batches > 0 and batches >= args.max_train_batches:
                    break
        values = retrieval_metrics(model, val_loader, device, args.amp)
        average_loss = total_loss / max(batches, 1)
        improved = values["map"] > best_map + 0.02
        if improved:
            best_map = values["map"]
            wait = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "held_sequence": args.held_seq, "val_map": best_map},
                output_dir / "model_best.pth",
            )
        else:
            wait += 1
        with metrics_path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow([
                epoch,
                f"{average_loss:.6f}",
                f"{values['rank1']:.6f}",
                f"{values['map']:.6f}",
                int(values["queries"]),
                f"{optimizer.param_groups[0]['lr']:.8e}",
                f"{optimizer.param_groups[1]['lr']:.8e}",
                batches,
            ])
        print(
            f"[M23-24] held={args.held_seq} epoch={epoch}/{args.epochs} loss={average_loss:.5f} "
            f"rank1={values['rank1']:.3f} map={values['map']:.3f} best={best_map:.3f} batches={batches}",
            flush=True,
        )
        if wait >= args.early_stop_patience:
            break
    summary = {**protocol, "best_inner_val_map": best_map, "checkpoint": str(output_dir / "model_best.pth")}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"held_sequence": args.held_seq, "best_inner_val_map": best_map, "checkpoint": summary["checkpoint"]}), flush=True)


if __name__ == "__main__":
    main()
