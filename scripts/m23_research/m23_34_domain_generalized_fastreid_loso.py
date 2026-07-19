#!/usr/bin/env python3
from __future__ import annotations

"""M23-34 domain-generalized FastReID under strict sequence LOSO.

The held MOT20 sequence is never read while constructing training/validation
sets or selecting a checkpoint.  Compared with M23-24, this trainer removes two
known cross-sequence shortcuts: natural sequence-frequency imbalance and pooled
validation retrieval that rewards camera/style separation.  It uses a
temperature-balanced per-sequence identity sampler, sequence-adversarial
gradient reversal, stronger style randomization, and checkpoint selection by a
robust aggregate of per-training-sequence retrieval mAP.
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
from torch import nn
from torch.autograd import Function
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
    def __init__(
        self,
        items: Sequence[Item],
        transform,
        box_expand: float,
        sequence_to_domain: Dict[str, int],
    ) -> None:
        self.items = list(items)
        self.transform = transform
        self.box_expand = float(box_expand)
        self.sequence_to_domain = dict(sequence_to_domain)

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
        return crop.mul(255.0), int(item.pid), int(self.sequence_to_domain[item.seq])


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


class SequenceBalancedIdentityBatchSampler(Sampler[List[int]]):
    """P x K sampler with sequence quotas proportional to identity_count**temperature."""

    def __init__(
        self,
        items: Sequence[Item],
        batch_size: int,
        num_instances: int,
        seed: int,
        sequence_temperature: float,
    ) -> None:
        if batch_size % num_instances != 0:
            raise ValueError("batch_size must be divisible by num_instances")
        self.items = list(items)
        self.batch_size = int(batch_size)
        self.num_instances = int(num_instances)
        self.pids_per_batch = self.batch_size // self.num_instances
        self.seed = int(seed)
        self.epoch = 0
        self.index_by_seq_pid: Dict[str, Dict[int, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, item in enumerate(self.items):
            self.index_by_seq_pid[item.seq][int(item.pid)].append(index)
        self.sequences = sorted(self.index_by_seq_pid)
        if self.pids_per_batch < len(self.sequences):
            raise ValueError("batch has fewer identity slots than training sequences")
        counts = np.asarray(
            [len(self.index_by_seq_pid[seq]) for seq in self.sequences], dtype=float
        )
        weights = np.power(np.maximum(counts, 1.0), float(sequence_temperature))
        available = self.pids_per_batch - len(self.sequences)
        raw_extra = available * weights / weights.sum()
        extra = np.floor(raw_extra).astype(int)
        for index in np.argsort(-(raw_extra - extra))[: available - int(extra.sum())]:
            extra[int(index)] += 1
        self.quotas = {
            seq: int(1 + extra[index]) for index, seq in enumerate(self.sequences)
        }
        self.length = max(1, math.ceil(len(self.items) / self.batch_size))

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        pid_lists = {
            seq: sorted(self.index_by_seq_pid[seq]) for seq in self.sequences
        }
        for _ in range(self.length):
            batch: List[int] = []
            for seq in self.sequences:
                pids = pid_lists[seq]
                quota = self.quotas[seq]
                selected = (
                    rng.sample(pids, quota)
                    if len(pids) >= quota
                    else rng.choices(pids, k=quota)
                )
                for pid in selected:
                    rows = self.index_by_seq_pid[seq][int(pid)]
                    chosen = (
                        rng.sample(rows, self.num_instances)
                        if len(rows) >= self.num_instances
                        else rng.choices(rows, k=self.num_instances)
                    )
                    batch.extend(chosen)
            rng.shuffle(batch)
            yield batch


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.scale * gradient, None


def gradient_reverse(values: torch.Tensor, scale: float) -> torch.Tensor:
    return GradientReversal.apply(values, float(scale))


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


def retrieval_from_tensors(feats: torch.Tensor, pids: torch.Tensor) -> Dict[str, float]:
    if not len(feats):
        return {"rank1": 0.0, "map": 0.0, "queries": 0.0}
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
            precision = positives.float().cumsum(0) / torch.arange(
                1, len(positives) + 1
            )
            ap_sum += float((precision * positives.float()).sum() / count)
    denominator = max(valid_queries, 1)
    return {
        "rank1": 100.0 * correct / denominator,
        "map": 100.0 * ap_sum / denominator,
        "queries": float(valid_queries),
    }


@torch.no_grad()
def retrieval_metrics(
    model,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    domain_to_sequence: Dict[int, str],
) -> Dict[str, object]:
    model.eval()
    features: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    domains: List[torch.Tensor] = []
    for images, pids, domain_ids in loader:
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            embedding = model(images)
        features.append(F.normalize(embedding.float(), dim=1).cpu())
        labels.append(pids.long().cpu())
        domains.append(domain_ids.long().cpu())
    if not features:
        return {
            "rank1": 0.0,
            "map": 0.0,
            "queries": 0.0,
            "per_sequence": {},
            "mean_sequence_map": 0.0,
            "worst_sequence_map": 0.0,
            "std_sequence_map": 0.0,
            "robust_sequence_map": 0.0,
        }
    feats = torch.cat(features)
    pids = torch.cat(labels)
    domain_ids = torch.cat(domains)
    pooled = retrieval_from_tensors(feats, pids)
    per_sequence: Dict[str, Dict[str, float]] = {}
    for domain, seq in sorted(domain_to_sequence.items()):
        mask = domain_ids == int(domain)
        per_sequence[seq] = retrieval_from_tensors(feats[mask], pids[mask])
    maps = np.asarray([row["map"] for row in per_sequence.values()], dtype=float)
    mean_map = float(maps.mean()) if len(maps) else 0.0
    std_map = float(maps.std()) if len(maps) else 0.0
    return {
        **pooled,
        "per_sequence": per_sequence,
        "mean_sequence_map": mean_map,
        "worst_sequence_map": float(maps.min()) if len(maps) else 0.0,
        "std_sequence_map": std_map,
        "robust_sequence_map": mean_map - 0.5 * std_map,
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
    parser.add_argument("--sequence-temperature", type=float, default=0.5)
    parser.add_argument("--domain-adversarial-weight", type=float, default=0.15)
    parser.add_argument("--domain-head-dim", type=int, default=256)
    parser.add_argument("--domain-warmup-epochs", type=int, default=1)
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
        "status": "running",
        "experiment": "M23-34 strict sequence-LOSO domain-generalized FastReID",
        "held_sequence": args.held_seq,
        "training_sequences": train_sequences,
        "held_sequence_read_during_training": False,
        "checkpoint_selection": (
            "maximize mean per-training-sequence retrieval mAP minus 0.5 standard "
            "deviation; pooled retrieval is diagnostic only"
        ),
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
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.30, hue=0.06)],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.08),
        transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 1.5))], p=0.15),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.18), value="random"),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((384, 128), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    sequence_to_domain = {seq: index for index, seq in enumerate(train_sequences)}
    domain_to_sequence = {index: seq for seq, index in sequence_to_domain.items()}
    train_dataset = CropDataset(
        train_items, train_transform, args.box_expand, sequence_to_domain
    )
    val_dataset = CropDataset(
        val_items, val_transform, args.box_expand, sequence_to_domain
    )

    sampler = SequenceBalancedIdentityBatchSampler(
        train_items,
        args.batch_size,
        args.num_instances,
        args.seed,
        args.sequence_temperature,
    )
    protocol["sequence_to_domain"] = sequence_to_domain
    protocol["sampler_sequence_identity_quotas"] = sampler.quotas
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )
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
    domain_head = nn.Sequential(
        nn.Linear(int(cfg.MODEL.BACKBONE.FEAT_DIM), args.domain_head_dim),
        nn.GELU(),
        nn.Dropout(0.20),
        nn.Linear(args.domain_head_dim, len(train_sequences)),
    ).to(device)

    backbone_parameters = list(model.backbone.parameters())
    head_parameters = list(model.heads.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.base_lr},
            {"params": head_parameters, "lr": args.head_lr},
            {"params": domain_head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(
            [
                "epoch",
                "train_loss",
                "identity_loss",
                "domain_loss",
                "val_rank1",
                "val_map",
                "val_mean_sequence_map",
                "val_worst_sequence_map",
                "val_std_sequence_map",
                "val_robust_sequence_map",
                "val_per_sequence_json",
                "queries",
                "backbone_lr",
                "head_lr",
                "domain_lr",
                "batches",
            ]
        )

    best_map = -1.0
    best_worst_map = -1.0
    best_values: Dict[str, object] = {}
    wait = 0
    for epoch in range(1, args.epochs + 1):
        cosine = 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / max(args.epochs, 1)))
        optimizer.param_groups[0]["lr"] = 0.0 if epoch <= args.freeze_backbone_epochs else args.base_lr * cosine
        optimizer.param_groups[1]["lr"] = args.head_lr * cosine
        optimizer.param_groups[2]["lr"] = args.head_lr * cosine
        model.train()
        domain_head.train()
        total_loss = 0.0
        total_identity_loss = 0.0
        total_domain_loss = 0.0
        batches = 0
        with EventStorage(start_iter=(epoch - 1) * max(len(train_loader), 1)) as storage:
            for images, pids, domain_ids in train_loader:
                images = images.to(device, non_blocking=True)
                pids = pids.to(device, non_blocking=True)
                domain_ids = domain_ids.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    normalized = model.preprocess_image({"images": images})
                    backbone_features = model.backbone(normalized)
                    outputs = model.heads(backbone_features, pids)
                    losses = model.losses(outputs, pids)
                    identity_loss = sum(losses.values())
                    progress = (
                        (epoch - 1) * max(len(train_loader), 1) + batches
                    ) / max(args.epochs * max(len(train_loader), 1), 1)
                    reversal_scale = (
                        0.0
                        if epoch <= args.domain_warmup_epochs
                        else 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
                    )
                    domain_logits = domain_head(
                        gradient_reverse(
                            F.normalize(outputs["features"].float(), dim=1),
                            reversal_scale,
                        )
                    )
                    domain_loss = F.cross_entropy(domain_logits, domain_ids)
                    loss = identity_loss + args.domain_adversarial_weight * domain_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach())
                total_identity_loss += float(identity_loss.detach())
                total_domain_loss += float(domain_loss.detach())
                batches += 1
                storage.step()
                if batches == 1 or batches % 100 == 0:
                    print(
                        f"[M23-34 train] held={args.held_seq} epoch={epoch}/{args.epochs} "
                        f"batch={batches}/{len(train_loader)} running_loss={total_loss / batches:.5f}",
                        flush=True,
                    )
                if args.max_train_batches > 0 and batches >= args.max_train_batches:
                    break
        values = retrieval_metrics(
            model, val_loader, device, args.amp, domain_to_sequence
        )
        average_loss = total_loss / max(batches, 1)
        average_identity_loss = total_identity_loss / max(batches, 1)
        average_domain_loss = total_domain_loss / max(batches, 1)
        robust_map = float(values["robust_sequence_map"])
        worst_map = float(values["worst_sequence_map"])
        improved = robust_map > best_map + 0.02 or (
            abs(robust_map - best_map) <= 0.02 and worst_map > best_worst_map + 0.02
        )
        if improved:
            best_map = robust_map
            best_worst_map = worst_map
            best_values = values
            wait = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "domain_head": domain_head.state_dict(),
                    "epoch": epoch,
                    "held_sequence": args.held_seq,
                    "val_robust_sequence_map": best_map,
                    "val_worst_sequence_map": best_worst_map,
                    "val_per_sequence": values["per_sequence"],
                },
                output_dir / "model_best.pth",
            )
        else:
            wait += 1
        with metrics_path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow([
                epoch,
                f"{average_loss:.6f}",
                f"{average_identity_loss:.6f}",
                f"{average_domain_loss:.6f}",
                f"{values['rank1']:.6f}",
                f"{values['map']:.6f}",
                f"{values['mean_sequence_map']:.6f}",
                f"{values['worst_sequence_map']:.6f}",
                f"{values['std_sequence_map']:.6f}",
                f"{values['robust_sequence_map']:.6f}",
                json.dumps(values["per_sequence"], sort_keys=True),
                int(values["queries"]),
                f"{optimizer.param_groups[0]['lr']:.8e}",
                f"{optimizer.param_groups[1]['lr']:.8e}",
                f"{optimizer.param_groups[2]['lr']:.8e}",
                batches,
            ])
        print(
            f"[M23-34] held={args.held_seq} epoch={epoch}/{args.epochs} loss={average_loss:.5f} "
            f"rank1={values['rank1']:.3f} pooled_map={values['map']:.3f} "
            f"robust_map={robust_map:.3f} worst_map={worst_map:.3f} "
            f"best={best_map:.3f} batches={batches}",
            flush=True,
        )
        if wait >= args.early_stop_patience:
            break
    summary = {
        **protocol,
        "status": "completed",
        "best_inner_val_robust_sequence_map": best_map,
        "best_inner_val_worst_sequence_map": best_worst_map,
        "best_inner_val_metrics": best_values,
        "checkpoint": str(output_dir / "model_best.pth"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "held_sequence": args.held_seq,
                "best_inner_val_robust_sequence_map": best_map,
                "checkpoint": summary["checkpoint"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
