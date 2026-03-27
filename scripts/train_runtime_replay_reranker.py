#!/usr/bin/env python3
from __future__ import annotations

import argparse
import faulthandler
import json
import random
import sys
import traceback
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.runtime_replay_assoc import RuntimeReplayAssociationAdapter, save_runtime_replay_checkpoint  # noqa: E402


@dataclass(frozen=True)
class GroupRef:
    shard_path: str
    group_index: int
    seq: str
    group_id: str
    is_ambiguous: bool
    is_background: bool
    is_recoverable: bool
    rank_top1_correct: bool
    positive_in_topk: bool


class PreparedShard:
    def __init__(self, path: Path) -> None:
        self.path = str(path)
        z = np.load(path, allow_pickle=True)
        self.group_ids = [str(x) for x in z["group_ids"].tolist()]
        self.seq = [str(x) for x in z["seq"].tolist()]
        self.det_frame = np.asarray(z["det_frame"], dtype=np.int32) if "det_frame" in z.files else np.zeros((len(self.group_ids),), dtype=np.int32)
        self.group_offsets = np.asarray(z["group_offsets"], dtype=np.int64)
        self.group_size = np.asarray(z["group_size"], dtype=np.int32)
        self.candidate_count_total = np.asarray(z["candidate_count_total"], dtype=np.int32)
        self.group_is_ambiguous = np.asarray(z["group_is_ambiguous"], dtype=np.uint8)
        self.group_is_background = np.asarray(z["group_is_background"], dtype=np.uint8)
        self.group_is_recoverable = np.asarray(z["group_is_recoverable"], dtype=np.uint8)
        self.rank_top1_correct = np.asarray(z["rank_top1_correct"], dtype=np.uint8)
        self.positive_in_topk = np.asarray(z["positive_in_topk"], dtype=np.uint8)
        self.det_feat = np.asarray(z["det_feat"], dtype=np.float32)
        self.det_box = np.asarray(z["det_box"], dtype=np.float32)
        self.det_score = np.asarray(z["det_score"], dtype=np.float32)
        self.track_rank = np.asarray(z["track_rank"], dtype=np.int32)
        self.track_id = np.asarray(z["track_id"], dtype=np.int32)
        self.label = np.asarray(z["label"], dtype=np.int64)
        self.valid_train_row = np.asarray(z["valid_train_row"], dtype=np.uint8)
        self.base_score = np.asarray(z["base_score"], dtype=np.float32)
        self.refined_score = np.asarray(z["refined_score"], dtype=np.float32)
        self.motion_score = np.asarray(z["motion_score"], dtype=np.float32)
        self.teacher_score = np.asarray(z["teacher_score"], dtype=np.float32)
        self.scalar_feat = np.asarray(z["scalar_feat"], dtype=np.float32)
        self.hist_feat = np.asarray(z["hist_feat"], dtype=np.float32)
        self.hist_mask = np.asarray(z["hist_mask"], dtype=np.uint8)
        self.hist_time = np.asarray(z["hist_time"], dtype=np.int32)
        self.track_box = np.asarray(z["track_box"], dtype=np.float32)
        rank_score_col = z["rank_score_col"].tolist()
        if isinstance(rank_score_col, list):
            self.rank_score_col = str(rank_score_col[0])
        else:
            self.rank_score_col = str(rank_score_col)
        feature_names = z["feature_names"].tolist()
        self.feature_names = [str(x) for x in feature_names]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train the complete runtime-replay learned reranker.")
    ap.add_argument("--input-dir", required=True, help="Directory containing runtime_replay_shard_*.npz files.")
    ap.add_argument("--out-ckpt", required=True)
    ap.add_argument("--metrics-path", default="")
    ap.add_argument("--init-ckpt", default="", help="Optional checkpoint to initialize model weights from.")
    ap.add_argument("--save-every-epoch-dir", default="", help="Optional directory to save per-epoch checkpoints.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--batch-groups", type=int, default=24)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--valid-only", action="store_true")
    ap.add_argument("--train-seqs", default="")
    ap.add_argument("--val-seqs", default="")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--max-train-groups", type=int, default=0)
    ap.add_argument("--max-val-groups", type=int, default=0)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--scalar-hidden", type=int, default=96)
    ap.add_argument("--scalar-out", type=int, default=96)
    ap.add_argument("--temporal-hidden", type=int, default=96)
    ap.add_argument("--temporal-out", type=int, default=96)
    ap.add_argument("--token-dim", type=int, default=160)
    ap.add_argument("--duel-dim", type=int, default=160)
    ap.add_argument("--group-hidden", type=int, default=128)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--delta-scale", type=float, default=1.0)
    ap.add_argument("--margin-threshold", type=float, default=0.10)
    ap.add_argument("--margin-temperature", type=float, default=0.03)
    ap.add_argument("--gate-cap", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--duel-margin", type=float, default=0.20)
    ap.add_argument("--loss-duel-weight", type=float, default=0.5)
    ap.add_argument("--loss-safe-weight", type=float, default=0.2)
    ap.add_argument("--loss-distill-weight", type=float, default=0.25)
    ap.add_argument("--loss-gate-weight", type=float, default=0.05)
    ap.add_argument("--gate-positive-target", type=float, default=0.20)
    ap.add_argument("--gate-positive-scale", type=float, default=1.0)
    ap.add_argument("--distill-temperature", type=float, default=1.0)
    ap.add_argument("--ambiguous-weight", type=float, default=2.0)
    ap.add_argument("--hard-positive-weight", type=float, default=4.0)
    ap.add_argument("--easy-weight", type=float, default=0.5)
    ap.add_argument("--background-weight", type=float, default=0.75)
    ap.add_argument("--select-amb-weight", type=float, default=1.5)
    ap.add_argument("--select-hard-weight", type=float, default=2.5)
    ap.add_argument("--select-bg-weight", type=float, default=0.10)
    ap.add_argument("--select-easy-weight", type=float, default=0.05)
    ap.add_argument("--train-groups-per-epoch", type=int, default=0)
    ap.add_argument("--val-groups-per-epoch", type=int, default=0)
    ap.add_argument("--fixed-val-sample", action="store_true", help="Reuse the same sampled validation groups every epoch.")
    ap.add_argument("--sample-ambiguous-weight", type=float, default=3.0)
    ap.add_argument("--sample-hard-positive-weight", type=float, default=8.0)
    ap.add_argument("--sample-easy-weight", type=float, default=0.35)
    ap.add_argument("--sample-background-weight", type=float, default=0.5)
    ap.add_argument("--sample-groups-per-shard", type=int, default=96)
    positive_group = ap.add_mutually_exclusive_group()
    positive_group.add_argument(
        "--force-positive-into-topk",
        dest="force_positive_into_topk",
        action="store_true",
        default=True,
        help="Legacy behavior: inject a valid positive back into the selected top-k if it exists outside top-k.",
    )
    positive_group.add_argument(
        "--honest-topk",
        dest="force_positive_into_topk",
        action="store_false",
        help="Use the actual runtime top-k only; do not inject the positive candidate back into the selected set.",
    )
    ap.add_argument(
        "--exclude-unrecoverable-positive-loss",
        action="store_true",
        help="Exclude groups whose valid positive exists outside the selected top-k from training losses and report them separately.",
    )
    return ap.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_seq_set(raw: str) -> set[str]:
    return {x.strip() for x in str(raw or "").split(",") if x.strip()}


def _discover_shards(root: Path) -> list[Path]:
    return sorted(root.glob("runtime_replay_shard_*.npz"))


def _read_npz_npy_shape(path: Path, member: str) -> tuple[int, ...]:
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open(member, "r") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
            else:
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
    return tuple(int(x) for x in shape)


def _infer_model_dims(shard_path: Path) -> tuple[int, int]:
    scalar_shape = _read_npz_npy_shape(shard_path, "scalar_feat.npy")
    det_shape = _read_npz_npy_shape(shard_path, "det_feat.npy")
    if len(scalar_shape) != 2 or len(det_shape) != 2:
        raise RuntimeError(
            f"Unexpected shard tensor shapes in {shard_path}: scalar_feat={scalar_shape}, det_feat={det_shape}"
        )
    return int(scalar_shape[1]), int(det_shape[1])


def _collect_refs(shard_paths: list[Path]) -> list[GroupRef]:
    refs: list[GroupRef] = []
    for path in shard_paths:
        z = np.load(path, allow_pickle=True)
        gids = [str(x) for x in z["group_ids"].tolist()]
        seqs = [str(x) for x in z["seq"].tolist()]
        ambiguous = np.asarray(z["group_is_ambiguous"], dtype=np.uint8)
        background = np.asarray(z["group_is_background"], dtype=np.uint8)
        recoverable = np.asarray(z["group_is_recoverable"], dtype=np.uint8)
        top1_correct = np.asarray(z["rank_top1_correct"], dtype=np.uint8)
        positive_in_topk = np.asarray(z["positive_in_topk"], dtype=np.uint8)
        for idx, (gid, seq) in enumerate(zip(gids, seqs)):
            refs.append(
                GroupRef(
                    shard_path=str(path),
                    group_index=idx,
                    seq=seq,
                    group_id=gid,
                    is_ambiguous=bool(int(ambiguous[idx])),
                    is_background=bool(int(background[idx])),
                    is_recoverable=bool(int(recoverable[idx])),
                    rank_top1_correct=bool(int(top1_correct[idx])),
                    positive_in_topk=bool(int(positive_in_topk[idx])),
                )
            )
    return refs


def _split_refs(refs: list[GroupRef], train_seqs: set[str], val_seqs: set[str], val_ratio: float, seed: int) -> tuple[list[GroupRef], list[GroupRef]]:
    if train_seqs or val_seqs:
        train_refs: list[GroupRef] = []
        val_refs: list[GroupRef] = []
        for ref in refs:
            if ref.seq in val_seqs:
                val_refs.append(ref)
            elif train_seqs:
                if ref.seq in train_seqs:
                    train_refs.append(ref)
            else:
                train_refs.append(ref)
        return train_refs, val_refs

    rng = np.random.default_rng(seed)
    order = np.arange(len(refs))
    rng.shuffle(order)
    split = int(round((1.0 - val_ratio) * float(len(refs))))
    split = max(1, min(split, len(refs) - 1))
    train_refs = [refs[idx] for idx in order[:split]]
    val_refs = [refs[idx] for idx in order[split:]]
    return train_refs, val_refs


def _limit_refs(refs: list[GroupRef], max_groups: int, seed: int) -> list[GroupRef]:
    if int(max_groups) <= 0 or len(refs) <= int(max_groups):
        return refs
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(refs))[: int(max_groups)]
    keep = sorted(int(i) for i in order.tolist())
    return [refs[idx] for idx in keep]


def _group_ref_map(refs: list[GroupRef]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for ref in refs:
        out[ref.shard_path].append(int(ref.group_index))
    return out


def _unique_seq_list(refs: list[GroupRef]) -> list[str]:
    return sorted({str(ref.seq) for ref in refs if str(ref.seq)})


def _is_hard_positive_ref(ref: GroupRef) -> bool:
    return (not ref.is_background) and (
        ref.is_recoverable or (ref.is_ambiguous and (not ref.rank_top1_correct) and ref.positive_in_topk)
    )


def _sample_ref_weight(ref: GroupRef, args: argparse.Namespace) -> float:
    if ref.is_background:
        return max(float(args.sample_background_weight), 1e-6)
    if _is_hard_positive_ref(ref):
        return max(float(args.sample_hard_positive_weight), 1e-6)
    if ref.is_ambiguous:
        return max(float(args.sample_ambiguous_weight), 1e-6)
    return max(float(args.sample_easy_weight), 1e-6)


def _sample_epoch_refs(
    refs: list[GroupRef],
    groups_per_epoch: int,
    args: argparse.Namespace,
    epoch: int,
    seed_offset: int,
) -> list[GroupRef]:
    target = int(groups_per_epoch)
    if target <= 0 or len(refs) <= target:
        return list(refs)
    weights = np.asarray([_sample_ref_weight(ref, args) for ref in refs], dtype=np.float64)
    rng = np.random.default_rng(int(args.seed) + int(seed_offset) + int(epoch) * 1009)
    shard_to_ref_indices: dict[str, list[int]] = defaultdict(list)
    for ref_idx, ref in enumerate(refs):
        shard_to_ref_indices[ref.shard_path].append(int(ref_idx))

    shard_paths = list(shard_to_ref_indices.keys())
    sample_groups_per_shard = max(int(args.sample_groups_per_shard), 1)
    target_shards = min(len(shard_paths), max(1, int(np.ceil(float(target) / float(sample_groups_per_shard)))))

    shard_weights = np.asarray(
        [weights[np.asarray(shard_to_ref_indices[shard_path], dtype=np.int64)].sum() for shard_path in shard_paths],
        dtype=np.float64,
    )
    shard_weights = shard_weights / np.clip(shard_weights.sum(), 1e-12, None)
    chosen_shards = rng.choice(len(shard_paths), size=target_shards, replace=False, p=shard_weights)

    sampled_indices: list[int] = []
    remaining = int(target)
    remaining_shards = int(target_shards)
    for shard_pos, shard_idx in enumerate(chosen_shards.tolist(), start=1):
        shard_path = shard_paths[int(shard_idx)]
        ref_indices = np.asarray(shard_to_ref_indices[shard_path], dtype=np.int64)
        local_weights = weights[ref_indices]
        local_weights = local_weights / np.clip(local_weights.sum(), 1e-12, None)
        if shard_pos == target_shards:
            take = int(remaining)
        else:
            take = min(
                len(ref_indices),
                max(sample_groups_per_shard, int(np.ceil(float(remaining) / float(max(remaining_shards, 1))))),
            )
        replace = int(take) > int(len(ref_indices))
        choice = rng.choice(ref_indices, size=int(take), replace=replace, p=local_weights)
        sampled_indices.extend(int(idx) for idx in choice.tolist())
        remaining -= int(take)
        remaining_shards -= 1
        if remaining <= 0:
            break

    if len(sampled_indices) < target:
        global_weights = weights / np.clip(weights.sum(), 1e-12, None)
        extra = rng.choice(len(refs), size=int(target - len(sampled_indices)), replace=False, p=global_weights)
        sampled_indices.extend(int(idx) for idx in extra.tolist())

    if len(sampled_indices) > target:
        sampled_indices = sampled_indices[:target]
    return [refs[idx] for idx in sampled_indices]


def _select_candidates(
    shard: PreparedShard,
    group_index: int,
    topk: int,
    valid_only: bool,
    force_positive_into_topk: bool,
) -> np.ndarray:
    start = int(shard.group_offsets[group_index])
    end = int(shard.group_offsets[group_index + 1])
    idx = np.arange(start, end, dtype=np.int64)
    if idx.size == 0:
        return idx
    if valid_only:
        idx = idx[shard.valid_train_row[idx] > 0]
        if idx.size == 0:
            return idx
    if shard.rank_score_col == "base_score":
        rank_scores = shard.base_score[idx]
    else:
        rank_scores = shard.refined_score[idx]
    order = np.argsort(-rank_scores, kind="mergesort")
    idx = idx[order]
    if topk > 0 and idx.size > topk:
        keep = idx[:topk].tolist()
        if force_positive_into_topk:
            label_mask = shard.label[idx] > 0
            pos_indices = idx[label_mask]
            if pos_indices.size > 0 and int(pos_indices[0]) not in keep:
                keep[-1] = int(pos_indices[0])
        if shard.rank_score_col == "base_score":
            keep = sorted(set(keep), key=lambda x: (-float(shard.base_score[x]), int(shard.track_rank[x])))
        else:
            keep = sorted(set(keep), key=lambda x: (-float(shard.refined_score[x]), int(shard.track_rank[x])))
        idx = np.asarray(keep, dtype=np.int64)
    return idx


def _collate_batch(
    shard: PreparedShard,
    group_indices: list[int],
    topk: int,
    device: torch.device,
    valid_only: bool,
    force_positive_into_topk: bool,
) -> dict[str, torch.Tensor]:
    selected_all = [
        _select_candidates(
            shard,
            group_idx,
            topk=topk,
            valid_only=valid_only,
            force_positive_into_topk=force_positive_into_topk,
        )
        for group_idx in group_indices
    ]
    keep_pairs = [(group_idx, cand_idx) for group_idx, cand_idx in zip(group_indices, selected_all) if int(cand_idx.size) > 0]
    if not keep_pairs:
        return {
            "anchor_scores": torch.zeros((0, 0), device=device, dtype=torch.float32),
            "scalar_feat": torch.zeros((0, 0, shard.scalar_feat.shape[1]), device=device, dtype=torch.float32),
            "det_feat": torch.zeros((0, shard.det_feat.shape[1]), device=device, dtype=torch.float32),
            "hist_feat": torch.zeros((0, 0, shard.hist_feat.shape[1], shard.det_feat.shape[1]), device=device, dtype=torch.float32),
            "hist_mask": torch.zeros((0, 0, shard.hist_feat.shape[1]), device=device, dtype=torch.bool),
            "hist_time": torch.zeros((0, 0, shard.hist_feat.shape[1]), device=device, dtype=torch.int64),
            "det_times": torch.zeros((0,), device=device, dtype=torch.int64),
            "candidate_mask": torch.zeros((0, 0), device=device, dtype=torch.bool),
            "supervised_mask": torch.zeros((0, 0), device=device, dtype=torch.bool),
            "teacher_score": torch.zeros((0, 0), device=device, dtype=torch.float32),
            "det_scores": torch.zeros((0,), device=device, dtype=torch.float32),
            "target_index": torch.zeros((0,), device=device, dtype=torch.long),
            "group_is_ambiguous": torch.zeros((0,), device=device, dtype=torch.bool),
            "group_is_background": torch.zeros((0,), device=device, dtype=torch.bool),
            "group_is_recoverable": torch.zeros((0,), device=device, dtype=torch.bool),
            "rank_top1_correct": torch.zeros((0,), device=device, dtype=torch.bool),
            "positive_in_topk": torch.zeros((0,), device=device, dtype=torch.bool),
            "group_has_valid_positive": torch.zeros((0,), device=device, dtype=torch.bool),
        }
    group_indices = [g for g, _ in keep_pairs]
    selected = [c for _, c in keep_pairs]
    max_cand = max(int(x.size) for x in selected)
    hist_steps = int(shard.hist_feat.shape[1])
    feat_dim = int(shard.det_feat.shape[1])
    scalar_dim = int(shard.scalar_feat.shape[1])
    batch = len(group_indices)

    anchor_scores = torch.full((batch, max_cand), 1e-4, device=device, dtype=torch.float32)
    scalar_feat = torch.zeros((batch, max_cand, scalar_dim), device=device, dtype=torch.float32)
    det_feat = torch.zeros((batch, feat_dim), device=device, dtype=torch.float32)
    hist_feat = torch.zeros((batch, max_cand, hist_steps, feat_dim), device=device, dtype=torch.float32)
    hist_mask = torch.ones((batch, max_cand, hist_steps), device=device, dtype=torch.bool)
    hist_time = torch.zeros((batch, max_cand, hist_steps), device=device, dtype=torch.int64)
    det_times = torch.zeros((batch,), device=device, dtype=torch.int64)
    candidate_mask = torch.zeros((batch, max_cand), device=device, dtype=torch.bool)
    supervised_mask = torch.zeros((batch, max_cand), device=device, dtype=torch.bool)
    teacher_score = torch.zeros((batch, max_cand), device=device, dtype=torch.float32)
    det_scores = torch.zeros((batch,), device=device, dtype=torch.float32)
    target_index = torch.full((batch,), -1, device=device, dtype=torch.long)
    group_is_ambiguous = torch.zeros((batch,), device=device, dtype=torch.bool)
    group_is_background = torch.zeros((batch,), device=device, dtype=torch.bool)
    group_is_recoverable = torch.zeros((batch,), device=device, dtype=torch.bool)
    rank_top1_correct = torch.zeros((batch,), device=device, dtype=torch.bool)
    positive_in_topk = torch.zeros((batch,), device=device, dtype=torch.bool)
    group_has_valid_positive = torch.zeros((batch,), device=device, dtype=torch.bool)

    for batch_idx, (group_idx, cand_idx) in enumerate(zip(group_indices, selected)):
        cand_count = int(cand_idx.size)
        if cand_count <= 0:
            continue
        group_start = int(shard.group_offsets[group_idx])
        group_end = int(shard.group_offsets[group_idx + 1])
        full_idx = np.arange(group_start, group_end, dtype=np.int64)
        if valid_only:
            full_idx = full_idx[shard.valid_train_row[full_idx] > 0]
        group_has_valid_positive[batch_idx] = bool(full_idx.size > 0 and np.any(shard.label[full_idx] > 0))
        det_feat[batch_idx] = torch.from_numpy(shard.det_feat[group_idx]).to(device=device, dtype=torch.float32)
        det_scores[batch_idx] = float(shard.det_score[group_idx])
        det_times[batch_idx] = int(shard.det_frame[group_idx])
        group_is_ambiguous[batch_idx] = bool(shard.group_is_ambiguous[group_idx] > 0)
        group_is_background[batch_idx] = bool(shard.group_is_background[group_idx] > 0)
        group_is_recoverable[batch_idx] = bool(shard.group_is_recoverable[group_idx] > 0)
        rank_top1_correct[batch_idx] = bool(shard.rank_top1_correct[group_idx] > 0)
        positive_in_topk[batch_idx] = bool(shard.positive_in_topk[group_idx] > 0)

        candidate_mask[batch_idx, :cand_count] = True
        hist_feat[batch_idx, :cand_count] = torch.from_numpy(shard.hist_feat[cand_idx]).to(device=device, dtype=torch.float32)
        hist_mask[batch_idx, :cand_count] = torch.from_numpy(shard.hist_mask[cand_idx] > 0).to(device=device, dtype=torch.bool)
        hist_time[batch_idx, :cand_count] = torch.from_numpy(shard.hist_time[cand_idx]).to(device=device, dtype=torch.long)
        scalar_feat[batch_idx, :cand_count] = torch.from_numpy(shard.scalar_feat[cand_idx]).to(device=device, dtype=torch.float32)
        teacher_score[batch_idx, :cand_count] = torch.from_numpy(shard.teacher_score[cand_idx]).to(device=device, dtype=torch.float32)
        if shard.rank_score_col == "base_score":
            anchor_np = shard.base_score[cand_idx]
        else:
            anchor_np = shard.refined_score[cand_idx]
        anchor_scores[batch_idx, :cand_count] = torch.from_numpy(anchor_np).to(device=device, dtype=torch.float32)

        valid_np = np.ones((cand_count,), dtype=np.bool_)
        if valid_only:
            valid_np = shard.valid_train_row[cand_idx] > 0
        supervised_mask[batch_idx, :cand_count] = torch.from_numpy(valid_np).to(device=device, dtype=torch.bool)
        label_positions = np.where((shard.label[cand_idx] > 0) & valid_np)[0]
        if label_positions.size > 0:
            target_index[batch_idx] = int(label_positions[0])

    return {
        "anchor_scores": anchor_scores.clamp(min=1e-4, max=1.0 - 1e-4),
        "scalar_feat": scalar_feat,
        "det_feat": det_feat,
        "hist_feat": hist_feat,
        "hist_mask": hist_mask,
        "hist_time": hist_time,
        "det_times": det_times,
        "candidate_mask": candidate_mask,
        "supervised_mask": supervised_mask,
        "teacher_score": teacher_score.clamp(min=0.0, max=1.0),
        "det_scores": det_scores.clamp(min=0.0, max=1.0),
        "target_index": target_index,
        "group_is_ambiguous": group_is_ambiguous,
        "group_is_background": group_is_background,
        "group_is_recoverable": group_is_recoverable,
        "rank_top1_correct": rank_top1_correct,
        "positive_in_topk": positive_in_topk,
        "group_has_valid_positive": group_has_valid_positive,
    }


def _new_stats() -> dict[str, float]:
    return {
        "groups": 0.0,
        "pos_groups": 0.0,
        "amb_groups": 0.0,
        "easy_groups": 0.0,
        "bg_groups": 0.0,
        "hard_groups": 0.0,
        "recoverable_groups": 0.0,
        "recovered_groups": 0.0,
        "valid_positive_groups": 0.0,
        "selected_positive_groups": 0.0,
        "unrecoverable_positive_groups": 0.0,
        "loss_sum": 0.0,
        "list_loss_sum": 0.0,
        "duel_loss_sum": 0.0,
        "safe_loss_sum": 0.0,
        "distill_loss_sum": 0.0,
        "gate_loss_sum": 0.0,
        "base_top1_sum": 0.0,
        "final_top1_sum": 0.0,
        "amb_base_top1_sum": 0.0,
        "amb_final_top1_sum": 0.0,
        "easy_base_top1_sum": 0.0,
        "easy_final_top1_sum": 0.0,
        "hard_base_top1_sum": 0.0,
        "hard_final_top1_sum": 0.0,
        "activation_sum": 0.0,
        "active_groups_sum": 0.0,
        "margin_sum": 0.0,
        "entropy_sum": 0.0,
        "score_shift_sum": 0.0,
        "easy_shift_sum": 0.0,
        "amb_shift_sum": 0.0,
        "bg_shift_sum": 0.0,
        "bg_max_base_sum": 0.0,
        "bg_max_final_sum": 0.0,
        "bg_suppression_sum": 0.0,
    }


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) > 0.0 else 0.0


def _selection_score(metrics: dict[str, float], args: argparse.Namespace) -> float:
    return (
        float(metrics["top1_gain"])
        + float(args.select_amb_weight) * float(metrics["amb_top1_gain"])
        + float(args.select_hard_weight) * float(metrics["hard_top1_gain"])
        + float(args.select_bg_weight) * float(metrics["bg_suppression"])
        - float(args.select_easy_weight) * float(metrics["easy_shift_mean"])
    )


def _group_weight(
    is_positive: torch.Tensor,
    is_ambiguous: torch.Tensor,
    hard_positive_mask: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    bg = torch.full_like(is_positive, float(args.background_weight), dtype=torch.float32)
    easy = torch.full_like(is_positive, float(args.easy_weight), dtype=torch.float32)
    amb = torch.full_like(is_positive, float(args.ambiguous_weight), dtype=torch.float32)
    hard = torch.full_like(is_positive, float(args.hard_positive_weight), dtype=torch.float32)
    positive_weight = torch.where(hard_positive_mask, hard, torch.where(is_ambiguous, amb, easy))
    return torch.where(~is_positive, bg, positive_weight)


def _compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    candidate_logits = outputs["candidate_logits"]
    anchor_logits = outputs["anchor_logits"]
    null_logit = outputs["null_logit"]
    candidate_scores = outputs["candidate_scores"]
    valid_mask = batch["candidate_mask"]
    supervised_mask = batch["supervised_mask"] & valid_mask
    target_index = batch["target_index"]
    batch_size, cand = candidate_logits.shape
    null_class = cand

    joint_logits = outputs["joint_logits"]
    target_class = torch.where(target_index >= 0, target_index, torch.full_like(target_index, null_class))
    list_loss = F.cross_entropy(joint_logits, target_class, reduction="none")

    valid_positive_mask = batch["group_has_valid_positive"]
    positive_mask = target_index >= 0
    unrecoverable_positive_mask = valid_positive_mask & (~positive_mask)
    ambiguous_mask = positive_mask & batch["group_is_ambiguous"]
    easy_mask = positive_mask & (~batch["group_is_ambiguous"])
    background_mask = batch["group_is_background"]
    recoverable_mask = batch["group_is_recoverable"] & positive_mask
    hard_positive_mask = recoverable_mask | (
        ambiguous_mask & (~batch["rank_top1_correct"]) & batch["positive_in_topk"]
    )

    duel_loss = torch.zeros_like(list_loss)
    if bool(torch.any(ambiguous_mask)):
        rows = torch.nonzero(ambiguous_mask, as_tuple=False).view(-1)
        pos_logit = candidate_logits[rows, target_index[rows]]
        neg_mask = supervised_mask[rows].clone()
        neg_mask.scatter_(1, target_index[rows].view(-1, 1), False)
        hardest_neg = candidate_logits[rows].masked_fill(~neg_mask, -1e9).max(dim=1).values
        duel_loss[rows] = F.relu(float(args.duel_margin) - pos_logit + hardest_neg)

    safe_loss = torch.zeros_like(list_loss)
    if bool(torch.any(easy_mask)):
        rows = torch.nonzero(easy_mask, as_tuple=False).view(-1)
        diff = (candidate_logits[rows] - anchor_logits[rows]) ** 2
        denom = supervised_mask[rows].to(dtype=diff.dtype).sum(dim=1).clamp(min=1.0)
        safe_loss[rows] = (diff * supervised_mask[rows].to(dtype=diff.dtype)).sum(dim=1) / denom

    teacher_score = batch["teacher_score"].clamp(min=1e-4, max=1.0 - 1e-4)
    teacher_logits = torch.logit(teacher_score, eps=1e-4)
    teacher_logits = teacher_logits.masked_fill(~supervised_mask, -1e9)
    student_logits = candidate_logits.masked_fill(~supervised_mask, -1e9)
    teacher_available = supervised_mask.any(dim=1) & (teacher_score.max(dim=1).values > 1e-4)
    distill_loss = torch.zeros_like(list_loss)
    if bool(torch.any(teacher_available)):
        rows = torch.nonzero(teacher_available, as_tuple=False).view(-1)
        t = max(float(args.distill_temperature), 1e-4)
        student_logp = torch.log_softmax(student_logits[rows] / t, dim=1)
        teacher_prob = torch.softmax(teacher_logits[rows] / t, dim=1)
        distill_loss[rows] = F.kl_div(student_logp, teacher_prob, reduction="none").sum(dim=1) * (t * t)

    gate_loss = torch.zeros_like(list_loss)
    if bool(torch.any(background_mask)):
        gate_loss[background_mask] = outputs["group_activation"][background_mask]
    if bool(torch.any(easy_mask)):
        gate_loss[easy_mask] = gate_loss[easy_mask] + outputs["group_activation"][easy_mask]
    if bool(torch.any(unrecoverable_positive_mask)):
        gate_loss[unrecoverable_positive_mask] = gate_loss[unrecoverable_positive_mask] + outputs["group_activation"][unrecoverable_positive_mask]
    if bool(torch.any(hard_positive_mask)):
        target = float(args.gate_positive_target)
        scale = float(args.gate_positive_scale)
        gate_loss[hard_positive_mask] = gate_loss[hard_positive_mask] + scale * F.relu(
            target - outputs["group_activation"][hard_positive_mask]
        )

    sample_weight = _group_weight(
        is_positive=positive_mask,
        is_ambiguous=batch["group_is_ambiguous"],
        hard_positive_mask=hard_positive_mask,
        args=args,
    ).to(candidate_logits.device)
    if bool(args.exclude_unrecoverable_positive_loss):
        sample_weight = sample_weight.masked_fill(unrecoverable_positive_mask, 0.0)
    total = sample_weight * (
        list_loss
        + float(args.loss_duel_weight) * duel_loss
        + float(args.loss_safe_weight) * safe_loss
        + float(args.loss_distill_weight) * distill_loss
        + float(args.loss_gate_weight) * gate_loss
    )
    loss = total.sum() / sample_weight.sum().clamp(min=1e-6)

    return {
        "loss": loss,
        "list_loss": list_loss.mean(),
        "duel_loss": duel_loss.mean(),
        "safe_loss": safe_loss.mean(),
        "distill_loss": distill_loss.mean(),
        "gate_loss": gate_loss.mean(),
        "loss_per_group": total.detach(),
        "positive_mask": positive_mask,
        "valid_positive_mask": valid_positive_mask,
        "ambiguous_mask": ambiguous_mask,
        "easy_mask": easy_mask,
        "background_mask": background_mask,
        "unrecoverable_positive_mask": unrecoverable_positive_mask,
        "hard_positive_mask": hard_positive_mask,
        "joint_logits": joint_logits.detach(),
    }


def _update_stats(
    stats: dict[str, float],
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    losses: dict[str, torch.Tensor],
) -> None:
    candidate_scores = outputs["candidate_scores"]
    anchor_scores = batch["anchor_scores"]
    supervised_mask = batch["supervised_mask"] & batch["candidate_mask"]
    joint_logits = losses["joint_logits"]
    target_index = batch["target_index"]
    null_class = candidate_scores.shape[1]
    target_class = torch.where(target_index >= 0, target_index, torch.full_like(target_index, null_class))
    final_pred = torch.argmax(joint_logits, dim=1)
    base_anchor = anchor_scores.masked_fill(~supervised_mask, -1e9)
    base_pred = torch.argmax(base_anchor, dim=1)
    activation = outputs["group_activation"]

    score_shift = torch.abs(candidate_scores - anchor_scores) * batch["candidate_mask"].to(dtype=candidate_scores.dtype)
    score_shift = score_shift.sum(dim=1) / batch["candidate_mask"].to(dtype=candidate_scores.dtype).sum(dim=1).clamp(min=1.0)
    bg_max_base = base_anchor.masked_fill(base_anchor < -1e8, 0.0).max(dim=1).values
    bg_max_final = (candidate_scores * batch["candidate_mask"].to(dtype=candidate_scores.dtype)).max(dim=1).values

    positive_mask = losses["positive_mask"]
    valid_positive_mask = losses["valid_positive_mask"]
    ambiguous_mask = losses["ambiguous_mask"]
    easy_mask = losses["easy_mask"]
    background_mask = losses["background_mask"]
    unrecoverable_positive_mask = losses["unrecoverable_positive_mask"]
    hard_positive_mask = losses["hard_positive_mask"]
    recoverable_mask = batch["group_is_recoverable"] & positive_mask

    base_correct = (base_pred == target_index) & positive_mask
    final_correct = (final_pred == target_class) & positive_mask

    stats["groups"] += float(candidate_scores.shape[0])
    stats["pos_groups"] += float(positive_mask.sum().item())
    stats["amb_groups"] += float(ambiguous_mask.sum().item())
    stats["easy_groups"] += float(easy_mask.sum().item())
    stats["bg_groups"] += float(background_mask.sum().item())
    stats["hard_groups"] += float(hard_positive_mask.sum().item())
    stats["recoverable_groups"] += float(recoverable_mask.sum().item())
    stats["recovered_groups"] += float((final_correct & recoverable_mask).sum().item())
    stats["valid_positive_groups"] += float(valid_positive_mask.sum().item())
    stats["selected_positive_groups"] += float(positive_mask.sum().item())
    stats["unrecoverable_positive_groups"] += float(unrecoverable_positive_mask.sum().item())
    stats["loss_sum"] += float(losses["loss"].item()) * float(candidate_scores.shape[0])
    stats["list_loss_sum"] += float(losses["list_loss"].item()) * float(candidate_scores.shape[0])
    stats["duel_loss_sum"] += float(losses["duel_loss"].item()) * float(candidate_scores.shape[0])
    stats["safe_loss_sum"] += float(losses["safe_loss"].item()) * float(candidate_scores.shape[0])
    stats["distill_loss_sum"] += float(losses["distill_loss"].item()) * float(candidate_scores.shape[0])
    stats["gate_loss_sum"] += float(losses["gate_loss"].item()) * float(candidate_scores.shape[0])
    stats["base_top1_sum"] += float(base_correct.sum().item())
    stats["final_top1_sum"] += float(final_correct.sum().item())
    stats["amb_base_top1_sum"] += float(base_correct[ambiguous_mask].sum().item())
    stats["amb_final_top1_sum"] += float(final_correct[ambiguous_mask].sum().item())
    stats["easy_base_top1_sum"] += float(base_correct[easy_mask].sum().item())
    stats["easy_final_top1_sum"] += float(final_correct[easy_mask].sum().item())
    stats["hard_base_top1_sum"] += float(base_correct[hard_positive_mask].sum().item())
    stats["hard_final_top1_sum"] += float(final_correct[hard_positive_mask].sum().item())
    stats["activation_sum"] += float(activation.sum().item())
    stats["active_groups_sum"] += float((activation > 0.10).sum().item())
    stats["margin_sum"] += float(outputs["margin"].sum().item())
    stats["entropy_sum"] += float(outputs["entropy"].sum().item())
    stats["score_shift_sum"] += float(score_shift.sum().item())
    stats["amb_shift_sum"] += float(score_shift[ambiguous_mask].sum().item())
    stats["easy_shift_sum"] += float(score_shift[easy_mask].sum().item())
    stats["bg_shift_sum"] += float(score_shift[background_mask].sum().item())
    stats["bg_max_base_sum"] += float(bg_max_base[background_mask].sum().item())
    stats["bg_max_final_sum"] += float(bg_max_final[background_mask].sum().item())
    stats["bg_suppression_sum"] += float((bg_max_base[background_mask] - bg_max_final[background_mask]).sum().item())


def _finalize_stats(stats: dict[str, float]) -> dict[str, float]:
    groups = stats["groups"]
    pos_groups = stats["pos_groups"]
    amb_groups = stats["amb_groups"]
    easy_groups = stats["easy_groups"]
    bg_groups = stats["bg_groups"]
    hard_groups = stats["hard_groups"]
    recoverable_groups = stats["recoverable_groups"]
    valid_positive_groups = stats["valid_positive_groups"]
    return {
        "groups": groups,
        "pos_groups": pos_groups,
        "amb_groups": amb_groups,
        "easy_groups": easy_groups,
        "bg_groups": bg_groups,
        "hard_groups": hard_groups,
        "valid_positive_groups": valid_positive_groups,
        "selected_positive_groups": stats["selected_positive_groups"],
        "unrecoverable_positive_groups": stats["unrecoverable_positive_groups"],
        "loss": _safe_div(stats["loss_sum"], groups),
        "list_loss": _safe_div(stats["list_loss_sum"], groups),
        "duel_loss": _safe_div(stats["duel_loss_sum"], groups),
        "safe_loss": _safe_div(stats["safe_loss_sum"], groups),
        "distill_loss": _safe_div(stats["distill_loss_sum"], groups),
        "gate_loss": _safe_div(stats["gate_loss_sum"], groups),
        "base_top1": _safe_div(stats["base_top1_sum"], pos_groups),
        "final_top1": _safe_div(stats["final_top1_sum"], pos_groups),
        "top1_gain": _safe_div(stats["final_top1_sum"] - stats["base_top1_sum"], pos_groups),
        "amb_base_top1": _safe_div(stats["amb_base_top1_sum"], amb_groups),
        "amb_final_top1": _safe_div(stats["amb_final_top1_sum"], amb_groups),
        "amb_top1_gain": _safe_div(stats["amb_final_top1_sum"] - stats["amb_base_top1_sum"], amb_groups),
        "easy_base_top1": _safe_div(stats["easy_base_top1_sum"], easy_groups),
        "easy_final_top1": _safe_div(stats["easy_final_top1_sum"], easy_groups),
        "easy_top1_gain": _safe_div(stats["easy_final_top1_sum"] - stats["easy_base_top1_sum"], easy_groups),
        "hard_base_top1": _safe_div(stats["hard_base_top1_sum"], hard_groups),
        "hard_final_top1": _safe_div(stats["hard_final_top1_sum"], hard_groups),
        "hard_top1_gain": _safe_div(stats["hard_final_top1_sum"] - stats["hard_base_top1_sum"], hard_groups),
        "recoverable_rate": _safe_div(stats["recovered_groups"], recoverable_groups),
        "selected_positive_rate": _safe_div(stats["selected_positive_groups"], valid_positive_groups),
        "unrecoverable_positive_rate": _safe_div(stats["unrecoverable_positive_groups"], valid_positive_groups),
        "active_rate": _safe_div(stats["active_groups_sum"], groups),
        "activation_mean": _safe_div(stats["activation_sum"], groups),
        "margin_mean": _safe_div(stats["margin_sum"], groups),
        "entropy_mean": _safe_div(stats["entropy_sum"], groups),
        "score_shift_mean": _safe_div(stats["score_shift_sum"], groups),
        "amb_shift_mean": _safe_div(stats["amb_shift_sum"], amb_groups),
        "easy_shift_mean": _safe_div(stats["easy_shift_sum"], easy_groups),
        "bg_shift_mean": _safe_div(stats["bg_shift_sum"], bg_groups),
        "bg_max_base": _safe_div(stats["bg_max_base_sum"], bg_groups),
        "bg_max_final": _safe_div(stats["bg_max_final_sum"], bg_groups),
        "bg_suppression": _safe_div(stats["bg_suppression_sum"], bg_groups),
    }


def _run_epoch(
    model: RuntimeReplayAssociationAdapter,
    shard_map: dict[str, list[int]],
    device: torch.device,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
    phase: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    stats = _new_stats()

    shard_items = list(shard_map.items())
    if is_train:
        random.shuffle(shard_items)

    total_shards = len(shard_items)
    for shard_idx, (shard_path, group_indices) in enumerate(shard_items, start=1):
        print(
            f"[{phase}] shard {shard_idx}/{total_shards} "
            f"path={Path(shard_path).name} groups={len(group_indices)}",
            flush=True,
        )
        shard = PreparedShard(Path(shard_path))
        group_indices = list(group_indices)
        if is_train:
            random.shuffle(group_indices)
        for start in range(0, len(group_indices), int(args.batch_groups)):
            batch_indices = group_indices[start:start + int(args.batch_groups)]
            batch = _collate_batch(
                shard=shard,
                group_indices=batch_indices,
                topk=int(args.topk),
                device=device,
                valid_only=bool(args.valid_only),
                force_positive_into_topk=bool(args.force_positive_into_topk),
            )
            if not bool(batch["candidate_mask"].any()):
                continue
            model_valid_mask = batch["supervised_mask"] if bool(args.valid_only) else batch["candidate_mask"]
            with torch.set_grad_enabled(is_train):
                outputs = model(
                    anchor_scores=batch["anchor_scores"],
                    scalar_features=batch["scalar_feat"],
                    det_features=batch["det_feat"],
                    hist_features=batch["hist_feat"],
                    hist_masks=batch["hist_mask"],
                    hist_times=batch["hist_time"],
                    det_times=batch["det_times"],
                    det_scores=batch["det_scores"],
                    valid_mask=model_valid_mask,
                )
                losses = _compute_losses(outputs=outputs, batch=batch, args=args)
                loss = losses["loss"]

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if float(args.grad_clip) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                optimizer.step()

            _update_stats(stats=stats, outputs=outputs, batch=batch, losses=losses)
    return _finalize_stats(stats)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _save_checkpoint_snapshot(
    model: RuntimeReplayAssociationAdapter,
    path: Path,
    epoch: int,
    selection_score: float,
    extra: dict[str, Any] | None = None,
) -> None:
    payload_extra = {
        "snapshot_epoch": int(epoch),
        "snapshot_selection_score": float(selection_score),
    }
    if extra:
        payload_extra.update(extra)
    save_runtime_replay_checkpoint(model=model, path=path, extra=payload_extra)


def main() -> None:
    faulthandler.enable(all_threads=True)
    args = parse_args()
    _set_seed(int(args.seed))
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    shard_paths = _discover_shards(Path(args.input_dir).resolve())
    if not shard_paths:
        raise FileNotFoundError(f"No runtime replay shards found under {args.input_dir}")

    refs = _collect_refs(shard_paths)
    all_seq_list = _unique_seq_list(refs)
    train_refs, val_refs = _split_refs(
        refs=refs,
        train_seqs=_parse_seq_set(args.train_seqs),
        val_seqs=_parse_seq_set(args.val_seqs),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    train_refs = _limit_refs(train_refs, int(args.max_train_groups), seed=int(args.seed) + 17)
    val_refs = _limit_refs(val_refs, int(args.max_val_groups), seed=int(args.seed) + 31)
    if not train_refs:
        raise RuntimeError("Training split is empty.")
    if not val_refs:
        val_refs = list(train_refs)

    train_seq_list = _unique_seq_list(train_refs)
    val_seq_list = _unique_seq_list(val_refs)
    split_mode = "explicit_seq_split" if (args.train_seqs or args.val_seqs) else "random_group_split"
    print(
        f"[split] mode={split_mode} total_groups={len(refs)} "
        f"train_groups={len(train_refs)} val_groups={len(val_refs)} "
        f"train_seqs={','.join(train_seq_list) if train_seq_list else '<none>'} "
        f"val_seqs={','.join(val_seq_list) if val_seq_list else '<none>'}",
        flush=True,
    )
    train_hard = sum(1 for ref in train_refs if _is_hard_positive_ref(ref))
    val_hard = sum(1 for ref in val_refs if _is_hard_positive_ref(ref))
    print(
        f"[split] hard_positive train={train_hard}/{len(train_refs)} "
        f"val={val_hard}/{len(val_refs)} "
        f"train_epoch_budget={int(args.train_groups_per_epoch) or len(train_refs)} "
        f"val_epoch_budget={int(args.val_groups_per_epoch) or len(val_refs)}",
        flush=True,
    )
    if split_mode == "random_group_split" and len(all_seq_list) > 1:
        print(
            "[warning] Using random group split across multiple sequences; validation may be optimistic "
            "because nearby groups from the same sequence can appear in both train and val.",
            flush=True,
        )

    metrics_path = Path(args.metrics_path) if args.metrics_path else Path(args.out_ckpt).with_suffix(".metrics.jsonl")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("", encoding="utf-8")
    out_ckpt_path = Path(args.out_ckpt).resolve()
    latest_ckpt_path = out_ckpt_path.with_name(out_ckpt_path.stem + ".latest.pt")
    best_ckpt_path = out_ckpt_path.with_name(out_ckpt_path.stem + ".best.pt")
    crash_ckpt_path = out_ckpt_path.with_name(out_ckpt_path.stem + ".crash.pt")
    epoch_ckpt_dir = Path(args.save_every_epoch_dir).resolve() if args.save_every_epoch_dir else out_ckpt_path.parent / "epoch_ckpts"
    epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)
    error_path = metrics_path.with_suffix(".error.log")
    if error_path.exists():
        error_path.unlink()
    _append_jsonl(
        metrics_path,
        {
            "event": "startup",
            "input_dir": str(Path(args.input_dir).resolve()),
            "train_groups": int(len(train_refs)),
            "val_groups": int(len(val_refs)),
            "train_sequences": train_seq_list,
            "val_sequences": val_seq_list,
            "latest_checkpoint_path": str(latest_ckpt_path),
            "best_checkpoint_path": str(best_ckpt_path),
            "epoch_checkpoint_dir": str(epoch_ckpt_dir),
            "fixed_val_sample": bool(args.fixed_val_sample),
        },
    )

    print(f"[stage] infer model dims from shard={Path(shard_paths[0]).name}", flush=True)
    scalar_dim, feat_dim = _infer_model_dims(shard_paths[0])
    print(f"[stage] inferred dims scalar_dim={scalar_dim} feat_dim={feat_dim}", flush=True)
    model = RuntimeReplayAssociationAdapter(
        scalar_dim=int(scalar_dim),
        feat_dim=int(feat_dim),
        scalar_hidden=int(args.scalar_hidden),
        scalar_out=int(args.scalar_out),
        temporal_hidden=int(args.temporal_hidden),
        temporal_out=int(args.temporal_out),
        token_dim=int(args.token_dim),
        duel_dim=int(args.duel_dim),
        group_hidden=int(args.group_hidden),
        num_heads=int(args.num_heads),
        topk=int(args.topk),
        delta_scale=float(args.delta_scale),
        margin_threshold=float(args.margin_threshold),
        margin_temperature=float(args.margin_temperature),
        gate_cap=float(args.gate_cap),
        dropout=float(args.dropout),
    ).to(device)
    if args.init_ckpt:
        init_ckpt = torch.load(str(Path(args.init_ckpt).resolve()), map_location="cpu")
        state_dict = init_ckpt["state_dict"] if isinstance(init_ckpt, dict) and "state_dict" in init_ckpt else init_ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"[stage] init_ckpt={Path(args.init_ckpt).resolve()} "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    train_shard_map = _group_ref_map(train_refs)
    val_shard_map = _group_ref_map(val_refs)

    best_state = None
    best_epoch = 0
    best_score = -float("inf")
    bad_epochs = 0

    print(
        f"[stage] split ready train_groups={len(train_refs)} val_groups={len(val_refs)} "
        f"train_shards={len(train_shard_map)} val_shards={len(val_shard_map)}",
        flush=True,
    )

    fixed_epoch_val_refs: list[GroupRef] | None = None
    if bool(args.fixed_val_sample):
        fixed_epoch_val_refs = _sample_epoch_refs(
            refs=val_refs,
            groups_per_epoch=int(args.val_groups_per_epoch),
            args=args,
            epoch=1,
            seed_offset=20000,
        )
        fixed_val_hard = sum(1 for ref in fixed_epoch_val_refs if _is_hard_positive_ref(ref))
        print(
            f"[stage] fixed_val enabled val_groups={len(fixed_epoch_val_refs)} hard={fixed_val_hard}",
            flush=True,
        )

    last_finished_epoch = 0
    try:
        for epoch in range(1, int(args.epochs) + 1):
            epoch_train_refs = _sample_epoch_refs(
                refs=train_refs,
                groups_per_epoch=int(args.train_groups_per_epoch),
                args=args,
                epoch=epoch,
                seed_offset=10000,
            )
            if fixed_epoch_val_refs is None:
                epoch_val_refs = _sample_epoch_refs(
                    refs=val_refs,
                    groups_per_epoch=int(args.val_groups_per_epoch),
                    args=args,
                    epoch=epoch,
                    seed_offset=20000,
                )
            else:
                epoch_val_refs = list(fixed_epoch_val_refs)
            epoch_train_shard_map = _group_ref_map(epoch_train_refs)
            epoch_val_shard_map = _group_ref_map(epoch_val_refs)
            epoch_train_hard = sum(1 for ref in epoch_train_refs if _is_hard_positive_ref(ref))
            epoch_val_hard = sum(1 for ref in epoch_val_refs if _is_hard_positive_ref(ref))
            print(
                f"[stage] epoch {epoch:02d} sampled train_groups={len(epoch_train_refs)} hard={epoch_train_hard} "
                f"val_groups={len(epoch_val_refs)} hard={epoch_val_hard}",
                flush=True,
            )
            print(f"[stage] epoch {epoch:02d} train_start", flush=True)
            train_metrics = _run_epoch(
                model=model,
                shard_map=epoch_train_shard_map,
                device=device,
                args=args,
                optimizer=optimizer,
                phase=f"train e{epoch:02d}",
            )
            print(f"[stage] epoch {epoch:02d} val_start", flush=True)
            val_metrics = _run_epoch(
                model=model,
                shard_map=epoch_val_shard_map,
                device=device,
                args=args,
                optimizer=None,
                phase=f"val e{epoch:02d}",
            )
            selection = _selection_score(val_metrics, args)

            print(
                f"[epoch {epoch:02d}] "
                f"train_loss={train_metrics['loss']:.6f} base_top1={train_metrics['base_top1']:.4f} "
                f"final_top1={train_metrics['final_top1']:.4f} gain={train_metrics['top1_gain']:+.4f} "
                f"amb_gain={train_metrics['amb_top1_gain']:+.4f} hard_gain={train_metrics['hard_top1_gain']:+.4f} "
                f"topk_pos={train_metrics['selected_positive_rate']:.4f} "
                f"unrec={train_metrics['unrecoverable_positive_rate']:.4f} "
                f"active={train_metrics['active_rate']:.4f}",
                flush=True,
            )
            print(
                f"[epoch {epoch:02d}] "
                f"val_loss={val_metrics['loss']:.6f} base_top1={val_metrics['base_top1']:.4f} "
                f"final_top1={val_metrics['final_top1']:.4f} gain={val_metrics['top1_gain']:+.4f} "
                f"amb_gain={val_metrics['amb_top1_gain']:+.4f} hard_gain={val_metrics['hard_top1_gain']:+.4f} "
                f"topk_pos={val_metrics['selected_positive_rate']:.4f} "
                f"unrec={val_metrics['unrecoverable_positive_rate']:.4f} "
                f"easy_shift={val_metrics['easy_shift_mean']:.6f} "
                f"bg_suppress={val_metrics['bg_suppression']:+.6f} recover={val_metrics['recoverable_rate']:.4f} "
                f"sel={selection:+.6f}",
                flush=True,
            )

            _append_jsonl(
                metrics_path,
                {
                    "epoch": int(epoch),
                    "selection_score": float(selection),
                    "train": train_metrics,
                    "val": val_metrics,
                },
            )

            _save_checkpoint_snapshot(
                model=model,
                path=latest_ckpt_path,
                epoch=epoch,
                selection_score=selection,
                extra={
                    "kind": "latest",
                    "train_group_count": int(len(train_refs)),
                    "val_group_count": int(len(val_refs)),
                },
            )
            _save_checkpoint_snapshot(
                model=model,
                path=epoch_ckpt_dir / f"epoch_{epoch:02d}.pt",
                epoch=epoch,
                selection_score=selection,
                extra={
                    "kind": "epoch",
                    "epoch": int(epoch),
                    "train_group_count": int(len(train_refs)),
                    "val_group_count": int(len(val_refs)),
                },
            )

            if selection > best_score + 1e-8:
                best_score = float(selection)
                best_epoch = int(epoch)
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                _save_checkpoint_snapshot(
                    model=model,
                    path=best_ckpt_path,
                    epoch=epoch,
                    selection_score=selection,
                    extra={
                        "kind": "best",
                        "train_group_count": int(len(train_refs)),
                        "val_group_count": int(len(val_refs)),
                    },
                )
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(args.patience):
                    print(f"[early-stop] epoch={epoch} best_epoch={best_epoch} best_sel={best_score:+.6f}", flush=True)
                    last_finished_epoch = int(epoch)
                    break

            last_finished_epoch = int(epoch)
    except BaseException as exc:
        tb = traceback.format_exc()
        error_path.write_text(tb, encoding="utf-8")
        _append_jsonl(
            metrics_path,
            {
                "event": "crash",
                "epoch": int(last_finished_epoch),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "error_log": str(error_path),
            },
        )
        if best_state is not None:
            model.load_state_dict(best_state, strict=True)
            _save_checkpoint_snapshot(
                model=model,
                path=crash_ckpt_path,
                epoch=best_epoch,
                selection_score=best_score,
                extra={
                    "kind": "crash_best",
                    "train_group_count": int(len(train_refs)),
                    "val_group_count": int(len(val_refs)),
                },
            )
        print(tb, file=sys.stderr, flush=True)
        raise

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    save_runtime_replay_checkpoint(
        model=model,
        path=out_ckpt_path,
        extra={
            "best_epoch": int(best_epoch),
            "best_selection_score": float(best_score),
            "train_group_count": int(len(train_refs)),
            "val_group_count": int(len(val_refs)),
        },
    )

    summary_path = metrics_path.with_suffix(".summary.json")
    summary = {
        "input_dir": str(Path(args.input_dir).resolve()),
        "checkpoint_path": str(Path(args.out_ckpt).resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "split_mode": split_mode,
        "all_sequences": all_seq_list,
        "train_sequences": train_seq_list,
        "val_sequences": val_seq_list,
        "train_groups": int(len(train_refs)),
        "val_groups": int(len(val_refs)),
        "best_epoch": int(best_epoch),
        "best_selection_score": float(best_score),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[saved] {args.out_ckpt}", flush=True)
    print(f"[summary] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
