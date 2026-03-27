#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.mtcr_assoc import MTCRAssociationAdapter
from train_haca_v1_from_gt_tracks import group_segments, load_datasets, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MTCR from GT pseudo-track NPZ groups.")
    parser.add_argument("--train-npz", nargs="+", required=True)
    parser.add_argument("--val-npz", nargs="*", default=[])
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--batch-groups", type=int, default=0, help="Only used for shard loading; 0 keeps full shard order.")
    parser.add_argument("--hist-hidden", type=int, default=16)
    parser.add_argument("--comp-hidden", type=int, default=64)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--margin-threshold", type=float, default=-1.0, help="If < 0, fit from train anchor margins.")
    parser.add_argument("--margin-quantile", type=float, default=0.35)
    parser.add_argument("--margin-temperature", type=float, default=0.03)
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--decay-scales", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--duel-margin", type=float, default=0.20)
    parser.add_argument("--loss-duel-weight", type=float, default=0.50)
    parser.add_argument("--loss-safe-weight", type=float, default=0.20)
    parser.add_argument("--loss-bg-weight", type=float, default=0.75)
    parser.add_argument("--loss-gate-weight", type=float, default=0.10)
    parser.add_argument("--ambiguous-weight", type=float, default=2.0)
    parser.add_argument("--easy-weight", type=float, default=0.5)
    parser.add_argument("--background-weight", type=float, default=0.75)
    parser.add_argument("--gate-cap", type=float, default=0.20)
    parser.add_argument("--bg-scale", type=float, default=0.75)
    parser.add_argument("--select-amb-weight", type=float, default=1.5)
    parser.add_argument("--select-bg-weight", type=float, default=0.10)
    parser.add_argument("--select-easy-weight", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--metrics-path", type=str, default="", help="Optional JSONL path for per-epoch metrics.")
    return parser.parse_args()


def _ctx_anchor_column(ctx_feat: np.ndarray) -> int:
    # Legacy shards only stored spatial_sim as column 0. New shards append fused_sim at column 8.
    return 8 if ctx_feat.ndim == 2 and ctx_feat.shape[1] >= 9 else 0


def _group_batch(data, start: int, end: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    det_feat = torch.from_numpy(data.det_feat[start:start + 1]).to(device=device, dtype=torch.float32)
    hist_feat = torch.from_numpy(data.hist_feat[start:end]).to(device=device, dtype=torch.float32)
    hist_mask = torch.from_numpy(data.hist_mask[start:end] < 0.5).to(device=device, dtype=torch.bool)
    anchor_col = _ctx_anchor_column(data.ctx_feat)
    anchor_scores = torch.from_numpy(data.ctx_feat[start:end, anchor_col]).to(device=device, dtype=torch.float32).view(1, -1)
    motion_scores = torch.from_numpy(data.ctx_feat[start:end, 1]).to(device=device, dtype=torch.float32).view(1, -1)
    det_score = torch.from_numpy(data.ctx_feat[start:start + 1, 2]).to(device=device, dtype=torch.float32).view(1)
    track_gaps = torch.from_numpy(np.maximum(np.expm1(data.track_feat[start:end, 0]).astype(np.float32), 0.0)).to(device=device, dtype=torch.float32)
    labels = torch.from_numpy(data.label[start:end]).to(device=device, dtype=torch.float32)
    return det_feat, hist_feat, hist_mask, anchor_scores, motion_scores, det_score, track_gaps, labels


def _fit_margin_threshold(train_sets, quantile: float) -> float:
    margins: list[float] = []
    for data in train_sets:
        anchor_col = _ctx_anchor_column(data.ctx_feat)
        group_ids = torch.from_numpy(data.group_id)
        for start, end in group_segments(group_ids):
            scores = np.asarray(data.ctx_feat[start:end, anchor_col], dtype=np.float32)
            if scores.size == 0:
                continue
            if scores.size == 1:
                margins.append(float(scores[0]))
            else:
                top2 = np.sort(scores)[-2:]
                margins.append(float(top2[-1] - top2[-2]))
    if not margins:
        return 0.10
    q = min(max(float(quantile), 0.05), 0.95)
    return float(np.quantile(np.asarray(margins, dtype=np.float32), q))


def _group_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    margin_threshold: float,
    temperature: float,
    duel_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, bool, bool]:
    logits = torch.logit(outputs["final_scores"].view(-1).clamp(min=1e-4, max=1.0 - 1e-4), eps=1e-4) / max(float(temperature), 1e-4)
    base_logits = torch.logit(outputs["pair_tokens"][0, :, 0].clamp(min=1e-4, max=1.0 - 1e-4), eps=1e-4) / max(float(temperature), 1e-4)
    base_scores = outputs["pair_tokens"][0, :, 0]
    final_scores = outputs["final_scores"].view(-1)
    group_gate = outputs.get("group_gate", final_scores.new_zeros((1, 1))).view(-1)[0]
    bg_prob = outputs.get("bg_prob", final_scores.new_zeros((1, 1))).view(-1)[0]
    top1 = float(torch.argmax(final_scores).item() == int(torch.argmax(labels).item())) if bool(torch.any(labels > 0.5)) else 0.0

    if not bool(torch.any(labels > 0.5)):
        bg_loss = F.binary_cross_entropy(bg_prob.clamp(min=1e-4, max=1.0 - 1e-4), bg_prob.new_tensor(1.0))
        bg_loss = bg_loss + (-torch.log1p(-final_scores.max().clamp(max=1.0 - 1e-4)))
        gate_loss = group_gate.square()
        return logits.new_tensor(0.0), logits.new_tensor(0.0), logits.new_tensor(0.0), bg_loss, gate_loss, top1, False, False

    pos_idx = int(torch.argmax(labels).item())
    list_loss = F.cross_entropy(logits.view(1, -1), torch.tensor([pos_idx], device=logits.device))
    bg_loss = F.binary_cross_entropy(bg_prob.clamp(min=1e-4, max=1.0 - 1e-4), bg_prob.new_tensor(0.0))
    if logits.numel() <= 1:
        return list_loss, logits.new_tensor(0.0), logits.new_tensor(0.0), bg_loss, group_gate.square(), top1, True, False

    margin = outputs["comp_margin"][0, 0] if outputs["comp_margin"].numel() > 0 else outputs["final_scores"].view(-1).new_tensor(0.0)
    pred_idx = int(torch.argmax(base_scores).item())
    ambiguous = bool(pred_idx != pos_idx or float(margin.item()) < float(margin_threshold))
    if ambiguous:
        neg_mask = torch.ones_like(labels, dtype=torch.bool)
        neg_mask[pos_idx] = False
        duel_loss = F.relu(logits.new_tensor(float(duel_margin)) - logits[pos_idx] + logits[neg_mask].max())
        safe_loss = logits.new_tensor(0.0)
        gate_loss = logits.new_tensor(0.0)
    else:
        duel_loss = logits.new_tensor(0.0)
        safe_loss = torch.mean((logits - base_logits) ** 2)
        gate_loss = group_gate.square()
    return list_loss, duel_loss, safe_loss, bg_loss, gate_loss, top1, True, ambiguous


def _new_epoch_stats() -> dict[str, float]:
    return {
        "groups": 0.0,
        "pos_groups": 0.0,
        "amb_groups": 0.0,
        "easy_groups": 0.0,
        "bg_groups": 0.0,
        "loss_sum": 0.0,
        "list_loss_sum": 0.0,
        "duel_loss_sum": 0.0,
        "safe_loss_sum": 0.0,
        "bg_loss_sum": 0.0,
        "gate_loss_sum": 0.0,
        "base_top1_sum": 0.0,
        "final_top1_sum": 0.0,
        "improved_sum": 0.0,
        "worsened_sum": 0.0,
        "amb_base_top1_sum": 0.0,
        "amb_final_top1_sum": 0.0,
        "easy_base_top1_sum": 0.0,
        "easy_final_top1_sum": 0.0,
        "activation_sum": 0.0,
        "active_groups_sum": 0.0,
        "entropy_sum": 0.0,
        "margin_sum": 0.0,
        "score_shift_sum": 0.0,
        "residual_abs_sum": 0.0,
        "amb_activation_sum": 0.0,
        "amb_shift_sum": 0.0,
        "easy_shift_sum": 0.0,
        "bg_shift_sum": 0.0,
        "bg_max_base_sum": 0.0,
        "bg_max_final_sum": 0.0,
        "bg_suppression_sum": 0.0,
        "group_gate_sum": 0.0,
        "bg_prob_sum": 0.0,
    }


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) > 0.0 else 0.0


def _collect_group_stats(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    margin_threshold: float,
) -> dict[str, float]:
    stats = _new_epoch_stats()
    final_scores = outputs["final_scores"].view(-1)
    base_scores = outputs["pair_tokens"][0, :, 0]
    activation = float(outputs["comp_active"][0, 0].item()) if outputs["comp_active"].numel() > 0 else 0.0
    margin = float(outputs["comp_margin"][0, 0].item()) if outputs["comp_margin"].numel() > 0 else 0.0
    entropy = float(outputs["comp_entropy"][0, 0].item()) if outputs["comp_entropy"].numel() > 0 else 0.0
    score_shift = float(torch.mean(torch.abs(final_scores - base_scores)).item()) if final_scores.numel() > 0 else 0.0
    residual_abs = float(torch.mean(torch.abs(outputs["comp_residual"])).item()) if outputs["comp_residual"].numel() > 0 else 0.0
    group_gate = float(outputs["group_gate"].view(-1)[0].item()) if "group_gate" in outputs else 0.0
    bg_prob = float(outputs["bg_prob"].view(-1)[0].item()) if "bg_prob" in outputs else 0.0

    stats["groups"] = 1.0
    stats["activation_sum"] = activation
    stats["active_groups_sum"] = 1.0 if activation > 0.10 else 0.0
    stats["entropy_sum"] = entropy
    stats["margin_sum"] = margin
    stats["score_shift_sum"] = score_shift
    stats["residual_abs_sum"] = residual_abs
    stats["group_gate_sum"] = group_gate
    stats["bg_prob_sum"] = bg_prob

    is_positive = bool(torch.any(labels > 0.5))
    if not is_positive:
        bg_max_base = float(base_scores.max().item()) if base_scores.numel() > 0 else 0.0
        bg_max_final = float(final_scores.max().item()) if final_scores.numel() > 0 else 0.0
        stats["bg_groups"] = 1.0
        stats["bg_shift_sum"] = score_shift
        stats["bg_max_base_sum"] = bg_max_base
        stats["bg_max_final_sum"] = bg_max_final
        stats["bg_suppression_sum"] = bg_max_base - bg_max_final
        return stats

    pos_idx = int(torch.argmax(labels).item())
    base_pred = int(torch.argmax(base_scores).item())
    final_pred = int(torch.argmax(final_scores).item())
    base_correct = 1.0 if base_pred == pos_idx else 0.0
    final_correct = 1.0 if final_pred == pos_idx else 0.0
    ambiguous = 1.0 if (base_pred != pos_idx or margin < float(margin_threshold)) else 0.0
    easy = 1.0 - ambiguous

    stats["pos_groups"] = 1.0
    stats["base_top1_sum"] = base_correct
    stats["final_top1_sum"] = final_correct
    stats["improved_sum"] = 1.0 if final_correct > base_correct else 0.0
    stats["worsened_sum"] = 1.0 if final_correct < base_correct else 0.0

    if ambiguous > 0.5:
        stats["amb_groups"] = 1.0
        stats["amb_base_top1_sum"] = base_correct
        stats["amb_final_top1_sum"] = final_correct
        stats["amb_activation_sum"] = activation
        stats["amb_shift_sum"] = score_shift
    else:
        stats["easy_groups"] = 1.0
        stats["easy_base_top1_sum"] = base_correct
        stats["easy_final_top1_sum"] = final_correct
        stats["easy_shift_sum"] = score_shift
    return stats


def _merge_epoch_stats(dst: dict[str, float], src: dict[str, float]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0.0) + float(value)


def _finalize_epoch_stats(stats: dict[str, float]) -> dict[str, float]:
    groups = stats["groups"]
    pos_groups = stats["pos_groups"]
    amb_groups = stats["amb_groups"]
    easy_groups = stats["easy_groups"]
    bg_groups = stats["bg_groups"]
    result = {
        "groups": groups,
        "pos_groups": pos_groups,
        "amb_groups": amb_groups,
        "easy_groups": easy_groups,
        "bg_groups": bg_groups,
        "loss": _safe_div(stats["loss_sum"], groups),
        "list_loss": _safe_div(stats["list_loss_sum"], groups),
        "duel_loss": _safe_div(stats["duel_loss_sum"], groups),
        "safe_loss": _safe_div(stats["safe_loss_sum"], groups),
        "bg_loss": _safe_div(stats["bg_loss_sum"], groups),
        "gate_loss": _safe_div(stats["gate_loss_sum"], groups),
        "base_top1": _safe_div(stats["base_top1_sum"], pos_groups),
        "final_top1": _safe_div(stats["final_top1_sum"], pos_groups),
        "top1_gain": _safe_div(stats["final_top1_sum"] - stats["base_top1_sum"], pos_groups),
        "improved_rate": _safe_div(stats["improved_sum"], pos_groups),
        "worsened_rate": _safe_div(stats["worsened_sum"], pos_groups),
        "amb_base_top1": _safe_div(stats["amb_base_top1_sum"], amb_groups),
        "amb_final_top1": _safe_div(stats["amb_final_top1_sum"], amb_groups),
        "amb_top1_gain": _safe_div(stats["amb_final_top1_sum"] - stats["amb_base_top1_sum"], amb_groups),
        "easy_base_top1": _safe_div(stats["easy_base_top1_sum"], easy_groups),
        "easy_final_top1": _safe_div(stats["easy_final_top1_sum"], easy_groups),
        "easy_top1_gain": _safe_div(stats["easy_final_top1_sum"] - stats["easy_base_top1_sum"], easy_groups),
        "active_rate": _safe_div(stats["active_groups_sum"], groups),
        "activation_mean": _safe_div(stats["activation_sum"], groups),
        "amb_activation_mean": _safe_div(stats["amb_activation_sum"], amb_groups),
        "margin_mean": _safe_div(stats["margin_sum"], groups),
        "entropy_mean": _safe_div(stats["entropy_sum"], groups),
        "score_shift_mean": _safe_div(stats["score_shift_sum"], groups),
        "residual_abs_mean": _safe_div(stats["residual_abs_sum"], groups),
        "residual_gate_mean": _safe_div(stats["group_gate_sum"], groups),
        "bg_prob_mean": _safe_div(stats["bg_prob_sum"], groups),
        "amb_shift_mean": _safe_div(stats["amb_shift_sum"], amb_groups),
        "easy_shift_mean": _safe_div(stats["easy_shift_sum"], easy_groups),
        "bg_shift_mean": _safe_div(stats["bg_shift_sum"], bg_groups),
        "bg_max_base": _safe_div(stats["bg_max_base_sum"], bg_groups),
        "bg_max_final": _safe_div(stats["bg_max_final_sum"], bg_groups),
        "bg_suppression": _safe_div(stats["bg_suppression_sum"], bg_groups),
    }
    return result


def _append_metrics_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _selection_score(metrics: dict[str, float], args: argparse.Namespace) -> float:
    return (
        float(metrics["top1_gain"])
        + float(args.select_amb_weight) * float(metrics["amb_top1_gain"])
        + float(args.select_bg_weight) * float(metrics["bg_suppression"])
        - float(args.select_easy_weight) * float(metrics["easy_shift_mean"])
    )


def _group_weight(args: argparse.Namespace, is_positive: bool, ambiguous: bool) -> float:
    if not is_positive:
        return float(args.background_weight)
    if ambiguous:
        return float(args.ambiguous_weight)
    return float(args.easy_weight)


def _run_epoch(
    model: MTCRAssociationAdapter,
    datasets,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    margin_threshold: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    stats = _new_epoch_stats()

    for shard_idx, data in enumerate(datasets, start=1):
        group_slices = group_segments(torch.from_numpy(data.group_id))
        if is_train:
            random.shuffle(group_slices)
        for start, end in group_slices:
            det_feat, hist_feat, hist_mask, anchor_scores, motion_scores, det_score, track_gaps, labels = _group_batch(
                data=data,
                start=start,
                end=end,
                device=device,
            )
            force_include_indices = None
            if is_train and bool(torch.any(labels > 0.5)):
                force_include_indices = torch.tensor([int(torch.argmax(labels).item())], device=device, dtype=torch.long)
            with torch.set_grad_enabled(is_train):
                outputs = model(
                    anchor_scores=anchor_scores,
                    det_features=det_feat,
                    track_history_features=hist_feat,
                    track_history_masks=hist_mask,
                    motion_scores=motion_scores,
                    det_scores=det_score,
                    track_gaps=track_gaps,
                    force_include_indices=force_include_indices,
                )
                list_loss, duel_loss, safe_loss, bg_loss, gate_loss, top1, is_positive, ambiguous = _group_loss(
                    outputs=outputs,
                    labels=labels,
                    margin_threshold=margin_threshold,
                    temperature=args.temperature,
                    duel_margin=args.duel_margin,
                )
                sample_weight = _group_weight(args=args, is_positive=is_positive, ambiguous=ambiguous)
                loss = sample_weight * (
                    list_loss
                    + args.loss_duel_weight * duel_loss
                    + args.loss_safe_weight * safe_loss
                    + args.loss_bg_weight * bg_loss
                    + args.loss_gate_weight * gate_loss
                )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if loss.requires_grad:
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
                    optimizer.step()

            group_stats = _collect_group_stats(outputs=outputs, labels=labels, margin_threshold=margin_threshold)
            group_stats["loss_sum"] = float(loss.item())
            group_stats["list_loss_sum"] = float(list_loss.item())
            group_stats["duel_loss_sum"] = float(duel_loss.item())
            group_stats["safe_loss_sum"] = float(safe_loss.item())
            group_stats["bg_loss_sum"] = float(bg_loss.item())
            group_stats["gate_loss_sum"] = float(gate_loss.item())
            _merge_epoch_stats(stats, group_stats)

        shard_metrics = _finalize_epoch_stats(stats)

        print(
            f"[{'train' if is_train else 'eval'}] shard {shard_idx}/{len(datasets)} "
            f"path={data.path} avg_loss={shard_metrics['loss']:.6f} "
            f"base_top1={shard_metrics['base_top1']:.4f} mtcr_top1={shard_metrics['final_top1']:.4f} "
            f"amb_gain={shard_metrics['amb_top1_gain']:+.4f} easy_shift={shard_metrics['easy_shift_mean']:.4f} "
            f"bg_suppress={shard_metrics['bg_suppression']:+.4f} "
            f"active={shard_metrics['active_rate']:.4f} gate={shard_metrics['residual_gate_mean']:.4f}",
            flush=True,
        )
    return _finalize_epoch_stats(stats)


def _export_checkpoint(model: MTCRAssociationAdapter, out_path: Path) -> None:
    state = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in model.state_dict().items()}
    payload = {
        "version": np.asarray(["mtcr_v2"], dtype=object),
        "hist_hidden": np.asarray([model.hist_hidden], dtype=np.int32),
        "comp_hidden": np.asarray([model.comp_hidden], dtype=np.int32),
        "comp_topk": np.asarray([model.topk], dtype=np.int32),
        "comp_margin_threshold": np.asarray([model.margin_threshold], dtype=np.float32),
        "comp_margin_temperature": np.asarray([model.margin_temperature], dtype=np.float32),
        "comp_delta_scale": np.asarray([model.delta_scale], dtype=np.float32),
        "comp_gate_cap": np.asarray([model.gate_cap], dtype=np.float32),
        "comp_bg_scale": np.asarray([model.bg_scale], dtype=np.float32),
        "comp_gate_init_logit": np.asarray([model.gate_init_logit], dtype=np.float32),
        "comp_bg_init_logit": np.asarray([model.bg_init_logit], dtype=np.float32),
        "min_history": np.asarray([model.min_history], dtype=np.int32),
        "decay_scales": model.decay_scales.detach().cpu().numpy().astype(np.float32),
        "W_hist1": state["hist_fc1.weight"].T,
        "b_hist1": state["hist_fc1.bias"],
        "W_hist2": state["hist_fc2.weight"].T,
        "b_hist2": state["hist_fc2.bias"],
        "W_hist_attn": state["hist_attn.weight"].T,
        "b_hist_attn": state["hist_attn.bias"],
        "W_duel1": state["duel_fc1.weight"].T,
        "b_duel1": state["duel_fc1.bias"],
        "W_duel2": state["duel_fc2.weight"].T,
        "b_duel2": state["duel_fc2.bias"],
        "W_attn": state["attn_fc.weight"].T,
        "b_attn": state["attn_fc.bias"],
        "W_comp1": state["comp_fc1.weight"].T,
        "b_comp1": state["comp_fc1.bias"],
        "W_comp2": state["comp_fc2.weight"].T,
        "b_comp2": state["comp_fc2.bias"],
        "W_group1": state["group_fc1.weight"].T,
        "b_group1": state["group_fc1.bias"],
        "W_group2": state["group_fc2.weight"].T,
        "b_group2": state["group_fc2.bias"],
        "W_gate": state["gate_head.weight"].T,
        "b_gate": state["gate_head.bias"],
        "W_bg": state["bg_head.weight"].T,
        "b_bg": state["bg_head.bias"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
    print(f"[saved] {out_path}", flush=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    train_sets = load_datasets(args.train_npz, batch_groups=args.batch_groups, split_name="train")
    val_sets = load_datasets(args.val_npz, batch_groups=args.batch_groups, split_name="val") if args.val_npz else []
    if not train_sets:
        raise ValueError("No training shards were provided.")

    margin_threshold = float(args.margin_threshold)
    if margin_threshold < 0.0:
        margin_threshold = _fit_margin_threshold(train_sets, quantile=args.margin_quantile)
    print(f"[margin-threshold] {margin_threshold:.6f}", flush=True)

    model = MTCRAssociationAdapter(
        hist_hidden=args.hist_hidden,
        comp_hidden=args.comp_hidden,
        topk=args.topk,
        margin_threshold=margin_threshold,
        margin_temperature=args.margin_temperature,
        delta_scale=args.delta_scale,
        min_history=args.min_history,
        decay_scales=args.decay_scales,
        gate_cap=args.gate_cap,
        bg_scale=args.bg_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    best_score = -float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    metrics_path = Path(args.metrics_path) if args.metrics_path else Path(args.out_npz).with_suffix(".metrics.jsonl")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("", encoding="utf-8")
    print(f"[metrics] {metrics_path}", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            datasets=train_sets,
            device=device,
            optimizer=optimizer,
            args=args,
            margin_threshold=margin_threshold,
        )
        if val_sets:
            val_metrics = _run_epoch(
                model=model,
                datasets=val_sets,
                device=device,
                optimizer=None,
                args=args,
                margin_threshold=margin_threshold,
            )
        else:
            val_metrics = dict(train_metrics)

        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_metrics['loss']:.6f} base_top1={train_metrics['base_top1']:.4f} "
            f"mtcr_top1={train_metrics['final_top1']:.4f} gain={train_metrics['top1_gain']:+.4f} "
            f"amb_gain={train_metrics['amb_top1_gain']:+.4f} easy_shift={train_metrics['easy_shift_mean']:.4f} "
            f"active={train_metrics['active_rate']:.4f}",
            flush=True,
        )
        print(
            f"[epoch {epoch:02d}] "
            f"val_loss={val_metrics['loss']:.6f} base_top1={val_metrics['base_top1']:.4f} "
            f"mtcr_top1={val_metrics['final_top1']:.4f} gain={val_metrics['top1_gain']:+.4f} "
            f"amb_gain={val_metrics['amb_top1_gain']:+.4f} easy_shift={val_metrics['easy_shift_mean']:.4f} "
            f"bg_suppress={val_metrics['bg_suppression']:+.4f} "
            f"bg_prob={val_metrics['bg_prob_mean']:.4f} sel={_selection_score(val_metrics, args):+.6f} "
            f"active={val_metrics['active_rate']:.4f}",
            flush=True,
        )
        selection_score = _selection_score(val_metrics, args)
        _append_metrics_record(
            metrics_path,
            {
                "epoch": int(epoch),
                "margin_threshold": float(margin_threshold),
                "selection_score": float(selection_score),
                "train": train_metrics,
                "val": val_metrics,
            },
        )

        if selection_score > best_score + 1e-8:
            best_val = val_metrics["loss"]
            best_score = float(selection_score)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(
                    f"[early-stop] epoch={epoch} best_epoch={best_epoch} "
                    f"best_val={best_val:.6f} best_score={best_score:+.6f}",
                    flush=True,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    _export_checkpoint(model=model, out_path=Path(args.out_npz))
    summary_path = metrics_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val),
                "best_selection_score": float(best_score),
                "checkpoint_path": str(Path(args.out_npz)),
                "metrics_path": str(metrics_path),
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"[metrics-summary] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
