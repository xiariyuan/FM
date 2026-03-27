# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import torch.nn.functional as F
import torchvision
import copy
import math
import torch.nn as nn

from utils.misc import is_main_process, is_distributed


# Several calculation functions that are used in multiple model structures:

def pos_to_pos_embed(pos, num_pos_feats: int = 64, temperature: int = 10000, scale: float = 2 * math.pi):
    pos = pos * scale
    dim_i = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_i = temperature ** (2 * (torch.div(dim_i, 2, rounding_mode="trunc")) / num_pos_feats)
    pos_embed = pos[..., None] / dim_i      # (N, M, n_feats) or (B, N, M, n_feats)
    pos_embed = torch.stack((pos_embed[..., 0::2].sin(), pos_embed[..., 1::2].cos()), dim=-1)
    pos_embed = torch.flatten(pos_embed, start_dim=-3)
    return pos_embed


def label_to_one_hot(labels: torch.Tensor, n_classes: int, dtype=torch.float32):
    labels_fixed = labels.clone().to(torch.long)
    labels_fixed[labels_fixed < 0] = n_classes - 1
    labels_fixed = labels_fixed.clamp(min=0, max=n_classes - 1)
    one_hot = F.one_hot(labels_fixed, num_classes=n_classes).to(dtype)
    return one_hot


def inverse_sigmoid(x, eps=1e-5):
    """
    if      x = 1/(1+exp(-y))
    then    y = ln(x/(1-x))
    Args:
        x:
        eps:

    Returns:

    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    # if float(torchvision.__version__[:3]) < 0.7:
    #     if input.numel() > 0:
    #         return torch.nn.functional.interpolate(
    #             input, size, scale_factor, mode, align_corners
    #         )
    #
    #     output_shape = _output_size(2, input, size, scale_factor)
    #     output_shape = list(input.shape[:-2]) + list(output_shape)
    #     if float(torchvision.__version__[:3]) < 0.5:
    #         return _NewEmptyTensorOp.apply(input, output_shape)
    #     return _new_empty_tensor(input, output_shape)
    # else:
    return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    num_classes = output.size(1)
    if num_classes == 0:
        return [torch.zeros([], device=output.device) for _ in topk]
    maxk = min(max(topk), num_classes)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        k = min(k, num_classes)
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def _torch_load_compat(path, map_location):
    """Compatibility wrapper for torch.load with/without weights_only."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support weights_only argument.
        return torch.load(path, map_location=map_location)


def load_detr_pretrain(model: nn.Module, pretrain_path: str, num_classes: int | None, default_class_idx: int | None = None):
    pretrain_model = _torch_load_compat(pretrain_path, map_location=lambda storage, loc: storage)
    pretrain_state_dict = pretrain_model["model"]
    detr_state_dict = dict()
    model_state_dict = model.state_dict()
    for k, v in pretrain_state_dict.items():
        # Avoid double prefix if the checkpoint already has "detr." in keys.
        _k = k if k.startswith("detr.") else f"detr.{k}"
        detr_state_dict[_k] = v      # add the prefix for the detr model (in MOTIP).

    keys_to_delete = []
    for k, v in detr_state_dict.items():
        if "class_embed" in k:
            if num_classes is None:
                num_classes = len(detr_state_dict[k])
            if num_classes == len(detr_state_dict[k]):    # Just fine for the classifier:
                pass
            elif num_classes == 1 and len(detr_state_dict[k]) > 1:
                # Reduce the classifier to a single category (default: person in COCO-style datasets).
                if default_class_idx is None:
                    default_class_idx = 1 if len(detr_state_dict[k]) >= 91 else 0
                detr_state_dict[k] = detr_state_dict[k][default_class_idx:default_class_idx+1]
            else:
                raise NotImplementedError(
                    f"Pretrained detr has a class head for {len(detr_state_dict[k])} classes, "
                    f"we do not support this pretrained model."
                )
        # # For Detect Query:
        # if "query_embed" in k:
        #     if len(detr_state_dict[k]) != len(model_state_dict[k]):
        #         # missmatch for num of det queries
        #         print(">>>> Because the num of det queries is not matched, "
        #               "we only use a part of the pretrained query embed.")
        #         detr_state_dict[k] = model_state_dict[k]
        #     else:
        #         pass
        # for DINO:
        if "label_enc" in k:
            if len(detr_state_dict[k]) != len(model_state_dict[k]):
                # mismatch for num classes
                if len(model_state_dict[k]) == 2:   # 1 class (person + background)
                    if default_class_idx is None:
                        default_class_idx = 1 if len(detr_state_dict[k]) >= 91 else 0
                    detr_state_dict[k] = torch.cat(
                        (detr_state_dict[k][default_class_idx:default_class_idx+1], detr_state_dict[k][-1:]), dim=0
                    )
                elif num_classes == 1 and len(detr_state_dict[k]) > 1:
                    # Same logic as class_embed: reduce to single class
                    if default_class_idx is None:
                        default_class_idx = 1 if len(detr_state_dict[k]) >= 91 else 0
                    # For label_enc in DINO, we need person class + background class
                    detr_state_dict[k] = torch.cat(
                        (detr_state_dict[k][default_class_idx:default_class_idx+1], detr_state_dict[k][-1:]), dim=0
                    )
                else:
                    # Skip label_enc if size mismatch and cannot be handled
                    # This allows using pretrained weights with different labelbook_size
                    print(f"[WARNING] Skipping label_enc loading due to size mismatch: "
                          f"model={len(model_state_dict[k])}, pretrain={len(detr_state_dict[k])}")
                    keys_to_delete.append(k)

    # Delete keys that couldn't be handled
    for k in keys_to_delete:
        del detr_state_dict[k]

    # Transfer the pre-trained parameters to the model state dict.
    for k, v in detr_state_dict.items():
        if k in model_state_dict:
            model_state_dict[k] = v
        else:
            # Skip unmatched keys (for partially compatible checkpoints)
            continue
    # Load the model state dict.
    model.load_state_dict(state_dict=model_state_dict, strict=True)
    return


def save_checkpoint(model, path, states: dict, optimizer, scheduler, only_detr: bool = False):
    if is_main_process():   # only save the model in the main process.
        model = get_model(model)
        if only_detr:
            model = model.detr
        save_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "states": states,
        }
        torch.save(save_state, path)
    return


def load_checkpoint(model, path, states=None, optimizer=None, scheduler=None, strict_check: bool = True):
    """
    Load checkpoint with explicit missing/unexpected key checking.

    Args:
        model: The model to load weights into.
        path: Path to the checkpoint file.
        states: Optional dict to update with checkpoint states.
        optimizer: Optional optimizer to load state.
        scheduler: Optional scheduler to load state.
        strict_check: If True, will print warnings for missing/unexpected keys
                     and raise error if critical DETR keys are missing.
    """
    load_state = _torch_load_compat(path, map_location=lambda storage, loc: storage)
    model_state = load_state["model"]

    if "bbox_embed.0.layers.0.weight" in model_state:
        load_detr_pretrain(model=model, pretrain_path=path, num_classes=None)
        return
    else:
        model_state_dict = model.state_dict()
        loadable_state = {}
        skipped_keys = []
        for k, v in model_state.items():
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                loadable_state[k] = v
            else:
                skipped_keys.append(k)

        if len(loadable_state) == 0:
            raise RuntimeError("No matching parameters found when loading checkpoint.")

        # Calculate missing and unexpected keys
        model_keys = set(model_state_dict.keys())
        ckpt_keys = set(model_state.keys())
        loaded_keys = set(loadable_state.keys())

        missing_keys = sorted(list(model_keys - loaded_keys))
        unexpected_keys = sorted(list(ckpt_keys - model_keys))
        shape_mismatch_keys = sorted([k for k in skipped_keys if k in model_keys])

        # Print checkpoint loading statistics
        print(f"[Checkpoint Loading Statistics]")
        print(f"  Checkpoint keys: {len(ckpt_keys)}")
        print(f"  Model keys: {len(model_keys)}")
        print(f"  Successfully loaded: {len(loaded_keys)}")
        print(f"  Missing in checkpoint: {len(missing_keys)}")
        print(f"  Unexpected in checkpoint: {len(unexpected_keys)}")
        print(f"  Shape mismatch: {len(shape_mismatch_keys)}")

        if strict_check:
            # Check for critical DETR keys
            missing_detr = [k for k in missing_keys if k.startswith("detr.")]
            if missing_detr:
                print(f"[WARNING] Missing DETR keys ({len(missing_detr)}):")
                for k in missing_detr[:10]:
                    print(f"    - {k}")
                if len(missing_detr) > 10:
                    print(f"    ... and {len(missing_detr) - 10} more")

            # Critical keys that MUST be present
            critical_keys = [
                "detr.transformer.encoder.layers.0.self_attn.in_proj_weight",
                "detr.transformer.decoder.layers.0.self_attn.in_proj_weight",
                "detr.backbone.0.body.layer1.0.conv1.weight",
            ]
            missing_critical = [k for k in critical_keys if k in model_keys and k not in loaded_keys]
            if missing_critical:
                print(f"[CRITICAL ERROR] Missing critical keys:")
                for k in missing_critical:
                    print(f"    - {k}")
                raise RuntimeError(
                    f"Critical model keys are missing from checkpoint! "
                    f"This will cause random initialization and poor performance. "
                    f"Missing: {missing_critical}"
                )

            if shape_mismatch_keys:
                print(f"[WARNING] Shape mismatch keys:")
                for k in shape_mismatch_keys[:5]:
                    print(f"    - {k}: model={model_state_dict[k].shape}, ckpt={model_state[k].shape}")

        model.load_state_dict(loadable_state, strict=False)

    if optimizer is not None:
        opt_state = load_state.get("optimizer")
        if opt_state is not None:
            try:
                optimizer.load_state_dict(opt_state)
            except (ValueError, TypeError, KeyError):
                pass
    if scheduler is not None:
        sched_state = load_state.get("scheduler")
        if sched_state is not None:
            try:
                scheduler.load_state_dict(sched_state)
            except (ValueError, TypeError, KeyError):
                pass
    if states is not None:
        states.update(load_state["states"])
    return


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_model(model):
    return model.module if is_distributed() else model

# For previous version of MOTIP models:
def load_previous_checkpoint(model, path, states=None, optimizer=None, scheduler=None):
    assert states is None and optimizer is None and scheduler is None, \
        "The states, optimizer, and scheduler should be None for the previous version of MOTIP models."

    load_state = torch.load(path, map_location=lambda storage, loc: storage)
    model_state = load_state["model"]
    transfer_states = dict()

    if "bbox_embed.0.layers.0.weight" in model_state:
        load_detr_pretrain(model=model, pretrain_path=path, num_classes=None)
        return
    else:
        for k, v in model_state.items():
            if "detr" in k:
                transfer_states[k] = v
            elif "seq_decoder" in k:
                transfer_k = k
                transfer_k = transfer_k.replace("seq_decoder.", "")
                if "trajectory_feature_adapter" in transfer_k:
                    transfer_k = transfer_k.replace("trajectory_feature_adapter", "trajectory_modeling.adapter")
                    if "norm" in transfer_k:
                        transfer_k = transfer_k.replace("adapter.", "")
                elif "trajectory_augmentation" in transfer_k:
                    transfer_k = transfer_k.replace("trajectory_augmentation.trajectory_ffn", "trajectory_modeling.ffn")
                    if "ffn.norm" in transfer_k:
                        transfer_k = transfer_k.replace("ffn.norm", "ffn_norm")
                elif "related_temporal_embeds" in transfer_k:
                    transfer_k = transfer_k.replace("related_temporal_embeds", "rel_pos_embeds")
                elif "embed_to_word" in transfer_k:
                    for _ in range(0, 6):
                        _transfer_k = transfer_k
                        _transfer_k = _transfer_k.replace("embed_to_word", f"embed_to_word_layers.{_}")
                        if _transfer_k in transfer_states:
                            print(f"Key '{_transfer_k}' is already in the transfer states.")
                        transfer_states[_transfer_k] = v
                    continue
                elif "decoder_layers" in transfer_k:
                    transfer_k = transfer_k.replace("decoder_layers", "cross_attn_layers")
                elif ".norm_layers" in transfer_k:
                    transfer_k = transfer_k.replace("norm_layers", "cross_attn_norm_layers")
                elif "self_attn_layers" in transfer_k:
                    pass
                elif "self_norm_layers" in transfer_k:
                    transfer_k = transfer_k.replace("self_norm_layers", "self_attn_norm_layers")
                elif "ffn_layers" in transfer_k:
                    if "norm" in transfer_k:
                        transfer_k = transfer_k.replace("ffn_layers", "ffn_norm_layers")
                        transfer_k = transfer_k.replace("norm.", "")
                        pass
                else:
                    pass
                if transfer_k in transfer_states:
                    print(f"Key '{transfer_k}' is already in the transfer states.")
                transfer_states[transfer_k] = v
                pass
            else:
                pass
        model.load_state_dict(transfer_states)

    if optimizer is not None:
        optimizer.load_state_dict(load_state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(load_state["scheduler"])
    if states is not None:
        states.update(load_state["states"])
    return
