# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
import math
import torch
import einops
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.state import PartialState
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from collections import defaultdict
from torchvision.transforms import v2
from typing import Any, Generator, List

from models.motip import build as build_motip
from models.motip.id_criterion import build as build_id_criterion
from runtime_option import runtime_option
from utils.misc import yaml_to_dict, set_seed
from configs.util import load_super_config, update_config
from log.logger import Logger
from data import build_dataset
from data.naive_sampler import NaiveSampler
from data.util import collate_fn
from log.log import TPS, Metrics
from models.misc import load_detr_pretrain, save_checkpoint, load_checkpoint
from models.misc import get_model
from utils.nested_tensor import NestedTensor
from submit_and_evaluate import submit_and_evaluate_one_model

# Make TORCH_HOME portable across environments.
# If the user already set TORCH_HOME externally, respect it.
if "TORCH_HOME" not in os.environ:
    os.environ["TORCH_HOME"] = os.path.join(os.path.expanduser("~"), ".cache", "torch")

def train_engine(config: dict):
    # Init some settings:
    assert "EXP_NAME" in config and config["EXP_NAME"] is not None, "Please set the experiment name."
    outputs_dir = config["OUTPUTS_DIR"] if config["OUTPUTS_DIR"] is not None \
        else os.path.join("./outputs/", config["EXP_NAME"])

    # Init Accelerator at beginning:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    state = PartialState()
    # Disable checkpointing under DDP to avoid re-entrant backward issues.
    if accelerator.num_processes > 1:
        config["USE_DECODER_CHECKPOINT"] = False
        config["DETR_NUM_CHECKPOINT_FRAMES"] = 0
    # Also, we set the seed:
    set_seed(config["SEED"])
    # Set the sharing strategy (to avoid error: too many open files):
    torch.multiprocessing.set_sharing_strategy('file_system')   # if not, raise error: too many open files.

    # Init Logger:
    logger = Logger(
        logdir=os.path.join(outputs_dir, "train"),
        use_wandb=config["USE_WANDB"],
        config=config,
        exp_owner=config["EXP_OWNER"],
        exp_project=config["EXP_PROJECT"],
        exp_group=config["EXP_GROUP"],
        exp_name=config["EXP_NAME"],
    )
    logger.info(f"We init the logger at {logger.logdir}.")
    if config["USE_WANDB"] is False:
        logger.warning("The wandb is not used in this experiment.")
    logger.info(f"The distributed type is {state.distributed_type}.")
    logger.config(config=config)

    # ===================== Save effective config for reproducibility =====================
    # GPT recommendation: dump the final merged config to output directory
    # Only main process should save to avoid race conditions in multi-GPU setup
    import yaml
    import subprocess
    if accelerator.is_main_process:
        config_save_path = os.path.join(outputs_dir, "config_effective.yaml")
        os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
        with open(config_save_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info(f"Saved effective config to: {config_save_path}")

        # Save git commit hash if this is a git repo
        try:
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                 stderr=subprocess.DEVNULL).decode('ascii').strip()
            git_commit_path = os.path.join(outputs_dir, "git_commit.txt")
            with open(git_commit_path, 'w') as f:
                f.write(f"Git commit: {git_commit}\n")
            logger.info(f"Git commit: {git_commit}")
        except:
            logger.warning("Not a git repository or git not available.")

    # Wait for all processes to ensure config is saved before proceeding
    accelerator.wait_for_everyone()
    # ==================================================================================

    # Build training dataset:
    train_dataset = build_dataset(config=config, is_validation=False)
    logger.dataset(train_dataset)

    # Log train/val split info if configured
    val_sequences = config.get("VAL_SEQUENCES", None)
    if val_sequences is not None and len(val_sequences) > 0:
        logger.info(f"Train/Val Split enabled. VAL_SEQUENCES: {val_sequences}")
        logger.info(f"Training will exclude these sequences.")
        # Hard-print actual sequences loaded in training dataset (GPT requirement)
        train_loaded_seqs = train_dataset.get_loaded_sequences()
        for dataset_name in train_loaded_seqs:
            for split in train_loaded_seqs[dataset_name]:
                seqs = train_loaded_seqs[dataset_name][split]
                logger.info(f"[TRAIN] {dataset_name}/{split}: {len(seqs)} sequences loaded: {seqs}")

        # ===================== Hard assert: VAL leakage into TRAIN =====================
        if isinstance(val_sequences, str):
            val_sequences = [val_sequences]

        def _match_base(seq_name: str, bases) -> bool:
            # Use startswith instead of 'in' to avoid false matches like "-1" matching "-10"
            # For MOT17-02-DPM, this will match base "MOT17-02"
            return any(seq_name.startswith(base) for base in bases)

        overlaps = []
        for dataset_name, splits in train_loaded_seqs.items():
            for split, seqs in splits.items():
                for s in seqs:
                    if _match_base(s, val_sequences):
                        overlaps.append((dataset_name, split, s))

        if overlaps:
            logger.error(f"❌ VAL leakage detected! VAL_SEQUENCES={val_sequences}")
            logger.error(f"Overlaps ({len(overlaps)}), examples: {overlaps[:20]}")
            raise RuntimeError(f"VAL_SEQUENCES leaked into TRAIN dataset: {overlaps[:20]}")
        else:
            logger.info("✅ Confirmed: VAL_SEQUENCES are excluded from TRAIN dataset.")
        # ================================================================================
    else:
        logger.info("Train/Val Split not configured. Using full training set.")

    # Build training data sampler:
    if "DATASET_WEIGHTS" in config:
        data_weights = defaultdict(lambda: defaultdict())
        for _ in range(len(config["DATASET_WEIGHTS"])):
            data_weights[config["DATASETS"][_]][config["DATASET_SPLITS"][_]] = config["DATASET_WEIGHTS"][_]
        data_weights = dict(data_weights)
    else:
        data_weights = None
    train_sampler = NaiveSampler(
        data_source=train_dataset,
        sample_steps=config["SAMPLE_STEPS"],
        sample_lengths=config["SAMPLE_LENGTHS"],
        sample_intervals=config["SAMPLE_INTERVALS"],
        length_per_iteration=config["LENGTH_PER_ITERATION"],
        data_weights=data_weights,
        min_legal_ratio=config.get("MIN_LEGAL_RATIO", 1.0),
    )
    # Build training data loader:
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=config["BATCH_SIZE"],
        num_workers=config["NUM_WORKERS"],
        prefetch_factor=config["PREFETCH_FACTOR"] if config["NUM_WORKERS"] > 0 else None,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Init the training states:
    train_states = {
        "start_epoch": 0,
        "global_step": 0
    }

    # ============= GPT recommendation: Track best model on validation set =============
    best_val_metric = {
        "epoch": -1,
        "MOTA": -float('inf'),  # Track MOTA as primary metric
        "IDF1": -float('inf'),
        "HOTA": -float('inf'),
    }
    # ==================================================================================

    # Build MOTIP model:
    model, detr_criterion = build_motip(config=config)
    # Load the pre-trained DETR:
    load_detr_pretrain(
        model=model, pretrain_path=config["DETR_PRETRAIN"], num_classes=config["NUM_CLASSES"],
        default_class_idx=config["DETR_DEFAULT_CLASS_IDX"] if "DETR_DEFAULT_CLASS_IDX" in config else None,
    )
    logger.success(
        log=f"Load the pre-trained DETR from '{config['DETR_PRETRAIN']}'. "
    )
    # Build Loss Function:
    id_criterion = build_id_criterion(config=config)

    # Build Optimizer:
    if config["DETR_NUM_TRAIN_FRAMES"] == 0:
        for n, p in model.named_parameters():
            if "detr" in n:
                p.requires_grad = False     # only train the MOTIP part.
    param_groups = get_param_groups(model, config)
    optimizer = AdamW(
        params=param_groups,
        lr=config["LR"],
        weight_decay=config["WEIGHT_DECAY"],
    )
    scheduler = MultiStepLR(
        optimizer=optimizer,
        milestones=config["SCHEDULER_MILESTONES"],
        gamma=config["SCHEDULER_GAMMA"],
    )

    # Other infos:
    only_detr = config["ONLY_DETR"]

    # Resuming:
    if config["RESUME_MODEL"] is not None:
        load_checkpoint(
            model=model,
            path=config["RESUME_MODEL"],
            optimizer=optimizer if config["RESUME_OPTIMIZER"] else None,
            scheduler=scheduler if config["RESUME_SCHEDULER"] else None,
            states=train_states,
        )
        # Different processing on scheduler:
        if config["RESUME_SCHEDULER"]:
            scheduler.step()
        else:
            for _ in range(0, train_states["start_epoch"]):
                scheduler.step()
        logger.success(
            log=f"Resume the model from '{config['RESUME_MODEL']}', "
                f"optimizer={config['RESUME_OPTIMIZER']}, "
                f"scheduler={config['RESUME_SCHEDULER']}, "
                f"states={train_states}. "
                f"Start from epoch {train_states['start_epoch']}, step {train_states['global_step']}."
        )

    train_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, model, optimizer,
        # device_placement=[False]        # whether to place the data on the device
    )

    for epoch in range(train_states["start_epoch"], config["EPOCHS"]):
        logger.info(log=f"Start training epoch {epoch}.")
        epoch_start_timestamp = TPS.timestamp()
        # Prepare the sampler for the current epoch:
        train_sampler.prepare_for_epoch(epoch=epoch)
        # --- LFD: importance temperature schedule (softmax(logits / tau)) ---
        if config.get("USE_IMPORTANCE_TAU_SCHEDULE", True):
            tau_start = float(config.get("IMPORTANCE_TAU_START", 2.0))
            tau_end = float(config.get("IMPORTANCE_TAU_END", 1.0))
            denom = max(config["EPOCHS"] - 1, 1)
            prog = float(epoch) / float(denom)
            importance_tau = tau_start + (tau_end - tau_start) * prog
        else:
            importance_tau = float(config.get("IMPORTANCE_TAU", 1.0))

        def _set_importance_tau(m):
            if hasattr(m, "importance_tau"):
                try:
                    m.importance_tau.fill_(float(importance_tau))
                except Exception:
                    pass

        model.apply(_set_importance_tau)
        if accelerator.is_main_process:
            logger.info(log=f"[LFD] importance_tau={importance_tau:.4f}")
        # Train one epoch:
        train_metrics = train_one_epoch(
            accelerator=accelerator,
            logger=logger,
            states=train_states,
            epoch=epoch,
            dataloader=train_dataloader,
            model=model,
            detr_criterion=detr_criterion,
            id_criterion=id_criterion,
            optimizer=optimizer,
            only_detr=only_detr,
            lr_warmup_epochs=config["LR_WARMUP_EPOCHS"],
            lr_warmup_tgt_lr=config["LR"],
            detr_num_train_frames=config["DETR_NUM_TRAIN_FRAMES"],
            detr_num_checkpoint_frames=config["DETR_NUM_CHECKPOINT_FRAMES"],
            detr_criterion_batch_len=config.get("DETR_CRITERION_BATCH_LEN", 10),
            use_decoder_checkpoint=config["USE_DECODER_CHECKPOINT"],
            accumulate_steps=config["ACCUMULATE_STEPS"],
            separate_clip_norm=config.get("SEPARATE_CLIP_NORM", True),
            max_clip_norm=config.get("MAX_CLIP_NORM", 0.1),
            use_accelerate_clip_norm=config.get("USE_ACCELERATE_CLIP_NORM", True),
            freq_ortho_loss_weight=config.get("FREQ_ORTHO_LOSS_WEIGHT", 1.0),
            freq_consistency_loss_weight=config.get("FREQ_CONSISTENCY_LOSS_WEIGHT", 0.1),
            save_lfd_diagnostics=config.get("SAVE_LFD_DIAGNOSTICS", False),
            lfd_diag_interval=int(config.get("LFD_DIAG_INTERVAL", 500)),
            # For multi last checkpoints:
            outputs_dir=outputs_dir,
            is_last_epochs=(epoch == config["EPOCHS"] - 1),
            multi_last_checkpoints=config["MULTI_LAST_CHECKPOINTS"],
        )

        # Get learning rate:
        lr = optimizer.state_dict()["param_groups"][-1]["lr"]
        train_metrics["lr"].update(lr)
        train_metrics["lr"].sync()
        time_per_epoch = TPS.format(TPS.timestamp() - epoch_start_timestamp)
        logger.metrics(
            log=f"[Finish epoch: {epoch}] [Time: {time_per_epoch}] ",
            metrics=train_metrics,
            fmt="{global_average:.4f}",
            statistic="global_average",
            global_step=train_states["global_step"],
            prefix="epoch",
            x_axis_step=epoch,
            x_axis_name="epoch",
        )

        # Save checkpoint:
        if (epoch + 1) % config["SAVE_CHECKPOINT_PER_EPOCH"] == 0:
            save_checkpoint(
                model=model,
                path=os.path.join(outputs_dir, f"checkpoint_{epoch}.pth"),
                states=train_states,
                optimizer=optimizer,
                scheduler=scheduler,
                only_detr=only_detr,
            )
            if config["INFERENCE_DATASET"] is not None:
                assert config["INFERENCE_SPLIT"] is not None, f"Please set the INFERENCE_SPLIT for inference."
                # Get VAL_SEQUENCES for evaluation filtering
                val_sequences = config.get("VAL_SEQUENCES", None)
                if val_sequences is not None and len(val_sequences) > 0:
                    logger.info(f"Evaluating on VAL_SEQUENCES: {val_sequences}")

                # ============= GPT recommendation: Dual-threshold evaluation =============
                # Run two sets of thresholds to diagnose detection vs gating issues
                eval_configs = []

                # 1. Diagnostic thresholds (high recall, avoid gating artifacts)
                if config.get("EVAL_DIAGNOSTIC_THRESHOLDS", True):
                    eval_configs.append({
                        "name": "diagthr",
                        "det_thresh": 0.1,
                        "newborn_thresh": 0.0,
                        "id_thresh": 0.0,
                        "desc": "Diagnostic (high-recall)"
                    })

                # 2. Default thresholds (actual deployment settings)
                eval_configs.append({
                    "name": "default",
                    "det_thresh": config["DET_THRESH"],
                    "newborn_thresh": config["NEWBORN_THRESH"],
                    "id_thresh": config["ID_THRESH"],
                    "desc": "Default"
                })

                for eval_cfg in eval_configs:
                    logger.info(f"[Eval] Running with {eval_cfg['desc']} thresholds: "
                               f"DET={eval_cfg['det_thresh']}, NEWBORN={eval_cfg['newborn_thresh']}, ID={eval_cfg['id_thresh']}")

                    eval_metrics = submit_and_evaluate_one_model(
                        is_evaluate=True,
                        accelerator=accelerator,
                        state=state,
                        logger=logger,
                        model=model,
                        data_root=config["DATA_ROOT"],
                        dataset=config["INFERENCE_DATASET"],
                        data_split=config["INFERENCE_SPLIT"],
                        outputs_dir=os.path.join(outputs_dir, "train", "eval_during_train", f"epoch_{epoch}_{eval_cfg['name']}"),
                        image_max_longer=config["INFERENCE_MAX_LONGER"],
                        size_divisibility=config.get("SIZE_DIVISIBILITY", 0),
                        miss_tolerance=config["MISS_TOLERANCE"],
                        use_sigmoid=config["USE_FOCAL_LOSS"] if "USE_FOCAL_LOSS" in config else False,
                        assignment_protocol=config["ASSIGNMENT_PROTOCOL"] if "ASSIGNMENT_PROTOCOL" in config else "hungarian",
                        det_thresh=eval_cfg["det_thresh"],
                        newborn_thresh=eval_cfg["newborn_thresh"],
                        id_thresh=eval_cfg["id_thresh"],
                        area_thresh=config["AREA_THRESH"],
                        inference_only_detr=config["INFERENCE_ONLY_DETR"] if config["INFERENCE_ONLY_DETR"] is not None
                        else config["ONLY_DETR"],
                        sequence_include=val_sequences if val_sequences and len(val_sequences) > 0 else None,
                        val_sequences=val_sequences,
                        assert_eval_only_val=config.get("ASSERT_EVAL_ONLY_VAL", True),
                        detector_filter=config.get("DETECTOR_FILTER", None),
                    )
                    eval_metrics.sync()
                    logger.metrics(
                        log=f"[Eval epoch: {epoch}] [{eval_cfg['desc']}] ",
                        metrics=eval_metrics,
                    fmt="{global_average:.4f}",
                    statistic="global_average",
                    global_step=train_states["global_step"],
                    prefix="epoch",
                    x_axis_step=epoch,
                    x_axis_name="epoch",
                )

                    # ============= GPT recommendation: Save best model on validation set =============
                    # Only track best model for "default" thresholds (not diagnostic)
                    if eval_cfg['name'] == 'default':
                        current_mota = eval_metrics["MOTA"].global_average
                        if current_mota > best_val_metric["MOTA"]:
                            best_val_metric["MOTA"] = current_mota
                            best_val_metric["IDF1"] = eval_metrics["IDF1"].global_average
                            best_val_metric["HOTA"] = eval_metrics["HOTA"].global_average
                            best_val_metric["epoch"] = epoch

                            # Save best model
                            best_model_path = os.path.join(outputs_dir, "checkpoint_best.pth")
                            save_checkpoint(
                                model=model,
                                path=best_model_path,
                                states=train_states,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                only_detr=only_detr,
                            )
                            logger.success(f"[Best Model] Saved new best model at epoch {epoch}: "
                                          f"MOTA={current_mota:.4f}, IDF1={best_val_metric['IDF1']:.4f}, "
                                          f"HOTA={best_val_metric['HOTA']:.4f}")
                    # ==================================================================================

        logger.success(log=f"Finish training epoch {epoch}.")
        # Prepare for next step:
        scheduler.step()
    pass


def train_one_epoch(
        # Infos:
        accelerator: Accelerator,
        logger: Logger,
        states: dict,
        epoch: int,
        dataloader: DataLoader,
        model,
        detr_criterion,
        id_criterion,
        optimizer,
        only_detr,
        lr_warmup_epochs: int,
        lr_warmup_tgt_lr: float,
        detr_num_train_frames: int,
        detr_num_checkpoint_frames: int,
        detr_criterion_batch_len: int,
        use_decoder_checkpoint: bool,
        accumulate_steps: int = 1,
        separate_clip_norm: bool = True,
        max_clip_norm: float = 0.1,
        use_accelerate_clip_norm: bool = True,
        logging_interval: int = 20,
        freq_ortho_loss_weight: float = 1.0,
        freq_consistency_loss_weight: float = 0.1,
        # LFD diagnostics (optional evidence chain dumps)
        save_lfd_diagnostics: bool = False,
        lfd_diag_interval: int = 500,
        # For multi last checkpoints:
        outputs_dir: str = None,
        is_last_epochs: bool = False,
        multi_last_checkpoints: int = 0,
):
    current_last_checkpoint_idx = 0

    model.train()
    tps = TPS()     # time per step
    metrics = Metrics()
    optimizer.zero_grad(set_to_none=True)
    step_timestamp = tps.timestamp()
    device = accelerator.device
    _B = dataloader.batch_sampler.batch_size
    _num_gts_per_frame = 0

    # Prepare for gradient clip norm:
    model_without_ddp = get_model(model)
    detr_framework = model_without_ddp.detr_framework
    detr_params = []
    other_params = []
    for name, param in model_without_ddp.named_parameters():
        if "detr" in name:
            detr_params.append(param)
        else:
            other_params.append(param)

    for step, samples in enumerate(dataloader):
        images, annotations, metas = samples["images"], samples["annotations"], samples["metas"]
        # Normalize the images:
        # (Normally, it should be done in the dataloader, but here we do it in the training loop (on cuda).)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        images.tensors = v2.functional.to_dtype(images.tensors, dtype=torch.float32, scale=True)
        images.tensors = v2.functional.normalize(images.tensors, mean=mean, std=std)
        # A hack implementation to recover 0.0 in the masked regions:
        images.tensors = images.tensors * (~images.mask[:, :, None, ...]).to(torch.float32)
        images.tensors = images.tensors.contiguous()

        # Learning rate warmup:
        if epoch < lr_warmup_epochs:
            # Do warmup:
            lr_warmup(
                optimizer=optimizer,
                epoch=epoch, curr_iter=step, tgt_lr=lr_warmup_tgt_lr,
                warmup_epochs=lr_warmup_epochs, num_iter_per_epoch=len(dataloader),
            )

        _B, _T = len(annotations), len(annotations[0])
        detr_num_train_frames = min(detr_num_train_frames, _T)

        # Prepare the DETR targets from the annotations:
        detr_targets_flatten = annotations_to_flatten_detr_targets(annotations=annotations, device=device)

        # Select the training and no_grad frames:
        random_frame_idxs = torch.randperm(_T, device=device)   # use these random indices to select the frames.
        go_back_frame_idxs = torch.argsort(random_frame_idxs)   # use these indices to go back to the original order.
        go_back_frame_idxs_flatten = torch.cat([
            go_back_frame_idxs + _T * b for b in range(_B)
        ])      # only used for the DETR's criterion.
        # Split random_frame_idxs into training and no_grad frame indices:
        detr_train_frame_idxs = random_frame_idxs[:detr_num_train_frames]
        detr_no_grad_frame_idxs = random_frame_idxs[detr_num_train_frames:]

        detr_outputs_flatten_idxs = torch.arange(_B * _T, device=device)
        detr_outputs_flatten_idxs = einops.rearrange(detr_outputs_flatten_idxs, "(b t) -> b t", b=_B)
        detr_outputs_flatten_idxs = torch.cat([
            einops.rearrange(detr_outputs_flatten_idxs[:, :detr_num_train_frames], "b t -> (b t)"),
            einops.rearrange(detr_outputs_flatten_idxs[:, detr_num_train_frames:], "b t -> (b t)"),
        ], dim=0)
        detr_outputs_flatten_go_back_idxs = torch.argsort(detr_outputs_flatten_idxs)
        pass
        # Select the training and no_grad frames:
        detr_train_frames = nested_tensor_index_select(images, dim=1, index=detr_train_frame_idxs)
        detr_no_grad_frames = nested_tensor_index_select(images, dim=1, index=detr_no_grad_frame_idxs)

        # Prepare for the DETR forward function, turn the (B, T, ...) images to (B*T, ...) (or said flatten):
        detr_train_frames.tensors = einops.rearrange(detr_train_frames.tensors, "b t c h w -> (b t) c h w").contiguous()
        detr_train_frames.mask = einops.rearrange(detr_train_frames.mask, "b t h w -> (b t) h w").contiguous()
        detr_no_grad_frames.tensors = einops.rearrange(detr_no_grad_frames.tensors, "b t c h w -> (b t) c h w").contiguous()
        detr_no_grad_frames.mask = einops.rearrange(detr_no_grad_frames.mask, "b t h w -> (b t) h w").contiguous()

        detr_train_targets = None
        if detr_framework == "dino" and detr_num_train_frames > 0:
            detr_train_targets = []
            for b in range(_B):
                for idx in detr_train_frame_idxs.tolist():
                    detr_train_targets.append(detr_targets_flatten[b * _T + idx])

        # DETR forward:
        # 1. no_grad frames:
        if _T > detr_num_train_frames:      # do have no_grad frames (if not, skip this part)
            with torch.no_grad():
                if detr_num_checkpoint_frames == 0 or detr_num_checkpoint_frames * 4 >= len(detr_no_grad_frames):
                    # Directly forward the no_grad frames:
                    with accelerator.autocast():
                        detr_no_grad_outputs = model(frames=detr_no_grad_frames, part="detr")
                else:
                    # Split the no_grad frames into batched iterations (reduce the memory usage):
                    detr_no_grad_outputs_list = []
                    for batch_samples in batch_iterator(
                        detr_num_checkpoint_frames * 4,
                        detr_no_grad_frames,
                    ):
                        batch_frames = batch_samples[0]
                        with accelerator.autocast():
                            detr_no_grad_outputs_list.append(model(frames=batch_frames, part="detr"))
                    detr_no_grad_outputs = tensor_dict_cat_list(detr_no_grad_outputs_list, dim=0)
                    del detr_no_grad_outputs_list
        else:                               # no no_grad frames
            detr_no_grad_outputs = None

        # 2. training frames:
        if detr_num_train_frames > 0:
            if detr_num_checkpoint_frames == 0 or detr_num_checkpoint_frames >= len(detr_train_frames):
                # Directly forward the training frames:
                with accelerator.autocast():
                    if detr_framework == "dino":
                        detr_train_outputs = model(frames=detr_train_frames, part="detr", targets=detr_train_targets)
                    else:
                        detr_train_outputs = model(frames=detr_train_frames, part="detr")
            else:
                # Split the training frames into batched iterations (reduce the memory usage):
                detr_train_outputs_list = []
                if detr_framework == "dino" and detr_train_targets is not None:
                    for batch_frames, batch_targets in batch_iterator(
                        detr_num_checkpoint_frames,
                        detr_train_frames,
                        detr_train_targets,
                    ):
                        with accelerator.autocast():
                            detr_train_outputs_list.append(
                                model(frames=batch_frames, part="detr", use_checkpoint=True, targets=batch_targets)
                            )
                else:
                    for batch_samples in batch_iterator(
                        detr_num_checkpoint_frames,
                        detr_train_frames,
                    ):
                        batch_frames = batch_samples[0]
                        with accelerator.autocast():
                            detr_train_outputs_list.append(
                                model(frames=batch_frames, part="detr", use_checkpoint=True)
                            )
                detr_train_outputs = tensor_dict_cat_list(detr_train_outputs_list, dim=0)
                del detr_train_outputs_list
        else:
            detr_train_outputs = None

        # Combine training and no_grad outputs:
        detr_outputs = tensor_dict_cat(detr_train_outputs, detr_no_grad_outputs, dim=0)
        # Recover the order of the outputs:
        detr_outputs = tensor_dict_index_select(detr_outputs, index=detr_outputs_flatten_go_back_idxs, dim=0)
        detr_outputs = tensor_dict_index_select(detr_outputs, index=go_back_frame_idxs_flatten, dim=0)

        # DETR criterion:
        if detr_framework == "dino":
            detr_loss_dict, detr_indices = detr_criterion(
                outputs=detr_outputs,
                targets=detr_targets_flatten,
                return_indices=True,
            )
        else:
            detr_loss_dict, detr_indices = detr_criterion(
                outputs=detr_outputs, targets=detr_targets_flatten, batch_len=detr_criterion_batch_len
            )

        # Whether to only train the DETR, OR to train the MOTIP together:
        if not only_detr:
            _G, _, _N = annotations[0][0]["trajectory_id_labels"].shape
            # Need to prepare for MOTIP:
            seq_info = prepare_for_motip(
                detr_outputs=detr_outputs, annotations=annotations, detr_indices=detr_indices,
            )
            with accelerator.autocast():
                seq_info = model(seq_info=seq_info, part="trajectory_modeling")
                id_decoder_output = model(
                    seq_info=seq_info,
                    part="id_decoder",
                    use_decoder_checkpoint=use_decoder_checkpoint,
                )
            # Unpack possible freq-aware outputs
            if isinstance(id_decoder_output, tuple) and len(id_decoder_output) == 4:
                id_logits, id_gts, id_masks, freq_extra_losses = id_decoder_output
            else:
                id_logits, id_gts, id_masks = id_decoder_output
                freq_extra_losses = None
            # ID loss（支持 decoder 提供的分支/层权重）
            if freq_extra_losses is not None and isinstance(freq_extra_losses, dict) and 'loss_weights' in freq_extra_losses:
                weights = freq_extra_losses.get('loss_weights', None)
                if (
                    isinstance(weights, (list, tuple))
                    and len(weights) > 0
                    and id_logits is not None
                    and id_masks is not None
                    and id_logits.shape[0] % len(weights) == 0
                ):
                    k = len(weights)
                    chunk = id_logits.shape[0] // k
                    loss_sum = 0.0
                    weight_sum = 0.0
                    for i, w in enumerate(weights):
                        if w is None:
                            continue
                        w_val = float(w)
                        if w_val == 0.0:
                            continue
                        logits_i = id_logits[i * chunk:(i + 1) * chunk]
                        labels_i = id_gts[i * chunk:(i + 1) * chunk] if id_gts is not None else None
                        masks_i = id_masks[i * chunk:(i + 1) * chunk]
                        loss_i = id_criterion(id_logits=logits_i, id_labels=labels_i, id_masks=masks_i)
                        loss_sum = loss_sum + w_val * loss_i
                        weight_sum += w_val
                    if weight_sum > 0:
                        id_loss = loss_sum / weight_sum
                    else:
                        id_loss = id_criterion(id_logits=id_logits, id_labels=id_gts, id_masks=id_masks)
                else:
                    id_loss = id_criterion(id_logits=id_logits, id_labels=id_gts, id_masks=id_masks)
            else:
                id_loss = id_criterion(id_logits=id_logits, id_labels=id_gts, id_masks=id_masks)
            if id_gts is not None:
                _num_gts_per_frame = max(_num_gts_per_frame, id_gts.shape[-1])
            # print(f"Num of GTs per frame: {_num_gts_per_frame}")
            pass
        else:
            id_loss = None
            freq_extra_losses = None
            seq_info = {"freq_losses": {}}

        # Backward:
        with accelerator.autocast():
            detr_weight_dict = detr_criterion.weight_dict
            detr_loss = sum(
                detr_loss_dict[k] * detr_weight_dict[k] for k in detr_loss_dict.keys() if k in detr_weight_dict
            )
            freq_losses = seq_info.get("freq_losses", {}) if not only_detr else {}
            # Optional: dump LFD diagnostics tensors for plots (FFT response / overlap heatmap / importance stats)
            if (not only_detr) and save_lfd_diagnostics and (outputs_dir is not None) and accelerator.is_main_process:
                try:
                    if int(states.get("global_step", 0)) % max(int(lfd_diag_interval), 1) == 0:
                        diag_dir = os.path.join(outputs_dir, "lfd_diag")
                        os.makedirs(diag_dir, exist_ok=True)
                        freq_info = (seq_info.get("freq_info", {}) or {}).get("decomposition_info", {})
                        band_features = (freq_info.get("band_features", {}) or {})
                        diag = {
                            "epoch": int(epoch),
                            "global_step": int(states.get("global_step", 0)),
                        }
                        tb = seq_info.get("trajectory_boxes", None)
                        tm = seq_info.get("trajectory_masks", None)
                        if isinstance(tb, torch.Tensor):
                            diag["trajectory_boxes"] = tb.detach().to("cpu")
                            try:
                                B_, G_, T_, N_, _ = tb.shape
                                diag["B"], diag["G"], diag["T"], diag["N"] = int(B_), int(G_), int(T_), int(N_)
                            except Exception:
                                pass
                        if isinstance(tm, torch.Tensor):
                            diag["trajectory_masks"] = tm.detach().to("cpu")
                        for k in ["filters_detached", "center_freqs", "sigmas", "importance", "importance_logits", "importance_tau"]:
                            v = band_features.get(k, None)
                            if isinstance(v, torch.Tensor):
                                diag[k] = v.detach().to("cpu")
                        # Also record current losses (cpu scalar)
                        ol = freq_info.get("ortho_loss", None)
                        if isinstance(ol, torch.Tensor):
                            diag["ortho_loss"] = ol.detach().to("cpu")
                        torch.save(diag, os.path.join(diag_dir, f"e{epoch}_g{states.get('global_step',0)}.pt"))
                except Exception:
                    pass
            freq_ortho_loss = freq_losses.get("ortho_loss", 0.0)
            if not torch.is_tensor(freq_ortho_loss):
                freq_ortho_loss = torch.tensor(freq_ortho_loss, device=device, dtype=detr_loss.dtype)
            # 限制频率损失范围，防止梯度爆炸
            freq_ortho_loss = torch.clamp(freq_ortho_loss, min=0.0, max=10.0)

            freq_consistency_loss = 0.0
            if freq_extra_losses is not None and isinstance(freq_extra_losses, dict):
                freq_consistency_loss = freq_extra_losses.get("consistency_loss", 0.0)
            if not torch.is_tensor(freq_consistency_loss):
                freq_consistency_loss = torch.tensor(freq_consistency_loss, device=device, dtype=detr_loss.dtype)
            # 限制频率损失范围，防止梯度爆炸
            freq_consistency_loss = torch.clamp(freq_consistency_loss, min=0.0, max=10.0)

            loss = detr_loss \
                   + (id_loss if id_loss is not None else 0) * id_criterion.weight \
                   + freq_ortho_loss_weight * freq_ortho_loss \
                   + freq_consistency_loss_weight * freq_consistency_loss
            # Touch freq/id params to avoid DDP unused-parameter mismatch across ranks when a branch is skipped.
            # Skip this when decoder checkpointing is on to avoid double-ready errors in reentrant backward.
            if not use_decoder_checkpoint:
                force_use = torch.zeros([], device=loss.device, dtype=loss.dtype)
                for n, p in model_without_ddp.named_parameters():
                    if ("id_decoder" in n or "trajectory_modeling" in n) and p.requires_grad:
                        force_use = force_use + p.view(-1)[0] * 0
                loss = loss + force_use
            # Logging losses:
            metrics.update(name="loss", value=loss.item())
            metrics.update(name="detr_loss", value=detr_loss.item())
            if id_loss is not None:
                metrics.update(name="id_loss", value=id_loss.item())
            if not only_detr:
                metrics.update(name="freq_ortho_loss", value=freq_ortho_loss.item())
                metrics.update(name="freq_consistency_loss", value=freq_consistency_loss.item())
            for k, v in detr_loss_dict.items():
                metrics.update(name=k, value=v.item())
            loss /= accumulate_steps
            accelerator.backward(loss)  # use this line to replace loss.backward()
            if (step + 1) % accumulate_steps == 0:
                if use_accelerate_clip_norm:
                    if separate_clip_norm:
                        # Unscale once, then clip separately to avoid double-unscale errors.
                        accelerator.unscale_gradients()
                        detr_grad_norm = torch.nn.utils.clip_grad_norm_(detr_params, max_clip_norm)
                        other_grad_norm = torch.nn.utils.clip_grad_norm_(other_params, max_clip_norm)
                    else:
                        detr_grad_norm = other_grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=max_clip_norm)
                else:
                    if separate_clip_norm:
                        accelerator.unscale_gradients()
                        detr_grad_norm = torch.nn.utils.clip_grad_norm_(detr_params, max_clip_norm)
                        other_grad_norm = torch.nn.utils.clip_grad_norm_(other_params, max_clip_norm)
                    else:
                        accelerator.unscale_gradients()
                        detr_grad_norm = other_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip_norm)
                # Hack implementation to log grad_norm
                metrics.update(name="detr_grad_norm", value=detr_grad_norm.item())
                metrics.update(name="other_grad_norm", value=other_grad_norm.item())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # Logging:
        tps.update(tps=tps.timestamp() - step_timestamp)
        step_timestamp = tps.timestamp()
        # Logging:
        if step % logging_interval == 0:
            # logger.info(f"[Epoch: {epoch}] [{step}/{total_steps}] [tps: {tps.average:.2f}s]")
            # Get learning rate for current step:
            _lr = optimizer.state_dict()["param_groups"][-1]["lr"]
            # Get the GPU memory usage:
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
                _cuda_memory = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            else:
                _cuda_memory = 0.0
            _cuda_memory = torch.tensor([_cuda_memory], device=device)
            # _cuda_memory_reduce = accelerator.reduce(_cuda_memory, reduction="none")
            _gathered_cuda_memory = accelerator.gather(_cuda_memory)
            _max_cuda_memory = _gathered_cuda_memory.max().item()
            accelerator.wait_for_everyone()
            # Clear some values:
            metrics["lr"].clear()  # clear the learning rate value from last step
            metrics["max_cuda_mem(MB)"].clear()
            # Update them to the metrics:
            metrics.update(name="lr", value=_lr)
            metrics.update(name="max_cuda_mem(MB)", value=_max_cuda_memory)
            # Sync the metrics:
            metrics.sync()
            eta = tps.eta(total_steps=len(dataloader), current_steps=step)
            logger.metrics(
                log=f"[Epoch: {epoch}] [{step}/{len(dataloader)}] "
                    f"[tps: {tps.average:.2f}s] [eta: {TPS.format(eta)}] ",
                metrics=metrics,
                global_step=states["global_step"],
            )
        # For multi last checkpoints:
        if is_last_epochs and multi_last_checkpoints > 0:
            if (step + 1) == int(math.ceil((len(dataloader) / multi_last_checkpoints) * (current_last_checkpoint_idx + 1))):
                _dir = os.path.join(outputs_dir, "multi_last_checkpoints")
                os.makedirs(_dir, exist_ok=True)
                save_checkpoint(
                    model=model,
                    path=os.path.join(_dir, f"last_checkpoint_{current_last_checkpoint_idx}.pth"),
                    states=states,
                    optimizer=None,
                    scheduler=None,
                    only_detr=only_detr,
                )
                logger.info(
                    log=f"Save the last checkpoint {current_last_checkpoint_idx} at step {step}."
                )
                current_last_checkpoint_idx += 1
        # Update the counters:
        states["global_step"] += 1
    states["start_epoch"] += 1
    return metrics


def get_param_groups(model, config) -> list[dict]:
    def _match_names(_name, _key_names):
        for _k in _key_names:
            if _k in _name:
                return True
        return False

    # Keywords:
    backbone_names = config["LR_BACKBONE_NAMES"]
    linear_proj_names = config["LR_LINEAR_PROJ_NAMES"]
    dictionary_names = config["LR_DICTIONARY_NAMES"]
    freq_names = config.get("LR_FREQ_NAMES", [])
    pass
    # Param groups:
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if _match_names(n, backbone_names) and p.requires_grad],
            "lr_scale": config["LR_BACKBONE_SCALE"],
            "lr": config["LR"] * config["LR_BACKBONE_SCALE"]
        },
        {
            "params": [p for n, p in model.named_parameters() if _match_names(n, linear_proj_names) and p.requires_grad],
            "lr_scale": config["LR_LINEAR_PROJ_SCALE"],
            "lr": config["LR"] * config["LR_LINEAR_PROJ_SCALE"]
        },
        {
            "params": [p for n, p in model.named_parameters() if _match_names(n, dictionary_names) and p.requires_grad],
            "lr_scale": config["LR_DICTIONARY_SCALE"],
            "lr": config["LR"] * config["LR_DICTIONARY_SCALE"]
        },
        {
            "params": [p for n, p in model.named_parameters() if _match_names(n, freq_names) and p.requires_grad],
            "lr_scale": config.get("LR_FREQ_SCALE", 1.0),
            "lr": config["LR"] * config.get("LR_FREQ_SCALE", 1.0)
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if not _match_names(n, backbone_names)
                       and not _match_names(n, linear_proj_names)
                       and not _match_names(n, dictionary_names)
                       and not _match_names(n, freq_names)
                       and p.requires_grad],
        }
    ]
    return param_groups


def lr_warmup(optimizer, epoch: int, curr_iter: int, tgt_lr: float, warmup_epochs: int, num_iter_per_epoch: int):
    # min_lr = 1e-8
    total_warmup_iters = warmup_epochs * num_iter_per_epoch
    current_lr_ratio = (epoch * num_iter_per_epoch + curr_iter + 1) / total_warmup_iters
    current_lr = tgt_lr * current_lr_ratio
    for param_grop in optimizer.param_groups:
        if "lr_scale" in param_grop:
            param_grop["lr"] = current_lr * param_grop["lr_scale"]
        else:
            param_grop["lr"] = current_lr
        pass
    return


def annotations_to_flatten_detr_targets(annotations: list, device):
    """
    Args:
        annotations: annotations from the dataloader.
        device: move the targets to the device.

    Returns:
        A list of targets for the DETR model supervision, len=(B*T).
    """
    targets = []
    for annotation in annotations:      # scan by batch
        for ann in annotation:          # scan by frame
            targets.append(
                {
                    "boxes": ann["bbox"].to(device),
                    "labels": ann["category"].to(device),
                }
            )
    return targets


def nested_tensor_index_select(nested_tensor: NestedTensor, dim: int, index: torch.Tensor):
    tensors, mask = nested_tensor.decompose()
    _device = tensors.device
    index = index.to(_device)
    selected_tensors = torch.index_select(input=tensors, dim=dim, index=index).contiguous()
    selected_mask = torch.index_select(input=mask, dim=dim, index=index).contiguous()
    return NestedTensor(tensors=selected_tensors, mask=selected_mask)


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size: (b + 1) * batch_size] for arg in args]


def tensor_dict_cat(tensor_dict1, tensor_dict2, dim=0):
    if tensor_dict1 is None or tensor_dict2 is None:
        assert tensor_dict1 is not None or tensor_dict2 is not None, "One of the tensor dict should be not None."
        return tensor_dict1 if tensor_dict2 is None else tensor_dict2
    else:
        res_tensor_dict = defaultdict()
        for k in tensor_dict1.keys():
            if isinstance(tensor_dict1[k], torch.Tensor):
                res_tensor_dict[k] = torch.cat([tensor_dict1[k], tensor_dict2[k]], dim=dim)
            elif isinstance(tensor_dict1[k], dict):
                res_tensor_dict[k] = tensor_dict_cat(tensor_dict1[k], tensor_dict2[k], dim=dim)
            elif isinstance(tensor_dict1[k], list):
                assert len(tensor_dict1[k]) == len(tensor_dict2[k]), "The list should have the same length."
                res_tensor_dict[k] = [
                    tensor_dict_cat(tensor_dict1[k][_], tensor_dict2[k][_], dim=dim)
                    for _ in range(len(tensor_dict1[k]))
                ]
            else:
                # Keep the value from the first dict for unsupported types.
                res_tensor_dict[k] = tensor_dict1[k]
        return dict(res_tensor_dict)


def tensor_dict_cat_list(tensor_dicts, dim=0):
    """Concatenate a list of (nested) tensor dicts once, avoiding O(n^2) cat in loops."""
    if tensor_dicts is None or len(tensor_dicts) == 0:
        return None
    res_tensor_dict = defaultdict()
    keys = tensor_dicts[0].keys()
    for k in keys:
        v0 = tensor_dicts[0][k]
        if isinstance(v0, torch.Tensor):
            res_tensor_dict[k] = torch.cat([td[k] for td in tensor_dicts], dim=dim)
        elif isinstance(v0, dict):
            res_tensor_dict[k] = tensor_dict_cat_list([td[k] for td in tensor_dicts], dim=dim)
        elif isinstance(v0, list):
            assert all(len(td[k]) == len(v0) for td in tensor_dicts), "The list should have the same length."
            res_tensor_dict[k] = [
                tensor_dict_cat_list([td[k][i] for td in tensor_dicts], dim=dim)
                for i in range(len(v0))
            ]
        else:
            res_tensor_dict[k] = v0
    return dict(res_tensor_dict)


def tensor_dict_index_select(tensor_dict, index, dim=0):
    res_tensor_dict = defaultdict()
    for k in tensor_dict.keys():
        if isinstance(tensor_dict[k], torch.Tensor):
            res_tensor_dict[k] = torch.index_select(tensor_dict[k], index=index, dim=dim).contiguous()
        elif isinstance(tensor_dict[k], dict):
            res_tensor_dict[k] = tensor_dict_index_select(tensor_dict[k], index=index, dim=dim)
        elif isinstance(tensor_dict[k], list):
            res_tensor_dict[k] = [
                tensor_dict_index_select(tensor_dict[k][_], index=index, dim=dim)
                for _ in range(len(tensor_dict[k]))
            ]
        else:
            res_tensor_dict[k] = tensor_dict[k]
    return dict(res_tensor_dict)


def prepare_for_motip(detr_outputs, annotations, detr_indices):
    _B, _T = len(annotations), len(annotations[0])
    _G, _, _N = annotations[0][0]["trajectory_id_labels"].shape
    _device = detr_outputs["pred_logits"].device
    _feature_dim = detr_outputs["outputs"].shape[-1]
    _feature_dtype = detr_outputs["outputs"].dtype
    # Init corresponding variables:
    trajectory_id_labels = - torch.ones((_B, _G, _T, _N), dtype=torch.int64, device=_device)
    trajectory_times = - torch.ones((_B, _G, _T, _N), dtype=torch.int64, device=_device)
    trajectory_masks = torch.ones((_B, _G, _T, _N), dtype=torch.bool, device=_device)
    trajectory_boxes = torch.zeros((_B, _G, _T, _N, 4), dtype=torch.float32, device=_device)
    trajectory_features = torch.zeros((_B, _G, _T, _N, _feature_dim), dtype=_feature_dtype, device=_device)
    unknown_id_labels = - torch.ones((_B, _G, _T, _N), dtype=torch.int64, device=_device)
    unknown_times = - torch.ones((_B, _G, _T, _N), dtype=torch.int64, device=_device)
    unknown_masks = torch.ones((_B, _G, _T, _N), dtype=torch.bool, device=_device)
    unknown_boxes = torch.zeros((_B, _G, _T, _N, 4), dtype=torch.float32, device=_device)
    unknown_features = torch.zeros((_B, _G, _T, _N, _feature_dim), dtype=_feature_dtype, device=_device)
    for b in range(_B):
        for t in range(_T):
            flatten_idx = b * _T + t
            go_back_detr_idxs = torch.argsort(detr_indices[flatten_idx][1])
            detr_output_embeds = detr_outputs["outputs"][flatten_idx][detr_indices[flatten_idx][0][go_back_detr_idxs]]
            detr_boxes = detr_outputs["pred_boxes"][flatten_idx][detr_indices[flatten_idx][0][go_back_detr_idxs]]
            _num_matched = detr_output_embeds.shape[0]  # DETR匹配的目标数量
            for group in range(_G):
                _curr_traj_ann_idxs = annotations[b][t]["trajectory_ann_idxs"][group, 0, :]
                _curr_unk_ann_idxs = annotations[b][t]["unknown_ann_idxs"][group, 0, :]
                _curr_traj_masks = annotations[b][t]["trajectory_id_masks"][group, 0, :]
                _curr_unk_masks = annotations[b][t]["unknown_id_masks"][group, 0, :]
                # Fill the fields:
                trajectory_id_labels[b, group, t] = annotations[b][t]["trajectory_id_labels"][group, 0, :]
                unknown_id_labels[b, group, t] = annotations[b][t]["unknown_id_labels"][group, 0, :]
                trajectory_times[b, group, t] = annotations[b][t]["trajectory_times"][group, 0, :]
                unknown_times[b, group, t] = annotations[b][t]["unknown_times"][group, 0, :]
                trajectory_masks[b, group, t] = _curr_traj_masks
                unknown_masks[b, group, t] = _curr_unk_masks
                # Bounds check: 只处理索引在有效范围内的目标
                _valid_traj = (~_curr_traj_masks) & (_curr_traj_ann_idxs >= 0)
                _traj_idxs = _curr_traj_ann_idxs[_valid_traj]
                _traj_in_bounds = _traj_idxs < _num_matched
                if _traj_in_bounds.any():
                    _valid_traj_positions = _valid_traj.nonzero(as_tuple=True)[0][_traj_in_bounds]
                    trajectory_features[b, group, t, _valid_traj_positions] = detr_output_embeds[_traj_idxs[_traj_in_bounds]]
                    trajectory_boxes[b, group, t, _valid_traj_positions] = detr_boxes[_traj_idxs[_traj_in_bounds]]
                _valid_unk = (~_curr_unk_masks) & (_curr_unk_ann_idxs >= 0)
                _unk_idxs = _curr_unk_ann_idxs[_valid_unk]
                _unk_in_bounds = _unk_idxs < _num_matched
                if _unk_in_bounds.any():
                    _valid_unk_positions = _valid_unk.nonzero(as_tuple=True)[0][_unk_in_bounds]
                    unknown_features[b, group, t, _valid_unk_positions] = detr_output_embeds[_unk_idxs[_unk_in_bounds]]
                    unknown_boxes[b, group, t, _valid_unk_positions] = detr_boxes[_unk_idxs[_unk_in_bounds]]
                pass
            pass
    return {
        "trajectory_id_labels": trajectory_id_labels,
        "trajectory_times": trajectory_times,
        "trajectory_masks": trajectory_masks,
        "trajectory_boxes": trajectory_boxes,
        "trajectory_features": trajectory_features,
        "unknown_id_labels": unknown_id_labels,
        "unknown_times": unknown_times,
        "unknown_masks": unknown_masks,
        "unknown_boxes": unknown_boxes,
        "unknown_features": unknown_features,
    }


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # from issue: https://github.com/pytorch/pytorch/issues/11201
    # import torch.multiprocessing
    # torch.multiprocessing.set_sharing_strategy('file_system')

    # Get runtime option:
    opt = runtime_option()
    cfg = yaml_to_dict(opt.config_path)

    # Loading super config:
    if opt.super_config_path is not None:   # the runtime option is priority
        cfg = load_super_config(cfg, opt.super_config_path)
    else:                                   # if not, use the default super config path in the config file
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])

    # Combine the config and runtime into config dict:
    cfg = update_config(config=cfg, option=opt)

    # Call the "train_engine" function:
    train_engine(config=cfg)
