# Copyright (c) Ruopeng Gao. All Rights Reserved.
# About: Submit or evaluate the model.

import os
import time
import torch
import subprocess
from accelerate import Accelerator
from accelerate.state import PartialState
from torch.utils.data import DataLoader

from runtime_option import runtime_option
from utils.misc import yaml_to_dict
from configs.util import load_super_config, update_config
from log.logger import Logger
from data.joint_dataset import dataset_classes
from data.seq_dataset import SeqDataset
from models.runtime_tracker import RuntimeTracker
from log.log import Metrics
from models.motip import build as build_motip
from models.misc import load_checkpoint


def submit_and_evaluate(config: dict):
    # Init Accelerator at beginning:
    accelerator = Accelerator()
    state = PartialState()

    mode = config["INFERENCE_MODE"]
    assert mode in ["submit", "evaluate"], f"Mode {mode} is not supported."
    # Generate the output dir:
    assert "OUTPUTS_DIR" in config and config["OUTPUTS_DIR"] is not None, "OUTPUTS_DIR is not set."
    outputs_dir = config["OUTPUTS_DIR"]
    inference_group = config["INFERENCE_GROUP"]
    inference_dataset = config["INFERENCE_DATASET"]
    inference_split = config["INFERENCE_SPLIT"]
    inference_model = config["INFERENCE_MODEL"]
    _inference_model_name = os.path.split(inference_model)[-1][:-4]
    outputs_dir = os.path.join(
        outputs_dir, mode, inference_group, inference_dataset, inference_split, _inference_model_name
    )
    _is_outputs_dir_exist = os.path.exists(outputs_dir)
    accelerator.wait_for_everyone()
    os.makedirs(outputs_dir, exist_ok=True)

    # Init Logger, do not use wandb:
    logger = Logger(
        logdir=str(outputs_dir),
        use_wandb=False,
        config=config,
        # exp_owner=config["EXP_OWNER"],
        # exp_project=config["EXP_PROJECT"],
        # exp_group=config["EXP_GROUP"],
        # exp_name=config["EXP_NAME"],
    )
    # Log runtime config:
    logger.config(config=config)
    # Log other infos:
    logger.info(
        f"{mode.capitalize()} model: {inference_model}, inference dataset: {inference_dataset}, "
        f"inference split: {inference_split}, inference group: {inference_group}."
    )
    if _is_outputs_dir_exist:
        logger.warning(f"Outputs dir '{outputs_dir}' already exists, may overwrite the existing files.")
        # Only main process sleeps to give user a chance to cancel
        # Other processes wait at barrier
        if accelerator.is_main_process:
            time.sleep(5)   # wait for 5 seconds, give the user a chance to cancel.
        accelerator.wait_for_everyone()
    else:
        logger.info(f"Outputs dir '{outputs_dir}' created.")

    model, _ = build_motip(config=config)

    use_previous_checkpoint = config.get("USE_PREVIOUS_CHECKPOINT", False)
    if not use_previous_checkpoint:
        load_checkpoint(model, path=config["INFERENCE_MODEL"])
    else:
        from models.misc import load_previous_checkpoint
        load_previous_checkpoint(model, path=config["INFERENCE_MODEL"])

    model = accelerator.prepare(model)

    metrics = submit_and_evaluate_one_model(
        is_evaluate=config["INFERENCE_MODE"] == "evaluate",
        accelerator=accelerator,
        state=state,
        logger=logger,
        model=model,
        data_root=config["DATA_ROOT"],
        dataset=config["INFERENCE_DATASET"],
        data_split=config["INFERENCE_SPLIT"],
        outputs_dir=outputs_dir,
        image_max_shorter=config.get("INFERENCE_MAX_SHORTER", 800),
        image_max_longer=config["INFERENCE_MAX_LONGER"],    # the max shorter side of the image is set to 800 by default
        size_divisibility=config.get("SIZE_DIVISIBILITY", 0),
        use_sigmoid=config.get("USE_FOCAL_LOSS", False),
        assignment_protocol=config.get("ASSIGNMENT_PROTOCOL", "hungarian"),
        miss_tolerance=config["MISS_TOLERANCE"],
        det_thresh=config["DET_THRESH"],
        newborn_thresh=config["NEWBORN_THRESH"],
        id_thresh=config["ID_THRESH"],
        area_thresh=config.get("AREA_THRESH", 0),
        min_track_len=config.get("MIN_TRACK_LEN", 0),
        inference_only_detr=config["INFERENCE_ONLY_DETR"] if config["INFERENCE_ONLY_DETR"] is not None
        else config["ONLY_DETR"],
        dtype=config.get("INFERENCE_DTYPE", "FP32"),
    )

    if metrics is not None:
        metrics.sync()
        logger.metrics(
            log=f"Finish evaluation for model '{inference_model}', dataset '{inference_dataset}', "
                f"split '{inference_split}', group '{inference_group}': ",
            metrics=metrics,
            fmt="{global_average:.4f}",
        )
    return


def submit_and_evaluate_one_model(
        is_evaluate: bool,
        accelerator: Accelerator,
        state: PartialState,
        logger: Logger,
        model,
        data_root: str,
        dataset: str,
        data_split: str,
        # Outputs:
        outputs_dir: str,
        # Parameters with defaults:
        image_max_shorter: int = 800,
        image_max_longer: int = 1536,
        size_divisibility: int = 0,
        use_sigmoid: bool = False,
        assignment_protocol: str = "hungarian",
        miss_tolerance: int = 30,
        det_thresh: float = 0.5,
        newborn_thresh: float = 0.5,
        id_thresh: float = 0.1,
        area_thresh: int = 0,
        min_track_len: int = 0,
        inference_only_detr: bool = False,
        dtype: str = "FP32",
        sequence_include: list = None,  # For train/val split: only evaluate these sequences
        val_sequences: list = None,  # VAL_SEQUENCES for assertion
        assert_eval_only_val: bool = True,  # Enable hard assert for eval-only-val
        detector_filter: list = None,  # Filter by detector versions (e.g., ['FRCNN'])
):
    # Protect against string input for sequence lists (should be list)
    if isinstance(sequence_include, str):
        sequence_include = [sequence_include]
    if isinstance(val_sequences, str):
        val_sequences = [val_sequences]

    # Build the datasets:
    if dataset not in dataset_classes:
        # Fallback: reuse MOT17 dataset class for MOT20 if no dedicated class is registered.
        if dataset.lower() == "mot20" and "MOT17" in dataset_classes:
            dataset_classes[dataset] = dataset_classes["MOT17"]
        else:
            raise KeyError(f"Dataset {dataset} is not registered.")

    # Try to create dataset with sequence_include, fallback if not supported
    try:
        inference_dataset = dataset_classes[dataset](
            data_root=data_root,
            split=data_split,
            load_annotation=False,
            sequence_include=sequence_include,  # Pass sequence filter
            detector_filter=detector_filter,  # Pass detector filter
        )
    except TypeError as e:
        # Only fallback if the error is about unexpected keyword argument
        # Otherwise re-raise to catch real bugs (missing params, None values, etc.)
        if "unexpected keyword argument" in str(e) and ("sequence_include" in str(e) or "detector_filter" in str(e)):
            # Dataset class doesn't support sequence_include or detector_filter parameter
            inference_dataset = dataset_classes[dataset](
                data_root=data_root,
                split=data_split,
                load_annotation=False,
            )
            # Manual filtering will be applied later if sequence_include is provided
        else:
            # Real TypeError - re-raise for debugging
            raise

    # ============= GPT recommendation: Diagnostic stats collection =============
    # Track aggregated diagnostic statistics across all sequences
    global_diagnostic_stats = {
        "total_frames": 0,
        "raw_detections": 0,
        "after_score_filter": 0,
        "after_newborn_filter": 0,
        "unknown_count": 0,
        "newborn_count": 0,
    }
    # ===========================================================================

    # Set the dtype during inference:
    match dtype:
        case "FP32": dtype=torch.float32
        case "FP16": dtype=torch.float16
        case _: raise ValueError(f"Unknown dtype '{dtype}'.")
    # Filter out the sequences that will not be processed in this GPU (if we have multiple GPUs):
    _inference_sequence_names = list(inference_dataset.sequence_infos.keys())
    _inference_sequence_names.sort()
    # Log actual sequences being evaluated (GPT requirement)
    if sequence_include:
        logger.info(f"[EVAL] Evaluating on {len(_inference_sequence_names)} sequences: {_inference_sequence_names}")
    else:
        logger.info(f"[EVAL] Evaluating on all {len(_inference_sequence_names)} sequences in {dataset}/{data_split}")

    # ===================== Optional hard assert: Eval only VAL =====================
    if assert_eval_only_val and val_sequences:
        if isinstance(val_sequences, str):
            val_sequences = [val_sequences]

        def _match_base(seq_name: str, bases) -> bool:
            # Use startswith instead of 'in' to avoid false matches like "-1" matching "-10"
            # For MOT17-02-DPM, this will match base "MOT17-02"
            return any(seq_name.startswith(base) for base in bases)

        unexpected = [s for s in _inference_sequence_names if not _match_base(s, val_sequences)]
        if unexpected:
            logger.error(f"❌ Eval is NOT restricted to VAL_SEQUENCES={val_sequences}")
            logger.error(f"Unexpected eval sequences ({len(unexpected)}): {unexpected[:20]}")
            raise RuntimeError(f"Eval leaked non-VAL sequences: {unexpected[:20]}")
        else:
            logger.info(f"✅ Confirmed: Eval sequences are restricted to VAL_SEQUENCES={val_sequences}")
    # ================================================================================

    # Restrict dataset to the selected sequence(s)
    inference_dataset.sequence_infos = {
        k: v for k, v in inference_dataset.sequence_infos.items() if k in _inference_sequence_names
    }
    inference_dataset.image_paths = {
        k: v for k, v in inference_dataset.image_paths.items() if k in _inference_sequence_names
    }
    if len(_inference_sequence_names) == 0:
        raise ValueError(
            f"No sequences found for dataset '{dataset}' split '{data_split}' at '{inference_dataset.data_dir}'. "
            f"Please check data_root/split path and dataset name."
        )
    # If we have multiple GPUs, we need to filter out the sequences that will not be processed in this GPU:
    # However, there is a special case that the number of GPUs is larger than the number of sequences:
    if len(_inference_sequence_names) <= state.process_index:
        logger.info(
            log=f"Number of sequences is smaller than the number of processes, "
                f"a fake sequence will be processed on process {state.process_index}.",
            only_main=False,
        )
        inference_dataset.sequence_infos = {
            _inference_sequence_names[0]: inference_dataset.sequence_infos[_inference_sequence_names[0]]
        }
        inference_dataset.image_paths = {
            _inference_sequence_names[0]: inference_dataset.image_paths[_inference_sequence_names[0]]
        }
        is_fake = True
    else:
        for _ in range(len(_inference_sequence_names)):
            if _ % state.num_processes != state.process_index:
                inference_dataset.sequence_infos.pop(_inference_sequence_names[_])
                inference_dataset.image_paths.pop(_inference_sequence_names[_])
        is_fake = False

    # Process each sequence:
    for sequence_name in inference_dataset.sequence_infos.keys():
        # break
        sequence_dataset = SeqDataset(
            seq_info=inference_dataset.sequence_infos[sequence_name],
            image_paths=inference_dataset.image_paths[sequence_name],
            max_shorter=image_max_shorter,
            max_longer=image_max_longer,
            size_divisibility=size_divisibility,
            dtype=dtype,
        )
        sequence_loader = DataLoader(
            dataset=sequence_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda x: x[0],
        )
        # sequence_loader = accelerator.prepare(sequence_loader)
        sequence_wh = sequence_dataset.seq_hw()
        runtime_tracker = RuntimeTracker(
            model=model,
            sequence_hw=sequence_wh,
            use_sigmoid=use_sigmoid,
            assignment_protocol=assignment_protocol,
            miss_tolerance=miss_tolerance,
            det_thresh=det_thresh,
            newborn_thresh=newborn_thresh,
            id_thresh=id_thresh,
            area_thresh=area_thresh,
            only_detr=inference_only_detr,
            dtype=dtype,
        )
        # Enable diagnostics for this tracker (GPT recommendation)
        runtime_tracker.enable_diagnostics = True
        runtime_tracker.reset_diagnostics()
        if is_fake:
            logger.info(
                f"Fake submitting sequence {sequence_name} with {len(sequence_loader)} frames.",
                only_main=False
            )
        else:
            logger.info(f"Submitting sequence {sequence_name} with {len(sequence_loader)} frames.", only_main=False)
        sequence_results, sequence_fps = get_results_of_one_sequence(
            runtime_tracker=runtime_tracker,
            sequence_loader=sequence_loader,
            logger=logger,
        )
        # Write the results to the submit file:
        if dataset in ["DanceTrack", "SportsMOT", "MOT17", "MOT20", "PersonPath22_Inference", "BFT"]:
            tracker_name = "tracker_default"
            
            # Step 1: Count track lengths (for min_track_len filtering)
            track_counts = {}
            for t in range(len(sequence_results)):
                for obj_id in sequence_results[t]["id"]:
                    _oid = obj_id.item()
                    track_counts[_oid] = track_counts.get(_oid, 0) + 1
            
            # Step 2: Build results with filtering
            sequence_tracker_results = []
            id_remap = {}  # keep tracker IDs dense to avoid huge id ranges in eval
            for t in range(len(sequence_results)):
                for obj_id, score, category, bbox in zip(
                        sequence_results[t]["id"],
                        sequence_results[t]["score"],
                        sequence_results[t]["category"],
                        sequence_results[t]["bbox"],    # [x, y, w, h]
                ):
                    _oid = obj_id.item()
                    # Filter short tracks
                    if min_track_len > 0 and track_counts[_oid] <= min_track_len:
                        continue
                    mapped_id = id_remap.setdefault(_oid, len(id_remap) + 1)
                    sequence_tracker_results.append(
                        f"{t + 1},{mapped_id},"
                        f"{bbox[0].item():.2f},{bbox[1].item():.2f},{bbox[2].item():.2f},{bbox[3].item():.2f},"
                        f"1,-1,-1,-1\n"
                    )
            if not is_fake:
                # Save under tracker_name/data to match TrackEval default layout
                tracker_seq_dir = os.path.join(outputs_dir, "tracker", tracker_name, "data")
                os.makedirs(tracker_seq_dir, exist_ok=True)
                with open(os.path.join(tracker_seq_dir, f"{sequence_name}.txt"), "w") as submit_file:
                    submit_file.writelines(sequence_tracker_results)
                logger.success(f"Submit sequence {sequence_name} done, FPS: {sequence_fps:.2f}. "
                               f"Saved to {os.path.join(tracker_seq_dir, f'{sequence_name}.txt')}.",
                               only_main=False)

                # ============= Collect diagnostic stats for this sequence =============
                seq_stats = runtime_tracker.diagnostic_stats
                for key in global_diagnostic_stats:
                    global_diagnostic_stats[key] += seq_stats[key]
                # =======================================================================
            else:
                logger.success(f"Fake submit sequence {sequence_name} done, FPS: {sequence_fps:.2f}.", only_main=False)
            pass
        else:
            raise NotImplementedError(f"Do not support to submit the results for dataset '{dataset}'.")

    # Post-process for submitting and evaluation:
    accelerator.wait_for_everyone()
    if not is_evaluate:
        logger.success(
            log=f"Submit done. Saved to {os.path.join(outputs_dir, 'tracker')}",
            only_main=True,
        )
        return None
    else:
        if accelerator.is_main_process:
            logger.info(
                log=f"Start evaluation...",
                only_main=True,
            )
            # Prepare for evaluation:
            if dataset in ["DanceTrack", "SportsMOT", "MOT17", "MOT20", "BFT"]:
                gt_dir = os.path.join(data_root, dataset, data_split)
                tracker_dir = os.path.join(outputs_dir, "tracker")
            elif dataset in ["PersonPath22_Inference"]:
                gt_dir = os.path.join(data_root, dataset, "gts", "person_path_22-test")
                tracker_dir = os.path.join(outputs_dir, "tracker")
            else:
                raise NotImplementedError(f"Do not support to find the gt_dir for dataset '{dataset}'.")
            allowed_mot_splits = {"train", "val", "test"}
            if dataset in ["DanceTrack", "SportsMOT", "BFT"] or (dataset in ["MOT17", "MOT20"] and data_split in allowed_mot_splits):
                tracker_name = "tracker_default"
                # Build a seqmap containing ALL sequences (not just this process's subset)
                # to ensure correct evaluation in multi-GPU setup
                seqmap_path = os.path.join(outputs_dir, "seqmap_eval.txt")
                with open(seqmap_path, "w") as f:
                    f.write("name\n")  # TrackEval skips the first line as header
                    # Use _inference_sequence_names (complete list before multi-GPU split)
                    for name in _inference_sequence_names:
                        f.write(f"{name}\n")
                args = {
                    "--SPLIT_TO_EVAL": data_split,
                    "--METRICS": ["HOTA", "CLEAR", "Identity"],
                    "--GT_FOLDER": gt_dir,
                    "--SEQMAP_FILE": seqmap_path,
                    "--SKIP_SPLIT_FOL": "True",
                    "--TRACKERS_TO_EVAL": tracker_name,
                    "--TRACKER_SUB_FOLDER": "data",
                    "--USE_PARALLEL": "False",
                    "--NUM_PARALLEL_CORES": "8",
                    "--PLOT_CURVES": "False",
                    "--TRACKERS_FOLDER": tracker_dir,
                    "--BENCHMARK": dataset,
                }
                cmd = ["python", "TrackEval/scripts/run_mot_challenge.py"]
            elif dataset in ["PersonPath22_Inference"]:
                args = {
                    "--SPLIT_TO_EVAL": data_split,
                    "--METRICS": ["HOTA", "CLEAR", "Identity"],
                    "--GT_FOLDER": gt_dir,
                    "--USE_PARALLEL": "True",
                    "--NUM_PARALLEL_CORES": "8",
                    "--TRACKERS_FOLDER": tracker_dir,
                    "--BENCHMARK": "person_path_22",
                    "--SEQMAP_FILE": os.path.join(data_root, dataset, "gts", "seqmaps", "person_path_22-test.txt"),
                    "--SKIP_SPLIT_FOL": "True",
                    "--TRACKER_SUB_FOLDER": "",
                    "--TRACKERS_TO_EVAL": "",
                }
                cmd = ["python", "TrackEval/scripts/run_person_path_22.py"]
            else:
                raise NotImplementedError(
                    f"Do not support to eval the results for dataset '{dataset}' split '{data_split}'."
                )
            for k, v in args.items():
                cmd.append(k)
                if isinstance(v, list):
                    cmd += v
                else:
                    cmd.append(v)
            # Run the eval script:
            _ = subprocess.run(
                cmd,
            )
            # Check if the eval script is done:
            if _.returncode == 0:
                logger.success("Evaluation script is done.", only_main=True)
            else:
                raise RuntimeError("Evaluation script failed.")
        # Wait for all processes:
        accelerator.wait_for_everyone()
        # Get the metrics:
        eval_metrics_path = os.path.join(outputs_dir, "tracker", "tracker_default", "pedestrian_summary.txt")
        eval_metrics_dict = get_eval_metrics_dict(metric_path=eval_metrics_path)
        metrics = Metrics()
        metrics["HOTA"].update(eval_metrics_dict["HOTA"])
        metrics["DetA"].update(eval_metrics_dict["DetA"])
        metrics["AssA"].update(eval_metrics_dict["AssA"])
        metrics["DetPr"].update(eval_metrics_dict["DetPr"])
        metrics["DetRe"].update(eval_metrics_dict["DetRe"])
        metrics["AssPr"].update(eval_metrics_dict["AssPr"])
        metrics["AssRe"].update(eval_metrics_dict["AssRe"])
        metrics["MOTA"].update(eval_metrics_dict["MOTA"])
        metrics["IDF1"].update(eval_metrics_dict["IDF1"])
        logger.success(
            log=f"Get evaluation metrics from {eval_metrics_path}.",
            only_main=True,
        )

        # ============= GPT recommendation: Print diagnostic statistics =============
        # Print aggregated diagnostics to help diagnose detection vs gating issues
        # CRITICAL: ALL processes must call reduce() to avoid deadlock
        # Convert to tensors for reduction across GPUs
        stats_tensor = torch.tensor([
            global_diagnostic_stats["total_frames"],
            global_diagnostic_stats["raw_detections"],
            global_diagnostic_stats["after_score_filter"],
            global_diagnostic_stats["after_newborn_filter"],
            global_diagnostic_stats["unknown_count"],
            global_diagnostic_stats["newborn_count"],
        ], dtype=torch.float32, device=accelerator.device)

        # Reduce across all processes (collective op - ALL processes must participate)
        aggregated_stats = accelerator.reduce(stats_tensor, reduction="sum")

        # Only main process prints, and only if there are actual frames
        if accelerator.is_main_process:
            total_frames = int(aggregated_stats[0].item())
            if total_frames > 0:
                raw_detections = int(aggregated_stats[1].item())
                after_score = int(aggregated_stats[2].item())
                after_newborn = int(aggregated_stats[3].item())
                unknown_count = int(aggregated_stats[4].item())
                newborn_count = int(aggregated_stats[5].item())

                avg_raw = raw_detections / total_frames
                avg_after_score = after_score / total_frames
                avg_after_newborn = after_newborn / total_frames
                keep_ratio = (after_newborn / after_score * 100) if after_score > 0 else 0
                unknown_ratio = (unknown_count / after_score * 100) if after_score > 0 else 0

                logger.info(f"[DIAG][VAL] det_before={after_score} "
                           f"det_after={after_newborn} "
                           f"keep={keep_ratio:.1f}% unknown={unknown_ratio:.1f}% "
                           f"avg_det/frame={avg_after_newborn:.1f}")
        # ============================================================================

        return metrics


@torch.no_grad()
def get_results_of_one_sequence(
        logger: Logger,
        runtime_tracker: RuntimeTracker,
        sequence_loader: DataLoader,
):
    tracker_results = []
    # Removed strict assertion to support short sequences (<=10 frames)
    warmup_frames = min(10, len(sequence_loader) - 1) if len(sequence_loader) > 1 else 0
    begin_time = None

    try:
        device = next(runtime_tracker.model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    for t, (image, image_path) in enumerate(sequence_loader):
        if t == warmup_frames:
            begin_time = time.time()
        image.tensors = image.tensors.to(device)
        image.mask = image.mask.to(device)
        # image = nested_tensor_from_tensor_list(tensor_list=[image[0]])
        runtime_tracker.update(image=image)
        _results = runtime_tracker.get_track_results()
        tracker_results.append(_results)

    # Calculate FPS with fallback for short sequences
    if begin_time is not None and len(sequence_loader) > warmup_frames:
        fps = (len(sequence_loader) - warmup_frames) / (time.time() - begin_time)
    else:
        fps = 0.0  # Not enough frames for meaningful FPS calculation

    return tracker_results, fps


def get_eval_metrics_dict(metric_path: str):
    if not os.path.isfile(metric_path):
        raise FileNotFoundError(
            f"Evaluation metrics file not found: {metric_path}. "
            "Please check TrackEval logs for errors."
        )
    with open(metric_path) as f:
        metric_names = f.readline()[:-1].split(" ")
        metric_values = f.readline()[:-1].split(" ")
    metrics = {
        n: float(v) for n, v in zip(metric_names, metric_values)
    }
    return metrics


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Get runtime option:
    opt = runtime_option()
    cfg = yaml_to_dict(opt.config_path)

    # Loading super config:
    if opt.super_config_path is not None:  # the runtime option is priority
        cfg = load_super_config(cfg, opt.super_config_path)
    else:  # if not, use the default super config path in the config file
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])

    # Combine the config and runtime into config dict:
    cfg = update_config(config=cfg, option=opt)

    # Call the "train_engine" function:
    submit_and_evaluate(config=cfg)
