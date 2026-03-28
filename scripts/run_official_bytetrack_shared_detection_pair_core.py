#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.backends.cudnn as cudnn
from loguru import logger


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
BYTE_ROOT = REPO_ROOT / "third_party" / "ByteTrack"

if str(BYTE_ROOT) not in sys.path:
    sys.path.insert(0, str(BYTE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yolox.exp import get_exp  # noqa: E402
from yolox.tracker.basetrack import BaseTrack  # noqa: E402
from yolox.tracker.byte_tracker import BYTETracker  # noqa: E402
from yolox.tracker.byte_tracker_local_conflict import ByteTrackerLocalConflict  # noqa: E402
from yolox.utils import fuse_model, get_model_info, postprocess  # noqa: E402


def write_results(filename: Path, results: Sequence[tuple[int, list[list[float]], list[int], list[float]]]) -> None:
    save_format = "{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n"
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", encoding="utf-8") as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if int(track_id) < 0:
                    continue
                x1, y1, w, h = tlwh
                f.write(
                    save_format.format(
                        frame=int(frame_id),
                        id=int(track_id),
                        x1=round(float(x1), 1),
                        y1=round(float(y1), 1),
                        w=round(float(w), 1),
                        h=round(float(h), 1),
                        s=round(float(score), 2),
                    )
                )
    logger.info("save results to {}", str(filename))


def write_tracker_diagnostics(filename: Path, payload: Dict[str, object]) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("save tracker diagnostics to {}", str(filename))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official ByteTrack host-only and plugin trackers on shared detector outputs."
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--exp-file", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", default="/gemini/code/datasets")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--nms", type=float, default=0.7)
    parser.add_argument("--track-thresh", type=float, default=0.6)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.9)
    parser.add_argument("--min-box-area", type=float, default=100.0)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--fuse", action="store_true", default=False)
    parser.add_argument("--host-variant", default="official_bytetrack")
    parser.add_argument(
        "--plugin-mode",
        choices=["learned_commit", "posthost_one_edit_oracle", "posthost_one_edit_hierarchical"],
        default="learned_commit",
    )
    parser.add_argument("--graph-ckpt", default="")
    parser.add_argument("--graph-topk", type=int, default=8)
    parser.add_argument("--graph-min-detections", type=int, default=2)
    parser.add_argument("--graph-min-committed-matches", type=int, default=1)
    parser.add_argument("--graph-max-detections", type=int, default=8)
    parser.add_argument("--graph-max-tracks", type=int, default=32)
    parser.add_argument("--graph-cluster-gate-thresh", type=float, default=0.5)
    parser.add_argument("--graph-cluster-gate-temp", type=float, default=1.0)
    parser.add_argument("--graph-cluster-gate-bias", type=float, default=0.0)
    parser.add_argument("--graph-max-commits-per-cluster", type=int, default=1)
    parser.add_argument("--graph-replacement-budget-ratio", type=float, default=0.05)
    parser.add_argument("--graph-max-replaced-clusters", type=int, default=0)
    parser.add_argument("--graph-min-commit-margin", type=float, default=0.05)
    parser.add_argument("--posthost-oracle-data-root", default="")
    parser.add_argument("--posthost-oracle-min-iou", type=float, default=0.5)
    parser.add_argument("--posthost-hierarchical-keep-thresh", type=float, default=0.5)
    parser.add_argument("--posthost-hierarchical-swap-thresh", type=float, default=0.5)
    parser.add_argument("--posthost-hierarchical-candidate-min-refined-score", type=float, default=0.10)
    parser.add_argument("--posthost-hierarchical-host-summary-prior-alpha", type=float, default=0.0)
    return parser.parse_args()


def apply_video_profile(args: argparse.Namespace, *, video_name: str, ori_thresh: float, ori_buffer: int) -> None:
    if video_name in {"MOT17-05-FRCNN", "MOT17-06-FRCNN"}:
        args.track_buffer = 14
    elif video_name in {"MOT17-13-FRCNN", "MOT17-14-FRCNN"}:
        args.track_buffer = 25
    else:
        args.track_buffer = int(ori_buffer)

    if video_name in {"MOT17-01-FRCNN", "MOT17-06-FRCNN"}:
        args.track_thresh = 0.65
    elif video_name == "MOT17-12-FRCNN":
        args.track_thresh = 0.7
    elif video_name == "MOT17-14-FRCNN":
        args.track_thresh = 0.67
    elif video_name in {"MOT20-06", "MOT20-08"}:
        args.track_thresh = 0.3
    else:
        args.track_thresh = float(ori_thresh)


def build_tracker_args(args: argparse.Namespace, *, plugin: bool) -> argparse.Namespace:
    tracker_args = copy.deepcopy(args)
    tracker_args.mot20 = False
    if plugin:
        tracker_args.use_local_conflict = str(args.plugin_mode) == "learned_commit"
        tracker_args.use_posthost_oracle_edit = str(args.plugin_mode) == "posthost_one_edit_oracle"
        tracker_args.use_posthost_hierarchical_edit = str(args.plugin_mode) == "posthost_one_edit_hierarchical"
        tracker_args.local_conflict_checkpoint = (
            str(Path(args.graph_ckpt).resolve())
            if str(args.plugin_mode) in {"learned_commit", "posthost_one_edit_hierarchical"}
            else ""
        )
        tracker_args.local_conflict_topk = int(args.graph_topk)
        tracker_args.local_conflict_min_detections = int(args.graph_min_detections)
        tracker_args.local_conflict_min_committed_matches = int(args.graph_min_committed_matches)
        tracker_args.local_conflict_max_detections = int(args.graph_max_detections)
        tracker_args.local_conflict_max_tracks = int(args.graph_max_tracks)
        tracker_args.local_conflict_cluster_gate_thresh = float(args.graph_cluster_gate_thresh)
        tracker_args.local_conflict_cluster_gate_temp = float(args.graph_cluster_gate_temp)
        tracker_args.local_conflict_cluster_gate_bias = float(args.graph_cluster_gate_bias)
        tracker_args.local_conflict_max_commits_per_cluster = int(args.graph_max_commits_per_cluster)
        tracker_args.local_conflict_replacement_budget_ratio = float(args.graph_replacement_budget_ratio)
        tracker_args.local_conflict_max_replaced_clusters = int(args.graph_max_replaced_clusters)
        tracker_args.local_conflict_min_commit_margin = float(args.graph_min_commit_margin)
        tracker_args.local_conflict_host_variant = str(args.host_variant)
        tracker_args.local_conflict_dump_dir = ""
        tracker_args.local_conflict_dump_topk = int(args.graph_topk)
        tracker_args.local_conflict_dump_min_score = 0.0
        tracker_args.posthost_oracle_data_root = str(args.posthost_oracle_data_root or args.data_root)
        tracker_args.posthost_oracle_min_iou = float(args.posthost_oracle_min_iou)
        tracker_args.posthost_hierarchical_keep_thresh = float(args.posthost_hierarchical_keep_thresh)
        tracker_args.posthost_hierarchical_swap_thresh = float(args.posthost_hierarchical_swap_thresh)
        tracker_args.posthost_hierarchical_candidate_min_refined_score = float(
            args.posthost_hierarchical_candidate_min_refined_score
        )
        tracker_args.posthost_hierarchical_host_summary_prior_alpha = float(
            args.posthost_hierarchical_host_summary_prior_alpha
        )
    else:
        tracker_args.use_local_conflict = False
        tracker_args.use_posthost_oracle_edit = False
        tracker_args.use_posthost_hierarchical_edit = False
        tracker_args.local_conflict_checkpoint = ""
        tracker_args.local_conflict_topk = int(args.graph_topk)
        tracker_args.local_conflict_min_detections = int(args.graph_min_detections)
        tracker_args.local_conflict_min_committed_matches = int(args.graph_min_committed_matches)
        tracker_args.local_conflict_max_detections = int(args.graph_max_detections)
        tracker_args.local_conflict_max_tracks = int(args.graph_max_tracks)
        tracker_args.local_conflict_cluster_gate_thresh = float(args.graph_cluster_gate_thresh)
        tracker_args.local_conflict_cluster_gate_temp = float(args.graph_cluster_gate_temp)
        tracker_args.local_conflict_cluster_gate_bias = float(args.graph_cluster_gate_bias)
        tracker_args.local_conflict_max_commits_per_cluster = int(args.graph_max_commits_per_cluster)
        tracker_args.local_conflict_replacement_budget_ratio = float(args.graph_replacement_budget_ratio)
        tracker_args.local_conflict_max_replaced_clusters = int(args.graph_max_replaced_clusters)
        tracker_args.local_conflict_min_commit_margin = float(args.graph_min_commit_margin)
        tracker_args.local_conflict_host_variant = str(args.host_variant)
        tracker_args.local_conflict_dump_dir = ""
        tracker_args.local_conflict_dump_topk = int(args.graph_topk)
        tracker_args.local_conflict_dump_min_score = 0.0
        tracker_args.posthost_oracle_data_root = ""
        tracker_args.posthost_oracle_min_iou = float(args.posthost_oracle_min_iou)
        tracker_args.posthost_hierarchical_keep_thresh = float(args.posthost_hierarchical_keep_thresh)
        tracker_args.posthost_hierarchical_swap_thresh = float(args.posthost_hierarchical_swap_thresh)
        tracker_args.posthost_hierarchical_candidate_min_refined_score = float(
            args.posthost_hierarchical_candidate_min_refined_score
        )
        tracker_args.posthost_hierarchical_host_summary_prior_alpha = float(
            args.posthost_hierarchical_host_summary_prior_alpha
        )
    return tracker_args


def filter_online_targets(online_targets, *, min_box_area: float):
    online_tlwhs = []
    online_ids = []
    online_scores = []
    for track in online_targets:
        tlwh = track.tlwh
        tid = track.track_id
        vertical = tlwh[2] / tlwh[3] > 1.6
        if tlwh[2] * tlwh[3] > min_box_area and not vertical:
            online_tlwhs.append([float(x) for x in tlwh])
            online_ids.append(int(tid))
            online_scores.append(float(track.score))
    return online_tlwhs, online_ids, online_scores


def flush_sequence(
    *,
    sequence_name: str,
    host_results: list[tuple[int, list[list[float]], list[int], list[float]]],
    plugin_results: list[tuple[int, list[list[float]], list[int], list[float]]],
    out_dir: Path,
    plugin_tracker: ByteTrackerLocalConflict | None,
) -> None:
    host_path = out_dir / "00_host_only" / "track_results" / f"{sequence_name}.txt"
    plugin_path = out_dir / "01_host_plus_plugin" / "track_results" / f"{sequence_name}.txt"
    write_results(host_path, host_results)
    write_results(plugin_path, plugin_results)
    if plugin_tracker is not None:
        diagnostics = plugin_tracker.get_local_conflict_diagnostics()
        write_tracker_diagnostics(
            out_dir / "01_host_plus_plugin" / "diagnostics" / f"{sequence_name}.json",
            diagnostics,
        )
        if hasattr(plugin_tracker, "close_local_conflict_dump"):
            plugin_tracker.close_local_conflict_dump()


def main() -> int:
    args = parse_args()
    if int(args.devices) != 1:
        raise RuntimeError("shared-detection pair core currently supports only --devices 1")
    if int(args.batch_size) != 1:
        raise RuntimeError("shared-detection pair core currently supports only --batch-size 1")
    if not Path(args.exp_file).is_file():
        raise FileNotFoundError(f"Missing exp file: {args.exp_file}")
    if not Path(args.ckpt).is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt}")
    if str(args.plugin_mode) in {"learned_commit", "posthost_one_edit_hierarchical"} and not Path(args.graph_ckpt).is_file():
        raise FileNotFoundError(f"Missing graph checkpoint: {args.graph_ckpt}")

    out_dir = Path(args.out_dir).resolve()
    (out_dir / "00_host_only" / "track_results").mkdir(parents=True, exist_ok=True)
    (out_dir / "01_host_plus_plugin" / "track_results").mkdir(parents=True, exist_ok=True)
    (out_dir / "01_host_plus_plugin" / "diagnostics").mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )
    cudnn.benchmark = True

    exp = get_exp(str(Path(args.exp_file).resolve()), None)
    exp.test_conf = float(args.conf)
    exp.nmsthre = float(args.nms)
    model = exp.get_model()
    logger.info("Model Summary: {}", get_model_info(model, exp.test_size))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    model.to(device)
    model.eval()

    logger.info("loading checkpoint")
    ckpt = torch.load(str(Path(args.ckpt).resolve()), map_location=device)
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("Fusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()

    val_loader = exp.get_eval_loader(int(args.batch_size), False, False)

    host_args = build_tracker_args(args, plugin=False)
    plugin_args = build_tracker_args(args, plugin=True)
    ori_thresh = float(args.track_thresh)
    ori_buffer = int(args.track_buffer)

    host_tracker = None
    plugin_tracker = None
    id_counter = {
        "host_only": 0,
        "host_plus_plugin": 0,
    }
    current_sequence = ""
    host_results: list[tuple[int, list[list[float]], list[int], list[float]]] = []
    plugin_results: list[tuple[int, list[list[float]], list[int], list[float]]] = []
    video_names = defaultdict(str)

    for cur_iter, (imgs, _, info_imgs, ids) in enumerate(val_loader):
        del cur_iter
        frame_id = int(info_imgs[2].item())
        video_id = int(info_imgs[3].item())
        img_file_name = info_imgs[4]
        video_name = str(img_file_name[0]).split("/")[0]
        video_names[video_id] = video_name

        apply_video_profile(host_args, video_name=video_name, ori_thresh=ori_thresh, ori_buffer=ori_buffer)
        apply_video_profile(plugin_args, video_name=video_name, ori_thresh=ori_thresh, ori_buffer=ori_buffer)

        if frame_id == 1:
            if current_sequence and host_tracker is not None and plugin_tracker is not None:
                flush_sequence(
                    sequence_name=current_sequence,
                    host_results=host_results,
                    plugin_results=plugin_results,
                    out_dir=out_dir,
                    plugin_tracker=plugin_tracker,
                )
                host_results = []
                plugin_results = []
            host_tracker = BYTETracker(copy.deepcopy(host_args))
            plugin_tracker = ByteTrackerLocalConflict(copy.deepcopy(plugin_args))
            current_sequence = video_name
        elif host_tracker is None or plugin_tracker is None:
            raise RuntimeError(f"Tracker state was not initialized before frame {frame_id} of {video_name}")

        host_tracker.args.track_thresh = float(host_args.track_thresh)
        host_tracker.args.track_buffer = int(host_args.track_buffer)
        plugin_tracker.args.track_thresh = float(plugin_args.track_thresh)
        plugin_tracker.args.track_buffer = int(plugin_args.track_buffer)
        if hasattr(plugin_tracker, "set_sequence_name"):
            plugin_tracker.set_sequence_name(video_name)

        with torch.no_grad():
            imgs = imgs.to(device=device, non_blocking=True)
            imgs = imgs.half() if args.fp16 else imgs.float()
            outputs = model(imgs)
            outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)

        output_tensor = outputs[0]
        if output_tensor is None:
            continue

        host_input = output_tensor.detach().clone()
        plugin_input = output_tensor.detach().clone()
        BaseTrack._count = int(id_counter["host_only"])
        host_online = host_tracker.update(host_input, info_imgs, exp.test_size)
        id_counter["host_only"] = int(BaseTrack._count)
        BaseTrack._count = int(id_counter["host_plus_plugin"])
        plugin_online = plugin_tracker.update(plugin_input, info_imgs, exp.test_size)
        id_counter["host_plus_plugin"] = int(BaseTrack._count)

        host_tlwhs, host_ids, host_scores = filter_online_targets(
            host_online,
            min_box_area=float(args.min_box_area),
        )
        plugin_tlwhs, plugin_ids, plugin_scores = filter_online_targets(
            plugin_online,
            min_box_area=float(args.min_box_area),
        )
        host_results.append((frame_id, host_tlwhs, host_ids, host_scores))
        plugin_results.append((frame_id, plugin_tlwhs, plugin_ids, plugin_scores))

    if current_sequence and host_tracker is not None and plugin_tracker is not None:
        flush_sequence(
            sequence_name=current_sequence,
            host_results=host_results,
            plugin_results=plugin_results,
            out_dir=out_dir,
            plugin_tracker=plugin_tracker,
        )

    summary = {
        "status": "success",
        "plugin_mode": str(args.plugin_mode),
        "host_results_dir": str((out_dir / "00_host_only" / "track_results").resolve()),
        "plugin_results_dir": str((out_dir / "01_host_plus_plugin" / "track_results").resolve()),
        "plugin_diagnostics_dir": str((out_dir / "01_host_plus_plugin" / "diagnostics").resolve()),
    }
    (out_dir / "shared_pair_core_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
