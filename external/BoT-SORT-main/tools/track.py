import argparse
import configparser
import hashlib
import json
import os
import sys
import os.path as osp
import subprocess
from datetime import datetime, timezone
import cv2
import numpy as np
import torch

sys.path.append('.')

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking

from tracker.tracking_utils.timer import Timer
from tracker.bot_sort import BoTSORT

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
PROFILE_META_KEYS = {"description", "settings"}

# Global
trackerTimer = Timer()
timer = Timer()


def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT Tracks For Evaluation!")

    parser.add_argument("path", help="path to dataset under evaluation")
    parser.add_argument("--benchmark", dest="benchmark", type=str, default='MOT17', help="benchmark to evaluate: MOT17 | MOT20 | DanceTrack")
    parser.add_argument("--eval", dest="split_to_eval", type=str, default='test', help="split to evaluate: train | val | test")
    parser.add_argument("--seq-ids", dest="seq_ids", nargs='+', type=int, default=None, help="optional subset of sequence ids to evaluate, e.g. --seq-ids 3 5")
    parser.add_argument("--mot17-detector-exts", dest="mot17_detector_exts", nargs='+', type=str, default=None, help="optional MOT17 detector tags to evaluate, e.g. --mot17-detector-exts FRCNN")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("--exp-profile", dest="exp_profile", type=str, default="", help="named runtime profile under configs/profiles or an explicit json path")
    parser.add_argument("--profile-root", dest="profile_root", type=str, default="", help="optional root directory for runtime profiles")
    parser.add_argument("--run-manifest-path", dest="run_manifest_path", type=str, default="", help="optional output path for run_manifest.json")
    parser.add_argument("--default-parameters", dest="default_parameters", default=False, action="store_true", help="use the default parameters as in the paper")
    parser.add_argument("--save-frames", dest="save_frames", default=False, action="store_true", help="save sequences with tracks.")

    # Detector
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')

    # CMC
    parser.add_argument("--cmc-method", default="file", type=str, help="cmc method: files (Vidstab GMC) | sparseOptFlow | orb | ecc | none")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use Re-ID flag.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    # Frequency-guided appearance gate (plug-and-play)
    parser.add_argument('--freq-gate', dest='freq_gate', default=False, action='store_true', help='enable frequency-guided appearance gate')
    parser.add_argument('--freq-gate-min', dest='freq_gate_min', type=float, default=0.2, help='min gate for appearance scaling')
    parser.add_argument('--freq-gate-max', dest='freq_gate_max', type=float, default=1.0, help='max gate for appearance scaling')
    parser.add_argument('--laplace-assoc', dest='laplace_assoc', default=False, action='store_true', help='enable Laplace-guided temporal reliability association')
    parser.add_argument(
        '--laplace-primary-only',
        dest='laplace_primary_only',
        default=False,
        action='store_true',
        help='apply Laplace association only in the primary (high-score) association stage',
    )
    parser.add_argument('--laplace-weight', dest='laplace_weight', type=float, default=0.35, help='blend weight for Laplace temporal signatures')
    parser.add_argument('--laplace-decay-scales', dest='laplace_decay_scales', nargs='+', type=float, default=[1.0, 2.0, 4.0], help='decay scales for Laplace temporal prototypes')
    parser.add_argument('--laplace-min-history', dest='laplace_min_history', type=int, default=3, help='minimum track history length to enable Laplace temporal prototypes')
    parser.add_argument('--laplace-proto-mode', dest='laplace_proto_mode', type=str, default='multi', choices=['multi', 'single', 'mean'], help='prototype construction for temporal appearance')
    parser.add_argument('--laplace-no-reliability', dest='laplace_no_reliability', default=False, action='store_true', help='disable reliability-controlled fusion and only refine appearance distance')
    parser.add_argument('--laplace-no-det-score', dest='laplace_no_det_score', default=False, action='store_true', help='disable detection-score modulation in reliability')
    parser.add_argument('--laplace-disable-pole-bank', dest='laplace_disable_pole_bank', default=False, action='store_true', help='disable learned pole-bank gating and use fixed temporal prototypes')
    parser.add_argument('--laplace-reliability-scale', dest='laplace_reliability_scale', type=float, default=1.0, help='global scale factor for reliability strength')
    parser.add_argument(
        '--laplace-agreement-mode',
        dest='laplace_agreement_mode',
        type=str,
        default='absdiff',
        choices=['absdiff', 'scaled_absdiff', 'min', 'prod'],
        help='agreement function between spatial and temporal appearance similarities',
    )
    parser.add_argument(
        '--laplace-use-history-len',
        dest='laplace_use_history_len',
        default=False,
        action='store_true',
        help='down-weight reliability for short track histories',
    )
    parser.add_argument(
        '--laplace-history-len-gamma',
        dest='laplace_history_len_gamma',
        type=float,
        default=1.0,
        help='exponent for history-length reliability scaling (only if --laplace-use-history-len is set)',
    )
    parser.add_argument(
        '--laplace-calibrator',
        dest='laplace_calibrator',
        type=str,
        default='',
        help='optional learned alpha/r calibrator weights (.npz) to override heuristic fusion',
    )
    parser.add_argument(
        '--laplace-assoc-mode',
        dest='laplace_assoc_mode',
        type=str,
        default='auto',
        choices=['auto', 'heuristic', 'current_learned', 'haca_v1', 'haca_v2', 'haca_v3'],
        help='association-time learned mode: auto keeps old behavior unless a HACA checkpoint is provided',
    )
    parser.add_argument(
        '--laplace-haca-checkpoint',
        dest='laplace_haca_checkpoint',
        type=str,
        default='',
        help='HACA-v1 checkpoint (.npz) for primary association only',
    )
    parser.add_argument(
        '--laplace-haca-no-set-encoder',
        dest='laplace_haca_no_set_encoder',
        default=False,
        action='store_true',
        help='disable HACA candidate-set encoder and keep only the pair projector',
    )
    parser.add_argument(
        '--laplace-haca-no-background',
        dest='laplace_haca_no_background',
        default=False,
        action='store_true',
        help='disable HACA background head and only apply bounded residual correction',
    )
    parser.add_argument(
        '--laplace-haca-delta-scale',
        dest='laplace_haca_delta_scale',
        type=float,
        default=float('nan'),
        help='optional override for HACA residual scale; default uses the checkpoint value',
    )
    parser.add_argument('--laplace-analysis-dir', dest='laplace_analysis_dir', type=str, default='', help='optional output directory for LTRA pairwise analysis logs')

    return parser


def repo_root_from_track_py():
    return osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))


def default_profile_root():
    return osp.join(repo_root_from_track_py(), "configs", "profiles")


def resolve_profile_path(profile_name, profile_root):
    candidate = profile_name
    if osp.isfile(candidate):
        return osp.abspath(candidate)
    root = profile_root or default_profile_root()
    json_candidate = osp.join(root, f"{profile_name}.json")
    if osp.isfile(json_candidate):
        return osp.abspath(json_candidate)
    bare_candidate = osp.join(root, profile_name)
    if osp.isfile(bare_candidate):
        return osp.abspath(bare_candidate)
    raise FileNotFoundError(f"Unable to resolve exp profile '{profile_name}' under {root}")


def collect_explicit_dests(parser, argv):
    explicit = {"path"}
    option_to_dest = {}
    for action in parser._actions:
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest
    for token in argv:
        if not token.startswith("-"):
            continue
        opt = token.split("=", 1)[0]
        dest = option_to_dest.get(opt)
        if dest:
            explicit.add(dest)
    return explicit


def apply_exp_profile(args, parser, argv):
    if not args.exp_profile:
        return None
    profile_path = resolve_profile_path(args.exp_profile, args.profile_root)
    with open(profile_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    settings = payload.get("settings", payload)
    explicit_dests = collect_explicit_dests(parser, argv)
    for key, value in settings.items():
        if key in PROFILE_META_KEYS or key in explicit_dests:
            continue
        if not hasattr(args, key):
            raise KeyError(f"Profile '{args.exp_profile}' contains unknown arg '{key}'")
        setattr(args, key, value)
    args.profile_root = args.profile_root or default_profile_root()
    return {
        "name": args.exp_profile,
        "path": profile_path,
        "description": payload.get("description", ""),
        "settings": settings,
        "explicit_overrides": sorted(explicit_dests),
    }


def compute_file_md5(path_value):
    if not path_value:
        return ""
    abs_path = osp.abspath(path_value)
    if not osp.isfile(abs_path):
        return ""
    digest = hashlib.md5()
    with open(abs_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(repo_root):
    info = {"commit": "", "dirty": None}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
        info["dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                text=True,
            ).strip()
        )
    except Exception:
        pass
    return info


def build_sequence_manifest_entry(args, spec):
    return {
        "name": spec["name"],
        "path": spec["path"],
        "ablation": bool(spec["ablation"]),
        "mot20": bool(spec["mot20"]),
        "fps": int(spec["fps"]),
        "exp_file": args.exp_file,
        "ckpt": args.ckpt,
        "track_high_thresh": float(args.track_high_thresh),
        "track_low_thresh": float(args.track_low_thresh),
        "new_track_thresh": float(args.new_track_thresh),
        "track_buffer": int(args.track_buffer),
        "match_thresh": float(args.match_thresh),
        "proximity_thresh": float(args.proximity_thresh),
        "appearance_thresh": float(args.appearance_thresh),
        "assoc_mode": args.laplace_assoc_mode,
        "laplace_assoc": bool(args.laplace_assoc),
        "laplace_primary_only": bool(args.laplace_primary_only),
        "with_reid": bool(args.with_reid),
        "fast_reid_config": args.fast_reid_config,
        "fast_reid_weights": args.fast_reid_weights,
        "cmc_method": args.cmc_method,
    }


def write_run_manifest(manifest_path, args, dataset_root, sequence_specs, resolved_sequences, profile_payload, output_dir):
    repo_root = repo_root_from_track_py()
    manifest = {
        "manifest_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": repo_root,
        "git": git_metadata(repo_root),
        "profile": profile_payload or {},
        "dataset_root": dataset_root,
        "benchmark": args.benchmark,
        "split_to_eval": args.split_to_eval,
        "experiment_name": args.experiment_name,
        "output_dir": output_dir,
        "mot17_detector_exts": list(args.mot17_detector_exts) if args.mot17_detector_exts is not None else None,
        "detector": {
            "exp_file": args.exp_file,
            "ckpt": args.ckpt,
            "ckpt_md5": compute_file_md5(args.ckpt),
            "device": str(args.device),
            "fp16": bool(args.fp16),
            "fuse": bool(args.fuse),
        },
        "reid": {
            "enabled": bool(args.with_reid),
            "config": args.fast_reid_config,
            "weights": args.fast_reid_weights,
            "weights_md5": compute_file_md5(args.fast_reid_weights),
        },
        "tracking": {
            "track_high_thresh": float(args.track_high_thresh),
            "track_low_thresh": float(args.track_low_thresh),
            "new_track_thresh": float(args.new_track_thresh),
            "track_buffer": int(args.track_buffer),
            "match_thresh": float(args.match_thresh),
            "aspect_ratio_thresh": float(args.aspect_ratio_thresh),
            "min_box_area": float(args.min_box_area),
            "proximity_thresh": float(args.proximity_thresh),
            "appearance_thresh": float(args.appearance_thresh),
            "cmc_method": args.cmc_method,
        },
        "association": {
            "freq_gate": bool(args.freq_gate),
            "laplace_assoc": bool(args.laplace_assoc),
            "laplace_assoc_mode": args.laplace_assoc_mode,
            "laplace_primary_only": bool(args.laplace_primary_only),
            "laplace_calibrator": args.laplace_calibrator,
            "laplace_haca_checkpoint": args.laplace_haca_checkpoint,
            "laplace_disable_pole_bank": bool(args.laplace_disable_pole_bank),
            "laplace_no_reliability": bool(args.laplace_no_reliability),
            "laplace_no_det_score": bool(args.laplace_no_det_score),
            "laplace_haca_no_set_encoder": bool(args.laplace_haca_no_set_encoder),
            "laplace_haca_no_background": bool(args.laplace_haca_no_background),
        },
        "sequence_specs": sequence_specs,
        "resolved_sequences": resolved_sequences,
    }
    os.makedirs(osp.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def read_seqinfo_value(seq_dir, key, default_value):
    seqinfo_path = osp.join(seq_dir, "seqinfo.ini")
    if not osp.isfile(seqinfo_path):
        return default_value
    parser = configparser.ConfigParser()
    parser.read(seqinfo_path)
    if "Sequence" not in parser or key not in parser["Sequence"]:
        return default_value
    return parser["Sequence"].get(key, fallback=default_value)


def build_sequence_specs(data_path, benchmark, split_to_eval, seq_ids=None, mot17_detector_exts=None):
    if benchmark == 'MOT20':
        train_seqs = [1, 2, 3, 5]
        test_seqs = [4, 6, 7, 8]
        seqs_ext = ['']
        mot = 20
    elif benchmark == 'MOT17':
        train_seqs = [2, 4, 5, 9, 10, 11, 13]
        test_seqs = [1, 3, 6, 7, 8, 12, 14]
        seqs_ext = ['FRCNN', 'DPM', 'SDP']
        if mot17_detector_exts is not None:
            seqs_ext = mot17_detector_exts
        mot = 17
    elif benchmark == 'DanceTrack':
        split_dir = osp.join(data_path, split_to_eval)
        if not osp.isdir(split_dir):
            raise ValueError("Error: Missing DanceTrack split directory: " + split_dir)
        seq_names = sorted(
            seq_name for seq_name in os.listdir(split_dir)
            if osp.isdir(osp.join(split_dir, seq_name))
        )
        if seq_ids is not None:
            selected = set(seq_ids)
            filtered = []
            for seq_name in seq_names:
                numeric = ''.join(ch for ch in seq_name if ch.isdigit())
                if numeric and int(numeric) in selected:
                    filtered.append(seq_name)
            seq_names = filtered
        if not seq_names:
            raise ValueError("Error: No DanceTrack sequences selected.")
        specs = []
        for seq_name in seq_names:
            seq_dir = osp.join(split_dir, seq_name)
            fps = int(read_seqinfo_value(seq_dir, "frameRate", 20))
            specs.append({
                "name": seq_name,
                "path": osp.join(seq_dir, "img1"),
                "ablation": False,
                "mot20": False,
                "fps": fps,
            })
        return specs
    else:
        raise ValueError("Error: Unsupported benchmark:" + benchmark)

    ablation = False
    if split_to_eval == 'train':
        seqs = train_seqs
    elif split_to_eval == 'val':
        # NOTE:
        # - In the upstream BoT-SORT/ByteTrack codebase, `ablation=True` is used to switch
        #   to the MOT17 "ablation" protocol (different exp/ckpt and GMC file folder).
        # - MOT20 does not have an "ablation" benchmark in the same sense, and treating
        #   MOT20 val as `ablation=True` breaks CMC file loading (looks under MOT17_ablation).
        #
        # For MOT20, we still evaluate on the train sequences (1/2/3/5) as a val split,
        # but we keep `ablation=False` so that:
        #   - default_parameters uses the standard MOT20 exp/ckpt, and
        #   - GMC file-based CMC reads from `tracker/GMC_files/MOTChallenge`.
        seqs = train_seqs
        if benchmark == 'MOT17':
            ablation = True
    elif split_to_eval == 'test':
        seqs = test_seqs
    else:
        raise ValueError("Error: Unsupported split to evaluate:" + split_to_eval)

    if seq_ids is not None:
        selected = set(seq_ids)
        seqs = [seq_id for seq_id in seqs if seq_id in selected]
        if not seqs:
            raise ValueError("Error: No sequences selected after applying --seq-ids")

    specs = []
    for ext in seqs_ext:
        for seq_id in seqs:
            if seq_id < 10:
                seq = 'MOT' + str(mot) + '-0' + str(seq_id)
            else:
                seq = 'MOT' + str(mot) + '-' + str(seq_id)

            if ext != '':
                seq += '-' + ext

            split = 'train' if seq_id in train_seqs else 'test'
            seq_dir = osp.join(data_path, split, seq)
            fps = int(read_seqinfo_value(seq_dir, "frameRate", 30))
            specs.append({
                "name": seq,
                "path": osp.join(seq_dir, "img1"),
                "ablation": ablation,
                "mot20": mot == 20,
                "fps": fps,
            })
    return specs


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            device=torch.device("cpu"),
            fp16=False
    ):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        if img is None:
            raise ValueError("Empty image: ", img_info["file_name"])

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)

        return outputs, img_info


def image_track(predictor, vis_folder, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()

    if args.ablation:
        files = files[len(files) // 2 + 1:]

    num_frames = len(files)

    # Tracker
    tracker = BoTSORT(args, frame_rate=args.fps)

    results = []

    for frame_id, img_path in enumerate(files, 1):

        # Detect objects
        outputs, img_info = predictor.inference(img_path, timer)
        scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))

        if outputs[0] is not None:
            outputs = outputs[0].cpu().numpy()
            detections = outputs[:, :7]
            detections[:, :4] /= scale

            trackerTimer.tic()
            online_targets = tracker.update(detections, img_info["raw_img"])
            trackerTimer.toc()

            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)

                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        if args.save_frames:
            save_folder = osp.join(vis_folder, args.name)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {}/{} ({:.2f} fps)'.format(frame_id, num_frames, 1. / max(1e-5, timer.average_time)))

    res_file = osp.join(vis_folder, args.name + ".txt")

    with open(res_file, 'w') as f:
        f.writelines(results)
    logger.info(f"save results to {res_file}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    vis_folder = osp.join(output_dir, "track_results")
    os.makedirs(vis_folder, exist_ok=True)

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if args.ckpt is None:
        ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
    else:
        ckpt_file = args.ckpt
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")

    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    predictor = Predictor(model, exp, args.device, args.fp16)

    image_track(predictor, vis_folder, args)


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    profile_payload = apply_exp_profile(args, parser, sys.argv[1:])

    data_path = args.path
    fp16 = args.fp16
    device = args.device
    base_exp_file = args.exp_file
    base_ckpt = args.ckpt

    sequence_specs = build_sequence_specs(
        data_path=data_path,
        benchmark=args.benchmark,
        split_to_eval=args.split_to_eval,
        seq_ids=args.seq_ids,
        mot17_detector_exts=args.mot17_detector_exts,
    )

    mainTimer = Timer()
    mainTimer.tic()
    resolved_sequences = {}
    manifest_path = ""
    manifest_output_dir = ""

    for spec in sequence_specs:
        seq = spec["name"]

        args.name = seq

        args.ablation = spec["ablation"]
        args.mot20 = spec["mot20"]
        args.fps = spec["fps"]
        args.device = device
        args.fp16 = fp16
        args.batch_size = 1
        args.trt = False
        args.path = spec["path"]

        if args.benchmark == 'DanceTrack' and args.cmc_method in ['file', 'files']:
            logger.warning("DanceTrack has no precomputed GMC files; switching CMC to none.")
            args.cmc_method = 'none'

        if args.default_parameters:

            if args.benchmark == 'MOT20':  # MOT20
                args.exp_file = r'./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py'
                args.ckpt = r'./pretrained/bytetrack_x_mot20.pth.tar'
                args.match_thresh = 0.7
            else:  # MOT17 / DanceTrack
                if args.ablation:
                    args.exp_file = r'./yolox/exps/example/mot/yolox_x_ablation.py'
                    args.ckpt = r'./pretrained/bytetrack_ablation.pth.tar'
                else:
                    args.exp_file = r'./yolox/exps/example/mot/yolox_x_mix_det.py'
                    args.ckpt = r'./pretrained/bytetrack_x_mot17.pth.tar'

            exp = get_exp(args.exp_file, args.name)

            args.track_high_thresh = 0.6
            args.track_low_thresh = 0.1
            args.track_buffer = 30

            if seq == 'MOT17-05-FRCNN' or seq == 'MOT17-06-FRCNN':
                args.track_buffer = 14
            elif seq == 'MOT17-13-FRCNN' or seq == 'MOT17-14-FRCNN':
                args.track_buffer = 25
            else:
                args.track_buffer = 30

            if seq == 'MOT17-01-FRCNN':
                args.track_high_thresh = 0.65
            elif seq == 'MOT17-06-FRCNN':
                args.track_high_thresh = 0.65
            elif seq == 'MOT17-12-FRCNN':
                args.track_high_thresh = 0.7
            elif seq == 'MOT17-14-FRCNN':
                args.track_high_thresh = 0.67
            elif seq in ['MOT20-06', 'MOT20-08']:
                args.track_high_thresh = 0.3
                exp.test_size = (736, 1920)

            args.new_track_thresh = args.track_high_thresh + 0.1
        else:
            args.exp_file = base_exp_file
            args.ckpt = base_ckpt
            exp = get_exp(args.exp_file, args.name)

        exp.test_conf = max(0.001, args.track_low_thresh - 0.01)
        if not args.experiment_name:
            args.experiment_name = exp.exp_name
        if not manifest_output_dir:
            manifest_output_dir = osp.join(exp.output_dir, args.experiment_name)
            manifest_path = args.run_manifest_path or osp.join(manifest_output_dir, "run_manifest.json")
        resolved_sequences[seq] = build_sequence_manifest_entry(args, spec)
        write_run_manifest(
            manifest_path=manifest_path,
            args=args,
            dataset_root=data_path,
            sequence_specs=sequence_specs,
            resolved_sequences=resolved_sequences,
            profile_payload=profile_payload,
            output_dir=manifest_output_dir,
        )
        main(exp, args)

    mainTimer.toc()
    print("TOTAL TIME END-to-END (with loading networks and images): ", mainTimer.total_time)
    print("TOTAL TIME (Detector + Tracker): " + str(timer.total_time) + ", FPS: " + str(1.0 /timer.average_time))
    print("TOTAL TIME (Tracker only): " + str(trackerTimer.total_time) + ", FPS: " + str(1.0 / trackerTimer.average_time))
