import argparse
import configparser
import csv
import gc
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


def _read_image_robust(img_path):
    """
    Read an image using cv2.imread first, then fall back to byte-based decode.

    Some filesystem/runtime combinations intermittently return None from
    cv2.imread even when the file exists and is valid. A byte-based fallback
    makes evaluation less brittle.
    """
    img = cv2.imread(img_path)
    if img is not None:
        return img
    try:
        data = np.fromfile(img_path, dtype=np.uint8)
        if data.size:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    except Exception as exc:
        logger.warning(f"robust image read fallback failed for {img_path}: {exc}")
    return None


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
    parser.add_argument(
        "--skip-existing-results",
        dest="skip_existing_results",
        default=False,
        action="store_true",
        help="skip a sequence if its final track result already exists",
    )
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
    parser.add_argument('--fcaa-enable', dest='fcaa_enable', default=False, action='store_true', help='enable FCAA selective appearance rescoring')
    parser.add_argument('--fcaa-scorer-checkpoint', dest='fcaa_scorer_checkpoint', type=str, default='', help='trained FCAA scorer checkpoint (.pt)')
    parser.add_argument('--fcaa-trigger-mode', dest='fcaa_trigger_mode', type=str, default='row_margin', choices=['row_margin', 'shared_det_top1', 'shared_det_top1_margin'], help='FCAA trigger grouping mode')
    parser.add_argument('--fcaa-trigger-margin', dest='fcaa_trigger_margin', type=float, default=0.05, help='best-vs-second-best similarity margin threshold for FCAA triggering')
    parser.add_argument('--fcaa-lambda', dest='fcaa_lambda', type=float, default=0.3, help='blend weight for FCAA refined match probability')
    parser.add_argument('--fcaa-topk', dest='fcaa_topk', type=int, default=3, help='top-k candidate detections to rescore when FCAA triggers')
    parser.add_argument('--fcaa-crop-height', dest='fcaa_crop_height', type=int, default=128, help='crop height for fixed frequency descriptors')
    parser.add_argument('--fcaa-crop-width', dest='fcaa_crop_width', type=int, default=64, help='crop width for fixed frequency descriptors')
    parser.add_argument('--fcaa-analysis-dir', dest='fcaa_analysis_dir', type=str, default='', help='optional output directory for FCAA runtime diagnostics')
    parser.add_argument('--fgas-enable', dest='fgas_enable', default=False, action='store_true', help='enable FGAS block-level primary association refinement')
    parser.add_argument('--fgas-resolver-checkpoint', dest='fgas_resolver_checkpoint', type=str, default='', help='trained FGAS block resolver checkpoint (.pt)')
    parser.add_argument('--fgas-topk', dest='fgas_topk', type=int, default=5, help='top-k candidate detections kept per track for FGAS blocks')
    parser.add_argument('--fgas-max-rows', dest='fgas_max_rows', type=int, default=3, help='max rows per FGAS conflict block')
    parser.add_argument('--fgas-max-cols', dest='fgas_max_cols', type=int, default=3, help='max cols per FGAS conflict block')
    parser.add_argument('--fgas-blend-weight', dest='fgas_blend_weight', type=float, default=0.5, help='blend weight for FGAS refined block scores')
    parser.add_argument('--fgas-assignment-mode', dest='fgas_assignment_mode', type=str, default='blend', choices=['blend', 'replace'], help='FGAS assignment update mode')
    parser.add_argument('--fgas-row-nomatch-weight', dest='fgas_row_nomatch_weight', type=float, default=0.0, help='extra down-weight applied when FGAS predicts row no-match')
    parser.add_argument('--fgas-controller-enable', dest='fgas_controller_enable', default=False, action='store_true', help='enable FGAS controller actions before Hungarian')
    parser.add_argument('--fgas-controller-edge-thresh', dest='fgas_controller_edge_thresh', type=float, default=0.6, help='minimum FGAS edge probability for forced match proposals')
    parser.add_argument('--fgas-controller-row-defer-thresh', dest='fgas_controller_row_defer_thresh', type=float, default=0.6, help='minimum row no-match probability for deferring a track from primary association')
    parser.add_argument('--fgas-controller-col-newborn-thresh', dest='fgas_controller_col_newborn_thresh', type=float, default=0.6, help='minimum column newborn probability for reserving a detection from primary association')
    parser.add_argument('--fgas-controller-margin-thresh', dest='fgas_controller_margin_thresh', type=float, default=0.05, help='margin required between controller action logit and competing alternatives')
    parser.add_argument('--fgas-controller-ambiguity-margin', dest='fgas_controller_ambiguity_margin', type=float, default=0.05, help='base similarity margin that defines an ambiguous row for FGAS controller actions')
    parser.add_argument('--fgas-crop-height', dest='fgas_crop_height', type=int, default=128, help='crop height for FGAS frequency descriptors when used')
    parser.add_argument('--fgas-crop-width', dest='fgas_crop_width', type=int, default=64, help='crop width for FGAS frequency descriptors when used')
    parser.add_argument('--owneralt-competition-enable', dest='owneralt_competition_enable', default=False, action='store_true', help='enable standalone owner-challenger competition rewrite before Hungarian')
    parser.add_argument('--owneralt-competition-min-time-since-update', dest='owneralt_competition_min_time_since_update', type=int, default=2, help='minimum challenger staleness in frames')
    parser.add_argument('--owneralt-competition-max-time-since-update', dest='owneralt_competition_max_time_since_update', type=int, default=8, help='maximum challenger staleness in frames')
    parser.add_argument('--owneralt-competition-min-tracklet-len', dest='owneralt_competition_min_tracklet_len', type=int, default=20, help='minimum challenger tracklet length')
    parser.add_argument('--owneralt-competition-min-box-iou', dest='owneralt_competition_min_box_iou', type=float, default=0.75, help='minimum challenger-track box IoU on the contested detection')
    parser.add_argument('--owneralt-competition-gap1-min-box-iou', dest='owneralt_competition_gap1_min_box_iou', type=float, default=-1.0, help='optional stricter challenger-track box IoU for gap=1 challengers; negative disables')
    parser.add_argument('--owneralt-competition-owner-max-tracklet-len', dest='owneralt_competition_owner_max_tracklet_len', type=int, default=8, help='maximum owner tracklet length allowed for release')
    parser.add_argument('--owneralt-competition-owner-alt-det-min-score', dest='owneralt_competition_owner_alt_det_min_score', type=float, default=0.0, help='minimum detection score for the owner fallback detection')
    parser.add_argument('--owneralt-competition-owner-alt-det-min-box-iou', dest='owneralt_competition_owner_alt_det_min_box_iou', type=float, default=0.0, help='minimum owner-vs-fallback box IoU')
    parser.add_argument('--owneralt-competition-gap1-owner-alt-det-min-box-iou', dest='owneralt_competition_gap1_owner_alt_det_min_box_iou', type=float, default=-1.0, help='optional stricter owner-vs-fallback box IoU for gap=1 challengers; negative disables')
    parser.add_argument('--owneralt-competition-max-owner-edge-deficit', dest='owneralt_competition_max_owner_edge_deficit', type=float, default=0.10, help='maximum allowed challenger-vs-owner cost deficit on the contested detection')
    parser.add_argument('--owneralt-competition-gap1-max-owner-edge-deficit', dest='owneralt_competition_gap1_max_owner_edge_deficit', type=float, default=-1.0, help='optional stricter challenger-vs-owner cost deficit for gap=1 challengers; negative disables')
    parser.add_argument('--owneralt-competition-evidence-mode', dest='owneralt_competition_evidence_mode', type=str, default='legacy', choices=['legacy', 'joint'], help='how OwnerAlt ranks accepted challenger-owner-alt triples')
    parser.add_argument('--owneralt-competition-max-joint-penalty', dest='owneralt_competition_max_joint_penalty', type=float, default=-1.0, help='maximum allowed local total-cost increase for any accepted OwnerAlt rewrite; negative disables')
    parser.add_argument('--owneralt-competition-gap1-max-joint-penalty', dest='owneralt_competition_gap1_max_joint_penalty', type=float, default=-1.0, help='optional stricter local total-cost increase for gap=1 challengers; negative disables')
    parser.add_argument('--owneralt-competition-owner-alt-bonus', dest='owneralt_competition_owner_alt_bonus', type=float, default=0.10, help='cost bonus applied to the owner fallback edge')
    parser.add_argument('--owneralt-competition-block-owner-on-reclaim', dest='owneralt_competition_block_owner_on_reclaim', default=False, action='store_true', help='block the original owner from reclaiming the contested detection after reroute')
    parser.add_argument('--reentry-memory-enable', dest='reentry_memory_enable', default=False, action='store_true', help='enable long-term archived-track recovery before newborn creation')
    parser.add_argument('--reentry-memory-max-gap', dest='reentry_memory_max_gap', type=int, default=60, help='max frames after removal to keep a track eligible for re-entry recovery')
    parser.add_argument('--reentry-memory-max-size', dest='reentry_memory_max_size', type=int, default=256, help='maximum number of archived tracks kept for recovery')
    parser.add_argument('--reentry-memory-min-similarity', dest='reentry_memory_min_similarity', type=float, default=0.60, help='minimum similarity required to reactivate an archived track')
    parser.add_argument('--reentry-memory-confirm-streak', dest='reentry_memory_confirm_streak', type=int, default=2, help='number of consecutive confirmations required before committing a recovered track')
    parser.add_argument('--reentry-memory-confirm-gap', dest='reentry_memory_confirm_gap', type=int, default=2, help='maximum frame gap allowed between consecutive confirmations')
    parser.add_argument('--reentry-memory-confirm-min-similarity', dest='reentry_memory_confirm_min_similarity', type=float, default=0.65, help='minimum similarity required for a confirmation frame')
    parser.add_argument('--reentry-memory-min-det-score', dest='reentry_memory_min_det_score', type=float, default=0.10, help='minimum detection confidence considered for recovery')
    parser.add_argument('--reentry-memory-appearance-weight', dest='reentry_memory_app_weight', type=float, default=0.55, help='appearance weight in the re-entry score')
    parser.add_argument('--reentry-memory-iou-weight', dest='reentry_memory_iou_weight', type=float, default=0.25, help='IoU/motion weight in the re-entry score')
    parser.add_argument('--reentry-memory-score-weight', dest='reentry_memory_score_weight', type=float, default=0.10, help='detection-score weight in the re-entry score')
    parser.add_argument('--reentry-memory-gap-weight', dest='reentry_memory_gap_weight', type=float, default=0.10, help='gap-decay weight in the re-entry score')
    parser.add_argument('--reentry-memory-use-low-score', dest='reentry_memory_use_low_score', default=False, action='store_true', help='also run recovery on low-score unmatched detections')
    parser.add_argument('--reentry-memory-compete-primary', dest='reentry_memory_compete_primary', default=False, action='store_true', help='let removed tracks compete with active tracks in primary association')
    parser.add_argument('--reentry-engine-enable', dest='reentry_engine_enable', default=False, action='store_true', help='enable the database-style re-entry query engine')
    parser.add_argument('--reentry-engine-hilbert-order', dest='reentry_engine_hilbert_order', type=int, default=8, help='Hilbert order used by the archive spatial router')
    parser.add_argument('--reentry-engine-bf-threshold', dest='reentry_engine_bf_threshold', type=int, default=50, help='archive size threshold for brute-force query planning')
    parser.add_argument('--reentry-engine-spatial-radius', dest='reentry_engine_spatial_radius', type=int, default=2, help='base spatial routing radius')
    parser.add_argument('--reentry-engine-max-spatial-radius', dest='reentry_engine_max_spatial_radius', type=int, default=4, help='maximum spatial routing radius')
    parser.add_argument('--reentry-engine-short-gap-threshold', dest='reentry_engine_short_gap_threshold', type=int, default=0, help='skip engine commit for re-entries with gap <= this threshold (0=disabled)')
    parser.add_argument('--reentry-engine-num-prototypes', dest='reentry_engine_num_prototypes', type=int, default=1, help='number of appearance prototypes per archived identity (1=single smooth_feat, 3=multi-prototype)')
    parser.add_argument('--reentry-engine-recent-score-margin', dest='reentry_engine_recent_score_margin', type=float, default=0.0, help='if top candidate scores are within this margin, allow recent-exit reranking (0=disabled)')
    parser.add_argument('--reentry-engine-recent-min-exit-frame-advantage', dest='reentry_engine_recent_min_exit_frame_advantage', type=int, default=0, help='minimum exit-frame recency advantage required to override the top score (0=disabled)')
    parser.add_argument('--reentry-engine-dump-matches', dest='reentry_engine_dump_matches', default=False, action='store_true', help='dump top-k match candidates for each engine query (for offline ceiling analysis)')
    parser.add_argument('--bc-lost-track-promote', dest='bc_lost_track_promote', default=False, action='store_true', help='enable branch-consistency v2: promote recently-lost tracks in primary cost matrix when appearance is strong')
    parser.add_argument('--bc-lost-track-max-gap', dest='bc_lost_track_max_gap', type=int, default=30, help='max time_since_update for a lost track to be eligible for promotion')
    parser.add_argument('--bc-appearance-thresh', dest='bc_appearance_thresh', type=float, default=0.45, help='embedding distance threshold below which a lost track is considered appearance-matching')
    parser.add_argument('--bc-cost-margin', dest='bc_cost_margin', type=float, default=0.15, help='max cost delta between lost track and best alternative for promotion')
    parser.add_argument('--bc-promotion-weight', dest='bc_promotion_weight', type=float, default=0.3, help='multiplicative cost discount applied to promoted lost tracks (0.3 = 30%% reduction)')
    parser.add_argument('--bc-dump-trace', dest='bc_dump_trace', default=False, action='store_true', help='dump per-promotion trace log for IDSW attribution analysis')
    parser.add_argument('--graph-assoc-enable', dest='graph_assoc_enable', default=False, action='store_true', help='enable standalone local competition-graph reassociation before Hungarian')
    parser.add_argument('--graph-assoc-no-col-only-blocks', dest='graph_assoc_allow_col_only_blocks', default=True, action='store_false', help='disable graph-association blocks triggered only by column ambiguity with no ambiguous rows')
    parser.add_argument('--graph-assoc-require-row-involved-strict-reclaim', dest='graph_assoc_require_row_involved_strict_reclaim', default=False, action='store_true', help='require row-involved graph blocks to strictly increase reclaim matches before acceptance')
    parser.add_argument('--graph-assoc-top-k', dest='graph_assoc_top_k', type=int, default=3, help='top-k candidate detections kept per track when building local competition graphs')
    parser.add_argument('--graph-assoc-max-rows', dest='graph_assoc_max_rows', type=int, default=4, help='max number of track rows allowed in a local competition graph')
    parser.add_argument('--graph-assoc-max-cols', dest='graph_assoc_max_cols', type=int, default=4, help='max number of detection cols allowed in a local competition graph')
    parser.add_argument('--graph-assoc-row-margin', dest='graph_assoc_row_margin', type=float, default=0.03, help='row ambiguity margin between best and second-best cost')
    parser.add_argument('--graph-assoc-col-margin', dest='graph_assoc_col_margin', type=float, default=0.03, help='column ambiguity margin between best and second-best owner cost')
    parser.add_argument('--graph-assoc-min-reclaim-time-since-update', dest='graph_assoc_min_reclaim_time_since_update', type=int, default=1, help='minimum gap for a track to be treated as reclaimable')
    parser.add_argument('--graph-assoc-max-reclaim-time-since-update', dest='graph_assoc_max_reclaim_time_since_update', type=int, default=8, help='maximum gap for a track to be treated as reclaimable')
    parser.add_argument('--graph-assoc-min-reclaim-tracklet-len', dest='graph_assoc_min_reclaim_tracklet_len', type=int, default=20, help='minimum tracklet length for reclaimable rows')
    parser.add_argument('--graph-assoc-recent-owner-max-time-since-update', dest='graph_assoc_recent_owner_max_time_since_update', type=int, default=1, help='maximum gap for a row to be treated as a recent owner')
    parser.add_argument('--graph-assoc-recent-owner-max-tracklet-len', dest='graph_assoc_recent_owner_max_tracklet_len', type=int, default=8, help='maximum tracklet length for a row to be treated as a recent owner')
    parser.add_argument('--graph-assoc-protect-young-active-rows', dest='graph_assoc_protect_young_active_rows', default=False, action='store_true', help='protect short active baseline owners from being replaced by same-gap reclaim rows')
    parser.add_argument('--graph-assoc-young-active-max-time-since-update', dest='graph_assoc_young_active_max_time_since_update', type=int, default=1, help='maximum gap for a row to be treated as a protected young active owner')
    parser.add_argument('--graph-assoc-young-active-max-tracklet-len', dest='graph_assoc_young_active_max_tracklet_len', type=int, default=20, help='maximum tracklet length for a row to be treated as a protected young active owner')
    parser.add_argument('--graph-assoc-young-active-min-reclaim-gap', dest='graph_assoc_young_active_min_reclaim_gap', type=int, default=2, help='minimum gap required for an introduced reclaim row when suppressing a protected young active owner')
    parser.add_argument('--graph-assoc-young-active-max-cost-delta', dest='graph_assoc_young_active_max_cost_delta', type=float, default=-1.0, help='optional stricter max cost increase allowed when suppressing a protected young active owner; negative disables')
    parser.add_argument('--graph-assoc-protect-stale-lost-owner-rows', dest='graph_assoc_protect_stale_lost_owner_rows', default=False, action='store_true', help='protect long-gap stale lost baseline owners from being replaced by weak active introduced rows')
    parser.add_argument('--graph-assoc-stale-lost-owner-min-time-since-update', dest='graph_assoc_stale_lost_owner_min_time_since_update', type=int, default=9, help='minimum gap for a suppressed baseline row to be treated as a stale lost owner')
    parser.add_argument('--graph-assoc-stale-lost-owner-min-tracklet-len', dest='graph_assoc_stale_lost_owner_min_tracklet_len', type=int, default=100, help='minimum tracklet length for a suppressed baseline row to be treated as a stale lost owner')
    parser.add_argument('--graph-assoc-stale-lost-owner-active-max-time-since-update', dest='graph_assoc_stale_lost_owner_active_max_time_since_update', type=int, default=1, help='maximum gap for an introduced row to be treated as an active replacement in stale-owner protection')
    parser.add_argument('--graph-assoc-stale-lost-owner-min-introduced-edge-utility', dest='graph_assoc_stale_lost_owner_min_introduced_edge_utility', type=float, default=0.0, help='minimum raw edge utility required for active introduced rows when suppressing a stale lost owner')
    parser.add_argument('--graph-assoc-min-box-iou', dest='graph_assoc_min_box_iou', type=float, default=0.6, help='minimum box IoU for edges considered inside the local reassociation graph')
    parser.add_argument('--graph-assoc-reclaim-bonus', dest='graph_assoc_reclaim_bonus', type=float, default=0.08, help='bonus added to reclaimable rows inside local graph scoring')
    parser.add_argument('--graph-assoc-recent-owner-penalty', dest='graph_assoc_recent_owner_penalty', type=float, default=0.05, help='penalty applied to very recent short owner rows inside local graph scoring')
    parser.add_argument('--graph-assoc-iou-bonus', dest='graph_assoc_iou_bonus', type=float, default=0.04, help='bonus weight for box IoU inside local graph scoring')
    parser.add_argument('--graph-assoc-score-bonus', dest='graph_assoc_score_bonus', type=float, default=0.02, help='bonus weight for detection confidence inside local graph scoring')
    parser.add_argument('--graph-assoc-min-assignment-gain', dest='graph_assoc_min_assignment_gain', type=float, default=0.01, help='minimum local utility gain required to accept a reassociation block')
    parser.add_argument('--graph-assoc-max-cost-delta', dest='graph_assoc_max_cost_delta', type=float, default=0.05, help='maximum allowed raw local cost increase over the baseline block assignment')
    parser.add_argument('--graph-assoc-row-involved-min-assignment-gain', dest='graph_assoc_row_involved_min_assignment_gain', type=float, default=0.01, help='minimum gain required for blocks that include ambiguous rows')
    parser.add_argument('--graph-assoc-col-only-min-assignment-gain', dest='graph_assoc_col_only_min_assignment_gain', type=float, default=0.01, help='minimum gain required for column-only ambiguity blocks')
    parser.add_argument('--graph-assoc-col-only-max-cost-delta', dest='graph_assoc_col_only_max_cost_delta', type=float, default=0.05, help='maximum cost increase allowed for column-only ambiguity blocks')
    parser.add_argument('--graph-assoc-force-match-cost', dest='graph_assoc_force_match_cost', type=float, default=0.0, help='cost written onto chosen graph-association matches after accepting a block')
    parser.add_argument('--graph-assoc-allow-match-count-drop', dest='graph_assoc_allow_match_count_drop', default=False, action='store_true', help='allow graph reassociation blocks to reduce the number of matched pairs; disabled by default')
    parser.add_argument('--graph-assoc-dump-candidate-rows', dest='graph_assoc_dump_candidate_rows', default=False, action='store_true', help='dump candidate graph-association blocks, including rejected ones, for learned block acceptance training')
    parser.add_argument('--graph-assoc-candidate-rerank-top-k', dest='graph_assoc_candidate_rerank_top_k', type=int, default=6, help='number of candidate assignments to enumerate and rerank inside each local graph block')
    parser.add_argument('--graph-assoc-learned-commit-rerank-candidates', dest='graph_assoc_learned_commit_rerank_candidates', default=False, action='store_true', help='use the learned commit scorer to rerank enumerated local graph assignments before deciding the accepted match set')
    parser.add_argument('--graph-assoc-commit-checkpoint', dest='graph_assoc_commit_checkpoint', type=str, default='', help='optional learned checkpoint for graph-association commit gating')
    parser.add_argument('--graph-assoc-commit-device', dest='graph_assoc_commit_device', type=str, default='', help='device used by the graph-association commit scorer; empty follows the tracker device')
    parser.add_argument('--graph-assoc-commit-score-margin', dest='graph_assoc_commit_score_margin', type=float, default=0.0, help='minimum learned chosen-vs-baseline score delta required for learned graph-association commit acceptance')
    parser.add_argument('--graph-assoc-commit-gate-only', dest='graph_assoc_commit_gate_only', default=False, action='store_true', help='apply the learned graph-association commit scorer only as a conservative post-rule veto gate')
    parser.add_argument('--graph-assoc-commit-replace-rules', dest='graph_assoc_commit_replace_rules', default=False, action='store_true', help='replace graph-association hand-crafted commit rules with the learned commit scorer')
    parser.add_argument(
        '--graph-assoc-commit-decision-mode',
        dest='graph_assoc_commit_decision_mode',
        type=str,
        default='',
        help='optional override for dual-head graph-association gate decision mode',
    )
    parser.add_argument(
        '--graph-assoc-commit-threshold',
        dest='graph_assoc_commit_threshold',
        type=float,
        default=float('nan'),
        help='optional override for the learned graph-association gate acceptance threshold',
    )
    parser.add_argument(
        '--graph-assoc-commit-neutral-risk-weight',
        dest='graph_assoc_commit_neutral_risk_weight',
        type=float,
        default=float('nan'),
        help='optional override for dual-head gain-vs-neutral weighting',
    )
    parser.add_argument(
        '--graph-assoc-commit-positive-threshold',
        dest='graph_assoc_commit_positive_threshold',
        type=float,
        default=float('nan'),
        help='optional positive-head threshold used by dual-threshold graph-association gating',
    )
    parser.add_argument(
        '--graph-assoc-commit-neutral-threshold',
        dest='graph_assoc_commit_neutral_threshold',
        type=float,
        default=float('nan'),
        help='optional neutral-head threshold used by dual-threshold graph-association gating',
    )
    parser.add_argument(
        '--graph-assoc-commit-safety-min-gain',
        dest='graph_assoc_commit_safety_min_gain',
        type=float,
        default=float('nan'),
        help='optional minimum raw utility gain floor retained even when learned commit replacement is enabled',
    )
    parser.add_argument(
        '--graph-assoc-commit-safety-max-cost-delta',
        dest='graph_assoc_commit_safety_max_cost_delta',
        type=float,
        default=float('nan'),
        help='optional maximum raw cost increase floor retained even when learned commit replacement is enabled',
    )
    parser.add_argument(
        '--graph-assoc-commit-safety-require-reclaim-improve',
        dest='graph_assoc_commit_safety_require_reclaim_improve',
        default=False,
        action='store_true',
        help='require learned replacement candidates to preserve reclaim-improvement structure over the baseline block',
    )
    parser.add_argument(
        '--graph-assoc-commit-safety-require-same-match-count',
        dest='graph_assoc_commit_safety_require_same_match_count',
        default=False,
        action='store_true',
        help='require learned replacement candidates to keep the same local match count as the baseline block',
    )
    parser.add_argument('--graph-assoc-analysis-dir', dest='graph_assoc_analysis_dir', type=str, default='', help='optional output directory for graph-association runtime diagnostics')
    parser.add_argument('--owneralt-analysis-dir', dest='owneralt_analysis_dir', type=str, default='', help='optional output directory for OwnerAlt runtime diagnostics')
    parser.add_argument('--incremental-write-interval', dest='incremental_write_interval', type=int, default=20, help='flush tracking results and owneralt diagnostics every N frames to avoid losing long-sequence progress')
    parser.add_argument('--rgsa-dump-dir', dest='rgsa_dump_dir', type=str, default='', help='optional output directory for RGSA oracle dump (HACA debug per frame)')
    parser.add_argument('--rgsa-enable', dest='rgsa_enable', default=False, action='store_true', help='enable RGSA stage1/stage2 runtime cost rewriting before Hungarian')
    parser.add_argument('--rgsa-stage1-checkpoint', dest='rgsa_stage1_checkpoint', type=str, default='', help='trained RGSA Stage 1 checkpoint (.pt)')
    parser.add_argument('--rgsa-stage2-checkpoint', dest='rgsa_stage2_checkpoint', type=str, default='', help='trained RGSA Stage 2 checkpoint (.pt)')
    parser.add_argument('--rgsa-device', dest='rgsa_device', type=str, default='', help='device used by RGSA runtime heads; empty follows tracker device')
    parser.add_argument('--rgsa-topk', dest='rgsa_topk', type=int, default=5, help='top-k local candidates passed from Stage 1 to Stage 2')
    parser.add_argument('--rgsa-stage1-lambda-defer', dest='rgsa_stage1_lambda_defer', type=float, default=0.3, help='soft host-edge cost penalty for Stage 1 defer')
    parser.add_argument('--rgsa-stage1-lambda-reject', dest='rgsa_stage1_lambda_reject', type=float, default=0.8, help='soft host-edge rejection strength for Stage 1')
    parser.add_argument('--rgsa-stage2-rewrite-gain', dest='rgsa_stage2_rewrite_gain', type=float, default=0.35, help='cost discount applied to Stage 2 rewritten edge')
    parser.add_argument('--rgsa-stage2-haca-mode', dest='rgsa_stage2_haca_mode', type=str, default='learned', choices=['learned', 'dual_haca'], help='Stage2 strategy: learned head or dual-param HACA re-scoring')
    parser.add_argument('--rgsa-stage2-comp-delta-scale', dest='rgsa_stage2_comp_delta_scale', type=float, default=2.0, help='aggressive HACA delta_scale for deferred re-scoring')
    parser.add_argument('--rgsa-stage2-comp-margin-temperature', dest='rgsa_stage2_comp_margin_temperature', type=float, default=0.05, help='aggressive HACA margin_temperature for deferred re-scoring')
    parser.add_argument('--rgsa-stage2-rewrite-threshold', dest='rgsa_stage2_rewrite_threshold', type=float, default=0.05, help='min score improvement to accept a Stage2 rewrite')
    parser.add_argument('--rgsa-stage2-cost-discount', dest='rgsa_stage2_cost_discount', type=float, default=0.2, help='cost discount for Stage2 rewritten edges in dual_haca mode')
    parser.add_argument('--rgsa-verifier-mode', dest='rgsa_verifier_mode', type=str, default='none', choices=['none', 'heuristic', 'heuristic_tight'], help='Stage2 verifier mode: none=legacy, heuristic=default rules, heuristic_tight=conservative rules')
    parser.add_argument('--ccrc-enable', dest='ccrc_enable', default=False, action='store_true', help='enable CCRC Platt calibrator for commit/abstain')
    parser.add_argument('--ccrc-platt-checkpoint', dest='ccrc_platt_checkpoint', type=str, default='', help='path to Platt calibrator checkpoint (.pt)')
    parser.add_argument('--ccrc-tau-commit', dest='ccrc_tau_commit', type=float, default=0.0, help='commit threshold: p_correct >= tau -> commit, else abstain')

    # TCGAU: Track-Coherence-Gated Appearance Update
    parser.add_argument('--tcgau-enable', dest='tcgau_enable', default=False, action='store_true', help='enable TCGAU three-tier appearance update gating')
    parser.add_argument('--tcgau-freeze-thresh', dest='tcgau_freeze_thresh', type=float, default=0.30, help='q_update <= this -> freeze appearance memory')
    parser.add_argument('--tcgau-soft-thresh', dest='tcgau_soft_thresh', type=float, default=0.70, help='q_update <= this -> soft update with reduced alpha')
    parser.add_argument('--tcgau-soft-alpha', dest='tcgau_soft_alpha', type=float, default=0.97, help='EMA alpha for soft mode (higher = more conservative)')
    parser.add_argument('--tcgau-margin-thresh', dest='tcgau_margin_thresh', type=float, default=0.05, help='minimum margin to allow q_update > 0')
    parser.add_argument('--tcgau-history-norm-denom', dest='tcgau_history_norm_denom', type=float, default=10.0, help='denominator for history normalization')
    parser.add_argument('--tcgau-log-pairs', dest='tcgau_log_pairs', default=False, action='store_true', help='log per-pair TCGAU diagnostics')

    # TOS-Track (Track Occlusion Shadow): analysis-only mode
    parser.add_argument('--tos-enable', dest='tos_enable', default=False, action='store_true', help='enable TOS-Track occlusion shadow system')
    parser.add_argument('--tos-analysis-only', dest='tos_analysis_only', default=False, action='store_true', help='TOS analysis mode: instrument tracks, output CSV, no behavioral changes')
    parser.add_argument('--tos-analysis-dir', dest='tos_analysis_dir', type=str, default='', help='output directory for TOS analysis CSV files')
    parser.add_argument('--tos-hold-buffer', dest='tos_hold_buffer', type=int, default=30, help='TOS hold buffer: number of frames to freeze features after losing detection')
    parser.add_argument('--tos-newborn-delay', dest='tos_newborn_delay', type=int, default=5, help='TOS newborn delay: frames to defer new track creation')
    parser.add_argument('--tos-memory-frames', dest='tos_memory_frames', type=int, default=150, help='TOS shadow memory: frames to keep archived tracks before pruning')
    parser.add_argument('--tos-reconnect-gap-max', dest='tos_reconnect_gap_max', type=int, default=60, help='TOS max gap for reconnection (frames since shadow track end)')
    parser.add_argument('--tos-reconnect-min-similarity', dest='tos_reconnect_min_similarity', type=float, default=0.70, help='TOS min appearance similarity for reconnection')
    parser.add_argument('--tos-occlusion-thresh', dest='tos_occlusion_thresh', type=float, default=0.5, help='TOS occlusion score threshold to trigger hold mode')
    parser.add_argument('--tos-freeze-on-occlusion', dest='tos_freeze_on_occlusion', default=False, action='store_true', help='TOS: freeze features when occlusion detected (implies --tos-enable)')
    parser.add_argument('--tos-disable-reentry', dest='tos_disable_reentry', default=False, action='store_true', help='TOS: disable reentry_memory when TOS is active')

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
        "graph_assoc_enable": bool(getattr(args, "graph_assoc_enable", False)),
        "graph_assoc_top_k": int(getattr(args, "graph_assoc_top_k", 3)),
        "graph_assoc_max_rows": int(getattr(args, "graph_assoc_max_rows", 4)),
        "graph_assoc_max_cols": int(getattr(args, "graph_assoc_max_cols", 4)),
        "graph_assoc_row_margin": float(getattr(args, "graph_assoc_row_margin", 0.03)),
        "graph_assoc_col_margin": float(getattr(args, "graph_assoc_col_margin", 0.03)),
        "graph_assoc_min_reclaim_time_since_update": int(getattr(args, "graph_assoc_min_reclaim_time_since_update", 1)),
        "graph_assoc_max_reclaim_time_since_update": int(getattr(args, "graph_assoc_max_reclaim_time_since_update", 8)),
        "graph_assoc_min_reclaim_tracklet_len": int(getattr(args, "graph_assoc_min_reclaim_tracklet_len", 20)),
        "graph_assoc_recent_owner_max_time_since_update": int(getattr(args, "graph_assoc_recent_owner_max_time_since_update", 1)),
        "graph_assoc_recent_owner_max_tracklet_len": int(getattr(args, "graph_assoc_recent_owner_max_tracklet_len", 8)),
        "graph_assoc_min_box_iou": float(getattr(args, "graph_assoc_min_box_iou", 0.6)),
        "graph_assoc_reclaim_bonus": float(getattr(args, "graph_assoc_reclaim_bonus", 0.08)),
        "graph_assoc_recent_owner_penalty": float(getattr(args, "graph_assoc_recent_owner_penalty", 0.05)),
        "graph_assoc_iou_bonus": float(getattr(args, "graph_assoc_iou_bonus", 0.04)),
        "graph_assoc_score_bonus": float(getattr(args, "graph_assoc_score_bonus", 0.02)),
        "graph_assoc_min_assignment_gain": float(getattr(args, "graph_assoc_min_assignment_gain", 0.01)),
        "graph_assoc_max_cost_delta": float(getattr(args, "graph_assoc_max_cost_delta", 0.05)),
        "graph_assoc_force_match_cost": float(getattr(args, "graph_assoc_force_match_cost", 0.0)),
        "graph_assoc_allow_match_count_drop": bool(getattr(args, "graph_assoc_allow_match_count_drop", False)),
        "graph_assoc_commit_checkpoint": str(getattr(args, "graph_assoc_commit_checkpoint", "") or ""),
        "graph_assoc_commit_device": str(getattr(args, "graph_assoc_commit_device", "") or ""),
        "graph_assoc_commit_score_margin": float(getattr(args, "graph_assoc_commit_score_margin", 0.0)),
        "graph_assoc_commit_gate_only": bool(getattr(args, "graph_assoc_commit_gate_only", False)),
        "graph_assoc_commit_replace_rules": bool(getattr(args, "graph_assoc_commit_replace_rules", False)),
        "graph_assoc_commit_decision_mode": str(getattr(args, "graph_assoc_commit_decision_mode", "") or ""),
        "graph_assoc_commit_threshold": float(getattr(args, "graph_assoc_commit_threshold", float("nan"))),
        "graph_assoc_commit_neutral_risk_weight": float(getattr(args, "graph_assoc_commit_neutral_risk_weight", float("nan"))),
        "graph_assoc_commit_positive_threshold": float(getattr(args, "graph_assoc_commit_positive_threshold", float("nan"))),
        "graph_assoc_commit_neutral_threshold": float(getattr(args, "graph_assoc_commit_neutral_threshold", float("nan"))),
        "owneralt_competition_enable": bool(getattr(args, "owneralt_competition_enable", False)),
        "owneralt_competition_min_time_since_update": int(getattr(args, "owneralt_competition_min_time_since_update", 2)),
        "owneralt_competition_max_time_since_update": int(getattr(args, "owneralt_competition_max_time_since_update", 8)),
        "owneralt_competition_min_tracklet_len": int(getattr(args, "owneralt_competition_min_tracklet_len", 20)),
        "owneralt_competition_min_box_iou": float(getattr(args, "owneralt_competition_min_box_iou", 0.75)),
        "owneralt_competition_gap1_min_box_iou": float(getattr(args, "owneralt_competition_gap1_min_box_iou", -1.0)),
        "owneralt_competition_owner_max_tracklet_len": int(getattr(args, "owneralt_competition_owner_max_tracklet_len", 8)),
        "owneralt_competition_owner_alt_det_min_score": float(getattr(args, "owneralt_competition_owner_alt_det_min_score", 0.0)),
        "owneralt_competition_owner_alt_det_min_box_iou": float(getattr(args, "owneralt_competition_owner_alt_det_min_box_iou", 0.0)),
        "owneralt_competition_gap1_owner_alt_det_min_box_iou": float(getattr(args, "owneralt_competition_gap1_owner_alt_det_min_box_iou", -1.0)),
        "owneralt_competition_max_owner_edge_deficit": float(getattr(args, "owneralt_competition_max_owner_edge_deficit", 0.10)),
        "owneralt_competition_gap1_max_owner_edge_deficit": float(getattr(args, "owneralt_competition_gap1_max_owner_edge_deficit", -1.0)),
        "owneralt_competition_evidence_mode": str(getattr(args, "owneralt_competition_evidence_mode", "legacy")),
        "owneralt_competition_max_joint_penalty": float(getattr(args, "owneralt_competition_max_joint_penalty", -1.0)),
        "owneralt_competition_gap1_max_joint_penalty": float(getattr(args, "owneralt_competition_gap1_max_joint_penalty", -1.0)),
        "owneralt_competition_owner_alt_bonus": float(getattr(args, "owneralt_competition_owner_alt_bonus", 0.10)),
        "owneralt_competition_block_owner_on_reclaim": bool(getattr(args, "owneralt_competition_block_owner_on_reclaim", False)),
        "reentry_memory_enable": bool(getattr(args, "reentry_memory_enable", False)),
        "reentry_memory_use_low_score": bool(getattr(args, "reentry_memory_use_low_score", False)),
        "reentry_memory_compete_primary": bool(getattr(args, "reentry_memory_compete_primary", False)),
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
            "graph_assoc_enable": bool(getattr(args, "graph_assoc_enable", False)),
            "graph_assoc_top_k": int(getattr(args, "graph_assoc_top_k", 3)),
            "graph_assoc_max_rows": int(getattr(args, "graph_assoc_max_rows", 4)),
            "graph_assoc_max_cols": int(getattr(args, "graph_assoc_max_cols", 4)),
            "graph_assoc_row_margin": float(getattr(args, "graph_assoc_row_margin", 0.03)),
            "graph_assoc_col_margin": float(getattr(args, "graph_assoc_col_margin", 0.03)),
            "graph_assoc_min_reclaim_time_since_update": int(getattr(args, "graph_assoc_min_reclaim_time_since_update", 1)),
            "graph_assoc_max_reclaim_time_since_update": int(getattr(args, "graph_assoc_max_reclaim_time_since_update", 8)),
            "graph_assoc_min_reclaim_tracklet_len": int(getattr(args, "graph_assoc_min_reclaim_tracklet_len", 20)),
            "graph_assoc_recent_owner_max_time_since_update": int(getattr(args, "graph_assoc_recent_owner_max_time_since_update", 1)),
            "graph_assoc_recent_owner_max_tracklet_len": int(getattr(args, "graph_assoc_recent_owner_max_tracklet_len", 8)),
            "graph_assoc_min_box_iou": float(getattr(args, "graph_assoc_min_box_iou", 0.6)),
            "graph_assoc_reclaim_bonus": float(getattr(args, "graph_assoc_reclaim_bonus", 0.08)),
            "graph_assoc_recent_owner_penalty": float(getattr(args, "graph_assoc_recent_owner_penalty", 0.05)),
            "graph_assoc_iou_bonus": float(getattr(args, "graph_assoc_iou_bonus", 0.04)),
            "graph_assoc_score_bonus": float(getattr(args, "graph_assoc_score_bonus", 0.02)),
            "graph_assoc_min_assignment_gain": float(getattr(args, "graph_assoc_min_assignment_gain", 0.01)),
            "graph_assoc_max_cost_delta": float(getattr(args, "graph_assoc_max_cost_delta", 0.05)),
            "graph_assoc_force_match_cost": float(getattr(args, "graph_assoc_force_match_cost", 0.0)),
            "graph_assoc_allow_match_count_drop": bool(getattr(args, "graph_assoc_allow_match_count_drop", False)),
            "graph_assoc_commit_checkpoint": str(getattr(args, "graph_assoc_commit_checkpoint", "") or ""),
            "graph_assoc_commit_device": str(getattr(args, "graph_assoc_commit_device", "") or ""),
            "graph_assoc_commit_score_margin": float(getattr(args, "graph_assoc_commit_score_margin", 0.0)),
            "graph_assoc_commit_gate_only": bool(getattr(args, "graph_assoc_commit_gate_only", False)),
            "graph_assoc_commit_replace_rules": bool(getattr(args, "graph_assoc_commit_replace_rules", False)),
            "graph_assoc_commit_decision_mode": str(getattr(args, "graph_assoc_commit_decision_mode", "") or ""),
            "graph_assoc_commit_threshold": float(getattr(args, "graph_assoc_commit_threshold", float("nan"))),
            "graph_assoc_commit_neutral_risk_weight": float(getattr(args, "graph_assoc_commit_neutral_risk_weight", float("nan"))),
            "graph_assoc_commit_positive_threshold": float(getattr(args, "graph_assoc_commit_positive_threshold", float("nan"))),
            "graph_assoc_commit_neutral_threshold": float(getattr(args, "graph_assoc_commit_neutral_threshold", float("nan"))),
            "owneralt_competition_enable": bool(getattr(args, "owneralt_competition_enable", False)),
            "owneralt_competition_min_time_since_update": int(getattr(args, "owneralt_competition_min_time_since_update", 2)),
            "owneralt_competition_max_time_since_update": int(getattr(args, "owneralt_competition_max_time_since_update", 8)),
            "owneralt_competition_min_tracklet_len": int(getattr(args, "owneralt_competition_min_tracklet_len", 20)),
            "owneralt_competition_min_box_iou": float(getattr(args, "owneralt_competition_min_box_iou", 0.75)),
            "owneralt_competition_gap1_min_box_iou": float(getattr(args, "owneralt_competition_gap1_min_box_iou", -1.0)),
            "owneralt_competition_owner_max_tracklet_len": int(getattr(args, "owneralt_competition_owner_max_tracklet_len", 8)),
            "owneralt_competition_owner_alt_det_min_score": float(getattr(args, "owneralt_competition_owner_alt_det_min_score", 0.0)),
            "owneralt_competition_owner_alt_det_min_box_iou": float(getattr(args, "owneralt_competition_owner_alt_det_min_box_iou", 0.0)),
            "owneralt_competition_gap1_owner_alt_det_min_box_iou": float(getattr(args, "owneralt_competition_gap1_owner_alt_det_min_box_iou", -1.0)),
            "owneralt_competition_max_owner_edge_deficit": float(getattr(args, "owneralt_competition_max_owner_edge_deficit", 0.10)),
            "owneralt_competition_gap1_max_owner_edge_deficit": float(getattr(args, "owneralt_competition_gap1_max_owner_edge_deficit", -1.0)),
            "owneralt_competition_evidence_mode": str(getattr(args, "owneralt_competition_evidence_mode", "legacy")),
            "owneralt_competition_max_joint_penalty": float(getattr(args, "owneralt_competition_max_joint_penalty", -1.0)),
            "owneralt_competition_gap1_max_joint_penalty": float(getattr(args, "owneralt_competition_gap1_max_joint_penalty", -1.0)),
            "owneralt_competition_owner_alt_bonus": float(getattr(args, "owneralt_competition_owner_alt_bonus", 0.10)),
            "owneralt_competition_block_owner_on_reclaim": bool(getattr(args, "owneralt_competition_block_owner_on_reclaim", False)),
            "reentry_memory_enable": bool(getattr(args, "reentry_memory_enable", False)),
            "reentry_memory_use_low_score": bool(getattr(args, "reentry_memory_use_low_score", False)),
            "reentry_memory_compete_primary": bool(getattr(args, "reentry_memory_compete_primary", False)),
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


def write_single_row_csv(path, row):
    os.makedirs(osp.dirname(path), exist_ok=True)
    fieldnames = list(row.keys())
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def write_jsonl(path, rows):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def append_jsonl(path, rows):
    if not rows:
        return
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def append_lines(path, lines):
    if not lines:
        return
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.writelines(lines)


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
            img_info["img_path"] = img
            img = _read_image_robust(img)
        else:
            img_info["file_name"] = None

        if img is None:
            raise ValueError(
                "Empty image: ",
                img_info.get("img_path", img_info["file_name"]),
            )

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
    flush_interval = max(1, int(getattr(args, "incremental_write_interval", 20)))
    res_file = osp.join(vis_folder, args.name + ".txt")
    res_partial_file = res_file + ".partial"
    if osp.isfile(res_partial_file):
        os.remove(res_partial_file)
    open(res_partial_file, 'w', encoding='utf-8').close()
    pending_results = []

    owneralt_enabled = bool(getattr(args, "owneralt_competition_enable", False)) and hasattr(tracker, "get_owneralt_summary")
    owneralt_analysis_dir = args.owneralt_analysis_dir or osp.join(osp.dirname(vis_folder), "owneralt_analysis")
    owneralt_summary_path = osp.join(owneralt_analysis_dir, f"{args.name}_summary.csv")
    owneralt_event_path = osp.join(owneralt_analysis_dir, f"{args.name}_events.jsonl")
    owneralt_event_partial_path = owneralt_event_path + ".partial"
    if owneralt_enabled and osp.isfile(owneralt_event_partial_path):
        os.remove(owneralt_event_partial_path)
    reentry_enabled = bool(getattr(args, "reentry_memory_enable", False)) and hasattr(tracker, "get_reentry_summary")
    reentry_analysis_dir = osp.join(osp.dirname(vis_folder), "reentry_analysis")
    reentry_summary_path = osp.join(reentry_analysis_dir, f"{args.name}_summary.csv")
    tos_enabled = bool(getattr(args, "tos_enable", False)) and hasattr(tracker, "tos_stats")
    tos_analysis_only = bool(getattr(args, "tos_analysis_only", False))
    tos_analysis_dir = getattr(args, "tos_analysis_dir", "") or osp.join(osp.dirname(vis_folder), "tos_analysis")
    tos_analysis_path = osp.join(tos_analysis_dir, f"{args.name}_frames.csv")
    rgsa_enabled = bool(getattr(args, "rgsa_enable", False)) and hasattr(tracker, "get_rgsa_summary")
    tcgau_enabled = bool(getattr(args, "tcgau_enable", False))
    rgsa_or_tcgau = rgsa_enabled or tcgau_enabled
    rgsa_analysis_dir = osp.join(osp.dirname(vis_folder), "rgsa_analysis")
    rgsa_summary_path = osp.join(rgsa_analysis_dir, f"{args.name}_summary.csv")
    graph_assoc_enabled = bool(getattr(args, "graph_assoc_enable", False)) and hasattr(tracker, "get_graph_assoc_summary")
    graph_assoc_analysis_dir = args.graph_assoc_analysis_dir or osp.join(osp.dirname(vis_folder), "graph_assoc_analysis")
    graph_assoc_summary_path = osp.join(graph_assoc_analysis_dir, f"{args.name}_summary.csv")
    graph_assoc_event_path = osp.join(graph_assoc_analysis_dir, f"{args.name}_events.jsonl")
    graph_assoc_event_partial_path = graph_assoc_event_path + ".partial"
    graph_assoc_candidate_path = osp.join(graph_assoc_analysis_dir, f"{args.name}_candidates.jsonl")
    graph_assoc_candidate_partial_path = graph_assoc_candidate_path + ".partial"
    if graph_assoc_enabled and osp.isfile(graph_assoc_event_partial_path):
        os.remove(graph_assoc_event_partial_path)
    if graph_assoc_enabled and bool(getattr(args, "graph_assoc_dump_candidate_rows", False)) and osp.isfile(graph_assoc_candidate_partial_path):
        os.remove(graph_assoc_candidate_partial_path)
    engine_match_dump_enabled = bool(getattr(args, "reentry_engine_dump_matches", False)) and hasattr(tracker, "reentry_engine") and tracker.reentry_engine is not None
    engine_match_dump_dir = osp.join(osp.dirname(vis_folder), "engine_match_dumps")
    engine_match_dump_path = osp.join(engine_match_dump_dir, f"{args.name}_matches.jsonl") if engine_match_dump_enabled else ""
    engine_match_dump_partial_path = engine_match_dump_path + ".partial" if engine_match_dump_enabled else ""
    if engine_match_dump_enabled and osp.isfile(engine_match_dump_partial_path):
        os.remove(engine_match_dump_partial_path)
    bc_trace_enabled = bool(getattr(args, "bc_dump_trace", False)) and hasattr(tracker, "bc_stats") and tracker.bc_stats.get("enabled", False)
    bc_trace_dir = osp.join(osp.dirname(vis_folder), "bc_traces")
    bc_trace_path = osp.join(bc_trace_dir, f"{args.name}_bc_traces.jsonl") if bc_trace_enabled else ""
    bc_trace_partial_path = bc_trace_path + ".partial" if bc_trace_enabled else ""
    if bc_trace_enabled and osp.isfile(bc_trace_partial_path):
        os.remove(bc_trace_partial_path)

    def flush_tracking_checkpoint(frame_id, final=False):
        append_lines(res_partial_file, pending_results)
        pending_results.clear()
        if owneralt_enabled:
            owneralt_events = tracker.drain_owneralt_event_rows() if hasattr(tracker, "drain_owneralt_event_rows") else []
            append_jsonl(owneralt_event_partial_path, owneralt_events)
            owneralt_summary = tracker.get_owneralt_summary()
            owneralt_summary.update(
                {
                    "seq_name": args.name,
                    "benchmark": args.benchmark,
                    "split_to_eval": args.split_to_eval,
                    "owneralt_competition_enable": bool(getattr(args, "owneralt_competition_enable", False)),
                    "with_reid": bool(getattr(args, "with_reid", False)),
                    "result_file": res_file if final else res_partial_file,
                    "num_frames": int(num_frames),
                    "checkpoint_frame": int(frame_id),
                    "checkpoint_complete": int(bool(final)),
                }
            )
            owneralt_summary["event_jsonl"] = owneralt_event_path if final else owneralt_event_partial_path
            write_single_row_csv(owneralt_summary_path, owneralt_summary)
        if reentry_enabled:
            reentry_summary = tracker.get_reentry_summary()
            reentry_summary.update(
                {
                    "seq_name": args.name,
                    "benchmark": args.benchmark,
                    "split_to_eval": args.split_to_eval,
                    "reentry_memory_enable": bool(getattr(args, "reentry_memory_enable", False)),
                    "reentry_memory_use_low_score": bool(getattr(args, "reentry_memory_use_low_score", False)),
                    "reentry_memory_compete_primary": bool(getattr(args, "reentry_memory_compete_primary", False)),
                    "with_reid": bool(getattr(args, "with_reid", False)),
                    "result_file": res_file if final else res_partial_file,
                    "num_frames": int(num_frames),
                    "checkpoint_frame": int(frame_id),
                    "checkpoint_complete": int(bool(final)),
                }
            )
            reentry_summary["reentry_confirm_streak"] = int(reentry_summary.get("confirm_streak", 0))
            reentry_summary["reentry_confirm_gap"] = int(reentry_summary.get("confirm_gap", 0))
            reentry_summary["reentry_confirm_min_similarity"] = float(reentry_summary.get("confirm_min_similarity", 0.0))
            if hasattr(tracker, "bc_stats"):
                reentry_summary["bc_promote"] = json.dumps(tracker.bc_stats)
            write_single_row_csv(reentry_summary_path, reentry_summary)
        if rgsa_or_tcgau:
            rgsa_summary = tracker.get_rgsa_summary()
            rgsa_summary.update(
                {
                    "seq_name": args.name,
                    "benchmark": args.benchmark,
                    "split_to_eval": args.split_to_eval,
                    "rgsa_enable": bool(getattr(args, "rgsa_enable", False)),
                    "with_reid": bool(getattr(args, "with_reid", False)),
                    "laplace_assoc": bool(getattr(args, "laplace_assoc", False)),
                    "result_file": res_file if final else res_partial_file,
                    "num_frames": int(num_frames),
                    "checkpoint_frame": int(frame_id),
                    "checkpoint_complete": int(bool(final)),
                }
            )
            write_single_row_csv(rgsa_summary_path, rgsa_summary)
        if bc_trace_enabled and hasattr(tracker, "drain_bc_trace_log"):
            bc_traces = tracker.drain_bc_trace_log()
            if bc_traces:
                append_jsonl(bc_trace_partial_path, bc_traces)
        if engine_match_dump_enabled and hasattr(tracker.reentry_engine, "drain_match_log"):
            match_rows = tracker.reentry_engine.drain_match_log()
            if match_rows:
                append_jsonl(engine_match_dump_partial_path, match_rows)
        if graph_assoc_enabled:
            graph_assoc_events = tracker.drain_graph_assoc_event_rows() if hasattr(tracker, "drain_graph_assoc_event_rows") else []
            append_jsonl(graph_assoc_event_partial_path, graph_assoc_events)
            if bool(getattr(args, "graph_assoc_dump_candidate_rows", False)):
                graph_assoc_candidates = tracker.drain_graph_assoc_candidate_rows() if hasattr(tracker, "drain_graph_assoc_candidate_rows") else []
                append_jsonl(graph_assoc_candidate_partial_path, graph_assoc_candidates)
            graph_assoc_summary = tracker.get_graph_assoc_summary()
            graph_assoc_summary.update(
                {
                    "seq_name": args.name,
                    "benchmark": args.benchmark,
                    "split_to_eval": args.split_to_eval,
                    "graph_assoc_enable": bool(getattr(args, "graph_assoc_enable", False)),
                    "graph_assoc_commit_checkpoint": str(getattr(args, "graph_assoc_commit_checkpoint", "") or ""),
                    "graph_assoc_commit_device": str(getattr(args, "graph_assoc_commit_device", "") or ""),
                    "graph_assoc_commit_score_margin": float(getattr(args, "graph_assoc_commit_score_margin", 0.0)),
                    "graph_assoc_commit_gate_only": bool(getattr(args, "graph_assoc_commit_gate_only", False)),
                    "graph_assoc_commit_replace_rules": bool(getattr(args, "graph_assoc_commit_replace_rules", False)),
                    "graph_assoc_commit_decision_mode": str(getattr(args, "graph_assoc_commit_decision_mode", "") or ""),
                    "graph_assoc_commit_threshold": float(getattr(args, "graph_assoc_commit_threshold", float("nan"))),
                    "graph_assoc_commit_neutral_risk_weight": float(getattr(args, "graph_assoc_commit_neutral_risk_weight", float("nan"))),
                    "graph_assoc_commit_positive_threshold": float(getattr(args, "graph_assoc_commit_positive_threshold", float("nan"))),
                    "graph_assoc_commit_neutral_threshold": float(getattr(args, "graph_assoc_commit_neutral_threshold", float("nan"))),
                    "with_reid": bool(getattr(args, "with_reid", False)),
                    "result_file": res_file if final else res_partial_file,
                    "num_frames": int(num_frames),
                    "checkpoint_frame": int(frame_id),
                    "checkpoint_complete": int(bool(final)),
                }
            )
            graph_assoc_summary["event_jsonl"] = graph_assoc_event_path if final else graph_assoc_event_partial_path
            graph_assoc_summary["candidate_jsonl"] = (
                graph_assoc_candidate_path if final else graph_assoc_candidate_partial_path
            ) if bool(getattr(args, "graph_assoc_dump_candidate_rows", False)) else ""
            write_single_row_csv(graph_assoc_summary_path, graph_assoc_summary)

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
                    pending_results.append(
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
        if frame_id % flush_interval == 0:
            flush_tracking_checkpoint(frame_id)

    flush_tracking_checkpoint(num_frames)
    os.replace(res_partial_file, res_file)
    logger.info(f"save results to {res_file}")
    fcaa_summary = tracker.get_fcaa_summary() if hasattr(tracker, "get_fcaa_summary") else {}
    fcaa_summary.update(
        {
            "seq_name": args.name,
            "benchmark": args.benchmark,
            "split_to_eval": args.split_to_eval,
            "fcaa_enable": bool(getattr(args, "fcaa_enable", False)),
            "fgas_enable": bool(getattr(args, "fgas_enable", False)),
            "with_reid": bool(getattr(args, "with_reid", False)),
            "result_file": res_file,
            "num_frames": int(num_frames),
        }
    )
    analysis_dir = args.fcaa_analysis_dir or osp.join(osp.dirname(vis_folder), "fcaa_analysis")
    summary_path = osp.join(analysis_dir, f"{args.name}_summary.csv")
    trigger_path = osp.join(analysis_dir, f"{args.name}_triggers.jsonl")
    fcaa_summary["trigger_jsonl"] = trigger_path
    write_single_row_csv(summary_path, fcaa_summary)
    trigger_rows = tracker.get_fcaa_trigger_rows() if hasattr(tracker, "get_fcaa_trigger_rows") else []
    write_jsonl(trigger_path, trigger_rows)
    logger.info(f"save FCAA analysis to {summary_path}")
    if bool(getattr(args, "fgas_enable", False)) and hasattr(tracker, "get_fgas_summary"):
        fgas_summary = tracker.get_fgas_summary()
        fgas_summary.update(
            {
                "seq_name": args.name,
                "benchmark": args.benchmark,
                "split_to_eval": args.split_to_eval,
                "fgas_enable": bool(getattr(args, "fgas_enable", False)),
                "with_reid": bool(getattr(args, "with_reid", False)),
                "result_file": res_file,
                "num_frames": int(num_frames),
            }
        )
        fgas_analysis_dir = osp.join(osp.dirname(vis_folder), "fgas_analysis")
        fgas_summary_path = osp.join(fgas_analysis_dir, f"{args.name}_summary.csv")
        write_single_row_csv(fgas_summary_path, fgas_summary)
        logger.info(f"save FGAS analysis to {fgas_summary_path}")
    if bool(getattr(args, "owneralt_competition_enable", False)) and hasattr(tracker, "get_owneralt_summary"):
        flush_tracking_checkpoint(num_frames, final=True)
        if osp.isfile(owneralt_event_partial_path):
            os.replace(owneralt_event_partial_path, owneralt_event_path)
        elif not osp.isfile(owneralt_event_path):
            write_jsonl(owneralt_event_path, [])
        logger.info(f"save OwnerAlt analysis to {owneralt_summary_path}")
    if reentry_enabled:
        flush_tracking_checkpoint(num_frames, final=True)
        logger.info(f"save re-entry analysis to {reentry_summary_path}")
    if tos_enabled:
        flush_tracking_checkpoint(num_frames, final=True)
        tos_summary = tracker.get_tos_summary() if hasattr(tracker, "get_tos_summary") else {}
        tos_summary["seq_name"] = str(args.name)
        tos_summary["benchmark"] = str(args.benchmark)
        tos_summary["split_to_eval"] = str(args.split_to_eval)
        tos_summary["num_frames"] = int(num_frames)
        tos_summary["result_file"] = str(res_file) if res_file else ""
        tos_summary_dir = osp.join(osp.dirname(vis_folder), "tos_analysis")
        tos_summary_path = osp.join(tos_summary_dir, f"{args.name}_summary.csv")
        os.makedirs(tos_summary_dir, exist_ok=True)
        write_single_row_csv(tos_summary_path, tos_summary)
        logger.info(f"save TOS analysis to {tos_summary_path}")
    if rgsa_or_tcgau:
        flush_tracking_checkpoint(num_frames, final=True)
        logger.info(f"save RGSA/TCGAU analysis to {rgsa_summary_path}")
        # Save TCGAU pair log if requested
        if tcgau_enabled and getattr(tracker, "tcgau_log_pairs", False) and getattr(tracker, "tcgau_pair_log", None):
            tcgau_log_dir = osp.join(osp.dirname(vis_folder), "rgsa_analysis")
            os.makedirs(tcgau_log_dir, exist_ok=True)
            tcgau_pairs_path = osp.join(tcgau_log_dir, f"{args.name}_tcgau_pairs.csv")
            if tracker.tcgau_pair_log:
                import csv as _csv
                with open(tcgau_pairs_path, "w", newline="") as _f:
                    _w = _csv.DictWriter(_f, fieldnames=list(tracker.tcgau_pair_log[0].keys()))
                    _w.writeheader()
                    _w.writerows(tracker.tcgau_pair_log)
                logger.info(f"save TCGAU pair log ({len(tracker.tcgau_pair_log)} pairs) to {tcgau_pairs_path}")
    # TOS-Track: drain analysis rows to CSV
    if tos_enabled and hasattr(tracker, "_tos_analysis_rows"):
        import csv as _csv
        os.makedirs(tos_analysis_dir, exist_ok=True)
        rows = getattr(tracker, "_tos_analysis_rows", [])
        if rows:
            with open(tos_analysis_path, "w", newline="") as _f:
                _w = _csv.DictWriter(_f, fieldnames=list(rows[0].keys()))
                _w.writeheader()
                _w.writerows(rows)
            logger.info(f"save TOS analysis ({len(rows)} rows) to {tos_analysis_path}")
        else:
            # Write empty file with headers
            _header = ["frame", "track_id", "det_id", "track_state", "track_len", "hist_len",
                        "gap", "det_score", "app_sim", "pair_rel", "q_update", "update_mode",
                        "tos_occlusion", "tos_shadow", "tos_shadow_dur",
                        "haca_active", "haca_margin", "haca_entropy", "haca_bg"]
            with open(tos_analysis_path, "w", newline="") as _f:
                _f.write(",".join(_header) + "\n")
            logger.info(f"save TOS analysis (0 rows) to {tos_analysis_path}")
        # Clear rows after draining
        tracker._tos_analysis_rows = []
    if bool(getattr(args, "graph_assoc_enable", False)) and hasattr(tracker, "get_graph_assoc_summary"):
        flush_tracking_checkpoint(num_frames, final=True)
        if osp.isfile(graph_assoc_event_partial_path):
            os.replace(graph_assoc_event_partial_path, graph_assoc_event_path)
        elif not osp.isfile(graph_assoc_event_path):
            write_jsonl(graph_assoc_event_path, [])
        if bool(getattr(args, "graph_assoc_dump_candidate_rows", False)):
            if osp.isfile(graph_assoc_candidate_partial_path):
                os.replace(graph_assoc_candidate_partial_path, graph_assoc_candidate_path)
            elif not osp.isfile(graph_assoc_candidate_path):
                write_jsonl(graph_assoc_candidate_path, [])
        logger.info(f"save graph association analysis to {graph_assoc_summary_path}")
    if engine_match_dump_enabled:
        if osp.isfile(engine_match_dump_partial_path):
            os.replace(engine_match_dump_partial_path, engine_match_dump_path)
        elif not osp.isfile(engine_match_dump_path):
            write_jsonl(engine_match_dump_path, [])
        logger.info(f"save engine match dump to {engine_match_dump_path}")
    if bc_trace_enabled:
        if osp.isfile(bc_trace_partial_path):
            os.replace(bc_trace_partial_path, bc_trace_path)
        elif not osp.isfile(bc_trace_path):
            write_jsonl(bc_trace_path, [])
        logger.info(f"save bc trace to {bc_trace_path}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    vis_folder = osp.join(output_dir, "track_results")
    os.makedirs(vis_folder, exist_ok=True)
    result_file = osp.join(vis_folder, args.name + ".txt")

    if bool(getattr(args, "skip_existing_results", False)) and osp.isfile(result_file):
        logger.info(f"Skipping existing result file {result_file}")
        return

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

    try:
        image_track(predictor, vis_folder, args)
    finally:
        # Release per-sequence state before the next sequence is evaluated in the
        # same Python process. This prevents CUDA cache growth across MOT20-02/MOT20-05
        # style multi-sequence runs and keeps the wrapper eval stable.
        del predictor
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
