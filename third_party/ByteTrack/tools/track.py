from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
from yolox.evaluators import MOTEvaluator

import argparse
import os
import random
import warnings
import glob
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="local rank for dist training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # det args
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument(
        "--use-local-conflict",
        dest="use_local_conflict",
        default=False,
        action="store_true",
        help="Enable local-conflict partial-commit plugin before official ByteTrack first-stage Hungarian.",
    )
    parser.add_argument(
        "--local-conflict-checkpoint",
        type=str,
        default="",
        help="Checkpoint for the local-conflict set predictor.",
    )
    parser.add_argument("--local-conflict-topk", type=int, default=8)
    parser.add_argument("--local-conflict-min-detections", type=int, default=2)
    parser.add_argument("--local-conflict-min-committed-matches", type=int, default=1)
    parser.add_argument("--local-conflict-max-detections", type=int, default=8)
    parser.add_argument("--local-conflict-max-tracks", type=int, default=32)
    parser.add_argument("--local-conflict-cluster-gate-thresh", type=float, default=0.5)
    parser.add_argument("--local-conflict-cluster-gate-temp", type=float, default=1.0)
    parser.add_argument("--local-conflict-cluster-gate-bias", type=float, default=0.0)
    parser.add_argument(
        "--local-conflict-max-commits-per-cluster",
        type=int,
        default=1,
        help="Conservative cap on committed pairs per cluster.",
    )
    parser.add_argument(
        "--local-conflict-replacement-budget-ratio",
        type=float,
        default=0.05,
        help="Global fuse limiting plugin replacement coverage across eligible clusters.",
    )
    parser.add_argument("--local-conflict-max-replaced-clusters", type=int, default=0)
    parser.add_argument("--local-conflict-min-commit-margin", type=float, default=0.05)
    parser.add_argument(
        "--local-conflict-host-variant",
        type=str,
        default="official_bytetrack",
        help="Host token passed to the local-conflict model.",
    )
    parser.add_argument(
        "--local-conflict-diagnostics-dir",
        type=str,
        default="",
        help="Optional output directory for per-sequence local-conflict diagnostics JSON files.",
    )
    parser.add_argument(
        "--local-conflict-dump-dir",
        type=str,
        default="",
        help="Optional output directory for official ByteTrack first-stage runtime candidate dumps.",
    )
    parser.add_argument(
        "--local-conflict-dump-topk",
        type=int,
        default=8,
        help="Top-k first-stage candidates to dump per detection. Use <=0 to dump all candidates.",
    )
    parser.add_argument(
        "--local-conflict-dump-min-score",
        type=float,
        default=0.0,
        help="Minimum refined score required for a dumped candidate row.",
    )
    parser.add_argument(
        "--use-posthost-oracle-edit",
        dest="use_posthost_oracle_edit",
        default=False,
        action="store_true",
        help="Enable post-host one-edit oracle ceiling after official ByteTrack first-stage matching.",
    )
    parser.add_argument(
        "--use-posthost-hierarchical-edit",
        dest="use_posthost_hierarchical_edit",
        default=False,
        action="store_true",
        help="Enable learned post-host hierarchical one-edit plugin after official ByteTrack first-stage matching.",
    )
    parser.add_argument(
        "--use-posthost-rule-edit",
        dest="use_posthost_rule_edit",
        default=False,
        action="store_true",
        help="Enable test-legal rule-based post-host one-edit plugin after official ByteTrack first-stage matching.",
    )
    parser.add_argument(
        "--posthost-oracle-data-root",
        type=str,
        default="",
        help="Dataset root used to load MOT17 half-val GT for post-host oracle evaluation.",
    )
    parser.add_argument(
        "--posthost-oracle-min-iou",
        type=float,
        default=0.5,
        help="Minimum IoU required to attach a detection to GT for post-host oracle evaluation.",
    )
    parser.add_argument(
        "--posthost-oracle-allowed-actions",
        type=str,
        default="all",
        help="Oracle post-host action filter. Supported: all, defer_only.",
    )
    parser.add_argument(
        "--posthost-hierarchical-keep-thresh",
        type=float,
        default=0.97,
        help="Probability threshold for keep-vs-edit gate in learned post-host hierarchical mode.",
    )
    parser.add_argument(
        "--posthost-hierarchical-swap-thresh",
        type=float,
        default=0.5,
        help="Probability threshold for choosing swap over defer in learned post-host hierarchical mode.",
    )
    parser.add_argument(
        "--posthost-hierarchical-candidate-min-refined-score",
        type=float,
        default=0.10,
        help="Minimum refined score required for non-host add/swap candidates in learned post-host hierarchical mode.",
    )
    parser.add_argument(
        "--posthost-hierarchical-host-summary-prior-alpha",
        type=float,
        default=0.0,
        help="Weak prior coefficient applied to runtime host-summary positive_count/score in learned post-host hierarchical mode.",
    )
    parser.add_argument(
        "--posthost-rule-large-only",
        dest="posthost_rule_large_only",
        default=True,
        action="store_true",
        help="Restrict rule-based post-host edits to large local components only.",
    )
    parser.add_argument(
        "--posthost-rule-allow-small",
        dest="posthost_rule_large_only",
        action="store_false",
        help="Allow rule-based post-host edits on small local components as well.",
    )
    parser.add_argument(
        "--posthost-rule-defer-refined-max",
        type=float,
        default=0.35,
        help="Maximum host refined score allowed for a rule-based defer candidate.",
    )
    parser.add_argument(
        "--posthost-rule-defer-iou-max",
        type=float,
        default=0.55,
        help="Maximum host IoU score allowed for a rule-based defer candidate.",
    )
    parser.add_argument(
        "--posthost-rule-defer-row-margin-max",
        type=float,
        default=0.15,
        help="Maximum host row-margin allowed for a rule-based defer candidate.",
    )
    parser.add_argument(
        "--posthost-rule-defer-track-hist-max",
        type=float,
        default=4.2,
        help="Maximum host track-history feature allowed for a rule-based defer candidate.",
    )
    parser.add_argument(
        "--posthost-rule-require-second-stage-rescue",
        dest="posthost_rule_require_second_stage_rescue",
        default=False,
        action="store_true",
        help="Require a deferred track to have a viable second-stage rescue path before the rule plugin can drop the host pair.",
    )
    parser.add_argument(
        "--posthost-rule-second-stage-iou-min",
        type=float,
        default=0.5,
        help="Minimum second-stage IoU rescue score required when second-stage rescue gating is enabled.",
    )
    parser.add_argument(
        "--posthost-rule-unconfirmed-fuse-min",
        type=float,
        default=0.0,
        help="Optional unconfirmed-track fused-sim fallback required when second-stage rescue gating is enabled.",
    )
    parser.add_argument(
        "--posthost-rule-scorecard-json",
        type=str,
        default="",
        help="Optional oracle-guided linear scorecard JSON used by the post-host rule plugin.",
    )
    parser.add_argument(
        "--posthost-rule-score-thresh",
        type=float,
        default=0.0,
        help="Optional probability threshold override for the post-host rule scorecard. <=0 uses the JSON default.",
    )
    parser.add_argument(
        "--posthost-rule-use-legacy-prefilter",
        dest="posthost_rule_use_legacy_prefilter",
        default=True,
        action="store_true",
        help="Keep the legacy refined/iou/margin/history prefilter before scorecard ranking.",
    )
    parser.add_argument(
        "--posthost-rule-no-legacy-prefilter",
        dest="posthost_rule_use_legacy_prefilter",
        action="store_false",
        help="Disable the legacy prefilter and rely on the scorecard gate for defer candidate selection.",
    )
    return parser


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    cudnn.benchmark = True

    rank = args.local_rank
    # rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    results_folder = os.path.join(file_name, "track_results")
    os.makedirs(results_folder, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    #logger.info("Model Structure:\n{}".format(str(model)))

    val_loader = exp.get_eval_loader(args.batch_size, is_distributed, args.test)
    evaluator = MOTEvaluator(
        args=args,
        dataloader=val_loader,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        num_classes=exp.num_classes,
        )

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    # start evaluate
    *_, summary = evaluator.evaluate(
        model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder
    )
    logger.info("\n" + summary)

    # evaluate MOTA
    mm.lap.default_solver = 'lap'

    if exp.val_ann == 'val_half.json':
        gt_type = '_val_half'
    else:
        gt_type = ''
    print('gt_type', gt_type)
    if args.mot20:
        gtfiles = glob.glob(os.path.join('datasets/MOT20/train', '*/gt/gt{}.txt'.format(gt_type)))
    else:
        gtfiles = glob.glob(os.path.join('datasets/mot/train', '*/gt/gt{}.txt'.format(gt_type)))
    print('gt_files', gtfiles)
    tsfiles = [f for f in glob.glob(os.path.join(results_folder, '*.txt')) if not os.path.basename(f).startswith('eval')]

    logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logger.info('Loading files.')
    
    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in tsfiles])    
    
    mh = mm.metrics.create()    
    accs, names = compare_dataframes(gt, ts)
    
    logger.info('Running metrics')
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
               'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
               'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    # summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    # print(mm.io.render_summary(
    #   summary, formatters=mh.formatters, 
    #   namemap=mm.io.motchallenge_metric_names))
    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                       'partially_tracked', 'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']
    print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logger.info('Completed')


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )
