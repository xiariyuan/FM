"""
@Author: Du Yunhao
@Filename: opts.py
@Contact: dyh_bupt@163.com
@Time: 2022/2/28 19:41
@Discription: opts
"""
import json
import argparse
from os.path import join

data = {
    'MOT17': {
        'val':[
            'MOT17-02-FRCNN',
            'MOT17-04-FRCNN',
            'MOT17-05-FRCNN',
            'MOT17-09-FRCNN',
            'MOT17-10-FRCNN',
            'MOT17-11-FRCNN',
            'MOT17-13-FRCNN'
        ],
        'test':[
            'MOT17-01-FRCNN',
            'MOT17-03-FRCNN',
            'MOT17-06-FRCNN',
            'MOT17-07-FRCNN',
            'MOT17-08-FRCNN',
            'MOT17-12-FRCNN',
            'MOT17-14-FRCNN'
        ]
    },
    'MOT20': {
        'test':[
            'MOT20-04',
            'MOT20-06',
            'MOT20-07',
            'MOT20-08'
        ]
    }
}

class opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            'dataset',
            type=str,
            help='MOT17 or MOT20',
        )
        self.parser.add_argument(
            'mode',
            type=str,
            help='val or test',
        )
        self.parser.add_argument(
            '--sequences',
            nargs='*',
            default=[],
            help='Optional explicit sequence list override'
        )
        self.parser.add_argument(
            '--BoT',
            action='store_true',
            help='Replacing the original feature extractor with BoT'
        )
        self.parser.add_argument(
            '--ECC',
            action='store_true',
            help='CMC model'
        )
        self.parser.add_argument(
            '--NSA',
            action='store_true',
            help='NSA Kalman filter'
        )
        self.parser.add_argument(
            '--EMA',
            action='store_true',
            help='EMA feature updating mechanism'
        )
        self.parser.add_argument(
            '--MC',
            action='store_true',
            help='Matching with both appearance and motion cost'
        )
        self.parser.add_argument(
            '--woC',
            action='store_true',
            help='Replace the matching cascade with vanilla matching'
        )
        self.parser.add_argument(
            '--AFLink',
            action='store_true',
            help='Appearance-Free Link'
        )
        self.parser.add_argument(
            '--GSI',
            action='store_true',
            help='Gaussian-smoothed Interpolation'
        )
        self.parser.add_argument(
            '--root_dataset',
            default='/data/dyh/data/MOTChallenge'
        )
        self.parser.add_argument(
            '--path_AFLink',
            default='/data/dyh/results/StrongSORT_Git/AFLink_epoch20.pth'
        )
        self.parser.add_argument(
            '--dir_save',
            default='/data/dyh/results/StrongSORT_Git/tmp'
        )
        self.parser.add_argument(
            '--EMA_alpha',
            default=0.9
        )
        self.parser.add_argument(
            '--MC_lambda',
            default=0.98
        )
        self.parser.add_argument(
            '--LAPLACE',
            action='store_true',
            help='Laplace-guided temporal reliability plug-in for appearance matching'
        )
        self.parser.add_argument(
            '--laplace_weight',
            type=float,
            default=0.35
        )
        self.parser.add_argument(
            '--laplace_decay_scales',
            nargs='+',
            type=float,
            default=[1.0, 2.0, 4.0]
        )
        self.parser.add_argument(
            '--laplace_min_history',
            type=int,
            default=3
        )
        self.parser.add_argument(
            '--laplace_max_history',
            type=int,
            default=30
        )
        self.parser.add_argument(
            '--laplace-calibrator',
            dest='laplace_calibrator',
            type=str,
            default=''
        )
        self.parser.add_argument(
            '--laplace-assoc-mode',
            dest='laplace_assoc_mode',
            type=str,
            default='auto',
            choices=['auto', 'heuristic', 'current_learned', 'haca_v1', 'haca_v2', 'haca_v3'],
            help='association-time mode: heuristic/current learned/HACA'
        )
        self.parser.add_argument(
            '--laplace-haca-checkpoint',
            dest='laplace_haca_checkpoint',
            type=str,
            default='',
            help='HACA checkpoint (.npz) for primary association'
        )
        self.parser.add_argument(
            '--laplace-haca-no-set-encoder',
            dest='laplace_haca_no_set_encoder',
            action='store_true',
            help='Disable HACA candidate-set encoder'
        )
        self.parser.add_argument(
            '--laplace-haca-no-background',
            dest='laplace_haca_no_background',
            action='store_true',
            help='Disable HACA background head'
        )
        self.parser.add_argument(
            '--laplace-haca-delta-scale',
            dest='laplace_haca_delta_scale',
            type=float,
            default=float('nan'),
            help='Optional override for HACA residual scale'
        )
        self.parser.add_argument(
            '--laplace-disable-pole-bank',
            dest='laplace_disable_pole_bank',
            action='store_true',
            help='Disable learned pole-bank gating and use fixed temporal prototypes instead'
        )
        self.parser.add_argument(
            '--laplace-analysis-dir',
            dest='laplace_analysis_dir',
            type=str,
            default=''
        )
        self.parser.add_argument(
            '--dir_dets',
            type=str,
            default=''
        )
        self.parser.add_argument(
            '--path_ECC',
            type=str,
            default=''
        )

    def parse(self, args=''):
        if args == '':
          opt = self.parser.parse_args()
        else:
          opt = self.parser.parse_args(args)
        opt.min_confidence = 0.6
        opt.nms_max_overlap = 1.0
        opt.min_detection_height = 0
        if opt.BoT:
            opt.max_cosine_distance = 0.4
            default_dir_dets = '/data/dyh/results/StrongSORT_Git/{}_{}_YOLOX+BoT'.format(opt.dataset, opt.mode)
        else:
            opt.max_cosine_distance = 0.3
            default_dir_dets = '/data/dyh/results/StrongSORT_Git/{}_{}_YOLOX+simpleCNN'.format(opt.dataset, opt.mode)
        if not opt.dir_dets:
            opt.dir_dets = default_dir_dets
        if opt.MC:
            opt.max_cosine_distance += 0.05
        if opt.EMA:
            opt.nn_budget = 1
        else:
            opt.nn_budget = 100
        if opt.ECC:
            path_ECC = opt.path_ECC if opt.path_ECC else '/data/dyh/results/StrongSORT_Git/{}_ECC_{}.json'.format(opt.dataset, opt.mode)
            opt.ecc = json.load(open(path_ECC))
        opt.sequences = opt.sequences if opt.sequences else data[opt.dataset][opt.mode]
        opt.dir_dataset = join(
            opt.root_dataset,
            opt.dataset,
            'train' if opt.mode == 'val' else 'test'
        )
        return opt

opt = opts().parse()
