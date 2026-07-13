#!/usr/bin/env python3
import argparse, sys, torch
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'external/BoT-SORT-main'))
from yolox.exp import get_exp
from yolox.utils import load_ckpt

ap=argparse.ArgumentParser()
ap.add_argument('-f','--exp-file', required=True)
ap.add_argument('-c','--ckpt', required=True)
ap.add_argument('-b','--batch-size', type=int, default=1)
ap.add_argument('--half', action='store_true')
args=ap.parse_args()
exp=get_exp(args.exp_file, None)
model=exp.get_model().cuda().eval()
ckpt=torch.load(args.ckpt, map_location='cuda')
state=ckpt.get('model', ckpt)
model=load_ckpt(model, state)
if args.half:
    model=model.half()
evaluator=exp.get_evaluator(args.batch_size, is_distributed=False, testdev=False)
ap5095, ap50, summary=evaluator.evaluate(model, distributed=False, half=args.half, test_size=exp.test_size)
print(summary)
print(f'RESULT AP={ap5095:.6f} AP50={ap50:.6f}')
