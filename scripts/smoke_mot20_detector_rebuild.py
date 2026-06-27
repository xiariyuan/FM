#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]
BOT = REPO / 'external' / 'BoT-SORT-main'
sys.path.insert(0, str(BOT))
from yolox.exp import get_exp
from yolox.utils import load_ckpt

EXP_FILE = BOT / 'yolox/exps/example/mot/yolox_x_mot20_rebuild_trainhalf.py'
CKPT = BOT / 'pretrained/bytetrack_x_mot20.pth.tar'

os.environ.setdefault('MOT20_DET_SMOKE_TRAIN_IMAGES', '8')
os.environ.setdefault('MOT20_DET_SMOKE_VAL_IMAGES', '4')
os.environ.setdefault('MOT20_DET_WORKERS', '0')
os.environ.setdefault('MOT20_DET_MAX_EPOCH', '1')

exp = get_exp(str(EXP_FILE), None)
print('exp', exp.exp_name, 'train_ann', exp.train_ann, 'val_ann', exp.val_ann, 'input', exp.input_size, 'test', exp.test_size)
loader = exp.get_data_loader(batch_size=1, is_distributed=False, no_aug=True)
t0 = time.time()
imgs, targets, info, ids = next(iter(loader))
print('train_batch', tuple(imgs.shape), tuple(targets.shape), 'ids', ids.flatten()[:4].tolist(), 'sec', round(time.time()-t0, 3))
vloader = exp.get_eval_loader(batch_size=1, is_distributed=False)
t0 = time.time()
vim, _, vinfo, vids = next(iter(vloader))
print('val_batch', tuple(vim.shape), 'vids', vids.flatten()[:4].tolist(), 'info', vinfo[:2] if isinstance(vinfo, tuple) else type(vinfo), 'sec', round(time.time()-t0, 3))
model = exp.get_model()
model.eval()
ckpt = torch.load(str(CKPT), map_location='cpu')
state = ckpt.get('model', ckpt)
missing, unexpected = model.load_state_dict(state, strict=False)
print('model_loaded', 'missing', len(missing), 'unexpected', len(unexpected))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
with torch.no_grad():
    out = model(imgs.float().to(device))
print('forward_ok', type(out).__name__, 'device', device)
