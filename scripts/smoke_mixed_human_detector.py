#!/usr/bin/env python3
import os, sys, torch
sys.path.insert(0, 'external/BoT-SORT-main')
from yolox.exp import get_exp

exp_file='external/BoT-SORT-main/yolox/exps/example/mot/yolox_x_mixed_human_quick.py'
exp=get_exp(exp_file, None)
print('exp', exp.exp_name, 'input', exp.input_size, 'test', exp.test_size, 'train_ann', exp.train_ann, 'val_ann', exp.val_ann)
loader=exp.get_data_loader(batch_size=int(os.getenv('SMOKE_BATCH','1')), is_distributed=False, no_aug=False)
imgs, targets, infos, ids = next(iter(loader))
print('batch imgs', tuple(imgs.shape), imgs.dtype, 'targets', tuple(targets.shape), targets.dtype)
print('infos sample', infos[:2] if isinstance(infos, list) else infos)
model=exp.get_model().cuda().train()
imgs=imgs.cuda(non_blocking=True).float()
targets=targets.cuda(non_blocking=True).float()
with torch.cuda.amp.autocast(enabled=True):
    loss=model(imgs, targets)
print('loss keys', sorted(loss.keys()))
print('total_loss', float(loss['total_loss'].detach().cpu()))
print('OK')
