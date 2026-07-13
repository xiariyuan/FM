#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

REPO = Path('/gemini/code/FMtrack-main/FM-Track')
DFINE = REPO / 'external/detectors/D-FINE'
BOTSORT = REPO / 'external/BoT-SORT-main'
MOT20 = Path('/gemini/code/datasets/MOT20')

sys.path.insert(0, str(DFINE))
from src.core import YAMLConfig  # noqa: E402

sys.path.insert(0, str(BOTSORT))
from tools.track import make_parser as make_botsort_parser  # noqa: E402
from tracker.bot_sort import BoTSORT  # noqa: E402


def load_dfine(config: str, weights: str, device: str):
    cfg = YAMLConfig(config, resume=weights)
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    checkpoint = torch.load(weights, map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model().to(device).eval()
    return model


def parse_seqinfo(seq_dir: Path):
    fps = 25
    width = height = None
    p = seq_dir / 'seqinfo.ini'
    if p.exists():
        for line in p.read_text(errors='ignore').splitlines():
            line = line.strip()
            if line.startswith('frameRate='):
                fps = int(float(line.split('=',1)[1]))
            elif line.startswith('imWidth='):
                width = int(float(line.split('=',1)[1]))
            elif line.startswith('imHeight='):
                height = int(float(line.split('=',1)[1]))
    return fps, width, height


def make_tracker_args(seq_dir: Path, fps: int, high: float, low: float, new: float, match: float, with_reid: bool, cmc_method: str):
    parser = make_botsort_parser()
    args = parser.parse_args([
        str(MOT20), '--benchmark', 'MOT20', '--eval', 'val', '--seq-ids', '5',
        '--track_high_thresh', str(high), '--track_low_thresh', str(low), '--new_track_thresh', str(new),
        '--match_thresh', str(match), '--track_buffer', '30', '--proximity_thresh', '0.5', '--appearance_thresh', '0.25',
        '--cmc-method', cmc_method,
    ] + (['--with-reid', '--fast-reid-config', 'fast_reid/configs/MOT20/sbs_S50.yml', '--fast-reid-weights', 'pretrained/mot20_sbs_S50.pth'] if with_reid else []))
    args.name = 'MOT20-05'
    args.ablation = False
    args.mot20 = True
    args.path = str(seq_dir / 'img1')
    args.fps = fps
    args.device = 'gpu'
    return args


def clamp_boxes_xyxy(boxes: np.ndarray, w: int, h: int):
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
    return boxes, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='MOT20-05')
    ap.add_argument('--config', default=str(DFINE / 'configs/dfine/objects365/dfine_hgnetv2_l_obj2coco.yml'))
    ap.add_argument('--weights', default=str(DFINE / 'weights/dfine_l_obj2coco_e25.pth'))
    ap.add_argument('--out-dir', default=str(REPO / 'outputs/detector_plugins/dfine_botsort_mot20_05'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--resize', type=int, default=640)
    ap.add_argument('--person-label', type=int, default=0)
    ap.add_argument('--score-thresh', type=float, default=0.001, help='Keep very low scores; BoT-SORT thresholds will filter.')
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--with-reid', action='store_true')
    ap.add_argument('--cmc-method', default='file')
    ap.add_argument('--track-high', type=float, default=0.55)
    ap.add_argument('--track-low', type=float, default=0.1)
    ap.add_argument('--new-track', type=float, default=0.65)
    ap.add_argument('--match-thresh', type=float, default=0.7)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / 'dfine_raw_detections.jsonl'
    det_path = out_dir / 'dfine_mot20_05_det.txt'
    track_dir = out_dir / 'track_results'
    track_dir.mkdir(exist_ok=True)
    track_path = track_dir / f'{args.seq}.txt'
    stats_path = out_dir / 'dfine_botsort_stats.json'

    seq_dir = MOT20 / 'train' / args.seq
    img_dir = seq_dir / 'img1'
    frames = sorted(img_dir.glob('*.jpg'))
    if args.max_frames > 0:
        frames = frames[:args.max_frames]
    fps, sw, sh = parse_seqinfo(seq_dir)

    print('[load] D-FINE', args.config, args.weights, flush=True)
    model = load_dfine(args.config, args.weights, args.device)
    transforms = T.Compose([T.Resize((args.resize, args.resize)), T.ToTensor()])

    print('[load] BoT-SORT tracker with_reid=', args.with_reid, 'cmc=', args.cmc_method, flush=True)
    bargs = make_tracker_args(seq_dir, fps, args.track_high, args.track_low, args.new_track, args.match_thresh, args.with_reid, args.cmc_method)
    tracker = BoTSORT(bargs, frame_rate=fps)

    det_rows = 0
    kept_rows = 0
    track_rows = 0
    t0 = time.time()

    with raw_path.open('w', encoding='utf-8') as raw_f, det_path.open('w', encoding='utf-8') as det_f, track_path.open('w', encoding='utf-8') as trk_f:
        for idx, img_path in enumerate(frames, start=1):
            frame_id = int(img_path.stem)
            raw_img = cv2.imread(str(img_path))
            if raw_img is None:
                data = np.fromfile(str(img_path), dtype=np.uint8)
                raw_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if raw_img is None:
                print('[warn] cannot read', img_path, flush=True)
                continue
            h, w = raw_img.shape[:2]
            im_pil = Image.fromarray(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
            orig_size = torch.tensor([[w, h]], device=args.device)
            im_data = transforms(im_pil).unsqueeze(0).to(args.device)
            with torch.no_grad():
                labels, boxes, scores = model(im_data, orig_size)
            labels_np = labels[0].detach().cpu().numpy().astype(np.int64)
            boxes_np = boxes[0].detach().cpu().numpy().astype(np.float32)
            scores_np = scores[0].detach().cpu().numpy().astype(np.float32)

            mask = (labels_np == args.person_label) & (scores_np >= args.score_thresh)
            boxes_np = boxes_np[mask]
            scores_np = scores_np[mask]
            labels_keep = labels_np[mask]
            if boxes_np.size:
                boxes_np, good = clamp_boxes_xyxy(boxes_np, w, h)
                scores_np = scores_np[good]
                labels_keep = labels_keep[good]
            else:
                boxes_np = boxes_np.reshape(0, 4)

            det_rows += int(mask.sum())
            kept_rows += len(scores_np)
            for b, s in zip(boxes_np, scores_np):
                x1,y1,x2,y2 = [float(v) for v in b]
                det_f.write(f'{frame_id},-1,{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{float(s):.6f},-1,-1,-1\n')
            raw_f.write(json.dumps({
                'frame': frame_id,
                'image': str(img_path),
                'num_person': int(len(scores_np)),
                'scores_top5': [float(x) for x in sorted(scores_np.tolist(), reverse=True)[:5]],
            }) + '\n')

            if len(scores_np):
                # BoT-SORT update() has two valid detector shapes:
                #   5 columns: x1,y1,x2,y2,score
                #   >=6 columns: x1,y1,x2,y2,obj_conf,cls_conf,class_id and internally score=obj_conf*cls_conf
                # Earlier we accidentally used 6 columns [box, score, class=0], so score became score*0=0.
                # Use the clean 5-column path to preserve D-FINE scores exactly.
                dets = np.concatenate([boxes_np, scores_np[:, None]], axis=1).astype(np.float32)
            else:
                dets = np.empty((0, 5), dtype=np.float32)
            online_targets = tracker.update(dets, raw_img)
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > bargs.aspect_ratio_thresh if tlwh[3] > 0 else True
                if tlwh[2] * tlwh[3] > bargs.min_box_area and not vertical:
                    trk_f.write(f'{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.6f},-1,-1,-1\n')
                    track_rows += 1
            if idx % 20 == 0 or idx == 1:
                elapsed = time.time() - t0
                fps_now = idx / elapsed if elapsed > 0 else 0
                print(f'[progress] {idx}/{len(frames)} frame={frame_id} dets={len(scores_np)} tracks={len(online_targets)} fps={fps_now:.2f}', flush=True)
                det_f.flush(); trk_f.flush(); raw_f.flush()

    stats = {
        'seq': args.seq,
        'frames': len(frames),
        'det_rows_preclip': det_rows,
        'det_rows': kept_rows,
        'track_rows': track_rows,
        'det_file': str(det_path),
        'track_file': str(track_path),
        'raw_file': str(raw_path),
        'elapsed_sec': time.time() - t0,
        'params': vars(args),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
