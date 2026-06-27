#!/usr/bin/env python3
"""Fast MOT20-to-COCO converter using seqinfo dimensions instead of cv2.imread."""
from __future__ import annotations
import argparse
import configparser
import json
from pathlib import Path
import numpy as np

KEEP_PERSON = {1}
IGNORE_PERSON = {2, 7, 8, 12}
NON_PERSON = {3, 4, 5, 6, 9, 10, 11}


def seq_info(seq_dir: Path) -> dict:
    cfg = configparser.ConfigParser()
    cfg.read(seq_dir / 'seqinfo.ini')
    s = cfg['Sequence']
    return {
        'name': s.get('name', seq_dir.name),
        'seqLength': int(s.get('seqLength')),
        'imWidth': int(s.get('imWidth')),
        'imHeight': int(s.get('imHeight')),
        'imExt': s.get('imExt', '.jpg'),
    }


def frame_range(num_images: int, split: str) -> tuple[int, int]:
    if 'half' in split:
        return (0, num_images // 2) if 'train' in split else (num_images // 2 + 1, num_images - 1)
    return (0, num_images - 1)


def load_txt(path: Path) -> np.ndarray:
    arr = np.loadtxt(str(path), dtype=np.float32, delimiter=',')
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def write_split_gt_det(seq_dir: Path, split: str, start0: int, end0: int) -> None:
    if 'half' not in split:
        return
    gt_path = seq_dir / 'gt' / 'gt.txt'
    det_path = seq_dir / 'det' / 'det.txt'
    if gt_path.exists():
        anns = load_txt(gt_path)
        anns_out = np.array([r for r in anns if start0 <= int(r[0]) - 1 <= end0], dtype=np.float32)
        if len(anns_out):
            anns_out[:, 0] -= start0
        out = seq_dir / 'gt' / f'gt_{split}.txt'
        with out.open('w') as f:
            for r in anns_out:
                f.write('{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                    int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4]), int(r[5]), int(r[6]), int(r[7]), float(r[8])
                ))
    if det_path.exists():
        dets = load_txt(det_path)
        dets_out = np.array([r for r in dets if start0 <= int(r[0]) - 1 <= end0], dtype=np.float32)
        if len(dets_out):
            dets_out[:, 0] -= start0
        out = seq_dir / 'det' / f'det_{split}.txt'
        with out.open('w') as f:
            for r in dets_out:
                f.write('{:d},{:d},{:.1f},{:.1f},{:.1f},{:.1f},{:.6f}\n'.format(
                    int(r[0]), int(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[6])
                ))


def convert(root: Path, split: str) -> dict:
    source = root / ('test' if split == 'test' else 'train')
    out = {'images': [], 'annotations': [], 'videos': [], 'categories': [{'id': 1, 'name': 'pedestrian'}]}
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    tid_curr = 0
    tid_last = -1
    for seq_dir in sorted([p for p in source.iterdir() if p.is_dir() and p.name.startswith('MOT20-')]):
        info = seq_info(seq_dir)
        video_cnt += 1
        out['videos'].append({'id': video_cnt, 'file_name': seq_dir.name})
        num_images = info['seqLength']
        start0, end0 = frame_range(num_images, split)
        for i in range(num_images):
            if i < start0 or i > end0:
                continue
            out['images'].append({
                'file_name': f'{seq_dir.name}/img1/{i+1:06d}{info["imExt"]}',
                'id': image_cnt + i + 1,
                'frame_id': i + 1 - start0,
                'prev_image_id': image_cnt + i if i > 0 else -1,
                'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                'video_id': video_cnt,
                'height': info['imHeight'],
                'width': info['imWidth'],
            })
        print(f'{seq_dir.name}: {num_images} images, selected {end0-start0+1}')
        if split != 'test':
            write_split_gt_det(seq_dir, split, start0, end0)
            anns = load_txt(seq_dir / 'gt' / 'gt.txt')
            for r in anns:
                frame_id = int(r[0])
                if frame_id - 1 < start0 or frame_id - 1 > end0:
                    continue
                if int(r[6]) != 1:
                    continue
                cat_id = int(r[7])
                if cat_id in NON_PERSON or cat_id in IGNORE_PERSON:
                    continue
                track_id = int(r[1])
                if track_id != tid_last:
                    tid_curr += 1
                    tid_last = track_id
                ann_cnt += 1
                out['annotations'].append({
                    'id': ann_cnt,
                    'category_id': 1,
                    'image_id': image_cnt + frame_id,
                    'track_id': tid_curr,
                    'bbox': [float(x) for x in r[2:6]],
                    'conf': float(r[6]),
                    'iscrowd': 0,
                    'area': float(r[4] * r[5]),
                })
        image_cnt += num_images
    print(f'loaded {split}: {len(out["images"])} images, {len(out["annotations"])} annotations, {len(out["videos"])} videos')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/gemini/code/datasets/MOT20')
    args = ap.parse_args()
    root = Path(args.root)
    ann_dir = root / 'annotations'
    ann_dir.mkdir(exist_ok=True)
    for split in ['train_half', 'val_half', 'train', 'test']:
        out = convert(root, split)
        tmp = ann_dir / f'{split}.json.tmp'
        final = ann_dir / f'{split}.json'
        tmp.write_text(json.dumps(out))
        tmp.replace(final)
        print(final, final.stat().st_size)

if __name__ == '__main__':
    main()
