#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random
from pathlib import Path

ROOT = Path('/gemini/code/datasets')
CH_ROOT = ROOT / 'CrowdHuman/OpenDataLab___CrowdHuman/raw/CrowdHuman/crowdhuman'
MOT17_ROOT = ROOT / 'MOT17'
MOT20_ROOT = ROOT / 'MOT20'
BOT_DATA = Path('external/BoT-SORT-main/datasets')
MIX_ROOT = BOT_DATA / 'MIX_CH_MOT17_MOT20'


def load_json(p: Path):
    return json.loads(p.read_text())


def ensure_links():
    MIX_ROOT.mkdir(parents=True, exist_ok=True)
    (MIX_ROOT / 'annotations').mkdir(exist_ok=True)
    links = {
        'crowdhuman_train': CH_ROOT / 'train',
        'crowdhuman_val': CH_ROOT / 'val',
        'MOT17_train': MOT17_ROOT / 'train',
        'MOT20_train': MOT20_ROOT / 'train',
    }
    for name, target in links.items():
        dst = MIX_ROOT / name
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() and Path(dst.resolve()) == target:
                continue
            dst.unlink() if dst.is_symlink() else None
        if not dst.exists():
            dst.symlink_to(target)


def add_source(out, src, prefix, source_name, max_images=0, rng=None):
    images = src['images']
    if max_images and max_images < len(images):
        rng = rng or random.Random(0)
        keep = set(rng.sample([im['id'] for im in images], max_images))
        images = [im for im in images if im['id'] in keep]
    else:
        keep = set(im['id'] for im in images)
    old_to_new = {}
    for im in images:
        new_im = dict(im)
        old_id = im['id']
        new_id = len(out['images']) + 1
        old_to_new[old_id] = new_id
        new_im['id'] = new_id
        new_im['file_name'] = f'{prefix}/{im["file_name"]}'
        new_im['source'] = source_name
        new_im['frame_id'] = int(new_im.get('frame_id', new_id))
        # Use a deterministic source-level video id for image-only datasets.
        new_im['video_id'] = int(new_im.get('video_id', abs(hash(source_name)) % 100000 + 1))
        out['images'].append(new_im)
    for ann in src.get('annotations', []):
        if ann['image_id'] not in old_to_new:
            continue
        x,y,w,h = ann['bbox']
        if w <= 1 or h <= 1:
            continue
        new_ann = dict(ann)
        new_ann['id'] = len(out['annotations']) + 1
        new_ann['image_id'] = old_to_new[ann['image_id']]
        new_ann['category_id'] = 1
        new_ann['iscrowd'] = int(bool(new_ann.get('iscrowd', 0)))
        new_ann['area'] = float(w*h)
        new_ann['source'] = source_name
        # MOTDataset requires track_id. For detection-only sources, give each box a unique pseudo track id.
        new_ann['track_id'] = int(new_ann.get('track_id', new_ann['id']))
        out['annotations'].append(new_ann)
    return len(images)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ch', type=int, default=3000)
    ap.add_argument('--mot17', type=int, default=1500)
    ap.add_argument('--mot20', type=int, default=1500)
    ap.add_argument('--val-mot20', type=int, default=1200)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--tag', default='quick')
    args = ap.parse_args()
    ensure_links()
    rng = random.Random(args.seed)
    out = {'images': [], 'annotations': [], 'videos': [], 'categories': [{'id':1,'name':'pedestrian'}]}
    sources = [
        ('crowdhuman_train', 'crowdhuman_train', CH_ROOT / 'annotations/crowdhuman_train.json', args.ch),
        ('MOT17_train', 'MOT17_train', MOT17_ROOT / 'annotations/train.json', args.mot17),
        ('MOT20_train', 'MOT20_train', MOT20_ROOT / 'annotations/train.json', args.mot20),
    ]
    counts = {}
    for source_name, prefix, path, maxn in sources:
        counts[source_name] = add_source(out, load_json(path), prefix, source_name, maxn, rng)
    train_path = MIX_ROOT / 'annotations' / f'train_{args.tag}.json'
    train_path.write_text(json.dumps(out))
    val = {'images': [], 'annotations': [], 'videos': [], 'categories': [{'id':1,'name':'pedestrian'}]}
    counts['MOT20_val_half'] = add_source(val, load_json(MOT20_ROOT / 'annotations/val_half.json'), 'MOT20_train', 'MOT20_val_half', args.val_mot20, rng)
    val_path = MIX_ROOT / 'annotations' / f'val_mot20_{args.tag}.json'
    val_path.write_text(json.dumps(val))
    print('MIX_ROOT', MIX_ROOT)
    print('train_json', train_path, train_path.stat().st_size)
    print('val_json', val_path, val_path.stat().st_size)
    print('counts', counts)
    print('train images', len(out['images']), 'anns', len(out['annotations']))
    print('val images', len(val['images']), 'anns', len(val['annotations']))
    print('first train image', out['images'][0])
    print('first val image', val['images'][0])

if __name__ == '__main__':
    main()
