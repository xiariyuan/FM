#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'external/BoT-SORT-main'))
from yolox.exp import get_exp
from yolox.utils import load_ckpt, postprocess


def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-9, aa + bb - inter)


def area_bucket(area):
    # COCO buckets: small <32^2, medium <96^2, large >=96^2
    if area < 32 * 32:
        return 'small'
    if area < 96 * 96:
        return 'medium'
    return 'large'


def load_model(exp_file, ckpt):
    exp = get_exp(exp_file, None)
    model = exp.get_model().cuda().eval()
    blob = torch.load(ckpt, map_location='cuda')
    state = blob.get('model', blob)
    model = load_ckpt(model, state)
    return exp, model


def infer(exp, model, batch_size=1, half=True, max_images=0):
    # env truncation is handled by exp via MIX_DET_SMOKE_VAL_IMAGES. Keep max_images for extra safety.
    loader = exp.get_eval_loader(batch_size=batch_size, is_distributed=False, testdev=False)
    if max_images and hasattr(loader.dataset, 'ids'):
        loader.dataset.ids = loader.dataset.ids[:max_images]
        loader.dataset.annotations = loader.dataset.annotations[:max_images]
    tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
    if half:
        model = model.half()
    model.eval()
    preds = []
    t0 = time.time()
    for imgs, _, info_imgs, ids in tqdm(loader, desc='infer', ncols=80):
        with torch.no_grad():
            imgs = imgs.type(tensor_type)
            outputs = model(imgs)
            outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)
        preds.extend(convert_to_coco(outputs, info_imgs, ids))
    return preds, time.time() - t0


def convert_to_coco(outputs, info_imgs, ids):
    # Adapted from YOLOX COCOEvaluator.convert_to_coco_format.
    data_list = []
    for output, img_h, img_w, img_id in zip(outputs, info_imgs[0], info_imgs[1], ids):
        if output is None:
            continue
        output = output.cpu()
        bboxes = output[:, 0:4]
        # preprocessing ratio in ValTransform: resized to test size with aspect preserved.
        scale = min(float(info_imgs[2][0]) / float(img_h), float(info_imgs[3][0]) / float(img_w)) if len(info_imgs) >= 4 else None
        # In this repo info_imgs can be (height,width,frame_id,video_id,file_name); evaluator uses img_h/img_w and test_size.
        # Safer: infer scale from exp.test_size unavailable here, but info_imgs[2]/[3] are frame/video, not resized dims for MOTDataset.
        # Therefore follow YOLOX evaluator by reconstructing scale from global test size stored in closure? We patch later by using exp in caller? 
        data_list.append(('RAW', output, int(img_id)))
    return data_list


def proper_convert(raw_preds, exp, dataset):
    # raw_preds entries store unscaled output; reproduce evaluator conversion with dataset img metadata.
    out = []
    id_to_img = {int(i): dataset.coco.loadImgs(int(i))[0] for i in dataset.ids}
    for tag, output, img_id in raw_preds:
        im = id_to_img[int(img_id)]
        h, w = float(im['height']), float(im['width'])
        scale = min(exp.test_size[0] / h, exp.test_size[1] / w)
        bboxes = output[:, 0:4] / scale
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        for box, c, s in zip(bboxes, cls, scores):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if bw <= 0 or bh <= 0:
                continue
            out.append({
                'image_id': int(img_id),
                'category_id': 1,
                'bbox': [x1, y1, bw, bh],
                'score': float(s),
            })
    return out


def greedy_match(gts, dets, iou_thr=0.5):
    # dets sorted high score first.
    used = set()
    matches = []
    fp = []
    for di, d in enumerate(dets):
        db = xywh_to_xyxy(d['bbox'])
        best_i, best_iou = -1, 0.0
        for gi, g in enumerate(gts):
            if gi in used:
                continue
            val = iou_xyxy(db, xywh_to_xyxy(g['bbox']))
            if val > best_iou:
                best_i, best_iou = gi, val
        if best_i >= 0 and best_iou >= iou_thr:
            used.add(best_i)
            matches.append((best_i, di, best_iou))
        else:
            fp.append(di)
    fn = [gi for gi in range(len(gts)) if gi not in used]
    return matches, fp, fn


def summarize(name, preds, coco, score_thr):
    gt_by_img = defaultdict(list)
    for ann in coco.dataset['annotations']:
        if int(ann.get('iscrowd', 0)):
            continue
        gt_by_img[int(ann['image_id'])].append(ann)
    pred_by_img = defaultdict(list)
    for d in preds:
        if d['score'] >= score_thr:
            pred_by_img[int(d['image_id'])].append(d)
    total_gt = total_det = total_tp50 = total_fp50 = total_fn50 = 0
    total_tp75 = total_fp75 = total_fn75 = 0
    ious50 = []
    score_stats = []
    bucket = {b: {'gt':0,'tp50':0,'fn50':0,'tp75':0,'fn75':0} for b in ['small','medium','large']}
    img_rows = []
    for img_id in coco.getImgIds():
        gts = gt_by_img.get(int(img_id), [])
        dets = sorted(pred_by_img.get(int(img_id), []), key=lambda x: x['score'], reverse=True)
        total_gt += len(gts); total_det += len(dets)
        m50, fp50, fn50 = greedy_match(gts, dets, 0.50)
        m75, fp75, fn75 = greedy_match(gts, dets, 0.75)
        total_tp50 += len(m50); total_fp50 += len(fp50); total_fn50 += len(fn50)
        total_tp75 += len(m75); total_fp75 += len(fp75); total_fn75 += len(fn75)
        for gi, di, iv in m50:
            ious50.append(iv); score_stats.append(dets[di]['score'])
        for g in gts:
            bucket[area_bucket(float(g['area']))]['gt'] += 1
        matched50 = {gi for gi,_,_ in m50}; matched75 = {gi for gi,_,_ in m75}
        for gi,g in enumerate(gts):
            b=area_bucket(float(g['area']))
            if gi in matched50: bucket[b]['tp50'] += 1
            else: bucket[b]['fn50'] += 1
            if gi in matched75: bucket[b]['tp75'] += 1
            else: bucket[b]['fn75'] += 1
        if len(gts) or len(dets):
            img_rows.append({'image_id': int(img_id), 'gt':len(gts), 'det':len(dets), 'tp50':len(m50), 'fp50':len(fp50), 'fn50':len(fn50), 'tp75':len(m75), 'fp75':len(fp75), 'fn75':len(fn75)})
    prec50 = total_tp50 / max(1, total_tp50 + total_fp50)
    rec50 = total_tp50 / max(1, total_gt)
    prec75 = total_tp75 / max(1, total_tp75 + total_fp75)
    rec75 = total_tp75 / max(1, total_gt)
    return {
        'name': name,
        'score_thr': score_thr,
        'images': len(coco.getImgIds()),
        'gt': total_gt,
        'detections': total_det,
        'tp50': total_tp50, 'fp50': total_fp50, 'fn50': total_fn50, 'precision50': prec50, 'recall50': rec50,
        'tp75': total_tp75, 'fp75': total_fp75, 'fn75': total_fn75, 'precision75': prec75, 'recall75': rec75,
        'mean_iou_matched50': float(np.mean(ious50)) if ious50 else 0.0,
        'median_iou_matched50': float(np.median(ious50)) if ious50 else 0.0,
        'mean_score_matched50': float(np.mean(score_stats)) if score_stats else 0.0,
        'bucket': bucket,
        'worst_images': sorted(img_rows, key=lambda r: (r['fn50'], r['fp50']), reverse=True)[:30]
    }


def compare_image_level(sum_a, sum_b):
    a = {r['image_id']: r for r in sum_a['worst_images']}
    # not enough for full compare; handled separately if needed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp-file', default='external/BoT-SORT-main/yolox/exps/example/mot/yolox_x_mixed_human_quick.py')
    ap.add_argument('--baseline-ckpt', default='external/BoT-SORT-main/pretrained/bytetrack_x_mot20.pth.tar')
    ap.add_argument('--mixed-ckpt', default='external/BoT-SORT-main/YOLOX_outputs/MIXED_HUMAN_QUICK/last_epoch_ckpt.pth.tar')
    ap.add_argument('--out-dir', default='outputs/detector_audit/mixed_vs_baseline_quick')
    ap.add_argument('--score-thr', type=float, default=0.001)
    ap.add_argument('--half', action='store_true')
    ap.add_argument('--batch-size', type=int, default=1)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, ckpt in [('baseline', args.baseline_ckpt), ('mixed', args.mixed_ckpt)]:
        print('LOAD', name, ckpt, flush=True)
        exp, model = load_model(args.exp_file, ckpt)
        loader = exp.get_eval_loader(batch_size=args.batch_size, is_distributed=False, testdev=False)
        dataset = loader.dataset
        tensor_type = torch.cuda.HalfTensor if args.half else torch.cuda.FloatTensor
        if args.half: model = model.half()
        model.eval()
        raw = []
        t0=time.time()
        for imgs, _, info_imgs, ids in tqdm(loader, desc=f'infer-{name}', ncols=80):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)
                outputs = model(imgs)
                outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)
            for output, img_id in zip(outputs, ids):
                if output is None:
                    continue
                raw.append(('RAW', output.cpu(), int(img_id)))
        preds = proper_convert(raw, exp, dataset)
        (out_dir / f'{name}_preds.json').write_text(json.dumps(preds))
        summ = summarize(name, preds, dataset.coco, args.score_thr)
        summ['seconds'] = time.time()-t0
        results[name] = summ
        (out_dir / f'{name}_summary.json').write_text(json.dumps(summ, indent=2))
        print('SUMMARY', name, json.dumps({k:summ[k] for k in ['detections','tp50','fp50','fn50','precision50','recall50','tp75','fp75','fn75','precision75','recall75','mean_iou_matched50']}), flush=True)
        del model
        torch.cuda.empty_cache()

    delta = {}
    for k in ['detections','tp50','fp50','fn50','precision50','recall50','tp75','fp75','fn75','precision75','recall75','mean_iou_matched50','median_iou_matched50']:
        delta[k] = results['mixed'][k] - results['baseline'][k]
    # bucket deltas
    bd = {}
    for b in ['small','medium','large']:
        bd[b] = {k: results['mixed']['bucket'][b][k]-results['baseline']['bucket'][b][k] for k in results['baseline']['bucket'][b]}
    final = {'baseline': results['baseline'], 'mixed': results['mixed'], 'delta_mixed_minus_baseline': delta, 'bucket_delta': bd}
    (out_dir / 'audit_summary.json').write_text(json.dumps(final, indent=2))
    print('FINAL_DELTA', json.dumps(delta, indent=2), flush=True)
    print('BUCKET_DELTA', json.dumps(bd, indent=2), flush=True)
    print('OUT_DIR', out_dir, flush=True)

if __name__ == '__main__':
    main()
