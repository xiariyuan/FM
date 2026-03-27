"""LFD importance binning diagnostics (speed / occlusion buckets).

This script is part of the "evidence chain" for top-tier review:
- Show that band-importance is not a static global gate.
- Demonstrate correlation between band usage and MOT phenomena (speed / occlusion).

It expects a .pt dump produced by train.py when SAVE_LFD_DIAGNOSTICS=True.
To enable speed/occlusion binning, the dump should contain:
  - trajectory_boxes: (B, G, T, N, 4)
  - trajectory_masks: (B, G, T, N)  (bool, True=masked)
  - importance:        (B*G, T, N, K) or (B, G, T, N, K)

Usage:
  python tools/lfd_importance_binning.py \
      --diag_dir outputs/exp_xxx/lfd_diag \
      --out_dir outputs/exp_xxx/lfd_diag_bins \
      --pick latest

Notes:
- Boxes are assumed to be DETR-style normalized (cx, cy, w, h) by default.
- "Speed" is computed as mean center displacement per frame (normalized units).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


def _list_pt_files(diag_dir: Path):
    return sorted([p for p in diag_dir.glob('*.pt') if p.is_file()])


def _pick_files(files, pick: str, max_files: int | None = None):
    if not files:
        raise FileNotFoundError('No .pt files found')
    if pick == 'latest':
        out = [files[-1]]
    elif pick == 'all':
        out = files
    elif pick.isdigit():
        idx = int(pick)
        idx = max(0, min(idx, len(files) - 1))
        out = [files[idx]]
    else:
        p = Path(pick)
        if p.exists():
            out = [p]
        else:
            raise ValueError(f'Unknown pick: {pick}')
    if max_files is not None:
        out = out[: max(1, int(max_files))]
    return out


def _parse_bins(bins_str: str | None):
    if not bins_str:
        return None
    try:
        return [float(x) for x in bins_str.split(',') if x.strip()]
    except Exception:
        return None


def _quantile_bins(values: np.ndarray, n_bins: int = 5):
    if values.size == 0:
        return None
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, qs)
    # ensure strictly increasing
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return edges.tolist()


def _centers_from_boxes(boxes: np.ndarray, box_format: str = 'cxcywh') -> Tuple[np.ndarray, np.ndarray]:
    box_format = box_format.lower()
    if box_format in ('cxcywh', 'cxywh', 'detr'):
        cx, cy = boxes[..., 0], boxes[..., 1]
        return cx, cy
    if box_format in ('xyxy',):
        x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5
    if box_format in ('xywh',):
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        return x + 0.5 * w, y + 0.5 * h
    raise ValueError(f'Unknown box_format: {box_format}')


def _compute_speed(boxes: np.ndarray, masks: np.ndarray, box_format: str) -> np.ndarray:
    """Mean per-target speed over time.

    Returns:
      speed: (B, G, N) in normalized units/frame.
    """
    valid = ~masks
    cx, cy = _centers_from_boxes(boxes, box_format=box_format)

    valid_pair = valid[:, :, 1:, :] & valid[:, :, :-1, :]
    dx = cx[:, :, 1:, :] - cx[:, :, :-1, :]
    dy = cy[:, :, 1:, :] - cy[:, :, :-1, :]
    dist = np.sqrt(dx * dx + dy * dy)

    dist = dist * valid_pair.astype(dist.dtype)
    denom = valid_pair.sum(axis=2)
    denom = np.maximum(denom, 1)
    speed = dist.sum(axis=2) / denom
    return speed


def _compute_occ_ratio(masks: np.ndarray) -> np.ndarray:
    """Per-target occlusion ratio over time (masked fraction)."""
    return masks.astype(np.float32).mean(axis=2)


def _bucket_means(values: np.ndarray, bucket: np.ndarray, n_bins: int):
    # values: (M, K)
    K = values.shape[-1]
    means = np.zeros((n_bins, K), dtype=np.float64)
    counts = np.zeros((n_bins,), dtype=np.int64)
    for b in range(n_bins):
        idx = bucket == b
        counts[b] = int(idx.sum())
        if idx.any():
            means[b] = values[idx].mean(axis=0)
    return means, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag_dir', type=str, required=True)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--pick', type=str, default='latest', help='latest | all | integer index | path')
    ap.add_argument('--max_files', type=int, default=20)
    ap.add_argument('--box_format', type=str, default='cxcywh')
    ap.add_argument('--speed_bins', type=str, default=None)
    ap.add_argument('--speed_nbins', type=int, default=5)
    ap.add_argument('--occ_bins', type=str, default=None)
    args = ap.parse_args()

    diag_dir = Path(args.diag_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _list_pt_files(diag_dir)
    picked = _pick_files(files, args.pick, max_files=args.max_files)

    all_imp = []
    all_speed = []
    all_occ = []
    all_valid = []

    for pt_path in picked:
        diag = torch.load(pt_path, map_location='cpu')
        imp = diag.get('importance', None)
        boxes = diag.get('trajectory_boxes', None)
        masks = diag.get('trajectory_masks', None)
        if not (isinstance(imp, torch.Tensor) and isinstance(boxes, torch.Tensor) and isinstance(masks, torch.Tensor)):
            continue

        imp_np = imp.float().cpu().numpy()
        boxes_np = boxes.float().cpu().numpy()
        masks_np = masks.bool().cpu().numpy()

        # boxes: (B,G,T,N,4) ; masks: (B,G,T,N)
        if boxes_np.ndim != 5 or masks_np.ndim != 4:
            continue
        B, G, T, N, _ = boxes_np.shape

        # importance can be (B*G, T, N, K) OR (B,G,T,N,K)
        if imp_np.ndim == 4:
            BG, Ti, Ni, K = imp_np.shape
            if BG != B * G or Ti != T or Ni != N:
                Bb = int(diag.get('B', B))
                Gg = int(diag.get('G', G))
                Tt = int(diag.get('T', T))
                Nn = int(diag.get('N', N))
                if BG == Bb * Gg and Ti == Tt and Ni == Nn:
                    B, G, T, N = Bb, Gg, Tt, Nn
                else:
                    continue
            imp_np = imp_np.reshape(B, G, T, N, K)
        elif imp_np.ndim == 5:
            if imp_np.shape[:4] != (B, G, T, N):
                continue
            K = imp_np.shape[-1]
        else:
            continue

        speed = _compute_speed(boxes_np, masks_np, args.box_format)  # (B,G,N)
        occ = _compute_occ_ratio(masks_np)  # (B,G,N)

        # Per-target mean importance over visible frames
        valid = (~masks_np).astype(np.float32)  # (B,G,T,N)
        denom = np.maximum(valid.sum(axis=2, keepdims=True), 1.0)  # (B,G,1,N)
        imp_mean = (imp_np * valid[..., None]).sum(axis=2) / denom[..., None]  # (B,G,N,K)

        imp_flat = imp_mean.reshape(-1, K)
        speed_flat = speed.reshape(-1)
        occ_flat = occ.reshape(-1)
        vis_cnt_flat = valid.sum(axis=2).reshape(-1)

        all_imp.append(imp_flat)
        all_speed.append(speed_flat)
        all_occ.append(occ_flat)
        all_valid.append(vis_cnt_flat)

    if not all_imp:
        raise RuntimeError('No valid diagnostics found for binning')

    imp_all = np.concatenate(all_imp, axis=0)
    speed_all = np.concatenate(all_speed, axis=0)
    occ_all = np.concatenate(all_occ, axis=0)
    vis_cnt_all = np.concatenate(all_valid, axis=0)

    # filter invalid targets (no visible frames)
    keep = vis_cnt_all > 0.0
    imp_all = imp_all[keep]
    speed_all = speed_all[keep]
    occ_all = occ_all[keep]

    K = imp_all.shape[-1]

    # ---------- speed bins ----------
    speed_bins = _parse_bins(args.speed_bins)
    if speed_bins is None:
        speed_bins = _quantile_bins(speed_all, n_bins=max(2, int(args.speed_nbins)))
    speed_bins = sorted(speed_bins)
    speed_bins[-1] = max(speed_bins[-1], float(speed_all.max()) + 1e-6)

    speed_bucket = np.digitize(speed_all, speed_bins[1:-1], right=False)
    speed_nb = len(speed_bins) - 1
    speed_means, speed_counts = _bucket_means(imp_all, speed_bucket, speed_nb)

    # ---------- occlusion bins ----------
    occ_bins = _parse_bins(args.occ_bins) or [0.0, 0.1, 0.3, 0.6, 1.0]
    occ_bins = sorted(occ_bins)
    occ_bins[0] = min(occ_bins[0], 0.0)
    occ_bins[-1] = max(occ_bins[-1], 1.0)
    occ_bucket = np.digitize(occ_all, occ_bins[1:-1], right=False)
    occ_nb = len(occ_bins) - 1
    occ_means, occ_counts = _bucket_means(imp_all, occ_bucket, occ_nb)

    # ---------- entropy (optional sanity) ----------
    eps = 1e-12
    ent = -(imp_all * np.log(np.clip(imp_all, eps, 1.0))).sum(axis=-1)
    ent_speed = np.zeros((speed_nb,), dtype=np.float64)
    ent_occ = np.zeros((occ_nb,), dtype=np.float64)
    for b in range(speed_nb):
        idx = speed_bucket == b
        ent_speed[b] = float(ent[idx].mean()) if idx.any() else 0.0
    for b in range(occ_nb):
        idx = occ_bucket == b
        ent_occ[b] = float(ent[idx].mean()) if idx.any() else 0.0

    # ---------- save arrays ----------
    np.save(out_dir / 'speed_bins.npy', np.array(speed_bins, dtype=np.float64))
    np.save(out_dir / 'speed_means.npy', speed_means)
    np.save(out_dir / 'speed_counts.npy', speed_counts)

    np.save(out_dir / 'occ_bins.npy', np.array(occ_bins, dtype=np.float64))
    np.save(out_dir / 'occ_means.npy', occ_means)
    np.save(out_dir / 'occ_counts.npy', occ_counts)

    # ---------- write csv summary ----------
    def _fmt_edges(edges: List[float], i: int) -> str:
        return f'[{edges[i]:.6f}, {edges[i+1]:.6f})'

    lines = []
    header = ['bucket_type', 'bucket', 'count', 'entropy'] + [f'band_{i}' for i in range(K)]
    lines.append(','.join(header))

    for i in range(speed_nb):
        row = ['speed', _fmt_edges(speed_bins, i), str(int(speed_counts[i])), f'{ent_speed[i]:.6f}']
        row += [f'{speed_means[i, j]:.6f}' for j in range(K)]
        lines.append(','.join(row))

    for i in range(occ_nb):
        row = ['occlusion', _fmt_edges(occ_bins, i), str(int(occ_counts[i])), f'{ent_occ[i]:.6f}']
        row += [f'{occ_means[i, j]:.6f}' for j in range(K)]
        lines.append(','.join(row))

    (out_dir / 'importance_binning_summary.csv').write_text('\n'.join(lines), encoding='utf-8')

    # ---------- plots ----------
    import matplotlib.pyplot as plt

    plt.figure()
    for k in range(K):
        plt.plot(np.arange(speed_nb), speed_means[:, k], marker='o', label=f'band_{k}')
    plt.xticks(np.arange(speed_nb), [f'{i}' for i in range(speed_nb)])
    plt.xlabel('Speed bucket index')
    plt.ylabel('Mean importance')
    plt.title('Band importance vs speed buckets')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'importance_vs_speed.png', dpi=200)
    plt.close()

    plt.figure()
    for k in range(K):
        plt.plot(np.arange(occ_nb), occ_means[:, k], marker='o', label=f'band_{k}')
    plt.xticks(np.arange(occ_nb), [f'{i}' for i in range(occ_nb)])
    plt.xlabel('Occlusion bucket index')
    plt.ylabel('Mean importance')
    plt.title('Band importance vs occlusion buckets')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'importance_vs_occlusion.png', dpi=200)
    plt.close()

    meta = [
        f'files: {len(picked)}',
        f'K(num_bands): {K}',
        f'speed_bins: {speed_bins}',
        f'occ_bins: {occ_bins}',
        f'total_targets: {imp_all.shape[0]}',
    ]
    (out_dir / 'meta.txt').write_text('\n'.join(meta), encoding='utf-8')

    print(f'[OK] Saved binning results to: {out_dir}')


if __name__ == '__main__':
    main()
