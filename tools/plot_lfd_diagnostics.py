"""Plot LFD diagnostics dumps.

This script reads *.pt files produced by train.py when SAVE_LFD_DIAGNOSTICS=True
and outputs:
  1) per-band frequency response curves (FFT power)
  2) overlap heatmap (cosine similarity on normalized power)
  3) importance statistics (mean + histogram)

Usage:
  python tools/plot_lfd_diagnostics.py \
      --diag_dir outputs/exp_xxx/lfd_diag \
      --out_dir outputs/exp_xxx/lfd_diag_plots \
      --pick latest

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def _list_pt_files(diag_dir: Path):
    return sorted([p for p in diag_dir.glob('*.pt') if p.is_file()])


def _pick_file(files, pick: str):
    if not files:
        raise FileNotFoundError('No .pt files found')
    if pick == 'latest':
        return files[-1]
    if pick.isdigit():
        idx = int(pick)
        idx = max(0, min(idx, len(files) - 1))
        return files[idx]
    p = Path(pick)
    if p.exists():
        return p
    raise ValueError(f'Unknown pick: {pick}')


def _fft_power(filters: np.ndarray, n_fft: int = 256):
    # filters: (K, k)
    fft = np.fft.rfft(filters, n=n_fft, axis=-1)
    power = (fft.real ** 2 + fft.imag ** 2)
    freqs = np.fft.rfftfreq(n_fft, d=1.0)  # cycles/sample in [0,0.5]
    return freqs, power


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag_dir', type=str, required=True)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--pick', type=str, default='latest', help='latest | integer index | path')
    ap.add_argument('--n_fft', type=int, default=256)
    args = ap.parse_args()

    diag_dir = Path(args.diag_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _list_pt_files(diag_dir)
    pt_path = _pick_file(files, args.pick)
    diag = torch.load(pt_path, map_location='cpu')

    # --------------- filters -> FFT curves + overlap heatmap ---------------
    filters = diag.get('filters_detached', None)
    if isinstance(filters, torch.Tensor):
        filt = filters.squeeze(1).float().numpy()  # (K,k)
        freqs, power = _fft_power(filt, n_fft=args.n_fft)  # (F,), (K,F)
        # normalize for comparison
        power_norm = power / (power.sum(axis=-1, keepdims=True) + 1e-12)

        # Save curves data
        np.save(out_dir / 'freqs.npy', freqs)
        np.save(out_dir / 'power_norm.npy', power_norm)

        # Plot curves
        import matplotlib.pyplot as plt
        plt.figure()
        for i in range(power_norm.shape[0]):
            plt.plot(freqs, power_norm[i], label=f'band_{i}')
        plt.xlabel('Frequency (cycles / sample)')
        plt.ylabel('Normalized power')
        plt.title(f'Band frequency responses\n{pt_path.name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'fft_responses.png', dpi=200)
        plt.close()

        # Overlap heatmap: cosine similarity on normalized power
        p = power_norm
        p = p / (np.linalg.norm(p, axis=-1, keepdims=True) + 1e-12)
        sim = p @ p.T
        plt.figure()
        plt.imshow(sim, vmin=0.0, vmax=1.0)
        plt.colorbar()
        plt.xlabel('Band j')
        plt.ylabel('Band i')
        plt.title('Overlap heatmap (cosine similarity)')
        plt.tight_layout()
        plt.savefig(out_dir / 'overlap_heatmap.png', dpi=200)
        plt.close()

    # --------------- importance stats ---------------
    imp = diag.get('importance', None)
    if isinstance(imp, torch.Tensor):
        imp_np = imp.float().numpy()  # (...,K)
        K = imp_np.shape[-1]
        imp_flat = imp_np.reshape(-1, K)
        imp_mean = imp_flat.mean(axis=0)
        np.save(out_dir / 'importance_mean.npy', imp_mean)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.bar(np.arange(K), imp_mean)
        plt.xlabel('Band index')
        plt.ylabel('Mean importance')
        plt.title('Importance mean across (B,T,N)')
        plt.tight_layout()
        plt.savefig(out_dir / 'importance_mean.png', dpi=200)
        plt.close()

        # histogram per band
        plt.figure()
        for i in range(K):
            plt.hist(imp_flat[:, i], bins=50, alpha=0.4, label=f'band_{i}')
        plt.xlabel('Importance')
        plt.ylabel('Count')
        plt.title('Importance histograms')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'importance_hist.png', dpi=200)
        plt.close()

    # --------------- meta ---------------
    with open(out_dir / 'meta.txt', 'w', encoding='utf-8') as f:
        f.write(f'source_pt: {pt_path}\n')
        for k in ['epoch', 'global_step']:
            if k in diag:
                f.write(f'{k}: {diag[k]}\n')
        if 'importance_tau' in diag:
            tau = diag['importance_tau']
            if isinstance(tau, torch.Tensor):
                try:
                    f.write(f'importance_tau: {float(tau.item())}\n')
                except Exception:
                    pass

    print(f'[OK] Plots saved to: {out_dir}')


if __name__ == '__main__':
    main()
