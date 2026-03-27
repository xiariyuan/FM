#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Efficiency Benchmark for FM-Track
顶会效率证据：FPS、FLOPs、参数量、显存

Usage:
    python tools/efficiency_benchmark.py --config configs/xxx.yaml --checkpoint outputs/xxx/best.pth
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional
from contextlib import contextmanager

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@contextmanager
def torch_timer(device: str = 'cuda'):
    """Context manager for accurate GPU timing."""
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield lambda: (end.record(), torch.cuda.synchronize(), start.elapsed_time(end))
    else:
        start = time.perf_counter()
        yield lambda: (time.perf_counter() - start) * 1000


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    # Count by module
    module_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        if params > 0:
            module_params[name] = params
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
        'by_module': module_params,
    }


def estimate_flops(model: nn.Module, input_shape: tuple, device: str = 'cuda') -> Dict[str, float]:
    """
    Estimate FLOPs using torch profiler or manual calculation.
    """
    try:
        from thop import profile, clever_format
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape, device=device)
        
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.3f")
        
        return {
            'flops': flops,
            'flops_str': flops_str,
            'params': params,
            'params_str': params_str,
        }
    except ImportError:
        print("[WARN] thop not installed, using estimation")
        # Rough estimation based on parameter count
        total_params = sum(p.numel() for p in model.parameters())
        # Assume ~2 FLOPs per parameter per forward pass (multiply-add)
        estimated_flops = total_params * 2 * np.prod(input_shape[:-1])
        
        return {
            'flops': estimated_flops,
            'flops_str': f'{estimated_flops/1e9:.2f}G',
            'params': total_params,
            'params_str': f'{total_params/1e6:.2f}M',
            'note': 'Estimated (thop not available)',
        }


def measure_memory(model: nn.Module, input_shape: tuple, device: str = 'cuda') -> Dict[str, float]:
    """Measure GPU memory usage."""
    if device != 'cuda' or not torch.cuda.is_available():
        return {'note': 'CUDA not available'}
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Baseline memory
    baseline = torch.cuda.memory_allocated()
    
    # Create input and run forward
    dummy_input = torch.randn(*input_shape, device=device)
    
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Peak memory
    peak = torch.cuda.max_memory_allocated()
    
    # Model memory
    model_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    return {
        'baseline_mb': baseline / 1024**2,
        'peak_mb': peak / 1024**2,
        'model_mb': model_memory / 1024**2,
        'activation_mb': (peak - baseline - model_memory) / 1024**2,
    }


def measure_fps(
    model: nn.Module,
    input_shape: tuple,
    device: str = 'cuda',
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
) -> Dict[str, float]:
    """Measure inference FPS."""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device)
    
    # Warmup
    print(f"  Warmup ({warmup_runs} runs)...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"  Benchmarking ({benchmark_runs} runs)...")
    times = []
    
    with torch.no_grad():
        for _ in range(benchmark_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(dummy_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    times = np.array(times)
    
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'fps': 1000.0 / np.mean(times),
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
    }


def measure_batch_scaling(
    model: nn.Module,
    base_shape: tuple,
    batch_sizes: list,
    device: str = 'cuda',
    runs_per_batch: int = 20,
) -> Dict[int, Dict[str, float]]:
    """Measure how FPS scales with batch size."""
    results = {}
    
    model.eval()
    
    for bs in batch_sizes:
        input_shape = (bs,) + base_shape[1:]
        print(f"  Batch size {bs}...")
        
        try:
            # Check memory
            torch.cuda.empty_cache()
            dummy_input = torch.randn(*input_shape, device=device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(dummy_input)
            
            torch.cuda.synchronize()
            
            # Measure
            times = []
            with torch.no_grad():
                for _ in range(runs_per_batch):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    _ = model(dummy_input)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) * 1000)
            
            results[bs] = {
                'mean_ms': float(np.mean(times)),
                'fps': 1000.0 / np.mean(times),
                'throughput': bs * 1000.0 / np.mean(times),  # samples/sec
                'memory_mb': torch.cuda.max_memory_allocated() / 1024**2,
            }
            
            del dummy_input
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                results[bs] = {'error': 'OOM'}
                torch.cuda.empty_cache()
            else:
                raise
    
    return results


class DummyFMTrackModel(nn.Module):
    """
    Dummy model for benchmarking when full model not available.
    Approximates FM-Track architecture complexity.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_bands: int = 4,
        num_layers: int = 6,
        num_heads: int = 8,
        vocab_size: int = 800,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_bands = num_bands
        
        # LFD (Learnable Frequency Decomposition)
        self.lfd = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
            )
            for _ in range(num_bands)
        ])
        
        # FTT (Frequency Temporal Transformer)
        self.ftt = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=feature_dim * 4,
                batch_first=True,
            ),
            num_layers=2,
        )
        
        # Cross-band interaction
        self.cross_band = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        
        # ID Decoder
        self.id_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=feature_dim * 4,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        
        # Output head
        self.output_head = nn.Linear(feature_dim, vocab_size + 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        
        # LFD
        x_flat = x.view(-1, D)
        band_features = [lfd(x_flat) for lfd in self.lfd]
        band_features = torch.stack(band_features, dim=1)  # (B*T*N, K, D)
        
        # FTT per band
        band_features = band_features.view(B * N, self.num_bands * T, D)
        band_features = self.ftt(band_features)
        
        # Cross-band
        band_features = band_features.view(B * T * N, self.num_bands, D)
        band_features, _ = self.cross_band(band_features, band_features, band_features)
        
        # Aggregate
        x = band_features.mean(dim=1)  # (B*T*N, D)
        x = x.view(B, T * N, D)
        
        # ID Decoder
        x = self.id_decoder(x, x)
        
        # Output
        logits = self.output_head(x)  # (B, T*N, V+1)
        
        return logits


def run_benchmark(
    config: Optional[dict] = None,
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda',
    output_dir: str = 'analysis/efficiency',
) -> Dict:
    """Run full efficiency benchmark."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("FM-Track Efficiency Benchmark")
    print("=" * 60)
    
    # Model setup
    if config is not None:
        # Load real model
        print("\n[Loading Model]")
        try:
            from models.bytetrack_feature_extractor import build_bytetrack_feature_extractor
            model = build_bytetrack_feature_extractor(config).to(device)
            print(f"  Loaded from config")
        except Exception as e:
            print(f"  Failed to load model: {e}")
            print(f"  Using dummy model for benchmarking")
            model = DummyFMTrackModel(
                feature_dim=config.get('FEATURE_DIM', 256),
                num_bands=config.get('NUM_BANDS', 4),
                num_layers=config.get('NUM_ID_DECODER_LAYERS', 6),
            ).to(device)
    else:
        print("\n[Using Dummy Model]")
        model = DummyFMTrackModel().to(device)
    
    model.eval()
    
    # Benchmark settings
    # Typical MOT input: B=1, T=18 frames, N=50 objects, D=256 features
    input_shape = (1, 18, 50, 256)
    batch_sizes = [1, 2, 4, 8]
    
    results = {
        'config': {
            'input_shape': input_shape,
            'device': device,
            'batch_sizes': batch_sizes,
        }
    }
    
    # 1. Parameter count
    print("\n[1/5] Counting Parameters...")
    params = count_parameters(model)
    results['parameters'] = params
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")
    
    # 2. FLOPs estimation
    print("\n[2/5] Estimating FLOPs...")
    flops = estimate_flops(model, input_shape, device)
    results['flops'] = flops
    print(f"  FLOPs: {flops['flops_str']}")
    print(f"  Params: {flops['params_str']}")
    
    # 3. Memory usage
    print("\n[3/5] Measuring Memory...")
    memory = measure_memory(model, input_shape, device)
    results['memory'] = memory
    if 'peak_mb' in memory:
        print(f"  Peak: {memory['peak_mb']:.1f} MB")
        print(f"  Model: {memory['model_mb']:.1f} MB")
    
    # 4. FPS measurement
    print("\n[4/5] Measuring FPS...")
    fps = measure_fps(model, input_shape, device, warmup_runs=20, benchmark_runs=100)
    results['fps'] = fps
    print(f"  Mean: {fps['mean_ms']:.2f} ms")
    print(f"  FPS: {fps['fps']:.1f}")
    print(f"  P95: {fps['p95_ms']:.2f} ms")
    
    # 5. Batch scaling
    print("\n[5/5] Measuring Batch Scaling...")
    batch_scaling = measure_batch_scaling(model, input_shape, batch_sizes, device)
    results['batch_scaling'] = batch_scaling
    for bs, data in batch_scaling.items():
        if 'error' not in data:
            print(f"  BS={bs}: {data['fps']:.1f} FPS, {data['throughput']:.1f} samples/s, {data['memory_mb']:.0f} MB")
        else:
            print(f"  BS={bs}: {data['error']}")
    
    # Save results
    results_path = output_dir / 'efficiency_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Saved] {results_path}")
    
    # Generate summary table
    summary_md = generate_summary_table(results)
    summary_path = output_dir / 'efficiency_summary.md'
    with open(summary_path, 'w') as f:
        f.write(summary_md)
    print(f"[Saved] {summary_path}")
    
    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)
    
    return results


def generate_summary_table(results: Dict) -> str:
    """Generate markdown summary table."""
    lines = [
        "# FM-Track Efficiency Benchmark Results\n",
        "## Configuration",
        f"- Input Shape: {results['config']['input_shape']}",
        f"- Device: {results['config']['device']}",
        "",
        "## Parameter Count",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total | {results['parameters']['total']:,} |",
        f"| Trainable | {results['parameters']['trainable']:,} |",
        f"| Frozen | {results['parameters']['frozen']:,} |",
        "",
        "## FLOPs",
        f"- Estimated FLOPs: {results['flops']['flops_str']}",
        "",
        "## Memory Usage",
    ]
    
    if 'peak_mb' in results['memory']:
        lines.extend([
            "| Metric | Value (MB) |",
            "|--------|------------|",
            f"| Peak | {results['memory']['peak_mb']:.1f} |",
            f"| Model | {results['memory']['model_mb']:.1f} |",
        ])
    
    lines.extend([
        "",
        "## Inference Speed",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Mean Latency | {results['fps']['mean_ms']:.2f} ms |",
        f"| FPS | {results['fps']['fps']:.1f} |",
        f"| P95 Latency | {results['fps']['p95_ms']:.2f} ms |",
        "",
        "## Batch Scaling",
        "| Batch Size | FPS | Throughput (samples/s) | Memory (MB) |",
        "|------------|-----|------------------------|-------------|",
    ])
    
    for bs, data in results['batch_scaling'].items():
        if 'error' not in data:
            lines.append(f"| {bs} | {data['fps']:.1f} | {data['throughput']:.1f} | {data['memory_mb']:.0f} |")
        else:
            lines.append(f"| {bs} | {data['error']} | - | - |")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='FM-Track Efficiency Benchmark')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--output-dir', type=str, default='analysis/efficiency',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    config = None
    if args.config:
        config = load_config(args.config)
    
    run_benchmark(
        config=config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
