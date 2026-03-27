# FM-Track Efficiency Benchmark Results

## Configuration
- Input Shape: (1, 18, 50, 256)
- Device: cuda

## Parameter Count
| Metric | Value |
|--------|-------|
| Total | 8,895,521 |
| Trainable | 8,895,521 |
| Frozen | 0 |

## FLOPs
- Estimated FLOPs: 3.504G

## Memory Usage
| Metric | Value (MB) |
|--------|------------|
| Peak | 72.1 |
| Model | 33.9 |

## Inference Speed
| Metric | Value |
|--------|-------|
| Mean Latency | 7.22 ms |
| FPS | 138.6 |
| P95 Latency | 12.95 ms |

## Batch Scaling
| Batch Size | FPS | Throughput (samples/s) | Memory (MB) |
|------------|-----|------------------------|-------------|
| 1 | 124.6 | 124.6 | 72 |
| 2 | 70.6 | 141.2 | 102 |
| 4 | 43.3 | 173.2 | 159 |
| 8 | 21.8 | 174.2 | 275 |