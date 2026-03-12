<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

# GEMM All-Scatter Benchmark

Tile-based GEMM with all-scatter communication using Iris. Each rank computes a local tile of the output matrix and scatters results to all peer GPUs using Iris's `store` primitive, enabling communication/computation overlap.

## Algorithm

Each rank holds a shard of matrix B (N/world_size columns) and computes the full A×B product for its local columns. Results are scattered in-kernel via remote stores to reconstruct the global output matrix across all GPUs.

## Usage

```bash
# Run with torchrun (2 GPUs)
torchrun --nproc_per_node 2 examples/07_gemm_all_scatter/benchmark.py --benchmark

# Validate correctness
torchrun --nproc_per_node 2 examples/07_gemm_all_scatter/benchmark.py --validate

# Custom matrix dimensions
torchrun --nproc_per_node 2 examples/07_gemm_all_scatter/benchmark.py \
    --benchmark -m 8192 -n 4608 -k 36864 --datatype fp16
```

## Benchmark Results

Measured on **2× AMD MI300X (gfx942)**, FP16, default dimensions (M=8192, N=4608, K=36864):

| Metric          | Value         |
|-----------------|---------------|
| Performance     | ~611 TFLOPS   |
| Total time      | ~4.55 ms      |
| GEMM kernel     | ~4.41 ms      |
| World size      | 2 GPUs        |
| Block size M    | 256           |
| Block size N    | 64            |
| Block size K    | 64            |
| SMs used        | 304 (all CUs) |

## IntelliKit Profiling (Metrix)

Profiled with [Metrix](https://github.com/amd/metrix) on 2× AMD MI300X (gfx942), M=8192, N=4608, K=36864, FP16, across 126 benchmark iterations.

### Memory Profile

| Metric                       | Value           |
|------------------------------|-----------------|
| HBM Bandwidth Utilization    | 3.2%            |
| HBM Read Bandwidth           | 167.4 GB/s      |
| HBM Write Bandwidth          | 2.5 GB/s        |
| Bytes Transferred (HBM)      | ~4.9 GB         |
| L1 Hit Rate                  | 52.6%           |
| L2 Hit Rate                  | 81.3%           |
| L2 Bandwidth                 | 911.8 GB/s      |
| Coalescing Efficiency        | 25.0%           |
| Global Load Efficiency       | 12.5%           |
| Global Store Efficiency      | 6.25%           |
| LDS Bank Conflicts           | 0               |
| Atomic Latency               | 0               |

### Compute Profile

| Metric                       | Value           |
|------------------------------|-----------------|
| Total FLOPs per dispatch     | ~1.39 TFLOPS    |
| Throughput (HBM-normalized)  | ~315,327 GFLOPS |
| HBM Arithmetic Intensity     | 267.7 FLOP/Byte |
| L2 Arithmetic Intensity      | 49.8 FLOP/Byte  |
| L1 Arithmetic Intensity      | 24.2 FLOP/Byte  |
| Avg kernel duration          | ~4.4 ms         |

### Analysis

- **Compute-bound**: Low HBM bandwidth utilization (~3.2%) combined with high arithmetic intensity (~268 FLOP/Byte) confirms the kernel is compute-bound, which is optimal for GEMM workloads.
- **Good L2 reuse**: 81.3% L2 hit rate indicates effective blocking and data reuse through the L2 cache.
- **Zero LDS conflicts and atomic latency**: The blocking strategy avoids shared memory bank conflicts, and the all-scatter communication via Iris remote stores incurs no detectable atomic latency overhead.
- **Coalescing**: 25% coalescing efficiency and 12.5% global load efficiency are expected for this tiled GEMM access pattern with large tiles (BLK_M=256, BLK_N=64, BLK_K=64).
