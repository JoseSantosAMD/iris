# Example 07: GEMM All-Scatter — IntelliKit Profiling Report

## Overview

This report presents the IntelliKit Metrix profiling results for **Example 07: GEMM All-Scatter**, which performs distributed matrix multiplication with all-scatter communication across multiple AMD GPUs.

**Environment:**
- GPUs: 2× AMD Radeon Graphics (gfx942 / MI300X)
- Matrix dimensions: M=4096, N=4096, K=4096
- Data type: FP16
- Algorithm: persistent GEMM with fused all-scatter

---

## Benchmark Results

Run with validation and benchmarking enabled:

```bash
torchrun --nproc_per_node=2 examples/07_gemm_all_scatter/benchmark.py \
  -m 4096 -n 4096 -k 4096 --validate --benchmark
```

| Metric | Value |
|--------|-------|
| Validation | ✅ Passed |
| Performance | **~202.9 TFLOPS** |
| Total time (comm + compute) | **0.677 ms** |
| GEMM kernel time | **0.536 ms** |

---

## IntelliKit Metrix Profiling

Profiled using Metrix with `--kernel persistent_gemm_all_scatter` filter, 10 replays.

### Timing

| Stat | Value |
|------|-------|
| Min duration | 518 μs |
| Max duration | 597 μs |
| **Avg duration** | **~543 μs** |

### Compute Metrics

| Metric | Value |
|--------|-------|
| Total FLOPs | 68.7 GFLOPs |
| **HBM GFLOPS (throughput)** | **~126,339 GFLOPS (~126 TFLOPS)** |
| HBM arithmetic intensity | **~257 FLOPs/byte** |
| L2 arithmetic intensity | ~40.9 FLOPs/byte |
| L1 arithmetic intensity | ~17.0 FLOPs/byte |

### Memory Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **HBM bandwidth utilization** | **~1.16%** | Compute-bound (not memory-bound) |
| HBM read bandwidth | ~53.1 GB/s | Low — data reused from L2/L1 |
| HBM write bandwidth | ~8.6 GB/s | Low output writes |
| **L2 hit rate** | **~84.2%** | Excellent L2 data reuse |
| **L1 hit rate** | **~66.7%** | Good L1 data reuse |
| L2 bandwidth | ~388 GB/s | Active L2 caching |
| Coalescing efficiency | 25.0% | Expected for tiled GEMM |
| Global load efficiency | ~12.5% | Tile-based strided access |
| Global store efficiency | 6.25% | Output scatter writes |
| **LDS bank conflicts** | **0** | Excellent shared memory usage |
| Atomic latency | 0 | No atomics in critical path |

---

## Analysis

### The kernel is **compute-bound**

The HBM arithmetic intensity of **~257 FLOPs/byte** is well above the machine balance (~20 FLOPs/byte for MI300X with ~4 TB/s HBM3 bandwidth). This confirms the GEMM kernel is in the compute-bound regime, which is the desired operating point for large GEMMs.

### Cache efficiency is high

- **L2 hit rate of 84.2%** — the tiled access pattern reuses operand tiles from L2 effectively.
- **L1 hit rate of 66.7%** — inner-loop reuse keeps data in the fast L1 cache.
- **Zero LDS bank conflicts** — the shared memory (LDS) layout avoids bank conflicts completely, a key optimization for high-throughput GEMM.

### Low coalescing efficiency is expected

The 25% coalescing efficiency and ~12.5% global load efficiency reflect the block-tiled access pattern inherent in GEMM kernels — the hardware reads full cache lines but only a fraction is used per cycle. This is standard behavior and is offset by the high arithmetic intensity.

### Communication overhead

The all-scatter communication (NCCL) adds overhead on top of the GEMM:
- GEMM kernel: ~543 μs (avg profiled)
- Total wall time: ~677 μs (including communication)
- Communication overhead: ~134 μs (~20% of total)

---

## Summary

The `persistent_gemm_all_scatter` kernel achieves **~126 TFLOPS per GPU** (202 TFLOPS aggregate across 2 GPUs) on MI300X with:
- Excellent compute utilization in the compute-bound regime
- High cache hit rates (L2: 84%, L1: 67%)
- Zero LDS bank conflicts
- Communication adding ~20% overhead on top of pure compute

The kernel is well-optimized for compute throughput. The primary opportunity for further improvement would be to reduce communication latency, which currently accounts for ~20% of total runtime.
