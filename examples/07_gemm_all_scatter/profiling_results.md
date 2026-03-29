# Example 07 – GEMM All-Scatter: IntelliKit Profiling Results

## System Configuration

| Property | Value |
|---|---|
| GPU | AMD Instinct MI300X (gfx942) |
| Compute Units | 304 |
| HBM Capacity | 256 GB |
| GPUs Used | 2 |
| Framework | Iris (Triton-based) |

---

## Benchmark Results

### Small Problem (M=4096, N=4096, K=4096)

| Metric | Value |
|---|---|
| Kernel | `persistent_gemm_all_scatter` |
| World Size | 2 |
| Local N per rank | 2048 |
| **Performance** | **205.8 TFLOPS** |
| Total Latency | 0.668 ms |
| GEMM Latency | 0.530 ms |
| Tile Config | BLK_M=256, BLK_N=64, BLK_K=64 |
| Total Tiles | 512 |

### Large Problem (M=8192, N=4608, K=36864) — Default Sizes

| Metric | Value |
|---|---|
| Kernel | `persistent_gemm_all_scatter` |
| World Size | 2 |
| Local N per rank | 2304 |
| **Performance** | **612.2 TFLOPS** |
| Total Latency | 4.546 ms |
| GEMM Latency | 4.408 ms |
| Tile Config | BLK_M=256, BLK_N=64, BLK_K=64 |
| Total Tiles | 1152 |

---

## IntelliKit Metrix Profiling

### Memory Profile – Small Problem (M=4096, N=4096, K=4096)

| Metric | Min | Max | Avg |
|---|---|---|---|
| **Kernel Duration** | 532.4 μs | 541.9 μs | **537.1 μs** |
| HBM Bandwidth Utilization | 1.18% | 1.18% | **1.18%** |
| HBM Read Bandwidth | 53.8 GB/s | 53.9 GB/s | **53.8 GB/s** |
| HBM Write Bandwidth | 8.6 GB/s | 8.8 GB/s | **8.7 GB/s** |
| Total HBM Bytes Transferred | 266 MB | 270 MB | **268 MB** |
| L1 Hit Rate | 66.73% | 66.73% | **66.73%** |
| L2 Hit Rate | 84.19% | 84.22% | **84.20%** |
| L2 Bandwidth | 389 GB/s | 394 GB/s | **392 GB/s** |
| Coalescing Efficiency | 25.0% | 25.0% | **25.0%** |
| Global Load Efficiency | 12.54% | 12.54% | **12.54%** |
| Global Store Efficiency | 6.25% | 6.25% | **6.25%** |
| LDS Bank Conflicts | 0.0 | 0.0 | **0.0** |
| Atomic Latency | 0.0 | 0.0 | **0.0** |

### Compute Profile – Small Problem (M=4096, N=4096, K=4096)

| Metric | Min | Max | Avg |
|---|---|---|---|
| **Kernel Duration** | 537.2 μs | 565.3 μs | **547.6 μs** |
| Total FLOPS | 68.7 GFLOPS | 68.7 GFLOPS | **68.7 GFLOPS** |
| Compute Throughput (HBM-normalized) | 121.6 TFLOPS | 127.9 TFLOPS | **125.5 TFLOPS** |
| HBM Arithmetic Intensity | 254.3 FLOP/byte | 258.4 FLOP/byte | **256.9 FLOP/byte** |
| L2 Arithmetic Intensity | 40.9 FLOP/byte | 40.9 FLOP/byte | **40.9 FLOP/byte** |
| L1 Arithmetic Intensity | 17.0 FLOP/byte | 17.0 FLOP/byte | **17.0 FLOP/byte** |

### Memory Profile – Large Problem (M=8192, N=4608, K=36864)

| Metric | Min | Max | Avg |
|---|---|---|---|
| **Kernel Duration** | 4609.9 μs | 4731.7 μs | **4683.2 μs** |
| HBM Bandwidth Utilization | 3.40% | 3.43% | **3.41%** |
| HBM Read Bandwidth | 177.2 GB/s | 178.8 GB/s | **177.9 GB/s** |
| HBM Write Bandwidth | 2.72 GB/s | 2.74 GB/s | **2.72 GB/s** |
| Total HBM Bytes Transferred | 5.10 GB | 5.16 GB | **5.13 GB** |
| L1 Hit Rate | 52.65% | 52.65% | **52.65%** |
| L2 Hit Rate | 81.81% | 81.88% | **81.84%** |
| L2 Bandwidth | 975 GB/s | 994 GB/s | **985 GB/s** |
| Coalescing Efficiency | 25.0% | 25.0% | **25.0%** |
| Global Load Efficiency | 12.50% | 12.50% | **12.50%** |
| Global Store Efficiency | 6.25% | 6.25% | **6.25%** |
| LDS Bank Conflicts | 0.0 | 0.0 | **0.0** |
| Atomic Latency | 0.0 | 0.0 | **0.0** |

### Compute Profile – Large Problem (M=8192, N=4608, K=36864)

| Metric | Min | Max | Avg |
|---|---|---|---|
| **Kernel Duration** | 5010.5 μs | 5196.3 μs | **5076.4 μs** |
| Total FLOPS | 1.39 TFLOPS | 1.39 TFLOPS | **1.39 TFLOPS** |
| Compute Throughput (HBM-normalized) | 267.8 TFLOPS | 277.7 TFLOPS | **274.1 TFLOPS** |
| HBM Arithmetic Intensity | 271.6 FLOP/byte | 276.1 FLOP/byte | **274.5 FLOP/byte** |
| L2 Arithmetic Intensity | 49.8 FLOP/byte | 49.8 FLOP/byte | **49.8 FLOP/byte** |
| L1 Arithmetic Intensity | 24.2 FLOP/byte | 24.2 FLOP/byte | **24.2 FLOP/byte** |

---

## Analysis & Observations

### Performance
- The GEMM all-scatter kernel achieves **~612 TFLOPS** at the default large problem size on 2× MI300X GPUs, with communication overhead kept to ~3% of total time (`gemm_ms ≈ 4.41 ms`, `total_ms ≈ 4.55 ms`).
- At small problem sizes (4k³), performance is **~206 TFLOPS**, as expected for memory-bound workloads at this scale.

### Memory Behavior
- **L2 hit rate is consistently high** (~81–84%), demonstrating effective reuse of data through the L2 cache across the K-dimension loop.
- **L1 hit rate is ~53–67%**, showing moderate reuse; this declines for larger problems as working sets exceed L1 capacity.
- **Coalescing efficiency of 25%** and **global load efficiency of ~12.5%** indicate that memory access patterns could benefit from improved tiling strategies. The 64-element BLK_N × fp16 stores translate to 128-byte transactions, but the scatter pattern for cross-rank writes contributes to suboptimal coalescing.
- **No LDS bank conflicts** and **no atomic latency** detected, confirming efficient shared-memory and atomic usage.
- **HBM bandwidth utilization is low** (1–3%), confirming this kernel is **compute-bound** at these problem sizes.

### Compute Intensity
- **HBM arithmetic intensity of ~257–274 FLOP/byte** well above the roofline compute-bound threshold for MI300X (~200 FLOP/byte at fp16 peak), confirming **compute-bound** behavior.
- The L2 arithmetic intensity (~41–50 FLOP/byte) and L1 arithmetic intensity (~17–24 FLOP/byte) decrease up the memory hierarchy, as expected.

### Optimization Opportunities
1. **Coalescing** (25% efficiency): BLK_N=64 with fp16 produces 128-byte writes. Increasing BLK_N or using larger tile sizes may improve coalescing for the scatter writes.
2. **Global store efficiency** (6.25%): The iris remote stores use per-element scatter semantics; batching writes or using larger BLOCK_SIZE_N would increase store granularity.
3. **L1 hit rate decline** at large sizes: Increasing `num_stages` for software pipelining could improve L1 data reuse for larger K dimensions.

---

*Profiled with IntelliKit Metrix on AMD MI300X (gfx942), 2 GPUs*
