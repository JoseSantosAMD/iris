#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
Profile Example 07 (GEMM All-Scatter) with IntelliKit Metrix.

Usage:
    python profile_with_metrix.py [--output results.json] [--m M] [--n N] [--k K]

This script profiles the persistent_gemm_all_scatter kernel using Metrix
and saves memory + compute metrics to a JSON file.
"""

import argparse
import json
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile GEMM All-Scatter with IntelliKit Metrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", type=int, default=4096, help="Number of rows in matrix A")
    parser.add_argument("-n", type=int, default=2048, help="Number of columns in matrix B")
    parser.add_argument("-k", type=int, default=4096, help="Common dimension")
    parser.add_argument("--num_ranks", type=int, default=2, help="Number of GPU ranks")
    parser.add_argument("--num_replays", type=int, default=3, help="Number of profiling replays")
    parser.add_argument("--master_port", type=int, default=29510, help="Torchrun master port")
    parser.add_argument(
        "--output",
        type=str,
        default="metrix_profiling_results.json",
        help="Output JSON file for profiling results",
    )
    return parser.parse_args()


def run_profile(profile_type, benchmark_cmd, output_file, num_replays, timeout=600):
    """Run metrix profiling with the given profile type."""
    cmd = [
        "metrix",
        "profile",
        "--profile",
        profile_type,
        "--output",
        output_file,
        "--num-replays",
        str(num_replays),
        "--timeout",
        str(timeout),
        "--kernel",
        "persistent_gemm",
        benchmark_cmd,
    ]
    env = os.environ.copy()
    env["HSA_NO_SCRATCH_RECLAIM"] = "1"
    print(f"[profile_with_metrix] Running {profile_type} profile...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[profile_with_metrix] WARNING: {profile_type} profiling failed (exit {result.returncode})")
        print(result.stderr[-2000:] if result.stderr else "")
        return None
    return output_file


def extract_kernel_data(json_file, kernel_pattern="persistent_gemm"):
    """Extract kernel metrics from a Metrix JSON output file."""
    if not json_file or not os.path.exists(json_file):
        return {}
    with open(json_file) as f:
        data = json.load(f)
    for key, value in data.items():
        if kernel_pattern in key:
            return {key: value}
    return {}


def main():
    args = parse_args()

    example_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    benchmark_script = os.path.join(example_dir, "benchmark.py")

    benchmark_cmd = (
        f"torchrun --nproc_per_node={args.num_ranks} --master_port={args.master_port} "
        f"{benchmark_script} -m {args.m} -n {args.n} -k {args.k}"
    )

    # Run benchmark without profiling first to get timing
    print("[profile_with_metrix] Running benchmark for timing...")
    env = os.environ.copy()
    env["HSA_NO_SCRATCH_RECLAIM"] = "1"
    benchmark_result = subprocess.run(
        benchmark_cmd.split() + ["--benchmark"],
        env=env,
        capture_output=True,
        text=True,
    )
    timing_data = {}
    if benchmark_result.returncode == 0:
        for line in benchmark_result.stdout.splitlines():
            if "tflops" in line.lower():
                print(f"  {line.strip()}")
        # Try to parse log.json if it was written
        if os.path.exists("log.json"):
            with open("log.json") as f:
                timing_data = json.load(f)
    else:
        print("[profile_with_metrix] Benchmark run failed:", benchmark_result.stderr[-500:])

    # Memory profile
    mem_output = args.output.replace(".json", "_memory.json")
    mem_file = run_profile("memory", benchmark_cmd, mem_output, args.num_replays)
    mem_data = extract_kernel_data(mem_file) if mem_file else {}

    # Compute profile
    compute_output = args.output.replace(".json", "_compute.json")
    compute_file = run_profile("compute", benchmark_cmd, compute_output, args.num_replays)
    compute_data = extract_kernel_data(compute_file) if compute_file else {}

    # Combine results
    results = {
        "benchmark": {
            "description": "GEMM All-Scatter benchmark (Example 07)",
            "matrix_size": {"M": args.m, "N": args.n, "K": args.k},
            "dtype": "fp16",
            "world_size": args.num_ranks,
            "algorithm": "persistent_gemm_all_scatter",
        },
        "timing": timing_data,
        "kernel_profiling": {},
    }

    # Merge memory and compute metrics per kernel
    all_kernel_keys = set(list(mem_data.keys()) + list(compute_data.keys()))
    for kernel_key in all_kernel_keys:
        merged_metrics = {}
        duration = None
        if kernel_key in mem_data:
            duration = mem_data[kernel_key].get("duration_us")
            merged_metrics.update(mem_data[kernel_key].get("metrics", {}))
        if kernel_key in compute_data:
            if duration is None:
                duration = compute_data[kernel_key].get("duration_us")
            merged_metrics.update(compute_data[kernel_key].get("metrics", {}))
        results["kernel_profiling"][kernel_key] = {
            "duration_us": duration,
            "metrics": merged_metrics,
        }

    # Save combined results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[profile_with_metrix] Results saved to: {args.output}")

    # Print summary
    print("\n" + "=" * 70)
    print("IntelliKit Metrix Profiling Summary: GEMM All-Scatter (Example 07)")
    print("=" * 70)
    print(f"  Matrix: M={args.m}, N={args.n}, K={args.k} | fp16 | {args.num_ranks} GPUs")
    if timing_data:
        tflops = timing_data.get("tflops")
        total_ms = timing_data.get("total_ms")
        print(f"  Throughput: {tflops:.2f} TFLOPS" if isinstance(tflops, float) else "  Throughput: N/A")
        print(f"  Latency:    {total_ms:.3f} ms" if isinstance(total_ms, float) else "  Latency:    N/A")

    for kernel_key, kdata in results["kernel_profiling"].items():
        kernel_name = kernel_key.split(":")[-1] if ":" in kernel_key else kernel_key
        print(f"\nKernel: {kernel_name}")
        dur = kdata.get("duration_us")
        if dur:
            print(f"  Duration: avg={dur['avg']:.1f} µs, min={dur['min']:.1f} µs, max={dur['max']:.1f} µs")
        for metric_name, stats in (kdata.get("metrics") or {}).items():
            avg = stats.get("avg", "N/A")
            print(f"  {metric_name}: {avg:.4f}" if isinstance(avg, float) else f"  {metric_name}: {avg}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
