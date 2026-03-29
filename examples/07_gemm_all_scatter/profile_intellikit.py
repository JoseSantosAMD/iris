#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
IntelliKit Metrix profiling script for the GEMM all-scatter benchmark (Example 7).

Profiles Example 7 using Metrix to collect GPU hardware metrics such as compute
throughput, memory bandwidth, and cache hit rates.

Usage:
    python profile_intellikit.py [options]

Examples:
    # Profile with all metrics (2 GPUs)
    python profile_intellikit.py --num_ranks 2

    # Profile with memory metrics only
    python profile_intellikit.py --num_ranks 2 --profile memory

    # Filter to show only GEMM kernels, save results to JSON
    python profile_intellikit.py --num_ranks 2 --kernel gemm --output profile.json

    # Quick timing-only profile
    python profile_intellikit.py --num_ranks 2 --time_only
"""

import argparse
import os
import sys

from metrix import Metrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile the GEMM all-scatter benchmark (Example 7) with IntelliKit Metrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Benchmark parameters forwarded to benchmark.py
    parser.add_argument("-m", type=int, default=8192, help="Number of rows in matrix A")
    parser.add_argument("-n", type=int, default=4608, help="Number of columns in matrix B")
    parser.add_argument("-k", type=int, default=36864, help="Common dimension between matrices A and B")
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Datatype of computation",
    )
    parser.add_argument("--BLK_M", type=int, default=256, help="Block size M")
    parser.add_argument("--BLK_N", type=int, default=64, help="Block size N")
    parser.add_argument("--BLK_K", type=int, default=64, help="Block size K")
    parser.add_argument("--gsize_m", type=int, default=6, help="L2-cache locality swizzle parameter")
    parser.add_argument("--num_stages", type=int, default=2, help="Number of pipeline stages")
    parser.add_argument("--heap_size", type=int, default=1 << 33, help="Iris heap size")
    parser.add_argument(
        "--gemm_sms",
        type=int,
        default=None,
        help="Number of SMs for persistent GEMM algorithm (default: auto-detected)",
    )
    parser.add_argument("-r", "--num_ranks", type=int, default=2, help="Number of ranks/GPUs")

    # Metrix profiler options
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["quick", "memory", "compute"],
        help="Use a preset Metrix profile",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Comma-separated list of Metrix metrics to collect (e.g. memory.l2_hit_rate,compute.hbm_gflops)",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default=None,
        help="Filter profiled kernels by name substring (regex)",
    )
    parser.add_argument(
        "--time_only",
        action="store_true",
        help="Collect kernel timing only, skip hardware counters",
    )
    parser.add_argument(
        "--num_replays",
        type=int,
        default=1,
        help="Number of profiling replays for metric aggregation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save profiling results to file (.json, .csv, or .txt)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Display only the top K kernels by duration",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="GPU architecture override (e.g. gfx942). Auto-detected if not specified.",
    )

    return parser.parse_args()


def build_benchmark_command(args) -> str:
    """Build the torchrun command for benchmark.py."""
    benchmark_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.py")

    cmd_parts = [
        f"torchrun --nproc_per_node {args.num_ranks}",
        benchmark_path,
        f"-m {args.m}",
        f"-n {args.n}",
        f"-k {args.k}",
        f"--datatype {args.datatype}",
        f"--BLK_M {args.BLK_M}",
        f"--BLK_N {args.BLK_N}",
        f"--BLK_K {args.BLK_K}",
        f"--gsize_m {args.gsize_m}",
        f"--num_stages {args.num_stages}",
        f"--heap_size {args.heap_size}",
        "--benchmark",
    ]

    if args.gemm_sms is not None:
        cmd_parts.append(f"--gemm_sms {args.gemm_sms}")

    return " ".join(cmd_parts)


def print_results(results, top_k=None):
    """Print profiling results in a human-readable format."""
    print()
    print("=" * 80)
    print("Metrix Profile: GEMM All-Scatter Benchmark")
    print(f"Total kernels profiled: {results.total_kernels}")
    print("=" * 80)

    kernels = results.kernels
    if top_k is not None:
        kernels = sorted(kernels, key=lambda k: k.duration_us.avg if k.duration_us else 0, reverse=True)[:top_k]
        print(f"(Showing top {top_k} kernels by duration)")

    for kernel in kernels:
        print()
        print("-" * 80)
        print(f"Kernel: {kernel.name}")
        if kernel.duration_us is not None:
            print(
                f"  Duration: {kernel.duration_us.avg:.2f} μs  (min={kernel.duration_us.min:.2f}, max={kernel.duration_us.max:.2f})"
            )

        if kernel.metrics:
            # Group metrics by category
            categories = {}
            for metric_name, stats in kernel.metrics.items():
                category = metric_name.split(".")[0].upper()
                categories.setdefault(category, {})[metric_name] = stats

            for category, metrics in sorted(categories.items()):
                print(f"  {category}:")
                for metric_name, stats in sorted(metrics.items()):
                    short_name = metric_name.split(".", 1)[-1].replace("_", " ").title()
                    print(f"    {short_name:<45} {stats.avg:>12.2f}  (min={stats.min:.2f}, max={stats.max:.2f})")

    print()
    print("=" * 80)


def save_results(results, output_path):
    """Save profiling results to a file."""
    import json

    ext = os.path.splitext(output_path)[-1].lower()

    if ext == ".json":
        data = {
            "command": results.command,
            "total_kernels": results.total_kernels,
            "kernels": [
                {
                    "name": k.name,
                    "duration_us": {
                        "avg": k.duration_us.avg if k.duration_us else None,
                        "min": k.duration_us.min if k.duration_us else None,
                        "max": k.duration_us.max if k.duration_us else None,
                    }
                    if k.duration_us
                    else None,
                    "metrics": {metric: {"avg": s.avg, "min": s.min, "max": s.max} for metric, s in k.metrics.items()},
                }
                for k in results.kernels
            ],
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {output_path}")
    elif ext == ".csv":
        import csv

        rows = []
        for k in results.kernels:
            row = {"kernel": k.name, "duration_us_avg": k.duration_us.avg if k.duration_us else None}
            for metric, s in k.metrics.items():
                row[metric] = s.avg
            rows.append(row)
        if rows:
            fieldnames = list(rows[0].keys())
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        print(f"Results saved to {output_path}")
    else:
        # Plain text
        with open(output_path, "w") as f:
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                print_results(results)
            f.write(buf.getvalue())
        print(f"Results saved to {output_path}")


def main():
    args = parse_args()

    # Build benchmark command
    command = build_benchmark_command(args)
    print(f"Profiling command: {command}")

    # Parse metrics list if provided
    metrics = None
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",")]

    # Initialize Metrix profiler
    profiler = Metrix(arch=args.arch)

    # Run benchmark under Metrix profiling
    results = profiler.profile(
        command=command,
        metrics=metrics,
        profile=args.profile,
        kernel_filter=args.kernel,
        time_only=args.time_only,
        num_replays=args.num_replays,
        aggregate_by_kernel=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    # Display results
    print_results(results, top_k=args.top)

    # Save results to file if requested
    if args.output:
        save_results(results, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
