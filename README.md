<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
-->

<p align="center">
  <img src="docs/images/logo.png" width="300px" />
</p>

# Iris: First-Class Multi-GPU Programming Experience in Triton

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/ROCm/iris/blob/main/.github/workflows/lint.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://github.com/ROCm/iris/actions/workflows/iris-tests.yml"><img src="https://github.com/ROCm/iris/actions/workflows/iris-tests.yml/badge.svg" alt="Iris Tests"></a>
  <a href="https://doi.org/10.5281/zenodo.17382307"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.17382307.svg" alt="DOI"></a>
  <a href="https://doi.org/10.48550/arXiv.2511.12500"><img src="https://img.shields.io/badge/cs.DC%2C%20cs.LG-arXiv%3A2511.12500-B31B1B.svg" alt="DOI"></a>
</p>

Iris is a Triton-based framework for Remote Memory Access (RMA) operations developed by AMD's Research and Advanced Development team. Iris provides SHMEM-like APIs within Triton for Multi-GPU programming. Iris' goal is to make Multi-GPU programming a first-class citizen in Triton while retaining Triton's programmability and performance.

## Table of Contents

- [Latest Updates](#latest-with-iris-)
- [Key Features](#key-features)
- [Quick Start](#quick-start-guide)
- [API Examples](#api-example)
- [Documentation](#documentation)
- [Supported GPUs](#supported-gpus)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Support](#support)
- [How to Cite](#how-to-cite)
- [License](#license)

## Latest with Iris 🔥

- [16/11/2025] Paper released: _[Iris: First-Class Multi-GPU Programming Experience in Triton](https://arxiv.org/abs/2511.12500)_
- [04/11/2025] Paper released: _[Eliminating Multi-GPU Performance Taxes: A Systems Approach to Efficient Distributed LLMs](https://arxiv.org/abs/2511.02168)_
- [02/10/2025] [Iris + Gluon Released](https://rocm.github.io/iris/reference/gluon/overview.html)
- [18/09/2025] [FlashDecode with Iris](https://github.com/ROCm/iris/tree/main/examples/13_flash_decode)
- [16/09/2025] Iris was presented in [Chinese](https://youtu.be/wW14w1QNrY8) for participants of the AMD Distributed Inference Kernel Contest
- [12/09/2025] Presented Iris at GPUMode [[talk](https://www.youtube.com/watch?v=i6Y2EelEC04)] | [[slides](https://github.com/ROCm/iris/blob/main/docs/slides/Awad-Osama-Potter%20-%20Iris%20Multi-GPU%20Programming%20Made%20Easier%20(GPU%20Mode).pdf)]
- [27/08/2025] [AMD's GPU Mode Competition Announced](https://amdchallenge2025.datamonsters.com/)
- [14/08/2025] Iris All-Scatter Taxonomy Released [[documentation](https://rocm.github.io/iris/conceptual/taxonomy.html)] | [[video](https://youtu.be/fYMdPe9UpHE)]
- [25/06/2025] Iris Released

## Key Features

- **SHMEM-like RMA**: Iris provides SHMEM-like RMA support in Triton.
- **Simple and Intuitive API**: Iris provides simple and intuitive RMA APIs. Writing multi-GPU programs is as easy as writing single-GPU programs.
- **Triton-based**: Iris is built on top of Triton and inherits Triton's performance and capabilities.
- **Triton Gluon-based backend (Experimental)**: Includes an optional backend built on Triton’s Gluon language, a lower-level GPU programming model that exposes explicit control over layouts, memory, and data movement—ideal for users seeking maximal performance and hardware-level optimization.

## Documentation

- [Setup Alternatives](https://rocm.github.io/iris/getting-started/installation.html)
- [Examples](https://rocm.github.io/iris/reference/examples.html)
- [Programming Model](https://rocm.github.io/iris/conceptual/programming-model.html)
- [Taxonomy of Multi-GPU Programming Patterns](https://rocm.github.io/iris/conceptual/taxonomy.html)
- [Fine-grained GEMM & Communication Overlap](https://rocm.github.io/iris/conceptual/finegrained-overlap.html)
- [API Reference](https://rocm.github.io/iris/reference/api-reference.html)

## API Example

### Basic Remote Memory Operations

Here's a simple example showing how to perform remote memory operations between GPUs using Iris. This example demonstrates the core concept of one GPU (rank 0) directly writing to another GPU's (rank 1) memory:

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import triton.language as tl
import iris

# Device-side APIs
@triton.jit
def kernel(buffer, buffer_size: tl.constexpr, block_size: tl.constexpr, heap_bases_ptr):
    # Compute start index of this block
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)

    # Guard for out-of-bounds accesses
    mask = offsets < buffer_size

    # Store 1 in the target buffer at each offset
    source_rank = 0
    target_rank = 1
    iris.store(buffer + offsets, 1,
            source_rank, target_rank,
            heap_bases_ptr, mask=mask)

def _worker(rank, world_size):
    # Torch distributed initialization
    device_id = rank % torch.cuda.device_count()
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method="tcp://127.0.0.1:29500",
        device_id=torch.device(f"cuda:{device_id}")
    )

    # Iris initialization
    heap_size = 2**30   # 1GiB symmetric heap for inter-GPU communication
    iris_ctx = iris.iris(heap_size)
    cur_rank = iris_ctx.get_rank()

    # Iris tensor allocation
    buffer_size = 4096  # 4K elements buffer
    buffer = iris_ctx.zeros(buffer_size, device="cuda", dtype=torch.float32)

    # Launch the kernel on rank 0
    block_size = 1024
    grid = lambda meta: (triton.cdiv(buffer_size, meta["block_size"]),)
    source_rank = 0
    if cur_rank == source_rank:
        kernel[grid](
            buffer,
            buffer_size,
            block_size,
            iris_ctx.get_heap_bases(),
        )

    # Synchronize all ranks
    iris_ctx.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2  # Using two ranks
    mp.spawn(_worker, args=(world_size,), nprocs=world_size, join=True)
```

This example shows the key concepts:
- **Symmetric Heap**: Each GPU allocates a symmetric heap for remote access
- **Direct RMA**: Rank 0 can directly store to Rank 1's memory without explicit message passing
- **Triton Integration**: Standard Triton kernel with Iris store operations

### Gluon-style API (Experimental)

Iris also provides an experimental API using Triton's Gluon language for users who need finer-grained control over memory layouts and data movement. The Gluon backend offers a cleaner API by encapsulating heap bases within a device context:

> [!NOTE]
> **Requirements for Gluon backend**: ROCm 7.0+ and Triton commit [aafec417bded34db6308f5b3d6023daefae43905](https://github.com/triton-lang/triton/tree/aafec417bded34db6308f5b3d6023daefae43905) or later are required to use the experimental Gluon APIs.

Key differences from the standard API:
- Device context encapsulates heap bases
- More explicit control over memory layouts
- Ideal for performance-critical applications

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import iris.experimental.iris_gluon as iris_gl

# Device-side APIs - context encapsulates heap_bases
@gluon.jit
def kernel(IrisDeviceCtx: gl.constexpr, context_tensor,
          buffer, buffer_size: gl.constexpr, block_size: gl.constexpr):
    # Initialize device context from tensor
    ctx = IrisDeviceCtx.initialize(context_tensor)
    
    pid = gl.program_id(0)
    block_start = pid * block_size
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    offsets = block_start + gl.arange(0, block_size, layout=layout)
    mask = offsets < buffer_size

    # Store 1 in the target buffer - no need to pass heap_bases separately!
    target_rank = 1
    ctx.store(buffer + offsets, 1, target_rank, mask=mask)

def _worker(rank, world_size):
    # Torch distributed initialization
    device_id = rank % torch.cuda.device_count()
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method="tcp://127.0.0.1:29500",
        device_id=torch.device(f"cuda:{device_id}")
    )

    # Iris initialization
    heap_size = 2**30   # 1GiB symmetric heap
    iris_ctx = iris_gl.iris(heap_size)
    context_tensor = iris_ctx.get_device_context()  # Get encoded context
    cur_rank = iris_ctx.get_rank()
    
    # Iris tensor allocation
    buffer_size = 4096  # 4K elements buffer
    buffer = iris_ctx.zeros(buffer_size, device="cuda", dtype=torch.float32)
    
    # Launch the kernel on rank 0
    block_size = 1024
    grid = (buffer_size + block_size - 1) // block_size
    source_rank = 0
    if cur_rank == source_rank:
        kernel[(grid,)](iris_gl.IrisDeviceCtx, context_tensor, 
                       buffer, buffer_size, block_size, num_warps=1)

    # Synchronize all ranks
    iris_ctx.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2  # Using two ranks
    mp.spawn(_worker, args=(world_size,), nprocs=world_size, join=True)
```

The Gluon API provides cleaner syntax by eliminating the need to pass `heap_bases_ptr` separately, making the code more maintainable and easier to read.

## Quick Start Guide

### Prerequisites

Before installing Iris, ensure you have:

- **Python**: Version 3.10 or higher
- **PyTorch**: Version 2.0+ with ROCm support
- **ROCm/HIP Runtime**: Version 6.3.1 or higher
- **Triton**: Compatible with your ROCm version
- **Build Tools**: setuptools>=61
- **Hardware**: AMD GPU (MI300X, MI350X, MI355X recommended)

> [!TIP]
> Don't have an AMD GPU? You can still contribute! See our [Contributing Guide](docs/CONTRIBUTING.md) for details on developing without GPU access.

### Quick Installation

For a quick installation directly from the repository:

```shell
pip install git+https://github.com/ROCm/iris.git
```

### Docker Compose (Recommended for Development)

The recommended way to get started is using Docker Compose, which provides a complete development environment with all dependencies pre-installed:

```shell
# Start the development container (first build takes 45-60 minutes)
docker compose up --build -d

# or depending on your docker version
docker-compose up --build -d

# Attach to the running container
docker attach iris-dev

# Install Iris in development mode
cd iris && pip install -e ".[dev]"
```

> [!IMPORTANT]
> The initial Docker build can take 45-60 minutes. Please be patient and do not cancel the build process.

### Running Your First Example

Once installed, try running a simple example:

```shell
# Run a basic load operation example
python examples/00_load/load_bench.py

# Run tests
pytest tests/unittests/
```

### Next Steps

- Explore the [examples](examples/) directory for ready-to-run scripts
- Check out [Programming Model](https://rocm.github.io/iris/conceptual/programming-model.html) documentation
- Read the [API Reference](https://rocm.github.io/iris/reference/api-reference.html)

For more installation options (baremetal, Docker, Apptainer), see the [Installation Guide](https://rocm.github.io/iris/getting-started/installation.html).

## Supported GPUs

Iris currently supports:

- MI300X, MI350X & MI355X

> [!NOTE]
> Iris may work on other AMD GPUs with ROCm compatibility.

## Roadmap

We plan to extend Iris with the following features:

- **Extended GPU Support**: Testing and optimization for additional AMD GPUs with ROCm compatibility
- **RDMA Support**: Multi-node support using Remote Direct Memory Access (RDMA) for distributed computing across multiple machines
- **End-to-End Integration**: Comprehensive examples covering various use cases and production deployment patterns
- **Performance Optimizations**: Continued improvements to kernel performance and memory efficiency
- **Enhanced Documentation**: More tutorials, best practices guides, and performance tuning tips

## Troubleshooting

### Common Issues

#### Build Failures

**Issue**: Docker build takes too long or appears stuck
- **Solution**: The initial build can take 45-60 minutes. This is normal. Do not cancel the process.

**Issue**: ROCm/HIP compilation errors
- **Solution**: Ensure you have ROCm 6.3.1 or higher installed. Check your ROCm version with `rocm-smi --version`.

#### Runtime Errors

**Issue**: `RuntimeError: No GPU available`
- **Solution**: Verify AMD GPU is detected with `rocm-smi`. Ensure ROCm drivers are properly installed.

**Issue**: `ImportError: cannot import name 'iris'`
- **Solution**: Ensure Iris is installed with `pip install -e ".[dev]"` in the repository directory.

**Issue**: NCCL initialization failures
- **Solution**: Check that PyTorch is built with ROCm support. Verify with `python -c "import torch; print(torch.version.hip)"`.

#### Development Without GPU

**Issue**: I don't have access to an AMD GPU
- **Solution**: You can still contribute! Make code changes locally and rely on CI testing. Run linting locally with `ruff check . --fix && ruff format .`.

### Getting Help

If you encounter issues not covered here:

1. Check the [documentation](https://rocm.github.io/iris/)
2. Search [existing issues](https://github.com/ROCm/iris/issues)
3. Open a [new issue](https://github.com/ROCm/iris/issues/new/choose) with:
   - Your environment details (ROCm version, GPU model, OS)
   - Steps to reproduce the problem
   - Full error messages and logs

# Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details on how to set up your development environment and contribute to the project.

## Support

Need help? We're here to support you! Here are a few ways to get in touch:

1. **Open an Issue**: Found a bug or have a feature request? [Open an issue](https://github.com/ROCm/iris/issues/new/choose) on GitHub
2. **Contact the Team**: If GitHub issues aren't working for you or you need to reach us directly, feel free to contact our development team

We welcome your feedback and contributions!

## How to Cite

If you use Iris or reference it in your research, please cite our work:

```bibtex
@misc{Awad:2025:IFM,
  author        = {Muhammad Awad and Muhammad Osama and Brandon Potter},
  title         = {Iris: First-Class Multi-{GPU} Programming Experience in {Triton}},
  year          = {2025},
  archivePrefix = {arXiv},
  eprint        = {2511.12500},
  primaryClass  = {cs.DC},
  doi           = {10.48550/arXiv.2511.12500}
}

@misc{Trifan:2025:EMT,
  author        = {Octavian Alexandru Trifan and Karthik Sangaiah and Muhammad Awad and Muhammad Osama and Sumanth Gudaparthi and Alexandru Nicolau and Alexander Veidenbaum and Ganesh Dasika},
  title         = {Eliminating Multi-{GPU} Performance Taxes: A Systems Approach to Efficient Distributed {LLMs}},
  year          = {2025},
  archivePrefix = {arXiv},
  eprint        = {2511.02168},
  primaryClass  = {cs.DC},
  doi           = {10.48550/arXiv.2511.02168}
}

@software{Awad:2025:IFM:Software,
  author        = {Muhammad Awad and Muhammad Osama and Brandon Potter},
  title         = {Iris: First-Class Multi-{GPU} Programming Experience in {Triton}},
  year          = 2025,
  month         = oct,
  doi           = {10.5281/zenodo.17382307},
  url           = {https://github.com/ROCm/iris}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
