# LightNeuron

![GitHub Actions status](https://github.com/jssonx/lightneuron/workflows/test/badge.svg)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/2938278352a24dba9edb07e2e1d0738a)](https://app.codacy.com/gh/jssonx/lightneuron/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
![License](https://img.shields.io/badge/license-MIT-green)

![lightneuron](./img/lightneuron.png)

LightNeuron is a highly efficient, educational neural network library designed for x86-64 architectures in C. It aims to provide insights into neural network mechanics, profiling, and optimization, with a special focus on the efficiency of General Matrix Multiply (GEMM) operations.

## Overview

Targeted primarily at students, researchers, and developers, LightNeuron offers a CNN inference framework capable of processing HDF5 model files. This facilitates the integration with models trained on frameworks like PyTorch and TensorFlow. Key features include:

- Convolutional Layer Computation (conv())
- Matrix Multiplication (matmul())
- Activation Functions (relu())
- Pooling (pooling())
- Forward Pass Operations (forwardPass())
- Feature Extraction and Interpretation
- Prediction (softmax(), predict())

![framework](./img/framework.png)

## Development Environment Specifications

LightNeuron is optimized for x86-64 architectures, ensuring compatibility and efficiency on a wide range of systems. Below are the specifications of the primary development environment, which can serve as a benchmark for expected performance:

- Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz
- Microarchitecture: Comet Lake
- 1.6 GHz is the base frequency of the CPU
- 4 cores, 2 threads per core
- 16 DP FLOPS/cycle (AVX2, FP64)
- Single core theoretical peak performance = 1.6 GHz \* 16 FLOPS/cycle = 25.6 GFLOPS
- Multi-core theoretical peak performance = 25.6 GFLOPS \* 4 cores = 102.4 GFLOPS
- References:
  - [What Every Computational Physicist Should Know AboutComputer Architecture](https://indico.cern.ch/event/814979/contributions/3401193/attachments/1831477/3105158/comp_arch_codas_2019.pdf)
  - [FLOPS, wikipedia](https://en.wikipedia.org/wiki/FLOPS)

## Prerequisites

Ensure your system is ready for LightNeuron by installing the `perf` tool:

```bash
sudo apt-get install linux-tools-$(uname -r) linux-cloud-tools-$(uname -r)
```

Configure your system by editing `/etc/sysctl.conf`:

```bash
kernel.perf_event_paranoid = -1
kernel.nmi_watchdog = 0
```

Activate the changes:

```bash
sudo sysctl -p
```

## Getting Started

1. **Clone the Repository**:

   ```bash
   git clone [repository-url]
   ```

2. **Download MNIST Dataset**:

   ```bash
   python get_data.py
   ```

3. **Compile and Run Labs**:
   ```bash
   make lab && ./lab
   ```

## Performance Profiling

Profile GEMM operations with specific targets and cache levels:

```bash
make perf TARGET=[your-target] CACHE_LEVEL=[your-cache-level] USE_PMU=1
```

- Replace `TARGET` with the GEMM implementation (e.g., `matmul_naive`).
- Set `CACHE_LEVEL` to desired cache level (e.g., `L1`, `L2`, `L3`).

Example:

```bash
make perf TARGET=matmul_naive CACHE_LEVEL=L1 USE_PMU=1
```

`USE_PMU=1` activates the Performance Monitoring Unit for detailed hardware-level performance insights.

## GEMM Optimization

LightNeuron places a strong emphasis on optimizing General Matrix Multiply (GEMM) operations. This optimization leads to significant performance improvements, as measured in GFLOPS (Giga Floating Point Operations Per Second), particularly noticeable across a range of matrix dimensions. Key strategies employed in this optimization include:



- **Intricate Loop Unrolling**: Enhances computational efficiency by reducing loop overhead.
- **Proactive Data Prefetching**: Improves data access speeds from memory.
- **Strategic Cache Management**: Optimizes the use of CPU cache to minimize data retrieval delays.
- **Precise Data Alignment**: Ensures data is appropriately aligned in memory, reducing access times.
- **Concurrent Multithreading**: Leverages parallel processing capabilities of modern CPUs.
- **Specialized Instruction Sets Utilization**: Takes advantage of x86 architecture-specific features like AVX2 and SSE to accelerate computations.

- **Loop Interchange**: Reorders nested loops to enhance memory access patterns and improve cache performance, eg. ijk -> kji.
- **Compiler Optimization Flags**: Employs -O2/-O3 levels for code efficiency.
- **Parallel Loops**: Uses OpenMP directives to distribute loop execution across multiple CPU threads.
- **Loop Tiling (Blocking)**: Optimizes spatial and temporal locality for caches.
- **Divide-and-Conqure**: Splits large matrices into smaller sub-matrices for better cache performance.
- **SIMD Intrinsics with Data Alignment**: Uses AVX2 instructions and aligns data to boost vectorized operations and memory throughput.

The result of these enhancements is a notable increase in CPU computational efficiency, boosting the performance of matrix multiplication operations considerably.

| Implementation                  | Cache references (millions) | L1-d cache misses (millions) | LL cache misses (millions) |
|---------------------------------|----------------------------:|----------------------------:|--------------------------:|
| +parallel loops                 | 4934.44                     | 406.47                      | 404.9                     |
| +tiling                         | 5010.46                     | 620.66                      | 13.29                     |
| +parallel divide-and-conquer    | 1881.06                     | 152.97                      | 5.21                      |

Tiling achieves a 96% reduction in last-level cache misses, and parallel divide-and-conquer further lowers overall cache references and minimizes cache misses.

### Performance Benchmark at Matrix Size 1200x1200

The following table showcases the GFLOPs performance of various kernels compared to Intel MKL, at a matrix size of 1200x1200.

GFLOPs at matrix size 1200x1200:

<!-- benchmark_results -->

| Version Implementation          | Running Times (ms) | Relative Speedup | Absolute Speedup | GFLOPS | Percent of Peak |
|---------------------------------|-------------------:|-----------------:|-----------------:|-------:|----------------:|
| naive                           | 11190.93           | 1.00             | 1.00             | 0.19   | 0.19%           |
| +interchange loops              | 4267.47            | 2.62             | 2.62             | 0.50   | 0.49%           |
| +optimization flags             | 675.76             | 6.32             | 16.56            | 3.18   | 3.10%           |
| +parallel loops                 | 147.87             | 4.57             | 75.68            | 14.52  | 14.18%          |
| +tiling                         | 101.3              | 1.46             | 110.47           | 21.20  | 20.70%          |
| +parallel devide-and-conquer    | 89.4               | 1.13             | 125.18           | 24.02  | 23.46%          |
| +avx2 intrinsics+data alignment | 81.02              | 1.10             | 138.13           | 26.51  | 25.88%          |
| +compiler vectorization         | 74.31              | 1.09             | 150.60           | 28.90  | 28.22%          |
| Intel MKL                       | 27.54              | 2.70             | 406.35           | 77.98  | 76.15%          |

<!-- benchmark_results -->
