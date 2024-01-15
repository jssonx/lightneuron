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

- **Loop Interchange**: Reorders nested loops to enhance memory access patterns and improve cache performance, eg. ijk -> kji.
- **Compiler Optimization Flags**: Employs -O2/-O3 levels for code efficiency.
- **Parallel Loops**: Uses OpenMP directives to distribute loop execution across multiple CPU threads.
- **Loop Tiling (Blocking)**: Optimizes spatial and temporal locality for caches.
- **Divide-and-Conquer**: Splits large matrices into smaller sub-matrices for better cache performance.
- **SIMD Intrinsics with Data Alignment**: Uses AVX2 instructions and aligns data to boost vectorized operations and memory throughput.

The result of these enhancements is a notable increase in CPU computational efficiency, boosting the performance of matrix multiplication operations considerably.

| Implementation               | Cache References (millions) | L1-d Cache Misses (millions) | LL Cache Misses (millions) |
| ---------------------------- | --------------------------: | ---------------------------: | -------------------------: |
| +parallel loops              |                     4934.44 |                       406.47 |                      404.9 |
| +tiling                      |                     5010.46 |                       620.66 |                      13.29 |
| +parallel divide-and-conquer |                     1881.06 |                       152.97 |                       5.21 |

Tiling achieves a 96% reduction in last-level cache misses, and parallel divide-and-conquer further lowers overall cache references and minimizes cache misses.

### Performance Benchmark at Matrix Size 1200x1200

The following table showcases the GFLOPs performance of various kernels compared to Intel MKL, at a matrix size of 1200x1200.

GFLOPs at matrix size 1200x1200:

<!-- benchmark_results -->

| Version Implementation                                                                                                       | Running Times (ms) | Relative Speedup | Absolute Speedup | GFLOPS | Percent of Peak | Percent of Intel MKL |
| ---------------------------------------------------------------------------------------------------------------------------- | -----------------: | ---------------: | ---------------: | -----: | --------------: | -------------------: |
| naive                                                                                                                        |           11190.93 |             1.00 |             1.00 |   0.19 |           0.19% |                0.25% |
| naive + interchange loops                                                                                                    |            4267.47 |             2.62 |             2.62 |   0.50 |           0.49% |                0.65% |
| naive + interchange loops + optimization flags                                                                               |             675.76 |             6.32 |            16.56 |   3.18 |           3.10% |                4.08% |
| naive + interchange loops + optimization flags + parallel loops                                                              |             147.87 |             4.57 |            75.68 |  14.52 |          14.18% |               18.62% |
| naive + interchange loops + optimization flags + parallel tiling                                                             |              101.3 |             1.46 |           110.47 |  21.20 |          20.70% |               27.19% |
| naive + interchange loops + optimization flags + parallel divide-and-conquer                                                 |              89.52 |             1.13 |           125.01 |  23.99 |          23.43% |               30.76% |
| naive + interchange loops + optimization flags + parallel divide-and-conquer + avx2 intrinsics + data alignment              |              71.11 |             1.26 |           157.37 |  30.20 |          29.49% |               38.73% |
| naive + interchange loops + optimization flags + parallel tiling + avx2 intrinsics + data alignment                          |              62.41 |             1.14 |           179.31 |  34.41 |          33.60% |               44.13% |
| naive + interchange loops + optimization flags + parallel divide-and-conquer + avx2 intrinsics + data alignment + coarsening |              43.62 |             1.43 |           256.56 |  49.23 |          48.08% |               63.14% |
| Intel MKL                                                                                                                    |              27.54 |             1.58 |           406.35 |  77.98 |          76.15% |              100.00% |


<!-- benchmark_results -->
