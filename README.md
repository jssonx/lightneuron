# LightNeuron 

![GitHub Actions status](https://github.com/jssonx/lightneuron/workflows/test/badge.svg) ![License](https://img.shields.io/badge/license-MIT-green)

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

- **CPU Architecture**: x86-64
- **Processor**: Intel Core i5-10210U
- **Number of Cores / Threads**: 4 / 8
- **Cache Configuration**:
  - L1 Cache: 256 KiB
  - L2 Cache: 1 MiB
  - L3 Cache: 6 MiB

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

The result of these enhancements is a notable increase in CPU computational efficiency, boosting the performance of matrix multiplication operations considerably.

![gflops_performance](./img/gflops_performance.png)

### Performance Benchmark at Matrix Size 1200x1200

The following table showcases the GFLOPs performance of various kernels compared to OpenBLAS, a widely used optimized BLAS library, at a matrix size of 1200x1200.

GFLOPs at matrix size 1200x1200:
<!-- benchmark_results -->
| Kernel | GFLOPs/s | Performance relative to OpenBLAS |
|:-------|---------:|:-------------------------------|
| OpenBLAS | `13.4` | 100.0% |
| gemm_4x4_v16 | `12.2` | 91.0% |
| gemm_4x4_v14 | `12.0` | 89.2% |
| gemm_4x4_v15 | `11.9` | 88.6% |
| gemm_4x4_v12 | `10.2` | 75.6% |
| gemm_4x4_v13 | `6.9` | 51.1% |
| gemm_4x4_v11 | `3.8` | 28.0% |
| gemm_v4 | `3.5` | 25.9% |
| gemm_4x4_v7 | `2.4` | 18.2% |
| gemm_4x4_v8 | `2.4` | 17.9% |
| gemm_4x4_v10 | `2.4` | 17.9% |
| gemm_4x4_v9 | `2.4` | 17.6% |
| gemm_1x4_v9 | `2.2` | 16.1% |
| gemm_1x4_v7 | `2.2` | 16.0% |
| gemm_1x4_v8 | `2.1` | 15.9% |
| gemm_1x4_v10 | `2.1` | 15.7% |
| gemm_4x4_v6 | `2.0` | 14.5% |
| gemm_1x4_v6 | `1.6` | 11.9% |
| gemm_4x4_v5 | `1.3` | 9.8% |
| gemm_4x4_v4 | `1.3` | 9.6% |
| gemm_1x4_v5 | `1.3` | 9.6% |
| gemm_v3 | `1.3` | 9.5% |
| gemm_1x4_v4 | `1.2` | 9.3% |
| gemm_v1 | `1.2` | 8.8% |
| gemm_v2 | `0.9` | 6.7% |
<!-- benchmark_results -->

This benchmark demonstrates the effective performance gains achieved through our optimization techniques, with several of the GEMM implementations nearing industry-standard performance levels.
