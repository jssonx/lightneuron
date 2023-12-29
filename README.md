# LightNeuron 

![GitHub Actions status](https://github.com/jssonx/lightneuron/workflows/test/badge.svg) ![License](https://img.shields.io/badge/license-MIT-green)

![lightneuron](./img/lightneuron.png)

LightNeuron is an educational neural network library, meticulously crafted in C for x86-64 architectures. Its primary focus is to serve as both an educational and practical resource in the realms of neural network mechanics, profiling, and optimization, particularly emphasizing the efficiency of General Matrix Multiply (GEMM) operations.

## Inference Framework

![framework](./img/framework.png)

At its core, LightNeuron functions as a CNN inference framework, adeptly handling HDF5 model files for inference. This feature ensures seamless integration with models initially trained using prominent frameworks like PyTorch and TensorFlow. LightNeuron encompasses a comprehensive suite of operations fundamental to CNN inference. These include convolutional layer computation (conv()), matrix multiplication (matmul()), activation functions (relu()), and pooling (pooling()). The framework efficiently orchestrates the forward pass (forwardPass()) through the neural network, concluding with a combination of flattening (flatten()) and a fully connected layer (linear()). This process culminates in the extraction and interpretation of features from convolutional layers. Subsequently, a softmax function (softmax()) is applied to transform the outputs into a normalized probability distribution, facilitating accurate predictions (predict()). LightNeuron exemplifies not only the practical deployment of deep learning models but also provides deep insights into neural network operations and structures.

### Prerequisites

To ensure your system's compatibility and performance optimization, it's crucial to verify and install the `perf` tool, aligning with your kernel version. Use the following commands:

```bash
sudo apt-get install linux-tools-$(uname -r) linux-cloud-tools-$(uname -r)
```
Adjusting `perf_event_paranoid` and disabling `nmi_watchdog`. This setting manages the access of non-privileged users to CPU performance events. Modify it by adding or altering this line in `/etc/sysctl.conf`:

```bash
kernel.perf_event_paranoid = -1
kernel.nmi_watchdog = 0
```

To activate these changes, execute:

```bash
sudo sysctl -p
```

### Usage Guide for Inference Framework

1. **Cloning the Repository**: Initiate by cloning this repository.
2. **Acquiring MNIST Dataset**: Use the following command:
   ```bash
   python get_data.py
   ```
3. **Compiling and Running Labs**: Execute the lab files for each session:
   ```bash
   make lab && ./lab
   ```

### Profiling Guide for GEMM in Inference Framework

To profile the performance of General Matrix Multiply (GEMM) operations with a focus on specific targets and cache levels, you can use a makefile command with a specified target and cache level argument. The command format is as follows:

```bash
make perf TARGET=[your-target] CACHE_LEVEL=[your-cache-level] USE_PMU=1
```

Here's how you might use the command:

- `TARGET` should be replaced with the specific GEMM implementation you wish to test (for example, `matmul_naive`).
- `CACHE_LEVEL` should be set to the cache level you're interested in analyzing (such as `L1`, `L2`, or `L3`).

For instance, if you want to profile a naive matrix multiplication implementation focusing on the L1 cache, the command would look like this:

```bash
make perf TARGET=matmul_naive CACHE_LEVEL=L1 USE_PMU=1
```

In this command, `USE_PMU=1` likely activates the Performance Monitoring Unit (PMU), which is a set of special-purpose registers for monitoring hardware-level events related to performance. 

## GEMM Optimization

![gflops_performance](./img/gflops_performance.png)

A significant aspect of the project is the meticulous optimization of the GEMM operation, exhibiting a diverse performance range in GFLOPS across various matrix dimensions. This is achieved through a combination of sophisticated techniques: intricate loop unrolling, proactive data prefetching, strategic cache management, precise data alignment, concurrent multithreading, and the utilization of specialized x86 architecture instruction sets, including Streaming SIMD Extensions (SSE). These enhancements are strategically designed to maximize CPU computational efficiency, substantially boosting the performance of matrix multiplication operations.