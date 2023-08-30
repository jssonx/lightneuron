# LightNeuron

![lightneuron](./img/lightneuron.png)

LightNeuron is a lightweight neural network library, written in C and optimized for x86-64 systems. It's designed to be a learning tool for understanding the inner workings of neural networks, as well as a practical guide for learning profiling and improving key computations such as General Matrix Multiply (GEMM) and other operations.

![framework](./img/framework.png)

## Configuration for Optimal Environment

### Verification and Installation of `perf`

To ensure your system's compatibility and performance optimization, it's crucial to verify and install the `perf` tool, aligning with your kernel version. Use the following commands:

```bash
sudo apt-get install linux-tools-$(uname -r) linux-cloud-tools-$(uname -r)
```

### Adjusting `perf_event_paranoid`

This setting manages the access of non-privileged users to CPU performance events. Modify it by adding or altering this line in `/etc/sysctl.conf`:

```bash
kernel.perf_event_paranoid = -1
```

To activate these changes, execute:

```bash
sudo sysctl -p
```

### Disabling NMI Watchdog

To minimize interference in performance monitoring, add this to `/etc/sysctl.conf`:

```bash
kernel.nmi_watchdog = 0
```

Implement the change with:

```bash
sudo sysctl -p
```

## Usage Guide for CNN Model

1. **Cloning the Repository**: Initiate by cloning this repository.
2. **Acquiring MNIST Dataset**: Use the following command:
   ```
   python get_data.py
   ```
3. **Compiling and Running Labs**: Execute the lab files for each session:
   ```
   make lab && ./lab
   ```
4. **Repeating Processes**: Repeat the above step for every lab.
5. **Cleanup**: Post-completion, clear the build files:
   ```
   make clean
   ```

## Testing Your Code

Run the following for initiating code tests:

```
make test
```

## GEMM Performance Profiling with `perf`

Profile the performance of General Matrix Multiply (GEMM) operations, especially focusing on specific targets and cache levels. Use the command format `make perf [target] [cache level]`. For example:

```
make perf matmul_naive l2
```


gcc ./perf/gemm_perf.c ./kernel/gemm/gemm_v1.c && ./a.out