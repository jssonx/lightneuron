# Auto-detect platform
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    # macOS
    CC = clang
    CFLAGS = -I. -lm
    HDF5_PREFIX = $(shell brew --prefix hdf5 2>/dev/null || echo "/usr/local/opt/hdf5")
    HDF5_FLAGS = -I$(HDF5_PREFIX)/include -L$(HDF5_PREFIX)/lib -lhdf5
    LIBOMP_PREFIX = $(shell brew --prefix libomp 2>/dev/null || echo "/usr/local/opt/libomp")
    OMP_FLAGS = -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include -L$(LIBOMP_PREFIX)/lib -lomp
    BLAS_FLAGS = -framework Accelerate
else
    # Linux
    CC = gcc
    CFLAGS = -I. -lm -lblas
    HDF5_FLAGS = -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5
    OMP_FLAGS = -fopenmp
    BLAS_FLAGS = -lblas
endif
COMMON_HDRS = ./utils/data_utils.h ./kernel/conv.h ./kernel/matrix_ops.h ./kernel/linear.h ./kernel/functional.h ./kernel/nn.h
COMMON_SRC = ./utils/data_utils.c ./kernel/conv.c ./kernel/functional.c ./kernel/matrix_ops.c ./kernel/linear.c ./kernel/nn.c

# Unity test framework
UNITY_FILES = ./tests/unity/unity.c
TEST_FILES = $(wildcard ./tests/*.c)
TEST_EXECUTABLES = $(patsubst %.c,%,$(TEST_FILES))

# Performance tests
# Core implementations (parameterized)
PERF_CORE = matmul_naive matmul_blocking matmul_blas matmul_sparse matmul_thread linear_naive linear_blocking gemm_perf

# Parameterized size targets (for backward compatibility)
SIZES = 64 128 256 512 1024
MATMUL_SIZE_TARGETS = $(foreach size,$(SIZES),matmul_naive_$(size) matmul_blocking_$(size) matmul_sparse_$(size) matmul_thread_$(size))

LEVEL ?= l1

BINS = lab

.PHONY: all
all: $(BINS)

.PHONY: lab
lab: lab.c $(COMMON_HDRS)
	$(CC) -o $@ lab.c $(COMMON_SRC) $(CFLAGS) $(HDF5_FLAGS) $(OMP_FLAGS) $(BLAS_FLAGS)

.PHONY: test
test: all_tests

.PHONY: all_tests
all_tests:
	$(CC) -o tests/$@ $(TEST_FILES) $(UNITY_FILES) $(COMMON_SRC) $(CFLAGS) $(HDF5_FLAGS) $(OMP_FLAGS) $(BLAS_FLAGS)
	./tests/$@

# Core implementations (direct build)
.PHONY: $(PERF_CORE)
$(PERF_CORE):
	@if [ "$@" = "gemm_perf" ]; then \
		GEMM_SRC="./kernel/gemm/gemm_naive.c ./kernel/gemm/gemm_interchange_loops.c ./kernel/gemm/gemm_parallel_loops.c ./kernel/gemm/gemm_tiling.c ./kernel/gemm/gemm_recursive_tiling.c ./kernel/gemm/gemm_simd.c ./kernel/gemm/gemm_cuda_style.c ./kernel/gemm/gemm_dispatch.c"; \
		$(CC) -o $@ ./perf/$@.c $(COMMON_SRC) $$GEMM_SRC $(CFLAGS) $(HDF5_FLAGS) $(OMP_FLAGS) $(BLAS_FLAGS); \
	else \
		$(CC) -o $@ ./perf/$@.c $(COMMON_SRC) $(CFLAGS) $(HDF5_FLAGS) $(OMP_FLAGS) $(BLAS_FLAGS); \
	fi
	@if [ "$(USE_PMU)" = "1" ] && [ "$(UNAME_S)" != "Darwin" ]; then \
		/usr/local/pmu-tools/pmu-tools/toplev.py --core S0-C0 -$(LEVEL) -v --no-desc taskset -c 0 ./$@; \
	else \
		./$@; \
	fi

# Parameterized size targets (build base and run with size argument)
.PHONY: $(MATMUL_SIZE_TARGETS) matmul_sparse_2048 matmul_sparse_4096
$(MATMUL_SIZE_TARGETS) matmul_sparse_2048 matmul_sparse_4096:
	@SIZE=$$(echo $@ | grep -oE "[0-9]+$$"); \
	BASE=$$(echo $@ | sed -E 's/_[0-9]+$$//'); \
	echo "Building $$BASE and running with size $$SIZE..."; \
	$(CC) -o $$BASE ./perf/$$BASE.c $(COMMON_SRC) $(CFLAGS) $(HDF5_FLAGS) $(OMP_FLAGS) $(BLAS_FLAGS); \
	if [ "$(USE_PMU)" = "1" ] && [ "$(UNAME_S)" != "Darwin" ]; then \
		/usr/local/pmu-tools/pmu-tools/toplev.py --core S0-C0 -$(LEVEL) -v --no-desc taskset -c 0 ./$$BASE $$SIZE; \
	else \
		./$$BASE $$SIZE; \
	fi

.PHONY: clean
clean:
	rm -f a.out $(BINS) $(TEST_EXECUTABLES) $(PERF_CORE)

.PHONY: help
help:
	@echo "LightNeuron Makefile - Platform: $(UNAME_S)"
	@echo ""
	@echo "Main targets:"
	@echo "  make lab                  - Build and run CNN inference (MNIST)"
	@echo "  make test                 - Run unit tests"
	@echo "  make clean                - Clean build artifacts"
	@echo ""
	@echo "Performance benchmarks:"
	@echo "  make matmul_naive         - Naive implementation (default: 1300x1300)"
	@echo "  make matmul_blocking      - Blocking/tiling optimization"
	@echo "  make matmul_thread        - Multi-threaded (pthread)"
	@echo "  make matmul_sparse        - Sparse matrix (CSR format)"
	@echo "  make matmul_blas          - BLAS library (Accelerate/OpenBLAS)"
	@echo ""
	@echo "Parameterized tests (specify size):"
	@echo "  make matmul_naive_64      - Run naive with 64x64 matrix"
	@echo "  make matmul_blocking_512  - Run blocking with 512x512 matrix"
	@echo "  Supported sizes: 64, 128, 256, 512, 1024, 2048, 4096"
	@echo ""
	@echo "Advanced (Linux only):"
	@echo "  make matmul_naive_64 USE_PMU=1 LEVEL=l1  - Profile with PMU tools"