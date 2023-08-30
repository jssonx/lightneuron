CC = gcc
CFLAGS = -I. -lm -lblas

HDF5_FLAGS = -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5
COMMON_HDRS = ./utils/data_utils.h ./kernel/conv.h ./kernel/matrix_ops.h ./kernel/linear.h ./kernel/functional.h ./kernel/nn.h
COMMON_SRC = ./utils/data_utils.c ./kernel/conv.c ./kernel/functional.c ./kernel/matrix_ops.c ./kernel/linear.c ./kernel/nn.c \
			 ./kernel/gemm/gemm_v3.c

# Unity test framework
UNITY_FILES = ./tests/unity/unity.c
TEST_FILES = $(wildcard ./tests/*.c)
TEST_EXECUTABLES = $(patsubst %.c,%,$(TEST_FILES))

# Performance
MATMUL_TARGETS = matmul_naive matmul_blocking matmul_blas matmul_sparse matmul_thread linear_naive linear_blocking \
                 matmul_naive_64 matmul_naive_128 matmul_naive_256 matmul_naive_512 matmul_naive_1024 \
				 matmul_blocking_64 matmul_blocking_128 matmul_blocking_256 matmul_blocking_512 matmul_blocking_1024 \
				 matmul_sparse_64 matmul_sparse_128 matmul_sparse_256 matmul_sparse_512 matmul_sparse_1024 matmul_sparse_2048 \
				 matmul_thread_64 matmul_thread_128 matmul_thread_256 matmul_thread_512 matmul_thread_1024 \
				 gemm_perf

LEVEL ?= l1

BINS = lab

.PHONY: all
all: $(BINS)

.PHONY: lab
lab: lab.c $(COMMON_HDRS)
	$(CC) -o $@ lab.c $(COMMON_SRC) $(CFLAGS) $(HDF5_FLAGS)

.PHONY: test
test: all_tests

.PHONY: all_tests
all_tests: 
	$(CC) -o tests/$@ $(TEST_FILES) $(UNITY_FILES) $(COMMON_SRC) $(CFLAGS) $(HDF5_FLAGS) 
	./tests/$@

# make matmul_naive_64 LEVEL=l1 USE_PMU=1
.PHONY: $(MATMUL_TARGETS)
$(MATMUL_TARGETS):
ifeq ($(USE_PMU),1)
	$(CC) -o $@ ./perf/$@.c $(COMMON_SRC) $(CFLAGS) $(HDF5_FLAGS)
	/usr/local/pmu-tools/pmu-tools/toplev.py --core S0-C0 -$(LEVEL) -v --no-desc taskset -c 0 ./$@
else
	$(CC) -o $@ ./perf/$@.c $(COMMON_SRC) $(CFLAGS) $(HDF5_FLAGS)
	./$@
endif

.PHONY: clean
clean:
	rm -f $(BINS) $(TEST_EXECUTABLES) $(MATMUL_TARGETS)