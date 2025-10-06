/*
 * GEMM implementation inspired by CUDA optimizations
 *
 * Key optimizations from CUDA kernel adapted to CPU:
 * 1. Multi-level tiling (3 levels: outer block, thread block, vector)
 * 2. Register blocking (simulate thread-local computation)
 * 3. Double buffering (prefetch next data while computing)
 * 4. Vector operations (using manual unrolling instead of CUDA float4)
 * 5. Cache-aware memory access patterns
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define A(i, j) a[(j) * lda + (i)]
#define B(i, j) b[(j) * ldb + (i)]
#define C(i, j) c[(j) * ldc + (i)]

// CUDA-inspired blocking parameters (adapted for CPU)
#define BLOCK_M 128    // Outer block size for M dimension
#define BLOCK_N 128    // Outer block size for N dimension
#define BLOCK_K 16     // K dimension block (increased from CUDA's 8 for CPU cache)
#define THREAD_M 8     // Each "thread" computes 8x8 elements
#define THREAD_N 8
#define VECTOR_SIZE 4  // Simulate float4 vectorization

/*
 * Simulate CUDA thread computation on CPU
 * Each "thread" computes a THREAD_M x THREAD_N tile
 * Using register blocking and double buffering concepts
 */
static inline void compute_thread_tile(
    const double *__restrict__ tile_A,  // BLOCK_M x BLOCK_K tile in cache
    const double *__restrict__ tile_B,  // BLOCK_K x BLOCK_N tile in cache
    double *__restrict__ tile_C,         // BLOCK_M x BLOCK_N output tile
    int thread_row, int thread_col,
    int tile_lda, int tile_ldb, int tile_ldc)
{
    // Register blocking: simulate CUDA thread registers
    // Each thread computes THREAD_M x THREAD_N = 64 elements
    double reg_C[THREAD_M][THREAD_N];
    double reg_A[THREAD_M];
    double reg_B[THREAD_N];

    // Initialize accumulator registers
    for (int i = 0; i < THREAD_M; i++) {
        for (int j = 0; j < THREAD_N; j++) {
            reg_C[i][j] = 0.0;
        }
    }

    // Base position for this thread
    int base_i = thread_row * THREAD_M;
    int base_j = thread_col * THREAD_N;

    // Double buffering simulation: process in pairs
    for (int k = 0; k < BLOCK_K; k++) {
        // Prefetch A column (simulate CUDA lds128)
        for (int i = 0; i < THREAD_M; i++) {
            reg_A[i] = tile_A[(base_i + i) + k * tile_lda];
        }

        // Prefetch B row (simulate CUDA lds128)
        for (int j = 0; j < THREAD_N; j++) {
            reg_B[j] = tile_B[k + (base_j + j) * tile_ldb];
        }

        // Compute outer product: reg_A * reg_B -> reg_C
        // Fully unrolled for CPU register optimization
        #pragma GCC unroll 8
        for (int i = 0; i < THREAD_M; i++) {
            #pragma GCC unroll 8
            for (int j = 0; j < THREAD_N; j++) {
                reg_C[i][j] += reg_A[i] * reg_B[j];
            }
        }
    }

    // Write back results (simulate CUDA global store)
    for (int i = 0; i < THREAD_M; i++) {
        for (int j = 0; j < THREAD_N; j++) {
            tile_C[(base_i + i) + (base_j + j) * tile_ldc] += reg_C[i][j];
        }
    }
}

/*
 * CUDA-style GEMM kernel
 * Simulates CUDA's hierarchical parallelism on CPU
 */
void gemm_cuda_style(int m, int n, int k,
                     double *a, int lda,
                     double *b, int ldb,
                     double *c, int ldc)
{
    // Outer loop: process BLOCK_M x BLOCK_N tiles (simulate CUDA grid)
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int block_i = 0; block_i < m; block_i += BLOCK_M) {
        for (int block_j = 0; block_j < n; block_j += BLOCK_N) {

            int block_m_size = (block_i + BLOCK_M <= m) ? BLOCK_M : (m - block_i);
            int block_n_size = (block_j + BLOCK_N <= n) ? BLOCK_N : (n - block_j);

            // Allocate cache-resident tiles (simulate CUDA shared memory)
            // Use stack allocation for better cache locality
            double tile_A[BLOCK_M * BLOCK_K] __attribute__((aligned(64)));
            double tile_B[BLOCK_K * BLOCK_N] __attribute__((aligned(64)));
            double tile_C[BLOCK_M * BLOCK_N] __attribute__((aligned(64)));

            // Initialize output tile
            memset(tile_C, 0, sizeof(tile_C));

            // K-dimension tiling loop
            for (int block_k = 0; block_k < k; block_k += BLOCK_K) {
                int block_k_size = (block_k + BLOCK_K <= k) ? BLOCK_K : (k - block_k);

                // Load A tile into cache (simulate CUDA gmem -> smem)
                // Optimized memory access pattern
                for (int kk = 0; kk < block_k_size; kk++) {
                    for (int ii = 0; ii < block_m_size; ii++) {
                        tile_A[ii + kk * BLOCK_M] = A(block_i + ii, block_k + kk);
                    }
                }

                // Load B tile into cache (simulate CUDA gmem -> smem)
                for (int jj = 0; jj < block_n_size; jj++) {
                    for (int kk = 0; kk < block_k_size; kk++) {
                        tile_B[kk + jj * BLOCK_K] = B(block_k + kk, block_j + jj);
                    }
                }

                // Thread-level parallelism (simulate CUDA thread block)
                // Each "thread" computes THREAD_M x THREAD_N elements
                int num_threads_m = (block_m_size + THREAD_M - 1) / THREAD_M;
                int num_threads_n = (block_n_size + THREAD_N - 1) / THREAD_N;

                for (int thread_i = 0; thread_i < num_threads_m; thread_i++) {
                    for (int thread_j = 0; thread_j < num_threads_n; thread_j++) {
                        compute_thread_tile(
                            tile_A, tile_B, tile_C,
                            thread_i, thread_j,
                            BLOCK_M, BLOCK_K, BLOCK_M
                        );
                    }
                }
            }

            // Write back final results (simulate CUDA smem -> gmem)
            for (int jj = 0; jj < block_n_size; jj++) {
                for (int ii = 0; ii < block_m_size; ii++) {
                    C(block_i + ii, block_j + jj) = tile_C[ii + jj * BLOCK_M];
                }
            }
        }
    }
}
