#ifndef GEMM_H
#define GEMM_H

#include <stdint.h>

// GEMM implementation types
typedef enum {
    GEMM_NAIVE,              // v1: Basic ijk loop
    GEMM_INTERCHANGE,        // v2: Loop interchange (jpi)
    GEMM_PARALLEL,           // v4: OpenMP parallel loops
    GEMM_TILING,             // v5: Loop tiling/blocking
    GEMM_REC_TILING,         // v6: Recursive divide-and-conquer
    GEMM_SIMD,               // v7: AVX2 SIMD + recursive
    GEMM_BLAS,               // v10: BLAS library (cblas_dgemm)
    GEMM_CUDA_STYLE          // v11: CUDA-inspired multi-level tiling
} GemmType;

// Individual implementations
void gemm_naive(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_interchange_loops(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_parallel_loops(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_tiling(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_rec_tiling(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_simd(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_cuda_style(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);

// Unified dispatcher function
void gemm_dispatch(GemmType type, int m, int n, int k,
                   double *a, int lda, double *b, int ldb,
                   double *c, int ldc);

// Get name of GEMM implementation
const char* gemm_get_name(GemmType type);

#endif // GEMM_H