#include "gemm.h"
#include <stdio.h>
#include <string.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

// Dispatcher function that calls the appropriate GEMM implementation
void gemm_dispatch(GemmType type, int m, int n, int k,
                   double *a, int lda, double *b, int ldb,
                   double *c, int ldc)
{
    switch (type) {
        case GEMM_NAIVE:
            gemm_naive(m, n, k, a, lda, b, ldb, c, ldc);
            break;

        case GEMM_INTERCHANGE:
            gemm_interchange_loops(m, n, k, a, lda, b, ldb, c, ldc);
            break;

        case GEMM_PARALLEL:
            gemm_parallel_loops(m, n, k, a, lda, b, ldb, c, ldc);
            break;

        case GEMM_TILING:
            gemm_tiling(m, n, k, a, lda, b, ldb, c, ldc);
            break;

        case GEMM_REC_TILING:
            gemm_rec_tiling(m, n, k, a, lda, b, ldb, c, ldc);
            break;

        case GEMM_SIMD:
            gemm_simd(m, n, k, a, lda, b, ldb, c, ldc);
            break;

        case GEMM_BLAS:
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                       m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
            break;

        case GEMM_CUDA_STYLE:
            gemm_cuda_style(m, n, k, a, lda, b, ldb, c, ldc);
            break;

        default:
            fprintf(stderr, "Unknown GEMM type: %d\n", type);
            break;
    }
}

// Get human-readable name of GEMM implementation
const char* gemm_get_name(GemmType type)
{
    switch (type) {
        case GEMM_NAIVE:       return "naive";
        case GEMM_INTERCHANGE: return "interchange_loops";
        case GEMM_PARALLEL:    return "parallel_loops";
        case GEMM_TILING:      return "tiling";
        case GEMM_REC_TILING:  return "rec_tiling";
        case GEMM_SIMD:        return "simd";
        case GEMM_BLAS:        return "blas";
        case GEMM_CUDA_STYLE:  return "cuda_style";
        default:               return "unknown";
    }
}
