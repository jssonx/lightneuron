#include <immintrin.h>  // For AVX2
#include "gemm.h"

// gcc ./perf/gemm_perf.c ./kernel/gemm/gemm_v4.c -mavx2 -mfma && ./a.out

/* Create macros so that the matrices are stored in column-major order */

#define A(i, j) a[(j) * lda + (i)]
#define B(i, j) b[(j) * ldb + (i)]
#define C(i, j) c[(j) * ldc + (i)]

void gemm_v4(int m, int n, int k,
             double *a, int lda,
             double *b, int ldb,
             double *c, int ldc) {
    int i, j, p;

    for (i = 0; i < m; i++) { /* Loop over the rows of C */
        for (j = 0; j < n; j++) { /* Loop over the columns of C */
            __m256d cij = _mm256_setzero_pd(); // Initialize the vector register

            for (p = 0; p <= k - 4; p += 4) { /* Update C(i,j) with the inner product */
                __m256d ai = _mm256_loadu_pd(&A(i, p)); // Load 4 elements from A
                __m256d bj = _mm256_loadu_pd(&B(p, j)); // Load 4 elements from B
                cij = _mm256_fmadd_pd(ai, bj, cij);     // Fused multiply-add operation
            }

            // Reduce the vector sum to a single sum
            double temp[4];
            _mm256_storeu_pd(temp, cij);
            for (int x = 0; x < 4; x++) {
                C(i, j) += temp[x];
            }

            // Handle the remaining elements if k is not a multiple of 4
            for (; p < k; p++) {
                C(i, j) += A(i, p) * B(p, j);
            }
        }
    }
}
