#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <assert.h>
#include <immintrin.h>

#define THRESHOLD 64
#define A(i, j) a[(j) * lda + (i)]
#define B(i, j) b[(j) * ldb + (i)]
#define C(i, j) c[(j) * ldc + (i)]

void gemm_helper(double *a, double *b, double *c, int rowA, int colA, int rowB, int colB, int rowC, int colC, int size, int lda, int ldb, int ldc)
{
        if (size <= THRESHOLD)
        {
                // Base case: perform matrix multiplication with AVX
                // Transpose B
                for (int k = 0; k < size; k += 4)
                {
                        for (int j = 0; j < size; j++)
                        {
                                for (int i = 0; i <= size - 4; i += 4)
                                {
                                        __m256d c_vec = _mm256_load_pd(&C(rowC + i, colC + j));

                                        for (int l = 0; l < 4; ++l)
                                        {
                                                // Since B is transposed, adjust indexing accordingly
                                                // __m256d b_val = _mm256_broadcast_sd(&B(colB + j, rowB + k + l));
                                                __m256d b_val = _mm256_broadcast_sd(&B(rowB + k + l, colB + j));
                                                __m256d a_vec = _mm256_load_pd(&A(rowA + i, colA + k + l));

                                                // Multiply a_vec with the broadcasted b_val and add to c_vec
                                                c_vec = _mm256_fmadd_pd(a_vec, b_val, c_vec);
                                        }

                                        _mm256_store_pd(&C(rowC + i, colC + j), c_vec);
                                }
                        }
                }
        }
        else
        {
                int newSize = size / 2;

/* Method 1 */
// Multiply four quadrants and get A00B00, A00B01, A10B10, A10B01
#pragma omp task
                gemm_helper(a, b, c, rowA, colA, rowB, colB, rowC, colC, newSize, lda, ldb, ldc);
#pragma omp task
                gemm_helper(a, b, c, rowA, colA, rowB, colB + newSize, rowC, colC + newSize, newSize, lda, ldb, ldc);
#pragma omp task
                gemm_helper(a, b, c, rowA + newSize, colA, rowB, colB, rowC + newSize, colC, newSize, lda, ldb, ldc);
#pragma omp task
                gemm_helper(a, b, c, rowA + newSize, colA, rowB, colB + newSize, rowC + newSize, colC + newSize, newSize, lda, ldb, ldc);
#pragma omp taskwait

                // Multiply four quadrants and get A01B10, A01B11, A11B10, A11B11
                // Add to the above result, get C00, C01, C10, C11 = A00B00 + A01B10, A00B01 + A01B11, A10B00 + A11B10, A10B01 + A11B11

#pragma omp task
                gemm_helper(a, b, c, rowA, colA + newSize, rowB + newSize, colB, rowC, colC, newSize, lda, ldb, ldc);
#pragma omp task
                gemm_helper(a, b, c, rowA, colA + newSize, rowB + newSize, colB + newSize, rowC, colC + newSize, newSize, lda, ldb, ldc);
#pragma omp task
                gemm_helper(a, b, c, rowA + newSize, colA + newSize, rowB + newSize, colB, rowC + newSize, colC, newSize, lda, ldb, ldc);
#pragma omp task
                gemm_helper(a, b, c, rowA + newSize, colA + newSize, rowB + newSize, colB + newSize, rowC + newSize, colC + newSize, newSize, lda, ldb, ldc);
#pragma omp taskwait
        }
}

void gemm_simd(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
        // Initialize OpenMP task group
        omp_set_num_threads(16);

#pragma omp parallel
        {
#pragma omp single
                {
                        gemm_helper(a, b, c, 0, 0, 0, 0, 0, 0, m, lda, ldb, ldc);
                }
        }
}
