#include <omp.h>
#include <assert.h>
#include <immintrin.h>

#define THRESHOLD 16
#define A(i, j) a[(j) * lda + (i)]
#define B(i, j) b[(j) * ldb + (i)]
#define C(i, j) c[(j) * ldc + (i)]

void gemm_omp(int m, int n, int k,
              double *a, int lda,
              double *b, int ldb,
              double *c, int ldc)
{
    // Check if m, n, k are powers of 2
    assert((m & (-m)) == m);
    assert((n & (-n)) == n);
    assert((k & (-k)) == k);

    if (m <= THRESHOLD || n <= THRESHOLD || k <= THRESHOLD) {
        // Base case: perform matrix multiplication with AVX
        for (int j = 0; j < n; ++j) {
            for (int p = 0; p < k; ++p) {
                __m256d b_vec = _mm256_set1_pd(B(p, j)); // 256 = 8 * 32 = 4 * 64 (4 doubles)
                for (int i = 0; i <= m - 4; i += 4) {
                    __m256d a_vec = _mm256_loadu_pd(&A(i, p));
                    __m256d c_vec = _mm256_loadu_pd(&C(i, j));

                    c_vec = _mm256_add_pd(c_vec, _mm256_mul_pd(a_vec, b_vec));

                    _mm256_storeu_pd(&C(i, j), c_vec);
                }

                // Handle remaining elements for cases where m is not a multiple of 4
                for (int i = m - m % 4; i < m; ++i) {
                    C(i, j) += A(i, p) * B(p, j);
                }
            }
        }
    } else {
        // Divide and conquer by dividing the matrices into four sub-matrices of m/2, n/2, k/2
        // Create OpenMP tasks to execute recursive calls in parallel
        
        // Top left quadrant
        #pragma omp task
        gemm_omp(m/2, n/2, k/2, a, lda, b, ldb, c, ldc);

        // Top right quadrant
        #pragma omp task
        gemm_omp(m/2, n/2, k/2, a, lda, b + ldb*k/2, ldb, c + ldc*n/2, ldc);

        // Bottom left quadrant
        #pragma omp task
        gemm_omp(m/2, n/2, k/2, a + lda*m/2, lda, b, ldb, c + m/2, ldc);

        // Bottom right quadrant
        #pragma omp task
        gemm_omp(m/2, n/2, k/2, a + lda*m/2, lda, b + ldb*k/2, ldb, c + ldc*n/2 + m/2, ldc);

        #pragma omp taskwait
    }
}

void gemm_simd(int m, int n, int k,
                   double *a, int lda,
                   double *b, int ldb,
                   double *c, int ldc)
{
    #pragma omp parallel
    {
        #pragma omp single
        {
            gemm_omp(m, n, k, a, lda, b, ldb, c, ldc);
        }
    }
}
