#include <omp.h>
#include <assert.h>

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
        // Base case: perform matrix multiplication
        for (int j = 0; j < n; ++j) {
            for (int p = 0; p < k; ++p) {
                for (int i = 0; i < m; ++i) {
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

void gemm_rec_tiling(int m, int n, int k,
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
