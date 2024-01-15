#include <omp.h>

#define TILE_SIZE 32
#define A(i, j) a[(j) * lda + (i)]
#define B(i, j) b[(j) * ldb + (i)]
#define C(i, j) c[(j) * ldc + (i)]

void gemm_tiling(int m, int n, int k,
              double *a, int lda,
              double *b, int ldb,
              double *c, int ldc)
{
    int i, j, p;
    int ii, jj, kk;
    #pragma omp parallel for private(i, j, p, ii, jj, kk) collapse(2)
    for (int jj = 0; jj < n; jj += TILE_SIZE) {
        for (int kk = 0; kk < k; kk += TILE_SIZE) {
            for (int ii = 0; ii < m; ii += TILE_SIZE) {
                for (int j = 0; j < TILE_SIZE; ++j) {
                    for (int p = 0; p < TILE_SIZE; ++p) {
                        for (int i = 0; i < TILE_SIZE; ++i) {
                            C(i+ii, j+jj) += A(i+ii, p+kk) * B(p+kk, j+jj);
                            // C(i+ii, j+jj) += A(i+ii, p+kk) * B(j+jj, p+kk);
                        }
                    }
                }
            }
        }
    }
}
