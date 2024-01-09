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
                for (int p = 0; p < TILE_SIZE; ++p) {
                    for (int j = 0; j < TILE_SIZE; ++j) {
                        for (int i = 0; i < TILE_SIZE; ++i) {
                            C(i+ii, j+jj) += A(i+ii, p+kk) * B(p+kk, j+jj);
                        }
                    }
                }
            }
        }
    }
}


// #include <omp.h>

// #define TILE_SIZE 16
// #define A(i, j) a[(j) * lda + (i)]
// #define B(i, j) b[(j) * ldb + (i)]
// #define C(i, j) c[(j) * ldc + (i)]

// void gemm_omp(int m, int n, int k,
//               double *a, int lda,
//               double *b, int ldb,
//               double *c, int ldc)
// {
//     #pragma omp parallel for collapse(2)
//     for (int jj = 0; jj < n; jj += TILE_SIZE) {
//         for (int ii = 0; ii < m; ii += TILE_SIZE) {
//             for (int kk = 0; kk < k; kk += TILE_SIZE) {
//                 for (int j = jj; j < jj + TILE_SIZE && j < n; ++j) {
//                     for (int i = ii; i < ii + TILE_SIZE && i < m; ++i) {
//                         for (int p = kk; p < kk + TILE_SIZE && p < k; ++p) {
//                             C(i, j) += A(i, p) * B(p, j);
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

