#ifndef GEMM_H
#define GEMM_H

#include <stdint.h>

void gemm_naive(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_interchange_loops(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_parallel_loops(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_tiling(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_rec_tiling(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_simd(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);

#endif // GEMM_H