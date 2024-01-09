#ifndef GEMM_H
#define GEMM_H

#include <stdint.h>

void gemm_v1(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_v2(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_v3(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_v4(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);

void gemm_1x4_v4(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_1x4_v5(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_1x4_v6(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_1x4_v7(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_1x4_v8(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_1x4_v9(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_1x4_v10(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);

void gemm_4x4_v4(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_4x4_v5(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_4x4_v6(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_4x4_v7(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_4x4_v8(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_4x4_v9(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_4x4_v10(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_4x4_v11(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_4x4_v12(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_4x4_v13(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_4x4_v14(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_4x4_v15(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void gemm_4x4_v16(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);

#endif // GEMM_H