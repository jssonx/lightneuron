#ifndef GEMM_H
#define GEMM_H

#include <stdint.h>

void gemm_v1(int m, int n, int k, float *A, int lda, float *B, int ldb, float *C, int ldc);
void gemm_v2(int m, int n, int k, float *A, int lda, float *B, int ldb, float *C, int ldc);

#endif