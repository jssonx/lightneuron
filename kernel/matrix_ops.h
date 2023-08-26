#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <pthread.h> 
#include <cblas.h>

float **matmul(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols);
float **matmul_blocking(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols);
float **matmul_blas(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols);
float **matmul_sparse(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols);
float **matmul_thread(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols);

void gemm_v1(int m, int n, int k, float *A, int lda, float *B, int ldb, float *C, int ldc);

#endif /* MATRIX_OPS_H */