#include "gemm.h"

#define A(i,j) A[(i)*lda + (j)]
#define B(i,j) B[(i)*ldb + (j)]
#define C(i,j) C[(i)*ldc + (j)]

void AddDot(int k, float *x, int incx, float *y, float *gamma);

void gemm_v2(int m, int n, int k,
             float *A, int lda, 
             float *B, int ldb,
             float *C, int ldc)
{
    for (int i = 0; i < m; i++) // Loop over the rows of C
    {
        for (int j = 0; j < n; j++) // Loop over the columns of C
        {
            C(i, j) = 0.0; // Initialize C[i, j]
            AddDot(k, &A(i, 0), lda, &B(0, j), &C(i, j));
        }
    }
}

#define X(i) x[(i)*incx]

void AddDot(int k, float *x, int incx, float *y, float *gamma)
{
    for (int p = 0; p < k; p++)
    {
        *gamma += X(p) * y[p];
    }
}



