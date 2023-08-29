#include "gemm.h"

#define A(i,j) A[(i)*lda + (j)]
#define B(i,j) B[(i)*ldb + (j)]
#define C(i,j) C[(i)*ldc + (j)]

void gemm_v1(int m, int n, int k,
            float *A, int lda, 
            float *B, int ldb,
            float *C, int ldc)
{
    for (int i = 0; i < m; i++) // Loop over the rows of C
    {
        for (int j = 0; j < n; j++) // Loop over the columns of C
        {
            C(i, j) = 0.0; // Initialize C[i, j]
            for (int p = 0; p < k; p++) // Update C(i, j) with the inner product of the ith row of A and the jth column of B
            {
                C(i, j) += A(i, p) * B(p, j);
            }
        }
    }
}