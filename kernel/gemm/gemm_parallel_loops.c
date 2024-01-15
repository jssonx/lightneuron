#include <omp.h>

#define A(i, j) a[(j) * lda + (i)]
#define B(i, j) b[(j) * ldb + (i)]
#define C(i, j) c[(j) * ldc + (i)]

/* Routine for computing C = A * B + C */

/* p, j, i */
void gemm_parallel_loops(int m, int n, int k,
             double *a, int lda,
             double *b, int ldb,
             double *c, int ldc)
{
    int i, j, p;
    
    #pragma omp parallel for
    for (j = 0; j < n; ++j)
    {
        for (p = 0; p < k; ++p)
        {   
            for (i = 0; i < m; ++i)
            {
                // C(i, j) += A(i, p) * B(j, p);
                C(i, j) += A(i, p) * B(p, j);
            }
        }
    }
}

// -fopenmp