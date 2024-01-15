
/* Create macros so that the matrices are stored in column-major order */

#define A(i, j) a[(j) * lda + (i)]
#define B(i, j) b[(j) * ldb + (i)]
#define C(i, j) c[(j) * ldc + (i)]


void gemm_interchange_loops(int m, int n, int k,
             double *a, int lda,
             double *b, int ldb,
             double *c, int ldc)
{
    int i, j, p;

    for (j = 0; j < n; ++j)
    {
        for (p = 0; p < k; ++p)
        {
            for (i = 0; i < m; ++i)
            {
                C(i, j) += A(i, p) * B(p, j);
                // C(i, j) += A(i, p) * B(j, p);
                // C(i, j) += A(p, i) * B(p, j);
            }
        }
    }
}
