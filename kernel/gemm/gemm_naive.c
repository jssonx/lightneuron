
/* Create macros so that the matrices are stored in column-major order */

#define A(i, j) a[(j) * lda + (i)]
#define B(i, j) b[(j) * ldb + (i)]
#define C(i, j) c[(j) * ldc + (i)]

/* Routine for computing C = A * B + C */

/* i, j, p */
void gemm_naive(int m, int n, int k,
             double *a, int lda,
             double *b, int ldb,
             double *c, int ldc)
{
    int i, j, p;

    for (i = 0; i < m; ++i) // Loop over the rows of C
    {
        for (j = 0; j < n; ++j) // Loop over the columns of C
        {
            for (p = 0; p < k; ++p) // Update C( i,j ) with the inner product of the ith row of A and the jth column of B
            {
                C(i, j) += A(i, p) * B(p, j);
            }
        }
    }
}


/* p, j, i */
// void gemm_naive(int m, int n, int k,
//              double *a, int lda,
//              double *b, int ldb,
//              double *c, int ldc)
// {
//     int i, j, p;

//     for (p = 0; p < k; ++p)
//     {
//         for (j = 0; j < n; ++j)
//         {
//             for (i = 0; i < m; ++i)
//             {
//                 C(i, j) += A(i, p) * B(p, j);
//             }
//         }
//     }
// }


/* p, i, j */
// void gemm_naive(int m, int n, int k,
//              double *a, int lda,
//              double *b, int ldb,
//              double *c, int ldc)
// {
//     int i, j, p;

//     for (p = 0; p < k; ++p)
//     {
//         for (i = 0; i < m; ++i)
//         {
//             for (j = 0; j < n; ++j)
//             {
//                 C(i, j) += A(i, p) * B(p, j);
//             }
//         }
//     }
// }

/* j, p, i */
// void gemm_naive(int m, int n, int k,
//              double *a, int lda,
//              double *b, int ldb,
//              double *c, int ldc)
// {
//     int i, j, p;

//     for (j = 0; j < n; ++j)
//     {
//         for (p = 0; p < k; ++p)
//         {
//             for (i = 0; i < m; ++i)
//             {
//                 C(i, j) += A(i, p) * B(p, j);
//             }
//         }
//     }
// }


/* j, i, p */
// void gemm_naive(int m, int n, int k,
//              double *a, int lda,
//              double *b, int ldb,
//              double *c, int ldc)
// {
//     int i, j, p;

//     for (j = 0; j < n; ++j)
//     {
//         for (i = 0; i < m; ++i)
//         {
//             for (p = 0; p < k; ++p)
//             {
//                 C(i, j) += A(i, p) * B(p, j);
//             }
//         }
//     }
// }


/* i, p, j */
// void gemm_naive(int m, int n, int k,
//              double *a, int lda,
//              double *b, int ldb,
//              double *c, int ldc)
// {
//     int i, j, p;

//     for (i = 0; i < m; ++i)
//     {
//         for (p = 0; p < k; ++p)
//         {
//             for (j = 0; j < n; ++j)
//             {
//                 C(i, j) += A(i, p) * B(p, j);
//             }
//         }
//     }
// }


