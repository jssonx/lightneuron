#include "matrix_ops.h"

#ifndef BLOCK_SIZE_I
#define BLOCK_SIZE_I 169
#endif

#ifndef BLOCK_SIZE_J
#define BLOCK_SIZE_J 32
#endif

#ifndef BLOCK_SIZE_K
#define BLOCK_SIZE_K 9
#endif

/* Create macros so that the matrices are stored in row-major order */

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



float **matmul(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    if (A_cols != B_rows)
    {
        printf("Matrix dimensions incompatible for multiplication.\n");
        return NULL;
    }

    float **C = (float **)malloc(A_rows * sizeof(float *));
    for (int i = 0; i < A_rows; i++)
    {
        C[i] = (float *)malloc(B_cols * sizeof(float));
    }

    for (int i = 0; i < A_rows; i++)
    {
        for (int j = 0; j < B_cols; j++)
        {
            C[i][j] = 0.0;
            for (int k = 0; k < A_cols; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

// Matmul with blocking optimization
float **matmul_blocking(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    if (A_cols != B_rows)
    {
        printf("Matrix dimensions incompatible for multiplication.\n");
        return NULL;
    }

    float **C = (float **)malloc(A_rows * sizeof(float *));
    for (int i = 0; i < A_rows; i++)
    {
        C[i] = (float *)calloc(B_cols, sizeof(float));
    }

    for (int ii = 0; ii < A_rows; ii += BLOCK_SIZE_I)
    {
        for (int jj = 0; jj < B_cols; jj += BLOCK_SIZE_J)
        {
            for (int kk = 0; kk < A_cols; kk += BLOCK_SIZE_K)
            {
                for (int i = ii; i < ii + BLOCK_SIZE_I && i < A_rows; i++)
                {
                    for (int j = jj; j < jj + BLOCK_SIZE_J && j < B_cols; j++)
                    {
                        float sum = C[i][j];
                        for (int k = kk; k < kk + BLOCK_SIZE_K && k < A_cols; k++)
                        {
                            sum += A[i][k] * B[k][j];
                        }
                        C[i][j] = sum;
                    }
                }
            }
        }
    }
    return C;
}

// Matmul with BLAS optimization
float **matmul_blas(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    if (A_cols != B_rows)
    {
        printf("Matrix dimensions incompatible for multiplication.\n");
        return NULL;
    }

    // Convert 2D arrays to 1D arrays for use with CBLAS
    float *a = (float *)malloc(A_rows * A_cols * sizeof(float));
    float *b = (float *)malloc(B_rows * B_cols * sizeof(float));
    for (int i = 0; i < A_rows; i++)
    {
        for (int j = 0; j < A_cols; j++)
        {
            a[i * A_cols + j] = A[i][j];
        }
    }
    for (int i = 0; i < B_rows; i++)
    {
        for (int j = 0; j < B_cols; j++)
        {
            b[i * B_cols + j] = B[i][j];
        }
    }

    float *c = (float *)malloc(A_rows * B_cols * sizeof(float));
    // Matrix multiplication using CBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A_rows, B_cols, A_cols, 1.0, a, A_cols, b, B_cols, 0.0, c, B_cols);

    // Convert 1D array back to 2D array for return
    float **C = (float **)malloc(A_rows * sizeof(float *));
    for (int i = 0; i < A_rows; i++)
    {
        C[i] = (float *)malloc(B_cols * sizeof(float));
        for (int j = 0; j < B_cols; j++)
        {
            C[i][j] = c[i * B_cols + j];
        }
    }

    // Free 1D arrays
    free(a);
    free(b);
    free(c);

    return C;
}

// Matmul with sparsity
typedef struct
{
    int *row_ptr;
    int *col_idx;
    float *values;
    int nnz;
} CSRMatrix;

CSRMatrix convert_to_csr(float **A, int rows, int cols)
{
    CSRMatrix csr;
    csr.nnz = 0;

    // Count non-zero elements
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (A[i][j] != 0.0)
            {
                csr.nnz++;
            }
        }
    }

    csr.row_ptr = (int *)malloc((rows + 1) * sizeof(int));
    csr.col_idx = (int *)malloc(csr.nnz * sizeof(int));
    csr.values = (float *)malloc(csr.nnz * sizeof(float));

    int k = 0;
    csr.row_ptr[0] = 0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (A[i][j] != 0.0)
            {
                csr.col_idx[k] = j;
                csr.values[k] = A[i][j];
                k++;
            }
        }
        csr.row_ptr[i + 1] = k;
    }

    return csr;
}

float **matmul_sparse(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    if (A_cols != B_rows)
    {
        printf("Matrix dimensions incompatible for multiplication.\n");
        return NULL;
    }

    CSRMatrix A_csr = convert_to_csr(A, A_rows, A_cols);
    CSRMatrix B_csr = convert_to_csr(B, B_rows, B_cols);

    float **C = (float **)malloc(A_rows * sizeof(float *));
    for (int i = 0; i < A_rows; i++)
    {
        C[i] = (float *)calloc(B_cols, sizeof(float));
        for (int k = A_csr.row_ptr[i]; k < A_csr.row_ptr[i + 1]; k++)
        {
            for (int l = B_csr.row_ptr[A_csr.col_idx[k]]; l < B_csr.row_ptr[A_csr.col_idx[k] + 1]; l++)
            {
                C[i][B_csr.col_idx[l]] += A_csr.values[k] * B_csr.values[l];
            }
        }
    }

    free(A_csr.row_ptr);
    free(A_csr.col_idx);
    free(A_csr.values);
    free(B_csr.row_ptr);
    free(B_csr.col_idx);
    free(B_csr.values);

    return C;
}

// Matmul with pthreads
typedef struct
{
    float **A;
    float **B;
    float **C;
    int start;
    int end;
    int A_cols;
    int B_cols;
} ThreadData;

void *multiply(void *arg)
{
    ThreadData *data = (ThreadData *)arg;
    for (int i = data->start; i < data->end; ++i)
    {
        for (int j = 0; j < data->B_cols; ++j)
        {
            data->C[i][j] = 0;
            for (int k = 0; k < data->A_cols; ++k)
                data->C[i][j] += data->A[i][k] * data->B[k][j];
        }
    }
    pthread_exit(0);
}

float **matmul_thread(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    if (A_cols != B_rows)
    {
        printf("Matrix dimensions incompatible for multiplication.\n");
        return NULL;
    }

    float **C = (float **)malloc(A_rows * sizeof(float *));
    for (int i = 0; i < A_rows; i++)
    {
        C[i] = (float *)malloc(B_cols * sizeof(float));
    }

    int num_threads = 8;
    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    ThreadData *thread_data = (ThreadData *)malloc(num_threads * sizeof(ThreadData));

    int rows_per_thread = A_rows / num_threads;

    for (int t = 0; t < num_threads; t++)
    {
        thread_data[t].A = A;
        thread_data[t].B = B;
        thread_data[t].C = C;
        thread_data[t].A_cols = A_cols;
        thread_data[t].B_cols = B_cols;
        thread_data[t].start = t * rows_per_thread;
        thread_data[t].end = (t == num_threads - 1) ? A_rows : (t + 1) * rows_per_thread;

        pthread_create(&threads[t], NULL, multiply, &thread_data[t]);
    }

    for (int t = 0; t < num_threads; t++)
    {
        pthread_join(threads[t], NULL);
    }

    free(threads);
    free(thread_data);

    return C;
}