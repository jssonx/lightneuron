#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../kernel/gemm/gemm.h"
#include "helpers.h"
#include <immintrin.h>
#include "mkl.h"

double *generate_random_matrix(int rows, int cols)
{
    // Ensure alignment for AVX operations (32-byte alignment for AVX)
    double *matrix = (double *)_mm_malloc(rows * cols * sizeof(double), 32); // 32 bytes is the width of the avx2 register
    if (matrix == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for the matrix\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows * cols; i++)
    {
        matrix[i] = (double)rand() / (double)RAND_MAX;
    }
    return matrix;
}

void free_matrix(double *matrix)
{
    _mm_free(matrix);
}

void print_matrix_col_major(const char *name, double *matrix, int rows, int cols)
{
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++)
    {
        printf("[ ");
        for (int j = 0; j < cols; j++)
        {
            // ColMajor, element index = j*rows + i
            printf("%f ", matrix[j * rows + i]);
        }
        printf("]\n");
    }
    printf("\n");
}

void transpose_matrix(double *matrix, double *transposed, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[i * cols + j] = matrix[j * rows + i];
        }
    }
}


int main()
{
    // srand(time(NULL));
    srand(42);

    FILE *f = fopen("./bench/OpenBLAS.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    for (int N = 1024; N <= 1024; N += 40)
    {
        double *A = generate_random_matrix(N, N);
        double *B = generate_random_matrix(N, N);
        double *A_transposed = (double *)_mm_malloc(N * N * sizeof(double), 32);
        double *B_transposed = (double *)_mm_malloc(N * N * sizeof(double), 32);
        double *C = (double *)_mm_malloc(N * N * sizeof(double), 32);
        memset(C, 0.0, N * N * sizeof(double));

        transpose_matrix(A, A_transposed, N, N);
        transpose_matrix(B, B_transposed, N, N);

        printf("Performing multiplication for N = %d...\n", N);
        uint64_t start = nanos();
        // gemm_tiling(N, N, N, A, N, B, N, C, N);
        int num_iterations = 3;
        for (int i = 0; i < num_iterations; ++i)
        {
            memset(C, 0.0, N * N * sizeof(double)); // Reset C to zero to check the result
            // gemm_interchange_loops(N, N, N, A, N, B_transposed, N, C, N);
            gemm_simd(N, N, N, A, N, B_transposed, N, C, N);
            // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);
        }
        uint64_t end = nanos();

        double gflop = (2.0 * (double)N * (double)N * (double)N) * 1e-9;
        double s = (double)(end - start) * 1e-9 / (double)num_iterations;
        printf("%f GFLOPS -- %.2f ms\n", gflop / s, s * 1e3);
        fprintf(f, "%d, %f\n", N, gflop / s);

        // Add a module to check the result using CBLAS
        double *C_cblas = (double *)_mm_malloc(N * N * sizeof(double), 32);
        memset(C_cblas, 0.0, N * N * sizeof(double));
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C_cblas, N);

        const double tolerance = 1e-6;
        double max_diff = 0.0;
        for (int i = 0; i < N * N; ++i)
        {
            double diff = fabs(C[i] - C_cblas[i]);
            if (diff > max_diff)
            {
                max_diff = diff;
            }
        }

        // Check if the max difference is within the tolerance
        if (max_diff > tolerance)
        {
            printf("Max difference between C and C_cblas: %f\n", max_diff);
            printf("The result may not be accurate.\n");
        }
        else
        {
            printf("The result is accurate within the tolerance.\n");
        }

#if 0
        // print A, B, C
        print_matrix_col_major("A", A, N, N);
        print_matrix_col_major("B", B, N, N);
        print_matrix_col_major("C", C, N, N);
#endif
        free_matrix(A);
        free_matrix(B);
        free_matrix(C);
        free_matrix(A_transposed);
        free_matrix(B_transposed);
    }

    fclose(f);
    printf("Multiplications and measurements completed.\n");

    return 0;
}
