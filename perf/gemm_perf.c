#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
// #include <cblas.h>
#include "../kernel/gemm/gemm.h"
#include "helpers.h"
#include <immintrin.h>
#include "mkl.h"

/*
 * Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz
 * Microarchitecture: Comet Lake
 * 1.6 GHz is the base frequency of the CPU
 * 4 cores, 2 threads per core
 * 16 DP FLOPS/cycle (AVX2, FP64)
 * Single core theoretical peak performance = 1.6 GHz * 16 FLOPS/cycle = 25.6 GFLOPS
 * Multi-core theoretical peak performance = 25.6 GFLOPS * 4 cores = 102.4 GFLOPS
 * reference:
 *  - https://indico.cern.ch/event/814979/contributions/3401193/attachments/1831477/3105158/comp_arch_codas_2019.pdf
 *  - https://en.wikipedia.org/wiki/FLOPS
*/

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

        double *C = (double *)_mm_malloc(N * N * sizeof(double), 32);
        memset(C, 0.0, N * N * sizeof(double));

        printf("Performing multiplication for N = %d...\n", N);
        uint64_t start = nanos();

        int num_iterations = 3;
        for (int i = 0; i < num_iterations; ++i)
        {
            memset(C, 0.0, N * N * sizeof(double)); // Reset C to zero
            gemm_simd(N, N, N, A, N, B, N, C, N);
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

        // print_matrix_col_major("C_cblas", C_cblas, N, N);

#if 0
        // print A, B, C
        print_matrix_col_major("A", A, N, N);
        print_matrix_col_major("B", B, N, N);
        print_matrix_col_major("C", C, N, N);
#endif
        free_matrix(A);
        free_matrix(B);
        free_matrix(C);
    }

    fclose(f);
    printf("Multiplications and measurements completed.\n");

    return 0;
}

// v1-3
// gcc ./perf/gemm_perf.c ./kernel/gemm/gemm_v1.c -O2 -o gemm && ./gemm
// gcc ./perf/gemm_perf.c ./kernel/gemm/gemm_v4.c -mavx2 -mfma -O2 -o gemm && ./gemm

// gcc ./perf/gemm_perf.c ./kernel/gemm/gemm_1x4_v4.c -O2 -o gemm && ./gemm

// gcc ./perf/gemm_perf.c ./kernel/gemm/gemm_4x4_v11.c -O2 -msse3 -o gemm && ./gemm

// gcc ./perf/gemm_perf.c -lcblas -lopenblas -o gemm && ./gemm

// -lmkl_rt

// valgrind --tool=cachegrind ./gemm

// double *generate_random_matrix(int rows, int cols)
// {
//     // Ensure alignment for AVX operations (32-byte alignment for AVX)
//     double *matrix = (double *)_mm_malloc(rows * cols * sizeof(double), 32);
//     if (matrix == NULL)
//     {
//         fprintf(stderr, "Failed to allocate memory for the matrix\n");
//         exit(EXIT_FAILURE);
//     }

//     for (int i = 0; i < rows * cols; i++)
//     {
//         // Generate a random integer between 0 and 100, then store it as a double
//         matrix[i] = (double)(rand() % 11);
//     }
//     return matrix;
// }
