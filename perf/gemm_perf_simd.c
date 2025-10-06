#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include "../kernel/gemm/gemm.h"
#include "helpers.h"
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#define HAS_MM_MALLOC 1
#else
#define HAS_MM_MALLOC 0
#endif
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

double *generate_random_matrix(int rows, int cols)
{
    // Ensure alignment for AVX operations (32-byte alignment for AVX)
#if HAS_MM_MALLOC
    double *matrix = (double *)_mm_malloc(rows * cols * sizeof(double), 32); // 32 bytes is the width of the avx2 register
#else
    double *matrix = (double *)aligned_alloc(32, rows * cols * sizeof(double));
#endif
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
#if HAS_MM_MALLOC
    _mm_free(matrix);
#else
    free(matrix);
#endif
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

    // ============================================================
    // CONFIGURATION: Choose GEMM implementation to test
    // ============================================================
    GemmType gemm_type = GEMM_SIMD;  // 👈 Change this to test different implementations

    // Available options:
    // GEMM_NAIVE       - v1: Basic ijk loop
    // GEMM_INTERCHANGE - v2: Loop interchange (jpi)
    // GEMM_PARALLEL    - v4: OpenMP parallel loops
    // GEMM_TILING      - v5: Loop tiling/blocking
    // GEMM_REC_TILING  - v6: Recursive divide-and-conquer
    // GEMM_SIMD        - v7: AVX2 SIMD + recursive
    // GEMM_BLAS        - v10: BLAS library
    // GEMM_CUDA_STYLE  - v11: CUDA-inspired multi-level tiling
    // ============================================================

    // Create bench directory if it doesn't exist
#ifdef _WIN32
    _mkdir("./bench");
#else
    mkdir("./bench", 0755);
#endif

    FILE *f = fopen("./bench/OpenBLAS.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    printf("Testing GEMM implementation: %s\n\n", gemm_get_name(gemm_type));

    for (int N = 1024; N <= 1024; N += 40)
    {
        double *A = generate_random_matrix(N, N);
        double *B = generate_random_matrix(N, N);
#if HAS_MM_MALLOC
        double *A_transposed = (double *)_mm_malloc(N * N * sizeof(double), 32);
        double *B_transposed = (double *)_mm_malloc(N * N * sizeof(double), 32);
        double *C = (double *)_mm_malloc(N * N * sizeof(double), 32);
#else
        double *A_transposed = (double *)aligned_alloc(32, N * N * sizeof(double));
        double *B_transposed = (double *)aligned_alloc(32, N * N * sizeof(double));
        double *C = (double *)aligned_alloc(32, N * N * sizeof(double));
#endif
        memset(C, 0.0, N * N * sizeof(double));

        transpose_matrix(A, A_transposed, N, N);
        transpose_matrix(B, B_transposed, N, N);

        printf("Performing multiplication for N = %d...\n", N);
        uint64_t start = nanos();
        int num_iterations = 3;
        for (int i = 0; i < num_iterations; ++i)
        {
            memset(C, 0.0, N * N * sizeof(double)); // Reset C to zero to check the result

            // Use dispatcher to call the selected implementation
            if (gemm_type == GEMM_BLAS) {
                // BLAS uses different matrix layout (not transposed)
                gemm_dispatch(gemm_type, N, N, N, A, N, B, N, C, N);
            } else {
                // Other implementations use transposed B
                gemm_dispatch(gemm_type, N, N, N, A, N, B_transposed, N, C, N);
            }
        }
        uint64_t end = nanos();

        double gflop = (2.0 * (double)N * (double)N * (double)N) * 1e-9;
        double s = (double)(end - start) * 1e-9 / (double)num_iterations;
        printf("%f GFLOPS -- %.2f ms\n", gflop / s, s * 1e3);
        fprintf(f, "%d, %f\n", N, gflop / s);

        // Add a module to check the result using CBLAS
#if HAS_MM_MALLOC
        double *C_cblas = (double *)_mm_malloc(N * N * sizeof(double), 32);
#else
        double *C_cblas = (double *)aligned_alloc(32, N * N * sizeof(double));
#endif
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
