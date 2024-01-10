#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include "../kernel/gemm/gemm.h"
#include "helpers.h"
#include <immintrin.h>

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
        gemm_rec_tiling(N, N, N, A, N, B, N, C, N);
        // int num_iterations = 3;
        // for (int i = 0; i < num_iterations; ++i)
        // {
        //     memset(C, 0.0, N * N * sizeof(double)); // Reset C to zero
        //     gemm_rec_tiling(N, N, N, A, N, B, N, C, N);
        //     // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);
        // }
        uint64_t end = nanos();

        // double gflop = (2.0 * (double)N * (double)N * (double)N) * 1e-9 * (double)num_iterations;
        double gflop = (2.0 * (double)N * (double)N * (double)N) * 1e-9;
        double s = (double)(end - start) * 1e-9;
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

// valgrind --tool=callgrind ./gemm
// valgrind --tool=cachegrind ./gemm

/* DEBUG
        double A_data[] = {
            0.771436, 0.311803, 0.419769, 0.506543, 0.689725, 0.538201, 0.362538, 0.301766,
            0.923273, 0.970391, 0.846840, 0.484018, 0.403277, 0.236746, 0.569909, 0.148699,
            0.633787, 0.642778, 0.950191, 0.611267, 0.567098, 0.033122, 0.567154, 0.350387,
            0.155592, 0.151729, 0.592983, 0.600836, 0.537108, 0.352212, 0.945186, 0.308544,
            0.664015, 0.364955, 0.815087, 0.353739, 0.903156, 0.177626, 0.655505, 0.826429,
            0.148016, 0.502345, 0.310447, 0.551294, 0.739091, 0.880355, 0.699992, 0.372878,
            0.523133, 0.650183, 0.984145, 0.090231, 0.683305, 0.551299, 0.440618, 0.838897,
            0.703028, 0.033601, 0.439733, 0.240136, 0.385813, 0.384919, 0.548681, 0.049827};

        double B_data[] = {
            0.749874, 0.363768, 0.403567, 0.653030, 0.541394, 0.059072, 0.479459, 0.689410,
            0.561417, 0.789906, 0.240703, 0.300508, 0.670262, 0.940696, 0.673386, 0.193395,
            0.590879, 0.657531, 0.283626, 0.274183, 0.208830, 0.724243, 0.113080, 0.911858,
            0.757844, 0.552813, 0.151994, 0.143657, 0.937732, 0.700675, 0.193484, 0.687606,
            0.064443, 0.597051, 0.340636, 0.605837, 0.656123, 0.820095, 0.295246, 0.217540,
            0.610002, 0.535950, 0.518048, 0.280263, 0.476645, 0.191435, 0.473658, 0.067524,
            0.848966, 0.757283, 0.341707, 0.057795, 0.481527, 0.454787, 0.969653, 0.239371,
            0.007600, 0.121648, 0.383027, 0.945332, 0.822323, 0.576512, 0.632937, 0.886766};

        // 使用_mm_malloc确保内存对齐
        double *A = (double *)_mm_malloc(8 * 8 * sizeof(double), 32);
        memcpy(A, A_data, 8 * 8 * sizeof(double));

        double *B = (double *)_mm_malloc(8 * 8 * sizeof(double), 32);
        memcpy(B, B_data, 8 * 8 * sizeof(double));

*/


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
