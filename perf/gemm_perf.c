#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../kernel/gemm/gemm.h"
#include "helpers.h"

double *generate_random_matrix(int rows, int cols) {
    double *matrix = (double *)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)rand() / (double)RAND_MAX;
    }
    return matrix;
}

void free_matrix(double *matrix) {
    free(matrix);
}

int main() {
    srand(time(NULL));

    FILE *f = fopen("./bench/gemm_4x4_v16.txt", "w");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    for (int N = 40; N <= 1000; N+=40) {
        double *A = generate_random_matrix(N, N);
        double *B = generate_random_matrix(N, N);
        double *C = (double *)malloc(N * N * sizeof(double));

        printf("Performing multiplication for N = %d...\n", N);
        uint64_t start = nanos();
        gemm_4x4_v16(N, N, N, A, N, B, N, C, N);
        uint64_t end = nanos();

        double gflop = (2.0 * N * N * N) * 1e-9;
        double s = (end - start) * 1e-9;
        printf("%f GFLOPS -- %.2f ms\n", gflop / s, s * 1e3);
        fprintf(f, "%d, %f\n", N, gflop / s);

        free_matrix(A);
        free_matrix(B);
        free_matrix(C);
    }

    fclose(f);
    printf("Multiplications and measurements completed.\n");

    return 0;
}
