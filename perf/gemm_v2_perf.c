#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kernel/gemm/gemm.h"
#include "helpers.h"

float *generate_random_matrix(int rows, int cols) {
    float *matrix = (float *)malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
    return matrix;
}

void free_matrix(float *matrix) {
    free(matrix);
}

int main() {
    srand(time(NULL));

    FILE *f = fopen("./bench/gflops_v1.txt", "w");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    for (int N = 1; N <= 1000; N++) {
        float *A = generate_random_matrix(N, N);
        float *B = generate_random_matrix(N, N);
        float *C = (float *)malloc(N * N * sizeof(float));

        printf("Performing multiplication for N = %d...\n", N);
        uint64_t start = nanos();
        gemm_v2(N, N, N, A, N, B, N, C, N);
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
