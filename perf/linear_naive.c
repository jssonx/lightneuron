#include <stdio.h>
#include <stdlib.h>
#include "kernel/linear.h"

#define INPUT_SIZE 25088
#define OUTPUT_SIZE 128

float *generate_fixed_array(int size, float value) {
    float *arr = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        arr[i] = value;
    }
    return arr;
}

float **generate_fixed_matrix(int rows, int cols, float value) {
    float **matrix = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++) {
        matrix[i] = generate_fixed_array(cols, value);
    }
    return matrix;
}

int main() {

    float *input = generate_fixed_array(INPUT_SIZE, 1.0f);
    float *biases = generate_fixed_array(OUTPUT_SIZE, 1.0f);
    float **weights = generate_fixed_matrix(OUTPUT_SIZE, INPUT_SIZE, 1.0f);

    float *output_naive = linear(input, weights, biases, INPUT_SIZE, OUTPUT_SIZE);

    // Cleanup
    free(input);
    free(biases);
    free(output_naive);
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        free(weights[i]);
    }
    free(weights);

    return 0;
}
