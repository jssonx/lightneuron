#include "linear.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "functional.h"
#include "matrix_ops.h"

float *linear(float *input, float **weights, float *biases, int inputSize, int outputSize)
{
  // Check for empty input
  if (input == NULL || inputSize == 0 || outputSize == 0)
  {
    return NULL;
  }

  float *output = malloc(outputSize * sizeof(*output));

  for (int i = 0; i < outputSize; i++)
  {
    output[i] = biases[i];
    for (int j = 0; j < inputSize; j++)
    {
      output[i] += weights[i][j] * input[j];
    }
  }

  return output;
}

#define TILE_SIZE_I 32
#define TILE_SIZE_J 256

float *linear_blocking(float *input, float **weights, float *biases, int inputSize, int outputSize) {
    if (input == NULL || inputSize == 0 || outputSize == 0) {
        return NULL;
    }

    float *output = malloc(outputSize * sizeof(*output));

    // Initialize output with biases
    for (int i = 0; i < outputSize; i++) {
        output[i] = biases[i];
    }

    // Outer two loops iterate over tiles
    for (int i = 0; i < outputSize; i += TILE_SIZE_I) {
        for (int j = 0; j < inputSize; j += TILE_SIZE_J) {
            // Inner two loops iterate within tiles
            for (int k = i; k < i + TILE_SIZE_I && k < outputSize; k++) {
                for (int l = j; l < j + TILE_SIZE_J && l < inputSize; l++) {
                    output[k] += weights[k][l] * input[l];
                }
            }
        }
    }

    return output;
}
