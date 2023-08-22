#ifndef LINEAR_H
#define LINEAR_H

#include <float.h>
#include <stdlib.h>

float *linear(float *input, float **weights, float *biases, int inputSize, int outputSize);
float *linear_blocking(float *input, float **weights, float *biases, int inputSize, int outputSize);

#endif // LINEAR_H