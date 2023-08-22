#ifndef FUNCTIONAL_H
#define FUNCTIONAL_H

#include <math.h>
#include <float.h>
#include <stdlib.h>

float relu(float x);

void applyRelu(float *input, int inputSize);

float *softmax(float *input, int inputSize);

#endif // FUNCTIONAL_H