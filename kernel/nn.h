#ifndef NN_H
#define NN_H

#include "functional.h"
#include "linear.h"
#include "matrix_ops.h"
#include "conv.h"

float *flatten(float ***input, int inputSize, int depth);
int forwardPass(float ***image, int numChannels, float ****conv1WeightsData, float **fc1WeightsData, float **fc2WeightsData, float *conv1BiasData, float *fc1BiasData, float *fc2BiasData);
int predict(float *probabilityVector, int numClasses);

#endif // NN_H