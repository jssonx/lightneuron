#include "functional.h"

float relu(float x)
{
    return fmaxf(0.0f, x);
}
// -O3

void applyRelu(float *input, int inputSize)
{
    for (int i = 0; i < inputSize; i++)
    {
        input[i] = relu(input[i]);
    }
}

float *softmax(float *input, int inputSize)
{
    float *softmaxOutput = malloc(inputSize * sizeof(*softmaxOutput));
    float maxInput = -FLT_MAX;
    float sumExp = 0.0;

    // Find maximum of input vector
    for (int i = 0; i < inputSize; i++)
    {
        if (input[i] > maxInput)
        {
            maxInput = input[i];
        }
    }

    // Compute exp of input - maxInput to avoid underflow
    for (int i = 0; i < inputSize; i++)
    {
        softmaxOutput[i] = expf(input[i] - maxInput);
        sumExp += softmaxOutput[i];
    }

    // Normalise and apply log
    for (int i = 0; i < inputSize; i++)
    {
        softmaxOutput[i] = logf(softmaxOutput[i] / sumExp);
    }

    return softmaxOutput;
}