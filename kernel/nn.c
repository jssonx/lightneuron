#include "nn.h"

float *flatten(float ***input, int inputSize, int depth)
{
  if (inputSize <= 0 || depth <= 0)
  {
    return NULL;
  }

  int outputSize = inputSize * inputSize * depth;
  float *output = malloc(outputSize * sizeof(*output));
  int idx = 0;

  for (int k = 0; k < depth; k++)
  {
    for (int i = 0; i < inputSize; i++)
    {
      for (int j = 0; j < inputSize; j++)
      {
        output[idx++] = input[k][i][j];
      }
    }
  }

  return output;
}

void destroyConvOutput(float ***convOutput, int convOutputSize)
{
  for (int i = 0; i < 32; i++)
  {
    for (int j = 0; j < convOutputSize; j++)
    {
      free(convOutput[i][j]);
    }
    free(convOutput[i]);
  }
  free(convOutput);
}

int forwardPass(float ***image, int numChannels, float ****conv1WeightsData, float **fc1WeightsData, float **fc2WeightsData, float *conv1BiasData, float *fc1BiasData, float *fc2BiasData)
{
  // 1. Perform the convolution operation
  float ***convOutput = convolution(image, numChannels, conv1WeightsData, conv1BiasData, 32, 28, 3);
  // float ***convOutput = convolution_im2col(image, numChannels, conv1WeightsData, conv1BiasData, 32, 28, 3, MATMUL_BLOCKING);
  // float ***convOutput = convolution_im2col(image, numChannels, conv1WeightsData, conv1BiasData, 32, 28, 3, MATMUL_BASE);
  // float ***convOutput = convolution_im2col(image, numChannels, conv1WeightsData, conv1BiasData, 32, 28, 3, MATMUL_BLAS);
  // float ***convOutput = convolution_im2col(image, numChannels, conv1WeightsData, conv1BiasData, 32, 28, 3, MATMUL_THREAD);
  // float ***convOutput = convolution_im2col(image, numChannels, conv1WeightsData, conv1BiasData, 32, 28, 3, MATMUL_SPARSE);

  // 2. Flatten the output
  float *fcInput = flatten(convOutput, 26, 32);

  // 3. Perform the fully connected operations
  float *fcOutput1 = linear(fcInput, fc1WeightsData, fc1BiasData, 26 * 26 * 32, 128);
  applyRelu(fcOutput1, 128);
  float *fcOutput2 = linear(fcOutput1, fc2WeightsData, fc2BiasData, 128, 10);

  // 4. Apply the final softmax activation
  float *softmaxOutput = softmax(fcOutput2, 10);

  // 5. Make predictions
  int predictedClass = predict(softmaxOutput, 10);

  // Clean up the memory usage
  destroyConvOutput(convOutput, 26);
  free(fcInput);
  free(fcOutput1);
  free(fcOutput2);
  free(softmaxOutput);

  return predictedClass;
}

int predict(float *probabilityVector, int numClasses)
{
  int predictedClass = 0;
  float maxProbability = probabilityVector[0];

  for (int i = 1; i < numClasses; i++)
  {
    if (probabilityVector[i] > maxProbability)
    {
      maxProbability = probabilityVector[i];
      predictedClass = i;
    }
  }
  return predictedClass;
}
