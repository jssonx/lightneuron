#include "conv.h"

// Im2col algorithm
float **im2col(float ***image, int numChannels, int imageSize, int kernelSize, int stride, int *outputSize)
{
    int outputRows = kernelSize * kernelSize * numChannels;
    int outputCols = ((imageSize - kernelSize) / stride + 1) * ((imageSize - kernelSize) / stride + 1);

    float **output = (float **)malloc(outputRows * sizeof(float *));
    for (int i = 0; i < outputRows; i++)
    {
        output[i] = (float *)malloc(outputCols * sizeof(float));
    }

    int colIdx = 0;
    for (int row = 0; row <= imageSize - kernelSize; row += stride)
    {
        for (int col = 0; col <= imageSize - kernelSize; col += stride)
        {
            for (int c = 0; c < numChannels; c++)
            {
                for (int i = 0; i < kernelSize; i++)
                {
                    for (int j = 0; j < kernelSize; j++)
                    {
                        int outputRow = c * kernelSize * kernelSize + i * kernelSize + j;
                        output[outputRow][colIdx] = image[c][row + i][col + j];
                    }
                }
            }
            colIdx++;
        }
    }

    *outputSize = outputCols;
    return output;
}

// Im2col algorithm's inverse
float ***col2im(float **result, int num_kernels, int conv_rows, int conv_cols)
{
    float ***convolutions = (float ***)malloc(num_kernels * sizeof(float **));
    for (int k = 0; k < num_kernels; k++)
    {
        convolutions[k] = (float **)malloc(conv_rows * sizeof(float *));
        for (int i = 0; i < conv_rows; i++)
        {
            convolutions[k][i] = (float *)malloc(conv_cols * sizeof(float));
        }
    }

    for (int k = 0; k < num_kernels; k++)
    {
        for (int i = 0; i < conv_rows; i++)
        {
            for (int j = 0; j < conv_cols; j++)
            {
                convolutions[k][i][j] = result[k][i * conv_cols + j];
            }
        }
    }

    return convolutions;
}

float **kernel_flatten(float ****kernel, int num_kernels, int kernel_size)
{
    float **flattened_kernels = (float **)malloc(num_kernels * sizeof(float *));
    for (int i = 0; i < num_kernels; i++)
    {
        flattened_kernels[i] = (float *)malloc(kernel_size * kernel_size * sizeof(float));
    }

    for (int k = 0; k < num_kernels; k++)
    {
        for (int i = 0; i < kernel_size; i++)
        {
            for (int j = 0; j < kernel_size; j++)
            {
                flattened_kernels[k][i * kernel_size + j] = kernel[k][0][i][j];
            }
        }
    }

    return flattened_kernels;
}

// Basic convolution operation
float ***convolution(float ***image, int numChannels, float ****kernel, float *biasData, int numFilters, int inputSize, int kernelSize)
{
    int outputSize = inputSize - kernelSize + 1;

    // Allocate memory for the convolution output
    float ***convOutput = malloc(numFilters * sizeof(*convOutput));
    for (int i = 0; i < numFilters; i++)
    {
        convOutput[i] = malloc(outputSize * sizeof(*convOutput[i]));
        for (int j = 0; j < outputSize; j++)
        {
            convOutput[i][j] = malloc(outputSize * sizeof(*convOutput[i][j]));
        }
    }

    // Perform the convolution operation
    for (int i = 0; i < numFilters; i++)
    {
        for (int j = 0; j < outputSize; j++)
        {
            for (int k = 0; k < outputSize; k++)
            {
                convOutput[i][j][k] = 0;
                for (int c = 0; c < numChannels; c++)
                {
                    for (int m = 0; m < kernelSize; m++)
                    {
                        for (int n = 0; n < kernelSize; n++)
                        {
                            convOutput[i][j][k] += image[c][j + m][k + n] * kernel[i][c][m][n];
                        }
                    }
                }
                // Add the bias term for the i-th filter
                convOutput[i][j][k] += biasData[i];

                // Apply ReLU activation function
                convOutput[i][j][k] = relu(convOutput[i][j][k]);
            }
        }
    }

    return convOutput;
}

// Convolution with im2col algorithm
float ***convolution_im2col(float ***image, int numChannels, float ****kernel, float *biasData, int numFilters, int inputSize, int kernelSize, MatmulType matmul_type)
{
    // Flatten kernel
    float **flattened_kernel = kernel_flatten(kernel, numFilters, kernelSize);

    // im2col
    int outputSize;
    float **im2col_output = im2col(image, numChannels, inputSize, kernelSize, 1, &outputSize);

    // matmul
    float **matmul_output;
    switch (matmul_type)
    {
    case MATMUL_BLOCKING:
        matmul_output = matmul_blocking(flattened_kernel, im2col_output, numFilters, kernelSize * kernelSize, kernelSize * kernelSize, outputSize);
        break;
    case MATMUL_BASE:
        matmul_output = matmul(flattened_kernel, im2col_output, numFilters, kernelSize * kernelSize, kernelSize * kernelSize, outputSize);
        break;
    case MATMUL_BLAS:
        matmul_output = matmul_blas(flattened_kernel, im2col_output, numFilters, kernelSize * kernelSize, kernelSize * kernelSize, outputSize);
        break;
    case MATMUL_SPARSE:
        matmul_output = matmul_sparse(flattened_kernel, im2col_output, numFilters, kernelSize * kernelSize, kernelSize * kernelSize, outputSize);
        break;
    case MATMUL_THREAD:
        matmul_output = matmul_thread(flattened_kernel, im2col_output, numFilters, kernelSize * kernelSize, kernelSize * kernelSize, outputSize);
        break;
    default:
        matmul_output = matmul(flattened_kernel, im2col_output, numFilters, kernelSize * kernelSize, kernelSize * kernelSize, outputSize);
        break;
    }

    float ***col2im_output = col2im(matmul_output, numFilters, inputSize - kernelSize + 1, inputSize - kernelSize + 1);

    // add bias and relu
    for (int k = 0; k < numFilters; k++)
    {
        for (int j = 0; j < inputSize - kernelSize + 1; j++)
        {
            for (int l = 0; l < inputSize - kernelSize + 1; l++)
            {
                col2im_output[k][j][l] += biasData[k];
                col2im_output[k][j][l] = relu(col2im_output[k][j][l]);
            }
        }
    }

    // Cleanup

    // cleanup flattened kernel
    for (int i = 0; i < numFilters; i++)
    {
        free(flattened_kernel[i]);
    }
    free(flattened_kernel);
    // cleanup im2col output
    for (int i = 0; i < kernelSize * kernelSize; i++)
    {
        free(im2col_output[i]);
    }
    free(im2col_output);
    // cleanup matmul output
    for (int i = 0; i < numFilters; i++)
    {
        free(matmul_output[i]);
    }
    free(matmul_output);

    return col2im_output;
}