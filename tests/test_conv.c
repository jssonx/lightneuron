#include "unity/unity.h"
#include "../kernel/kernel.h"
#include "../utils/data_utils.h"
#include "test_conv.h"


#define EPSILON 0.000001f

void free_image(float*** image, int numChannels, int inputSize) {
    for (int c = 0; c < numChannels; c++) {
        for (int i = 0; i < inputSize; i++) {
            free(image[c][i]);
        }
        free(image[c]);
    }
    free(image);
}

void free_kernel(float**** kernel, int numFilters, int kernelSize) {
    for (int i = 0; i < numFilters; i++) {
        for (int j = 0; j < 1; j++) {
            for (int k = 0; k < kernelSize; k++) {
                free(kernel[i][j][k]);
            }
            free(kernel[i][j]);
        }
        free(kernel[i]);
    }
    free(kernel);
}

void free_convOutput(float*** convOutput, int numFilters, int outputSize) {
    for (int i = 0; i < numFilters; i++) {
        for (int j = 0; j < outputSize; j++) {
            free(convOutput[i][j]);
        }
        free(convOutput[i]);
    }
    free(convOutput);
}


void assert_float_array_equal_conv(float ***expected, float ***actual, int depth, int rows, int cols)
{
    for (int i = 0; i < depth; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            for (int k = 0; k < cols; k++)
            {
                UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, expected[i][j][k], actual[i][j][k], __LINE__, "Arrays Not Equal!");
            }
        }
    }
}

void test_conv(void)
{
    // Setup
    float image_data[1][5][5] = {
        {
            {0, 0, 0, 0, 0},
            {0, 1, 1, 1, 0},
            {0, 1, 0, 1, 0},
            {0, 1, 1, 1, 0},
            {0, 0, 0, 0, 0}
        }
    };
    int numChannels = 1;
    float ***image = init_image(image_data, 5, numChannels);


    float kernel_data[1][1][3][3] = {
        {
            {
                {1, 0, 1},
                {2, 0, 2},
                {1, 0, 1}
            }
        }
    };
    int numFilters = 1;
    int kernelSize = 3;
    float ****kernel = init_kernel(kernel_data, numFilters, kernelSize);

    // Initialize the bias
    float *biasData = (float *)malloc(1 * sizeof(float));
    biasData[0] = 0;

    float ***expected = (float ***)malloc(1 * sizeof(float **));
    for(int i = 0; i < 1; i++) {
        expected[i] = (float **)malloc(3 * sizeof(float *));
        for(int j = 0; j < 3; j++) {
            expected[i][j] = (float *)malloc(3 * sizeof(float));
        }
    }
    
    // Initialize the expected result
    float expected_values[1][3][3] = {
        {
            {2, 6, 2},
            {2, 8, 2},
            {2, 6, 2}
        }
    };
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                expected[i][j][k] = expected_values[i][j][k];
            }
        }
    }

    // Run function under test
    float ***convOutput = convolution(image, numChannels, kernel, biasData, 1, 5, 3);

    // Check expectations
    assert_float_array_equal_conv(expected, convOutput, 1, 3, 3);

    // Cleanup
    free_image(image, numChannels, 5);
    free_kernel(kernel, 1, 3);
    free(biasData);
    for(int i = 0; i < 1; i++) {
        for(int j = 0; j < 3; j++) {
            free(expected[i][j]);
        }
        free(expected[i]);
    }
    free(expected);
    free_convOutput(convOutput, numFilters, kernelSize);
}

