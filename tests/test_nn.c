#include "unity/unity.h"
#include "../kernel/kernel.h"
#include "../utils/data_utils.h"

void test_flatten_basic(void)
{
    int numChannels = 3;
    int inputSize = 5;
    float image_data[3][5][5] = {
        // Channel 1
        {{0, 0, 0, 0, 0},
         {0, 1, 1, 1, 0},
         {0, 1, 0, 1, 0},
         {0, 1, 1, 1, 0},
         {0, 0, 0, 0, 0}},
        // Channel 2
        {{0, 0, 0, 0, 0},
         {0, 1, 1, 1, 0},
         {0, 1, 0, 1, 0},
         {0, 1, 1, 1, 0},
         {0, 0, 0, 0, 0}},
        // Channel 3
        {{0, 0, 0, 0, 0},
         {0, 1, 1, 1, 0},
         {0, 1, 0, 1, 0},
         {0, 1, 1, 1, 0},
         {0, 0, 0, 0, 0}}};

    float ***image = init_image(image_data, inputSize, numChannels);
    float *output = flatten(image, inputSize, numChannels);
    const int outputSize = numChannels * inputSize * inputSize;

    float expected_output[75] = {// Expected output size adjusted for 3 channels
                                 // Channel 1
                                 0, 0, 0, 0, 0,
                                 0, 1, 1, 1, 0,
                                 0, 1, 0, 1, 0,
                                 0, 1, 1, 1, 0,
                                 0, 0, 0, 0, 0,
                                 // Channel 2
                                 0, 0, 0, 0, 0,
                                 0, 1, 1, 1, 0,
                                 0, 1, 0, 1, 0,
                                 0, 1, 1, 1, 0,
                                 0, 0, 0, 0, 0,
                                 // Channel 3
                                 0, 0, 0, 0, 0,
                                 0, 1, 1, 1, 0,
                                 0, 1, 0, 1, 0,
                                 0, 1, 1, 1, 0,
                                 0, 0, 0, 0, 0};

    // Compare output to expected_output
    for (int i = 0; i < outputSize; i++)
    {
        TEST_ASSERT_EQUAL_FLOAT(expected_output[i], output[i]);
    }

    // Cleanup
    free(output);
    for (int k = 0; k < numChannels; k++)
    {
        for (int i = 0; i < inputSize; i++)
        {
            free(image[k][i]);
        }
        free(image[k]);
    }
    free(image);
}

// Testing with a simple array
void test_predict_simple_array(void)
{
    float probabilities[] = {0.1f, 0.3f, 0.6f};
    int num_classes = sizeof(probabilities) / sizeof(probabilities[0]);
    int expected_class = 2;
    int predicted_class = predict(probabilities, num_classes);
    TEST_ASSERT_EQUAL_INT(expected_class, predicted_class);
}

// Testing with all values the same
void test_predict_all_same_values(void)
{
    float probabilities[] = {0.3f, 0.3f, 0.3f};
    int num_classes = sizeof(probabilities) / sizeof(probabilities[0]);
    int expected_class = 0; // In case of tie, the first class is selected
    int predicted_class = predict(probabilities, num_classes);
    TEST_ASSERT_EQUAL_INT(expected_class, predicted_class);
}

// Testing with negatives and positives
void test_predict_mix_of_negatives_and_positives(void)
{
    float probabilities[] = {-0.1f, -0.2f, 0.3f, -0.4f};
    int num_classes = sizeof(probabilities) / sizeof(probabilities[0]);
    int expected_class = 2;
    int predicted_class = predict(probabilities, num_classes);
    TEST_ASSERT_EQUAL_INT(expected_class, predicted_class);
}

// Add more test cases as needed
