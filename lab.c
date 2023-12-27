#include "lab.h"

int main(int argc, char *argv[])
{
    hid_t file_id;
    herr_t status;
    float *conv1BiasData, *fc1BiasData, *fc2BiasData;
    float **fc1WeightsData, **fc2WeightsData;
    int conv1BiasSize, fc1BiasSize, fc2BiasSize;
    int fc1WeightsDim1, fc1WeightsDim2, fc2WeightsDim1, fc2WeightsDim2;

    float ****conv1WeightsData;
    int conv1WeightsDim1, conv1WeightsDim2, conv1WeightsDim3, conv1WeightsDim4;

    file_id = H5Fopen(PARAMS_FILENAME, H5F_ACC_RDONLY, H5P_DEFAULT);

    read_float_1d_params(file_id, "conv1.bias", &conv1BiasData, &conv1BiasSize);
    read_float_1d_params(file_id, "fc1.bias", &fc1BiasData, &fc1BiasSize);
    read_float_1d_params(file_id, "fc2.bias", &fc2BiasData, &fc2BiasSize);
    read_float_2d_params(file_id, "fc1.weight", &fc1WeightsData, &fc1WeightsDim1, &fc1WeightsDim2);
    read_float_2d_params(file_id, "fc2.weight", &fc2WeightsData, &fc2WeightsDim1, &fc2WeightsDim2);
    read_float_4d_params(file_id, "conv1.weight", &conv1WeightsData, &conv1WeightsDim1, &conv1WeightsDim2, &conv1WeightsDim3, &conv1WeightsDim4);

    // Load in images and labels
    int numImages = 100;
    int numChannels = 1;
    float ****images = loadImages("mnist_data/MNIST/raw/t10k-images-idx3-ubyte", numImages, numChannels);
    int numLabels = numImages;
    int *labels = loadLabels("mnist_data/MNIST/raw/t10k-labels-idx1-ubyte", numLabels);

    int numCorrect = 0;
    printf("IMAGE\t\tPREDICTION\tLABEL\n");
    for (int i = 0; i < numImages; i++)
    {
        int prediction = forwardPass(images[i], numChannels, conv1WeightsData, fc1WeightsData, fc2WeightsData, conv1BiasData, fc1BiasData, fc2BiasData);

        printf("Image: %d\tPrediction: %d\tLabel: %d\n", i, prediction, labels[i]);
        if (prediction == labels[i])
        {
            numCorrect++;
        }
    }

    float accuracy = numCorrect * 1.0 / numLabels;
    printf("Total Accuracy: %f%%\n", accuracy * 100);

    status = H5Fclose(file_id);

    cleanup_float_1d(conv1BiasData);
    cleanup_float_1d(fc1BiasData);
    cleanup_float_1d(fc2BiasData);
    cleanup_float_2d(fc1WeightsData, fc1WeightsDim1);
    cleanup_float_2d(fc2WeightsData, fc2WeightsDim1);
    cleanup_float_4d(conv1WeightsData, conv1WeightsDim1, conv1WeightsDim2, conv1WeightsDim3);
    destroyImages(images, numImages, numChannels);
    free(labels);

    return EXIT_SUCCESS;
}
