#include "data_utils.h"

float ***init_image(float image_data[][5][5], int inputSize, int channelSize)
{
    float ***image = malloc(channelSize * sizeof(*image));
    for (int c = 0; c < channelSize; c++)
    {
        image[c] = malloc(inputSize * sizeof(**image));
        for (int i = 0; i < inputSize; i++)
        {
            image[c][i] = malloc(inputSize * sizeof(***image));
            for (int j = 0; j < inputSize; j++)
            {
                image[c][i][j] = image_data[c][i][j];
            }
        }
    }
    return image;
}

float ****init_kernel(float kernel_data[][1][3][3], int numFilters, int kernelSize)
{
    float ****kernel = malloc(numFilters * sizeof(*kernel));
    for (int i = 0; i < numFilters; i++)
    {
        kernel[i] = malloc(1 * sizeof(*kernel[i]));
        for (int j = 0; j < 1; j++)
        {
            kernel[i][j] = malloc(kernelSize * sizeof(*kernel[i][j]));
            for (int k = 0; k < kernelSize; k++)
            {
                kernel[i][j][k] = malloc(kernelSize * sizeof(*kernel[i][j][k]));
                for (int m = 0; m < kernelSize; m++)
                {
                    kernel[i][j][k][m] = kernel_data[i][j][k][m];
                }
            }
        }
    }
    return kernel;
}

// loadImages returns a 4D array where each element of the array is a 28x28 image loaded from
// the training dataset.
float ****loadImages(const char *filename, int numImages, int numChannels)
{
    // Image pre-processing (let's just use 1000 for now)
    float ****images = malloc(numImages * sizeof(*images)); // Dimensions: numImages x numChannels x 28 x 28
    for (int i = 0; i < numImages; i++)
    {
        images[i] = malloc(numChannels * sizeof(*images[i]));
        for (int c = 0; c < numChannels; c++)
        {
            images[i][c] = malloc(28 * sizeof(**images[i]));
            for (int j = 0; j < 28; j++)
            {
                images[i][c][j] = malloc(28 * sizeof(***images[i]));
            }
        }
    }

    FILE *file = fopen(filename, "rb");
    if (file == NULL)
    {
        fprintf(stderr, "Error: Unable to open file %s\n", filename);
        exit(1);
    }

    // Skip the magic number and the size headers
    fseek(file, 16, SEEK_SET);
    for (int i = 0; i < numImages; i++)
    {
        for (int c = 0; c < numChannels; c++)
        {
            for (int j = 0; j < 28; j++)
            {
                for (int k = 0; k < 28; k++)
                {
                    unsigned char pixel;
                    fread(&pixel, 1, 1, file);
                    images[i][c][j][k] = pixel / 255.0;
                }
            }
        }
    }

    fclose(file);

    return images;
}

void destroyImages(float ****images, int numImages, int numChannels)
{
    for (int i = 0; i < numImages; i++)
    {
        for (int c = 0; c < numChannels; c++)
        {
            for (int j = 0; j < 28; j++)
            {
                free(images[i][c][j]);
            }
            free(images[i][c]);
        }
        free(images[i]);
    }
    free(images);
}

int *loadLabels(const char *filename, int numLabels)
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL)
    {
        printf("Error: Unable to open file %s\n", filename);
        exit(1);
    }

    // Skip the magic number and the size header
    fseek(file, 8, SEEK_SET);

    int *labels = malloc(numLabels * sizeof(*labels));
    for (int i = 0; i < numLabels; i++)
    {
        unsigned char label;
        fread(&label, 1, 1, file);
        labels[i] = (int)label;
    }

    fclose(file);
    return labels;
}

void read_float_1d_params(hid_t file_id, const char *datasetname, float **data, int *size)
{
    hid_t dataset_id;
    hid_t space_id;
    herr_t status;
    hsize_t dims_out[1];

    dataset_id = H5Dopen2(file_id, datasetname, H5P_DEFAULT);
    space_id = H5Dget_space(dataset_id);
    H5Sget_simple_extent_dims(space_id, dims_out, NULL);

    *size = (int)dims_out[0];
    *data = (float *)malloc(sizeof(float) * (*size));
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, *data);

    status = H5Sclose(space_id);
    status = H5Dclose(dataset_id);
}

void read_float_2d_params(hid_t file_id, const char *datasetname, float ***data, int *dim1, int *dim2)
{
    hid_t dataset_id;
    hid_t space_id;
    herr_t status;
    hsize_t dims_out[2];
    int i;

    dataset_id = H5Dopen2(file_id, datasetname, H5P_DEFAULT);
    space_id = H5Dget_space(dataset_id);
    H5Sget_simple_extent_dims(space_id, dims_out, NULL);

    *dim1 = (int)dims_out[0];
    *dim2 = (int)dims_out[1];
    float *temp_data = (float *)malloc(sizeof(float) * (*dim1) * (*dim2));
    *data = (float **)malloc(sizeof(float *) * (*dim1));
    for (i = 0; i < *dim1; i++)
    {
        (*data)[i] = &(temp_data[i * (*dim2)]);
    }

    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_data);

    status = H5Sclose(space_id);
    status = H5Dclose(dataset_id);
}

void read_float_4d_params(hid_t file_id, const char *datasetname, float *****data, int *dim1, int *dim2, int *dim3, int *dim4)
{
    hid_t dataset_id;
    hid_t space_id;
    herr_t status;
    hsize_t dims_out[4];
    int i, j, k;

    dataset_id = H5Dopen2(file_id, datasetname, H5P_DEFAULT);
    space_id = H5Dget_space(dataset_id);
    H5Sget_simple_extent_dims(space_id, dims_out, NULL);

    *dim1 = (int)dims_out[0];
    *dim2 = (int)dims_out[1];
    *dim3 = (int)dims_out[2];
    *dim4 = (int)dims_out[3];
    float *temp_data = (float *)malloc(sizeof(float) * (*dim1) * (*dim2) * (*dim3) * (*dim4));
    *data = (float ****)malloc(sizeof(float ****) * (*dim1));
    for (i = 0; i < *dim1; i++)
    {
        (*data)[i] = (float ***)malloc(sizeof(float ***) * (*dim2));
        for (j = 0; j < *dim2; j++)
        {
            (*data)[i][j] = (float **)malloc(sizeof(float **) * (*dim3));
            for (k = 0; k < *dim3; k++)
            {
                (*data)[i][j][k] = &(temp_data[i * (*dim2) * (*dim3) * (*dim4) + j * (*dim3) * (*dim4) + k * (*dim4)]);
            }
        }
    }

    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp_data);

    status = H5Sclose(space_id);
    status = H5Dclose(dataset_id);
}

void cleanup_float_1d(float *data)
{
    free(data);
}

void cleanup_float_2d(float **data, int dim1)
{
    free(data[0]);
    free(data);
}

void cleanup_float_4d(float ****data, int dim1, int dim2, int dim3)
{
    // First, free temp_data
    free(data[0][0][0]);

    for (int i = 0; i < dim1; i++)
    {
        for (int j = 0; j < dim2; j++)
        {
            for (int k = 0; k < dim3; k++)
            {
                // You don't need to free data[i][j][k] here, because it points to an address in temp_data
                // free(data[i][j][k]);
            }
            // Free the third dimension
            free(data[i][j]);
        }
        // Free the second dimension
        free(data[i]);
    }
    // Free the first dimension
    free(data);
}
