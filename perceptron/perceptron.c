/*======================================================================
 *  perceptron.c  ―  Perceptron (Supervized learning)
 *
 *  Author      :  ItzKarizma  <https://github.com/ItzKarizma>
 *  Created     :  03 Jul 2025
 *  Last update :  01 Sep 2025
 *
 *  Build       :  gcc -lm perceptron.c -o perceptron
 *  Usage       :  ./perceptron
 *
 *  Description :
 *      A simple implementation of the Perceptron algorithm for binary
 *      classification using supervised learning.
 *
 *  Todo / notes:
 *      WARNING, May overfit to your expectations.
 *      Jokes aside, there are a lot of things I could improve on
 *      but it works so that's something.
 *
 *  License     :  Custom MIT  (see LICENSE.md)
 *====================================================================*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <stdbool.h>

#define MAX_PATH 256

typedef struct
{
    double *features; // Array of input features (x1, ..., xn)
    int label; // The correct guess (supervised learning)
} Sample;


typedef struct
{
    Sample *samples;
    size_t numberOfSamples;
}
Dataset;

typedef struct
{
    double *weights;
    double bias;
    double learningRate;
    size_t numberOfFeatures; // All samples have the same number of features (== number of weights too)
}
Perceptron;

/**
 * @brief Reads data from a specific file.
 *
 * @param file_name The name of the file to read.
 * @param numberOfFeatures The number of features.
 *
 * @returns A pointer to a `Dataset` object containing the samples and the number of samples.
 */
Dataset *read_data_from_file(char *fileName, size_t numberOfFeatures, bool isLabeled)
{
    FILE *dataFile = fopen(fileName, "r"); // Does NOT throw, returns NULL upon failure
    if (dataFile == NULL) // File doesn't exist or maybe insufficient permissions to read ?
    {
        fprintf(stderr, "[read_data_from_file] Error opening file `%s`: %s\n", fileName, strerror(errno));
        exit(1);
    }

    Dataset *dataset = calloc(1, sizeof(Dataset));
    if (dataset == NULL)
    {
        perror("[read_data_from_file] Failed to allocate memory for dataset");
        exit(1);
    }

    double *tempFeaturesBuffer = malloc(numberOfFeatures * sizeof(double));
    const size_t maxTokenLength = 100;
    char *tokenBuffer = malloc(maxTokenLength);

    if (tempFeaturesBuffer == NULL || tokenBuffer == NULL)
    {
        perror("[read_data_from_file] Failed to allocate memory for tempFeaturesBuffer or tokenBuffer");
        exit(1);
    }
    tokenBuffer[0] = '\0';
    size_t tokenBufferSize = 0;

    char ch;
    size_t tokensRead = 0;

    while((ch = fgetc(dataFile)) != EOF) // Keep reading the next character until end of file
    {
        if (ch == '\n' || ch == ' ' || tokenBufferSize >= maxTokenLength - 1) // 1 space for null-terminator
        {
            storeLastLine:

            if (tokensRead < numberOfFeatures)
            {
                tempFeaturesBuffer[tokensRead] = strtod(tokenBuffer, NULL);
                ++tokensRead;
            }

            if (ch == '\n' || ch == EOF) // Check for end of line (or EOF if came from goto)
            {
                dataset->samples = realloc(dataset->samples, (dataset->numberOfSamples + 1) * sizeof(Sample)); // realloc can act as malloc too in case of initialization
                if (dataset->samples == NULL) // check
                {
                    perror("[read_data_from_file] Failed reallocating memory for dataset->samples");
                    exit(1);
                }

                dataset->samples[dataset->numberOfSamples].features = malloc(numberOfFeatures * sizeof(double));
                if (dataset->samples[dataset->numberOfSamples].features == NULL)
                {
                    perror("[read_data_from_file] Failed allocating memory for dataset->samples[dataset->numberOfSamples].features");
                    exit(1);
                }
                memcpy(dataset->samples[dataset->numberOfSamples].features, tempFeaturesBuffer, numberOfFeatures * sizeof(double));

                if (isLabeled == true)
                {
                    dataset->samples[dataset->numberOfSamples].label = atoi(tokenBuffer);
                }
                else
                {
                    dataset->samples[dataset->numberOfSamples].label = -1;
                }

                ++dataset->numberOfSamples;
                tokensRead = 0;
            }
            tokenBuffer[0] = '\0';
            tokenBufferSize = 0;
        }
        else // We're still in the same token
        {
            tokenBuffer[tokenBufferSize] = ch;
            ++tokenBufferSize;
            tokenBuffer[tokenBufferSize] = '\0';
        }
    }
    // This makes sure to store the last line of data
    // if EOF was encountered instead of a newline
    if (tokenBufferSize > 0)
    {
        goto storeLastLine;
    }
    // Prevent memory leaks
    free(tempFeaturesBuffer);
    free(tokenBuffer);
    fclose(dataFile);

    return dataset;
}

/**
 * @brief Saves the model's parameters to a file.
 *
 * @attention Overwrites the file. Old file data is lost!
 *
 * @param fileName The name of the file to write to.
 * @param perceptron A pointer to the `Perceptron` model to save.
 *
 * @note Saves the number of features/weights, the weights, bias, and learning rate.
 */
void save_model(const char *fileName, const Perceptron *perceptron)
{
    FILE *file = fopen(fileName, "w");
    if (file == NULL)
    {
        fprintf(stderr, "[save_model] Failed to open/create file '%s': %s", fileName, strerror(errno));
        exit(1);
    }

    fprintf(file, "%zu\n", perceptron->numberOfFeatures);
    for (size_t iWeight = 0; iWeight < perceptron->numberOfFeatures; ++iWeight)
    {
        fprintf(file, "%lf\n", perceptron->weights[iWeight]);
    }
    fprintf(file, "%lf\n%lf\n", perceptron->bias, perceptron->learningRate);
    fclose(file);
}

/**
 * @brief Loads a model's parameters from a file.
 *
 * @param fileName The model's path (e.g.: models/model.txt).
 *
 * @returns A pointer to the initialized perceptron object.
 */
Perceptron *load_model(const char *fileName)
{
    FILE *file = fopen(fileName, "r");
    if (file == NULL)
    {
        fprintf(stderr, "[load_model] Failed to open/read file '%s': %s", fileName, strerror(errno));
        exit(1);
    }

    Perceptron *perceptron = calloc(1, sizeof(Perceptron));
    if (perceptron == NULL)
    {
        perror("[load_model] Failed to allocate memory for perceptron");
        exit(1);
    }

    // Number of inputs
    if (fscanf(file, "%zu", &perceptron->numberOfFeatures) != 1)
    {
        fprintf(stderr, "[load_model] Failed to read data from file (numberOfInputs)");
        exit(1);
    }

    // Weights
    perceptron->weights = malloc(perceptron->numberOfFeatures * sizeof(double));
    if (perceptron->weights == NULL)
    {
        perror("[load_model] Failed to allocate memory for perceptron->weights");
        exit(1);
    }
    for (size_t iWeight = 0; iWeight < perceptron->numberOfFeatures; ++iWeight)
    {
        if (fscanf(file, "%lf", &perceptron->weights[iWeight]) != 1)
        {
            fprintf(stderr, "[load_model] Failed to read data from file (weight %zu)", iWeight);
            exit(1);
        }
    }

    // Bias
    if (fscanf(file, "%lf", &perceptron->bias) != 1)
    {
        fprintf(stderr, "[load_model] Failed to read data from file (bias)");
        exit(1);
    }

    // Learning rate
    if (fscanf(file, "%lf", &perceptron->learningRate) != 1)
    {
        fprintf(stderr, "[load_model] Failed to read data from file (learningRate)");
        exit(1);
    }

    fclose(file);
    return perceptron;
}

/**
 * @brief Frees a `Dataset` object.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 */
void free_dataset(Dataset *dataset)
{
    for (size_t iSample = 0; iSample < dataset->numberOfSamples; ++iSample)
    {
        free(dataset->samples[iSample].features);
    }
    free(dataset->samples);
    free(dataset);
}

/**
 * @brief Makes a deep copy of a `Dataset` object.
 *
 * @param dataset A pointer to the `Dataset` object to copy.
 * @param numberOfFeatures The number of features.
 *
 * @returns A pointer to a newly allocated and copied `Dataset` object.
 */
Dataset *deep_copy_dataset(const Dataset *dataset, size_t numberOfFeatures)
{
    Dataset *copy = malloc(sizeof(Dataset));
    if (copy == NULL)
    {
        perror("[deep_copy_dataset] Failed to allocate memory for copy");
        exit(1);
    }

    copy->samples = malloc(dataset->numberOfSamples * sizeof(Sample));
    if (copy->samples == NULL)
    {
        perror("[deep_copy_dataset] Failed to allocate memory for copy->samples");
        exit(1);
    }
    copy->numberOfSamples = dataset->numberOfSamples;

    for (size_t iSample = 0; iSample < dataset->numberOfSamples; ++iSample)
    {
        copy->samples[iSample].features = malloc(numberOfFeatures * sizeof(double));
        if (copy->samples[iSample].features == NULL)
        {
            perror("[deep_copy_dataset] Failed to allocate memory for copy->samples[iSample].features");
            exit(1);
        }
        copy->samples[iSample].label = dataset->samples[iSample].label;

        for (size_t iFeature = 0; iFeature < numberOfFeatures; ++iFeature)
        {
            copy->samples[iSample].features[iFeature] = dataset->samples[iSample].features[iFeature];
        }
    }
    return copy;
}

/**
 * @brief Updates weights and bias in-place.
 *
 * @param perceptron Pointer to the perceptron whose weights/bias will be updated.
 * @param sample Pointer to the training sample that triggered the update.
 * @param error Signed error term: (label − prediction). Values are −1, 0, or +1.
 */
void update_parameters(Perceptron *perceptron, const Sample *sample, int error)
{
    // Update weights
    for (size_t i = 0; i < perceptron->numberOfFeatures; ++i)
    {
        perceptron->weights[i] += perceptron->learningRate * error * sample->features[i];
    }
    // Update bias
    perceptron->bias += perceptron->learningRate * error;
}

/**
 * @brief Computes the perceptron’s output for one sample.
 *
 * @param perceptron Pointer to the trained (or partially trained) model.
 * @param sample Pointer to the input whose class you want to predict.
 *
 * @returns `1` if output is bigger or equal to 0, otherwise `0`.
 */
int compute_prediction(const Perceptron *perceptron, const Sample *sample)
{
    double output = perceptron->bias;
    for (size_t i = 0; i < perceptron->numberOfFeatures; ++i)
    {
        output += perceptron->weights[i] * sample->features[i];
    }
    return (output >= 0) ? 1 : 0;
}

/**
 * @brief Creates a perceptron sample out of user's input.
 *
 * @param numberOfFeatures The number of features.
 *
 * @returns A pointer to a `Sample` object.
 */
Sample *get_sample(size_t numberOfFeatures)
{
    Sample *sample = malloc(sizeof(Sample));
    if (sample == NULL)
    {
        perror("[get_sample] Failed to allocate memory for sample");
        exit(1);
    }

    sample->features = malloc(numberOfFeatures * sizeof(double));
    if (sample->features == NULL)
    {
        perror("[get_sample] Failed to allocate memory for sample->features");
        exit(1);
    }

    for (size_t iFeature = 0; iFeature < numberOfFeatures; ++iFeature)
    {
        printf("Feature %zu: ", iFeature);
        scanf("%lf", &sample->features[iFeature]);
    }

    printf("Label: ");
    scanf("%d", &sample->label);
    return sample;
}

/**
 * @brief Calculates the mean for each feature.
 *
 * @attention Assumes dataset->numberOfSamples is greater than 0.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param numberOfFeatures The number of features.
 *
 * @returns An array containing the mean value of each feature.
 */
double *calculate_mean(const Dataset *dataset, size_t numberOfFeatures)
{
    // Mean value for each feature (initialized with 0s)
    double *mean = calloc(numberOfFeatures, sizeof(double));
    if (mean == NULL) // check
    {
        perror("[calculate_mean] Failed allocating memory for `mean`.");
        exit(1);
    }

    // Loop through the points and sum the features respectively
    for (size_t iSample = 0; iSample < dataset->numberOfSamples; ++iSample)
    {
        for (size_t iFeature = 0; iFeature < numberOfFeatures; ++iFeature)
        {
            mean[iFeature] += dataset->samples[iSample].features[iFeature];
        }
    }

    // Finally, divide each value by the number of samples
    for (size_t iFeature = 0; iFeature < numberOfFeatures; ++iFeature)
    {
        mean[iFeature] /= dataset->numberOfSamples;
    }

    return mean;
}

/**
 * @brief Calculates the standard deviation for each feature.
 *
 * @attention Assumes `dataset->numberOfSamples` is greater than 0.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param mean An array containing the standard deviation of each feature.
 * @param numberOfFeatures The number of features.
 *
 * @returns An array containing the standard deviation of every feature.
 */
double *calculate_std_deviation(const Dataset *dataset, const double *mean, size_t numberOfFeatures)
{
    // Standard deviation value for each feature (initialized with 0s)
    double *standardDeviation = calloc(numberOfFeatures, sizeof(double));
    if (standardDeviation == NULL) // check
    {
        perror("[calculate_std_deviation] Failed allocating memory for standardDeviation");
        exit(1);
    }

    // Loop through the points and sum all the squared differences (feature - mean)^2
    for (size_t iSample = 0; iSample < dataset->numberOfSamples; ++iSample)
    {
        for (size_t iFeature = 0; iFeature < numberOfFeatures; ++iFeature)
        {
            double diff = dataset->samples[iSample].features[iFeature] - mean[iFeature];
            standardDeviation[iFeature] += diff * diff;
            // PS: pow() would've been slower :P
        }
    }

    // Finally, divide each value by the number of points to get the variance
    // and then use `sqrt` to get the standard deviation
    for (size_t iFeature = 0; iFeature < numberOfFeatures; ++iFeature)
    {
        standardDeviation[iFeature] = sqrt(standardDeviation[iFeature] / dataset->numberOfSamples);
    }

    return standardDeviation;
}

/**
 * @brief Standardizes the features of the whole dataset.
 *
 * x(standardized) = (x − μ) / σ
 *
 * @attention Modifies all sample features in-place
 *
 * instead of returning a new `Sample *` with the modified values.
 *
 * Exits with an error message if a division by 0 occurs.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param numberOfFeatures The number of features.
 */
void standardize_data(Dataset *dataset, size_t numberOfFeatures)
{
    // Check the number of samples because there's a division with it as denominator
    if (dataset->numberOfSamples == 0)
    {
        perror("[standardize_data] Failed to standardize data. \
            No data was found ? Avoided division by 0");
        exit(133);
    }

    double *mean = calculate_mean(dataset, numberOfFeatures);
    double *standardDeviation = calculate_std_deviation(dataset, mean, numberOfFeatures);

    for (size_t iSample = 0; iSample < dataset->numberOfSamples; ++iSample)
    {
        for (size_t iFeature = 0; iFeature < numberOfFeatures; ++iFeature)
        {
            if (standardDeviation[iFeature] == 0) // Again, avoid dividing by 0...
            {
                fprintf(stderr, "[standardize_data] Feature %zu has 0 for standard deviation. Avoided division by 0\n", iFeature);
                exit(133);
            }
            // x_standardized = (x - mu) / standard_deviation
            double x = dataset->samples[iSample].features[iFeature];
            double mu = mean[iFeature];
            double standard_deviation = standardDeviation[iFeature];

            // x_standardized
            dataset->samples[iSample].features[iFeature] = (x - mu) / standard_deviation;
        }
    }

    // Avoid memory leaking if this function is called multiple times
    free(mean);
    free(standardDeviation);
}

/**
 * @brief Initializes the perceptron's weights with random values.
 *
 * @param perceptron Pointer to the `Perceptron` model whose weights will be initialized with random values.
 *
 * @note Weights will be random but centered around 0.
 */
void init_random_weights(Perceptron *perceptron)
{
    perceptron->weights = malloc(perceptron->numberOfFeatures * sizeof(double));
    if (perceptron->weights == NULL)
    {
        perror("[init_weights] Failed to allocate memory for perceptron->weights");
        exit(1);
    }

    for (size_t iWeight = 0; iWeight < perceptron->numberOfFeatures; ++iWeight)
    {
        perceptron->weights[iWeight] = ((double)rand() / RAND_MAX) * 2 - 1; // random, centered around 0
    }
}

int main(int argc, char *argv[])
{
    srand((unsigned int)time(NULL));
    Perceptron *perceptron;

    // Some input variables that will be used for user input...
    unsigned int inputINT = 0;
    double inputDBL = 0.0;
    char path[MAX_PATH] = "";

    // Load a model if necessary
    printf("Would you like to load a model (0 -> no, 1 -> yes)? ");
    scanf("%u", &inputINT);

    if (inputINT == 1)
    {
        printf("Model path: ");
        scanf("%255s", path);
        perceptron = load_model(path);
        printf("\nModel loaded successfully (learning rate: %lf)\n\n", perceptron->learningRate);
    }
    else
    {
        perceptron = malloc(sizeof(Perceptron));
        if (perceptron == NULL)
        {
            perror("[main] Failed to allocate memory for perceptron");
            exit(1);
        }

        printf("\nNumber of features (label not included): ");
        scanf("%zu", &perceptron->numberOfFeatures);
        if (perceptron->numberOfFeatures == 0)
        {
            fprintf(stderr, "[main] The number of features cannot be equal to 0\n");
            exit(1);
        }
        init_random_weights(perceptron);

        printf("Set a bias value (default 1.0): ");
        scanf("%lf", &inputDBL);
        perceptron->bias = inputDBL;
    }

    printf("Would you like to standardize the values (0 -> no; 1 -> yes)? ");
    int standardizeValues; scanf("%d", &standardizeValues);

    // Training
    printf("\nTrain the model on a data set (0 -> no, 1 -> yes)? ");
    scanf("%u", &inputINT);

    if (inputINT == 1)
    {
        printf("Set a learning rate value (default 0.1): ");
        scanf("%lf", &inputDBL);
        perceptron->learningRate = inputDBL;

        printf("Data file's path: ");
        scanf("%255s", path);

        Dataset *trainDataset = read_data_from_file(path, perceptron->numberOfFeatures, true);
        if (standardizeValues == 1) standardize_data(trainDataset, perceptron->numberOfFeatures);

        printf("Set the number of epochs (default 100): ");
        scanf("%u", &inputINT);

        size_t epochs = (size_t)inputINT;
        for (size_t epoch = 0; epoch < epochs; ++epoch)
        {
            for (size_t iSample = 0; iSample < trainDataset->numberOfSamples; ++iSample)
            {
                Sample *sample = &trainDataset->samples[iSample];
                update_parameters(
                    perceptron,
                    sample,
                    sample->label - compute_prediction(perceptron, sample) // error value
                );
            }
        }
        printf("\nModel trained succesfully.\n");
        free_dataset(trainDataset);
    }

    // Inference

    // Labeled
    printf("\nTest the accuracy of the model on a 'labeled' data set (0 -> no, 1 -> yes)? ");
    scanf("%u", &inputINT);

    if (inputINT == 1)
    {
        printf("Data file's path: ");
        scanf("%255s", path);

        Dataset *testDataset = read_data_from_file(path, perceptron->numberOfFeatures, true);
        if (standardizeValues == 1) standardize_data(testDataset, perceptron->numberOfFeatures);

        size_t correctGuesses = 0;
        for (size_t iSample = 0; iSample < testDataset->numberOfSamples; ++iSample)
        {
            Sample *sample = &testDataset->samples[iSample];
            int output = compute_prediction(perceptron, sample);
            printf("Guessed %s\n", (output == sample->label) ? "correctly" : "incorrectly");
            correctGuesses += (output == sample->label) ? 1 : 0;
        }
        printf("\nModel accuracy: %.2f%%\n", (float)correctGuesses / testDataset->numberOfSamples * 100);
        free_dataset(testDataset);
    }

    // Unlabeled
    printf("\nRun the model on an 'unlabeled' data set (0 -> no, 1 -> yes)? ");
    scanf("%u", &inputINT);

    if (inputINT == 1)
    {
        printf("Data file's path: ");
        scanf("%255s", path);

        Dataset *unlabeledDataset = read_data_from_file(path, perceptron->numberOfFeatures, false);
        Dataset *unlabeledDatasetCopy = deep_copy_dataset(unlabeledDataset, perceptron->numberOfFeatures);
        if (standardizeValues == 1) standardize_data(unlabeledDataset, perceptron->numberOfFeatures);

        for (size_t iSample = 0; iSample < unlabeledDataset->numberOfSamples; ++iSample)
        {
            Sample *sample = &unlabeledDataset->samples[iSample];
            sample->label = compute_prediction(perceptron, sample); // it's unlabeled so prediction is the only label
            printf("Guessed %d\n", sample->label);
        }

        printf("\n\nWould you like to save the results to a file (0 -> no; 1 -> yes)? ");
        scanf("%d", &inputINT);

        if (inputINT == 1) // Save
        {
            printf("File path (directory must exist): ");
            scanf("%255s", path);
            FILE* file = fopen(path, "w");

            for (size_t iSample = 0; iSample < unlabeledDataset->numberOfSamples; ++iSample)
            {
                for (size_t iFeature = 0; iFeature < perceptron->numberOfFeatures; ++iFeature)
                {
                    fprintf(file, "%lf ", unlabeledDatasetCopy->samples[iSample].features[iFeature]); // Use copy to get the non-standardized features
                }
                fprintf(file, "%d\n", unlabeledDataset->samples[iSample].label);
            }
            fclose(file);
            printf("\nResults saved to file.\n");
        }
        else
        {
            printf("\nResults were not saved.\n");
        }

        free_dataset(unlabeledDataset);
        free_dataset(unlabeledDatasetCopy);
    }

    // Save the model if necessary
    printf("\nSave the model (0 -> no, 1 -> yes)? ");
    scanf("%u", &inputINT);

    if (inputINT == 1)
    {
        printf("Model path (directory must exist): ");
        scanf("%255s", path);
        save_model(path, perceptron);
        printf("\nModel saved succesfully.\n");
    }

    free(perceptron->weights);
    free(perceptron);
    return 0;
}
