/*======================================================================
 *  knn.c  ―  Simple k-Nearest Neighbours (lazy learner; supervized)
 *
 *  Author      :  ItzKarizma  <https://github.com/ItzKarizma>
 *  Created     :  25 Jun 2025
 *
 *  Build       :  gcc -lm knn.c -o knn
 *  Usage       :  ./knn
 *
 *  Description :
 *      Reads N-dimensional samples from a text file (final token = label),
 *      standardises the data in-place, returns the K nearest
 *      neighbours for a query sample and outputs what the sample
 *      was classified as.
 *
 *  License     :  Custom MIT  (see LICENSE.md)
 *====================================================================*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <stdbool.h>

#define MAX_PATH 256

typedef struct
{
    double *features; // A pointer containing double values to keep the number of features dynamic
    char *label; // The label of the sample (what it's classified as)
}
Sample;

// Keep the array of samples and its size together
typedef struct
{
    Sample *samples;
    size_t numberOfSamples;
}
Dataset;

// Facilitates the counting of different labels
// and their respective frequencies
typedef struct
{
    char **labels;
    size_t *frequencies;
    size_t numberOfLabels;
}
LabelStats;

/**
 * @brief Reads data from a specific file.
 *
 * My hard working char-by-char method.
 * Even God is confused how this function works.
 *
 * @attention It assumes that if the number of features are N,
 *
 * there are N+1 separate values (label included) per line.
 *
 * For example, for numberOfFeatures = 2, a line would be "feature1 feature2 label".
 *
 * @param fileName The name of the file to read.
 * @param numberOfFeatures The number of features.
 *
 * @returns A pointer to a `Dataset` object containing the samples and its size.
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
                    dataset->samples[dataset->numberOfSamples].label = strdup(tokenBuffer);
                }
                else
                {
                    dataset->samples[dataset->numberOfSamples].label = NULL;
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
 * @brief Frees a `Dataset` object.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 */
void free_dataset(Dataset *dataset)
{
    for (size_t iSample = 0; iSample < dataset->numberOfSamples; ++iSample)
    {
        free(dataset->samples[iSample].features);
        if (dataset->samples[iSample].label != NULL) free(dataset->samples[iSample].label);
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
        if (dataset->samples[iSample].label != NULL) copy->samples[iSample].label = strdup(dataset->samples[iSample].label);
        else copy->samples[iSample].label = NULL;

        for (size_t iFeature = 0; iFeature < numberOfFeatures; ++iFeature)
        {
            copy->samples[iSample].features[iFeature] = dataset->samples[iSample].features[iFeature];
        }
    }
    return copy;
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
        perror("[calculate_mean] Failed to allocate memory for mean");
        exit(1);
    }

    // Loop through the samples and sum the features respectively
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
 * @attention Assumes dataset->numberOfSamples is greater than 0.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param mean An array containing the mean value of each feature.
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
        perror("[calculate_std_deviation] Failed to allocate memory for standardDeviation");
        exit(1);
    }

    // Loop through the samples and sum all the squared differences (feature - mean)^2
    for (size_t iSample = 0; iSample < dataset->numberOfSamples; ++iSample)
    {
        for (size_t iFeature = 0; iFeature < numberOfFeatures; ++iFeature)
        {
            double diff = dataset->samples[iSample].features[iFeature] - mean[iFeature];
            standardDeviation[iFeature] += diff * diff;
            // pow() would've been slower :P
        }
    }

    // Finally, divide each value by the number of samples to get the variance
    // and then use `sqrt` to get the standard deviation
    for (size_t iFeature = 0; iFeature < numberOfFeatures; ++iFeature)
    {
        standardDeviation[iFeature] = sqrt(standardDeviation[iFeature] / dataset->numberOfSamples);
    }

    return standardDeviation;
}

/**
 * @brief Standardizes the features of the whole dataset and those of the new sample.
 *
 * x(standardized​) = (x − μ)​ / σ
 *
 * @attention Modifies all sample features in-place
 *
 * instead of returning a new `Sample` with the modified values.
 *
 * Exits with an error message if a division by 0 occurs.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param numberOfFeatures The number of features.
 */
void standardize_data(Dataset *dataset, const double *mean, const double *standardDeviation, size_t numberOfFeatures)
{
    // Check size because there's a division with size as denominator
    if (dataset->numberOfSamples == 0)
    {
        perror("[standardize_data] Failed to standardize data. No data was found ? Avoided division by 0");
        exit(133);
    }

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
}

/**
 * @brief Calculates the Euclidian distance between two samples.
 *
 * @param sample1 The first sample.
 * @param sample2 The second sample.
 * @param numberOfFeatures The number of features.
 *
 * @returns The Euclidian distance between `sample1` and `sample2`.
 */
double calculate_euclidian_distance(const Sample *sample1, const Sample *sample2, size_t numberOfFeatures)
{
    // Calculate the euclidian distance
    double euclidianDistance = 0;

    for (size_t iFeature = 0; iFeature < numberOfFeatures; ++iFeature)
    {
        double diff = sample1->features[iFeature] - sample2->features[iFeature];
        euclidianDistance += diff * diff;
    }
    return sqrt(euclidianDistance);
}

/**
 * @brief Finds the farthest sample in an array of samples.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param sample The main sample.
 * @param numberOfFeatures The number of features.
 *
 * @returns An array containing the index and euclidian distance of the farthest sample in `dataset->samples`.
 */
double *find_farthest_sample(const Dataset *dataset, const Sample *sample, size_t numberOfFeatures)
{
    // The index and euclidian distance of the farthest sample in the samples array
    // Initialized with 0s
    double *farthestSample = calloc(2, sizeof(double));

    for (size_t iSample = 0; iSample < dataset->numberOfSamples; ++iSample)
    {
        // Euclidean distance
        double euclidianDistance = calculate_euclidian_distance(sample, &dataset->samples[iSample], numberOfFeatures);

        // Store the sample's index and distance if necessary
        if (euclidianDistance > farthestSample[1])
        {
            farthestSample[0] = (double)iSample;
            farthestSample[1] = euclidianDistance;
        }
    }
    return farthestSample;
}

/**
 * @brief Returns the K nearest samples (neighbors).
 *
 * @attention Assumes `K` is smaller or equal to the total number of samples.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param sample The main sample.
 * @param numberOfFeatures The number of features.
 * @param K The number of the neighboring (nearest) samples to check for.
 *
 * @returns `Dataset *neighbors`, all the nearest neighbors of the `sample`.
 */
Dataset *get_K_nearest_neighbors(const Dataset *dataset, const Sample *sample, size_t numberOfFeatures, size_t K)
{
    Dataset *neighbors = malloc(sizeof(Dataset));
    if (neighbors == NULL)
    {
        perror("[get_K_nearest_neighbors] Failed to allocate memory for neighbors");
        exit(1);
    }

    neighbors->samples = malloc(K * sizeof(Sample));
    if (neighbors->samples == NULL)
    {
        perror("[get_K_nearest_neighbors] Failed to allocate memory for neighbors->samples");
        exit(1);
    }
    neighbors->numberOfSamples = 0;

    for (size_t iSample = 0; iSample < dataset->numberOfSamples; ++iSample)
    {
        if (neighbors->numberOfSamples >= K)
        {
            double *farthestSample = find_farthest_sample(neighbors, sample, numberOfFeatures);
            size_t iFarthestSample = (size_t)farthestSample[0];
            double farthestSampleDistance = farthestSample[1];

            if (calculate_euclidian_distance(sample, &dataset->samples[iSample], numberOfFeatures) < farthestSampleDistance)
            {
                free(neighbors->samples[iFarthestSample].label);
                neighbors->samples[iFarthestSample].label = strdup(dataset->samples[iSample].label);

                for (size_t iFeature = 0; iFeature < numberOfFeatures; ++iFeature)
                {
                    neighbors->samples[iFarthestSample].features[iFeature] = dataset->samples[iSample].features[iFeature];
                }
            }
            free(farthestSample);
        }
        else
        {
            neighbors->samples[neighbors->numberOfSamples].features = malloc(numberOfFeatures * sizeof(double));
            if (neighbors->samples[neighbors->numberOfSamples].features == NULL)
            {
                perror("[get_K_nearest_neighbors] Failed to allocate memory for neighbors->samples[neighbors->numberOfSamples].features");
                exit(1);
            }
            neighbors->samples[neighbors->numberOfSamples].label = strdup(dataset->samples[iSample].label);

            for (size_t iFeature = 0; iFeature < numberOfFeatures; ++iFeature)
            {
                neighbors->samples[neighbors->numberOfSamples].features[iFeature] = dataset->samples[iSample].features[iFeature];
            }
            ++neighbors->numberOfSamples;
        }
    }
    return neighbors;
}

/**
 * @brief Counts the number of different data labels.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 *
 * @returns A `LabelStats` object containing all the different labels and their frequencies.
 */
LabelStats get_label_stats(const Dataset *dataset)
{
    char **labels = NULL;
    size_t *labelFrequencies = NULL;
    size_t labelsSize = 0;

    for (int iSample = 0; iSample < dataset->numberOfSamples; ++iSample)
    {
        // Find if the label already exists & increment that label's counter
        size_t iDataType = 0;
        while (iDataType < labelsSize && strcmp(dataset->samples[iSample].label, labels[iDataType]) != 0)
        {
            ++iDataType;
        }

        if (iDataType < labelsSize)
        {
            ++labelFrequencies[iDataType];
        }
        else // Increase the size of the array if needed & store the label
        {
            // Apparently, `realloc(NULL, size) si the same as malloc(size)` :)
            labels = realloc(labels, (labelsSize + 1) * sizeof(char*));
            labelFrequencies = realloc(labelFrequencies, (labelsSize + 1) * sizeof(size_t));

            if (labels != NULL && labelFrequencies != NULL)
            {
                labels[labelsSize] = strdup(dataset->samples[iSample].label); // Copy the sample's label
                labelFrequencies[labelsSize] = 1; // Equals 1 instead of incrementing because it's a NEW data label, with 1 sample
                ++labelsSize;
            }
            else
            {
                fprintf(stderr, "[count_data_types()] NULL pointer(s) after memory re-allocation at iSample %d. %s\n", iSample, strerror(errno));
                exit(1);
            }
        }
    }
    return (LabelStats){
        .labels = labels,
        .frequencies = labelFrequencies,
        .numberOfLabels = labelsSize
    };
}

/**
 * @brief Classifies a sample based on the highest label frequencies.
 *
 * @attention K should be an odd number because the classification
 * is a majority vote ('even' numbers do not work well).
 *
 * @param neighbors A pointer to a `Dataset` object containing the array of K nearest samples and its count.
 *
 * @returns The predicted label for the new sample.
 */
char *classify_data(const Dataset *neighbors)
{
    LabelStats stats = get_label_stats(neighbors);
    size_t majorityVote = 0;
    size_t winnerIndex = -1;

    for (size_t iLabel = 0; iLabel < stats.numberOfLabels; ++iLabel)
    {
        if (stats.frequencies[iLabel] > majorityVote)
        {
            majorityVote = stats.frequencies[iLabel];
            winnerIndex = iLabel;
        }
    }

    char *predictedLabel = NULL;
    if (winnerIndex == -1)
    {
        predictedLabel = strdup("Unlabeled");
    }
    else
    {
        predictedLabel = strdup(stats.labels[winnerIndex]);
    }

    for (size_t iLabel = 0; iLabel < stats.numberOfLabels; ++iLabel)
    {
        free(stats.labels[iLabel]);
    }
    free(stats.labels);
    free(stats.frequencies);
    return predictedLabel;
}

int main(int argc, char *argv[])
{
    unsigned int answer = 0; // for questions

    printf("Number of features (label not included): ");
    size_t numberOfFeatures = 0; scanf("%zu", &numberOfFeatures); // The number of features (label is not included in this number)

    // If `K >= samples available`, it's simply a majority vote between all samples...
    printf("K neighbors: ");
    size_t K; scanf("%zu", &K); // K neighbors

    char path[MAX_PATH];

    printf("\nData file's path: ");
    scanf("%255s", path);
    Dataset *dataset = read_data_from_file(path, numberOfFeatures, true);

    double *mean = calculate_mean(dataset, numberOfFeatures); // freed at the end of main function
    double *standardDeviation = calculate_std_deviation(dataset, mean, numberOfFeatures);  // freed at the end of main function

    printf("Would you like to standardize the values (0 -> no; 1 -> yes)? ");
    int standardizeValues; scanf("%d", &standardizeValues);
    if (standardizeValues == 1) standardize_data(dataset, mean, standardDeviation, numberOfFeatures);

    printf("Would you like to do an accuracy test (0 -> no; 1 -> yes)? ");
    scanf("%d", &answer);

    if (answer == 1)
    {
        printf("\nTest data file's path: ");
        scanf("%255s", path);
        Dataset *datasetTest = read_data_from_file(path, numberOfFeatures, true);
        if (standardizeValues == 1) standardize_data(datasetTest, mean, standardDeviation, numberOfFeatures);

        size_t correctPredictions = 0;
        repeatAccuracyTest:
        for (size_t iSample = 0; iSample < datasetTest->numberOfSamples; ++iSample)
        {
            Dataset *neighbors = get_K_nearest_neighbors(dataset, &datasetTest->samples[iSample], numberOfFeatures, K);
            char *predictedLabel = classify_data(neighbors);

            int prediction = (strcmp(datasetTest->samples[iSample].label, predictedLabel) == 0)? 1 : 0;
            correctPredictions += prediction;
            printf("\nSample %zu was labeled %s\n", iSample+1, (prediction == 1)? "CORRECTLY" : "INCORRECTLY");
            free(predictedLabel);
            free_dataset(neighbors);
        }

        printf("\n\nAccuracy with K=%zu: %0.2f%%", K, (float)correctPredictions / datasetTest->numberOfSamples * 100);
        correctPredictions = 0;

        printf("\nUpdate K and retry (0 -> no; 1 -> yes)? ");
        scanf("%d", &answer);
        if (answer == 1)
        {
            printf("K neighbors: ");
            scanf("%zu", &K); // K neighbors
            goto repeatAccuracyTest;
        }

        free_dataset(datasetTest);
    }

    printf("Would you like to run the algorithm on unlabeled data (0 -> no; 1 -> yes)? ");
    scanf("%d", &answer);

    if (answer == 1)
    {
        Dataset *unlabeledDataset;

        printf("\nLoad unlabeled samples from a data file or enter the samples manually (0 -> data file; 1 -> manual input)? ");
        scanf("%d", &answer);

        if (answer == 0)
        {
            printf("\nUnlabeled data file's path: ");
            scanf("%255s", path);
            unlabeledDataset = read_data_from_file(path, numberOfFeatures, false);
        }
        else
        {
            printf("\nNumber of samples: ");
            size_t numberOfSamples; scanf("%zu", &numberOfSamples);

            // Allocate memory for the dataset and the samples

            unlabeledDataset = malloc(numberOfSamples * sizeof(Sample));
            if (unlabeledDataset == NULL)
            {
                perror("[main] Failed to allocate memory for unlabeledDataset");
                exit(1);
            }

            unlabeledDataset->samples = malloc(sizeof(Dataset));
            if (unlabeledDataset->samples == NULL)
            {
                perror("[main] Failed to allocate memory for unlabeledDataset->samples");
                exit(1);
            }

            unlabeledDataset->numberOfSamples = 0;

            // Fill the dataset with unlabeled samples
            for (size_t iSample = 0; iSample < numberOfSamples; ++iSample)
            {
                unlabeledDataset->samples[iSample].label = NULL;
                unlabeledDataset->samples[iSample].features = malloc(numberOfFeatures * sizeof(double));
                if (unlabeledDataset->samples[iSample].features == NULL)
                {
                    perror("[main] Failed to allocate memory for unlabeledDataset->samples[iSample].features");
                    exit(1);
                }

                printf("\n-- Enter the features of the sample %zu --\n", iSample+1);

                size_t iFeature = 0;
                while (iFeature < numberOfFeatures)
                {
                    printf("Feature %zu: ", iFeature+1);
                    scanf("%lf", &unlabeledDataset->samples[iSample].features[iFeature]);
                    ++iFeature;
                }
                ++unlabeledDataset->numberOfSamples;
            }
        }
        Dataset *unlabeledDatasetCopy = deep_copy_dataset(unlabeledDataset, numberOfFeatures); // To save samples with non-standardized features and their respective label, if needed
        if (standardizeValues == 1) standardize_data(unlabeledDataset, mean, standardDeviation, numberOfFeatures);

        for (size_t iSample = 0; iSample < unlabeledDataset->numberOfSamples; ++iSample)
        {
            Dataset *neighbors = get_K_nearest_neighbors(dataset, &unlabeledDataset->samples[iSample], numberOfFeatures, K);
            char *predictedLabel = classify_data(neighbors);

            // If there was a previous label that was dynamically allocated, free it first
            if (unlabeledDataset->samples[iSample].label != NULL)
            {
                free(unlabeledDataset->samples[iSample].label);
            }
            unlabeledDataset->samples[iSample].label = strdup(predictedLabel);

            printf("\nSample %zu was labeled as: %s\n", iSample+1, predictedLabel);
            free(predictedLabel);
            free_dataset(neighbors);
        }

        printf("\n\nWould you like to save the results to a file (0 -> no; 1 -> yes)? ");
        scanf("%d", &answer); // re-use variable answer

        if (answer == 1) // Save
        {
            printf("File path (directory must exist): ");
            scanf("%255s", path);
            FILE *file = fopen(path, "w");

            for (size_t iSample = 0; iSample < unlabeledDataset->numberOfSamples; ++iSample)
            {
                for (size_t iFeature = 0; iFeature < numberOfFeatures; ++iFeature)
                {
                    fprintf(file, "%lf ", unlabeledDatasetCopy->samples[iSample].features[iFeature]); // Use copy to get the non-standardized features
                }
                fprintf(file, "%s\n", unlabeledDataset->samples[iSample].label);
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
    free(mean);
    free(standardDeviation);
    free_dataset(dataset);

    return 0;
}
