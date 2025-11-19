/*======================================================================
 *  kmeans.c  ―  K-Means (Unsupervized learning)
 *
 *  Author      :  ItzKarizma  <https://github.com/ItzKarizma>
 *  Created     :  27 Jun 2025
 *
 *  Build       :  gcc -lm kmeans.c -o kmeans
 *  Usage       :  ./kmeans
 *
 *  Description :
 *      Reads N-dimensional samples from a text file,
 *      partitions a dataset into K distinct, non-overlapping clusters,
 *      where each sample belongs to the cluster with the nearest mean (or "centroid").
 *      It returns the best K value based on the elbow method (if a specific K value is undesired).
 *
 *  License     :  Custom MIT  (see LICENSE.md)
 *====================================================================*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <errno.h>

#define MAX_PATH 256

/* Global variables (todo: remove from global and simply pass to functions when needed...) */
int globalSaveOutput = 0;
int globalStandardize = 0;
int globalMinFeature = 0;
int globalMaxFeature = 1;
/* End of global variables */

typedef struct Sample Sample;

struct Sample
{
    double *features; // A pointer containing double values to keep the number of features dynamic
    Sample *centroid; // A pointer containing the centroid (unused for centroids themselves)
};

// Keep the array of samples and its size together
typedef struct
{
    Sample *samples;
    size_t numberOfSamples;
}
Dataset;

/**
 * @brief Reads data from a specific file.
 *
 * My hard working char-by-char method.
 *
 * @param fileName The name of the file to read.
 * @param numberOfFeatures The number of features.
 *
 * @note It also stores the minimum and maximum values.
 *
 * @returns A pointer to a `Dataset` object containing the samples and its size.
 */
Dataset *read_data_from_file(char *fileName, size_t numberOfFeatures)
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
        perror("[read_data_from_file] Failed to allocate memory for samples");
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

    int ch;
    size_t tokensRead = 0;

    int minMaxDefaults = 1;
    while((ch = fgetc(dataFile)) != EOF) // Keep reading the next character until end of file
    {
        if (ch == '\n' || ch == ' ' || tokenBufferSize >= maxTokenLength - 1) // 1 space for null-terminator
        {
            storeLastLine:

            if (tokensRead < numberOfFeatures)
            {
                tempFeaturesBuffer[tokensRead] = strtod(tokenBuffer, NULL);

                // Check for min and max feature (needed for better random centroids?)
                if (minMaxDefaults == 1)
                {
                    globalMaxFeature = tempFeaturesBuffer[tokensRead];
                    globalMinFeature = tempFeaturesBuffer[tokensRead];
                    minMaxDefaults = 0;
                }
                else
                {
                    if (tempFeaturesBuffer[tokensRead] > globalMaxFeature)
                        globalMaxFeature = tempFeaturesBuffer[tokensRead];
                    else if (tempFeaturesBuffer[tokensRead] < globalMinFeature)
                        globalMinFeature = tempFeaturesBuffer[tokensRead];
                }
                ++tokensRead;
            }

            if (ch == '\n' || ch == EOF) // Check for end of line (or file if came from goto)
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
    }
    free(dataset->samples);
    free(dataset);
}

/**
 * @brief Calculates the mean for each feature.
 *
 * @attention Assumes `dataset->numberOfSamples` is greater than 0.
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
        perror("[calculate_mean] Failed allocating memory for mean");
        exit(1);
    }

    // Loop through the samples and sum the features respectively
    for (size_t iSample = 0; iSample < dataset->numberOfSamples; ++iSample)
    {
        for (int iFeature = 0; iFeature < numberOfFeatures; ++iFeature)
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
 * @brief Standardizes the features of the whole dataset.
 *
 * x(standardized​) = (x − μ)​ / σ
 *
 * @note Modifies all sample features in-place
 * instead of returning a new `Sample` with the modified values.
 *
 * Exits with an error message if a division by 0 occurs.
 *
 * Stores max and min feature values.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param numberOfFeatures The number of features.
 */
void standardize_data(Dataset *dataset, size_t numberOfFeatures)
{
    // Check size because there's a division with size as denominator
    if (dataset->numberOfSamples == 0)
    {
        perror("[standardize_data] Failed to standardize data. \
            No data was found ? Avoided division by 0");
        exit(133);
    }

    double *mean = calculate_mean(dataset, numberOfFeatures);
    double *standardDeviation = calculate_std_deviation(dataset, mean, numberOfFeatures);

    int minMaxDefaults = 1;
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
            double x_standardized = (x - mu) / standard_deviation;
            dataset->samples[iSample].features[iFeature] = x_standardized;

            // Check for min and max feature (needed for better random centroids?)
            if (minMaxDefaults == 1)
            {
                globalMaxFeature = x_standardized;
                globalMinFeature = x_standardized;
                minMaxDefaults = 0;
            }
            else
            {
                if (x_standardized > globalMaxFeature)
                    globalMaxFeature = x_standardized;
                else if (x_standardized < globalMinFeature)
                    globalMinFeature = x_standardized;
            }
        }
    }
    // Avoid memory leaking if this function is called multiple times
    free(mean);
    free(standardDeviation);
}

/**
 * @brief Finds the nearest sample in an array of samples.
 *
 * @param sample The main sample.
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param numberOfFeatures The number of features.
 *
 * @returns The index of the nearest sample in `dataset->samples`.
 */
size_t find_nearest_point(const Sample *sample, const Dataset *dataset, size_t numberOfFeatures)
{
    // The index of the nearest sample in the samples array
    size_t iNearestPoint = 0;
    double nearestPointDistance = __DBL_MAX__;

    for (size_t iSample = 0; iSample < dataset->numberOfSamples; ++iSample)
    {
        // Euclidean distance
        double euclidianDistance = calculate_euclidian_distance(sample, &dataset->samples[iSample], numberOfFeatures);

        // Store the sample's index and distance if necessary
        if (euclidianDistance < nearestPointDistance)
        {
            iNearestPoint = iSample;
            nearestPointDistance = euclidianDistance; // Needed only for the condition
        }
    }
    return iNearestPoint;
}

/**
 * @brief Finds the farthest sample in an array of samples.
 *
 * @param sample The main sample.
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param numberOfFeatures The number of features.
 *
 * @returns The index of the nearest sample in `dataset->samples`.
 */
size_t find_furthest_sample(const Sample *sample, const Dataset *dataset, size_t numberOfFeatures)
{
    // The index of the nearest sample in the samples array
    size_t iFarthestPoint = 0;
    double farthestPointDistance = __DBL_MAX__;

    for (size_t iSample = 0; iSample < dataset->numberOfSamples; ++iSample)
    {
        // Euclidean distance
        double euclidianDistance = calculate_euclidian_distance(sample, &dataset->samples[iSample], numberOfFeatures);

        // Store the sample's index and distance if necessary
        if (euclidianDistance > farthestPointDistance)
        {
            iFarthestPoint = iSample;
            farthestPointDistance = euclidianDistance; // needed only for the condition
        }
    }
    return iFarthestPoint;
}

/**
 * @brief Calculates the inertia of each centroid.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param centroids A pointer to a `Dataset` object containing the array of centroids and its count.
 * @param numberOfFeatures The number of features.
 *
 * @returns The total inertia value for the desired number of centroids (K).
 */
double *calculate_inertia(const Dataset *dataset, const Dataset *centroids, size_t numberOfFeatures)
{
    double *inertiaPerCentroid = calloc(centroids->numberOfSamples, sizeof(double));
    if (inertiaPerCentroid == NULL)
    {
        perror("[calculate_inertia] Failed to allocate memory for inertiaPerCentroid");
        exit(1);
    }

    for (size_t iSample = 0; iSample < dataset->numberOfSamples; ++iSample)
    {
        size_t iCentroid = 0;
        while (iCentroid < centroids->numberOfSamples && dataset->samples[iSample].centroid != &centroids->samples[iCentroid])
        {
            ++iCentroid;
        }

        if (iCentroid == centroids->numberOfSamples)
        {
            fprintf(stderr, "[calculate_inertia] Centroid for sample %zu was not found... (wtf?)", iSample);
            exit(1);
        }

        double distance = calculate_euclidian_distance(
            &dataset->samples[iSample],
            dataset->samples[iSample].centroid,
            numberOfFeatures
        );
        inertiaPerCentroid[iCentroid] += distance * distance;
    }

    return inertiaPerCentroid;
}

/**
 * @brief Re-initializes 0-sample centroids by giving them a random features.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param centroids A pointer to a `Dataset` object containing the array of centroids and its count.
 * @param samplesPerCentroid An array of integers containing the number of samples assigned to each centroid.
 * @param numberOfFeatures The number of features.
 *
 * @returns The number of centroids that got 'revived'.
 *
 * @note This forces the data to be split into K clusters,
 * exactly as indicated by the user. From my POV, the best option is to randomize
 * the features of the centroids that have no samples assigned,
 * as if the algorithm was executed from 0 (better than assigning a random sample to it!).
 *
 * PS: I changed my mind, assigning a sample is better to avoid infinite loops... it should've been obvious!
 */
size_t handle_empty_centroids(Dataset *dataset, const Dataset *centroids, size_t *samplesPerCentroid, size_t numberOfFeatures)
{
    size_t revived = 0;

    for (size_t iCentroid = 0; iCentroid < centroids->numberOfSamples; ++iCentroid)
    {
        if (samplesPerCentroid[iCentroid] == 0)
        {
            /* Centroid Randomization
            for (int iFeature = 0; iFeature < numberOfFeatures; ++iFeature)
            {
                centroids->samples[iCentroid].features[iFeature] = (double)drand48();
            }
            */

            /* Centroid to its furthest sample
            size_t iSample = find_furthest_sample(&centroids->samples[iCentroid], dataset, numberOfFeatures);
            dataset->samples[iSample].centroid = &centroids->samples[iCentroid];
            */

            /* Centroid gets assigned the sample that will lower the total inertia as much as possible */
            double *inertiaPerCentroid = calculate_inertia(dataset, centroids, numberOfFeatures);
            size_t highestInertiaIndex = 0; // Assuming at least one valid cluster, avoids invalid continue if no clusters >1
            double highestInertia = -1; // -1 for continue, meaning no valid cluster to steal from
            for (size_t iInertia = 0; iInertia < centroids->numberOfSamples; ++iInertia)
            {
                // The last condition seems useless, I might remove it after testing thoroughly (future update? I'm lazy xD)
                // It seems logical to me that if highest inertia is this centroid, it means that it has at least 2 samples else its inertia would be 0!?
                if (inertiaPerCentroid[iInertia] > highestInertia && samplesPerCentroid[iInertia] > 1)
                {
                    highestInertia = inertiaPerCentroid[iInertia];
                    highestInertiaIndex = iInertia;
                }
            }
            free(inertiaPerCentroid);
            if (highestInertia == -1) continue;

            // Collect pointers to samples in the donor cluster
            Sample **clusterSamples = malloc(samplesPerCentroid[highestInertiaIndex] * sizeof(Sample *));
            if (clusterSamples == NULL) {
                perror("[handle_empty_centroids] Failed to allocate memory for clusterSamples");
                exit(1);
            }
            size_t clusterCount = 0;
            for (size_t iSample = 0; iSample < dataset->numberOfSamples; ++iSample) {
                if (dataset->samples[iSample].centroid == &centroids->samples[highestInertiaIndex]) {
                    clusterSamples[clusterCount++] = &dataset->samples[iSample];
                }
            }

            // Temp dataset for search (no malloc for Dataset, just stack init)
            Dataset tempCluster = { .samples = *clusterSamples, .numberOfSamples = clusterCount };

            size_t localIndex = find_furthest_sample(&centroids->samples[highestInertiaIndex], &tempCluster, numberOfFeatures);

            // Reassign
            clusterSamples[localIndex]->centroid = &centroids->samples[iCentroid];

            // Update counts (old is always highestInertiaIndex)
            --samplesPerCentroid[highestInertiaIndex];
            ++samplesPerCentroid[iCentroid];

            ++revived;
            free(clusterSamples);  // Just free the array of pointers, using free_dataset() frees features!
        }
    }
    return revived;
}

/**
 * @brief Adjusts each centroid based on the mean value of their samples.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param centroids A pointer to a `Dataset` object containing the array of centroids and its count.
 * @param numberOfFeatures The number of features.
 * @param threshold The minimum amount of distance the new centroid's features should be for it to move .
 *
 * @returns The number of centroids that have changed features (>threshold).
 *
 * @note `threshold` should be between 0 and 1.
 */
size_t adjust_centroids(const Dataset *dataset, Dataset *centroids, size_t numberOfFeatures, double threshold)
{
    Dataset *samplesPerCentroid = calloc(centroids->numberOfSamples, sizeof(Dataset));
    if (samplesPerCentroid == NULL)
    {
        perror("[adjust_centroids] Failed to allocate memory for samplesPerCentroid");
        exit(1);
    }

    for (size_t iSample = 0; iSample < dataset->numberOfSamples; ++iSample)
    {
        size_t iCentroid = 0;
        while(iCentroid < centroids->numberOfSamples && dataset->samples[iSample].centroid != &centroids->samples[iCentroid])
        {
            ++iCentroid;
        }

        if (iCentroid == centroids->numberOfSamples)
        {
            fprintf(stderr, "[adjust_centroids] No centroid assigned to this sample. Call `assign_centroid_to_samples` first");
            exit(1);
        }

        samplesPerCentroid[iCentroid].samples = realloc(samplesPerCentroid[iCentroid].samples, (samplesPerCentroid[iCentroid].numberOfSamples + 1) * sizeof(Sample));
        if (samplesPerCentroid[iCentroid].samples == NULL)
        {
            perror("[adjust_centroids] Failed to (re)allocate memory for samplesPerCentroid[iCentroid].samples");
            exit(1);
        }
        samplesPerCentroid[iCentroid].samples[samplesPerCentroid[iCentroid].numberOfSamples] = dataset->samples[iSample];
        ++samplesPerCentroid[iCentroid].numberOfSamples;
    }

    size_t adjustements = 0;
    for (size_t iCentroid = 0; iCentroid < centroids->numberOfSamples; ++iCentroid)
    {
        if (samplesPerCentroid[iCentroid].numberOfSamples == 0)
        {
            continue;
        }

        Sample newCentroidPosition;
        newCentroidPosition.features = calculate_mean(&samplesPerCentroid[iCentroid], numberOfFeatures);

        if (calculate_euclidian_distance(&centroids->samples[iCentroid], &newCentroidPosition, numberOfFeatures) > threshold)
        {
            free(centroids->samples[iCentroid].features); // Memory leak if not freed...
            centroids->samples[iCentroid].features = newCentroidPosition.features;
            ++adjustements;
        }
        else
        {
            // Not used anymore, free the memory
            free(newCentroidPosition.features);
            newCentroidPosition.features = NULL;
        }

        // Free the top-level pointer for samples
        free(samplesPerCentroid[iCentroid].samples); // Do NOT free the actual samples used by the main Dataset!
    }

    // Finally, free the top-level pointer
    free(samplesPerCentroid);

    return adjustements;
}

/**
 * @brief Assigns centroids to their nearest samples.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param centroids A pointer to a `Dataset` object containing the array of centroids and its count.
 * @param numberOfFeatures The number of features.
 *
 * @returns The number of samples assigned to each centroid.
 */
size_t *assign_centroid_to_samples(Dataset *dataset, const Dataset *centroids, size_t numberOfFeatures)
{
    if (centroids->numberOfSamples == 0)
    {
        fprintf(stderr, "[assign_centroid_to_samples] Centroids' size is 0. Cannot assign any samples");
        exit(1);
    }

    size_t *samplesPerCentroid = calloc(centroids->numberOfSamples, sizeof(size_t));
    if (samplesPerCentroid == NULL)
    {
        perror("[assign_centroid_to_samples] Failed to allocate memory for samplesPerCentroid");
        exit(1);
    }

    for (size_t iSample = 0; iSample < dataset->numberOfSamples; ++iSample)
    {
        size_t iNearestCentroid = find_nearest_point(&dataset->samples[iSample], centroids, numberOfFeatures);
        dataset->samples[iSample].centroid = &centroids->samples[iNearestCentroid];
        ++samplesPerCentroid[iNearestCentroid];
    }

    return samplesPerCentroid;
}

/**
 * @brief Outputs the features values of all samples.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param numberOfFeatures The number of features.
 *
 * @note I wrote this for easier debugging...
 */
void output_all_sample_features(const Dataset *dataset, size_t numberOfFeatures)
{
    printf("\n-- SAMPLES OUTPUT --\n");
    for (size_t iSample = 0; iSample < dataset->numberOfSamples; ++iSample)
    {
        printf("\n[Sample %zu] Features:\n\n", iSample);
        for (size_t iFeature = 0; iFeature < numberOfFeatures; ++iFeature)
        {
            printf("- Feature %zu: %lf\n",
                iFeature,
                dataset->samples[iSample].features[iFeature]);
        }
    }
    printf("\nIteration finished.\n\n");
}

/**
 * @brief Outputs the distance difference after the centroids were adjusted.
 *
 * @param centroids A pointer to a `Dataset` object containing the array of centroids and its count.
 * @param oldCentroids A pointer to a `Dataset` object containing the array of centroids at their older state and its count.
 * @param samplesPerCentroid The number of samples assigned to each centroid.
 * @param numberOfFeatures The number of features.
 *
 * @note I wrote this for easier debugging...
 *
 * You can pass NULL for `oldCentroids` and/or `samplesPerCentroid` if they're not needed.
 */
void output_centroids(const Dataset *centroids, const Dataset *oldCentroids, const size_t *samplesPerCentroid, size_t numberOfFeatures)
{
    FILE *file = NULL;
    if (globalSaveOutput == 1)
    {
        file = fopen("output.txt", "a");
        fprintf(file, "K-Means Iteration Output for K=%zu:\n", centroids->numberOfSamples);
    }

    printf("\n### OUTPUT ###\n");
    if (globalSaveOutput == 1) fprintf(file, "\n### OUTPUT ###\n");

    for (size_t iCentroid = 0; iCentroid < centroids->numberOfSamples; ++iCentroid)
    {
        double distance = (oldCentroids != NULL) ? calculate_euclidian_distance(
            &oldCentroids->samples[iCentroid],
            &centroids->samples[iCentroid],
            numberOfFeatures
        ) : 0;

        printf("\n[Centroid %zu] Features:\n", iCentroid);
        if (globalSaveOutput == 1) fprintf(file, "\n[Centroid %zu] Features:\n", iCentroid);

        for (size_t iFeature = 0; iFeature < numberOfFeatures; ++iFeature)
        {
            if (oldCentroids != NULL)
            {
                printf(
                    "- Feature %zu: %lf (old) -> %lf (new)\n",
                    iFeature,
                    oldCentroids->samples[iCentroid].features[iFeature],
                    centroids->samples[iCentroid].features[iFeature]
                );
                if (globalSaveOutput == 1) fprintf(
                    file,
                    "- Feature %zu: %lf (old) -> %lf (new)\n",
                    iFeature,
                    oldCentroids->samples[iCentroid].features[iFeature],
                    centroids->samples[iCentroid].features[iFeature]
                );
            }
            else
            {
                printf(
                    "- Feature %zu: %lf\n",
                    iFeature,
                    centroids->samples[iCentroid].features[iFeature]
                );
                if (globalSaveOutput == 1) fprintf(
                    file,
                    "- Feature %zu: %lf\n",
                    iFeature,
                    centroids->samples[iCentroid].features[iFeature]
                );
            }
        }

        if (samplesPerCentroid != NULL)
        {
            printf("\nSamples assigned: %zu\n", samplesPerCentroid[iCentroid]);
            if (globalSaveOutput == 1) fprintf(file, "\nSamples assigned: %zu\n", samplesPerCentroid[iCentroid]);
        }
        if (oldCentroids != NULL)
        {
            printf("\n- Distance difference: %lf\n\n", distance);
            if (globalSaveOutput == 1) fprintf(file, "\n- Distance difference: %lf\n\n", distance);
        }
    }
    printf("\n");
    if (globalSaveOutput == 1)
    {
        fprintf(file, "\n\n\n");
        fclose(file);
    }
}

/**
 * @brief Clones a `Dataset` object.
 *
 * @param source A pointer to a `Dataset` object which will be cloned.
 * @param numberOfFeatures The number of features.
 *
 * @returns A pointer to a `Dataset` object, a clone of `source`.
 */
Dataset *clone_dataset_object(const Dataset *source, size_t numberOfFeatures)
{
    Dataset *destination = malloc(sizeof(Dataset));
    if (destination == NULL)
    {
        perror("[clone_dataset_object] Failed to allocate memory for destination");
        exit(1);
    }
    destination->numberOfSamples = source->numberOfSamples;
    destination->samples = malloc(destination->numberOfSamples * sizeof(Sample));

    for (size_t iSample = 0; iSample < destination->numberOfSamples; ++iSample) {

        destination->samples[iSample].features = malloc(numberOfFeatures * sizeof(double));
        if (destination->samples[iSample].features == NULL)
        {
            perror("[clone_dataset_object] Failed to allocate memory for destination->samples[iSample].features");
            exit(1);
        }

        memcpy(destination->samples[iSample].features,
               source->samples[iSample].features,
               numberOfFeatures * sizeof(double));

        destination->samples[iSample].centroid = NULL; // not used
    }
    return destination;
}

/**
 * @brief Creates a`Dataset`object containing K randomly initialized centroids.
 *
 * @param numberOfFeatures The number of features (dimensions) for each centroid.
 * @param K The number of centroids to create.
 *
 * @returns A`Dataset`object containing K randomly initialized centroids.
 */
Dataset *create_random_centroids(size_t numberOfFeatures, size_t K)
{
    Dataset *centroids = malloc(sizeof(Dataset));
    if (centroids == NULL)
    {
        perror("[create_random_centroids] Failed to allocate memory for centroids");
        exit(1);
    }
    centroids->samples = malloc(K * sizeof(Sample));
    if (centroids->samples == NULL)
    {
        perror("[create_random_centroids] Failed to allocate memory for centroids->samples");
        exit(1);
    }
    centroids->numberOfSamples = K;

    for (size_t iCentroid = 0; iCentroid < K; ++iCentroid)
    {
        centroids->samples[iCentroid].features = malloc(numberOfFeatures * sizeof(double));
        if (centroids->samples[iCentroid].features == NULL)
        {
            perror("[create_random_centroids] Failed to allocate memory for centroids->samples[iCentroid].features");
            exit(1);
        }
        centroids->samples[iCentroid].centroid = NULL;

        for (size_t iFeature = 0; iFeature < numberOfFeatures; ++iFeature)
        {
            double random = (double)rand() / RAND_MAX;
            centroids->samples[iCentroid].features[iFeature] = globalMaxFeature * random + globalMinFeature; // (((double)rand() / RAND_MAX) * 2 - 1)
        }
    }
    return centroids;
}

/**
 * @brief Runs the K-Means clustering algorithm until centroids converge.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param centroids A pointer to a `Dataset` object containing the array of centroids and its count.
 * @param numberOfFeatures The number of features.
 * @param tolerance The minimum amount of distance the new centroid's features should be for it to move.
 *
 * @note This function stops when centroids no longer move significantly.
 */
size_t *run_kmeans_to_convergence(Dataset *dataset, Dataset *centroids, size_t numberOfFeatures, double tolerance)
{
    size_t *samplesPerCentroid;
    size_t centroidsAdjusted;
    do
    {
        size_t revivedCentroids = 0; // Reset for the new iteration

        samplesPerCentroid = assign_centroid_to_samples(dataset, centroids, numberOfFeatures);
        revivedCentroids = handle_empty_centroids(dataset, centroids, samplesPerCentroid, numberOfFeatures);

        // Track modifications by keeping a temporary backup of the data before adjustments are made
        // Dataset *centroidsClone = clone_dataset_object(centroids, numberOfFeatures);
        // Number of adjusted centroids
        centroidsAdjusted = adjust_centroids(dataset, centroids, numberOfFeatures, tolerance);

        //output_centroids(centroids, centroidsClone, samplesPerCentroid, numberOfFeatures);
        
        //free_dataset(centroidsClone);
        
        // keep the samples-per-centroid data if adjustements were made & escape loop
        if (revivedCentroids == 0 && centroidsAdjusted == 0) break;
        free(samplesPerCentroid);
    }
    while (1);
    return samplesPerCentroid;
}

/**
 * @brief Finds the 'best' K value.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param numberOfFeatures The number of features.
 * @param startK The number of centroids to start with.
 * @param endK The number of centroids the loop ends with.
 * @param tolerance The minimum amount of distance the new centroid's features should be for it to move.
 * @param dropThreshold This is the inertia's percentage difference (0.01 = 1%) between current and last execution.
 * @param restarts To avoid as much as possible a bad local minimum with high inertia,
 * restart the algorithm and pick the best.
 *
 * @note This function executes the K-means algorithm in a loop,
 * with K in {`startK`, ..., `endK`}.
 */
void elbow_method(Dataset *dataset, size_t numberOfFeatures, size_t startK, size_t endK, double tolerance, double dropThreshold, int restarts)
{
    int answerReachMaxK = 0;
    double lastInertia = -1.0;

    for (size_t K = startK; K <= endK; ++K)
    {
        double bestInertia = __DBL_MAX__;
        Dataset *bestResultCentroids = NULL;
        size_t *bestResultSamplesPerCentroid = NULL;

        for (int r = 0; r < restarts; ++r)
        {
            Dataset *centroids = create_random_centroids(numberOfFeatures, K);
            size_t *samplesPerCentroid = run_kmeans_to_convergence(dataset, centroids, numberOfFeatures, tolerance);
            double *inertiaPerCentroid = calculate_inertia(dataset, centroids, numberOfFeatures);

            double inertia = 0; // total inertia
            for (size_t iCentroid = 0; iCentroid < centroids->numberOfSamples; ++iCentroid)
            {
                inertia += inertiaPerCentroid[iCentroid];
            }
            free(inertiaPerCentroid);

            if (bestInertia > inertia)
            {
                if (bestResultCentroids != NULL)
                {
                    free_dataset(bestResultCentroids);
                    free(bestResultSamplesPerCentroid);
                }

                bestInertia = inertia;
                bestResultCentroids = centroids;
                bestResultSamplesPerCentroid = samplesPerCentroid;
            }
            else
            {
                free_dataset(centroids);
                free(samplesPerCentroid);
            }
        }
        output_centroids(bestResultCentroids, NULL, bestResultSamplesPerCentroid, numberOfFeatures);

        // Output the calculated total Inertia (WCSS) value
        printf("K = %zu, Inertia = %.6f, Per-Sample SSE = %lf\n", K, bestInertia, bestInertia/dataset->numberOfSamples);

        if (lastInertia > 0)
        {
            // How much did the inertia value drop by
            double dropK = (lastInertia - bestInertia) / lastInertia; // 0.1 means 10%

            if (dropK < dropThreshold && answerReachMaxK == 0)
            {
                printf("Found inertia gains under the drop threshold with K value: %zu (Drop threshold: %lf)\n\n", K, dropThreshold);

                if (K != endK)
                {
                    printf("Would you like to continue until reaching the maximum K desired nonetheless (0 -> no; 1 -> yes)? ");
                    scanf("%d", &answerReachMaxK);
                }
                if (answerReachMaxK == 0) break;
            }
        }
        lastInertia = bestInertia;
        free_dataset(bestResultCentroids);
        free(bestResultSamplesPerCentroid);
    }
}

int main(int argc, char *argv[])
{
    // Initialize the randomizer using the current timestamp as seed
    srand((unsigned int)time(NULL));

    printf("Number of features: ");
    size_t numberOfFeatures; scanf("%zu", &numberOfFeatures); // The number of features

    printf("Tolerance, minimum value for a centroid's movement to be recognized (default 0.001): ");
    double tolerance; scanf("%lf", &tolerance); // The tolerance value

    printf("Restarts, higher accuracy but heavier/slower (default 5): ");
    int restarts; scanf("%d", &restarts); // The numbers of time the algorithm restarts for a specific K value

    char path[MAX_PATH];
    printf("Data file's path: ");
    scanf("%255s", path);

    Dataset *dataset = read_data_from_file(path, numberOfFeatures);
    printf("Would you like to standardize the values (0 -> no; 1 -> yes)? ");
    scanf("%d", &globalStandardize);
    if (globalStandardize == 1) standardize_data(dataset, numberOfFeatures);
    //output_all_sample_features(dataset, numberOfFeatures); // For debugging

    printf("Would you like to save the output (0 -> no; 1 -> yes)? ");
    scanf("%d", &globalSaveOutput);
    (globalSaveOutput == 1) ? printf("Output will be saved in current directory as 'output.txt'.\n") : printf("Output will not be saved.\n");

    printf("Would you like to use the elbow method (0 -> no; 1 -> yes)? ");
    int useElbow; scanf("%d", &useElbow);

    size_t K = 1; // default K value
    if (useElbow == 1)
    {
        printf("Enter the maximum K value (default 10): ");
        size_t maxK; scanf("%zu", &maxK);

        printf("Enter the drop threshold for the elbow method (default 0.1 for 10%%): ");
        double dropThreshold = 0.1; scanf("%lf", &dropThreshold); // The difference between the inertia values of K-1 and K

        elbow_method(dataset, numberOfFeatures, K, maxK, tolerance, dropThreshold, restarts);
    }
    else // Don't use the elbow method
    {
        printf("K centroids: ");
        scanf("%zu", &K); // The number of centroids

        if (K == 0)
        {
            fprintf(stderr, "[main] Failed to run K-means algorithm: K value equals 0.\n");
            exit(1);
        }
        else if (K >= dataset->numberOfSamples)
        {
            fprintf(stderr, "[main] Failed to run K-means algorithm: K value (%zu) is bigger or equal to the total number of samples.\n", K);
            exit(1);
        }

        double bestInertia = __DBL_MAX__;
        Dataset *bestResultCentroids = NULL;
        size_t *bestResultSamplesPerCentroid = NULL;
        for (int r = 0; r < restarts; ++r)
        {
            Dataset *centroids = create_random_centroids(numberOfFeatures, K);
            size_t *samplesPerCentroid = run_kmeans_to_convergence(dataset, centroids, numberOfFeatures, tolerance);
            double *inertiaPerCentroid = calculate_inertia(dataset, centroids, numberOfFeatures);

            double inertia = 0; // total inertia
            for (size_t iCentroid = 0; iCentroid < centroids->numberOfSamples; ++iCentroid)
            {
                inertia += inertiaPerCentroid[iCentroid];
            }
            free(inertiaPerCentroid);

            if (bestInertia > inertia)
            {
                if (bestResultCentroids != NULL)
                {
                    free_dataset(bestResultCentroids);
                    free(bestResultSamplesPerCentroid);
                }

                bestInertia = inertia;
                bestResultCentroids = centroids;
                bestResultSamplesPerCentroid = samplesPerCentroid;
            }
            else
            {
                free_dataset(centroids);
                free(samplesPerCentroid);
            }
        }
        output_centroids(bestResultCentroids, NULL, bestResultSamplesPerCentroid, numberOfFeatures);

        // Output the calculated total Inertia (WCSS) value
        printf("K = %zu, Inertia = %.6f, Per-Sample SSE = %lf\n", K, bestInertia, bestInertia / dataset->numberOfSamples); // K value, Total inertia value, AVG per-sample inertia

        free_dataset(bestResultCentroids);
        free(bestResultSamplesPerCentroid);
    }

    free_dataset(dataset);
    return 0;
}
