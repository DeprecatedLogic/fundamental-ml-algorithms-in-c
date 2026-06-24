/*======================================================================
 *  K-Means (unsupervised learning)
 *
 *  Author      :  DeprecatedLogic  <https://www.github.com/DeprecatedLogic>
 *  Created     :  27 Jun 2025
 *
 *  Description :
 *      Reads N-dimensional samples from a text file,
 *      partitions a dataset into K distinct, non-overlapping clusters,
 *      where each sample belongs to the cluster with the nearest mean (or "centroid").
 *      It returns the best K value based on the elbow method (if a specific K value is undesired).
 *====================================================================*/

#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define CMD_PRINT(ANSI_COLOR, PROGRAM_NAME, ARGS) printf("~>%s %s %s %s\n", ANSI_COLOR, PROGRAM_NAME, ARGS, ANSI_COLOR_RESET)

#define MAX_PATH 256
#define MAX_DELIM 16
#define SUCCESS 0
#define FAILURE 1

typedef struct
{
    char dataset_path[MAX_PATH]; // Dataset file path
    size_t number_of_features; // Number of features per sample (default: 1)
    double tolerance; // Minimum value for a centroid's movement to be recognized (default: 0.001)
    size_t restarts; // Higher accuracy but slower (default: 5)
    size_t K; // Start algorithm at K_begin value (default: 1)
    size_t K_end;  // End algorithm at K_end value (default: 10)
    double drop_threshold; // Difference between the inertia values of K-1 and K, used with elbow method (default: 0.1 meaning 10%)
    bool standardize; // Standardize all sample features (false by default)
    // todo: enter a path instead of a predefined file name for the output/results ?
    bool save_output; // Save output in current directory as 'output.txt' (false by default) 
    bool use_elbow_method; // Use the elbow method to find the most optimal K (false by default)
    bool force_reach_K; // Continue until reaching the max K desired, used with elbow method (false by default)
    char token_delimiter[MAX_DELIM];
    char decimals_delimiter;
    char comment_delimiter[MAX_DELIM];
    bool debug; // Verbose, used for debugging (false by default)
} Args;

/* Global variables */
double global_min_feature = 0;
double global_max_feature = 0;
Args global_args;
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
    size_t number_of_samples;
}
Dataset;

/**
 * @brief Frees a `Dataset` object.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 */
void free_dataset(Dataset *dataset)
{
    if (dataset == NULL)
    {
        if (global_args.debug) printf("[free_dataset] Dataset is NULL\n"); // Debugging
        return;
    }

    if (dataset->samples != NULL)
    {
        for (size_t sample_index = 0; sample_index < dataset->number_of_samples; ++sample_index)
        {
            if (dataset->samples[sample_index].features != NULL)
                free(dataset->samples[sample_index].features);
        }
        free(dataset->samples);
    }

    free(dataset);
    if (global_args.debug) printf("[free_dataset] Dataset freed\n"); // Debugging
}

bool is_number(char *number, char decimals_delimiter)
{
    if (*number == '-') ++number;
    if (*number == '\0') return false;
    
    static const unsigned char DIGIT_LUT[256] = {
        ['1'] = 1, ['2'] = 1, ['3'] = 1,
        ['4'] = 1, ['5'] = 1, ['6'] = 1,
        ['7'] = 1, ['8'] = 1, ['9'] = 1,
        ['0'] = 1 // `{['0'  '9']=1}` possible but less portable
    };

    bool delimiter_found = false;
    while (*number)
    {
        char c = *number++;

        if (DIGIT_LUT[(unsigned char)c]) continue;
        if (c == decimals_delimiter)
        {
            if (delimiter_found) return false;
            delimiter_found = true;
            continue;
        }
        return false;
    }
    return true;
}

Dataset *read_data_from_file(
    const char *file_path, size_t number_of_features,
    char *token_delimiter, char decimals_delimiter, char *comment_delimiter
)
{
    if (token_delimiter == NULL) strcpy(token_delimiter, " ");
    if (comment_delimiter == NULL) strcpy(comment_delimiter, "#");
    if (strcmp(comment_delimiter, comment_delimiter) == 1)
    {
        fprintf(stderr, "[read_data_from_file] An error occured, the delimiter cannot be `%s`\n", comment_delimiter);
        return NULL;
    }

    FILE *data_file = fopen(file_path, "r"); // Does NOT throw, returns NULL upon failure
    if (data_file == NULL) // File doesn't exist or maybe insufficient permissions to read ?
    {
        fprintf(stderr, "[read_data_from_file] Failed to open file: %s\n", file_path);
        return NULL;
    }

    Dataset *dataset = (Dataset *)calloc(1, sizeof(Dataset));
    if (dataset == NULL)
    {
        perror("[read_data_from_file] Failed to allocate memory for dataset\n");
        fclose(data_file);
        return NULL;
    }

    const size_t feature_max_digits = 40; // a max of 40 digits per feature should be enough (the maximum precision for double is 15 or 17 anyway?)
    const size_t label_max_chars = 100; // a max of 100 chars available for the label, enough to go crazy with the labels
    const size_t line_size = (number_of_features * feature_max_digits) + label_max_chars + 1; // +1 for '\0'
    
    size_t line_number = 1;
    char line[line_size];
    bool bom_checked = false;
    while(fgets(line, (int)(line_size), data_file))
    {
        if (!bom_checked)
        {
            unsigned char *p_line = (unsigned char *)line;
            if (p_line[0] == 0xEF && p_line[1] == 0xBB && p_line[2] == 0xBF)
                memmove(line, line + 3, strlen(line + 3) + 1);
            
            bom_checked = true;
        }
        
        if (strncmp(line, comment_delimiter, strlen(comment_delimiter)) == 0)
        {
            if (global_args.debug) printf("[read_data_from_file] Line %zu skipped (fully commented line)\n", line_number++);
            continue;
        }

        char *uncommented_line = strtok(line, comment_delimiter);
        
        // Strip newline
        uncommented_line[strcspn(uncommented_line, "\r\n")] = '\0';
        if (uncommented_line == NULL || *uncommented_line == '\0')
        {
            if (global_args.debug) printf("[read_data_from_file] Line %zu skipped (empty line)\n", line_number);
            continue;
        }

        if (global_args.debug) printf("[read_data_from_file] uncommented_line (line %zu): %s\n", line_number, uncommented_line);

        char *token_buffer = strtok(uncommented_line, token_delimiter);
        size_t token_counter = 0;
        
        Sample sample = {
            .features = (double *)malloc(sizeof(double) * number_of_features),
            .centroid = NULL
        };
        if (sample.features == NULL)
        {
            perror("[read_data_from_file] Failed to allocate memory for sample.features\n");
            free_dataset(dataset);
            fclose(data_file);
            return NULL;
        }
        size_t current_number_of_features = 0;
            
        while(token_buffer != NULL)
        {
            if (global_args.debug) printf("[read_data_from_file] token_buffer: %s (token_counter: %zu)\n", token_buffer, token_counter);
            
            // If true, we can stop here but we won't be able to find the exact number of features the sample has
            // which requires the user to check by themself
            //if (token_counter > number_of_features) break; // encountered bad line format probably; 

            if (current_number_of_features < number_of_features)
            {
                if (!is_number(token_buffer, decimals_delimiter)) // encountered a value that is not a number
                {
                    fprintf(
                        stderr,
                        "[read_data_from_file] Encoutered a bad sample at line '%zu' (feature value is not a number ?)\n",
                        line_number
                    );
                    free_dataset(dataset);
                    free(sample.features);
                    fclose(data_file);
                    return NULL;
                }
                sample.features[current_number_of_features++] = strtod(token_buffer, NULL);
            }

            token_buffer = strtok(NULL, token_delimiter);
            ++token_counter;
        }
        if (token_counter != number_of_features)
        {
            fprintf(
                stderr,
                "[read_data_from_file] Encountered a mismatch: features counted [%zu] != expected [%zu] (line %zu)\n",
                token_counter,
                number_of_features,
                line_number
            );
            free_dataset(dataset);
            free(sample.features);
            fclose(data_file);
            return NULL;
        }

        Sample *temp_samples = (Sample *)realloc(dataset->samples, sizeof(Sample) * (dataset->number_of_samples + 1));
        if (temp_samples == NULL)
        {
            fprintf(stderr, "[read_data_from_file] Failed to re-allocate memory for temp_samples\n");
            free_dataset(dataset);
        }
        dataset->samples = temp_samples;
        dataset->samples[dataset->number_of_samples] = sample;
        ++dataset->number_of_samples;
        
        ++line_number;
    }
    fclose(data_file);
    return dataset;
}

/**
 * @brief Calculates the mean for each feature.
 *
 * @attention Assumes `dataset->number_of_samples` is greater than 0.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param number_of_features The number of features.
 *
 * @returns An array containing the mean value of each feature.
 */
double *calculate_mean(const Dataset *dataset, size_t number_of_features)
{
    // Mean value for each feature (initialized with 0s)
    double *mean = calloc(number_of_features, sizeof(double));
    if (mean == NULL) // check
    {
        perror("[calculate_mean] Failed to allocate memory for mean & initialize all members to 0\n");
        return NULL;
    }

    // Loop through the samples and sum the features respectively
    for (size_t sample_index = 0; sample_index < dataset->number_of_samples; ++sample_index)
    {
        for (int feature_index = 0; feature_index < number_of_features; ++feature_index)
            mean[feature_index] += dataset->samples[sample_index].features[feature_index];
    }

    // Finally, divide each value by the number of samples
    for (size_t feature_index = 0; feature_index < number_of_features; ++feature_index)
    {
        mean[feature_index] /= dataset->number_of_samples;
    }

    return mean;
}

/**
 * @brief Calculates the standard deviation for each feature.
 *
 * @attention Assumes `dataset->number_of_samples` is greater than 0.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param mean An array containing the standard deviation of each feature.
 * @param number_of_features The number of features.
 *
 * @returns An array containing the standard deviation of every feature.
 */
double *calculate_std_deviation(const Dataset *dataset, const double *mean, size_t number_of_features)
{
    // Standard deviation value for each feature (initialized with 0s)
    double *standard_deviation = calloc(number_of_features, sizeof(double));
    if (standard_deviation == NULL) // check
    {
        perror("[calculate_std_deviation] Failed to allocate memory for standard_deviation & initialize all members to 0\n");
        return NULL;
    }

    // Loop through the samples and sum all the squared differences (feature - mean)^2
    for (size_t sample_index = 0; sample_index < dataset->number_of_samples; ++sample_index)
    {
        for (size_t feature_index = 0; feature_index < number_of_features; ++feature_index)
        {
            double diff = dataset->samples[sample_index].features[feature_index] - mean[feature_index];
            standard_deviation[feature_index] += diff * diff;
            // pow() would've been slower :P
        }
    }
    // Finally, divide each value by the number of samples to get the variance
    // and then use `sqrt` to get the standard deviation
    for (size_t feature_index = 0; feature_index < number_of_features; ++feature_index)
    {
        standard_deviation[feature_index] = sqrt(standard_deviation[feature_index] / dataset->number_of_samples);
    }
    return standard_deviation;
}

/**
 * @brief Calculates the Euclidian distance between two samples.
 *
 * @param sample1 The first sample.
 * @param sample2 The second sample.
 * @param number_of_features The number of features.
 *
 * @returns The Euclidian distance between `sample1` and `sample2`.
 */
double calculate_euclidian_distance(const Sample *sample1, const Sample *sample2, size_t number_of_features)
{
    // Calculate the euclidian distance
    double euclidian_distance = 0;

    for (size_t feature_index = 0; feature_index < number_of_features; ++feature_index)
    {
        double diff = sample1->features[feature_index] - sample2->features[feature_index];
        euclidian_distance += diff * diff;
    }
    return sqrt(euclidian_distance);
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
 * @param number_of_features The number of features.
 */
int standardize_data(Dataset *dataset, const double *mean, const double *standard_deviation, size_t number_of_features)
{
    // Check size because there's a division with size as denominator
    if (dataset->number_of_samples == 0)
    {
        perror("[standardize_data] Failed to standardize data. Avoided division by 0 (no data found ?)\n");
        return FAILURE;
    }

    bool min_max_defaults = true;
    for (size_t sample_index = 0; sample_index < dataset->number_of_samples; ++sample_index)
    {
        for (size_t feature_index = 0; feature_index < number_of_features; ++feature_index)
        {
            if (standard_deviation[feature_index] == 0) // Again, avoid dividing by 0...
            {
                fprintf(stderr, "[standardize_data] Feature %zu has 0 for standard deviation. Avoided division by 0\n", feature_index);
                return FAILURE;
            }
            // x_standardized = (x - mu) / standard_deviation
            double x = dataset->samples[sample_index].features[feature_index];
            double mu = mean[feature_index];
            double std = standard_deviation[feature_index];

            // x_standardized
            double x_standardized = (x - mu) / std;
            dataset->samples[sample_index].features[feature_index] = x_standardized;

            // Check for min and max feature (needed for better random centroids?)
            if (min_max_defaults)
            {
                global_max_feature = x_standardized;
                global_min_feature = x_standardized;
                min_max_defaults = false;
            }
            else
            {
                if (x_standardized > global_max_feature)
                    global_max_feature = x_standardized;
                else if (x_standardized < global_min_feature)
                    global_min_feature = x_standardized;
            }
        }
    }

    return SUCCESS;
}

/**
 * @brief Finds the nearest sample in an array of samples.
 *
 * @param sample The main sample.
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param number_of_features The number of features.
 *
 * @returns The index of the nearest sample in `dataset->samples`.
 */
size_t find_nearest_point(const Sample *sample, const Dataset *dataset, size_t number_of_features)
{
    // The index of the nearest sample in the samples array
    size_t nearest_point_index = 0;
    double nearest_point_distance = __DBL_MAX__;

    for (size_t sample_index = 0; sample_index < dataset->number_of_samples; ++sample_index)
    {
        // Euclidean distance
        double euclidian_distance = calculate_euclidian_distance(sample, &dataset->samples[sample_index], number_of_features);

        // Store the sample's index and distance if necessary
        if (euclidian_distance < nearest_point_distance)
        {
            nearest_point_index = sample_index;
            nearest_point_distance = euclidian_distance; // Needed only for the condition
        }
    }
    return nearest_point_index;
}

/**
 * @brief Finds the farthest sample in an array of samples.
 *
 * @param sample The main sample.
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param number_of_features The number of features.
 *
 * @returns The index of the farthest sample in `dataset->samples`.
 */
size_t find_farthest_sample(const Sample *sample, const Dataset *dataset, size_t number_of_features)
{
    // The index of the farthest sample in the samples array
    size_t farthest_point_index = dataset->number_of_samples;
    double farthest_point_distance = -__DBL_MAX__;

    for (size_t sample_index = 0; sample_index < dataset->number_of_samples; ++sample_index)
    {
        // Euclidean distance
        double euclidian_distance = calculate_euclidian_distance(sample, &dataset->samples[sample_index], number_of_features);

        // Store the sample's index and distance if necessary
        if (euclidian_distance > farthest_point_distance)
        {
            farthest_point_index = sample_index;
            farthest_point_distance = euclidian_distance; // needed only for the condition
        }
    }
    return farthest_point_index;
}

/**
 * @brief Calculates the inertia of each centroid.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param centroids A pointer to a `Dataset` object containing the array of centroids and its count.
 * @param number_of_features The number of features.
 *
 * @returns The total inertia value for the desired number of centroids (K).
 */
double *calculate_inertia(const Dataset *dataset, const Dataset *centroids, size_t number_of_features)
{
    double *inertia_per_centroid = calloc(centroids->number_of_samples, sizeof(double));
    if (inertia_per_centroid == NULL)
    {
        perror("[calculate_inertia] Failed to allocate memory for inertia_per_centroid & initialize all members to 0\n");
        return NULL;
    }

    for (size_t sample_index = 0; sample_index < dataset->number_of_samples; ++sample_index)
    {
        size_t centroid_index = 0;
        while (centroid_index < centroids->number_of_samples && dataset->samples[sample_index].centroid != &centroids->samples[centroid_index])
        {
            ++centroid_index;
        }

        if (centroid_index == centroids->number_of_samples)
        {
            fprintf(stderr, "[calculate_inertia] Centroid for sample %zu was not found... (wtf?)\n", sample_index);
            free(inertia_per_centroid);
            return NULL;
        }

        double distance = calculate_euclidian_distance(
            &dataset->samples[sample_index],
            dataset->samples[sample_index].centroid,
            number_of_features
        );
        inertia_per_centroid[centroid_index] += distance * distance;
    }

    return inertia_per_centroid;
}

/**
 * @brief Re-initializes 0-sample centroids by giving them a random features.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param centroids A pointer to a `Dataset` object containing the array of centroids and its count.
 * @param samples_per_centroid An array of integers containing the number of samples assigned to each centroid.
 * @param number_of_features The number of features.
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
int handle_empty_centroids(Dataset *dataset, const Dataset *centroids, size_t *samples_per_centroid, size_t number_of_features, size_t *out_revived_count)
{
    size_t revived = 0;
    for (size_t centroid_index = 0; centroid_index < centroids->number_of_samples; ++centroid_index)
    {
        if (samples_per_centroid[centroid_index] == 0)
        {
            /* Centroid Randomization
            for (int feature_index = 0; feature_index < number_of_features; ++feature_index)
            {
                centroids->samples[centroid_index].features[feature_index] = (double)drand48();
            }
            */

            /* Centroid to the farthest sample
            size_t sample_index = find_farthest_sample(&centroids->samples[centroid_index], dataset, number_of_features);
            if (sample_index == dataset->number_of_samples)
            {
                fprintf(stderr, "[handle_empty_centroids] Failed to find the farthest sample");
                return FAILURE;
            }
            dataset->samples[sample_index].centroid = &centroids->samples[centroid_index];
            */

            /* Centroid gets assigned the sample that will lower the total inertia as much as possible */
            double *inertia_per_centroid = calculate_inertia(dataset, centroids, number_of_features);
            if (inertia_per_centroid == NULL)
            {
                fprintf(stderr, "[handle_empty_centroids] Failed to calculate inertia\n");
                return FAILURE;
            }
            size_t highest_inertia_index = 0; // Assuming at least one valid cluster, avoids invalid "continue" (later on) if no clusters > 1
            double highest_inertia = -1; // -1 for continue, meaning no valid cluster to steal from
            for (size_t inertia_index = 0; inertia_index < centroids->number_of_samples; ++inertia_index)
            {
                // The last condition seems useless, I might remove it after testing thoroughly (future update? xD)
                // It seems logical to me that if highest inertia is this centroid, it means that it has at least 2 samples else its inertia would be 0!?
                if (inertia_per_centroid[inertia_index] > highest_inertia && samples_per_centroid[inertia_index] > 1)
                {
                    highest_inertia = inertia_per_centroid[inertia_index];
                    highest_inertia_index = inertia_index;
                }
            }
            free(inertia_per_centroid);
            if (highest_inertia == -1) continue;

            // Collect pointers to samples in the donor cluster
            Sample **cluster_samples = calloc(samples_per_centroid[highest_inertia_index], sizeof(Sample *));
            if (cluster_samples == NULL)
            {
                perror("[handle_empty_centroids] Failed to allocate memory for cluster_samples\n");
                free(inertia_per_centroid);
                return FAILURE;
            }
            size_t cluster_count = 0;
            for (size_t sample_index = 0; sample_index < dataset->number_of_samples; ++sample_index)
            {
                if (dataset->samples[sample_index].centroid == &centroids->samples[highest_inertia_index])
                {
                    cluster_samples[cluster_count++] = &dataset->samples[sample_index];
                }
            }

            // Temp dataset for search (no malloc for Dataset, just stack init)
            Dataset temp_cluster = { .samples = *cluster_samples, .number_of_samples = cluster_count };

            size_t local_index = find_farthest_sample(&centroids->samples[highest_inertia_index], &temp_cluster, number_of_features);

            // Reassign
            cluster_samples[local_index]->centroid = &centroids->samples[centroid_index];

            // Update counts (old is always highest_inertia_index)
            --samples_per_centroid[highest_inertia_index];
            ++samples_per_centroid[centroid_index];

            ++revived;
            free(cluster_samples);  // Free just the array of pointers, using free_dataset() frees features too!
        }
    }
    *out_revived_count = revived;
    return SUCCESS;
}

/**
 * @brief Adjusts each centroid based on the mean value of their samples.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param centroids A pointer to a `Dataset` object containing the array of centroids and its count.
 * @param number_of_features The number of features.
 * @param threshold The minimum amount of distance the new centroid's features should be for it to move .
 *
 * @returns The number of centroids that have changed features (>threshold).
 *
 * @note `threshold` should be between 0 and 1.
 */
int adjust_centroids(const Dataset *dataset, Dataset *centroids, size_t number_of_features, double threshold, size_t *out_centroids_adjusted)
{
    Dataset *samples_per_centroid = calloc(centroids->number_of_samples, sizeof(Dataset));
    if (samples_per_centroid == NULL)
    {
        perror("[adjust_centroids] Failed to allocate memory for samples_per_centroid & initialize all members to 0\n");
        return FAILURE;
    }

    for (size_t sample_index = 0; sample_index < dataset->number_of_samples; ++sample_index)
    {
        size_t centroid_index = 0;
        while(centroid_index < centroids->number_of_samples && dataset->samples[sample_index].centroid != &centroids->samples[centroid_index])
        {
            ++centroid_index;
        }

        if (centroid_index == centroids->number_of_samples)
        {
            fprintf(stderr, "[adjust_centroids] No centroid assigned to this sample. Call `assign_centroid_to_samples` first\n");
            free_dataset(samples_per_centroid);
            return FAILURE;
        }

        Sample *temp_samples = realloc(samples_per_centroid[centroid_index].samples, (samples_per_centroid[centroid_index].number_of_samples + 1) * sizeof(Sample));
        if (temp_samples == NULL)
        {
            perror("[adjust_centroids] Failed to re-allocate memory for samples_per_centroid[centroid_index].samples\n");
            free_dataset(samples_per_centroid);
            return FAILURE;
        }
        samples_per_centroid[centroid_index].samples = temp_samples;
        samples_per_centroid[centroid_index].samples[samples_per_centroid[centroid_index].number_of_samples] = dataset->samples[sample_index];
        ++samples_per_centroid[centroid_index].number_of_samples;
    }

    size_t adjustements = 0;
    for (size_t centroid_index = 0; centroid_index < centroids->number_of_samples; ++centroid_index)
    {
        if (samples_per_centroid[centroid_index].number_of_samples == 0)
        {
            continue;
        }

        Sample new_centroid_position;
        new_centroid_position.features = calculate_mean(&samples_per_centroid[centroid_index], number_of_features);
        if (new_centroid_position.features == NULL)
        {
            // Do NOT free the actual samples (features) used by the main Dataset!
            free(samples_per_centroid[centroid_index].samples);
            free(samples_per_centroid);
            return FAILURE;
        }

        if (calculate_euclidian_distance(&centroids->samples[centroid_index], &new_centroid_position, number_of_features) > threshold)
        {
            free(centroids->samples[centroid_index].features); // Memory leak if not freed...
            centroids->samples[centroid_index].features = new_centroid_position.features;
            ++adjustements;
        }
        else
        {
            // Not used anymore, free the memory
            free(new_centroid_position.features);
            new_centroid_position.features = NULL;
        }

        // Free the top-level pointer for samples
        free(samples_per_centroid[centroid_index].samples); // Do NOT free the actual samples used by the main Dataset!
    }

    // Finally, free the top-level pointer
    free(samples_per_centroid);

    *out_centroids_adjusted = adjustements;
    return SUCCESS;
}

/**
 * @brief Assigns centroids to their nearest samples.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param centroids A pointer to a `Dataset` object containing the array of centroids and its count.
 * @param number_of_features The number of features.
 *
 * @returns The number of samples assigned to each centroid.
 */
size_t *assign_centroid_to_samples(Dataset *dataset, const Dataset *centroids, size_t number_of_features)
{
    if (centroids->number_of_samples == 0)
    {
        perror("[assign_centroid_to_samples] Centroids' size is 0. Cannot assign any samples\n");
        return NULL;
    }

    size_t *samples_per_centroid = calloc(centroids->number_of_samples, sizeof(size_t));
    if (samples_per_centroid == NULL)
    {
        perror("[assign_centroid_to_samples] Failed to allocate memory for samples_per_centroid & initialize all members to 0\n");
        return NULL;
    }

    for (size_t sample_index = 0; sample_index < dataset->number_of_samples; ++sample_index)
    {
        size_t nearest_centroid_index = find_nearest_point(&dataset->samples[sample_index], centroids, number_of_features);
        dataset->samples[sample_index].centroid = &centroids->samples[nearest_centroid_index];
        ++samples_per_centroid[nearest_centroid_index];
    }

    return samples_per_centroid;
}

/**
 * @brief Outputs the features values of all samples.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param number_of_features The number of features.
 *
 * @note I wrote this for easier debugging...
 */
void output_all_sample_features(const Dataset *dataset, size_t number_of_features)
{
    printf("\n-- SAMPLES OUTPUT --\n");
    for (size_t sample_index = 0; sample_index < dataset->number_of_samples; ++sample_index)
    {
        printf("\n[Sample %zu] Features:\n\n", sample_index);
        for (size_t feature_index = 0; feature_index < number_of_features; ++feature_index)
        {
            printf("- Feature %zu: %g\n",
                feature_index,
                dataset->samples[sample_index].features[feature_index]);
        }
    }
    printf("\nIteration finished.\n\n");
}

/**
 * @brief Outputs the distance difference after the centroids were adjusted.
 *
 * @param centroids A pointer to a `Dataset` object containing the array of centroids and its count.
 * @param old_centroids A pointer to a `Dataset` object containing the array of centroids at their older state and its count.
 * @param samples_per_centroid The number of samples assigned to each centroid.
 * @param number_of_features The number of features.
 *
 * @note I wrote this for easier debugging...
 *
 * You can pass NULL for `old_centroids` and/or `samples_per_centroid` if they're not needed.
 */
int output_centroids(const Dataset *centroids, const Dataset *old_centroids, const size_t *samples_per_centroid, size_t number_of_features, bool save_output)
{
    FILE *file = NULL;
    if (save_output)
    {
        file = fopen("output.txt", "a");
        if (file == NULL)
        {
            fprintf(stderr, "[output_centroids] Failed to open file 'output.txt'\n");
            return FAILURE;
        }
        fprintf(file, "K-Means Iteration Output for K=%zu:\n", centroids->number_of_samples);
    }

    printf("\n### OUTPUT ###\n");
    if (save_output) fprintf(file, "\n### OUTPUT ###\n");

    for (size_t centroid_index = 0; centroid_index < centroids->number_of_samples; ++centroid_index)
    {
        double distance = (old_centroids != NULL) ? calculate_euclidian_distance(
            &old_centroids->samples[centroid_index],
            &centroids->samples[centroid_index],
            number_of_features
        ) : 0;

        printf("\n[Centroid %zu] Features:\n", centroid_index);
        if (save_output) fprintf(file, "\n[Centroid %zu] Features:\n", centroid_index);

        for (size_t feature_index = 0; feature_index < number_of_features; ++feature_index)
        {
            if (old_centroids != NULL)
            {
                printf(
                    "- Feature %zu: %g (old) -> %g (new)\n",
                    feature_index,
                    old_centroids->samples[centroid_index].features[feature_index],
                    centroids->samples[centroid_index].features[feature_index]
                );
                if (save_output) fprintf(
                    file,
                    "- Feature %zu: %g (old) -> %g (new)\n",
                    feature_index,
                    old_centroids->samples[centroid_index].features[feature_index],
                    centroids->samples[centroid_index].features[feature_index]
                );
            }
            else
            {
                printf(
                    "- Feature %zu: %g\n",
                    feature_index,
                    centroids->samples[centroid_index].features[feature_index]
                );
                if (save_output) fprintf(
                    file,
                    "- Feature %zu: %g\n",
                    feature_index,
                    centroids->samples[centroid_index].features[feature_index]
                );
            }
        }

        if (samples_per_centroid != NULL)
        {
            printf("\nSamples assigned: %zu\n", samples_per_centroid[centroid_index]);
            if (save_output) fprintf(file, "\nSamples assigned: %zu\n", samples_per_centroid[centroid_index]);
        }
        if (old_centroids != NULL)
        {
            printf("\n- Distance difference: %g\n\n", distance);
            if (save_output) fprintf(file, "\n- Distance difference: %g\n\n", distance);
        }
    }
    printf("\n");
    if (file != NULL)
    {
        fprintf(file, "\n\n\n");
        fclose(file);
    }

    return SUCCESS;
}

/**
 * @brief Clones a `Dataset` object.
 *
 * @param source A pointer to a `Dataset` object which will be cloned.
 * @param number_of_features The number of features.
 *
 * @returns A pointer to a `Dataset` object, a clone of `source`.
 */
Dataset *clone_dataset_object(const Dataset *source, size_t number_of_features)
{
    Dataset *destination = malloc(sizeof(Dataset));
    if (destination == NULL)
    {
        perror("[clone_dataset_object] Failed to allocate memory for destination\n");
        return NULL;
    }
    destination->number_of_samples = source->number_of_samples;
    destination->samples = malloc(destination->number_of_samples * sizeof(Sample));

    for (size_t sample_index = 0; sample_index < destination->number_of_samples; ++sample_index) {

        destination->samples[sample_index].features = malloc(number_of_features * sizeof(double));
        if (destination->samples[sample_index].features == NULL)
        {
            fprintf(stderr, "[clone_dataset_object] Failed to allocate memory for destination->samples[%zu].features\n", sample_index);
            free_dataset(destination);
            return NULL;
        }

        memcpy(destination->samples[sample_index].features,
               source->samples[sample_index].features,
               number_of_features * sizeof(double));

        destination->samples[sample_index].centroid = NULL; // not used
    }
    return destination;
}

/**
 * @brief Creates a`Dataset`object containing K randomly initialized centroids.
 *
 * @param number_of_features The number of features (dimensions) for each centroid.
 * @param K The number of centroids to create.
 *
 * @returns A`Dataset`object containing K randomly initialized centroids.
 */
Dataset *create_random_centroids(size_t number_of_features, size_t K)
{
    Dataset *centroids = malloc(sizeof(Dataset));
    if (centroids == NULL)
    {
        perror("[create_random_centroids] Failed to allocate memory for centroids\n");
        return NULL;
    }
    centroids->samples = malloc(K * sizeof(Sample));
    if (centroids->samples == NULL)
    {
        perror("[create_random_centroids] Failed to allocate memory for centroids->samples\n");
        free_dataset(centroids);
        return NULL;
    }
    centroids->number_of_samples = K;

    for (size_t centroid_index = 0; centroid_index < K; ++centroid_index)
    {
        centroids->samples[centroid_index].features = malloc(number_of_features * sizeof(double));
        if (centroids->samples[centroid_index].features == NULL)
        {
            fprintf(stderr, "[create_random_centroids] Failed to allocate memory for centroids->samples[%zu].features\n", centroid_index);
            free_dataset(centroids);
            return NULL;
        }
        centroids->samples[centroid_index].centroid = NULL;

        for (size_t feature_index = 0; feature_index < number_of_features; ++feature_index)
        {
            double random = (double)rand() / RAND_MAX;
            centroids->samples[centroid_index].features[feature_index] = global_max_feature * random + global_min_feature; // (((double)rand() / RAND_MAX) * 2 - 1)
        }
    }
    return centroids;
}

/**
 * @brief Runs the K-Means clustering algorithm until centroids converge.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param centroids A pointer to a `Dataset` object containing the array of centroids and its count.
 * @param number_of_features The number of features.
 * @param tolerance The minimum amount of distance the new centroid's features should be for it to move.
 *
 * @note This function stops when centroids no longer move significantly.
 */
size_t *run_kmeans_to_convergence(Dataset *dataset, Dataset *centroids, size_t number_of_features, double tolerance)
{
    size_t *samples_per_centroid = NULL;
    size_t centroids_adjusted;
    do
    {
        size_t revived_centroids = 0; // Reset for the new iteration

        samples_per_centroid = assign_centroid_to_samples(dataset, centroids, number_of_features);
        if (samples_per_centroid == NULL)
        {
            fprintf(stderr, "[run_kmeans_to_convergence] Failed to assign centroid to samples\n");
            return NULL;
        }

        if (handle_empty_centroids(dataset, centroids, samples_per_centroid,
            number_of_features, &revived_centroids) == FAILURE)
        {
            fprintf(stderr, "[run_kmeans_to_convergence] Failed to handle empty centroids\n");
            free(samples_per_centroid);
            return NULL;
        }

        // Track modifications by keeping a temporary backup of the data before adjustments are made
        //Dataset *centroids_clone = clone_dataset_object(centroids, number_of_features);

        // Number of adjusted centroids
        
        if (adjust_centroids(dataset, centroids, number_of_features,
            tolerance, &centroids_adjusted) == FAILURE)
        {
            fprintf(stderr, "");
            free(samples_per_centroid);
            return NULL;
        }

        //output_centroids(centroids, centroids_clone, samples_per_centroid, number_of_features);
        
        //free_dataset(centroids_clone);
        
        // Keep the samples-per-centroid data if adjustements were made & escape loop
        if (revived_centroids == 0 && centroids_adjusted == 0) break;
        free(samples_per_centroid);
        samples_per_centroid = NULL;
    }
    while (true);
    return samples_per_centroid;
}

/**
 * @brief Finds the 'best' K value.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param number_of_features The number of features.
 * @param K_begin The number of centroids to start with.
 * @param K_end The number of centroids the loop ends with.
 * @param tolerance The minimum amount of distance the new centroid's features should be for it to move.
 * @param drop_threshold This is the inertia's percentage difference (0.01 = 1%) between current and last execution.
 * @param restarts To avoid as much as possible a bad local minimum with high inertia,
 * restart the algorithm and pick the best.
 *
 * @note This function executes the K-means algorithm in a loop,
 * with K in {`K_begin`, ..., `K_end`}.
 */
int elbow_method(Dataset *dataset, size_t number_of_features, size_t K_begin, size_t K_end, double tolerance, double drop_threshold, int restarts, bool save_output)
{
    int answer_reach_max_K = 0;
    double last_inertia = -1.0;

    for (size_t K = K_begin; K <= K_end; ++K)
    {
        double best_inertia = __DBL_MAX__;
        Dataset *best_result_centroids = NULL;
        size_t *best_result_samples_per_centroid = NULL;

        for (int r = 0; r < restarts; ++r)
        {
            Dataset *centroids = create_random_centroids(number_of_features, K);
            if (centroids == NULL)
            {
                fprintf(stderr, "");
                return FAILURE;
            }
            size_t *samples_per_centroid = run_kmeans_to_convergence(dataset, centroids, number_of_features, tolerance);
            if (samples_per_centroid == NULL)
            {
                fprintf(stderr, "");
                free_dataset(centroids);
                return FAILURE;
            }
            double *inertia_per_centroid = calculate_inertia(dataset, centroids, number_of_features);
            if (centroids == NULL)
            {
                fprintf(stderr, "");
                free(samples_per_centroid);
                free_dataset(centroids);
                return FAILURE;
            }

            double inertia = 0; // total inertia
            for (size_t centroid_index = 0; centroid_index < centroids->number_of_samples; ++centroid_index)
            {
                inertia += inertia_per_centroid[centroid_index];
            }
            free(inertia_per_centroid);

            if (best_inertia > inertia)
            {
                if (best_result_centroids != NULL)
                {
                    free_dataset(best_result_centroids);
                    free(best_result_samples_per_centroid);
                }

                best_inertia = inertia;
                best_result_centroids = centroids;
                best_result_samples_per_centroid = samples_per_centroid;
            }
            else
            {
                free_dataset(centroids);
                free(samples_per_centroid);
            }
        }
        if (output_centroids(best_result_centroids, NULL, 
            best_result_samples_per_centroid, number_of_features, save_output) == FAILURE)
        {
            fprintf(stderr, "[elbow_method] Failed to output centroids info");
            if (best_result_samples_per_centroid != NULL) free(best_result_samples_per_centroid);
            free_dataset(best_result_centroids);
            return FAILURE;
        }

        // Output the calculated total Inertia (WCSS) value
        printf("K = %zu, Inertia = %.6f, Per-Sample SSE = %g\n", K, best_inertia, best_inertia/dataset->number_of_samples);

        if (last_inertia > 0)
        {
            // How much did the inertia value drop by
            double drop_K = (last_inertia - best_inertia) / last_inertia; // 0.1 means 10%

            if (drop_K < drop_threshold)
            {
                printf("Found inertia gains under the drop threshold with K value: %zu (Drop threshold: %g)\n\n", K, drop_threshold);
                if (!global_args.force_reach_K)
                {
                    if (best_result_samples_per_centroid != NULL) free(best_result_samples_per_centroid);
                    free_dataset(best_result_centroids);
                    break;
                }
            }
        }
        last_inertia = best_inertia;
        free_dataset(best_result_centroids);
        if (best_result_samples_per_centroid != NULL) free(best_result_samples_per_centroid);
    }

    return SUCCESS;
}

void print_usage_examples(const char* program_name)
{
    printf("=== %s Usage Examples ===\n", program_name);

    printf("\n1. Run KMeans on a dataset with 3 features per sample\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/kmeans/train -n 3");

    printf("\n2. Standardize features before clustering\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/kmeans/train -n 3 --std");

    printf("\n3. Set K manually to 5 clusters\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/kmeans/train -n 3 -k 5");

    printf("\n4. Use elbow method to find optimal K automatically\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/kmeans/train -n 3 --elbow");

    printf("\n5. Use elbow method with forced max K = 15\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/kmeans/train -n 3 --elbow --force-k -e 15");

    printf("\n6. Fine-tune algorithm with smaller centroid movement threshold and higher accuracy\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/kmeans/train -n 3 -t 0.0001 -r 10");

    printf("\n7. Save output to current directory\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/kmeans/train -n 3 -o");

    printf("\n8. Custom token & decimal delimiters\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/kmeans/train.ssv -n 3 --tdel ; --ddel ,");

    printf("\n9. Run a demo using the repo's sample dataset:\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/kmeans/test -n 3 -k 5 --std --elbow");

    printf("\n10. Show all options and usage info\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "help");

    printf("\nNotes:\n");
    printf("- Options order does not matter.\n");
    printf("- Combine multiple options as needed (e.g., training + standardization + elbow method).\n");
}

void print_usage(const char *program_name)
{
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -f <string>           Dataset file path\n");
    printf("  -n <integer>          Number of features per sample (default: 1)\n");
    printf("  -t <float>            Minimum value for a centroid's movement to be recognized (default: 0.001)\n");
    printf("  -r <integer>          Higher accuracy but slower (default: 5)\n");
    printf("  -k <integer>          Set the K value (default: 1)\n");
    printf("  -e <integer>          End algorithm at K_end value, used with elbow method (default: 10)\n");
    printf("  -d <float>            Difference between the inertia values of K-1 and K, used with elbow method (default: 0.1 meaning 10%%)\n");
    printf("  -o                    Save output in current directory as 'output.txt' (false by default)\n");
    printf("  --std                 Standardize all sample features (false by default)\n");
    printf("  --elbow               Use the elbow method to find the most optimal K (false by default)\n");
    printf("  --force-k             Continue until reaching the max K desired (K_end), used with elbow method (false by default)\n");
    printf("  --tdel <string>       Token delimiter (default: space)\n");
    printf("  --ddel <char>         Decimal delimiter (default: .)\n");
    printf("  --cdel <string>       Comment delimiter (default: #)\n");
    printf("\n");
    printf("  -v, --debug           Verbose, used for debugging (false by default)\n");
    printf("  examples              Shows a list of usage examples\n");
    printf("  help                  Usage and options menu (this command)\n");
}

Args parse_args(int argc, char **argv)
{
    Args args;

    // Default values
    strncpy(args.dataset_path, "", MAX_PATH-1);
    args.number_of_features = 1;
    args.tolerance = 0.001;
    args.restarts = 1;
    args.K = 1;
    args.K_end = 1;
    args.drop_threshold = 0.1;
    args.standardize = false;
    args.save_output = false;
    args.use_elbow_method = false;
    args.force_reach_K = false;
    strncpy(args.token_delimiter, " ", MAX_DELIM-1);
    args.decimals_delimiter = '.';
    strncpy(args.comment_delimiter, "#", MAX_DELIM-1);
    args.debug = false;

    bool elbow_auto_enabled = false;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-f") == 0 && i+1 < argc)
            strncpy(args.dataset_path, argv[++i], MAX_PATH-1);
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc)
        {
            char *endptr;
            args.number_of_features = strtoul(argv[++i], &endptr, 10);
            if (*endptr != '\0') {
                fprintf(stderr, "Invalid number of features: %s\n", argv[i]);
                exit(EXIT_FAILURE);
            }
        }
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc)
        {
            char *endptr;
            args.tolerance = strtod(argv[++i], &endptr);
            if (*endptr != '\0') {
                fprintf(stderr, "Invalid tolerance value: %s\n", argv[i]);
                exit(EXIT_FAILURE);
            }
        }
        else if (strcmp(argv[i], "-r") == 0 && i+1 < argc)
        {
            char *endptr;
            args.restarts = strtoul(argv[++i], &endptr, 10);
            if (*endptr != '\0') {
                fprintf(stderr, "Invalid restarts value: %s\n", argv[i]);
                exit(EXIT_FAILURE);
            }
        }
        else if (strcmp(argv[i], "-k") == 0 && i+1 < argc)
        {
            char *endptr;
            args.K = strtoul(argv[++i], &endptr, 10);
            if (*endptr != '\0') {
                fprintf(stderr, "Invalid K value: %s\n", argv[i]);
                exit(EXIT_FAILURE);
            }
        }
        else if (strcmp(argv[i], "-e") == 0 && i+1 < argc)
        {
            char *endptr;
            args.K_end = strtoul(argv[++i], &endptr, 10);
            if (*endptr != '\0') {
                fprintf(stderr, "Invalid max K value: %s\n", argv[i]);
                exit(EXIT_FAILURE);
            }
            args.use_elbow_method = true; // enable automatically
            elbow_auto_enabled = true;
        }
        else if (strcmp(argv[i], "-d") == 0 && i+i < argc)
        {
            char *endptr;
            args.drop_threshold = strtod(argv[++i], &endptr);
            if (*endptr != '\0') {
                fprintf(stderr, "Invalid value for drop threshold: %s\n", argv[i]);
                exit(EXIT_FAILURE);
            }
            args.use_elbow_method = true; // enable automatically
            elbow_auto_enabled = true;
        }
        else if (strcmp(argv[i], "-o") == 0)
            args.save_output = true;
        else if (strcmp(argv[i], "--std") == 0)
            args.standardize = true;
        else if (strcmp(argv[i], "--elbow") == 0)
            args.use_elbow_method = true;
        else if (strcmp(argv[i], "--force-k") == 0)
        {
            args.force_reach_K = true;
            args.use_elbow_method = true; // enable automatically
            elbow_auto_enabled = true;
        }
        else if (strcmp(argv[i], "--tdel") == 0 && i+1 < argc)
            strncpy(args.token_delimiter, argv[++i], MAX_DELIM-1);
        else if (strcmp(argv[i], "-ddel") == 0 && i+1 < argc)
            args.decimals_delimiter = argv[++i][0];
        else if (strcmp(argv[i], "-cdel") == 0 && i+1 < argc)
            strncpy(args.comment_delimiter, argv[++i], MAX_DELIM-1);
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--debug") == 0)
            args.debug = true;
        else if (strcmp(argv[i], "examples") == 0)
        {
            print_usage_examples(argv[0]);
            exit(EXIT_SUCCESS);
        }
        else
        {
            print_usage(argv[0]);
            exit((strcmp(argv[i], "help") == 0) ? EXIT_SUCCESS : EXIT_FAILURE);
        }
    }
    if (elbow_auto_enabled)
        printf("Elbow method enabled automatically (used an arg that requires --elbow)\n");
    if (args.use_elbow_method && args.K_end < args.K)
    {
        args.K_end = args.K * 2;
        printf("Detected a bigger K value than max K. Using %zu (K*2) for max K\n", args.K_end);
    }
    return args;
}

int main(int argc, char *argv[])
{
    // Parse arguments
    global_args = parse_args(argc, argv);
    if (global_args.debug)
    {
        printf("Debugging: true\n");
        printf("\n");
        printf("Dataset path: %s\n",  global_args.dataset_path);
        printf("Number of features: %zu\n",  global_args.number_of_features);
        printf("Tolerance: %g\n",  global_args.tolerance);
        printf("Number of restarts: %zu\n",  global_args.restarts);
        printf("K value: %zu\n",  global_args.K);
        printf("Max K: %zu%s\n", global_args.K_end, global_args.use_elbow_method ? "" : " (unused)");
        printf("Drop threshold: %g%s\n",  global_args.drop_threshold, global_args.use_elbow_method ? "" : " (unused)");
        printf("Standardize: %s\n", global_args.standardize ? "true" : "false");
        printf("Save output: %s\n", global_args.save_output ? "true" : "false");
        printf("Use elbow method: %s\n", global_args.use_elbow_method ? "true" : "false");
        printf("Force reach K: %s%s\n", global_args.force_reach_K ? "true" : "false", global_args.use_elbow_method ? "" : " (unused)");
        printf("Token delimiter: '%s'\n", global_args.token_delimiter);
        printf("Decimal delimiter: '%c'\n", global_args.decimals_delimiter);
        printf("Comment delimiter: '%s'\n", global_args.comment_delimiter);
        printf("\n");
    }
    
    // Initialize the randomizer using the current timestamp as seed
    srand((unsigned int)time(NULL));

    if (global_args.debug) printf("[main] Loading dataset\n");
    Dataset *dataset = read_data_from_file(
        global_args.dataset_path, global_args.number_of_features,
        global_args.token_delimiter, global_args.decimals_delimiter, global_args.comment_delimiter
    );
    if (dataset == NULL)
    {
        fprintf(stderr, "[main] Failed to load dataset\n");
        return EXIT_FAILURE;
    }

    double *mean = calculate_mean(dataset, global_args.number_of_features); // freed at the end of main function
    if (mean == NULL)
    {
        free_dataset(dataset);
        return EXIT_FAILURE;
    }
    double *standard_deviation = calculate_std_deviation(
        dataset, mean, global_args.number_of_features
    );  // freed at the end of main function
    if (standard_deviation == NULL)
    {
        fprintf(stderr, "[main] Failed to calculate standard deviation\n");
        free(mean);
        free_dataset(dataset);
        return EXIT_FAILURE;
    }

    if (global_args.standardize)
    {
        if (global_args.debug) printf("[main] Standardizing sample features\n");
        if (standardize_data(dataset, mean, standard_deviation, global_args.number_of_features) == FAILURE)
        {
            fprintf(stderr, "[main] Failed to standardize data\n");
            free(standard_deviation);
            free(mean);
            free_dataset(dataset);
            return EXIT_FAILURE;
        }
        if (global_args.debug) printf("[main] Done\n");
    }
    if (global_args.debug) output_all_sample_features(dataset, global_args.number_of_features); // For debugging

    if (global_args.K == 0)
    {
        fprintf(stderr, "[main] Failed to run K-means algorithm: K value equals 0.\n");
        free(standard_deviation);
        free(mean);
        free_dataset(dataset);
        return EXIT_FAILURE;
    }
    else if (global_args.K >= dataset->number_of_samples)
    {
        fprintf(
            stderr,
            "[main] Failed to run K-means algorithm: K value (%zu) is bigger or equal to the total number of samples.\n",
            global_args.K
        );
        return EXIT_FAILURE;
    }

    if (global_args.use_elbow_method)
    {
        if (global_args.debug) printf("[main] Executing elbow method\n");
        if (elbow_method(
            dataset, global_args.number_of_features,
            global_args.K, global_args.K_end,
            global_args.tolerance, global_args.drop_threshold,
            global_args.restarts, global_args.save_output) == FAILURE)
        {
            fprintf(stderr, "[main] An error occured while executing the elbow method\n");
            free(standard_deviation);
            free(mean);
            free_dataset(dataset);
            return EXIT_FAILURE;
        }
        if (global_args.debug) printf("[main] Finished\n");
    }
    else // Don't use the elbow method
    {
        if (global_args.debug)
            printf("[main] Running the algorithm with K value %zu (restarts: %zu)\n", global_args.K, global_args.restarts);
        
        double best_inertia = __DBL_MAX__;
        Dataset *best_result_centroids = NULL;
        size_t *best_result_samples_per_centroid = NULL;
        for (size_t r = 0; r < global_args.restarts; ++r)
        {
            if (global_args.debug) printf("[main] Loop number %zu\n", r);

            if (global_args.debug) printf("[main] Creating random centroids\n");
            Dataset *centroids = create_random_centroids(global_args.number_of_features, global_args.K);
            if (centroids == NULL)
            {
                free(standard_deviation);
                free(mean);
                free_dataset(dataset);
                return EXIT_FAILURE;
            }
            if (global_args.debug) printf("[main] Running the algorithm until all centroids converge\n");
            size_t *samples_per_centroid = run_kmeans_to_convergence(
                dataset, centroids, global_args.number_of_features, global_args.tolerance
            );
            if (centroids == NULL)
            {
                free_dataset(centroids);
                free(standard_deviation);
                free(mean);
                free_dataset(dataset);
                return EXIT_FAILURE;
            }

            if (global_args.debug) printf("[main] Calculating inertia per centroid\n");
            double *inertia_per_centroid = calculate_inertia(dataset, centroids, global_args.number_of_features);
            if (inertia_per_centroid == NULL)
            {
                free(samples_per_centroid);
                free_dataset(centroids);
                free(standard_deviation);
                free(mean);
                free_dataset(dataset);
                return EXIT_FAILURE;
            }

            double inertia = 0; // total inertia
            for (size_t centroid_index = 0; centroid_index < centroids->number_of_samples; ++centroid_index)
                inertia += inertia_per_centroid[centroid_index];
            if (global_args.debug) printf("[main] Total inertia calculated: %g\n", inertia);

            free(inertia_per_centroid);

            if (best_inertia > inertia)
            {
                if (global_args.debug) printf("[main] Smaller inertia found (inertia < best_inertia)\n");
                if (best_result_centroids != NULL)
                {
                    free_dataset(best_result_centroids);
                    free(best_result_samples_per_centroid);
                }

                best_inertia = inertia;
                best_result_centroids = centroids;
                best_result_samples_per_centroid = samples_per_centroid;
            }
            else
            {
                free_dataset(centroids);
                free(samples_per_centroid);
            }
        }
        if (output_centroids(
            best_result_centroids, NULL, best_result_samples_per_centroid,
            global_args.number_of_features, global_args.save_output) == FAILURE)
        {
            free(standard_deviation);
            free(mean);
            free_dataset(dataset);
            return EXIT_FAILURE;
        }

        // Output the calculated total Inertia (WCSS) value
        printf(
            "K = %zu, Inertia = %.6f, Per-Sample SSE = %g\n",
            global_args.K, best_inertia, best_inertia / dataset->number_of_samples
        ); // K value, Total inertia value, AVG per-sample inertia

        free_dataset(best_result_centroids);
        free(best_result_samples_per_centroid);
    }

    free(standard_deviation);
    free(mean);
    free_dataset(dataset);
    return EXIT_SUCCESS;
}
