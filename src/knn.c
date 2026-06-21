/*======================================================================
 *  k-Nearest Neighbors (lazy learner; supervised)
 *
 *  Author      :  DeprecatedLogic  <https://www.github.com/DeprecatedLogic>
 *  Created     :  25 Jun 2025
 *
 *  Description :
 *      Reads N-dimensional samples from a text file (final token = label),
 *      standardises the data in-place, returns the K nearest
 *      neighbors for a query sample and outputs what the sample
 *      was classified as.
 *====================================================================*/

#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
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
    double *features; // A pointer containing double values to keep the number of features dynamic
    char *label; // The label of the sample (what it's classified as)
}
Sample;

// Keep the array of samples and its size together
typedef struct
{
    Sample *samples;
    size_t number_of_samples;
}
Dataset;

// Facilitates the counting of different labels
// and their respective frequencies
typedef struct
{
    char **labels;
    size_t *frequencies;
    size_t number_of_labels;
}
LabelStats;

typedef struct
{
    char dataset_path[MAX_PATH]; // Dataset file path
    char acc_test_dataset_path[MAX_PATH]; // Test the accuracy of the algorithm with a labeled dataset & specific K value
    char prediction_dataset_path[MAX_PATH]; // Unlabeled dataset file path containing samples to predict/label
    char prediction_output_path[MAX_PATH]; // Save output at this path
    size_t number_of_features;
    size_t K; // K neighbors; if `K >= samples available`, it's simply a majority vote between all samples...
    bool standardize; // Standardize all sample features (false by default)
    bool manual_input; // Enter samples to predict manually, usually used for quick tests (false by default)
    char token_delimiter[MAX_DELIM];
    char decimals_delimiter;
    char comment_delimiter[MAX_DELIM];
    bool debug; // Verbose, used for debugging (false by default)
} Args;
Args global_args;

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
            if (dataset->samples[sample_index].label != NULL)
                free(dataset->samples[sample_index].label);
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
    const char *file_path, size_t number_of_features, bool is_labeled,
    char *token_delimiter, char decimals_delimiter, char *comment_delimiter
)
{
    if (token_delimiter == NULL) strcpy(token_delimiter, " ");
    if (comment_delimiter == NULL) strcpy(comment_delimiter, "#");
    if (strcmp(token_delimiter, comment_delimiter) == 0)
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
        fprintf(stderr, "[read_data_from_file] Failed to allocate memory for dataset\n");
        fclose(data_file);
        return NULL;
    }

    const size_t feature_max_digits = 40; // a max of 40 digits per feature should be enough (the maximum precision for double is 15 or 17 ?)
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
            .label = NULL
        };
        if (sample.features == NULL)
        {
            fprintf(stderr, "[read_data_from_file] Failed to allocate memory for sample.features\n");
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
            else if (is_labeled)
            {
                sample.label = strdup(token_buffer);
            }

            token_buffer = strtok(NULL, token_delimiter);
            ++token_counter;
        }
        if (token_counter-1 != number_of_features) // -1 because of the label
        {
            fprintf(
                stderr,
                "[read_data_from_file] Encountered a mismatch: features counted [%zu] != expected [%zu] (line %zu)\n",
                token_counter-1,
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
 * @brief Makes a deep copy of a `Dataset` object.
 *
 * @param dataset A pointer to the `Dataset` object to copy.
 * @param number_of_features The number of features.
 *
 * @returns A pointer to a newly allocated and copied `Dataset` object.
 */
Dataset *deep_copy_dataset(const Dataset *dataset, size_t number_of_features)
{
    Dataset *copy = malloc(sizeof(Dataset));
    if (copy == NULL)
    {
        fprintf(stderr, "[deep_copy_dataset] Failed to allocate memory for copy\n");
        return NULL;
    }

    copy->samples = malloc(dataset->number_of_samples * sizeof(Sample));
    if (copy->samples == NULL)
    {
        fprintf(stderr, "[deep_copy_dataset] Failed to allocate memory for copy->samples\n");
        free_dataset(copy);
        return NULL;
    }
    copy->number_of_samples = dataset->number_of_samples;

    for (size_t sample_index = 0; sample_index < dataset->number_of_samples; ++sample_index)
    {
        copy->samples[sample_index].features = malloc(number_of_features * sizeof(double));
        if (copy->samples[sample_index].features == NULL)
        {
            fprintf(stderr, "[deep_copy_dataset] Failed to allocate memory for copy->samples[sample_index].features\n");
            free_dataset(copy);
            return NULL;
        }
        if (dataset->samples[sample_index].label != NULL) copy->samples[sample_index].label = strdup(dataset->samples[sample_index].label);
        else copy->samples[sample_index].label = NULL;

        for (size_t feature_index = 0; feature_index < number_of_features; ++feature_index)
        {
            copy->samples[sample_index].features[feature_index] = dataset->samples[sample_index].features[feature_index];
        }
    }
    return copy;
}

/**
 * @brief Calculates the mean for each feature.
 *
 * @attention Assumes dataset->number_of_samples is greater than 0.
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
        fprintf(stderr, "[calculate_mean] Failed to allocate memory for mean\n");
        return NULL;
    }

    // Loop through the samples and sum the features respectively
    for (size_t sample_index = 0; sample_index < dataset->number_of_samples; ++sample_index)
    {
        for (size_t feature_index = 0; feature_index < number_of_features; ++feature_index)
        {
            mean[feature_index] += dataset->samples[sample_index].features[feature_index];
        }
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
 * @attention Assumes dataset->number_of_samples is greater than 0.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param mean An array containing the mean value of each feature.
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
        fprintf(stderr, "[calculate_std_deviation] Failed to allocate memory for standard_deviation\n");
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
 * @param number_of_features The number of features.
 */
int standardize_data(Dataset *dataset, const double *mean, const double *standard_deviation, size_t number_of_features)
{
    // Check size because there's a division with size as denominator
    if (dataset->number_of_samples == 0)
    {
        fprintf(stderr, "[standardize_data] Failed to standardize data. No data was found ? Avoided division by 0\n");
        return FAILURE;
    }

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
            dataset->samples[sample_index].features[feature_index] = (x - mu) / std;
        }
    }

    return SUCCESS;
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
 * @brief Finds the farthest sample in an array of samples.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param sample The main sample.
 * @param number_of_features The number of features.
 *
 * @returns An array containing the index and euclidian distance of the farthest sample in `dataset->samples`.
 */
double *find_farthest_sample(const Dataset *dataset, const Sample *sample, size_t number_of_features)
{
    // The index and euclidian distance of the farthest sample in the samples array
    // Initialized with 0s
    double *farthest_sample = calloc(2, sizeof(double));
    if (farthest_sample == NULL)
    {
        fprintf(stderr, "[find_farthest_sample] Failed to allocate memory for farthest_sample");
        return NULL;
    }

    for (size_t sample_index = 0; sample_index < dataset->number_of_samples; ++sample_index)
    {
        // Euclidean distance
        double euclidian_distance = calculate_euclidian_distance(sample, &dataset->samples[sample_index], number_of_features);

        // Store the sample's index and distance if necessary
        if (euclidian_distance > farthest_sample[1])
        {
            farthest_sample[0] = (double)sample_index;
            farthest_sample[1] = euclidian_distance;
        }
    }
    return farthest_sample;
}

/**
 * @brief Returns the K nearest samples (neighbors).
 *
 * @attention Assumes `K` is smaller or equal to the total number of samples.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param sample The main sample.
 * @param number_of_features The number of features.
 * @param K The number of the neighboring (nearest) samples to check for.
 *
 * @returns `Dataset *neighbors`, all the nearest neighbors of the `sample`.
 */
Dataset *get_K_nearest_neighbors(const Dataset *dataset, const Sample *sample, size_t number_of_features, size_t K)
{
    Dataset *neighbors = malloc(sizeof(Dataset));
    if (neighbors == NULL)
    {
        fprintf(stderr, "[get_K_nearest_neighbors] Failed to allocate memory for neighbors\n");
        return NULL;
    }

    neighbors->samples = malloc(K * sizeof(Sample));
    if (neighbors->samples == NULL)
    {
        fprintf(stderr, "[get_K_nearest_neighbors] Failed to allocate memory for neighbors->samples\n");
        free_dataset(neighbors);
        return NULL;
    }
    neighbors->number_of_samples = 0;

    for (size_t sample_index = 0; sample_index < dataset->number_of_samples; ++sample_index)
    {
        if (neighbors->number_of_samples >= K)
        {
            double *farthest_sample = find_farthest_sample(neighbors, sample, number_of_features);
            if (farthest_sample == NULL)
            {
                fprintf(stderr, "[get_K_nearest_neighbors] Failed to find farthest sample\n");
                free_dataset(neighbors);
                return NULL;
            }
            size_t farthest_sample_index = (size_t)farthest_sample[0];
            double farthest_sample_distance = farthest_sample[1];

            if (calculate_euclidian_distance(sample, &dataset->samples[sample_index], number_of_features) < farthest_sample_distance)
            {
                free(neighbors->samples[farthest_sample_index].label);
                if (dataset->samples[sample_index].label == NULL)
                {
                    fprintf(stderr, "[get_K_nearest_neighbors] Sample with index %zu has a missing label\n", sample_index);
                    free(farthest_sample);
                    free_dataset(neighbors);
                    return NULL;
                }
                neighbors->samples[farthest_sample_index].label = strdup(dataset->samples[sample_index].label);

                for (size_t feature_index = 0; feature_index < number_of_features; ++feature_index)
                {
                    neighbors->samples[farthest_sample_index].features[feature_index] = dataset->samples[sample_index].features[feature_index];
                }
            }
            free(farthest_sample);
        }
        else
        {
            neighbors->samples[neighbors->number_of_samples].features = malloc(number_of_features * sizeof(double));
            if (neighbors->samples[neighbors->number_of_samples].features == NULL)
            {
                fprintf(stderr, "[get_K_nearest_neighbors] Failed to allocate memory for neighbors->samples[%zu].features\n", neighbors->number_of_samples);
                free_dataset(neighbors);
                return NULL;
            }
            neighbors->samples[neighbors->number_of_samples].label = strdup(dataset->samples[sample_index].label);

            for (size_t feature_index = 0; feature_index < number_of_features; ++feature_index)
            {
                neighbors->samples[neighbors->number_of_samples].features[feature_index] = dataset->samples[sample_index].features[feature_index];
            }
            ++neighbors->number_of_samples;
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
    size_t *label_frequencies = NULL;
    size_t labels_size = 0;

    for (int sample_index = 0; sample_index < dataset->number_of_samples; ++sample_index)
    {
        // Find if the label already exists & increment that label's counter
        size_t label_index = 0;
        while (label_index < labels_size && strcmp(dataset->samples[sample_index].label, labels[label_index]) != 0)
        {
            ++label_index;
        }

        if (label_index < labels_size)
        {
            ++label_frequencies[label_index];
        }
        else // Increase the size of the array if needed & store the label
        {
            // Apparently, `realloc(NULL, size) is the same as malloc(size)` :)
            char **temp_labels = realloc(labels, (labels_size + 1) * sizeof(char*));
            if (temp_labels == NULL && labels != NULL)
            {
                for (size_t label_index = 0; label_index < labels_size; ++label_index)
                    free(labels[label_index]);
                free(labels);
            }
            labels = temp_labels;

            size_t *temp_label_frequencies = realloc(label_frequencies, (labels_size + 1) * sizeof(size_t));
            if (temp_label_frequencies == NULL && label_frequencies != NULL)
                free(label_frequencies);
            label_frequencies = temp_label_frequencies;

            if (temp_labels != NULL && temp_label_frequencies != NULL)
            {    
                labels[labels_size] = strdup(dataset->samples[sample_index].label); // Copy the sample's label
                label_frequencies[labels_size] = 1; // Equals 1 instead of incrementing because it's a NEW data label, with 1 sample
                ++labels_size;
            }
            else
            {
                fprintf(stderr, "[get_label_stats] NULL pointer(s) after memory re-allocation at sample_index %d.\n", sample_index);
                return (LabelStats){
                    .labels = NULL,
                    .frequencies = NULL,
                    .number_of_labels = 0
                };
            }
        }
    }
    return (LabelStats){
        .labels = labels,
        .frequencies = label_frequencies,
        .number_of_labels = labels_size
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
    if (stats.labels == NULL)
    {
        fprintf(stderr, "[classify_data] Failed to get label stats\n");
        return NULL;
    }
    size_t majority_vote = 0;
    size_t winner_index = -1;

    for (size_t label_index = 0; label_index < stats.number_of_labels; ++label_index)
    {
        if (stats.frequencies[label_index] > majority_vote)
        {
            majority_vote = stats.frequencies[label_index];
            winner_index = label_index;
        }
    }

    char *predicted_label = NULL;
    if (winner_index == -1)
    {
        predicted_label = strdup("Unlabeled");
    }
    else
    {
        predicted_label = strdup(stats.labels[winner_index]);
    }

    for (size_t label_index = 0; label_index < stats.number_of_labels; ++label_index)
    {
        free(stats.labels[label_index]);
    }
    free(stats.labels);
    free(stats.frequencies);
    return predicted_label;
}

void print_usage_examples(const char* program_name)
{
    printf("=== %s Usage Examples ===\n", program_name);

    printf("\n1. Run KNN on a dataset with 3 features per sample\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/knn/train -n 3");

    printf("\n2. Standardize features before classification\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/knn/train -n 3 --std");

    printf("\n3. Set K manually to 5 neighbors\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/knn/train -n 3 -k 5");

    printf("\n4. Predict on an unlabeled dataset\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/knn/train -n 3 -p datasets/knn/test");

    printf("\n5. Test accuracy using a labeled dataset\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/knn/train -n 3 -a datasets/knn/test");

    printf("\n6. Enter samples manually for prediction\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/knn/train -n 3 --manual");

    printf("\n7. Save predictions to a file\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/knn/train -n 3 -p datasets/knn/test -o output.txt");

    printf("\n8. Custom token & decimal delimiters\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/knn/train.ssv -n 3 --tdel ; --ddel ,");

    printf("\n9. Run a demo using the repo's sample dataset:\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/knn/train -a datasets/knn/test -n 5 -k 5 --std");

    printf("\n10. Show all options and usage info\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "help");

    printf("\nNotes:\n");
    printf("- Options order does not matter.\n");
    printf("- Combine multiple options as needed (e.g., training + standardization + elbow method).\n");
}

void print_usage(const char *program_name)
{
    printf("Usage: %s [Options]\n", program_name);
    printf("Options:\n");
    printf("  -f <string>           Dataset file path\n");
    printf("  -a <string>           Labeled dataset used to test accuracy with preferred K neighbors\n");
    printf("  -p <string>           Unlabeled dataset file path containing the samples to predict\n");
    printf("  -o <string>           Save output file containing samples with predicted labels\n");
    printf("  -n <integer>          Number of features per sample (default: 1)\n");
    printf("  -k <integer>          Set the K value (default: 3)\n");
    printf("  --std                 Standardize all sample features (false by default)\n");
    printf("  --manual              Enter the samples to predict manually (false by default)\n");
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
    strncpy(args.acc_test_dataset_path, "", MAX_PATH-1);
    strncpy(args.prediction_dataset_path, "", MAX_PATH-1);
    strncpy(args.prediction_output_path, "", MAX_PATH-1);
    args.number_of_features = 3;
    args.K = 2;
    args.standardize = false;
    args.manual_input = false;
    strncpy(args.token_delimiter, " ", MAX_DELIM-1);
    args.decimals_delimiter = '.';
    strncpy(args.comment_delimiter, "#", MAX_DELIM-1);
    args.debug = false;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-f") == 0 && i+1 < argc)
            strncpy(args.dataset_path, argv[++i], MAX_PATH-1);
        else if (strcmp(argv[i], "-a") == 0 && i+1 < argc)
            strncpy(args.acc_test_dataset_path, argv[++i], MAX_PATH-1);
        else if (strcmp(argv[i], "-p") == 0 && i+1 < argc)
            strncpy(args.prediction_dataset_path, argv[++i], MAX_PATH-1);
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc)
            strncpy(args.prediction_output_path, argv[++i], MAX_PATH-1);
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc)
        {
            char *endptr;
            args.number_of_features = strtoul(argv[++i], &endptr, 10);
            if (*endptr != '\0') {
                fprintf(stderr, "Invalid number of features: %s\n", argv[i]);
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
        else if (strcmp(argv[i], "--std") == 0)
            args.standardize = true;
        else if (strcmp(argv[i], "--manual") == 0)
            args.manual_input = true;
        else if (strcmp(argv[i], "--tdel") == 0 && i+1 < argc)
            strncpy(args.token_delimiter, argv[++i], MAX_DELIM-1);
        else if (strcmp(argv[i], "--ddel") == 0 && i+1 < argc)
            args.decimals_delimiter = argv[++i][0];
        else if (strcmp(argv[i], "--cdel") == 0 && i+1 < argc)
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
            printf("EXIT EXIT_FAILURE ? argv[%d]: %s\n", i, argv[i]);
            print_usage(argv[0]);
            exit((strcmp(argv[i], "help") == 0) ? EXIT_SUCCESS : EXIT_FAILURE);
        }
    }

    return args;
}

int main(int argc, char *argv[])
{
    global_args = parse_args(argc, argv);
    if (global_args.debug)
    {
        printf("Debugging: true\n");
        printf("\n");
        printf("Dataset path: %s\n",  global_args.dataset_path);
        printf("Accuracy test dataset path: %s\n",  global_args.acc_test_dataset_path);
        printf("Prediction dataset path: %s\n",  global_args.prediction_dataset_path);
        printf("Prediction output path: %s\n",  global_args.prediction_output_path);
        printf("Number of features: %zu\n",  global_args.number_of_features);
        printf("K value: %zu\n",  global_args.K);
        printf("Standardize: %s\n", global_args.standardize ? "true" : "false");
        printf("Manual input: %s\n", global_args.manual_input ? "true" : "false");
        printf("Token delimiter: '%s'\n", global_args.token_delimiter);
        printf("Decimal delimiter: '%c'\n", global_args.decimals_delimiter);
        printf("Comment delimiter: '%s'\n", global_args.comment_delimiter);
        printf("\n");
    }
    
    if (global_args.debug) printf("[main] Loading dataset\n");
    Dataset *dataset = read_data_from_file(
        global_args.dataset_path,
        global_args.number_of_features,
        true,
        global_args.token_delimiter,
        global_args.decimals_delimiter,
        global_args.comment_delimiter
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

    if (strcmp(global_args.acc_test_dataset_path, "") != 0)
    {
        Dataset *dataset_test = read_data_from_file(
            global_args.acc_test_dataset_path,
            global_args.number_of_features,
            true,
            global_args.token_delimiter,
            global_args.decimals_delimiter,
            global_args.comment_delimiter
        );
        if (dataset_test == NULL)
        {
            free(standard_deviation);
            free(mean);
            free_dataset(dataset);
            return EXIT_FAILURE;
        }
        if (global_args.standardize)
        {
            if (standardize_data(dataset_test, mean, standard_deviation, global_args.number_of_features) == FAILURE)
            {

                free_dataset(dataset_test);
                free(standard_deviation);
                free(mean);
                free_dataset(dataset);
                return EXIT_FAILURE;
            }
        }

        char user_input[16] = "";
        bool retry = false;
        size_t correct_predictions = 0;
        size_t temp_K = global_args.K;
        do
        {
            for (size_t sample_index = 0; sample_index < dataset_test->number_of_samples; ++sample_index)
            {
                Dataset *neighbors = get_K_nearest_neighbors(
                    dataset, &dataset_test->samples[sample_index],
                    global_args.number_of_features, temp_K
                );
                if (neighbors == NULL)
                {
                    free_dataset(dataset_test);
                    free(standard_deviation);
                    free(mean);
                    free_dataset(dataset);
                    return EXIT_FAILURE;
                }
                char *predicted_label = classify_data(neighbors);
                if (predicted_label == NULL)
                {
                    free_dataset(neighbors);
                    free_dataset(dataset_test);
                    free(standard_deviation);
                    free(mean);
                    free_dataset(dataset);
                    return EXIT_FAILURE;
                }
                bool is_correct = (strcmp(dataset_test->samples[sample_index].label, predicted_label) == 0) ? true : false;
                correct_predictions += (size_t)is_correct;
                printf("\nSample %zu was labeled %s\n", sample_index + 1, is_correct ? "CORRECTLY" : "INCORRECTLY");

                free(predicted_label);
                free_dataset(neighbors);
            }

            printf("\n\nAccuracy with K=%zu: %0.2f%%", temp_K, (float)correct_predictions / dataset_test->number_of_samples * 100);
            correct_predictions = 0;

            printf("\nUpdate K and retry (y/n)? ");
            scanf("%15s", user_input);
            retry = user_input[0] == 'y' ? true : false;
            
            while (retry)
            {
                printf("K neighbors: ");
                scanf("%15s", user_input);
                if (is_number(user_input, global_args.decimals_delimiter))
                {
                    temp_K = (size_t)strtod(user_input, NULL);
                    if (temp_K > 0) break; // Do not set answer to true, it will update K but not retry
                }
            }

        } while (retry);

        free_dataset(dataset_test);
    }

    if (strcmp(global_args.prediction_dataset_path, "") != 0 || global_args.manual_input)
    {
        Dataset *unlabeled_dataset = NULL;

        if (global_args.manual_input)
        {
            printf("\nNumber of samples: ");
            size_t number_of_samples; scanf("%zu", &number_of_samples);

            // Allocate memory for the dataset and the samples

            unlabeled_dataset = calloc(1, sizeof(Dataset));
            if (unlabeled_dataset == NULL)
            {
                fprintf(stderr, "[main] Failed to allocate memory for unlabeled_dataset\n");
                free(standard_deviation);
                free(mean);
                free_dataset(dataset);
                return EXIT_FAILURE;
            }

            unlabeled_dataset->samples = calloc(number_of_samples, sizeof(Sample));
            if (unlabeled_dataset->samples == NULL)
            {
                fprintf(stderr, "[main] Failed to allocate memory for unlabeled_dataset->samples\n");
                free_dataset(unlabeled_dataset);
                free(standard_deviation);
                free(mean);
                free_dataset(dataset);
                return EXIT_FAILURE;
            }

            // Fill the dataset with unlabeled samples
            for (size_t sample_index = 0; sample_index < number_of_samples; ++sample_index)
            {
                unlabeled_dataset->samples[sample_index].label = NULL;
                unlabeled_dataset->samples[sample_index].features = malloc(global_args.number_of_features * sizeof(double));
                if (unlabeled_dataset->samples[sample_index].features == NULL)
                {
                    fprintf(stderr, "[main] Failed to allocate memory for unlabeled_dataset->samples[sample_index].features\n");
                    free_dataset(unlabeled_dataset);
                    free(standard_deviation);
                    free(mean);
                    free_dataset(dataset);
                    return EXIT_FAILURE;
                }

                printf("\n-- Enter the features of the sample %zu --\n", sample_index+1);

                size_t feature_index = 0;
                while (feature_index < global_args.number_of_features)
                {
                    printf("Feature %zu: ", feature_index+1);
                    scanf("%lf", &unlabeled_dataset->samples[sample_index].features[feature_index]);
                    ++feature_index;
                }
                ++unlabeled_dataset->number_of_samples;
            }
        }
        else
        {
            unlabeled_dataset = read_data_from_file(
                global_args.prediction_dataset_path,
                global_args.number_of_features,
                false,
                global_args.token_delimiter,
                global_args.decimals_delimiter,
                global_args.comment_delimiter
            );
            if (unlabeled_dataset == NULL)
            {
                free(standard_deviation);
                free(mean);
                free_dataset(dataset);
                return EXIT_FAILURE;
            }
        }

        // Deep copy dataset before standardizing to save samples with non-standardized features and their respective label, if needed
        Dataset *unlabeled_dataset_copy = NULL;
        if (global_args.standardize)
        {
            unlabeled_dataset_copy = deep_copy_dataset(unlabeled_dataset, global_args.number_of_features);
            if (standardize_data(unlabeled_dataset, mean, standard_deviation, global_args.number_of_features) == EXIT_FAILURE)
            {
                free_dataset(unlabeled_dataset);
                free(standard_deviation);
                free(mean);
                free_dataset(dataset);
                return EXIT_FAILURE;
            }
        }

        for (size_t sample_index = 0; sample_index < unlabeled_dataset->number_of_samples; ++sample_index)
        {
            Dataset *neighbors = get_K_nearest_neighbors(dataset, &unlabeled_dataset->samples[sample_index], global_args.number_of_features, global_args.K);
            if (neighbors == NULL)
            {
                free_dataset(unlabeled_dataset_copy);
                free_dataset(unlabeled_dataset);
                free(standard_deviation);
                free(mean);
                free_dataset(dataset);
                return EXIT_FAILURE;
            }
            char *predicted_label = classify_data(neighbors);
            if (predicted_label == NULL)
            {
                free_dataset(neighbors);
                free_dataset(unlabeled_dataset_copy);
                free_dataset(unlabeled_dataset);
                free(standard_deviation);
                free(mean);
                free_dataset(dataset);
                return EXIT_FAILURE;
            }

            // If there was a previous label that was dynamically allocated, free it first
            if (unlabeled_dataset->samples[sample_index].label != NULL)
            {
                free(unlabeled_dataset->samples[sample_index].label);
            }
            unlabeled_dataset->samples[sample_index].label = strdup(predicted_label);

            printf("\nSample %zu was labeled as: %s\n", sample_index+1, predicted_label);
            free(predicted_label);
            free_dataset(neighbors);
        }

        if (strcmp(global_args.prediction_output_path, "") != 0) // Save
        {
            FILE *file = fopen(global_args.prediction_output_path, "w");
            if (file == NULL)
            {
                fprintf(stderr, "[main] Failed to open in write mode the file at path: %s\n", global_args.prediction_output_path);
                free_dataset(unlabeled_dataset_copy);
                free_dataset(unlabeled_dataset);
                free(standard_deviation);
                free(mean);
                free_dataset(dataset);
                return EXIT_FAILURE;
            }
            
            Dataset *temp_unlabeled_dataset = unlabeled_dataset;
            if (unlabeled_dataset_copy != NULL) temp_unlabeled_dataset = unlabeled_dataset_copy; // Use copy to get original feature values

            for (size_t sample_index = 0; sample_index < unlabeled_dataset->number_of_samples; ++sample_index)
            {
                for (size_t feature_index = 0; feature_index < global_args.number_of_features; ++feature_index)
                    fprintf(file, "%g ", temp_unlabeled_dataset->samples[sample_index].features[feature_index]);
                
                fprintf(file, "%s\n", unlabeled_dataset->samples[sample_index].label);
            }
            fclose(file);

            printf("\nResults saved to file.\n");
        }
        else printf("\nResults were discarded.\n");
        
        free_dataset(unlabeled_dataset);
        free_dataset(unlabeled_dataset_copy);
    }
    free(mean);
    free(standard_deviation);
    free_dataset(dataset);

    return EXIT_SUCCESS;
}
