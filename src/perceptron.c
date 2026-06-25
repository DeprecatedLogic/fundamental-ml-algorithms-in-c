/*======================================================================
 *  Perceptron (supervized learning)
 *
 *  Author      :  DeprecatedLogic  <https://www.github.com/DeprecatedLogic>
 *  Created     :  03 Jul 2025
 *
 *  Description :
 *      A simple implementation of the Perceptron algorithm for binary
 *      classification using supervised learning.
 *====================================================================*/

// todo:
// - initialize mean and std in main, use that when standardizing other datasets.
// - store the mean and std values when saving the model, update load function accordingly.
// - return NULL or an integer instead of stopping the program mid-function,
//   to free allocated memory manually.

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

#define MAX_WEIGHTS 64
#define MAX_PATH 256
#define MAX_DELIM 16
#define SUCCESS 0
#define FAILURE -1

typedef struct
{
    char train_dataset_path[MAX_PATH]; // Dataset file path
    char acc_test_dataset_path[MAX_PATH]; // Test the accuracy of the algorithm with a labeled dataset & specific K value
    char prediction_dataset_path[MAX_PATH];
    char prediction_output_path[MAX_PATH];
    size_t number_of_features; // Number of features per sample (default: 1)
    char weight_tokens[MAX_WEIGHTS][32];
    size_t weight_token_count;
    double bias;
    double learning_rate;
    size_t epochs;
    bool standardize; // Standardize all sample features (false by default)
    char save_model_path[MAX_PATH];
    char load_model_path[MAX_PATH];
    char token_delimiter[MAX_DELIM];
    char decimals_delimiter;
    char comment_delimiter[MAX_DELIM];
    bool output_parameters;
    bool debug; // Verbose, used for debugging (false by default)
} Args;

Args global_args;

typedef struct
{
    double *features; // Array of input features (x1, ..., xn)
    size_t label; // The correctly guessed label (supervised learning)
} Sample;


typedef struct
{
    Sample *samples;
    size_t number_of_samples;

    char **labels;
    size_t number_of_labels;
}
Dataset;

typedef struct
{
    double *weights;
    double bias;
    double learning_rate;
    size_t number_of_features; // All samples have the same number of features (== size of the arrays too)
    double *mean;
    double *standard_deviation;
    char **labels;
    size_t number_of_labels;
}
Perceptron;

/**
 * @brief Frees a `Dataset` object.
 *
 * @param dataset A pointer to a `Dataset` object  
 * containing two arrays, for samples and labels.
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

    if (dataset->labels != NULL)
    {
        for (size_t label_index = 0; label_index < dataset->number_of_labels; ++label_index)
        {
            if (dataset->labels[label_index] != NULL)
                free(dataset->labels[label_index]);
        }
        free(dataset->labels);
    }

    free(dataset);
    if (global_args.debug) printf("[free_dataset] Dataset freed\n"); // Debugging
}

void free_perceptron(Perceptron *perceptron)
{
    if (perceptron == NULL)
    {
        if (global_args.debug) printf("[free_perceptron] Perceptron is NULL\n");
        return;
    }

    if (perceptron->mean != NULL) free(perceptron->mean);
    if (perceptron->standard_deviation != NULL) free(perceptron->standard_deviation);
    if (perceptron->weights != NULL) free(perceptron->weights);
    
    if (perceptron->labels != NULL)
    {
        for (size_t i = 0; i < perceptron->number_of_labels; ++i)
        {
            if (perceptron->labels[i] != NULL)
                free(perceptron->labels[i]);
        }
        free(perceptron->labels);
    }

    free(perceptron);
    if (global_args.debug) printf("[free_perceptron] Perceptron freed\n");
}

/**
 * @brief Checks if a given string represents a valid number.
 *
 * Uses a Look-Up Table (LUT) for fast digit validation. Handles
 * negative signs and a custom decimal delimiter.
 *
 * @param number The null-terminated string to check.
 * @param decimals_delimiter The character used as the decimal separator.
 * @return true if the string is a valid number, false otherwise.
 */
bool is_number(char *number, char decimals_delimiter)
{
    if (global_args.debug) fprintf(
        stderr,
        "[is_number] Number string: %s (first character: %c)\n",
        number,
        number[0]
    );

    if (*number == '-')
    {
        if (global_args.debug) fprintf(
            stderr,
            "[is_number] Encountered negative sign, skipping\n"
        );
        ++number;
    }
    if (*number == '\0')
    {
        if (global_args.debug) fprintf(
            stderr,
            "[is_number] Encountered end of string, returning false\n"
        );
        return false;
    }
 
    if (global_args.debug) fprintf(
        stderr,
        "[is_number] Starting with character '%c'\n",
        *number
    );

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
        if (global_args.debug) fprintf(
            stderr,
            "[is_number] Character value is '%c', returning false\n",
            c
        );
        return false;
    }
    return true;
}

/**
 * @brief Encodes a string label into a size_t index.
 *
 * If the label doesn't exist in the dataset, it dynamically adds it
 * to the dataset's labels array.
 *
 * @param dataset A pointer to the Dataset object.
 * @param label The string label to encode.
 * @param out_label_index Pointer to store the resulting encoded label index.
 * @return SUCCESS if successful, FAILURE on memory allocation error.
 */
int encode_label(Dataset *dataset, char *label, size_t *out_label_index)
{
    size_t label_index = 0;
    while (label_index < dataset->number_of_labels && strcmp(dataset->labels[label_index], label) != 0)
    {
        ++label_index;
    }
    
    if (label_index == dataset->number_of_labels)
    {
        char **temp_labels = realloc(dataset->labels, sizeof(char*) * (dataset->number_of_labels + 1));
        if (temp_labels == NULL)
        {
            fprintf(stderr, "[encoded_label] Failed to re-allocate memory for dataset->labels\n");
            return FAILURE;
        }
        dataset->labels = temp_labels;

        char *copy = strdup(label);
        if (copy == NULL) return FAILURE;
        dataset->labels[dataset->number_of_labels++] = copy;
    }
    if (global_args.debug) printf("[encoded_label] label: '%s' (length: %zu)\n", label, strlen(label));

    *out_label_index = label_index;
    return SUCCESS;
}

/**
 * @brief Reads sample data from a delimited text file and constructs a Dataset.
 *
 * Parses a file line by line, skipping comments and empty lines. It handles BOM 
 * stripping, tokenizes the features using the provided delimiter, and encodes 
 * the label if the dataset is supervised.
 *
 * @param file_path The path to the dataset file.
 * @param number_of_features The expected number of features per sample.
 * @param is_labeled Boolean indicating if the last token on each line is a label.
 * @param token_delimiter The string used to separate features/labels.
 * @param decimals_delimiter The character used for decimals in floating-point numbers.
 * @param comment_delimiter The string that marks the start of a comment.
 * @return A pointer to the allocated Dataset, or NULL if an error occurs.
 */
Dataset *read_data_from_file(
    const char *file_path, size_t number_of_features, bool is_labeled,
    char *token_delimiter, char decimals_delimiter, char *comment_delimiter
)
{
    if (global_args.debug) printf("[read_data_from_file] Reading data from file: %s\n", file_path);

    if (strcmp(token_delimiter, comment_delimiter) == 0)
    {
        fprintf(
            stderr,
            "[read_data_from_file] An error occured, token and comment delimiters cannot be the same (Token:%s | Comment:%s)\n",
            token_delimiter,
            comment_delimiter
        );
        return NULL;
    }
    if (token_delimiter[0] == decimals_delimiter || comment_delimiter[0] == decimals_delimiter)
    {
        fprintf(
            stderr,
            "[read_data_from_file] An error occured, token/comment delimiter cannot be the same as the decimals delimiter (%c)\n",
            decimals_delimiter
        );
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
        perror("[read_data_from_file] Failed to allocate memory for dataset");
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
        if (strchr(line, '\n') == NULL && !feof(data_file))
        {
            fprintf(stderr, "[read_data_from_file] Line %zu is too long\n", line_number);
            free_dataset(dataset);
            fclose(data_file);
            return NULL;
        }

        if (!bom_checked)
        {
            unsigned char *p_line = (unsigned char *)line;
            
            if (strlen(line) >= 3 && p_line[0] == 0xEF && p_line[1] == 0xBB && p_line[2] == 0xBF)
                memmove(line, line + 3, strlen(line + 3) + 1);
            
            bom_checked = true;
        }

        if (strncmp(line, comment_delimiter, strlen(comment_delimiter)) == 0)
        {
            if (global_args.debug) printf("[read_data_from_file] Line %zu skipped (fully commented line)\n", line_number);
            ++line_number;
            continue;
        }

        char *uncommented_line = strtok(line, comment_delimiter);
        
        // Empty line
        if (uncommented_line == NULL)
        {
            if (global_args.debug) printf("[read_data_from_file] Line %zu skipped (empty line)\n", line_number);
            ++line_number;
            continue;
        }
        
        // Strip newline
        uncommented_line[strcspn(uncommented_line, "\r\n")] = '\0';

        // Skip leading whitespace
        while (*uncommented_line == ' ' || *uncommented_line == '\t') ++uncommented_line;

        if (*uncommented_line == '\0')
        {
            if (global_args.debug) printf("[read_data_from_file] Line %zu skipped (whitespace-only line)\n", line_number);
            ++line_number;
            continue;
        }

        if (global_args.debug) printf("[read_data_from_file] uncommented_line (line %zu): %s\n", line_number, uncommented_line);

        char *token_buffer = strtok(uncommented_line, token_delimiter);
        size_t token_counter = 0;
        
        Sample sample = {
            .features = (double *)malloc(sizeof(double) * number_of_features),
            .label = 0
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
                        "[read_data_from_file] Encoutered a bad sample at line '%zu' (feature value '%s' is not a number ?)\n",
                        line_number,
                        token_buffer
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
                if (encode_label(dataset, token_buffer, &sample.label) == FAILURE)
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
            }

            token_buffer = strtok(NULL, token_delimiter);
            ++token_counter;
        }

        size_t features_counted = (is_labeled && token_counter > 0) ? token_counter - 1 : token_counter; // -1 because of the label
        if (features_counted != number_of_features)
        {

            fprintf(
                stderr,
                "[read_data_from_file] Encountered a mismatch: features counted [%zu] != expected [%zu] (line %zu)\n",
                features_counted,
                number_of_features,
                line_number
            );
            free_dataset(dataset);
            free(sample.features);
            fclose(data_file);
            return NULL;
        }

        Sample* temp_samples = (Sample *)realloc(dataset->samples, sizeof(Sample) * (dataset->number_of_samples + 1));
        if (temp_samples == NULL)
        {
            fprintf(stderr, "[read_data_from_file] Failed to re-allocate memory for temp_samples\n");
            free_dataset(dataset);
            free(sample.features);
            fclose(data_file);
            return NULL;
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
 * @brief Saves the perceptron's parameters to a binary file.
 *
 * @param file_path The path to save the perceptron model.
 * @param perceptron A pointer to the `Perceptron` model to save.
 *
 * @note Saves version, number of features, weights, bias, learning rate, 
 *       and the standardization parameters (mean and std).
 */
int save_model(const char *file_path, const Perceptron *perceptron)
{
    if (global_args.debug) printf("[save_model] Saving model as binary: %s\n", file_path);
    
    FILE *file = fopen(file_path, "wb");
    if (file == NULL)
    {
        fprintf(stderr, "[save_model] Failed to open/create file: %s\n", file_path);
        return FAILURE;
    }

    // Write encoding version (to ensure compatibility!)
    size_t version = 0x76312E30; // v1.0
    fwrite(&version, sizeof(size_t), 1, file);

    // Write number of features
    fwrite(&perceptron->number_of_features, sizeof(size_t), 1, file);

    // Write weights
    fwrite(perceptron->weights, sizeof(double), perceptron->number_of_features, file);

    // Write bias and learning rate
    fwrite(&perceptron->bias, sizeof(double), 1, file);
    fwrite(&perceptron->learning_rate, sizeof(double), 1, file);

    // Write standardization parameters
    fwrite(perceptron->mean, sizeof(double), perceptron->number_of_features, file);
    fwrite(perceptron->standard_deviation, sizeof(double), perceptron->number_of_features, file);

    // Write labels
    fwrite(&perceptron->number_of_labels, sizeof(size_t), 1, file);
    for (size_t iLabel = 0; iLabel < perceptron->number_of_labels; ++iLabel)
    {
        size_t label_length = strlen(perceptron->labels[iLabel]);
        fwrite(&label_length, sizeof(size_t), 1, file);
        fwrite(perceptron->labels[iLabel], sizeof(char), label_length, file);
    }

    if (global_args.debug) printf("[save_model] Binary model saved successfully.\n");

    fclose(file);
    return SUCCESS;
}

/**
 * @brief Loads a perceptron's parameters from a binary file.
 *
 * @param file_path The model's path.
 * @returns A pointer to the initialized perceptron object.
 */
Perceptron *load_model(const char *file_path)
{
    if (global_args.debug) printf("[load_model] Loading model from: %s\n", file_path);

    FILE *file = fopen(file_path, "rb");
    if (file == NULL)
    {
        fprintf(stderr, "[load_model] Failed to open/read file: %s\n", file_path);
        return NULL;
    }

    Perceptron *perceptron = calloc(1, sizeof(Perceptron));
    if (perceptron == NULL)
    {
        perror("[load_model] Failed to allocate memory for perceptron");
        fclose(file);
        return NULL;
    }

    // Check version
    size_t version = 0;
    if (fread(&version, sizeof(size_t), 1, file) != 1 || version != 0x76312E30) // 0x76312E30 = v1.0
    {
        fprintf(stderr, "[load_lamp] Version mismatch or invalid format!\n");
        free(perceptron);
        fclose(file);
        return NULL;
    }

    // Read number of features
    if (fread(&perceptron->number_of_features, sizeof(size_t), 1, file) != 1)
    {
        fprintf(stderr, "[load_model] Failed to read number of features\n");
        goto error_cleanup;
    }

    // Verify feature count matches global expectation
    if (perceptron->number_of_features != global_args.number_of_features)
    {
        fprintf(stderr, "[load_model] Feature mismatch! Model has %zu, expected %zu\n", 
                perceptron->number_of_features, global_args.number_of_features);
        goto error_cleanup;
    }

    // Allocate and Read Weights
    perceptron->weights = malloc(perceptron->number_of_features * sizeof(double));
    if (fread(perceptron->weights, sizeof(double), perceptron->number_of_features, file) != perceptron->number_of_features)
    {
        fprintf(stderr, "[load_model] Failed to read weights\n");
        goto error_cleanup;
    }

    // Read Bias and Learning Rate
    if (fread(&perceptron->bias, sizeof(double), 1, file) != 1 ||
        fread(&perceptron->learning_rate, sizeof(double), 1, file) != 1)
    {
        fprintf(stderr, "[load_model] Failed to read bias or learning rate\n");
        goto error_cleanup;
    }

    // Read standardization parameters
    perceptron->mean = malloc(perceptron->number_of_features * sizeof(double));
    perceptron->standard_deviation = malloc(perceptron->number_of_features * sizeof(double));
    
    if (fread(perceptron->mean, sizeof(double), perceptron->number_of_features, file) != perceptron->number_of_features ||
        fread(perceptron->standard_deviation, sizeof(double), perceptron->number_of_features, file) != perceptron->number_of_features)
    {
        fprintf(stderr, "[load_model] Failed to read mean or std deviation\n");
        goto error_cleanup;
    }

    // Read labels
    if (fread(&perceptron->number_of_labels, sizeof(size_t), 1, file) != 1)
    {
        fprintf(stderr, "[load_model] Failed to read number of labels\n");
        goto error_cleanup;
    }
    
    if (perceptron->number_of_labels > 0)
    {
        perceptron->labels = malloc(sizeof(char*) * perceptron->number_of_labels);
        if (perceptron->labels == NULL) goto error_cleanup;

        for (size_t i = 0; i < perceptron->number_of_labels; ++i)
        {
            size_t len = 0;
            if (fread(&len, sizeof(size_t), 1, file) != 1) goto error_cleanup;
            
            perceptron->labels[i] = malloc(len + 1);
            if (perceptron->labels[i] == NULL) goto error_cleanup;
            
            if (fread(perceptron->labels[i], sizeof(char), len, file) != len) goto error_cleanup;
            perceptron->labels[i][len] = '\0';
        }
    }

    if (global_args.debug) printf("[load_model] Model loaded successfully from binary!\n");
    fclose(file);
    return perceptron;

error_cleanup:
    if (perceptron) free_perceptron(perceptron);
    fclose(file);
    return NULL;
}

/**
 * @brief Makes a deep copy of a `Dataset` object.
 *
 * @attention Labels are not copied because they are not used!
 *
 * @param dataset A pointer to the `Dataset` object to copy.
 * @param number_of_features The number of features.
 *
 * @returns A pointer to a newly allocated and copied `Dataset` object.
 */
Dataset *deep_copy_dataset(const Dataset *dataset, size_t number_of_features)
{
    Dataset *copy = calloc(1, sizeof(Dataset));
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
    copy->number_of_samples = 0;

    for (size_t sample_index = 0; sample_index < dataset->number_of_samples; ++sample_index)
    {
        copy->samples[sample_index].features = malloc(number_of_features * sizeof(double));
        if (copy->samples[sample_index].features == NULL)
        {
            fprintf(stderr, "[deep_copy_dataset] Failed to allocate memory for copy->samples[sample_index].features\n");
            free_dataset(copy);
            return NULL;
        }
        
        for (size_t feature_index = 0; feature_index < number_of_features; ++feature_index)
        {
            copy->samples[sample_index].features[feature_index] = dataset->samples[sample_index].features[feature_index];
        }

        copy->samples[sample_index].label = dataset->samples[sample_index].label;

        ++copy->number_of_samples;
    }

    if (global_args.debug) printf("[deep_copy_dataset] Successfully deep-copied dataset\n");
    return copy;
}

/**
 * @brief Updates weights and bias in-place.
 *
 * @param perceptron Pointer to the perceptron whose weights/bias will be updated.
 * @param sample Pointer to the training sample that triggered the update.
 * @param error Signed error term: (label − prediction). Values are −1, 0, or 1.
 */
void update_parameters(Perceptron *perceptron, const Sample *sample, int error)
{
    // Update weights
    for (size_t i = 0; i < perceptron->number_of_features; ++i)
    {
        perceptron->weights[i] += perceptron->learning_rate * error * sample->features[i];
    }
    // Update bias
    perceptron->bias += perceptron->learning_rate * error;
}

/**
 * @brief Computes the perceptron’s output for one sample.
 *
 * @param perceptron Pointer to the trained (or partially trained) model.
 * @param sample Pointer to the input whose class you want to predict.
 *
 * @returns `1` if output is greater than or equal to 0, otherwise `0`.
 */
size_t compute_prediction(const Perceptron *perceptron, const Sample *sample)
{
    double output = perceptron->bias;
    for (size_t i = 0; i < perceptron->number_of_features; ++i)
    {
        output += perceptron->weights[i] * sample->features[i];
    }
    return (output >= 0) ? 1 : 0;
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
        fprintf(stderr, "[calculate_mean] Failed allocating memory for mean\n");
        return NULL;
    }

    // Loop through the points and sum the features respectively
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

    if (global_args.debug) printf("[calculate_mean] Calculated mean successfully\n");
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
        perror("[calculate_std_deviation] Failed allocating memory for standard_deviation");
        return NULL;
    }

    // Loop through the points and sum all the squared differences (feature - mean)^2
    for (size_t sample_index = 0; sample_index < dataset->number_of_samples; ++sample_index)
    {
        for (size_t feature_index = 0; feature_index < number_of_features; ++feature_index)
        {
            double diff = dataset->samples[sample_index].features[feature_index] - mean[feature_index];
            standard_deviation[feature_index] += diff * diff;
        }
    }

    // Finally, divide each value by the number of points to get the variance
    // and then use `sqrt` to get the standard deviation
    for (size_t feature_index = 0; feature_index < number_of_features; ++feature_index)
    {
        standard_deviation[feature_index] = sqrt(standard_deviation[feature_index] / dataset->number_of_samples);
    }

    if (global_args.debug) printf("[calculate_std_deviation] Calculated standard deviation successfully\n");
    return standard_deviation;
}

/**
 * @brief Standardizes the features of the whole dataset and those of the new sample.
 *
 * x(standardized​) = (x − μ)​ / σ
 *
 * Modifies all sample features in-place  
 * instead of returning a new `Sample` with the modified values.
 *
 * @attention Exits with an error message if a division by 0 occurs.
 *
 * @param dataset A pointer to a `Dataset` object containing the array of samples and its count.
 * @param mean The mean value from the train dataset
 * @param standard_deviation The standard deviation from the train dataset.
 * @param number_of_features The number of features each sample has.
 * 
 * @returns SUCCESS if features get standardized, otherise FAILURE.
 */
int standardize_data(Dataset *dataset, size_t number_of_features, const double* mean, const double *standard_deviation)
{
    // Check the number of samples because there's a division with it as denominator
    if (dataset->number_of_samples == 0)
    {
        fprintf(stderr, "[standardize_data] Failed to standardize data. \
            No data was found ? Avoided division by 0");
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

    if (global_args.debug) printf("[standardize_data] Features standardized successfully\n");
    return SUCCESS;
}

/**
 * @brief Initializes the perceptron's weights with random values.
 *
 * @param perceptron Pointer to the `Perceptron` model whose weights will be initialized with random values.
 *
 * @note Weights will be random but centered around 0.
 */
int init_random_weights(Perceptron *perceptron)
{
    perceptron->weights = (double*)malloc(perceptron->number_of_features * sizeof(double));
    if (perceptron->weights == NULL)
    {
        perror("[init_weights] Failed to allocate memory for perceptron->weights");
        free(perceptron);
        return FAILURE;
    }

    for (size_t weight_index = 0; weight_index < perceptron->number_of_features; ++weight_index)
    {
        perceptron->weights[weight_index] = ((double)rand() / RAND_MAX) * 2 - 1; // random, centered around 0
        if (global_args.debug)
            printf(
        "[init_random_weights] perceptron->weights[%zu]: %g\n",
                weight_index, perceptron->weights[weight_index]
            );
    }

    return SUCCESS;
}

int apply_indexed_weights(Perceptron *perceptron)
{
    for (size_t i = 0; i < global_args.weight_token_count; i++)
    {
        char *token = global_args.weight_tokens[i];

        char *colon = strchr(token, ':');
        if (!colon)
        {
            fprintf(stderr, "Invalid weight format: %s\n", token);
            return FAILURE;
        }

        *colon = '\0';
        char *index_str = token;
        char *value_str = colon + 1;

        char *endptr;

        size_t index = strtoul(index_str, &endptr, 10);
        if (*endptr != '\0' || index == 0 || index > perceptron->number_of_features)
        {
            fprintf(stderr, "Invalid weight index: %s\n", index_str);
            return FAILURE;
        }

        double value = strtod(value_str, &endptr);
        if (*endptr != '\0')
        {
            fprintf(stderr, "Invalid weight value: %s\n", value_str);
            return FAILURE;
        }

        perceptron->weights[index - 1] = value; // starts at 1, so we remove 1 to get the actual weight index
    }
    
    return SUCCESS;
}

void output_parameters(Perceptron *perceptron)
{
    printf("Learning rate: %g\n", perceptron->learning_rate);
    printf("Bias: %g\n", perceptron->bias);
    printf("Number of weights: %zu\n", perceptron->number_of_features);
    printf("Weights:");
    for (size_t weight_index = 0; weight_index < perceptron->number_of_features; ++weight_index)
        printf(" %g", perceptron->weights[weight_index]);
    printf("\n");

    printf("\n== Per-feature mean & standard deviation ==\n");
    for (size_t feature_index = 0; feature_index < perceptron->number_of_features; ++feature_index)
    {
        printf(
            "Feature %zu: mean %g | std %g\n",
            feature_index+1,
            perceptron->mean[feature_index],
            perceptron->standard_deviation[feature_index]);
    }
}

void print_usage_examples(const char* program_name)
{
    printf("=== %s Usage Examples ===\n", program_name);

    printf("\n1. Train on a dataset with 2 features per sample and set custom values for weights 1 & 5:\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/perceptron/train -n 2 -w 1:3.45 5:-9.2");

    printf("\n2. Train and standardize features:\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/perceptron/train -n 2 --std");

    printf("\n3. Use a semicolon as token delimiter:\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/perceptron/train.ssv -n 2 --tdel ;");

    printf("\n4. Floating-point numbers with comma decimal delimiter:\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/perceptron/train -n 2 --ddel ,");

    printf("\n5. Train and save a perceptron model:\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/perceptron/train -n 2 --save models/perceptron_model");

    printf("\n6. Load a pre-trained model and test accuracy:\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "--load models/perceptron_model -a datasets/perceptron/test -n 2");

    printf("\n7. Predict labels for an unlabeled dataset:\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-p datasets/perceptron/unlabeled -o predictions.txt --load models/perceptron_model -n 2");

    printf("\n8. Run a demo using the repo's sample dataset:\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/perceptron/train -a datasets/perceptron/test -n 2");

    printf("\n9. Save sample features with their predicted labels:\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/perceptron/train -p datasets/perceptron/unlabeled -n 2 -o output.txt");

    printf("\n10. Show all available options:\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "help");

    printf("\nNotes:\n");
    printf("- Options order does not matter.\n");
    printf("- Combine multiple options as needed (e.g., training + save + standardization).\n");
}

void print_usage(const char *program_name)
{
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -f <string>           Dataset file path\n");
    printf("  -a <string>           Labeled dataset used to test accuracy\n");
    printf("  -p <string>           Unlabeled dataset containing the samples to predict\n");
    printf("  -o <string>           Save output file containing samples with predicted labels\n");
    printf("  -n <integer>          Number of features per sample (default: 1)\n");
    printf("  -w <string>           Initial weight values (random if unset)\n");
    printf("  -b <float>            Bias (default: 0)\n");
    printf("  -l <float>            Learning rate (default: 0.1)\n");
    printf("  -e <integer>          Episodes, the amount of times the model trains on the dataset (default: 1)\n");
    printf("  --std                 Standardize all sample features (false by default)\n");
    printf("  --save <string>       Path to save the trained perceptron\n");
    printf("  --load <string>       Perceptron model to load for inference\n");
    printf("  --tdel <string>       Token delimiter (default: space)\n");
    printf("  --ddel <char>         Decimal delimiter (default: .)\n");
    printf("  --cdel <string>       Comment delimiter (default: #)\n");
    printf("  --params              Output perceptron parameters at the end (false by default)\n");
    printf("\n");
    printf("  -v, --debug           Verbose, used for debugging (false by default)\n");
    printf("  examples              Shows a list of usage examples\n");
    printf("  help                  Usage and options menu (this command)\n");
}

Args parse_args(int argc, char **argv)
{
    Args args;

    // Default values
    strncpy(args.train_dataset_path, "", MAX_PATH-1);
    strncpy(args.acc_test_dataset_path, "", MAX_PATH-1);
    strncpy(args.prediction_dataset_path, "", MAX_PATH-1);
    strncpy(args.prediction_output_path, "", MAX_PATH-1);
    args.number_of_features = 1;
    args.weight_token_count = 0;
    args.bias = 0;
    args.learning_rate = 0.1;
    args.epochs = 100;
    args.standardize = false;
    strncpy(args.save_model_path, "", MAX_PATH-1);
    strncpy(args.load_model_path, "", MAX_PATH-1);
    strncpy(args.token_delimiter, " ", MAX_DELIM-1);
    args.decimals_delimiter = '.';
    strncpy(args.comment_delimiter, "#", MAX_DELIM-1);
    args.output_parameters = false;
    args.debug = false;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-f") == 0 && i+1 < argc)
            strncpy(args.train_dataset_path, argv[++i], MAX_PATH-1);
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
        else if (strcmp(argv[i], "-w") == 0)
        {
            while (i+1 < argc && argv[i+1][0] != '-')
            {
                if (args.weight_token_count >= MAX_WEIGHTS)
                {
                    fprintf(stderr, "Too many weight arguments\n");
                    exit(EXIT_FAILURE);
                }

                strncpy(
                    args.weight_tokens[args.weight_token_count++],
                    argv[++i],
                    31
                );
            }
        }
        else if (strcmp(argv[i], "-b") == 0 && i+1 < argc)
        {
            char *endptr;
            args.bias = strtod(argv[++i], &endptr);
            if (*endptr != '\0') {
                fprintf(stderr, "Invalid bias value: %s\n", argv[i]);
                exit(EXIT_FAILURE);
            }
        }
        else if (strcmp(argv[i], "-l") == 0 && i+1 < argc)
        {
            char *endptr;
            args.learning_rate = strtod(argv[++i], &endptr);
            if (*endptr != '\0') {
                fprintf(stderr, "Invalid learning rate value: %s\n", argv[i]);
                exit(EXIT_FAILURE);
            }
        }
        else if (strcmp(argv[i], "-e") == 0 && i+1 < argc)
        {
            char *endptr;
            args.epochs = strtoul(argv[++i], &endptr, 10);
            if (*endptr != '\0') {
                fprintf(stderr, "Invalid number of epochs: %s\n", argv[i]);
                exit(EXIT_FAILURE);
            }
        }
        else if (strcmp(argv[i], "--std") == 0)
            args.standardize = true;
        else if (strcmp(argv[i], "--save") == 0 && i+1 < argc)
            strncpy(args.save_model_path, argv[++i], MAX_PATH-1);
        else if (strcmp(argv[i], "--load") == 0 && i+1 < argc)
            strncpy(args.load_model_path, argv[++i], MAX_PATH-1);
        else if (strcmp(argv[i], "--tdel") == 0 && i+1 < argc)
            strncpy(args.token_delimiter, argv[++i], MAX_DELIM-1);
        else if (strcmp(argv[i], "--ddel") == 0 && i+1 < argc)
            args.decimals_delimiter = argv[++i][0];
        else if (strcmp(argv[i], "--cdel") == 0 && i+1 < argc)
            strncpy(args.comment_delimiter, argv[++i], MAX_DELIM-1);
        else if (strcmp(argv[i], "--params") == 0)
            args.output_parameters = true;
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

    return args;
}

/**
 * @brief Validates that a dataset adheres to the binary classification contract.
 *
 * Ensures the dataset contains exactly two unique labels. This is a critical
 * guardrail for algorithms like the Perceptron, which mathematically require
 * strictly binary targets (0 or 1) to function correctly.
 *
 * @param dataset Pointer to the Dataset structure to validate.
 * @param dataset_name A descriptive string (e.g., "train", "test") used 
 * for error reporting.
 * @note If the validation fails, the function prints a descriptive error 
 * message to stderr.
 *
 * @returns SUCCESS if valid, otherwise FAILURE.
 */
int validate_binary_dataset(const Dataset *dataset, const char *dataset_name)
{
    if (dataset->number_of_labels != 2)
    {
        fprintf(
            stderr, 
            ANSI_COLOR_RED "Error: Perceptron requires exactly 2 unique labels for binary classification (found %zu for %s dataset).\n" ANSI_COLOR_RESET, 
            dataset->number_of_labels,
            dataset_name
        );
        return FAILURE;
    }
    return SUCCESS;
}

int main(int argc, char *argv[])
{
    global_args = parse_args(argc, argv);
    if (global_args.debug)
    {
        printf("Debugging: true\n");
        printf("\n");
        printf("Dataset path: %s\n",  global_args.train_dataset_path);
        printf("Accuracy test dataset path: %s\n",  global_args.acc_test_dataset_path);
        printf("Prediction dataset path: %s\n",  global_args.prediction_dataset_path);
        printf("Prediction output path: %s\n",  global_args.prediction_output_path);
        printf("Number of features: %zu\n", global_args.number_of_features);
        printf("Custom Weights: ");
        for (size_t weight_index = 0; weight_index < global_args.weight_token_count; ++weight_index)
            printf("%s", global_args.weight_tokens[weight_index]);
        printf("\n");
        printf("Bias: %g\n", global_args.bias);
        printf("Learning rate: %g\n", global_args.learning_rate);
        printf("Epochs: %zu\n", global_args.epochs);
        printf("Standardize: %s\n", global_args.standardize ? "true" : "false");
        printf("Save model path: %s\n",  global_args.save_model_path);
        printf("Load model path: %s\n",  global_args.load_model_path);
        printf("Token delimiter: '%s'\n", global_args.token_delimiter);
        printf("Decimal delimiter: '%c'\n", global_args.decimals_delimiter);
        printf("Comment delimiter: '%s'\n", global_args.comment_delimiter);
        printf("\n");
    }

    srand((unsigned int)time(NULL));
    
    Perceptron *perceptron = NULL;

    // Load a model ?
    if (strcmp(global_args.load_model_path, "") != 0)
    {
        perceptron = load_model(global_args.load_model_path);
        if (perceptron == NULL)
        {
            fprintf(stderr, "Failed to load the model at path: %s\n", global_args.load_model_path);
            return EXIT_FAILURE;
        }
        printf("\nModel loaded successfully (learning rate: %g)\n\n", perceptron->learning_rate);
    }
    else if (strcmp(global_args.train_dataset_path, "") != 0) // Training
    {
        perceptron = (Perceptron*)malloc(sizeof(Perceptron));
        if (perceptron == NULL)
        {
            fprintf(stderr, "[main] Failed to allocate memory for perceptron\n");
            return EXIT_FAILURE;
        }

        if (global_args.number_of_features == 0)
        {
            fprintf(stderr, "[main] The number of features cannot be equal to 0\n");
            free_perceptron(perceptron);
            return EXIT_FAILURE;
        }

        perceptron->number_of_features = global_args.number_of_features;
        perceptron->bias = global_args.bias;
        perceptron->learning_rate = global_args.learning_rate;
        if (init_random_weights(perceptron) == FAILURE)
        {
            fprintf(stderr, "[main] Failed to initialize random weights\n");
            free_perceptron(perceptron);
            return EXIT_FAILURE;
        }
        if (apply_indexed_weights(perceptron) == FAILURE)
        {
            fprintf(stderr, "[main] Failed to set custom weights\n");
            free_perceptron(perceptron);
            return EXIT_FAILURE;
        }

        if (global_args.debug)
        {
            printf("[main] Current parameters:\n- Bias: %g\n- Weights:", perceptron->bias);
            for (size_t weight_index = 0; weight_index < perceptron->number_of_features; ++weight_index)
                printf(" %g",perceptron->weights[weight_index]);
            printf("\n");
        }

        Dataset *train_dataset = read_data_from_file(
            global_args.train_dataset_path,
            perceptron->number_of_features,
            true,
            global_args.token_delimiter,
            global_args.decimals_delimiter,
            global_args.comment_delimiter
        );
        if (train_dataset == NULL)
        {
            fprintf(stderr, "[main] Failed to load dataset\n");
            free_perceptron(perceptron);
            return EXIT_FAILURE;
        }

        if (validate_binary_dataset(train_dataset, "train") == FAILURE)
        {
            free_perceptron(perceptron);
            free_dataset(train_dataset);
            return EXIT_FAILURE;
        }

        perceptron->mean = calculate_mean(train_dataset, perceptron->number_of_features);
        if (perceptron->mean == NULL)
        {
            fprintf(stderr, "[main] Failed to calculate mean\n");
            free_perceptron(perceptron);
            free_dataset(train_dataset);
            return EXIT_FAILURE;
        }
        perceptron->standard_deviation = calculate_std_deviation(train_dataset, perceptron->mean, perceptron->number_of_features);
        if (perceptron->standard_deviation == NULL)
        {
            fprintf(stderr, "[main] Failed to calculate standard deviation\n");
            free_perceptron(perceptron);
            free_dataset(train_dataset);
            return EXIT_FAILURE;
        }

        if (global_args.standardize)
        {
            if (standardize_data(train_dataset, perceptron->number_of_features, 
                perceptron->mean, perceptron->standard_deviation) == FAILURE)
            {
                fprintf(stderr, "[main] Failed to standardize features in train_dataset\n");
                free_perceptron(perceptron);
                free_dataset(train_dataset);
                return EXIT_FAILURE;
            }
        }

        if (global_args.debug) printf("[main] Training perceptron model\n");
        for (size_t epoch = 0; epoch < global_args.epochs; ++epoch)
        {
            if (global_args.debug)
            {
                size_t label_index = train_dataset->samples[0].label;
                printf(
                    "Epoch %zu (first sample: enc_label=%zu label=%s pred=%zu)\n",
                    epoch,
                    label_index,
                    train_dataset->labels[label_index],
                    compute_prediction(perceptron, &train_dataset->samples[0])
                );
            }
            
            for (size_t sample_index = 0; sample_index < train_dataset->number_of_samples; ++sample_index)
            {
                Sample *sample = &train_dataset->samples[sample_index];
                int error_value = compute_prediction(perceptron, sample);
                update_parameters(
                    perceptron,
                    sample,
                    sample->label - error_value
                );
            }

            if (global_args.debug)
            {
                printf("Updated parameters:\n- Bias: %g\n- Weights:", perceptron->bias);
                for (size_t weight_index = 0; weight_index < perceptron->number_of_features; ++weight_index)
                    printf(" %g",perceptron->weights[weight_index]);
                printf("\n");
            }

        }
        printf("\nModel trained succesfully.\n");

        // Transfer label mapping to the model
        perceptron->number_of_labels = train_dataset->number_of_labels;
        perceptron->labels = malloc(sizeof(char*) * train_dataset->number_of_labels);
        for (size_t i = 0; i < train_dataset->number_of_labels; ++i)
        {
            perceptron->labels[i] = strdup(train_dataset->labels[i]);
        }

        free_dataset(train_dataset);
    }
    else
    {
        fprintf(stderr, "[main] Load a model or use a dataset for training\n");
        return EXIT_FAILURE;
    }

    // === Inference ===

    // Accuracy test with labeled dataset
    if (strcmp(global_args.acc_test_dataset_path, "") != 0)
    {
        Dataset *test_dataset = read_data_from_file(
            global_args.acc_test_dataset_path,
            perceptron->number_of_features,
            true,
            global_args.token_delimiter,
            global_args.decimals_delimiter,
            global_args.comment_delimiter
        );
        if (test_dataset == NULL)
        {
            free_perceptron(perceptron);
            return EXIT_FAILURE;
        }

        if (validate_binary_dataset(test_dataset, "test") == FAILURE)
        {
            free_perceptron(perceptron);
            free_dataset(test_dataset);
            return EXIT_FAILURE;
        }

        if (global_args.standardize)
        {
            if (standardize_data(test_dataset,perceptron->number_of_features,
                perceptron->mean, perceptron->standard_deviation) == FAILURE)
            {
                fprintf(stderr, "[main] Failed to standardize features in test_dataset\n");
                free_perceptron(perceptron);
                free_dataset(test_dataset);
                return EXIT_FAILURE;
            }
        }

        printf("\nRunning accuracy test...\n");
        size_t correct_guesses = 0;

        for (size_t sample_index = 0; sample_index < test_dataset->number_of_samples; ++sample_index)
        {
            Sample *sample = &test_dataset->samples[sample_index];
            size_t output = compute_prediction(perceptron, sample); // 0 or 1

            // Resolve the actual strings from their respective dictionaries
            char *predicted_label_str = perceptron->labels[output];
            char *actual_label_str = test_dataset->labels[sample->label];

            bool is_correct = (strcmp(predicted_label_str, actual_label_str) == 0);
            
            printf(
                "Guessed %s for sample %s%zu%s (Expected: %s, Predicted: %s)\n",
                is_correct ? ANSI_COLOR_GREEN "correctly" ANSI_COLOR_RESET : ANSI_COLOR_RED "incorrectly" ANSI_COLOR_RESET,
                ANSI_COLOR_MAGENTA,
                sample_index,
                ANSI_COLOR_RESET,
                actual_label_str, 
                predicted_label_str
            );

            correct_guesses += is_correct ? 1 : 0;
        }
        printf("\nModel accuracy: %.2f%%\n\n", (float)correct_guesses / test_dataset->number_of_samples * 100);

        free_dataset(test_dataset);
    }

    // Predict unlabeled samples
    if (strcmp(global_args.prediction_dataset_path, "") != 0)
    {
        Dataset *unlabeled_dataset = read_data_from_file(
            global_args.prediction_dataset_path,
            perceptron->number_of_features,
            false,
            global_args.token_delimiter,
            global_args.decimals_delimiter,
            global_args.comment_delimiter
        );
        if (unlabeled_dataset == NULL)
        {
            free_perceptron(perceptron);
            return EXIT_FAILURE;
        }

        // Deep copy dataset before standardizing to save samples with non-standardized features and their respective label, if needed
        Dataset *unlabeled_dataset_copy = NULL;

        if (global_args.standardize)
        {
            unlabeled_dataset_copy = deep_copy_dataset(unlabeled_dataset, perceptron->number_of_features);
            if (unlabeled_dataset_copy == NULL)
            {
                free_perceptron(perceptron);
                free_dataset(unlabeled_dataset);
                return EXIT_FAILURE;
            }
            if (standardize_data(unlabeled_dataset, perceptron->number_of_features,
                    perceptron->mean, perceptron->standard_deviation) == FAILURE)
            {
                fprintf(stderr, "[main] Failed to standardize features in unlabeled_dataset\n");
                free_perceptron(perceptron);
                free_dataset(unlabeled_dataset);
                return EXIT_FAILURE;
            }
        }

        printf("\nComputing predictions...\n");

        for (size_t sample_index = 0; sample_index < unlabeled_dataset->number_of_samples; ++sample_index)
        {
            Sample *sample = &unlabeled_dataset->samples[sample_index];
            sample->label = compute_prediction(perceptron, sample); // sample is unlabeled so the only label is what the algorithm predicts
            
            printf(
                "Guessed %s%zu%s (%s%s%s)\n",
                ANSI_COLOR_CYAN,
                sample->label,
                ANSI_COLOR_RESET,
                ANSI_COLOR_BLUE,
                perceptron->labels[sample->label],
                ANSI_COLOR_RESET
            );
        }

        if (strcmp(global_args.prediction_output_path, "") != 0) // Save
        {
            FILE* file = fopen(global_args.prediction_output_path, "w");
            if (file == NULL)
            {
                fprintf(stderr, "[main] Failed to open in write mode the file at path: %s\n", global_args.prediction_output_path);
                free_dataset(unlabeled_dataset_copy);
                free_dataset(unlabeled_dataset);
                free_perceptron(perceptron);
                return EXIT_FAILURE;
            }

            Dataset *temp_unlabeled_dataset = unlabeled_dataset;
            if (unlabeled_dataset_copy != NULL) temp_unlabeled_dataset = unlabeled_dataset_copy; // Use copy to get original feature values

            for (size_t sample_index = 0; sample_index < unlabeled_dataset->number_of_samples; ++sample_index)
            {
                for (size_t feature_index = 0; feature_index < perceptron->number_of_features; ++feature_index)
                    fprintf(file, "%g ", temp_unlabeled_dataset->samples[sample_index].features[feature_index]);
                
                if (unlabeled_dataset->samples[sample_index].label < perceptron->number_of_labels)
                    fprintf(
                file,
                "%zu (%s)\n",
                        unlabeled_dataset->samples[sample_index].label,
                        perceptron->labels[unlabeled_dataset->samples[sample_index].label]
                    );
                else
                    fprintf(file, "%zu\n", unlabeled_dataset->samples[sample_index].label);
            }
            fclose(file);
            printf("\nResults saved to file.\n\n");
        }
        else printf("\nResults were discarded.\n");

        free_dataset(unlabeled_dataset);
        free_dataset(unlabeled_dataset_copy);
    }

    // Save the model if necessary
    if (strcmp(global_args.save_model_path, "") != 0)
    {
        if (save_model(global_args.save_model_path, perceptron) == SUCCESS)
            printf("\nModel saved succesfully.\n\n");
        else
            printf("\nFailed to save model.\n\n");
    }

    if (global_args.output_parameters) output_parameters(perceptron);
    free_perceptron(perceptron);
    return EXIT_SUCCESS;
}
