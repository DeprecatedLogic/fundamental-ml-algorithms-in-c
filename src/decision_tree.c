/*======================================================================
 *  Decision Tree (supervized learning)
 *
 *  Author      :  DeprecatedLogic  <https://www.github.com/DeprecatedLogic>
 *  Created     :  03 Mar 2026
 *
 *  Description :
 *      A simple implementation of the Perceptron algorithm for binary
 *      classification using supervised learning.
 *====================================================================*/

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define CMD_PRINT(ANSI_COLOR, PROGRAM_NAME, ARGS) printf("~>  %s %s %s %s\n", ANSI_COLOR, PROGRAM_NAME, ARGS, ANSI_COLOR_RESET)

#define MAX_PATH 256
#define MAX_DELIM 16
#define SUCCESS 0
#define FAILURE 1

typedef struct
{
    double *features;   // A pointer containing double values to keep the number of features dynamic 
    size_t encoded_label; // The encoded label of the sample (what it's classified as)
}
Sample;

typedef struct
{
    Sample *samples;
    size_t number_of_samples;
    size_t number_of_features;

    char **labels;
    size_t number_of_labels;
}
Dataset;

// Facilitates the counting of different encoded labels (size_t)
typedef struct
{
    size_t *frequencies;
    size_t size;
}
LabelFrequencies;

typedef struct Node Node;
struct Node
{   
    // Child nodes
    Node *left;
    Node *right;

    size_t *indices;
    size_t number_of_indices;

    size_t feature_index;
    double threshold;
    double gini;
    size_t majority_encoded_label;

    bool is_leaf;
    size_t depth;
    size_t index;
};

// Used for saving and loading a pre-trained decision tree.
//
// It would be cool to visualise the decision tree one day ?
// (C front-end is a territory I don't wanna dive in to be honest...)
typedef struct
{
    Node *root;         // We can find the rest easily with the first Node
    Dataset *dataset;   // We also need the dataset, since nodes only store the indices
                        // but not the actual values
}
DecisionTree; 

typedef struct
{
    size_t majority_label;
    size_t feature_index;
    double threshold;
    size_t is_leaf;
    int left_index;
    int right_index;
} NodeDisk;

typedef struct
{
    double value;
    size_t label;
} Tuple;

typedef struct
{
    char train_dataset_path[MAX_PATH];
    char prediction_dataset_path[MAX_PATH];
    char prediction_output_path[MAX_PATH];
    char save_model_path[MAX_PATH];
    char load_model_path[MAX_PATH];
    size_t number_of_features;
    char token_delimiter[MAX_DELIM];
    char decimals_delimiter;
    char comment_delimiter[MAX_DELIM];
    bool debug;
    size_t min_samples;
    size_t max_depth;
} Args;
Args global_args;

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

void free_node(Node *node)
{
    if (node == NULL) return;

    // Recursively go through all the nodes

    free_node(node->left);
    free_node(node->right);
    
    if(node->indices != NULL) free(node->indices);
    free(node);
    if (global_args.debug) printf("[free_node] Node freed\n"); // Debugging
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

size_t encode_label(Dataset *dataset, char *label)
{
    size_t label_index = 0;
    while (label_index < dataset->number_of_labels && strcmp(dataset->labels[label_index], label) != 0)
    {
        ++label_index;
    }
    
    if (label_index == dataset->number_of_labels)
    {
        dataset->labels = realloc(dataset->labels, sizeof(char*) * (dataset->number_of_labels + 1));
        if (dataset->labels == NULL)
        {
            fprintf(stderr, "[encoded_label] Failed to re-allocate memory for dataset->labels\n");
            exit(EXIT_FAILURE);
        }
        dataset->labels[dataset->number_of_labels++] = strdup(label);
    }
    if (global_args.debug) printf("[encoded_label] label: '%s' (length: %zu)\n", label, strlen(label));
    return label_index;
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

    /* Time to write some good quality code (lol) */

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
            .encoded_label = 0
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
                sample.encoded_label = encode_label(dataset, token_buffer);
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

        dataset->samples = (Sample *)realloc(dataset->samples, sizeof(Sample) * (dataset->number_of_samples + 1));
        dataset->samples[dataset->number_of_samples] = sample;
        ++dataset->number_of_samples;
        
        ++line_number;
    }
    dataset->number_of_features = number_of_features;
    fclose(data_file);
    return dataset;
}

Tuple *build_tuples_array(Dataset *dataset, Node *root, size_t feature_index)
{
    // Build an array of value-label tuples from root->indices
    Tuple *value_label_arr = (Tuple*)malloc(sizeof(Tuple) * root->number_of_indices);
    if (value_label_arr == NULL)
    {
        fprintf(stderr, "[build_tuples_array] Failed to allocate memory for value_label_arr\n");
        return NULL;
    }
    for (size_t i = 0; i < root->number_of_indices; ++i)
    {
        size_t root_sample_index = root->indices[i];
        Sample *sample = &dataset->samples[root_sample_index];

        value_label_arr[i] = (Tuple){
            .value = sample->features[feature_index],
            .label = sample->encoded_label
        };
    }

    return value_label_arr;
}

void sweep(
    Tuple *tuples, size_t number_of_tuples, size_t number_of_labels,
    double *threshold, double *gini
)
{
    LabelFrequencies right_counts = {
        .frequencies = (size_t*)calloc(number_of_labels, sizeof(size_t)),
        .size = number_of_labels
    };
    LabelFrequencies left_counts = {
        .frequencies = (size_t*)calloc(number_of_labels, sizeof(size_t)),
        .size = number_of_labels
    };

    for (size_t tuple_index = 0; tuple_index < number_of_tuples; ++tuple_index)
    {
        size_t label = tuples[tuple_index].label;
        ++right_counts.frequencies[label];
    }

    double best_gini = (double)INFINITY;
    double best_threshold = 0;

    for (size_t tuple_index = 0; tuple_index < number_of_tuples-1; ++tuple_index)
    {
        size_t current_label = tuples[tuple_index].label;

        // Move sample from right to left
        ++left_counts.frequencies[current_label];
        --right_counts.frequencies[current_label];

        // Skip if the next value is the same, meaning no valid split
        if (tuples[tuple_index].value == tuples[tuple_index+1].value) continue;

        // Keep track of frequencies size
        size_t left_size = tuple_index + 1;
        size_t right_size = number_of_tuples - left_size;

        // Compute Gini (left)
        double gini_left = 1.0;
        for (size_t j = 0; j < number_of_labels; ++j)
        {
            double p = left_counts.frequencies[j] / (double)left_size;
            gini_left -= p * p;
        }

        // Compute Gini (right)
        double gini_right = 1.0;
        for (size_t j = 0; j < number_of_labels; ++j)
        {
            double p = right_counts.frequencies[j] / (double)right_size;
            gini_right -= p * p;
        }

        // Weighted Gini
        double gini = ((double)left_size / number_of_tuples) * gini_left +
            ((double)right_size / number_of_tuples) * gini_right;

        if (gini < best_gini)
        {
            best_gini = gini;
            best_threshold = (tuples[tuple_index].value + tuples[tuple_index+1].value) / 2; // value in-between
        }
    }

    *threshold = best_threshold;
    *gini = best_gini;

    free(right_counts.frequencies);
    free(left_counts.frequencies);
}

LabelFrequencies *get_label_frequencies(Dataset *dataset, Node *node)
{
    // Build an array with encoded label frequencies
    LabelFrequencies *lf = (LabelFrequencies*)malloc(sizeof(LabelFrequencies));
    if (lf == NULL)
    {
        fprintf(stderr, "[get_label_frequencies] Failed to allocate memory for lf\n");
        return NULL;
    }
    lf->frequencies = (size_t*)calloc(dataset->number_of_labels, sizeof(size_t));
    if (lf->frequencies == NULL)
    {
        fprintf(stderr, "[get_label_frequencies] Failed to allocate memory for lf->frequencies and initialize all members to 0\n");
        free(lf);
        return NULL;
    }
    lf->size = dataset->number_of_labels;

    if (global_args.debug) printf("[get_label_frequencies] Counting the label frequencies\n"); // Debugging
    for (size_t i = 0; i < node->number_of_indices; ++i)
    {
        size_t node_sample_index = node->indices[i];
        if (global_args.debug) printf("[get_label_frequencies] node->indices[%zu]: %zu\n",
            i, node_sample_index
        ); // Debugging

        Sample *node_sample = &dataset->samples[node_sample_index];
        if (node_sample->encoded_label >= dataset->number_of_labels)
        {
            fprintf(
                stderr,
                "[get_label_frequencies] Sample has an encoded label (%zu) >= the number of labels (%zu) available in the dataset\n",
                node_sample->encoded_label,
                dataset->number_of_labels
            );
            free(lf->frequencies);
            free(lf);
            return NULL;
        } // Debugging
        
        if (global_args.debug) printf(
            "[get_label_frequencies] Incrementing lf->frequencies[%zu] by 1\n",
            node_sample->encoded_label
        ); // Debugging
        ++lf->frequencies[node_sample->encoded_label];

        if (global_args.debug) printf(
            "[get_label_frequencies] Checking lf->frequencies[%zu] value: %zu\n",
            node_sample->encoded_label,
            lf->frequencies[node_sample->encoded_label]
        ); // Debugging
    }

    return lf;
}

void merge_sort(Tuple *tuples, size_t size)
{
    // todo: implement the merge sort algorithm (in-place sorting)
    // and replace the qsort in train_node()
}

int compare_tuple(const void *a, const void *b)
{
    double diff = ((Tuple*)a)->value - ((Tuple*)b)->value;
    return (diff > 0) - (diff < 0);
}

bool should_stop_splitting(Node *node)
{
    return (
        node->is_leaf ||
        node->number_of_indices < global_args.min_samples ||
        node->depth >= global_args.max_depth
    );
}

int train_node(Dataset *dataset, Node *root)
{
    // Build an array with encoded label frequencies
    LabelFrequencies *lf = get_label_frequencies(dataset, root);
    if (lf == NULL)
    {
        return FAILURE;
    }
    
    // Find the majority label & number of non-zero labels
    size_t majority_encoded_label = 0;
    size_t max_frequency = 0;
    size_t non_zero_labels = 0;
    if (global_args.debug) printf("[train_node] Looping through the label frequencies\n"); // Debugging
    for (size_t frequency_index = 0; frequency_index < lf->size; ++frequency_index)
    {
        if (global_args.debug) printf("[train_node] [encoded label: %zu] [frequency: %zu]\n",
            frequency_index, lf->frequencies[frequency_index]
        ); // Debugging
        if (lf->frequencies[frequency_index] > 0)
        {
            ++non_zero_labels;

            if (lf->frequencies[frequency_index] > max_frequency)
            {
                max_frequency = lf->frequencies[frequency_index];
                majority_encoded_label = frequency_index;
            }
        }
    }
    root->majority_encoded_label = majority_encoded_label;
    root->is_leaf = (non_zero_labels == 1);
    if (global_args.debug) printf("[train_node] Node is %sa leaf\n", (root->is_leaf) ? "" : "not "); // Debugging
    
    // Free LabelFrequencies allocated memory
    free(lf->frequencies);
    free(lf);

    // Stop ?
    if (should_stop_splitting(root))
    {
        if (global_args.debug) printf("[train_node] Stopped splitting\n"); // Debugging
        return SUCCESS;
    }

    // Find the best split
    double best_gini = (double)INFINITY;
    double best_threshold = 0;
    size_t best_feature_index = 0;
    for (size_t feature_index = 0; feature_index < dataset->number_of_features; ++feature_index)
    {
        Tuple *tuples = build_tuples_array(dataset, root, feature_index);
        if (tuples == NULL)
        {
            fprintf(stderr, "[train_node] Failed to build an array of tuples\n");
            return FAILURE;
        }
        //merge_sort(tuples, root->number_of_indices);
        // instead of merge_sort for now:
        qsort(tuples, root->number_of_indices, sizeof(Tuple), compare_tuple);

        double out_threshold = 0;
        double out_gini = 0;
        sweep(
            tuples,
            root->number_of_indices,
            dataset->number_of_labels,
            &out_threshold,
            &out_gini
        );

        if (out_gini < best_gini)
        {
            best_gini = out_gini;
            best_threshold = out_threshold;
            best_feature_index = feature_index;
        }
        free(tuples);
    }
    root->feature_index = best_feature_index;
    root->threshold = best_threshold;

    // Initialize child nodes
    Node *left = (Node*)calloc(1, sizeof(Node));
    if (left == NULL)
    {
        fprintf(stderr, "[train_node] Failed to allocate memory for node left and initialize all members to 0\n");
        return FAILURE;
    }
    Node *right = (Node*)calloc(1, sizeof(Node));
    if (right == NULL)
    {
        fprintf(stderr, "[train_node] Failed to allocate memory for node right and initialize all members to 0\n");
        free(left);
        return FAILURE;
    }
    
    left->depth = root->depth + 1;
    right->depth = root->depth + 1;
    
    left->indices = (size_t*)malloc(sizeof(size_t) * root->number_of_indices);
    if (left->indices == NULL)
    {
        fprintf(stderr, "[train_node] Failed to allocate memory for node left's indices\n");
        free(left);
        free(right);
        return FAILURE;
    }
    right->indices = (size_t*)malloc(sizeof(size_t) * root->number_of_indices);
    if (left->indices == NULL)
    {
        fprintf(stderr, "[train_node] Failed to allocate memory for node right's indices\n");
        free(left->indices);
        free(left);
        free(right);
        return FAILURE;
    }

    // Partition root->indices into left & right sets
    for (size_t i = 0; i < root->number_of_indices; ++i)
    {
        size_t root_sample_index = root->indices[i];
        Sample *sample = &dataset->samples[root_sample_index];
        Node *selected = (sample->features[root->feature_index] <= root->threshold) ? left : right;
        selected->indices[selected->number_of_indices++] = root_sample_index;
    }
    // todo: verify these 2 reallocs (I'm too tired right now...)
    left->indices = realloc(left->indices, sizeof(size_t) * left->number_of_indices);
    right->indices = realloc(right->indices, sizeof(size_t) * right->number_of_indices);
    
    // Let's not forget to assign the child nodes to their parent node too :)
    root->left = left;
    root->right = right;

    // Debugging
    if (global_args.debug) printf("[train_node] Node depth %zu: feature=%zu threshold=%f samples=%zu\n",
       root->depth,
       root->feature_index,
       root->threshold,
       root->number_of_indices
    );

    // Call train_node on both child nodes
    train_node(dataset, left);
    train_node(dataset, right);

    return SUCCESS;
}

void output_tree_info(Dataset *dataset, Node *node)
{
    if (node == NULL) return;

    // Indentation
    for (size_t i = 0; i < node->depth; ++i)
        printf("  ");

    printf("Node (depth: %zu, samples: %zu) ", node->depth, node->number_of_indices);

    if (node->is_leaf)
    {
        char *majority_label = dataset->labels[node->majority_encoded_label];
        printf("[LEAF] label: %zu (%s)\n", node->majority_encoded_label, majority_label);
    }
    else
    {
        printf("[SPLIT] feature: %zu | threshold: %.5f\n", node->feature_index, node->threshold);

        output_tree_info(dataset, node->left);
        output_tree_info(dataset, node->right);
    }
}

// Unused but I like recursive B)
size_t recursively_predict_sample(const Node *node, const Sample *sample)
{
    if (node->is_leaf) return node->majority_encoded_label;

    if (sample->features[node->feature_index] <= node->threshold)
        return recursively_predict_sample(node->left, sample);
    return recursively_predict_sample(node->right, sample);
}

size_t predict_sample(const Node *node, const Sample *sample)
{
    while(!node->is_leaf)
        node = (sample->features[node->feature_index] <= node->threshold) ? node->left : node->right;

    return node->majority_encoded_label;
}

int predict_samples_in_dataset(const Dataset *dataset, const Node *root, const Dataset *pred_dataset)
{
    FILE *out = NULL;
    if (strcmp(global_args.prediction_output_path, "") != 0)
    {
        out = fopen(global_args.prediction_output_path, "w");
        if (out == NULL) return FAILURE;
    }

    for (size_t sample_index = 0; sample_index < pred_dataset->number_of_samples; ++sample_index)
    {
        Sample *sample = &pred_dataset->samples[sample_index];
        size_t predicted_encoded_label = predict_sample(root, sample);
        
        fprintf(stdout, "Prediction for sample %zu: ", sample_index);

        for (size_t iFeature = 0; iFeature < dataset->number_of_features; ++iFeature)
        {
            fprintf(stdout, "%g ", sample->features[iFeature]);
            if (out != NULL) fprintf(out, "%.17g ", sample->features[iFeature]);
        }

        fprintf(stdout, "[%s]\n", dataset->labels[predicted_encoded_label]);
        if (out != NULL) fprintf(out, "[%s]\n", dataset->labels[predicted_encoded_label]);
    }
    if (out != NULL) fclose(out);
    
    return SUCCESS;
}

void collect_all_nodes(Node *node, Node **collection, size_t *index)
{
    // NULL instead of checking is_leaf because parents are NOT leaves but still need to be assigned an index
    if (node == NULL) return;

    node->index = *index;
    collection[*index] = node;
    ++(*index);

    collect_all_nodes(node->left, collection, index);
    collect_all_nodes(node->right, collection, index);
}

size_t total_nodes(Node *node)
{
    return (node->is_leaf) ? 1 : 1 + total_nodes(node->left) + total_nodes(node->right);
}

int save_model(const DecisionTree *model, const char *path)
{
    if (global_args.debug) printf("[save_model] Saving model as: %s\n", path);
    
    // Open the file in writing mode
    FILE* model_file = fopen(path, "wb");
    if (model_file == NULL) // Invalid path or maybe insufficient permissions to write ?
    {
        fprintf(stderr, "[save_model] Failed to open file: %s\n", path);
        return FAILURE;
    }

    // Write encoding version (to ensure compatibility)
    size_t version = 0x76312E30; // v1.0
    if (global_args.debug) printf("[save_model] Writing encoding version: %zu\n", version);
    fwrite(&version, sizeof(size_t), 1, model_file);

    // Write the number of features
    if (global_args.debug)
        printf("[save_model] Writing the number of features (%zu)\n", model->dataset->number_of_features);
    fwrite(&model->dataset->number_of_features, sizeof(size_t), 1, model_file);

    // Write dataset labels
    if (global_args.debug) printf("[save_model] Writing dataset labels\n");
    fwrite(&model->dataset->number_of_labels, sizeof(size_t), 1, model_file);
    for (size_t label_index = 0; label_index < model->dataset->number_of_labels; ++label_index)
    {
        char *label = model->dataset->labels[label_index];
        size_t len = strlen(label);
        fwrite(&len, sizeof(size_t), 1, model_file);
        fwrite(label, sizeof(char), len, model_file);
    }

    // Collect all nodes in pre-order
    if (global_args.debug) printf("[save_model] Collecting nodes in pre-order\n");
    size_t nodes_count = total_nodes(model->root);
    Node **collection = (Node**)malloc(sizeof(Node*) * nodes_count);
    if (collection == NULL)
    {
        fprintf(stderr, "[save_model] Failed to allocate memory for collection\n");
        fclose(model_file);
        remove(path);
        return FAILURE;
    }

    size_t collection_index = 0;
    collect_all_nodes(model->root, collection, &collection_index);

    // Write nodes
    if (global_args.debug) printf("[save_model] Writing the number of nodes: %zu\n", nodes_count);
    fwrite(&nodes_count, sizeof(size_t), 1, model_file);
    
    for (size_t node_index = 0; node_index < nodes_count; ++node_index)
    {
        Node *node = collection[node_index];
        if (global_args.debug)
            printf("[save_model] Creating a node disk out of collection[%zu] (node->index %zu)\n", node_index, node->index);
        
        NodeDisk disk;
        disk.majority_label = node->majority_encoded_label;
        disk.feature_index = node->feature_index;
        disk.threshold = node->threshold;
        disk.is_leaf = node->is_leaf;
        if (node->is_leaf)
        {
            disk.left_index = -1;
            disk.right_index = -1;
        }
        else
        {
            disk.left_index = node->left->index;
            disk.right_index = node->right->index;
        }

        if (global_args.debug)
            printf("[save_model] Node disk created\nWriting node disk to file\n");

        fwrite(&disk, sizeof(NodeDisk), 1, model_file);

        if (global_args.debug)
            printf("[save_model] Wrote to file (%s) successfully\n", path);
    }
    if (global_args.debug) printf("[save_model] %zu NodeDisk objects were written successfully\n", nodes_count);
    
    free(collection);
    fclose(model_file);
    if (global_args.debug) printf("[save_model] Collection memory freed and file closed\n");

    return SUCCESS;
}

DecisionTree *load_model(const char *path)
{
    if (global_args.debug) printf("[load_model] Loading model from: %s\n", path);

    // Allocate memory for the decision tree
    DecisionTree *model = (DecisionTree*)malloc(sizeof(DecisionTree));
    if (model == NULL)
    {
        fprintf(stderr, "[load_model] Failed to allocate memory for model\n");
        return NULL;
    }
    // Allocate memory for dataset
    model->dataset = (Dataset*)calloc(1, sizeof(Dataset));
    if (model->dataset == NULL)
    {
        fprintf(stderr, "[load_model] Failed to allocate memory for model\n");
        free(model);
        return NULL;
    }
    if (global_args.debug) printf("[load_model] Allocated memory for model and model->dataset\n");
    
    // Open file
    FILE* model_file = fopen(path, "rb");
    if (model_file == NULL) // File doesn't exist or maybe insufficient permissions ?
    {
        fprintf(stderr, "[load_model] Failed to open file: %s\n", path);
        free_dataset(model->dataset);
        free(model);
        return NULL;
    }
    if (global_args.debug) printf("[load_model] Opened model file in 'rb' mode\n");

    // Read the encoding version
    if (global_args.debug) printf("[load_model] Reading the encoding version\n");
    size_t version = 0;
    fread(&version, sizeof(size_t), 1, model_file);
    

    if (global_args.debug) printf("[load_model] Checking if the encoding version matches v1.0: %s\n", (version == 0x76312E30) ? "true" : "false");
    if (version == 0x76312E30)
    {
        // Read the number of features
        if (global_args.debug) printf("[load_model] Reading the number of features\n");
        fread(&model->dataset->number_of_features, sizeof(size_t), 1, model_file);
        if (global_args.number_of_features != model->dataset->number_of_features)
        {
            printf("[load_model] Feature count mismatch detected. Using model value (%zu).\n", model->dataset->number_of_features);
            global_args.number_of_features = model->dataset->number_of_features;
        }
        if (global_args.debug) printf("[load_model] Number of features %zu\n", model->dataset->number_of_features);

        // Read the number of labels
        if (global_args.debug) printf("[load_model] Reading the number of labels\n");
        fread(&model->dataset->number_of_labels, sizeof(size_t), 1, model_file);
        model->dataset->labels = (char**)malloc(sizeof(char*) * model->dataset->number_of_labels);
        if (model->dataset->labels == NULL)
        {
            fprintf(stderr, "[load_model] Failed to allocate memory for model->dataset->labels\n");
            free_dataset(model->dataset);
            free(model);
            fclose(model_file);
            return NULL;
        }
        if (global_args.debug) printf("[load_model] Number of labels %zu\n", model->dataset->number_of_labels);

        // Read the labels
        if (global_args.debug) printf("[load_model] Reading the labels\n");
        for (size_t label_index = 0; label_index < model->dataset->number_of_labels; ++label_index)
        {
            // Read the length
            if (global_args.debug) printf("[load_model] Reading the length of the label %zu\n", label_index);
            size_t len = 0;
            fread(&len, sizeof(size_t), 1, model_file);
            if (global_args.debug) printf("[load_model] Length %zu\n", len);

            // Allocate char * length memory
            model->dataset->labels[label_index] = (char*)malloc(sizeof(char) * len + 1);
            if (model->dataset->labels[label_index] == NULL)
            {
                fprintf(stderr, "[load_model] Failed to allocate memory for model->dataset->labels[%zu]", label_index);
                free_dataset(model->dataset);
                free(model);
                fclose(model_file);
                return NULL;
            }
            if (global_args.debug) printf("[load_model] Allocated memory for the label\n");

            // Read len amount of chars
            if (global_args.debug) printf("[load_model] Reading %zu character(s)\n", len);
            fread(model->dataset->labels[label_index], sizeof(char), len, model_file);

            if (global_args.debug) printf("[load_model] Adding null terminator\n");
            model->dataset->labels[label_index][len] = '\0'; // null terminate

            if (global_args.debug) printf("[load_model] Label: %s\n", model->dataset->labels[label_index]);
        }

        // Read the amount of nodes
        if (global_args.debug) printf("[load_model] Reading the amount of nodes\n");
        size_t nodes_count = 0;
        fread(&nodes_count, sizeof(size_t), 1, model_file);
        if (global_args.debug) printf("[load_model] Nodes read: %zu\n", nodes_count);
        
        if (global_args.debug) printf("[load_model] Allocating memory for disk_nodes\n");
        NodeDisk *disk_nodes = (NodeDisk*)malloc(sizeof(NodeDisk) * nodes_count);
        if (disk_nodes == NULL)
        {
            fprintf(stderr, "[load_model] Failed to allocate memory for disk_nodes\n");
            free_dataset(model->dataset);
            free(model);
            fclose(model_file);
            return NULL;
        }
        fread(disk_nodes, sizeof(NodeDisk), nodes_count, model_file);
        if (global_args.debug) printf("[load_model] Disk nodes read successfully\n");

        if (global_args.debug) printf("[load_model] Allocating memory for collection\n");
        Node **collection = (Node**)malloc(sizeof(Node*) * nodes_count);
        if (collection == NULL)
        {
            fprintf(stderr, "[load_model] Failed to allocate memory for collection\n");
            free(disk_nodes);
            free_dataset(model->dataset);
            free(model);
            fclose(model_file);
            return NULL;
        }
        size_t collection_index = 0;

        // Assign data to nodes
        if (global_args.debug) printf("[load_model] Assigning data to nodes\n");
        for (size_t node_index = 0; node_index < nodes_count; ++node_index)
        {
            if (global_args.debug) printf("[load_model] Working on node at collection[%zu]\n", node_index);
            collection[node_index] = (Node*)calloc(1, sizeof(Node));
            
            NodeDisk disk = disk_nodes[node_index];
            Node *node = collection[node_index];

            if (global_args.debug) printf("[load_model] Assigning encoded majority label: %zu\n", disk.majority_label);
            node->majority_encoded_label = disk.majority_label;
            if (global_args.debug) printf("[load_model] Assigning feature index: %zu\n", disk.feature_index);
            node->feature_index = disk.feature_index;
            if (global_args.debug) printf("[load_model] Assigning threshold: %g\n", disk.threshold);
            node->threshold = disk.threshold;
            if (global_args.debug) printf("[load_model] Assigning feature index: %s\n", (disk.is_leaf) ? "true" : "false");
            node->is_leaf = disk.is_leaf;

            if (global_args.debug) printf("[load_model] Setting node's children to NULL\n");
            node->left = NULL;
            node->right = NULL;
        }
        if (global_args.debug) printf("[load_model] Data assigned successfully\n");

        // Build tree
        if (global_args.debug) printf("[load_model] Building decision tree by connecting nodes\n");
        for (size_t node_index = 0; node_index < nodes_count; ++node_index)
        {
            int left_index = disk_nodes[node_index].left_index;
            int right_index = disk_nodes[node_index].right_index;

            if (left_index != -1)
            {
                collection[node_index]->left = collection[left_index];
                collection[node_index]->left->depth = collection[node_index]->depth + 1;
            }
            if (right_index != -1)
            {
                collection[node_index]->right = collection[right_index];
                collection[node_index]->right->depth = collection[node_index]->depth + 1;
            }
        }
        if (global_args.debug) printf("[load_model] Tree built successfully\n");
        if (global_args.debug) printf("[load_model] Setting the the node with index 0 as root\n");
        model->root = collection[0]; // pre-order so root is index 0
        if (global_args.debug) printf("[load_model] Done\n");

        if (global_args.debug) printf("[load_model] Freeing allocated memory for disk_nodes and collection\n");
        free(disk_nodes);
        free(collection);

        if (global_args.debug) printf("[load_model] Closing file\n");
        fclose(model_file);
        if (global_args.debug) printf("[load_model] Done\n");

        return model;
    }
    if (global_args.debug) printf("[load_model] Version not compatible, closing file\n");
    fclose(model_file);
    if (global_args.debug) printf("[load_model] Done\n");
    return NULL;
}

void print_usage_examples(const char* program_name)
{
    printf("=== %s Usage Examples ===\n", program_name);

    printf("\n1. Train a decision tree on a dataset with 3 features per sample\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/decision_tree/train -n 3");

    printf("\n2. Use semicolon-separated values & standardize tokens\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/decision_tree/train.ssv -n 3 --tdel ;");

    printf("\n3. Floating-point numbers use comma as decimal delimiter\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/decision_tree/train -n 3 --ddel ,");

    printf("\n4. Train & save a decision tree model\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/decision_tree/train -n 3 --save models/decision_tree_model");

    printf("\n5. Load a pre-trained decision tree with 4 features per sample\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "--load models/decision_tree_model -n 4");

    printf("\n6. Predict unlabeled samples and save the output\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-p datasets/decision_tree/unlabeled -o predictions.txt --load models/decision_tree_model -n 3");

    printf("\n7. Train with custom tree constraints (min samples per node & max depth)\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/decision_tree/train -n 3 --min-samples 5 --max-depth 10");

    printf("\n8. Run a demo using the repo's sample dataset:\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/decision_tree/1feat_train -p datasets/decision_tree/1feat_test -n 1");

    printf("\n9. Train with debug/verbose output\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "-f datasets/decision_tree/train -n 3 --debug");

    printf("\n10. Show all available options\n");
    CMD_PRINT(ANSI_COLOR_GREEN, program_name, "help");

    printf("\nNotes:\n");
    printf("- Options order does not matter.\n");
    printf("- Combine multiple options as needed (e.g., training + testing + save).\n");
}

void print_usage(const char *program_name)
{
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -f <string>                   Dataset file path\n");
    printf("  -p <string>                   Unlabeled dataset file path containing the samples to predict\n");
    printf("  -o <string>                   Save output file containing samples with predicted labels\n");
    printf("  -n <string>                   Number of features per sample (default: 1)\n");
    printf("  --min-samples <integer>       The minimum number of samples required for a node to split (default: 2)\n");
    printf("  --max-depth <integer>         The maximum depth the tree can reach (default: 999)\n");
    printf("  --save <string>               Path to save the trained decision tree\n");
    printf("  --load <string>               Model to load for inference\n");
    printf("  --tdel <string>               Token delimiter (default: space)\n");
    printf("  --ddel <char>                 Decimal delimiter (default: .)\n");
    printf("  --cdel <string>               Comment delimiter (default: #)\n");
    printf("\n");
    printf("  -v, --debug                   Verbose, used for debugging (false by default)\n");
    printf("  examples                      Shows a list of usage examples\n");
    printf("  help                          Usage and options menu (this command)\n");
}

Args parse_args(int argc, char **argv)
{
    Args args;

    // Default values
    strncpy(args.train_dataset_path, "", MAX_PATH-1);
    strncpy(args.prediction_dataset_path, "", MAX_PATH-1);
    strncpy(args.prediction_output_path, "", MAX_PATH-1);
    args.number_of_features = 1;
    args.min_samples = 2;
    args.max_depth = 999;
    strncpy(args.save_model_path, "", MAX_PATH-1);
    strncpy(args.load_model_path, "", MAX_PATH-1);
    strncpy(args.token_delimiter, " ", MAX_DELIM-1);
    args.decimals_delimiter = '.';
    strncpy(args.comment_delimiter, "#", MAX_DELIM-1);
    args.debug = false;
    

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-f") == 0 && i+1 < argc)
            strncpy(args.train_dataset_path, argv[++i], MAX_PATH-1);
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
        else if (strcmp(argv[i], "--min-samples") == 0 && i+1 < argc)
        {
            char *endptr;
            args.min_samples = strtoul(argv[++i], &endptr, 10);
            if (*endptr != '\0') {
                fprintf(stderr, "Invalid value for minimum samples: %s\n", argv[i]);
                exit(EXIT_FAILURE);
            }
        }
        else if (strcmp(argv[i], "--max-depth") == 0 && i+1 < argc)
        {
            char *endptr;
            args.max_depth = strtoul(argv[++i], &endptr, 10);
            if (*endptr != '\0') {
                fprintf(stderr, "Invalid value for max depth: %s\n", argv[i]);
                exit(EXIT_FAILURE);
            }
        }
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

int main(int argc, char *argv[])
{
    global_args = parse_args(argc, argv);
    if (global_args.debug)
    {
        printf("Debugging: true\n");
        printf("\n");
        printf("Dataset path: %s\n",  global_args.train_dataset_path);
        printf("Prediction dataset path: %s\n",  global_args.prediction_dataset_path);
        printf("Prediction output path: %s\n",  global_args.prediction_output_path);
        printf("Save model path: %s\n",  global_args.save_model_path);
        printf("Load model path: %s\n",  global_args.load_model_path);
        printf("Number of features: %zu\n", global_args.number_of_features);
        printf("Token delimiter: '%s'\n", global_args.token_delimiter);
        printf("Decimal delimiter: '%c'\n", global_args.decimals_delimiter);
        printf("Comment delimiter: '%s'\n", global_args.comment_delimiter);
        printf("Minimum samples: '%zu'\n", global_args.min_samples);
        printf("Maximum tree depth: '%zu'\n", global_args.max_depth);
        printf("\n");
    }

    Dataset *dataset = NULL;
    Node *root = NULL;

    DecisionTree *decision_tree = NULL;
    if (strcmp(global_args.load_model_path, "") != 0)
    {
        if (global_args.debug) printf("[main] Loading decision tree model\n");
        decision_tree = load_model(global_args.load_model_path);
        if (decision_tree == NULL || decision_tree->dataset == NULL || decision_tree->root == NULL)
        {
            fprintf(stderr, "[main] Failed to load the decision tree model with the following path: %s\n", global_args.load_model_path);
            return EXIT_FAILURE;
        }
        if (global_args.debug) printf("[main] Loaded successfully\n");
        dataset = decision_tree->dataset;
        root = decision_tree->root;
    }
    else
    {
        if (global_args.debug) printf("[main] Loading dataset\n");
        // Read the data while doing in-place label encoding per sample
        dataset = read_data_from_file(
            global_args.train_dataset_path,
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
        if (global_args.debug) printf("[main] Dataset loaded\nCreating root node\n");
        root = (Node*)calloc(1, sizeof(Node));
        if (root == NULL)
        {
            fprintf(stderr, "[main] Failed to allocate memory for root and initialize all members to 0\n");
            free_dataset(dataset);
            return EXIT_FAILURE;
        }

        root->indices = (size_t*)malloc(sizeof(size_t) * dataset->number_of_samples);
        if (root->indices == NULL)
        {
            fprintf(stderr, "[main] Failed to allocate memory for root->indices\n");
            free_node(root);
            free_dataset(dataset);
            return EXIT_FAILURE;
        }

        for (size_t sample_index = 0; sample_index < dataset->number_of_samples; ++sample_index)
            root->indices[sample_index] = sample_index;

        root->number_of_indices = dataset->number_of_samples;
        if (global_args.debug) printf("[main] Root note created\nTraining tree\n");

        if (train_node(dataset, root) == FAILURE)
        {
            fprintf(stderr, "[main] Failed to train tree\n");
            free_node(root);
            free_dataset(dataset);
            return EXIT_FAILURE;
        }
        if (global_args.debug) printf("[main] Tree trained\n");

        if (global_args.debug) printf("[main] Allocating memory for decision tree and assigning dataset and root node\n");
        decision_tree = malloc(sizeof(DecisionTree));
        if (decision_tree == NULL)
        {
            fprintf(stderr, "[main] Failed to allocate memory for decision_tree\n");
            free_node(root);
            free_dataset(dataset);
            return EXIT_FAILURE;
        }
        decision_tree->root = root;
        decision_tree->dataset = dataset;
    }
    
    output_tree_info(dataset, root);

    if (global_args.debug) printf("[main] Checking if prediction dataset path is not empty: %s\n", global_args.prediction_dataset_path);
    if (strcmp(global_args.prediction_dataset_path, "") != 0)
    {
        if (global_args.debug) printf("[main] Loading dataset with samples to predict\n");
        Dataset *prediction_dataset = read_data_from_file(
            global_args.prediction_dataset_path,
            global_args.number_of_features,
            false,
            global_args.token_delimiter,
            global_args.decimals_delimiter,
            global_args.comment_delimiter
        );
        if (prediction_dataset == NULL)
        {
            fprintf(stderr, "[main] Failed to load prediction dataset\n");
            free_node(root);
            free_dataset(dataset);
            free(decision_tree);
            return EXIT_FAILURE;
        }
        if (global_args.debug) printf("[main] Dataset loaded\nRunning algorithm to predict labels for dataset samples\n");
        predict_samples_in_dataset(dataset, root, prediction_dataset);
        if (global_args.debug) printf("[main] Algorithm ended\nFreeing dataset\n");
        free_dataset(prediction_dataset);
        if (global_args.debug) printf("[main] Prediction dataset freed\n");
    }
    
    if (global_args.debug) printf("[main] Checking if save model path is not empty: %s\n", global_args.save_model_path);
    if (strcmp(global_args.save_model_path, "") != 0)
    {
        if (global_args.debug) printf("[main] Saving model\n");
        if (save_model(decision_tree, global_args.save_model_path) == FAILURE)
        {
            fprintf(stderr, "[main] Failed to save model\n");
            free_node(root);
            free_dataset(dataset);
            free(decision_tree);
            return EXIT_FAILURE;
        }
        if (global_args.debug) printf("[main] Model saved\n");
    }

    if (global_args.debug) printf("[main] Freeing memory\n");
    free_node(root);
    free_dataset(dataset);
    free(decision_tree);
    if (global_args.debug) printf("[main] Memory freed successfully\n");
    return EXIT_SUCCESS;
}
