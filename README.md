## Fundamental Machine Learning Algorithms in C

This project implements the following ML algorithms:

* **K-Nearest Neighbors (KNN)**
* **K-Means Clustering**
* **Perceptron (Binary Classification)**
* **Decision Tree**

This repository contains my personal implementations of various machine learning algorithms, built from scratch in C.
The goal is to deeply understand the underlying mechanics of these algorithms.

---

### How to Run ?

Replace *knn* with the desired algorithm (*kmeans*, *perceptron*, or *decision_tree*):

```bash
git clone https://github.com/DeprecatedLogic/fundamental-ml-algorithms-in-c.git
cd ./fundamental-ml-algorithms-in-c/src
gcc -lm knn.c -o knn
./knn
```

Make sure you have a properly formatted file.
Data files should be space-separated, with features first and the label (if necessary) last for each line/row.
Examples are already included in the repository, specifically in the *datasets* folder.

---

### KNN
A supervised algorithm for classification.

* Takes user input for **features** and **K**.
* Predicts based on the closest neighbors.
* Handles invalid data robustly (as long as you don't try it).

---

### K-Means
An unsupervised clustering algorithm.

* Supports automatic **K** detection using the **Elbow Method**.
* Handling of empty clusters is not entirely implemented but it's there (I'm lazy to finish this).
* Highly configurable: thresholds, custom K, max K, etc.

---

### Perceptron
A supervised binary classifier using the perceptron update rule.

* Supports adjustable learning rate.
* Supports saving and loading a model.

---

### Decision Tree
A supervised classifier that partitions data based on feature thresholds to create a hierarchical decision structure.

* Features a custom, efficient file reader that handles large datasets with configurable delimiters and automatic comment stripping.
* Includes binary serialization to save and load trained models, allowing for consistent deployment and inspection.
* Built-in utilities for traversing and printing the learned node structure to the console for model transparency.

---

### Features

* Fully written in C with no external ML libraries.
* Manual memory management (it's C after all).
* Support for CLI-based configuration without hardcoded values.
* Clear comments and docstrings (honestly, for future me in case one day I come back to this :P).

---

### Why ?

Built as a learning project to deeply understand how these algorithms work under the hood, not just to *use* machine learning, but to *build* it from scratch.

This was originally required for my AI studies (1st year of a Bachelor's degree), where we were asked to implement parts of these three (KMeans, KNN, Perceptron) algorithms. Just coding *some* functions wasn’t satisfying enough for me, so I decided to start completely from zero and make everything more dynamic and to my own liking...

---

### Author Notes

Made with caffeine and occasional frustration by [DeprecatedLogic](https://github.com/DeprecatedLogic).
