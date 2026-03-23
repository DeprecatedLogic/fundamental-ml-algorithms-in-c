## Fundamental Machine Learning Algorithms in C

This project implements three core machine learning algorithms from scratch in C:

* **K-Nearest Neighbors (KNN)**
* **K-Means Clustering**
* **Perceptron (Binary Classification)**

This repository contains my personal implementations of various machine learning algorithms, built from scratch in C.
The goal is to deeply understand the underlying mechanics of these algorithms.

---

### How to Run ?

Replace *knn* with the desired algorithm (*kmeans* or *perceptron*):

```bash
git clone https://github.com/ItzKarizma/fundamental-ml-algorithms-in-c.git
cd ./fundamental-ml-algorithms-in-c/knn
gcc -lm knn.c -o knn
./knn
```

Make sure you have a file with data (raw / not-standardized) and properly formatted.
Data files should be space-separated, with features first and the label (if necessary) last for each line.
Examples are already included in the repository, specifically in the *datasets* folder.

---

### KNN

* Supervised algorithm for classification.
* Takes user input for **features** and **K**.
* Predicts based on the closest neighbors.
* Handles invalid data robustly (as long as you don't try it).

---

### K-Means

* Unsupervised clustering algorithm.
* Supports automatic **K** detection using the **Elbow Method**.
* Handling of empty clusters is not entirely implemented but it's there (I'm lazy to finish this).
* Highly configurable: thresholds, custom K, max K, etc.

---

### Perceptron

* Supervised binary classifier using the perceptron update rule.
* Supports adjustable learning rate.
* Supports saving and loading a model.

---

### Features

* Fully written in C with no external ML libraries.
* Manual memory management (it's C after all).
* Standardized input parsing.
* Clear comments and docstrings (honestly, for future me in case one day I come back to this :P).

---

### Why ?

Built as a learning project to deeply understand how these algorithms work under the hood, not just to *use* machine learning, but to *build* it from scratch.

This was originally required for my AI studies (1st year of a Bachelor's degree), where we were asked to implement parts of these 3 (KMeans, KNN, Perceptron) algorithms. Just coding *some* functions wasn’t satisfying enough for me, so I decided to start completely from zero and make everything more dynamic and to my own liking...

---

### Author Notes

Made with caffeine and occasional frustration by [DeprecatedLogic](https://github.com/DeprecatedLogic).
