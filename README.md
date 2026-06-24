# Fundamental ML Algorithms from Scratch (in C)
A personal repository dedicated to building machine learning algorithms from the ground up. No external ML libraries, no high-level abstractions, just raw C, manual memory management, and the occasional battle with pointers.

## Why?
This repository started as a requirement for my first-year Bachelor’s studies in Artificial Intelligence.  
While the assignment asked for basic implementations of three core algorithms (KNN, K-Means, and Perceptron), I found myself wanting to understand the mechanics at a deeper level.  
I decided to start from zero to build something dynamic, robust, and entirely under my own control.  

This project is a practical deep-dive into the "black box" of AI, proving that you don't need heavy frameworks to implement intelligent behavior, just C and a lot of **patience**.

## The Algorithms
* **KNN**: A lazy learner that classifies data based on proximity.
* **K-Means**: An unsupervised clustering algorithm that supports automatic K detection via the Elbow Method.
* **Perceptron**: A binary classifier powered by the perceptron update rule, complete with model saving/loading and adjustable learning rates.
* **Decision Tree**: A hierarchical classifier featuring custom file parsing, binary serialization, and console-based tree visualization.  

## How to Run
Everything is built for a standard C environment.

Clone and navigate to `/src`:
```bash
git clone https://github.com/DeprecatedLogic/fundamental-ml-algorithms-in-c.git
cd ./fundamental-ml-algorithms-in-c/src
```

Compile and run:
```bash
# Example for KNN
gcc -lm knn.c -o knn
./knn
```

**Data Format:**
* Space-separated values (configurable via command line).
* Features first, label last (where applicable).
* Check the `/datasets` folder for examples, the file parser is quite robust too.  

## Technical Highlights
Written in pure C, zero dependencies.  
Manual memory management with extensive `valgrind` testing to ensure zero leaks.  
To avoid hardcoded values, an argument parser was implemented so that you can fully configure the algorithms directly via the command line.  

Includes Doxygen-style documentation and comments, primarily because I have a very short memory and I know future-me will love a reminder on what's going on!

---

## Author
Made with caffeine, persistence, and occasional frustration by [DeprecatedLogic](https://github.com/DeprecatedLogic).
