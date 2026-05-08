# Project 2: Learning from Data

## 1. Core Idea

This project is about **Learning from Data**, where we use observed data to train an AI model and then apply the model to unseen data.

The basic workflow is:

1. We have a **training set**, which contains observed data.
2. A model is learned from the training set.
3. The model is then used on unseen data, namely the **test set**.

The key assumption is:

> The training set and test set follow the same data distribution, or at least share the same hidden pattern.

Therefore, the AI model can be regarded as an abstraction of the data distribution.

---

## 2. Typical Pipeline

A typical learning-from-data pipeline contains four main stages:

### 2.1 Feature Extraction

Raw data, such as images or dialog text, is converted into structured feature vectors.

For example, an image may be transformed into a vector:

$$
\mathbf{x} \in \mathbb{R}^d
$$

where $d$ is the feature dimension.

### 2.2 Feature Selection

Some features may be removed to reduce the input dimension.

The goal is to keep useful information while making the problem easier and faster to solve.

### 2.3 Training

A model is generated from the training set.

### 2.4 Deployment

The trained model is used to process unseen test data.

---

## 3. How to Avoid Overfitting

Since the test set is not accessible during training, we need to avoid overfitting the training data.

A common method is:

1. Split the original training set into:
    - a smaller training set;
    - a validation set.
2. Train the model on the smaller training set.
3. Evaluate the model on the validation set.
4. Tune the model parameters according to validation performance.
5. Finally, train the model again from scratch on the whole training set using the selected parameters.

This process helps make sure the model does not only memorize the training samples.

---

## 4. Supervised Learning

In supervised learning, the training data contains:

- input vectors;
- corresponding target vectors.

The test data only contains input vectors.

The goal is to learn a model that predicts the target vector of unseen input data.

There are two common supervised learning tasks:

- **Regression**: the target is continuous.
- **Classification**: the target is discrete.

---

## 5. Unsupervised Learning

In unsupervised learning, the training data only contains input vectors.

There are no given target labels.

Typical unsupervised learning tasks include:

- **Clustering**: finding groups according to similarity.
- **Density Estimation**: estimating the data distribution.
- **Visualization**: reducing dimension for better observation.

---

## 6. Feature Selection

Feature selection tries to select useful dimensions from the original input features.

### Pros

- It makes the problem easier.
- It speeds up training.
- It reduces input dimension.

### Cons

- Removing features also removes part of the original information.
- If important features are discarded, the final accuracy may drop.

The main trade-off is:

$$
\text{High Accuracy} \quad \text{vs.} \quad \text{Low Input Dimension}
$$

Also, the training set and test set must use the same feature selection process.

---

## 7. Project Overview

This project contains three sub-tasks:

1. **Sub-task 1: Supervised Learning**
2. **Sub-task 2: Unsupervised Learning**
3. **Sub-task 3: Feature Selection**

The general pipeline is:

- For image classification:
    - use image vectors as input;
    - train a classification model;
    - predict labels for test images;
    - submit predictions to the server;
    - the server calculates the prediction accuracy.

- For image retrieval:
    - use a query image vector;
    - search similar images from the image repository;
    - return an image list;
    - the server calculates the retrieval accuracy.

- For feature selection:
    - select a subset of feature dimensions;
    - generate a feature mask;
    - the fixed training and testing process evaluates the selected features.

---

## 8. Sub-task 1: Supervised Learning for Image Classification

### 8.1 Task Description

This task requires us to generate a model that predicts the label of an input image.

However, the raw image is not provided.

Instead, the preprocessed image vector is provided.

The basic form is:

$$
\text{Image Vector} \rightarrow \text{Model} \rightarrow \text{Label}
$$

### 8.2 Baseline: SoftMax Regression

The baseline model is **SoftMax Regression**.

First, a linear model is used:

$$
\mathbf{o} = \mathbf{W}\mathbf{x} + \mathbf{b}
$$

where:

$$
\mathbf{x} \in \mathbb{R}^{d}
$$

$$
\mathbf{W} \in \mathbb{R}^{q \times d}
$$

$$
\mathbf{b} \in \mathbb{R}^{q}
$$

Here:

- $d$ is the input feature dimension.
- $q$ is the number of classes.
- $\mathbf{o}$ is the output score vector.

Then SoftMax is applied:

$$
\hat{\mathbf{y}} = \text{softmax}(\mathbf{o})
$$

For the $i$-th class:

$$
\hat{y}_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)}
$$

The loss function is cross-entropy loss:

$$
L(\mathbf{y}, \hat{\mathbf{y}})
=
-\sum_{j=1}^{q} y_j \log \hat{y}_j
$$

### 8.3 Provided Files

Training set:

- `classification_train_data.pkl`
- `classification_train_label.pkl`

Test set:

- `classification_test_data.pkl`

Python scripts:

- `image_load_demo.ipynb`
    - demonstrates how to load training data and labels.
- `image_classification_demo.ipynb`
    - provides the baseline for this task.

---

## 9. Sub-task 2: Unsupervised Learning for Image Retrieval

### 9.1 Task Description

This task requires us to find similar images from the image repository, given a test image vector.

Again, the raw image is not provided.

The training set is the image repository.

The basic form is:

$$
\text{Test Image Vector} \rightarrow \text{Model} \rightarrow \text{Image ID List}
$$

### 9.2 Baseline: Nearest Neighbor Search

The baseline method is **Nearest Neighbor Search**, or **NNS**.

It uses Euclidean distance as the similarity measure.

Given two image vectors $\mathbf{x}$ and $\mathbf{z}$, their Euclidean distance is:

$$
d(\mathbf{x}, \mathbf{z})
=
\sqrt{
\sum_{i=1}^{d}
(x_i - z_i)^2
}
$$

For each query image, we find the top-$K$ images with the smallest distance in the repository.

That is, we retrieve:

$$
\operatorname{TopK}_{\mathbf{z} \in \mathcal{R}}
\left(
-d(\mathbf{x}, \mathbf{z})
\right)
$$

where $\mathcal{R}$ is the image repository.

### 9.3 Provided Files

Image repository:

- `image_retrieval_repository_data.pkl`

Test set:

- `image_retrieval_test_data.pkl`

Python scripts:

- `image_load_demo.ipynb`
    - demonstrates how to load repository data.
- `image_retrieval_demo.ipynb`
    - provides the baseline for this task.

---

## 10. Sub-task 3: Feature Selection

### 10.1 Task Description

This task requires us to select exactly **30 dimensions** from the original input dimension.

The training process and test process are fixed.

Therefore, the quality of selected features is evaluated by classification accuracy.

The basic form is:

$$
\text{Original Feature Vector}
\rightarrow
\text{Feature Selection}
\rightarrow
\text{Selected 30-D Feature Vector}
\rightarrow
\text{Fixed Classification Process}
$$

### 10.2 Baseline: Random Selection

The baseline method is random feature selection.

The process is:

1. Select a fixed random seed.
2. Randomly select 30 feature dimensions.
3. Generate a feature mask for these 30 selected dimensions.

A feature mask can be written as:

$$
\mathbf{m} \in \{0,1\}^{d}
$$

where:

$$
\sum_{i=1}^{d} m_i = 30
$$

If $m_i = 1$, then the $i$-th feature is selected.

If $m_i = 0$, then the $i$-th feature is discarded.

### 10.3 Provided Files

Validation set:

- `classification_validation_data.pkl`
- `classification_validation_label.pkl`

Python scripts:

- `feature_selection.ipynb`
    - demonstrates how to select features from validation data.
- `image_recognition.ipynb`
    - evaluates the quality of selected features using a fixed image classification process.

---

## 11. Final Summary

Project 2 contains three main tasks:

| Sub-task | Type | Goal | Baseline |
|---|---|---|---|
| Sub-task 1 | Supervised Learning | Predict image labels | SoftMax Regression |
| Sub-task 2 | Unsupervised Learning | Retrieve similar images | Nearest Neighbor Search |
| Sub-task 3 | Feature Selection | Select 30 useful features | Random Selection |

Most trivially, this project is not mainly about processing raw images.

Instead, it focuses on learning from preprocessed image vectors.

Therefore, the key points are:

1. For classification, we need to train a good supervised model.
2. For retrieval, we need to define a good similarity measurement or retrieval method.
3. For feature selection, we need to find 30 dimensions that preserve useful classification information.

The final evaluation is based on prediction accuracy, retrieval accuracy, and the quality of the selected feature subset.