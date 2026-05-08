# Project 2 Evaluation Details Summary

## 1. Environment Preparation

The recommended Python version is:

$$
\text{Python} = 3.10
$$

A separate environment is recommended, so that Project 2 is isolated from Project 1.

```bash
conda create -n python310 python=3.10
conda activate python310
conda install jupyter
pip install -r requirements.txt
````

Required packages include:

```txt
matplotlib==3.7.1
numpy==1.26.1
tqdm==4.64.1
scikit-learn==1.5.2
```

---

## 2. Subtask 1: Image Classification

### 2.1 Requirement

In this subtask, we need to independently train a model using the provided training set, and then use the trained model to predict labels for the test set.

The provided demo is:

```txt
image_classification_demo.ipynb
```

It shows the baseline image classification procedure.

We are allowed to customize the demo and use our own method.

### 2.2 Provided Files

```txt
.
├── README.md
├── requirements.txt
├── classification_train_data.pkl
├── classification_train_label.pkl
├── image_classification_demo.ipynb
├── SoftmaxRegression.py
├── classifier.py
└── util.py
```

Explanation:

* `classification_train_data.pkl`: image features of the training set.
* `classification_train_label.pkl`: labels of the training set.
* `image_classification_demo.ipynb`: baseline program for Subtask 1.
* `SoftmaxRegression.py`: implementation of the Softmax Regression baseline.
* `classifier.py`: the file we need to implement and submit.
* `util.py`: common utility functions.

### 2.3 Grading

To be graded, we need to implement:

```txt
classifier.py
```

The grading rule is binary:

$$
\text{Score} =
\begin{cases}
\text{Full points}, & \text{if test accuracy exceeds the baseline} \
0, & \text{otherwise}
\end{cases}
$$

Therefore, the main goal of this subtask is to make the classifier better than the given baseline.

---

## 3. Subtask 2: Image Retrieval

### 3.1 Requirement

In this subtask, we need to train or design a model based on the provided image repository.

Given each query image from the test set, the model should retrieve 5 similar images from the image repository.

For each query image, the output should be a list of image IDs.

The provided demo is:

```txt
image_retrieval_demo.ipynb
```

It shows the baseline image retrieval procedure.

We are also allowed to customize the demo and use our own method.

### 3.2 Provided Files

```txt
.
├── README.md
├── image_retrieval_repository_data.pkl
├── image_retrieval_demo.ipynb
├── NNS.py
├── retrieval.py
└── util.py
```

Explanation:

* `image_retrieval_repository_data.pkl`: image features in the repository.
* `image_retrieval_demo.ipynb`: baseline program for Subtask 2.
* `NNS.py`: implementation of the Nearest Neighbor Search baseline.
* `retrieval.py`: the file we need to implement and submit.
* `util.py`: common utility functions.

### 3.3 Accuracy Formula

For each test image, suppose we submit:

$$
n = 5
$$

similar images.

If $m$ submitted images are considered similar by the evaluation process, then the accuracy for this query is:

$$
\text{Accuracy}
===============

\frac{m}{n}
\times 100%
$$

Since $n = 5$, this becomes:

$$
\text{Accuracy}
===============

\frac{m}{5}
\times 100%
$$

The final test accuracy is the average accuracy over the whole test set:

$$
\text{Final Accuracy}
=====================

\frac{1}{T}
\sum_{i=1}^{T}
\text{Accuracy}_i
$$

where $T$ is the number of test queries.

### 3.4 Grading

To be graded, we need to implement:

```txt
retrieval.py
```

The grading rule is also binary:

$$
\text{Score} =
\begin{cases}
\text{Full points}, & \text{if test accuracy exceeds the baseline} \
0, & \text{otherwise}
\end{cases}
$$

Therefore, the key point is to retrieve more correct similar images than the baseline.

---

## 4. Subtask 3: Feature Selection

### 4.1 Requirement

In this subtask, we need to write an algorithm to select no more than 30 features from the image features of the classification validation set.

That is, if the original feature dimension is $n$, we need to construct a binary feature mask:

$$
\mathbf{mask} \in {0,1}^{n}
$$

The number of selected features should satisfy:

$$
\sum_{i=1}^{n} \mathbf{mask}_i \le 30
$$

For each input image vector:

$$
\mathbf{x} \in \mathbb{R}^{n}
$$

the same binary mask is applied to produce the actual input:

$$
\check{\mathbf{x}}
==================

\mathbf{x} \odot \mathbf{mask}
$$

Here, $\odot$ means element-wise multiplication.

If a mask value is $0$, the corresponding feature dimension is ignored.

### 4.2 Example

Suppose:

$$
n = 3
$$

and the input vector is:

$$
\mathbf{x}
==========

\langle 112, 345, 321 \rangle
$$

The mask is:

$$
\mathbf{mask}
=============

\langle 1, 0, 1 \rangle
$$

Then the masked vector is:

$$
\check{\mathbf{x}}
==================

\langle 112, 345, 321 \rangle
\odot
\langle 1, 0, 1 \rangle
=======================

\langle 112, 0, 321 \rangle
$$

The second feature, whose original value is $345$, is masked out and will not affect the trained model.

### 4.3 Provided Files

```txt
.
├── README.md
├── classification_validation_data.pkl
├── classification_validation_label.pkl
├── image_recognition_model_weights.pkl
├── feature_selection.ipynb
├── selector.py
└── image_recognition.ipynb
```

Explanation:

* `classification_validation_data.pkl`: image features of the validation set.
* `classification_validation_label.pkl`: labels of the validation set.
* `image_recognition_model_weights.pkl`: fixed weights of a baseline image classification model.
* `feature_selection.ipynb`: demonstrates how to select features.
* `selector.py`: the file we need to implement and submit.
* `image_recognition.ipynb`: evaluates the selected features on the validation set.

### 4.4 Evaluation Procedure

The offline validation procedure is:

1. Run `feature_selection.ipynb`.
2. Generate the feature mask file:

```txt
mask_code.pkl
```

3. Run `image_recognition.ipynb`.
4. Load the fixed classification model from:

```txt
image_recognition_model_weights.pkl
```

5. Apply the selected mask to the validation data.
6. Predict labels.
7. Compare predictions with:

```txt
classification_validation_label.pkl
```

8. Compute the classification accuracy.

During official testing, the procedure is similar, but the validation data is replaced by the hidden test data.

### 4.5 Grading

To be graded, we need to implement:

```txt
selector.py
```

The grading rule is:

$$
\text{Score} =
\begin{cases}
\text{Full points}, & \text{if test accuracy exceeds the baseline} \
0, & \text{otherwise}
\end{cases}
$$

Therefore, this subtask is mainly about finding a better feature subset than random selection.

---

## 5. Final Summary

Project 2 contains three evaluated subtasks:

| Subtask   | Task                 | Submit File     | Goal                                     | Full-score Condition      |
| --------- | -------------------- | --------------- | ---------------------------------------- | ------------------------- |
| Subtask 1 | Image Classification | `classifier.py` | Predict test image labels                | Accuracy exceeds baseline |
| Subtask 2 | Image Retrieval      | `retrieval.py`  | Retrieve 5 similar images for each query | Accuracy exceeds baseline |
| Subtask 3 | Feature Selection    | `selector.py`   | Select at most 30 useful features        | Accuracy exceeds baseline |

Most trivially, all three subtasks are baseline-beating tasks.

The grading is not based on how complex the method is. Instead, the key requirement is:

$$
\text{Our Method Accuracy} > \text{Baseline Accuracy}
$$

So, for full score, we should focus on practical improvement over the provided baseline programs.

In short:

1. For classification, improve the model in `classifier.py`.
2. For retrieval, improve the retrieval strategy in `retrieval.py`.
3. For feature selection, improve the selected mask in `selector.py`.

