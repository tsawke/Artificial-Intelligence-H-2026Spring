# CS311 Project 2 Report

## General Notes

This project has three tasks: image classification, image retrieval, and feature selection. In all tasks, the first column is an id column. I do not use it as an image feature. The real input has 256 feature values.

All methods use only the given data. I do not submit any extra model file or mask file. For Task 1, `classifier.py` reads the train data and trains the model when `Classifier` is made. For Task 3, `selector.py` reads the validation data and fixed model weights, then makes or checks the mask.

## Subtask 1: Image Classification

### 1. Introduction

This is a supervised learning task. Each image is given as a 256-feature vector. Each train image has a label from 0 to 9. The goal is to train a model and predict labels for test images.

The baseline is Softmax Regression. It is a linear model. I tried several linear models and MLP models. The final choice is an MLP, because it gives higher validation accuracy.

### 2. Method

The final model is a one-hidden-layer MLP:

1. Load `classification_train_data.pkl` and `classification_train_label.pkl`.
2. Remove the id column.
3. Compute mean and std on the train data.
4. Use z-score.
5. Train `MLPClassifier` with 1024 hidden units and ReLU.
6. Use the trained model to predict test labels.

Main settings:

| Item | Value |
|---|---:|
| Hidden units | 1024 |
| Solver | Adam |
| Alpha | 0.0001 |
| Batch size | 256 |
| Max iter | 120 |
| Seed | 321 |

The model is trained once when the class is made. Then `inference()` only does prediction.

### 3. Experiments

The score is accuracy. I used a stratified 80/20 train-validation split to choose the model.

| Model | Main settings | Seed | Validation accuracy |
|---|---:|---:|---:|
| Softmax baseline | lr=0.1, iter=10000 | 123 | 0.5115046018 |
| RidgeClassifier | alpha=1.0 | 123 | 0.4914965986 |
| SGD logistic | alpha=0.0001 | 123 | 0.4872949180 |
| LogisticRegression | C=0.01 | 123 | 0.5080032013 |
| LogisticRegression | C=1.0 | 123 | 0.5055022009 |
| MLP | hidden=(128,) | 123 | 0.5253101240 |
| MLP | hidden=(256,) | 123 | 0.5341136455 |
| MLP | hidden=(512,) | 123 | 0.5495198079 |
| MLP | hidden=(1024,) | 123 | 0.5600240096 |
| MLP | hidden=(1024,) | 321 | 0.5678271309 |
| MLP | hidden=(1024,) | 456 | 0.5579231693 |

The 1024-unit MLP is always above the baseline in my validation runs. The final seed is 321.

### 4. Further Thoughts

If I had more time, I would try several MLP models and average their results. I did not use it here because one MLP is already above the baseline and can run within the time limit.

## Subtask 2: Image Retrieval

### 1. Introduction

This is an unsupervised learning task. Given one query image, the system should return 5 similar images from the repository. The baseline uses nearest neighbor search with Euclidean distance.

The true rule for "similar" is not given. So I keep most of the baseline result and change only the last place.

### 2. Method

The method returns row numbers in the repository, as the TA said. The id column is not used as a feature.

The steps are:

1. Remove the id column from the repository.
2. Compute raw Euclidean distance.
3. Also check shifts, rotations, flips, and blur.
4. Keep the top 4 raw Euclidean neighbors.
5. For the 5th answer, choose the best non-raw-top-5 result from the extra checks.
6. If the raw nearest distance is large, use a wider shift check, because the query may be a moved image.
7. If the query is exactly one repository row, remove that row and then use the normal raw order. I do not skip any raw rank.

This keeps most of the baseline behavior, but the whole top-5 set is not the same as the baseline. It may help when the same image is shifted or changed a little.

### 3. Experiments

The hidden test labels are not given, so I used three local checks.

First, I used a label check on query images not in the repository. I did not use labels in the final method.

| Method | Local check result |
|---|---:|
| Raw Euclidean | 0.101009 |
| Final method | 0.101085 |
| Z-score Euclidean | 0.09840 |
| Raw cosine | 0.10016 |
| Z-score cosine | 0.09960 |
| Repository LogisticRegression label accuracy | 0.11000 |
| Repository MLP(128) label accuracy | 0.09900 |
| Repository MLP(256) label accuracy | 0.10300 |
| Repository MLP(512) label accuracy | 0.08900 |

Second, I used a self-query label check on repository rows, with self-match removed.

| Query set | Raw Euclidean | Final method |
|---|---:|---:|
| First 256 rows | 0.10859 | 0.10859 |
| First 1000 rows | 0.09740 | 0.09760 |
| All 5000 rows | 0.09860 | 0.09868 |

Third, I used transform tests.

| Test | Raw top-5 | Final top-5 |
|---|---:|---:|
| Shift right 1 | 0.086 | 0.984 |
| Shift down 1 | 0.008 | 0.984 |
| Shift right 2 | 0.000 | 0.965 |
| Shift down 2 | 0.101 | 0.801 |
| Shift right 3 | 0.000 | 0.933 |
| Shift left 3 | 0.000 | 0.702 |
| Shift down 3 | 0.007 | 0.479 |
| Shift up 3 | 0.001 | 0.999 |
| 90-degree rotation | 0.000 | 1.000 |
| 3 by 3 blur | 0.133 | 0.980 |
| Raw top-5 overlap on normal queries | 1.000 | 0.9990 |

The output shape is `(a, 5)`. The values are `int64` row numbers. The checked rows have no repeated index.

### 4. Further Thoughts

Task 2 still has the most risk, because the true similar rule is hidden. If I had real feedback, I would change this rule. I did not use a label model because local checks showed it was not safe.

## Subtask 3: Feature Selection

### 1. Introduction

This task is feature selection for a fixed classifier. The model weights are given and cannot be changed. The goal is to choose no more than 30 features from 256 features.

The baseline uses a random 30-feature mask. I need a mask that gives higher accuracy.

### 2. Method

The selector makes the mask when `Selector` is made:

1. Load `classification_validation_data.pkl`.
2. Load `classification_validation_label.pkl`.
3. Load `image_recognition_model_weights.pkl`.
4. Remove the id column.
5. Start from the best 30-feature set found by feature search.
6. Run local swap search on the given data.
7. Return a binary mask with shape `(1, 256)`.

In swap search, I try to remove one selected feature and add one new feature. I only keep the swap if accuracy becomes higher. This keeps the mask valid and avoids using an extra mask file.

### 3. Experiments

The score is validation accuracy after applying the mask.

| Method | Feature number | Validation accuracy |
|---|---:|---:|
| No feature | 0 | 0.099140 |
| Random baseline, seed 42 | 30 | 0.1648659464 |
| All features, not allowed | 256 | 0.5150060024 |
| First 30 features | 30 | 0.4423769508 |
| Greedy forward + swap | 30 | 0.4561824730 |
| Final mask + swap check | 30 | 0.4658863545 |

The selected features are:

`[0, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 24, 26, 28, 29, 33, 46, 47, 54, 59, 62, 67, 71, 205]`

Mask check:

| Item | Result |
|---|---:|
| Shape | `(1, 256)` |
| Feature number | 30 |
| Validation accuracy | 0.4658863545 |
| Gain over random baseline | 0.3010204082 |

This is far above the random baseline.


## Final Check

The final submission has four files: `classifier.py`, `retrieval.py`, `selector.py`, and `report.md`.

Task 1 trains from the given train data. Task 2 returns repository row numbers. Task 3 computes a valid 30-feature mask from the given validation data and fixed model weights.
