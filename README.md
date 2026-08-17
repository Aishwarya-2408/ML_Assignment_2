# Dry Bean Classification using Machine Learning


## 1. Problem Statement

The objective of this project is to implement and compare multiple machine learning classification algorithms on the UCI Dry Bean Dataset.

The implemented models are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest Classifier


The models are evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)


The trained models are deployed through an interactive Streamlit application.


---

# 2. Dataset Description


## Dataset Name

UCI Dry Bean Dataset


## Source

UCI Machine Learning Repository


## Dataset Characteristics

Number of instances: 13611

Number of features: 16

Number of classes: 7


## Target Classes

- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SEKER
- SIRA


## Features

The dataset contains image-based measurements of dry beans:

- Area
- Perimeter
- MajorAxisLength
- MinorAxisLength
- AspectRation
- Eccentricity
- ConvexArea
- EquivDiameter
- Extent
- Solidity
- roundness
- Compactness
- ShapeFactor1
- ShapeFactor2
- ShapeFactor3
- ShapeFactor4


---

# 3. GitHub Repository Link

(Add your GitHub link here)


---

# 4. Models Used and Evaluation


| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression |0.9214|0.9934|0.9222|0.9214|0.9216|0.9050|
| Decision Tree |0.8983|0.9556|0.8988|0.8983|0.8984|0.8770|
| KNN |0.9166|0.9839|0.9173|0.9166|0.9168|0.8992|
| Naive Bayes |0.7639|0.9644|0.7654|0.7639|0.7615|0.7154|
| Random Forest |0.9221|0.9920|0.9222|0.9221|0.9221|0.9058|


---

# 5. Model Performance Observations


## Logistic Regression

Logistic Regression achieved strong performance with an AUC score of 0.9934.

The dataset has good class separability, allowing a linear model to perform effectively.


## Decision Tree

Decision Tree achieved reasonable performance but was lower compared with ensemble approaches.

A single decision tree can suffer from higher variance.


## KNN

KNN performed well after feature scaling.

However, prediction becomes computationally expensive for larger datasets.


## Naive Bayes

Naive Bayes obtained the lowest performance.

This is because the algorithm assumes feature independence, which may not completely hold for image-derived measurements.


## Random Forest

Random Forest achieved the best overall performance.

The ensemble approach improves generalization by combining multiple decision trees.


## Overall Winner

Random Forest was selected as the overall winner because it achieved the highest average performance across evaluation metrics.


---

# 6. Streamlit Application


Live Application Link:

(Add Streamlit URL here)


Application Features:

- Upload test CSV file
- Select trained ML model
- Display evaluation metrics
- Display classification report
- Display confusion matrix

