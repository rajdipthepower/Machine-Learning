# Bank Customer Churn Prediction using Neural Networks

## Overview

This project aims to predict which customers are likely to churn (leave the bank) using historical customer data. Customer churn is a critical metric for banks, and accurate prediction allows for proactive retention strategies. The core of this solution utilizes neural networks (specifically, a Multilayer Perceptron), with a strong emphasis on clean preprocessing and handling data imbalance correctly without introducing data leakage.

## Key Technologies

* **Python**
* **Pandas** and **NumPy** for data manipulation and analysis
* **Scikit-learn** for pipeline scaling, feature encoding, and evaluation metrics
* **Imbalanced-learn (SMOTE)** for synthetic minority oversampling
* **TensorFlow** / **Keras** for building and training the neural networks

## Methodology & Pipeline Protection

Imbalanced datasets can lead to models that perform well on the majority class but poorly on the minority (churning) class. This project employs a rigorous, leakage-free workflow to handle this issue:

1. **One-Hot Encoding:** Categorical features (`Geography`, `Gender`) are transformed using dummy variables, dropping the first category to avoid the dummy variable trap.
2. **Train-Test-Validation Split:** The dataset is split into training, validation, and test sets using stratification to preserve the original class distribution across all splits.
3. **Leakage-Free Feature Scaling:** Continuous features (`CreditScore`, `Age`, `Tenure`, `Balance`, `EstimatedSalary`) are normalized using scikit-learn's `MinMaxScaler`. The scaler is fitted strictly on the training set, and the learned parameters are then applied to transform the validation and test sets.
4. **Addressing Class Imbalance:** Two distinct training approaches are evaluated:
* **Class Weights:** Computing balanced class weights directly from the training data and applying them during the loss optimization step to penalize minority class misclassifications.
* **SMOTE (Synthetic Minority Over-sampling Technique):** Synthetically balancing the training dataset prior to model training. SMOTE is applied exclusively to the training set to prevent validation/test leakage.



## Model: Neural Network Architecture

A feed-forward neural network (Multilayer Perceptron) is implemented using Keras. To ensure reproducible benchmarks, the architecture is completely re-instantiated and re-compiled between experiments to guarantee a fresh initialization of weights:

* **Input Layer:** Dynamically shapes to the feature space , consisting of 32 units with `ReLU` activation.
* **Hidden Layers:** Two dense layers consisting of 100 units each with `ReLU` activations to map complex, non-linear patterns.
* **Output Layer:** A single neuron with a `Sigmoid` activation function to output the probability of churn.
* **Optimization & Callbacks:** Compiled with the Adam optimizer and Binary Cross-Entropy loss. Training utilizes `EarlyStopping` monitoring validation accuracy with a patience of 5 epochs and weight restoration enabled.

## Results & Performance Metrics

Because simple classification accuracy can be misleading on imbalanced data, the model is evaluated using comprehensive classification reports highlighting Precision, Recall, and F1-Scores.

### Experiment 1: Cost-Sensitive Learning (Class Weights)

Adjusting the loss function penalty to accommodate the $0.256$ class imbalance ratio yielded a robust balance between capturing churning customers and preserving majority class accuracy:

```text
              precision    recall  f1-score   support

           0       0.91      0.84      0.88      1593
           1       0.53      0.69      0.60       407

    accuracy                           0.81      2000
   macro avg       0.72      0.77      0.74      2000
weighted avg       0.84      0.81      0.82      2000

```

### Experiment 2: Oversampling (SMOTE)

Training a completely fresh network on the SMOTE-balanced training distribution resulted in a highly precise model for identifying churn risks, raising the minority class precision significantly to $0.69$:

```text
              precision    recall  f1-score   support

           0       0.88      0.94      0.91      1593
           1       0.69      0.51      0.59       407

    accuracy                           0.85      2000
   macro avg       0.79      0.73      0.75      2000
weighted avg       0.84      0.85      0.85      2000

```
