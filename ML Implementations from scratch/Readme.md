# Machine Learning: First Principles & Implementations

This repository contains a rigorous exploration of machine learning algorithms, transitioning from raw mathematical theory to practical implementation. The core philosophy is to build complex models—such as Decision Trees and the EM algorithm—entirely from scratch using **NumPy** before benchmarking them against industrial standards like **Scikit-Learn**.

---

## 📂 Repository Structure

> **Note:** All assignment files are located in the following directory:  
> `Machine-Learning-Assignments/`

---

## 1. `ML_Assignment_1.ipynb` (Foundations & Manual Logic)

- **Extensive EDA:** Detailed initial data visualization, correlation heatmaps, and statistical profiling were performed to understand feature relationships before modeling.
- **Linear Regression:** Manual implementation via the Normal Equation (OLS), complemented by Forward and Backward Stepwise Selection.
- **Regularization:** In-depth study of Ridge (L2) and Lasso (L1) regression on the Diabetes dataset.
- **Logistic Regression:** Multi-class classification implemented for the **Iris Dataset**.
- **Decision Tree from Scratch:** High-performance implementation tested on the **MNIST Handwritten Digits Dataset**.
- **Clustering:** K-Means implementation on the Wine Dataset with a focus on Feature Scaling impact.

---

## 2. `ML_Assignment_2.ipynb` (Probabilistic & Deep Learning)

- **EM Algorithm for GMM:** A full from-scratch implementation of the Expectation-Maximization algorithm for Gaussian Mixture Models.
- **Neural Networks (ANN/FFNN):** A multi-layer perceptron architecture (`H=2, K₁=100, K₂=50`) for MNIST digit classification, featuring L2 Regularization experiments.
- **Support Vector Machines (SVM):** Implementation using Linear, Polynomial, and RBF kernels on custom-generated linearly separable data.

---

# 🏆 Featured Implementation: Decision Tree from Scratch

The centerpiece of this project is a robust **Decision Tree Classifier** built from first principles. Designed to handle the high-dimensional MNIST dataset, this implementation utilizes custom stopping conditions that allow it to outperform the standard Scikit-Learn model.

---

## 📊 Performance Benchmark

| Model | Root Feature | Root Impurity | Test Accuracy |
|------|------|------|------|
| **Custom (Gini)** | `pixel_5_2` | 0.9 | **84.97%** |
| **Custom (Entropy)** | `pixel_5_2` | 3.32 | **84.97%** |
| **Sklearn (Gini)** | `pixel_2_5` | 0.9 | 78.10% |
| **Sklearn (Entropy)** | `pixel_5_2` | 3.32 | 83.67% |

---

## 🔍 Why the Custom Model Outperformed Sklearn

- **Thresholding & Early Stopping:**  
  While Scikit-Learn uses `min_impurity_decrease`, this implementation utilizes a custom `info_gain_thres`. This allowed the model to explore deeper splits, capturing finer pixel nuances that default pruning might miss.

- **Information Gain Nuance:**  
  Precise handling of continuous pixel thresholds and Information Gain formulas allowed this model to maintain higher predictive power on the test set without significant overfitting.

---

# 🚀 Technical Highlights

- **Data-First Approach (EDA):**  
  No model was built without first performing comprehensive Exploratory Data Analysis to identify outliers, multi-collinearity, and class imbalances.

- **ANN Optimization:**  
  The Feed-Forward Neural Network achieved an impressive **~96.67% accuracy** on MNIST. The addition of L2 regularization proved vital in narrowing the generalization gap.

- **Mathematical Convergence:**  
  The EM algorithm for GMM successfully converged to true distribution parameters with a minimal MSE of **~0.071**, validating the stability of the manual implementation.

- **The Scaling Verdict:**  
  In K-Means clustering, analysis proved that while unscaled data might show lower inertia, scaled data is statistically superior for identifying distinct classes (**ARI ≈ 0.716**).

---

# 🛠️ Technical Stack

- **Mathematics:** NumPy (Backbone for all "from scratch" logic)
- **Analysis & EDA:** Pandas, Matplotlib, Seaborn
- **Benchmarking:** Scikit-Learn

---

# 🏃 Execution

1. Clone the repository.
2. Ensure all notebooks and the assignment PDF are in the `Machine-Learning-Assignments/` folder.
3. Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

4. Launch Jupyter Notebook:

```bash
jupyter notebook
```

5. Open the notebooks inside the `Machine-Learning-Assignments/` directory and run all cells.

---

# 📌 Key Learning Outcomes

- Building machine learning algorithms completely from scratch using mathematical foundations.
- Understanding optimization, convergence, impurity measures, and regularization deeply rather than relying solely on libraries.
- Benchmarking custom implementations against production-grade frameworks like Scikit-Learn.
- Developing intuition for preprocessing, feature scaling, model selection, and overfitting control.

---

# 📈 Results Summary

| Algorithm | Key Achievement |
|---|---|
| Decision Tree | Outperformed Scikit-Learn on MNIST |
| ANN / FFNN | Achieved ~96.67% MNIST Accuracy |
| EM for GMM | Converged with MSE ~0.071 |
| K-Means | Demonstrated importance of feature scaling |
| SVM | Compared Linear, Polynomial, and RBF kernels |

---

# 📚 Future Improvements

- Add Random Forest and Gradient Boosting implementations from scratch.
- Implement CNNs for improved MNIST performance.
- Add PCA and dimensionality reduction experiments.
- Introduce hyperparameter optimization pipelines.
- Convert notebooks into modular Python packages.

---

# 🤝 Acknowledgment

This repository was built as a deep academic and practical exploration of machine learning fundamentals, emphasizing mathematical intuition, manual implementation, and empirical validation.
