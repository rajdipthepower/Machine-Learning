# Machine Learning: First Principles & Implementations

This repository contains a rigorous exploration of machine learning algorithms, transitioning from raw mathematical theory to practical implementation. The core philosophy is to build complex models—such as Decision Trees and the EM algorithm—entirely from scratch using **NumPy** before benchmarking them against industrial standards like **Scikit-Learn**.

## 📂 Repository Structure
**Note:** All assignment files should be deposited in the following directory:  
`Machine-Learning-Assignments/`

### 1. ML_Assignment_1.ipynb (Foundations & Manual Logic)
* **Exploratory Data Analysis (EDA):** Extensive initial data visualization and statistical analysis to understand feature distributions and correlations.
* **Linear Regression:** Manual implementation via the Normal Equation (OLS), complemented by Forward and Backward Stepwise Selection.
* **Regularization:** In-depth study of Ridge (L2) and Lasso (L1) regression.
* **Logistic Regression:** Multi-class classification implemented for the **Iris Dataset**.
* **Decision Tree from Scratch:** High-performance implementation tested on the **MNIST Dataset**.
* **Clustering:** K-Means implementation on the Wine Dataset with a focus on Feature Scaling impact.

### 2. ML_Assignment_2.ipynb (Probabilistic & Deep Learning)
* **EM Algorithm for GMM:** A full from-scratch implementation of the Expectation-Maximization algorithm for Gaussian Mixture Models.
* **Neural Networks (ANN/FFNN):** A multi-layer perceptron architecture ($H=2, K_1=100, K_2=50$) for MNIST digit classification, featuring L2 Regularization experiments.
* **Support Vector Machines (SVM):** Implementation using Linear, Polynomial, and RBF kernels on custom-generated linearly separable data.

---

## 🏆 Featured Implementation: Decision Tree from Scratch
The centerpiece of this project is a robust **Decision Tree Classifier** built from first principles. It was specifically designed to handle the complexity of the MNIST digits dataset, and through careful tuning of stopping conditions, it successfully outperformed the standard Scikit-Learn implementation.

### Performance Benchmark
$$\Large
\begin{array}{|l|c|c|c|}
\hline
\textbf{Model} & \textbf{Root Feature} & \textbf{Root Impurity} & \textbf{Test Accuracy} \\
\hline
\text{Custom (Gini)} & \text{pixel\_5\_2} & 0.9 & \mathbf{84.97\%} \\
\hline
\text{Custom (Entropy)} & \text{pixel\_5\_2} & 3.32 & \mathbf{84.97\%} \\
\hline
\text{Sklearn (Gini)} & \text{pixel\_2\_5} & 0.9 & 78.10\% \\
\hline
\text{Sklearn (Entropy)} & \text{pixel\_5\_2} & 3.32 & 83.67\% \\
\hline
\end{array}$$

### Why the Custom Model Outperformed Sklearn:
* **Thresholding & Early Stopping:** While Sklearn uses `min_impurity_decrease`, this implementation utilizes a custom `info_gain_thres`. This allowed the model to split to a more optimal depth, capturing finer pixel nuances that Sklearn's default pruning might miss.
* **Information Gain Nuance:** Differences in the precision of Information Gain calculations and the specific handling of continuous pixel thresholds allowed this model to achieve higher predictive power on the test set.

---

## 🚀 Technical Highlights

* **Extensive EDA:** Before any modeling, each dataset underwent rigorous Exploratory Data Analysis to identify outliers, multi-collinearity, and class imbalances.
* **ANN Optimization:** The Feed-Forward Neural Network achieved an impressive **~96.67% accuracy** on MNIST. L2 regularization was used to significantly narrow the generalization gap.
* **Mathematical Convergence:** The EM algorithm for GMM successfully converged to true distribution parameters with a minimal MSE of **~0.071**, proving the stability of the manual implementation.
* **The Scaling Verdict:** In K-Means clustering, analysis proved that while unscaled data might show lower inertia, scaled data is statistically superior for identifying distinct classes (ARI ~0.716).

---

## 🛠️ Technical Stack
* **Mathematics:** NumPy (Backbone for all "from scratch" logic)
* **Analysis & EDA:** Pandas, Matplotlib, Seaborn
* **Benchmarking:** Scikit-Learn

## 🏃 Execution
To run these notebooks:
1. Clone the repository.
2. Ensure all notebooks are in the `Machine-Learning-Assignments/` folder.
3. Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
