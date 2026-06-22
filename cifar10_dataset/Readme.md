# CIFAR-10 Classification: Architectural Efficiency & Redundancy Analysis

This directory contains a comparative study of image classification on the CIFAR-10 dataset. It explores the transition from standard Feed-Forward Neural Networks (ANN) to Convolutional Neural Networks (CNN), with a specific focus on how different architectures handle high-dimensional RGB data and spatial correlations.

This study also serves as a conceptual foundation for understanding why modern vision architectures move from dense networks → convolutional priors → transformer-based models for visual understanding.

---

## 📁 Folder Structure
* `ANN_CIFAR10.ipynb`: A study on the limitations of dense architectures, documenting the "Curse of Dimensionality" and parameter explosion.
* `CNN_CIFAR10.ipynb`: An optimized implementation utilizing spatial hierarchies and translation invariance for superior classification accuracy.

---

## 🛠️ Technical Workflow

### 1. The Dimensionality Challenge
* **Feature Analysis:** Comparative study of Grayscale (1,024 features) vs. RGB (3,072 features).
* **Redundancy Mapping:** Analysis of why FFNNs fail to exploit inter-channel correlations, leading to inevitable overfitting on high-dimensional inputs.

🔎 **Additional Insight:**  
This step highlights that flattening images destroys spatial locality, forcing ANN models to learn spatial relationships implicitly rather than structurally.

---

### 2. Architectural Implementations

* **ANN Prototype:** A multi-layer dense network used to benchmark the baseline performance of standard neural layers on complex image data.
* **CNN Prototype:** A hierarchical feature-extraction model using `Conv2D` and `MaxPooling2D` layers to reduce degrees of freedom and focus on local spatial patterns.

🔎 **Additional Insight:**  
CNNs introduce **inductive bias (local connectivity + weight sharing)**, which drastically reduces parameter redundancy compared to fully connected layers.

---

### 3. Key Findings

* **Structural Bias:** Demonstrated that without convolutional filters, parameter count increases without a proportional gain in information density.
* **Invariance:** CNNs successfully captured spatial features that were lost in the flattened input of the ANN model.

🔎 **Additional Insight:**  
Pooling operations contribute to **translation invariance**, allowing the model to recognize objects regardless of position shifts in the image.

---

## 📊 Performance Comparison

| Architecture | Approach | Key Observation |
| :--- | :--- | :--- |
| **ANN** | Flattened Vector | High redundancy; restricted generalization. |
| **CNN** | Spatial Tensors | Efficient feature extraction; translation invariance. |

🔎 **Additional Insight:**  
The performance gap between ANN and CNN grows significantly with image complexity and channel depth, reinforcing CNNs as the preferred baseline for vision tasks.

---

## 📂 Tech Stack

* **Framework:** TensorFlow / Keras
* **Libraries:** NumPy, Matplotlib, Scikit-learn
* **Hardware:** Optimized for T4 GPU acceleration

---

## 🚀 Extended Context (Modern Perspective)

While this repository focuses on ANN vs CNN, it forms a conceptual bridge toward modern architectures:

- CNNs → introduce spatial priors  
- Vision Transformers → replace convolution with self-attention  
- Hybrid models → combine both paradigms  

This progression explains why architectures evolve from **handcrafted inductive bias → data-driven global attention mechanisms**.

---
