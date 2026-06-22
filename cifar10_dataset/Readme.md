# CIFAR-10 Classification: Architectural Efficiency & Redundancy Analysis

This directory contains a comparative study of image classification on the CIFAR-10 dataset. It explores the transition from standard Feed-Forward Neural Networks (ANN) to Convolutional Neural Networks (CNN), with a specific focus on how different architectures handle high-dimensional RGB data and spatial correlations.

This repository has now been extended to include **Transformer-based vision modeling (ViT)**, enabling a comparison between classical deep learning approaches (ANN/CNN) and modern attention-based architectures.

---

## 📁 Folder Structure
* `ANN_CIFAR10.ipynb`: A study on the limitations of dense architectures, documenting the "Curse of Dimensionality" and parameter explosion.
* `CNN_CIFAR10.ipynb`: An optimized implementation utilizing spatial hierarchies and translation invariance for superior classification accuracy.
* `ViT_CIFAR10.ipynb`: A Vision Transformer-based implementation using pretrained **ViT-Base-Patch16-224** with **LoRA fine-tuning** for efficient adaptation on CIFAR-10.

---

## 🛠️ Technical Workflow

### 1. The Dimensionality Challenge
* **Feature Analysis:** Comparative study of Grayscale (1,024 features) vs. RGB (3,072 features).
* **Redundancy Mapping:** Analysis of why FFNNs fail to exploit inter-channel correlations, leading to inevitable overfitting on high-dimensional inputs.

---

### 2. Architectural Implementations
* **ANN Prototype:** A multi-layer dense network used to benchmark the baseline performance of standard neural layers on complex image data.
* **CNN Prototype:** A hierarchical feature-extraction model using `Conv2D` and `MaxPooling2D` layers to reduce degrees of freedom and focus on local spatial patterns.
* **ViT Prototype:** A transformer-based vision model leveraging patch embeddings and self-attention mechanisms, fine-tuned using **LoRA (Low-Rank Adaptation)** for parameter-efficient training.

---

### 3. Key Findings
* **Structural Bias:** Demonstrated that without convolutional filters, parameter count increases without a proportional gain in information density.
* **Invariance:** CNNs successfully captured spatial features that were lost in the flattened input of the ANN model.
* **Global Context Learning:** Vision Transformers improve performance by modeling **long-range dependencies** between image patches using self-attention instead of convolution.

---

## 📊 Performance Comparison

| Architecture | Approach | Key Observation |
| :--- | :--- | :--- |
| **ANN** | Flattened Vector | High redundancy; restricted generalization. |
| **CNN** | Spatial Tensors | Efficient feature extraction; translation invariance. |
| **ViT** | Patch + Attention | Strong global feature modeling; scalable with pretraining. |

---

## 📂 Tech Stack
* **Framework:** TensorFlow / Keras (ANN & CNN), PyTorch (ViT)
* **Libraries:** NumPy, Matplotlib, Scikit-learn, torchvision
* **Transformer Stack:** Hugging Face Transformers + PEFT (LoRA)
* **Hardware:** Optimized for T4 GPU acceleration
