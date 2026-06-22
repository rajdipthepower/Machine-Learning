# 🚀 Vision Transformer (ViT) Fine-Tuning with LoRA on CIFAR-10

This project demonstrates **high-performance image classification** on the CIFAR-10 dataset using a pretrained Vision Transformer (ViT) model from Hugging Face, fine-tuned using **Low-Rank Adaptation (LoRA)** for efficient training.

The entire pipeline is implemented in a **Google Colab Jupyter Notebook**, covering data preprocessing, augmentation, training, evaluation, and visualization.

---

## 📌 Project Overview

We fine-tune the **ViT-Base-Patch16-224** model on the CIFAR-10 dataset using a hybrid approach:

- Frozen Vision Transformer backbone  
- Trainable classification head  
- LoRA-based parameter-efficient fine-tuning  
- Custom PyTorch data pipeline  

---

## 📊 Dataset: CIFAR-10

The dataset used is the standard **CIFAR-10 benchmark dataset**, which contains:

- 60,000 RGB images (32×32)
- 10 balanced classes:
  - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 6,000 images per class

---

## 🧠 Model Architecture

### 🔹 Base Model

We use a pretrained Vision Transformer:

```python
ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

From Hugging Face Transformers

🔹 Key Components
Patch-based image embedding (16×16 patches)
Transformer encoder layers
Classification head replaced for CIFAR-10 (10 classes)
Frozen backbone during initial training
🔹 LoRA Fine-Tuning

We apply Low-Rank Adaptation (LoRA):

Rank: r = 8
Alpha: 16
Target modules:
q_proj
k_proj
v_proj
o_proj
Dropout: 0.1

This reduces trainable parameters significantly while preserving performance.

📁 Data Preprocessing Pipeline
🔹 Image Loading
Supports .jpg, .png, .jfif
Recursive directory traversal using rglob
🔹 Sample Structure

Each data sample is structured as:

{
  "path": "image_path",
  "label": "class_name",
  "id": "file_stem"
}
🔹 Label Encoding
Labels are sorted alphabetically
Converted into integer IDs

Mappings:

label_to_id
id_to_label
🔀 Dataset Splitting

We use stratified splitting via Scikit-learn:

Training set: 85%
Validation set: 15% of training
Test set: separate holdout set

Ensures balanced class distribution across splits.

🎨 Data Augmentation

Implemented using torchvision:

Training Augmentations
Random Horizontal Flip (p=0.5)
Random Rotation (±45°)
Color Jitter:
brightness
contrast
saturation
hue
Random Grayscale (p=0.1)
Validation / Test Preprocessing
Resize → 256×256
Center Crop → 224×224
🧩 Custom Dataset Class

Built using PyTorch:

Key features:

Robust image loading with error handling
Automatic retry on corrupted images
On-the-fly transformation support
Label encoding integration
⚙️ DataLoader Configuration
batch_size = 32
num_workers = 2
pin_memory = True
prefetch_factor = 2

Custom collate_fn:

Uses ViT image processor
Converts batch → tensor format [B, 3, 224, 224]
🏋️ Training Pipeline
🔹 Optimizer
AdamW
Learning rate: 3e-4
Weight decay: 1e-4
🔹 Scheduler
Cosine Annealing LR
Smooth decay over epochs
🔹 Loss Function

Handled internally by Hugging Face ViT:

CrossEntropyLoss
🧪 Training Strategy
Phase 1: Feature Extraction
Freeze ViT backbone
Train classifier head only
Phase 2: LoRA Fine-Tuning
Inject LoRA into attention layers
Train only low-rank adapters
🧠 Early Stopping
Monitors validation loss
Saves best model checkpoint:
best_model_classifier_vit.pth
Patience-based stopping strategy
📈 Evaluation Metrics

We evaluate using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
🧾 Test Results
Metric	Score
Accuracy	~98%
Macro F1	~0.98
Loss	~0.053
📊 Observations
Strong performance across all classes
Minor confusion between visually similar classes (cat vs dog)
Transformer backbone generalizes extremely well
📉 Visualizations

The notebook includes:

Training vs Validation Loss curves
Training vs Validation Accuracy curves
Confusion Matrix heatmap (Seaborn)
🧪 Inference Pipeline

Steps:

Load trained model checkpoint
Preprocess image using ViT processor
Forward pass through model
Apply argmax on logits
🛠️ Tech Stack
PyTorch – Deep learning framework
Hugging Face Transformers – Vision Transformer + processor
torchvision – Image augmentations & transforms
Scikit-learn – Data splitting & evaluation
NumPy – Numerical computing
Matplotlib – Plotting
Seaborn – Visualization
tqdm – Progress bars
📌 Notebook Environment

This project is fully implemented in:

Google Colab Notebook
PyTorch GPU runtime
Jupyter-based experimentation workflow
🚀 Key Learnings
Vision Transformers outperform CNNs when properly fine-tuned
LoRA drastically reduces training cost
Proper preprocessing is critical for ViT performance
Transfer learning is extremely effective on CIFAR-10
🔮 Future Improvements
Try ViT-Large / DeiT architectures
Add MixUp / CutMix augmentation
Deploy using FastAPI or Flask
Export model to ONNX for faster inference
Add Grad-CAM / attention visualization
📎 Author Notes

This project was built as a hands-on deep learning experiment to understand:

Transformer-based vision models
Efficient fine-tuning techniques
End-to-end training pipelines in PyTorch
