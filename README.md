# Comparative Analysis of CNN Architectures (ResNet-50 vs. MobileNetV2) for Batik Parang Classification

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/ragerone/CNN_BATIK_PARANG)

![Banner](https://img.freepik.com/premium-photo/batik-parang-indonesian-traditional-clothes_41969-2313.jpg)

## 📌 Project Overview

This project presents a comprehensive comparative analysis of two Convolutional Neural Network (CNN) architectures, **ResNet-50** and **MobileNetV2**, utilized for the fine-grained classification of **Batik Parang** sub-motifs (`Parang Rusak` vs `Non-Parang Rusak`).

This work is based on the methodology outlined in the thesis: *"Analisis Komparatif Arsitektur CNN (ResNet-50 vs. MobileNetV2) untuk Klasifikasi Citra Fine-Grained pada Sub-Motif Batik Rumpun Parang"*.

### Research Objective

The study aims to address the trade-off between **classification accuracy** and **computational efficiency** (inference latency and model size), determining the optimal architecture for both high-end analysis and mobile deployment scenarios.

## 🔬 Methodology

### 1. Dataset & Preprocessing

- **Source**: `parang_1` (Controlled Labeled Data) and `parang_2` (Wild/Unlabeled Data).
- **Split**: 80% Training / 20% Validation (Stratified).
- **Augmentation**: Rotation (20°), Width/Height Shift (0.2), Shear, Zoom, and Horizontal Flip were applied to the training set to improve generalization.
- **Resolution**: All images resized to `224x224` pixels.

### 2. Architectures Implemented

1. **ResNet-50**: chosen for its deep residual learning capabilities, expected to capture intricate high-frequency details of Batik patterns.
2. **MobileNetV2**: chosen for its inverted residual blocks and linear bottlenecks, optimized for mobile efficiency.

### 3. Training Configuration

- **Weights**: Pre-trained on ImageNet (Transfer Learning).
- **Head**: Global Average Pooling -> Dense (1024, ReLU) -> Dropout (0.5) -> Softmax.
- **Optimizer**: Adam (LR=1e-4).
- **Loss**: Categorical Cross-Entropy.

## 📊 Experimental Results

We evaluated the models on the validation set using **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **Efficiency Metrics**.

| Metric | MobileNetV2 | ResNet-50 | Analysis |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 78.3% | **100%** | ResNet-50 achieved perfect classification on the validation set. |
| **F1-Score** | 0.76 | **1.00** | MobileNet struggled with certain complex sub-patterns. |
| **Recall** | 0.78 | **1.00** | ResNet-50 missed zero positive samples. |
| **Model Size** | **24.17 MB** | 114.58 MB | MobileNetV2 is ~5x smaller, ideal for mobile apps. |
| **Latency** | **69.75 ms** | 132.32 ms | MobileNetV2 is ~2x faster in inference. |

### Visual Analysis (Confusion Matrix)

- **ResNet-50**: Showed a perfect diagonal, indicating zero misclassifications.
- **MobileNetV2**: Showed confusion primarily on patterns with heavy visual noise, mistaking some `Parang Rusak` variants for `Non-Parang`.

## 🛠️ How to Run Locally

### 1. Clone Repository

```bash
git clone https://github.com/Rageronee/CNN-BATIK-PARANGCNN_BATIK_PARANG.git
cd CNN_BATIK_PARANG
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Application

We have built a custom Streamlit app to demonstrate "Wild" inference (using data outside the training distribution).

```bash
streamlit run src/app.py
```

## 🧠 Deployment (Hugging Face)

This model is deployed using **Hugging Face Spaces**.

- **Platform**: Spaces (Streamlit SDK)
- **Link**: [Click here to view Demo](https://huggingface.co/spaces/ragerone/CNN_BATIK_PARANG)

## 📁 Repository Structure

```
├── src/
│   ├── app.py          # Main Streamlit Application
│   ├── data_loader.py  # Data preprocessing and modification pipeline
│   ├── evaluate.py     # Evaluation script for generating metrics
│   └── train_model.py  # Training logic for ResNet and MobileNet
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

---
*Created by Muhammad Afnan Risandi, 2026.*
