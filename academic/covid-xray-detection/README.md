
<div align="center">

![COVID-19 X-Ray Detection](assets/covid.png)

# 🦠 COVID-19 X-Ray Detection

> Multi-class classification system to detect COVID-19, Viral Pneumonia, and Normal cases from chest X-ray images using deep learning.

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

</div>

---

## 📋 Project Overview

This project implements a computer vision system to automatically classify chest X-ray images into three categories: COVID-19, Viral Pneumonia, and Normal (healthy). The project uses a **systematic multi-model comparison approach**, implementing 4 different models with increasing complexity to understand the impact of image preprocessing and neural network architecture.

**🎯 Use Case**: Automated COVID-19 screening and differential diagnosis support for healthcare professionals.

---

## 📊 Dataset

- **Total Images**: (see data/raw, update with actual count)
- **Classes**: 3 (COVID-19, Viral Pneumonia, Normal)
- **Image Type**: Chest X-rays (grayscale, some RGB)
- **Distribution**: (update with actual numbers)
- **Split**:
	- 🎓 Train: 70%
	- ✅ Validation: 15%
	- 🧪 Test: 15%

---

## 🧠 Model Architectures

This project implements **4 different models** to systematically compare approaches:

### Model 1: ANN with RGB Images
- **Input**: RGB X-ray images
- **Architecture**: Artificial Neural Network (fully connected)
- **Config**: `config/model1_ann_rgb.yaml`
- **Checkpoint**: `models/model1_ann_rgb_best.h5`
- **Purpose**: Baseline using raw RGB images

### Model 2: ANN with Grayscale Images
- **Input**: Grayscale X-ray images
- **Architecture**: Artificial Neural Network (fully connected)
- **Config**: `config/model2_ann_grayscale.yaml`
- **Checkpoint**: `models/model2_ann_grayscale_best.h5`
- **Purpose**: Effect of grayscale preprocessing

### Model 3: ANN with Gaussian-blurred Images
- **Input**: Gaussian-blurred X-ray images
- **Architecture**: Artificial Neural Network (fully connected)
- **Config**: `config/model3_ann_blur.yaml`
- **Checkpoint**: `models/model3_ann_blur_best.h5`
- **Purpose**: Effect of blur preprocessing

### Model 4: ANN with Laplacian-filtered Images
- **Input**: Laplacian-filtered X-ray images
- **Architecture**: Artificial Neural Network (fully connected)
- **Config**: `config/model4_ann_laplacian.yaml`
- **Checkpoint**: `models/model4_ann_laplacian_best.h5`
- **Purpose**: Effect of edge enhancement

---

## 🏆 Key Takeaways

- Systematic comparison of preprocessing techniques (RGB, grayscale, blur, edge enhancement) for X-ray image classification.
- Simple ANNs can achieve strong performance with proper data preparation.
- Preprocessing can significantly impact model accuracy and generalization.
- Modular codebase enables easy experimentation and extension.

---

## 📊 Model Comparison

| Model | Preprocessing         | Best Accuracy | Notes                       |
|-------|----------------------|---------------|-----------------------------|
| 1     | RGB                  | (fill in)     | Baseline ANN                |
| 2     | Grayscale            | (fill in)     | Simpler input, less noise   |
| 3     | Gaussian Blur        | (fill in)     | Smoother, less detail       |
| 4     | Laplacian (Edges)    | (fill in)     | Edge-focused, more contrast |

> _Update the table with your actual results._

---

## 💡 Best Practices

- Always validate data splits and class balance before training.
- Use modular configuration files for reproducibility.
- Track experiments and results for each model variant.
- Visualize predictions and errors to gain insights.
- Document all preprocessing and training steps.

---

## 🌍 Real-World Applications

- Rapid COVID-19 screening in clinical settings.
- Triage support for radiologists and healthcare workers.
- Research on the impact of preprocessing in medical imaging.
- Educational tool for deep learning and medical AI.

---

---

## 📁 Project Structure

```
covid-xray-detection/
├── assets/                  # Project images and banners
├── config/                  # Model configurations
│   ├── model1_ann_rgb.yaml
│   ├── model2_ann_grayscale.yaml
│   ├── model3_ann_blur.yaml
│   └── model4_ann_laplacian.yaml
├── data/
│   ├── raw/                # Original X-ray data
│   └── processed/          # Preprocessed splits
├── models/                 # Trained model checkpoints
│   ├── model1_ann_rgb_best.h5
│   ├── model2_ann_grayscale_best.h5
│   ├── model3_ann_blur_best.h5
│   └── model4_ann_laplacian_best.h5
├── src/
│   ├── models/            # Model architectures
│   ├── data.py            # Data loading
│   ├── train_all_models.py
│   └── compare_models.py
├── notebooks/              # EDA and experiments
├── docs/                   # Documentation
├── requirements.txt        # Dependencies
└── train_all_models.sh     # Training script
```

---

## 🚀 Quick Start

### 1️⃣ Setup Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Add Data

Place your data files in `data/raw/`

### 3️⃣ Train All Models

```bash
python src/train_all_models.py
```

### 4️⃣ Generate Comparison Report

```bash
python src/compare_models.py
```

---

## 📝 License

Academic project - for educational purposes only.

---

## 👤 Author

**Sudhir Shivaram**  
📧 Email: shivaram.sudhir@gmail.com  
🔗 GitHub: [@sushiva](https://github.com/sushiva)

