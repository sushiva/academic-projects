# 🏗️ Safety Helmet Detection

> Binary image classification system to detect whether a person is wearing a safety helmet using deep learning.

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen.svg)](https://github.com/sushiva/academic-projects)

---

## 📋 Project Overview

This project implements a computer vision model to automatically detect safety helmet usage in images. The model uses **transfer learning** with ResNet18 architecture and achieves **perfect accuracy** on a balanced dataset of 631 images.

**🎯 Use Case**: Automated safety compliance monitoring in construction sites, factories, and other work environments.

---

## 📊 Dataset

- **Total Images**: 631 (200×200×3 RGB images)
- **Classes**: 2 (With Helmet, Without Helmet)
- **Distribution**:
  - 🔴 Without Helmet: 320 images (50.71%)
  - 🟢 With Helmet: 311 images (49.29%)
- **Split**:
  - 🎓 Train: 70% (441 images)
  - ✅ Validation: 15% (95 images)
  - 🧪 Test: 15% (95 images)

---

## 🧠 Model Architecture

- **Base Model**: ResNet18 (pretrained on ImageNet)
- **Transfer Learning**: Fine-tuning all layers
- **Parameters**: 11.3M total, all trainable
- **Custom Classifier Head**:
  ```
  Dropout(0.5) → Linear(512→256) → ReLU → Dropout(0.25) → Linear(256→2)
  ```

---

## 📁 Project Structure

```
safety-helmet-detection/
├── 📝 config/
│   └── config.yaml              # All hyperparameters and settings
├── 💾 data/
│   ├── raw/                     # Original data (not in git)
│   │   ├── images_proj.npy
│   │   └── Labels_proj.csv
│   └── processed/               # Preprocessed splits
├── 🤖 models/
│   ├── best_model.pth          # Best trained model
│   └── checkpoints/            # Training checkpoints
├── 📓 notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   └── eda_analysis.py         # EDA Python script
├── 📈 outputs/
│   ├── eda/                    # EDA visualizations
│   ├── training/               # Training curves, logs
│   └── evaluation/             # Evaluation metrics, plots
├── 🔧 src/
│   ├── data.py                 # Data loading and preprocessing
│   ├── model.py                # Model architecture
│   ├── train.py                # Training pipeline
│   └── evaluate.py             # Evaluation pipeline
└── 📄 requirements.txt          # Python dependencies
```

---

## 🚀 Quick Start

### 1️⃣ Setup Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install torch torchvision scikit-learn pyyaml tqdm numpy pandas matplotlib seaborn
```

### 2️⃣ Add Data

Place your data files in `data/raw/`:
- `images_proj.npy`
- `Labels_proj.csv`

### 3️⃣ Train Model

```bash
# Default training (30 epochs, early stopping)
python src/train.py
```

**⏱️ Training Time:**
- 💻 CPU: ~1-2 hours
- 🍎 MPS (Apple Silicon): ~20-30 minutes
- 🎮 CUDA GPU: ~10-15 minutes

### 4️⃣ Evaluate Model

```bash
python src/evaluate.py
```

Results saved to `outputs/evaluation/`

---

## ⚙️ Configuration

Edit `config/config.yaml` to customize training parameters:

```yaml
# Key settings
training:
  batch_size: 32
  epochs: 30
  learning_rate: 0.001
  device: "cpu"  # cpu, cuda, or mps

model:
  architecture: "resnet18"
  pretrained: true
```

---

## 🎯 Results

### 🏆 Final Performance (Test Set):
- **Accuracy**: 100.00% ✨
- **Precision**: 100.00% 🎯
- **Recall**: 100.00% 📊
- **F1-Score**: 100.00% ��

### 📊 Per-Class Results:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 🔴 Without Helmet | 1.00 | 1.00 | 1.00 | 48 |
| 🟢 With Helmet | 1.00 | 1.00 | 1.00 | 47 |

### ⏱️ Training Details:
- **Training Time**: 10.79 minutes
- **Epochs**: 12/30 (early stopping)
- **Best Epoch**: 2
- **Validation Accuracy**: 100.00%
- **Device**: CPU

---

## 📸 Visualizations

### Training Curves
![Training History](../../outputs/training/plots/training_history.png)

### Confusion Matrix
![Confusion Matrix](../../outputs/evaluation/confusion_matrix.png)

### ROC Curve (AUC = 1.000)
![ROC Curve](../../outputs/evaluation/roc_curve.png)

---

## 🔬 Data Augmentation

Applied during training:
- 🔄 Random rotation (±15°)
- ↔️ Random horizontal flip
- 🌈 Color jitter (brightness ±20%, contrast ±20%)
- 📐 Normalization (ImageNet statistics)

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| 🧠 Framework | PyTorch 2.0+ |
| 🤖 Model | ResNet18 (torchvision) |
| 📊 Data Processing | NumPy, Pandas |
| 📈 Visualization | Matplotlib, Seaborn |
| 📏 Metrics | scikit-learn |
| 📝 Config | PyYAML |
| ⏳ Progress | tqdm |

---

## 💡 Key Features

✅ **Transfer Learning** from ImageNet
✅ **Early Stopping** (patience: 10 epochs)
✅ **Learning Rate Scheduling** (ReduceLROnPlateau)
✅ **Model Checkpointing** (saves best model)
✅ **Real-time Progress** tracking with tqdm
✅ **Automatic Visualization** generation
✅ **Comprehensive Evaluation** metrics

---

## 🎓 Reproducing Results

```bash
# Ensure reproducibility
# config.yaml has random_seed: 42

# Run training
python src/train.py

# Run evaluation
python src/evaluate.py
```

---

## 🚀 Future Improvements

Potential enhancements:
1. 🎨 Add Grad-CAM visualization for model interpretability
2. 🏗️ Try different architectures (ResNet50, EfficientNet, MobileNet)
3. 📹 Implement real-time detection with webcam
4. 🌐 Deploy as REST API with FastAPI
5. 🎪 Create web demo with Gradio
6. 🔀 Add data augmentation strategies (mixup, cutout)
7. 🤝 Experiment with ensemble methods

---

## 📚 Documentation

For detailed setup instructions across different machines, see:
- 📖 [Setup Guide](../../../SETUP_GUIDE.md) - Complete setup instructions
- ❓ [FAQ](../../../FAQ.md) - Common questions and answers

---

## 📝 License

Academic project - for educational purposes only.

---

## 🙏 Acknowledgments

- **Dataset**: SafeGuard Corp helmet detection dataset
- **Pretrained Models**: torchvision (ImageNet weights)
- **Framework**: PyTorch

---

## 👤 Author

**Sudhir Shivaram**
📧 Email: shivaram.sudhir@gmail.com
🔗 GitHub: [@sushiva](https://github.com/sushiva)

---

<div align="center">

**Made with ❤️ for Academic Excellence**

[⭐ Star this repo](https://github.com/sushiva/academic-projects) | [🐛 Report Bug](https://github.com/sushiva/academic-projects/issues) | [💡 Request Feature](https://github.com/sushiva/academic-projects/issues)

</div>
