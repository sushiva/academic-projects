# 🏗️ Safety Helmet Detection

> Binary image classification system to detect whether a person is wearing a safety helmet using deep learning.

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen.svg)](https://github.com/sushiva/academic-projects)

---

## 📋 Project Overview

This project implements a computer vision system to automatically detect safety helmet usage in images. The project uses a **systematic multi-model comparison approach**, implementing 4 different models with increasing complexity to understand the impact of transfer learning, classifier architecture, and data augmentation.

All models achieve **perfect accuracy** on a balanced dataset of 631 images, demonstrating that transfer learning with minimal trainable parameters can match custom CNNs trained from scratch.

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

## 🧠 Model Architectures

This project implements **4 different models** to systematically compare approaches:

### Model 1: SimpleCNN (Baseline)
- **Trained from scratch** (no pretrained weights)
- **Architecture**: 3 Conv blocks (32→64→128 filters) + Classifier
- **Parameters**: 20.6M trainable
- **Purpose**: Baseline to compare against transfer learning

### Model 2: ResNet18 + Frozen Backbone
- **Base**: ResNet18 pretrained on ImageNet
- **Frozen**: All backbone layers (11.17M params frozen)
- **Trainable**: Only final classifier (1,026 params)
- **Purpose**: Demonstrate power of transfer learning

### Model 3: ResNet18 + Deep Classifier
- **Base**: ResNet18 pretrained on ImageNet (frozen)
- **Classifier**: Deep multi-layer (512→256→128→2)
- **Parameters**: 427K trainable
- **Purpose**: Show impact of classifier depth

### Model 4: ResNet18 + Fine-tuning + Heavy Augmentation
- **Base**: ResNet18 pretrained on ImageNet
- **Training**: All layers trainable (11.6M params)
- **Augmentation**: Heavy (rotation, flips, color, perspective)
- **Purpose**: Full pipeline for maximum robustness

---

## 📁 Project Structure

```
safety-helmet-detection/
├── 📝 config/
│   ├── model1_simple_cnn.yaml      # Model 1 config
│   ├── model2_resnet_base.yaml     # Model 2 config
│   ├── model3_resnet_deep.yaml     # Model 3 config
│   ├── model4_resnet_augmented.yaml # Model 4 config
│   └── config.yaml                 # Legacy single model config
├── 💾 data/
│   ├── raw/                        # Original data (not in git)
│   │   ├── images_proj.npy
│   │   └── Labels_proj.csv
│   └── processed/                  # Preprocessed splits
├── 🤖 models/
│   ├── model1_simple_cnn_best.pth  # Model 1 checkpoint
│   ├── model2_resnet_base_best.pth # Model 2 checkpoint
│   ├── model3_resnet_deep_best.pth # Model 3 checkpoint
│   └── model4_resnet_augmented_best.pth # Model 4 checkpoint
├── 📓 notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   └── eda_analysis.py            # EDA Python script
├── 📈 outputs/
│   ├── eda/                       # EDA visualizations
│   ├── training/                  # Training curves, logs
│   ├── evaluation/                # Evaluation metrics, plots
│   └── comparison/                # Multi-model comparison
│       ├── model_comparison.png
│       ├── confusion_matrices.png
│       └── comparison_report.md
├── 🔧 src/
│   ├── models/                    # Model architecture classes
│   │   ├── simple_cnn.py         # Model 1 architecture
│   │   ├── resnet_base.py        # Model 2 architecture
│   │   ├── resnet_deep.py        # Model 3 architecture
│   │   └── resnet_augmented.py   # Model 4 architecture
│   ├── data.py                    # Data loading (4 augmentation levels)
│   ├── train_all_models.py        # Train all 4 models sequentially
│   ├── compare_models.py          # Generate comparison report
│   ├── train.py                   # Single model training
│   └── evaluate.py                # Evaluation pipeline
├── 📚 docs/
│   └── FAQ.md                     # Transfer learning & ML concepts
└── 📄 requirements.txt             # Python dependencies
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

### 3️⃣ Train All Models (Recommended)

```bash
# Train all 4 models sequentially
python src/train_all_models.py
```

This will train all 4 models one after another and save their checkpoints.

**⏱️ Total Training Time:**
- 💻 CPU: ~50-60 minutes (all 4 models)
- 🍎 MPS (Apple Silicon): ~40-50 minutes
- 🎮 CUDA GPU: ~30-40 minutes

**Alternative: Train Single Model**
```bash
python src/train.py  # Uses config/config.yaml
```

### 4️⃣ Generate Comparison Report

```bash
python src/compare_models.py
```

This creates:
- Model comparison visualizations
- Confusion matrices for all models
- ROC curves
- Comprehensive markdown report

Results saved to `outputs/comparison/`

### 5️⃣ Evaluate Individual Model

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

## 🔬 Multi-Model Comparison

This project implemented **4 different approaches** to systematically compare:
- Training from scratch vs transfer learning
- Frozen backbone vs fine-tuning
- Simple vs deep classifiers
- Different data augmentation strategies

### 📊 Comparative Results

| Model | Approach | Trainable Params | Training Time | Test Accuracy |
|-------|----------|-----------------|---------------|---------------|
| **Model 1** | SimpleCNN (from scratch) | 20.6M | 14.36 min | 100% |
| **Model 2** | ResNet18 + Frozen Backbone | 1,026 | 14.59 min | 100% |
| **Model 3** | ResNet18 + Deep Classifier | 427K | 11.50 min | 100% |
| **Model 4** | ResNet18 + Fine-tuning + Heavy Aug | 11.6M | ~15 min | 100% |

### 📈 Comparison Visualizations

![Model Comparison](outputs/comparison/model_comparison.png)
*Performance, efficiency, and training time comparison across all models*

![Confusion Matrices](outputs/comparison/confusion_matrices.png)
*Side-by-side confusion matrices showing perfect classification*

![Progressive Improvement](outputs/comparison/progressive_improvement.png)
*How each technique contributed to the final model*

**📄 Detailed Report**: See [Comparison Report](outputs/comparison/comparison_report.md) for complete analysis

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
- **Training Time**: 10.79 minutes (single model) | ~50 minutes (all 4 models)
- **Epochs**: 12/30 (early stopping)
- **Best Epoch**: 2
- **Validation Accuracy**: 100.00%
- **Device**: CPU

---

## 🎓 Key Takeaways from Multi-Model Comparison

This assignment implemented a **progressive model comparison approach**, training 4 models with increasing complexity to understand the impact of different techniques:

### 💡 Critical Insights

#### 1. **Transfer Learning is Incredibly Powerful** 🚀
- **Model 2** achieved 100% accuracy with only **1,026 trainable parameters**
- That's **20,000x fewer** parameters than training from scratch!
- Started at **97.89% validation accuracy** on epoch 1 vs Model 1's 80%
- **Lesson**: Always start with pre-trained weights when possible

#### 2. **More Parameters ≠ Better Performance** ⚖️
- Model 1 (20.6M params) = same accuracy as Model 2 (1,026 params)
- **Parameter efficiency** is crucial for deployment
- Smaller models → faster inference, less memory, easier deployment
- **Lesson**: Optimize for efficiency, not just raw parameter count

#### 3. **Classifier Architecture Matters** 🧠
- Model 3's deep classifier (512→256→128→2) with **427K params**
- Faster convergence (11.50 min) compared to Models 1 & 2
- Adding layers to the classifier helps model adapt to specific task
- **Lesson**: Don't just use a single linear layer - experiment with depth

#### 4. **Data Augmentation Adds Robustness** 🔄
- Model 4 used heavy augmentation (rotation, flips, color jitter, perspective)
- Lower initial accuracy (78.95% vs 97.89%) but caught up by epoch 3
- Augmentation makes training harder but improves generalization
- **Lesson**: Augmentation is essential for real-world robustness

#### 5. **Progressive Development > Big Bang Approach** 🎯
- Building models incrementally helped isolate impact of each technique:
  - Model 1→2: Isolated transfer learning benefit
  - Model 2→3: Isolated deep classifier benefit
  - Model 3→4: Isolated fine-tuning + augmentation benefit
- **Lesson**: Systematic experimentation reveals what actually works

#### 6. **Training Time Insights** ⏱️
- Frozen backbone (Model 2): 14.59 min for 30 epochs
- Deep classifier (Model 3): **11.50 min** (fastest!)
- Full fine-tuning (Model 4): ~15 min for 30 epochs
- **Lesson**: Freezing backbone doesn't always save time; optimizer has less to update but each epoch is similar

#### 7. **Early Convergence with Transfer Learning** 📈
- Model 1: Reached 100% at epoch 4
- Model 2: Reached 100% at **epoch 3** (faster!)
- Model 3: Reached 100% at **epoch 2** (even faster!)
- Model 4: Reached 100% at **epoch 3**
- **Lesson**: Transfer learning + good architecture = rapid convergence

### 🏆 Best Practices Learned

✅ **Always use transfer learning** for image classification tasks
✅ **Start simple** (frozen backbone) before fine-tuning everything
✅ **Experiment with classifier depth** - it's cheap and effective
✅ **Use data augmentation** even if your training accuracy is perfect
✅ **Track everything** - log parameters, time, and metrics for comparison
✅ **Progressive experimentation** beats random hyperparameter tuning
✅ **Smaller can be better** - optimize for deployment constraints

### 🎯 Real-World Applications

These insights directly apply to production scenarios:
- **Edge Deployment**: Model 2 (43 MB) vs Model 1 (236 MB) - 5.5x smaller!
- **Inference Speed**: Fewer parameters = faster predictions
- **Mobile/IoT**: Model 2 could run on resource-constrained devices
- **Cost**: Smaller models = lower cloud hosting costs

### 📝 What Would I Do Differently?

1. **Try even smaller backbones** (MobileNet, EfficientNet-B0) for edge deployment
2. **Experiment with learning rates** per layer (discriminative fine-tuning)
3. **Test on out-of-distribution data** to verify robustness
4. **Implement Grad-CAM** to visualize what models actually learned
5. **Ensemble top models** for potentially even better accuracy

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

Potential enhancements for production deployment:

### Already Completed ✅
- ✅ Multi-model comparison (4 different approaches)
- ✅ Progressive evaluation framework
- ✅ Transfer learning implementation
- ✅ Multiple data augmentation strategies

### Next Steps 🎯
1. 🎪 **Web Demo with Gradio/Streamlit** - Interactive UI for portfolio
   - Upload image → Get prediction
   - Compare all 4 models side-by-side
   - Show confidence scores
2. 🎨 **Grad-CAM Visualization** - Model interpretability
   - Highlight which regions models focus on
   - Compare attention maps across models
3. 📹 **Real-time Detection** - Webcam integration
   - Live helmet detection
   - Performance optimization for real-time
4. 🌐 **REST API Deployment** - FastAPI backend
   - Serve best model (Model 2 or 3 for efficiency)
   - Docker containerization
5. 🏗️ **Lightweight Architectures** - Edge deployment
   - MobileNetV3, EfficientNet-B0
   - Quantization for mobile devices
6. 🤝 **Model Ensemble** - Combine predictions
   - Soft voting across top models
   - Uncertainty quantification
7. 📊 **MLOps Pipeline** - Production monitoring
   - MLflow experiment tracking
   - Model registry and versioning
   - A/B testing framework

---

## 📚 Documentation

For detailed information and learning resources:
- 📖 [FAQ - Transfer Learning & ML Concepts](docs/FAQ.md) - What is transfer learning? Fine-tuning? Data augmentation? Industry best practices
- 📊 [Model Comparison Report](outputs/comparison/comparison_report.md) - Detailed multi-model analysis

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
