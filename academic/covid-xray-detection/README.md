# 🦠 COVID-19 X-Ray Detection

> Multi-class classification system to detect COVID-19, Viral Pneumonia, and Normal cases from chest X-ray images using deep learning.

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

---

## 📋 Project Overview

This project implements a computer vision system to automatically classify chest X-ray images into three categories: COVID-19, Viral Pneumonia, and Normal (healthy). The project uses a **systematic multi-model comparison approach**, implementing 4 different models with increasing complexity.

**🎯 Use Case**: Automated COVID-19 screening and differential diagnosis support for healthcare professionals.

---

## 📊 Dataset

- **Total Images**: TBD
- **Classes**: 3 (COVID-19, Viral Pneumonia, Normal)
- **Image Type**: Chest X-rays (grayscale)
- **Distribution**: TBD
- **Split**: 70/15/15 (train/val/test)

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

## 📁 Project Structure

```
covid-xray-detection/
├── config/                  # Model configurations
├── data/
│   ├── raw/                # Original X-ray data
│   └── processed/          # Preprocessed splits
├── models/                 # Trained model checkpoints
├── src/
│   ├── models/            # Model architectures
│   ├── data.py            # Data loading
│   ├── train_all_models.py
│   └── compare_models.py
├── outputs/
│   ├── training/
│   ├── evaluation/
│   └── comparison/
└── docs/                   # Documentation
```

---

## 📝 License

Academic project - for educational purposes only.

---

## 👤 Author

**Sudhir Shivaram**
📧 Email: shivaram.sudhir@gmail.com
🔗 GitHub: [@sushiva](https://github.com/sushiva)
