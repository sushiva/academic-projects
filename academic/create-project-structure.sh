#!/bin/bash

# COVID-19 X-Ray Detection Project Structure Setup Script
# This script creates the complete directory structure for the project

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  COVID-19 X-Ray Detection Project${NC}"
echo -e "${BLUE}  Structure Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Base project directory
PROJECT_DIR="/home/bhargav/academic-projects/academic/covid-xray-detection"

# Create base directory
echo -e "${GREEN}Creating base project directory...${NC}"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR" || exit

# Create main directory structure
echo -e "${GREEN}Creating directory structure...${NC}"

# Config directory
mkdir -p config

# Data directories
mkdir -p data/raw
mkdir -p data/processed

# Models directory
mkdir -p models

# Notebooks directory
mkdir -p notebooks

# Outputs directories
mkdir -p outputs/eda
mkdir -p outputs/training/logs
mkdir -p outputs/training/plots
mkdir -p outputs/evaluation
mkdir -p outputs/comparison

# Source code directories
mkdir -p src/models

# Documentation directory
mkdir -p docs

# Tests directory
mkdir -p tests

# Create __init__.py files for Python packages
echo -e "${GREEN}Creating Python package files...${NC}"
touch src/__init__.py
touch src/models/__init__.py

# Create placeholder files
echo -e "${GREEN}Creating placeholder configuration files...${NC}"

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Data files (too large for git)
data/raw/*.npy
data/raw/*.csv
data/processed/

# Model checkpoints
models/*.pth
models/*.pt

# Outputs
outputs/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Environment
.env
EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core ML/DL
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Data processing
pandas>=2.0.0
Pillow>=10.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Configuration
pyyaml>=6.0

# Progress bars
tqdm>=4.65.0

# Jupyter (optional)
jupyter>=1.0.0
ipykernel>=6.25.0

# Optional: Web demo
# gradio>=3.50.0
# streamlit>=1.28.0
EOF

# Create README.md template
cat > README.md << 'EOF'
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
EOF

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Project structure created successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Project location: $PROJECT_DIR"
echo ""
echo "Next steps:"
echo "1. Copy your COVID X-ray dataset to: $PROJECT_DIR/data/raw/"
echo "2. Activate virtual environment: cd $PROJECT_DIR && python3 -m venv .venv && source .venv/bin/activate"
echo "3. Install dependencies: pip install -r requirements.txt"
echo ""
echo "Directory structure:"
tree -L 2 "$PROJECT_DIR" 2>/dev/null || ls -R "$PROJECT_DIR"
