# Complete Setup Guide - Academic Projects

This guide covers everything you need to work with your academic projects across different machines.

## Table of Contents
1. [Initial GitHub Setup](#initial-github-setup)
2. [Cloning to Your Mac](#cloning-to-your-mac)
3. [Environment Setup](#environment-setup)
4. [Running the Safety Helmet Detection Project](#running-the-safety-helmet-detection-project)
5. [Using GPU Acceleration (Mac M1/M2/M3)](#using-gpu-acceleration-mac-m1m2m3)
6. [Adding New Projects](#adding-new-projects)
7. [Common Commands](#common-commands)
8. [Troubleshooting](#troubleshooting)

---

## Initial GitHub Setup

### 1. Create Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `academic-projects`
3. Description: "Collection of academic ML/AI projects"
4. Choose Public or Private
5. **Don't** initialize with README (we already have one)
6. Click "Create repository"

### 2. Push from Linux Machine

```bash
cd /home/bhargav/academic-projects

# Set default branch to main
git branch -M main

# Add your GitHub repo (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/academic-projects.git

# Push to GitHub
git push -u origin main
```

---

## Cloning to Your Mac

### 1. Clone the Repository

```bash
# Navigate to where you want the projects
cd ~/Documents  # or ~/Desktop, or wherever you prefer

# Clone the repository
git clone https://github.com/YOUR_USERNAME/academic-projects.git

# Navigate into the project
cd academic-projects/academic/safety-helmet-detection
```

### 2. Copy Your Data Files

Since data files are excluded from git (too large), you need to add them manually:

**Option A: Copy from External Drive**
```bash
# Assuming data is on external drive or USB
cp /Volumes/YourDrive/data/raw/*.npy data/raw/
cp /Volumes/YourDrive/data/raw/*.csv data/raw/
```

**Option B: Download from Cloud**
- Upload data to Google Drive / Dropbox from Linux
- Download to Mac
- Copy to `data/raw/` folder

---

## Environment Setup

### On Mac (M1/M2/M3)

```bash
# Navigate to project
cd ~/Documents/academic-projects/academic/safety-helmet-detection

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install torch torchvision scikit-learn pyyaml tqdm numpy pandas matplotlib seaborn

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"
```

### On Linux

```bash
# Navigate to project
cd ~/academic-projects/academic/safety-helmet-detection

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Safety Helmet Detection Project

### 1. Check Your Setup

```bash
# Make sure you're in project directory
cd academic-projects/academic/safety-helmet-detection

# Activate environment
source .venv/bin/activate

# Verify data exists
ls -lh data/raw/

# Should see:
# - images_proj.npy
# - Labels_proj.csv
```

### 2. Run Training

```bash
# For CPU training
python src/train.py

# For background training (can close terminal)
nohup python src/train.py > training_output.log 2>&1 &

# Check training progress
tail -f training_output.log

# Or check last 50 lines
tail -50 training_output.log
```

### 3. Run Evaluation

After training completes:

```bash
# Run evaluation
python src/evaluate.py

# Results will be saved to outputs/evaluation/
ls -lh outputs/evaluation/
```

### 4. View Results

```bash
# Open evaluation results
open outputs/evaluation/confusion_matrix.png
open outputs/evaluation/roc_curve.png
open outputs/evaluation/per_class_metrics.png

# View metrics
cat outputs/evaluation/metrics.txt
cat outputs/evaluation/classification_report.txt
```

---

## Using GPU Acceleration (Mac M1/M2/M3)

If you have Apple Silicon Mac, you can use MPS for GPU acceleration!

### 1. Check MPS Availability

```bash
python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"
```

If it shows `True`, you can use GPU acceleration.

### 2. Enable MPS in Config

Edit `config/config.yaml`, line 54:

```yaml
# Change from:
device: "cpu"

# To:
device: "mps"
```

### 3. Train with GPU

```bash
python src/train.py
```

Training will be **3-5x faster** on MPS compared to CPU!

**Expected Times:**
- CPU: 1-2 hours for 30 epochs
- MPS (Apple Silicon): 20-30 minutes for 30 epochs
- CUDA GPU: 10-15 minutes for 30 epochs

---

## Adding New Projects

When you complete a new academic project:

### 1. Add to Repository

```bash
# Navigate to academic folder
cd ~/Documents/academic-projects/academic

# Copy your new project
cp -r ~/path/to/new-project ./

# Or create new project directly
mkdir new-project-name
cd new-project-name
# ... work on project ...

# Commit and push
cd ~/Documents/academic-projects
git add academic/new-project-name
git commit -m "Add new project: [project name]"
git push
```

### 2. Update Main README

Edit `README.md` to add your new project to the list:

```markdown
### 2. New Project Name
Brief description of the project.

- **Tech Stack**: Technologies used
- **Dataset**: Dataset info
- **Results**: Key results
- **Location**: `academic/new-project-name/`
```

---

## Common Commands

### Git Commands

```bash
# Check status
git status

# Add all changes
git add .

# Commit changes
git commit -m "Your commit message"

# Push to GitHub
git push

# Pull latest changes
git pull

# View commit history
git log --oneline
```

### Python Environment

```bash
# Activate virtual environment
source .venv/bin/activate

# Deactivate virtual environment
deactivate

# Install package
pip install package-name

# Install from requirements
pip install -r requirements.txt

# Save current packages
pip freeze > requirements.txt

# List installed packages
pip list
```

### Project Management

```bash
# Check GPU/Device
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available()}')"

# Test data loading
python src/data.py

# Test model
python src/model.py

# Monitor training
tail -f training_output.log

# Kill training process
pkill -f "python src/train.py"
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Install PyTorch
pip install torch torchvision
```

### Issue: "FileNotFoundError: data/raw/images_proj.npy"

**Solution:**
- Data files are not in git (too large)
- Copy data files manually to `data/raw/` folder
- Check with: `ls -lh data/raw/`

### Issue: Training is slow on Mac

**Solution:**
- Check if you have Apple Silicon (M1/M2/M3)
- Run: `uname -m` (should show `arm64`)
- If yes, change `device: "mps"` in config.yaml
- Will be 3-5x faster than CPU

### Issue: "MPS backend not available"

**Solution:**
- Make sure you're on macOS 12.3+
- Update PyTorch: `pip install --upgrade torch torchvision`
- If still doesn't work, use `device: "cpu"`

### Issue: Git push requires password

**Solution:**
```bash
# Set up SSH keys (recommended)
# Or use personal access token instead of password
# See: https://docs.github.com/en/authentication
```

### Issue: Out of memory during training

**Solution:**
1. Reduce batch size in `config/config.yaml`:
   ```yaml
   training:
     batch_size: 16  # reduce from 32
   ```
2. Or use a smaller model:
   ```yaml
   model:
     architecture: "resnet18"  # smaller than resnet50
   ```

---

## Quick Reference: Complete Workflow

### First Time Setup (Linux)
```bash
cd /home/bhargav/academic-projects
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/academic-projects.git
git push -u origin main
```

### Clone to Mac
```bash
cd ~/Documents
git clone https://github.com/YOUR_USERNAME/academic-projects.git
cd academic-projects/academic/safety-helmet-detection
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision scikit-learn pyyaml tqdm numpy pandas matplotlib seaborn
```

### Run Project
```bash
cd ~/Documents/academic-projects/academic/safety-helmet-detection
source .venv/bin/activate
python src/train.py
python src/evaluate.py
```

### Update and Sync
```bash
# After making changes
git add .
git commit -m "Description of changes"
git push

# On other machine, get latest
git pull
```

---

## Additional Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **Git Cheat Sheet**: https://education.github.com/git-cheat-sheet-education.pdf
- **Python Virtual Environments**: https://docs.python.org/3/tutorial/venv.html
- **Apple Silicon GPU**: https://pytorch.org/docs/stable/notes/mps.html

---

## Contact & Support

If you encounter issues:
1. Check this guide's Troubleshooting section
2. Search for error message online
3. Check project's GitHub Issues (if public)
4. Ask on Stack Overflow with relevant tags

---

**Last Updated**: December 2024
**Version**: 1.0
