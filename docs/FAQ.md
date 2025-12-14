# Frequently Asked Questions (FAQ)

Common questions and answers about the safety helmet detection project and setup.

---

## General Questions

### Q: How do I verify if my laptop has a GPU?

**On Linux:**
```bash
nvidia-smi
lspci | grep -i vga
```

**On Mac:**
```bash
uname -m  # arm64 = Apple Silicon (M1/M2/M3), x86_64 = Intel
system_profiler SPDisplaysDataType
```

**Using Python:**
```bash
source .venv/bin/activate
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

---

## Training Questions

### Q: Which machines have GPUs for deep learning?

**Free Options (Best for Students):**
1. **Google Colab** - Free GPU (Tesla T4)
   - URL: https://colab.research.google.com
   - 12 hours GPU time per session
   - Training: ~10-15 minutes instead of 2 hours

2. **Kaggle Notebooks** - Free GPU (P100 or T4)
   - URL: https://kaggle.com/code
   - 30 hours/week GPU quota

3. **Your University**
   - GPU computing cluster
   - Research lab machines
   - Computer science department GPUs

**Personal Machines:**
- Gaming laptops/desktops with NVIDIA GPUs (GTX/RTX series)
- Workstations with NVIDIA Quadro/Tesla cards
- Mac with Apple Silicon (M1/M2/M3) - uses MPS, not CUDA

**Cloud Platforms (Paid):**
- AWS (p3 instances)
- Google Cloud Platform (GPU instances)
- Azure (GPU VMs)
- Paperspace Gradient

### Q: How do I use GPU on Apple Silicon Mac (M1/M2/M3)?

**1. Verify MPS is available:**
```bash
python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"
```

**2. Enable in config:**
Edit `config/config.yaml`:
```yaml
training:
  device: "mps"  # change from "cpu"
```

**3. Expected speedup:**
- CPU: 1-2 hours for 30 epochs
- MPS: 20-30 minutes for 30 epochs
- 3-5x faster than CPU

### Q: Can I see training progress while it's running?

Yes! Use one of these commands:

```bash
# View last 50 lines
tail -50 training_output.log

# Follow in real-time
tail -f training_output.log

# View specific number of lines
tail -100 training_output.log
```

---

## Model Questions

### Q: Should the model be saved as .pkl or .pth file?

**Recommendation: Use `.pth` (what we use by default)**

**`.pth` (PyTorch format):**
- PyTorch's native format
- Uses `torch.save()` and `torch.load()`
- More efficient for PyTorch models
- Industry standard
- Better version compatibility

**`.pkl` (Pickle format):**
- Python's general serialization format
- Can save any Python object
- Less efficient for deep learning models
- Not recommended for PyTorch models

**Current setup:**
```python
# In train.py
torch.save(checkpoint, 'models/best_model.pth')

# In evaluate.py
checkpoint = torch.load('models/best_model.pth')
```

**If you need both formats:**
You can save both:
```python
import pickle
# Save as .pth
torch.save(model.state_dict(), 'model.pth')
# Save as .pkl
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

---

## Git & GitHub Questions

### Q: How do I organize multiple academic projects in one GitHub repo?

**Structure:**
```
academic-projects/
├── README.md
├── .gitignore
└── academic/
    ├── safety-helmet-detection/
    ├── nlp-project/
    └── recommendation-system/
```

**Steps:**
1. Create parent repo
2. Add projects under `academic/` folder
3. Push to GitHub
4. Clone on other machines

**Commands:**
```bash
# Initial setup
cd academic-projects
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/USERNAME/academic-projects.git
git push -u origin main

# Clone on other machine
git clone https://github.com/USERNAME/academic-projects.git
```

### Q: What files should be excluded from git?

Our `.gitignore` excludes:
- Virtual environments (`.venv/`, `venv/`)
- Python cache (`__pycache__/`, `*.pyc`)
- Data files (`*.npy`, `*.csv` in data folders)
- Model files (`*.pth`, `*.pkl`)
- Log files (`*.log`)
- OS files (`.DS_Store`)
- IDE files (`.vscode/`, `.idea/`)

**Why exclude these?**
- **Data/Models**: Too large for git (use cloud storage)
- **Logs**: Generated files, not source code
- **Cache/temp**: Can be regenerated
- **OS/IDE**: Machine-specific

---

## Environment & Dependencies

### Q: How do I set up the environment on a new machine?

**Mac/Linux:**
```bash
cd academic-projects/academic/safety-helmet-detection

# Create virtual environment
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Install dependencies
pip install torch torchvision scikit-learn pyyaml tqdm numpy pandas matplotlib seaborn

# Verify
python -c "import torch; print(torch.__version__)"
```

**Windows:**
```bash
cd academic-projects\academic\safety-helmet-detection

# Create virtual environment
python -m venv .venv

# Activate
.venv\Scripts\activate

# Install dependencies
pip install torch torchvision scikit-learn pyyaml tqdm numpy pandas matplotlib seaborn
```

### Q: How do I know if packages are installed correctly?

```bash
# Activate environment
source .venv/bin/activate

# Check installed packages
pip list | grep -E "torch|numpy|pandas"

# Test imports
python -c "import torch, torchvision, numpy, pandas; print('All imports successful!')"
```

---

## Project Workflow

### Q: What's the complete workflow from setup to submission?

**Day 1: Setup & Training**
1. Set up environment
2. Run EDA (already done)
3. Start training (10-120 minutes depending on device)
4. Training completes automatically

**Day 2: Evaluation & Improvement**
1. Run evaluation on test set
2. Review results
3. If accuracy < 85%, try improvements:
   - Different architecture (ResNet50, MobileNet)
   - More data augmentation
   - Adjust hyperparameters
4. Generate visualizations

**Day 3: Documentation & Submission**
1. Update README with results
2. Clean up code
3. Push to GitHub
4. Prepare presentation materials
5. Submit assignment

### Q: What commands do I run to complete the project?

**Training:**
```bash
cd academic-projects/academic/safety-helmet-detection
source .venv/bin/activate
python src/train.py
```

**Evaluation:**
```bash
python src/evaluate.py
```

**View results:**
```bash
cat outputs/evaluation/metrics.txt
open outputs/evaluation/confusion_matrix.png
```

**Push to GitHub:**
```bash
cd ~/academic-projects
git add .
git commit -m "Update with training results"
git push
```

---

## Results & Interpretation

### Q: What does 100% validation accuracy mean?

**Good news:**
- Model learned the task perfectly on validation set
- Strong performance indicator

**Things to check:**
1. **Test set performance**: Run evaluation to verify on unseen data
2. **Overfitting**: If test accuracy << validation accuracy, model overfit
3. **Data leakage**: Verify train/val/test splits are correct

**Next steps:**
- Run `python src/evaluate.py` to test on held-out test set
- Check confusion matrix for any patterns
- Review misclassified examples (if any)

### Q: What results do I need for my assignment?

**Essential outputs:**
1. **Metrics** (`outputs/evaluation/metrics.txt`):
   - Accuracy, Precision, Recall, F1-Score

2. **Visualizations**:
   - Training curves (`outputs/training/plots/training_history.png`)
   - Confusion matrix (`outputs/evaluation/confusion_matrix.png`)
   - ROC curve (`outputs/evaluation/roc_curve.png`)
   - Sample predictions

3. **Model**:
   - Trained model file (`models/best_model.pth`)

4. **Documentation**:
   - README with project description
   - Results and interpretation
   - How to reproduce

---

## Troubleshooting

### Q: Training is taking too long

**Solutions:**
1. **Use GPU**: Switch to MPS (Mac) or CUDA (NVIDIA)
2. **Reduce epochs**: Change `epochs: 30` to `epochs: 10` in config
3. **Reduce batch size**: If out of memory
4. **Use smaller model**: Switch from ResNet50 to ResNet18
5. **Use Google Colab**: Free GPU, much faster

### Q: Model accuracy is too low (< 85%)

**Try these improvements:**

1. **More training:**
   ```yaml
   training:
     epochs: 50  # increase from 30
   ```

2. **Different architecture:**
   ```yaml
   model:
     architecture: "resnet50"  # try larger model
   ```

3. **More data augmentation:**
   ```yaml
   data:
     augmentation:
       rotation: 20  # increase from 15
       brightness: 0.3  # increase from 0.2
   ```

4. **Lower learning rate:**
   ```yaml
   training:
     learning_rate: 0.0001  # reduce from 0.001
   ```

### Q: Evaluation script fails

**Common issues:**

1. **Model file not found:**
   ```bash
   # Check if model exists
   ls -lh models/best_model.pth

   # If missing, training didn't complete
   # Re-run training
   ```

2. **Import errors:**
   ```bash
   # Make sure environment is activated
   source .venv/bin/activate

   # Reinstall packages
   pip install torch torchvision scikit-learn
   ```

---

## Performance Optimization

### Q: How can I make training faster?

**Hardware:**
1. Use GPU (MPS on Mac, CUDA on NVIDIA)
2. Use cloud GPU (Google Colab, Kaggle)
3. Increase batch size if you have more memory

**Software:**
1. Reduce image size (if acceptable)
2. Use smaller model (ResNet18 vs ResNet50)
3. Reduce number of workers in DataLoader
4. Disable some augmentations

**Example config for faster training:**
```yaml
training:
  batch_size: 64  # increase if memory allows
  epochs: 20      # reduce
  device: "mps"   # use GPU

data:
  augmentation:
    enabled: false  # disable for faster training (not recommended)
```

---

## Tips & Best Practices

### Q: What should I include in my assignment report?

**Essential sections:**
1. **Introduction**: Problem statement, objectives
2. **Dataset**: Description, size, distribution
3. **Methodology**:
   - Model architecture
   - Training procedure
   - Hyperparameters
4. **Results**:
   - Metrics (accuracy, precision, recall, F1)
   - Visualizations (confusion matrix, ROC curve)
   - Training curves
5. **Discussion**:
   - Interpretation of results
   - Strengths and limitations
   - Future improvements
6. **Conclusion**: Summary of findings
7. **References**: Dataset source, frameworks used

### Q: How do I make my project more impressive?

**Quick wins:**
1. **Add Grad-CAM visualization**: Show what model is looking at
2. **Try multiple models**: Compare ResNet18, ResNet50, MobileNet
3. **Error analysis**: Analyze misclassified examples
4. **Create demo**: Simple web interface with Gradio
5. **Document well**: Clear README, comments in code

**Code quality:**
1. Follow PEP 8 style guidelines
2. Add docstrings to functions
3. Use meaningful variable names
4. Keep functions small and focused

---

## Quick Command Reference

**Environment:**
```bash
source .venv/bin/activate    # Activate
deactivate                   # Deactivate
pip list                     # List packages
```

**Training:**
```bash
python src/train.py          # Normal run
nohup python src/train.py > training.log 2>&1 &  # Background
tail -f training.log         # Monitor progress
```

**Evaluation:**
```bash
python src/evaluate.py       # Run evaluation
cat outputs/evaluation/metrics.txt  # View metrics
```

**Git:**
```bash
git status                   # Check status
git add .                    # Stage all changes
git commit -m "message"      # Commit
git push                     # Push to GitHub
git pull                     # Pull latest changes
```

**File operations:**
```bash
ls -lh models/              # List model files
du -sh data/                # Check data size
find . -name "*.pth"        # Find all model files
```

---

## ML Engineering Best Practices

### Q: What's the industry standard for ML configuration management?

**Short Answer**: Separate config files per experiment (what we're using) OR base config + experiment overrides with tools like Hydra.

**Industry Approaches**:

1. **Separate Configs (Current approach - ✅ Recommended for students)**
   ```
   config/
   ├── model1_baseline.yaml
   ├── model2_transfer.yaml
   └── model3_optimized.yaml
   ```
   - **Used by**: Most companies for clarity
   - **Pros**: Clear, git-friendly, easy to track experiments
   - **Cons**: Some duplication of common settings

2. **Hydra (Meta/Facebook standard)**
   ```
   config/
   ├── base.yaml
   └── experiments/
       ├── baseline.yaml
       └── transfer.yaml
   ```
   ```bash
   # Run with overrides
   python train.py model=resnet18 lr=0.001
   ```
   - **Used by**: Meta, Uber, Airbnb
   - **Pros**: Config composition, command-line overrides
   - **Cons**: Requires learning Hydra

3. **MLflow (Experiment tracking)**
   ```python
   import mlflow
   mlflow.log_params({"model": "resnet18", "lr": 0.001})
   ```
   - **Used by**: Netflix, Databricks
   - **Pros**: Track and compare experiments
   - **Cons**: Additional infrastructure

4. **Weights & Biases (Research favorite)**
   ```python
   import wandb
   wandb.init(config={"model": "resnet18"})
   ```
   - **Used by**: OpenAI, Toyota Research
   - **Pros**: Beautiful dashboards, team collaboration
   - **Cons**: Requires account (free for academics)

**What Big Companies Use**:

| Company | Approach | Tool |
|---------|----------|------|
| Google | Separate configs + Protocol Buffers | Internal |
| Meta | Config composition | Hydra |
| OpenAI | Separate configs | Custom |
| Uber | Config composition + Tracking | Hydra + MLflow |
| Netflix | Experiment tracking | MLflow |

**Our Approach**: Separate configs (industry-standard, simple, clear)

**Best Practices**:
- ✅ Version control all configs (git)
- ✅ One config per experiment
- ✅ Document config changes in commits
- ✅ Use meaningful names (model1_baseline.yaml)
- ✅ Add comments explaining unusual values
- ❌ Don't hardcode parameters in code
- ❌ Don't mix config and code logic
- ❌ Don't use environment variables for everything

**For Future Projects**: Consider adding:
1. **Hydra** for config composition
2. **MLflow** or **W&B** for experiment tracking
3. **Pydantic** for config validation
4. **DVC** for data versioning

**Example Hydra Setup** (for reference):
```yaml
# config/base.yaml
defaults:
  - data: default
  - training: default

data:
  batch_size: 32
  train_split: 0.7

# config/experiments/resnet.yaml
defaults:
  - /base

model:
  type: ResNet18
  pretrained: true
```

```bash
# Run experiments
python train.py experiment=resnet
python train.py experiment=resnet model.pretrained=false  # Override
```

**Resources**:
- Hydra: https://hydra.cc/
- MLflow: https://mlflow.org/
- Weights & Biases: https://wandb.ai/
- DVC: https://dvc.org/

---

**Last Updated**: December 2024
**Version**: 1.1

For more detailed instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md)
