# ML Project Options & Techniques Guide

A comprehensive guide to different options, techniques, and tools you can use in machine learning projects.

---

## 1. Model Architectures (Computer Vision)

### Transfer Learning Models

**ResNet (Residual Networks)**
- **Variants**: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
- **When to use**: General-purpose, proven architecture
- **Pros**: Skip connections prevent vanishing gradients, well-tested
- **Cons**: Larger models (ResNet50+) are slower
- **Best for**: When accuracy is priority over speed

**EfficientNet**
- **Variants**: B0 (smallest) to B7 (largest)
- **When to use**: Need best accuracy-to-speed ratio
- **Pros**: State-of-art accuracy with fewer parameters
- **Cons**: Slightly more complex to fine-tune
- **Best for**: Production deployment where both accuracy and speed matter

**MobileNet**
- **Variants**: MobileNetV2, MobileNetV3
- **When to use**: Mobile/edge devices, real-time applications
- **Pros**: Very fast, small model size
- **Cons**: Slightly lower accuracy than ResNet/EfficientNet
- **Best for**: Mobile apps, embedded systems, real-time inference

**VGG (VGG16, VGG19)**
- **When to use**: Educational purposes, baseline comparisons
- **Pros**: Simple architecture, easy to understand
- **Cons**: Very large, slow, outdated
- **Best for**: Learning, not recommended for production

**DenseNet**
- **When to use**: When you need maximum information flow
- **Pros**: Dense connections, good gradient flow
- **Cons**: Memory intensive
- **Best for**: Research, when GPU memory is abundant

**Vision Transformer (ViT)**
- **When to use**: Large datasets, cutting-edge research
- **Pros**: State-of-art on large datasets
- **Cons**: Needs lots of data, computationally expensive
- **Best for**: Large-scale projects with abundant data

### Custom Architectures

**When to build custom:**
- Very specific problem domain
- Need exact control over model size/speed
- Research purposes

**Example use cases:**
- Specialized medical imaging
- Custom object detection
- Novel computer vision tasks

---

## 2. Training Techniques

### Data Augmentation

**Basic Augmentations** (What we used):
```python
- Random rotation
- Horizontal/vertical flip
- Color jitter (brightness, contrast)
- Random crop
- Normalization
```

**Advanced Augmentations**:
```python
- Mixup: Blend two images together
- Cutout: Random rectangular masks
- CutMix: Cut and paste from another image
- AutoAugment: ML-learned augmentation policies
- RandAugment: Random augmentation selection
- Albumentations library: Professional augmentations
```

**When to use advanced:**
- Limited training data
- Model is overfitting
- Need to simulate different conditions

### Learning Rate Strategies

**ReduceLROnPlateau** (What we used):
- Reduces LR when metric stops improving
- Simple, effective
- Good default choice

**Other options:**
- **CosineAnnealingLR**: Smooth cosine decay
- **StepLR**: Drop LR at specific epochs
- **ExponentialLR**: Exponential decay
- **OneCycleLR**: Modern, fast convergence
- **CyclicLR**: Cyclical learning rates

**Learning Rate Finder:**
- Find optimal LR before training
- Tools: fastai's lr_find()

### Regularization Techniques

**Dropout** (What we used):
- Randomly drop neurons during training
- Prevents overfitting

**Other options:**
- **L1/L2 Regularization**: Weight decay
- **Batch Normalization**: Normalize layer inputs
- **Label Smoothing**: Soften one-hot labels
- **Early Stopping**: Stop when validation stops improving
- **Data Augmentation**: Implicit regularization

### Optimizers

**Adam** (What we used):
- Adaptive learning rates
- Good default choice
- Works well out-of-box

**Other options:**
- **SGD with Momentum**: Classic, often better final accuracy
- **AdamW**: Adam with proper weight decay
- **RAdam**: More stable than Adam
- **Ranger**: RAdam + Lookahead
- **LAMB**: For very large batch sizes

---

## 3. Experiment Tracking

### TensorBoard (Free, PyTorch native)
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')
writer.add_scalar('Loss/train', loss, epoch)
writer.add_scalar('Accuracy/train', acc, epoch)
writer.add_images('predictions', images, epoch)
```

**Pros**: Free, integrated with PyTorch, good visualization
**Cons**: Limited collaboration features
**Best for**: Solo projects, local development

### Weights & Biases (wandb)
```python
import wandb

wandb.init(project="helmet-detection")
wandb.log({"loss": loss, "accuracy": acc})
wandb.watch(model)
```

**Pros**: Cloud-based, beautiful UI, easy sharing, hyperparameter sweep
**Cons**: Requires account, some features paid
**Best for**: Team projects, sharing results, hyperparameter tuning

### MLflow
```python
import mlflow

mlflow.log_param("learning_rate", 0.001)
mlflow.log_metric("accuracy", acc)
mlflow.pytorch.log_model(model, "model")
```

**Pros**: Open source, model registry, production-oriented
**Cons**: More complex setup
**Best for**: Production ML pipelines, model versioning

### Neptune.ai
Similar to W&B, more enterprise-focused

### Comet.ml
Similar to W&B, good for experimentation

---

## 4. Model Interpretability

### Grad-CAM (Gradient-weighted Class Activation Mapping)
```python
# Shows what parts of image the model looks at
from pytorch_grad_cam import GradCAM

cam = GradCAM(model, target_layer)
heatmap = cam(input_image)
```

**What it shows**: Highlights important regions in image
**When to use**:
- Debugging model decisions
- Building trust in model
- Understanding failures
**Great for presentations**: Visual proof model works correctly

### Saliency Maps
- Shows pixel importance
- Simpler than Grad-CAM

### LIME (Local Interpretable Model-agnostic Explanations)
- Explains any model's predictions
- Model-agnostic

### SHAP (SHapley Additive exPlanations)
- Game theory-based explanations
- More rigorous than LIME

**When interpretability matters:**
- Medical applications
- Financial decisions
- Regulatory requirements
- Debugging model behavior

---

## 5. Deployment Options

### REST API with FastAPI

**Example:**
```python
from fastapi import FastAPI, File, UploadFile
import torch

app = FastAPI()
model = torch.load('model.pth')

@app.post("/predict")
async def predict(file: UploadFile):
    image = load_image(file)
    prediction = model(image)
    return {"class": prediction}
```

**Pros**: Fast, modern, automatic docs, async
**Best for**: Production APIs, microservices

**Alternative: Flask**
- Simpler, more established
- Good for simple APIs

### Gradio (Interactive Demos)

**Example:**
```python
import gradio as gr

def predict(image):
    return model.predict(image)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label()
)
demo.launch()
```

**Pros**: Zero web development needed, beautiful UI, shareable
**Best for**: Demos, prototypes, non-technical users

**Alternative: Streamlit**
- More customizable
- Better for data apps

### Docker Containerization

**Why Docker:**
- Reproducibility
- Easy deployment
- Environment consistency

**Example Dockerfile:**
```dockerfile
FROM python:3.13
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "src/train.py"]
```

**When to use:**
- Production deployment
- Sharing with others
- Cloud deployment

### Cloud Deployment Options

**HuggingFace Spaces** (Easiest):
- Free hosting for Gradio/Streamlit
- Automatic from Git
- Great for demos

**AWS SageMaker**:
- Enterprise-grade
- Scalable
- More complex

**Google Cloud AI Platform**:
- Similar to SageMaker
- Good integration with GCP

**Azure ML**:
- Microsoft's offering
- Good for enterprise

**Heroku** (Deprecated for free tier):
- Was popular for simple apps
- Now mostly paid

**Railway/Render**:
- Modern alternatives to Heroku
- Simple deployment

---

## 6. Hyperparameter Tuning

### Manual Tuning (What we did)
- Set hyperparameters based on experience
- Simple, quick for known problems

### Grid Search
```python
from sklearn.model_selection import GridSearchCV

params = {
    'lr': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64]
}
# Try all combinations
```

**Pros**: Systematic, thorough
**Cons**: Expensive, slow

### Random Search
```python
# Random sampling from ranges
# Often better than grid search
```

**Pros**: More efficient than grid
**Cons**: Might miss optimal

### Bayesian Optimization (Optuna)
```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1)
    model = train(lr=lr)
    return model.accuracy

study = optuna.create_study()
study.optimize(objective, n_trials=100)
```

**Pros**: Smart search, finds optima faster
**Best for**: Expensive training, limited budget

### Population-based Training
- Evolves hyperparameters during training
- State-of-art but complex

---

## 7. Model Evaluation Metrics

### Classification Metrics (What we used)

**Accuracy**: Overall correctness
- Good when classes balanced
- Misleading with imbalanced data

**Precision**: Of predicted positives, how many correct?
- Important when false positives are costly

**Recall**: Of actual positives, how many caught?
- Important when false negatives are costly

**F1-Score**: Harmonic mean of precision and recall
- Good balance metric

**Confusion Matrix**: Shows all prediction types
- Visual understanding of errors

**ROC-AUC**: Threshold-independent performance
- Good for comparing models

### Advanced Metrics

**PR-AUC**: Precision-Recall AUC
- Better for imbalanced datasets

**Matthews Correlation Coefficient**:
- Good for imbalanced data

**Cohen's Kappa**:
- Agreement vs chance

### Regression Metrics

**MAE** (Mean Absolute Error): Average absolute difference
**MSE** (Mean Squared Error): Penalizes large errors
**RMSE** (Root MSE): Same units as target
**R²**: Explained variance

---

## 8. Testing & Validation

### Unit Tests (pytest)
```python
def test_model_output_shape():
    model = HelmetClassifier(config)
    output = model(dummy_input)
    assert output.shape == (1, 2)

def test_data_loader():
    loader = create_dataloaders(config)
    assert len(loader) > 0
```

**Why test:**
- Catch bugs early
- Ensure reproducibility
- Professional code

### Integration Tests
- Test full pipeline
- End-to-end validation

### Model Performance Tests
```python
def test_accuracy_threshold():
    accuracy = evaluate(model)
    assert accuracy > 0.95
```

---

## 9. Advanced Techniques

### Ensemble Methods

**Voting**:
- Combine multiple models
- Average predictions

**Stacking**:
- Train meta-model on predictions
- Often better than voting

**When to use:**
- Competition settings
- When accuracy is critical
- Have computational resources

### Self-Supervised Learning
- Train without labels
- Use pretext tasks

### Few-Shot Learning
- Learn from very few examples
- Meta-learning approaches

### Active Learning
- Model selects most informative samples
- Efficient labeling

### Knowledge Distillation
- Train smaller model from larger
- Deployment optimization

---

## 10. Production Best Practices

### Model Versioning
- Track model versions
- MLflow model registry
- DVC for large files

### A/B Testing
- Compare model versions in production
- Gradual rollout

### Monitoring
- Track prediction distribution
- Detect model drift
- Alert on anomalies

### CI/CD for ML
- Automated testing
- Automated deployment
- GitHub Actions, Jenkins

### Model Cards
```markdown
# Model Card

## Model Details
- Architecture: ResNet18
- Training date: 2024-12-12
- Accuracy: 100%

## Intended Use
- Safety helmet detection
- Construction sites

## Limitations
- Requires clear images
- 200x200 input size

## Evaluation
- Test accuracy: 100%
- Balanced dataset
```

---

## 11. Common Pitfalls to Avoid

### Data Leakage
- Don't augment before splitting
- Don't use test data for any training decisions

### Overfitting
- Model memorizes training data
- Solutions: regularization, more data, augmentation

### Underfitting
- Model too simple
- Solutions: bigger model, train longer

### Class Imbalance
- Unequal class distribution
- Solutions: weighted loss, oversampling, undersampling

### Not Using Validation Set
- Test set contamination
- Always have separate validation

---

## 12. When to Use What

### Small Dataset (<1000 images)
- Transfer learning (ResNet18, MobileNet)
- Heavy augmentation
- Simple architectures

### Medium Dataset (1k-100k)
- Transfer learning with fine-tuning
- Standard augmentation
- ResNet, EfficientNet

### Large Dataset (>100k)
- Can train from scratch
- Larger models (ResNet50, EfficientNet)
- Consider Vision Transformers

### Limited Compute
- MobileNet, EfficientNet-B0
- Smaller batch sizes
- Mixed precision training

### Production Deployment
- MobileNet for edge
- EfficientNet for cloud
- Docker + FastAPI
- Monitoring setup

### Research/Exploration
- Try multiple architectures
- Experiment tracking (W&B)
- Interpretability tools

---

## Quick Decision Tree

```
Need to deploy?
├─ Yes
│  ├─ Mobile/Edge → MobileNet + Gradio/FastAPI
│  ├─ Cloud → EfficientNet + Docker + FastAPI
│  └─ Demo → Gradio on HuggingFace Spaces
│
└─ No (Research/Assignment)
   ├─ Limited time → ResNet18 + Basic augmentation
   ├─ Want to learn → Try multiple models + Experiment tracking
   └─ Need interpretability → Add Grad-CAM
```

---

## Resources for Learning

**Courses:**
- Fast.ai - Practical Deep Learning
- deeplearning.ai - Andrew Ng's courses
- PyTorch tutorials - Official docs

**Books:**
- "Deep Learning for Coders" (fastai)
- "Hands-On Machine Learning" (Géron)
- "Deep Learning" (Goodfellow)

**Papers:**
- ResNet: "Deep Residual Learning"
- EfficientNet: "Rethinking Model Scaling"
- Vision Transformer: "An Image is Worth 16x16 Words"

**Tools to Explore:**
- PyTorch Lightning - Less boilerplate
- Hugging Face Transformers - Pre-trained models
- timm (PyTorch Image Models) - Model zoo

---

## Summary: What to Try Next Project

**Beginner → Intermediate:**
1. Add TensorBoard logging
2. Try different augmentations
3. Compare 2-3 architectures
4. Add Grad-CAM visualization

**Intermediate → Advanced:**
1. Implement Gradio demo
2. Build FastAPI endpoint
3. Use Weights & Biases
4. Docker deployment
5. Try ensemble methods

**Advanced → Production:**
1. Full CI/CD pipeline
2. Model monitoring
3. A/B testing framework
4. Hyperparameter tuning with Optuna
5. Knowledge distillation

---

**Remember**: Start simple, add complexity as needed. A working simple solution beats a broken complex one!

**Last Updated**: December 2024
