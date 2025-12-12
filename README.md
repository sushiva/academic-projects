# Academic Projects

Collection of machine learning and AI projects completed during my academic coursework.

## Projects

### 1. Safety Helmet Detection
Binary image classification system to detect whether a person is wearing a safety helmet using deep learning (ResNet18 transfer learning).

- **Tech Stack**: PyTorch, ResNet18, Python
- **Dataset**: 631 images (200x200x3)
- **Accuracy**: [To be updated after training]
- **Location**: `academic/safety-helmet-detection/`

## Setup

Each project contains its own README with specific setup instructions. Generally:

```bash
cd academic/[project-name]
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Structure

```
academic-projects/
├── academic/
│   └── safety-helmet-detection/
│       ├── config/
│       ├── data/
│       ├── models/
│       ├── notebooks/
│       ├── outputs/
│       ├── src/
│       └── README.md
└── README.md
```
