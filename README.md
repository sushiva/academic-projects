# Academic Projects

Collection of machine learning and AI projects completed during my academic coursework.

## Projects

### 1. Safety Helmet Detection
Binary image classification system to detect whether a person is wearing a safety helmet using deep learning (ResNet18 transfer learning).

- **Tech Stack**: PyTorch, ResNet18, Python
- **Dataset**: 631 images (200x200x3)
- **Accuracy**: 100% (test set)
- **Location**: [academic/safety-helmet-detection/](academic/safety-helmet-detection/)
- **Details**: See project [README](academic/safety-helmet-detection/README.md)

## Setup

Each project contains its own README with specific setup instructions. Generally:

```bash
cd academic/[project-name]
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Documentation & Resources

This repository includes comprehensive guides and references in the [docs/](docs/) folder:

- **[SUMMARY.md](docs/SUMMARY.md)** - **START HERE** - Project overview and next steps
- [SETUP_GUIDE.md](docs/SETUP_GUIDE.md) - Complete setup instructions for all machines
- [FAQ.md](docs/FAQ.md) - Frequently asked questions and troubleshooting
- [ML_OPTIONS_GUIDE.md](docs/ML_OPTIONS_GUIDE.md) - Comprehensive ML techniques reference
- [APPROACH_COMPARISON.md](docs/APPROACH_COMPARISON.md) - Detailed comparison of different ML approaches
- [QUICK_COMPARISON.md](docs/QUICK_COMPARISON.md) - Visual quick reference for approach comparison

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
├── docs/                        ⭐ Documentation
│   ├── SUMMARY.md              (Start here)
│   ├── SETUP_GUIDE.md
│   ├── FAQ.md
│   ├── ML_OPTIONS_GUIDE.md
│   ├── APPROACH_COMPARISON.md
│   └── QUICK_COMPARISON.md
└── README.md
```
