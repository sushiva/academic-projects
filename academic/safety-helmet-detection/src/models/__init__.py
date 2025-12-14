"""
Model architectures for safety helmet detection.

This package contains different model implementations for comparative analysis:
- SimpleCNN: Baseline convolutional neural network
- ResNet18Base: Transfer learning with frozen ResNet18 backbone
- ResNet18Deep: ResNet18 with deep classifier head
- ResNet18Augmented: Full pipeline with heavy augmentation
"""

from .simple_cnn import SimpleCNN
from .resnet_base import ResNet18Base
from .resnet_deep import ResNet18Deep
from .resnet_augmented import ResNet18Augmented

__all__ = [
    'SimpleCNN',
    'ResNet18Base',
    'ResNet18Deep',
    'ResNet18Augmented',
]
