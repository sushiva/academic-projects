"""
Model 2: ResNet18 Base (Transfer Learning)

ResNet18 with frozen backbone and simple classifier to demonstrate
the power of transfer learning.

Expected Performance: 90-95% accuracy
Purpose: Show impact of pre-trained features vs training from scratch
Improvement over Model 1: +10-15%
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet18Base(nn.Module):
    """
    ResNet18 with frozen backbone and simple classifier.

    Architecture:
        - ResNet18 Backbone (frozen, pre-trained on ImageNet)
        - Simple Classifier: Flatten → Dense(2)

    Only the final classification layer is trainable (~1K parameters).
    This demonstrates pure transfer learning without fine-tuning.

    Args:
        config: Configuration dictionary containing model parameters
    """

    def __init__(self, config):
        super(ResNet18Base, self).__init__()

        self.num_classes = config['model']['num_classes']
        self.pretrained = config['model'].get('pretrained', True)
        self.freeze_backbone = config['model'].get('freeze_backbone', True)

        # Load pre-trained ResNet18
        if self.pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)

        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features

        # Remove the original classification layer
        self.backbone.fc = nn.Identity()

        # Freeze backbone if specified
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Simple classifier (just one linear layer)
        self.classifier = nn.Linear(num_features, self.num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 200, 200)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Extract features using frozen backbone
        features = self.backbone(x)  # Shape: (batch_size, 512)

        # Classify
        output = self.classifier(features)  # Shape: (batch_size, num_classes)

        return output

    def get_num_parameters(self, trainable_only=True):
        """
        Return number of parameters.

        Args:
            trainable_only: If True, return only trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def unfreeze_backbone(self):
        """Unfreeze the backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False


if __name__ == "__main__":
    # Test the model
    config = {
        'model': {
            'num_classes': 2,
            'pretrained': True,
            'freeze_backbone': True
        }
    }

    model = ResNet18Base(config)
    print(f"Model: ResNet18Base")
    print(f"Total parameters: {model.get_num_parameters(trainable_only=False):,}")
    print(f"Trainable parameters: {model.get_num_parameters(trainable_only=True):,}")

    # Test forward pass
    x = torch.randn(4, 3, 200, 200)  # Batch of 4 images
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✓ Model test passed!")
