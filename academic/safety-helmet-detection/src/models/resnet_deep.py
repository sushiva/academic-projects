"""
Model 3: ResNet18 + Deep Classifier

ResNet18 with frozen backbone and deep multi-layer classifier to demonstrate
the importance of classifier capacity.

Expected Performance: 95-98% accuracy
Purpose: Show that a deeper classifier can better leverage pre-trained features
Improvement over Model 2: +5-8%
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet18Deep(nn.Module):
    """
    ResNet18 with frozen backbone and deep classifier head.

    Architecture:
        - ResNet18 Backbone (frozen, pre-trained on ImageNet)
        - Deep Classifier:
            - Dense(512 → 512) → ReLU → Dropout(0.5)
            - Dense(512 → 256) → ReLU → Dropout(0.3)
            - Dense(256 → 128) → ReLU → Dropout(0.2)
            - Dense(128 → 2)

    The deeper classifier provides more capacity to learn complex decision boundaries
    from the pre-trained features.

    Args:
        config: Configuration dictionary containing model parameters
    """

    def __init__(self, config):
        super(ResNet18Deep, self).__init__()

        self.num_classes = config['model']['num_classes']
        self.pretrained = config['model'].get('pretrained', True)
        self.freeze_backbone = config['model'].get('freeze_backbone', True)

        # Classifier architecture (can be customized via config)
        self.classifier_layers = config['model'].get('classifier_layers', [512, 256, 128])
        self.dropout_rates = config['model'].get('dropout_rates', [0.5, 0.3, 0.2])

        # Load pre-trained ResNet18
        if self.pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)

        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features  # 512 for ResNet18

        # Remove the original classification layer
        self.backbone.fc = nn.Identity()

        # Freeze backbone if specified
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Build deep classifier
        self.classifier = self._build_classifier(num_features)

    def _build_classifier(self, input_features):
        """
        Build deep classifier head.

        Args:
            input_features: Number of input features from backbone

        Returns:
            nn.Sequential: Deep classifier module
        """
        layers = []
        in_features = input_features

        # Add intermediate layers
        for hidden_size, dropout_rate in zip(self.classifier_layers, self.dropout_rates):
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            in_features = hidden_size

        # Add final classification layer
        layers.append(nn.Linear(in_features, self.num_classes))

        return nn.Sequential(*layers)

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

        # Pass through deep classifier
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
            'freeze_backbone': True,
            'classifier_layers': [512, 256, 128],
            'dropout_rates': [0.5, 0.3, 0.2]
        }
    }

    model = ResNet18Deep(config)
    print(f"Model: ResNet18Deep")
    print(f"Total parameters: {model.get_num_parameters(trainable_only=False):,}")
    print(f"Trainable parameters: {model.get_num_parameters(trainable_only=True):,}")

    # Test forward pass
    x = torch.randn(4, 3, 200, 200)  # Batch of 4 images
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✓ Model test passed!")
