"""
Model 1: Simple CNN (Baseline)

A simple convolutional neural network trained from scratch to establish baseline performance.
No transfer learning, no pre-trained weights.

Expected Performance: 75-85% accuracy
Purpose: Demonstrate the challenge of training from scratch on small datasets
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for binary image classification.

    Architecture:
        - Conv Block 1: Conv2D(32) → BatchNorm → ReLU → MaxPool
        - Conv Block 2: Conv2D(64) → BatchNorm → ReLU → MaxPool
        - Conv Block 3: Conv2D(128) → BatchNorm → ReLU → MaxPool
        - Classifier: Flatten → Dense(256) → Dropout(0.5) → Dense(2)

    Args:
        config: Configuration dictionary containing model parameters
    """

    def __init__(self, config):
        super(SimpleCNN, self).__init__()

        self.num_classes = config['model']['num_classes']

        # Convolutional Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 200x200 → 100x100
        )

        # Convolutional Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 100x100 → 50x50
        )

        # Convolutional Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 50x50 → 25x25
        )

        # Calculate the size after convolutions
        # Input: 200x200, after 3 maxpools (2x2): 200/8 = 25
        # So we have 128 channels of 25x25 = 128 * 25 * 25 = 80,000 features
        self.flatten_size = 128 * 25 * 25

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 200, 200)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Initialize network weights using He initialization for ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_num_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    config = {
        'model': {
            'num_classes': 2
        }
    }

    model = SimpleCNN(config)
    print(f"Model: SimpleCNN")
    print(f"Total parameters: {model.get_num_parameters():,}")

    # Test forward pass
    x = torch.randn(4, 3, 200, 200)  # Batch of 4 images
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✓ Model test passed!")
