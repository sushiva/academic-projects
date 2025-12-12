import torch
import torch.nn as nn
from torchvision import models


class HelmetClassifier(nn.Module):
    """Helmet detection classifier using transfer learning"""

    def __init__(self, config):
        super(HelmetClassifier, self).__init__()

        self.config = config
        model_name = config['model']['architecture']
        num_classes = config['model']['num_classes']
        pretrained = config['model']['pretrained']
        dropout = config['model']['dropout']

        # Load pretrained model
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif model_name == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        else:
            raise ValueError(f"Unknown model architecture: {model_name}")

        # Freeze backbone if specified
        if config['model']['freeze_backbone']:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen for fine-tuning")


def create_model(config):
    """Factory function to create model"""
    model = HelmetClassifier(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel: {config['model']['architecture']}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


if __name__ == "__main__":
    # Test model creation
    import yaml
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    config_path = project_root / "config/config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = create_model(config)
    print(f"\nModel architecture:\n{model}")

    # Test forward pass
    dummy_input = torch.randn(1, 3, 200, 200)
    output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}")
