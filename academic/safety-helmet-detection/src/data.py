import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from pathlib import Path
import yaml


class HelmetDataset(Dataset):
    """Dataset class for helmet detection"""

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to PIL Image for transforms
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(config, is_training=True):
    """
    Get data transforms based on augmentation level.

    Augmentation Levels:
        - none: No augmentation (validation/test or baseline training)
        - light: Minimal augmentation (horizontal flip only)
        - medium: Moderate augmentation (rotation, flip, color jitter)
        - heavy: Maximum augmentation (all techniques)

    Args:
        config: Configuration dictionary
        is_training: Whether transforms are for training set

    Returns:
        torchvision.transforms.Compose: Composed transforms
    """

    # Get augmentation level from config (default to 'medium' for backward compatibility)
    aug_level = config['data'].get('augmentation_level', 'medium')

    # Base normalization (always applied)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet statistics
        std=[0.229, 0.224, 0.225]
    )

    # If not training or no augmentation, return just normalization
    if not is_training or aug_level == 'none':
        return transforms.Compose([normalize])

    # Build augmentation based on level
    if aug_level == 'light':
        # Light augmentation: Only horizontal flip
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            normalize
        ])

    elif aug_level == 'medium':
        # Medium augmentation: Rotation, flip, color jitter
        return transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            normalize
        ])

    elif aug_level == 'heavy':
        # Heavy augmentation: All techniques
        return transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            normalize
        ])

    else:
        raise ValueError(f"Unknown augmentation level: {aug_level}. "
                        f"Choose from: 'none', 'light', 'medium', 'heavy'")


def load_and_split_data(config):
    """Load data and create train/val/test splits"""

    # Get project root
    project_root = Path(__file__).parent.parent

    # Load data
    print("Loading dataset...")
    # Support both old and new config formats
    images_path = project_root / config['data'].get('images_path', config['data'].get('raw_images', 'data/raw/images_proj.npy'))
    labels_path = project_root / config['data'].get('labels_path', config['data'].get('raw_labels', 'data/raw/labels_proj.csv'))

    images = np.load(images_path)
    labels_df = pd.read_csv(labels_path)
    labels = labels_df['Label'].values

    # Normalize images to [0, 1]
    images = images.astype(np.float32) / 255.0

    print(f"Loaded {len(images)} images with shape {images.shape}")
    print(f"Class distribution: {np.bincount(labels)}")

    # Split data
    random_seed = config['data']['random_seed']
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    test_split = config['data']['test_split']

    # First split: separate test set
    train_val_images, test_images, train_val_labels, test_labels = train_test_split(
        images, labels,
        test_size=test_split,
        random_state=random_seed,
        stratify=labels
    )

    # Second split: separate train and validation
    val_ratio = val_split / (train_split + val_split)
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_val_images, train_val_labels,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=train_val_labels
    )

    print(f"\nData split:")
    print(f"  Train: {len(train_images)} images ({len(train_images)/len(images)*100:.1f}%)")
    print(f"  Val:   {len(val_images)} images ({len(val_images)/len(images)*100:.1f}%)")
    print(f"  Test:  {len(test_images)} images ({len(test_images)/len(images)*100:.1f}%)")

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


def create_dataloaders(config):
    """Create PyTorch DataLoaders for train/val/test"""

    # Load and split data
    train_data, val_data, test_data = load_and_split_data(config)

    # Get transforms
    train_transform = get_transforms(config, is_training=True)
    val_transform = get_transforms(config, is_training=False)

    # Create datasets
    train_dataset = HelmetDataset(*train_data, transform=train_transform)
    val_dataset = HelmetDataset(*val_data, transform=val_transform)
    test_dataset = HelmetDataset(*test_data, transform=val_transform)

    # Create dataloaders
    batch_size = config['training']['batch_size']

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the data loading
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config/config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    train_loader, val_loader, test_loader = create_dataloaders(config)

    print(f"\nDataLoaders created successfully!")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
