"""
Train All Models Script

Trains all 4 models sequentially for comparative analysis:
1. Model 1: Simple CNN (Baseline)
2. Model 2: ResNet18 Base (Transfer Learning)
3. Model 3: ResNet18 + Deep Classifier
4. Model 4: ResNet18 + Heavy Augmentation

This script demonstrates progressive improvement in model performance.
"""

import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import model classes
from models import SimpleCNN, ResNet18Base, ResNet18Deep, ResNet18Augmented
from data import create_dataloaders

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ModelTrainer:
    """Trainer class for a single model"""

    def __init__(self, model, config, model_name):
        self.model = model
        self.config = config
        self.model_name = model_name
        self.device = torch.device(config['training']['device'])

        # Move model to device
        self.model.to(self.device)

        # Setup loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0)
        )

        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config['training']['scheduler']['patience'],
            factor=config['training']['scheduler']['factor'],
            min_lr=config['training']['scheduler']['min_lr']
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_no_improve = 0

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct / total

        return val_loss, val_acc

    def train(self, train_loader, val_loader, epochs):
        """Full training loop"""
        print(f"\nTraining {self.model_name}...")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print("=" * 80)

        start_time = time.time()

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)

            # Print progress
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.6f}")

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.epochs_no_improve = 0
                # Save best model
                self.save_model('best')
                print(f"  ✓ New best model! (Val Loss: {val_loss:.4f})")
            else:
                self.epochs_no_improve += 1

            # Early stopping
            if self.config['training']['early_stopping']['enabled']:
                patience = self.config['training']['early_stopping']['patience']
                if self.epochs_no_improve >= patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.2f} minutes")
        print(f"Best model at epoch {self.best_epoch} with Val Loss: {self.best_val_loss:.4f}")

        return training_time

    def save_model(self, suffix='best'):
        """Save model checkpoint"""
        project_root = Path(__file__).parent.parent
        models_dir = project_root / self.config['output']['models_dir']
        models_dir.mkdir(parents=True, exist_ok=True)

        model_path = models_dir / f"{self.model_name}_{suffix}.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }, model_path)

    def plot_training_history(self):
        """Plot training curves"""
        project_root = Path(__file__).parent.parent
        plots_dir = project_root / self.config['output']['plots_dir'] / self.model_name
        plots_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        axes[0].plot(self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(self.history['val_loss'], label='Val Loss', marker='s')
        axes[0].axvline(x=self.best_epoch-1, color='r', linestyle='--', label=f'Best Epoch ({self.best_epoch})')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{self.model_name} - Training Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Plot accuracy
        axes[1].plot(self.history['train_acc'], label='Train Acc', marker='o')
        axes[1].plot(self.history['val_acc'], label='Val Acc', marker='s')
        axes[1].axvline(x=self.best_epoch-1, color='r', linestyle='--', label=f'Best Epoch ({self.best_epoch})')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title(f'{self.model_name} - Training Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(plots_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()


def train_single_model(config_path):
    """Train a single model given its config file"""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['model']['name']
    model_type = config['model']['type']

    print(f"\n{'='*80}")
    print(f"Starting Training: {model_name}")
    print(f"Model Type: {model_type}")
    print(f"{'='*80}")

    # Create dataloaders
    print("\nPreparing data...")
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Create model
    print(f"\nInitializing {model_type} model...")
    if model_type == "SimpleCNN":
        model = SimpleCNN(config)
    elif model_type == "ResNet18Base":
        model = ResNet18Base(config)
    elif model_type == "ResNet18Deep":
        model = ResNet18Deep(config)
    elif model_type == "ResNet18Augmented":
        model = ResNet18Augmented(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create trainer
    trainer = ModelTrainer(model, config, model_name)

    # Train
    training_time = trainer.train(
        train_loader,
        val_loader,
        epochs=config['training']['epochs']
    )

    # Plot training history
    trainer.plot_training_history()

    # Save final model
    trainer.save_model('final')

    # Return results summary
    results = {
        'model_name': model_name,
        'model_type': model_type,
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'best_epoch': trainer.best_epoch,
        'best_val_loss': trainer.best_val_loss,
        'best_val_acc': max(trainer.history['val_acc']),
        'final_train_acc': trainer.history['train_acc'][-1],
        'training_time_mins': training_time / 60,
        'epochs_trained': len(trainer.history['train_loss'])
    }

    return results


def main():
    """Main function to train all models"""

    project_root = Path(__file__).parent.parent
    config_dir = project_root / "config"

    # List of config files in training order
    config_files = [
        "model1_simple_cnn.yaml",
        "model2_resnet_base.yaml",
        "model3_resnet_deep.yaml",
        "model4_resnet_augmented.yaml"
    ]

    all_results = []

    print("\n" + "="*80)
    print(" "*20 + "MULTI-MODEL TRAINING PIPELINE")
    print("="*80)
    print(f"\nTraining {len(config_files)} models sequentially...")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    total_start_time = time.time()

    for i, config_file in enumerate(config_files, 1):
        config_path = config_dir / config_file

        if not config_path.exists():
            print(f"\n⚠ Warning: Config file not found: {config_path}")
            continue

        print(f"\n[{i}/{len(config_files)}] Training model from: {config_file}")

        try:
            results = train_single_model(config_path)
            all_results.append(results)
            print(f"\n✓ {results['model_name']} completed successfully!")
            print(f"  Best Val Acc: {results['best_val_acc']:.2f}%")
            print(f"  Training Time: {results['training_time_mins']:.2f} minutes")

        except Exception as e:
            print(f"\n✗ Error training {config_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - total_start_time

    print("\n" + "="*80)
    print(" "*20 + "TRAINING COMPLETE")
    print("="*80)
    print(f"\nTotal training time: {total_time/60:.2f} minutes")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Save results summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = project_root / "outputs" / "training" / "all_models_summary.csv"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f"\n✓ Results summary saved to: {results_path}")

        # Print summary table
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(results_df[['model_name', 'best_val_acc', 'training_time_mins', 'trainable_params']].to_string(index=False))

    print("\n✓ All models trained successfully!")
    print(f"\nNext step: Run 'python src/compare_models.py' to generate comparison report\n")


if __name__ == "__main__":
    main()
