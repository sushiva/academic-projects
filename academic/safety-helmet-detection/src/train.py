import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
from pathlib import Path
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model import create_model
from data import create_dataloaders


class Trainer:
    """Training pipeline for helmet detection"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')

        # Create directories
        self.project_root = Path(__file__).parent.parent
        self.setup_directories()

        # Model
        self.model = create_model(config)
        self.model = self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        # Learning rate scheduler
        if config['training']['scheduler']['enabled']:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=config['training']['scheduler']['patience'],
                factor=config['training']['scheduler']['factor']
            )
        else:
            self.scheduler = None

        # Data loaders
        print("\nLoading data...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0

        print(f"\nUsing device: {self.device}")

    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            self.project_root / self.config['paths']['models'],
            self.project_root / self.config['paths']['checkpoints'],
            self.project_root / self.config['paths']['logs'],
            self.project_root / self.config['paths']['plots']
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
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
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{running_loss / (pbar.n + 1):.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': self.history['val_acc'][-1],
            'history': self.history
        }

        # Save checkpoint
        checkpoint_path = self.project_root / self.config['paths']['checkpoints'] / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.project_root / self.config['paths']['best_model']
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def plot_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(self.history['val_loss'], label='Val Loss', marker='o')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Acc', marker='o')
        axes[1].plot(self.history['val_acc'], label='Val Acc', marker='o')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plot_path = self.project_root / self.config['paths']['plots'] / 'training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training history plot saved to {plot_path}")

    def train(self):
        """Main training loop"""
        num_epochs = self.config['training']['epochs']
        early_stopping = self.config['training']['early_stopping']

        print(f"\nStarting training for {num_epochs} epochs...")
        print("=" * 70)

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 70)

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Save best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                print(f"  New best model! Val Acc: {val_acc:.2f}%")
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.epochs_no_improve += 1

            # Early stopping
            if early_stopping['enabled']:
                if self.epochs_no_improve >= early_stopping['patience']:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    print(f"Best val acc: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
                    break

        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"Total time: {elapsed_time / 60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
        print("=" * 70)

        # Plot training history
        self.plot_history()


def main():
    """Main training function"""
    # Load config
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config/config.yaml"

    print("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create trainer and train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
