import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

from model import create_model
from data import create_dataloaders


class Evaluator:
    """Model evaluation pipeline"""

    def __init__(self, config, model_path=None):
        self.config = config
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
        self.project_root = Path(__file__).parent.parent

        # Create output directory
        self.eval_dir = self.project_root / 'outputs' / 'evaluation'
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        self.model = create_model(config)

        if model_path is None:
            model_path = self.project_root / config['paths']['best_model']

        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load data
        _, _, self.test_loader = create_dataloaders(config)

        # Class names
        self.class_names = list(config['classes'].values())

        print(f"Using device: {self.device}")
        print(f"Test samples: {len(self.test_loader.dataset)}")

    def predict(self):
        """Get predictions on test set"""
        all_preds = []
        all_labels = []
        all_probs = []

        print("\nGenerating predictions...")
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader):
                images = images.to(self.device)

                # Forward pass
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_labels), np.array(all_preds), np.array(all_probs)

    def compute_metrics(self, y_true, y_pred):
        """Compute evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }

        return metrics

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        save_path = self.eval_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Confusion matrix saved to {save_path}")

        # Also plot normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Percentage'})
        plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        save_path = self.eval_dir / 'confusion_matrix_normalized.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Normalized confusion matrix saved to {save_path}")

    def plot_roc_curve(self, y_true, y_probs):
        """Plot ROC curve for binary classification"""
        if len(self.class_names) == 2:
            # Binary classification
            y_probs_positive = y_probs[:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_probs_positive)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                    label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('Receiver Operating Characteristic (ROC) Curve',
                     fontsize=16, fontweight='bold')
            plt.legend(loc="lower right", fontsize=12)
            plt.grid(alpha=0.3)
            plt.tight_layout()

            save_path = self.eval_dir / 'roc_curve.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"ROC curve saved to {save_path}")

    def plot_per_class_metrics(self, y_true, y_pred):
        """Plot per-class precision, recall, F1"""
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)

        x = np.arange(len(self.class_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision, width, label='Precision', color='#3498db')
        ax.bar(x, recall, width, label='Recall', color='#e74c3c')
        ax.bar(x + width, f1, width, label='F1-Score', color='#2ecc71')

        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Metrics', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])

        # Add value labels on bars
        for i, v in enumerate(precision):
            ax.text(i - width, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        for i, v in enumerate(recall):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        for i, v in enumerate(f1):
            ax.text(i + width, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_path = self.eval_dir / 'per_class_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Per-class metrics plot saved to {save_path}")

    def save_classification_report(self, y_true, y_pred):
        """Save classification report"""
        report = classification_report(y_true, y_pred,
                                      target_names=self.class_names,
                                      digits=4)

        report_path = self.eval_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 70 + "\n\n")
            f.write(report)

        print(f"Classification report saved to {report_path}")
        print("\n" + report)

    def evaluate(self):
        """Run full evaluation"""
        print("\n" + "=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)

        # Get predictions
        y_true, y_pred, y_probs = self.predict()

        # Compute metrics
        print("\nComputing metrics...")
        metrics = self.compute_metrics(y_true, y_pred)

        print("\nOverall Metrics:")
        print("-" * 70)
        for metric_name, value in metrics.items():
            print(f"{metric_name.capitalize():15s}: {value:.4f} ({value*100:.2f}%)")

        # Generate plots
        print("\nGenerating visualizations...")
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_roc_curve(y_true, y_probs)
        self.plot_per_class_metrics(y_true, y_pred)

        # Save classification report
        self.save_classification_report(y_true, y_pred)

        # Save metrics to file
        metrics_path = self.eval_dir / 'metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write("Evaluation Metrics\n")
            f.write("=" * 70 + "\n\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name.capitalize():15s}: {value:.4f} ({value*100:.2f}%)\n")

        print(f"\nMetrics saved to {metrics_path}")

        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE!")
        print(f"All results saved to {self.eval_dir}")
        print("=" * 70)

        return metrics


def main():
    """Main evaluation function"""
    # Load config
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config/config.yaml"

    print("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create evaluator and evaluate
    evaluator = Evaluator(config)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
