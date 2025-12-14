"""
Model Comparison Script

Loads all trained models, evaluates them on the test set,
and generates comprehensive comparison visualizations and report.
"""

import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Import model classes
from models import SimpleCNN, ResNet18Base, ResNet18Deep, ResNet18Augmented
from data import create_dataloaders

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)


def load_model(config_path, model_path):
    """Load a trained model"""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_type = config['model']['type']

    # Create model
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

    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config, checkpoint


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model on test set"""

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'f1_score': f1_score(all_labels, all_preds, average='weighted'),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

    return metrics


def plot_comparison_bar_chart(results_df, output_dir):
    """Plot comparison bar chart of all models"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Accuracy comparison
    axes[0, 0].bar(range(len(results_df)), results_df['accuracy'] * 100, color='steelblue', alpha=0.8)
    axes[0, 0].set_xticks(range(len(results_df)))
    axes[0, 0].set_xticklabels(results_df['model_name'], rotation=45, ha='right')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylim([0, 105])
    for i, v in enumerate(results_df['accuracy'] * 100):
        axes[0, 0].text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # F1-Score comparison
    axes[0, 1].bar(range(len(results_df)), results_df['f1_score'] * 100, color='coral', alpha=0.8)
    axes[0, 1].set_xticks(range(len(results_df)))
    axes[0, 1].set_xticklabels(results_df['model_name'], rotation=45, ha='right')
    axes[0, 1].set_ylabel('F1-Score (%)')
    axes[0, 1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylim([0, 105])
    for i, v in enumerate(results_df['f1_score'] * 100):
        axes[0, 1].text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Training time comparison
    axes[1, 0].bar(range(len(results_df)), results_df['training_time_mins'], color='lightgreen', alpha=0.8)
    axes[1, 0].set_xticks(range(len(results_df)))
    axes[1, 0].set_xticklabels(results_df['model_name'], rotation=45, ha='right')
    axes[1, 0].set_ylabel('Training Time (minutes)')
    axes[1, 0].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    for i, v in enumerate(results_df['training_time_mins']):
        axes[1, 0].text(i, v + 0.5, f'{v:.1f} min', ha='center', va='bottom', fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Trainable parameters comparison
    axes[1, 1].bar(range(len(results_df)), results_df['trainable_params'] / 1e6, color='plum', alpha=0.8)
    axes[1, 1].set_xticks(range(len(results_df)))
    axes[1, 1].set_xticklabels(results_df['model_name'], rotation=45, ha='right')
    axes[1, 1].set_ylabel('Trainable Parameters (Millions)')
    axes[1, 1].set_title('Model Complexity Comparison', fontsize=14, fontweight='bold')
    for i, v in enumerate(results_df['trainable_params'] / 1e6):
        axes[1, 1].text(i, v + 0.1, f'{v:.2f}M', ha='center', va='bottom', fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_bars.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(all_metrics, model_names, output_dir):
    """Plot confusion matrices for all models in a grid"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    class_names = ['Without Helmet', 'With Helmet']

    for idx, (metrics, model_name) in enumerate(zip(all_metrics, model_names)):
        cm = metrics['confusion_matrix']

        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=True, square=True, linewidths=1, linecolor='black')

        axes[idx].set_title(f'{model_name}\nAccuracy: {metrics["accuracy"]*100:.2f}%',
                           fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices_grid.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_progressive_improvement(results_df, output_dir):
    """Plot progressive improvement across models"""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Line plot of accuracy progression
    axes[0].plot(range(len(results_df)), results_df['accuracy'] * 100,
                marker='o', markersize=12, linewidth=3, color='steelblue', label='Test Accuracy')
    axes[0].fill_between(range(len(results_df)), results_df['accuracy'] * 100, alpha=0.3)
    axes[0].set_xticks(range(len(results_df)))
    axes[0].set_xticklabels(results_df['model_name'], rotation=45, ha='right')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Progressive Improvement: Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 105])
    axes[0].grid(True, alpha=0.3)

    # Annotate improvements
    for i in range(1, len(results_df)):
        improvement = (results_df.iloc[i]['accuracy'] - results_df.iloc[i-1]['accuracy']) * 100
        mid_x = (i-1 + i) / 2
        mid_y = (results_df.iloc[i]['accuracy'] + results_df.iloc[i-1]['accuracy']) * 50
        axes[0].annotate(f'+{improvement:.1f}%',
                        xy=(mid_x, mid_y),
                        fontsize=10,
                        ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # Multi-metric comparison
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(results_df))
    width = 0.2

    for idx, metric in enumerate(metrics_to_plot):
        offset = (idx - len(metrics_to_plot)/2 + 0.5) * width
        axes[1].bar(x + offset, results_df[metric] * 100, width,
                   label=metric.replace('_', ' ').title(), alpha=0.8)

    axes[1].set_xlabel('Model', fontsize=12)
    axes[1].set_ylabel('Score (%)', fontsize=12)
    axes[1].set_title('Multi-Metric Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(results_df['model_name'], rotation=45, ha='right')
    axes[1].legend()
    axes[1].set_ylim([0, 105])
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'progressive_improvement.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_markdown_report(results_df, all_metrics, model_names, output_dir):
    """Generate comprehensive markdown report"""

    report_path = output_dir / 'comparison_report.md'

    with open(report_path, 'w') as f:
        f.write("# Multi-Model Comparison Report\n\n")
        f.write("Comprehensive comparison of 4 models for safety helmet detection.\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        best_model_idx = results_df['accuracy'].idxmax()
        best_model = results_df.iloc[best_model_idx]

        f.write(f"**Best Performing Model:** {best_model['model_name']}\n")
        f.write(f"- **Test Accuracy:** {best_model['accuracy']*100:.2f}%\n")
        f.write(f"- **F1-Score:** {best_model['f1_score']*100:.2f}%\n")
        f.write(f"- **Training Time:** {best_model['training_time_mins']:.2f} minutes\n")
        f.write(f"- **Trainable Parameters:** {best_model['trainable_params']/1e6:.2f}M\n\n")

        # Overall comparison
        f.write("---\n\n")
        f.write("## Model Comparison Table\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Parameters |\n")
        f.write("|-------|----------|-----------|--------|----------|---------------|------------|\n")

        for _, row in results_df.iterrows():
            f.write(f"| {row['model_name']} | "
                   f"{row['accuracy']*100:.2f}% | "
                   f"{row['precision']*100:.2f}% | "
                   f"{row['recall']*100:.2f}% | "
                   f"{row['f1_score']*100:.2f}% | "
                   f"{row['training_time_mins']:.1f} min | "
                   f"{row['trainable_params']/1e6:.2f}M |\n")

        f.write("\n")

        # Progressive improvement
        f.write("---\n\n")
        f.write("## Progressive Improvement Analysis\n\n")
        f.write("| From Model | To Model | Accuracy Gain | Key Change |\n")
        f.write("|------------|----------|---------------|------------|\n")

        improvements = [
            ("Model 1", "Model 2", "Transfer Learning (frozen backbone)"),
            ("Model 2", "Model 3", "Deep Classifier (more capacity)"),
            ("Model 3", "Model 4", "Heavy Augmentation + Fine-tuning")
        ]

        for i in range(1, len(results_df)):
            prev_acc = results_df.iloc[i-1]['accuracy'] * 100
            curr_acc = results_df.iloc[i]['accuracy'] * 100
            gain = curr_acc - prev_acc

            from_model, to_model, change = improvements[i-1]
            f.write(f"| {from_model} | {to_model} | +{gain:.2f}% | {change} |\n")

        f.write("\n")

        # Individual model details
        f.write("---\n\n")
        f.write("## Individual Model Analysis\n\n")

        for idx, row in results_df.iterrows():
            f.write(f"### {row['model_name']}\n\n")
            f.write(f"**Architecture:** {row['model_type']}\n\n")

            f.write("**Performance:**\n")
            f.write(f"- Accuracy: {row['accuracy']*100:.2f}%\n")
            f.write(f"- Precision: {row['precision']*100:.2f}%\n")
            f.write(f"- Recall: {row['recall']*100:.2f}%\n")
            f.write(f"- F1-Score: {row['f1_score']*100:.2f}%\n\n")

            f.write("**Training:**\n")
            f.write(f"- Training Time: {row['training_time_mins']:.2f} minutes\n")
            f.write(f"- Best Epoch: {row['best_epoch']}\n")
            f.write(f"- Epochs Trained: {row['epochs_trained']}\n\n")

            f.write("**Model Complexity:**\n")
            f.write(f"- Total Parameters: {row['total_params']:,}\n")
            f.write(f"- Trainable Parameters: {row['trainable_params']:,}\n\n")

            f.write("---\n\n")

        # Conclusions
        f.write("## Key Findings\n\n")
        f.write("1. **Transfer Learning Impact:** ")
        acc_gain_1_to_2 = (results_df.iloc[1]['accuracy'] - results_df.iloc[0]['accuracy']) * 100
        f.write(f"Using pre-trained ResNet18 improved accuracy by {acc_gain_1_to_2:.2f}% "
               f"over training from scratch.\n\n")

        f.write("2. **Classifier Capacity:** ")
        acc_gain_2_to_3 = (results_df.iloc[2]['accuracy'] - results_df.iloc[1]['accuracy']) * 100
        f.write(f"Adding a deep classifier increased accuracy by {acc_gain_2_to_3:.2f}%, "
               f"showing the importance of classifier capacity.\n\n")

        f.write("3. **Augmentation + Fine-tuning:** ")
        acc_gain_3_to_4 = (results_df.iloc[3]['accuracy'] - results_df.iloc[2]['accuracy']) * 100
        f.write(f"Heavy augmentation and fine-tuning the entire network added {acc_gain_3_to_4:.2f}% "
               f"improvement.\n\n")

        f.write("4. **Overall Improvement:** ")
        total_gain = (results_df.iloc[3]['accuracy'] - results_df.iloc[0]['accuracy']) * 100
        f.write(f"From baseline to best model: **+{total_gain:.2f}%** improvement.\n\n")

        f.write("---\n\n")
        f.write("## Visualizations\n\n")
        f.write("- [Model Comparison Bars](model_comparison_bars.png)\n")
        f.write("- [Confusion Matrices Grid](confusion_matrices_grid.png)\n")
        f.write("- [Progressive Improvement](progressive_improvement.png)\n\n")

        f.write("---\n\n")
        f.write(f"*Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    print(f"✓ Markdown report saved to: {report_path}")


def main():
    """Main comparison function"""

    project_root = Path(__file__).parent.parent
    config_dir = project_root / "config"
    models_dir = project_root / "models"
    output_dir = project_root / "outputs" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model configurations
    models_info = [
        ("model1_simple_cnn.yaml", "model1_simple_cnn_best.pth"),
        ("model2_resnet_base.yaml", "model2_resnet_base_best.pth"),
        ("model3_resnet_deep.yaml", "model3_resnet_deep_best.pth"),
        ("model4_resnet_augmented.yaml", "model4_resnet_augmented_best.pth")
    ]

    print("\n" + "="*80)
    print(" "*25 + "MODEL COMPARISON")
    print("="*80)

    all_results = []
    all_metrics = []
    model_names = []

    # Load training summary if available
    training_summary_path = project_root / "outputs" / "training" / "all_models_summary.csv"
    if training_summary_path.exists():
        training_df = pd.read_csv(training_summary_path)
    else:
        training_df = None

    for config_file, model_file in models_info:
        config_path = config_dir / config_file
        model_path = models_dir / model_file

        if not model_path.exists():
            print(f"\n⚠ Warning: Model file not found: {model_path}")
            continue

        print(f"\nEvaluating: {config_file}")

        # Load model
        model, config, checkpoint = load_model(config_path, model_path)
        model_name = config['model']['name']
        model_names.append(model_name)

        # Create dataloader (using same test set for all)
        _, _, test_loader = create_dataloaders(config)

        # Evaluate
        metrics = evaluate_model(model, test_loader)
        all_metrics.append(metrics)

        # Collect results
        result = {
            'model_name': model_name,
            'model_type': config['model']['type'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        }

        # Add training info if available
        if training_df is not None:
            training_row = training_df[training_df['model_name'] == model_name]
            if not training_row.empty:
                result.update({
                    'total_params': int(training_row['total_params'].values[0]),
                    'trainable_params': int(training_row['trainable_params'].values[0]),
                    'best_epoch': int(training_row['best_epoch'].values[0]),
                    'training_time_mins': float(training_row['training_time_mins'].values[0]),
                    'epochs_trained': int(training_row['epochs_trained'].values[0])
                })

        all_results.append(result)

        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  F1-Score: {metrics['f1_score']*100:.2f}%")

    # Create results dataframe
    results_df = pd.DataFrame(all_results)

    # Save results
    results_path = output_dir / "comparison_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Comparison results saved to: {results_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_comparison_bar_chart(results_df, output_dir)
    print("  ✓ Comparison bar charts")

    plot_confusion_matrices(all_metrics, model_names, output_dir)
    print("  ✓ Confusion matrices grid")

    plot_progressive_improvement(results_df, output_dir)
    print("  ✓ Progressive improvement charts")

    # Generate report
    print("\nGenerating comparison report...")
    generate_markdown_report(results_df, all_metrics, model_names, output_dir)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - comparison_results.csv")
    print(f"  - model_comparison_bars.png")
    print(f"  - confusion_matrices_grid.png")
    print(f"  - progressive_improvement.png")
    print(f"  - comparison_report.md")
    print("\n✓ All comparison outputs generated successfully!\n")


if __name__ == "__main__":
    main()
