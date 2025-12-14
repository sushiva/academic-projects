import os
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import CovidDataLoader
from evaluate import model_performance_classification


def load_trained_model(model_name):
    model_path = f"models/{model_name}_best.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(model_path)


def compare_models():
    print("Loading datasets...")
    data_loader = CovidDataLoader(
        images_path="data/raw/CovidImages-1.npy",
        labels_path="data/raw/CovidLabels-1.csv",
        test_size=0.3,
        val_size=0.5,
        random_state=812
    )

    datasets, labels = data_loader.prepare_all_datasets()

    models_info = [
        {
            'name': 'model1_ann_rgb',
            'display_name': 'ANN with RGB Images',
            'preprocessing': 'rgb'
        },
        {
            'name': 'model2_ann_grayscale',
            'display_name': 'ANN with Grayscale Images',
            'preprocessing': 'grayscale'
        },
        {
            'name': 'model3_ann_blur',
            'display_name': 'ANN with Gaussian-blurred Images',
            'preprocessing': 'blur'
        },
        {
            'name': 'model4_ann_laplacian',
            'display_name': 'ANN with Laplacian-Filtered Images',
            'preprocessing': 'laplacian'
        }
    ]

    train_performances = []
    val_performances = []
    test_performances = []
    model_names = []

    for model_info in models_info:
        print(f"\nEvaluating {model_info['display_name']}...")

        try:
            model = load_trained_model(model_info['name'])
            X_train, X_val, X_test, y_train, y_val, y_test = datasets[model_info['preprocessing']]

            train_perf = model_performance_classification(model, X_train, y_train)
            val_perf = model_performance_classification(model, X_val, y_val)
            test_perf = model_performance_classification(model, X_test, y_test)

            train_performances.append(train_perf)
            val_performances.append(val_perf)
            test_performances.append(test_perf)
            model_names.append(model_info['display_name'])

            print(f"Train Accuracy: {train_perf['Accuracy'].values[0]:.4f}")
            print(f"Val Accuracy: {val_perf['Accuracy'].values[0]:.4f}")
            print(f"Test Accuracy: {test_perf['Accuracy'].values[0]:.4f}")

        except FileNotFoundError as e:
            print(f"Skipping {model_info['name']}: {e}")
            continue

    if not train_performances:
        print("No models found for comparison!")
        return

    train_comp_df = pd.concat([perf.T for perf in train_performances], axis=1)
    train_comp_df.columns = model_names

    val_comp_df = pd.concat([perf.T for perf in val_performances], axis=1)
    val_comp_df.columns = model_names

    test_comp_df = pd.concat([perf.T for perf in test_performances], axis=1)
    test_comp_df.columns = model_names

    os.makedirs('outputs/comparison', exist_ok=True)

    print("\n" + "="*80)
    print("TRAINING PERFORMANCE COMPARISON")
    print("="*80)
    print(train_comp_df)

    print("\n" + "="*80)
    print("VALIDATION PERFORMANCE COMPARISON")
    print("="*80)
    print(val_comp_df)

    print("\n" + "="*80)
    print("TEST PERFORMANCE COMPARISON")
    print("="*80)
    print(test_comp_df)

    print("\n" + "="*80)
    print("TRAIN vs VALIDATION DIFFERENCE")
    print("="*80)
    diff_df = train_comp_df - val_comp_df
    print(diff_df)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        data = pd.DataFrame({
            'Model': model_names * 3,
            'Split': ['Train'] * len(model_names) + ['Validation'] * len(model_names) + ['Test'] * len(model_names),
            metric: list(train_comp_df.loc[metric]) + list(val_comp_df.loc[metric]) + list(test_comp_df.loc[metric])
        })

        sns.barplot(x='Model', y=metric, hue='Split', data=data, ax=ax)
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(metric, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Split')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/comparison/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved to outputs/comparison/model_comparison.png")

    with open('outputs/comparison/comparison_report.md', 'w') as f:
        f.write("# COVID-19 X-ray Classification - Model Comparison Report\n\n")
        f.write("## Training Performance\n\n")
        f.write(train_comp_df.to_markdown())
        f.write("\n\n## Validation Performance\n\n")
        f.write(val_comp_df.to_markdown())
        f.write("\n\n## Test Performance\n\n")
        f.write(test_comp_df.to_markdown())
        f.write("\n\n## Train vs Validation Difference\n\n")
        f.write(diff_df.to_markdown())
        f.write("\n\n## Key Findings\n\n")

        best_val_model = val_comp_df.loc['Accuracy'].idxmax()
        best_val_acc = val_comp_df.loc['Accuracy'].max()
        best_test_model = test_comp_df.loc['Accuracy'].idxmax()
        best_test_acc = test_comp_df.loc['Accuracy'].max()

        f.write(f"- Best validation accuracy: **{best_val_model}** ({best_val_acc:.4f})\n")
        f.write(f"- Best test accuracy: **{best_test_model}** ({best_test_acc:.4f})\n")
        f.write("\n## Conclusion\n\n")
        f.write("The comparison shows which preprocessing technique works best for COVID-19 X-ray classification.\n")
        f.write("Medical X-ray images are naturally grayscale, so grayscale preprocessing is expected to perform well.\n")

    print("Comparison report saved to outputs/comparison/comparison_report.md")


if __name__ == '__main__':
    compare_models()
