import os
import sys
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import CovidDataLoader
from models.ann import build_ann_rgb, build_ann_grayscale, build_ann_blur, build_ann_laplacian
from evaluate import model_performance_classification, plot_confusion_matrix, plot_training_history


def train_model(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    tf.keras.utils.set_random_seed(config['seed'])
    tf.config.experimental.enable_op_determinism()

    print(f"Loading data for {config['model']['name']}...")
    data_loader = CovidDataLoader(
        images_path=config['data']['images_path'],
        labels_path=config['data']['labels_path'],
        test_size=config['data']['test_split'] + config['data']['val_split'],
        val_size=config['data']['val_split'] / (config['data']['test_split'] + config['data']['val_split']),
        random_state=config['seed']
    )

    datasets, labels = data_loader.prepare_all_datasets()
    preprocessing_type = config['model']['preprocessing_type']
    X_train, X_val, X_test, y_train, y_val, y_test = datasets[preprocessing_type]

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")

    num_classes = labels['Label'].nunique()
    input_shape = X_train.shape[1:]

    print(f"\nBuilding {config['model']['name']}...")
    if preprocessing_type == 'rgb':
        model = build_ann_rgb(input_shape, num_classes)
    elif preprocessing_type == 'grayscale':
        model = build_ann_grayscale(input_shape, num_classes)
    elif preprocessing_type == 'blur':
        model = build_ann_blur(input_shape, num_classes)
    elif preprocessing_type == 'laplacian':
        model = build_ann_laplacian(input_shape, num_classes)

    adam = optimizers.Adam()
    model.compile(loss=losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy'])

    print("\nModel Summary:")
    model.summary()

    print("\nTraining model...")
    train_datagen = ImageDataGenerator()
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=config['training']['batch_size']),
        validation_data=(X_val, y_val),
        epochs=config['training']['epochs'],
        verbose=2
    )

    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs/training/plots', exist_ok=True)
    os.makedirs('outputs/evaluation', exist_ok=True)

    model_path = f"models/{config['model']['name']}_best.h5"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    print("\nEvaluating on training set...")
    train_perf = model_performance_classification(model, X_train, y_train)
    print(train_perf)

    print("\nEvaluating on validation set...")
    val_perf = model_performance_classification(model, X_val, y_val)
    print(val_perf)

    print("\nEvaluating on test set...")
    test_perf = model_performance_classification(model, X_test, y_test)
    print(test_perf)

    plot_path = f"outputs/training/plots/{config['model']['name']}_history.png"
    plot_training_history(history, save_path=plot_path)
    print(f"Training history plot saved to {plot_path}")

    cm_path = f"outputs/evaluation/{config['model']['name']}_confusion_matrix.png"
    plot_confusion_matrix(model, X_test, y_test, save_path=cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    results = {
        'train_performance': train_perf.to_dict('records')[0],
        'val_performance': val_perf.to_dict('records')[0],
        'test_performance': test_perf.to_dict('records')[0]
    }

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train COVID-19 X-ray classification model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    train_model(args.config)
