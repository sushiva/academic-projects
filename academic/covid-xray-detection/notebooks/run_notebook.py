import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras import losses, optimizers, Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

tf.keras.utils.set_random_seed(812)
tf.config.experimental.enable_op_determinism()

print("="*80)
print("LOADING DATA")
print("="*80)

rgb_images = np.load('../data/raw/CovidImages-1.npy')
labels = pd.read_csv('../data/raw/CovidLabels-1.csv')

print(f"Images shape: {rgb_images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"\nClass distribution:\n{labels['Label'].value_counts()}")

print("\n" + "="*80)
print("PREPROCESSING")
print("="*80)

gray_images = []
for i in range(len(rgb_images)):
    gray_images.append(cv2.cvtColor(rgb_images[i], cv2.COLOR_RGB2GRAY))

gaus_blur_images = []
for i in range(len(rgb_images)):
    gaus_blur_images.append(cv2.GaussianBlur(rgb_images[i], (3,3), 0))

edge_images = []
for i in range(len(gray_images)):
    edge_images.append(cv2.Laplacian(gray_images[i],cv2.CV_64F))

print("Preprocessing complete")

print("\n" + "="*80)
print("SPLITTING DATA")
print("="*80)

X_train_rgb, X_temp_rgb, y_train_rgb, y_temp_rgb = train_test_split(
    np.array(rgb_images), labels, test_size=0.2, random_state=42, stratify=labels
)
X_val_rgb, X_test_rgb, y_val_rgb, y_test_rgb = train_test_split(
    X_temp_rgb, y_temp_rgb, test_size=0.5, random_state=42, stratify=y_temp_rgb
)

X_train_gray, X_temp_gray, y_train, y_temp = train_test_split(
    np.array(gray_images), labels, test_size=0.2, random_state=42, stratify=labels
)
X_val_gray, X_test_gray, y_val, y_test = train_test_split(
    X_temp_gray, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

X_train_blur, X_temp_blur, y_train_blur, y_temp_blur = train_test_split(
    np.array(gaus_blur_images), labels, test_size=0.2, random_state=42, stratify=labels
)
X_val_blur, X_test_blur, y_val_blur, y_test_blur = train_test_split(
    X_temp_blur, y_temp_blur, test_size=0.5, random_state=42, stratify=y_temp_blur
)

X_train_edge, X_temp_edge, y_train_edge, y_temp_edge = train_test_split(
    np.array(edge_images), labels, test_size=0.2, random_state=42, stratify=labels
)
X_val_edge, X_test_edge, y_val_edge, y_test_edge = train_test_split(
    X_temp_edge, y_temp_edge, test_size=0.5, random_state=42, stratify=y_temp_edge
)

print(f"RGB - Train: {X_train_rgb.shape}, Val: {X_val_rgb.shape}, Test: {X_test_rgb.shape}")
print(f"Gray - Train: {X_train_gray.shape}, Val: {X_val_gray.shape}, Test: {X_test_gray.shape}")

enc = LabelBinarizer()
y_train_encoded = enc.fit_transform(y_train_rgb)
y_val_encoded = enc.transform(y_val_rgb)
y_test_encoded = enc.transform(y_test_rgb)

X_train_rgb = X_train_rgb.astype('float32')/255.0
X_val_rgb = X_val_rgb.astype('float32')/255.0
X_test_rgb = X_test_rgb.astype('float32')/255.0

X_train_gray = np.expand_dims(X_train_gray.astype('float32')/255.0, axis=-1)
X_val_gray = np.expand_dims(X_val_gray.astype('float32')/255.0, axis=-1)
X_test_gray = np.expand_dims(X_test_gray.astype('float32')/255.0, axis=-1)

X_train_blur = X_train_blur.astype('float32')/255.0
X_val_blur = X_val_blur.astype('float32')/255.0
X_test_blur = X_test_blur.astype('float32')/255.0

X_train_edge = np.expand_dims(X_train_edge.astype('float32')/255.0, axis=-1)
X_val_edge = np.expand_dims(X_val_edge.astype('float32')/255.0, axis=-1)
X_test_edge = np.expand_dims(X_test_edge.astype('float32')/255.0, axis=-1)

print("Data normalized and encoded")

def model_performance_classification(model, predictors, target):
    pred = model.predict(predictors, verbose=0).argmax(axis=1)
    target = target.argmax(axis=1)
    acc = accuracy_score(target, pred)
    recall = recall_score(target, pred, average='weighted')
    precision = precision_score(target, pred, average='weighted', zero_division=0)
    f1 = f1_score(target, pred, average='weighted')
    df_perf = pd.DataFrame({"Accuracy": acc, "Recall": recall, "Precision": precision, "F1 Score": f1}, index=[0])
    return df_perf

results = {}

print("\n" + "="*80)
print("MODEL 1: ANN WITH RGB IMAGES")
print("="*80)

num_classes = labels['Label'].nunique()
image_size = X_train_rgb[0].size
shape = X_train_rgb.shape[1:]

model_1 = Sequential()
model_1.add(Input(shape=(shape[0],shape[1],shape[2])))
model_1.add(Flatten())
model_1.add(Dense(20, activation='relu',kernel_initializer='he_uniform',input_shape=(image_size,)))
model_1.add(Dense(10, activation='relu',kernel_initializer='he_uniform'))
model_1.add(Dense(5, activation='relu',kernel_initializer='he_uniform'))
model_1.add(Dense(num_classes, activation='softmax'))

adam = optimizers.Adam()
model_1.compile(loss=losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy'])

train_datagen = ImageDataGenerator()
history_1 = model_1.fit(
    train_datagen.flow(X_train_rgb, y_train_encoded, batch_size=128),
    validation_data=(X_val_rgb, y_val_encoded),
    epochs=15,
    verbose=0
)

model_1_test_perf = model_performance_classification(model_1, X_test_rgb, y_test_encoded)
print(f"Model 1 Test Performance:\n{model_1_test_perf}")
results['Model 1 RGB'] = model_1_test_perf

print("\n" + "="*80)
print("MODEL 2: ANN WITH GRAYSCALE IMAGES")
print("="*80)

shape = X_train_gray.shape[1:]
model_2 = Sequential()
model_2.add(Input(shape=(shape[0],shape[1],shape[2])))
model_2.add(Flatten())
model_2.add(Dense(50, activation='relu',kernel_initializer='he_uniform'))
model_2.add(Dense(20, activation='relu',kernel_initializer='he_uniform'))
model_2.add(Dense(10, activation='relu',kernel_initializer='he_uniform'))
model_2.add(Dense(5, activation='relu',kernel_initializer='he_uniform'))
model_2.add(Dense(num_classes, activation='softmax'))

adam = optimizers.Adam()
model_2.compile(loss=losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy'])

train_datagen = ImageDataGenerator()
history_2 = model_2.fit(
    train_datagen.flow(X_train_gray, y_train_encoded, batch_size=128),
    validation_data=(X_val_gray, y_val_encoded),
    epochs=10,
    verbose=0
)

model_2_test_perf = model_performance_classification(model_2, X_test_gray, y_test_encoded)
print(f"Model 2 Test Performance:\n{model_2_test_perf}")
results['Model 2 Grayscale'] = model_2_test_perf

print("\n" + "="*80)
print("MODEL 3: ANN WITH GAUSSIAN-BLURRED IMAGES")
print("="*80)

shape = X_train_blur.shape[1:]
model_3 = Sequential()
model_3.add(Input(shape=(shape[0],shape[1],shape[2])))
model_3.add(Flatten())
model_3.add(Dense(50, activation='relu',kernel_initializer='he_uniform'))
model_3.add(Dense(20, activation='relu',kernel_initializer='he_uniform'))
model_3.add(Dense(10, activation='relu',kernel_initializer='he_uniform'))
model_3.add(Dense(5, activation='relu',kernel_initializer='he_uniform'))
model_3.add(Dense(num_classes, activation='softmax'))

adam = optimizers.Adam()
model_3.compile(loss=losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy'])

train_datagen = ImageDataGenerator()
history_3 = model_3.fit(
    train_datagen.flow(X_train_blur, y_train_encoded, batch_size=128),
    validation_data=(X_val_blur, y_val_encoded),
    epochs=10,
    verbose=0
)

model_3_test_perf = model_performance_classification(model_3, X_test_blur, y_test_encoded)
print(f"Model 3 Test Performance:\n{model_3_test_perf}")
results['Model 3 Blur'] = model_3_test_perf

print("\n" + "="*80)
print("MODEL 4: ANN WITH LAPLACIAN-FILTERED IMAGES")
print("="*80)

shape = X_train_edge.shape[1:]
model_4 = Sequential()
model_4.add(Input(shape=(shape[0],shape[1],shape[2])))
model_4.add(Flatten())
model_4.add(Dense(50, activation='relu',kernel_initializer='he_uniform'))
model_4.add(Dense(20, activation='relu',kernel_initializer='he_uniform'))
model_4.add(Dense(10, activation='relu',kernel_initializer='he_uniform'))
model_4.add(Dense(5, activation='relu',kernel_initializer='he_uniform'))
model_4.add(Dense(num_classes, activation='softmax'))

adam = optimizers.Adam()
model_4.compile(loss=losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy'])

train_datagen = ImageDataGenerator()
history_4 = model_4.fit(
    train_datagen.flow(X_train_edge, y_train_encoded, batch_size=128),
    validation_data=(X_val_edge, y_val_encoded),
    epochs=10,
    verbose=0
)

model_4_test_perf = model_performance_classification(model_4, X_test_edge, y_test_encoded)
print(f"Model 4 Test Performance:\n{model_4_test_perf}")
results['Model 4 Laplacian'] = model_4_test_perf

print("\n" + "="*80)
print("COMPARISON OF ALL MODELS")
print("="*80)

comparison_df = pd.concat([perf.T for perf in results.values()], axis=1)
comparison_df.columns = results.keys()
print(comparison_df)

print("\n" + "="*80)
print("BEST MODEL")
print("="*80)
best_model = comparison_df.loc['Accuracy'].idxmax()
best_accuracy = comparison_df.loc['Accuracy'].max()
print(f"Best performing model: {best_model}")
print(f"Test Accuracy: {best_accuracy:.4f}")
