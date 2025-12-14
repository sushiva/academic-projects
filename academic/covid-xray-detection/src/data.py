import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


class CovidDataLoader:
    def __init__(self, images_path, labels_path, test_size=0.2, val_size=0.5, random_state=42):
        self.images_path = images_path
        self.labels_path = labels_path
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.label_encoder = LabelBinarizer()

    def load_data(self):
        rgb_images = np.load(self.images_path)
        labels = pd.read_csv(self.labels_path)
        return rgb_images, labels

    def preprocess_grayscale(self, rgb_images):
        gray_images = []
        for i in range(len(rgb_images)):
            gray_images.append(cv2.cvtColor(rgb_images[i], cv2.COLOR_RGB2GRAY))
        return np.array(gray_images)

    def preprocess_gaussian_blur(self, rgb_images):
        blur_images = []
        for i in range(len(rgb_images)):
            blur_images.append(cv2.GaussianBlur(rgb_images[i], (3, 3), 0))
        return np.array(blur_images)

    def preprocess_laplacian(self, gray_images):
        edge_images = []
        for i in range(len(gray_images)):
            edge_images.append(cv2.Laplacian(gray_images[i], cv2.CV_64F))
        return np.array(edge_images)

    def split_and_normalize(self, images, labels, preprocessing_type='rgb'):
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y_temp
        )

        if preprocessing_type in ['grayscale', 'laplacian']:
            X_train = np.expand_dims(X_train, axis=-1)
            X_val = np.expand_dims(X_val, axis=-1)
            X_test = np.expand_dims(X_test, axis=-1)

        X_train = X_train.astype('float32') / 255.0
        X_val = X_val.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)

        return X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded

    def prepare_all_datasets(self):
        rgb_images, labels = self.load_data()

        gray_images = self.preprocess_grayscale(rgb_images)
        blur_images = self.preprocess_gaussian_blur(rgb_images)
        edge_images = self.preprocess_laplacian(gray_images)

        datasets = {}

        datasets['rgb'] = self.split_and_normalize(
            np.array(rgb_images), labels, preprocessing_type='rgb'
        )

        datasets['grayscale'] = self.split_and_normalize(
            gray_images, labels, preprocessing_type='grayscale'
        )

        datasets['blur'] = self.split_and_normalize(
            blur_images, labels, preprocessing_type='rgb'
        )

        datasets['laplacian'] = self.split_and_normalize(
            edge_images, labels, preprocessing_type='laplacian'
        )

        return datasets, labels
