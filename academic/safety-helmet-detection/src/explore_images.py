import numpy as np
import pandas as pd

# Load data
images = np.load('data/raw/images_proj.npy')
labels = pd.read_csv('data/raw/Labels_proj.csv')

print("Images shape:", images.shape)
print("\nLabels preview:")
print(labels.head(10))
print("\nLabels info:")
print(labels.info())
print("\nClass distribution:")
print(labels.value_counts())