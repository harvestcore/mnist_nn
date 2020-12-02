from mnist import MNIST
import numpy as np

# Load the datasets.
# Makes use of the MNIST library (https://pypi.org/project/python-mnist/).
def load_datasets():
    data = MNIST('./datasets')
    images, labels = data.load_training()
    return images, labels

# Normalize a dataset.
# Makes use of the normalization function from numpy library.
def normalize(dataset):
    normalized = []
    for array in dataset:
        normalized.append(array / np.linalg.norm(array))

    return normalized

