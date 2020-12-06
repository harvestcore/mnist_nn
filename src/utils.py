from mnist import MNIST
import numpy as np
from datetime import datetime
import csv

# Normalize a dataset.
# Makes use of the normalization function from numpy library.
def normalize(dataset):
    normalized = []
    for array in dataset:
        normalized.append(array / np.linalg.norm(array))

    return normalized

# Load the datasets.
# Makes use of the MNIST library (https://pypi.org/project/python-mnist/).
def load_datasets():
    data = MNIST('./datasets')
    images, labels = data.load_training()
    return normalize(images), labels

def to_csv(filename, data):
    with open(filename, mode='a') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(data)

def get_filename(epochs, rate, layers, activation):
    filename = 'epochs-' + str(epochs)
    filename += '_rate-' + str(rate)
    filename += '_activation-' + activation
    filename += '_layers-' + ','.join(map(str, layers))
    filename += '_' + str(datetime.now())
    filename += '.csv'

    return filename
