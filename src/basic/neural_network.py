import numpy as np
import time

import activation_functions as af
from activation_functions import ACTIVATION_FUNCTIONS

class NeuralNetwork:
    def __init__(self, layers, epochs, rate, activation='sigmoid'):
        self.layers = layers
        self.noof_layers = len(layers) - 1
        self.input_layer = layers[0]
        self.epochs = epochs
        self.rate = rate
        self.activation = getattr(af, activation) if activation in ACTIVATION_FUNCTIONS else getattr(af, 'sigmoid')
        self.weights = self.generate_weights()

    def generate_weights(self):
        weights = {}

        current_layer = self.input_layer
        for i in range(self.noof_layers):
            index = i + 1
            key = 'W' + str(index)
            weights[key] = np.random.randn(self.layers[index], current_layer) * np.sqrt(1. / self.layers[index])
            current_layer = self.layers[index]

        return weights

    def back_propagation(self, values, output):
        weights = self.weights
        changes = {}

        # Last layer
        z_key = 'Z' + str(self.noof_layers)
        w_key = 'W' + str(self.noof_layers)
        a_key = 'A' + str(self.noof_layers - 1)

        # Latest layer error
        error = 2 * (output - values) / output.shape[0] * self.activation(weights[z_key])
        changes[w_key] = np.outer(error, weights[a_key])

        # Previous layers
        for i in range(self.noof_layers, 1, -1):
            w_key = 'W' + str(i)
            w_previous_key = 'W' + str(i - 1)
            z_key = 'Z' + str(i - 1)
            a_key = 'A' + str(i - 2)

            if i - 2 > 0:
                error = np.dot(weights[w_key].T, error) * self.activation(weights[z_key])
                changes[w_previous_key] = np.outer(error, weights[a_key])

        self.update(changes)

    def forward_propagation(self, values):
        weights = self.weights

        # First layer values
        weights['A0'] = values

        for i in range(1, self.noof_layers + 1, 1):
            keyLastA = 'A' + str(i - 1)
            keyW = 'W' + str(i)
            keyA = 'A' + str(i)
            keyZ = 'Z' + str(i)

            weights[keyZ] = np.dot(weights[keyW], weights[keyLastA])
            weights[keyA] = self.activation(weights[keyZ])

        return weights['A' + str(self.noof_layers)]

    def update(self, changes):       
        for key, value in changes.items():
            self.weights[key] -= self.rate * value

    def accuracy(self, x, y):
        acc = []

        for _x, _y in zip(x, y):
            output = self.forward_propagation(_x)
            pred = np.argmax(output)
            acc.append(pred == np.argmax(_y))
        
        return np.mean(acc) * 100


    def train(self, x_train, y_train, x, y):
        results = []
        start_time = time.time()
        for epoch in range(1, self.epochs + 1, 1):
            for _x, _y in zip(x_train, y_train):
                output = self.forward_propagation(_x)
                self.back_propagation(_y, output)
       
            results.append([
                epoch,                      # Epoch
                self.accuracy(x, y),        # Accuracy
                time.time() - start_time    # Time spent
            ])
            
        return results
