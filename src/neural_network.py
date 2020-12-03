from math import sqrt
from random import randrange

import activation_functions as af
from activation_functions import ACTIVATION_FUNCTIONS

class NeuralNetwork:
    def __init__(self, layers, epochs, rate, activation='sigmoid'):
        self.layers = layers
        self.noof_layers = len(layers) - 1
        self.input_layer = layers[0]
        self.output_layer = layers[len(layers) - 1]
        self.hidden_layers = layers[1:len(layers) - 1]
        self.epochs = epochs
        self.rate = rate
        self.activation = getattr(af, activation) if activation in ACTIVATION_FUNCTIONS else getattr(af, 'sigmoid')
        self.weights = self.generate_weights()

    def generate_weights(self):
        weights = {}

        for i in range(self.noof_layers):
            index = i + 1
            key = 'w' + str(index)
            a = self.layers[index]
            b = self.input_layer

            if a > b:
                a = self.input_layer
                b = self.layers[index]
                
            weights[key] = randrange(a, b) * sqrt(1.0 / self.layers[index])

        return weights

    def forward_propagation(self):
        return 1

    def back_propagation(self):
        return 1
