import numpy as np
from math import sqrt, tanh, pi

ACTIVATION_FUNCTIONS = ['sigmoid', 'sigmoid_derivative', 'relu', 'elu', 'lrelu', 'softmax', 'gelu']

def sigmoid(value):
    return 1 / (1 + np.exp(-value))

def sigmoid_derivative(value):
    return (np.exp(-value))/((np.exp(-value)+1)**2)

def relu(value):
    return max(0, value)

def elu(value, alpha=0.2):
    if value > 0:
        return value
    
    return alpha * (np.exp(value) - 1)

def lrelu(value, alpha=0.2):
    if value > 0:
        return value

    return alpha * value

def softmax(values):
    xp = np.exp(values - max(values))
    return xp / sum(xp)

def gelu(value):
    return 0.5 * value * (1 + tanh(sqrt(2/pi) * (value + 0.044715 * pow(value, 3))))
