from datetime import datetime

from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from neural_network import NeuralNetwork
from activation_functions import ACTIVATION_FUNCTIONS
from utils import to_csv, get_filename

x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
x = (x/255).astype('float32')
y = to_categorical(y)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)

def main():
    layers = [784, 128, 64, 10]
    rates = [0.001, 0.01, 0.1, 0.25, 0.5, 1]

    for activation in ACTIVATION_FUNCTIONS:
        for rate in rates:
            for epochs in range (10, 60, 10):
                print(f"[!] Starting training: Epochs: {epochs}, Activation: {activation}, Rate: {rate}")
                nn = NeuralNetwork(layers=layers, epochs=epochs, rate=rate, activation=activation)
                results = nn.train(x_train, y_train, x_val, y_val)
                filename = get_filename(epochs=epochs, rate=rate, layers=layers, activation=activation)
                print(f"[!] Writing results to: {filename}")
                to_csv('output/' + filename, results)   

if __name__ == "__main__":
    main()