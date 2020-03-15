import numpy as np
from math import exp

class Model(object):
    def __init__(self):
        self.layers = []
        self.bias = []

    def sigmoid(self, x):
        return 1/(1+exp(-x))

    def softmax(self):
        pass

    def _differentiation(self, func, x):
        if func == self.sigmoid:
            return exp(x)/(1+exp(x))**2

    def add_layer(self, neurons, func):
        layer = np.random.rand(neurons, 1)
        bias = np.random.rand(neurons, 1)
        self.layers.append([layer, func])
        self.bias.append(bias)

    def train(self, data, epoch=10, learn_rate=0.001, batch_size=2):
        data = np.array(data, dtype=np.float32)
        if data.shape != self.layers[0][0].shape:
            raise ValueError("Shape does not match")
        self.layers[0][0] = np.copy(data)