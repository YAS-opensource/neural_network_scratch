import numpy as np
from math import exp


class Model(object):
    def __init__(self):
        self.layers = []
        self.bias = []
        self.weights = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self):
        pass

    def _differentiation(self, func, x):
        if func == self.sigmoid:
            return x * (1 - x)

    def add_layer(self, neurons, func):
        layer = np.random.rand(neurons, 1)
        bias = np.random.rand(neurons, 1)
        bias_size = len(self.bias)
        if bias_size > 0:
            m = self.bias[bias_size - 1].shape[0]
            weight = np.random.rand(neurons, m)
            self.weights.append(weight)
        self.layers.append([layer, func])
        self.bias.append(bias)

    def _error_calculation(self, data, label):
        total = data-label
        total = total.sum(axis=1)
        # total = total/total.shape[0]
        return total

    def _forward_feed(self, data, label, batch_size):
        steps = data.shape[1] // batch_size + 1
        error = 0
        for i in range(steps):
            train_layers = []
            data_matrix = data[
                np.ix_(
                    list(range(data.shape[0])),
                    list(
                        range(i * batch_size, min((i + 1) * batch_size, data.shape[1]))
                    ),
                )
            ]
            if len(data_matrix) > 0:
                layers = len(self.bias)
                data_len = data_matrix.shape[1]
                current_layer = data_matrix
                train_layers.append(data_matrix)
                for j in range(layers-1):
                    hidden_layer = self.layers[j][1](np.dot(self.weights[j], current_layer)+np.tile(self.bias[j+1], data_len))
                    train_layers.append(hidden_layer)
                    current_layer = hidden_layer
                result = train_layers[-1]
                y = np.zeros(result.shape)
                rows = label[i * batch_size: min((i + 1) * batch_size, data.shape[1])]
                y[rows, list(range(result.shape[1]))] = 1.0
                error = self._error_calculation(result, y)
                raw_error = np.dot(np.transpose(error), error)
                print(raw_error)

    def train(self, data, label, epoch=10, learn_rate=0.001, batch_size=2):
        data = np.transpose(data)

        for i in range(epoch):
            self._forward_feed(data, label, batch_size)
