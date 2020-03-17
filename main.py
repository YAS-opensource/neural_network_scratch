from model.model import Model
import numpy as np

if __name__ == ("__main__"):
    model = Model()
    data = [[1, 2, 7], [4, 5, 9], [3, 8, 6]]
    label = [0, 1, 3]
    data = np.array(data, dtype=np.float32)
    # mean = data.mean(axis=0)
    # std = data.std(axis=0)
    # data = data - mean
    # data = data/std
    print(data)
    model.add_layer(3, model.sigmoid)
    model.add_layer(4, model.sigmoid)
    model.train(data, label, 1, 0.001, 2)
    # print(model.layers)
