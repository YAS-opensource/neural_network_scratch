from model.model import Model

if __name__ == ("__main__"):
    model = Model()
    model.add_layer(3, model.sigmoid)
    model.train([[1], [2], [3]])