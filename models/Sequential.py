import numpy as np
import h5py


class Sequential():

    layers = []
    loss = ""
    metrics = []

    def __init__(self):
        pass

    def add(self, layer):
        self.layers.append(layer)
        return

    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        return

    def evaluate(self, x, y, batch_size):
        pass

    def fit(self, x, y, epochs, batch_size, validation_split):
        pass

    def predict(self, x):
        pass

    def save(self, path_name):
        pass