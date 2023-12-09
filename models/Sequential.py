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

    def compile(self, optimizer='adam', loss='MSE', metrics=['accuracy']):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        return

    def evaluate(self, x, y, batch_size):
        pass

    def fit(self, x, y, epochs=1, batch_size=1, validation_split=0.1):
        pass

    def predict(self, x):
        pass

    def save(self, path_name):
        pass