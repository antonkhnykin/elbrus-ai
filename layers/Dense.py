import numpy as np


class Dense:

    def __init__(self, units, input_shape, activation=None):
        self.units = units
        self.a_function = activation
        self.input_shape = input_shape
        self.weights = np.random.rand(units, input_shape)
        self.biases = np.random.rand(units, 1)
        self.activations = np.random.rand(units, 1)
        self.outputs = np.random.rand(units, 1)

        if self.units < 0:
            raise ValueError(
                "Received an invalid value for `units`, expected "
                f"a positive integer. Received: units={units}"
            )
