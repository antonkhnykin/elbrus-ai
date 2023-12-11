import numpy as np


class Dense:

    units = 0
    activation = ""
    input_shape = 0
    W = np.array(0.0)
    b = np.array(0.0)
    outputs = np.array(0.0)

    def __init__(self, units, input_shape, activation=None):
        self.units = units
        self.activation = activation
        self.input_shape = input_shape
        self.W = np.random.rand(units, input_shape)
        self.b = np.random.rand(units, 1)
        self.outputs = np.random.rand(units, 1)

        if self.units < 0:
            raise ValueError(
                "Received an invalid value for `units`, expected "
                f"a positive integer. Received: units={units}"
            )