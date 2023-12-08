import numpy as np

class Dense():

    units = 0
    activation = ""
    input_shape = ()
    W = np.zeros(0, np.float64)
    b = np.zeros(0, np.float64)
    outputs = np.zeros(0, np.float64)

    def __init(self, units, input_shape, activation=None):
        self.units = units
        self.activation = activation
        self.input_shape = input_shape
        self.W = np.reshape(self.W, (input_shape[0], units))
        self.b = np.reshape(self.b, units)
        self.outputs = np.reshape(self.outputs, units)

        if self.units < 0:
            raise ValueError(
                "Received an invalid value for `units`, expected "
                f"a positive integer. Received: units={units}"
            )