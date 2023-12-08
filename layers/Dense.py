

class Dense():

    units = 0
    activation = ""
    input_shape = 0

    def __init(self, units, input_shape, activation=None, ):
        self.units = units
        self.activation = activation
        self.input_shape = input_shape

        if self.units < 0:
            raise ValueError(
                "Received an invalid value for `units`, expected "
                f"a positive integer. Received: units={units}"
            )