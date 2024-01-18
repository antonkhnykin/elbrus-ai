import numpy as np


class Dense:

    def __init__(self, units, input_shape, activation=None):
        self.units = units  # Число выходов слоя
        self.a_function = activation  # Активационная функция слоя
        self.input_shape = input_shape  # Число входов слоя
        self.weights = np.random.rand(units, input_shape)  # По умолчанию заполняем массив весов случайными значениями от 0 до 1
        print("weigths", self.weights)
        self.biases = np.random.rand(units, 1)  # По умолчанию заполняем массив смещений случайными значениями от 0 до 1
        self.activations = np.zeros(units)  # По умолчанию заполняем массив активаций нулями
        self.outputs = np.zeros(units)  # По умолчанию заполняем массив выходов слоя нулями

        if self.units < 0:
            raise ValueError(
                "Received an invalid value for `units`, expected "
                f"a positive integer. Received: units={units}"
            )
