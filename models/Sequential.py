import numpy as np
import h5py


class Sequential:

    def __init__(self):
        self.layers = []
        return


    def add(self, layer):
        self.layers.append(layer)
        return


    def build(self):
        pass


    def compile(self, optimizer='adam', loss='MSE', metrics=['accuracy']):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        return


    def evaluate(self, x, y, batch_size):
        pass


    def backward(self):
        pass


    def fit(self, x, y, epochs=1, batch_size=None, validation_split=0.1):
        print("Fit start")
        self.x = x.to_numpy()
        self.y = y.to_numpy()

        # Начинаем цикл по всем входным наблюдениям
        for num in range(len(self.x)):
            z_prev = self.x[num].reshape(len(self.x[num]), 1)

            # Начинаем цикл по всем скрытым слоям нейронной сети
            for num_layer in range(len(self.layers)):
                self.layers[num_layer].activations = np.dot(self.layers[num_layer].W, z_prev) + self.layers[num_layer].b

                # Реализация активации ReLU
                if self.layers[num_layer].activation == "relu":
                    for activation_num in range(len(self.layers[num_layer].activations)):
                        self.layers[num_layer].outputs[activation_num] = max(0, self.layers[num_layer].activations[activation_num])

                # Реализация активации Sigmoid
                if self.layers[num_layer].activation == "sigmoid":
                    for activation_num in range(len(self.layers[num_layer].activations)):
                        self.layers[num_layer].outputs[activation_num] = 1 / (1 + np.exp(-self.layers[num_layer].activations[activation_num]))

                z_prev = self.layers[num_layer].outputs  # Сохраняем выходы текущего слоя.
                                                         # На следующей итерации они будут выходами предыдущего слоя
                print(num_layer, self.layers[num_layer].outputs)
            print("Error =", self.y[num] - sum(self.layers[len(self.layers) - 1].outputs))

        print("Fit finish")
        return


    def predict(self, x):
        pass


    def save(self, path_name):
        pass
