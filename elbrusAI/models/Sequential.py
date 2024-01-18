import numpy as np
from elbrusAI.layers.activations import relu, sigmoid, tanh
from elbrusAI.optimizers import SGD


class Sequential:

    def __init__(self):
        print('eeeeee')
        self.layers = []
        return


    def add(self, layer):
        self.layers.append(layer)
        return


    def build(self):
        pass


    def compile(self, optimizer='SGD', loss='MSE', metrics=None):
        if metrics is None:
            metrics = ['accuracy']
        if optimizer == "SGD":
            self.optimizer = SGD()
        self.loss = loss
        self.metrics = metrics

        return


    def evaluate(self, x, y, batch_size):
        pass


    def f_sigmoida(self, x):
        """Вычисление активационной функции."""
        res = np.array(len(x))
        for i in range(res):
            res[i] = 1 / (1 + np.exp(-x[i]))

        return res


    # def diff(self, layer):
    #     """Вычисление производной активационной функции."""
    #     if layer.a_function == "sigmoid":
    #         return self.f_sigmoida(layer.outputs) * (1 - self.f_sigmoida(layer))


    def diff(self, layer):
        """Вычисление производной активационной функции."""
        if layer.a_function == "sigmoid":
            return self.f_sigmoida(layer.outputs) * (1 - self.f_sigmoida(layer))


    def activate(self, activations, a_function):
        """Вычисление активационной функции."""

        # Реализация активации через ReLU
        if a_function == "relu":
            return np.asarray(list(map(relu, activations)))

        # Реализация активации через Sigmoid
        if a_function == "sigmoid":
            print("activ:", activations)
            return np.asarray(list(map(sigmoid, activations)))

        # Реализация активации через Tanh
        if a_function == "tanh":
            return np.asarray(list(map(tanh, activations)))

        return 0


    def show_layers(self):
        print("---- Start showing layers ----")
        for num_layer in range(len(self.layers)):
            print("Output", num_layer, self.layers[num_layer].outputs)
            print("Weights", num_layer, self.layers[num_layer].weights)
            print("              -------")
        print("---- Finish showing layers ----")


    def forward(self, input_layer):
        """Прямой проход по нейронной сети для вычисления выхода нейронной сети."""
        z_prev = input_layer  # Значения входного слоя становятся значениями предыдущего
        for num_layer in range(len(self.layers)):
            print('weigts', self.layers[num_layer].weights)
            print('biases', self.layers[num_layer].biases)
            print('z_prev', z_prev)
            # Вычисляем активации слоя (выходы до активационной функции)
            self.layers[num_layer].activations = np.matmul(self.layers[num_layer].weights, z_prev) + self.layers[num_layer].biases
            # print('activations', layers[num_layer].activations)
            # print(num_layer)
            # print('-----------------------')
            self.layers[num_layer].outputs = self.activate(self.layers[num_layer].activations, self.layers[num_layer].a_function)
            z_prev = self.layers[num_layer].outputs  # Сохраняем выходы текущего слоя.
                                                # На следующей итерации они будут выходами предыдущего слоя
            print(num_layer, self.layers[num_layer].outputs)
        return 0


    def backward(self, x, y_true):
        """Обратный проход по нейронной сети - backpropagation."""
        if self.loss == "MSE":
            delta = (self.layers[-1].outputs[0] - y_true).reshape(len(self.layers[-1].outputs[0]), 1)
            print("self.layers[-1].outputs", self.layers[-1].outputs)
            print("delta", delta)
            for num_layer in range(len(self.layers)-1, -1, -1):
#                layers[num_layer].weights -= self.optimizer.learning_rate * delta * layers[num_layer - 1].activations
#                for num_neuron in range(layers[num_layer].units):
                self.show_layers()
                print('num_layer', num_layer)
                print(self.optimizer.learning_rate * np.matmul(self.layers[num_layer - 1].outputs, delta))
                print('layers[num_layer].weights1', self.layers[num_layer].weights)
                print('layers[num_layer - 1].outputs', self.layers[num_layer - 1].outputs)
                print('delta', delta)
                self.layers[num_layer].weights -= (self.optimizer.learning_rate * np.matmul(self.layers[num_layer - 1].outputs, delta))[0]
                print('layers[num_layer].weights2', self.layers[num_layer].weights)
                if num_layer != 0:  # Для входного слоя считать локальный градиент не нужно
                    delta *= self.layers[num_layer].weights * self.diff(self.layers[num_layer-1])
                print('----')
        return 0


    def fit(self, x, y, epochs=1, batch_size=None, validation_split=0.1):
        """Обучение модели."""

        print("Fit start")
        self.x_train = x.to_numpy()  # Конвертируем датасет в numpy-массив
        self.y_train = y.to_numpy()  # Конвертируем датасет в numpy-массив
        print('x', self.x_train)
        print('x_reshape', self.x_train[0].reshape(len(self.x_train[0]), 1))
        print('x_reshape', self.x_train[0])
        print('y', self.y_train)
        # Начинаем обучение по эпохам
        for _ in range(epochs):
            # Делаем первый проход с рандомными весами чтобы получить первую ошибку
            #self.layers = self.forward(self.layers, self.x[0].reshape(len(self.x[0]), 1))
            ####self.forward(self.x[0].reshape(len(self.x[0]), 1))

            # Начинаем цикл по всем входным наблюдениям
            for num in range(len(self.x_train)):
                # Делаем проход вперед для вычисления нового значения выхода из нейронной сети
                self.forward(self.x_train[num].reshape(len(self.x_train[num]), 1))
                print('Forward finished')
                # Делаем backprop для обновления весов
                self.backward(self.x_train[num].reshape(len(self.x_train[num]), 1), self.y_train[num]) #.reshape(len(self.x[num]), 1), self.y[num])
                print('Backward finished')
                print("Error =", self.y_train[num] - sum(self.layers[-1].outputs))

        print("Fit finish")
        return


    def predict(self, x):
        pass


    def save(self, path_name):
        pass
