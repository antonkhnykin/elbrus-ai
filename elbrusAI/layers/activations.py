import numpy as np


def relu(x):
    return max(0, x)


def sigmoid(x):
    return 1 + np.exp(-x)


def tanh(x):
    return 2 / (1 + np.exp(-x)) - 1