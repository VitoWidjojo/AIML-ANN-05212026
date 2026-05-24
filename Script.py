import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid(x1, w1, x2, w2, bias):
    return sigmoid(x1 * w1 + x2 * w2 - bias)

