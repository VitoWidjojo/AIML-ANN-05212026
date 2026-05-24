import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid(x1, w1, x2, w2, x3, w3, bias):
    return sigmoid(x1 * w1 + x2 * w2 + x3 * w3 - bias)

def sigmoid(x1, w1, x2, w2, bias):
    return sigmoid(x1 * w1 + x2 * w2 - bias)

#3 input x1 x2 x3

# 6 7
# 4 5
#1 2 3

w14 = 0.5      
w24 = 0.4
w34 = -0.2
w15 = 0.7
w25 = 0.6
w35 = 0.8
w46 = 0.6
w56 = 0.7
w47 = 0.3
w57 = -0.2

bias4 = 0.8
bias5 = -0.1
bias6 = 0.3
bias7 = 0.4

#use x1 x2 x3 as input later
#now, placeholder values for x1 x2 x3
x1 = 1
x2 = 0
x3 = 0

x4 = sigmoid(x1, w14, x2, w24, x3, w34, bias4)
x5 = sigmoid(x1, w15, x2, w25, x3, w35, bias5)

x6 = sigmoid(x4, w46, x5, w56, bias6)
x7 = sigmoid(x4, w47, x5, w57, bias7)