import numpy as np

# Source - https://stackoverflow.com/a/78793583
# Posted by Leolo, modified by community. See post 'Timeline' for change history
# Retrieved 2026-05-25, License - CC BY-SA 4.0

np.set_printoptions(legacy='1.25')


def sigmoid(arr, bias):
    total = 0
    for i in range(len(arr)):
        total += arr[i][0] * arr[i][1]
    return 1 / (1 + np.exp(total - bias))


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

theta4 = -1
theta5 = -1
theta6 = -1
theta7 = -1

learning_rate = 0.1

#use x1 x2 x3 as input later
#now, placeholder values for x1 x2 x3
x1 = 1
x2 = 0
x3 = 0

x1= int(input("Enter value for x1: "))
x2= int(input("Enter value for x2: "))
x3= int(input("Enter value for x3: "))

y6= int(input("Enter desired value for y6: "))
y7= int(input("Enter desired value for y7: "))

x4 = sigmoid([(x1, w14), (x2, w24), (x3, w34)], bias4)
x5 = sigmoid([(x1, w15), (x2, w25), (x3, w35)], bias5)

x6 = sigmoid([(x4, w46), (x5, w56)], bias6)
x7 = sigmoid([(x4, w47), (x5, w57)], bias7)

print("-----")
print("from training:")
print([x6, x7])
print("-----")
print("your desired values:")
print([y6, y7])
print("-----")
print("error:")
error6 = y6 - x6
error7 = y7 - x7
print([error6, error7]) 
print("-----")
gradient_error6 = error6 * x6 * (1 - x6)
gradient_error7 = error7 * x7 * (1 - x7)
print("gradient error for node 6:", gradient_error6)
print("gradient error for node 7:", gradient_error7)
print()
print("weight corrections for node 6:")
print("w46:     ", learning_rate * gradient_error6 * x4)
print("w56:     ", learning_rate * gradient_error6 * x5)
print("theta6:  ", learning_rate * gradient_error6 * theta6)
w46 = learning_rate * gradient_error6 * x4 + w46
w56 = learning_rate * gradient_error6 * x5 + w56
bias6 = learning_rate * gradient_error6 * theta6 + bias6
print()
print("new weights for node 6:")
print("w46:     ", w46)
print("w56:     ", w56)
print("bias6:   ", bias6)
print()
print("weight corrections for node 7:") 
print("w47:     ", learning_rate * gradient_error7 * x4)
print("w57:     ", learning_rate * gradient_error7 * x5)
print("bias7:   ", learning_rate * gradient_error7 * theta7)
w47 = learning_rate * gradient_error7 * x4 + w47
w57 = learning_rate * gradient_error7 * x5 + w57
bias7 = learning_rate * gradient_error7 * theta7 + bias7
print()
print("new weights for node 7:")
print("w47:     ", w47)
print("w57:     ", w57)
print("bias7:   ", bias7)

#Please if you are continuing the sequence, just copy paste it and change the node it connects to.
#also, loop 8 times.