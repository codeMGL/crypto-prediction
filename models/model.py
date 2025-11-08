# xs: (variables, data_length)     --> (6, 1438)
# ys: (data_length,)               --> (1438,)
# w: (w1, variables), (w1, output) -->
# b: (w1, 1), (w2, 1), ?(1, 1)     -->
import numpy as np

# def feedForward(self, inputs, checking_error = False):
#     # Feed the inputs to the NN
#     self.a[0] = inputs

#     # Coping values
#     z = self.z.copy()
#     a = self.a.copy()
#     w = self.w.copy()
#     b = self.b.copy()

#     # Feeding forward each layer
#     for i in range(1, self.len):
#         z[i] = np.dot(w[i - 1], a[i - 1])
#         z[i] = np.add(z[i], b[i - 1])
#         # Applying the activation function
#         # if i != self.len: ERROR
#         if i < self.len - 1:
#             # ReLU for intermediate layers
#             a[i] = self.ReLU(z[i], i)
#         else:
#             # Applying softmax to the output layer
#             # a[i] = self.softmax(z[i])
#             # ERROR: Sin activación
#             a[i] = z[i]

#     if not checking_error:
#         # Modifying arrays
#         self.z = z.copy()
#         self.a = a.copy()
#         self.w = w.copy()
#         self.b = b.copy()
#     return a[self.len - 1]


class NeuralNetwork:

    def __init__(self, layers, data_num, learningRate=0.2):
        # Weights and biases connect a layer and the next one
        self.w = []
        self.b = []
        self.lr = learningRate
        self.m = data_num
        for i in range(len(layers) - 1):
            self.w.append(np.random.rand(layers[i + 1], layers[i]))
            self.b.append(np.random.rand(layers[i + 1], 1))

        self.len = len(layers)
        # Neuron values before activation function (raw)
        self.z = [None] * (len(layers))
        self.dz = [None] * (len(layers))
        # Value after activation function (ReLU and softmax)
        self.a = [None] * (len(layers))
        self.da = [None] * (len(layers))
        # Derivative of the weights and biases layers to calculate the error
        self.dw = [None] * (len(layers))
        self.db = [None] * (len(layers))

    def train(self, dataSamples, outputsMatrix, steps=100):
        # print("TRAINING...")
        if steps <= 0:
            # print("///////////////////////////////")
            print("Finished training")
            return
        self.feedForward(dataSamples)
        self.backwardsPropagation(outputsMatrix)
        steps -= 1
        self.train(dataSamples, outputsMatrix, steps)

    # Forwards propagation
    def feedForward(self, inputs):
        print("Forwards propagation")
        # Feed the inputs to the NN
        self.a[0] = inputs

        # Feeding forward each layer
        for i in range(1, self.len):
            self.z[i] = np.dot(self.w[i - 1], self.a[i - 1])
            self.z[i] = np.add(self.z[i], self.b[i - 1])
            # Applying the activation function
            if i != self.len:
                # ReLU for intermediate layers
                self.a[i] = self.ReLU(self.z[i], i)
            else:
                # Applying softmax to the output layer
                self.a[i] = self.softmax(self.z[i])
                # size(self.a[i], "OUTPUT after softmax:")

    def backwardsPropagation(self, outMatrix):
        print("Backwards propagation")
        last = self.len - 1
        # Calculating the error, based on the expected output (outMatrix)

        # Loop to create dz, dw, db
        for i in range(last, 0, -1):
            # Substracting the output minus the correct data (outMatrix)
            # or the output of each layer (z) applying the inverse act. fn
            if i == last:
                self.dz[last] = self.a[last] - outMatrix
            else:
                w_T = np.transpose(self.w[i])
                self.dz[i] = np.dot(w_T, self.dz[i + 1])
                self.dz[i] *= self.ReLU_prime(self.z[i])
            # size(self.dz[i], f"dz{i}")
            self.dw[i] = (1 / self.m) * np.dot(self.dz[i], np.transpose(self.a[i - 1]))
            # size(self.dw[i], f"dw{i}")
            # Bias gradient
            self.db[i] = (1 / self.m) * np.sum(self.dz[i], axis=1, keepdims=True)
            # size(self.db[i], f"db{i}")

        # Updating parameters based on the error (dw, db) and the learning rate
        for n in range(1, last):
            self.w[n] -= self.lr * self.dw[n + 1]
            self.b[n] -= self.lr * self.db[n + 1]

    def ReLU(self, matrix, i):
        # Using numpy: return np.maximum(0, matrix)
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                elt = matrix[i][j]
                matrix[i][j] = 0 if elt < 0 else elt
        return matrix

    def ReLU_prime(self, matrix):
        # As ReLU outputs a straight line, its derivative is 1 if x > 0
        # Using numpy: return (Z > 0).astype(float)
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                elt = matrix[i][j]
                matrix[i][j] = 0 if elt <= 0 else 1
        return matrix

    def softmax(self, matrix):
        e_x = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
