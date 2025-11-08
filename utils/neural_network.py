import numpy as np
from utils.functions import unNormalizeData


class NeuralNetwork:

    def __init__(self, layers, data_num, learningRate):
        # Weights and biases connect a layer and the next one
        self.w = []
        self.b = []
        self.lr = learningRate
        # Number of samples
        self.m = data_num
        for i in range(len(layers) - 1):
            # self.w.append(np.random.rand(layers[i + 1], layers[i]))
            # self.b.append(np.random.rand(layers[i + 1], 1))
            # Para ReLU (He initialization)
            # ERROR: Mejora Claude
            self.w.append(
                np.random.randn(layers[i + 1], layers[i]) * np.sqrt(2.0 / layers[i])
            )
            # Para bias
            self.b.append(np.zeros((layers[i + 1], 1)))

        # print("w")
        # for i in range(len(self.w)):
        #     print(self.w[i].shape, end=" ")
        # print("\nb")
        # for j in range(len(self.b)):
        #     print(self.b[j].shape, end=" ")
        # print()
        self.len = len(layers)

        # -- Inicializing the lists storing every layer matrix --
        # Neuron values before activation function (raw)
        self.z = [None] * (len(layers))
        # ERRORES
        self.dz = [None] * (len(layers))
        # Value after activation function (ReLU and softmax)
        self.a = [None] * (len(layers))
        self.da = [None] * (len(layers))
        # Derivative of the weights and biases layers to calculate the error
        self.dw = [None] * (len(layers))
        self.db = [None] * (len(layers))

    def train(self, X_train, Y_train, X_test, Y_test, norm_price, steps) -> None:
        # print("TRAINING...")
        for step in range(steps):
            self.feedForward(X_train)
            self.backwardsPropagation(Y_train)
            if (step - 1) % int(steps / 10) == 0:
                # Testing
                self.test(X_test, Y_test, norm_price, step - 1)


    # Forwards propagation
    def feedForward(self, inputs):
        # Feed the inputs to the NN
        self.a[0] = inputs

        # Feeding forward each layer
        for i in range(1, self.len):
            self.z[i] = np.dot(self.w[i - 1], self.a[i - 1])
            self.z[i] = np.add(self.z[i], self.b[i - 1])
            # Applying the activation function
            # if i != self.len: ERROR
            if i < self.len - 1:
                # ReLU for intermediate layers
                self.a[i] = self.ReLU(self.z[i], i)
            else:
                # Applying softmax to the output layer
                # a[i] = self.softmax(z[i])
                # ERROR: Sin activación
                self.a[i] = self.sigmoid(self.z[i])

        return self.a[self.len - 1]

    def backwardsPropagation(self, outMatrix) -> None:
        last = self.len - 1
        # Calculating the error, based on the expected output (outMatrix)

        # Loop to create dz, dw, db
        # print("dz", len(self.dz))
        for i in range(last, 0, -1):
            # print("i", i, last)
            # Substracting the output minus the correct data (outMatrix)
            # or the output of each layer (z) applying the inverse act. fn
            if i == last:
                # Derivada: (pred - true) * sigmoid'(z) ERROR, SIN sigmoid_prime
                self.dz[i] = (self.a[i] - outMatrix)
            else:
                w_T = np.transpose(self.w[i])
                self.dz[i] = np.dot(w_T, self.dz[i + 1])
                self.dz[i] *= self.ReLU_prime(self.z[i])
            self.dw[i] = (1 / self.m) * np.dot(self.dz[i], np.transpose(self.a[i - 1]))
            # Bias gradient
            self.db[i] = (1 / self.m) * np.sum(self.dz[i], axis=1, keepdims=True)

        # Updating parameters based on the error (dw, db) and the learning rate
        # print("CON (1, last) Y EL BLOQUE SUPERIOR NO TIENE ERRORES, PERO TAMPOCO MEJORA")
        # print("CON AXIS=0 SE VA EL ERROR PERO NO MEJORA")
        for n in range(0, last):
            self.w[n] -= self.lr * self.dw[n + 1]
            self.b[n] -= self.lr * self.db[n + 1]

    def test(self, X_test, Y_test, norm_price, step=-1) -> None:
        """
        Args:
            X_test: NORMALIZADO
            Y_test: NORMALIZADO
            norm_price: dict con min/max para desnormalizar
        """
        # Normalized metrics
        predictions_norm = self.feedForward(X_test)

        # # Verifica predicciones normalizadas
        # print("*"*10, "predictions_norm stats:", "*"*10)
        # print(f"  min: {predictions_norm.min():.6f}")
        # print(f"  max: {predictions_norm.max():.6f}")
        # print(f"  mean: {predictions_norm.mean():.6f}")

        mse_norm = np.mean((predictions_norm - Y_test) ** 2)
        mae_norm = np.mean(np.abs(predictions_norm - Y_test))

        # Real metrics
        predictions_real = unNormalizeData(predictions_norm, norm_price)
        Y_test_real = unNormalizeData(Y_test, norm_price)

        rmse_real = np.sqrt(np.mean((Y_test_real - predictions_real) ** 2))
        mae_real = np.mean(np.abs(Y_test_real - predictions_real))
        mape = np.mean(np.abs((Y_test_real - predictions_real) / (np.abs(Y_test_real) + 1e-10))) * 100


        mape = (
            np.mean(np.abs((Y_test_real - predictions_real) / (Y_test_real + 1e-10))) * 100
        )
        if step != -1:
            print(f"\nItineration {step}")
        print(
            f"Normalized:   MSE: {mse_norm:05.4f}       MAE: {mae_norm:05.4f} \t"
        )
        print(
            f"Real scale:   RMSE: {rmse_real:05.4f}$   MAE: {mae_real:05.4f}$   MAPE: {mape:05.4f}%"
        )
        if step == -1: print("MSE < 0.01        MAE < 0.03       MAPE < 10%")

    def ReLU(self, matrix, i):
        activated_matrix = matrix.copy()
        # Using numpy: return np.maximum(0, matrix)
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                elt = matrix[i][j]
                activated_matrix[i][j] = 0 if elt < 0 else elt
        return activated_matrix

    def ReLU_prime(self, matrix):
        # As ReLU outputs a straight line, its derivative is 1 if x > 0
        # Using numpy: return (Z > 0).astype(float)
        relu = matrix.copy()
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                elt = matrix[i][j]
                relu[i][j] = 0 if elt <= 0 else 1
        return relu

    # ERROR: Para clasificación
    def softmax(self, matrix):
        # Posible ERROR: axis=0 (antes, axis=1)
        if not np.isfinite(matrix).all():
            print("⚠️ Non-finite values in matrix:", matrix)

        e_x = np.exp(matrix - np.max(matrix, axis=0, keepdims=True))
        if np.isnan(e_x).any():
            print("⚠️ NaN detected in softmax output")

        return e_x / np.sum(e_x, axis=0, keepdims=True)

    def sigmoid(self, matrix):
        return 1 / (1 + np.exp(-np.clip(matrix, -500, 500)))  # clip evita overflow

    def sigmoid_prime(self, matrix):
        s = self.sigmoid(matrix)
        return s * (1 - s)
