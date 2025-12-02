import numpy as np
import time
from utils.functions import unNormalizeData
from utils.graph import graph


class NeuralNetwork:

    def __init__(self, layers, data_num, learningRate=0.05):
        # Weights and biases connect a layer and the next one
        self.w = []
        self.b = []
        self.lr = learningRate
        # Number of samples
        self.m = data_num
        # Weights and biases intialization
        for i in range(len(layers) - 1):
            # self.w.append(np.random.rand(layers[i + 1], layers[i]))
            # self.b.append(np.random.rand(layers[i + 1], 1))
            # ERROR: Mejora Claude
            # Para ReLU (He initialization)
            self.w.append(
                np.random.randn(layers[i + 1], layers[i]) * np.sqrt(2.0 / layers[i])
            )
            self.b.append(np.zeros((layers[i + 1], 1)))

        # print("w")
        # for i in range(len(self.w)):
        #     print(self.w[i].shape, end=" ")
        # print("\nb")
        # for j in range(len(self.b)):
        #     print(self.b[j].shape, end=" ")
        # print()
        self.layers = layers
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

    def saveModel(self, name, norm_price) -> None:
        print("Saving model")
        # print("norm_price", norm_price)
        print("w", end=" ")
        for i in range(len(self.w)):
            print(self.w[i].shape, end=" ")
        print("\nb", end=" ")
        for j in range(len(self.b)):
            print(self.b[j].shape, end=" ")
        print()
        w = self.w.copy()
        b = self.b.copy()
        with open(f"{name}.txt", "w") as f:
            # Weights and biases
            txt = self.matrixToStr(w, "Weights")
            f.write(str(txt) + "\n")
            txt = self.matrixToStr(b, "Biases")
            f.write(str(txt) + "\n")
            # Layers
            # TO DO: Hace falta parsearlos?
            txt = "Layers\n"
            for layer in self.layers:
                txt += f"{layer:03.0f}, "
            txt = txt[:-2]
            f.write(txt + "\n")
            # Data number
            txt = f"\nData number\n{self.m:05.0f}"
            f.write(txt + "\n")
            # Normalising parameters
            print(f"Norm price: {norm_price}")
            txt = "\nX_mins\n"
            for mins in norm_price["X_mins"]:
                txt += str(mins) + ", "
            txt = txt[:-2] + "\nX_maxs\n"
            for maxs in norm_price["X_maxs"]:
                txt += str(maxs) + ", "
            txt = txt[:-2] + "\n"
            txt += f"Y_min\n{norm_price["Y_min"]}\nY_max\n{norm_price["Y_max"]}"
            f.write(txt)

        print("\nData loaded correctly")
        print(self)

    def matrixToStr(self, arr, name=""):
        # Params: arr of weigths or biases
        print(f"To str. Len:", len(arr))
        txt = f"{name}"
        for matrix in arr:
            shape = f"{matrix.shape[0]:03.0f}; {matrix.shape[1]:03.0f}"
            txt += f"\n\n{name[0]} ({shape})\n"
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    txt += str(matrix[i][j]) + ", "
                txt = txt[:-2] + "\n"
        return txt

    def __str__(self):
        txt = "MODEL:\n W: "
        for i in range(len(self.w)):
            txt += f"{self.w[i].shape}, "
        txt = txt[:-2]
        txt += "\n B:"
        for j in range(len(self.b)):
            txt += f"{self.b[j].shape}, "
        txt = txt[:-2]
        txt += f"\n Learning rate: {self.lr}  Layers: {self.layers} Training data length (m): {self.m}"
        # print("w", self.w[0], self.w[1])
        # print("b", self.b[0], self.b[1])
        return txt

    def train(self, X_train, Y_train, X_test, Y_test, X_test2, Y_test2, norm_price, total_steps) -> None:
        t = time.time()
        for step in range(total_steps):
            self.feedForward(X_train)
            self.backwardsPropagation(Y_train)
            # if (step - 1) % int(total_steps / 10) == 0:
            if step % (total_steps // 20) == 0 or step == total_steps - 1:
                # Testing and ploting data
                self.test(X_train, Y_train, X_test, Y_test, X_test2, Y_test2, norm_price, step - 1)
                print(f"Time elapsed: {np.round(time.time() - t, 2)} seconds")
                t = time.time()

    # NOT USED / REFACTOR: Weights and layers doesn't match up
    def trainBatches(
        self, X_train, Y_train, X_test, Y_test, norm_price, batchSize, epochs
    ) -> None:
        t = time.time()
        for epoch in range(epochs):
            for i in range(0, self.m, batchSize):
                X_batch = X_train[i : i + batchSize]
                Y_batch = Y_train[i : i + batchSize]
                self.feedForward(X_batch)
                self.backwardsPropagation(Y_batch)
                if (epoch - 1) % int(epochs / 10) == 0:
                    # Testing
                    self.test(X_train, Y_train, X_test, Y_test, norm_price, epoch - 1)
                    print(f"Time elapsed: {np.round(time.time() - t, 2)} seconds")
                    t = time.time()

    # Forwards propagation
    def feedForward(self, inputs):
        # Feed the inputs to the NN
        self.a[0] = inputs

        # Feeding forward each layer
        for i in range(1, self.len):
            self.z[i] = np.dot(self.w[i - 1], self.a[i - 1])
            self.z[i] = np.add(self.z[i], self.b[i - 1])
            # Applying the activation function
            # if i != self.len:
            # ERROR en el código antiguo
            if i < self.len - 1:
                # ReLU for intermediate layers
                self.a[i] = self.ReLU(self.z[i], i)
            else:
                # ERROR en el codigo antiguo: Sin activación
                # Applying softmax to the output layer
                # a[i] = self.softmax(z[i])
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
                # Derivada: (pred - true) * sigmoid'(z) ERROR, SIN sigmoid_prime DA IGUAL?
                self.dz[i] = self.a[i] - outMatrix
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

    # Tests model accuracy
    def test(self, X_train, Y_train, X_test, Y_test, X_test2, Y_test2, norm_price, step=-1) -> None:
        # Verifying both on train and test data to prevent overfitting

        # Normalized predictions
        pred_norm_train = self.feedForward(X_train)
        pred_norm_test = self.feedForward(X_test)
        pred_norm_test2 = self.feedForward(X_test2)

        # Normalized errors
        mse_norm_train = np.mean((pred_norm_train - Y_train) ** 2)
        mae_norm_train = np.mean(np.abs(pred_norm_train - Y_train))

        mse_norm_test = np.mean((pred_norm_test - Y_test) ** 2)
        mae_norm_test = np.mean(np.abs(pred_norm_test - Y_test))

        mse_norm_test2 = np.mean((pred_norm_test2 - Y_test2) ** 2)
        mae_norm_test2 = np.mean(np.abs(pred_norm_test2 - Y_test2))

        # Real prediction metrics
        pred_real_train = unNormalizeData(pred_norm_train, norm_price)
        Y_real_train = unNormalizeData(Y_train, norm_price)

        pred_real_test = unNormalizeData(pred_norm_test, norm_price)
        Y_real_test = unNormalizeData(Y_test, norm_price)

        pred_real_test2 = unNormalizeData(pred_norm_test2, norm_price)
        Y_real_test2 = unNormalizeData(Y_test2, norm_price)

        # Real errors
        mse_real_train = np.mean((Y_real_train - pred_real_train) ** 2)
        rmse_real_train = np.sqrt(mse_real_train)
        mae_real_train = np.mean(np.abs(Y_real_train - pred_real_train))
        mape_train = (
            np.mean(
                np.abs(
                    (Y_real_train - pred_real_train) / (np.abs(Y_real_train) + 1e-10)
                )
            )
            * 100
        )

        mse_real_test = np.mean((Y_real_test - pred_real_test) ** 2)
        rmse_real_test = np.sqrt(mse_real_test)
        mae_real_test = np.mean(np.abs(Y_real_test - pred_real_test))
        mape_test = (
            np.mean(
                np.abs((Y_real_test - pred_real_test) / (np.abs(Y_real_test) + 1e-10))
            )
            * 100
        )
        mse_real_test2 = np.mean((Y_real_test2 - pred_real_test2) ** 2)
        rmse_real_test2 = np.sqrt(mse_real_test2)
        mae_real_test2 = np.mean(np.abs(Y_real_test2 - pred_real_test2))
        mape_test2 = (
            np.mean(
                np.abs((Y_real_test2 - pred_real_test2) / (np.abs(Y_real_test2) + 1e-10))
            )
            * 100
        )

        if step != -1:
            print(f"\n--- Itineration {step} ---")
        print(
            f"Normalized (Train):   MSE: {mse_norm_train:05.4f}       MAE: {mae_norm_train:05.4f}"
        )
        print(
            f"Normalized (Test):    MSE: {mse_norm_test:05.4f}       MAE: {mae_norm_test:05.4f}"
        )
        print(
            f"Norm (Test 2025):     MSE: {mse_norm_test2:05.4f}       MAE: {mae_norm_test2:05.4f}"
        )
        print(
            f"Real scale (Train):   RMSE: {rmse_real_train:05.4f}$   MAE: {mae_real_train:05.4f}$   MAPE: {mape_train:05.4f}%"
        )
        print(
            f"Real scale (Test):    RMSE: {rmse_real_test:05.4f}$   MAE: {mae_real_test:05.4f}$   MAPE: {mape_test:05.4f}%"
        )
        print(
            f"Real (Test 2025):     RMSE: {rmse_real_test2:05.4f}$   MAE: {mae_real_test2:05.4f}$   MAPE: {mape_test2:05.4f}%"
        )
        if step == -1:
            print("MSE < 0.01      MAE < 0.03       MAPE < 10%")

        # REFACTOR, no se pintan todos los errores
        graph.updateAndPlot(
            mse_norm_train,
            mse_norm_test,
            mae_norm_train,
            mae_norm_test,
            mape_train,
            mape_test,
            step,
        )
        return (
            mse_norm_train,
            mse_norm_test,
            mae_norm_train,
            mae_norm_test,
            mape_train,
            mape_test)

    # Testing with 2025 data
    def testRealData(self, X_test, Y_test, norm_price, step=-1) -> None:
        """
        Args:
            X_test: NORMALIZADO
            Y_test: NORMALIZADO
            norm_price: dict con min/max para desnormalizar
        """
        # print("\nnorm_price", norm_price)
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
        mape = (
            np.mean(
                np.abs((Y_test_real - predictions_real) / (np.abs(Y_test_real) + 1e-10))
            )
            * 100
        )

        mape = (
            np.mean(np.abs((Y_test_real - predictions_real) / (Y_test_real + 1e-10)))
            * 100
        )
        if step != -1:
            print(f"\nItineration {step}")
        print(f"Normalized:   MSE: {mse_norm:05.4f}       MAE: {mae_norm:05.4f} \t")
        print(
            f"Real scale:   RMSE: {rmse_real:05.4f}$   MAE: {mae_real:05.4f}$   MAPE: {mape:05.4f}%"
        )

    def ReLU(self, matrix, i):
        activated_matrix = matrix.copy()
        # Using numpy: return np.maximum(0, matrix)
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                elt = matrix[i][j]
                activated_matrix[i][j] = 0 if elt < 0 else elt
        return activated_matrix

    # "ERRORes": Modificar la matriz original
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
