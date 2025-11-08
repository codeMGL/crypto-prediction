import time
import numpy as np

from utils.neural_network import NeuralNetwork
# from models.model import NeuralNetwork
import utils.functions as utils

# xs: (variables, data_length)     --> (6, 1438)
# ys: (data_length,)               --> (1438,)
# w: (w1, variables), (w1, output) -->
# b: (w1, 1), (w2, 1), ?(1, 1)     -->

# ---------------------------------------------------------------------
# To add new ETH prices, we need to preserve X_min and X_max


X_train, X_test, Y_train, Y_test = [], [], [], []

# --- Hyperparameters ---
# Percentage of data used in training
DATA_SPLIT = 70 # 70%
learning_rate = 0.005
training_steps = 3000

def main():
    # --- Loading and parsing data ---
    X_train, X_test, Y_train, Y_test = utils.loadData(DATA_SPLIT)
    # print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    return

    # --- Normalizing train data ---
    # Returning the min and max prices to reverse normalisation in MAPE
    X_train_norm, Y_train_norm, norm_price = utils.normalizeTrainData(X_train, Y_train)
    X_test_norm, Y_test_norm = utils.normalizeTestData(X_test, Y_test, norm_price)

    # --- Training the model ---
    # layers = [X_train.shape[0], 256, 256, 1]
    # layers = [X_train.shape[0], 64, 32, 1]
    # layers = [X_train.shape[0], 64, 64, 32, 1] # BUENA  MSE: 0.0007 MAE: 0.0220  | learning rate = 0.005, layers = [5, 64, 64, 32, 1]
    layers = [
        X_train.shape[0],
        64,
        64,
        64,
        1,
    ]  # MSE: 0.0006       MAE: 0.0190  MAPE: 88.9005%
    data_num = X_train.shape[1]

    model = NeuralNetwork(layers, data_num, learning_rate)
    print(f"TRAINING: {training_steps} steps, learning rate = {learning_rate}, layers = {layers}")
    t0 = time.time()
    model.train(X_train_norm, Y_train_norm, X_test_norm, Y_test_norm, norm_price, training_steps)
    t_f = time.time()
    print(f"Training took {np.round((t_f - t0) / 60, 2)} minutes!")

    # --- Predicting new data ---
    print("\nTEST:")
    # ERROR: Ya está normalizado
    # X_test_norm, Y_test_norm, norm_price_test = utils.normalizeData(X_test, Y_test)
    model.test(X_test_norm, Y_test_norm, norm_price)


if __name__ == "__main__":
    main()
