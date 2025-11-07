import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub

from utils.neural_network import NeuralNetwork
import utils.functions as utils

# xs: (variables, data_length)     --> (6, 1438)
# ys: (data_length,)               --> (1438,)
# w: (w1, variables), (w1, output) -->
# b: (w1, 1), (w2, 1), ?(1, 1)     -->
X_train, X_test, Y_train, Y_test = [], [], [], []

# Hyperparameters
# Percentage of data used in training
DATA_SPLIT = 70

def main():
    # --- Loading and parsing data ---
    X_train, X_test, Y_train, Y_test = utils.loadData("data/ETH_day.txt")
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


if __name__ == "__main__":
    main()
